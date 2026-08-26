"""Risk manager : dernier checkpoint avant TOUT ordre.

Ce module est independant de la logique de strategie et ne peut pas etre
contourne : l'executeur n'accepte un ordre que s'il porte un RiskVerdict
approuve, et seul le risk manager en emet.

Les verifications sont sequentielles et court-circuitantes. Un seul refus suffit
a annuler le trade — il n'y a pas de "score global" ou une bonne note
compenserait une violation. Les limites en dur (2 % par position, -15 % de
drawdown total) sont importees du code, jamais de la config.

Ordre des controles, du plus grave au plus anodin :
    kill switch -> drawdowns -> pauses -> circuit breakers -> regime ->
    devil advocate -> qualite du trade -> limites de position -> rate limits
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from trader.config import (
    HARD_MAX_DRAWDOWN_TOTAL_PCT,
    HARD_MAX_POSITION_PCT,
    Settings,
)
from trader.logging_setup import get_logger
from trader.models import (
    ContraRecommendation,
    RiskDecision,
    RiskVerdict,
    TradeIntent,
)
from trader.portfolio import Portfolio
from trader.risk.circuit_breaker import CircuitBreaker
from trader.risk.kill_switch import KillSwitch
from trader.risk.position_sizer import PositionSizer
from trader.utils.time_utils import to_utc, utc_now

log = get_logger(__name__)


@dataclass(slots=True)
class TradingPause:
    """Pause imposee suite a un drawdown."""

    reason: str
    until: datetime
    triggered_at: datetime

    def is_active(self, now: datetime) -> bool:
        """Vrai si la pause court toujours."""
        return now < self.until


@dataclass(slots=True)
class RiskState:
    """Etat consolide du risque, expose au monitoring."""

    kill_switch_active: bool = False
    paused_until: datetime | None = None
    pause_reason: str = ""
    orders_last_hour: int = 0
    orders_today: int = 0
    breaker_trips: int = 0
    rejections: dict[str, int] = field(default_factory=dict)


class RiskManager:
    """Valide, redimensionne ou refuse chaque intention de trade.

    Un refus n'est jamais negociable : il n'existe aucun parametre, aucun mode
    et aucun appel qui permette de passer outre.
    """

    def __init__(
        self,
        settings: Settings,
        portfolio: Portfolio,
        kill_switch: KillSwitch | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        sizer: PositionSizer | None = None,
        event_sink: Callable[[str, str, str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.settings = settings
        self.config = settings.risk
        self.portfolio = portfolio
        self.kill_switch = kill_switch or KillSwitch(settings.kill_switch, event_sink=event_sink)
        self.breaker = circuit_breaker or CircuitBreaker(
            settings.risk.circuit_breakers, event_sink=event_sink
        )
        self.sizer = sizer or PositionSizer(settings.risk)
        self.pause: TradingPause | None = None
        self._order_times: deque[datetime] = deque(maxlen=1000)
        self._rejections: dict[str, int] = {}

        # Verification defensive : si quelqu'un modifie la config a chaud pour
        # depasser les limites en dur, on refuse de demarrer.
        self._assert_hard_limits()

    def _assert_hard_limits(self) -> None:
        """Verifie au demarrage que la configuration respecte les limites en dur."""
        if self.config.max_position_pct > HARD_MAX_POSITION_PCT:
            raise ValueError(
                f"config corrompue : max_position_pct={self.config.max_position_pct} "
                f"depasse la limite en dur {HARD_MAX_POSITION_PCT}"
            )
        if self.config.max_drawdown_total_pct > HARD_MAX_DRAWDOWN_TOTAL_PCT:
            raise ValueError(
                f"config corrompue : max_drawdown_total_pct="
                f"{self.config.max_drawdown_total_pct} depasse la limite en dur "
                f"{HARD_MAX_DRAWDOWN_TOTAL_PCT}"
            )

    # ------------------------------------------------------------- controle

    def evaluate(
        self,
        intent: TradeIntent,
        prices: dict[str, float] | None = None,
        now: datetime | None = None,
        win_rate: float | None = None,
        win_loss_ratio: float | None = None,
    ) -> RiskVerdict:
        """Evalue une intention de trade et rend un verdict definitif."""
        stamp = to_utc(now or utc_now())
        reasons: list[str] = []
        checks: dict[str, object] = {}

        # 1. Kill switch : rien ne passe, jamais.
        if self.kill_switch.is_triggered():
            return self._reject(
                intent, f"kill switch arme : {self.kill_switch.reason()}", "kill_switch"
            )

        # 2. Drawdowns : les pertes deja subies limitent les risques suivants.
        drawdown = self.portfolio.drawdowns(stamp)
        checks["drawdown_daily_pct"] = round(drawdown.daily_pct, 3)
        checks["drawdown_weekly_pct"] = round(drawdown.weekly_pct, 3)
        checks["drawdown_total_pct"] = round(drawdown.total_pct, 3)

        if drawdown.total_pct >= self.config.max_drawdown_total_pct:
            self.kill_switch.trigger(
                f"drawdown total {drawdown.total_pct:.2f} % >= "
                f"{self.config.max_drawdown_total_pct:.2f} %",
                source="risk_manager",
                details={"equity": self.portfolio.equity(prices)},
            )
            return self._reject(
                intent,
                f"drawdown total {drawdown.total_pct:.2f} % : KILL SWITCH declenche",
                "drawdown_total",
            )
        if drawdown.weekly_pct >= self.config.max_drawdown_weekly_pct:
            self._pause(
                f"drawdown hebdomadaire {drawdown.weekly_pct:.2f} %",
                hours=self.config.weekly_pause_hours,
                now=stamp,
            )
        elif drawdown.daily_pct >= self.config.max_drawdown_daily_pct:
            self._pause(
                f"drawdown journalier {drawdown.daily_pct:.2f} %",
                hours=self.config.daily_pause_hours,
                now=stamp,
            )

        # 3. Pause en cours.
        if self.pause is not None and self.pause.is_active(stamp):
            return self._reject(
                intent,
                f"trading en pause jusqu'a {self.pause.until.isoformat()} ({self.pause.reason})",
                "pause",
            )

        # 4. Circuit breakers.
        breaker_status = self.breaker.status(intent.asset, stamp)
        if breaker_status.tripped:
            return self._reject(
                intent,
                f"circuit breaker actif : {'; '.join(breaker_status.reasons)}",
                "circuit_breaker",
            )

        # 5. Regime.
        if intent.regime.is_crisis:
            return self._reject(
                intent, "regime de crise : aucune nouvelle position", "regime_crisis"
            )

        # 6. DevilAdvocate : son verdict est contraignant, pas consultatif.
        size_multiplier = 1.0
        if intent.contra_report is not None:
            report = intent.contra_report
            checks["contra_score"] = round(report.contra_score, 3)
            if report.recommendation is ContraRecommendation.ABORT:
                return self._reject(
                    intent,
                    f"devil advocate : ABORT (score {report.contra_score:.2f}) — "
                    f"{'; '.join(report.contra_signals[:3])}",
                    "devil_advocate_abort",
                )
            if report.recommendation is ContraRecommendation.REDUCE_SIZE:
                size_multiplier *= self.settings.devil_advocate.reduce_factor
                reasons.append(
                    f"taille reduite par le devil advocate (score {report.contra_score:.2f})"
                )
        elif self.settings.devil_advocate.enabled:
            # Le module est obligatoire : une intention non auditee est refusee.
            return self._reject(
                intent,
                "aucun rapport du devil advocate : intention non auditee",
                "devil_advocate_missing",
            )

        # 7. Qualite intrinseque du trade.
        quality_rejection = self._check_trade_quality(intent, checks)
        if quality_rejection is not None:
            return self._reject(intent, *quality_rejection)

        # 8. Positions concurrentes et doublons.
        if self.portfolio.has_position(intent.asset):
            return self._reject(
                intent, f"position deja ouverte sur {intent.asset}", "duplicate_position"
            )
        if self.portfolio.open_count >= self.config.max_concurrent_positions:
            return self._reject(
                intent,
                f"{self.portfolio.open_count} positions ouvertes "
                f"(max {self.config.max_concurrent_positions})",
                "max_positions",
            )

        # 9. Cooldown sur le meme actif.
        last_trade = self.portfolio.last_trade_time(intent.asset)
        if last_trade is not None:
            elapsed_min = (stamp - last_trade).total_seconds() / 60.0
            if elapsed_min < self.config.cooldown_same_asset_min:
                return self._reject(
                    intent,
                    f"cooldown actif sur {intent.asset} "
                    f"({elapsed_min:.1f} min < {self.config.cooldown_same_asset_min} min)",
                    "cooldown",
                )

        # 10. Rate limits.
        rate_rejection = self._check_rate_limits(stamp)
        if rate_rejection is not None:
            return self._reject(intent, *rate_rejection)

        # 11. Dimensionnement.
        equity = self.portfolio.equity(prices)
        sizing = self.sizer.size(
            equity=equity,
            entry_price=intent.entry_price,
            stop_loss=intent.stop_loss,
            confidence=intent.confidence,
            regime=intent.regime,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            current_exposure_pct=self.portfolio.exposure_pct(prices),
        )
        reasons.extend(sizing.reasons)
        if not sizing.is_tradable:
            return self._reject(intent, f"taille nulle : {sizing.reasons[-1]}", "sizing_zero")

        size = sizing.size * size_multiplier
        notional = sizing.notional * size_multiplier

        # 12. Verification finale de la limite en dur, apres tous les ajustements.
        max_notional = equity * HARD_MAX_POSITION_PCT / 100.0
        if notional > max_notional + 1e-9:
            notional = max_notional
            size = notional / intent.entry_price
            reasons.append(f"notionnel ramene a la limite en dur ({HARD_MAX_POSITION_PCT} %)")

        decision = RiskDecision.REDUCED if size_multiplier < 1.0 else RiskDecision.APPROVED
        checks["equity"] = round(equity, 2)
        checks["exposure_pct"] = round(self.portfolio.exposure_pct(prices), 3)
        checks["notional"] = round(notional, 4)
        checks["fraction_of_equity"] = round(notional / equity * 100.0, 4) if equity > 0 else 0.0

        verdict = RiskVerdict(
            decision=decision,
            approved_size=float(size),
            approved_notional=float(notional),
            reasons=reasons,
            checks=checks,
            timestamp=stamp,
        )
        log.info(
            "risk_approved",
            asset=intent.asset,
            side=intent.side.value,
            decision=decision.value,
            checks={k: str(v) for k, v in checks.items()},
        )
        return verdict

    def _check_trade_quality(
        self, intent: TradeIntent, checks: dict[str, object]
    ) -> tuple[str, str] | None:
        """Verifie stop loss, distance de stop et ratio risk/reward."""
        if intent.stop_loss <= 0:
            return "stop loss absent ou invalide", "no_stop"

        direction = 1 if intent.side.value == "buy" else -1
        if direction > 0 and intent.stop_loss >= intent.entry_price:
            return "stop loss au-dessus du prix d'entree sur un achat", "invalid_stop"
        if direction < 0 and intent.stop_loss <= intent.entry_price:
            return "stop loss sous le prix d'entree sur une vente", "invalid_stop"

        stop_distance = intent.stop_distance_pct
        checks["stop_distance_pct"] = round(stop_distance, 3)
        if stop_distance > self.config.max_stop_distance_pct:
            return (
                f"stop trop eloigne ({stop_distance:.2f} % > "
                f"{self.config.max_stop_distance_pct:.2f} %)",
                "stop_too_far",
            )

        risk_reward = intent.risk_reward
        checks["risk_reward"] = round(risk_reward, 3) if risk_reward is not None else None
        if risk_reward is None:
            return "aucune cible de profit : ratio risk/reward incalculable", "no_target"
        if risk_reward < self.config.min_risk_reward:
            return (
                f"ratio risk/reward {risk_reward:.2f} < minimum {self.config.min_risk_reward:.2f}",
                "poor_risk_reward",
            )
        return None

    def _check_rate_limits(self, now: datetime) -> tuple[str, str] | None:
        """Verifie les limites de frequence d'ordres."""
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        last_hour = sum(1 for stamp in self._order_times if stamp >= hour_ago)
        last_day = sum(1 for stamp in self._order_times if stamp >= day_ago)

        if last_hour >= self.config.max_orders_per_hour:
            return (
                f"{last_hour} ordres dans l'heure (max {self.config.max_orders_per_hour})",
                "rate_limit_hour",
            )
        if last_day >= self.config.max_orders_per_day:
            return (
                f"{last_day} ordres aujourd'hui (max {self.config.max_orders_per_day})",
                "rate_limit_day",
            )
        return None

    # -------------------------------------------------------------- etat

    def record_order(self, now: datetime | None = None) -> None:
        """Enregistre un ordre effectivement transmis (alimente les rate limits).

        A appeler UNIQUEMENT apres transmission reelle : compter les ordres
        refuses fausserait les limites et bloquerait le systeme pour rien.
        """
        self._order_times.append(to_utc(now or utc_now()))

    def _pause(self, reason: str, hours: int, now: datetime) -> None:
        """Met le trading en pause pour une duree donnee."""
        until = now + timedelta(hours=hours)
        if self.pause is not None and self.pause.is_active(now) and self.pause.until >= until:
            return
        self.pause = TradingPause(reason=reason, until=until, triggered_at=now)
        log.warning("trading_paused", reason=reason, hours=hours, until=until.isoformat())

    def _reject(self, intent: TradeIntent, reason: str, code: str) -> RiskVerdict:
        """Construit un verdict de refus et l'enregistre."""
        self._rejections[code] = self._rejections.get(code, 0) + 1
        log.warning(
            "risk_rejected",
            asset=intent.asset,
            side=intent.side.value,
            code=code,
            reason=reason,
        )
        return RiskVerdict(
            decision=RiskDecision.REJECTED,
            approved_size=0.0,
            approved_notional=0.0,
            reasons=[reason],
            checks={"code": code},
        )

    def should_close_all(self, regime_is_crisis: bool) -> tuple[bool, str]:
        """Indique s'il faut liquider toutes les positions, et pourquoi."""
        if self.kill_switch.is_triggered():
            return True, f"kill switch : {self.kill_switch.reason()}"
        if regime_is_crisis and self.settings.risk.crisis_regime.close_existing:
            return True, "regime de crise : fermeture des positions"
        return False, ""

    def state(self, now: datetime | None = None) -> RiskState:
        """Etat consolide du risque, pour le monitoring."""
        stamp = to_utc(now or utc_now())
        hour_ago = stamp - timedelta(hours=1)
        day_ago = stamp - timedelta(days=1)
        active_pause = self.pause if self.pause and self.pause.is_active(stamp) else None
        return RiskState(
            kill_switch_active=self.kill_switch.is_triggered(),
            paused_until=active_pause.until if active_pause else None,
            pause_reason=active_pause.reason if active_pause else "",
            orders_last_hour=sum(1 for s in self._order_times if s >= hour_ago),
            orders_today=sum(1 for s in self._order_times if s >= day_ago),
            breaker_trips=len(self.breaker.status(now=stamp).trips),
            rejections=dict(self._rejections),
        )

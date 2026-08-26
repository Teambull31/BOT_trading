"""Detection du declin des strategies.

Une strategie ne meurt pas d'un coup : elle s'erode. Le but est de le voir AVANT
que ca coute cher, pas apres. Cinq signaux sont surveilles :

1. Sharpe court terme effondre par rapport au long terme (Sharpe_7j < 0.5 x Sharpe_30j) ;
2. hit rate sous 45 % sur 14 jours ;
3. profit factor sous 1.0 sur 14 jours ;
4. series de pertes consecutives ;
5. alpha qui derive vers zero — teste par un ADF sur le P&L cumule : une serie
   de P&L qui devient stationnaire autour de zero signale une strategie dont
   l'edge a disparu.

Etats : HEALTHY -> DEGRADING -> DEAD -> (retraining) -> ZOMBIE -> HEALTHY

Une strategie DEAD n'est JAMAIS supprimee : elle continue de tourner en shadow
mode, ses signaux sont logges, et si le regime redevient favorable elle repasse
en test avant reactivation. Les regimes changent ; ce qui ne marche plus
aujourd'hui peut redevenir pertinent demain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from trader.adaptation.evaluator import EvaluationSnapshot, StrategyEvaluator
from trader.config import DecayDetectionConfig
from trader.logging_setup import get_logger
from trader.models import StrategyHealth
from trader.utils.time_utils import utc_now

log = get_logger(__name__)


@dataclass(slots=True)
class DecayVerdict:
    """Diagnostic de sante d'une strategie."""

    strategy: str
    health: StrategyHealth
    signals: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    weight_multiplier: float = 1.0
    needs_retraining: bool = False
    timestamp: datetime = field(default_factory=utc_now)

    @property
    def is_healthy(self) -> bool:
        """Vrai si la strategie peut continuer a peser normalement."""
        return self.health is StrategyHealth.HEALTHY

    def to_dict(self) -> dict[str, object]:
        """Representation serialisable."""
        return {
            "strategy": self.strategy,
            "health": self.health.value,
            "signals": list(self.signals),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "weight_multiplier": self.weight_multiplier,
            "needs_retraining": self.needs_retraining,
            "timestamp": self.timestamp.isoformat(),
        }


WEIGHT_MULTIPLIER: dict[StrategyHealth, float] = {
    StrategyHealth.HEALTHY: 1.0,
    StrategyHealth.DEGRADING: 0.5,
    StrategyHealth.ZOMBIE: 0.0,
    StrategyHealth.DEAD: 0.0,
}
"""Poids accorde a une strategie selon son etat de sante."""


@dataclass(slots=True)
class DeadRecord:
    """Suivi d'une strategie declaree morte."""

    since: datetime
    retrained_at: datetime | None = None
    shadow_started: datetime | None = None


class StrategyDecayDetector:
    """Surveille l'erosion de l'edge de chaque strategie."""

    def __init__(
        self,
        config: DecayDetectionConfig,
        evaluator: StrategyEvaluator,
        shadow_mode_days: int = 14,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.shadow_mode_days = shadow_mode_days
        self.dead_since: dict[str, DeadRecord] = {}
        self.last_check: datetime | None = None

    def needs_check(self, now: datetime | None = None) -> bool:
        """Vrai si l'intervalle de verification est ecoule."""
        if self.last_check is None:
            return True
        reference = now or utc_now()
        return reference - self.last_check >= timedelta(hours=self.config.check_interval_hours)

    def check(
        self,
        strategy: str,
        now: datetime | None = None,
        trades: pd.DataFrame | None = None,
    ) -> DecayVerdict:
        """Diagnostique une strategie."""
        reference = now or utc_now()
        snapshot = self.evaluator.evaluate(strategy, reference, trades)
        signals: list[str] = []

        short = snapshot.window(7)
        medium = snapshot.window(14)
        long = snapshot.window(30)

        metrics = {
            "sharpe_7d": short.sharpe,
            "sharpe_30d": long.sharpe,
            "hit_rate_14d": medium.hit_rate,
            "profit_factor_14d": medium.profit_factor,
            "consecutive_losses": float(medium.consecutive_losses),
            "trades_30d": float(long.trades),
        }

        if long.trades < self.config.min_trades_for_verdict:
            # Pas assez de trades pour conclure : on ne condamne pas sur du bruit.
            # Mais une strategie deja morte ou en test ne recupere pas son poids
            # pour la seule raison qu'elle a cesse de trader.
            health = self._resurrect_or_keep(strategy, StrategyHealth.HEALTHY, reference)
            return DecayVerdict(
                strategy=strategy,
                health=health,
                signals=[
                    f"echantillon insuffisant ({long.trades} trades sur 30 j) : "
                    "aucun verdict de decay rendu"
                ],
                metrics=metrics,
                weight_multiplier=WEIGHT_MULTIPLIER[health],
            )

        # 1. Effondrement du Sharpe court terme.
        # Condition indispensable : que la fenetre courte soit elle-meme
        # significative. Sinon une strategie qui n'a simplement pas trade depuis
        # une semaine (Sharpe 7 j neutralise a 0) serait declaree en declin,
        # alors qu'elle n'a rien fait de mal.
        if (
            short.is_significant
            and long.sharpe > 0
            and short.sharpe < long.sharpe * self.config.sharpe_decay_ratio
        ):
            signals.append(
                f"Sharpe 7 j ({short.sharpe:.2f}) sous "
                f"{self.config.sharpe_decay_ratio:.0%} du Sharpe 30 j ({long.sharpe:.2f})"
            )

        # 2. Hit rate degrade.
        if medium.is_significant and medium.hit_rate < self.config.min_hit_rate_14d:
            signals.append(
                f"hit rate 14 j de {medium.hit_rate:.1%} sous le seuil "
                f"{self.config.min_hit_rate_14d:.0%}"
            )

        # 3. Profit factor.
        profit_factor_dead = (
            medium.is_significant and medium.profit_factor < self.config.min_profit_factor_14d
        )
        if profit_factor_dead:
            signals.append(
                f"profit factor 14 j de {medium.profit_factor:.2f} sous "
                f"{self.config.min_profit_factor_14d:.2f}"
            )

        # 4. Pertes consecutives.
        losses = medium.consecutive_losses
        losses_dead = losses > self.config.max_consecutive_losses
        if losses_dead:
            signals.append(
                f"{losses} pertes consecutives (seuil de desactivation "
                f"{self.config.max_consecutive_losses})"
            )
        elif losses > self.config.max_consecutive_losses_alert:
            signals.append(
                f"{losses} pertes consecutives (seuil d'alerte "
                f"{self.config.max_consecutive_losses_alert})"
            )

        # 5. Disparition de l'alpha (test de stationnarite).
        alpha_signal, alpha_stat = self._alpha_is_dying(strategy, reference, trades)
        if alpha_signal:
            signals.append(alpha_signal)
        metrics["adf_pvalue"] = alpha_stat

        health = self._classify(
            signals=signals,
            profit_factor_dead=profit_factor_dead,
            losses_dead=losses_dead,
            sharpe_30d=long.sharpe,
        )
        health = self._resurrect_or_keep(strategy, health, reference)
        multiplier = WEIGHT_MULTIPLIER[health]

        verdict = DecayVerdict(
            strategy=strategy,
            health=health,
            signals=signals,
            metrics=metrics,
            weight_multiplier=multiplier,
            needs_retraining=health in (StrategyHealth.DEGRADING, StrategyHealth.DEAD),
            timestamp=reference,
        )
        if health is not StrategyHealth.HEALTHY:
            log.warning("strategy_decay_detected", **verdict.to_dict())
        return verdict

    def _classify(
        self,
        signals: list[str],
        profit_factor_dead: bool,
        losses_dead: bool,
        sharpe_30d: float,
    ) -> StrategyHealth:
        """Traduit les signaux en etat de sante.

        Deux criteres tuent seuls (profit factor sous 1 = la strategie perd de
        l'argent ; trop de pertes consecutives = quelque chose est casse). Les
        autres degradent progressivement.
        """
        if profit_factor_dead or losses_dead:
            return StrategyHealth.DEAD
        if sharpe_30d < 0 and signals:
            return StrategyHealth.DEAD
        if len(signals) >= 2:
            return StrategyHealth.DEAD
        if signals:
            return StrategyHealth.DEGRADING
        return StrategyHealth.HEALTHY

    def _alpha_is_dying(
        self, strategy: str, now: datetime, trades: pd.DataFrame | None
    ) -> tuple[str | None, float]:
        """Teste si le P&L cumule perd sa derive (alpha qui tend vers zero).

        Un P&L cumule sain a une tendance : il derive vers le haut, donc la serie
        est NON stationnaire. Si le test ADF conclut a la stationnarite, c'est que
        la serie oscille autour d'une constante : il n'y a plus de derive, donc
        plus d'edge.
        """
        if trades is None:
            trades = self.evaluator.store.load_trades(
                strategy=strategy, since=now - timedelta(days=60), mode=self.evaluator.mode
            )
        if trades is None or len(trades) < 20:
            return None, float("nan")

        cumulative = trades["pnl"].cumsum().to_numpy(dtype=float)
        if np.allclose(cumulative, cumulative[0]):
            return None, float("nan")

        try:
            from statsmodels.tsa.stattools import adfuller

            pvalue = float(adfuller(cumulative, autolag="AIC")[1])
        except (ImportError, ValueError) as exc:
            log.debug("adf_unavailable", strategy=strategy, error=str(exc))
            return None, float("nan")

        # p < 0.05 : on rejette la racine unitaire, la serie est stationnaire.
        if pvalue < 0.05 and cumulative[-1] <= cumulative[len(cumulative) // 2]:
            return (
                f"P&L cumule devenu stationnaire (ADF p={pvalue:.3f}) : l'alpha a disparu"
            ), pvalue
        return None, pvalue

    def _resurrect_or_keep(
        self, strategy: str, health: StrategyHealth, now: datetime
    ) -> StrategyHealth:
        """Gere le cycle DEAD -> ZOMBIE -> HEALTHY.

        Une strategie morte depuis assez longtemps et qui redevient saine ne
        reprend pas directement du capital : elle passe d'abord en ZOMBIE, une
        periode de test en shadow mode. On ne refait pas confiance sur un
        rebond de quelques jours.
        """
        record = self.dead_since.get(strategy)

        if health is StrategyHealth.DEAD:
            if record is None:
                self.dead_since[strategy] = DeadRecord(since=now)
                log.warning("strategy_declared_dead", strategy=strategy)
            return StrategyHealth.DEAD

        if record is None:
            return health

        dead_days = (now - record.since).total_seconds() / 86400.0
        if record.shadow_started is None:
            if dead_days < self.config.dead_period_days:
                # Trop tot : un redressement passager ne rachete pas une strategie.
                return StrategyHealth.DEAD
            record.shadow_started = now
            log.info("strategy_enters_zombie_test", strategy=strategy, dead_days=dead_days)
            return StrategyHealth.ZOMBIE

        shadow_days = (now - record.shadow_started).total_seconds() / 86400.0
        if shadow_days < self.shadow_mode_days:
            return StrategyHealth.ZOMBIE

        del self.dead_since[strategy]
        log.info(
            "strategy_resurrected",
            strategy=strategy,
            shadow_days=round(shadow_days, 1),
            health=health.value,
        )
        return health

    def check_all(
        self, strategies: list[str], now: datetime | None = None
    ) -> dict[str, DecayVerdict]:
        """Diagnostique toutes les strategies et memorise l'heure du controle."""
        reference = now or utc_now()
        verdicts = {name: self.check(name, reference) for name in strategies}
        self.last_check = reference
        return verdicts

    def health_provider(self, strategy: str) -> StrategyHealth:
        """Callback pour le DevilAdvocate : etat de sante connu d'une strategie."""
        return self.check(strategy).health

    def snapshot(self) -> dict[str, EvaluationSnapshot]:
        """Dernieres evaluations connues."""
        return self.evaluator.last_evaluations()

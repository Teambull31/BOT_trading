"""Orchestrateur : la boucle principale du systeme.

Ordre d'un cycle, strictement respecte :

    kill switch -> donnees -> features -> regime -> sorties (stops/cibles/crise)
      -> ensemble -> devil advocate -> risk manager -> execution -> persistence

Deux invariants structurent tout le reste :

1. Les SORTIES sont traitees AVANT les entrees. Proteger le capital deja engage
   passe avant l'envie d'en engager davantage.
2. Aucun ordre ne part sans un RiskVerdict approuve, et seul le risk manager en
   emet. Le chemin d'execution ne peut pas etre court-circuite.

La boucle ne laisse jamais une exception d'un actif tuer le cycle des autres :
une erreur sur ETH ne doit pas empecher de fermer une position sur BTC.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from trader.adaptation.decay_detector import StrategyDecayDetector
from trader.adaptation.devil_advocate import DevilAdvocate
from trader.adaptation.evaluator import StrategyEvaluator
from trader.adaptation.retrainer import WalkForwardRetrainer
from trader.config import Mode, Settings
from trader.data.features import FeatureBuilder
from trader.data.ingester import DataIngester, PermanentError, TransientError
from trader.data.snapshot import build_snapshot
from trader.data.store import DataStore
from trader.execution.executor import OrderExecutor
from trader.logging_setup import get_logger
from trader.models import (
    ContraRecommendation,
    EnsembleDecision,
    MarketSnapshot,
    OrderSide,
    RegimeState,
    TradeIntent,
)
from trader.portfolio import Portfolio, position_from_fill
from trader.regime.detector import RegimeDetector
from trader.risk.kill_switch import KillSwitch
from trader.risk.manager import RiskManager
from trader.strategy.ensemble import StrategyEnsemble
from trader.utils.time_utils import utc_now

log = get_logger(__name__)


@dataclass(slots=True)
class CycleReport:
    """Compte rendu d'un cycle de trading."""

    timestamp: datetime
    regime_by_asset: dict[str, str] = field(default_factory=dict)
    decisions: dict[str, str] = field(default_factory=dict)
    orders_sent: int = 0
    positions_closed: int = 0
    errors: list[str] = field(default_factory=list)
    equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Representation serialisable du cycle."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "regimes": dict(self.regime_by_asset),
            "decisions": dict(self.decisions),
            "orders_sent": self.orders_sent,
            "positions_closed": self.positions_closed,
            "errors": list(self.errors),
            "equity": round(self.equity, 4),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }


class TradingOrchestrator:
    """Assemble tous les modules et execute la boucle de trading."""

    def __init__(
        self,
        settings: Settings,
        store: DataStore,
        ingester: DataIngester,
        ensemble: StrategyEnsemble,
        portfolio: Portfolio,
        risk_manager: RiskManager,
        executor: OrderExecutor,
        devil_advocate: DevilAdvocate | None = None,
        detector: RegimeDetector | None = None,
        feature_builder: FeatureBuilder | None = None,
        evaluator: StrategyEvaluator | None = None,
        decay_detector: StrategyDecayDetector | None = None,
        retrainer: WalkForwardRetrainer | None = None,
        alerter: Any | None = None,
        metrics: Any | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.ingester = ingester
        self.ensemble = ensemble
        self.portfolio = portfolio
        self.risk = risk_manager
        self.executor = executor
        self.evaluator = evaluator
        self.decay = decay_detector
        self.retrainer = retrainer
        self.devil_advocate = devil_advocate or DevilAdvocate(
            settings.devil_advocate,
            health_provider=decay_detector.health_provider if decay_detector else None,
        )
        # L'ensemble pondere selon la performance mesuree, pas selon une
        # confiance declaree : on branche l'evaluateur comme source de verite.
        if evaluator is not None and self.ensemble.metrics_provider is None:
            self.ensemble.metrics_provider = evaluator.metrics_provider
        self.timeframe = settings.data.primary_timeframe
        self.detector = detector or RegimeDetector(settings.regime, self.timeframe)
        self.features = feature_builder or FeatureBuilder(timeframe=self.timeframe)
        self.alerter = alerter
        self.metrics = metrics
        self.kill_switch: KillSwitch = risk_manager.kill_switch
        self._running = False
        self._cycles = 0
        self._market_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        self._last_retraining: datetime | None = None

    @property
    def assets(self) -> list[str]:
        """Univers d'actifs suivis."""
        return list(self.settings.universe.assets)

    # ------------------------------------------------------------- cycle

    async def run_cycle(self, now: datetime | None = None) -> CycleReport:
        """Execute un cycle complet sur tous les actifs."""
        stamp = now or utc_now()
        report = CycleReport(timestamp=stamp)
        self._cycles += 1

        if self.kill_switch.is_triggered():
            report.halted = True
            report.halt_reason = self.kill_switch.reason()
            log.critical("cycle_halted_by_kill_switch", reason=report.halt_reason)
            await self._liquidate_all(f"kill switch : {report.halt_reason}", report)
            return report

        prices: dict[str, float] = {}
        for asset in self.assets:
            try:
                await self._process_asset(asset, report, prices, stamp)
            except (TransientError, PermanentError) as exc:
                message = f"{asset}: donnees indisponibles ({exc})"
                report.errors.append(message)
                log.error("asset_data_error", asset=asset, error=str(exc))
            except Exception as exc:  # noqa: BLE001 - un actif ne tue pas le cycle
                message = f"{asset}: {type(exc).__name__}: {exc}"
                report.errors.append(message)
                log.error(
                    "asset_cycle_error", asset=asset, error=str(exc), error_type=type(exc).__name__
                )

        await self._run_adaptation(report, stamp)
        report.equity = self.portfolio.mark_to_market(prices, stamp)
        self._persist_cycle(report, prices)
        self.kill_switch.beat(stamp, {"cycle": self._cycles, "equity": round(report.equity, 2)})
        return report

    async def _process_asset(
        self,
        asset: str,
        report: CycleReport,
        prices: dict[str, float],
        now: datetime,
    ) -> None:
        """Traite un actif : donnees, regime, sorties, puis eventuellement entree."""
        ohlcv, features = await self._market_data(asset)
        if ohlcv.empty or features.empty:
            report.errors.append(f"{asset}: pas assez de donnees")
            return

        snapshot = await self._snapshot(asset, ohlcv, features)
        prices[asset] = snapshot.last_price

        # Circuit breakers sur les conditions de marche.
        self.risk.breaker.check_spread(asset, snapshot.spread_pct, now)
        self.risk.breaker.check_price_move(asset, snapshot.last_price, now)
        for exchange, latency in self.ingester.latencies.items():
            self.risk.breaker.check_latency(exchange, latency, now)

        if self.detector.needs_retrain(now):
            self.detector.fit(features, now=now)
        regime = self.detector.detect(features, ohlcv, now=now)
        report.regime_by_asset[asset] = regime.regime.value
        self.store.save_regime(regime, asset=asset)

        # 1. Sorties d'abord : proteger avant d'engager.
        closed = await self._handle_exits(asset, snapshot, regime)
        report.positions_closed += closed

        # 2. Entree eventuelle.
        decision = self.ensemble.decide(snapshot, regime)
        if decision.blocked_reason:
            report.decisions[asset] = f"no_trade: {decision.blocked_reason}"
            self.store.save_decision(asset, "ensemble", "blocked", decision.to_dict())
            return
        if not decision.is_actionable:
            report.decisions[asset] = "neutral"
            return

        contra = self.devil_advocate.review(decision, snapshot, regime)
        self.store.save_decision(
            asset, "devil_advocate", contra.recommendation.value, contra.to_dict()
        )
        if contra.recommendation is ContraRecommendation.ABORT:
            report.decisions[asset] = f"aborted: {contra.contra_signals[:2]}"
            return

        intent = self._build_intent(decision, regime, contra, snapshot)
        verdict = self.risk.evaluate(intent, prices, now=now)
        self.store.save_decision(asset, "risk", verdict.decision.value, verdict.to_dict())
        if not verdict.is_approved:
            report.decisions[asset] = f"rejected: {verdict.reasons[0]}"
            return

        result = await self.executor.execute(intent, verdict, snapshot)
        if not result.order.filled_size:
            report.decisions[asset] = f"unfilled: {result.error or 'aucun fill'}"
            self.risk.breaker.record_execution_error(now)
            return

        self.risk.record_order(now)
        self.portfolio.open_position(
            position_from_fill(
                asset=asset,
                side=intent.side,
                size=result.order.filled_size,
                fill_price=result.order.average_price,
                stop_loss=intent.stop_loss,
                target_price=intent.target_price,
                fees=result.order.fees,
                now=now,
                metadata={
                    "weights": decision.weights,
                    "regime": regime.regime.value,
                    "contra_score": contra.contra_score,
                },
            )
        )
        report.orders_sent += 1
        report.decisions[asset] = f"opened {intent.side.value} {result.order.filled_size:.6f}"
        await self._alert(
            "INFO",
            f"Position ouverte {asset} {intent.side.value} "
            f"{result.order.filled_size:.6f} @ {result.order.average_price:.2f}",
        )

    async def _run_adaptation(self, report: CycleReport, now: datetime) -> None:
        """Detecte le decay des strategies et declenche les retrainings.

        L'adaptation tourne APRES le trading du cycle : modifier les poids au
        milieu d'un cycle rendrait les decisions de ce cycle inexplicables a
        posteriori, ce qui casserait l'audit trail.
        """
        if self.decay is None or not self.decay.needs_check(now):
            return

        names = list(self.ensemble.records)
        try:
            verdicts = self.decay.check_all(names, now)
        except Exception as exc:  # noqa: BLE001 - l'adaptation ne tue pas le trading
            report.errors.append(f"decay: {exc}")
            log.error("decay_check_failed", error=str(exc))
            return

        degraded: list[str] = []
        for name, verdict in verdicts.items():
            self.ensemble.set_health(name, verdict.health)
            self.store.save_strategy_metrics(name, verdict.to_dict() | verdict.metrics)
            if verdict.health.value != "healthy":
                degraded.append(f"{name}={verdict.health.value}")
            if verdict.needs_retraining:
                await self._retrain(name, report, now)

        if degraded:
            await self._alert("WARNING", f"Sante des strategies degradee : {', '.join(degraded)}")

    async def _retrain(self, strategy_name: str, report: CycleReport, now: datetime) -> None:
        """Relance un retraining walk-forward pour une strategie en declin."""
        if self.retrainer is None:
            return
        if (
            self._last_retraining is not None
            and (now - self._last_retraining).total_seconds() < 3600
        ):
            # Un retraining est couteux et ne s'improvise pas a chaque cycle.
            return

        record = self.ensemble.records.get(strategy_name)
        cached = next(iter(self._market_cache.values()), None)
        if record is None or cached is None:
            return
        ohlcv, features = cached
        try:
            result = self.retrainer.retrain_strategy(
                record.strategy, ohlcv, features, self._retraining_score, now=now
            )
        except Exception as exc:  # noqa: BLE001 - un retraining rate n'arrete pas le systeme
            report.errors.append(f"retraining {strategy_name}: {exc}")
            log.error("retraining_failed", strategy=strategy_name, error=str(exc))
            return

        self.store.save_event(
            "INFO" if result.accepted else "WARNING",
            "retraining",
            f"{strategy_name}: {result.reason}",
            result.to_dict(),
        )
        self._last_retraining = now

    def _retraining_score(
        self, strategy: Any, ohlcv: pd.DataFrame, features: pd.DataFrame
    ) -> float:
        """Score d'un jeu de parametres : qualite des signaux produits.

        On n'execute pas un backtest complet par candidat (trop couteux en
        boucle) : on mesure la coherence des signaux avec le rendement futur
        de la barre suivante, ce qui est un proxy direct de l'edge.
        """
        from trader.data.snapshot import build_snapshot
        from trader.models import Regime, RegimeState

        if len(ohlcv) < 60 or features.empty:
            return 0.0
        neutral_regime = RegimeState(
            regime=Regime(self.detector.last_state.regime.value)
            if self.detector.last_state
            else Regime.RANGE_BOUND,
            confidence=0.7,
            agreement_score=1.0,
            transition_probability=0.2,
        )
        future_returns = ohlcv["close"].pct_change().shift(-1)
        score = 0.0
        count = 0
        for position in range(50, len(ohlcv) - 1, max(1, len(ohlcv) // 40)):
            data = build_snapshot("backtest", ohlcv, features, position=position)
            output = strategy.generate_signal(data, neutral_regime)
            if not output.is_actionable:
                continue
            realized = float(future_returns.iloc[position])
            if not pd.notna(realized):
                continue
            score += output.signal.direction * realized * output.confidence
            count += 1
        if count < 3:
            return 0.0
        return score / count * 10_000.0

    async def _handle_exits(self, asset: str, snapshot: MarketSnapshot, regime: RegimeState) -> int:
        """Ferme les positions dont le stop, la cible ou le regime l'exige."""
        position = self.portfolio.get_position(asset)
        if position is None:
            return 0

        price = snapshot.last_price
        reason = ""
        if position.should_stop_out(price):
            reason = "stop_loss"
        elif position.should_take_profit(price):
            reason = "take_profit"
        else:
            close_all, motive = self.risk.should_close_all(regime.is_crisis)
            if close_all:
                reason = motive

        if not reason:
            return 0

        exit_side = OrderSide.SELL if position.side is OrderSide.BUY else OrderSide.BUY
        result = await self.executor.close_position(
            asset, exit_side, position.size, snapshot, reason
        )
        if not result.order.filled_size:
            log.error("exit_failed", asset=asset, reason=reason, error=result.error)
            self.risk.breaker.record_execution_error()
            await self._alert(
                "CRITICAL", f"ECHEC de fermeture sur {asset} ({reason}) : {result.error}"
            )
            return 0

        trade = self.portfolio.close_position(
            asset,
            exit_price=result.order.average_price,
            fees=result.order.fees,
            reason=reason,
            regime=regime.regime.value,
        )
        self.store.save_trade(trade, mode=self.settings.general.mode.value)
        self.risk.record_order()
        level = "WARNING" if trade.pnl < 0 else "INFO"
        await self._alert(
            level,
            f"Position fermee {asset} ({reason}) : P&L {trade.pnl:+.2f} "
            f"({trade.return_pct:+.2f} %)",
        )
        return 1

    def _build_intent(
        self,
        decision: EnsembleDecision,
        regime: RegimeState,
        contra: Any,
        snapshot: MarketSnapshot,
    ) -> TradeIntent:
        """Assemble l'intention de trade soumise au risk manager."""
        side = OrderSide.BUY if decision.signal.direction > 0 else OrderSide.SELL
        return TradeIntent(
            asset=decision.asset,
            side=side,
            entry_price=snapshot.last_price,
            stop_loss=float(decision.stop_loss or 0.0),
            target_price=decision.target_price,
            confidence=decision.confidence,
            regime=regime,
            decision=decision,
            contra_report=contra,
        )

    async def _liquidate_all(self, reason: str, report: CycleReport) -> None:
        """Ferme toutes les positions ouvertes (kill switch, crise immediate)."""
        for asset in list(self.portfolio.positions):
            position = self.portfolio.get_position(asset)
            if position is None:
                continue
            try:
                ohlcv, features = await self._market_data(asset)
                snapshot = await self._snapshot(asset, ohlcv, features)
                exit_side = OrderSide.SELL if position.side is OrderSide.BUY else OrderSide.BUY
                result = await self.executor.close_position(
                    asset, exit_side, position.size, snapshot, reason
                )
                if result.order.filled_size:
                    trade = self.portfolio.close_position(
                        asset,
                        exit_price=result.order.average_price,
                        fees=result.order.fees,
                        reason=reason,
                    )
                    self.store.save_trade(trade, mode=self.settings.general.mode.value)
                    report.positions_closed += 1
            except Exception as exc:  # noqa: BLE001 - on tente toutes les positions
                report.errors.append(f"liquidation {asset}: {exc}")
                log.critical("liquidation_failed", asset=asset, error=str(exc))
        await self._alert("CRITICAL", f"Liquidation totale : {reason}")

    # -------------------------------------------------------------- donnees

    async def _market_data(self, asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Recupere les bougies a jour et recalcule les features."""
        result = await self.ingester.update_incremental(asset, self.timeframe)
        ohlcv = self.store.load_ohlcv(asset, self.timeframe)
        if ohlcv.empty:
            ohlcv = result.frame
        if ohlcv.empty:
            return ohlcv, pd.DataFrame()
        features = self.features.build(ohlcv)
        self._market_cache[asset] = (ohlcv, features)
        return ohlcv, features

    async def _snapshot(
        self, asset: str, ohlcv: pd.DataFrame, features: pd.DataFrame
    ) -> MarketSnapshot:
        """Construit le snapshot, enrichi du carnet d'ordres si disponible."""
        book: dict[str, Any] = {}
        with contextlib.suppress(TransientError, PermanentError):
            book = await self.ingester.fetch_order_book(asset)
        return build_snapshot(
            asset,
            ohlcv,
            features,
            bid=book.get("best_bid"),
            ask=book.get("best_ask"),
            order_book=book,
        )

    def _persist_cycle(self, report: CycleReport, prices: dict[str, float]) -> None:
        """Enregistre l'etat du portefeuille et les metriques du cycle."""
        try:
            self.store.save_equity(
                equity=report.equity,
                cash=self.portfolio.cash,
                exposure_pct=self.portfolio.exposure_pct(prices),
                open_positions=self.portfolio.open_count,
                mode=self.settings.general.mode.value,
            )
            for name, state in self.ensemble.snapshot().items():
                self.store.save_strategy_metrics(name, {**state, **state.get("metrics", {})})
        except Exception as exc:  # noqa: BLE001 - la persistence ne doit pas tuer la boucle
            log.error("cycle_persist_failed", error=str(exc))

        if self.metrics is not None:
            with contextlib.suppress(Exception):
                self.metrics.update_from_cycle(self, report, prices)

    async def _alert(self, level: str, message: str) -> None:
        """Envoie une alerte si un alerter est branche."""
        if self.alerter is None:
            return
        try:
            await self.alerter.send(level, message)
        except Exception as exc:  # noqa: BLE001 - une alerte ratee n'arrete pas le trading
            log.error("alert_failed", level=level, error=str(exc))

    # ---------------------------------------------------------------- boucle

    async def run_forever(self, max_cycles: int | None = None) -> None:
        """Boucle principale : cycles espaces, heartbeat, arret propre."""
        self._running = True
        interval = self.settings.general.loop_interval_sec
        log.info(
            "orchestrator_started",
            mode=self.settings.general.mode.value,
            assets=self.assets,
            interval_sec=interval,
            strategies=list(self.ensemble.records),
        )
        if self.settings.general.mode is Mode.LIVE:
            log.critical("mode_live_actif", capital=self.settings.general.initial_capital)

        cycles = 0
        try:
            while self._running:
                report = await self.run_cycle()
                cycles += 1
                if report.halted:
                    log.critical("orchestrator_halted", reason=report.halt_reason)
                    break
                if max_cycles is not None and cycles >= max_cycles:
                    break
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            log.info("orchestrator_cancelled")
            raise
        finally:
            self._running = False
            await self.shutdown()

    def stop(self) -> None:
        """Demande l'arret de la boucle apres le cycle courant."""
        self._running = False
        log.info("orchestrator_stop_requested")

    async def shutdown(self) -> None:
        """Arret propre : ferme les connexions, laisse les positions intactes.

        On ne liquide PAS a l'arret : un redemarrage planifie ne doit pas
        declencher de ventes. Seul le kill switch liquide.
        """
        log.info(
            "orchestrator_shutdown",
            open_positions=self.portfolio.open_count,
            equity=round(self.portfolio.equity(), 2),
        )
        with contextlib.suppress(Exception):
            await self.ingester.close()
        self.kill_switch.stop_http_server()

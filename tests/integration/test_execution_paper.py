"""Tests d'integration end-to-end en mode paper.

Le systeme complet tourne : donnees -> regime -> ensemble -> devil advocate ->
risk manager -> execution paper -> portefeuille -> persistence. Aucun reseau :
un faux exchange sert des bougies deterministes.

Ces tests verifient les proprietes qui doivent tenir sur le systeme ASSEMBLE,
notamment celle qui compte le plus : aucun ordre ne peut partir sans passer par
le risk manager.
"""

from __future__ import annotations

import warnings
from datetime import timedelta

import pytest

from tests.conftest import make_ohlcv
from tests.integration.test_data_pipeline import FakeExchange, to_ccxt
from tests.unit.test_ensemble import ScriptedStrategy
from trader.adaptation.devil_advocate import DevilAdvocate
from trader.config import load_settings
from trader.data.features import FeatureBuilder
from trader.data.ingester import DataIngester
from trader.data.snapshot import build_snapshot
from trader.data.store import DataStore
from trader.execution.executor import OrderExecutor, RiskBypassError
from trader.execution.paper import PaperBroker, build_order
from trader.models import (
    ContraRecommendation,
    ContraReport,
    EnsembleDecision,
    OrderSide,
    OrderType,
    Regime,
    RegimeState,
    RiskDecision,
    RiskVerdict,
    Signal,
    TradeIntent,
)
from trader.orchestrator import TradingOrchestrator
from trader.portfolio import Portfolio
from trader.risk.kill_switch import KillSwitch
from trader.risk.manager import RiskManager
from trader.strategy.ensemble import StrategyEnsemble
from trader.utils.time_utils import utc_now

warnings.filterwarnings("ignore")

ALL_REGIMES = [
    "bull_low_vol",
    "bull_high_vol",
    "bear_low_vol",
    "bear_high_vol",
    "range_bound",
    "uncertain",
]
CAPITAL = 10_000.0


@pytest.fixture
def paper_settings(tmp_path):
    return load_settings(
        "config/default.toml",
        "config/paper.toml",
        overrides={
            "general": {"initial_capital": CAPITAL, "loop_interval_sec": 1},
            "data": {"db_url": f"sqlite:///{tmp_path}/paper.db"},
            "universe": {"assets": ["ETH/USDT"]},
            "kill_switch": {"sentinel_path": str(tmp_path / "kill"), "http_enabled": False},
            "execution": {"simulate_latency": False},
            "monitoring": {"prometheus_enabled": False},
        },
    )


@pytest.fixture
def candles():
    start = utc_now() - timedelta(hours=800)
    return to_ccxt(make_ohlcv(n=800, drift=0.002, vol=0.008, seed=61, start=start))


def build_system(settings, candles, strategies=None, seed_data=True):
    """Assemble un systeme complet en paper trading, sans reseau."""
    store = DataStore(settings.data.db_url)
    exchange = FakeExchange(candles)
    ingester = DataIngester(settings, store=store, clients={"binance": exchange})
    portfolio = Portfolio(settings.general.initial_capital)
    kill_switch = KillSwitch(settings.kill_switch)
    risk = RiskManager(settings, portfolio, kill_switch=kill_switch)
    executor = OrderExecutor(settings, broker=PaperBroker(settings.execution, seed=1), store=store)
    ensemble = StrategyEnsemble(
        strategies
        or [
            ScriptedStrategy("a", Signal.BUY, regimes=ALL_REGIMES, stop_pct=0.03, target_pct=0.08),
            ScriptedStrategy("b", Signal.BUY, regimes=ALL_REGIMES, stop_pct=0.02, target_pct=0.07),
        ],
        settings.ensemble,
    )
    orchestrator = TradingOrchestrator(
        settings=settings,
        store=store,
        ingester=ingester,
        ensemble=ensemble,
        portfolio=portfolio,
        risk_manager=risk,
        executor=executor,
        devil_advocate=DevilAdvocate(settings.devil_advocate),
    )
    if seed_data:
        frame = build_frame(candles)
        store.save_ohlcv("binance", "ETH/USDT", settings.data.primary_timeframe, frame)
    return orchestrator, store, portfolio, risk


def build_frame(candles):
    import pandas as pd

    frame = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame.set_index("timestamp")


def snapshot_from(candles, position: int = -1):
    frame = build_frame(candles)
    features = FeatureBuilder(timeframe="1h").build(frame)
    return build_snapshot("ETH/USDT", frame, features, position=position)


# ---------------------------------------------------------------- cycle complet


async def test_full_cycle_opens_position(paper_settings, candles):
    orchestrator, store, portfolio, _ = build_system(paper_settings, candles)
    report = await orchestrator.run_cycle()

    assert not report.errors, report.errors
    assert report.regime_by_asset
    assert report.equity > 0
    if report.orders_sent:
        assert portfolio.open_count == 1
        position = portfolio.get_position("ETH/USDT")
        assert position.stop_loss > 0
        # La limite en dur de 2 % s'applique au systeme assemble.
        assert position.notional(position.entry_price) <= CAPITAL * 0.02 * 1.05


async def test_audit_trail_is_written(paper_settings, candles):
    """Chaque decision doit rester tracable a posteriori."""
    orchestrator, store, _, _ = build_system(paper_settings, candles)
    await orchestrator.run_cycle()
    decisions = store.load_decisions()
    assert decisions
    stages = {entry["stage"] for entry in decisions}
    assert stages & {"ensemble", "devil_advocate", "risk"}
    assert not store.load_equity().empty


async def test_kill_switch_halts_and_liquidates(paper_settings, candles):
    orchestrator, store, portfolio, risk = build_system(paper_settings, candles)
    await orchestrator.run_cycle()
    had_position = portfolio.open_count

    risk.kill_switch.trigger("test d'arret d'urgence", source="test")
    report = await orchestrator.run_cycle()

    assert report.halted
    assert "arret d'urgence" in report.halt_reason
    assert portfolio.open_count == 0
    if had_position:
        assert report.positions_closed == 1


async def test_stop_loss_closes_position(paper_settings, candles):
    """Une position dont le stop est touche doit etre fermee au cycle suivant."""
    orchestrator, store, portfolio, _ = build_system(paper_settings, candles)
    await orchestrator.run_cycle()
    if portfolio.open_count == 0:
        pytest.skip("aucune position ouverte sur cet echantillon")

    position = portfolio.get_position("ETH/USDT")
    # On remonte le stop au-dessus du prix courant : il est donc touche.
    position.stop_loss = position.entry_price * 1.5
    report = await orchestrator.run_cycle()
    assert report.positions_closed == 1
    assert portfolio.open_count == 0
    assert store.load_trades().shape[0] == 1


async def test_crisis_regime_liquidates_and_blocks(paper_settings, candles):
    orchestrator, _, portfolio, _ = build_system(paper_settings, candles)
    await orchestrator.run_cycle()

    crisis = RegimeState(
        regime=Regime.CRISIS, confidence=0.95, agreement_score=1.0, transition_probability=0.5
    )
    orchestrator.detector.detect = lambda *args, **kwargs: crisis  # type: ignore[assignment]
    report = await orchestrator.run_cycle()

    assert portfolio.open_count == 0
    assert report.decisions.get("ETH/USDT", "").startswith("no_trade")


async def test_errors_on_one_asset_do_not_break_cycle(paper_settings, candles):
    paper_settings.universe.assets = ["ETH/USDT", "GHOST/USDT"]
    orchestrator, _, _, _ = build_system(paper_settings, candles)
    report = await orchestrator.run_cycle()
    # L'actif fantome echoue, mais ETH est bien traite.
    assert "ETH/USDT" in report.regime_by_asset
    assert report.equity > 0


async def test_loop_runs_multiple_cycles(paper_settings, candles):
    orchestrator, _, _, _ = build_system(paper_settings, candles)
    paper_settings.general.loop_interval_sec = 0
    await orchestrator.run_forever(max_cycles=3)
    assert orchestrator._cycles == 3


# ------------------------------------------------------ garde-fou execution


async def test_execution_without_risk_approval_is_impossible(paper_settings, candles):
    """LE test central : on ne peut pas envoyer d'ordre sans verdict approuve."""
    executor = OrderExecutor(paper_settings, broker=PaperBroker(paper_settings.execution))
    snapshot = snapshot_from(candles)
    decision = EnsembleDecision(
        asset="ETH/USDT",
        signal=Signal.BUY,
        score=1.0,
        confidence=0.9,
        consensus=1.0,
        dispersion=0.0,
        weights={},
        contributions=[],
        stop_loss=snapshot.last_price * 0.97,
        target_price=snapshot.last_price * 1.08,
        entry_price=snapshot.last_price,
    )
    intent = TradeIntent(
        asset="ETH/USDT",
        side=OrderSide.BUY,
        entry_price=snapshot.last_price,
        stop_loss=snapshot.last_price * 0.97,
        target_price=snapshot.last_price * 1.08,
        confidence=0.9,
        regime=RegimeState(Regime.BULL_LOW_VOL, 0.9, 1.0, 0.1),
        decision=decision,
        contra_report=ContraReport([], 0.0, ContraRecommendation.PROCEED),
    )
    rejected = RiskVerdict(
        decision=RiskDecision.REJECTED,
        approved_size=0.0,
        approved_notional=0.0,
        reasons=["refus de test"],
    )
    with pytest.raises(RiskBypassError):
        await executor.execute(intent, rejected, snapshot)


async def test_paper_execution_applies_fees_and_slippage(paper_settings, candles):
    broker = PaperBroker(paper_settings.execution, seed=3)
    snapshot = snapshot_from(candles)
    order = build_order(
        "ETH/USDT",
        OrderSide.BUY,
        size=0.05,
        price=None,
        order_type=OrderType.MARKET,
        reason="test",
    )
    fill = await broker.execute(order, snapshot)
    assert fill.filled_size > 0
    assert fill.fees > 0
    assert fill.slippage_bps > 0
    # Un achat au marche est servi au-dessus du dernier prix.
    assert fill.average_price >= snapshot.last_price


async def test_paper_partial_fill_on_oversized_order(paper_settings, candles):
    """Un ordre trop gros pour le carnet n'est que partiellement servi."""
    broker = PaperBroker(paper_settings.execution, seed=3)
    snapshot = snapshot_from(candles)
    average_volume = float(snapshot.ohlcv["volume"].tail(20).mean())
    order = build_order(
        "ETH/USDT",
        OrderSide.BUY,
        size=average_volume * 10,
        price=None,
        order_type=OrderType.MARKET,
        reason="ordre surdimensionne",
    )
    fill = await broker.execute(order, snapshot)
    assert fill.filled_size < order.size
    assert "fill partiel" in fill.note


async def test_slippage_tracker_records_divergence(paper_settings, candles):
    orchestrator, _, _, _ = build_system(paper_settings, candles)
    await orchestrator.run_cycle()
    summary = orchestrator.executor.slippage.summary()
    assert set(summary) >= {"count", "mean_estimated_bps", "mean_realized_bps", "divergence_pct"}


async def test_orders_are_persisted(paper_settings, candles):
    orchestrator, store, portfolio, _ = build_system(paper_settings, candles)
    await orchestrator.run_cycle()
    if portfolio.open_count == 0:
        pytest.skip("aucun ordre passe sur cet echantillon")
    with store.session() as session:
        from trader.data.store import OrderRow

        assert session.query(OrderRow).count() >= 1


# ------------------------------------------------------- orchestrateur : bords


async def test_liquidation_failure_is_reported_not_swallowed(paper_settings, candles):
    """Si une liquidation echoue, le cycle doit le dire, pas l'ignorer."""
    orchestrator, _, portfolio, risk = build_system(paper_settings, candles)
    await orchestrator.run_cycle()
    if portfolio.open_count == 0:
        pytest.skip("aucune position ouverte sur cet echantillon")

    async def failing_close(*args, **kwargs):
        raise RuntimeError("exchange injoignable")

    orchestrator.executor.close_position = failing_close
    risk.kill_switch.trigger("arret", source="test")
    report = await orchestrator.run_cycle()
    assert report.halted
    assert any("liquidation" in error for error in report.errors)
    assert portfolio.open_count == 1  # la position reste, et on le sait


async def test_exit_failure_raises_critical_alert(paper_settings, candles):
    class RecordingAlerter:
        def __init__(self):
            self.alerts = []

        async def send(self, level, message, details=None):
            self.alerts.append((level, message))
            return True

    orchestrator, _, portfolio, _ = build_system(paper_settings, candles)
    alerter = RecordingAlerter()
    orchestrator.alerter = alerter
    await orchestrator.run_cycle()
    if portfolio.open_count == 0:
        pytest.skip("aucune position ouverte sur cet echantillon")

    position = portfolio.get_position("ETH/USDT")
    position.stop_loss = position.entry_price * 1.5  # stop touche

    from trader.execution.executor import ExecutionResult
    from trader.execution.paper import build_order
    from trader.models import OrderType

    async def unfilled_close(asset, side, size, data, reason):
        order = build_order(asset, side, size, None, OrderType.MARKET, reason=reason)
        return ExecutionResult(order=order, filled=False, attempts=3, error="carnet vide")

    orchestrator.executor.close_position = unfilled_close
    await orchestrator.run_cycle()
    assert any(level == "CRITICAL" for level, _ in alerter.alerts)
    assert portfolio.open_count == 1  # on ne fait jamais disparaitre une position non fermee


async def test_alerter_failure_does_not_break_cycle(paper_settings, candles):
    class BrokenAlerter:
        async def send(self, level, message, details=None):
            raise RuntimeError("telegram hors service")

    orchestrator, _, _, _ = build_system(paper_settings, candles)
    orchestrator.alerter = BrokenAlerter()
    report = await orchestrator.run_cycle()
    assert report.equity > 0


async def test_metrics_hook_is_called(paper_settings, candles):
    class RecordingMetrics:
        def __init__(self):
            self.calls = 0

        def update_from_cycle(self, orchestrator, report, prices):
            self.calls += 1

    orchestrator, _, _, _ = build_system(paper_settings, candles)
    metrics = RecordingMetrics()
    orchestrator.metrics = metrics
    await orchestrator.run_cycle()
    assert metrics.calls == 1


async def test_stop_during_loop_ends_it_after_current_cycle(paper_settings, candles):
    """stop() interrompt la boucle proprement, sans couper un cycle en cours."""

    class StoppingMetrics:
        def __init__(self, orchestrator):
            self.orchestrator = orchestrator

        def update_from_cycle(self, orchestrator, report, prices):
            self.orchestrator.stop()

    orchestrator, _, _, _ = build_system(paper_settings, candles)
    paper_settings.general.loop_interval_sec = 0
    orchestrator.metrics = StoppingMetrics(orchestrator)
    await orchestrator.run_forever(max_cycles=5)
    assert orchestrator._cycles == 1


async def test_shutdown_keeps_positions_open(paper_settings, candles):
    """Un arret planifie ne liquide pas : seul le kill switch le fait."""
    orchestrator, _, portfolio, _ = build_system(paper_settings, candles)
    await orchestrator.run_cycle()
    before = portfolio.open_count
    await orchestrator.shutdown()
    assert portfolio.open_count == before


async def test_adaptation_runs_and_updates_health(paper_settings, candles):
    """La boucle d'adaptation met a jour la sante des strategies apres le cycle."""
    from trader.adaptation.decay_detector import StrategyDecayDetector
    from trader.adaptation.evaluator import StrategyEvaluator

    orchestrator, store, _, _ = build_system(paper_settings, candles)
    evaluator = StrategyEvaluator(store, min_trades=1, mode="paper")
    orchestrator.evaluator = evaluator
    orchestrator.decay = StrategyDecayDetector(
        paper_settings.decay_detection, evaluator, shadow_mode_days=14
    )
    await orchestrator.run_cycle()
    assert orchestrator.decay.last_check is not None
    for record in orchestrator.ensemble.records.values():
        assert record.health is not None


async def test_decay_failure_is_contained(paper_settings, candles):
    class BrokenDecay:
        last_check = None

        def needs_check(self, now=None):
            return True

        def check_all(self, names, now=None):
            raise RuntimeError("evaluateur casse")

        def health_provider(self, name):
            from trader.models import StrategyHealth

            return StrategyHealth.HEALTHY

    orchestrator, _, _, _ = build_system(paper_settings, candles)
    orchestrator.decay = BrokenDecay()
    report = await orchestrator.run_cycle()
    assert any("decay" in error for error in report.errors)
    assert report.equity > 0  # le trading continue


async def test_retraining_is_triggered_by_decay(paper_settings, candles, tmp_path):
    """Une strategie en declin doit declencher un retraining, trace en base."""
    from trader.adaptation.retrainer import WalkForwardRetrainer
    from trader.models import StrategyHealth

    orchestrator, store, _, _ = build_system(paper_settings, candles)
    paper_settings.retraining.artifacts_dir = str(tmp_path / "retraining")

    class DecayStub:
        last_check = None

        def needs_check(self, now=None):
            return True

        def check_all(self, names, now=None):
            from trader.adaptation.decay_detector import DecayVerdict

            self.last_check = now
            return {
                name: DecayVerdict(
                    strategy=name,
                    health=StrategyHealth.DEGRADING,
                    signals=["hit rate en baisse"],
                    metrics={"sharpe_30d": -0.2},
                    weight_multiplier=0.5,
                    needs_retraining=(name == "a"),
                )
                for name in names
            }

        def health_provider(self, name):
            return StrategyHealth.DEGRADING

    orchestrator.decay = DecayStub()
    orchestrator.retrainer = WalkForwardRetrainer(paper_settings, max_candidates=2)
    await orchestrator.run_cycle()

    assert orchestrator.decay.last_check is not None
    assert orchestrator.ensemble.records["a"].health is StrategyHealth.DEGRADING
    events = [e for e in store.load_events() if e["source"] == "retraining"]
    assert events, "le retraining doit laisser une trace dans l'audit trail"


async def test_retraining_is_rate_limited(paper_settings, candles, tmp_path):
    """Un retraining est couteux : il ne se relance pas a chaque cycle."""
    from trader.adaptation.retrainer import WalkForwardRetrainer

    orchestrator, store, _, _ = build_system(paper_settings, candles)
    paper_settings.retraining.artifacts_dir = str(tmp_path / "retraining")
    orchestrator.retrainer = WalkForwardRetrainer(paper_settings, max_candidates=1)

    report = orchestrator_report()
    await orchestrator._market_data("ETH/USDT")
    await orchestrator._retrain("a", report, utc_now())
    first = len([e for e in store.load_events() if e["source"] == "retraining"])
    await orchestrator._retrain("a", report, utc_now())
    second = len([e for e in store.load_events() if e["source"] == "retraining"])
    assert second == first  # deuxieme appel immediat ignore


def orchestrator_report():
    from trader.orchestrator import CycleReport

    return CycleReport(timestamp=utc_now())


async def test_retraining_score_is_finite(paper_settings, candles):
    """Le score de retraining doit rester exploitable, meme sur peu de donnees."""
    orchestrator, _, _, _ = build_system(paper_settings, candles)
    ohlcv, features = await orchestrator._market_data("ETH/USDT")
    strategy = orchestrator.ensemble.records["a"].strategy
    score = orchestrator._retraining_score(strategy, ohlcv, features)
    assert isinstance(score, float)
    assert score == score  # non NaN


async def test_retraining_score_on_short_history(paper_settings, candles):
    orchestrator, _, _, _ = build_system(paper_settings, candles)
    ohlcv, features = await orchestrator._market_data("ETH/USDT")
    strategy = orchestrator.ensemble.records["a"].strategy
    assert orchestrator._retraining_score(strategy, ohlcv.head(30), features.head(30)) == 0.0


async def test_retraining_failure_is_contained(paper_settings, candles):
    class BrokenRetrainer:
        def retrain_strategy(self, *args, **kwargs):
            raise RuntimeError("optimiseur casse")

    orchestrator, _, _, _ = build_system(paper_settings, candles)
    orchestrator.retrainer = BrokenRetrainer()
    await orchestrator._market_data("ETH/USDT")
    report = orchestrator_report()
    await orchestrator._retrain("a", report, utc_now())
    assert any("retraining" in error for error in report.errors)

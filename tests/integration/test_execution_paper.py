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

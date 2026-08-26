"""Tests de l'executeur d'ordres : retry, escalade, mode live, garde-fou risque."""

from __future__ import annotations

import warnings

import pytest

from tests.conftest import make_ohlcv
from trader.config import load_settings
from trader.data.features import FeatureBuilder
from trader.data.snapshot import build_snapshot
from trader.execution.executor import ExecutionError, OrderExecutor, RiskBypassError
from trader.execution.paper import PaperBroker, PaperFill, build_order
from trader.execution.slippage import SlippageTracker, estimate_slippage
from trader.models import (
    ContraRecommendation,
    ContraReport,
    EnsembleDecision,
    OrderSide,
    OrderStatus,
    OrderType,
    Regime,
    RegimeState,
    RiskDecision,
    RiskVerdict,
    Signal,
    TradeIntent,
)

warnings.filterwarnings("ignore")


@pytest.fixture
def snapshot():
    frame = make_ohlcv(n=300, seed=91)
    features = FeatureBuilder(timeframe="1h").build(frame)
    price = float(frame["close"].iloc[-1])
    return build_snapshot("ETH/USDT", frame, features, bid=price * 0.9995, ask=price * 1.0005)


@pytest.fixture
def settings():
    return load_settings(
        "config/default.toml",
        overrides={"execution": {"simulate_latency": False, "retry_backoff_sec": 0.01}},
    )


def make_intent(snapshot, side: OrderSide = OrderSide.BUY) -> TradeIntent:
    price = snapshot.last_price
    direction = 1 if side is OrderSide.BUY else -1
    decision = EnsembleDecision(
        asset="ETH/USDT",
        signal=Signal.BUY if direction > 0 else Signal.SELL,
        score=0.8,
        confidence=0.8,
        consensus=1.0,
        dispersion=0.0,
        weights={},
        contributions=[],
        stop_loss=price * (1 - direction * 0.03),
        target_price=price * (1 + direction * 0.08),
        entry_price=price,
    )
    return TradeIntent(
        asset="ETH/USDT",
        side=side,
        entry_price=price,
        stop_loss=price * (1 - direction * 0.03),
        target_price=price * (1 + direction * 0.08),
        confidence=0.8,
        regime=RegimeState(Regime.BULL_LOW_VOL, 0.9, 1.0, 0.1),
        decision=decision,
        contra_report=ContraReport([], 0.1, ContraRecommendation.PROCEED),
    )


def approved(size: float = 0.05) -> RiskVerdict:
    return RiskVerdict(
        decision=RiskDecision.APPROVED,
        approved_size=size,
        approved_notional=size * 2000.0,
        reasons=["test"],
    )


# ------------------------------------------------------------- garde-fou


async def test_rejected_verdict_blocks_execution(settings, snapshot):
    executor = OrderExecutor(settings, broker=PaperBroker(settings.execution))
    verdict = RiskVerdict(RiskDecision.REJECTED, 0.0, 0.0, ["refus"])
    with pytest.raises(RiskBypassError):
        await executor.execute(make_intent(snapshot), verdict, snapshot)


async def test_zero_size_verdict_blocks_execution(settings, snapshot):
    executor = OrderExecutor(settings, broker=PaperBroker(settings.execution))
    verdict = RiskVerdict(RiskDecision.APPROVED, 0.0, 0.0, ["taille nulle"])
    with pytest.raises(RiskBypassError):
        await executor.execute(make_intent(snapshot), verdict, snapshot)


# --------------------------------------------------------------- execution


async def test_limit_order_uses_maker_fee(settings, snapshot):
    executor = OrderExecutor(settings, broker=PaperBroker(settings.execution, seed=1))
    result = await executor.execute(make_intent(snapshot), approved(), snapshot)
    assert result.filled
    assert result.order.order_type is OrderType.LIMIT
    expected = result.notional * settings.execution.maker_fee_bps / 10_000.0
    assert result.order.fees == pytest.approx(expected, rel=1e-6)


async def test_urgent_order_goes_to_market(settings, snapshot):
    executor = OrderExecutor(settings, broker=PaperBroker(settings.execution, seed=1))
    result = await executor.execute(make_intent(snapshot), approved(), snapshot, urgent=True)
    assert result.order.order_type is OrderType.MARKET


async def test_exit_orders_are_always_market(settings, snapshot):
    """Une sortie ne s'optimise pas : un limit non servi laisse une position ouverte."""
    executor = OrderExecutor(settings, broker=PaperBroker(settings.execution, seed=1))
    result = await executor.close_position("ETH/USDT", OrderSide.SELL, 0.05, snapshot, "stop_loss")
    assert result.order.order_type is OrderType.MARKET
    assert result.order.reason == "stop_loss"


async def test_retry_then_success(settings, snapshot):
    class FlakyBroker(PaperBroker):
        def __init__(self, config):
            super().__init__(config, seed=1)
            self.calls = 0

        async def execute(self, order, data):
            self.calls += 1
            if self.calls < 2:
                raise ConnectionError("exchange injoignable")
            return await super().execute(order, data)

    broker = FlakyBroker(settings.execution)
    executor = OrderExecutor(settings, broker=broker)
    result = await executor.execute(make_intent(snapshot), approved(), snapshot)
    assert result.filled
    assert result.attempts == 2


async def test_limit_escalates_to_market_after_retries(settings, snapshot):
    """Insister au meme prix limite ne sert a rien si le marche est parti."""

    class NeverFillsLimit(PaperBroker):
        async def execute(self, order, data):
            if order.order_type is OrderType.LIMIT:
                return PaperFill(0.0, 0.0, 0.0, 0.0, OrderStatus.CANCELED, 0.0, "non servi")
            return await super().execute(order, data)

    executor = OrderExecutor(settings, broker=NeverFillsLimit(settings.execution, seed=1))
    result = await executor.execute(make_intent(snapshot), approved(), snapshot)
    assert result.order.metadata.get("escalated_to_market")
    assert result.filled


async def test_definitive_failure_is_reported(settings, snapshot):
    class BrokenBroker(PaperBroker):
        async def execute(self, order, data):
            raise ConnectionError("exchange hors service")

    executor = OrderExecutor(settings, broker=BrokenBroker(settings.execution))
    result = await executor.execute(make_intent(snapshot), approved(), snapshot)
    assert not result.filled
    assert result.order.status is OrderStatus.FAILED
    assert "hors service" in result.error


async def test_orders_are_persisted(settings, snapshot, memory_store):
    executor = OrderExecutor(
        settings, broker=PaperBroker(settings.execution, seed=1), store=memory_store
    )
    await executor.execute(make_intent(snapshot), approved(), snapshot)
    with memory_store.session() as session:
        from trader.data.store import OrderRow

        assert session.query(OrderRow).count() == 1


async def test_persistence_failure_does_not_block_trading(settings, snapshot):
    class BrokenStore:
        def save_order(self, order, mode="paper"):
            raise RuntimeError("base indisponible")

    executor = OrderExecutor(
        settings, broker=PaperBroker(settings.execution, seed=1), store=BrokenStore()
    )
    result = await executor.execute(make_intent(snapshot), approved(), snapshot)
    assert result.filled  # l'ordre passe malgre l'echec de persistence


async def test_slippage_is_tracked(settings, snapshot):
    executor = OrderExecutor(settings, broker=PaperBroker(settings.execution, seed=1))
    await executor.execute(make_intent(snapshot), approved(), snapshot)
    assert executor.slippage.count == 1
    assert "divergence_pct" in executor.slippage.summary()


# ------------------------------------------------------------------- live


async def test_live_mode_without_client_fails_loudly(snapshot):
    live_settings = load_settings(
        "config/default.toml",
        "config/live.toml",
        overrides={"execution": {"simulate_latency": False, "retry_backoff_sec": 0.01}},
    )
    executor = OrderExecutor(live_settings, broker=PaperBroker(live_settings.execution))
    assert executor.is_live
    result = await executor.execute(make_intent(snapshot), approved(), snapshot)
    assert not result.filled
    assert "aucun client" in result.error


async def test_live_order_maps_ccxt_response(snapshot):
    live_settings = load_settings(
        "config/default.toml",
        "config/live.toml",
        overrides={"execution": {"simulate_latency": False, "retry_backoff_sec": 0.01}},
    )

    class FakeCcxt:
        def __init__(self):
            self.orders = []

        async def create_order(self, symbol, type_, side, amount, price=None, params=None):
            self.orders.append((symbol, type_, side, amount, price))
            return {
                "id": "abc123",
                "status": "closed",
                "filled": amount,
                "average": snapshot.last_price * 1.001,
                "fee": {"cost": 0.12},
            }

    client = FakeCcxt()
    executor = OrderExecutor(live_settings, exchange_client=client)
    result = await executor.execute(make_intent(snapshot), approved(0.02), snapshot)
    assert result.filled
    assert result.order.status is OrderStatus.FILLED
    assert result.order.fees == pytest.approx(0.12)
    assert client.orders[0][1] == "limit"


# -------------------------------------------------------------- slippage


def test_slippage_grows_with_size():
    small = estimate_slippage(
        2000.0, OrderSide.BUY, size=1.0, spread_pct=0.05, average_volume=1000.0
    )
    large = estimate_slippage(
        2000.0, OrderSide.BUY, size=500.0, spread_pct=0.05, average_volume=1000.0
    )
    assert large.slippage_bps > small.slippage_bps
    assert large.price > small.price


def test_slippage_direction_depends_on_side():
    buy = estimate_slippage(2000.0, OrderSide.BUY, 1.0, spread_pct=0.1)
    sell = estimate_slippage(2000.0, OrderSide.SELL, 1.0, spread_pct=0.1)
    assert buy.price > 2000.0 > sell.price


def test_fixed_bps_model():
    estimate = estimate_slippage(2000.0, OrderSide.BUY, 1.0, model="fixed_bps", fixed_bps=10.0)
    assert estimate.slippage_bps == 10.0
    assert estimate.impact_component_bps == 0.0


def test_invalid_reference_price_is_rejected():
    with pytest.raises(ValueError, match="prix de reference"):
        estimate_slippage(0.0, OrderSide.BUY, 1.0)


def test_tracker_divergence():
    tracker = SlippageTracker()
    assert tracker.divergence_pct() == 0.0
    for _ in range(10):
        tracker.record(estimated_bps=10.0, realized_bps=20.0)
    assert tracker.divergence_pct() == pytest.approx(100.0)
    assert tracker.summary()["count"] == 10


def test_build_order_generates_unique_ids():
    """Deux ordres emis dans la meme milliseconde doivent rester distincts."""
    ids = {
        build_order("ETH/USDT", OrderSide.BUY, 1.0, None, OrderType.MARKET).client_id
        for _ in range(50)
    }
    assert len(ids) == 50
    first = build_order("ETH/USDT", OrderSide.BUY, 1.0, None, OrderType.MARKET)
    assert first.fill_ratio == 0.0
    assert "ETHUSDT" in first.client_id


async def test_paper_rejects_invalid_price(settings):
    """Un snapshot sans prix exploitable produit un rejet, pas un fill fantome."""
    frame = make_ohlcv(n=100, seed=1)
    features = FeatureBuilder(timeframe="1h").build(frame)
    snapshot = build_snapshot("ETH/USDT", frame, features)
    object.__setattr__(snapshot, "last_price", 0.0)
    broker = PaperBroker(settings.execution)
    order = build_order("ETH/USDT", OrderSide.BUY, 1.0, None, OrderType.MARKET)
    fill = await broker.execute(order, snapshot)
    assert fill.status is OrderStatus.REJECTED


def test_execution_error_is_available():
    assert issubclass(ExecutionError, RuntimeError)


# ----------------------------------------------------- frictions du paper


async def test_latency_is_actually_simulated(snapshot):
    """La latence n'est pas cosmetique : le simulateur la subit vraiment."""
    import time

    from trader.config import load_settings as load

    settings = load(
        "config/default.toml",
        overrides={"execution": {"simulate_latency": True, "latency_ms_range": [100, 200]}},
    )
    broker = PaperBroker(settings.execution, seed=5)
    order = build_order("ETH/USDT", OrderSide.BUY, 0.01, None, OrderType.MARKET)
    started = time.monotonic()
    fill = await broker.execute(order, snapshot)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    assert 90 <= fill.latency_ms <= 210
    assert elapsed_ms >= 90


async def test_partial_fill_below_minimum_is_flagged(settings, snapshot):
    """Un fill trop partiel doit etre signale, pas presente comme un succes."""
    broker = PaperBroker(settings.execution, seed=5)
    average_volume = float(snapshot.ohlcv["volume"].tail(20).mean())
    order = build_order("ETH/USDT", OrderSide.BUY, average_volume * 50, None, OrderType.MARKET)
    fill = await broker.execute(order, snapshot)
    assert fill.status is OrderStatus.PARTIALLY_FILLED
    assert "minimum de remplissage" in fill.note


async def test_small_order_is_fully_filled(settings, snapshot):
    broker = PaperBroker(settings.execution, seed=5)
    order = build_order("ETH/USDT", OrderSide.BUY, 0.001, None, OrderType.MARKET)
    fill = await broker.execute(order, snapshot)
    assert fill.status is OrderStatus.FILLED
    assert fill.note == ""


async def test_sell_is_filled_at_the_bid(settings, snapshot):
    """On vend toujours du mauvais cote du spread."""
    broker = PaperBroker(settings.execution, seed=5)
    order = build_order("ETH/USDT", OrderSide.SELL, 0.01, None, OrderType.MARKET)
    fill = await broker.execute(order, snapshot)
    assert fill.average_price <= snapshot.last_price


async def test_paper_without_volume_history_still_fills(settings):
    import pandas as pd

    frame = make_ohlcv(n=60, seed=3).drop(columns=["volume"])
    frame["volume"] = pd.NA
    snapshot_no_volume = build_snapshot(
        "ETH/USDT",
        make_ohlcv(n=60, seed=3),
        FeatureBuilder(timeframe="1h").build(make_ohlcv(n=60, seed=3)),
    )
    object.__setattr__(snapshot_no_volume, "ohlcv", None)
    broker = PaperBroker(settings.execution, seed=5)
    order = build_order("ETH/USDT", OrderSide.BUY, 0.01, None, OrderType.MARKET)
    fill = await broker.execute(order, snapshot_no_volume)
    assert fill.filled_size == pytest.approx(0.01)

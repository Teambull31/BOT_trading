"""Tests de la couche de persistence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from tests.conftest import make_ohlcv
from trader.models import Order, OrderSide, OrderStatus, OrderType, Regime, RegimeState, Trade


def test_ohlcv_roundtrip_and_idempotence(memory_store):
    frame = make_ohlcv(n=50)
    assert memory_store.save_ohlcv("binance", "ETH/USDT", "1h", frame) == 50
    assert memory_store.save_ohlcv("binance", "ETH/USDT", "1h", frame) == 0  # pas de doublons

    loaded = memory_store.load_ohlcv("ETH/USDT", "1h")
    assert len(loaded) == 50
    assert loaded.index.is_monotonic_increasing
    pd.testing.assert_series_equal(
        loaded["close"].reset_index(drop=True),
        frame["close"].reset_index(drop=True),
        check_names=False,
    )


def test_last_timestamp_tracks_latest_candle(memory_store):
    frame = make_ohlcv(n=20)
    memory_store.save_ohlcv("binance", "ETH/USDT", "1h", frame)
    last = memory_store.last_ohlcv_timestamp("binance", "ETH/USDT", "1h")
    assert last == frame.index[-1].to_pydatetime()
    assert memory_store.last_ohlcv_timestamp("binance", "BTC/USDT", "1h") is None


def test_ohlcv_window_filtering(memory_store):
    frame = make_ohlcv(n=100)
    memory_store.save_ohlcv("binance", "ETH/USDT", "1h", frame)
    subset = memory_store.load_ohlcv("ETH/USDT", "1h", start=frame.index[40], end=frame.index[59])
    assert len(subset) == 20


def test_raw_snapshots_are_kept_for_audit(memory_store):
    memory_store.save_raw("binance", "ETH/USDT", "order_book", {"bids": [[1, 2]]})
    rows = memory_store.load_raw("ETH/USDT", "order_book")
    assert len(rows) == 1
    assert rows[0][1]["bids"] == [[1, 2]]


def test_order_and_trade_persistence(memory_store):
    order = Order(
        asset="ETH/USDT",
        side=OrderSide.BUY,
        size=0.5,
        order_type=OrderType.LIMIT,
        price=2000.0,
        status=OrderStatus.FILLED,
        filled_size=0.5,
        average_price=2001.0,
    )
    memory_store.save_order(order)

    opened = datetime(2024, 3, 1, tzinfo=UTC)
    trade = Trade(
        asset="ETH/USDT",
        side=OrderSide.BUY,
        size=0.5,
        entry_price=2000.0,
        exit_price=2100.0,
        pnl=50.0,
        fees=1.0,
        opened_at=opened,
        closed_at=opened + timedelta(hours=6),
        strategy="momentum",
    )
    memory_store.save_trade(trade)

    trades = memory_store.load_trades(strategy="momentum")
    assert len(trades) == 1
    assert trades.iloc[0]["pnl"] == 50.0
    assert memory_store.load_trades(strategy="mean_revert").empty


def test_equity_curve_persistence(memory_store):
    base = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(5):
        memory_store.save_equity(
            equity=10000.0 + i * 10, cash=9000.0, timestamp=base + timedelta(days=i)
        )
    curve = memory_store.load_equity()
    assert len(curve) == 5
    assert curve.iloc[-1] == 10040.0


def test_regime_history_persistence(memory_store):
    state = RegimeState(
        regime=Regime.BULL_LOW_VOL,
        confidence=0.8,
        agreement_score=1.0,
        transition_probability=0.1,
    )
    memory_store.save_regime(state, asset="ETH/USDT")
    history = memory_store.load_regimes()
    assert history.iloc[-1]["regime"] == "bull_low_vol"


def test_decision_audit_trail(memory_store):
    memory_store.save_decision("ETH/USDT", "risk", "rejected", {"reason": "drawdown"})
    decisions = memory_store.load_decisions()
    assert decisions[0]["stage"] == "risk"
    assert decisions[0]["payload"]["reason"] == "drawdown"


def test_events_are_filterable_by_level(memory_store):
    memory_store.save_event("INFO", "test", "ok")
    memory_store.save_event("CRITICAL", "risk", "kill switch")
    assert len(memory_store.load_events(level="CRITICAL")) == 1


def test_purge_clears_tables(memory_store):
    memory_store.save_event("INFO", "test", "ok")
    memory_store.purge(["events"])
    assert memory_store.load_events() == []

"""Tests du decoupage walk-forward et du moteur de backtest.

Ce que l'on verifie ici n'est pas la rentabilite (impossible a garantir) mais
l'HONNETETE du backtest : pas de look-ahead, execution a la barre suivante,
frais toujours appliques, stops respectes, purge entre train et validation.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from tests.conftest import make_ohlcv
from tests.unit.test_ensemble import ScriptedStrategy
from trader.backtest.engine import BacktestEngine, buy_and_hold
from trader.backtest.walk_forward import (
    WalkForwardResult,
    assert_no_overlap,
    walk_forward_splits,
)
from trader.config import RetrainingConfig, load_settings
from trader.models import Signal
from trader.strategy.ensemble import StrategyEnsemble

warnings.filterwarnings("ignore")


@pytest.fixture
def settings():
    return load_settings("config/default.toml", overrides={"general": {"initial_capital": 10000.0}})


def always_long_ensemble() -> StrategyEnsemble:
    """Ensemble qui achete des qu'il en a le droit : utile pour tester la mecanique."""
    regimes = [
        "bull_low_vol",
        "bull_high_vol",
        "bear_low_vol",
        "bear_high_vol",
        "range_bound",
        "uncertain",
    ]
    return StrategyEnsemble(
        [
            ScriptedStrategy("a", Signal.BUY, regimes=regimes, stop_pct=0.03, target_pct=0.05),
            ScriptedStrategy("b", Signal.BUY, regimes=regimes, stop_pct=0.04, target_pct=0.06),
        ]
    )


# ----------------------------------------------------------------- decoupage


def test_splits_respect_purge_gap():
    index = pd.date_range("2024-01-01", periods=400, freq="1D", tz="UTC")
    config = RetrainingConfig(
        train_window_days=90, validation_window_days=14, test_window_days=7, purge_gap_days=2
    )
    splits = walk_forward_splits(index, config)
    assert splits
    for split in splits:
        assert_no_overlap(split)
        gap_days = (split.validation_start - split.train_end).days
        assert gap_days == config.purge_gap_days


def test_splits_move_forward_in_time():
    index = pd.date_range("2024-01-01", periods=500, freq="1D", tz="UTC")
    splits = walk_forward_splits(index, RetrainingConfig())
    starts = [split.train_start for split in splits]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_short_history_yields_no_splits():
    index = pd.date_range("2024-01-01", periods=30, freq="1D", tz="UTC")
    assert walk_forward_splits(index, RetrainingConfig()) == []


def test_split_slicing_is_disjoint():
    index = pd.date_range("2024-01-01", periods=400, freq="1D", tz="UTC")
    frame = pd.DataFrame({"value": range(len(index))}, index=index)
    split = walk_forward_splits(index, RetrainingConfig())[0]
    train = split.slice_train(frame)
    validation = split.slice_validation(frame)
    assert train.index.max() < validation.index.min()


def test_oos_ratio_flags_overfitting():
    result = WalkForwardResult(
        in_sample=[{"sharpe": 2.0}, {"sharpe": 2.0}],
        out_of_sample=[{"sharpe": 0.2}, {"sharpe": 0.1}],
    )
    assert result.oos_ratio() < 0.2
    assert not result.is_robust(min_ratio=0.70)


def test_oos_ratio_accepts_robust_model():
    result = WalkForwardResult(
        in_sample=[{"sharpe": 1.0}, {"sharpe": 1.2}],
        out_of_sample=[{"sharpe": 0.9}, {"sharpe": 1.0}],
    )
    assert result.is_robust(min_ratio=0.70)


# -------------------------------------------------------------------- moteur


def test_engine_produces_trades_and_equity(settings):
    frame = make_ohlcv(n=1500, drift=0.001, vol=0.01, seed=31)
    result = BacktestEngine(settings, always_long_ensemble()).run(
        frame, warmup=300, retrain_every=10_000
    )
    assert result.trade_count > 0
    assert not result.equity.empty
    assert result.equity.index.is_monotonic_increasing
    assert set(result.metrics) >= {"sharpe", "max_drawdown_pct", "profit_factor"}


def test_every_trade_pays_fees(settings):
    """Un backtest sans frais est une fiction : chaque trade doit en payer."""
    frame = make_ohlcv(n=1200, drift=0.001, vol=0.01, seed=32)
    result = BacktestEngine(settings, always_long_ensemble()).run(
        frame, warmup=300, retrain_every=10_000
    )
    assert result.trade_count > 0
    assert all(trade.fees > 0 for trade in result.trades)


def test_stops_are_respected(settings):
    """Aucune perte ne doit depasser significativement la distance du stop."""
    frame = make_ohlcv(n=1500, drift=-0.001, vol=0.015, seed=33)
    result = BacktestEngine(settings, always_long_ensemble()).run(
        frame, warmup=300, retrain_every=10_000
    )
    for trade in result.trades:
        if trade.exit_reason == "stop_loss":
            loss_pct = trade.pnl / (trade.size * trade.entry_price) * 100.0
            assert loss_pct > -12.0, f"perte anormale sur un stop : {loss_pct:.2f} %"


def test_execution_happens_on_next_bar(settings):
    """On ne peut pas trader le prix de cloture qui vient de declencher le signal."""
    frame = make_ohlcv(n=800, drift=0.002, vol=0.008, seed=34)
    result = BacktestEngine(settings, always_long_ensemble()).run(
        frame, warmup=300, retrain_every=10_000
    )
    assert result.trade_count > 0
    for trade in result.trades[:5]:
        close_at_signal = float(frame.loc[trade.opened_at, "close"])
        # Le prix d'entree vient de l'ouverture suivante + slippage, jamais du close.
        assert trade.entry_price != close_at_signal


def test_position_size_respects_hard_limit(settings):
    """Sans risk manager branche, la limite en dur de 2 % s'applique quand meme."""
    frame = make_ohlcv(n=1000, drift=0.001, vol=0.01, seed=35)
    result = BacktestEngine(settings, always_long_ensemble()).run(
        frame, warmup=300, retrain_every=10_000
    )
    capital = settings.general.initial_capital
    for trade in result.trades:
        notional = trade.size * trade.entry_price
        assert notional <= capital * settings.risk.max_position_pct / 100.0 * 1.05


def test_risk_hook_can_veto_every_trade(settings):
    """Le hook de risque a le dernier mot : s'il refuse, rien ne passe."""
    frame = make_ohlcv(n=1000, drift=0.002, vol=0.01, seed=36)
    engine = BacktestEngine(
        settings, always_long_ensemble(), risk_hook=lambda decision, regime, equity: 0.0
    )
    result = engine.run(frame, warmup=300, retrain_every=10_000)
    assert result.trade_count == 0
    assert float(result.equity.iloc[-1]) == pytest.approx(settings.general.initial_capital)


def test_engine_rejects_too_short_history(settings):
    with pytest.raises(ValueError, match="trop court"):
        BacktestEngine(settings, always_long_ensemble()).run(make_ohlcv(n=50), warmup=300)


def test_regime_history_is_recorded(settings):
    """Le detecteur est reentraine en walk-forward et son historique est trace."""
    frame = make_ohlcv(n=900, drift=0.001, seed=37)
    result = BacktestEngine(settings, always_long_ensemble()).run(
        frame, warmup=300, retrain_every=200
    )
    assert not result.regimes.empty
    assert set(result.regimes.columns) >= {"timestamp", "regime", "confidence"}


def test_buy_and_hold_benchmark(settings):
    frame = make_ohlcv(n=800, drift=0.002, vol=0.005, seed=38)
    curve = buy_and_hold(frame, 1000.0, warmup=300)
    assert not curve.empty
    assert float(curve.iloc[0]) == pytest.approx(1000.0)


def test_backtest_is_deterministic(settings):
    """Deux executions identiques doivent donner exactement le meme resultat."""
    frame = make_ohlcv(n=900, drift=0.001, vol=0.01, seed=39)
    first = BacktestEngine(settings, always_long_ensemble()).run(
        frame, warmup=300, retrain_every=10_000
    )
    second = BacktestEngine(settings, always_long_ensemble()).run(
        frame, warmup=300, retrain_every=10_000
    )
    assert first.trade_count == second.trade_count
    assert float(first.equity.iloc[-1]) == pytest.approx(float(second.equity.iloc[-1]))

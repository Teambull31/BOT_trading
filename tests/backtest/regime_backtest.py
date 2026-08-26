"""Backtests par regime : verifier que le systeme se comporte differemment selon
le marche dans lequel il opere.

Un systeme adaptatif qui produit exactement le meme comportement en marche
haussier, en range et en crise n'est pas adaptatif : il est aveugle.
"""

from __future__ import annotations

import warnings

import pytest

from tests.conftest import make_ohlcv
from tests.unit.test_ensemble import ScriptedStrategy
from trader.backtest.engine import BacktestEngine
from trader.config import load_settings
from trader.data.features import FeatureBuilder
from trader.models import Regime, Signal
from trader.regime.detector import RegimeDetector
from trader.strategy.ensemble import StrategyEnsemble
from trader.strategy.mean_revert import MeanRevertStrategy
from trader.strategy.momentum import MomentumStrategy

warnings.filterwarnings("ignore")

ALL_REGIMES = [
    "bull_low_vol",
    "bull_high_vol",
    "bear_low_vol",
    "bear_high_vol",
    "range_bound",
    "uncertain",
]


@pytest.fixture
def settings():
    return load_settings("config/default.toml", overrides={"general": {"initial_capital": 10000.0}})


def permissive_ensemble() -> StrategyEnsemble:
    return StrategyEnsemble(
        [
            ScriptedStrategy("a", Signal.BUY, regimes=ALL_REGIMES),
            ScriptedStrategy("b", Signal.BUY, regimes=ALL_REGIMES),
        ]
    )


def test_crisis_regime_blocks_new_positions(settings):
    """En crise, aucune position ne doit etre ouverte."""
    frame = make_ohlcv(n=1200, vol=0.03, drift=-0.004, seed=51, jump_at=700, jump_size=-0.35)
    result = BacktestEngine(settings, permissive_ensemble()).run(
        frame, warmup=300, retrain_every=300
    )
    crisis_bars = result.regimes[result.regimes["regime"] == "crisis"]
    if not crisis_bars.empty:
        crisis_stamps = set(crisis_bars["timestamp"])
        opened_in_crisis = [t for t in result.trades if t.opened_at in crisis_stamps]
        assert not opened_in_crisis, "des positions ont ete ouvertes en regime de crise"


def test_uncertain_regime_halves_exposure(settings):
    """Le regime incertain doit reduire la taille des positions de moitie."""
    frame = make_ohlcv(n=900, drift=0.001, vol=0.012, seed=52)
    result = BacktestEngine(settings, permissive_ensemble()).run(
        frame, warmup=300, retrain_every=10_000
    )
    if result.trade_count == 0:
        pytest.skip("aucun trade genere sur cet echantillon")

    uncertain_stamps = set(result.regimes[result.regimes["regime"] == "uncertain"]["timestamp"])
    certain_stamps = set(result.regimes["timestamp"]) - uncertain_stamps
    uncertain_sizes = [
        t.size * t.entry_price for t in result.trades if t.opened_at in uncertain_stamps
    ]
    certain_sizes = [t.size * t.entry_price for t in result.trades if t.opened_at in certain_stamps]
    if uncertain_sizes and certain_sizes:
        multiplier = settings.risk.uncertain_regime.exposure_multiplier
        assert max(uncertain_sizes) <= max(certain_sizes) * multiplier * 1.05


def test_regime_gating_prevents_trades_without_quorum(settings):
    """Momentum et mean-reversion couvrent des regimes disjoints.

    Avec la regle "minimum 2 strategies actives", aucun trade n'est possible :
    c'est une propriete structurelle du pool, pas un bug. Elle documente
    pourquoi le pool doit couvrir chaque regime avec au moins deux strategies.
    """
    frame = make_ohlcv(n=900, drift=0.002, vol=0.008, seed=53)
    ensemble = StrategyEnsemble([MomentumStrategy(), MeanRevertStrategy()])
    result = BacktestEngine(settings, ensemble).run(frame, warmup=300, retrain_every=10_000)
    assert result.trade_count == 0
    assert any("minimum" in reason for reason in result.blocked_reasons)


@pytest.mark.parametrize(
    ("name", "kwargs", "expected"),
    [
        ("haussier", {"drift": 0.002, "vol": 0.006}, {Regime.BULL_LOW_VOL, Regime.BULL_HIGH_VOL}),
        ("range", {"mean_revert": 0.2, "vol": 0.005}, {Regime.RANGE_BOUND}),
    ],
)
def test_detector_labels_match_generated_market(name, kwargs, expected):
    """Le detecteur doit reconnaitre le marche qu'on lui a fabrique."""
    frame = make_ohlcv(n=1200, seed=19, **kwargs)
    features = FeatureBuilder(timeframe="1h").build(frame)
    detector = RegimeDetector(load_settings("config/default.toml").regime, "1h")
    detector.fit(features)
    state = detector.detect(features, frame)
    assert state.regime in expected | {Regime.UNCERTAIN}


def test_regime_distribution_is_recorded(settings):
    """Le backtest trace la distribution des regimes traverses."""
    frame = make_ohlcv(n=1000, drift=0.001, vol=0.01, seed=54)
    result = BacktestEngine(settings, permissive_ensemble()).run(
        frame, warmup=300, retrain_every=300
    )
    distribution = result.regimes["regime"].value_counts(normalize=True)
    assert not distribution.empty
    assert distribution.sum() == pytest.approx(1.0)
    assert all(label in {r.value for r in Regime} for label in distribution.index)


def test_performance_can_be_broken_down_by_regime(settings):
    """On doit pouvoir attribuer le P&L a chaque regime : sinon on n'apprend rien."""
    frame = make_ohlcv(n=1200, drift=0.001, vol=0.012, seed=55)
    result = BacktestEngine(settings, permissive_ensemble()).run(
        frame, warmup=300, retrain_every=400
    )
    if result.trade_count == 0:
        pytest.skip("aucun trade genere sur cet echantillon")

    regimes = result.regimes.set_index("timestamp")["regime"]
    attributed = {}
    for trade in result.trades:
        label = regimes.get(trade.opened_at, "inconnu")
        attributed[label] = attributed.get(label, 0.0) + trade.pnl
    assert attributed
    assert sum(attributed.values()) == pytest.approx(sum(t.pnl for t in result.trades), rel=1e-6)

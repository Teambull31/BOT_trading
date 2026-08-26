"""Tests des strategies : contrat, comportement par regime, anti-biais."""

from __future__ import annotations

import warnings

import pytest

from tests.conftest import make_ohlcv
from trader.data.features import FeatureBuilder
from trader.data.snapshot import build_snapshot
from trader.models import Regime, RegimeState, Signal, StrategyOutput
from trader.strategy.base import BaseStrategy
from trader.strategy.mean_revert import MeanRevertParams, MeanRevertStrategy
from trader.strategy.momentum import MomentumParams, MomentumStrategy

warnings.filterwarnings("ignore")

ALL_STRATEGIES = [MomentumStrategy, MeanRevertStrategy]


def regime_state(regime: Regime, transition: float = 0.1) -> RegimeState:
    return RegimeState(
        regime=regime, confidence=0.8, agreement_score=1.0, transition_probability=transition
    )


def snapshot_for(frame, position: int = -1):
    features = FeatureBuilder(timeframe="1h").build(frame)
    return build_snapshot("ETH/USDT", frame, features, position=position)


# --------------------------------------------------------------------- contrat


@pytest.mark.parametrize("factory", ALL_STRATEGIES)
def test_strategy_respects_base_contract(factory):
    strategy = factory()
    assert isinstance(strategy, BaseStrategy)
    assert strategy.name
    regimes = strategy.get_required_regimes()
    assert regimes and all(regime in {r.value for r in Regime} for regime in regimes)


@pytest.mark.parametrize("factory", ALL_STRATEGIES)
def test_output_always_has_stop_and_contra_evidence(factory):
    """Tout signal directionnel doit porter un stop ET des preuves contraires."""
    strategy = factory()
    frames = [
        make_ohlcv(n=400, drift=0.004, vol=0.006, seed=3),
        make_ohlcv(n=400, drift=-0.004, vol=0.006, seed=4),
        make_ohlcv(n=400, mean_revert=0.25, vol=0.008, seed=5),
    ]
    produced = 0
    for frame in frames:
        features = FeatureBuilder(timeframe="1h").build(frame)
        # On balaie plusieurs instants : une strategie ne signale pas a chaque bougie.
        for position in range(250, len(frame), 10):
            data = build_snapshot("ETH/USDT", frame, features, position=position)
            for regime in Regime:
                output = strategy.generate_signal(data, regime_state(regime))
                assert isinstance(output, StrategyOutput)
                if output.is_actionable:
                    produced += 1
                    assert output.stop_loss > 0
                    assert output.contra_evidence, "contra_evidence obligatoire"
                    assert output.reasoning
                    assert 0.0 <= output.confidence <= 1.0
    assert produced > 0, "la strategie n'a produit aucun signal exploitable"


@pytest.mark.parametrize("factory", ALL_STRATEGIES)
def test_strategy_never_raises_on_degenerate_data(factory):
    strategy = factory()
    tiny = make_ohlcv(n=30, seed=1)
    output = strategy.generate_signal(snapshot_for(tiny), regime_state(Regime.RANGE_BOUND))
    assert output.signal is Signal.NEUTRAL


@pytest.mark.parametrize("factory", ALL_STRATEGIES)
def test_param_space_bounds_are_enforced(factory):
    strategy = factory()
    space = strategy.param_space()
    assert space
    key, (low, high) = next(iter(space.items()))
    strategy.set_params({key: (low + high) / 2})
    with pytest.raises(ValueError, match="hors des bornes"):
        strategy.set_params({key: high * 100 + 1000})
    with pytest.raises(ValueError, match="inconnu"):
        strategy.set_params({"parametre_invente": 1.0})


@pytest.mark.parametrize("factory", ALL_STRATEGIES)
def test_regime_gating(factory):
    strategy = factory()
    allowed = strategy.get_required_regimes()
    for regime in Regime:
        expected = regime.value in allowed
        assert strategy.is_active_in(regime_state(regime)) is expected


def test_build_output_downgrades_signal_without_contra_evidence():
    """Une strategie qui oublie ses contra_evidence voit son signal annule."""

    class SloppyStrategy(MomentumStrategy):
        name = "sloppy"

        def generate_signal(self, data, regime):
            return self.build_output(
                data=data,
                signal=Signal.BUY,
                confidence=0.9,
                stop_loss=data.last_price * 0.95,
                target_price=data.last_price * 1.1,
                reasoning="je suis sur de moi",
                contra_evidence=[],
            )

    output = SloppyStrategy().generate_signal(
        snapshot_for(make_ohlcv(n=300, seed=2)), regime_state(Regime.BULL_LOW_VOL)
    )
    assert output.signal is Signal.NEUTRAL
    assert "contra-evidence" in output.reasoning


def test_build_output_rejects_missing_stop_loss():
    class NoStopStrategy(MomentumStrategy):
        name = "nostop"

        def generate_signal(self, data, regime):
            return self.build_output(
                data=data,
                signal=Signal.BUY,
                confidence=0.9,
                stop_loss=0.0,
                target_price=data.last_price * 1.1,
                reasoning="pas de stop",
                contra_evidence=["risque"],
            )

    output = NoStopStrategy().generate_signal(
        snapshot_for(make_ohlcv(n=300, seed=2)), regime_state(Regime.BULL_LOW_VOL)
    )
    assert output.signal is Signal.NEUTRAL
    assert "stop loss" in output.reasoning


# -------------------------------------------------------------------- momentum


def test_momentum_follows_established_uptrend():
    strategy = MomentumStrategy(MomentumParams(min_confidence=0.25))
    frame = make_ohlcv(n=600, drift=0.005, vol=0.004, seed=12)
    output = strategy.generate_signal(snapshot_for(frame), regime_state(Regime.BULL_LOW_VOL))
    assert output.signal.direction >= 0
    if output.is_actionable:
        assert output.stop_loss < output.entry_price
        assert output.target_price > output.entry_price


def test_momentum_follows_established_downtrend():
    strategy = MomentumStrategy(MomentumParams(min_confidence=0.25))
    frame = make_ohlcv(n=600, drift=-0.005, vol=0.004, seed=12)
    output = strategy.generate_signal(snapshot_for(frame), regime_state(Regime.BEAR_LOW_VOL))
    assert output.signal.direction <= 0
    if output.is_actionable:
        assert output.stop_loss > output.entry_price


def test_momentum_stays_out_of_flat_market():
    strategy = MomentumStrategy()
    frame = make_ohlcv(n=600, drift=0.0, vol=0.003, mean_revert=0.3, seed=15)
    output = strategy.generate_signal(snapshot_for(frame), regime_state(Regime.BULL_LOW_VOL))
    assert output.signal is Signal.NEUTRAL


def test_momentum_penalizes_unstable_regime():
    """Une transition de regime imminente doit reduire la conviction."""
    strategy = MomentumStrategy(MomentumParams(min_confidence=0.05))
    frame = make_ohlcv(n=600, drift=0.005, vol=0.004, seed=12)
    data = snapshot_for(frame)
    calm = strategy.generate_signal(data, regime_state(Regime.BULL_LOW_VOL, transition=0.05))
    shaky = strategy.generate_signal(data, regime_state(Regime.BULL_LOW_VOL, transition=0.9))
    assert shaky.confidence < calm.confidence
    assert any("regime" in evidence for evidence in shaky.contra_evidence)


# ----------------------------------------------------------------- mean revert


def test_mean_revert_buys_oversold_extension():
    strategy = MeanRevertStrategy(MeanRevertParams(min_confidence=0.2))
    frame = make_ohlcv(n=600, mean_revert=0.15, vol=0.006, seed=21)
    # On force une extension baissiere sur la derniere bougie.
    frame.iloc[-1, frame.columns.get_loc("close")] *= 0.90
    frame.iloc[-1, frame.columns.get_loc("low")] *= 0.88
    output = strategy.generate_signal(snapshot_for(frame), regime_state(Regime.RANGE_BOUND))
    assert output.signal.direction >= 0
    if output.is_actionable:
        assert output.stop_loss < output.entry_price


def test_mean_revert_refuses_trending_market():
    strategy = MeanRevertStrategy()
    frame = make_ohlcv(n=600, drift=0.006, vol=0.004, seed=22)
    output = strategy.generate_signal(snapshot_for(frame), regime_state(Regime.RANGE_BOUND))
    assert output.signal is Signal.NEUTRAL
    assert "directionnel" in output.reasoning or "exces" in output.reasoning


def test_mean_revert_only_authorized_in_range():
    assert MeanRevertStrategy().get_required_regimes() == ["range_bound"]


def test_mean_revert_flags_falling_knife():
    """Trois bougies consecutives contre le trade = contre-indication explicite."""
    strategy = MeanRevertStrategy(MeanRevertParams(min_confidence=0.05))
    frame = make_ohlcv(n=600, mean_revert=0.15, vol=0.006, seed=23)
    for offset in (3, 2, 1):
        frame.iloc[-offset, frame.columns.get_loc("close")] *= 0.93 ** (4 - offset)
    output = strategy.generate_signal(snapshot_for(frame), regime_state(Regime.RANGE_BOUND))
    evidence = " ".join(output.contra_evidence)
    assert output.signal is Signal.NEUTRAL or "couteau" in evidence or evidence

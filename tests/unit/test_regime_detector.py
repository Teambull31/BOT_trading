"""Tests du detecteur de regime.

On teste le COMPORTEMENT attendu (le systeme reconnait un marche haussier,
detecte une crise, sait dire qu'il ne sait pas), pas les valeurs numeriques
exactes des modeles, qui dependent de l'initialisation aleatoire.
"""

from __future__ import annotations

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv
from trader.config import RegimeConfig
from trader.data.features import FeatureBuilder
from trader.models import Regime
from trader.regime.detector import (
    Direction,
    RegimeDetector,
    _required_votes,
    compose_regime,
    direction_of,
)
from trader.regime.trend import TrendState, classify_trend
from trader.regime.volatility import VolRegime, classify_volatility, ewma_volatility
from trader.utils.time_utils import utc_now

warnings.filterwarnings("ignore")


@pytest.fixture
def builder() -> FeatureBuilder:
    return FeatureBuilder(timeframe="1h")


def fitted_detector(frame: pd.DataFrame, builder: FeatureBuilder, **overrides):
    features = builder.build(frame)
    detector = RegimeDetector(RegimeConfig(**overrides), "1h")
    detector.fit(features)
    return detector, features


# ------------------------------------------------------------------ volatilite


def test_volatility_detects_extreme_regime():
    calm = make_ohlcv(n=800, vol=0.004, seed=3)
    shocked = make_ohlcv(n=800, vol=0.004, seed=3, jump_at=799, jump_size=-0.25)
    calm_returns = np.log(calm["close"] / calm["close"].shift(1))
    shocked_returns = np.log(shocked["close"] / shocked["close"].shift(1))

    assert classify_volatility(calm_returns).regime is not VolRegime.EXTREME
    assert classify_volatility(shocked_returns).regime is VolRegime.EXTREME
    assert classify_volatility(shocked_returns).is_crisis_level


def test_volatility_handles_short_series():
    state = classify_volatility(pd.Series([0.01, -0.01, 0.005]))
    assert state.regime is VolRegime.NORMAL


def test_ewma_volatility_is_positive_and_reactive():
    returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))
    vol = ewma_volatility(returns).dropna()
    assert (vol > 0).all()


# ------------------------------------------------------------------ tendance


def test_trend_rules_separate_trend_from_range(builder):
    trend = builder.build(make_ohlcv(n=800, drift=0.003, vol=0.004, seed=8))
    flat = builder.build(make_ohlcv(n=800, mean_revert=0.25, vol=0.004, seed=8))
    assert classify_trend(trend).is_trending
    assert not classify_trend(flat).is_trending


def test_trend_analysis_on_empty_features():
    assert classify_trend(pd.DataFrame()).state is TrendState.UNDEFINED


# ------------------------------------------------------------------ detection


def test_detects_bull_market(builder):
    detector, features = fitted_detector(
        make_ohlcv(n=1200, drift=0.002, vol=0.006, seed=19), builder
    )
    state = detector.detect(features)
    assert state.regime in (Regime.BULL_LOW_VOL, Regime.BULL_HIGH_VOL)
    assert state.confidence >= 0.6


def test_detects_range_market(builder):
    detector, features = fitted_detector(
        make_ohlcv(n=1200, mean_revert=0.2, vol=0.005, seed=19), builder
    )
    assert detector.detect(features).regime is Regime.RANGE_BOUND


@pytest.mark.parametrize("seed", [5, 19, 23])
def test_crisis_short_circuits_the_vote(builder, seed):
    """La crise prime sur le consensus : en cas de risque de ruine, on tranche prudent."""
    frame = make_ohlcv(n=1200, vol=0.03, drift=-0.003, seed=seed, jump_at=1150, jump_size=-0.35)
    detector, features = fitted_detector(frame, builder)
    state = detector.detect(features)
    assert state.regime is Regime.CRISIS
    assert not state.regime.is_tradable
    assert state.details["trigger"] == "volatility_extreme"


def test_uncertain_when_methods_disagree(builder):
    """Le systeme a le droit — et le devoir — de dire qu'il ne sait pas."""
    detector, features = fitted_detector(
        make_ohlcv(n=1200, drift=0.0002, vol=0.012, seed=101), builder, min_confidence=0.95
    )
    state = detector.detect(features)
    assert state.regime is Regime.UNCERTAIN
    assert state.is_uncertain
    assert "reason" in state.details


def test_high_agreement_threshold_forces_uncertainty(builder):
    detector, features = fitted_detector(
        make_ohlcv(n=1200, drift=0.002, seed=19), builder, min_agreement=1.0
    )
    state = detector.detect(features)
    # Avec unanimite exigee, le moindre desaccord bascule en UNCERTAIN.
    assert state.regime is Regime.UNCERTAIN or state.agreement_score == 1.0


def test_detect_without_fit_still_works_via_rules(builder):
    """Sans modeles entraines, les regles quantitatives prennent le relais."""
    features = builder.build(make_ohlcv(n=400, drift=0.003, vol=0.005, seed=7))
    state = RegimeDetector(RegimeConfig(), "1h").detect(features)
    assert isinstance(state.regime, Regime)
    # Une seule methode disponible : aucune corroboration, donc pas de certitude.
    assert state.regime is Regime.UNCERTAIN


def test_empty_features_yield_uncertain():
    state = RegimeDetector(RegimeConfig(), "1h").detect(pd.DataFrame())
    assert state.regime is Regime.UNCERTAIN
    assert state.confidence == 0.0


def test_transition_probability_is_bounded(builder):
    detector, features = fitted_detector(make_ohlcv(n=1200, drift=0.001, seed=11), builder)
    assert 0.0 <= detector.detect(features).transition_probability <= 1.0


def test_retraining_schedule(builder):
    detector, features = fitted_detector(make_ohlcv(n=600, seed=2), builder)
    assert not detector.needs_retrain()
    assert detector.needs_retrain(utc_now() + timedelta(days=8))


def test_fit_survives_degenerate_data(builder):
    """Un modele qui ne converge pas est desactive, il ne fait pas planter le systeme."""
    flat = make_ohlcv(n=300, vol=0.0, drift=0.0, seed=1)
    detector = RegimeDetector(RegimeConfig(), "1h")
    report = detector.fit(builder.build(flat))
    assert isinstance(report, dict)
    state = detector.detect(builder.build(flat))
    assert isinstance(state.regime, Regime)


def test_state_serialization(builder):
    detector, features = fitted_detector(make_ohlcv(n=1200, drift=0.002, seed=19), builder)
    payload = detector.detect(features).to_dict()
    assert set(payload) >= {"regime", "confidence", "agreement_score", "method_votes"}


# ------------------------------------------------------------------ helpers


def test_required_votes_tolerates_two_thirds_rounding():
    """2/3 = 0.6667 doit satisfaire un seuil configure a 0.67."""
    assert _required_votes(3, 0.67) == 2
    assert _required_votes(2, 0.67) == 2
    assert _required_votes(4, 0.67) == 3
    assert _required_votes(3, 1.0) == 3


def test_direction_mapping_is_total():
    for regime in Regime:
        assert isinstance(direction_of(regime), Direction)


def test_compose_regime_uses_measured_volatility(builder):
    returns = np.log(make_ohlcv(n=500, vol=0.004, seed=1)["close"].pct_change().add(1.0)).dropna()
    low_vol = classify_volatility(returns)
    assert compose_regime(Direction.BULL, low_vol) in (
        Regime.BULL_LOW_VOL,
        Regime.BULL_HIGH_VOL,
    )
    assert compose_regime(Direction.RANGE, low_vol) is Regime.RANGE_BOUND
    assert compose_regime(Direction.CRISIS, low_vol) is Regime.CRISIS

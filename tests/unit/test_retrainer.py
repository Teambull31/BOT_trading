"""Tests du retraining walk-forward.

L'enjeu n'est pas de trouver de bons parametres : c'est de REFUSER les mauvais.
Un retrainer qui accepte tout est plus dangereux qu'un retrainer inexistant.
"""

from __future__ import annotations

import json
import warnings

import pytest

from tests.conftest import make_ohlcv
from trader.adaptation.retrainer import WalkForwardRetrainer
from trader.config import RetrainingConfig, load_settings
from trader.data.features import FeatureBuilder
from trader.strategy.momentum import MomentumParams, MomentumStrategy

warnings.filterwarnings("ignore")


@pytest.fixture
def settings(tmp_path):
    return load_settings(
        "config/default.toml",
        overrides={"retraining": {"artifacts_dir": str(tmp_path / "retraining")}},
    )


@pytest.fixture
def market():
    frame = make_ohlcv(n=24 * 200, freq="1h", drift=0.0005, vol=0.01, seed=71)
    return frame, FeatureBuilder(timeframe="1h").build(frame)


@pytest.fixture
def strategy():
    return MomentumStrategy(MomentumParams())


def constant_score(value: float):
    """Score identique partout : aucun candidat ne peut se distinguer."""
    return lambda strategy, ohlcv, features: value


def overfitting_score(strategy, ohlcv, features):
    """Excellent en train, mauvais en validation : signature du sur-apprentissage."""
    return 10.0 if len(ohlcv) > 24 * 60 else -1.0


def test_short_history_is_rejected(settings, strategy):
    retrainer = WalkForwardRetrainer(settings)
    frame = make_ohlcv(n=200, freq="1h", seed=1)
    features = FeatureBuilder(timeframe="1h").build(frame)
    result = retrainer.retrain_strategy(strategy, frame, features, constant_score(1.0))
    assert not result.accepted
    assert "trop court" in result.reason


def test_overfitted_candidate_is_rejected(settings, strategy, market):
    """Un candidat brillant en train et nul en validation doit etre refuse."""
    frame, features = market
    retrainer = WalkForwardRetrainer(settings, max_candidates=6)
    result = retrainer.retrain_strategy(strategy, frame, features, overfitting_score)
    assert not result.accepted
    assert strategy.get_params() == result.old_params


def test_params_unchanged_when_nothing_beats_current(settings, strategy, market):
    frame, features = market
    before = strategy.get_params()
    retrainer = WalkForwardRetrainer(settings, max_candidates=5)
    result = retrainer.retrain_strategy(strategy, frame, features, constant_score(1.0))
    assert not result.accepted
    assert strategy.get_params() == before


def test_genuine_improvement_is_accepted(settings, market):
    """Un candidat meilleur en train ET en validation doit etre adopte."""
    frame, features = market
    strategy = MomentumStrategy(MomentumParams(adx_threshold=15.0))

    def score(candidate, ohlcv, features_slice) -> float:
        # Score maximal pour un seuil ADX eleve, identique en train et validation :
        # amelioration reelle et robuste, pas un artefact d'echantillon.
        return float(candidate.params.adx_threshold) / 35.0

    retrainer = WalkForwardRetrainer(settings, max_candidates=8)
    result = retrainer.retrain_strategy(strategy, frame, features, score)
    assert result.accepted
    assert strategy.params.adx_threshold > 15.0
    assert result.oos_ratio >= settings.retraining.min_oos_ratio


def test_negative_out_of_sample_is_rejected(settings, strategy, market):
    frame, features = market
    retrainer = WalkForwardRetrainer(settings, max_candidates=4)
    result = retrainer.retrain_strategy(strategy, frame, features, constant_score(-2.0))
    assert not result.accepted


def test_candidates_stay_within_bounded_space(settings, strategy, market):
    """Le retrainer n'explore jamais au-dela des bornes declarees."""
    retrainer = WalkForwardRetrainer(settings, max_candidates=50)
    space = strategy.param_space()
    for candidate in retrainer._candidates(strategy):
        for name, value in candidate.items():
            low, high = space[name]
            assert low <= value <= high


def test_candidate_budget_is_capped(settings, strategy):
    retrainer = WalkForwardRetrainer(settings, max_candidates=7, values_per_param=5)
    assert len(retrainer._candidates(strategy)) <= 7


def test_artifact_is_written_for_audit(settings, strategy, market):
    frame, features = market
    retrainer = WalkForwardRetrainer(settings, max_candidates=3)
    result = retrainer.retrain_strategy(strategy, frame, features, constant_score(1.0))
    artifacts = list(retrainer.artifacts_dir.glob("*.json"))
    assert artifacts
    payload = json.loads(artifacts[0].read_text())
    assert payload["strategy"] == strategy.name
    assert payload["version"] == result.version
    assert "reason" in payload


def test_regime_detector_retraining_is_traced(settings, market):
    from trader.regime.detector import RegimeDetector

    _, features = market
    detector = RegimeDetector(settings.regime, "1h")
    retrainer = WalkForwardRetrainer(settings)
    artifact = retrainer.retrain_regime_detector(detector, features)
    assert artifact["kind"] == "regime_detector"
    assert detector.last_fit is not None
    assert list(retrainer.artifacts_dir.glob("regime_*.json"))


def test_summary_reports_acceptance_rate(settings, strategy, market):
    frame, features = market
    retrainer = WalkForwardRetrainer(settings, max_candidates=3)
    retrainer.retrain_strategy(strategy, frame, features, constant_score(1.0))
    summary = retrainer.summary()
    assert summary["runs"] == 1
    assert 0.0 <= summary["acceptance_rate"] <= 1.0


def test_purge_gap_is_enforced_in_splits(settings):
    from trader.backtest.walk_forward import walk_forward_splits

    frame = make_ohlcv(n=24 * 200, freq="1h", seed=5)
    config = RetrainingConfig(purge_gap_days=2)
    for split in walk_forward_splits(frame.index, config):
        assert (split.validation_start - split.train_end).days == 2

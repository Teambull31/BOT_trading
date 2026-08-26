"""Tests du module anti-biais de confirmation.

Ce module doit trouver des raisons de NE PAS trader. Les tests verifient qu'il
en trouve quand il y en a, qu'il n'en invente pas quand tout va bien, et qu'il
ne peut pas etre desactive.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv
from trader.adaptation.devil_advocate import DevilAdvocate
from trader.config import DevilAdvocateConfig
from trader.data.features import FeatureBuilder
from trader.data.snapshot import build_snapshot
from trader.models import (
    ContraRecommendation,
    EnsembleDecision,
    Regime,
    RegimeState,
    Signal,
    StrategyHealth,
    StrategyOutput,
)

warnings.filterwarnings("ignore")


def make_snapshot(frame=None, funding=None, **kwargs):
    frame = frame if frame is not None else make_ohlcv(n=500, drift=0.001, vol=0.006, seed=7)
    funding_series = pd.Series(funding, index=frame.index) if funding is not None else None
    features = FeatureBuilder(timeframe="1h").build(frame, funding_rates=funding_series)
    return build_snapshot("ETH/USDT", frame, features, **kwargs)


def make_decision(
    signal: Signal = Signal.BUY,
    consensus: float = 1.0,
    dispersion: float = 0.0,
    weights: dict[str, float] | None = None,
    contra_per_strategy: int = 1,
    price: float = 2000.0,
) -> EnsembleDecision:
    contributions = [
        StrategyOutput(
            signal=signal,
            confidence=0.8,
            stop_loss=price * 0.97,
            reasoning="test",
            contra_evidence=[f"doute {i}" for i in range(contra_per_strategy)],
            regime_affinity=["bull_low_vol"],
            strategy_name=name,
            asset="ETH/USDT",
            entry_price=price,
        )
        for name in (weights or {"a": 0.5, "b": 0.5})
    ]
    return EnsembleDecision(
        asset="ETH/USDT",
        signal=signal,
        score=0.7,
        confidence=0.8,
        consensus=consensus,
        dispersion=dispersion,
        weights=weights or {"a": 0.5, "b": 0.5},
        contributions=contributions,
        stop_loss=price * 0.97,
        target_price=price * 1.06,
        entry_price=price,
    )


def make_regime(
    regime: Regime = Regime.BULL_LOW_VOL, transition: float = 0.05, confidence: float = 0.9
) -> RegimeState:
    return RegimeState(
        regime=regime,
        confidence=confidence,
        agreement_score=1.0,
        transition_probability=transition,
    )


# --------------------------------------------------------------- obligatoire


def test_cannot_be_disabled():
    config = DevilAdvocateConfig()
    object.__setattr__(config, "enabled", False)  # contournement force
    with pytest.raises(ValueError, match="obligatoire"):
        DevilAdvocate(config)


def test_neutral_signal_needs_no_audit():
    report = DevilAdvocate().review(make_decision(Signal.NEUTRAL), make_snapshot(), make_regime())
    assert report.recommendation is ContraRecommendation.PROCEED
    assert report.contra_score == 0.0


# ------------------------------------------------------------------ detection


def test_regime_transition_is_flagged():
    calm = DevilAdvocate().review(make_decision(), make_snapshot(), make_regime(transition=0.05))
    shaky = DevilAdvocate().review(make_decision(), make_snapshot(), make_regime(transition=0.95))
    assert shaky.contra_score > calm.contra_score
    assert any("regime instable" in signal for signal in shaky.contra_signals)


def test_uncertain_regime_is_flagged():
    report = DevilAdvocate().review(make_decision(), make_snapshot(), make_regime(Regime.UNCERTAIN))
    assert any("regime non identifie" in signal for signal in report.contra_signals)
    assert report.checks["regime_uncertainty"] > 0.5


def test_dead_dominant_strategy_is_flagged():
    """Si la strategie qui porte le trade est morte, c'est un signal d'alarme."""
    advocate = DevilAdvocate(health_provider=lambda name: StrategyHealth.DEAD)
    report = advocate.review(
        make_decision(weights={"momentum": 0.6, "autre": 0.4}), make_snapshot(), make_regime()
    )
    assert any("morte" in signal for signal in report.contra_signals)
    assert report.checks["strategy_decay"] == 1.0


def test_degrading_strategy_is_flagged():
    advocate = DevilAdvocate(health_provider=lambda name: StrategyHealth.DEGRADING)
    report = advocate.review(make_decision(), make_snapshot(), make_regime())
    assert any("DEGRADING" in signal for signal in report.contra_signals)


def test_health_provider_failure_is_contained():
    def broken(name: str):
        raise RuntimeError("registre indisponible")

    advocate = DevilAdvocate(health_provider=broken)
    report = advocate.review(make_decision(), make_snapshot(), make_regime())
    assert report.checks["strategy_decay"] == 0.0


def test_crowded_trade_via_funding_rate():
    """Un funding tres positif signale un trade long surpeuple."""
    frame = make_ohlcv(n=500, drift=0.001, seed=7)
    normal = np.full(len(frame), 0.0001)
    extreme = normal.copy()
    extreme[-30:] = 0.02  # funding qui explose a la hausse

    calm = DevilAdvocate().review(
        make_decision(), make_snapshot(frame, funding=normal), make_regime()
    )
    crowded = DevilAdvocate().review(
        make_decision(), make_snapshot(frame, funding=extreme), make_regime()
    )
    assert crowded.checks["crowded_trade"] > calm.checks["crowded_trade"]


def test_overextension_is_flagged():
    """Acheter un sommet a +3 sigma doit etre signale."""
    frame = make_ohlcv(n=500, drift=0.0005, vol=0.005, seed=8)
    frame.iloc[-1, frame.columns.get_loc("close")] *= 1.20
    frame.iloc[-1, frame.columns.get_loc("high")] *= 1.22
    report = DevilAdvocate().review(make_decision(), make_snapshot(frame), make_regime())
    assert report.checks["overextension"] > 0.0


def test_weak_consensus_is_flagged():
    strong = DevilAdvocate().review(
        make_decision(consensus=1.0, dispersion=0.0), make_snapshot(), make_regime()
    )
    weak = DevilAdvocate().review(
        make_decision(consensus=0.55, dispersion=0.9), make_snapshot(), make_regime()
    )
    assert weak.contra_score > strong.contra_score


def test_many_internal_doubts_raise_score():
    few = DevilAdvocate().review(
        make_decision(contra_per_strategy=1), make_snapshot(), make_regime()
    )
    many = DevilAdvocate().review(
        make_decision(contra_per_strategy=6), make_snapshot(), make_regime()
    )
    assert many.contra_score > few.contra_score


# ------------------------------------------------------------ recommandations


def test_high_score_aborts_trade():
    """Cumul de signaux defavorables : le trade doit etre annule."""
    advocate = DevilAdvocate(
        DevilAdvocateConfig(abort_threshold=0.35, reduce_threshold=0.15),
        health_provider=lambda name: StrategyHealth.DEAD,
    )
    report = advocate.review(
        make_decision(consensus=0.5, dispersion=1.0, contra_per_strategy=6),
        make_snapshot(),
        make_regime(Regime.UNCERTAIN, transition=0.95, confidence=0.2),
    )
    assert report.recommendation is ContraRecommendation.ABORT
    assert len(report.contra_signals) >= 3


def test_moderate_score_reduces_size():
    advocate = DevilAdvocate(DevilAdvocateConfig(abort_threshold=0.9, reduce_threshold=0.1))
    report = advocate.review(
        make_decision(consensus=0.6), make_snapshot(), make_regime(transition=0.7)
    )
    assert report.recommendation is ContraRecommendation.REDUCE_SIZE


def test_clean_setup_proceeds():
    """Sans rien a redire, le module laisse passer : il n'invente pas de doutes."""
    frame = make_ohlcv(n=600, drift=0.002, vol=0.004, seed=44)
    report = DevilAdvocate().review(
        make_decision(consensus=1.0, contra_per_strategy=1),
        make_snapshot(frame),
        make_regime(transition=0.02, confidence=0.95),
    )
    assert report.recommendation is ContraRecommendation.PROCEED
    assert report.contra_score < 0.4


def test_report_is_serializable():
    report = DevilAdvocate().review(make_decision(), make_snapshot(), make_regime())
    payload = report.to_dict()
    assert set(payload) >= {"contra_score", "recommendation", "contra_signals", "checks"}


def test_score_is_always_bounded():
    advocate = DevilAdvocate(health_provider=lambda name: StrategyHealth.DEAD)
    for transition in (0.0, 0.5, 1.0):
        report = advocate.review(
            make_decision(consensus=0.0, dispersion=2.0, contra_per_strategy=10),
            make_snapshot(),
            make_regime(Regime.UNCERTAIN, transition=transition, confidence=0.0),
        )
        assert 0.0 <= report.contra_score <= 1.0

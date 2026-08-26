"""Tests du meta-modele d'ensemble.

Les regles testees ici sont des garde-fous : moins de 2 strategies actives,
desaccord fort, poids cape a 40 %, Sharpe negatif -> shadow mode.
"""

from __future__ import annotations

import warnings

import pytest

from tests.conftest import make_ohlcv
from trader.config import EnsembleConfig
from trader.data.features import FeatureBuilder
from trader.data.snapshot import build_snapshot
from trader.models import Regime, RegimeState, Signal, StrategyHealth, StrategyOutput
from trader.strategy.base import BaseStrategy, StrategyParams
from trader.strategy.ensemble import StrategyEnsemble
from trader.strategy.mean_revert import MeanRevertStrategy
from trader.strategy.momentum import MomentumStrategy

warnings.filterwarnings("ignore")


class ScriptedStrategy(BaseStrategy):
    """Strategie de test : produit un signal impose."""

    def __init__(
        self,
        name: str,
        signal: Signal,
        confidence: float = 0.8,
        regimes: list[str] | None = None,
        stop_pct: float = 0.03,
        target_pct: float = 0.06,
        raises: bool = False,
    ) -> None:
        super().__init__(StrategyParams(min_confidence=0.0))
        self.name = name
        self._signal = signal
        self._confidence = confidence
        self._regimes = regimes or ["bull_low_vol", "range_bound"]
        self._stop_pct = stop_pct
        self._target_pct = target_pct
        self._raises = raises

    def get_required_regimes(self) -> list[str]:
        return self._regimes

    def generate_signal(self, data, regime) -> StrategyOutput:
        if self._raises:
            raise RuntimeError("panne simulee de strategie")
        if self._signal is Signal.NEUTRAL:
            return self.neutral(data.asset, "neutre par construction")
        direction = self._signal.direction
        return self.build_output(
            data=data,
            signal=self._signal,
            confidence=self._confidence,
            stop_loss=data.last_price * (1 - direction * self._stop_pct),
            target_price=data.last_price * (1 + direction * self._target_pct),
            reasoning=f"{self.name} scripte",
            contra_evidence=[f"{self.name}: incertitude de marche"],
        )


@pytest.fixture
def snapshot():
    frame = make_ohlcv(n=400, drift=0.002, vol=0.006, seed=8)
    features = FeatureBuilder(timeframe="1h").build(frame)
    return build_snapshot("ETH/USDT", frame, features)


def regime_state(regime: Regime = Regime.BULL_LOW_VOL, transition: float = 0.1) -> RegimeState:
    return RegimeState(
        regime=regime, confidence=0.85, agreement_score=1.0, transition_probability=transition
    )


# --------------------------------------------------------- regles de blocage


def test_single_active_strategy_blocks_trade(snapshot):
    """Une seule strategie active = pas de consensus possible = pas de trade."""
    ensemble = StrategyEnsemble(
        [
            ScriptedStrategy("solo", Signal.STRONG_BUY),
            ScriptedStrategy("autre_regime", Signal.STRONG_BUY, regimes=["crisis"]),
        ]
    )
    decision = ensemble.decide(snapshot, regime_state())
    assert not decision.is_actionable
    assert "minimum" in decision.blocked_reason or "consensus" in decision.blocked_reason


def test_strong_disagreement_blocks_trade(snapshot):
    """Deux strategies a l'oppose : le systeme s'abstient."""
    ensemble = StrategyEnsemble(
        [
            ScriptedStrategy("haussiere", Signal.STRONG_BUY),
            ScriptedStrategy("baissiere", Signal.STRONG_SELL),
        ],
        EnsembleConfig(max_signal_dispersion=0.5),
    )
    decision = ensemble.decide(snapshot, regime_state())
    assert not decision.is_actionable
    assert "desaccord" in decision.blocked_reason or "consensus" in decision.blocked_reason


def test_crisis_regime_blocks_all_trades(snapshot):
    ensemble = StrategyEnsemble(
        [
            ScriptedStrategy("a", Signal.STRONG_BUY, regimes=["crisis"]),
            ScriptedStrategy("b", Signal.STRONG_BUY, regimes=["crisis"]),
        ]
    )
    decision = ensemble.decide(snapshot, regime_state(Regime.CRISIS))
    assert not decision.is_actionable
    assert "crise" in decision.blocked_reason


def test_neutral_strategies_block_trade(snapshot):
    ensemble = StrategyEnsemble(
        [
            ScriptedStrategy("a", Signal.NEUTRAL),
            ScriptedStrategy("b", Signal.NEUTRAL),
        ]
    )
    assert not ensemble.decide(snapshot, regime_state()).is_actionable


def test_consensus_threshold_is_enforced(snapshot):
    """Deux pour, une contre : le consensus de 2/3 ne suffit pas si le seuil est a 0.8."""
    ensemble = StrategyEnsemble(
        [
            ScriptedStrategy("a", Signal.BUY),
            ScriptedStrategy("b", Signal.BUY),
            ScriptedStrategy("c", Signal.SELL),
        ],
        EnsembleConfig(consensus_threshold=0.8, max_signal_dispersion=5.0),
    )
    decision = ensemble.decide(snapshot, regime_state())
    assert not decision.is_actionable
    assert "consensus" in decision.blocked_reason


# ------------------------------------------------------------ decision valide


def test_agreeing_strategies_produce_signal(snapshot):
    ensemble = StrategyEnsemble(
        [
            ScriptedStrategy("a", Signal.STRONG_BUY),
            ScriptedStrategy("b", Signal.BUY),
            ScriptedStrategy("c", Signal.STRONG_BUY),
        ]
    )
    decision = ensemble.decide(snapshot, regime_state())
    assert decision.is_actionable
    assert decision.signal.direction > 0
    assert decision.consensus == pytest.approx(1.0)
    assert decision.stop_loss < decision.entry_price < decision.target_price


def test_most_conservative_stop_is_retained(snapshot):
    """En cas de desaccord sur le stop, on garde toujours le plus protecteur."""
    ensemble = StrategyEnsemble(
        [
            ScriptedStrategy("large", Signal.BUY, stop_pct=0.10),
            ScriptedStrategy("serre", Signal.BUY, stop_pct=0.02),
        ]
    )
    decision = ensemble.decide(snapshot, regime_state())
    assert decision.is_actionable
    assert decision.stop_loss == pytest.approx(snapshot.last_price * 0.98)


def test_short_decision_keeps_stop_above_entry(snapshot):
    ensemble = StrategyEnsemble(
        [
            ScriptedStrategy("a", Signal.STRONG_SELL, stop_pct=0.05),
            ScriptedStrategy("b", Signal.STRONG_SELL, stop_pct=0.02),
        ]
    )
    decision = ensemble.decide(snapshot, regime_state())
    assert decision.is_actionable
    assert decision.signal.direction < 0
    assert decision.stop_loss == pytest.approx(snapshot.last_price * 1.02)


# ---------------------------------------------------------------- ponderation


def test_weights_are_capped_at_40_percent(snapshot):
    """Le systeme ne peut jamais dependre d'une seule strategie."""

    def metrics(name: str) -> dict[str, float]:
        return {"sharpe_30d": 5.0 if name == "star" else 0.1, "hit_rate": 0.9}

    ensemble = StrategyEnsemble(
        [
            ScriptedStrategy("star", Signal.BUY),
            ScriptedStrategy("b", Signal.BUY),
            ScriptedStrategy("c", Signal.BUY),
        ],
        metrics_provider=metrics,
    )
    decision = ensemble.decide(snapshot, regime_state())
    assert max(decision.weights.values()) <= 0.40 + 1e-9
    assert sum(decision.weights.values()) == pytest.approx(1.0)


def test_negative_sharpe_zeroes_weight_and_enables_shadow(snapshot):
    def metrics(name: str) -> dict[str, float]:
        return {"sharpe_30d": -1.5 if name == "perdante" else 1.0, "hit_rate": 0.5}

    ensemble = StrategyEnsemble(
        [
            ScriptedStrategy("perdante", Signal.BUY),
            ScriptedStrategy("bonne", Signal.BUY),
            ScriptedStrategy("autre", Signal.BUY),
        ],
        metrics_provider=metrics,
    )
    ensemble.decide(snapshot, regime_state())
    assert ensemble.records["perdante"].weight == 0.0
    assert ensemble.records["perdante"].shadow is True
    # Elle continue de tourner : son dernier signal est bien enregistre.
    assert ensemble.records["perdante"].last_output is not None


def test_dead_strategy_keeps_producing_shadow_signals(snapshot):
    """Une strategie morte n'est jamais supprimee : les regimes changent."""
    ensemble = StrategyEnsemble(
        [
            ScriptedStrategy("morte", Signal.STRONG_BUY),
            ScriptedStrategy("a", Signal.BUY),
            ScriptedStrategy("b", Signal.BUY),
        ]
    )
    ensemble.set_health("morte", StrategyHealth.DEAD)
    decision = ensemble.decide(snapshot, regime_state())
    assert "morte" not in decision.weights
    assert ensemble.records["morte"].last_output.signal is Signal.STRONG_BUY
    assert "morte" in ensemble.snapshot()


def test_metrics_provider_failure_does_not_break_decision(snapshot):
    def broken(name: str) -> dict[str, float]:
        raise RuntimeError("base de metriques indisponible")

    ensemble = StrategyEnsemble(
        [ScriptedStrategy("a", Signal.BUY), ScriptedStrategy("b", Signal.BUY)],
        metrics_provider=broken,
    )
    assert ensemble.decide(snapshot, regime_state()).is_actionable


def test_strategy_exception_is_contained(snapshot):
    """Une strategie qui plante ne tue pas la boucle de trading."""
    ensemble = StrategyEnsemble(
        [
            ScriptedStrategy("cassee", Signal.BUY, raises=True),
            ScriptedStrategy("a", Signal.BUY),
            ScriptedStrategy("b", Signal.BUY),
        ]
    )
    decision = ensemble.decide(snapshot, regime_state())
    assert decision.is_actionable
    assert ensemble.records["cassee"].last_output.signal is Signal.NEUTRAL


def test_flip_flop_penalizes_weight(snapshot):
    stable = ScriptedStrategy("stable", Signal.BUY)
    unstable = ScriptedStrategy("instable", Signal.BUY)
    ensemble = StrategyEnsemble([stable, unstable, ScriptedStrategy("c", Signal.BUY)])
    for value in [1.0, -1.0] * 10:
        ensemble.records["instable"].signal_history.append(value)
    for _ in range(20):
        ensemble.records["stable"].signal_history.append(1.0)
    weights = ensemble.compute_weights(list(ensemble.records.values()))
    assert weights["instable"] < weights["stable"]


# ------------------------------------------------------- integration reelle


def test_real_strategies_are_gated_by_regime():
    """En range, le momentum ne vote pas ; en tendance, la mean-reversion non plus."""
    ensemble = StrategyEnsemble([MomentumStrategy(), MeanRevertStrategy()])
    frame = make_ohlcv(n=500, drift=0.003, vol=0.005, seed=9)
    features = FeatureBuilder(timeframe="1h").build(frame)
    data = build_snapshot("ETH/USDT", frame, features)

    bull_active = {r.name for r in ensemble.eligible(regime_state(Regime.BULL_LOW_VOL))}
    range_active = {r.name for r in ensemble.eligible(regime_state(Regime.RANGE_BOUND))}
    assert bull_active == {"momentum"}
    assert range_active == {"mean_revert"}

    # Avec une seule strategie eligible par regime, aucun trade n'est possible.
    assert not ensemble.decide(data, regime_state(Regime.BULL_LOW_VOL)).is_actionable


def test_empty_pool_is_rejected():
    with pytest.raises(ValueError, match="au moins une strategie"):
        StrategyEnsemble([])


def test_duplicate_names_are_rejected():
    with pytest.raises(ValueError, match="dupliques"):
        StrategyEnsemble([ScriptedStrategy("x", Signal.BUY), ScriptedStrategy("x", Signal.BUY)])

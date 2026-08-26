"""Meta-modele d'ensemble : pondere, active et desactive les strategies.

Le principe directeur du systeme : AUCUNE STRATEGIE N'EST ETERNELLE. L'ensemble
n'a donc pas de favorite. Il applique des regles strictes :

- une strategie ne vote que si le regime courant figure dans ses regimes autorises ;
- moins de 2 strategies actives -> aucun trade (pas de consensus possible) ;
- desaccord fort entre strategies -> aucun trade ;
- le poids d'une strategie est cape a 40 % : le systeme ne peut jamais dependre
  d'une seule ;
- Sharpe glissant 30 j negatif -> poids 0, mais la strategie CONTINUE de tourner
  en shadow mode. Les regimes changent : une strategie morte aujourd'hui peut
  redevenir pertinente, et ses signaux fantomes servent a le detecter.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trader.config import EnsembleConfig
from trader.logging_setup import get_logger
from trader.models import (
    EnsembleDecision,
    MarketSnapshot,
    RegimeState,
    Signal,
    StrategyHealth,
    StrategyOutput,
)
from trader.strategy.base import BaseStrategy
from trader.utils.math_utils import EPSILON, normalize_weights

log = get_logger(__name__)

MetricsProvider = Callable[[str], dict[str, float]]
"""Fournit les metriques de performance d'une strategie (Sharpe, hit rate...)."""


@dataclass(slots=True)
class StrategyRecord:
    """Etat runtime d'une strategie dans l'ensemble."""

    strategy: BaseStrategy
    weight: float = 0.0
    health: StrategyHealth = StrategyHealth.HEALTHY
    shadow: bool = False
    signal_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    last_output: StrategyOutput | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Nom de la strategie."""
        return self.strategy.name

    @property
    def votes(self) -> bool:
        """Vrai si la strategie a le droit de peser sur la decision finale."""
        return not self.shadow and self.health is not StrategyHealth.DEAD

    def flip_flop_rate(self) -> float:
        """Frequence de changement de sens du signal (instabilite)."""
        history = [value for value in self.signal_history if abs(value) > EPSILON]
        if len(history) < 3:
            return 0.0
        flips = sum(1 for a, b in zip(history, history[1:], strict=False) if a * b < 0)
        return flips / (len(history) - 1)


class StrategyEnsemble:
    """Agrege les signaux d'un pool de strategies en une decision unique."""

    def __init__(
        self,
        strategies: Sequence[BaseStrategy],
        config: EnsembleConfig | None = None,
        metrics_provider: MetricsProvider | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("l'ensemble exige au moins une strategie")
        self.config = config or EnsembleConfig()
        self.metrics_provider = metrics_provider
        self.records: dict[str, StrategyRecord] = {
            strategy.name: StrategyRecord(strategy=strategy) for strategy in strategies
        }
        if len(self.records) != len(strategies):
            raise ValueError("noms de strategies dupliques dans l'ensemble")

    # ------------------------------------------------------------- pool

    def add(self, strategy: BaseStrategy) -> None:
        """Ajoute une strategie au pool."""
        if strategy.name in self.records:
            raise ValueError(f"strategie deja presente : {strategy.name}")
        self.records[strategy.name] = StrategyRecord(strategy=strategy)

    def set_health(self, name: str, health: StrategyHealth) -> None:
        """Met a jour l'etat de sante d'une strategie (appele par le decay detector).

        Une strategie DEAD n'est jamais supprimee : elle passe en shadow mode et
        continue de produire des signaux, logges pour analyse.
        """
        record = self.records[name]
        record.health = health
        record.shadow = health in (StrategyHealth.DEAD, StrategyHealth.ZOMBIE)
        log.info(
            "strategy_health_updated",
            strategy=name,
            health=health.value,
            shadow=record.shadow,
        )

    def eligible(self, regime: RegimeState) -> list[StrategyRecord]:
        """Strategies autorisees a trader dans le regime courant (hors shadow)."""
        return [
            record
            for record in self.records.values()
            if record.votes and record.strategy.is_active_in(regime)
        ]

    # ------------------------------------------------------------- decision

    def decide(self, data: MarketSnapshot, regime: RegimeState) -> EnsembleDecision:
        """Produit la decision agregee de l'ensemble pour un actif."""
        outputs = self._collect_outputs(data, regime)
        voting = [
            (record, output)
            for record, output in outputs
            if record.votes and record.strategy.is_active_in(regime)
        ]

        if regime.is_crisis:
            return self._blocked(data, "regime de crise : aucune nouvelle position", outputs)

        actionable = [(record, output) for record, output in voting if output.is_actionable]
        if len(voting) < self.config.min_active_strategies:
            return self._blocked(
                data,
                f"{len(voting)} strategie(s) active(s) < minimum "
                f"{self.config.min_active_strategies} : consensus impossible",
                outputs,
            )
        if len(actionable) < self.config.min_active_strategies:
            return self._blocked(
                data,
                f"seulement {len(actionable)} signal(aux) directionnel(s) : "
                "pas assez pour un consensus",
                outputs,
            )

        weights = self.compute_weights([record for record, _ in actionable])
        score, dispersion, consensus = self._aggregate(actionable, weights)

        if dispersion > self.config.max_signal_dispersion:
            return self._blocked(
                data,
                f"desaccord fort entre strategies (dispersion {dispersion:.2f} > "
                f"{self.config.max_signal_dispersion:.2f})",
                outputs,
                weights=weights,
                score=score,
                dispersion=dispersion,
                consensus=consensus,
            )
        if consensus < self.config.consensus_threshold:
            return self._blocked(
                data,
                f"consensus insuffisant ({consensus:.2f} < {self.config.consensus_threshold:.2f})",
                outputs,
                weights=weights,
                score=score,
                dispersion=dispersion,
                consensus=consensus,
            )

        signal = Signal.from_score(score * 2.0)
        if signal is Signal.NEUTRAL:
            return self._blocked(
                data,
                f"score agrege trop faible ({score:+.2f})",
                outputs,
                weights=weights,
                score=score,
                dispersion=dispersion,
                consensus=consensus,
            )

        direction = signal.direction
        agreeing = [
            (record, output)
            for record, output in actionable
            if output.signal.direction == direction
        ]
        stop_loss = self._consensus_stop(agreeing, data.last_price, direction)
        target = self._consensus_target(agreeing, weights, direction)
        confidence = float(
            sum(weights.get(record.name, 0.0) * output.confidence for record, output in agreeing)
        )

        decision = EnsembleDecision(
            asset=data.asset,
            signal=signal,
            score=score,
            confidence=min(1.0, confidence),
            consensus=consensus,
            dispersion=dispersion,
            weights=weights,
            contributions=[output for _, output in outputs],
            stop_loss=stop_loss,
            target_price=target,
            entry_price=data.last_price,
            timestamp=data.timestamp,
        )
        log.info("ensemble_decision", **decision.to_dict())
        return decision

    def _collect_outputs(
        self, data: MarketSnapshot, regime: RegimeState
    ) -> list[tuple[StrategyRecord, StrategyOutput]]:
        """Interroge TOUTES les strategies, y compris celles en shadow mode.

        Les strategies inactives sont interrogees quand meme : leurs signaux sont
        traces pour mesurer ce qu'elles auraient fait. C'est ce qui permet de
        ressusciter une strategie quand le regime redevient favorable.
        """
        outputs: list[tuple[StrategyRecord, StrategyOutput]] = []
        for record in self.records.values():
            try:
                output = record.strategy.generate_signal(data, regime)
            except Exception as exc:  # noqa: BLE001 - une strategie ne doit jamais tuer la boucle
                log.error(
                    "strategy_failed",
                    strategy=record.name,
                    asset=data.asset,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                output = record.strategy.neutral(data.asset, f"erreur interne : {exc}")
            record.last_output = output
            record.signal_history.append(float(output.signal.value))
            if record.shadow or not record.strategy.is_active_in(regime):
                log.debug(
                    "shadow_signal",
                    strategy=record.name,
                    asset=data.asset,
                    signal=output.signal.name,
                    regime=regime.regime.value,
                    reason="shadow" if record.shadow else "regime_mismatch",
                )
            outputs.append((record, output))
        return outputs

    def compute_weights(self, records: Sequence[StrategyRecord]) -> dict[str, float]:
        """Calcule le meta-score de chaque strategie et en deduit les poids.

        Composantes :
        - performance recente (Sharpe glissant 30 j) ;
        - taux de predictions correctes (hit rate) ;
        - bonus de diversification (decorrelation vis-a-vis des autres) ;
        - penalite d'instabilite (strategie qui change d'avis sans arret).
        """
        raw: dict[str, float] = {}
        for record in records:
            metrics = self._metrics(record)
            record.metrics = metrics
            sharpe = metrics.get("sharpe_30d", 0.0)

            if sharpe < 0.0:
                # Regle non negociable : Sharpe negatif -> poids nul, shadow mode.
                raw[record.name] = 0.0
                if not record.shadow:
                    record.shadow = True
                    log.warning(
                        "strategy_zero_weight",
                        strategy=record.name,
                        sharpe_30d=round(sharpe, 3),
                        reason="sharpe glissant negatif",
                    )
                continue

            performance = float(np.clip(0.5 + sharpe / 4.0, 0.1, 1.5))
            hit_rate = metrics.get("hit_rate", 0.5)
            accuracy = float(np.clip(0.5 + (hit_rate - 0.5) * 2.0, 0.2, 1.5))
            diversification = 1.0 + self.config.diversification_bonus * self._diversification(
                record, records
            )
            stability = 1.0 - self.config.flip_flop_penalty * record.flip_flop_rate()
            raw[record.name] = max(0.0, performance * accuracy * diversification * stability)

        weights = normalize_weights(raw, cap=self.config.max_weight_single)
        for record in records:
            record.weight = weights.get(record.name, 0.0)
        return weights

    def _metrics(self, record: StrategyRecord) -> dict[str, float]:
        """Metriques de performance d'une strategie (via le provider injecte).

        Sans provider (demarrage a froid), toutes les strategies sont a egalite :
        on ne prete a aucune une performance qu'on n'a pas mesuree.
        """
        if self.metrics_provider is None:
            return {"sharpe_30d": 0.0, "hit_rate": 0.5, "trades": 0.0}
        try:
            return dict(self.metrics_provider(record.name))
        except Exception as exc:  # noqa: BLE001 - le provider ne doit pas casser la boucle
            log.error("metrics_provider_failed", strategy=record.name, error=str(exc))
            return {"sharpe_30d": 0.0, "hit_rate": 0.5, "trades": 0.0}

    def _diversification(self, record: StrategyRecord, peers: Sequence[StrategyRecord]) -> float:
        """Bonus de decorrelation : 0 (clone des autres) a 1 (totalement decorrelee)."""
        own = list(record.signal_history)
        correlations: list[float] = []
        for peer in peers:
            if peer.name == record.name:
                continue
            other = list(peer.signal_history)
            length = min(len(own), len(other))
            if length < 5:
                continue
            left = np.asarray(own[-length:], dtype=float)
            right = np.asarray(other[-length:], dtype=float)
            if left.std() < EPSILON or right.std() < EPSILON:
                continue
            correlations.append(abs(float(np.corrcoef(left, right)[0, 1])))
        if not correlations:
            return 0.5
        return float(np.clip(1.0 - np.mean(correlations), 0.0, 1.0))

    def _aggregate(
        self,
        actionable: Sequence[tuple[StrategyRecord, StrategyOutput]],
        weights: dict[str, float],
    ) -> tuple[float, float, float]:
        """Agrege les signaux : (score pondere, dispersion, consensus)."""
        values: list[float] = []
        used_weights: list[float] = []
        score = 0.0
        for record, output in actionable:
            weight = weights.get(record.name, 0.0)
            # Signal normalise dans [-1, 1] et module par la confiance propre.
            value = (output.signal.value / 2.0) * output.confidence
            score += weight * value
            values.append(output.signal.value / 2.0)
            used_weights.append(weight)

        if not values or sum(used_weights) < EPSILON:
            return 0.0, float("inf"), 0.0

        array = np.asarray(values)
        weight_array = np.asarray(used_weights)
        mean = float(np.average(array, weights=weight_array))
        dispersion = float(np.sqrt(np.average((array - mean) ** 2, weights=weight_array)))

        direction = np.sign(score)
        if abs(direction) < EPSILON:
            return score, dispersion, 0.0
        consensus = float(
            sum(
                weight
                for (record, output), weight in zip(actionable, used_weights, strict=True)
                if np.sign(output.signal.value) == direction
            )
            / sum(used_weights)
        )
        return score, dispersion, consensus

    @staticmethod
    def _consensus_stop(
        agreeing: Sequence[tuple[StrategyRecord, StrategyOutput]],
        price: float,
        direction: int,
    ) -> float | None:
        """Stop loss retenu : le PLUS PRUDENT des stops proposes.

        En cas de desaccord sur le stop, on prend systematiquement celui qui
        protege le plus. Le risque prime sur l'esperance de gain.
        """
        stops = [output.stop_loss for _, output in agreeing if output.stop_loss > 0]
        if not stops:
            return None
        return max(stops) if direction > 0 else min(stops)

    @staticmethod
    def _consensus_target(
        agreeing: Sequence[tuple[StrategyRecord, StrategyOutput]],
        weights: dict[str, float],
        direction: int,
    ) -> float | None:
        """Cible retenue : moyenne ponderee des cibles proposees."""
        pairs = [
            (weights.get(record.name, 0.0), output.target_price)
            for record, output in agreeing
            if output.target_price is not None and output.target_price > 0
        ]
        pairs = [(weight, target) for weight, target in pairs if weight > 0]
        if not pairs:
            return None
        total = sum(weight for weight, _ in pairs)
        if total < EPSILON:
            return None
        return float(sum(weight * target for weight, target in pairs) / total)

    def _blocked(
        self,
        data: MarketSnapshot,
        reason: str,
        outputs: Sequence[tuple[StrategyRecord, StrategyOutput]],
        weights: dict[str, float] | None = None,
        score: float = 0.0,
        dispersion: float = 0.0,
        consensus: float = 0.0,
    ) -> EnsembleDecision:
        """Construit une decision de non-trade documentee."""
        decision = EnsembleDecision(
            asset=data.asset,
            signal=Signal.NEUTRAL,
            score=score,
            confidence=0.0,
            consensus=consensus,
            dispersion=dispersion,
            weights=weights or {},
            contributions=[output for _, output in outputs],
            entry_price=data.last_price,
            blocked_reason=reason,
            timestamp=data.timestamp,
        )
        log.info("ensemble_no_trade", asset=data.asset, reason=reason)
        return decision

    # -------------------------------------------------------------- etat

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Etat courant de toutes les strategies (pour monitoring et persistence)."""
        return {
            record.name: {
                "weight": round(record.weight, 4),
                "health": record.health.value,
                "shadow": record.shadow,
                "flip_flop_rate": round(record.flip_flop_rate(), 3),
                "regimes": record.strategy.get_required_regimes(),
                "params": record.strategy.get_params(),
                "metrics": {k: round(v, 4) for k, v in record.metrics.items()},
                "last_signal": record.last_output.signal.name if record.last_output else None,
            }
            for record in self.records.values()
        }

"""Retraining walk-forward automatique.

Le retraining est l'operation la plus dangereuse du systeme : c'est le moment ou
l'on peut, en toute bonne foi, remplacer un modele mediocre mais honnete par un
modele sur-appris qui va exceller sur le passe et echouer sur le futur.

Six garde-fous, tous obligatoires :

1. FENETRES DISJOINTES avec purge. Le train et la validation ne se touchent
   jamais ; un gap absorbe les features a fenetre longue.
2. ESPACE DE RECHERCHE BORNE. Chaque strategie declare des bornes etroites
   (`param_space`). On n'explore pas au-dela.
3. BUDGET D'ESSAIS LIMITE. Plus on teste de combinaisons, plus on a de chances
   d'en trouver une bonne par hasard. Le budget est plafonne.
4. SEUIL OUT-OF-SAMPLE. La performance hors echantillon doit atteindre au moins
   `min_oos_ratio` de la performance en echantillon, sinon on rejette.
5. COMPARAISON AU MODELE ACTUEL ET AU BENCHMARK. Un nouveau modele qui ne bat
   pas l'ancien en out-of-sample n'est pas adopte. On garde l'ancien.
6. TRACABILITE. Chaque retraining ecrit un artefact JSON versionne : parametres
   testes, retenus, performances, verdict.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trader.backtest.walk_forward import WalkForwardSplit, walk_forward_splits
from trader.config import RetrainingConfig, Settings
from trader.logging_setup import get_logger
from trader.strategy.base import BaseStrategy
from trader.utils.time_utils import utc_now

log = get_logger(__name__)

ScoreFunction = Callable[[BaseStrategy, pd.DataFrame, pd.DataFrame], float]
"""Evalue une strategie parametree sur (ohlcv, features) et rend un score."""


@dataclass(slots=True)
class RetrainingResult:
    """Resultat d'un retraining pour une strategie."""

    strategy: str
    accepted: bool
    reason: str
    old_params: dict[str, float] = field(default_factory=dict)
    new_params: dict[str, float] = field(default_factory=dict)
    in_sample_score: float = 0.0
    out_of_sample_score: float = 0.0
    baseline_score: float = 0.0
    oos_ratio: float = 0.0
    candidates_tested: int = 0
    folds: int = 0
    timestamp: datetime = field(default_factory=utc_now)
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Representation serialisable pour l'artefact d'audit."""
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


class WalkForwardRetrainer:
    """Reoptimise les strategies et le detecteur de regime en walk-forward."""

    def __init__(
        self,
        settings: Settings,
        config: RetrainingConfig | None = None,
        max_candidates: int = 24,
        values_per_param: int = 3,
    ) -> None:
        self.settings = settings
        self.config = config or settings.retraining
        self.max_candidates = max_candidates
        self.values_per_param = values_per_param
        self.artifacts_dir = Path(self.config.artifacts_dir)
        self.history: list[RetrainingResult] = []

    # ------------------------------------------------------------ strategies

    def retrain_strategy(
        self,
        strategy: BaseStrategy,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        score_function: ScoreFunction,
        now: datetime | None = None,
    ) -> RetrainingResult:
        """Cherche de meilleurs hyperparametres, et n'adopte que s'ils tiennent hors echantillon."""
        reference = now or utc_now()
        old_params = strategy.get_params()
        splits = walk_forward_splits(ohlcv.index, self.config)

        if not splits:
            return self._reject(
                strategy, old_params, "historique trop court pour un walk-forward", reference
            )

        candidates = self._candidates(strategy)
        if not candidates:
            return self._reject(strategy, old_params, "aucun hyperparametre a optimiser", reference)

        baseline = self._score_across_folds(
            strategy, old_params, splits, ohlcv, features, score_function
        )
        best_params = old_params
        best_in_sample = baseline["in_sample"]
        best_out_of_sample = baseline["out_of_sample"]
        tested = 0

        for params in candidates:
            tested += 1
            scores = self._score_across_folds(
                strategy, params, splits, ohlcv, features, score_function
            )
            if scores["out_of_sample"] > best_out_of_sample:
                best_params = params
                best_in_sample = scores["in_sample"]
                best_out_of_sample = scores["out_of_sample"]

        strategy.set_params(old_params)  # on ne modifie rien avant validation
        oos_ratio = best_out_of_sample / best_in_sample if abs(best_in_sample) > 1e-9 else 0.0

        result = RetrainingResult(
            strategy=strategy.name,
            accepted=False,
            reason="",
            old_params=old_params,
            new_params=best_params,
            in_sample_score=best_in_sample,
            out_of_sample_score=best_out_of_sample,
            baseline_score=baseline["out_of_sample"],
            oos_ratio=oos_ratio,
            candidates_tested=tested,
            folds=len(splits),
            timestamp=reference,
            version=reference.strftime("%Y%m%dT%H%M%S"),
        )

        if best_params == old_params:
            result.reason = "aucun jeu de parametres ne bat l'actuel hors echantillon"
        elif best_out_of_sample <= 0:
            result.reason = f"performance out-of-sample non positive ({best_out_of_sample:.3f})"
        elif oos_ratio < self.config.min_oos_ratio:
            result.reason = (
                f"ratio OOS/IS de {oos_ratio:.2f} sous le seuil "
                f"{self.config.min_oos_ratio:.2f} : signature de sur-apprentissage"
            )
        elif best_out_of_sample <= baseline["out_of_sample"]:
            result.reason = (
                f"le candidat ({best_out_of_sample:.3f}) ne bat pas le modele actuel "
                f"({baseline['out_of_sample']:.3f})"
            )
        else:
            strategy.set_params(best_params)
            result.accepted = True
            result.reason = (
                f"adopte : OOS {best_out_of_sample:.3f} > actuel "
                f"{baseline['out_of_sample']:.3f}, ratio OOS/IS {oos_ratio:.2f}"
            )

        self.history.append(result)
        self._write_artifact(result)
        log.info("strategy_retrained", **{k: str(v) for k, v in result.to_dict().items()})
        return result

    def _candidates(self, strategy: BaseStrategy) -> list[dict[str, float]]:
        """Genere les combinaisons a tester dans l'espace BORNE de la strategie.

        Le budget est plafonne : chaque essai supplementaire augmente la chance
        de trouver un bon score par pur hasard.
        """
        space = strategy.param_space()
        current = strategy.get_params()
        grids: dict[str, list[float]] = {}
        for name, (low, high) in space.items():
            if name not in current:
                continue
            grids[name] = [float(value) for value in np.linspace(low, high, self.values_per_param)]
        if not grids:
            return []

        names = sorted(grids)[: self.settings.retraining.max_params_per_strategy]
        combinations = list(itertools.product(*(grids[name] for name in names)))
        if len(combinations) > self.max_candidates:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(combinations), size=self.max_candidates, replace=False)
            combinations = [combinations[int(i)] for i in sorted(indices)]
        return [dict(zip(names, combo, strict=True)) for combo in combinations]

    def _score_across_folds(
        self,
        strategy: BaseStrategy,
        params: dict[str, float],
        splits: Sequence[WalkForwardSplit],
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        score_function: ScoreFunction,
    ) -> dict[str, float]:
        """Evalue un jeu de parametres sur toutes les fenetres."""
        try:
            strategy.set_params(params)
        except ValueError as exc:
            log.debug("invalid_params_skipped", strategy=strategy.name, error=str(exc))
            return {"in_sample": float("-inf"), "out_of_sample": float("-inf")}

        in_sample: list[float] = []
        out_of_sample: list[float] = []
        for split in splits:
            train_ohlcv = split.slice_train(ohlcv)
            validation_ohlcv = split.slice_validation(ohlcv)
            if len(train_ohlcv) < 50 or len(validation_ohlcv) < 20:
                continue
            in_sample.append(score_function(strategy, train_ohlcv, split.slice_train(features)))
            out_of_sample.append(
                score_function(strategy, validation_ohlcv, split.slice_validation(features))
            )
        if not out_of_sample:
            return {"in_sample": float("-inf"), "out_of_sample": float("-inf")}
        return {
            "in_sample": float(np.mean(in_sample)) if in_sample else 0.0,
            "out_of_sample": float(np.mean(out_of_sample)),
        }

    def _reject(
        self,
        strategy: BaseStrategy,
        old_params: dict[str, float],
        reason: str,
        now: datetime,
    ) -> RetrainingResult:
        """Construit un resultat de rejet documente."""
        result = RetrainingResult(
            strategy=strategy.name,
            accepted=False,
            reason=reason,
            old_params=old_params,
            new_params=old_params,
            timestamp=now,
            version=now.strftime("%Y%m%dT%H%M%S"),
        )
        self.history.append(result)
        self._write_artifact(result)
        log.warning("retraining_skipped", strategy=strategy.name, reason=reason)
        return result

    # --------------------------------------------------------------- regime

    def retrain_regime_detector(
        self, detector: Any, features: pd.DataFrame, now: datetime | None = None
    ) -> dict[str, Any]:
        """Reentraine le detecteur de regime sur la fenetre d'entrainement.

        Le detecteur n'a pas de performance out-of-sample directement mesurable
        (il n'y a pas de "vrai" label de regime), donc on ne peut pas le
        selectionner comme une strategie. On le reentraine sur des donnees
        recentes et on trace le resultat.
        """
        reference = now or utc_now()
        window = int(self.config.train_window_days * 24)
        report = detector.fit(features.tail(max(window, 200)), now=reference)
        artifact = {
            "kind": "regime_detector",
            "timestamp": reference.isoformat(),
            "version": reference.strftime("%Y%m%dT%H%M%S"),
            "report": report,
        }
        self._write_json(f"regime_{artifact['version']}.json", artifact)
        log.info("regime_detector_retrained", **{k: str(v) for k, v in report.items()})
        return artifact

    # ------------------------------------------------------------ artefacts

    def _write_artifact(self, result: RetrainingResult) -> None:
        """Ecrit l'artefact d'audit du retraining."""
        self._write_json(f"{result.strategy}_{result.version}.json", result.to_dict())

    def _write_json(self, filename: str, payload: dict[str, Any]) -> None:
        """Ecrit un artefact JSON, sans jamais casser le retraining en cas d'echec disque."""
        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            path = self.artifacts_dir / filename
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            log.error("artifact_write_failed", filename=filename, error=str(exc))

    def summary(self) -> dict[str, Any]:
        """Bilan des retrainings effectues."""
        accepted = [result for result in self.history if result.accepted]
        return {
            "runs": len(self.history),
            "accepted": len(accepted),
            "rejected": len(self.history) - len(accepted),
            "acceptance_rate": (len(accepted) / len(self.history) if self.history else 0.0),
            "last_reasons": [result.reason for result in self.history[-5:]],
        }

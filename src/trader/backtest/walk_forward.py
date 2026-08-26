"""Decoupage walk-forward avec purge (anti data-leakage).

Un backtest in-sample ne prouve rien. Le walk-forward avance dans le temps :
on entraine sur une fenetre passee, on valide sur la fenetre suivante JAMAIS vue,
puis on decale. Entre les deux, un `purge_gap` de quelques jours coupe la
contamination : les features a fenetre longue (vol 90 j, Hurst 100 barres)
chevauchent la frontiere et fuiteraient sinon de l'information d'entrainement
dans la validation.

    |<--- train 90j --->|<-gap 2j->|<- valid 14j ->|<- test 7j ->|
                                    (out-of-sample stricte)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from trader.config import RetrainingConfig
from trader.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    """Une fenetre walk-forward : train / (gap purge) / validation / test."""

    index: int
    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime
    test_end: datetime

    def slice_train(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Sous-ensemble d'entrainement."""
        return frame.loc[self.train_start : self.train_end]

    def slice_validation(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Sous-ensemble de validation (strictement out-of-sample)."""
        return frame.loc[self.validation_start : self.validation_end]

    def slice_test(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Sous-ensemble de test final."""
        return frame.loc[self.test_start : self.test_end]

    def to_dict(self) -> dict[str, str]:
        """Representation serialisable des bornes."""
        return {
            "index": str(self.index),
            "train": f"{self.train_start.date()} -> {self.train_end.date()}",
            "validation": f"{self.validation_start.date()} -> {self.validation_end.date()}",
            "test": f"{self.test_start.date()} -> {self.test_end.date()}",
        }


@dataclass(slots=True)
class WalkForwardResult:
    """Agregation des resultats de toutes les fenetres."""

    splits: list[WalkForwardSplit] = field(default_factory=list)
    in_sample: list[dict[str, float]] = field(default_factory=list)
    out_of_sample: list[dict[str, float]] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    def mean_metric(self, key: str, out_of_sample: bool = True) -> float:
        """Moyenne d'une metrique sur toutes les fenetres."""
        source = self.out_of_sample if out_of_sample else self.in_sample
        values = [float(item.get(key, 0.0)) for item in source if key in item]
        return sum(values) / len(values) if values else 0.0

    def oos_ratio(self, key: str = "sharpe") -> float:
        """Rapport performance out-of-sample / in-sample.

        C'est LE chiffre qui compte : un ratio proche de 1 signale un modele
        robuste, un ratio faible ou negatif signale du sur-apprentissage.
        """
        in_sample = self.mean_metric(key, out_of_sample=False)
        if abs(in_sample) < 1e-9:
            return 0.0
        return self.mean_metric(key, out_of_sample=True) / in_sample

    def is_robust(self, min_ratio: float = 0.70, key: str = "sharpe") -> bool:
        """Vrai si la performance out-of-sample tient ses promesses."""
        return self.mean_metric(key) > 0 and self.oos_ratio(key) >= min_ratio

    def summary(self) -> dict[str, float]:
        """Resume chiffre du walk-forward."""
        return {
            "folds": float(len(self.splits)),
            "sharpe_in_sample": round(self.mean_metric("sharpe", False), 3),
            "sharpe_out_of_sample": round(self.mean_metric("sharpe"), 3),
            "oos_ratio": round(self.oos_ratio(), 3),
            "return_out_of_sample_pct": round(self.mean_metric("total_return_pct"), 3),
            "max_drawdown_out_of_sample_pct": round(self.mean_metric("max_drawdown_pct"), 3),
            "trades_out_of_sample": round(self.mean_metric("trades"), 1),
        }


def walk_forward_splits(
    index: pd.DatetimeIndex,
    config: RetrainingConfig | None = None,
    step_days: int | None = None,
) -> list[WalkForwardSplit]:
    """Genere les fenetres walk-forward couvrant l'index temporel fourni.

    Args:
        index: index temporel des donnees disponibles.
        config: parametres de fenetrage (tailles, gap de purge).
        step_days: decalage entre deux fenetres (defaut : taille de validation).

    Returns:
        La liste des fenetres, vide si l'historique est trop court.
    """
    config = config or RetrainingConfig()
    if index is None or len(index) < 2:
        return []

    start = index[0].to_pydatetime()
    end = index[-1].to_pydatetime()
    train = timedelta(days=config.train_window_days)
    gap = timedelta(days=config.purge_gap_days)
    validation = timedelta(days=config.validation_window_days)
    test = timedelta(days=config.test_window_days)
    step = timedelta(days=step_days or config.validation_window_days)

    total_needed = train + gap + validation + test
    if end - start < total_needed:
        log.warning(
            "walk_forward_history_too_short",
            available_days=(end - start).days,
            required_days=total_needed.days,
        )
        return []

    splits: list[WalkForwardSplit] = []
    cursor = start
    position = 0
    while cursor + total_needed <= end:
        train_end = cursor + train
        validation_start = train_end + gap
        validation_end = validation_start + validation
        test_end = validation_end + test
        splits.append(
            WalkForwardSplit(
                index=position,
                train_start=cursor,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=validation_end,
                test_end=test_end,
            )
        )
        cursor += step
        position += 1

    log.info("walk_forward_splits", folds=len(splits), step_days=step.days)
    return splits


def assert_no_overlap(split: WalkForwardSplit) -> None:
    """Verifie qu'aucune fenetre ne chevauche une autre (leakage structurel)."""
    if split.train_end >= split.validation_start:
        raise ValueError(
            f"fenetre {split.index}: le train empiete sur la validation "
            f"({split.train_end} >= {split.validation_start})"
        )
    if split.validation_end > split.test_start:
        raise ValueError(f"fenetre {split.index}: la validation empiete sur le test")

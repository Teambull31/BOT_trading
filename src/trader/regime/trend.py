"""Classification trend / range / mean-reversion par regles quantitatives.

Ces regles sont volontairement simples et lisibles : c'est la methode de secours
quand les modeles statistiques (HMM, clustering) manquent de donnees ou divergent.
Un systeme dont on ne comprend pas la decision est un systeme qu'on ne peut pas
debrancher au bon moment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from trader.utils.math_utils import hurst_exponent


class TrendState(str, Enum):
    """Comportement directionnel du marche."""

    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGE = "range"
    MEAN_REVERTING = "mean_reverting"
    UNDEFINED = "undefined"


@dataclass(frozen=True, slots=True)
class TrendAnalysis:
    """Resultat de la classification directionnelle."""

    state: TrendState
    adx: float
    di_spread: float
    hurst: float
    bb_width_percentile: float
    slope: float
    strength: float

    @property
    def is_trending(self) -> bool:
        """Vrai si le marche est directionnel."""
        return self.state in (TrendState.UPTREND, TrendState.DOWNTREND)


def classify_trend(
    features: pd.DataFrame,
    adx_trend_threshold: float = 25.0,
    adx_range_threshold: float = 20.0,
    hurst_trend_threshold: float = 0.6,
    hurst_revert_threshold: float = 0.4,
    close: pd.Series | None = None,
) -> TrendAnalysis:
    """Classe le comportement directionnel a partir des features courantes.

    Regles :
    - ADX > seuil haut et |+DI - -DI| significatif -> tendance (sens donne par le DI).
    - ADX < seuil bas et compression des bandes de Bollinger -> range.
    - Hurst > 0.6 -> tendance persistante ; Hurst < 0.4 -> mean-reverting.
    """
    if features is None or features.empty:
        return TrendAnalysis(TrendState.UNDEFINED, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0)

    last = features.iloc[-1]
    adx_value = _value(last, "adx", 0.0)
    di_spread = _value(last, "di_spread", 0.0)
    hurst = _value(last, "hurst", float("nan"))
    if pd.isna(hurst) and close is not None and len(close) > 50:
        hurst = hurst_exponent(close.tail(200))
    hurst = 0.5 if pd.isna(hurst) else hurst

    # Percentile calcule sur une fenetre bornee : l'historique complet rendrait
    # le cout quadratique en backtest, sans rien apporter au diagnostic.
    bb_width = (
        features["bb_width"].dropna().tail(500)
        if "bb_width" in features
        else pd.Series(dtype=float)
    )
    if len(bb_width) > 10:
        current_width = float(bb_width.iloc[-1])
        width_percentile = float((bb_width <= current_width).mean())
    else:
        width_percentile = 0.5
    slope = _value(last, "trend_slope", 0.0)

    if adx_value >= adx_trend_threshold and abs(di_spread) > 2.0:
        state = TrendState.UPTREND if di_spread > 0 else TrendState.DOWNTREND
        strength = min(1.0, adx_value / 50.0)
    elif hurst >= hurst_trend_threshold and abs(slope) > 0.0:
        state = TrendState.UPTREND if slope > 0 else TrendState.DOWNTREND
        strength = min(1.0, (hurst - 0.5) * 2.0)
    elif hurst <= hurst_revert_threshold:
        state = TrendState.MEAN_REVERTING
        strength = min(1.0, (0.5 - hurst) * 2.0)
    elif adx_value <= adx_range_threshold and width_percentile <= 0.35:
        state = TrendState.RANGE
        strength = 1.0 - width_percentile
    elif adx_value <= adx_range_threshold:
        state = TrendState.RANGE
        strength = 0.4
    else:
        state = TrendState.UNDEFINED
        strength = 0.2

    return TrendAnalysis(
        state=state,
        adx=adx_value,
        di_spread=di_spread,
        hurst=hurst,
        bb_width_percentile=width_percentile,
        slope=slope,
        strength=float(strength),
    )


def _value(row: pd.Series, key: str, default: float) -> float:
    """Lecture defensive d'une feature (absente ou NaN -> defaut)."""
    if key not in row.index:
        return default
    value = row[key]
    return default if pd.isna(value) else float(value)

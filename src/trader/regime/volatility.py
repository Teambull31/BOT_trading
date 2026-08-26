"""Regime de volatilite : realisee, EWMA, et detection des chocs.

On n'a pas d'implicite en crypto mid-cap, donc le proxy "VIX-like" est le ratio
volatilite realisee court terme / volatilite historique long terme. Au-dela de
`crisis_sigma` ecarts-types, on considere que le marche est en crise.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from trader.utils.math_utils import EPSILON
from trader.utils.time_utils import annualization_factor


class VolRegime(str, Enum):
    """Classes de volatilite."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass(frozen=True, slots=True)
class VolatilityState:
    """Etat de volatilite courant."""

    regime: VolRegime
    realized_short: float
    realized_long: float
    ratio: float
    zscore: float
    percentile: float
    is_shock: bool

    @property
    def is_crisis_level(self) -> bool:
        """Vrai si la volatilite justifie a elle seule le mode crise."""
        return self.regime is VolRegime.EXTREME


def ewma_volatility(
    returns: pd.Series, lambda_: float = 0.94, periods_per_year: float = 8760.0
) -> pd.Series:
    """Volatilite EWMA facon RiskMetrics (approximation GARCH(1,1) a parametres fixes).

    Le choix d'un lambda fixe plutot que d'un GARCH estime est deliberé : moins
    de parametres, pas de reestimation instable, comportement previsible.
    """
    squared = returns.fillna(0.0) ** 2
    variance = squared.ewm(alpha=1.0 - lambda_, min_periods=10, adjust=False).mean()
    return np.sqrt(variance * periods_per_year)


def classify_volatility(
    returns: pd.Series,
    timeframe: str = "1h",
    short_window: int = 24 * 7,
    long_window: int = 24 * 90,
    crisis_sigma: float = 2.0,
    shock_sigma: float = 4.0,
) -> VolatilityState:
    """Classe le regime de volatilite courant a partir des returns.

    Args:
        returns: log-returns (index temporel trie).
        short_window: fenetre courte, en nombre de bougies.
        long_window: fenetre longue de reference.
        crisis_sigma: nombre d'ecarts-types du ratio au-dela duquel c'est une crise.
        shock_sigma: seuil de detection d'un choc ponctuel sur le dernier return.
    """
    clean = returns.dropna()
    periods_per_year = annualization_factor(timeframe)
    if len(clean) < 20:
        return VolatilityState(VolRegime.NORMAL, 0.0, 0.0, 1.0, 0.0, 0.5, False)

    short_window = min(short_window, max(10, len(clean) // 3))
    long_window = min(long_window, len(clean))

    short_series = clean.rolling(short_window, min_periods=max(5, short_window // 3)).std(
        ddof=1
    ) * np.sqrt(periods_per_year)
    long_vol = float(clean.iloc[-long_window:].std(ddof=1) * np.sqrt(periods_per_year))
    short_vol = float(short_series.iloc[-1]) if short_series.notna().any() else long_vol

    ratio = short_vol / long_vol if long_vol > EPSILON else 1.0
    ratio_series = (short_series / long_vol).dropna() if long_vol > EPSILON else pd.Series([1.0])
    ratio_std = float(ratio_series.std(ddof=1)) if len(ratio_series) > 2 else 0.0
    ratio_mean = float(ratio_series.mean()) if len(ratio_series) > 0 else 1.0
    zscore = (ratio - ratio_mean) / ratio_std if ratio_std > EPSILON else 0.0
    percentile = float((ratio_series <= ratio).mean()) if len(ratio_series) > 5 else 0.5

    last_return = float(clean.iloc[-1])
    bar_std = float(clean.iloc[-long_window:].std(ddof=1))
    is_shock = bar_std > EPSILON and abs(last_return) > shock_sigma * bar_std

    if zscore >= crisis_sigma or is_shock:
        regime = VolRegime.EXTREME
    elif percentile >= 0.80:
        regime = VolRegime.HIGH
    elif percentile <= 0.20:
        regime = VolRegime.LOW
    else:
        regime = VolRegime.NORMAL

    return VolatilityState(
        regime=regime,
        realized_short=short_vol,
        realized_long=long_vol,
        ratio=ratio,
        zscore=float(zscore),
        percentile=percentile,
        is_shock=is_shock,
    )

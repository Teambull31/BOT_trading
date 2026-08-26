"""Fonctions mathematiques et statistiques utilisees par les modules d'analyse.

Toutes les fonctions sont sans look-ahead : elles ne consomment que les valeurs
passees en argument, et les versions rolling utilisent des fenetres fermees a t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON: float = 1e-12


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division protegee contre le zero et les NaN."""
    if denominator is None or not np.isfinite(denominator) or abs(denominator) < EPSILON:
        return default
    result = numerator / denominator
    return float(result) if np.isfinite(result) else default


def log_returns(prices: pd.Series) -> pd.Series:
    """Log-returns d'une serie de prix (premiere valeur = NaN)."""
    clean = prices.astype(float).replace(0.0, np.nan)
    return np.log(clean / clean.shift(1))


def realized_volatility(returns: pd.Series, window: int, periods_per_year: float) -> pd.Series:
    """Volatilite realisee annualisee sur une fenetre glissante."""
    return returns.rolling(window, min_periods=max(2, window // 2)).std(ddof=1) * np.sqrt(
        periods_per_year
    )


def sharpe_ratio(
    returns: pd.Series | np.ndarray, periods_per_year: float, risk_free: float = 0.0
) -> float:
    """Sharpe ratio annualise. Retourne 0.0 si l'echantillon est insuffisant."""
    array = np.asarray(returns, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return 0.0
    excess = array - risk_free / periods_per_year
    std = float(np.std(excess, ddof=1))
    if std < EPSILON:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series | np.ndarray, periods_per_year: float, target: float = 0.0
) -> float:
    """Sortino ratio annualise (penalise uniquement la volatilite baissiere)."""
    array = np.asarray(returns, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return 0.0
    downside = array[array < target] - target
    if downside.size == 0:
        return 0.0
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    if downside_dev < EPSILON:
        return 0.0
    return float((np.mean(array) - target) / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    """Drawdown maximal d'une courbe d'equity, en fraction positive (0.15 = -15 %)."""
    array = np.asarray(equity, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return 0.0
    running_max = np.maximum.accumulate(array)
    drawdowns = np.where(running_max > EPSILON, (running_max - array) / running_max, 0.0)
    return float(np.max(drawdowns))


def calmar_ratio(returns: pd.Series | np.ndarray, periods_per_year: float) -> float:
    """Calmar ratio : rendement annualise / drawdown maximal."""
    array = np.asarray(returns, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return 0.0
    equity = np.cumprod(1.0 + array)
    drawdown = max_drawdown(equity)
    if drawdown < EPSILON:
        return 0.0
    total_return = float(equity[-1]) - 1.0
    years = array.size / periods_per_year
    if years < EPSILON:
        return 0.0
    annualized = (1.0 + total_return) ** (1.0 / years) - 1.0
    return float(annualized / drawdown)


def profit_factor(pnl: pd.Series | np.ndarray) -> float:
    """Profit factor : somme des gains / somme des pertes (en valeur absolue)."""
    array = np.asarray(pnl, dtype=float)
    array = array[np.isfinite(array)]
    gains = float(np.sum(array[array > 0]))
    losses = float(-np.sum(array[array < 0]))
    if losses < EPSILON:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def hit_rate(pnl: pd.Series | np.ndarray) -> float:
    """Taux de trades gagnants (les trades a zero comptent comme perdants)."""
    array = np.asarray(pnl, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    return float(np.mean(array > 0))


def max_consecutive_losses(pnl: pd.Series | np.ndarray) -> int:
    """Plus longue serie de trades perdants consecutifs."""
    array = np.asarray(pnl, dtype=float)
    worst = 0
    current = 0
    for value in array:
        if np.isfinite(value) and value <= 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return int(worst)


def hurst_exponent(series: pd.Series | np.ndarray, max_lag: int = 40) -> float:
    """Exposant de Hurst par la methode du rescaled range simplifiee.

    > 0.5 : serie persistante (tendance). < 0.5 : serie anti-persistante
    (mean-reverting). ~ 0.5 : marche aleatoire. Retourne 0.5 si l'echantillon
    est trop court pour conclure.
    """
    array = np.asarray(series, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < max_lag * 2:
        max_lag = max(4, array.size // 2)
    if array.size < 20:
        return 0.5
    lags = np.arange(2, max_lag)
    tau = []
    valid_lags = []
    for lag in lags:
        diff = array[lag:] - array[:-lag]
        std = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
        if std > EPSILON:
            tau.append(std)
            valid_lags.append(lag)
    if len(tau) < 3:
        return 0.5
    slope = np.polyfit(np.log(valid_lags), np.log(tau), 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def autocorrelation(series: pd.Series, lag: int) -> float:
    """Autocorrelation d'une serie a un lag donne."""
    clean = series.dropna()
    if clean.size <= lag + 2:
        return 0.0
    value = clean.autocorr(lag=lag)
    return float(value) if value is not None and np.isfinite(value) else 0.0


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Z-score glissant (fenetre fermee a t, aucun look-ahead)."""
    mean = series.rolling(window, min_periods=max(2, window // 2)).mean()
    std = series.rolling(window, min_periods=max(2, window // 2)).std(ddof=1)
    return (series - mean) / std.replace(0.0, np.nan)


def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Fraction de Kelly : f* = p - (1 - p) / R, bornee a [0, 1]."""
    if win_loss_ratio <= EPSILON:
        return 0.0
    raw = win_rate - (1.0 - win_rate) / win_loss_ratio
    return float(np.clip(raw, 0.0, 1.0))


def clamp(value: float, low: float, high: float) -> float:
    """Borne une valeur dans [low, high]."""
    return float(min(max(value, low), high))


def normalize_weights(weights: dict[str, float], cap: float | None = None) -> dict[str, float]:
    """Normalise des poids positifs pour qu'ils somment a 1, avec cap optionnel.

    Le cap est applique iterativement : le surplus des poids capes est
    redistribue aux autres, jusqu'a convergence.
    """
    positive = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(positive.values())
    if total < EPSILON:
        return {k: 0.0 for k in positive}
    normalized = {k: v / total for k, v in positive.items()}
    if cap is None:
        return normalized
    if cap * len(normalized) < 1.0 - EPSILON:
        # Cap impossible a respecter : repartition uniforme.
        uniform = 1.0 / len(normalized)
        return {k: uniform for k in normalized}
    for _ in range(50):
        excess = {k: v for k, v in normalized.items() if v > cap + EPSILON}
        if not excess:
            break
        surplus = sum(v - cap for v in excess.values())
        free = {k: v for k, v in normalized.items() if v <= cap + EPSILON}
        free_total = sum(free.values())
        for key in excess:
            normalized[key] = cap
        if free_total < EPSILON:
            break
        for key in free:
            normalized[key] += surplus * (normalized[key] / free_total)
    return normalized

"""Evaluation des strategies : Sharpe, Sortino, Calmar, hit rate, profit factor.

Alimente deux consommateurs :
- l'ensemble, qui pondere les strategies selon leur performance recente ;
- le detecteur de decay, qui compare les fenetres courtes aux longues.

Regle de prudence : en dessous de `min_trades`, on ne rend PAS de verdict de
performance. Un Sharpe calcule sur trois trades ne mesure rien, et l'utiliser
pour ponderer une strategie revient a laisser le hasard piloter le capital.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from trader.data.store import DataStore
from trader.logging_setup import get_logger
from trader.utils.math_utils import (
    calmar_ratio,
    hit_rate,
    max_consecutive_losses,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)
from trader.utils.time_utils import utc_now

log = get_logger(__name__)

TRADES_PER_YEAR: float = 252.0
"""Base d'annualisation des metriques calculees par trade (et non par bougie)."""


@dataclass(slots=True)
class StrategyMetrics:
    """Metriques de performance d'une strategie sur une fenetre donnee."""

    strategy: str
    trades: int = 0
    window_days: int = 0
    total_pnl: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0
    is_significant: bool = False

    def to_dict(self) -> dict[str, float | int | bool | str]:
        """Representation serialisable."""
        return {
            "strategy": self.strategy,
            "trades": self.trades,
            "window_days": self.window_days,
            "total_pnl": round(self.total_pnl, 4),
            "sharpe": round(self.sharpe, 4),
            "sortino": round(self.sortino, 4),
            "calmar": round(self.calmar, 4),
            "hit_rate": round(self.hit_rate, 4),
            "profit_factor": round(min(self.profit_factor, 1e6), 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "consecutive_losses": self.consecutive_losses,
            "win_loss_ratio": round(self.win_loss_ratio, 4),
            "is_significant": self.is_significant,
        }


@dataclass(slots=True)
class EvaluationSnapshot:
    """Metriques multi-fenetres d'une strategie."""

    strategy: str
    windows: dict[int, StrategyMetrics] = field(default_factory=dict)

    def window(self, days: int) -> StrategyMetrics:
        """Metriques d'une fenetre (vides si absente)."""
        return self.windows.get(days, StrategyMetrics(strategy=self.strategy, window_days=days))

    def as_provider_dict(self) -> dict[str, float]:
        """Format attendu par l'ensemble pour la ponderation."""
        long_window = self.window(30)
        short_window = self.window(7)
        return {
            "sharpe_30d": long_window.sharpe,
            "sharpe_7d": short_window.sharpe,
            "hit_rate": long_window.hit_rate,
            "profit_factor": min(long_window.profit_factor, 1e6),
            "trades": float(long_window.trades),
            "win_loss_ratio": long_window.win_loss_ratio,
        }


class StrategyEvaluator:
    """Calcule les metriques de performance des strategies a partir des trades."""

    DEFAULT_WINDOWS: tuple[int, ...] = (7, 14, 30)

    def __init__(
        self,
        store: DataStore,
        min_trades: int = 10,
        mode: str | None = None,
        windows: tuple[int, ...] = DEFAULT_WINDOWS,
    ) -> None:
        self.store = store
        self.min_trades = min_trades
        self.mode = mode
        self.windows = windows
        self._cache: dict[str, EvaluationSnapshot] = {}

    # --------------------------------------------------------------- calcul

    def evaluate(
        self, strategy: str, now: datetime | None = None, trades: pd.DataFrame | None = None
    ) -> EvaluationSnapshot:
        """Calcule les metriques d'une strategie sur toutes les fenetres."""
        reference = now or utc_now()
        if trades is None:
            longest = max(self.windows)
            trades = self.store.load_trades(
                strategy=strategy, since=reference - timedelta(days=longest), mode=self.mode
            )

        snapshot = EvaluationSnapshot(strategy=strategy)
        for days in self.windows:
            cutoff = reference - timedelta(days=days)
            window_trades = trades[trades["closed_at"] >= cutoff] if not trades.empty else trades
            snapshot.windows[days] = self._metrics(strategy, window_trades, days)
        self._cache[strategy] = snapshot
        return snapshot

    def _metrics(self, strategy: str, trades: pd.DataFrame, days: int) -> StrategyMetrics:
        """Calcule les metriques sur un echantillon de trades."""
        metrics = StrategyMetrics(strategy=strategy, window_days=days)
        if trades is None or trades.empty:
            return metrics

        pnl = trades["pnl"].to_numpy(dtype=float)
        metrics.trades = int(pnl.size)
        metrics.total_pnl = float(np.sum(pnl))
        metrics.hit_rate = hit_rate(pnl)
        metrics.profit_factor = profit_factor(pnl)
        metrics.consecutive_losses = max_consecutive_losses(pnl)

        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        metrics.avg_win = float(np.mean(wins)) if wins.size else 0.0
        metrics.avg_loss = float(np.mean(losses)) if losses.size else 0.0
        metrics.win_loss_ratio = (
            abs(metrics.avg_win / metrics.avg_loss) if metrics.avg_loss != 0 else 0.0
        )

        # Les returns sont normalises par le notionnel : comparer des P&L bruts
        # de tailles differentes n'a pas de sens.
        notional = (trades["size"] * trades["entry_price"]).replace(0.0, np.nan)
        returns = (trades["pnl"] / notional).dropna().to_numpy(dtype=float)
        if returns.size >= 2:
            metrics.sharpe = sharpe_ratio(returns, TRADES_PER_YEAR)
            metrics.sortino = sortino_ratio(returns, TRADES_PER_YEAR)
            metrics.calmar = calmar_ratio(returns, TRADES_PER_YEAR)
            metrics.max_drawdown_pct = max_drawdown(np.cumprod(1.0 + returns)) * 100.0

        metrics.is_significant = metrics.trades >= self.min_trades
        if not metrics.is_significant:
            # Echantillon trop petit : on neutralise les ratios plutot que de
            # laisser du bruit statistique piloter la ponderation.
            metrics.sharpe = 0.0
            metrics.sortino = 0.0
            metrics.calmar = 0.0
        return metrics

    # ------------------------------------------------------------ interface

    def metrics_provider(self, strategy: str) -> dict[str, float]:
        """Callback branche sur l'ensemble pour la ponderation dynamique."""
        return self.evaluate(strategy).as_provider_dict()

    def evaluate_all(
        self, strategies: list[str], now: datetime | None = None
    ) -> dict[str, EvaluationSnapshot]:
        """Evalue plusieurs strategies d'un coup."""
        return {name: self.evaluate(name, now) for name in strategies}

    def ranking(self, strategies: list[str], window: int = 30) -> list[tuple[str, float]]:
        """Classement des strategies par Sharpe sur une fenetre."""
        scored = [(name, self.evaluate(name).window(window).sharpe) for name in strategies]
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def last_evaluations(self) -> dict[str, EvaluationSnapshot]:
        """Dernieres evaluations calculees, par strategie."""
        return dict(self._cache)

    def portfolio_metrics(self, now: datetime | None = None) -> dict[str, float]:
        """Metriques globales du portefeuille, toutes strategies confondues."""
        reference = now or utc_now()
        equity = self.store.load_equity(mode=self.mode)
        trades = self.store.load_trades(mode=self.mode)
        if equity.empty:
            return {"equity": 0.0, "trades": float(len(trades))}

        returns = equity.pct_change().dropna()
        days = max(1.0, (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0)
        periods_per_year = len(returns) / days * 365.0 if days > 0 else TRADES_PER_YEAR
        pnl = trades["pnl"].to_numpy(dtype=float) if not trades.empty else np.array([])

        return {
            "equity": float(equity.iloc[-1]),
            "peak_equity": float(equity.max()),
            "max_drawdown_pct": max_drawdown(equity) * 100.0,
            "sharpe": sharpe_ratio(returns, periods_per_year),
            "sortino": sortino_ratio(returns, periods_per_year),
            "trades": float(pnl.size),
            "hit_rate": hit_rate(pnl) if pnl.size else 0.0,
            "profit_factor": min(profit_factor(pnl), 1e6) if pnl.size else 0.0,
            "days_running": days,
            "reference": reference.timestamp(),
        }

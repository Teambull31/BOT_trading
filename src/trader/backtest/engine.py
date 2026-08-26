"""Moteur de backtest evenementiel, barre par barre.

Regles de conception, toutes destinees a empecher un backtest menteur :

1. A chaque barre i, le systeme ne voit QUE les donnees 0..i (MarketSnapshot
   tranche). Le look-ahead est structurellement impossible, pas seulement
   "evite avec soin".
2. Les features sont calculees UNE fois sur toute la serie, mais chaque snapshot
   n'expose que son prefixe. Ce raccourci est valide parce que `assert_no_lookahead`
   garantit qu'une feature calculee sur la serie complete a la meme valeur en t
   que si elle avait ete calculee avec les seules donnees jusqu'a t.
3. Les ordres sont executes a la barre SUIVANTE (i+1), jamais au prix de cloture
   qui a declenche le signal : on ne peut pas trader un prix qu'on vient juste
   d'observer.
4. Frais et slippage sont toujours appliques. Un backtest sans couts est une
   fiction.
5. Les stops et les cibles sont evalues sur le haut/bas de la barre, avec la
   convention pessimiste : si la barre touche le stop ET la cible, le stop
   l'emporte.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from trader.config import Settings
from trader.data.features import FeatureBuilder
from trader.data.snapshot import build_snapshot
from trader.execution.slippage import estimate_slippage
from trader.logging_setup import get_logger
from trader.models import (
    EnsembleDecision,
    MarketSnapshot,
    OrderSide,
    Position,
    Regime,
    RegimeState,
    Trade,
)
from trader.regime.detector import RegimeDetector
from trader.strategy.ensemble import StrategyEnsemble
from trader.utils.math_utils import (
    calmar_ratio,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)
from trader.utils.time_utils import annualization_factor

log = get_logger(__name__)

RiskHook = Callable[[EnsembleDecision, RegimeState, float], float]
"""Hook de risque : (decision, regime, equity) -> notionnel autorise (0 = refus)."""


@dataclass(slots=True)
class BacktestResult:
    """Resultat complet d'un backtest."""

    equity: pd.Series
    trades: list[Trade]
    regimes: pd.DataFrame
    metrics: dict[str, float]
    blocked_reasons: dict[str, int] = field(default_factory=dict)
    initial_capital: float = 0.0

    @property
    def trade_count(self) -> int:
        """Nombre de trades clotures."""
        return len(self.trades)

    def trades_frame(self) -> pd.DataFrame:
        """Trades sous forme de DataFrame."""
        if not self.trades:
            return pd.DataFrame(columns=["asset", "pnl", "closed_at", "strategy", "regime"])
        return pd.DataFrame([trade.to_dict() for trade in self.trades])

    def summary(self) -> str:
        """Resume lisible du backtest."""
        lines = [
            f"Capital initial   : {self.initial_capital:,.2f}",
            f"Capital final     : {self.metrics.get('final_equity', 0.0):,.2f}",
            f"Rendement total   : {self.metrics.get('total_return_pct', 0.0):+.2f} %",
            f"Sharpe            : {self.metrics.get('sharpe', 0.0):.2f}",
            f"Sortino           : {self.metrics.get('sortino', 0.0):.2f}",
            f"Calmar            : {self.metrics.get('calmar', 0.0):.2f}",
            f"Max drawdown      : {self.metrics.get('max_drawdown_pct', 0.0):.2f} %",
            f"Trades            : {self.trade_count}",
            f"Hit rate          : {self.metrics.get('hit_rate', 0.0):.1%}",
            f"Profit factor     : {self.metrics.get('profit_factor', 0.0):.2f}",
        ]
        return "\n".join(lines)


class BacktestEngine:
    """Rejoue une strategie d'ensemble sur des donnees historiques."""

    def __init__(
        self,
        settings: Settings,
        ensemble: StrategyEnsemble,
        detector: RegimeDetector | None = None,
        risk_hook: RiskHook | None = None,
        feature_builder: FeatureBuilder | None = None,
        timeframe: str | None = None,
    ) -> None:
        self.settings = settings
        self.ensemble = ensemble
        self.timeframe = timeframe or settings.data.primary_timeframe
        self.detector = detector or RegimeDetector(settings.regime, self.timeframe)
        self.risk_hook = risk_hook
        self.feature_builder = feature_builder or FeatureBuilder(timeframe=self.timeframe)

    def run(
        self,
        ohlcv: pd.DataFrame,
        asset: str = "ETH/USDT",
        warmup: int = 200,
        initial_capital: float | None = None,
        features: pd.DataFrame | None = None,
        retrain_every: int | None = None,
    ) -> BacktestResult:
        """Execute le backtest barre par barre.

        Args:
            warmup: nombre de bougies reservees au calcul des features et au
                premier entrainement des modeles de regime.
            retrain_every: periodicite de reentrainement du detecteur, en bougies
                (par defaut : `regime.retrain_interval_days` converti en bougies).
        """
        if ohlcv is None or len(ohlcv) <= warmup + 10:
            raise ValueError(
                f"historique trop court : {0 if ohlcv is None else len(ohlcv)} bougies "
                f"pour un warmup de {warmup}"
            )

        default_capital = self.settings.general.initial_capital
        capital = float(initial_capital if initial_capital is not None else default_capital)
        features = features if features is not None else self.feature_builder.build(ohlcv)
        bars_per_day = annualization_factor(self.timeframe) / 365.0
        retrain_every = retrain_every or max(
            24, int(self.settings.regime.retrain_interval_days * bars_per_day)
        )

        equity = capital
        cash = capital
        position: Position | None = None
        trades: list[Trade] = []
        equity_points: list[tuple[datetime, float]] = []
        regime_rows: list[dict[str, Any]] = []
        blocked: dict[str, int] = {}
        last_fit = -1

        for index in range(warmup, len(ohlcv) - 1):
            timestamp = ohlcv.index[index].to_pydatetime()
            bar = ohlcv.iloc[index]
            next_bar = ohlcv.iloc[index + 1]
            price = float(bar["close"])

            if last_fit < 0 or index - last_fit >= retrain_every:
                # Reentrainement walk-forward : uniquement sur le passe visible.
                self.detector.fit(features.iloc[: index + 1], now=timestamp)
                last_fit = index

            snapshot = self._snapshot(asset, ohlcv, features, index)
            regime = self.detector.detect(
                features.iloc[: index + 1], ohlcv.iloc[: index + 1], now=timestamp
            )
            regime_rows.append(
                {
                    "timestamp": timestamp,
                    "regime": regime.regime.value,
                    "confidence": regime.confidence,
                    "agreement": regime.agreement_score,
                }
            )

            # 1. Sorties : stop, cible, ou fermeture forcee en crise.
            if position is not None:
                exit_price, reason = self._exit_price(position, next_bar, regime)
                if exit_price is not None:
                    trade, cash = self._close(
                        position, exit_price, cash, next_bar, timestamp, reason
                    )
                    trades.append(trade)
                    position = None

            # 2. Entrees.
            if position is None:
                decision = self.ensemble.decide(snapshot, regime)
                if decision.blocked_reason:
                    key = decision.blocked_reason.split(" (")[0][:60]
                    blocked[key] = blocked.get(key, 0) + 1
                elif decision.is_actionable and decision.stop_loss:
                    notional = self._sizing(decision, regime, equity)
                    if notional > 0:
                        position, cash = self._open(decision, notional, cash, next_bar, snapshot)

            # L'equity = cash + P&L latent. Les frais d'entree ont deja ete
            # deduits du cash a l'ouverture, ils ne sont pas comptes deux fois.
            equity = cash + (self._position_value(position, price) if position else 0.0)
            equity_points.append((timestamp, equity))

        # Cloture de la position residuelle au dernier prix connu.
        if position is not None:
            final_price = float(ohlcv["close"].iloc[-1])
            trade, cash = self._close(
                position,
                final_price,
                cash,
                ohlcv.iloc[-1],
                ohlcv.index[-1].to_pydatetime(),
                "fin_de_backtest",
            )
            trades.append(trade)
            equity_points.append((ohlcv.index[-1].to_pydatetime(), cash))

        curve = pd.Series(
            [value for _, value in equity_points],
            index=pd.DatetimeIndex([stamp for stamp, _ in equity_points]),
            name="equity",
        )
        return BacktestResult(
            equity=curve,
            trades=trades,
            regimes=pd.DataFrame(regime_rows),
            metrics=self.compute_metrics(curve, trades, capital),
            blocked_reasons=blocked,
            initial_capital=capital,
        )

    # ------------------------------------------------------------- mecanique

    def _snapshot(
        self, asset: str, ohlcv: pd.DataFrame, features: pd.DataFrame, index: int
    ) -> MarketSnapshot:
        """Construit le snapshot de la barre courante (donnees 0..index seulement)."""
        spread_pct = self._spread_pct(ohlcv, index)
        price = float(ohlcv["close"].iloc[index])
        half_spread = price * spread_pct / 200.0
        return build_snapshot(
            asset,
            ohlcv,
            features,
            position=index,
            bid=price - half_spread,
            ask=price + half_spread,
        )

    @staticmethod
    def _spread_pct(ohlcv: pd.DataFrame, index: int) -> float:
        """Spread estime a partir de l'amplitude recente (proxy raisonnable)."""
        window = ohlcv.iloc[max(0, index - 20) : index + 1]
        if window.empty:
            return 0.05
        amplitude = float(((window["high"] - window["low"]) / window["close"]).mean())
        return float(np.clip(amplitude * 5.0, 0.02, 1.0))

    def _sizing(self, decision: EnsembleDecision, regime: RegimeState, equity: float) -> float:
        """Determine le notionnel a engager (delegue au risk hook si fourni)."""
        if self.risk_hook is not None:
            return max(0.0, float(self.risk_hook(decision, regime, equity)))
        # Sans risk manager branche, on applique la limite en dur par position.
        notional = equity * self.settings.risk.max_position_pct / 100.0
        if regime.is_uncertain:
            notional *= self.settings.risk.uncertain_regime.exposure_multiplier
        return notional

    def _open(
        self,
        decision: EnsembleDecision,
        notional: float,
        cash: float,
        next_bar: pd.Series,
        snapshot: MarketSnapshot,
    ) -> tuple[Position, float]:
        """Ouvre une position au prix de la barre suivante, frais et slippage inclus."""
        side = OrderSide.BUY if decision.signal.direction > 0 else OrderSide.SELL
        reference = float(next_bar["open"])
        size = notional / reference if reference > 0 else 0.0
        estimate = estimate_slippage(
            reference_price=reference,
            side=side,
            size=size,
            spread_pct=snapshot.spread_pct,
            average_volume=float(snapshot.ohlcv["volume"].tail(20).mean()),
            model=self.settings.execution.slippage_model,
            fixed_bps=self.settings.execution.fixed_slippage_bps,
        )
        fees = notional * self.settings.execution.taker_fee_bps / 10_000.0
        position = Position(
            asset=decision.asset,
            side=side,
            size=size,
            entry_price=estimate.price,
            stop_loss=float(decision.stop_loss),
            target_price=decision.target_price,
            strategy="ensemble",
            opened_at=snapshot.timestamp,
            fees_paid=fees,
            metadata={"weights": decision.weights, "slippage_bps": estimate.slippage_bps},
        )
        return position, cash - fees

    def _close(
        self,
        position: Position,
        exit_price: float,
        cash: float,
        bar: pd.Series,
        timestamp: datetime,
        reason: str,
    ) -> tuple[Trade, float]:
        """Cloture une position et enregistre le trade."""
        side = OrderSide.SELL if position.side is OrderSide.BUY else OrderSide.BUY
        notional = position.size * exit_price
        estimate = estimate_slippage(
            reference_price=exit_price,
            side=side,
            size=position.size,
            spread_pct=None,
            average_volume=float(bar.get("volume", 0.0)),
            model=self.settings.execution.slippage_model,
            fixed_bps=self.settings.execution.fixed_slippage_bps,
        )
        fill_price = estimate.price
        fees = notional * self.settings.execution.taker_fee_bps / 10_000.0
        gross = (fill_price - position.entry_price) * position.size * position.direction
        pnl = gross - fees - position.fees_paid
        trade = Trade(
            asset=position.asset,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=fill_price,
            pnl=pnl,
            fees=fees + position.fees_paid,
            opened_at=position.opened_at,
            closed_at=timestamp,
            strategy=position.strategy,
            exit_reason=reason,
        )
        return trade, cash + pnl

    @staticmethod
    def _position_value(position: Position, price: float) -> float:
        """Valeur mark-to-market d'une position (P&L latent uniquement)."""
        return (price - position.entry_price) * position.size * position.direction

    def _exit_price(
        self, position: Position, next_bar: pd.Series, regime: RegimeState
    ) -> tuple[float | None, str]:
        """Determine si et a quel prix la position se ferme sur la barre suivante.

        Convention pessimiste : si la barre touche le stop ET la cible, on
        suppose que le stop a ete touche en premier.
        """
        high = float(next_bar["high"])
        low = float(next_bar["low"])

        if position.direction > 0:
            if low <= position.stop_loss:
                return position.stop_loss, "stop_loss"
            if position.target_price is not None and high >= position.target_price:
                return position.target_price, "take_profit"
        else:
            if high >= position.stop_loss:
                return position.stop_loss, "stop_loss"
            if position.target_price is not None and low <= position.target_price:
                return position.target_price, "take_profit"

        if regime.regime is Regime.CRISIS and self.settings.risk.crisis_regime.close_existing:
            return float(next_bar["open"]), "crisis_exit"
        return None, ""

    # -------------------------------------------------------------- metriques

    def compute_metrics(
        self, equity: pd.Series, trades: Sequence[Trade], initial_capital: float
    ) -> dict[str, float]:
        """Calcule les metriques de performance du backtest."""
        if equity.empty:
            return {"final_equity": initial_capital, "total_return_pct": 0.0}

        returns = equity.pct_change().dropna()
        periods = annualization_factor(self.timeframe)
        pnl = np.asarray([trade.pnl for trade in trades], dtype=float)
        final = float(equity.iloc[-1])

        return {
            "final_equity": final,
            "total_return_pct": (final / initial_capital - 1.0) * 100.0,
            "sharpe": sharpe_ratio(returns, periods),
            "sortino": sortino_ratio(returns, periods),
            "calmar": calmar_ratio(returns, periods),
            "max_drawdown_pct": max_drawdown(equity) * 100.0,
            "volatility_annual_pct": float(returns.std(ddof=1) * np.sqrt(periods) * 100.0)
            if len(returns) > 2
            else 0.0,
            "trades": float(len(trades)),
            "hit_rate": hit_rate(pnl),
            "profit_factor": min(profit_factor(pnl), 1e6),
            "avg_trade_pnl": float(np.mean(pnl)) if pnl.size else 0.0,
            "best_trade": float(np.max(pnl)) if pnl.size else 0.0,
            "worst_trade": float(np.min(pnl)) if pnl.size else 0.0,
        }


def buy_and_hold(ohlcv: pd.DataFrame, initial_capital: float, warmup: int = 200) -> pd.Series:
    """Courbe d'equity d'un buy & hold, benchmark de reference.

    Toute strategie qui ne bat pas le buy & hold ne merite pas sa complexite.
    """
    window = ohlcv.iloc[warmup:]
    if window.empty:
        return pd.Series(dtype=float)
    normalized = window["close"] / float(window["close"].iloc[0])
    return (normalized * initial_capital).rename("buy_and_hold")

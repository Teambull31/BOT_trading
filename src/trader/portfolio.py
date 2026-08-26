"""Etat du portefeuille : cash, positions, equity, drawdowns.

Source unique de verite sur "ou j'en suis". Le risk manager y lit les drawdowns,
l'exposition et le nombre de positions ouvertes ; il n'a pas le droit de les
recalculer dans son coin, sinon deux composants finissent par croire des choses
differentes sur le meme portefeuille.

Les drawdowns sont calcules par rapport a des pics DATES (jour, semaine, global),
pas par rapport au capital initial : perdre 3 % apres avoir gagne 20 % n'est pas
la meme chose que perdre 3 % du premier jour.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from trader.logging_setup import get_logger
from trader.models import OrderSide, Position, Trade
from trader.utils.math_utils import EPSILON
from trader.utils.time_utils import to_utc, utc_now

log = get_logger(__name__)


@dataclass(slots=True)
class EquityPoint:
    """Point date de la courbe d'equity."""

    timestamp: datetime
    equity: float


@dataclass(slots=True)
class DrawdownState:
    """Etat des drawdowns a un instant donne."""

    daily_pct: float
    weekly_pct: float
    total_pct: float
    peak_equity: float
    daily_peak: float
    weekly_peak: float

    def worst(self) -> float:
        """Pire drawdown des trois horizons."""
        return max(self.daily_pct, self.weekly_pct, self.total_pct)


class Portfolio:
    """Suit le capital, les positions ouvertes et l'historique des trades."""

    def __init__(self, initial_capital: float, max_history: int = 20000) -> None:
        if initial_capital <= 0:
            raise ValueError("le capital initial doit etre strictement positif")
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[Trade] = []
        self.equity_history: deque[EquityPoint] = deque(maxlen=max_history)
        self.peak_equity = float(initial_capital)
        self._last_prices: dict[str, float] = {}
        self._last_trade_time: dict[str, datetime] = {}
        self.equity_history.append(EquityPoint(utc_now(), float(initial_capital)))

    # ------------------------------------------------------------- valeur

    def mark_to_market(self, prices: dict[str, float], now: datetime | None = None) -> float:
        """Met a jour la valeur du portefeuille aux prix courants."""
        self._last_prices.update({k: float(v) for k, v in prices.items() if v > 0})
        value = self.equity(prices)
        stamp = to_utc(now or utc_now())
        self.equity_history.append(EquityPoint(stamp, value))
        self.peak_equity = max(self.peak_equity, value)
        return value

    def equity(self, prices: dict[str, float] | None = None) -> float:
        """Valeur totale : cash + P&L latent des positions ouvertes."""
        marks = {**self._last_prices, **(prices or {})}
        unrealized = 0.0
        for asset, position in self.positions.items():
            price = marks.get(asset, position.entry_price)
            unrealized += position.unrealized_pnl(price)
        return self.cash + unrealized

    def exposure(self, prices: dict[str, float] | None = None) -> float:
        """Exposition brute en valeur notionnelle."""
        marks = {**self._last_prices, **(prices or {})}
        return sum(
            position.notional(marks.get(asset, position.entry_price))
            for asset, position in self.positions.items()
        )

    def exposure_pct(self, prices: dict[str, float] | None = None) -> float:
        """Exposition brute en % de l'equity."""
        equity = self.equity(prices)
        if equity <= EPSILON:
            return 0.0
        return self.exposure(prices) / equity * 100.0

    # ---------------------------------------------------------- positions

    def open_position(self, position: Position, fees: float = 0.0) -> None:
        """Enregistre une nouvelle position (une seule par actif)."""
        if position.asset in self.positions:
            raise ValueError(f"position deja ouverte sur {position.asset}")
        self.positions[position.asset] = position
        self.cash -= fees
        self._last_prices[position.asset] = position.entry_price
        self._last_trade_time[position.asset] = to_utc(position.opened_at)
        log.info(
            "position_opened",
            asset=position.asset,
            side=position.side.value,
            size=position.size,
            entry=position.entry_price,
            stop=position.stop_loss,
        )

    def close_position(
        self,
        asset: str,
        exit_price: float,
        fees: float = 0.0,
        reason: str = "",
        regime: str = "",
        now: datetime | None = None,
    ) -> Trade:
        """Cloture une position et enregistre le trade correspondant."""
        position = self.positions.pop(asset, None)
        if position is None:
            raise KeyError(f"aucune position ouverte sur {asset}")

        gross = (exit_price - position.entry_price) * position.size * position.direction
        total_fees = fees + position.fees_paid
        pnl = gross - total_fees
        self.cash += pnl
        stamp = to_utc(now or utc_now())
        trade = Trade(
            asset=asset,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=float(exit_price),
            pnl=pnl,
            fees=total_fees,
            opened_at=position.opened_at,
            closed_at=stamp,
            strategy=position.strategy,
            regime=regime,
            exit_reason=reason,
        )
        self.closed_trades.append(trade)
        self._last_trade_time[asset] = stamp
        log.info(
            "position_closed",
            asset=asset,
            exit=exit_price,
            pnl=round(pnl, 4),
            reason=reason,
        )
        return trade

    def get_position(self, asset: str) -> Position | None:
        """Position ouverte sur un actif, ou None."""
        return self.positions.get(asset)

    def has_position(self, asset: str) -> bool:
        """Vrai si une position est ouverte sur cet actif."""
        return asset in self.positions

    @property
    def open_count(self) -> int:
        """Nombre de positions ouvertes."""
        return len(self.positions)

    def last_trade_time(self, asset: str) -> datetime | None:
        """Horodatage de la derniere activite sur un actif (pour le cooldown)."""
        return self._last_trade_time.get(asset)

    def positions_to_stop(self, prices: dict[str, float]) -> list[tuple[Position, str]]:
        """Positions dont le stop ou la cible est atteint au prix courant."""
        hits: list[tuple[Position, str]] = []
        for asset, position in self.positions.items():
            price = prices.get(asset)
            if price is None:
                continue
            if position.should_stop_out(price):
                hits.append((position, "stop_loss"))
            elif position.should_take_profit(price):
                hits.append((position, "take_profit"))
        return hits

    # --------------------------------------------------------- drawdowns

    def drawdowns(self, now: datetime | None = None) -> DrawdownState:
        """Calcule les drawdowns journalier, hebdomadaire et total.

        Chaque drawdown est mesure depuis le PIC de sa fenetre, pas depuis le
        capital initial : c'est ce qui protege les gains deja acquis.
        """
        reference = to_utc(now or utc_now())
        current = self.equity()
        history = list(self.equity_history)

        def peak_since(delta: timedelta) -> float:
            cutoff = reference - delta
            values = [point.equity for point in history if point.timestamp >= cutoff]
            values.append(current)
            return max(values)

        daily_peak = peak_since(timedelta(days=1))
        weekly_peak = peak_since(timedelta(days=7))
        total_peak = max(self.peak_equity, current)

        def drawdown(peak: float) -> float:
            if peak <= EPSILON:
                return 0.0
            return max(0.0, (peak - current) / peak * 100.0)

        return DrawdownState(
            daily_pct=drawdown(daily_peak),
            weekly_pct=drawdown(weekly_peak),
            total_pct=drawdown(total_peak),
            peak_equity=total_peak,
            daily_peak=daily_peak,
            weekly_peak=weekly_peak,
        )

    # ------------------------------------------------------------ synthese

    def realized_pnl(self, since: datetime | None = None) -> float:
        """P&L realise cumule depuis une date."""
        trades = self.closed_trades
        if since is not None:
            cutoff = to_utc(since)
            trades = [trade for trade in trades if to_utc(trade.closed_at) >= cutoff]
        return float(sum(trade.pnl for trade in trades))

    def snapshot(self, prices: dict[str, float] | None = None) -> dict[str, float | int]:
        """Etat courant du portefeuille, pour le monitoring et la persistence."""
        equity = self.equity(prices)
        drawdown = self.drawdowns()
        return {
            "equity": round(equity, 4),
            "cash": round(self.cash, 4),
            "initial_capital": self.initial_capital,
            "total_return_pct": round((equity / self.initial_capital - 1.0) * 100.0, 4),
            "exposure_pct": round(self.exposure_pct(prices), 4),
            "open_positions": self.open_count,
            "closed_trades": len(self.closed_trades),
            "drawdown_daily_pct": round(drawdown.daily_pct, 4),
            "drawdown_weekly_pct": round(drawdown.weekly_pct, 4),
            "drawdown_total_pct": round(drawdown.total_pct, 4),
            "realized_pnl": round(self.realized_pnl(), 4),
        }

    def reset_to(self, equity: float) -> None:
        """Reinitialise le portefeuille (usage tests et redemarrage a froid)."""
        self.cash = float(equity)
        self.positions.clear()
        self._last_prices.clear()
        self.peak_equity = float(equity)
        self.equity_history.clear()
        self.equity_history.append(EquityPoint(utc_now(), float(equity)))


def position_from_fill(
    asset: str,
    side: OrderSide,
    size: float,
    fill_price: float,
    stop_loss: float,
    target_price: float | None = None,
    strategy: str = "ensemble",
    fees: float = 0.0,
    now: datetime | None = None,
    metadata: dict | None = None,
) -> Position:
    """Construit une Position a partir d'une execution."""
    return Position(
        asset=asset,
        side=side,
        size=float(size),
        entry_price=float(fill_price),
        stop_loss=float(stop_loss),
        target_price=target_price,
        strategy=strategy,
        opened_at=to_utc(now or utc_now()),
        fees_paid=float(fees),
        metadata=metadata or {},
    )

"""Moteur de backtest actions, barre par barre, sans look-ahead possible.

Regles d'execution, toutes destinees a empecher un backtest menteur :

1. Le signal est evalue sur la CLOTURE de la barre t, avec les seules donnees
   0..t. L'ordre est execute a l'OUVERTURE de la barre t+1. On ne trade jamais
   le prix qui vient de declencher le signal.
2. Les stops sont evalues sur le bas (ou le haut) de la barre suivante. Si la
   barre ouvre deja au-dela du stop (gap), la sortie se fait a l'ouverture — pas
   au niveau theorique du stop, qui n'a jamais existe ce jour-la.
3. Frais de courtage et slippage sont appliques a chaque entree et chaque sortie.
4. Le dimensionnement se fait sur le RISQUE : la perte si le stop initial est
   touche vaut `risk_per_trade_pct` du capital. C'est la seule facon de comparer
   des titres a 30 $ et a 900 $.

Le portefeuille est multi-titres : plusieurs positions peuvent coexister, dans la
limite de `max_positions` et de l'exposition totale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from trader.equities.strategy import (
    TrendParams,
    compute_indicators,
    entry_signal,
    exit_price_level,
    initial_stop,
)
from trader.logging_setup import get_logger
from trader.utils.math_utils import (
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)

log = get_logger(__name__)

TRADING_DAYS_PER_YEAR: float = 252.0


@dataclass(frozen=True, slots=True)
class ExecutionCosts:
    """Frictions d'execution appliquees a chaque ordre.

    Valeurs volontairement prudentes pour un courtier retail europeen sur
    actions americaines : un backtest optimiste sur les couts est un backtest
    qui ment sur le resultat.
    """

    commission_pct: float = 0.10
    slippage_pct: float = 0.05
    min_commission: float = 1.0

    def entry_price(self, price: float) -> float:
        """Prix reellement paye a l'achat."""
        return price * (1.0 + self.slippage_pct / 100.0)

    def exit_price(self, price: float) -> float:
        """Prix reellement obtenu a la vente."""
        return price * (1.0 - self.slippage_pct / 100.0)

    def commission(self, notional: float) -> float:
        """Commission sur un ordre."""
        return max(self.min_commission, notional * self.commission_pct / 100.0)


@dataclass(frozen=True, slots=True)
class RiskParams:
    """Regles de risque du portefeuille actions."""

    risk_per_trade_pct: float = 1.0
    max_position_pct: float = 35.0
    max_exposure_pct: float = 100.0
    max_positions: int = 3
    allow_fractional_shares: bool = True
    sizing_mode: str = "risk"
    """'risk' : la perte au stop vaut risk_per_trade_pct du capital.
    'target_weight' : chaque position vise max_position_pct du capital.

    Le mode 'risk' borne la perte mais laisse le capital majoritairement en
    cash quand les stops sont larges — sur un titre a 40 % de volatilite, il
    n'engage qu'une dizaine de pour cent du capital et ne capte presque rien
    d'une hausse. Le mode 'target_weight' investit reellement ; c'est le stop
    qui borne la perte, pas la taille.
    """


@dataclass(slots=True)
class EquityTrade:
    """Trade cloture."""

    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    costs: float
    exit_reason: str

    @property
    def return_pct(self) -> float:
        """Rendement du trade en % du notionnel engage."""
        notional = self.shares * self.entry_price
        return self.pnl / notional * 100.0 if notional > 0 else 0.0

    @property
    def holding_days(self) -> int:
        """Duree de detention en jours calendaires."""
        return (self.exit_date - self.entry_date).days

    def to_dict(self) -> dict:
        """Representation serialisable."""
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date.date().isoformat(),
            "exit_date": self.exit_date.date().isoformat(),
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "shares": round(self.shares, 6),
            "pnl": round(self.pnl, 2),
            "costs": round(self.costs, 2),
            "return_pct": round(self.return_pct, 2),
            "holding_days": self.holding_days,
            "exit_reason": self.exit_reason,
        }


@dataclass(slots=True)
class OpenPosition:
    """Position ouverte."""

    symbol: str
    entry_date: datetime
    entry_price: float
    shares: float
    stop: float
    highest_close: float
    entry_costs: float


@dataclass(slots=True)
class BacktestReport:
    """Resultat complet d'un backtest actions."""

    equity: pd.Series
    trades: list[EquityTrade]
    initial_capital: float
    start: date
    end: date
    benchmark: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metrics: dict[str, float] = field(default_factory=dict)
    open_positions: list[str] = field(default_factory=list)
    open_details: list[dict] = field(default_factory=list)
    """Positions encore ouvertes a la fin de la fenetre.

    C'est l'information la plus actionnable du rapport : elle dit ou en est le
    capital MAINTENANT et a quel niveau chaque position sortirait. Un rapport
    qui ne montre que les trades clotures cache la moitie de l'exposition.
    """

    @property
    def final_equity(self) -> float:
        """Capital final."""
        return float(self.equity.iloc[-1]) if len(self.equity) else self.initial_capital

    @property
    def total_return_pct(self) -> float:
        """Rendement total en %."""
        return (self.final_equity / self.initial_capital - 1.0) * 100.0

    def summary(self, currency: str = "EUR") -> str:
        """Resume lisible."""
        metrics = self.metrics
        benchmark_line = ""
        if len(self.benchmark):
            benchmark_return = (float(self.benchmark.iloc[-1]) / self.initial_capital - 1.0) * 100.0
            benchmark_dd = self.metrics.get("benchmark_max_drawdown_pct", 0.0)
            benchmark_line = (
                f"\n  Buy & hold equipondere : {float(self.benchmark.iloc[-1]):>10,.2f} {currency} "
                f"({benchmark_return:+.2f} %, drawdown max {benchmark_dd:.2f} %)"
            )
        return (
            f"Periode                : {self.start} -> {self.end}\n"
            f"  Capital initial      : {self.initial_capital:>10,.2f} {currency}\n"
            f"  Capital final        : {self.final_equity:>10,.2f} {currency} "
            f"({self.total_return_pct:+.2f} %)"
            f"{benchmark_line}\n"
            f"  Trades clotures      : {len(self.trades)}"
            f" (positions encore ouvertes : {len(self.open_positions)})\n"
            f"  Taux de reussite     : {metrics.get('hit_rate', 0.0):.0%}\n"
            f"  Profit factor        : {metrics.get('profit_factor', 0.0):.2f}\n"
            f"  Sharpe annualise     : {metrics.get('sharpe', 0.0):+.2f}\n"
            f"  Sortino annualise    : {metrics.get('sortino', 0.0):+.2f}\n"
            f"  Drawdown maximal     : {metrics.get('max_drawdown_pct', 0.0):.2f} %\n"
            f"  Frais totaux payes   : {metrics.get('total_costs', 0.0):,.2f} {currency}"
        )


class EquityBacktester:
    """Rejoue la strategie sur un panier de titres."""

    def __init__(
        self,
        params: TrendParams | None = None,
        risk: RiskParams | None = None,
        costs: ExecutionCosts | None = None,
    ) -> None:
        self.params = params or TrendParams()
        self.risk = risk or RiskParams()
        self.costs = costs or ExecutionCosts()

    def run(
        self,
        frames: dict[str, pd.DataFrame],
        start: date,
        end: date,
        initial_capital: float = 1000.0,
    ) -> BacktestReport:
        """Execute le backtest sur la fenetre [start, end].

        Les donnees ANTERIEURES a `start` presentes dans `frames` servent
        uniquement a alimenter les indicateurs (moyenne 200 jours, ATR). Aucun
        trade n'est pris avant `start`, et aucune donnee posterieure a la barre
        courante n'est jamais lue.
        """
        if not frames:
            raise ValueError("aucun titre fourni")

        indicators = {
            symbol: self._prepare(frame) for symbol, frame in frames.items() if len(frame) > 30
        }
        calendar = self._trading_calendar(frames, start, end)
        if len(calendar) < 2:
            raise ValueError(f"calendrier de trading trop court entre {start} et {end}")

        cash = float(initial_capital)
        positions: dict[str, OpenPosition] = {}
        trades: list[EquityTrade] = []
        equity_points: list[tuple[datetime, float]] = []
        total_costs = 0.0
        pending_entries: list[str] = []
        last_exit_index: dict[str, int] = {}

        for index in range(len(calendar) - 1):
            today = calendar[index]
            tomorrow = calendar[index + 1]

            # 1. Executer les entrees decidees hier, a l'ouverture d'aujourd'hui.
            for symbol in pending_entries:
                opened, spent, cost = self._open_position(
                    symbol, frames[symbol], indicators[symbol], today, cash, positions
                )
                if opened is not None:
                    positions[symbol] = opened
                    cash -= spent
                    total_costs += cost
            pending_entries = []

            # 2. Gerer les sorties sur la seance du jour.
            for symbol in list(positions):
                closed, proceeds, cost = self._check_exit(
                    positions[symbol], frames[symbol], indicators[symbol], today
                )
                if closed is not None:
                    trades.append(closed)
                    cash += proceeds
                    total_costs += cost
                    del positions[symbol]
                    last_exit_index[symbol] = index

            # 3. Decider des entrees de demain, sur les donnees closes ce soir.
            equity = cash + self._positions_value(positions, frames, today)
            pending_entries = self._select_entries(
                frames, indicators, positions, today, equity, cash, last_exit_index, index
            )
            equity_points.append((today, equity))

            if tomorrow is None:  # pragma: no cover - securite
                break

        # Cloture finale : on marque les positions restantes au dernier cours.
        last_day = calendar[-1]
        final_equity = cash + self._positions_value(positions, frames, last_day)
        equity_points.append((last_day, final_equity))

        curve = pd.Series(
            [value for _, value in equity_points],
            index=pd.DatetimeIndex([stamp for stamp, _ in equity_points]),
            name="equity",
        )
        report = BacktestReport(
            equity=curve,
            trades=trades,
            initial_capital=float(initial_capital),
            start=start,
            end=end,
            benchmark=self._benchmark(frames, calendar, initial_capital),
            open_positions=sorted(positions),
            open_details=self._open_details(positions, frames, last_day),
        )
        report.metrics = self._metrics(report, total_costs)
        return report

    # ------------------------------------------------------------- mecanique

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Calcule les indicateurs d'un titre."""
        indicators = compute_indicators(frame, self.params)
        indicators.attrs["min_adx"] = self.params.min_adx
        return indicators

    @staticmethod
    def _trading_calendar(
        frames: dict[str, pd.DataFrame], start: date, end: date
    ) -> list[datetime]:
        """Calendrier commun des seances, borne a la fenetre demandee."""
        index = None
        for frame in frames.values():
            index = frame.index if index is None else index.union(frame.index)
        if index is None:
            return []
        window = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
        return list(window.to_pydatetime())

    def _select_entries(
        self,
        frames: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
        positions: dict[str, OpenPosition],
        today: datetime,
        equity: float,
        cash: float,
        last_exit_index: dict[str, int] | None = None,
        day_index: int = 0,
    ) -> list[str]:
        """Selectionne les titres a acheter a l'ouverture suivante.

        En cas de signaux simultanes, on privilegie la tendance la plus etablie
        (ADX le plus eleve) : arbitrer par ordre alphabetique introduirait un
        biais silencieux vers le debut de l'alphabet.

        Un titre stoppe recemment est ecarte pendant `reentry_cooldown_days`
        seances : le racheter immediatement revient a repayer les frais pour
        reprendre le mouvement qui vient justement de faire sauter le stop.
        """
        if len(positions) >= self.risk.max_positions:
            return []

        cooldown = self.params.reentry_cooldown_days
        exits = last_exit_index or {}

        candidates: list[tuple[float, str]] = []
        for symbol in frames:
            if symbol in positions or symbol not in indicators:
                continue
            if cooldown > 0 and day_index - exits.get(symbol, -(10**9)) < cooldown:
                continue
            history = indicators[symbol].loc[:today]
            if len(history) < self.params.min_history:
                continue
            if entry_signal(history, position=0, mode=self.params.entry_mode):
                candidates.append((float(history["adx"].iloc[-1]), symbol))

        candidates.sort(reverse=True)
        free_slots = self.risk.max_positions - len(positions)
        return [symbol for _, symbol in candidates[:free_slots]]

    def _open_position(
        self,
        symbol: str,
        frame: pd.DataFrame,
        indicators: pd.DataFrame,
        today: datetime,
        cash: float,
        positions: dict[str, OpenPosition],
    ) -> tuple[OpenPosition | None, float, float]:
        """Ouvre une position a l'ouverture du jour."""
        if today not in frame.index:
            return None, 0.0, 0.0

        raw_open = float(frame.loc[today, "open"])
        history = indicators.loc[:today]
        previous = history.iloc[-2] if len(history) >= 2 else history.iloc[-1]
        atr_value = float(previous["atr"])
        if not np.isfinite(atr_value) or atr_value <= 0 or raw_open <= 0:
            return None, 0.0, 0.0

        fill_price = self.costs.entry_price(raw_open)
        stop = initial_stop(fill_price, atr_value, self.params)
        stop_distance = fill_price - stop
        if stop_distance <= 0:
            return None, 0.0, 0.0

        equity = cash + sum(
            position.shares * float(frame.loc[today, "close"])
            for position in positions.values()
            if position.symbol == symbol
        )
        equity = max(equity, cash)

        # Dimensionnement : par le risque, ou par allocation cible.
        max_notional = min(equity * self.risk.max_position_pct / 100.0, cash * 0.98)
        if self.risk.sizing_mode == "target_weight":
            shares = max_notional / fill_price
        else:
            risk_amount = equity * self.risk.risk_per_trade_pct / 100.0
            shares = min(risk_amount / stop_distance, max_notional / fill_price)
        if not self.risk.allow_fractional_shares:
            shares = float(int(shares))
        if shares <= 0:
            return None, 0.0, 0.0

        notional = shares * fill_price
        commission = self.costs.commission(notional)
        if notional + commission > cash:
            return None, 0.0, 0.0

        position = OpenPosition(
            symbol=symbol,
            entry_date=today,
            entry_price=fill_price,
            shares=shares,
            stop=stop,
            highest_close=float(frame.loc[today, "close"]),
            entry_costs=commission,
        )
        log.debug(
            "equity_position_opened",
            symbol=symbol,
            date=str(today.date()),
            price=round(fill_price, 2),
            shares=round(shares, 4),
            stop=round(stop, 2),
        )
        return position, notional + commission, commission

    def _check_exit(
        self,
        position: OpenPosition,
        frame: pd.DataFrame,
        indicators: pd.DataFrame,
        today: datetime,
    ) -> tuple[EquityTrade | None, float, float]:
        """Verifie si la position doit etre fermee sur la seance du jour."""
        if today not in frame.index or today <= position.entry_date:
            return None, 0.0, 0.0

        bar = frame.loc[today]
        low, open_price, close = float(bar["low"]), float(bar["open"]), float(bar["close"])
        history = indicators.loc[:today]
        if len(history) < 2:
            return None, 0.0, 0.0

        # Le stop suiveur du jour se calcule sur les donnees d'HIER : il etait
        # deja connu a l'ouverture, contrairement au cours du jour meme.
        yesterday = history.iloc[:-1]
        trailing = exit_price_level(yesterday, self.params, position.highest_close)
        stop_level = max(position.stop, trailing)

        exit_raw: float | None = None
        reason = ""
        if open_price <= stop_level:
            # Gap d'ouverture sous le stop : on sort au prix reel, pas au niveau
            # theorique du stop qui n'a jamais ete cote.
            exit_raw, reason = open_price, "gap_sous_stop"
        elif low <= stop_level:
            exit_raw, reason = stop_level, "stop_suiveur"

        if exit_raw is None:
            position.highest_close = max(position.highest_close, close)
            position.stop = stop_level
            return None, 0.0, 0.0

        fill_price = self.costs.exit_price(exit_raw)
        notional = position.shares * fill_price
        commission = self.costs.commission(notional)
        pnl = (
            (fill_price - position.entry_price) * position.shares
            - commission
            - position.entry_costs
        )
        trade = EquityTrade(
            symbol=position.symbol,
            entry_date=position.entry_date,
            exit_date=today,
            entry_price=position.entry_price,
            exit_price=fill_price,
            shares=position.shares,
            pnl=pnl,
            costs=commission + position.entry_costs,
            exit_reason=reason,
        )
        log.debug(
            "equity_position_closed",
            symbol=position.symbol,
            date=str(today.date()),
            pnl=round(pnl, 2),
            reason=reason,
        )
        return trade, notional - commission, commission

    def _open_details(
        self,
        positions: dict[str, OpenPosition],
        frames: dict[str, pd.DataFrame],
        today: datetime,
    ) -> list[dict]:
        """Etat detaille des positions encore ouvertes a la derniere seance."""
        details: list[dict] = []
        for symbol, position in sorted(positions.items()):
            frame = frames[symbol]
            price = (
                float(frame.loc[today, "close"]) if today in frame.index else position.entry_price
            )
            notional = position.shares * price
            unrealized = (price - position.entry_price) * position.shares - position.entry_costs
            details.append(
                {
                    "symbol": symbol,
                    "entry_date": position.entry_date.date().isoformat(),
                    "entry_price": round(position.entry_price, 2),
                    "cours": round(price, 2),
                    "shares": round(position.shares, 6),
                    "valeur": round(notional, 2),
                    "pnl_latent": round(unrealized, 2),
                    "pnl_latent_%": round(
                        unrealized / (position.shares * position.entry_price) * 100.0, 2
                    )
                    if position.shares * position.entry_price > 0
                    else 0.0,
                    "stop": round(position.stop, 2),
                    "marge_avant_stop_%": round((position.stop / price - 1.0) * 100.0, 2)
                    if price > 0
                    else 0.0,
                }
            )
        return details

    @staticmethod
    def _positions_value(
        positions: dict[str, OpenPosition], frames: dict[str, pd.DataFrame], today: datetime
    ) -> float:
        """Valeur mark-to-market des positions ouvertes."""
        total = 0.0
        for symbol, position in positions.items():
            frame = frames[symbol]
            if today in frame.index:
                total += position.shares * float(frame.loc[today, "close"])
            else:
                total += position.shares * position.entry_price
        return total

    def _benchmark(
        self, frames: dict[str, pd.DataFrame], calendar: list[datetime], capital: float
    ) -> pd.Series:
        """Buy & hold equipondere sur l'univers, frais d'entree inclus."""
        if not calendar:
            return pd.Series(dtype=float)
        usable = {
            symbol: frame
            for symbol, frame in frames.items()
            if calendar[0] in frame.index and calendar[-1] in frame.index
        }
        if not usable:
            return pd.Series(dtype=float)

        allocation = capital / len(usable)
        shares = {}
        for symbol, frame in usable.items():
            entry = self.costs.entry_price(float(frame.loc[calendar[0], "open"]))
            commission = self.costs.commission(allocation)
            shares[symbol] = max(0.0, (allocation - commission)) / entry

        values = []
        for day in calendar:
            total = 0.0
            for symbol, quantity in shares.items():
                frame = usable[symbol]
                price = float(frame.loc[day, "close"]) if day in frame.index else np.nan
                total += quantity * price if np.isfinite(price) else 0.0
            values.append(total)
        return pd.Series(values, index=pd.DatetimeIndex(calendar), name="buy_and_hold")

    def _metrics(self, report: BacktestReport, total_costs: float) -> dict[str, float]:
        """Metriques de performance du backtest."""
        equity = report.equity
        returns = equity.pct_change().dropna()
        pnl = np.array([trade.pnl for trade in report.trades], dtype=float)
        return {
            "sharpe": sharpe_ratio(returns, TRADING_DAYS_PER_YEAR),
            "sortino": sortino_ratio(returns, TRADING_DAYS_PER_YEAR),
            "max_drawdown_pct": max_drawdown(equity) * 100.0,
            "hit_rate": hit_rate(pnl) if pnl.size else 0.0,
            "profit_factor": min(profit_factor(pnl), 1e6) if pnl.size else 0.0,
            "trades": float(pnl.size),
            "avg_trade_pnl": float(np.mean(pnl)) if pnl.size else 0.0,
            "best_trade": float(np.max(pnl)) if pnl.size else 0.0,
            "worst_trade": float(np.min(pnl)) if pnl.size else 0.0,
            "total_costs": total_costs,
            "exposure_days": float(sum(t.holding_days for t in report.trades)),
            "benchmark_max_drawdown_pct": (
                max_drawdown(report.benchmark) * 100.0 if len(report.benchmark) else 0.0
            ),
            "benchmark_return_pct": (
                (float(report.benchmark.iloc[-1]) / report.initial_capital - 1.0) * 100.0
                if len(report.benchmark)
                else 0.0
            ),
        }

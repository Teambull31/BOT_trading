"""Simulation a effet de levier, longs et shorts, avec appel de marge reel.

Trois mecanismes sont modelises parce qu'ils decident du resultat a fort levier,
et qu'un backtest qui les omet ment de plusieurs ordres de grandeur :

1. LIQUIDATION. Le courtier ferme la position des que les fonds propres passent
   sous la marge de maintenance. A levier L, la marge initiale vaut 1/L du
   notionnel ; un mouvement adverse de quelques pour cent suffit donc a tout
   effacer. A L=30, la marge initiale est de 3.3 % : un titre qui recule de
   3.3 % dans la journee liquide le compte.

2. GAP D'OUVERTURE. La liquidation n'est pas garantie au niveau theorique. Si
   la seance OUVRE au-dela du seuil, la position est fermee a l'ouverture, plus
   bas. Les fonds propres peuvent alors devenir NEGATIFS : on doit de l'argent
   au courtier. Le simulateur l'autorise explicitement au lieu de plancher a
   zero, sans quoi le risque reel disparait du resultat.

3. PORTAGE. A levier L, on emprunte (L-1) fois ses fonds propres. A 5 % l'an et
   L=30, cela coute 29 x 5 % = 145 % des fonds propres PAR AN, soit environ
   0.40 % par jour de detention. Les shorts paient en plus un loyer de titre.
   C'est le cout que les demonstrations de levier oublient le plus souvent.

Les frictions (commission, slippage) s'appliquent au NOTIONNEL, pas aux fonds
propres : a levier 30, un slippage de 0.05 % coute 1.5 % du compte a chaque
aller-retour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

TRADING_DAYS: float = 252.0


@dataclass(frozen=True, slots=True)
class LeverageParams:
    """Reglages du compte a levier."""

    leverage: float = 1.0
    annual_financing_pct: float = 5.0
    """Taux d'emprunt annuel sur la part financee (L-1 fois les fonds propres)."""
    short_borrow_pct: float = 1.0
    """Loyer annuel du titre emprunte, paye en plus sur les positions vendeuses."""
    maintenance_ratio: float = 0.5
    """Marge de maintenance en fraction de la marge initiale. 0.5 = liquidation
    quand la moitie de la marge initiale a ete consommee."""
    commission_pct: float = 0.02
    slippage_pct: float = 0.05

    @property
    def initial_margin_pct(self) -> float:
        """Marge initiale en % du notionnel."""
        return 100.0 / self.leverage

    @property
    def ruin_move_pct(self) -> float:
        """Mouvement adverse, en %, qui declenche la liquidation."""
        return self.initial_margin_pct * self.maintenance_ratio


@dataclass(slots=True)
class LeverageResult:
    """Resultat d'une simulation a levier."""

    final_equity: float
    initial_capital: float
    ruined: bool
    ruin_date: datetime | None
    liquidations: int
    trades: int
    max_drawdown_pct: float
    financing_paid: float
    friction_paid: float
    worst_equity: float
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def total_return_pct(self) -> float:
        """Rendement total en %. Peut descendre sous -100 % (dette)."""
        return (self.final_equity / self.initial_capital - 1.0) * 100.0


def simulate_leveraged(
    frame: pd.DataFrame,
    direction: pd.Series,
    params: LeverageParams,
    initial_capital: float = 1000.0,
) -> LeverageResult:
    """Rejoue une serie de positions longues/courtes a levier.

    `direction` vaut +1 (long), -1 (short) ou 0 (hors marche) pour chaque
    seance. La decision de la seance t est EXECUTEE a l'ouverture de t+1 :
    comme partout ailleurs dans ce projet, on ne trade jamais le prix qui vient
    de produire le signal.
    """
    dates = frame.index
    open_, high, low, close = (
        frame["open"].to_numpy(),
        frame["high"].to_numpy(),
        frame["low"].to_numpy(),
        frame["close"].to_numpy(),
    )
    wanted = direction.reindex(dates).fillna(0.0).to_numpy()

    equity = float(initial_capital)
    position = 0.0  # nombre de titres, signe
    side = 0
    entry_notional = 0.0
    liquidation_price = np.nan

    curve: list[float] = []
    peak = equity
    worst = equity
    max_dd = 0.0
    liquidations = 0
    trades = 0
    financing_paid = 0.0
    friction_paid = 0.0
    ruin_date: datetime | None = None
    daily_financing = params.annual_financing_pct / 100.0 / TRADING_DAYS
    daily_borrow = params.short_borrow_pct / 100.0 / TRADING_DAYS

    for index in range(len(dates) - 1):
        # 1. Executer a l'ouverture d'aujourd'hui la decision prise hier soir.
        target = int(wanted[index - 1]) if index > 0 else 0
        if target != side and equity > 0:
            if side != 0:
                equity, cost = _close(equity, position, side, open_[index], entry_notional, params)
                friction_paid += cost
                position, side, entry_notional = 0.0, 0, 0.0
            if target != 0 and equity > 0:
                notional = equity * params.leverage
                fill = open_[index] * (1.0 + target * params.slippage_pct / 100.0)
                position = notional / fill * target
                side = target
                entry_notional = notional
                cost = notional * params.commission_pct / 100.0
                equity -= cost
                friction_paid += cost
                trades += 1
                liquidation_price = _liquidation_price(fill, equity, position, side, params)

        # 2. La seance se deroule : liquidation eventuelle, au pire prix reel.
        if side != 0:
            gapped = (side > 0 and open_[index] <= liquidation_price) or (
                side < 0 and open_[index] >= liquidation_price
            )
            breached = (side > 0 and low[index] <= liquidation_price) or (
                side < 0 and high[index] >= liquidation_price
            )
            if gapped or breached:
                # Sur gap, le seuil theorique n'a jamais existe : on sort a
                # l'ouverture, ce qui peut laisser les fonds propres negatifs.
                fill = open_[index] if gapped else liquidation_price
                equity, cost = _close(equity, position, side, fill, entry_notional, params)
                friction_paid += cost
                position, side, entry_notional = 0.0, 0, 0.0
                liquidations += 1
                if equity <= 0 and ruin_date is None:
                    ruin_date = dates[index]

        # 3. Portage de la nuit, sur la part empruntee.
        if side != 0:
            borrowed = max(0.0, abs(position) * close[index] - equity)
            charge = borrowed * daily_financing
            if side < 0:
                charge += abs(position) * close[index] * daily_borrow
            equity -= charge
            financing_paid += charge

        marked = (
            equity
            if side == 0
            else _mark_to_market(equity, position, side, close[index], entry_notional)
        )
        curve.append(marked)
        peak = max(peak, marked)
        worst = min(worst, marked)
        if peak > 0:
            max_dd = max(max_dd, (peak - marked) / peak * 100.0)
        if marked <= 0:
            if ruin_date is None:
                ruin_date = dates[index]
            break

    # Cloture finale au dernier cours traite, pour ne pas laisser une position
    # ouverte valorisee a un prix que la simulation n'a jamais atteint.
    if side != 0 and equity > 0 and curve:
        equity, cost = _close(equity, position, side, close[len(curve) - 1], entry_notional, params)
        friction_paid += cost

    return LeverageResult(
        final_equity=float(equity),
        initial_capital=float(initial_capital),
        ruined=ruin_date is not None,
        ruin_date=ruin_date,
        liquidations=liquidations,
        trades=trades,
        max_drawdown_pct=float(max_dd),
        financing_paid=float(financing_paid),
        friction_paid=float(friction_paid),
        worst_equity=float(worst),
        equity=pd.Series(curve, index=dates[: len(curve)], name="equity"),
    )


def _entry_price(position: float, entry_notional: float) -> float:
    """Prix d'entree reconstitue depuis le notionnel."""
    return entry_notional / abs(position) if position else 0.0


def _mark_to_market(
    equity: float, position: float, side: int, price: float, entry_notional: float
) -> float:
    """Fonds propres valorises au cours du jour."""
    entry = _entry_price(position, entry_notional)
    return equity + position * (price - entry)


def _close(
    equity: float,
    position: float,
    side: int,
    price: float,
    entry_notional: float,
    params: LeverageParams,
) -> tuple[float, float]:
    """Ferme la position au prix donne, frictions incluses."""
    fill = price * (1.0 - side * params.slippage_pct / 100.0)
    entry = _entry_price(position, entry_notional)
    pnl = position * (fill - entry)
    notional = abs(position) * fill
    cost = notional * params.commission_pct / 100.0
    return equity + pnl - cost, cost


def _liquidation_price(
    entry: float, equity: float, position: float, side: int, params: LeverageParams
) -> float:
    """Cours auquel le courtier ferme d'office.

    Les fonds propres absorbent la perte jusqu'a la marge de maintenance ; le
    reste du notionnel est finance. La distance au seuil ne depend donc que du
    levier, pas du titre — d'ou le fait qu'un titre volatil soit liquide bien
    plus souvent a levier identique.
    """
    if position == 0:
        return np.nan
    maintenance = equity * params.maintenance_ratio
    distance = (equity - maintenance) / abs(position)
    return entry - side * distance


def direction_from_probability(
    probability: pd.Series, long_threshold: float = 0.55, short_threshold: float = 0.45
) -> pd.Series:
    """Traduit une probabilite de hausse en position longue, courte ou nulle.

    Au-dessus de `long_threshold` on achete, en dessous de `short_threshold` on
    vend a decouvert, entre les deux on reste hors marche. Un seuil symetrique
    autour de 0.5 evite d'introduire un biais directionnel implicite.
    """
    return pd.Series(
        np.where(
            probability > long_threshold, 1.0, np.where(probability < short_threshold, -1.0, 0.0)
        ),
        index=probability.index,
    ).where(probability.notna(), 0.0)

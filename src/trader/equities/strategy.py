"""Strategie de suivi de tendance actions — parametres standards, non optimises.

Pourquoi ce style : le trend-following est le seul style documente comme ayant
survecu plusieurs decennies sur des marches tres differents. Il ne cherche pas a
predire : il constate une tendance et la suit tant qu'elle dure. C'est ce qui le
rend PERENNE — il ne repose sur aucune inefficience precise qui pourrait
disparaitre, seulement sur le fait que les marches produisent parfois des
mouvements soutenus.

Les parametres sont ceux de la litterature classique, NON optimises sur les
donnees testees :
- filtre de tendance SMA 200 jours (Faber, tactical asset allocation) ;
- cassure Donchian 20 jours (systeme des Turtles, 1983) ;
- stop suiveur a 3 x ATR(14) sous le plus haut atteint (chandelier exit, Le Beau).

Ne pas optimiser ces valeurs est un CHOIX, pas une paresse : quatre parametres
ajustes sur trois mois de donnees produiraient une courbe magnifique et sans
aucune valeur predictive.

Toutes les series d'indicateurs sont CAUSALES : la valeur en t ne depend que des
observations jusqu'a t incluse. La verification est faite empiriquement par
`assert_signals_causal`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trader.data.features import adx, atr


@dataclass(frozen=True, slots=True)
class TrendParams:
    """Parametres de la strategie. Valeurs standards de la litterature."""

    trend_filter_days: int = 200
    breakout_days: int = 20
    exit_days: int = 10
    atr_period: int = 14
    trailing_atr: float = 3.0
    initial_stop_atr: float = 2.5
    adx_period: int = 14
    min_adx: float = 15.0
    min_history: int = 210
    reentry_cooldown_days: int = 0
    """Seances d'attente avant de racheter un titre qui vient d'etre stoppe.

    Laisse a 0 (desactive) apres mesure. L'intuition est pourtant solide : sans
    delai, un titre qui oscille autour de sa moyenne longue est rachete des le
    lendemain de sa sortie, stoppe a nouveau, rachete encore, chaque aller-retour
    coutant des frais et une petite perte.

    Le balayage in-sample (2023-10 -> 2025-12, cinq univers, valeurs 0/3/5/10/15/
    20/30) donne une courbe NON MONOTONE : 0 est bon, 3 a 20 sont tous moins bons,
    30 est le meilleur. Une vraie amelioration serait progressive ; cette forme
    en U signale du bruit. Verification par sous-periodes : le delai de 30 jours
    ne gagne que 2 semestres sur 5, et tout son avantage vient de 2024H2, un
    marche sans direction ou un delai aide mecaniquement. En periode de tendance
    il fait manquer des reprises et coute.

    Retenir 30 reviendrait a choisir la meilleure de sept valeurs testees sur une
    seule fenetre : exactement l'optimisation qui produit une belle courbe
    passee et aucune performance future.
    """
    entry_mode: str = "breakout"
    """'breakout' : cassure Donchian. 'trend' : rester investi au-dessus de la SMA200.

    Le mode 'trend' est le systeme de Faber : on detient le titre tant qu'il est
    au-dessus de sa moyenne longue, on sort quand il repasse dessous. Beaucoup
    moins de trades, il capture l'essentiel des grandes tendances au lieu de se
    faire sortir par le bruit a chaque respiration du marche.
    """

    def describe(self) -> str:
        """Description lisible des reglages."""
        if self.entry_mode == "trend":
            return (
                f"suivi de tendance SMA{self.trend_filter_days} (entree des que le "
                f"cours repasse au-dessus, sortie quand il repasse dessous), "
                f"stop de securite {self.initial_stop_atr}xATR / {self.trailing_atr}xATR"
            )
        return (
            f"SMA{self.trend_filter_days} + cassure Donchian {self.breakout_days}j, "
            f"sortie {self.exit_days}j, stop initial {self.initial_stop_atr}xATR, "
            f"stop suiveur {self.trailing_atr}xATR, ADX>{self.min_adx:.0f}"
        )


def compute_indicators(frame: pd.DataFrame, params: TrendParams) -> pd.DataFrame:
    """Calcule les indicateurs causaux necessaires aux signaux.

    Les extremes Donchian sont decales d'une barre (`shift(1)`) : le plus haut
    des 20 derniers jours ne doit PAS inclure la barre courante, sinon toute
    cloture devient mecaniquement une cassure de son propre plus haut.
    """
    out = pd.DataFrame(index=frame.index)
    close, high, low = frame["close"], frame["high"], frame["low"]

    out["close"] = close
    out["sma_trend"] = close.rolling(
        params.trend_filter_days, min_periods=params.trend_filter_days
    ).mean()
    out["donchian_high"] = (
        high.rolling(params.breakout_days, min_periods=params.breakout_days).max().shift(1)
    )
    out["donchian_low"] = low.rolling(params.exit_days, min_periods=params.exit_days).min().shift(1)
    out["atr"] = atr(high, low, close, params.atr_period)
    out["adx"] = adx(high, low, close, params.adx_period)[0]
    out["atr_pct"] = out["atr"] / close * 100.0
    return out


def entry_signal(indicators: pd.DataFrame, position: int, mode: str = "breakout") -> bool:
    """Vrai si les conditions d'entree longue sont reunies a la derniere barre.

    Mode 'breakout' — trois conditions cumulatives :
    1. le titre est au-dessus de sa moyenne 200 jours (regime haussier) ;
    2. la cloture depasse le plus haut des 20 seances precedentes (cassure) ;
    3. l'ADX confirme qu'une tendance existe (filtre anti-marche sans direction).

    Mode 'trend' — une seule condition : le titre est au-dessus de sa moyenne
    200 jours. On accepte d'entrer "en retard" pour ne jamais rater une grande
    tendance ; sur actions, la hausse se concentre dans un petit nombre de
    seances, et un systeme trop selectif les manque.
    """
    if position != 0 or indicators.empty:
        return False
    last = indicators.iloc[-1]
    if last[["sma_trend", "atr"]].isna().any() or last["atr"] <= 0:
        return False
    above_trend = bool(last["close"] > last["sma_trend"])
    if mode == "trend":
        return above_trend
    if last[["donchian_high", "adx"]].isna().any():
        return False
    return bool(
        above_trend
        and last["close"] > last["donchian_high"]
        and last["adx"] >= _min_adx(indicators)
    )


def _min_adx(indicators: pd.DataFrame) -> float:
    """Seuil d'ADX attache aux indicateurs (evite de repasser les params partout)."""
    return float(indicators.attrs.get("min_adx", 15.0))


def exit_price_level(indicators: pd.DataFrame, params: TrendParams, highest_close: float) -> float:
    """Niveau de sortie courant.

    Mode 'breakout' : le plus PROTECTEUR du stop suiveur (chandelier) et du
    plancher Donchian 10 jours — le premier protege le gain latent, le second
    sort quand la structure de tendance se casse.

    Mode 'trend' : la moyenne 200 jours sert de sortie, doublee d'un stop de
    securite tres large. Un stop serre annulerait tout l'interet du mode, qui
    est justement de laisser respirer la position.
    """
    last = indicators.iloc[-1]
    chandelier = highest_close - params.trailing_atr * float(last["atr"])
    if params.entry_mode == "trend":
        sma = float(last["sma_trend"]) if not pd.isna(last["sma_trend"]) else -np.inf
        return max(chandelier, sma)
    donchian = float(last["donchian_low"]) if not pd.isna(last["donchian_low"]) else -np.inf
    return max(chandelier, donchian)


def initial_stop(entry_price: float, atr_value: float, params: TrendParams) -> float:
    """Stop initial, place sous le prix d'entree a `initial_stop_atr` x ATR."""
    return entry_price - params.initial_stop_atr * atr_value


def assert_signals_causal(
    frame: pd.DataFrame, params: TrendParams, cut_ratio: float = 0.7
) -> list[str]:
    """Verifie empiriquement qu'aucun indicateur ne lit le futur.

    Meme principe que pour les features crypto : recalculer sur un prefixe des
    donnees doit donner exactement les memes valeurs sur ce prefixe. Une
    divergence prouve une fuite d'information future.
    """
    cut = int(len(frame) * cut_ratio)
    full = compute_indicators(frame, params)
    prefix = compute_indicators(frame.iloc[:cut], params)

    offenders: list[str] = []
    for column in prefix.columns:
        left = prefix[column]
        right = full[column].iloc[:cut]
        valid = left.notna() & right.notna()
        if not valid.any():
            continue
        diff = (left[valid] - right[valid]).abs()
        scale = right[valid].abs().clip(lower=1.0)
        if bool(((diff / scale) > 1e-9).any()):
            offenders.append(column)
    return offenders

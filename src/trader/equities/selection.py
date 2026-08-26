"""Selection de l'univers de trading.

Principe : la selection ne doit utiliser QUE des donnees anterieures a la
fenetre de test. Choisir aujourd'hui les titres qui ont bien performe en
juin-aout serait la forme de triche la plus rentable et la plus inutile qui
soit — un backtest qui ne dit rien du futur.

Criteres, dans l'ordre :
1. liquidite suffisante (volume median en dollars) : un systeme qui trade des
   titres illiquides paie un slippage sans rapport avec le modele ;
2. DECORRELATION vis-a-vis des titres imposes. MU et ASML sont deux
   semi-conducteurs : leur correlation est elevee, et ajouter deux semis de plus
   reviendrait a miser tout le capital sur un seul cycle sectoriel. Un portefeuille
   qui ne diversifie pas n'est pas perenne, il est juste chanceux ou malchanceux.
3. TENDANCIALITE. La decorrelation seule ne suffit pas : un titre parfaitement
   decorrele mais qui oscille sans jamais tendre ne produira aucun signal
   exploitable par une strategie de suivi de tendance. Il occuperait une place
   dans l'univers sans jamais rien apporter. On mesure donc la part des seances
   passees au-dessus de la moyenne 200 jours, et le score final combine les deux :

       score = tendancialite x (1 - |correlation|)

   Ce score privilegie les titres qui tendent ET qui apportent de la
   diversification, plutot que l'un au detriment de l'autre.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from trader.equities.data import load_universe
from trader.logging_setup import get_logger

log = get_logger(__name__)

# Panier de candidats : grandes capitalisations tres liquides, secteurs varies.
CANDIDATES: tuple[str, ...] = (
    "NVDA",  # semi-conducteurs / IA
    "AMAT",  # equipement semi-conducteurs
    "TSM",  # fonderie
    "JNJ",  # sante
    "XOM",  # energie
    "PG",  # consommation de base
    "JPM",  # banque
    "WMT",  # distribution
    "KO",  # boissons
    "CAT",  # industrie lourde
    "UNH",  # assurance sante
    "GLD",  # or (ETF, decorrelation classique)
)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Evaluation d'un candidat sur la fenetre de selection."""

    symbol: str
    correlation: float
    dollar_volume_m: float
    annual_vol_pct: float
    trend_days_pct: float
    eligible: bool
    reason: str = ""

    @property
    def score(self) -> float:
        """Score combine : tendancialite ponderee par l'apport de diversification."""
        if not self.eligible:
            return 0.0
        return self.trend_days_pct * (1.0 - abs(self.correlation))

    def to_row(self) -> dict:
        """Ligne de tableau lisible."""
        return {
            "symbole": self.symbol,
            "correl_impose": round(self.correlation, 3),
            "volume_M$/j": round(self.dollar_volume_m, 1),
            "vol_annuelle_%": round(self.annual_vol_pct, 1),
            "jours_en_tendance_%": round(self.trend_days_pct, 1),
            "score": round(self.score, 1),
            "eligible": self.eligible,
            "motif": self.reason,
        }


def score_candidates(
    imposed: dict[str, pd.DataFrame],
    candidates: dict[str, pd.DataFrame],
    min_dollar_volume_m: float = 200.0,
    min_history: int = 250,
) -> list[CandidateScore]:
    """Evalue chaque candidat sur la fenetre de selection fournie."""
    imposed_returns = pd.DataFrame(
        {
            symbol: np.log(frame["close"] / frame["close"].shift(1))
            for symbol, frame in imposed.items()
        }
    ).dropna(how="all")
    reference = imposed_returns.mean(axis=1)

    scores: list[CandidateScore] = []
    for symbol, frame in candidates.items():
        if len(frame) < min_history:
            scores.append(
                CandidateScore(symbol, 1.0, 0.0, 0.0, 0.0, False, "historique insuffisant")
            )
            continue

        returns = np.log(frame["close"] / frame["close"].shift(1)).dropna()
        aligned = pd.concat([returns, reference], axis=1, join="inner").dropna()
        correlation = (
            float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) > 30 else 1.0
        )
        dollar_volume = float((frame["close"] * frame["volume"]).median()) / 1e6
        annual_vol = float(returns.std(ddof=1) * np.sqrt(252) * 100.0)

        # Part des seances passees au-dessus de la moyenne 200 jours : mesure a
        # quel point le titre offre des tendances exploitables par la strategie.
        sma = frame["close"].rolling(200, min_periods=200).mean()
        above = (frame["close"] > sma).dropna()
        trend_days = float(above.mean() * 100.0) if len(above) else 0.0

        eligible = dollar_volume >= min_dollar_volume_m
        reason = "" if eligible else f"liquidite {dollar_volume:.0f} M$/j insuffisante"
        scores.append(
            CandidateScore(
                symbol, correlation, dollar_volume, annual_vol, trend_days, eligible, reason
            )
        )
    return sorted(scores, key=lambda score: score.score, reverse=True)


def select_universe(
    imposed_symbols: list[str],
    selection_start: date,
    selection_end: date,
    count: int = 2,
    candidates: tuple[str, ...] = CANDIDATES,
) -> tuple[list[str], list[CandidateScore]]:
    """Choisit `count` titres complementaires, sur donnees anterieures uniquement.

    Args:
        selection_end: DERNIERE date utilisable. Doit etre anterieure au debut
            de la fenetre de backtest, sans quoi la selection regarde le futur.
    """
    imposed = load_universe(imposed_symbols, selection_start, selection_end)
    pool = load_universe(list(candidates), selection_start, selection_end)
    scores = score_candidates(imposed, pool)

    chosen = [score.symbol for score in scores if score.eligible and score.score > 0][:count]
    log.info(
        "universe_selected",
        imposed=imposed_symbols,
        chosen=chosen,
        selection_window=f"{selection_start} -> {selection_end}",
    )
    return chosen, scores

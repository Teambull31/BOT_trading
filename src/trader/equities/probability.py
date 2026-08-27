"""Probabilite de reussite conditionnee au faisceau de signaux — estimee causalement.

Le piege central de cet exercice : construire la table "score de signaux ->
probabilite de hausse" sur tout l'historique, puis s'en servir pour trader ce
meme historique. La table connait alors le resultat des trades qu'elle
recommande. C'est la meme tricherie que lire le cours de demain, en moins
visible — et elle produit des courbes spectaculaires et fausses.

Deux objets distincts sont donc produits ici, et il ne faut jamais confondre
leurs usages :

- `calibration_table` : DESCRIPTIVE. Mesure sur toute la periode ce que valait
  chaque score. Sert a comprendre, jamais a decider.
- `causal_probabilities` : DECISIONNELLE. A la seance t, n'utilise que les
  trades dont le resultat etait DEJA connu en t, c'est-a-dire ouverts au plus
  tard en t - horizon. C'est la seule version utilisable pour trader.

L'ecart entre les deux mesure exactement ce que l'on croirait gagner en
trichant. Le script d'etude l'affiche systematiquement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trader.equities.signals import compute_signals, forward_outcome

LAPLACE_ALPHA: float = 1.0
"""Lissage de Laplace : un score vu 3 fois avec 3 hausses ne vaut pas 100 %."""

MIN_SAMPLES: int = 30
"""En dessous, on ne croit pas l'estimation et on retombe sur le taux de base."""


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    """Une ligne de la table descriptive score -> resultat observe."""

    score: int
    samples: int
    win_rate: float
    base_rate: float
    mean_return_pct: float
    median_return_pct: float

    @property
    def edge_pct(self) -> float:
        """Ecart du taux de reussite par rapport au TAUX DE BASE.

        Comparer a 50 % serait la faute classique de cet exercice. Sur actions,
        une fenetre de dix seances est haussiere environ 59 % du temps sans
        rien faire : un signal a 58 % de reussite parait alors "+8 points au-
        dessus de pile ou face" alors qu'il fait MOINS BIEN que ne rien
        regarder. Le seul point de comparaison honnete est le taux observe sur
        l'ensemble des seances de la meme periode.
        """
        return (self.win_rate - self.base_rate) * 100.0


def calibration_table(
    frames: dict[str, pd.DataFrame], horizon: int = 10, min_samples: int = MIN_SAMPLES
) -> list[CalibrationRow]:
    """Table DESCRIPTIVE : ce que chaque score a valu, sur toute la periode.

    Ne pas utiliser pour decider un trade : elle a vu tout l'historique.
    """
    panel = build_panel(frames, horizon)
    if panel.empty:
        return []
    base_rate = float((panel["forward"] > 0).mean())
    rows: list[CalibrationRow] = []
    for score, group in panel.groupby("score"):
        if len(group) < min_samples:
            continue
        rows.append(
            CalibrationRow(
                score=int(score),
                samples=len(group),
                win_rate=float((group["forward"] > 0).mean()),
                base_rate=base_rate,
                mean_return_pct=float(group["forward"].mean() * 100.0),
                median_return_pct=float(group["forward"].median() * 100.0),
            )
        )
    return sorted(rows, key=lambda row: row.score)


def stability_by_year(
    frames: dict[str, pd.DataFrame], horizon: int = 10, threshold: int = 5
) -> pd.DataFrame:
    """L'ecart au taux de base tient-il annee apres annee ?

    Le test qui separe un edge d'un artefact. Un signal reellement informatif
    garde son avance quand le regime change ; un signal qui ne fait que suivre
    la hausse generale s'effondre l'annee ou le marche baisse. La colonne
    `base` permet de reperer ces annees : sous 50 %, le marche recule.
    """
    panel = build_panel(frames, horizon)
    if panel.empty:
        return pd.DataFrame()
    panel["annee"] = pd.DatetimeIndex(panel["date"]).year

    rows = []
    for year, group in panel.groupby("annee"):
        base = float((group["forward"] > 0).mean())
        row = {"annee": int(year), "base_rate": base, "observations": len(group)}
        for label, subset in (
            ("baissier", group[group["score"] <= -threshold]),
            ("haussier", group[group["score"] >= threshold]),
        ):
            if len(subset) < 20:
                row[f"{label}_ecart_pts"] = np.nan
                row[f"{label}_n"] = len(subset)
                continue
            row[f"{label}_ecart_pts"] = (float((subset["forward"] > 0).mean()) - base) * 100.0
            row[f"{label}_n"] = len(subset)
        rows.append(row)
    return pd.DataFrame(rows).set_index("annee")


def build_panel(frames: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    """Empile (date, titre, score, resultat futur) pour tout l'univers.

    Mutualiser les titres est indispensable : un score extreme n'apparait que
    quelques dizaines de fois par titre, trop peu pour estimer quoi que ce soit.
    La mutualisation se fait entre TITRES, jamais entre DATES — le decalage
    temporel reste impose en aval.
    """
    parts: list[pd.DataFrame] = []
    for symbol, frame in frames.items():
        signals = compute_signals(frame)
        outcome = forward_outcome(frame, horizon)
        part = pd.DataFrame(
            {
                "date": frame.index,
                "symbol": symbol,
                "score": signals["score"].to_numpy(),
                "forward": outcome.to_numpy(),
            }
        )
        parts.append(part.dropna(subset=["score", "forward"]))
    if not parts:
        return pd.DataFrame(columns=["date", "symbol", "score", "forward"])
    panel = pd.concat(parts, ignore_index=True)
    panel["score"] = panel["score"].astype(int)
    return panel.sort_values(["date", "symbol"], ignore_index=True)


def causal_probabilities(
    frames: dict[str, pd.DataFrame],
    horizon: int = 10,
    min_samples: int = MIN_SAMPLES,
) -> dict[str, pd.DataFrame]:
    """Probabilite de hausse estimee a chaque seance, sans lire le futur.

    Mecanique du decalage, qui est tout l'interet du module : le resultat d'un
    signal emis en s n'est connu qu'en s + horizon. Donc a la seance t, seuls
    les signaux emis jusqu'a t - horizon ont un resultat observable. La table
    utilisee en t est construite sur ces seuls echantillons.

    Renvoie, par titre, un tableau indexe par date avec le score du jour, la
    probabilite estimee, et le nombre d'echantillons qui la soutiennent.
    """
    panel = build_panel(frames, horizon)
    if panel.empty:
        return {symbol: pd.DataFrame() for symbol in frames}

    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    date_rank = pd.Series(range(len(dates)), index=dates)
    panel["rank"] = panel["date"].map(date_rank).astype(int)
    panel["win"] = (panel["forward"] > 0).astype(float)

    buckets = sorted(panel["score"].unique())
    # Comptages cumules par score, indexes par rang de date : counts[b][r] =
    # nombre d'echantillons de score b emis jusqu'au rang r inclus.
    counts = _cumulative_by_rank(panel, buckets, len(dates), "win")

    results: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        signals = compute_signals(frame)
        scores = signals["score"]
        ranks = pd.Series(frame.index.map(date_rank), index=frame.index)

        probability = np.full(len(frame), np.nan)
        support = np.zeros(len(frame), dtype=int)
        for position, (score, rank) in enumerate(zip(scores, ranks, strict=True)):
            if pd.isna(score) or pd.isna(rank):
                continue
            # Frontiere causale : seuls les signaux emis au moins `horizon`
            # seances plus tot ont un resultat connu aujourd'hui.
            usable = int(rank) - horizon
            if usable < 0:
                continue
            total, wins = counts["all"][usable]
            base_rate = (wins + LAPLACE_ALPHA) / (total + 2.0 * LAPLACE_ALPHA) if total else 0.5
            bucket = counts["by_score"].get(int(score))
            if bucket is None:
                probability[position] = base_rate
                continue
            n, w = bucket[usable]
            support[position] = int(n)
            if n < min_samples:
                probability[position] = base_rate
            else:
                probability[position] = (w + LAPLACE_ALPHA) / (n + 2.0 * LAPLACE_ALPHA)

        results[symbol] = pd.DataFrame(
            {
                "score": scores.to_numpy(),
                "probabilite": probability,
                "echantillons": support,
            },
            index=frame.index,
        )
    return results


def _cumulative_by_rank(
    panel: pd.DataFrame, buckets: list[int], n_dates: int, win_column: str
) -> dict:
    """Comptages cumules (total, victoires) par score et par rang de date."""
    by_score: dict[int, np.ndarray] = {}
    for bucket in buckets:
        subset = panel[panel["score"] == bucket]
        totals = np.zeros(n_dates)
        wins = np.zeros(n_dates)
        np.add.at(totals, subset["rank"].to_numpy(), 1.0)
        np.add.at(wins, subset["rank"].to_numpy(), subset[win_column].to_numpy())
        by_score[int(bucket)] = np.column_stack([totals.cumsum(), wins.cumsum()])

    totals = np.zeros(n_dates)
    wins = np.zeros(n_dates)
    np.add.at(totals, panel["rank"].to_numpy(), 1.0)
    np.add.at(wins, panel["rank"].to_numpy(), panel[win_column].to_numpy())
    return {"by_score": by_score, "all": np.column_stack([totals.cumsum(), wins.cumsum()])}


def assert_probabilities_causal(
    frames: dict[str, pd.DataFrame], horizon: int = 10, cut_ratio: float = 0.7
) -> list[str]:
    """Verifie qu'ajouter des donnees futures ne change pas les estimations passees.

    Meme protocole que pour les indicateurs : on recalcule sur un prefixe des
    donnees ; les probabilites du prefixe doivent etre IDENTIQUES a celles
    obtenues sur la serie complete. Une divergence prouve que la table lit des
    resultats qui n'etaient pas encore connus.
    """
    prefixed = {
        symbol: frame.iloc[: int(len(frame) * cut_ratio)] for symbol, frame in frames.items()
    }
    full = causal_probabilities(frames, horizon)
    partial = causal_probabilities(prefixed, horizon)

    offenders: list[str] = []
    for symbol, left in partial.items():
        right = full[symbol].loc[left.index]
        both = left["probabilite"].notna() & right["probabilite"].notna()
        if not both.any():
            continue
        gap = (left.loc[both, "probabilite"] - right.loc[both, "probabilite"]).abs()
        if bool((gap > 1e-9).any()):
            offenders.append(symbol)
    return offenders

"""Etude : probabilite de reussite selon le faisceau de signaux, puis test a levier.

Deux questions, traitees separement parce qu'elles n'ont pas la meme reponse :

1. Un faisceau de signaux concordants annonce-t-il vraiment une hausse ?
   -> table de calibration, version descriptive PUIS version causale.
2. Que donne ce signal joue a levier 30, en long comme en short ?
   -> balayage de levier avec appels de marge, gaps et cout de portage reels.

Utilisation :
    python scripts/signal_probability.py
    python scripts/signal_probability.py --horizon 20 --leverages 1,5,30
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trader.equities.data import load_universe  # noqa: E402
from trader.equities.leverage import (  # noqa: E402
    LeverageParams,
    direction_from_probability,
    simulate_leveraged,
)
from trader.equities.probability import (  # noqa: E402
    assert_probabilities_causal,
    calibration_table,
    causal_probabilities,
    stability_by_year,
)
from trader.equities.signals import SIGNAL_NAMES  # noqa: E402

UNIVERSE = ["MU", "ASML", "WMT", "GLD", "NVDA", "JPM", "XOM", "KO"]
IN_SAMPLE_END = date(2025, 12, 31)
HISTORY_START = date(2019, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=10, help="horizon d'evaluation en seances")
    parser.add_argument("--leverages", default="1,2,5,10,20,30")
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def section(title: str) -> None:
    print(f"\n{title}\n{'=' * 78}")


def main() -> int:
    args = parse_args()
    leverages = [float(value) for value in args.leverages.split(",")]

    frames = load_universe(UNIVERSE, HISTORY_START, IN_SAMPLE_END, refresh=args.refresh)
    if not frames:
        print("aucune donnee disponible", file=sys.stderr)
        return 1

    section("0. PERIMETRE")
    print(f"Univers   : {', '.join(sorted(frames))}")
    print(f"Periode   : {HISTORY_START} -> {IN_SAMPLE_END} (in-sample, hors fenetre 2026)")
    print(f"Horizon   : {args.horizon} seances")
    print(f"Signaux   : {', '.join(SIGNAL_NAMES)} (score de -7 a +7)")

    section("1. CONTROLE DE CAUSALITE")
    offenders = assert_probabilities_causal(frames, args.horizon)
    print(
        f"Titres dont les probabilites changent quand on ajoute du futur : {offenders or 'AUCUN'}"
    )
    if offenders:
        print("ECHEC : la table lit des resultats non encore connus.", file=sys.stderr)
        return 1

    section("2. TABLE DE CALIBRATION (descriptive — a NE PAS utiliser pour trader)")
    rows = calibration_table(frames, args.horizon)
    if not rows:
        print("echantillon insuffisant", file=sys.stderr)
        return 1
    base_rate = rows[0].base_rate
    print(
        f"Taux de base : {base_rate:.1%} des fenetres de {args.horizon} seances montent"
        " sans rien faire."
    )
    print("C'est LE point de comparaison. Un score sous cette barre detruit de l'information.\n")
    print(
        f"{'score':>6} | {'cas':>6} | {'hausse':>7} | {'vs base':>9} | {'moy %':>7} | {'med %':>7}"
    )
    print("-" * 62)
    for row in rows:
        verdict = "" if row.edge_pct > 0 else "  <- sous le taux de base"
        print(
            f"{row.score:>+6} | {row.samples:>6} | {row.win_rate:>6.1%} |"
            f" {row.edge_pct:>+8.1f} | {row.mean_return_pct:>+7.2f} |"
            f" {row.median_return_pct:>+7.2f}{verdict}"
        )

    bullish = [row for row in rows if row.score >= 5]
    if bullish:
        best = max(bullish, key=lambda row: row.edge_pct)
        print(
            f"\nMeilleur score HAUSSIER (>= +5) : {best.score:+d} -> {best.win_rate:.1%}"
            f" soit {best.edge_pct:+.1f} points par rapport au taux de base."
        )

    section("2 bis. L'ECART TIENT-IL CHAQUE ANNEE ? (le test qui separe edge et artefact)")
    stability = stability_by_year(frames, args.horizon)
    if not stability.empty:
        print(f"{'annee':>6} | {'base':>6} | {'signaux baissiers':>19} | {'signaux haussiers':>19}")
        print("-" * 60)
        for year, row in stability.iterrows():
            cells = {}
            for prefix in ("baissier", "haussier"):
                gap = row[f"{prefix}_ecart_pts"]
                cells[prefix] = (
                    f"{'trop peu de cas':>19}"
                    if pd.isna(gap)
                    else f"{gap:>+9.1f} pts n={int(row[f'{prefix}_n']):<5}"
                )
            cell = cells.__getitem__
            marker = "  <- marche BAISSIER" if row["base_rate"] < 0.5 else ""
            print(
                f"{int(year):>6} | {row['base_rate']:>5.1%} | {cell('baissier')} |"
                f" {cell('haussier')}{marker}"
            )
        print(
            "\nLire les annees ou le taux de base passe sous 50 % : ce sont les seules"
            "\nqui testent vraiment un signal. Un edge qui n'existe que les annees de"
            "\nhausse n'est pas un edge, c'est de l'exposition au marche deguisee."
        )

    section("3. LA MEME TABLE, ESTIMEE CAUSALEMENT (utilisable, elle)")
    causal = causal_probabilities(frames, args.horizon)
    merged = []
    for symbol, table in causal.items():
        part = table.dropna(subset=["probabilite"]).copy()
        part["symbol"] = symbol
        merged.append(part)
    panel = pd.concat(merged) if merged else pd.DataFrame()

    if not panel.empty:
        print(f"{'score':>6} | {'cas':>6} | {'proba estimee':>14} | {'vs base':>9}")
        print("-" * 46)
        for score, group in panel.groupby("score"):
            if len(group) < 30:
                continue
            mean_probability = float(group["probabilite"].mean())
            gap = (mean_probability - base_rate) * 100.0
            print(f"{int(score):>+6} | {len(group):>6} | {mean_probability:>13.1%} | {gap:>+8.1f}")
        print(
            "\nCes estimations sont plus plates que la table descriptive : c'est normal"
            "\net c'est le signe que le decalage causal fonctionne. La table descriptive"
            "\nconnait le resultat des cas qu'elle classe ; celle-ci ne l'apprend qu'apres."
        )

    section("4. TEST A LEVIER — long et short pilotes par la probabilite causale")
    print(
        "Hypotheses : financement 5 %/an sur la part empruntee, loyer de titre 1 %/an\n"
        "sur les shorts, commission 0.02 % et slippage 0.05 % du NOTIONNEL,\n"
        "liquidation a la moitie de la marge initiale, gaps d'ouverture honores.\n"
    )

    header = (
        f"{'levier':>7} | {'liquide a':>10} | {'titres ruines':>14} | "
        f"{'median %':>9} | {'pire %':>9} | {'portage':>8}"
    )
    print(header)
    print("-" * len(header))

    for leverage in leverages:
        params = LeverageParams(leverage=leverage)
        returns: list[float] = []
        ruined = 0
        financing: list[float] = []
        for symbol, frame in frames.items():
            probability = causal[symbol]["probabilite"]
            direction = direction_from_probability(probability)
            result = simulate_leveraged(frame, direction, params, args.capital)
            returns.append(result.total_return_pct)
            financing.append(result.financing_paid)
            ruined += result.ruined
        print(
            f"{leverage:>6.0f}x | {params.ruin_move_pct:>9.2f}% | {ruined:>4}/{len(frames):<9} |"
            f" {statistics.median(returns):>+8.1f} | {min(returns):>+8.1f} |"
            f" {statistics.fmean(financing):>7.0f}"
        )

    print(
        "\n'liquide a' = mouvement adverse suffisant pour effacer le compte."
        "\n'portage' = interets moyens payes par titre, en euros, pour 1000 EUR de depart."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

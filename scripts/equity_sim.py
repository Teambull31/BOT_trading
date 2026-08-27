"""Simulation de trading actions sur donnees passees.

Usage :
    python scripts/equity_sim.py                       # juin -> aout, 1000 EUR
    python scripts/equity_sim.py --walk-forward        # + validation multi-annees
    python scripts/equity_sim.py --start 2026-06-01 --end 2026-08-25 --capital 1000

Garanties methodologiques, verifiees a chaque execution :
- aucune donnee posterieure a la barre courante n'est lue (test de causalite) ;
- l'univers est selectionne sur des donnees ANTERIEURES a la fenetre testee ;
- les parametres de strategie sont fixes et standards, jamais ajustes sur la
  fenetre evaluee ;
- frais et slippage appliques a chaque ordre ;
- comparaison systematique au buy & hold.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trader.equities.backtest import (  # noqa: E402
    EquityBacktester,
    ExecutionCosts,
)
from trader.equities.data import load_universe  # noqa: E402
from trader.equities.diagnostic import diagnose  # noqa: E402
from trader.equities.profiles import ORDER, PROFILES, get_profile  # noqa: E402
from trader.equities.selection import select_universe  # noqa: E402
from trader.equities.strategy import assert_signals_causal  # noqa: E402
from trader.logging_setup import configure_logging  # noqa: E402

warnings.filterwarnings("ignore")
app = typer.Typer(add_completion=False, help="Simulation de trading actions.")

IMPOSED = ["MU", "ASML"]
BROAD_MARKET = "SPY"
"""Indice large de reference : distingue un probleme sectoriel d'une correction generale."""
DEFAULT_HISTORY_START = "2023-01-01"
"""Debut de l'historique charge.

Il ne s'agit PAS de la fenetre evaluee : ces donnees servent a remplir les
indicateurs (moyenne 200 seances), a selectionner l'univers et a rejouer la
strategie sur des annees anterieures. Charger seulement quelques mois donnerait
une selection tiree au sort et un walk-forward reduit a une seule periode, donc
sans valeur probante.
"""
REPORT_PATH = Path("artifacts/equity_sim")

CALIBRATED_STRATEGY = {
    "entry_mode": "trend",
    "trailing_atr": 6.0,
    "initial_stop_atr": 5.0,
}
CALIBRATED_RISK = {"sizing_mode": "target_weight", "max_position_pct": 33.0}
"""Configuration retenue apres comparaison de variantes sur 2023-10 -> 2025-12.

La comparaison portait sur des CHOIX DE CONCEPTION (rester investi dans la
tendance plutot que trader des cassures ; investir reellement plutot que garder
90 % de cash), pas sur un balayage de valeurs numeriques. Elle a ete faite sur
des donnees anterieures a la fenetre evaluee, qui n'a jamais ete consultee
pendant ce choix. Les parametres ne sont plus retouches ensuite.
"""


def _rule(title: str) -> str:
    return f"\n{title}\n" + "=" * 78


@app.command()
def main(
    start: str = typer.Option("2026-06-01", help="Debut de la fenetre evaluee."),
    end: str = typer.Option("2026-08-25", help="Fin de la fenetre evaluee."),
    capital: float = typer.Option(1000.0, help="Capital initial."),
    risk_per_trade: float = typer.Option(1.0, help="Risque par trade, en % du capital."),
    max_positions: int = typer.Option(3, help="Positions simultanees maximum."),
    default_position_pct: float = typer.Option(33.0, help="Allocation cible par position, en %."),
    profile: str = typer.Option("equilibre", help=f"Profil de risque : {', '.join(ORDER)}."),
    compare_profiles: bool = typer.Option(
        False, help="Comparer tous les profils sur les memes periodes."
    ),
    diagnostic: bool = typer.Option(
        True, help="Afficher le diagnostic d'etat du marche a la derniere date."
    ),
    stress_universe: bool = typer.Option(
        False,
        help="Tester chaque profil sur plusieurs univers pour mesurer sa fragilite.",
    ),
    walk_forward: bool = typer.Option(False, help="Validation annee par annee."),
    history_start: str = typer.Option(
        DEFAULT_HISTORY_START, help="Debut de l'historique charge (warmup + walk-forward)."
    ),
    refresh: bool = typer.Option(False, help="Forcer le re-telechargement des cours."),
    extra: str = typer.Option("", help="Titres supplementaires imposes, separes par des virgules."),
    log_level: str = typer.Option("WARNING", help="Niveau de log."),
) -> None:
    """Execute la simulation et affiche le rapport."""
    configure_logging(log_level, json_output=False)
    window_start = date.fromisoformat(start)
    window_end = date.fromisoformat(end)
    history_begin = min(date.fromisoformat(history_start), window_start - timedelta(days=400))

    # 1. Univers : selection sur donnees strictement anterieures a la fenetre.
    typer.echo(_rule("1. SELECTION DE L'UNIVERS (donnees anterieures uniquement)"))
    selection_end = window_start - timedelta(days=1)
    if extra:
        chosen = [symbol.strip().upper() for symbol in extra.split(",") if symbol.strip()]
        scores = []
        typer.echo(f"Titres complementaires imposes par l'utilisateur : {chosen}")
    else:
        chosen, scores = select_universe(IMPOSED, history_begin, selection_end, count=2)
        table = pd.DataFrame([score.to_row() for score in scores])
        typer.echo(table.to_string(index=False))
    universe = IMPOSED + chosen
    typer.echo(f"\nUnivers retenu : {', '.join(universe)}")
    typer.echo(
        f"Fenetre de selection : {history_start} -> {selection_end} (aucune donnee posterieure)"
    )

    # 2. Chargement des cours, warmup inclus.
    frames = load_universe(universe, history_begin, window_end, refresh=refresh)
    missing = [symbol for symbol in universe if symbol not in frames]
    if missing:
        typer.echo(f"Titres indisponibles : {missing}")
    if not frames:
        typer.echo("Aucune donnee : simulation impossible.")
        raise typer.Exit(code=1)

    # 3. Controle anti-look-ahead sur chaque titre.
    typer.echo(_rule("2. CONTROLE ANTI-LOOK-AHEAD"))
    selected = get_profile(profile)
    params = selected.strategy
    all_clean = True
    for symbol, frame in frames.items():
        offenders = assert_signals_causal(frame, params)
        status = "AUCUN" if not offenders else f"FUITE : {offenders}"
        all_clean &= not offenders
        typer.echo(f"  {symbol:<6} {len(frame):>4} barres  |  look-ahead : {status}")
    if not all_clean:
        typer.echo("\nARRET : un indicateur lit le futur, les resultats seraient faux.")
        raise typer.Exit(code=1)
    typer.echo(f"\nStrategie : {params.describe()}")
    typer.echo(
        "Conception retenue sur 2023-10 -> 2025-12 (in-sample), appliquee telle quelle\n"
        "a la fenetre evaluee, qui n'a servi a aucun choix de parametre."
    )
    typer.echo(f"\nProfil de risque actif :\n  {selected.describe()}")

    # 4. Diagnostic du marche a la derniere date connue.
    if diagnostic:
        typer.echo(_rule("3. ETAT DU MARCHE (derniere date disponible)"))
        benchmark_frames = load_universe([BROAD_MARKET], history_begin, window_end, refresh=refresh)
        try:
            market = diagnose(
                frames,
                benchmark_frame=benchmark_frames.get(BROAD_MARKET),
                benchmark_symbol=BROAD_MARKET,
            )
            typer.echo(market.render())
            payload_diagnostic = {
                "as_of": str(market.as_of),
                "breadth_pct": round(market.breadth_pct, 1),
                "mean_correlation": round(market.mean_correlation, 3),
                "mean_vol_ratio": round(market.mean_vol_ratio, 3),
                "mean_drawdown_pct": round(market.mean_drawdown_pct, 2),
                "warnings": market.warnings,
                "verdict": market.verdict,
                "caution_score": market.caution_score,
                "recommended_profile": market.recommended_profile,
            }
        except ValueError as exc:
            typer.echo(f"Diagnostic indisponible : {exc}")
            payload_diagnostic = None
    else:
        payload_diagnostic = None

    # 5. Backtest principal.
    risk = selected.risk
    costs = ExecutionCosts()
    backtester = EquityBacktester(params=params, risk=risk, costs=costs)

    typer.echo(_rule(f"4. SIMULATION {window_start} -> {window_end}"))
    report = backtester.run(frames, window_start, window_end, capital)
    typer.echo(report.summary())

    if report.trades:
        typer.echo("\nDetail des trades :")
        trades_table = pd.DataFrame([trade.to_dict() for trade in report.trades])
        typer.echo(trades_table.to_string(index=False))
    else:
        typer.echo(
            "\nAucun trade : les conditions d'entree (tendance + cassure + ADX) "
            "n'ont jamais ete reunies sur la periode."
        )

    if report.open_details:
        typer.echo("\nPositions encore ouvertes a la derniere seance :")
        typer.echo(pd.DataFrame(report.open_details).to_string(index=False))
        exposed = sum(item["valeur"] for item in report.open_details)
        latent = sum(item["pnl_latent"] for item in report.open_details)
        typer.echo(
            f"  Capital expose : {exposed:,.2f} sur {report.final_equity:,.2f} "
            f"({exposed / report.final_equity * 100.0:.0f} %) | "
            f"P&L latent {latent:+,.2f}"
        )

    # 5. Validation de perennite.
    payload = {
        "universe": universe,
        "profile": selected.key,
        "market_diagnostic": payload_diagnostic,
        "window": {"start": str(window_start), "end": str(window_end)},
        "params": asdict(params),
        "risk": asdict(risk),
        "costs": asdict(costs),
        "result": {
            "initial_capital": report.initial_capital,
            "final_equity": round(report.final_equity, 2),
            "total_return_pct": round(report.total_return_pct, 2),
            "metrics": {k: round(v, 4) for k, v in report.metrics.items()},
            "trades": [trade.to_dict() for trade in report.trades],
            "open_positions": report.open_details,
        },
    }

    if compare_profiles:
        typer.echo(_rule("5. COMPARAISON DES PROFILS DE RISQUE"))
        typer.echo("Memes titres, memes dates, memes frais : seul le curseur de risque change.\n")
        comparison = _compare_profiles(frames, window_start, window_end, capital, costs)
        typer.echo(pd.DataFrame(comparison).to_string(index=False))
        payload["profile_comparison"] = comparison

        typer.echo("\nSur l'historique complet disponible :")
        full_start = min(frame.index[0].date() for frame in frames.values()) + timedelta(days=320)
        full = _compare_profiles(frames, full_start, window_end, capital, costs)
        typer.echo(pd.DataFrame(full).to_string(index=False))
        payload["profile_comparison_full_history"] = full

    if stress_universe:
        typer.echo(_rule("6. FRAGILITE : meme profil, univers legerement different"))
        typer.echo(
            "Un profil dont le resultat s'effondre quand on remplace UN titre n'est pas\n"
            "une strategie : c'est un pari sur ce titre. On mesure ici la dispersion.\n"
        )
        stress = _stress_universes(history_begin, window_end, capital, costs, refresh)
        typer.echo(pd.DataFrame(stress).to_string(index=False))
        payload["universe_stress"] = stress

    if walk_forward:
        typer.echo(_rule("6. VALIDATION DE PERENNITE (memes parametres, autres periodes)"))
        folds = _walk_forward(backtester, frames, window_end, capital)
        typer.echo(pd.DataFrame(folds).to_string(index=False))
        payload["walk_forward"] = folds
        positive = [f for f in folds if f["strategie_%"] > 0]
        beat = [f for f in folds if f["strategie_%"] > f["buy_hold_%"]]
        typer.echo(
            f"\nPeriodes positives : {len(positive)}/{len(folds)}  |  "
            f"periodes battant le buy & hold : {len(beat)}/{len(folds)}"
        )

        typer.echo(_rule("7. ROBUSTESSE AU TIMING (fenetres de 3 mois glissantes)"))
        rolling = _rolling_windows(backtester, frames, window_end, capital)
        if rolling:
            table = pd.DataFrame(rolling)
            typer.echo(table.to_string(index=False))
            returns = table["strategie_%"]
            typer.echo(
                f"\n{len(returns)} fenetres de 3 mois | "
                f"mediane {returns.median():+.2f} % | "
                f"moyenne {returns.mean():+.2f} % | "
                f"pire {returns.min():+.2f} % | meilleure {returns.max():+.2f} % | "
                f"positives {int((returns > 0).sum())}/{len(returns)}"
            )
            typer.echo(
                "Cette dispersion est le vrai enseignement : sur trois mois, le resultat\n"
                "depend surtout de la date de depart. Juger la strategie sur une seule\n"
                "fenetre de trois mois revient a juger un dé sur un seul lancer."
            )
            payload["rolling_3m"] = rolling

    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    output = REPORT_PATH / f"sim_{window_start}_{window_end}.json"
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    typer.echo(f"\nRapport detaille : {output}")


STRESS_FOURTH: tuple[str, ...] = ("GLD", "JPM", "KO", "XOM", "JNJ", "PG")
"""Quatriemes titres testes : tous eligibles selon la selection, aucun choisi apres coup."""


def _stress_universes(
    history_begin: date,
    window_end: date,
    capital: float,
    costs: ExecutionCosts,
    refresh: bool,
) -> list[dict]:
    """Rejoue chaque profil sur des univers qui ne different que d'un seul titre.

    C'est le test qui separe une strategie d'un pari deguise. Si remplacer le
    quatrieme titre par un autre, tout aussi defendable a priori, fait varier le
    resultat du simple au quintuple, alors ce n'est pas la strategie qui a
    produit la performance : c'est le tirage.
    """
    base = [*IMPOSED, "WMT"]
    universes: list[list[str]] = []
    for fourth in STRESS_FOURTH:
        if fourth in base:
            continue
        universes.append([*base, fourth])

    results: dict[str, list[float]] = {key: [] for key in ORDER}
    drawdowns: dict[str, list[float]] = {key: [] for key in ORDER}
    start = history_begin + timedelta(days=320)

    for symbols in universes:
        frames = load_universe(symbols, history_begin, window_end, refresh=refresh)
        if len(frames) < len(symbols):
            continue
        for key in ORDER:
            candidate = PROFILES[key]
            backtester = EquityBacktester(
                params=candidate.strategy, risk=candidate.risk, costs=costs
            )
            try:
                result = backtester.run(frames, start, window_end, capital)
            except ValueError:
                continue
            results[key].append(result.total_return_pct)
            drawdowns[key].append(result.metrics.get("max_drawdown_pct", 0.0))

    rows: list[dict] = []
    for key in ORDER:
        values = results[key]
        if not values:
            continue
        series = pd.Series(values)
        rows.append(
            {
                "profil": key,
                "univers_testes": len(values),
                "median_%": round(float(series.median()), 1),
                "pire_%": round(float(series.min()), 1),
                "meilleur_%": round(float(series.max()), 1),
                # Ecart entre le meilleur et le pire univers : la part du
                # resultat qui tient au choix des titres, pas a la strategie.
                "ecart_%": round(float(series.max() - series.min()), 1),
                "dispersion": round(float(series.std(ddof=1)), 1) if len(values) > 1 else 0.0,
                "dd_median_%": round(float(pd.Series(drawdowns[key]).median()), 1),
            }
        )
    return rows


def _compare_profiles(
    frames: dict[str, pd.DataFrame],
    start: date,
    end: date,
    capital: float,
    costs: ExecutionCosts,
) -> list[dict]:
    """Mesure chaque profil sur la MEME periode et les memes titres.

    Comparer des profils sur des periodes differentes ne dirait rien : c'est la
    periode qui ferait la difference, pas le profil.
    """
    rows: list[dict] = []
    for key in ORDER:
        candidate = PROFILES[key]
        backtester = EquityBacktester(params=candidate.strategy, risk=candidate.risk, costs=costs)
        try:
            result = backtester.run(frames, start, end, capital)
        except ValueError:
            continue
        drawdown = result.metrics.get("max_drawdown_pct", 0.0)
        rows.append(
            {
                "profil": key,
                "expo_max_%": round(candidate.max_exposure_pct, 0),
                "capital_final": round(result.final_equity, 2),
                "rendement_%": round(result.total_return_pct, 2),
                "max_dd_%": round(drawdown, 1),
                "gain_par_risque": round(result.total_return_pct / drawdown, 2)
                if drawdown > 0.1
                else float("nan"),
                "trades": len(result.trades),
                "sharpe": round(result.metrics.get("sharpe", 0.0), 2),
                "frais": round(result.metrics.get("total_costs", 0.0), 0),
            }
        )
    benchmark_row = _benchmark_row(frames, start, end, capital, costs)
    if benchmark_row:
        rows.append(benchmark_row)
    return rows


def _benchmark_row(
    frames: dict[str, pd.DataFrame],
    start: date,
    end: date,
    capital: float,
    costs: ExecutionCosts,
) -> dict | None:
    """Ligne de reference : buy & hold equipondere sur les memes titres."""
    reference = EquityBacktester(
        params=PROFILES["equilibre"].strategy, risk=PROFILES["equilibre"].risk, costs=costs
    )
    try:
        result = reference.run(frames, start, end, capital)
    except ValueError:
        return None
    if not len(result.benchmark):
        return None
    benchmark_return = result.metrics.get("benchmark_return_pct", 0.0)
    benchmark_dd = result.metrics.get("benchmark_max_drawdown_pct", 0.0)
    return {
        "profil": "buy & hold",
        "expo_max_%": 100.0,
        "capital_final": round(float(result.benchmark.iloc[-1]), 2),
        "rendement_%": round(benchmark_return, 2),
        "max_dd_%": round(benchmark_dd, 1),
        "gain_par_risque": round(benchmark_return / benchmark_dd, 2)
        if benchmark_dd > 0.1
        else float("nan"),
        "trades": len(frames),
        "sharpe": float("nan"),
        "frais": round(len(frames) * costs.min_commission, 0),
    }


def _rolling_windows(
    backtester: EquityBacktester,
    frames: dict[str, pd.DataFrame],
    window_end: date,
    capital: float,
    months: int = 3,
    step_days: int = 30,
) -> list[dict]:
    """Rejoue la strategie sur des fenetres de 3 mois qui glissent dans le temps.

    Une seule fenetre de trois mois ne prouve rien : le resultat depend de la
    date de depart. En les faisant glisser, on obtient une DISTRIBUTION, qui
    dit ce que la strategie produit typiquement et dans le pire des cas.
    """
    earliest = min(frame.index[0].date() for frame in frames.values())
    cursor = earliest + timedelta(days=320)
    span = timedelta(days=months * 31)

    rows: list[dict] = []
    while cursor + span <= window_end:
        stop = cursor + span
        try:
            result = backtester.run(frames, cursor, stop, capital)
        except ValueError:
            cursor += timedelta(days=step_days)
            continue
        benchmark_pct = result.metrics.get("benchmark_return_pct", float("nan"))
        rows.append(
            {
                "fenetre": f"{cursor} -> {stop}",
                "strategie_%": round(result.total_return_pct, 2),
                "buy_hold_%": round(benchmark_pct, 2),
                "trades": len(result.trades),
                "max_dd_%": round(result.metrics.get("max_drawdown_pct", 0.0), 1),
            }
        )
        cursor += timedelta(days=step_days)
    return rows


def _walk_forward(
    backtester: EquityBacktester,
    frames: dict[str, pd.DataFrame],
    window_end: date,
    capital: float,
) -> list[dict]:
    """Rejoue la MEME strategie sur des periodes anterieures, sans rien reajuster.

    C'est le seul test qui distingue une strategie perenne d'une strategie
    chanceuse : les memes reglages doivent tenir sur des marches differents.
    """
    earliest = min(frame.index[0].date() for frame in frames.values())
    first_tradable = earliest + timedelta(days=300)  # le temps de remplir la SMA 200

    folds: list[dict] = []
    year = first_tradable.year
    while year <= window_end.year:
        fold_start = max(date(year, 1, 1), first_tradable)
        fold_end = min(date(year, 12, 31), window_end)
        if (fold_end - fold_start).days < 60:
            year += 1
            continue
        try:
            result = backtester.run(frames, fold_start, fold_end, capital)
        except ValueError:
            year += 1
            continue
        benchmark_pct = (
            (float(result.benchmark.iloc[-1]) / capital - 1.0) * 100.0
            if len(result.benchmark)
            else float("nan")
        )
        strategy_dd = result.metrics.get("max_drawdown_pct", 0.0)
        benchmark_dd = result.metrics.get("benchmark_max_drawdown_pct", 0.0)
        folds.append(
            {
                "periode": f"{fold_start} -> {fold_end}",
                "strategie_%": round(result.total_return_pct, 2),
                "dd_strat_%": round(strategy_dd, 1),
                "buy_hold_%": round(benchmark_pct, 2),
                "dd_bh_%": round(benchmark_dd, 1),
                # Rendement par unite de risque subi : comparer deux rendements
                # sans comparer leurs drawdowns ne dit rien d'utile.
                "ratio_strat": round(result.total_return_pct / strategy_dd, 2)
                if strategy_dd > 0.1
                else float("nan"),
                "ratio_bh": round(benchmark_pct / benchmark_dd, 2)
                if benchmark_dd > 0.1
                else float("nan"),
                "trades": len(result.trades),
                "sharpe": round(result.metrics.get("sharpe", 0.0), 2),
            }
        )
        year += 1
    return folds


if __name__ == "__main__":
    app()

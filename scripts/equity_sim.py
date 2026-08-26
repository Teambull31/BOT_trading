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
    RiskParams,
)
from trader.equities.data import load_universe  # noqa: E402
from trader.equities.selection import select_universe  # noqa: E402
from trader.equities.strategy import TrendParams, assert_signals_causal  # noqa: E402
from trader.logging_setup import configure_logging  # noqa: E402

warnings.filterwarnings("ignore")
app = typer.Typer(add_completion=False, help="Simulation de trading actions.")

IMPOSED = ["MU", "ASML"]
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
    params = TrendParams(**CALIBRATED_STRATEGY)
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

    # 4. Backtest principal.
    risk = RiskParams(
        sizing_mode=CALIBRATED_RISK["sizing_mode"],
        max_position_pct=default_position_pct,
        risk_per_trade_pct=risk_per_trade,
        max_positions=max_positions,
    )
    costs = ExecutionCosts()
    backtester = EquityBacktester(params=params, risk=risk, costs=costs)

    typer.echo(_rule(f"3. SIMULATION {window_start} -> {window_end}"))
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

    # 5. Validation de perennite.
    payload = {
        "universe": universe,
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
        },
    }

    if walk_forward:
        typer.echo(_rule("4. VALIDATION DE PERENNITE (memes parametres, autres periodes)"))
        folds = _walk_forward(backtester, frames, window_end, capital)
        typer.echo(pd.DataFrame(folds).to_string(index=False))
        payload["walk_forward"] = folds
        positive = [f for f in folds if f["strategie_%"] > 0]
        beat = [f for f in folds if f["strategie_%"] > f["buy_hold_%"]]
        typer.echo(
            f"\nPeriodes positives : {len(positive)}/{len(folds)}  |  "
            f"periodes battant le buy & hold : {len(beat)}/{len(folds)}"
        )

        typer.echo(_rule("5. ROBUSTESSE AU TIMING (fenetres de 3 mois glissantes)"))
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

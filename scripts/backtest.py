"""CLI de backtest : rejoue le systeme sur des donnees historiques.

Usage :
    python scripts/backtest.py --asset ETH/USDT --timeframe 1h --days 180
    python scripts/backtest.py --csv data/eth_1h.csv --walk-forward
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trader.backtest.engine import BacktestEngine, buy_and_hold  # noqa: E402
from trader.backtest.walk_forward import (  # noqa: E402
    WalkForwardResult,
    walk_forward_splits,
)
from trader.config import load_settings  # noqa: E402
from trader.data.features import FeatureBuilder  # noqa: E402
from trader.data.store import DataStore  # noqa: E402
from trader.logging_setup import configure_logging  # noqa: E402
from trader.strategy.ensemble import StrategyEnsemble  # noqa: E402
from trader.strategy.registry import build_default_pool  # noqa: E402

app = typer.Typer(add_completion=False, help="Backtest de l'agent de trading adaptatif.")


def load_frame(csv: Path | None, store: DataStore, asset: str, timeframe: str) -> pd.DataFrame:
    """Charge les donnees depuis un CSV ou depuis la base locale."""
    if csv is not None:
        frame = pd.read_csv(csv)
        stamp_column = "timestamp" if "timestamp" in frame.columns else frame.columns[0]
        frame[stamp_column] = pd.to_datetime(frame[stamp_column], utc=True)
        return frame.set_index(stamp_column).sort_index()
    return store.load_ohlcv(asset, timeframe)


@app.command()
def main(
    asset: str = typer.Option("ETH/USDT", help="Actif a backtester."),
    timeframe: str = typer.Option("1h", help="Timeframe des bougies."),
    csv: Path | None = typer.Option(None, help="Fichier CSV OHLCV (sinon : base locale)."),
    config: Path = typer.Option(Path("config/default.toml"), help="Configuration de base."),
    override: Path | None = typer.Option(None, help="Fichier d'override."),
    capital: float = typer.Option(10000.0, help="Capital initial du backtest."),
    warmup: int = typer.Option(300, help="Bougies reservees au calcul des features."),
    walk_forward: bool = typer.Option(False, help="Backtest walk-forward multi-fenetres."),
    log_level: str = typer.Option("WARNING", help="Niveau de log."),
) -> None:
    """Lance un backtest simple ou walk-forward et affiche les metriques."""
    configure_logging(level=log_level, json_output=False)
    settings = load_settings(config, override)
    store = DataStore(settings.data.db_url)
    frame = load_frame(csv, store, asset, timeframe)

    if frame.empty:
        typer.echo(
            f"Aucune donnee pour {asset} en {timeframe}. "
            "Lancez d'abord une ingestion (scripts/paper_trade.py --backfill-only)."
        )
        raise typer.Exit(code=1)

    features = FeatureBuilder(timeframe=timeframe).build(frame)
    ensemble = StrategyEnsemble(build_default_pool(), settings.ensemble)
    engine = BacktestEngine(settings, ensemble, timeframe=timeframe)

    if not walk_forward:
        result = engine.run(
            frame, asset=asset, warmup=warmup, initial_capital=capital, features=features
        )
        typer.echo(result.summary())
        benchmark = buy_and_hold(frame, capital, warmup)
        if not benchmark.empty:
            typer.echo(f"Buy & hold      : {float(benchmark.iloc[-1]):,.2f}")
        if result.blocked_reasons:
            typer.echo("\nRaisons de non-trade :")
            for reason, count in sorted(result.blocked_reasons.items(), key=lambda item: -item[1])[
                :5
            ]:
                typer.echo(f"  {count:>5} x {reason}")
        raise typer.Exit(code=0)

    splits = walk_forward_splits(frame.index, settings.retraining)
    if not splits:
        typer.echo("Historique trop court pour un walk-forward.")
        raise typer.Exit(code=1)

    aggregate = WalkForwardResult(splits=splits)
    for split in splits:
        train = split.slice_train(frame)
        validation = split.slice_validation(frame)
        if len(train) < warmup + 20 or len(validation) < 20:
            continue
        in_sample = engine.run(train, asset=asset, warmup=warmup, initial_capital=capital)
        out_sample = engine.run(
            pd.concat([train.tail(warmup), validation]),
            asset=asset,
            warmup=warmup,
            initial_capital=capital,
        )
        aggregate.in_sample.append(in_sample.metrics)
        aggregate.out_of_sample.append(out_sample.metrics)
        aggregate.details.append(split.to_dict())
        typer.echo(
            f"Fenetre {split.index:>2} | IS sharpe {in_sample.metrics['sharpe']:+.2f} "
            f"| OOS sharpe {out_sample.metrics['sharpe']:+.2f} "
            f"| OOS trades {out_sample.trade_count}"
        )

    typer.echo("\nSynthese walk-forward :")
    for key, value in aggregate.summary().items():
        typer.echo(f"  {key:<32} {value}")
    verdict = "ROBUSTE" if aggregate.is_robust(settings.retraining.min_oos_ratio) else "SUR-APPRIS"
    typer.echo(f"\nVerdict : {verdict} (seuil OOS/IS = {settings.retraining.min_oos_ratio:.0%})")


if __name__ == "__main__":
    app()

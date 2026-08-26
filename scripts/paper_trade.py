"""Lancement du paper trading (raccourci de `trader paper`).

Usage :
    python scripts/paper_trade.py                 # boucle continue
    python scripts/paper_trade.py --cycles 5      # 5 cycles puis arret
    python scripts/paper_trade.py --backfill-only # charge l'historique et sort
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trader.config import load_settings  # noqa: E402
from trader.logging_setup import configure_logging  # noqa: E402
from trader.main import build_orchestrator  # noqa: E402

app = typer.Typer(add_completion=False, help="Paper trading de l'agent adaptatif.")


@app.command()
def main(
    config: Path = typer.Option(Path("config/default.toml")),
    override: Path = typer.Option(Path("config/paper.toml")),
    cycles: int | None = typer.Option(None, help="Nombre de cycles a executer."),
    backfill_only: bool = typer.Option(False, help="Charger l'historique puis sortir."),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Demarre le paper trading."""
    settings = load_settings(config, override)
    configure_logging(log_level, log_file="logs/paper.log")
    orchestrator = build_orchestrator(settings)

    async def run() -> None:
        results = await orchestrator.ingester.backfill(timeframes=[settings.data.primary_timeframe])
        typer.echo(f"Historique charge : {len(results)} series")
        if backfill_only:
            await orchestrator.shutdown()
            return
        orchestrator.kill_switch.start_http_server()
        await orchestrator.run_forever(max_cycles=cycles)

    asyncio.run(run())


if __name__ == "__main__":
    app()

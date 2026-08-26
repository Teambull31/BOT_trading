"""Point d'entree : assemblage des modules et CLI.

Commandes :
    trader paper                 # paper trading (mode par defaut)
    trader live --i-understand-the-risk
    trader status                # etat du portefeuille et des strategies
    trader kill "raison"         # arret d'urgence
    trader watchdog              # surveillant externe du kill switch

Le mode live exige DEUX conditions : une configuration explicitement en
Mode.LIVE et le drapeau `--i-understand-the-risk`. Un fichier de configuration
oublie ne peut donc pas, a lui seul, envoyer de l'argent reel sur le marche.
"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path

import typer

from trader.adaptation.decay_detector import StrategyDecayDetector
from trader.adaptation.devil_advocate import DevilAdvocate
from trader.adaptation.evaluator import StrategyEvaluator
from trader.adaptation.retrainer import WalkForwardRetrainer
from trader.config import Mode, Settings, load_settings
from trader.data.features import FeatureBuilder
from trader.data.ingester import DataIngester
from trader.data.store import DataStore
from trader.execution.executor import OrderExecutor
from trader.execution.paper import PaperBroker
from trader.logging_setup import configure_logging, get_logger
from trader.orchestrator import TradingOrchestrator
from trader.portfolio import Portfolio
from trader.regime.detector import RegimeDetector
from trader.risk.kill_switch import KillSwitch, KillSwitchWatchdog
from trader.risk.manager import RiskManager
from trader.strategy.ensemble import StrategyEnsemble
from trader.strategy.registry import build_default_pool, log_coverage

log = get_logger(__name__)
cli = typer.Typer(add_completion=False, help="Agent de trading autonome adaptatif.")


def build_orchestrator(settings: Settings) -> TradingOrchestrator:
    """Assemble tous les modules du systeme.

    Point unique de cablage : c'est ici, et nulle part ailleurs, que l'on decide
    quel composant parle a quel autre.
    """
    store = DataStore(settings.data.db_url)
    ingester = DataIngester(settings, store=store)
    portfolio = Portfolio(settings.general.initial_capital)

    def event_sink(level: str, source: str, message: str, payload: dict) -> None:
        """Trace les evenements de protection dans l'audit trail."""
        store.save_event(level, source, message, payload)

    kill_switch = KillSwitch(settings.kill_switch, event_sink=event_sink)
    risk_manager = RiskManager(settings, portfolio, kill_switch=kill_switch, event_sink=event_sink)

    evaluator = StrategyEvaluator(
        store,
        min_trades=settings.decay_detection.min_trades_for_verdict,
        mode=settings.general.mode.value,
    )
    decay_detector = StrategyDecayDetector(
        settings.decay_detection, evaluator, settings.ensemble.shadow_mode_days
    )

    strategies = build_default_pool()
    log_coverage(strategies, settings.ensemble.min_active_strategies)
    ensemble = StrategyEnsemble(
        strategies, settings.ensemble, metrics_provider=evaluator.metrics_provider
    )

    executor = OrderExecutor(
        settings,
        broker=PaperBroker(settings.execution),
        store=store,
    )
    timeframe = settings.data.primary_timeframe
    return TradingOrchestrator(
        settings=settings,
        store=store,
        ingester=ingester,
        ensemble=ensemble,
        portfolio=portfolio,
        risk_manager=risk_manager,
        executor=executor,
        devil_advocate=DevilAdvocate(
            settings.devil_advocate, health_provider=decay_detector.health_provider
        ),
        detector=RegimeDetector(settings.regime, timeframe),
        feature_builder=FeatureBuilder(timeframe=timeframe),
        evaluator=evaluator,
        decay_detector=decay_detector,
        retrainer=WalkForwardRetrainer(settings),
    )


async def _run(settings: Settings, backfill: bool, max_cycles: int | None) -> None:
    """Prepare les donnees puis lance la boucle, avec arret propre sur SIGINT/SIGTERM."""
    orchestrator = build_orchestrator(settings)
    orchestrator.kill_switch.start_http_server()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, orchestrator.stop)
        except (NotImplementedError, RuntimeError):
            # Certaines plateformes ne supportent pas add_signal_handler.
            log.warning("signal_handler_unavailable", signal=sig.name)

    if backfill:
        log.info("backfill_started", assets=orchestrator.assets)
        await orchestrator.ingester.backfill(timeframes=[settings.data.primary_timeframe])

    await orchestrator.run_forever(max_cycles=max_cycles)


@cli.command()
def paper(
    config: Path = typer.Option(Path("config/default.toml"), help="Configuration de base."),
    override: Path = typer.Option(Path("config/paper.toml"), help="Override paper."),
    backfill: bool = typer.Option(True, help="Charger l'historique au demarrage."),
    max_cycles: int | None = typer.Option(None, help="Nombre de cycles (illimite par defaut)."),
) -> None:
    """Lance le systeme en paper trading (aucun argent reel engage)."""
    settings = load_settings(config, override)
    configure_logging(settings.general.log_level, log_file="logs/trader.log")
    if settings.general.mode is not Mode.PAPER:
        typer.echo("La commande `paper` exige une configuration en mode paper.")
        raise typer.Exit(code=1)
    typer.echo(
        f"Paper trading | capital {settings.general.initial_capital:,.2f} "
        f"{settings.general.base_currency} | actifs {settings.universe.assets}"
    )
    asyncio.run(_run(settings, backfill, max_cycles))


@cli.command()
def live(
    config: Path = typer.Option(Path("config/default.toml"), help="Configuration de base."),
    override: Path = typer.Option(Path("config/live.toml"), help="Override live."),
    i_understand_the_risk: bool = typer.Option(
        False,
        "--i-understand-the-risk",
        help="Confirme que le capital engage peut etre perdu integralement.",
    ),
    skip_checklist: bool = typer.Option(
        False, help="Ignorer la checklist go-live (fortement deconseille)."
    ),
    backfill: bool = typer.Option(True, help="Charger l'historique au demarrage."),
) -> None:
    """Lance le systeme en LIVE. Argent reel. Deux confirmations exigees."""
    settings = load_settings(config, override)
    configure_logging(settings.general.log_level, log_file="logs/trader_live.log")

    if settings.general.mode is not Mode.LIVE:
        typer.echo("La configuration fournie n'est pas en mode live.")
        raise typer.Exit(code=1)
    if not i_understand_the_risk:
        typer.echo(
            "Refus de demarrer en live sans --i-understand-the-risk.\n"
            "Le capital engage doit etre de l'argent que vous pouvez perdre "
            "INTEGRALEMENT."
        )
        raise typer.Exit(code=1)

    if not skip_checklist:
        from scripts.go_live_checklist import run_checklist

        report = run_checklist(settings)
        typer.echo(report.render())
        if not report.passed:
            typer.echo("\nChecklist go-live NON validee : demarrage refuse.")
            raise typer.Exit(code=1)

    typer.echo(
        f"MODE LIVE | capital {settings.general.initial_capital:,.2f} "
        f"{settings.general.base_currency} | actifs {settings.universe.assets}"
    )
    asyncio.run(_run(settings, backfill, None))


@cli.command()
def status(
    config: Path = typer.Option(Path("config/default.toml")),
    override: Path | None = typer.Option(None),
) -> None:
    """Affiche l'etat courant : equity, positions, regimes, evenements."""
    settings = load_settings(config, override)
    configure_logging("WARNING", json_output=False)
    store = DataStore(settings.data.db_url)

    equity = store.load_equity()
    trades = store.load_trades()
    regimes = store.load_regimes(limit=5)
    events = store.load_events(limit=5)

    typer.echo(f"Mode              : {settings.general.mode.value}")
    typer.echo(f"Capital initial   : {settings.general.initial_capital:,.2f}")
    if not equity.empty:
        current = float(equity.iloc[-1])
        peak = float(equity.max())
        typer.echo(f"Equity            : {current:,.2f}")
        typer.echo(f"Drawdown courant  : {(peak - current) / peak * 100.0:.2f} % (pic {peak:,.2f})")
    typer.echo(f"Trades clotures   : {len(trades)}")
    if not trades.empty:
        wins = int((trades["pnl"] > 0).sum())
        typer.echo(f"Hit rate          : {wins / len(trades):.1%}")
        typer.echo(f"P&L cumule        : {trades['pnl'].sum():+,.2f}")
    if not regimes.empty:
        last = regimes.iloc[-1]
        typer.echo(f"Dernier regime    : {last['regime']} (conf {last['confidence']:.2f})")

    switch = KillSwitch(settings.kill_switch)
    if switch.is_triggered():
        typer.echo(f"\nKILL SWITCH ARME  : {switch.reason()}")
    if events:
        typer.echo("\nDerniers evenements :")
        for event in events:
            typer.echo(f"  [{event['level']}] {event['source']}: {event['message']}")


@cli.command()
def kill(
    reason: str = typer.Argument("arret manuel", help="Motif de l'arret."),
    config: Path = typer.Option(Path("config/default.toml")),
    override: Path | None = typer.Option(None),
) -> None:
    """Arme le kill switch : le systeme s'arrete et liquide ses positions."""
    settings = load_settings(config, override)
    configure_logging("WARNING", json_output=False)
    KillSwitch(settings.kill_switch).trigger(reason, source="cli")
    typer.echo(f"Kill switch arme : {reason}")
    typer.echo(
        "Le systeme s'arretera au prochain cycle. Pour le desarmer, supprimez "
        f"manuellement {settings.kill_switch.sentinel_path}."
    )


@cli.command()
def watchdog(
    config: Path = typer.Option(Path("config/default.toml")),
    override: Path | None = typer.Option(None),
    once: bool = typer.Option(False, help="Un seul cycle de verification."),
) -> None:
    """Lance le surveillant externe (process separe du trader)."""
    settings = load_settings(config, override)
    configure_logging(settings.general.log_level, json_output=False)
    monitor = KillSwitchWatchdog(
        KillSwitch(settings.kill_switch),
        db_url=settings.data.db_url,
        max_drawdown_pct=settings.risk.max_drawdown_total_pct,
        initial_capital=settings.general.initial_capital,
    )
    if once:
        typer.echo(str(monitor.run_once()))
        return
    monitor.run_forever()


def main() -> None:
    """Point d'entree console."""
    cli()


if __name__ == "__main__":
    main()

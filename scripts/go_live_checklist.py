"""Checklist automatisee avant passage en argent reel.

Aucun critere n'est decoratif : chacun correspond a une facon concrete de perdre
de l'argent qu'on aurait pu eviter. La checklist echoue en bloc — un seul critere
au rouge suffit a refuser le passage en live.

Les criteres qui ne peuvent pas etre verifies par du code (avoir relu ses logs,
accepter de perdre le capital) exigent une attestation ecrite dans un fichier
`artifacts/go_live_manual.json`. C'est volontairement penible : signer une
attestation est un acte plus conscient que passer un drapeau `--force`.

Usage :
    python scripts/go_live_checklist.py
    python scripts/go_live_checklist.py --config config/default.toml
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trader.config import Settings, load_settings  # noqa: E402
from trader.data.store import DataStore  # noqa: E402
from trader.logging_setup import configure_logging  # noqa: E402
from trader.models import StrategyHealth  # noqa: E402
from trader.utils.math_utils import (  # noqa: E402
    max_drawdown,
    profit_factor,
    sharpe_ratio,
)
from trader.utils.time_utils import annualization_factor, utc_now  # noqa: E402

MANUAL_ATTESTATION_PATH = Path("artifacts/go_live_manual.json")
REQUIRED_ATTESTATIONS = (
    "logs_audited",
    "capital_is_expendable",
    "alerts_tested",
)

app = typer.Typer(add_completion=False, help="Checklist de passage en live.")


@dataclass(slots=True)
class Criterion:
    """Un critere de la checklist."""

    name: str
    passed: bool
    detail: str
    blocking: bool = True

    @property
    def symbol(self) -> str:
        """Marqueur visuel du resultat."""
        if self.passed:
            return "[OK]"
        return "[BLOQUANT]" if self.blocking else "[AVERTISSEMENT]"


@dataclass(slots=True)
class ChecklistReport:
    """Resultat complet de la checklist."""

    criteria: list[Criterion] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Vrai si aucun critere bloquant n'a echoue."""
        return all(criterion.passed for criterion in self.criteria if criterion.blocking)

    @property
    def failures(self) -> list[Criterion]:
        """Criteres bloquants en echec."""
        return [c for c in self.criteria if c.blocking and not c.passed]

    def render(self) -> str:
        """Rapport lisible en console."""
        lines = ["", "CHECKLIST GO-LIVE", "=" * 72]
        for criterion in self.criteria:
            lines.append(f"{criterion.symbol:<16} {criterion.name}")
            lines.append(f"{'':<16} {criterion.detail}")
        lines.append("=" * 72)
        if self.passed:
            lines.append("RESULTAT : tous les criteres bloquants sont valides.")
            lines.append(
                "Rappel : valider la checklist ne rend pas le systeme rentable. "
                "Elle garantit seulement qu'il est mesurable et arretable."
            )
        else:
            lines.append(f"RESULTAT : {len(self.failures)} critere(s) bloquant(s) en echec.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Rapport serialisable."""
        return {
            "passed": self.passed,
            "criteria": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "detail": c.detail,
                    "blocking": c.blocking,
                }
                for c in self.criteria
            ],
        }


def load_manual_attestations(path: Path = MANUAL_ATTESTATION_PATH) -> dict[str, bool]:
    """Charge les attestations manuelles signees par l'operateur."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {key: bool(value) for key, value in payload.items()}


def run_checklist(
    settings: Settings,
    store: DataStore | None = None,
    manual_path: Path = MANUAL_ATTESTATION_PATH,
) -> ChecklistReport:
    """Evalue tous les criteres de passage en live."""
    owned_store = store is None
    store = store or DataStore(settings.data.db_url)
    report = ChecklistReport()
    config = settings.paper_trading

    try:
        equity = store.load_equity(mode="paper")
        trades = store.load_trades(mode="paper")
        events = store.load_events(limit=1000)
        timeframe = settings.data.primary_timeframe

        # 1. Anciennete du paper trading.
        if equity.empty:
            report.criteria.append(
                Criterion(
                    "Paper trading actif",
                    False,
                    "aucune courbe d'equity en base : le systeme n'a jamais tourne",
                )
            )
            days = 0.0
        else:
            days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0
            report.criteria.append(
                Criterion(
                    f"Paper trading >= {config.min_days_before_live} jours",
                    days >= config.min_days_before_live,
                    f"{days:.1f} jours de paper trading enregistres",
                )
            )

        # 2. Nombre de trades (significativite statistique).
        trade_count = len(trades)
        enough = trade_count >= config.min_trades_for_live
        verdict = "echantillon suffisant" if enough else "trop petit pour conclure"
        report.criteria.append(
            Criterion(
                f"Trades >= {config.min_trades_for_live}",
                enough,
                f"{trade_count} trades clotures ({verdict})",
            )
        )

        # 3. Sharpe ratio.
        if equity.empty or len(equity) < 10:
            sharpe = 0.0
        else:
            recent = equity[equity.index >= utc_now() - timedelta(days=30)]
            series = recent if len(recent) > 10 else equity
            sharpe = sharpe_ratio(series.pct_change().dropna(), annualization_factor(timeframe))
        report.criteria.append(
            Criterion(
                f"Sharpe paper >= {config.min_sharpe_for_live}",
                sharpe >= config.min_sharpe_for_live,
                f"Sharpe mesure : {sharpe:.2f}",
            )
        )

        # 4. Drawdown maximal.
        drawdown_pct = max_drawdown(equity) * 100.0 if not equity.empty else 0.0
        report.criteria.append(
            Criterion(
                f"Max drawdown < {config.max_drawdown_for_live_pct} %",
                drawdown_pct < config.max_drawdown_for_live_pct and not equity.empty,
                f"drawdown maximal observe : {drawdown_pct:.2f} %",
            )
        )

        # 5. Profit factor.
        pnl = trades["pnl"].to_numpy() if not trades.empty else np.array([])
        factor = profit_factor(pnl) if pnl.size else 0.0
        report.criteria.append(
            Criterion(
                f"Profit factor > {config.min_profit_factor_for_live}",
                factor > config.min_profit_factor_for_live,
                f"profit factor : {factor:.2f}",
            )
        )

        # 6. Circuit breakers testes.
        breaker_events = {
            event["payload"].get("reason", "")
            for event in events
            if event["source"] == "circuit_breaker"
        }
        expected_breakers = {"spread_excessif", "latence_api", "mouvement_de_prix_extreme"}
        tested = expected_breakers & breaker_events
        report.criteria.append(
            Criterion(
                "Circuit breakers testes",
                tested == expected_breakers,
                f"{len(tested)}/{len(expected_breakers)} declenches au moins une fois "
                f"(manquants : {sorted(expected_breakers - tested) or 'aucun'})",
            )
        )

        # 7. Kill switch teste.
        kill_tested = any(
            event["source"].startswith("kill_switch") or "kill" in event["message"].lower()
            for event in events
        )
        report.criteria.append(
            Criterion(
                "Kill switch teste",
                kill_tested,
                "declenchement enregistre en base"
                if kill_tested
                else "aucun declenchement enregistre : testez-le avant le live",
            )
        )

        # 8. Ecart backtest / paper.
        backtest_sharpe = _stored_backtest_sharpe(events)
        if backtest_sharpe is None:
            report.criteria.append(
                Criterion(
                    "Coherence backtest / paper",
                    False,
                    "aucun backtest de reference enregistre pour comparaison",
                )
            )
        else:
            divergence = (
                abs(sharpe - backtest_sharpe) / abs(backtest_sharpe) * 100.0
                if abs(backtest_sharpe) > 1e-9
                else 100.0
            )
            report.criteria.append(
                Criterion(
                    f"Ecart backtest/paper < {config.max_backtest_divergence_pct} %",
                    divergence < config.max_backtest_divergence_pct,
                    f"Sharpe backtest {backtest_sharpe:.2f} vs paper {sharpe:.2f} "
                    f"(ecart {divergence:.1f} %)",
                )
            )

        # 9. Ecart de slippage.
        slippage_divergence = _stored_slippage_divergence(events)
        if slippage_divergence is None:
            report.criteria.append(
                Criterion(
                    "Slippage modelise vs realise",
                    False,
                    "aucune mesure de slippage enregistree",
                    blocking=False,
                )
            )
        else:
            report.criteria.append(
                Criterion(
                    f"Ecart de slippage < {config.max_slippage_divergence_pct} %",
                    slippage_divergence < config.max_slippage_divergence_pct,
                    f"ecart estime/realise : {slippage_divergence:.1f} %",
                )
            )

        # 10. Aucune strategie morte dans l'ensemble actif.
        dead = _dead_strategies(store)
        report.criteria.append(
            Criterion(
                "Aucune strategie DEAD active",
                not dead,
                f"strategies mortes : {sorted(dead)}"
                if dead
                else "toutes les strategies sont vivantes",
            )
        )

        # 11. Plafond de capital.
        capital = settings.general.initial_capital
        report.criteria.append(
            Criterion(
                f"Capital <= {config.max_live_capital:,.0f}",
                capital <= config.max_live_capital,
                f"capital configure : {capital:,.2f}",
            )
        )

        # 12-14. Attestations manuelles.
        attestations = load_manual_attestations(manual_path)
        labels = {
            "logs_audited": "Logs des 30 derniers jours audites manuellement",
            "capital_is_expendable": "Capital engage integralement perdable",
            "alerts_tested": "Alertes testees (message recu)",
        }
        for key in REQUIRED_ATTESTATIONS:
            signed = attestations.get(key, False)
            report.criteria.append(
                Criterion(
                    labels[key],
                    signed,
                    "atteste par l'operateur"
                    if signed
                    else f'non atteste : ajoutez "{key}": true dans {manual_path}',
                )
            )
    finally:
        if owned_store:
            store.close()

    return report


def _stored_backtest_sharpe(events: list[dict]) -> float | None:
    """Recupere le Sharpe du dernier backtest de reference enregistre."""
    for event in events:
        if event["source"] == "backtest" and "sharpe" in event["payload"]:
            return float(event["payload"]["sharpe"])
    return None


def _stored_slippage_divergence(events: list[dict]) -> float | None:
    """Recupere le dernier ecart de slippage enregistre."""
    for event in events:
        if event["source"] == "slippage" and "divergence_pct" in event["payload"]:
            return float(event["payload"]["divergence_pct"])
    return None


def _dead_strategies(store: DataStore) -> set[str]:
    """Strategies dont le dernier etat connu est DEAD."""
    from sqlalchemy import select

    from trader.data.store import StrategyMetricRow

    with store.session() as session:
        rows = (
            session.execute(
                select(StrategyMetricRow).order_by(StrategyMetricRow.timestamp.desc()).limit(200)
            )
            .scalars()
            .all()
        )

    latest: dict[str, str] = {}
    for row in rows:
        latest.setdefault(row.strategy, row.health)
    return {name for name, health in latest.items() if health == StrategyHealth.DEAD.value}


@app.command()
def main(
    config: Path = typer.Option(Path("config/default.toml")),
    override: Path | None = typer.Option(None),
    json_output: bool = typer.Option(False, "--json", help="Sortie JSON."),
) -> None:
    """Execute la checklist et sort en code 1 si elle echoue."""
    configure_logging("WARNING", json_output=False)
    settings = load_settings(config, override)
    report = run_checklist(settings)
    typer.echo(json.dumps(report.to_dict(), indent=2) if json_output else report.render())
    raise typer.Exit(code=0 if report.passed else 1)


if __name__ == "__main__":
    app()

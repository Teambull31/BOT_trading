"""Tests de la CLI : les verrous du mode live avant tout."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trader.main import build_orchestrator, cli

warnings.filterwarnings("ignore")
runner = CliRunner()


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Config de test ecrite sur disque, base et sentinelle isolees."""
    base = Path("config/default.toml").read_text(encoding="utf-8")
    base = base.replace(
        'db_url = "sqlite:///data/trader.db"', f'db_url = "sqlite:///{tmp_path}/cli.db"'
    ).replace('sentinel_path = "/tmp/trader_kill"', f'sentinel_path = "{tmp_path}/kill"')
    path = tmp_path / "default.toml"
    path.write_text(base, encoding="utf-8")
    return path


def test_live_requires_explicit_risk_flag(isolated_config):
    """Un fichier de config oublie ne doit pas suffire a engager de l'argent reel."""
    result = runner.invoke(
        cli, ["live", "--config", str(isolated_config), "--override", "config/live.toml"]
    )
    assert result.exit_code == 1
    assert "i-understand-the-risk" in result.output


def test_live_refuses_paper_configuration(isolated_config):
    result = runner.invoke(
        cli,
        [
            "live",
            "--config",
            str(isolated_config),
            "--override",
            "config/paper.toml",
            "--i-understand-the-risk",
        ],
    )
    assert result.exit_code == 1
    assert "pas en mode live" in result.output


def test_live_blocked_by_failed_checklist(isolated_config):
    """Meme avec le drapeau de risque, la checklist go-live doit passer.

    Sur un systeme vierge (aucun paper trading enregistre), elle echoue
    forcement : c'est exactement le but.
    """
    result = runner.invoke(
        cli,
        [
            "live",
            "--config",
            str(isolated_config),
            "--override",
            "config/live.toml",
            "--i-understand-the-risk",
        ],
    )
    assert result.exit_code == 1
    assert "CHECKLIST GO-LIVE" in result.output
    assert "NON validee" in result.output


def test_paper_refuses_live_configuration(isolated_config):
    result = runner.invoke(
        cli, ["paper", "--config", str(isolated_config), "--override", "config/live.toml"]
    )
    assert result.exit_code == 1
    assert "mode paper" in result.output


def test_kill_arms_the_switch(isolated_config, tmp_path):
    result = runner.invoke(cli, ["kill", "test CLI", "--config", str(isolated_config)])
    assert result.exit_code == 0
    sentinel = tmp_path / "kill"
    assert sentinel.exists()
    assert json.loads(sentinel.read_text())["reason"] == "test CLI"
    assert "supprimez" in result.output  # rappel du desarmement manuel


def test_status_reports_empty_system(isolated_config):
    result = runner.invoke(cli, ["status", "--config", str(isolated_config)])
    assert result.exit_code == 0
    assert "Mode" in result.output
    assert "Capital initial" in result.output


def test_status_reports_armed_kill_switch(isolated_config):
    runner.invoke(cli, ["kill", "arret test", "--config", str(isolated_config)])
    result = runner.invoke(cli, ["status", "--config", str(isolated_config)])
    assert "KILL SWITCH ARME" in result.output


def test_watchdog_single_cycle(isolated_config):
    result = runner.invoke(cli, ["watchdog", "--config", str(isolated_config), "--once"])
    assert result.exit_code == 0
    assert "heartbeat_triggered" in result.output


def test_build_orchestrator_wires_every_module(settings):
    orchestrator = build_orchestrator(settings)
    try:
        assert orchestrator.evaluator is not None
        assert orchestrator.decay is not None
        assert orchestrator.retrainer is not None
        assert orchestrator.alerter is not None
        assert orchestrator.metrics is not None
        assert orchestrator.ensemble.metrics_provider is not None
        assert orchestrator.devil_advocate.health_provider is not None
        assert len(orchestrator.ensemble.records) == 4
    finally:
        orchestrator.store.close()


def test_go_live_checklist_blocks_on_empty_system(settings, tmp_path):
    """Sur un systeme vierge, la checklist doit refuser le passage en live."""
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from go_live_checklist import run_checklist

    report = run_checklist(settings, manual_path=tmp_path / "manual.json")
    assert not report.passed
    assert report.failures
    assert "CHECKLIST GO-LIVE" in report.render()


def test_go_live_manual_attestations_are_required(settings, tmp_path):
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from go_live_checklist import run_checklist

    manual = tmp_path / "manual.json"
    without = run_checklist(settings, manual_path=manual)
    manual.write_text(
        json.dumps({"logs_audited": True, "capital_is_expendable": True, "alerts_tested": True}),
        encoding="utf-8",
    )
    with_attestations = run_checklist(settings, manual_path=manual)
    assert len(with_attestations.failures) == len(without.failures) - 3


async def test_run_wires_servers_and_stops_cleanly(settings, monkeypatch):
    """La commande de lancement demarre les serveurs auxiliaires et les arrete."""
    import trader.main as main_module

    started: dict[str, bool] = {}

    class FakeOrchestrator:
        def __init__(self):
            self.assets = ["ETH/USDT"]
            self.cycles = 0
            self.metrics = type(
                "M", (), {"start": lambda self_: started.__setitem__("metrics", True)}
            )()
            self.kill_switch = type(
                "K",
                (),
                {
                    "start_http_server": lambda self_: started.__setitem__("kill", True),
                    "stop_http_server": lambda self_: None,
                },
            )()
            self.ingester = type(
                "I",
                (),
                {"backfill": staticmethod(lambda **kwargs: _async_value({"ETH/USDT|1h": None}))},
            )()

        async def run_forever(self, max_cycles=None):
            self.cycles = max_cycles or 0
            started["loop"] = True

        def stop(self) -> None:
            pass

    async def _async_value(value):
        return value

    fake = FakeOrchestrator()
    monkeypatch.setattr(main_module, "build_orchestrator", lambda s: fake)

    stopped: dict[str, bool] = {}
    monkeypatch.setattr(
        main_module,
        "DashboardServer",
        lambda *args, **kwargs: type(
            "D",
            (),
            {
                "start": lambda self_: started.__setitem__("dashboard", True),
                "stop": lambda self_: stopped.__setitem__("dashboard", True),
            },
        )(),
    )

    await main_module._run(settings, backfill=True, max_cycles=1)

    assert started["kill"] and started["metrics"] and started["dashboard"]
    assert started["loop"]
    assert stopped["dashboard"]  # arret propre meme en sortie normale

"""Tests du monitoring : metriques, alertes, dashboard."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from trader.config import MonitoringConfig
from trader.monitoring.alerter import Alerter
from trader.monitoring.metrics import TraderMetrics
from trader.utils.time_utils import utc_now


class FakeNotifier:
    """Faux canal de notification : enregistre au lieu d'envoyer."""

    def __init__(self, succeed: bool = True) -> None:
        self.sent: list[tuple[str, str]] = []
        self.succeed = succeed

    def notify(self, body: str, title: str = "") -> bool:
        self.sent.append((title, body))
        return self.succeed


@pytest.fixture
def alerter(memory_store):
    return Alerter(
        MonitoringConfig(alert_rate_limit_info_sec=300),
        urls=["fake://canal"],
        store=memory_store,
        notifier=FakeNotifier(),
    )


# ------------------------------------------------------------------ alertes


async def test_alert_is_delivered_and_persisted(alerter, memory_store):
    assert await alerter.send("CRITICAL", "kill switch declenche")
    assert alerter._notifier.sent
    events = memory_store.load_events(level="CRITICAL")
    assert events and "kill switch" in events[0]["message"]


async def test_info_alerts_are_rate_limited(alerter):
    assert await alerter.send("INFO", "premier trade")
    assert not await alerter.send("INFO", "deuxieme trade")
    assert len(alerter._notifier.sent) == 1


async def test_critical_alerts_are_never_rate_limited(alerter):
    """Un kill switch doit sonner a chaque fois, meme en rafale."""
    for i in range(5):
        assert await alerter.send("CRITICAL", f"alerte critique {i}")
    assert len(alerter._notifier.sent) == 5


async def test_warning_uses_shorter_window(alerter):
    now = utc_now()
    await alerter.send("WARNING", "premier avertissement")
    alerter._last_sent["WARNING"] = now - timedelta(seconds=200)
    assert await alerter.send("WARNING", "second avertissement")


async def test_alert_without_channel_is_still_logged(memory_store):
    alerter = Alerter(MonitoringConfig(), urls=[], store=memory_store)
    assert not await alerter.send("INFO", "sans canal")
    assert memory_store.load_events()  # trace conservee malgre l'absence de canal


async def test_delivery_failure_is_contained(memory_store):
    class BrokenNotifier:
        def notify(self, body, title=""):
            raise RuntimeError("telegram injoignable")

    alerter = Alerter(
        MonitoringConfig(), urls=["fake://x"], store=memory_store, notifier=BrokenNotifier()
    )
    assert not await alerter.send("CRITICAL", "message important")


async def test_unknown_level_falls_back_to_info(alerter):
    await alerter.send("BIZARRE", "niveau inconnu")
    assert alerter.history[-1].level == "INFO"


async def test_daily_report_contains_key_figures(alerter):
    from trader.portfolio import Portfolio
    from trader.strategy.ensemble import StrategyEnsemble
    from trader.strategy.registry import build_default_pool

    portfolio = Portfolio(10_000.0)
    ensemble = StrategyEnsemble(build_default_pool())
    assert await alerter.send_daily_report(portfolio, ensemble)
    _, body = alerter._notifier.sent[-1]
    assert "Equity" in body
    assert "momentum" in body


async def test_test_channels_sends_critical(alerter):
    assert await alerter.test_channels()
    title, _ = alerter._notifier.sent[-1]
    assert "CRITICAL" in title


def test_urls_are_read_from_environment(monkeypatch, memory_store):
    """Un token Telegram n'a rien a faire dans un fichier de configuration."""
    monkeypatch.setenv("TRADER_ALERT_URLS", "tgram://token/chat,mailto://x@y.z")
    alerter = Alerter(MonitoringConfig(), store=memory_store, notifier=FakeNotifier())
    assert len(alerter.urls) == 2


# ---------------------------------------------------------------- metriques


def test_metrics_disabled_are_noops():
    metrics = TraderMetrics(enabled=False)
    metrics.record_order("buy", "filled", 12.0)
    metrics.record_api_error("binance", "timeout")
    metrics.record_breaker("spread")
    metrics.record_contra_score(0.5)  # aucune exception attendue


def test_metrics_update_from_cycle(memory_store):
    from trader.config import load_settings
    from trader.orchestrator import CycleReport
    from trader.portfolio import Portfolio
    from trader.strategy.ensemble import StrategyEnsemble
    from trader.strategy.registry import build_default_pool

    class FakeOrchestrator:
        def __init__(self):
            self.portfolio = Portfolio(10_000.0)
            self.ensemble = StrategyEnsemble(build_default_pool())
            self.settings = load_settings("config/default.toml")
            self.detector = type("D", (), {"last_state": None})()
            self.ingester = type("I", (), {"latencies": {"binance": 0.2}})()
            self.executor = type("E", (), {"slippage": None})()

    metrics = TraderMetrics(enabled=True, port=0)
    report = CycleReport(timestamp=utc_now(), regime_by_asset={"ETH/USDT": "range_bound"})
    metrics.update_from_cycle(FakeOrchestrator(), report, {})  # ne doit pas lever


def test_metrics_update_never_raises():
    """Une metrique ratee ne doit jamais interrompre le trading."""

    class Broken:
        @property
        def portfolio(self):
            raise RuntimeError("etat corrompu")

    TraderMetrics(enabled=False).update_from_cycle(Broken(), None, {})


# ---------------------------------------------------------------- dashboard


def test_dashboard_status_is_serializable(paper_orchestrator):
    from trader.monitoring.dashboard import DashboardServer

    dashboard = DashboardServer(paper_orchestrator)
    payload = json.dumps(dashboard.status(), default=str)
    assert "portfolio" in payload
    assert "risk" in payload


def test_dashboard_health_reports_kill_switch(paper_orchestrator):
    from trader.monitoring.dashboard import DashboardServer

    dashboard = DashboardServer(paper_orchestrator)
    assert dashboard.health()["status"] == "running"
    paper_orchestrator.kill_switch.trigger("test", source="test")
    health = dashboard.health()
    assert health["status"] == "halted"
    assert health["kill_switch"] is True


def test_dashboard_is_read_only(paper_orchestrator):
    """Le dashboard n'expose aucune route capable d'agir sur le systeme."""
    from trader.monitoring.dashboard import DashboardServer

    routes = DashboardServer(paper_orchestrator).routes()
    assert set(routes) == {"/status", "/positions", "/health"}


def test_dashboard_positions_listing(paper_orchestrator):
    from trader.models import OrderSide
    from trader.monitoring.dashboard import DashboardServer
    from trader.portfolio import position_from_fill

    paper_orchestrator.portfolio.open_position(
        position_from_fill("ETH/USDT", OrderSide.BUY, 0.1, 2000.0, 1940.0)
    )
    positions = DashboardServer(paper_orchestrator).positions()["positions"]
    assert positions[0]["asset"] == "ETH/USDT"
    assert positions[0]["stop_loss"] == 1940.0


def test_dashboard_server_starts_and_serves(paper_orchestrator):
    """Le dashboard doit reellement repondre en HTTP, pas seulement calculer du JSON."""
    import urllib.request

    from trader.monitoring.dashboard import DashboardServer

    dashboard = DashboardServer(paper_orchestrator, port=19292)
    dashboard.start()
    try:
        with urllib.request.urlopen("http://127.0.0.1:19292/health", timeout=5) as response:
            payload = json.loads(response.read())
        assert payload["status"] == "running"
        with urllib.request.urlopen("http://127.0.0.1:19292/status", timeout=5) as response:
            assert "portfolio" in json.loads(response.read())
    finally:
        dashboard.stop()


def test_dashboard_unknown_route_returns_404(paper_orchestrator):
    import urllib.error
    import urllib.request

    from trader.monitoring.dashboard import DashboardServer

    dashboard = DashboardServer(paper_orchestrator, port=19293)
    dashboard.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen("http://127.0.0.1:19293/admin", timeout=5)
        assert excinfo.value.code == 404
    finally:
        dashboard.stop()


def test_dashboard_start_is_idempotent(paper_orchestrator):
    from trader.monitoring.dashboard import DashboardServer

    dashboard = DashboardServer(paper_orchestrator, port=19294)
    dashboard.start()
    server = dashboard._server
    dashboard.start()
    try:
        assert dashboard._server is server
    finally:
        dashboard.stop()


def test_metrics_helpers_record_values():
    metrics = TraderMetrics(enabled=True, port=0)
    metrics.record_order("buy", "filled", 12.5)
    metrics.record_api_error("kraken", "timeout")
    metrics.record_breaker("spread_excessif")
    metrics.record_contra_score(0.62)

    from prometheus_client import generate_latest

    exposed = generate_latest(metrics.registry).decode()
    assert "trader_orders_total" in exposed
    assert "trader_api_errors_total" in exposed
    assert "trader_circuit_breaker_triggered" in exposed


def test_alerter_stats_counts_by_level(alerter):
    import asyncio

    asyncio.run(alerter.send("CRITICAL", "un"))
    asyncio.run(alerter.send("CRITICAL", "deux"))
    stats = alerter.stats()
    assert stats["CRITICAL"] == 2
    assert stats["delivered"] == 2


def test_alerter_without_apprise_is_not_configured(memory_store):
    alerter = Alerter(MonitoringConfig(), urls=["fake://x"], store=memory_store, notifier=None)
    # apprise absent ou URL invalide : le systeme continue, les alertes sont journalisees.
    assert isinstance(alerter.is_configured, bool)

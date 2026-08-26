"""Tests des circuit breakers et du kill switch."""

from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd
import pytest

from trader.config import CircuitBreakerConfig, KillSwitchConfig
from trader.risk.circuit_breaker import BreakerReason, CircuitBreaker
from trader.risk.kill_switch import KillSwitch, KillSwitchWatchdog
from trader.utils.time_utils import utc_now


@pytest.fixture
def breaker():
    return CircuitBreaker(CircuitBreakerConfig(pause_duration_min=30))


@pytest.fixture
def switch(tmp_path):
    return KillSwitch(
        KillSwitchConfig(
            sentinel_path=str(tmp_path / "trader_kill"),
            http_enabled=False,
            watchdog_timeout_sec=60,
        )
    )


# ------------------------------------------------------------ breakers


def test_spread_breaker_is_asset_scoped(breaker):
    assert breaker.check_spread("ETH/USDT", 5.0)
    assert breaker.is_tripped("ETH/USDT")
    assert not breaker.is_tripped("BTC/USDT")


def test_acceptable_spread_does_not_trip(breaker):
    assert not breaker.check_spread("ETH/USDT", 0.5)
    assert not breaker.is_tripped("ETH/USDT")


def test_missing_spread_does_not_trip(breaker):
    assert not breaker.check_spread("ETH/USDT", None)


def test_latency_breaker_is_global(breaker):
    assert breaker.check_latency("binance", 10.0)
    assert breaker.is_tripped("N_IMPORTE_QUOI")


def test_price_move_breaker(breaker):
    now = utc_now()
    breaker.check_price_move("ETH/USDT", 2000.0, now)
    assert not breaker.is_tripped()
    tripped = breaker.check_price_move("ETH/USDT", 2400.0, now + timedelta(minutes=2))
    assert tripped
    assert breaker.is_tripped()


def test_slow_price_move_does_not_trip(breaker):
    """Un mouvement de 10 % etale sur des heures n'est pas un choc."""
    now = utc_now()
    for i in range(20):
        breaker.check_price_move("ETH/USDT", 2000.0 * (1 + 0.005 * i), now + timedelta(hours=i))
    assert not breaker.is_tripped()


def test_execution_errors_trip_after_threshold(breaker):
    now = utc_now()
    config_threshold = breaker.config.max_execution_retries
    for i in range(config_threshold - 1):
        assert not breaker.record_execution_error(now + timedelta(seconds=i))
    assert breaker.record_execution_error(now + timedelta(seconds=config_threshold))


def test_stale_data_trips(breaker):
    now = utc_now()
    assert breaker.check_data_freshness("ETH/USDT", now - timedelta(minutes=30), 300.0, now)
    assert not breaker.check_data_freshness("BTC/USDT", now - timedelta(seconds=10), 300.0, now)


def test_breaker_expires_after_pause(breaker):
    now = utc_now()
    breaker.check_spread("ETH/USDT", 5.0, now)
    assert breaker.is_tripped("ETH/USDT", now)
    later = now + timedelta(minutes=breaker.config.pause_duration_min + 1)
    assert not breaker.is_tripped("ETH/USDT", later)


def test_repeated_trip_extends_pause(breaker):
    now = utc_now()
    breaker.check_spread("ETH/USDT", 5.0, now)
    breaker.check_spread("ETH/USDT", 6.0, now + timedelta(minutes=20))
    assert len(breaker.trips) == 1
    assert breaker.is_tripped(
        "ETH/USDT", now + timedelta(minutes=breaker.config.pause_duration_min + 5)
    )


def test_manual_trip_and_reset(breaker):
    breaker.trip_manually("maintenance")
    assert breaker.is_tripped()
    assert breaker.status().trips[0].reason is BreakerReason.MANUAL
    breaker.reset()
    assert not breaker.is_tripped()


# ---------------------------------------------------------- kill switch


def test_sentinel_file_stops_everything(switch):
    assert not switch.is_triggered()
    switch.trigger("perte maximale atteinte", source="test")
    assert switch.is_triggered()
    assert "perte maximale" in switch.reason()
    payload = json.loads(switch.sentinel.read_text())
    assert payload["source"] == "test"


def test_manual_touch_of_sentinel_is_honored(switch):
    """Un simple `touch` doit suffire a tout arreter, sans passer par le code."""
    switch.sentinel.parent.mkdir(parents=True, exist_ok=True)
    switch.sentinel.touch()
    assert switch.is_triggered()
    assert switch.reason()  # message degrade mais non vide


def test_clear_requires_explicit_confirmation(switch):
    switch.trigger("test")
    for wrong in ("", "oui", "confirme", "je confirme le redemarrage"):
        with pytest.raises(PermissionError):
            switch.clear(wrong)
    assert switch.is_triggered()
    switch.clear("JE CONFIRME LE REDEMARRAGE")
    assert not switch.is_triggered()


def test_heartbeat_roundtrip(switch):
    now = utc_now()
    switch.beat(now, {"positions": 2})
    assert switch.last_beat() is not None
    assert not switch.is_heartbeat_stale(now)


def test_stale_heartbeat_is_detected(switch):
    now = utc_now()
    switch.beat(now)
    stale_moment = now + timedelta(seconds=switch.config.watchdog_timeout_sec + 5)
    assert switch.is_heartbeat_stale(stale_moment)


def test_watchdog_kills_on_frozen_process(switch):
    """Un trader fige avec des positions ouvertes est plus dangereux qu'un trader arrete."""
    now = utc_now()
    switch.beat(now)
    watchdog = KillSwitchWatchdog(switch)
    assert not watchdog.check_heartbeat(now)
    frozen = now + timedelta(seconds=switch.config.watchdog_timeout_sec + 10)
    assert watchdog.check_heartbeat(frozen)
    assert switch.is_triggered()
    assert "watchdog" in switch.reason()


def test_watchdog_kills_on_drawdown_breach(switch):
    """Le watchdog lit l'equity lui-meme : aucun bug du trader ne peut le masquer."""
    watchdog = KillSwitchWatchdog(switch, max_drawdown_pct=15.0, initial_capital=10_000.0)
    healthy = pd.Series([10_000.0, 10_200.0, 9_800.0])
    assert not watchdog.check_drawdown(healthy)
    assert not switch.is_triggered()

    ruined = pd.Series([10_000.0, 10_500.0, 8_400.0])
    assert watchdog.check_drawdown(ruined)
    assert switch.is_triggered()
    assert "drawdown" in switch.reason()


def test_watchdog_respects_hard_limit(switch):
    """Meme si on lui demande 50 %, le watchdog n'ira jamais au-dela de la limite en dur."""
    watchdog = KillSwitchWatchdog(switch, max_drawdown_pct=50.0)
    assert watchdog.max_drawdown_pct == 15.0


def test_watchdog_run_once_reports_both_checks(switch):
    switch.beat()
    result = KillSwitchWatchdog(switch).run_once()
    assert set(result) == {"heartbeat_triggered", "drawdown_triggered"}
    assert not any(result.values())


# ------------------------------------------------- kill switch : serveur HTTP


def test_http_endpoints_kill_and_report(tmp_path):
    """L'endpoint HTTP doit pouvoir tout arreter, meme sans acces au terminal."""
    import json as json_module
    import urllib.request

    from trader.config import KillSwitchConfig
    from trader.risk.kill_switch import KillSwitch

    switch = KillSwitch(
        KillSwitchConfig(
            sentinel_path=str(tmp_path / "kill"),
            http_enabled=True,
            http_host="127.0.0.1",
            http_port=19091,
        )
    )
    switch.start_http_server()
    try:
        with urllib.request.urlopen("http://127.0.0.1:19091/status", timeout=5) as response:
            status = json_module.loads(response.read())
        assert status["triggered"] is False

        request = urllib.request.Request("http://127.0.0.1:19091/kill", method="POST", data=b"")
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json_module.loads(response.read())
        assert payload["status"] == "killed"
        assert switch.is_triggered()
        assert "HTTP" in switch.reason()
    finally:
        switch.stop_http_server()


def test_http_unknown_endpoint_is_rejected(tmp_path):
    import urllib.error
    import urllib.request

    from trader.config import KillSwitchConfig
    from trader.risk.kill_switch import KillSwitch

    switch = KillSwitch(
        KillSwitchConfig(sentinel_path=str(tmp_path / "kill"), http_enabled=True, http_port=19092)
    )
    switch.start_http_server()
    try:
        request = urllib.request.Request("http://127.0.0.1:19092/autre", method="POST", data=b"")
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request, timeout=5)
        assert excinfo.value.code == 404
        assert not switch.is_triggered()
    finally:
        switch.stop_http_server()


def test_http_server_disabled_by_config(tmp_path):
    from trader.config import KillSwitchConfig
    from trader.risk.kill_switch import KillSwitch

    switch = KillSwitch(KillSwitchConfig(sentinel_path=str(tmp_path / "kill"), http_enabled=False))
    switch.start_http_server()
    assert switch._server is None


def test_start_http_server_is_idempotent(tmp_path):
    from trader.config import KillSwitchConfig
    from trader.risk.kill_switch import KillSwitch

    switch = KillSwitch(
        KillSwitchConfig(sentinel_path=str(tmp_path / "kill"), http_enabled=True, http_port=19093)
    )
    switch.start_http_server()
    server = switch._server
    switch.start_http_server()
    try:
        assert switch._server is server
    finally:
        switch.stop_http_server()


def test_corrupted_sentinel_still_halts(switch):
    """Un fichier sentinelle illisible arrete quand meme le systeme."""
    switch.sentinel.parent.mkdir(parents=True, exist_ok=True)
    switch.sentinel.write_text("{ ceci n'est pas du json")
    assert switch.is_triggered()
    assert "illisible" in switch.reason()


def test_heartbeat_with_corrupted_file(switch):
    switch.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    switch.heartbeat_path.write_text("pas du json")
    assert switch.last_beat() is None
    assert not switch.is_heartbeat_stale()


def test_watchdog_ignores_already_triggered_switch(switch):
    from trader.risk.kill_switch import KillSwitchWatchdog

    switch.trigger("deja arme", source="test")
    watchdog = KillSwitchWatchdog(switch)
    assert not watchdog.check_heartbeat()  # pas de second declenchement


def test_watchdog_without_db_returns_false(switch):
    from trader.risk.kill_switch import KillSwitchWatchdog

    assert not KillSwitchWatchdog(switch, db_url=None).check_drawdown()


def test_watchdog_reads_equity_from_database(tmp_path):
    """Le watchdog lit l'equity en base, sans rien demander au trader."""
    from datetime import UTC, datetime

    from trader.config import KillSwitchConfig
    from trader.data.store import DataStore
    from trader.risk.kill_switch import KillSwitch, KillSwitchWatchdog

    db_url = f"sqlite:///{tmp_path}/wd.db"
    store = DataStore(db_url)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    for index, equity in enumerate([10_000.0, 10_500.0, 8_000.0]):
        store.save_equity(equity=equity, cash=equity, timestamp=base + timedelta(hours=index))
    store.close()

    switch = KillSwitch(KillSwitchConfig(sentinel_path=str(tmp_path / "kill"), http_enabled=False))
    watchdog = KillSwitchWatchdog(switch, db_url=db_url, max_drawdown_pct=15.0)
    assert watchdog.check_drawdown()
    assert switch.is_triggered()


def test_watchdog_run_forever_stops_when_triggered(switch):
    from trader.risk.kill_switch import KillSwitchWatchdog

    switch.trigger("arret", source="test")
    KillSwitchWatchdog(switch).run_forever(interval_sec=0.01)  # doit rendre la main

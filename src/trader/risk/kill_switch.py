"""Kill switch : arret d'urgence, independant de la boucle de trading.

Un kill switch qui vit dans le meme event loop que le trader ne sert a rien :
si la boucle est bloquee, le kill switch l'est aussi. Il y a donc quatre voies
d'arret, dont trois n'ont pas besoin que le process principal fonctionne :

1. FICHIER SENTINELLE — `touch /tmp/trader_kill` arrete tout. Marche meme si le
   reseau est coupe, meme si Python rame, meme sans terminal (un `touch` suffit).
2. ENDPOINT HTTP — `POST /kill` sur un serveur qui tourne dans son propre thread.
3. WATCHDOG — le trader ecrit un heartbeat toutes les N secondes ; si le fichier
   n'est plus rafraichi, le surveillant pose la sentinelle. Un trader fige avec
   des positions ouvertes est plus dangereux qu'un trader arrete.
4. SURVEILLANCE DU DRAWDOWN — un process externe lit l'equity en base et pose la
   sentinelle si le drawdown depasse la limite en dur, sans rien demander au
   process principal.

Le kill switch ne se desarme QUE manuellement : supprimer le fichier sentinelle
est un geste humain, deliberé. Aucun code du systeme ne l'efface tout seul.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from trader.config import HARD_MAX_DRAWDOWN_TOTAL_PCT, KillSwitchConfig
from trader.logging_setup import get_logger
from trader.utils.time_utils import to_utc, utc_now

log = get_logger(__name__)


class KillSwitch:
    """Interface du kill switch, cote process de trading.

    Le trader APPELLE `is_triggered()` a chaque iteration et s'arrete si c'est
    vrai. Il ne peut jamais le desactiver.
    """

    def __init__(
        self,
        config: KillSwitchConfig,
        event_sink: Callable[[str, str, str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.event_sink = event_sink
        self.sentinel = Path(config.sentinel_path)
        self.heartbeat_path = self.sentinel.with_suffix(".heartbeat")
        self._server: HTTPServer | None = None
        self._server_thread: threading.Thread | None = None

    # ------------------------------------------------------------ etat

    def is_triggered(self) -> bool:
        """Vrai si le kill switch est arme : le systeme doit s'arreter."""
        return self.sentinel.exists()

    def reason(self) -> str:
        """Motif de l'arret, tel qu'ecrit dans le fichier sentinelle."""
        if not self.sentinel.exists():
            return ""
        try:
            payload = json.loads(self.sentinel.read_text(encoding="utf-8"))
            return str(payload.get("reason", "inconnu"))
        except (OSError, json.JSONDecodeError):
            return "sentinelle presente (contenu illisible)"

    def trigger(self, reason: str, source: str = "system", details: dict | None = None) -> None:
        """Arme le kill switch. Irreversible sans intervention humaine."""
        payload = {
            "reason": reason,
            "source": source,
            "triggered_at": utc_now().isoformat(),
            "details": details or {},
        }
        self.sentinel.parent.mkdir(parents=True, exist_ok=True)
        self.sentinel.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.critical("kill_switch_triggered", reason=reason, source=source)
        if self.event_sink is not None:
            try:
                self.event_sink("CRITICAL", "kill_switch", reason, payload)
            except Exception as exc:  # noqa: BLE001 - l'arret prime sur son enregistrement
                log.error("kill_switch_event_persist_failed", error=str(exc))

    def clear(self, operator_confirmation: str) -> None:
        """Desarme le kill switch. Exige une confirmation explicite.

        Volontairement penible a appeler : ce geste ne doit jamais etre
        automatise, sous aucun pretexte.
        """
        if operator_confirmation != "JE CONFIRME LE REDEMARRAGE":
            raise PermissionError(
                "le kill switch ne se desarme qu'avec la confirmation explicite "
                "de l'operateur : 'JE CONFIRME LE REDEMARRAGE'"
            )
        self.sentinel.unlink(missing_ok=True)
        log.warning("kill_switch_cleared", operator=True)

    # ------------------------------------------------------- heartbeat

    def beat(self, now: datetime | None = None, payload: dict | None = None) -> None:
        """Ecrit le heartbeat du process principal (surveille par le watchdog)."""
        stamp = to_utc(now or utc_now())
        content = {"timestamp": stamp.isoformat(), **(payload or {})}
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.write_text(json.dumps(content), encoding="utf-8")

    def last_beat(self) -> datetime | None:
        """Horodatage du dernier heartbeat, ou None."""
        if not self.heartbeat_path.exists():
            return None
        try:
            payload = json.loads(self.heartbeat_path.read_text(encoding="utf-8"))
            return to_utc(datetime.fromisoformat(payload["timestamp"]))
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return None

    def is_heartbeat_stale(self, now: datetime | None = None) -> bool:
        """Vrai si le process principal n'a pas donne signe de vie a temps."""
        last = self.last_beat()
        if last is None:
            return False
        stamp = to_utc(now or utc_now())
        return stamp - last > timedelta(seconds=self.config.watchdog_timeout_sec)

    # ------------------------------------------------------ serveur HTTP

    def start_http_server(self) -> None:
        """Demarre l'endpoint HTTP d'arret dans un thread separe."""
        if not self.config.http_enabled or self._server is not None:
            return

        switch = self

        class Handler(BaseHTTPRequestHandler):
            """Endpoints : POST /kill (arret), GET /status (etat)."""

            def do_POST(self) -> None:  # noqa: N802 - impose par BaseHTTPRequestHandler
                if self.path.rstrip("/") != "/kill":
                    self.send_error(404, "endpoint inconnu")
                    return
                switch.trigger("arret demande via HTTP", source="http")
                body = json.dumps({"status": "killed", "reason": switch.reason()}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - impose par BaseHTTPRequestHandler
                body = json.dumps(
                    {
                        "triggered": switch.is_triggered(),
                        "reason": switch.reason(),
                        "last_beat": (
                            switch.last_beat().isoformat() if switch.last_beat() else None
                        ),
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                """Redirige les logs HTTP vers le logging structure."""
                log.debug("kill_switch_http", message=format % args)

        self._server = HTTPServer((self.config.http_host, self.config.http_port), Handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, name="kill-switch-http", daemon=True
        )
        self._server_thread.start()
        log.info(
            "kill_switch_http_started",
            host=self.config.http_host,
            port=self.config.http_port,
        )

    def stop_http_server(self) -> None:
        """Arrete l'endpoint HTTP."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._server_thread = None
            log.info("kill_switch_http_stopped")


class KillSwitchWatchdog:
    """Surveillant externe : watchdog de heartbeat et controle du drawdown.

    Concu pour tourner dans un PROCESS SEPARE (`python -m trader.risk.kill_switch`)
    ou en cron. Il ne partage rien avec le trader sauf le systeme de fichiers et
    la base de donnees : c'est precisement ce qui le rend fiable quand le trader
    ne repond plus.
    """

    def __init__(
        self,
        switch: KillSwitch,
        db_url: str | None = None,
        max_drawdown_pct: float = HARD_MAX_DRAWDOWN_TOTAL_PCT,
        initial_capital: float | None = None,
    ) -> None:
        self.switch = switch
        self.db_url = db_url
        self.max_drawdown_pct = min(max_drawdown_pct, HARD_MAX_DRAWDOWN_TOTAL_PCT)
        self.initial_capital = initial_capital

    def check_heartbeat(self, now: datetime | None = None) -> bool:
        """Arme le kill switch si le trader est fige. Retourne True s'il a arme."""
        if self.switch.is_triggered():
            return False
        if self.switch.is_heartbeat_stale(now):
            last = self.switch.last_beat()
            self.switch.trigger(
                f"watchdog : aucun heartbeat depuis {last.isoformat() if last else 'jamais'}",
                source="watchdog",
                details={"timeout_sec": self.switch.config.watchdog_timeout_sec},
            )
            return True
        return False

    def check_drawdown(self, equity_curve: Any = None) -> bool:
        """Arme le kill switch si le drawdown depasse la limite en dur.

        Lit l'equity directement en base : aucune dependance au process de trading,
        donc aucun moyen pour un bug du trader de masquer une perte.
        """
        curve = equity_curve
        if curve is None:
            if self.db_url is None:
                return False
            from trader.data.store import DataStore

            store = DataStore(self.db_url)
            try:
                curve = store.load_equity()
            finally:
                store.close()

        if curve is None or len(curve) < 2:
            return False

        peak = float(max(curve))
        current = float(curve.iloc[-1] if hasattr(curve, "iloc") else list(curve)[-1])
        reference = max(peak, self.initial_capital or peak)
        if reference <= 0:
            return False
        drawdown_pct = (reference - current) / reference * 100.0

        if drawdown_pct >= self.max_drawdown_pct and not self.switch.is_triggered():
            self.switch.trigger(
                f"drawdown total {drawdown_pct:.2f} % >= limite en dur "
                f"{self.max_drawdown_pct:.2f} %",
                source="watchdog_drawdown",
                details={
                    "peak_equity": round(reference, 2),
                    "current_equity": round(current, 2),
                    "drawdown_pct": round(drawdown_pct, 2),
                },
            )
            return True
        return False

    def run_once(self, now: datetime | None = None) -> dict[str, bool]:
        """Execute un cycle de surveillance."""
        return {
            "heartbeat_triggered": self.check_heartbeat(now),
            "drawdown_triggered": self.check_drawdown(),
        }

    def run_forever(self, interval_sec: float | None = None) -> None:
        """Boucle de surveillance (process separe). S'arrete quand le switch est arme."""
        import time

        interval = interval_sec or self.switch.config.check_interval_sec
        log.info("watchdog_started", interval_sec=interval, db_url=self.db_url)
        while True:
            result = self.run_once()
            if any(result.values()) or self.switch.is_triggered():
                log.critical(
                    "watchdog_kill_switch_active",
                    reason=self.switch.reason(),
                    **{k: str(v) for k, v in result.items()},
                )
                return
            time.sleep(interval)


def main() -> None:
    """Point d'entree du watchdog en process separe.

    Usage : python -m trader.risk.kill_switch --config config/paper.toml
    """
    import argparse

    from trader.config import load_settings
    from trader.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description="Watchdog externe du kill switch.")
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--once", action="store_true", help="Un seul cycle puis sortie.")
    args = parser.parse_args()

    settings = load_settings(args.config, args.override)
    configure_logging(settings.general.log_level)
    switch = KillSwitch(settings.kill_switch)
    watchdog = KillSwitchWatchdog(
        switch,
        db_url=settings.data.db_url,
        max_drawdown_pct=settings.risk.max_drawdown_total_pct,
        initial_capital=settings.general.initial_capital,
    )
    if args.once:
        log.info("watchdog_single_cycle", **{k: str(v) for k, v in watchdog.run_once().items()})
        return
    watchdog.run_forever()


if __name__ == "__main__":
    main()

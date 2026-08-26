"""Endpoints HTTP de supervision (source de donnees pour Grafana).

Volontairement en lecture seule : ce serveur ne peut RIEN declencher. Le seul
endpoint capable d'agir sur le systeme est le kill switch, qui vit dans son
propre module et ne sait faire qu'une chose — tout arreter.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from trader.logging_setup import get_logger
from trader.utils.time_utils import utc_now

log = get_logger(__name__)


class DashboardServer:
    """Sert l'etat du systeme en JSON sur quelques endpoints simples."""

    def __init__(self, orchestrator: Any, host: str = "127.0.0.1", port: int = 9092) -> None:
        self.orchestrator = orchestrator
        self.host = host
        self.port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ contenu

    def status(self) -> dict[str, Any]:
        """Etat global : portefeuille, risque, regime, strategies."""
        orchestrator = self.orchestrator
        portfolio = orchestrator.portfolio
        risk_state = orchestrator.risk.state()
        last_regime = getattr(orchestrator.detector, "last_state", None)
        return {
            "timestamp": utc_now().isoformat(),
            "mode": orchestrator.settings.general.mode.value,
            "cycles": orchestrator._cycles,
            "portfolio": portfolio.snapshot(),
            "risk": {
                "kill_switch": risk_state.kill_switch_active,
                "paused_until": (
                    risk_state.paused_until.isoformat() if risk_state.paused_until else None
                ),
                "pause_reason": risk_state.pause_reason,
                "orders_last_hour": risk_state.orders_last_hour,
                "orders_today": risk_state.orders_today,
                "breaker_trips": risk_state.breaker_trips,
                "rejections": risk_state.rejections,
            },
            "regime": last_regime.to_dict() if last_regime else None,
            "strategies": orchestrator.ensemble.snapshot(),
            "assets": orchestrator.assets,
        }

    def positions(self) -> dict[str, Any]:
        """Positions ouvertes."""
        return {
            "timestamp": utc_now().isoformat(),
            "positions": [
                {
                    "asset": position.asset,
                    "side": position.side.value,
                    "size": position.size,
                    "entry_price": position.entry_price,
                    "stop_loss": position.stop_loss,
                    "target_price": position.target_price,
                    "opened_at": position.opened_at.isoformat(),
                    "strategy": position.strategy,
                }
                for position in self.orchestrator.portfolio.positions.values()
            ],
        }

    def health(self) -> dict[str, Any]:
        """Sonde de vivacite : etat du kill switch et du heartbeat."""
        switch = self.orchestrator.kill_switch
        last_beat = switch.last_beat()
        return {
            "status": "halted" if switch.is_triggered() else "running",
            "kill_switch": switch.is_triggered(),
            "reason": switch.reason(),
            "last_heartbeat": last_beat.isoformat() if last_beat else None,
            "heartbeat_stale": switch.is_heartbeat_stale(),
            "cycles": self.orchestrator._cycles,
        }

    def routes(self) -> dict[str, Any]:
        """Table des endpoints disponibles."""
        return {
            "/status": self.status,
            "/positions": self.positions,
            "/health": self.health,
        }

    # ------------------------------------------------------------ serveur

    def start(self) -> None:
        """Demarre le serveur dans un thread separe."""
        if self._server is not None:
            return
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            """Handler en lecture seule."""

            def do_GET(self) -> None:  # noqa: N802 - impose par BaseHTTPRequestHandler
                route = dashboard.routes().get(self.path.rstrip("/") or "/status")
                if route is None:
                    self.send_error(404, "endpoint inconnu")
                    return
                try:
                    payload = json.dumps(route(), default=str).encode()
                except Exception as exc:  # noqa: BLE001 - le dashboard ne casse rien
                    log.error("dashboard_render_failed", path=self.path, error=str(exc))
                    self.send_error(500, "erreur interne")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                """Redirige les logs HTTP vers le logging structure."""
                log.debug("dashboard_http", message=format % args)

        self._server = HTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="dashboard-http", daemon=True
        )
        self._thread.start()
        log.info("dashboard_started", host=self.host, port=self.port)

    def stop(self) -> None:
        """Arrete le serveur."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None
            log.info("dashboard_stopped")

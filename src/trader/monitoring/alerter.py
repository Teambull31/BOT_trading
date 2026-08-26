"""Alertes multi-canal (Telegram, email, webhook) via apprise.

Trois niveaux, trois traitements :
- INFO : rate-limite. Un bot qui envoie un message par trade devient un bruit
  qu'on finit par ignorer, et le jour ou ca compte, on ne lit plus.
- WARNING : rate-limite plus souple.
- CRITICAL : JAMAIS rate-limite, jamais deduplique. Un kill switch doit sonner
  a chaque fois, meme si c'est la dixieme fois en une minute.

Les URLs de notification viennent d'une variable d'environnement, jamais du
fichier de configuration : un token Telegram n'a rien a faire dans un depot git.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from trader.config import MonitoringConfig
from trader.logging_setup import get_logger
from trader.utils.time_utils import to_utc, utc_now

log = get_logger(__name__)

LEVELS: dict[str, int] = {"INFO": 10, "WARNING": 20, "CRITICAL": 30}


@dataclass(slots=True)
class AlertRecord:
    """Alerte emise."""

    level: str
    message: str
    timestamp: datetime = field(default_factory=utc_now)
    delivered: bool = False


class Alerter:
    """Envoie des alertes sur les canaux configures."""

    def __init__(
        self,
        config: MonitoringConfig,
        urls: list[str] | None = None,
        store: Any | None = None,
        notifier: Any | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.history: list[AlertRecord] = []
        self._last_sent: dict[str, datetime] = {}
        self.urls = urls if urls is not None else self._urls_from_env()
        self._notifier = notifier
        if notifier is None and self.urls:
            self._notifier = self._build_notifier()

    def _urls_from_env(self) -> list[str]:
        """Lit les URLs de notification depuis l'environnement."""
        raw = os.getenv(self.config.alert_urls_env, "")
        return [url.strip() for url in raw.split(",") if url.strip()]

    def _build_notifier(self) -> Any | None:
        """Instancie apprise si disponible."""
        try:
            import apprise
        except ImportError:
            log.warning("apprise_absent", consequence="alertes seulement journalisees")
            return None
        notifier = apprise.Apprise()
        for url in self.urls:
            if not notifier.add(url):
                log.error("alert_url_invalid", url=url[:20] + "...")
        return notifier

    @property
    def is_configured(self) -> bool:
        """Vrai si au moins un canal est reellement branche."""
        return bool(self.urls) and self._notifier is not None

    # ------------------------------------------------------------- envoi

    async def send(self, level: str, message: str, details: dict[str, Any] | None = None) -> bool:
        """Envoie une alerte. Retourne True si elle a ete transmise."""
        level = level.upper()
        if level not in LEVELS:
            level = "INFO"

        record = AlertRecord(level=level, message=message)
        self.history.append(record)
        self._persist(level, message, details)

        if self._is_rate_limited(level, record.timestamp):
            log.debug("alert_rate_limited", level=level, message=message[:80])
            return False

        log_method = {
            "INFO": log.info,
            "WARNING": log.warning,
            "CRITICAL": log.critical,
        }[level]
        log_method("alert", level=level, message=message, **(details or {}))

        if not self.is_configured:
            return False

        delivered = await self._deliver(level, message)
        record.delivered = delivered
        if delivered:
            self._last_sent[level] = record.timestamp
        return delivered

    async def _deliver(self, level: str, message: str) -> bool:
        """Transmet l'alerte via apprise, sans bloquer la boucle asyncio."""
        title = f"[{level}] Trader adaptatif"
        try:
            return bool(await asyncio.to_thread(self._notifier.notify, body=message, title=title))
        except Exception as exc:  # noqa: BLE001 - une alerte ratee n'arrete pas le trading
            log.error("alert_delivery_failed", level=level, error=str(exc))
            return False

    def _is_rate_limited(self, level: str, now: datetime) -> bool:
        """Applique le rate limiting, sauf sur CRITICAL."""
        if level == "CRITICAL":
            return False
        last = self._last_sent.get(level)
        if last is None:
            return False
        window = self.config.alert_rate_limit_info_sec
        if level == "WARNING":
            window = max(1, window // 3)
        return to_utc(now) - last < timedelta(seconds=window)

    def _persist(self, level: str, message: str, details: dict[str, Any] | None) -> None:
        """Trace l'alerte en base, meme si l'envoi echoue."""
        if self.store is None:
            return
        try:
            self.store.save_event(level, "alerter", message, details or {})
        except Exception as exc:  # noqa: BLE001 - la persistence ne bloque pas l'alerte
            log.error("alert_persist_failed", error=str(exc))

    # ------------------------------------------------------------ rapports

    async def send_daily_report(
        self, portfolio: Any, ensemble: Any, extra: dict | None = None
    ) -> bool:
        """Envoie le rapport quotidien de synthese."""
        snapshot = portfolio.snapshot()
        strategies = ensemble.snapshot()
        lines = [
            "Rapport quotidien",
            f"Equity      : {snapshot['equity']:,.2f} ({snapshot['total_return_pct']:+.2f} %)",
            f"Drawdown    : {snapshot['drawdown_total_pct']:.2f} % "
            f"(jour {snapshot['drawdown_daily_pct']:.2f} %)",
            f"Positions   : {snapshot['open_positions']} "
            f"(exposition {snapshot['exposure_pct']:.1f} %)",
            f"Trades      : {snapshot['closed_trades']}",
            "Strategies  :",
        ]
        for name, state in strategies.items():
            lines.append(
                f"  - {name:<14} poids {state['weight']:.0%} | {state['health']}"
                + (" | shadow" if state["shadow"] else "")
            )
        for key, value in (extra or {}).items():
            lines.append(f"{key:<12}: {value}")
        return await self.send("INFO", "\n".join(lines))

    async def test_channels(self) -> bool:
        """Envoie un message de test (critere de la checklist go-live)."""
        if not self.is_configured:
            log.warning("alert_test_skipped", reason="aucun canal configure")
            return False
        return await self.send(
            "CRITICAL",
            "Test des alertes : si vous lisez ce message, le canal fonctionne.",
        )

    def stats(self) -> dict[str, int]:
        """Compte des alertes emises par niveau."""
        counts = {level: 0 for level in LEVELS}
        for record in self.history:
            counts[record.level] = counts.get(record.level, 0) + 1
        counts["delivered"] = sum(1 for record in self.history if record.delivered)
        return counts

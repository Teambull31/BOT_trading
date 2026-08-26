"""Circuit breakers : coupent le trading quand les conditions deviennent anormales.

Philosophie : ne pas chercher a etre malin quand le marche ou l'infrastructure
se comportent bizarrement. Un spread qui explose, une API qui rame, un prix qui
bouge de 10 % en cinq minutes : ce sont des situations ou l'on perd de l'argent
sans rien comprendre. On s'arrete, on attend, on reprend.

Chaque breaker a une portee : globale (tout le systeme) ou par actif.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from trader.config import CircuitBreakerConfig
from trader.logging_setup import get_logger
from trader.utils.time_utils import to_utc, utc_now

log = get_logger(__name__)


class BreakerReason(str, Enum):
    """Cause du declenchement d'un circuit breaker."""

    SPREAD = "spread_excessif"
    LATENCY = "latence_api"
    PRICE_MOVE = "mouvement_de_prix_extreme"
    EXECUTION_ERRORS = "erreurs_execution_repetees"
    DATA_STALE = "donnees_perimees"
    MANUAL = "declenchement_manuel"


@dataclass(slots=True)
class BreakerTrip:
    """Un declenchement actif."""

    reason: BreakerReason
    scope: str
    triggered_at: datetime
    until: datetime
    details: str = ""

    def is_active(self, now: datetime) -> bool:
        """Vrai si la pause est toujours en cours."""
        return now < self.until

    def remaining_sec(self, now: datetime) -> float:
        """Secondes restantes avant reprise."""
        return max(0.0, (self.until - now).total_seconds())


@dataclass(slots=True)
class BreakerStatus:
    """Etat consolide des circuit breakers."""

    tripped: bool
    trips: list[BreakerTrip] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        """Raisons lisibles des pauses en cours."""
        return [f"{trip.reason.value} ({trip.scope}): {trip.details}" for trip in self.trips]


class CircuitBreaker:
    """Surveille les conditions anormales et met le systeme en pause.

    GLOBAL : latence API, mouvement de prix extreme, erreurs d'execution.
    PAR ACTIF : spread excessif, donnees perimees.
    """

    GLOBAL: str = "__global__"

    def __init__(
        self,
        config: CircuitBreakerConfig,
        event_sink: Callable[[str, str, str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        # Les declenchements doivent laisser une trace durable : la checklist
        # go-live verifie que chaque breaker a deja ete eprouve au moins une fois.
        self.event_sink = event_sink
        self.trips: list[BreakerTrip] = []
        self._price_history: dict[str, deque[tuple[datetime, float]]] = {}
        self._execution_errors: deque[datetime] = deque(maxlen=50)

    # ----------------------------------------------------------- controles

    def check_spread(
        self, asset: str, spread_pct: float | None, now: datetime | None = None
    ) -> bool:
        """Verifie le spread bid/ask d'un actif. Retourne True si le breaker saute."""
        if spread_pct is None:
            return False
        if spread_pct > self.config.max_spread_pct:
            self._trip(
                BreakerReason.SPREAD,
                scope=asset,
                details=f"spread {spread_pct:.2f} % > {self.config.max_spread_pct:.2f} %",
                now=now,
            )
            return True
        return False

    def check_latency(self, exchange: str, latency_sec: float, now: datetime | None = None) -> bool:
        """Verifie la latence d'un exchange."""
        if latency_sec > self.config.max_api_latency_sec:
            self._trip(
                BreakerReason.LATENCY,
                scope=self.GLOBAL,
                details=f"{exchange} a repondu en {latency_sec:.2f} s",
                now=now,
            )
            return True
        return False

    def check_price_move(self, asset: str, price: float, now: datetime | None = None) -> bool:
        """Verifie l'amplitude du mouvement de prix sur les 5 dernieres minutes."""
        stamp = to_utc(now or utc_now())
        history = self._price_history.setdefault(asset, deque(maxlen=500))
        history.append((stamp, float(price)))

        cutoff = stamp - timedelta(minutes=5)
        window = [value for ts, value in history if ts >= cutoff]
        if len(window) < 2:
            return False

        low, high = min(window), max(window)
        if low <= 0:
            return False
        move_pct = (high - low) / low * 100.0
        if move_pct > self.config.max_price_move_5min_pct:
            self._trip(
                BreakerReason.PRICE_MOVE,
                scope=self.GLOBAL,
                details=f"{asset} a bouge de {move_pct:.1f} % en 5 min",
                now=stamp,
            )
            return True
        return False

    def record_execution_error(self, now: datetime | None = None) -> bool:
        """Enregistre une erreur d'execution ; declenche apres N erreurs rapprochees."""
        stamp = to_utc(now or utc_now())
        self._execution_errors.append(stamp)
        cutoff = stamp - timedelta(minutes=15)
        recent = [ts for ts in self._execution_errors if ts >= cutoff]
        if len(recent) >= self.config.max_execution_retries:
            self._trip(
                BreakerReason.EXECUTION_ERRORS,
                scope=self.GLOBAL,
                details=f"{len(recent)} erreurs d'execution en 15 min",
                now=stamp,
            )
            return True
        return False

    def check_data_freshness(
        self, asset: str, last_update: datetime, max_age_sec: float, now: datetime | None = None
    ) -> bool:
        """Verifie que les donnees d'un actif ne sont pas perimees.

        Trader sur des donnees figees est le meilleur moyen d'acheter un prix
        qui n'existe plus.
        """
        stamp = to_utc(now or utc_now())
        age = (stamp - to_utc(last_update)).total_seconds()
        if age > max_age_sec:
            self._trip(
                BreakerReason.DATA_STALE,
                scope=asset,
                details=f"donnees vieilles de {age:.0f} s (max {max_age_sec:.0f} s)",
                now=stamp,
            )
            return True
        return False

    def trip_manually(
        self, reason: str, scope: str | None = None, now: datetime | None = None
    ) -> None:
        """Declenchement manuel (usage operateur)."""
        self._trip(BreakerReason.MANUAL, scope=scope or self.GLOBAL, details=reason, now=now)

    # -------------------------------------------------------------- etat

    def _trip(
        self,
        reason: BreakerReason,
        scope: str,
        details: str,
        now: datetime | None = None,
    ) -> None:
        """Active une pause."""
        stamp = to_utc(now or utc_now())
        until = stamp + timedelta(minutes=self.config.pause_duration_min)
        existing = next(
            (trip for trip in self.trips if trip.reason is reason and trip.scope == scope), None
        )
        if existing is not None and existing.is_active(stamp):
            existing.until = max(existing.until, until)
            existing.details = details
            return
        trip = BreakerTrip(
            reason=reason, scope=scope, triggered_at=stamp, until=until, details=details
        )
        self.trips.append(trip)
        log.warning(
            "circuit_breaker_tripped",
            reason=reason.value,
            scope=scope,
            details=details,
            pause_min=self.config.pause_duration_min,
        )
        self._emit(reason, scope, details)

    def _emit(self, reason: BreakerReason, scope: str, details: str) -> None:
        """Enregistre le declenchement dans l'audit trail, sans jamais echouer."""
        if self.event_sink is None:
            return
        try:
            self.event_sink(
                "WARNING",
                "circuit_breaker",
                f"{reason.value} sur {scope} : {details}",
                {"reason": reason.value, "scope": scope, "details": details},
            )
        except Exception as exc:  # noqa: BLE001 - l'audit ne doit jamais bloquer la protection
            log.error("breaker_event_persist_failed", error=str(exc))

    def status(self, asset: str | None = None, now: datetime | None = None) -> BreakerStatus:
        """Etat des breakers, globalement ou pour un actif donne."""
        stamp = to_utc(now or utc_now())
        self.trips = [trip for trip in self.trips if trip.is_active(stamp)]
        relevant = [
            trip
            for trip in self.trips
            if trip.scope == self.GLOBAL or (asset is not None and trip.scope == asset)
        ]
        return BreakerStatus(tripped=bool(relevant), trips=relevant)

    def is_tripped(self, asset: str | None = None, now: datetime | None = None) -> bool:
        """Vrai si le trading est suspendu (globalement ou sur cet actif)."""
        return self.status(asset, now).tripped

    def reset(self, scope: str | None = None) -> None:
        """Leve les pauses (usage operateur ou tests)."""
        if scope is None:
            self.trips.clear()
        else:
            self.trips = [trip for trip in self.trips if trip.scope != scope]
        log.info("circuit_breaker_reset", scope=scope or "all")

    def snapshot(self, now: datetime | None = None) -> dict[str, object]:
        """Etat serialisable de TOUS les breakers actifs, toutes portees confondues."""
        stamp = to_utc(now or utc_now())
        self.trips = [trip for trip in self.trips if trip.is_active(stamp)]
        return {
            "active": len(self.trips),
            "trips": [
                {
                    "reason": trip.reason.value,
                    "scope": trip.scope,
                    "remaining_sec": round(trip.remaining_sec(stamp), 1),
                    "details": trip.details,
                }
                for trip in self.trips
            ],
        }

"""Utilitaires temporels : tout est en UTC, sans exception."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
    "1w": 604800,
}


def utc_now() -> datetime:
    """Retourne l'instant courant en UTC (timezone-aware)."""
    return datetime.now(UTC)


def to_utc(value: datetime) -> datetime:
    """Normalise un datetime en UTC (les naifs sont supposes UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def timeframe_to_seconds(timeframe: str) -> int:
    """Convertit un timeframe ccxt ('1h') en secondes."""
    try:
        return TIMEFRAME_SECONDS[timeframe]
    except KeyError as exc:
        raise ValueError(f"timeframe inconnu : {timeframe!r}") from exc


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    """Convertit un timeframe ccxt en timedelta."""
    return timedelta(seconds=timeframe_to_seconds(timeframe))


def bars_per_day(timeframe: str) -> float:
    """Nombre de bougies par jour pour un timeframe donne."""
    return 86400.0 / timeframe_to_seconds(timeframe)


def annualization_factor(timeframe: str) -> float:
    """Facteur d'annualisation des returns pour un timeframe (crypto : 365 j)."""
    return bars_per_day(timeframe) * 365.0


def to_millis(value: datetime) -> int:
    """Convertit un datetime en timestamp millisecondes (format ccxt)."""
    return int(to_utc(value).timestamp() * 1000)


def from_millis(value: int | float) -> datetime:
    """Convertit un timestamp millisecondes en datetime UTC."""
    return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)


def floor_to_timeframe(value: datetime, timeframe: str) -> datetime:
    """Arrondit un datetime au debut de la bougie de son timeframe."""
    seconds = timeframe_to_seconds(timeframe)
    epoch = int(to_utc(value).timestamp())
    return datetime.fromtimestamp(epoch - (epoch % seconds), tz=UTC)

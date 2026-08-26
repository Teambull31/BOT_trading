"""Fixtures partagees : generateurs de marches synthetiques et configuration de test.

Les generateurs produisent des regimes controles (tendance, range, crise) : c'est
la seule facon de tester deterministiquement un systeme qui doit reagir au regime.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from trader.config import Settings, load_settings
from trader.data.features import FeatureBuilder
from trader.data.store import DataStore

START = datetime(2024, 1, 1, tzinfo=UTC)


def make_ohlcv(
    n: int = 600,
    start_price: float = 2000.0,
    drift: float = 0.0,
    vol: float = 0.01,
    seed: int = 7,
    freq: str = "1h",
    start: datetime = START,
    mean_revert: float = 0.0,
    jump_at: int | None = None,
    jump_size: float = -0.25,
) -> pd.DataFrame:
    """Genere une serie OHLCV synthetique au comportement controle.

    Args:
        drift: derive par bougie (tendance).
        vol: volatilite par bougie.
        mean_revert: force de rappel vers la moyenne (0 = marche aleatoire).
        jump_at: index d'un choc de prix (simulation de crise).
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    log_price = np.log(start_price)
    anchor = log_price
    prices = np.empty(n)
    for i in range(n):
        shock = rng.normal(drift, vol)
        if jump_at is not None and i == jump_at:
            shock += jump_size
        log_price += shock + mean_revert * (anchor - log_price)
        prices[i] = np.exp(log_price)

    noise = np.abs(rng.normal(0.0, vol / 2.0, n))
    frame = pd.DataFrame(
        {
            "open": prices * (1.0 - noise / 2.0),
            "high": prices * (1.0 + noise),
            "low": prices * (1.0 - noise),
            "close": prices,
            "volume": rng.lognormal(5.0, 0.4, n),
        },
        index=index,
    )
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return frame


@pytest.fixture
def trending_market() -> pd.DataFrame:
    """Marche haussier net, volatilite moderee."""
    return make_ohlcv(n=800, drift=0.0016, vol=0.006, seed=11)


@pytest.fixture
def ranging_market() -> pd.DataFrame:
    """Marche en range, fort rappel vers la moyenne."""
    return make_ohlcv(n=800, drift=0.0, vol=0.005, mean_revert=0.12, seed=13)


@pytest.fixture
def crisis_market() -> pd.DataFrame:
    """Marche en crise : choc violent puis volatilite elevee."""
    return make_ohlcv(n=800, drift=-0.002, vol=0.03, seed=17, jump_at=600, jump_size=-0.3)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Configuration de test, isolee du systeme de fichiers reel."""
    return load_settings(
        "config/default.toml",
        overrides={
            "data": {"db_url": f"sqlite:///{tmp_path}/test.db", "history_days": 30},
            "general": {
                "initial_capital": 10000.0,
                "heartbeat_path": str(tmp_path / "heartbeat"),
            },
            "kill_switch": {"sentinel_path": str(tmp_path / "kill"), "http_enabled": False},
            "monitoring": {"prometheus_enabled": False},
        },
    )


@pytest.fixture
def store(settings: Settings) -> DataStore:
    """DataStore isole sur disque temporaire."""
    instance = DataStore(settings.data.db_url)
    yield instance
    instance.close()


@pytest.fixture
def memory_store() -> DataStore:
    """DataStore en memoire (tests rapides)."""
    instance = DataStore("sqlite:///:memory:")
    yield instance
    instance.close()


@pytest.fixture
def feature_builder() -> FeatureBuilder:
    """FeatureBuilder sur timeframe horaire."""
    return FeatureBuilder(timeframe="1h")


@pytest.fixture
def featured_market(trending_market: pd.DataFrame, feature_builder: FeatureBuilder):
    """Couple (ohlcv, features) pret a l'emploi."""
    return trending_market, feature_builder.build(trending_market)

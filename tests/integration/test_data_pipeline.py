"""Test d'integration du pipeline de donnees : ingester -> normalizer -> store -> features.

Aucun appel reseau : un faux client ccxt est injecte dans l'ingester.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from tests.conftest import make_ohlcv
from trader.data.features import FeatureBuilder, assert_no_lookahead
from trader.data.ingester import DataIngester, PermanentError, TransientError
from trader.utils.time_utils import utc_now


class FakeExchange:
    """Faux client ccxt : sert des bougies deterministes, sait echouer a la demande."""

    id = "fake"

    def __init__(self, candles: list[list[float]], fail_times: int = 0, permanent: bool = False):
        self.candles = candles
        self.fail_times = fail_times
        self.permanent = permanent
        self.calls = 0
        self.closed = False

    async def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise BadSymbol("symbole inconnu") if self.permanent else ConnectionError("timeout")
        rows = self.candles
        if since is not None:
            rows = [row for row in rows if row[0] >= since]
        return rows[: limit or 1000]

    @property
    def last_price(self) -> float:
        """Dernier prix des bougies servies : le carnet doit rester coherent avec elles."""
        return float(self.candles[-1][4]) if self.candles else 2000.0

    async def fetch_order_book(self, symbol, limit=None):
        mid = self.last_price
        tick = mid * 0.0001
        return {
            "bids": [[mid - tick * (i + 1), 1.0 + i] for i in range(limit or 20)],
            "asks": [[mid + tick * (i + 1), 0.5 + i] for i in range(limit or 20)],
            "timestamp": 1704067200000,
        }

    async def fetch_ticker(self, symbol):
        return {"last": self.last_price}

    async def fetch_funding_rate(self, symbol):
        return {"fundingRate": 0.0003}

    async def close(self):
        self.closed = True


class BadSymbol(Exception):
    """Imite ccxt.BadSymbol (erreur fonctionnelle, non retentable)."""


def to_ccxt(frame) -> list[list[float]]:
    return [
        [int(ts.timestamp() * 1000), row.open, row.high, row.low, row.close, row.volume]
        for ts, row in frame.iterrows()
    ]


@pytest.fixture
def candles():
    """Bougies se terminant a l'instant present : l'ingester demande `since` relatif a now."""
    start = utc_now() - timedelta(hours=300)
    return to_ccxt(make_ohlcv(n=300, drift=0.001, seed=5, start=start))


async def test_full_pipeline_ingest_to_features(settings, memory_store, candles):
    ingester = DataIngester(
        settings, store=memory_store, clients={"binance": FakeExchange(candles)}
    )
    result = await ingester.fetch_ohlcv("ETH/USDT", "1h", limit=300)

    assert not result.frame.empty
    assert result.rows_saved == len(result.frame)
    assert result.report.is_usable

    stored = memory_store.load_ohlcv("ETH/USDT", "1h")
    assert len(stored) == len(result.frame)

    builder = FeatureBuilder(timeframe="1h")
    features = builder.build(stored)
    assert not features.empty
    assert assert_no_lookahead(builder, stored) == []


async def test_raw_payload_is_archived_before_normalization(settings, memory_store, candles):
    ingester = DataIngester(
        settings, store=memory_store, clients={"binance": FakeExchange(candles)}
    )
    await ingester.fetch_ohlcv("ETH/USDT", "1h", limit=100)
    raw = memory_store.load_raw("ETH/USDT", "ohlcv_1h")
    assert raw, "la donnee brute doit etre archivee pour l'audit trail"


async def test_incremental_update_only_fetches_new_candles(settings, memory_store, candles):
    exchange = FakeExchange(candles)
    ingester = DataIngester(settings, store=memory_store, clients={"binance": exchange})
    await ingester.fetch_ohlcv("ETH/USDT", "1h", limit=300)
    before = len(memory_store.load_ohlcv("ETH/USDT", "1h"))

    result = await ingester.update_incremental("ETH/USDT", "1h")
    assert result.rows_saved == 0  # rien de nouveau cote exchange
    assert len(memory_store.load_ohlcv("ETH/USDT", "1h")) == before


async def test_transient_errors_are_retried(settings, memory_store, candles):
    exchange = FakeExchange(candles, fail_times=2)
    ingester = DataIngester(
        settings,
        store=memory_store,
        clients={"binance": exchange},
        max_retries=3,
        backoff_base_sec=0.01,
    )
    result = await ingester.fetch_ohlcv("ETH/USDT", "1h", limit=50)
    assert not result.frame.empty
    assert exchange.calls >= 3


async def test_permanent_errors_are_not_retried(settings, memory_store, candles):
    exchange = FakeExchange(candles, fail_times=5, permanent=True)
    ingester = DataIngester(
        settings,
        store=memory_store,
        clients={"binance": exchange},
        max_retries=3,
        backoff_base_sec=0.01,
    )
    with pytest.raises(PermanentError):
        await ingester.fetch_ohlcv("ETH/USDT", "1h", limit=50)
    assert exchange.calls == 1  # une seule tentative


async def test_transient_error_exhausts_retries(settings, memory_store, candles):
    exchange = FakeExchange(candles, fail_times=99)
    ingester = DataIngester(
        settings,
        store=memory_store,
        clients={"binance": exchange},
        max_retries=2,
        backoff_base_sec=0.01,
    )
    with pytest.raises(TransientError):
        await ingester.fetch_ohlcv("ETH/USDT", "1h", limit=50)


async def test_order_book_summary(settings, memory_store, candles):
    ingester = DataIngester(
        settings, store=memory_store, clients={"binance": FakeExchange(candles)}
    )
    book = await ingester.fetch_order_book("ETH/USDT", depth=10)
    assert book["best_bid"] < book["best_ask"]
    assert 0.0 < book["spread_pct"] < 1.0
    assert -1.0 <= book["imbalance"] <= 1.0


async def test_backfill_multiple_assets_and_timeframes(settings, memory_store, candles):
    ingester = DataIngester(
        settings, store=memory_store, clients={"binance": FakeExchange(candles)}
    )
    results = await ingester.backfill(assets=["ETH/USDT"], timeframes=["1h", "4h"])
    assert set(results) == {"ETH/USDT|1h", "ETH/USDT|4h"}


async def test_rate_limiter_spaces_calls(settings, memory_store, candles):
    settings.exchanges.binance = {"rate_limit_ms": 50, "sandbox": True}
    ingester = DataIngester(
        settings, store=memory_store, clients={"binance": FakeExchange(candles)}
    )
    start = asyncio.get_running_loop().time()
    await ingester.fetch_ohlcv("ETH/USDT", "1h", limit=10)
    await ingester.fetch_ohlcv("ETH/USDT", "1h", limit=10)
    assert asyncio.get_running_loop().time() - start >= 0.05

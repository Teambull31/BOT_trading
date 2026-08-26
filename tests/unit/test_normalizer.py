"""Tests de normalisation des donnees de marche."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tests.conftest import make_ohlcv
from trader.data.normalizer import (
    align_frames,
    detect_outliers,
    normalize_ohlcv,
    resample_ohlcv,
)

BASE_MS = 1704067200000  # 2024-01-01T00:00:00Z


def ccxt_rows(n: int = 24, step_ms: int = 3_600_000) -> list[list[float]]:
    return [
        [BASE_MS + i * step_ms, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0] for i in range(n)
    ]


LATER = datetime(2024, 1, 10, tzinfo=UTC)


def test_normalize_ccxt_payload():
    frame, report = normalize_ohlcv(ccxt_rows(), "1h", now=LATER)
    assert len(frame) == 24
    assert frame.index.tz is not None
    assert frame.index.is_monotonic_increasing
    assert report.is_usable


def test_duplicates_are_removed():
    rows = ccxt_rows(10)
    rows.append(rows[3])
    frame, report = normalize_ohlcv(rows, "1h", now=LATER)
    assert report.duplicates_removed == 1
    assert not frame.index.has_duplicates


def test_gaps_are_filled_conservatively():
    rows = ccxt_rows(10)
    del rows[5]
    frame, report = normalize_ohlcv(rows, "1h", now=LATER)
    assert report.gaps_filled == 1
    assert len(frame) == 10
    # Le volume manquant vaut 0, on n'invente pas d'activite.
    assert frame["volume"].min() == 0.0


def test_invalid_rows_are_dropped():
    rows = ccxt_rows(10)
    rows[2][4] = -5.0  # close negatif
    rows[3][5] = float("nan")  # volume NaN
    frame, report = normalize_ohlcv(rows, "1h", now=LATER, fill_gaps=False)
    assert report.invalid_rows_dropped == 2
    assert len(frame) == 8


def test_incoherent_ohlc_is_repaired():
    rows = ccxt_rows(5)
    rows[2][2] = 50.0  # high sous le close
    frame, report = normalize_ohlcv(rows, "1h", now=LATER)
    assert any("incoherentes" in issue for issue in report.issues)
    assert (frame["high"] >= frame["close"]).all()


def test_incomplete_last_candle_is_dropped():
    """Une bougie non close est une information partielle du futur."""
    now = datetime.fromtimestamp(BASE_MS / 1000, tz=UTC) + timedelta(hours=23, minutes=30)
    frame, report = normalize_ohlcv(ccxt_rows(24), "1h", now=now)
    assert report.incomplete_dropped is True
    assert len(frame) == 23


def test_empty_input_is_flagged_fatal():
    frame, report = normalize_ohlcv([], "1h", now=LATER)
    assert frame.empty
    assert not report.is_usable


def test_resample_to_higher_timeframe():
    frame, _ = normalize_ohlcv(ccxt_rows(24), "1h", now=LATER)
    resampled = resample_ohlcv(frame, "4h")
    assert len(resampled) == 6
    assert resampled["volume"].iloc[0] == pytest.approx(40.0)
    assert resampled["high"].iloc[0] == frame["high"].iloc[:4].max()


def test_align_frames_on_common_index():
    left = make_ohlcv(n=100, seed=1)
    right = make_ohlcv(n=100, seed=2).iloc[10:]
    aligned = align_frames({"a": left, "b": right})
    assert len(aligned["a"]) == len(aligned["b"]) == 90
    assert aligned["a"].index.equals(aligned["b"].index)


def test_detect_outliers_flags_extreme_moves():
    frame = make_ohlcv(n=300, vol=0.005, seed=9)
    frame.iloc[150, frame.columns.get_loc("close")] *= 3.0
    flagged = detect_outliers(frame, sigma=6.0)
    assert bool(flagged.iloc[150])
    assert flagged.sum() < 5

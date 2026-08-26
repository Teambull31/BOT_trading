"""Tests du feature engineering, look-ahead bias en tete."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv
from trader.data.features import (
    adx,
    assert_no_lookahead,
    atr,
    bollinger,
    macd,
    rsi,
)


def test_no_lookahead_on_any_feature(feature_builder, trending_market):
    """Le test le plus important du module : recalculer sur un prefixe doit
    donner exactement les memes valeurs. Toute divergence = fuite du futur."""
    offenders = assert_no_lookahead(feature_builder, trending_market)
    assert offenders == [], f"features avec look-ahead : {offenders}"


@pytest.mark.parametrize("market", ["trending", "ranging", "crisis"])
def test_no_lookahead_across_regimes(feature_builder, market):
    frames = {
        "trending": make_ohlcv(n=500, drift=0.002, vol=0.008, seed=3),
        "ranging": make_ohlcv(n=500, mean_revert=0.1, vol=0.006, seed=4),
        "crisis": make_ohlcv(n=500, vol=0.04, seed=5, jump_at=400),
    }
    assert assert_no_lookahead(feature_builder, frames[market]) == []


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.linspace(100, 200, 100))
    down = pd.Series(np.linspace(200, 100, 100))
    assert rsi(up, 14).dropna().between(0, 100).all()
    assert rsi(up, 14).iloc[-1] > 95
    assert rsi(down, 14).iloc[-1] < 5


def test_macd_signal_relationship():
    close = pd.Series(np.linspace(100, 200, 200))
    line, signal, hist = macd(close)
    assert np.allclose((line - signal).dropna(), hist.dropna())
    assert line.dropna().iloc[-1] > 0  # tendance haussiere -> MACD positif


def test_bollinger_ordering():
    close = pd.Series(np.random.default_rng(0).normal(100, 5, 200))
    lower, middle, upper = bollinger(close, 20)
    valid = lower.notna()
    assert (lower[valid] <= middle[valid]).all()
    assert (middle[valid] <= upper[valid]).all()


def test_atr_is_positive_and_reacts_to_volatility():
    calm = make_ohlcv(n=300, vol=0.002, seed=1)
    wild = make_ohlcv(n=300, vol=0.03, seed=1)
    calm_atr = atr(calm["high"], calm["low"], calm["close"]).dropna()
    wild_atr = atr(wild["high"], wild["low"], wild["close"]).dropna()
    assert (calm_atr > 0).all()
    assert wild_atr.mean() / wild["close"].mean() > calm_atr.mean() / calm["close"].mean()


def test_adx_detects_trend_versus_range():
    trend = make_ohlcv(n=600, drift=0.004, vol=0.004, seed=21)
    flat = make_ohlcv(n=600, drift=0.0, vol=0.004, mean_revert=0.25, seed=21)
    trend_adx = adx(trend["high"], trend["low"], trend["close"])[0].dropna().mean()
    flat_adx = adx(flat["high"], flat["low"], flat["close"])[0].dropna().mean()
    assert trend_adx > flat_adx


def test_builder_produces_expected_columns(featured_market):
    _, features = featured_market
    expected = {
        "rsi_14",
        "macd_hist",
        "bb_width",
        "atr_pct",
        "adx",
        "vwap",
        "obv",
        "hurst",
        "realized_vol_7d",
        "volume_ratio",
        "autocorr_1",
    }
    assert expected.issubset(set(features.columns))
    assert len(features) == len(featured_market[0])


def test_builder_rejects_unsorted_index(feature_builder, trending_market):
    shuffled = trending_market.sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError, match="trie"):
        feature_builder.build(shuffled)


def test_builder_rejects_missing_columns(feature_builder, trending_market):
    with pytest.raises(ValueError, match="manquantes"):
        feature_builder.build(trending_market.drop(columns=["volume"]))


def test_external_features_are_lagged(feature_builder, trending_market):
    """Funding et OI sont publies avec retard : ils doivent etre decales de 1."""
    funding = pd.Series(np.linspace(0.0, 0.01, len(trending_market)), index=trending_market.index)
    features = feature_builder.build(trending_market, funding_rates=funding)
    aligned = features["funding_rate"].dropna()
    assert np.allclose(aligned.to_numpy(), funding.shift(1).dropna().to_numpy())


def test_hurst_separates_trend_from_mean_reversion(feature_builder):
    trend = make_ohlcv(n=700, drift=0.003, vol=0.004, seed=31)
    revert = make_ohlcv(n=700, drift=0.0, vol=0.004, mean_revert=0.3, seed=31)
    trend_hurst = feature_builder.build(trend)["hurst"].dropna().mean()
    revert_hurst = feature_builder.build(revert)["hurst"].dropna().mean()
    assert trend_hurst > revert_hurst


def test_empty_input_returns_empty_frame(feature_builder):
    assert feature_builder.build(pd.DataFrame()).empty

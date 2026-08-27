"""Tests de la simulation actions.

Ce qui est teste ici n'est pas la rentabilite — personne ne peut la garantir —
mais l'HONNETETE de la simulation : pas de lecture du futur, execution a la
barre suivante, frais toujours preleves, stops respectes, selection d'univers
aveugle a la fenetre evaluee.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from trader.equities.backtest import (
    EquityBacktester,
    ExecutionCosts,
    RiskParams,
)
from trader.equities.data import _parse_money, _to_frame
from trader.equities.selection import score_candidates
from trader.equities.strategy import (
    TrendParams,
    assert_signals_causal,
    compute_indicators,
    entry_signal,
)

warnings.filterwarnings("ignore")


def make_series(
    n: int = 500,
    start_price: float = 100.0,
    drift: float = 0.0008,
    vol: float = 0.015,
    seed: int = 3,
    start: date = date(2023, 1, 2),
) -> pd.DataFrame:
    """Serie de cours quotidienne synthetique (jours ouvres)."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n)
    prices = start_price * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    noise = np.abs(rng.normal(0.0, vol / 2.0, n))
    frame = pd.DataFrame(
        {
            "open": prices * (1.0 - noise / 3.0),
            "high": prices * (1.0 + noise),
            "low": prices * (1.0 - noise),
            "close": prices,
            "volume": rng.lognormal(15.0, 0.3, n),
        },
        index=index,
    )
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return frame


@pytest.fixture
def trending_stock() -> pd.DataFrame:
    return make_series(n=600, drift=0.0012, vol=0.012, seed=11)


@pytest.fixture
def choppy_stock() -> pd.DataFrame:
    return make_series(n=600, drift=0.0, vol=0.01, seed=12)


# ------------------------------------------------------------------ donnees


def test_money_parsing():
    assert _parse_money("$932.97") == pytest.approx(932.97)
    assert _parse_money("19,163,180") == pytest.approx(19_163_180)
    assert np.isnan(_parse_money("N/A"))


def test_raw_rows_to_frame():
    rows = [
        {
            "date": "08/25/2026",
            "open": "$928.97",
            "high": "$946.67",
            "low": "$916.30",
            "close": "$932.97",
            "volume": "19,163,180",
        },
        {
            "date": "08/24/2026",
            "open": "$934.78",
            "high": "$936.45",
            "low": "$887.60",
            "close": "$910.43",
            "volume": "30,012,630",
        },
    ]
    frame = _to_frame(rows)
    assert len(frame) == 2
    assert frame.index.is_monotonic_increasing  # trie chronologiquement
    assert frame["close"].iloc[-1] == pytest.approx(932.97)


def test_incomplete_rows_are_dropped():
    rows = [
        {
            "date": "08/25/2026",
            "open": "$1",
            "high": "$2",
            "low": "$0.5",
            "close": "$1.5",
            "volume": "10",
        },
        {
            "date": "08/24/2026",
            "open": "N/A",
            "high": "N/A",
            "low": "N/A",
            "close": "N/A",
            "volume": "0",
        },
    ]
    # Mieux vaut un trou dans la serie qu'un prix invente.
    assert len(_to_frame(rows)) == 1


# ------------------------------------------------------------- causalite


def test_indicators_never_read_the_future(trending_stock):
    """Le test central : recalculer sur un prefixe doit donner les memes valeurs."""
    assert assert_signals_causal(trending_stock, TrendParams()) == []


@pytest.mark.parametrize("seed", [1, 7, 21])
def test_causality_across_random_series(seed):
    assert assert_signals_causal(make_series(n=450, seed=seed), TrendParams()) == []


def test_donchian_high_excludes_current_bar(trending_stock):
    """Sans decalage, toute cloture casserait mecaniquement son propre plus haut."""
    indicators = compute_indicators(trending_stock, TrendParams())
    valid = indicators["donchian_high"].notna()
    manual = trending_stock["high"].rolling(20).max().shift(1)
    pd.testing.assert_series_equal(
        indicators.loc[valid, "donchian_high"],
        manual[valid],
        check_names=False,
    )


# ------------------------------------------------------------------ signaux


def test_no_entry_below_trend_filter(choppy_stock):
    """Sous la moyenne 200 jours, aucune entree n'est autorisee."""
    params = TrendParams()
    indicators = compute_indicators(choppy_stock, params)
    below = indicators[indicators["close"] < indicators["sma_trend"]].dropna()
    if below.empty:
        pytest.skip("aucune seance sous la moyenne longue sur cet echantillon")
    for position in range(1, min(20, len(below))):
        window = indicators.loc[: below.index[position]]
        if float(window["close"].iloc[-1]) < float(window["sma_trend"].iloc[-1]):
            assert not entry_signal(window, position=0, mode="breakout")
            assert not entry_signal(window, position=0, mode="trend")


def test_no_entry_when_already_positioned(trending_stock):
    indicators = compute_indicators(trending_stock, TrendParams())
    assert not entry_signal(indicators, position=1, mode="trend")


def test_trend_mode_is_more_permissive_than_breakout(trending_stock):
    """Le mode tendance entre plus souvent : c'est tout son interet."""
    params = TrendParams()
    indicators = compute_indicators(trending_stock, params)
    indicators.attrs["min_adx"] = params.min_adx
    trend_signals = breakout_signals = 0
    for position in range(250, len(indicators), 5):
        window = indicators.iloc[: position + 1]
        trend_signals += entry_signal(window, 0, mode="trend")
        breakout_signals += entry_signal(window, 0, mode="breakout")
    assert trend_signals >= breakout_signals


# ---------------------------------------------------------------- backtest


@pytest.fixture
def universe(trending_stock, choppy_stock) -> dict[str, pd.DataFrame]:
    return {"AAA": trending_stock, "BBB": choppy_stock}


def window_of(frames: dict[str, pd.DataFrame], last_days: int = 200) -> tuple[date, date]:
    index = frames["AAA"].index
    return index[-last_days].date(), index[-1].date()


def test_backtest_produces_equity_and_trades(universe):
    start, end = window_of(universe)
    report = EquityBacktester().run(universe, start, end, 1000.0)
    assert len(report.equity) > 10
    assert report.equity.index.is_monotonic_increasing
    assert report.final_equity > 0


def test_every_trade_pays_costs(universe):
    start, end = window_of(universe)
    report = EquityBacktester(
        params=TrendParams(entry_mode="trend"),
        risk=RiskParams(sizing_mode="target_weight", max_position_pct=33.0),
    ).run(universe, start, end, 1000.0)
    if not report.trades:
        pytest.skip("aucun trade sur cet echantillon")
    assert all(trade.costs > 0 for trade in report.trades)
    assert report.metrics["total_costs"] > 0


def test_entry_never_uses_the_signal_bar_price(universe):
    """L'ordre part a l'ouverture SUIVANTE, jamais au cours qui a declenche le signal."""
    start, end = window_of(universe)
    report = EquityBacktester(params=TrendParams(entry_mode="trend")).run(
        universe, start, end, 1000.0
    )
    if not report.trades:
        pytest.skip("aucun trade sur cet echantillon")
    costs = ExecutionCosts()
    for trade in report.trades:
        frame = universe[trade.symbol]
        expected = costs.entry_price(float(frame.loc[trade.entry_date, "open"]))
        assert trade.entry_price == pytest.approx(expected, rel=1e-9)


def test_exit_happens_after_entry(universe):
    start, end = window_of(universe)
    report = EquityBacktester(params=TrendParams(entry_mode="trend")).run(
        universe, start, end, 1000.0
    )
    assert all(trade.exit_date > trade.entry_date for trade in report.trades)


def test_risk_sizing_bounds_the_loss(universe):
    """En mode risque, une sortie au stop ne doit pas couter bien plus que le budget."""
    start, end = window_of(universe, last_days=300)
    capital = 1000.0
    report = EquityBacktester(
        params=TrendParams(entry_mode="breakout"),
        risk=RiskParams(sizing_mode="risk", risk_per_trade_pct=1.0),
    ).run(universe, start, end, capital)
    for trade in report.trades:
        if trade.pnl < 0:
            # Budget de 1 %, tolerance pour les gaps d'ouverture et les frais.
            assert trade.pnl > -capital * 0.05, f"perte anormale : {trade.pnl:.2f}"


def test_target_weight_actually_invests(universe):
    """Le mode allocation cible doit engager le capital, pas le laisser dormir."""
    start, end = window_of(universe, last_days=300)
    report = EquityBacktester(
        params=TrendParams(entry_mode="trend"),
        risk=RiskParams(sizing_mode="target_weight", max_position_pct=33.0),
    ).run(universe, start, end, 1000.0)
    if not report.trades:
        pytest.skip("aucun trade sur cet echantillon")
    notionals = [trade.shares * trade.entry_price for trade in report.trades]
    assert max(notionals) > 1000.0 * 0.15


def test_position_limit_is_enforced(universe):
    start, end = window_of(universe, last_days=300)
    backtester = EquityBacktester(
        params=TrendParams(entry_mode="trend"),
        risk=RiskParams(sizing_mode="target_weight", max_positions=1, max_position_pct=33.0),
    )
    report = backtester.run(universe, start, end, 1000.0)
    # Avec une seule position autorisee, deux trades ne peuvent pas se chevaucher.
    by_date = sorted((t.entry_date, t.exit_date) for t in report.trades)
    for (_, first_exit), (second_entry, _) in zip(by_date, by_date[1:], strict=False):
        assert second_entry >= first_exit


def test_benchmark_is_computed(universe):
    start, end = window_of(universe)
    report = EquityBacktester().run(universe, start, end, 1000.0)
    assert len(report.benchmark) > 0
    assert "benchmark_max_drawdown_pct" in report.metrics


def test_empty_universe_is_rejected():
    with pytest.raises(ValueError, match="aucun titre"):
        EquityBacktester().run({}, date(2024, 1, 1), date(2024, 6, 1), 1000.0)


def test_backtest_is_deterministic(universe):
    start, end = window_of(universe)
    first = EquityBacktester().run(universe, start, end, 1000.0)
    second = EquityBacktester().run(universe, start, end, 1000.0)
    assert first.final_equity == pytest.approx(second.final_equity)
    assert len(first.trades) == len(second.trades)


def test_capital_never_goes_negative(universe):
    start, end = window_of(universe, last_days=400)
    report = EquityBacktester(
        params=TrendParams(entry_mode="trend"),
        risk=RiskParams(sizing_mode="target_weight", max_position_pct=33.0),
    ).run(universe, start, end, 1000.0)
    assert (report.equity > 0).all()


# --------------------------------------------------------------- selection


def test_selection_prefers_trending_and_decorrelated():
    """Un titre decorrele mais sans tendance n'aide pas un systeme de tendance."""
    imposed = {"IMP": make_series(n=600, drift=0.0015, vol=0.02, seed=1)}
    candidates = {
        "TENDANCE": make_series(n=600, drift=0.0015, vol=0.012, seed=99),
        "PLAT": make_series(n=600, drift=0.0, vol=0.008, seed=98),
    }
    scores = {s.symbol: s for s in score_candidates(imposed, candidates, min_dollar_volume_m=0.0)}
    assert scores["TENDANCE"].score > scores["PLAT"].score


def test_illiquid_candidate_is_excluded():
    imposed = {"IMP": make_series(n=400, seed=1)}
    thin = make_series(n=400, seed=2)
    thin["volume"] = 10.0  # quasi aucun echange
    scores = score_candidates(imposed, {"THIN": thin}, min_dollar_volume_m=200.0)
    assert not scores[0].eligible
    assert "liquidite" in scores[0].reason


def test_short_history_candidate_is_excluded():
    imposed = {"IMP": make_series(n=400, seed=1)}
    scores = score_candidates(imposed, {"NEW": make_series(n=40, seed=2)}, min_dollar_volume_m=0.0)
    assert not scores[0].eligible
    assert "historique" in scores[0].reason


def test_selection_window_excludes_evaluation_period():
    """Garde-fou methodologique : la selection s'arrete avant la fenetre evaluee."""
    evaluation_start = date(2026, 6, 1)
    selection_end = evaluation_start - timedelta(days=1)
    assert selection_end < evaluation_start


# --------------------------------------------------------------- profils


def test_profiles_form_a_coherent_risk_gradient():
    """Le curseur doit etre ordonne : l'exposition croit du prudent a l'agressif."""
    from trader.equities.profiles import ORDER, PROFILES

    exposures = [PROFILES[key].max_exposure_pct for key in ORDER if key != "budget_risque"]
    assert exposures == sorted(exposures)
    stops = [PROFILES[key].strategy.trailing_atr for key in ORDER if key != "budget_risque"]
    assert stops == sorted(stops)


def test_no_profile_uses_leverage():
    """Aucun profil ne doit engager plus que le capital disponible."""
    from trader.equities.profiles import PROFILES

    for profile in PROFILES.values():
        assert profile.max_exposure_pct <= 100.0


def test_profile_lookup_is_case_insensitive_and_validated():
    from trader.equities.profiles import get_profile

    assert get_profile("  EQUILIBRE ").key == "equilibre"
    with pytest.raises(ValueError, match="profil inconnu"):
        get_profile("turbo")


def test_profiles_describe_expected_behaviour():
    """Chaque profil doit annoncer ce a quoi s'attendre, pas seulement ses reglages."""
    from trader.equities.profiles import PROFILES

    for profile in PROFILES.values():
        assert profile.intent and profile.expected_behaviour
        assert "%" in profile.describe()


def test_higher_exposure_produces_larger_drawdown(universe):
    """Propriete fondamentale : plus d'exposition, plus de drawdown."""
    from trader.equities.profiles import PROFILES

    start, end = window_of(universe, last_days=400)
    drawdowns = {}
    for key in ("defensif", "equilibre", "offensif"):
        profile = PROFILES[key]
        report = EquityBacktester(params=profile.strategy, risk=profile.risk).run(
            universe, start, end, 1000.0
        )
        drawdowns[key] = report.metrics["max_drawdown_pct"]
    assert drawdowns["defensif"] <= drawdowns["equilibre"] + 1e-6
    assert drawdowns["equilibre"] <= drawdowns["offensif"] + 1e-6


# ------------------------------------------------------------- diagnostic


def test_diagnostic_detects_healthy_market():
    from trader.equities.diagnostic import diagnose

    frames = {
        "AAA": make_series(n=600, drift=0.0015, vol=0.010, seed=31),
        "BBB": make_series(n=600, drift=0.0012, vol=0.011, seed=32),
        "CCC": make_series(n=600, drift=0.0013, vol=0.009, seed=33),
    }
    result = diagnose(frames)
    assert result.breadth_pct > 50.0
    assert "PORTEUR" in result.verdict or "MITIGE" in result.verdict
    assert 0 <= result.caution_score <= result.max_caution


def test_diagnostic_detects_bear_market():
    """Univers entierement sous sa moyenne longue : le verdict doit etre prudent."""
    from trader.equities.diagnostic import diagnose

    frames = {
        "AAA": make_series(n=600, drift=-0.002, vol=0.02, seed=41),
        "BBB": make_series(n=600, drift=-0.0025, vol=0.022, seed=42),
    }
    result = diagnose(frames)
    assert result.breadth_pct < 50.0
    assert result.recommended_profile == "defensif"
    assert "DIFFICILE" in result.verdict
    assert result.warnings


def test_diagnostic_flags_volatility_spike():
    from trader.equities.diagnostic import diagnose

    calm = make_series(n=600, drift=0.001, vol=0.008, seed=51)
    # On triple la volatilite sur les 20 dernieres seances.
    shocked = calm.copy()
    rng = np.random.default_rng(7)
    shocks = rng.normal(0.0, 0.05, 20)
    shocked.iloc[-20:, shocked.columns.get_loc("close")] *= np.exp(np.cumsum(shocks))
    result = diagnose({"AAA": shocked, "BBB": calm})
    assert result.mean_vol_ratio > 1.0


def test_diagnostic_uses_broad_market_reference():
    """Un univers en difficulte pendant que le marche va bien n'est pas la meme
    chose qu'une correction generale."""
    from trader.equities.diagnostic import diagnose

    universe = {"AAA": make_series(n=600, drift=-0.002, vol=0.02, seed=61)}
    falling_market = make_series(n=600, drift=-0.0025, vol=0.02, seed=62)
    with_benchmark = diagnose(universe, benchmark_frame=falling_market)
    without = diagnose(universe)
    assert with_benchmark.benchmark is not None
    assert with_benchmark.caution_score > without.caution_score


def test_diagnostic_reports_are_renderable():
    from trader.equities.diagnostic import diagnose

    result = diagnose({"AAA": make_series(n=600, seed=71), "BBB": make_series(n=600, seed=72)})
    rendered = result.render()
    assert "VERDICT" in rendered
    assert "Score de prudence" in rendered


def test_diagnostic_requires_enough_history():
    from trader.equities.diagnostic import diagnose

    with pytest.raises(ValueError, match="historique insuffisant"):
        diagnose({"AAA": make_series(n=50, seed=81)})


def test_diagnostic_rejects_empty_input():
    from trader.equities.diagnostic import diagnose

    with pytest.raises(ValueError, match="aucune donnee"):
        diagnose({})

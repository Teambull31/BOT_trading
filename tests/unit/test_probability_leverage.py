"""Tests du faisceau de signaux, de la table de probabilite et du levier.

Ce qui est verifie ici n'est pas qu'un signal gagne — les mesures montrent
justement qu'aucun ne gagne — mais que la MESURE est honnete : pas de lecture
du futur dans la table de probabilite, et un simulateur a levier qui ne cache
ni les liquidations, ni les gaps, ni le cout de portage.
"""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

from trader.equities.leverage import (
    LeverageParams,
    direction_from_probability,
    simulate_leveraged,
)
from trader.equities.probability import (
    assert_probabilities_causal,
    build_panel,
    calibration_table,
    causal_probabilities,
    stability_by_year,
)
from trader.equities.signals import SIGNAL_NAMES, compute_signals, forward_outcome

warnings.filterwarnings("ignore")


def make_series(
    n: int = 900,
    start_price: float = 100.0,
    drift: float = 0.0005,
    vol: float = 0.015,
    seed: int = 5,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(date(2020, 1, 1), periods=n)
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
def universe() -> dict[str, pd.DataFrame]:
    return {
        "AAA": make_series(seed=1, drift=0.0008),
        "BBB": make_series(seed=2, drift=0.0),
        "CCC": make_series(seed=3, drift=-0.0004),
    }


# ------------------------------------------------------------------ signaux


def test_score_stays_within_bounds(universe):
    """Sept signaux a -1/0/+1 : le score ne peut pas sortir de [-7, +7]."""
    signals = compute_signals(universe["AAA"]).dropna(subset=["score"])
    assert not signals.empty
    assert signals["score"].between(-len(SIGNAL_NAMES), len(SIGNAL_NAMES)).all()


def test_signals_take_only_three_values(universe):
    signals = compute_signals(universe["AAA"])
    for name in SIGNAL_NAMES:
        values = set(signals[name].dropna().unique())
        assert values <= {-1.0, 0.0, 1.0}, f"{name} produit {values}"


def test_signals_do_not_read_the_future(universe):
    """Recalcul sur prefixe : les signaux passes ne doivent pas bouger."""
    frame = universe["AAA"]
    cut = int(len(frame) * 0.7)
    full = compute_signals(frame).iloc[:cut]
    prefix = compute_signals(frame.iloc[:cut])
    for name in (*SIGNAL_NAMES, "score"):
        both = full[name].notna() & prefix[name].notna()
        pd.testing.assert_series_equal(
            full.loc[both, name], prefix.loc[both, name], check_names=False
        )


def test_forward_outcome_looks_ahead_by_design(universe):
    """La variable a expliquer regarde le futur : c'est son role, pas un bug."""
    frame = universe["AAA"]
    outcome = forward_outcome(frame, 10)
    expected = frame["close"].iloc[10] / frame["close"].iloc[0] - 1.0
    assert outcome.iloc[0] == pytest.approx(expected)
    # Les dernieres barres n'ont pas d'avenir observable.
    assert outcome.iloc[-10:].isna().all()


# -------------------------------------------------------------- probabilites


def test_probability_table_is_causal(universe):
    """LE test central : ajouter des donnees futures ne change rien au passe."""
    assert assert_probabilities_causal(universe, horizon=10) == []


@pytest.mark.parametrize("horizon", [5, 20])
def test_probability_causal_across_horizons(universe, horizon):
    assert assert_probabilities_causal(universe, horizon=horizon) == []


def test_probabilities_stay_in_range(universe):
    tables = causal_probabilities(universe, horizon=10)
    for table in tables.values():
        valid = table["probabilite"].dropna()
        assert valid.between(0.0, 1.0).all()


def test_early_bars_have_no_probability(universe):
    """Avant d'avoir un horizon complet d'historique, aucune estimation possible."""
    tables = causal_probabilities(universe, horizon=10)
    table = tables["AAA"]
    assert table["probabilite"].iloc[:10].isna().all()


def test_calibration_compares_to_base_rate_not_to_coin_flip(universe):
    """L'ecart doit se mesurer au taux de base, sinon un signal nul parait bon."""
    rows = calibration_table(universe, horizon=10)
    if not rows:
        pytest.skip("echantillon synthetique trop petit")
    base = rows[0].base_rate
    assert all(row.base_rate == base for row in rows)
    for row in rows:
        assert row.edge_pct == pytest.approx((row.win_rate - base) * 100.0)


def test_panel_drops_bars_without_observable_outcome(universe):
    panel = build_panel(universe, horizon=10)
    assert panel["forward"].notna().all()
    assert panel["score"].notna().all()


def test_stability_reports_one_row_per_year(universe):
    table = stability_by_year(universe, horizon=10)
    assert not table.empty
    assert "base_rate" in table.columns
    assert table["base_rate"].between(0.0, 1.0).all()


# ------------------------------------------------------------------- levier


def test_liquidation_threshold_shrinks_with_leverage():
    """A levier 30, un mouvement de 1.67 % suffit a effacer le compte."""
    assert LeverageParams(leverage=1.0).ruin_move_pct == pytest.approx(50.0)
    assert LeverageParams(leverage=30.0).ruin_move_pct == pytest.approx(100.0 / 30.0 * 0.5)
    assert LeverageParams(leverage=30.0).ruin_move_pct < LeverageParams(leverage=10.0).ruin_move_pct


def test_flat_direction_never_trades(universe):
    frame = universe["AAA"]
    flat = pd.Series(0.0, index=frame.index)
    result = simulate_leveraged(frame, flat, LeverageParams(leverage=30.0), 1000.0)
    assert result.trades == 0
    assert result.final_equity == pytest.approx(1000.0)
    assert not result.ruined


def test_leverage_multiplies_losses_on_a_falling_stock():
    """Sur un titre qui baisse, plus de levier = plus de perte, jusqu'a la ruine."""
    frame = make_series(n=400, drift=-0.004, vol=0.01, seed=9)
    always_long = pd.Series(1.0, index=frame.index)
    low = simulate_leveraged(frame, always_long, LeverageParams(leverage=1.0), 1000.0)
    high = simulate_leveraged(frame, always_long, LeverageParams(leverage=30.0), 1000.0)
    assert high.total_return_pct < low.total_return_pct
    assert high.ruined


def test_financing_is_charged_while_positioned():
    """Le portage doit etre preleve : l'ignorer flatte massivement le levier."""
    frame = make_series(n=300, drift=0.0, vol=0.005, seed=4)
    always_long = pd.Series(1.0, index=frame.index)
    result = simulate_leveraged(frame, always_long, LeverageParams(leverage=10.0), 1000.0)
    if result.trades == 0:
        pytest.skip("aucune position ouverte")
    assert result.financing_paid > 0.0


def test_equity_can_go_negative_on_a_gap():
    """Un gap violent doit pouvoir laisser une DETTE, pas un compte a zero."""
    n = 60
    index = pd.bdate_range(date(2024, 1, 1), periods=n)
    close = np.full(n, 100.0)
    close[40:] = 50.0  # -50 % du jour au lendemain, sans cours intermediaire
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=index,
    )
    always_long = pd.Series(1.0, index=index)
    result = simulate_leveraged(frame, always_long, LeverageParams(leverage=30.0), 1000.0)
    assert result.ruined
    assert result.final_equity < 0.0, "un gap de -50 % a levier 30 doit creer une dette"
    assert result.total_return_pct < -100.0


def test_direction_thresholds_are_symmetric():
    probability = pd.Series([0.30, 0.50, 0.70, np.nan])
    direction = direction_from_probability(probability, 0.55, 0.45)
    assert list(direction) == [-1.0, 0.0, 1.0, 0.0]

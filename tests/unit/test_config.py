"""Tests de configuration : les hard limits doivent etre inviolables."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trader.config import (
    HARD_MAX_DRAWDOWN_TOTAL_PCT,
    HARD_MAX_POSITION_PCT,
    DevilAdvocateConfig,
    EnsembleConfig,
    Mode,
    RiskConfig,
    Settings,
    load_settings,
)


def test_default_config_loads():
    settings = load_settings("config/default.toml")
    assert settings.general.mode is Mode.PAPER
    assert settings.risk.max_position_pct <= HARD_MAX_POSITION_PCT
    assert settings.devil_advocate.enabled is True


def test_paper_and_live_overrides_load():
    paper = load_settings("config/default.toml", "config/paper.toml")
    live = load_settings("config/default.toml", "config/live.toml")
    assert paper.general.mode is Mode.PAPER
    assert live.general.mode is Mode.LIVE
    # Le live doit etre au moins aussi conservateur que le paper.
    assert live.risk.max_position_pct <= paper.risk.max_position_pct
    assert live.risk.max_drawdown_total_pct <= paper.risk.max_drawdown_total_pct


def test_position_size_hard_limit_cannot_be_relaxed():
    with pytest.raises(ValidationError, match="limite en dur"):
        RiskConfig(max_position_pct=HARD_MAX_POSITION_PCT + 0.5)


def test_drawdown_hard_limit_cannot_be_relaxed():
    with pytest.raises(ValidationError, match="limite en dur"):
        RiskConfig(max_drawdown_total_pct=HARD_MAX_DRAWDOWN_TOTAL_PCT + 1.0)


def test_devil_advocate_cannot_be_disabled():
    with pytest.raises(ValidationError, match="non desactivable"):
        DevilAdvocateConfig(enabled=False)


def test_devil_advocate_thresholds_must_be_ordered():
    with pytest.raises(ValidationError):
        DevilAdvocateConfig(abort_threshold=0.4, reduce_threshold=0.6)


def test_single_strategy_weight_is_capped_at_40pct():
    with pytest.raises(ValidationError):
        EnsembleConfig(max_weight_single=0.6)


def test_ensemble_requires_at_least_two_strategies():
    with pytest.raises(ValidationError):
        EnsembleConfig(min_active_strategies=1)


def test_drawdown_thresholds_must_be_ordered():
    with pytest.raises(ValidationError, match="drawdown"):
        RiskConfig(max_drawdown_daily_pct=8.0, max_drawdown_weekly_pct=4.0)


def test_crisis_regime_forbids_new_positions():
    with pytest.raises(ValidationError):
        RiskConfig(crisis_regime={"allow_new_positions": True})


def test_live_capital_is_capped():
    with pytest.raises(ValidationError, match="plafond"):
        Settings(
            general={"mode": "live", "initial_capital": 100_000.0},
            paper_trading={"max_live_capital": 500.0},
        )


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValidationError):
        RiskConfig(max_position_ptc=2.0)


def test_go_live_criteria_cannot_be_weakened():
    from trader.config import PaperTradingConfig

    with pytest.raises(ValidationError):
        PaperTradingConfig(min_days_before_live=5)
    with pytest.raises(ValidationError):
        PaperTradingConfig(min_sharpe_for_live=0.1)
    with pytest.raises(ValidationError):
        PaperTradingConfig(min_trades_for_live=10)


def test_exchange_settings_fallback():
    """Un exchange non declare dans la config recoit des valeurs par defaut sures."""
    settings = load_settings("config/default.toml")
    unknown = settings.exchanges.settings_for("exchange_inconnu")
    assert unknown.sandbox is True  # par defaut on ne touche jamais au reel
    assert unknown.api_key_env is None
    binance = settings.exchanges.settings_for("binance")
    assert binance.api_key_env == "BINANCE_API_KEY"


def test_all_exchange_names_include_primary_first():
    settings = load_settings("config/default.toml")
    names = settings.exchanges.all_names
    assert names[0] == settings.exchanges.primary
    assert len(names) == len(set(names))


def test_timeframe_helpers():
    """Les conversions de timeframe sont la base de toutes les annualisations."""
    from trader.utils.time_utils import (
        annualization_factor,
        bars_per_day,
        floor_to_timeframe,
        from_millis,
        timeframe_to_seconds,
        timeframe_to_timedelta,
        to_millis,
        to_utc,
        utc_now,
    )

    assert timeframe_to_seconds("1h") == 3600
    assert timeframe_to_timedelta("1d").days == 1
    assert bars_per_day("1h") == 24.0
    assert annualization_factor("1d") == 365.0

    with pytest.raises(ValueError, match="timeframe inconnu"):
        timeframe_to_seconds("3 semaines")

    now = utc_now()
    assert from_millis(to_millis(now)).replace(microsecond=0) == now.replace(microsecond=0)

    from datetime import datetime

    naive = datetime(2024, 1, 1, 12, 34)
    assert to_utc(naive).tzinfo is not None
    assert floor_to_timeframe(to_utc(naive), "1h").minute == 0

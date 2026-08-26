"""Tests du risk manager.

Le principe teste ici : un refus ne se negocie pas. Chaque regle doit bloquer,
seule, independamment de tout le reste — pas de compensation entre criteres.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from trader.config import (
    HARD_MAX_DRAWDOWN_TOTAL_PCT,
    HARD_MAX_POSITION_PCT,
    load_settings,
)
from trader.models import (
    ContraRecommendation,
    ContraReport,
    EnsembleDecision,
    OrderSide,
    Regime,
    RegimeState,
    RiskDecision,
    Signal,
)
from trader.portfolio import Portfolio, position_from_fill
from trader.risk.kill_switch import KillSwitch
from trader.risk.manager import RiskManager
from trader.risk.position_sizer import PositionSizer
from trader.utils.time_utils import utc_now

CAPITAL = 10_000.0
ENTRY = 2000.0


@pytest.fixture
def settings(tmp_path):
    return load_settings(
        "config/default.toml",
        overrides={
            "general": {"initial_capital": CAPITAL},
            "data": {"db_url": f"sqlite:///{tmp_path}/risk.db"},
            "kill_switch": {"sentinel_path": str(tmp_path / "kill"), "http_enabled": False},
        },
    )


@pytest.fixture
def portfolio():
    return Portfolio(CAPITAL)


@pytest.fixture
def manager(settings, portfolio):
    return RiskManager(settings, portfolio, kill_switch=KillSwitch(settings.kill_switch))


def make_regime(regime: Regime = Regime.BULL_LOW_VOL, transition: float = 0.1) -> RegimeState:
    return RegimeState(
        regime=regime, confidence=0.85, agreement_score=1.0, transition_probability=transition
    )


def make_intent(
    regime: RegimeState | None = None,
    entry: float = ENTRY,
    stop: float = ENTRY * 0.97,
    target: float | None = ENTRY * 1.08,
    side: OrderSide = OrderSide.BUY,
    contra: ContraReport | None = None,
    asset: str = "ETH/USDT",
    confidence: float = 0.7,
):
    from trader.models import TradeIntent

    regime = regime or make_regime()
    decision = EnsembleDecision(
        asset=asset,
        signal=Signal.BUY if side is OrderSide.BUY else Signal.SELL,
        score=0.8,
        confidence=confidence,
        consensus=1.0,
        dispersion=0.0,
        weights={"a": 0.5, "b": 0.5},
        contributions=[],
        stop_loss=stop,
        target_price=target,
        entry_price=entry,
    )
    return TradeIntent(
        asset=asset,
        side=side,
        entry_price=entry,
        stop_loss=stop,
        target_price=target,
        confidence=confidence,
        regime=regime,
        decision=decision,
        contra_report=contra
        or ContraReport(
            contra_signals=[], contra_score=0.1, recommendation=ContraRecommendation.PROCEED
        ),
    )


# ------------------------------------------------------------------ nominal


def test_valid_trade_is_approved(manager):
    verdict = manager.evaluate(make_intent(), {"ETH/USDT": ENTRY})
    assert verdict.is_approved
    assert verdict.decision is RiskDecision.APPROVED
    assert verdict.approved_notional > 0


def test_position_never_exceeds_hard_limit(manager):
    verdict = manager.evaluate(make_intent(confidence=1.0), {"ETH/USDT": ENTRY})
    assert verdict.approved_notional <= CAPITAL * HARD_MAX_POSITION_PCT / 100.0 + 1e-6


# ------------------------------------------------------------- kill switch


def test_kill_switch_blocks_everything(manager):
    manager.kill_switch.trigger("test", source="test")
    verdict = manager.evaluate(make_intent())
    assert not verdict.is_approved
    assert "kill switch" in verdict.reasons[0]


def test_kill_switch_cannot_be_cleared_without_confirmation(manager):
    manager.kill_switch.trigger("test", source="test")
    with pytest.raises(PermissionError):
        manager.kill_switch.clear("oui")
    assert manager.kill_switch.is_triggered()
    manager.kill_switch.clear("JE CONFIRME LE REDEMARRAGE")
    assert not manager.kill_switch.is_triggered()


# --------------------------------------------------------------- drawdowns


def test_total_drawdown_triggers_kill_switch(manager, portfolio):
    portfolio.cash = CAPITAL * (1.0 - HARD_MAX_DRAWDOWN_TOTAL_PCT / 100.0 - 0.01)
    verdict = manager.evaluate(make_intent())
    assert not verdict.is_approved
    assert manager.kill_switch.is_triggered()


def test_daily_drawdown_pauses_trading(manager, portfolio, settings):
    portfolio.cash = CAPITAL * (1.0 - settings.risk.max_drawdown_daily_pct / 100.0 - 0.005)
    verdict = manager.evaluate(make_intent())
    assert not verdict.is_approved
    assert manager.pause is not None
    assert "drawdown" in manager.pause.reason


def test_pause_expires(manager, portfolio, settings):
    portfolio.cash = CAPITAL * 0.96
    manager.evaluate(make_intent())
    assert manager.pause is not None
    portfolio.reset_to(CAPITAL)
    future = utc_now() + timedelta(hours=settings.risk.daily_pause_hours + 1)
    assert manager.evaluate(make_intent(), now=future).is_approved


# ------------------------------------------------------------- devil advocate


def test_devil_advocate_abort_blocks_trade(manager):
    report = ContraReport(
        contra_signals=["regime instable", "divergence"],
        contra_score=0.85,
        recommendation=ContraRecommendation.ABORT,
    )
    verdict = manager.evaluate(make_intent(contra=report))
    assert not verdict.is_approved
    assert "devil advocate" in verdict.reasons[0]


def test_devil_advocate_reduce_halves_size(manager, settings):
    normal = manager.evaluate(make_intent(), {"ETH/USDT": ENTRY})
    report = ContraReport(
        contra_signals=["volume faible"],
        contra_score=0.5,
        recommendation=ContraRecommendation.REDUCE_SIZE,
    )
    reduced = manager.evaluate(make_intent(contra=report), {"ETH/USDT": ENTRY})
    assert reduced.decision is RiskDecision.REDUCED
    assert reduced.approved_notional == pytest.approx(
        normal.approved_notional * settings.devil_advocate.reduce_factor, rel=1e-6
    )


def test_missing_devil_advocate_report_is_rejected(manager):
    """Une intention non auditee ne passe pas : le module est obligatoire."""
    intent = make_intent()
    unaudited = type(intent)(
        asset=intent.asset,
        side=intent.side,
        entry_price=intent.entry_price,
        stop_loss=intent.stop_loss,
        target_price=intent.target_price,
        confidence=intent.confidence,
        regime=intent.regime,
        decision=intent.decision,
        contra_report=None,
    )
    verdict = manager.evaluate(unaudited)
    assert not verdict.is_approved
    assert "devil advocate" in verdict.reasons[0]


# ----------------------------------------------------------------- regimes


def test_crisis_regime_blocks_new_positions(manager):
    verdict = manager.evaluate(make_intent(regime=make_regime(Regime.CRISIS)))
    assert not verdict.is_approved
    assert "crise" in verdict.reasons[0]


def test_uncertain_regime_halves_position(manager, settings):
    normal = manager.evaluate(make_intent(), {"ETH/USDT": ENTRY})
    uncertain = manager.evaluate(
        make_intent(regime=make_regime(Regime.UNCERTAIN)), {"ETH/USDT": ENTRY}
    )
    assert uncertain.is_approved
    assert uncertain.approved_notional == pytest.approx(
        normal.approved_notional * settings.risk.uncertain_regime.exposure_multiplier, rel=1e-6
    )


# ------------------------------------------------------- qualite des trades


def test_missing_stop_is_rejected(manager):
    verdict = manager.evaluate(make_intent(stop=0.0))
    assert not verdict.is_approved


def test_inverted_stop_is_rejected(manager):
    verdict = manager.evaluate(make_intent(stop=ENTRY * 1.03))
    assert not verdict.is_approved
    assert "stop loss au-dessus" in verdict.reasons[0]


def test_stop_too_far_is_rejected(manager, settings):
    distance = settings.risk.max_stop_distance_pct / 100.0 + 0.02
    verdict = manager.evaluate(make_intent(stop=ENTRY * (1 - distance)))
    assert not verdict.is_approved
    assert "trop eloigne" in verdict.reasons[0]


def test_poor_risk_reward_is_rejected(manager):
    # Stop a -3 %, cible a +2 % : ratio 0.67, sous le minimum de 1.5.
    verdict = manager.evaluate(make_intent(stop=ENTRY * 0.97, target=ENTRY * 1.02))
    assert not verdict.is_approved
    assert "risk/reward" in verdict.reasons[0]


def test_missing_target_is_rejected(manager):
    verdict = manager.evaluate(make_intent(target=None))
    assert not verdict.is_approved


# ---------------------------------------------------------------- limites


def test_duplicate_position_is_rejected(manager, portfolio):
    portfolio.open_position(position_from_fill("ETH/USDT", OrderSide.BUY, 0.1, ENTRY, ENTRY * 0.97))
    verdict = manager.evaluate(make_intent())
    assert not verdict.is_approved
    assert "deja ouverte" in verdict.reasons[0]


def test_max_concurrent_positions_is_enforced(manager, portfolio, settings):
    for i in range(settings.risk.max_concurrent_positions):
        portfolio.open_position(
            position_from_fill(f"ASSET{i}/USDT", OrderSide.BUY, 0.1, 100.0, 97.0)
        )
    verdict = manager.evaluate(make_intent())
    assert not verdict.is_approved
    assert "positions ouvertes" in verdict.reasons[0]


def test_cooldown_blocks_rapid_reentry(manager, portfolio, settings):
    portfolio.open_position(position_from_fill("ETH/USDT", OrderSide.BUY, 0.1, ENTRY, ENTRY * 0.97))
    portfolio.close_position("ETH/USDT", ENTRY * 1.01, reason="take_profit")
    verdict = manager.evaluate(make_intent())
    assert not verdict.is_approved
    assert "cooldown" in verdict.reasons[0]

    later = utc_now() + timedelta(minutes=settings.risk.cooldown_same_asset_min + 1)
    assert manager.evaluate(make_intent(), now=later).is_approved


def test_hourly_rate_limit(manager, settings):
    for _ in range(settings.risk.max_orders_per_hour):
        manager.record_order()
    verdict = manager.evaluate(make_intent())
    assert not verdict.is_approved
    assert "dans l'heure" in verdict.reasons[0]


def test_daily_rate_limit(manager, settings):
    now = utc_now()
    for i in range(settings.risk.max_orders_per_day):
        manager.record_order(now - timedelta(hours=2 + i % 20))
    verdict = manager.evaluate(make_intent())
    assert not verdict.is_approved
    assert "aujourd'hui" in verdict.reasons[0]


def test_total_exposure_limits_new_positions(manager, portfolio, settings):
    """Une fois l'exposition maximale atteinte, plus rien ne passe."""
    max_notional = CAPITAL * settings.risk.max_exposure_pct / 100.0
    portfolio.open_position(
        position_from_fill("OTHER/USDT", OrderSide.BUY, max_notional / 100.0, 100.0, 97.0)
    )
    verdict = manager.evaluate(make_intent(), {"OTHER/USDT": 100.0, "ETH/USDT": ENTRY})
    assert not verdict.is_approved


# -------------------------------------------------------- circuit breakers


def test_circuit_breaker_blocks_asset(manager):
    manager.breaker.check_spread("ETH/USDT", spread_pct=5.0)
    verdict = manager.evaluate(make_intent())
    assert not verdict.is_approved
    assert "circuit breaker" in verdict.reasons[0]


def test_global_circuit_breaker_blocks_all_assets(manager):
    manager.breaker.check_latency("binance", latency_sec=10.0)
    verdict = manager.evaluate(make_intent(asset="BTC/USDT"))
    assert not verdict.is_approved


def test_breaker_on_other_asset_does_not_block(manager):
    manager.breaker.check_spread("SOL/USDT", spread_pct=5.0)
    assert manager.evaluate(make_intent(asset="ETH/USDT")).is_approved


# ---------------------------------------------------------------- divers


def test_corrupted_config_is_refused_at_construction(settings, portfolio):
    settings.risk.max_position_pct = HARD_MAX_POSITION_PCT + 1.0
    with pytest.raises(ValueError, match="limite en dur"):
        RiskManager(settings, portfolio)


def test_rejections_are_counted(manager):
    manager.evaluate(make_intent(regime=make_regime(Regime.CRISIS)))
    manager.evaluate(make_intent(regime=make_regime(Regime.CRISIS)))
    assert manager.state().rejections["regime_crisis"] == 2


def test_should_close_all_on_crisis(manager):
    close, reason = manager.should_close_all(regime_is_crisis=True)
    assert close and "crise" in reason


def test_sizer_half_kelly_is_conservative(settings):
    sizer = PositionSizer(settings.risk)
    result = sizer.size(
        equity=CAPITAL,
        entry_price=ENTRY,
        stop_loss=ENTRY * 0.97,
        confidence=0.9,
        regime=make_regime(),
        win_rate=0.6,
        win_loss_ratio=2.0,
    )
    assert result.is_tradable
    assert result.fraction_of_equity <= HARD_MAX_POSITION_PCT / 100.0 + 1e-9
    assert "kelly" in result.method


def test_risk_based_size_matches_intended_loss(settings):
    sizer = PositionSizer(settings.risk)
    size = sizer.risk_based_size(CAPITAL, ENTRY, ENTRY * 0.98, risk_pct=1.0)
    loss_if_stopped = size * (ENTRY - ENTRY * 0.98)
    assert loss_if_stopped <= CAPITAL * 0.01 + 1e-6


def test_breaker_status_is_serializable(manager):
    manager.breaker.check_spread("ETH/USDT", spread_pct=5.0)
    snapshot = manager.breaker.snapshot()
    assert snapshot["active"] == 1
    assert snapshot["trips"][0]["reason"] == "spread_excessif"

"""Tests de la detection de decay et de l'evaluation des strategies."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from trader.adaptation.decay_detector import StrategyDecayDetector
from trader.adaptation.evaluator import StrategyEvaluator
from trader.config import DecayDetectionConfig
from trader.models import OrderSide, StrategyHealth, Trade
from trader.utils.time_utils import utc_now

NOW = utc_now()


def make_trades(
    store,
    strategy: str,
    pnls: list[float],
    start_days_ago: float = 20.0,
    span_days: float = 20.0,
    size: float = 0.1,
    price: float = 2000.0,
) -> None:
    """Enregistre une serie de trades etales dans le temps."""
    if not pnls:
        return
    step = span_days / max(1, len(pnls))
    for index, pnl in enumerate(pnls):
        closed = NOW - timedelta(days=start_days_ago - index * step)
        store.save_trade(
            Trade(
                asset="ETH/USDT",
                side=OrderSide.BUY,
                size=size,
                entry_price=price,
                exit_price=price + pnl / size,
                pnl=pnl,
                fees=0.1,
                opened_at=closed - timedelta(hours=4),
                closed_at=closed,
                strategy=strategy,
            ),
            mode="paper",
        )


@pytest.fixture
def evaluator(memory_store):
    return StrategyEvaluator(memory_store, min_trades=5, mode="paper")


@pytest.fixture
def detector(evaluator):
    return StrategyDecayDetector(
        DecayDetectionConfig(min_trades_for_verdict=10), evaluator, shadow_mode_days=14
    )


# ------------------------------------------------------------------ evaluateur


def test_metrics_on_winning_strategy(memory_store, evaluator):
    make_trades(memory_store, "gagnante", [10.0, 12.0, -5.0, 15.0, 8.0, -3.0, 11.0, 9.0])
    metrics = evaluator.evaluate("gagnante").window(30)
    assert metrics.trades == 8
    assert metrics.total_pnl == pytest.approx(57.0)
    assert metrics.hit_rate == pytest.approx(6 / 8)
    assert metrics.profit_factor > 1.0
    assert metrics.sharpe > 0


def test_metrics_on_losing_strategy(memory_store, evaluator):
    make_trades(memory_store, "perdante", [-10.0, -12.0, 5.0, -15.0, -8.0, 3.0, -11.0])
    metrics = evaluator.evaluate("perdante").window(30)
    assert metrics.total_pnl < 0
    assert metrics.profit_factor < 1.0
    assert metrics.sharpe < 0


def test_small_sample_neutralizes_ratios(memory_store):
    """Un Sharpe sur trois trades ne mesure rien : on refuse de le propager."""
    strict = StrategyEvaluator(memory_store, min_trades=20, mode="paper")
    make_trades(memory_store, "jeune", [10.0, 12.0, 11.0])
    metrics = strict.evaluate("jeune").window(30)
    assert metrics.trades == 3
    assert not metrics.is_significant
    assert metrics.sharpe == 0.0


def test_windows_are_independent(memory_store, evaluator):
    make_trades(memory_store, "s", [10.0] * 5, start_days_ago=25, span_days=3)
    make_trades(memory_store, "s", [-5.0] * 5, start_days_ago=3, span_days=2)
    snapshot = evaluator.evaluate("s")
    assert snapshot.window(7).total_pnl < 0
    assert snapshot.window(30).total_pnl > 0


def test_provider_format_matches_ensemble(memory_store, evaluator):
    make_trades(memory_store, "s", [5.0, -2.0, 6.0, 3.0, -1.0, 4.0])
    payload = evaluator.metrics_provider("s")
    assert set(payload) >= {"sharpe_30d", "sharpe_7d", "hit_rate", "trades"}


def test_unknown_strategy_returns_empty_metrics(evaluator):
    metrics = evaluator.evaluate("inexistante").window(30)
    assert metrics.trades == 0
    assert metrics.sharpe == 0.0


def test_ranking_orders_by_sharpe(memory_store, evaluator):
    make_trades(memory_store, "bonne", [10.0, 11.0, 9.0, 12.0, 10.0, 11.0])
    make_trades(memory_store, "mauvaise", [-10.0, -11.0, 5.0, -12.0, -10.0, -11.0])
    ranking = evaluator.ranking(["mauvaise", "bonne"])
    assert ranking[0][0] == "bonne"


# ------------------------------------------------------------------- decay


def test_healthy_strategy_stays_healthy(memory_store, detector):
    make_trades(
        memory_store, "saine", [8.0, 10.0, -3.0, 12.0, 9.0, -2.0, 11.0, 7.0, 10.0, 6.0, 9.0, 8.0]
    )
    verdict = detector.check("saine")
    assert verdict.health is StrategyHealth.HEALTHY
    assert verdict.weight_multiplier == 1.0


def test_low_profit_factor_kills_strategy(memory_store, detector):
    """Profit factor sous 1 : la strategie perd de l'argent, elle est desactivee."""
    make_trades(
        memory_store,
        "mourante",
        [-10.0, -8.0, 3.0, -12.0, -9.0, 2.0, -11.0, -7.0, 1.0, -10.0, -6.0, -8.0],
        start_days_ago=13,
        span_days=12,
    )
    verdict = detector.check("mourante")
    assert verdict.health is StrategyHealth.DEAD
    assert verdict.weight_multiplier == 0.0
    assert verdict.needs_retraining


def test_consecutive_losses_kill_strategy(memory_store, detector):
    make_trades(
        memory_store,
        "serie_noire",
        [12.0, 15.0, 14.0] + [-2.0] * 10,
        start_days_ago=13,
        span_days=12,
    )
    verdict = detector.check("serie_noire")
    assert verdict.health is StrategyHealth.DEAD
    assert any("consecutives" in signal for signal in verdict.signals)


def test_insufficient_sample_yields_no_verdict(memory_store, detector):
    make_trades(memory_store, "jeune", [5.0, -2.0, 3.0])
    verdict = detector.check("jeune")
    assert verdict.health is StrategyHealth.HEALTHY
    assert any("insuffisant" in signal for signal in verdict.signals)


def test_dead_strategy_is_not_resurrected_too_early(memory_store, detector):
    """Un redressement de quelques jours ne rachete pas une strategie morte."""
    make_trades(memory_store, "morte", [-10.0] * 12, start_days_ago=13, span_days=12)
    assert detector.check("morte").health is StrategyHealth.DEAD

    memory_store.purge(["trades"])
    make_trades(memory_store, "morte", [10.0] * 12, start_days_ago=5, span_days=4)
    # Toujours morte : la periode d'observation n'est pas ecoulee.
    assert detector.check("morte", NOW + timedelta(days=2)).health is StrategyHealth.DEAD


def test_dead_strategy_passes_through_zombie_before_returning(memory_store, detector):
    make_trades(memory_store, "phenix", [-10.0] * 12, start_days_ago=13, span_days=12)
    assert detector.check("phenix").health is StrategyHealth.DEAD

    memory_store.purge(["trades"])
    make_trades(memory_store, "phenix", [10.0] * 12, start_days_ago=5, span_days=4)

    after_dead_period = NOW + timedelta(days=31)
    assert detector.check("phenix", after_dead_period).health is StrategyHealth.ZOMBIE

    still_testing = after_dead_period + timedelta(days=5)
    assert detector.check("phenix", still_testing).health is StrategyHealth.ZOMBIE

    fully_tested = after_dead_period + timedelta(days=15)
    assert detector.check("phenix", fully_tested).health is StrategyHealth.HEALTHY


def test_zombie_weight_is_zero(memory_store, detector):
    make_trades(memory_store, "z", [-10.0] * 12, start_days_ago=13, span_days=12)
    detector.check("z")
    memory_store.purge(["trades"])
    make_trades(memory_store, "z", [10.0] * 12, start_days_ago=5, span_days=4)
    verdict = detector.check("z", NOW + timedelta(days=31))
    assert verdict.health is StrategyHealth.ZOMBIE
    assert verdict.weight_multiplier == 0.0


def test_check_interval_is_respected(detector):
    assert detector.needs_check(NOW)
    detector.check_all(["s"], NOW)
    assert not detector.needs_check(NOW + timedelta(hours=1))
    assert detector.needs_check(NOW + timedelta(hours=detector.config.check_interval_hours + 1))


def test_health_provider_interface(memory_store, detector):
    make_trades(memory_store, "s", [-10.0] * 12, start_days_ago=13, span_days=12)
    assert detector.health_provider("s") is StrategyHealth.DEAD


def test_alpha_decay_is_detected(memory_store, detector):
    """Un P&L cumule qui plafonne signale un alpha qui s'eteint."""
    rng = np.random.default_rng(3)
    growing = list(rng.normal(8.0, 1.0, 15))
    flat = list(rng.normal(0.0, 1.0, 20))
    make_trades(memory_store, "plateau", growing + flat, start_days_ago=29, span_days=28)
    verdict = detector.check("plateau")
    assert "adf_pvalue" in verdict.metrics


def test_verdict_is_serializable(memory_store, detector):
    make_trades(memory_store, "s", [5.0, -2.0] * 6)
    payload = detector.check("s").to_dict()
    assert set(payload) >= {"strategy", "health", "signals", "metrics"}

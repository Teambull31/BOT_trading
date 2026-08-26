"""Fabrique du pool de strategies.

Point unique ou l'on decide quelles strategies composent l'ensemble. Chaque
regime doit etre couvert par AU MOINS DEUX strategies : la regle de consensus
exige deux votes, donc un regime couvert par une seule strategie ne trade jamais.
`coverage_report()` verifie cette propriete et sert de garde-fou au demarrage.
"""

from __future__ import annotations

from collections.abc import Sequence

from trader.logging_setup import get_logger
from trader.models import Regime
from trader.strategy.base import BaseStrategy
from trader.strategy.breakout import BreakoutStrategy
from trader.strategy.mean_revert import MeanRevertStrategy
from trader.strategy.momentum import MomentumStrategy
from trader.strategy.sentiment import SentimentStrategy

log = get_logger(__name__)


def build_default_pool() -> list[BaseStrategy]:
    """Construit le pool de strategies par defaut.

    La composition n'est pas arbitraire : la regle de consensus exige DEUX
    strategies actives par regime. Momentum et mean-reversion couvrent des
    regimes disjoints et ne peuvent donc jamais former un quorum a elles deux.
    Breakout (range + tendance) et sentiment (tous regimes tradables) comblent
    ce trou : chaque regime tradable est desormais couvert par au moins deux
    strategies aux logiques differentes.
    """
    return [
        MomentumStrategy(),
        MeanRevertStrategy(),
        BreakoutStrategy(),
        SentimentStrategy(),
    ]


def coverage_report(strategies: Sequence[BaseStrategy]) -> dict[str, list[str]]:
    """Liste, pour chaque regime, les strategies autorisees a y trader."""
    coverage: dict[str, list[str]] = {regime.value: [] for regime in Regime}
    for strategy in strategies:
        for regime in strategy.get_required_regimes():
            coverage.setdefault(regime, []).append(strategy.name)
    return coverage


def uncovered_regimes(strategies: Sequence[BaseStrategy], min_strategies: int = 2) -> list[str]:
    """Regimes tradables couverts par moins de `min_strategies` strategies.

    Un regime sous-couvert n'est pas une erreur bloquante : le systeme s'abstient
    simplement de trader dans ce regime. Mais il faut le savoir, sinon on cherche
    pendant des jours pourquoi le bot ne prend aucune position.
    """
    coverage = coverage_report(strategies)
    # La crise est volontairement sans couverture : aucune position n'y est
    # ouverte, par regle de risque. UNCERTAIN, en revanche, doit etre couvert :
    # sinon la regle "diviser l'exposition par deux en regime incertain" ne
    # s'applique jamais, faute de trade a reduire.
    return [
        regime
        for regime, names in coverage.items()
        if regime != Regime.CRISIS.value and len(names) < min_strategies
    ]


def log_coverage(strategies: Sequence[BaseStrategy], min_strategies: int = 2) -> None:
    """Trace la couverture du pool au demarrage."""
    coverage = coverage_report(strategies)
    gaps = uncovered_regimes(strategies, min_strategies)
    log.info(
        "strategy_pool_coverage",
        strategies=[strategy.name for strategy in strategies],
        coverage={regime: names for regime, names in coverage.items() if names},
    )
    if gaps:
        log.warning(
            "regimes_sans_quorum",
            regimes=gaps,
            min_strategies=min_strategies,
            consequence="aucun trade ne sera pris dans ces regimes",
        )

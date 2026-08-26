"""Module anti-biais de confirmation.

Le probleme qu'il resout : un systeme de trading qui accumule des indicateurs
finit toujours par trouver de quoi justifier ce qu'il a envie de faire. Chaque
strategie produit deja ses propres contra_evidence, mais elle reste juge et
partie. Le DevilAdvocate, lui, ne connait pas la these du trade et n'a qu'une
mission : chercher ce qui cloche.

Il est appele APRES l'ensemble et AVANT le risk manager, et son verdict est
contraignant :
    contra_score > abort_threshold  (0.7) -> trade ANNULE
    contra_score > reduce_threshold (0.4) -> taille reduite de moitie

Ce module ne peut pas etre desactive par configuration (verifie dans config.py
et re-verifie ici a la construction).

Chaque controle retourne un score de 0 (rien a signaler) a 1 (tres defavorable),
pondere selon sa gravite. Un controle qui manque de donnees retourne 0 et le dit :
on ne condamne pas un trade sur une absence d'information, mais on ne l'absout
pas non plus.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trader.config import DevilAdvocateConfig
from trader.logging_setup import get_logger
from trader.models import (
    ContraRecommendation,
    ContraReport,
    EnsembleDecision,
    MarketSnapshot,
    RegimeState,
    StrategyHealth,
)
from trader.utils.math_utils import EPSILON

log = get_logger(__name__)


@dataclass(slots=True)
class Check:
    """Un controle a charge : son score, son poids et ce qu'il a trouve."""

    name: str
    score: float
    weight: float
    message: str = ""

    @property
    def contributes(self) -> bool:
        """Vrai si le controle a effectivement releve quelque chose."""
        return self.score > EPSILON and bool(self.message)


HealthProvider = Callable[[str], StrategyHealth]
"""Fournit l'etat de sante d'une strategie (injecte par le decay detector)."""


class DevilAdvocate:
    """Cherche activement les preuves CONTRE un trade propose."""

    def __init__(
        self,
        config: DevilAdvocateConfig | None = None,
        health_provider: HealthProvider | None = None,
    ) -> None:
        self.config = config or DevilAdvocateConfig()
        if not self.config.enabled:
            raise ValueError(
                "le DevilAdvocate ne peut pas etre desactive : ce module est "
                "structurellement obligatoire"
            )
        self.health_provider = health_provider

    def review(
        self,
        decision: EnsembleDecision,
        data: MarketSnapshot,
        regime: RegimeState,
    ) -> ContraReport:
        """Audite une decision et produit un rapport a charge."""
        direction = decision.signal.direction
        if direction == 0:
            return ContraReport(
                contra_signals=[],
                contra_score=0.0,
                recommendation=ContraRecommendation.PROCEED,
                checks={},
            )

        checks = [
            self._regime_transition(regime),
            self._regime_uncertainty(regime),
            self._price_indicator_divergence(data, direction),
            self._volume_confirmation(data, direction),
            self._correlation_contagion(data),
            self._strategy_decay(decision),
            self._crowded_trade(data, direction),
            self._overextension(data, direction),
            self._internal_contra_evidence(decision),
            self._weak_consensus(decision),
            self._volatility_spike(data),
        ]

        total_weight = sum(check.weight for check in checks)
        weighted = sum(check.score * check.weight for check in checks)
        contra_score = float(np.clip(weighted / total_weight, 0.0, 1.0)) if total_weight else 0.0

        if contra_score > self.config.abort_threshold:
            recommendation = ContraRecommendation.ABORT
        elif contra_score > self.config.reduce_threshold:
            recommendation = ContraRecommendation.REDUCE_SIZE
        else:
            recommendation = ContraRecommendation.PROCEED

        signals = [check.message for check in checks if check.contributes]
        report = ContraReport(
            contra_signals=signals,
            contra_score=contra_score,
            recommendation=recommendation,
            checks={check.name: round(check.score, 3) for check in checks},
        )
        log_method = log.warning if recommendation is not ContraRecommendation.PROCEED else log.info
        log_method(
            "devil_advocate_review",
            asset=decision.asset,
            signal=decision.signal.name,
            **report.to_dict(),
        )
        return report

    # ------------------------------------------------------------ controles

    def _regime_transition(self, regime: RegimeState) -> Check:
        """Le regime est-il en train de changer sous nos pieds ?"""
        probability = regime.transition_probability
        score = float(np.clip((probability - 0.3) / 0.5, 0.0, 1.0))
        message = (
            f"regime instable : {probability:.0%} de probabilite de transition" if score > 0 else ""
        )
        return Check("regime_transition", score, weight=1.5, message=message)

    def _regime_uncertainty(self, regime: RegimeState) -> Check:
        """Le regime est-il seulement identifie ?"""
        if regime.is_uncertain:
            return Check(
                "regime_uncertainty",
                0.8,
                weight=1.2,
                message="regime non identifie : le systeme ne sait pas dans quel marche il opere",
            )
        score = float(np.clip((0.75 - regime.confidence) / 0.5, 0.0, 1.0))
        message = f"confiance faible dans le regime ({regime.confidence:.2f})" if score > 0 else ""
        return Check("regime_uncertainty", score, weight=1.2, message=message)

    def _price_indicator_divergence(self, data: MarketSnapshot, direction: int) -> Check:
        """Le prix et les oscillateurs racontent-ils la meme histoire ?"""
        prices = self._series(data, "close", 40)
        rsi_series = self._series(data, "rsi_14", 40)
        if len(prices) < 20 or len(rsi_series) < 20:
            return Check("divergence", 0.0, weight=1.3)

        lookback = min(15, len(prices) - 1)
        price_change = float(prices.iloc[-1] - prices.iloc[-lookback])
        rsi_change = float(rsi_series.iloc[-1] - rsi_series.iloc[-lookback])

        if direction > 0 and price_change > 0 and rsi_change < -2.0:
            return Check(
                "divergence",
                float(np.clip(abs(rsi_change) / 15.0, 0.3, 1.0)),
                weight=1.3,
                message=(
                    f"divergence baissiere : prix +{price_change:.2f} mais RSI {rsi_change:+.1f}"
                ),
            )
        if direction < 0 and price_change < 0 and rsi_change > 2.0:
            return Check(
                "divergence",
                float(np.clip(rsi_change / 15.0, 0.3, 1.0)),
                weight=1.3,
                message=(
                    f"divergence haussiere : prix {price_change:.2f} mais RSI {rsi_change:+.1f}"
                ),
            )
        return Check("divergence", 0.0, weight=1.3)

    def _volume_confirmation(self, data: MarketSnapshot, direction: int) -> Check:
        """Le volume valide-t-il le mouvement, ou monte-t-il dans le vide ?"""
        volume_ratio = data.feature("volume_ratio")
        if volume_ratio is None:
            return Check("volume", 0.0, weight=1.0)
        if volume_ratio < 0.7:
            return Check(
                "volume",
                float(np.clip((0.7 - volume_ratio) / 0.5, 0.2, 1.0)),
                weight=1.0,
                message=f"volume anemique (x{volume_ratio:.2f}) : mouvement non confirme",
            )
        obv_slope = data.feature("obv_slope")
        if obv_slope is not None and direction * obv_slope < 0:
            return Check(
                "volume",
                0.5,
                weight=1.0,
                message="OBV oppose au sens du trade : les flux ne suivent pas",
            )
        return Check("volume", 0.0, weight=1.0)

    def _correlation_contagion(self, data: MarketSnapshot) -> Check:
        """Une correlation anormalement elevee signale un risque de contagion."""
        correlation = data.feature("corr_benchmark")
        if correlation is None:
            return Check("contagion", 0.0, weight=0.8)
        if abs(correlation) > 0.9:
            return Check(
                "contagion",
                float(np.clip((abs(correlation) - 0.9) / 0.1, 0.3, 1.0)),
                weight=0.8,
                message=(
                    f"correlation au benchmark de {correlation:.2f} : "
                    "risque de contagion, aucune diversification"
                ),
            )
        return Check("contagion", 0.0, weight=0.8)

    def _strategy_decay(self, decision: EnsembleDecision) -> Check:
        """La strategie qui pese le plus dans cette decision est-elle en declin ?"""
        if not decision.weights or self.health_provider is None:
            return Check("strategy_decay", 0.0, weight=1.4)
        dominant = max(decision.weights.items(), key=lambda item: item[1])
        name, weight = dominant
        try:
            health = self.health_provider(name)
        except Exception as exc:  # noqa: BLE001 - un provider casse ne bloque pas l'audit
            log.error("health_provider_failed", strategy=name, error=str(exc))
            return Check("strategy_decay", 0.0, weight=1.4)

        if health is StrategyHealth.DEAD:
            return Check(
                "strategy_decay",
                1.0,
                weight=1.4,
                message=f"la strategie dominante ({name}, poids {weight:.0%}) est morte",
            )
        if health is StrategyHealth.DEGRADING:
            return Check(
                "strategy_decay",
                0.6,
                weight=1.4,
                message=(
                    f"la strategie dominante ({name}, poids {weight:.0%}) perd son edge (DEGRADING)"
                ),
            )
        return Check("strategy_decay", 0.0, weight=1.4)

    def _crowded_trade(self, data: MarketSnapshot, direction: int) -> Check:
        """Le funding rate revele-t-il un trade surpeuple ?

        Un funding tres positif signifie que tout le monde est long et paie pour
        le rester. Acheter la-dedans, c'est rejoindre la sortie la plus encombree.
        """
        funding_z = data.feature("funding_zscore")
        funding = data.feature("funding_rate")
        if funding_z is None and funding is None:
            return Check("crowded_trade", 0.0, weight=1.1)

        if funding_z is not None and abs(funding_z) > 1.5 and direction * funding_z > 0:
            return Check(
                "crowded_trade",
                float(np.clip((abs(funding_z) - 1.5) / 2.0, 0.3, 1.0)),
                weight=1.1,
                message=(
                    f"funding rate a {funding_z:+.1f} sigma dans le sens du trade : "
                    "positionnement surpeuple"
                ),
            )
        if funding is not None and direction * funding > 0.001:
            return Check(
                "crowded_trade",
                0.5,
                weight=1.1,
                message=f"funding {funding:+.4f} contre l'entree : cout de portage defavorable",
            )
        return Check("crowded_trade", 0.0, weight=1.1)

    def _overextension(self, data: MarketSnapshot, direction: int) -> Check:
        """Achete-t-on un sommet (ou vend-on un creux) ?"""
        zscore = data.feature("price_zscore")
        bb_position = data.feature("bb_position")
        messages: list[str] = []
        score = 0.0

        if zscore is not None and direction * zscore > 2.0:
            score = max(score, float(np.clip((abs(zscore) - 2.0) / 2.0, 0.3, 1.0)))
            messages.append(f"prix a {zscore:+.1f} sigma de sa moyenne")
        if bb_position is not None:
            if direction > 0 and bb_position > 0.95:
                score = max(score, 0.6)
                messages.append("prix colle a la bande haute de Bollinger")
            if direction < 0 and bb_position < 0.05:
                score = max(score, 0.6)
                messages.append("prix colle a la bande basse de Bollinger")

        message = f"surextension : {', '.join(messages)}" if messages else ""
        return Check("overextension", score, weight=1.2, message=message)

    def _internal_contra_evidence(self, decision: EnsembleDecision) -> Check:
        """Que disent les strategies elles-memes contre leur propre signal ?"""
        direction = decision.signal.direction
        contributing = [
            output
            for output in decision.contributions
            if output.is_actionable and output.signal.direction == direction
        ]
        if not contributing:
            return Check("self_doubt", 0.0, weight=1.0)

        counts = [len(output.contra_evidence) for output in contributing]
        average = float(np.mean(counts))
        score = float(np.clip((average - 1.0) / 4.0, 0.0, 1.0))
        message = (
            f"les strategies porteuses listent {average:.1f} contre-indications en moyenne"
            if score > 0
            else ""
        )
        return Check("self_doubt", score, weight=1.0, message=message)

    def _weak_consensus(self, decision: EnsembleDecision) -> Check:
        """Le consensus est-il solide, ou le trade tient-il a un fil ?"""
        score = float(np.clip((0.85 - decision.consensus) / 0.5, 0.0, 1.0))
        message = (
            f"consensus fragile ({decision.consensus:.2f}) et dispersion {decision.dispersion:.2f}"
            if score > 0.2
            else ""
        )
        return Check("weak_consensus", score, weight=0.9, message=message)

    def _volatility_spike(self, data: MarketSnapshot) -> Check:
        """La volatilite explose-t-elle au moment d'entrer ?"""
        vol_z = data.feature("vol_zscore")
        if vol_z is None:
            return Check("volatility", 0.0, weight=1.0)
        if vol_z > 2.0:
            return Check(
                "volatility",
                float(np.clip((vol_z - 2.0) / 2.0, 0.3, 1.0)),
                weight=1.0,
                message=f"volatilite a {vol_z:+.1f} sigma : conditions d'execution degradees",
            )
        return Check("volatility", 0.0, weight=1.0)

    @staticmethod
    def _series(data: MarketSnapshot, name: str, length: int) -> pd.Series:
        """Fin d'une serie de features du snapshot."""
        if data.features is None or name not in getattr(data.features, "columns", []):
            return pd.Series(dtype=float)
        return data.features[name].dropna().tail(length)

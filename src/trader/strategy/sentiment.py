"""Strategie de sentiment derivee des donnees de derives.

Elle ne lit ni Twitter ni les news : le sentiment exploitable en crypto est
DEJA dans les donnees de marche, sous une forme mesurable et non narrative :

- funding rate : qui paie qui pour tenir sa position (positionnement de la foule) ;
- open interest : combien de levier est engage ;
- ratio volume/OI : rotation des positions ;
- liquidations recentes : ou la foule s'est fait sortir.

La these est CONTRARIENNE : quand le funding atteint un extreme, la foule est
massivement d'un cote, et ce cote est fragile. Cette strategie est donc
naturellement decorrelee des autres, ce qui est precisement ce qu'on attend d'un
membre d'ensemble : elle apporte de l'information, pas une copie.

Elle est autorisee dans tous les regimes tradables : le positionnement extreme
est une information valable partout — mais sa conviction est reduite en tendance
forte, ou "surpeuple" peut simplement vouloir dire "tout le monde a raison".
"""

from __future__ import annotations

from dataclasses import dataclass

from trader.models import MarketSnapshot, RegimeState, Signal, StrategyOutput
from trader.strategy.base import BaseStrategy, StrategyParams


@dataclass(slots=True)
class SentimentParams(StrategyParams):
    """Parametres de la strategie de sentiment."""

    funding_zscore_threshold: float = 1.8
    extreme_funding_rate: float = 0.0005
    oi_change_threshold: float = 0.15
    stop_atr_multiple: float = 2.0
    target_atr_multiple: float = 3.0
    min_confidence: float = 0.30


class SentimentStrategy(BaseStrategy):
    """Prend le contrepied des positionnements extremes."""

    name = "sentiment"
    description = "Contrarien sur funding rate, open interest et liquidations"

    def __init__(self, params: SentimentParams | None = None) -> None:
        super().__init__(params or SentimentParams())

    def get_required_regimes(self) -> list[str]:
        """Le positionnement extreme est exploitable dans tout regime tradable.

        Y compris UNCERTAIN : le funding rate mesure ce que fait la foule, pas
        ce que fait le marche. Cette information reste lisible meme quand le
        detecteur de regime, lui, ne conclut pas — et c'est precisement dans ces
        moments-la qu'un positionnement massif est le plus fragile. La taille
        reste divisee par deux par le risk manager.
        """
        return [
            "bull_low_vol",
            "bull_high_vol",
            "bear_low_vol",
            "bear_high_vol",
            "range_bound",
            "uncertain",
        ]

    def param_space(self) -> dict[str, tuple[float, float]]:
        """Bornes de recherche pour le retraining."""
        return super().param_space() | {
            "funding_zscore_threshold": (1.0, 3.0),
            "extreme_funding_rate": (0.0001, 0.002),
            "oi_change_threshold": (0.05, 0.40),
        }

    def generate_signal(self, data: MarketSnapshot, regime: RegimeState) -> StrategyOutput:
        """Produit un signal contrarien sur positionnement extreme, ou NEUTRAL."""
        params: SentimentParams = self.params  # type: ignore[assignment]
        funding_z = self.feature(data, "funding_zscore")
        funding = self.feature(data, "funding_rate")
        oi_change = self.feature(data, "oi_change")

        if funding_z is None and funding is None:
            return self.neutral(
                data.asset,
                "aucune donnee de derives (funding/OI) : cette strategie n'a rien a dire",
            )

        crowded_long = (funding_z is not None and funding_z >= params.funding_zscore_threshold) or (
            funding is not None and funding >= params.extreme_funding_rate
        )
        crowded_short = (
            funding_z is not None and funding_z <= -params.funding_zscore_threshold
        ) or (funding is not None and funding <= -params.extreme_funding_rate)

        if not (crowded_long or crowded_short):
            observed = funding_z if funding_z is not None else 0.0
            return self.neutral(data.asset, f"positionnement sans exces (funding z={observed:.2f})")

        # Contrarien : si la foule est longue, on regarde a la baisse.
        direction = -1 if crowded_long else 1
        confirmations: list[str] = []
        score = 0.35

        if funding_z is not None:
            confirmations.append(
                f"funding a {funding_z:+.2f} sigma : "
                f"{'longs' if crowded_long else 'shorts'} surpeuples"
            )
            score += min(0.25, abs(funding_z) / 10.0)
        if funding is not None:
            confirmations.append(f"funding rate {funding:+.5f} paye par la foule")

        if oi_change is not None and oi_change > params.oi_change_threshold:
            score += 0.20
            confirmations.append(f"open interest +{oi_change:.1%} : levier en forte accumulation")

        rsi_value = self.feature(data, "rsi_14")
        if rsi_value is not None:
            if direction < 0 and rsi_value > 70:
                score += 0.15
                confirmations.append(f"RSI {rsi_value:.0f} confirme l'exces haussier")
            if direction > 0 and rsi_value < 30:
                score += 0.15
                confirmations.append(f"RSI {rsi_value:.0f} confirme l'exces baissier")

        contra = self._contra_evidence(data, regime, direction)
        specific_contra = len(contra)
        contra = contra or self.residual_risk(data, direction)
        score = max(0.0, score - 0.15 * specific_contra)
        if score < params.min_confidence:
            return self.neutral(
                data.asset,
                f"exces de positionnement reel mais {specific_contra} contre-indications "
                f"(score {score:.2f})",
            )

        stop, target = self.atr_levels(data, direction)
        if stop is None:
            return self.neutral(data.asset, "ATR indisponible : impossible de placer un stop")

        signal = Signal.from_score(direction * 1.0)  # jamais STRONG : c'est un contrepied
        sense = "haussier" if direction > 0 else "baissier"
        return self.build_output(
            data=data,
            signal=signal,
            confidence=score,
            stop_loss=stop,
            target_price=target,
            reasoning=f"Contrepied {sense} sur positionnement extreme : "
            + " ; ".join(confirmations),
            contra_evidence=contra,
            metadata={"funding_zscore": funding_z, "funding_rate": funding, "score": score},
        )

    def _contra_evidence(
        self, data: MarketSnapshot, regime: RegimeState, direction: int
    ) -> list[str]:
        """Cherche activement ce qui contredit le contrepied."""
        contra: list[str] = []

        # Le risque majeur du contrarien : avoir raison trop tot dans une tendance.
        adx = self.feature(data, "adx")
        di_spread = self.feature(data, "di_spread")
        if adx is not None and adx > 30 and di_spread is not None and direction * di_spread < 0:
            contra.append(
                f"tendance forte a contre-sens (ADX {adx:.0f}) : "
                "un positionnement surpeuple peut le rester longtemps"
            )
        if regime.regime.value in ("bull_high_vol", "bear_high_vol"):
            contra.append(
                "regime directionnel volatil : le contrepied y est particulierement dangereux"
            )

        macd_hist = self.feature(data, "macd_hist_norm")
        if macd_hist is not None and direction * macd_hist < 0:
            contra.append("dynamique de prix encore opposee au contrepied")

        volume_ratio = self.feature(data, "volume_ratio")
        if volume_ratio is not None and volume_ratio > 2.5:
            contra.append(
                f"volume x{volume_ratio:.1f} : flux massif dans le sens de la foule, "
                "l'exces peut s'amplifier"
            )

        vol_z = self.feature(data, "vol_zscore")
        if vol_z is not None and vol_z > 2.0:
            contra.append(f"volatilite a {vol_z:+.1f} sigma : stops facilement balayes")

        if regime.transition_probability > 0.6:
            contra.append(
                f"regime instable ({regime.transition_probability:.0%}) : "
                "le positionnement peut se retourner sans prevenir"
            )

        hurst = self.feature(data, "hurst")
        if hurst is not None and hurst > 0.6:
            contra.append(f"Hurst {hurst:.2f} : serie persistante, defavorable a un contrepied")

        return contra

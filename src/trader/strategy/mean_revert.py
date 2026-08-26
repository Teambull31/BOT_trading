"""Strategie de retour a la moyenne.

Hypothese inverse du momentum : dans un marche sans direction (Hurst < 0.5, ADX
faible), les ecarts a la moyenne se referment. Elle ne trade donc QUE en range,
et jamais en tendance, ou elle se ferait laminer a contre-sens.

Garde-fou specifique : on ne prend une extension que si elle n'est pas un debut
de cassure. Le pire scenario d'une mean-reversion est d'acheter le premier tiers
d'un effondrement.
"""

from __future__ import annotations

from dataclasses import dataclass

from trader.models import MarketSnapshot, RegimeState, Signal, StrategyOutput
from trader.strategy.base import BaseStrategy, StrategyParams


@dataclass(slots=True)
class MeanRevertParams(StrategyParams):
    """Parametres de la strategie mean-reversion."""

    entry_zscore: float = 1.8
    exit_zscore: float = 0.3
    rsi_low: float = 30.0
    rsi_high: float = 70.0
    max_adx: float = 25.0
    max_bb_position: float = 0.05
    stop_atr_multiple: float = 1.5
    target_atr_multiple: float = 2.0


class MeanRevertStrategy(BaseStrategy):
    """Achete les exces baissiers et vend les exces haussiers, en range uniquement."""

    name = "mean_revert"
    description = "Retour a la moyenne sur exces de Bollinger / RSI / z-score"

    def __init__(self, params: MeanRevertParams | None = None) -> None:
        super().__init__(params or MeanRevertParams())

    def get_required_regimes(self) -> list[str]:
        """La mean-reversion ne trade qu'en marche sans direction etablie."""
        return ["range_bound"]

    def param_space(self) -> dict[str, tuple[float, float]]:
        """Bornes de recherche pour le retraining."""
        return super().param_space() | {
            "entry_zscore": (1.0, 3.0),
            "exit_zscore": (0.0, 1.0),
            "rsi_low": (15.0, 40.0),
            "rsi_high": (60.0, 85.0),
            "max_adx": (15.0, 30.0),
            "max_bb_position": (0.0, 0.25),
        }

    def generate_signal(self, data: MarketSnapshot, regime: RegimeState) -> StrategyOutput:
        """Produit un signal de retour a la moyenne, ou NEUTRAL."""
        params: MeanRevertParams = self.params  # type: ignore[assignment]
        zscore = self.feature(data, "price_zscore")
        bb_position = self.feature(data, "bb_position")
        rsi_value = self.feature(data, "rsi_14")
        adx = self.feature(data, "adx")
        hurst = self.feature(data, "hurst")

        if zscore is None or bb_position is None or rsi_value is None:
            return self.neutral(data.asset, "features de mean-reversion indisponibles")
        if adx is not None and adx > params.max_adx:
            return self.neutral(
                data.asset,
                f"marche trop directionnel pour une mean-reversion (ADX {adx:.1f})",
            )

        stretched_down = zscore <= -params.entry_zscore or bb_position <= params.max_bb_position
        stretched_up = zscore >= params.entry_zscore or bb_position >= 1.0 - params.max_bb_position
        if not (stretched_down or stretched_up):
            return self.neutral(data.asset, f"pas d'exces exploitable (z-score {zscore:+.2f})")

        direction = 1 if stretched_down else -1
        confirmations: list[str] = []
        score = 0.0

        if abs(zscore) >= params.entry_zscore:
            score += 0.35
            confirmations.append(f"prix a {zscore:+.2f} sigma de sa moyenne")
        if direction > 0 and bb_position <= params.max_bb_position:
            score += 0.25
            confirmations.append(f"prix colle a la bande basse de Bollinger ({bb_position:.2f})")
        if direction < 0 and bb_position >= 1.0 - params.max_bb_position:
            score += 0.25
            confirmations.append(f"prix colle a la bande haute de Bollinger ({bb_position:.2f})")
        if direction > 0 and rsi_value <= params.rsi_low:
            score += 0.20
            confirmations.append(f"RSI {rsi_value:.0f} en survente")
        if direction < 0 and rsi_value >= params.rsi_high:
            score += 0.20
            confirmations.append(f"RSI {rsi_value:.0f} en surachat")
        if hurst is not None and hurst < 0.5:
            score += 0.15
            confirmations.append(f"Hurst {hurst:.2f} < 0.5 : serie anti-persistante")

        contra = self._contra_evidence(data, regime, direction, adx, hurst)
        specific_contra = len(contra)
        contra = contra or self.residual_risk(data, direction)
        score = max(0.0, score - 0.15 * specific_contra)
        if score < params.min_confidence:
            return self.neutral(
                data.asset,
                f"exces reel mais {specific_contra} contre-indications (score {score:.2f})",
            )

        stop, _ = self.atr_levels(data, direction)
        if stop is None:
            return self.neutral(data.asset, "ATR indisponible : impossible de placer un stop")
        target = self._mean_target(data, direction)

        signal = Signal.from_score(direction * (2.0 if score >= 0.75 else 1.0))
        reasoning = (
            f"Retour a la moyenne {'haussier' if direction > 0 else 'baissier'} : "
            + " ; ".join(confirmations)
        )
        return self.build_output(
            data=data,
            signal=signal,
            confidence=score,
            stop_loss=stop,
            target_price=target,
            reasoning=reasoning,
            contra_evidence=contra,
            metadata={"zscore": zscore, "bb_position": bb_position, "score": score},
        )

    def _mean_target(self, data: MarketSnapshot, direction: int) -> float | None:
        """Cible = retour vers la moyenne mobile (bande mediane de Bollinger)."""
        middle = self.feature(data, "bb_middle")
        if middle is None or middle <= 0:
            _, target = self.atr_levels(data, direction)
            return target
        return float(middle)

    def _contra_evidence(
        self,
        data: MarketSnapshot,
        regime: RegimeState,
        direction: int,
        adx: float | None,
        hurst: float | None,
    ) -> list[str]:
        """Cherche activement ce qui contredit le pari de retour a la moyenne."""
        contra: list[str] = []

        if adx is not None and adx > 20.0:
            contra.append(f"ADX {adx:.1f} : une tendance est peut-etre en train de naitre")
        if hurst is not None and hurst > 0.55:
            contra.append(f"Hurst {hurst:.2f} > 0.5 : serie persistante, l'exces peut durer")

        volume_ratio = self.feature(data, "volume_ratio")
        if volume_ratio is not None and volume_ratio > 2.0:
            contra.append(
                f"volume x{volume_ratio:.1f} : mouvement soutenu, probable cassure plutot qu'exces"
            )

        bb_width = self.series(data, "bb_width", 30)
        if len(bb_width) >= 10 and float(bb_width.iloc[-1]) > float(bb_width.iloc[-10]) * 1.5:
            contra.append("expansion des bandes de Bollinger : regime de volatilite en hausse")

        atr_pct = self.feature(data, "atr_pct")
        if atr_pct is not None and atr_pct > 5.0:
            contra.append(f"ATR {atr_pct:.1f} % du prix : volatilite trop elevee pour un fade")

        if regime.transition_probability > 0.5:
            contra.append(
                f"regime instable (transition {regime.transition_probability:.0%}) : "
                "le range peut se rompre"
            )

        macd_hist = self.feature(data, "macd_hist_norm")
        if macd_hist is not None and direction * macd_hist < 0:
            contra.append("MACD oppose au trade : la dynamique court terme reste contraire")

        returns = self.series(data, "log_return", 5)
        if len(returns) >= 3 and all(direction * value < 0 for value in returns.iloc[-3:]):
            contra.append("trois bougies consecutives contre le trade : couteau qui tombe")

        return contra

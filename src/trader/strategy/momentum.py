"""Strategie momentum / trend-following.

Hypothese : dans un marche directionnel, les mouvements se prolongent (Hurst > 0.5).
Elle n'a donc le droit de trader QUE dans les regimes de tendance ; en range, la
meme logique produit du bruit et des pertes par aller-retours.

Anti-biais integre : la strategie cherche systematiquement ce qui contredit son
signal (divergence RSI, volume qui ne suit pas, surextension, tendance mure).
"""

from __future__ import annotations

from dataclasses import dataclass

from trader.models import MarketSnapshot, RegimeState, Signal, StrategyOutput
from trader.strategy.base import BaseStrategy, StrategyParams


@dataclass(slots=True)
class MomentumParams(StrategyParams):
    """Parametres de la strategie momentum."""

    adx_threshold: float = 22.0
    rsi_overbought: float = 78.0
    rsi_oversold: float = 22.0
    min_trend_slope: float = 0.001
    volume_confirmation: float = 0.8


class MomentumStrategy(BaseStrategy):
    """Suit la tendance etablie, confirmee par plusieurs signaux independants."""

    name = "momentum"
    description = "Trend-following multi-confirmation (MACD, ADX/DI, VWAP, pente)"

    def __init__(self, params: MomentumParams | None = None) -> None:
        super().__init__(params or MomentumParams())

    def get_required_regimes(self) -> list[str]:
        """Le momentum ne trade que dans les marches directionnels."""
        return ["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"]

    def param_space(self) -> dict[str, tuple[float, float]]:
        """Bornes de recherche pour le retraining."""
        return super().param_space() | {
            "adx_threshold": (15.0, 35.0),
            "rsi_overbought": (65.0, 85.0),
            "rsi_oversold": (15.0, 35.0),
            "min_trend_slope": (0.0, 0.01),
            "volume_confirmation": (0.5, 1.5),
        }

    def generate_signal(self, data: MarketSnapshot, regime: RegimeState) -> StrategyOutput:
        """Produit un signal de suivi de tendance, ou NEUTRAL si rien n'est net."""
        params: MomentumParams = self.params  # type: ignore[assignment]
        adx = self.feature(data, "adx")
        di_spread = self.feature(data, "di_spread")
        macd_hist = self.feature(data, "macd_hist_norm")
        rsi_value = self.feature(data, "rsi_14")
        slope = self.feature(data, "trend_slope")
        vwap_distance = self.feature(data, "vwap_distance_pct")
        volume_ratio = self.feature(data, "volume_ratio")

        if adx is None or di_spread is None or macd_hist is None:
            return self.neutral(data.asset, "features de tendance indisponibles")
        if adx < params.adx_threshold:
            return self.neutral(
                data.asset, f"tendance trop faible (ADX {adx:.1f} < {params.adx_threshold:.1f})"
            )

        direction = 1 if di_spread > 0 else -1
        confirmations: list[str] = [f"ADX {adx:.1f} confirme une tendance etablie"]
        score = 0.0

        if direction * di_spread > 0:
            score += 0.30
            confirmations.append(
                f"DI spread {di_spread:+.1f} oriente {'haut' if direction > 0 else 'bas'}"
            )
        if macd_hist is not None and direction * macd_hist > 0:
            score += 0.25
            confirmations.append(f"histogramme MACD {macd_hist:+.4f} dans le sens de la tendance")
        if slope is not None and direction * slope > params.min_trend_slope:
            score += 0.20
            confirmations.append(f"pente de prix {slope:+.4f} confirme")
        if vwap_distance is not None and direction * vwap_distance > 0:
            score += 0.15
            confirmations.append(f"prix du bon cote du VWAP ({vwap_distance:+.2f} %)")
        if volume_ratio is not None and volume_ratio >= params.volume_confirmation:
            score += 0.10
            confirmations.append(f"volume soutenu (x{volume_ratio:.2f} la moyenne)")

        contra = self._contra_evidence(data, regime, direction, rsi_value, volume_ratio, adx)
        specific_contra = len(contra)
        contra = contra or self.residual_risk(data, direction)
        # Chaque preuve contraire ampute la conviction : le doute a un cout.
        score = max(0.0, score - 0.12 * specific_contra)
        if score < params.min_confidence:
            return self.neutral(
                data.asset,
                f"momentum insuffisant apres {specific_contra} contre-indications "
                f"(score {score:.2f})",
            )

        stop, target = self.atr_levels(data, direction)
        if stop is None:
            return self.neutral(data.asset, "ATR indisponible : impossible de placer un stop")

        signal = Signal.from_score(direction * (2.0 if score >= 0.75 else 1.0))
        reasoning = (
            "Momentum "
            + ("haussier" if direction > 0 else "baissier")
            + " : "
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
            metadata={"adx": adx, "di_spread": di_spread, "score": score},
        )

    def _contra_evidence(
        self,
        data: MarketSnapshot,
        regime: RegimeState,
        direction: int,
        rsi_value: float | None,
        volume_ratio: float | None,
        adx: float,
    ) -> list[str]:
        """Cherche activement ce qui contredit le signal de momentum."""
        params: MomentumParams = self.params  # type: ignore[assignment]
        contra: list[str] = []

        if rsi_value is not None:
            if direction > 0 and rsi_value > params.rsi_overbought:
                contra.append(f"RSI {rsi_value:.0f} en surachat : entree tardive probable")
            if direction < 0 and rsi_value < params.rsi_oversold:
                contra.append(f"RSI {rsi_value:.0f} en survente : rebond technique probable")

        if volume_ratio is not None and volume_ratio < params.volume_confirmation:
            contra.append(f"volume {volume_ratio:.2f}x seulement : le mouvement n'est pas confirme")

        divergence = self._rsi_divergence(data, direction)
        if divergence:
            contra.append(divergence)

        if regime.transition_probability > 0.5:
            contra.append(
                f"probabilite de changement de regime elevee ({regime.transition_probability:.0%})"
            )

        adx_series = self.series(data, "adx", 20)
        if len(adx_series) >= 10 and adx_series.iloc[-1] < adx_series.iloc[-10]:
            contra.append("ADX en decroissance : la tendance perd de la force")

        if adx > 45:
            contra.append(f"ADX {adx:.0f} extreme : tendance mure, risque d'epuisement")

        extension = self.feature(data, "price_zscore")
        if extension is not None and direction * extension > 2.0:
            contra.append(f"prix a {extension:+.1f} sigma de sa moyenne : surextension")

        return contra

    def _rsi_divergence(self, data: MarketSnapshot, direction: int) -> str | None:
        """Detecte une divergence prix / RSI, signal classique d'essoufflement."""
        prices = self.series(data, "close", 30)
        rsi_series = self.series(data, "rsi_14", 30)
        if len(prices) < 20 or len(rsi_series) < 20:
            return None
        price_change = float(prices.iloc[-1] - prices.iloc[-15])
        rsi_change = float(rsi_series.iloc[-1] - rsi_series.iloc[-15])
        if direction > 0 and price_change > 0 and rsi_change < -3.0:
            return "divergence baissiere : le prix monte mais le RSI baisse"
        if direction < 0 and price_change < 0 and rsi_change > 3.0:
            return "divergence haussiere : le prix baisse mais le RSI monte"
        return None

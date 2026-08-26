"""Strategie de cassure de range.

Elle occupe la zone que momentum et mean-reversion laissent vide : le moment ou
un range se rompt. Elle trade donc en range (cassure anticipee) ET en tendance
(continuation apres consolidation).

Le piege connu de cette approche est la FAUSSE CASSURE : le prix depasse la
borne, declenche les stops, puis revient. Trois garde-fous :
- exiger une confirmation par le volume, une cassure sans volume est suspecte ;
- exiger une compression prealable (bandes de Bollinger serrees), car une
  cassure n'a de sens qu'apres une accumulation ;
- exiger que la bougie CLOTURE au-dela de la borne, pas seulement qu'elle la
  touche en seance.
"""

from __future__ import annotations

from dataclasses import dataclass

from trader.models import MarketSnapshot, RegimeState, Signal, StrategyOutput
from trader.strategy.base import BaseStrategy, StrategyParams


@dataclass(slots=True)
class BreakoutParams(StrategyParams):
    """Parametres de la strategie breakout."""

    lookback: int = 48
    min_volume_ratio: float = 1.3
    max_squeeze_percentile: float = 0.4
    breakout_buffer_pct: float = 0.15
    stop_atr_multiple: float = 1.8
    target_atr_multiple: float = 4.0


class BreakoutStrategy(BaseStrategy):
    """Achete les cassures de resistance et vend les cassures de support."""

    name = "breakout"
    description = "Cassure de range confirmee par le volume et la compression prealable"

    def __init__(self, params: BreakoutParams | None = None) -> None:
        super().__init__(params or BreakoutParams())

    def get_required_regimes(self) -> list[str]:
        """Cassure : pertinente en range (rupture), en tendance (continuation) et en
        regime incertain.

        Le regime UNCERTAIN est inclus deliberement : la these de cette strategie
        ne repose pas sur l'identification du regime mais sur une structure de
        prix observable (bornes de range, cassure confirmee par le volume). Elle
        s'auto-valide donc sans avoir besoin de savoir dans quel marche on est.
        Le risk manager divise de toute facon la taille par deux dans ce regime.
        """
        return [
            "range_bound",
            "bull_low_vol",
            "bull_high_vol",
            "bear_low_vol",
            "bear_high_vol",
            "uncertain",
        ]

    def param_space(self) -> dict[str, tuple[float, float]]:
        """Bornes de recherche pour le retraining."""
        return super().param_space() | {
            "lookback": (20.0, 100.0),
            "min_volume_ratio": (1.0, 2.5),
            "max_squeeze_percentile": (0.2, 0.7),
            "breakout_buffer_pct": (0.0, 0.5),
        }

    def generate_signal(self, data: MarketSnapshot, regime: RegimeState) -> StrategyOutput:
        """Produit un signal de cassure, ou NEUTRAL."""
        params: BreakoutParams = self.params  # type: ignore[assignment]
        lookback = int(params.lookback)
        ohlcv = data.ohlcv
        if ohlcv is None or len(ohlcv) < lookback + 5:
            return self.neutral(data.asset, "historique insuffisant pour delimiter un range")

        # Le range se calcule sur les bougies PRECEDENTES : inclure la bougie
        # courante rendrait toute cassure impossible par construction.
        window = ohlcv.iloc[-(lookback + 1) : -1]
        resistance = float(window["high"].max())
        support = float(window["low"].min())
        close = float(ohlcv["close"].iloc[-1])
        buffer_ratio = params.breakout_buffer_pct / 100.0

        broke_up = close > resistance * (1.0 + buffer_ratio)
        broke_down = close < support * (1.0 - buffer_ratio)
        if not (broke_up or broke_down):
            return self.neutral(
                data.asset,
                f"prix dans le range [{support:.2f}, {resistance:.2f}] : aucune cassure",
            )

        direction = 1 if broke_up else -1
        volume_ratio = self.feature(data, "volume_ratio") or 0.0
        squeeze_percentile = self._squeeze_percentile(data)
        confirmations: list[str] = [
            f"cloture a {close:.2f} au-dela de "
            f"{'la resistance' if direction > 0 else 'du support'} "
            f"({resistance if direction > 0 else support:.2f})"
        ]
        score = 0.35

        if volume_ratio >= params.min_volume_ratio:
            score += 0.30
            confirmations.append(f"volume x{volume_ratio:.2f} confirme la cassure")
        if squeeze_percentile is not None and squeeze_percentile <= params.max_squeeze_percentile:
            score += 0.20
            confirmations.append(
                f"compression prealable des bandes (percentile {squeeze_percentile:.2f})"
            )
        adx = self.feature(data, "adx")
        if adx is not None and adx > 20:
            score += 0.15
            confirmations.append(f"ADX {adx:.1f} : la cassure s'accompagne de directionnalite")

        contra = self._contra_evidence(
            data, regime, direction, volume_ratio, squeeze_percentile, resistance, support
        )
        specific_contra = len(contra)
        contra = contra or self.residual_risk(data, direction)
        score = max(0.0, score - 0.15 * specific_contra)
        if score < params.min_confidence:
            return self.neutral(
                data.asset,
                f"cassure detectee mais {specific_contra} contre-indications (score {score:.2f})",
            )

        stop, target = self.atr_levels(data, direction)
        if stop is None:
            return self.neutral(data.asset, "ATR indisponible : impossible de placer un stop")
        # Le stop se place de l'autre cote de la borne cassee : si le prix y
        # revient, la cassure etait fausse et la these est morte.
        boundary = resistance if direction > 0 else support
        stop = min(stop, boundary * 0.999) if direction > 0 else max(stop, boundary * 1.001)

        signal = Signal.from_score(direction * (2.0 if score >= 0.75 else 1.0))
        sense = "haussiere" if direction > 0 else "baissiere"
        return self.build_output(
            data=data,
            signal=signal,
            confidence=score,
            stop_loss=stop,
            target_price=target,
            reasoning=f"Cassure {sense} : " + " ; ".join(confirmations),
            contra_evidence=contra,
            metadata={
                "resistance": resistance,
                "support": support,
                "volume_ratio": volume_ratio,
                "score": score,
            },
        )

    def _squeeze_percentile(self, data: MarketSnapshot) -> float | None:
        """Percentile de la largeur de bandes : bas = compression prealable."""
        widths = self.series(data, "bb_width", 200)
        if len(widths) < 20:
            return None
        current = float(widths.iloc[-1])
        return float((widths <= current).mean())

    def _contra_evidence(
        self,
        data: MarketSnapshot,
        regime: RegimeState,
        direction: int,
        volume_ratio: float,
        squeeze_percentile: float | None,
        resistance: float,
        support: float,
    ) -> list[str]:
        """Cherche activement les signes d'une fausse cassure."""
        params: BreakoutParams = self.params  # type: ignore[assignment]
        contra: list[str] = []

        if volume_ratio < params.min_volume_ratio:
            contra.append(
                f"volume x{volume_ratio:.2f} sous le seuil {params.min_volume_ratio:.2f} : "
                "cassure non confirmee, risque de faux signal"
            )
        if squeeze_percentile is not None and squeeze_percentile > 0.8:
            contra.append(
                "aucune compression prealable : le marche etait deja etendu, "
                "la cassure peut etre un epuisement"
            )

        # Cassure a repetition : un range casse plusieurs fois dans les deux sens
        # est un range bruite, pas une rupture.
        closes = data.ohlcv["close"].tail(20) if data.ohlcv is not None else None
        if closes is not None and len(closes) >= 10:
            breaks = int((closes > resistance).sum() + (closes < support).sum())
            if breaks > 3:
                contra.append(
                    f"{breaks} incursions hors du range sur 20 bougies : bornes peu fiables"
                )

        atr_pct = self.feature(data, "atr_pct")
        if atr_pct is not None and atr_pct > 6.0:
            contra.append(f"ATR a {atr_pct:.1f} % du prix : le stop sera large, le ratio degrade")

        rsi_value = self.feature(data, "rsi_14")
        if rsi_value is not None:
            if direction > 0 and rsi_value > 80:
                contra.append(f"RSI {rsi_value:.0f} : cassure deja tres etiree")
            if direction < 0 and rsi_value < 20:
                contra.append(f"RSI {rsi_value:.0f} : cassure deja tres etiree")

        if regime.transition_probability > 0.5:
            contra.append(
                f"regime instable ({regime.transition_probability:.0%}) : "
                "les bornes du range perdent leur sens"
            )

        range_width = (resistance - support) / support * 100.0 if support > 0 else 0.0
        if range_width < 1.0:
            contra.append(
                f"range de {range_width:.2f} % seulement : cassure probablement insignifiante"
            )

        book_imbalance = self.feature(data, "book_imbalance")
        if book_imbalance is not None and direction * book_imbalance < -0.3:
            contra.append(f"carnet desequilibre a contre-sens ({book_imbalance:+.2f})")

        return contra

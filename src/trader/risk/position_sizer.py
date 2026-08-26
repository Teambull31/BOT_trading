"""Dimensionnement des positions.

Methode : Kelly fractionnaire (half-Kelly par defaut), plafonne par les limites
en dur. Kelly plein est mathematiquement optimal pour maximiser la croissance a
long terme MAIS suppose que l'on connait exactement ses probabilites. On ne les
connait pas : on les estime sur un echantillon fini et bruite. Surestimer son
edge de 20 % avec Kelly plein suffit a transformer une strategie gagnante en
ruine. On divise donc systematiquement par deux, et le resultat reste soumis au
plafond de 2 % du capital par position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trader.config import HARD_MAX_POSITION_PCT, RiskConfig
from trader.logging_setup import get_logger
from trader.models import RegimeState
from trader.utils.math_utils import EPSILON, clamp, kelly_fraction

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SizingResult:
    """Resultat du dimensionnement d'une position."""

    notional: float
    size: float
    fraction_of_equity: float
    method: str
    reasons: list[str]

    @property
    def is_tradable(self) -> bool:
        """Vrai si la taille calculee permet reellement de trader."""
        return self.size > 0 and self.notional > 0


class PositionSizer:
    """Calcule la taille d'une position selon le risque, pas selon la conviction."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        regime: RegimeState,
        win_rate: float | None = None,
        win_loss_ratio: float | None = None,
        current_exposure_pct: float = 0.0,
        min_notional: float = 0.0,
    ) -> SizingResult:
        """Determine le notionnel a engager sur un trade.

        Args:
            equity: capital total courant.
            entry_price: prix d'entree envisage.
            stop_loss: stop loss du trade (obligatoire, definit le risque).
            confidence: conviction de l'ensemble, dans [0, 1].
            regime: regime courant (module la taille).
            win_rate: taux de reussite historique, pour Kelly.
            win_loss_ratio: ratio gain moyen / perte moyenne, pour Kelly.
            current_exposure_pct: exposition deja engagee, en % de l'equity.
            min_notional: notionnel minimal impose par l'exchange.
        """
        reasons: list[str] = []
        if equity <= 0 or entry_price <= 0:
            return SizingResult(0.0, 0.0, 0.0, "invalid", ["equity ou prix invalide"])

        stop_distance = abs(entry_price - stop_loss) / entry_price
        if stop_distance < EPSILON:
            return SizingResult(0.0, 0.0, 0.0, "invalid", ["stop loss confondu avec l'entree"])

        # 1. Fraction de base : Kelly fractionnaire si l'historique le permet,
        #    sinon fixed-fractional prudent module par la confiance.
        if win_rate is not None and win_loss_ratio is not None and win_loss_ratio > 0:
            full_kelly = kelly_fraction(win_rate, win_loss_ratio)
            fraction = full_kelly * self.config.kelly_fraction
            method = f"kelly_{self.config.kelly_fraction:.2f}"
            reasons.append(
                f"Kelly plein {full_kelly:.3f} reduit a {fraction:.3f} "
                f"(facteur {self.config.kelly_fraction:.2f})"
            )
        else:
            fraction = self.config.max_position_pct / 100.0 * clamp(confidence, 0.0, 1.0)
            method = "fixed_fractional"
            reasons.append(
                f"pas d'historique exploitable : fixed-fractional module par la "
                f"confiance ({confidence:.2f})"
            )

        # 2. Modulation par le regime.
        if regime.is_crisis:
            return SizingResult(
                0.0, 0.0, 0.0, method, [*reasons, "regime de crise : aucune nouvelle position"]
            )
        if regime.is_uncertain:
            multiplier = self.config.uncertain_regime.exposure_multiplier
            fraction *= multiplier
            reasons.append(f"regime incertain : taille multipliee par {multiplier:.2f}")

        # 3. Plafonds : limite par position, puis exposition totale restante.
        cap = min(self.config.max_position_pct, HARD_MAX_POSITION_PCT) / 100.0
        if fraction > cap:
            reasons.append(f"fraction {fraction:.3f} ramenee au plafond par position {cap:.3f}")
            fraction = cap

        remaining = max(0.0, self.config.max_exposure_pct - current_exposure_pct) / 100.0
        if remaining <= EPSILON:
            return SizingResult(
                0.0,
                0.0,
                0.0,
                method,
                [*reasons, f"exposition totale deja a {current_exposure_pct:.1f} %"],
            )
        if fraction > remaining:
            reasons.append(f"fraction limitee par l'exposition totale restante ({remaining:.3f})")
            fraction = remaining

        notional = equity * fraction
        if min_notional > 0 and notional < min_notional:
            return SizingResult(
                0.0,
                0.0,
                0.0,
                method,
                [
                    *reasons,
                    f"notionnel {notional:.2f} sous le minimum de l'exchange ({min_notional:.2f})",
                ],
            )

        size = notional / entry_price
        risk_amount = notional * stop_distance
        reasons.append(
            f"risque en cas de stop : {risk_amount:.2f} "
            f"({risk_amount / equity * 100.0:.2f} % du capital)"
        )
        return SizingResult(
            notional=float(notional),
            size=float(size),
            fraction_of_equity=float(fraction),
            method=method,
            reasons=reasons,
        )

    def risk_based_size(
        self, equity: float, entry_price: float, stop_loss: float, risk_pct: float
    ) -> float:
        """Taille telle que toucher le stop coute exactement `risk_pct` % du capital.

        C'est la lecture la plus honnete du dimensionnement : on ne raisonne pas
        en "combien j'engage" mais en "combien je perds si j'ai tort".
        """
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance < EPSILON or equity <= 0 or entry_price <= 0:
            return 0.0
        risk_amount = equity * risk_pct / 100.0
        size = risk_amount / stop_distance
        max_size = (
            equity * min(self.config.max_position_pct, HARD_MAX_POSITION_PCT) / 100.0 / entry_price
        )
        return float(np.clip(size, 0.0, max_size))

"""Modelisation et suivi du slippage.

Le slippage est le poste de cout qui tue le plus de systemes en passant du
backtest au reel. On le modelise explicitement (spread + impact de marche), on
mesure l'ecart entre estime et realise, et cet ecart est un critere de go-live :
si le reel derape de plus de 50 % par rapport au modele, le modele est faux.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from trader.models import OrderSide

BPS: float = 10_000.0


@dataclass(frozen=True, slots=True)
class SlippageEstimate:
    """Estimation du slippage d'un ordre."""

    price: float
    slippage_bps: float
    spread_component_bps: float
    impact_component_bps: float

    @property
    def total_cost_pct(self) -> float:
        """Cout total en pourcentage du notionnel."""
        return self.slippage_bps / 100.0


def estimate_slippage(
    reference_price: float,
    side: OrderSide,
    size: float,
    spread_pct: float | None = None,
    average_volume: float | None = None,
    model: str = "spread_plus_impact",
    fixed_bps: float = 5.0,
    impact_coefficient: float = 10.0,
) -> SlippageEstimate:
    """Estime le prix d'execution reel d'un ordre.

    Modele `spread_plus_impact` :
        slippage = spread/2 + k * sqrt(taille / volume moyen)

    La racine carree est le modele d'impact standard (Almgren-Chriss simplifie) :
    doubler la taille ne double pas le cout, il le multiplie par ~1.41.

    Args:
        reference_price: prix mid de reference.
        side: sens de l'ordre (on paie toujours du mauvais cote).
        size: taille en unites de l'actif.
        spread_pct: spread bid/ask en % du mid.
        average_volume: volume moyen par bougie, pour l'impact.
        impact_coefficient: coefficient d'impact en bps.
    """
    if reference_price <= 0:
        raise ValueError("prix de reference invalide")

    if model == "fixed_bps":
        total_bps = fixed_bps
        spread_bps = fixed_bps
        impact_bps = 0.0
    else:
        spread_bps = (spread_pct or 0.05) * 100.0 / 2.0
        if average_volume and average_volume > 0 and size > 0:
            participation = min(size / average_volume, 1.0)
            impact_bps = impact_coefficient * float(np.sqrt(participation))
        else:
            impact_bps = fixed_bps / 2.0
        total_bps = spread_bps + impact_bps

    direction = 1.0 if side is OrderSide.BUY else -1.0
    price = reference_price * (1.0 + direction * total_bps / BPS)
    return SlippageEstimate(
        price=float(price),
        slippage_bps=float(total_bps),
        spread_component_bps=float(spread_bps),
        impact_component_bps=float(impact_bps),
    )


class SlippageTracker:
    """Compare le slippage estime au slippage realise.

    Un ecart systematique signale que le modele sous-estime les couts : le
    backtest est alors trop optimiste, et le passage en live serait dangereux.
    """

    def __init__(self, window: int = 200) -> None:
        self.estimated: deque[float] = deque(maxlen=window)
        self.realized: deque[float] = deque(maxlen=window)

    def record(self, estimated_bps: float, realized_bps: float) -> None:
        """Enregistre une paire (estime, realise)."""
        self.estimated.append(float(estimated_bps))
        self.realized.append(float(realized_bps))

    @property
    def count(self) -> int:
        """Nombre d'observations."""
        return len(self.realized)

    def mean_estimated(self) -> float:
        """Slippage estime moyen (bps)."""
        return float(np.mean(self.estimated)) if self.estimated else 0.0

    def mean_realized(self) -> float:
        """Slippage realise moyen (bps)."""
        return float(np.mean(self.realized)) if self.realized else 0.0

    def divergence_pct(self) -> float:
        """Ecart relatif entre realise et estime, en %."""
        estimated = self.mean_estimated()
        if abs(estimated) < 1e-9:
            return 0.0
        return abs(self.mean_realized() - estimated) / abs(estimated) * 100.0

    def summary(self) -> dict[str, float]:
        """Resume statistique du slippage."""
        return {
            "count": float(self.count),
            "mean_estimated_bps": round(self.mean_estimated(), 3),
            "mean_realized_bps": round(self.mean_realized(), 3),
            "divergence_pct": round(self.divergence_pct(), 2),
            "p95_realized_bps": (float(np.percentile(self.realized, 95)) if self.realized else 0.0),
        }

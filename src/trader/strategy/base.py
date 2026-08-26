"""Contrat que toute strategie doit respecter.

Trois obligations, non negociables :
1. Une strategie declare les regimes dans lesquels elle a le droit de trader.
   Hors de ces regimes, l'ensemble ne la sollicite pas.
2. Tout signal directionnel embarque un STOP LOSS. Une strategie qui ne sait pas
   ou elle a tort n'a pas le droit de proposer un trade.
3. Tout signal directionnel embarque des CONTRA_EVIDENCE : les elements qui vont
   contre le trade. C'est la premiere ligne de defense contre le biais de
   confirmation, avant meme le DevilAdvocate.

Les parametres exposent un espace de recherche BORNE (`param_space`) : le
retraining n'a pas le droit d'explorer n'importe quoi, sinon il sur-apprend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from typing import Any

import pandas as pd

from trader.logging_setup import get_logger
from trader.models import MarketSnapshot, Regime, RegimeState, Signal, StrategyOutput

log = get_logger(__name__)


@dataclass(slots=True)
class StrategyParams:
    """Parametres de base communs a toutes les strategies."""

    stop_atr_multiple: float = 2.0
    target_atr_multiple: float = 3.5
    min_confidence: float = 0.35

    def to_dict(self) -> dict[str, float]:
        """Parametres sous forme de dictionnaire."""
        return asdict(self)


class BaseStrategy(ABC):
    """Classe de base de toutes les strategies du pool."""

    name: str = "base"
    description: str = ""

    def __init__(self, params: StrategyParams | None = None) -> None:
        self.params = params or StrategyParams()

    # ------------------------------------------------------------- contrat

    @abstractmethod
    def generate_signal(self, data: MarketSnapshot, regime: RegimeState) -> StrategyOutput:
        """Produit un signal a partir des donnees ET du regime courant."""

    @abstractmethod
    def get_required_regimes(self) -> list[str]:
        """Regimes dans lesquels cette strategie est autorisee a trader."""

    # ------------------------------------------------------------- helpers

    def is_active_in(self, regime: RegimeState | Regime) -> bool:
        """Vrai si la strategie a le droit de trader dans ce regime."""
        label = regime.regime.value if isinstance(regime, RegimeState) else regime.value
        return label in self.get_required_regimes()

    def param_space(self) -> dict[str, tuple[float, float]]:
        """Espace de recherche BORNE des hyperparametres, pour le retraining.

        Volontairement etroit : un espace large invite au sur-apprentissage.
        """
        return {
            "stop_atr_multiple": (1.0, 4.0),
            "target_atr_multiple": (1.5, 6.0),
            "min_confidence": (0.2, 0.6),
        }

    def get_params(self) -> dict[str, float]:
        """Parametres courants."""
        return self.params.to_dict()

    def set_params(self, values: dict[str, float]) -> None:
        """Applique de nouveaux parametres, en refusant tout ce qui sort des bornes."""
        allowed = {field.name for field in fields(self.params)}
        space = self.param_space()
        for key, value in values.items():
            if key not in allowed:
                raise ValueError(f"{self.name}: parametre inconnu {key!r}")
            if key in space:
                low, high = space[key]
                if not low <= float(value) <= high:
                    raise ValueError(
                        f"{self.name}: {key}={value} hors des bornes autorisees [{low}, {high}]"
                    )
            setattr(self.params, key, float(value))

    def residual_risk(self, data: MarketSnapshot, direction: int) -> list[str]:
        """Risques residuels a mentionner quand aucune objection precise n'est trouvee.

        `contra_evidence` ne doit JAMAIS etre vide sur un signal directionnel.
        Ne rien avoir trouve n'est pas la meme chose que n'avoir aucun risque :
        c'est l'aveu que les controles effectues n'ont rien vu, ce qui reste une
        information — et un rappel que le trade peut echouer quand meme.
        """
        atr_pct = self.feature(data, "atr_pct")
        sense = "hausse" if direction > 0 else "baisse"
        risks = [
            f"aucune contre-indication detectee par {self.name} : "
            f"absence de preuve contre le trade, pas preuve de son bien-fonde"
        ]
        if atr_pct is not None:
            risks.append(
                f"volatilite courante de {atr_pct:.2f} % par bougie : le stop peut etre "
                f"touche par du bruit avant que la {sense} attendue se materialise"
            )
        return risks

    def neutral(self, asset: str, reason: str) -> StrategyOutput:
        """Signal neutre documente (aucune position, mais une explication)."""
        return StrategyOutput(
            signal=Signal.NEUTRAL,
            confidence=0.0,
            stop_loss=0.0,
            reasoning=reason,
            contra_evidence=[],
            regime_affinity=self.get_required_regimes(),
            strategy_name=self.name,
            asset=asset,
        )

    def build_output(
        self,
        data: MarketSnapshot,
        signal: Signal,
        confidence: float,
        stop_loss: float,
        target_price: float | None,
        reasoning: str,
        contra_evidence: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> StrategyOutput:
        """Construit une sortie valide, en refusant les signaux mal formes.

        Un signal sans contra_evidence est downgrade en NEUTRAL plutot que
        d'etre propage : mieux vaut ne pas trader que trader aveuglement.
        """
        if signal is not Signal.NEUTRAL:
            if stop_loss <= 0:
                log.error("strategy_missing_stop", strategy=self.name, asset=data.asset)
                return self.neutral(data.asset, f"{reasoning} (rejete : stop loss invalide)")
            if not contra_evidence:
                log.error("strategy_missing_contra", strategy=self.name, asset=data.asset)
                return self.neutral(
                    data.asset, f"{reasoning} (rejete : aucune contra-evidence produite)"
                )
            if confidence < self.params.min_confidence:
                return self.neutral(
                    data.asset,
                    f"{reasoning} (confiance {confidence:.2f} sous le seuil "
                    f"{self.params.min_confidence:.2f})",
                )
        return StrategyOutput(
            signal=signal,
            confidence=float(min(max(confidence, 0.0), 1.0)),
            stop_loss=float(stop_loss),
            target_price=target_price,
            entry_price=data.last_price,
            reasoning=reasoning,
            contra_evidence=contra_evidence,
            regime_affinity=self.get_required_regimes(),
            strategy_name=self.name,
            asset=data.asset,
            timestamp=data.timestamp,
            metadata=metadata or {},
        )

    # ---------------------------------------------------- acces aux features

    @staticmethod
    def feature(data: MarketSnapshot, name: str, default: float | None = None) -> float | None:
        """Derniere valeur d'une feature du snapshot."""
        return data.feature(name, default)

    @staticmethod
    def series(data: MarketSnapshot, name: str, length: int = 50) -> pd.Series:
        """Fin d'une serie de features (vide si absente)."""
        if data.features is None or name not in getattr(data.features, "columns", []):
            return pd.Series(dtype=float)
        return data.features[name].dropna().tail(length)

    def atr_levels(self, data: MarketSnapshot, direction: int) -> tuple[float | None, float | None]:
        """Stop loss et cible derives de l'ATR (volatilite-adaptatifs).

        Un stop en pourcentage fixe ignore la volatilite du moment : trop serre
        en marche agite, trop large en marche calme. L'ATR corrige cela.
        """
        atr_value = self.feature(data, "atr")
        price = data.last_price
        if atr_value is None or atr_value <= 0 or price <= 0:
            return None, None
        stop = price - direction * self.params.stop_atr_multiple * atr_value
        target = price + direction * self.params.target_atr_multiple * atr_value
        if stop <= 0:
            return None, None
        return float(stop), float(target)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, params={self.get_params()})"

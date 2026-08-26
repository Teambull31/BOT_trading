"""Types de domaine partages par tous les modules.

Ces objets circulent entre data -> regime -> strategy -> devil advocate ->
risk -> execution. Ils sont immuables autant que possible pour eviter qu'un
module en modifie un autre par effet de bord.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from trader.utils.time_utils import utc_now


class Signal(Enum):
    """Signal directionnel discret produit par une strategie."""

    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2

    @property
    def direction(self) -> int:
        """Direction du signal : 1 (long), -1 (short), 0 (neutre)."""
        return (self.value > 0) - (self.value < 0)

    @classmethod
    def from_score(cls, score: float, strong_threshold: float = 1.5) -> Signal:
        """Convertit un score continu en signal discret."""
        if score >= strong_threshold:
            return cls.STRONG_BUY
        if score >= 0.5:
            return cls.BUY
        if score <= -strong_threshold:
            return cls.STRONG_SELL
        if score <= -0.5:
            return cls.SELL
        return cls.NEUTRAL


class Regime(str, Enum):
    """Regimes de marche reconnus par le systeme."""

    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    RANGE_BOUND = "range_bound"
    CRISIS = "crisis"
    UNCERTAIN = "uncertain"

    @property
    def is_tradable(self) -> bool:
        """Faux pour les regimes ou aucune nouvelle position n'est permise."""
        return self is not Regime.CRISIS


class OrderSide(str, Enum):
    """Sens d'un ordre."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Type d'ordre transmis a l'exchange."""

    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    """Cycle de vie d'un ordre."""

    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    FAILED = "failed"


class RiskDecision(str, Enum):
    """Verdict du risk manager sur une intention de trade."""

    APPROVED = "approved"
    REDUCED = "reduced"
    REJECTED = "rejected"


class ContraRecommendation(str, Enum):
    """Recommandation du DevilAdvocate."""

    PROCEED = "proceed"
    REDUCE_SIZE = "reduce_size"
    ABORT = "abort"


class StrategyHealth(str, Enum):
    """Etat de sante d'une strategie dans l'ensemble."""

    HEALTHY = "healthy"
    DEGRADING = "degrading"
    DEAD = "dead"
    ZOMBIE = "zombie"


@dataclass(frozen=True, slots=True)
class RegimeState:
    """Etat de regime courant produit par le RegimeDetector."""

    regime: Regime
    confidence: float
    agreement_score: float
    transition_probability: float
    timestamp: datetime = field(default_factory=utc_now)
    method_votes: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_uncertain(self) -> bool:
        """Vrai si le regime courant est declare incertain."""
        return self.regime is Regime.UNCERTAIN

    @property
    def is_crisis(self) -> bool:
        """Vrai si le systeme est en mode crise."""
        return self.regime is Regime.CRISIS

    def to_dict(self) -> dict[str, Any]:
        """Representation serialisable pour les logs et la persistence."""
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "agreement_score": round(self.agreement_score, 4),
            "transition_probability": round(self.transition_probability, 4),
            "timestamp": self.timestamp.isoformat(),
            "method_votes": dict(self.method_votes),
        }


@dataclass(frozen=True, slots=True)
class StrategyOutput:
    """Sortie normalisee d'une strategie.

    `stop_loss` et `contra_evidence` sont OBLIGATOIRES : une strategie qui ne
    sait pas ou elle se trompe n'a pas le droit de proposer un trade.
    """

    signal: Signal
    confidence: float
    stop_loss: float
    reasoning: str
    contra_evidence: list[str]
    regime_affinity: list[str]
    strategy_name: str = "unknown"
    asset: str = ""
    target_price: float | None = None
    entry_price: float | None = None
    timestamp: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence hors [0,1] : {self.confidence}")
        if self.signal is not Signal.NEUTRAL:
            if self.stop_loss is None or self.stop_loss <= 0:
                raise ValueError("un signal directionnel exige un stop loss strictement positif")
            if not self.contra_evidence:
                raise ValueError(
                    "contra_evidence est obligatoire : chaque signal doit lister "
                    "les preuves qui vont contre lui"
                )

    @property
    def is_actionable(self) -> bool:
        """Vrai si le signal demande une action de trading."""
        return self.signal is not Signal.NEUTRAL

    @property
    def risk_reward(self) -> float | None:
        """Ratio risk/reward implicite, si entree et cible sont connues."""
        if self.entry_price is None or self.target_price is None:
            return None
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0:
            return None
        return abs(self.target_price - self.entry_price) / risk

    def to_dict(self) -> dict[str, Any]:
        """Representation serialisable pour l'audit trail."""
        return {
            "strategy": self.strategy_name,
            "asset": self.asset,
            "signal": self.signal.name,
            "confidence": round(self.confidence, 4),
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "reasoning": self.reasoning,
            "contra_evidence": list(self.contra_evidence),
            "regime_affinity": list(self.regime_affinity),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class EnsembleDecision:
    """Decision agregee de l'ensemble de strategies."""

    asset: str
    signal: Signal
    score: float
    confidence: float
    consensus: float
    dispersion: float
    weights: dict[str, float]
    contributions: list[StrategyOutput]
    stop_loss: float | None = None
    target_price: float | None = None
    entry_price: float | None = None
    blocked_reason: str | None = None
    timestamp: datetime = field(default_factory=utc_now)

    @property
    def is_actionable(self) -> bool:
        """Vrai si l'ensemble propose reellement un trade."""
        return self.signal is not Signal.NEUTRAL and self.blocked_reason is None

    def to_dict(self) -> dict[str, Any]:
        """Representation serialisable pour l'audit trail."""
        return {
            "asset": self.asset,
            "signal": self.signal.name,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "consensus": round(self.consensus, 4),
            "dispersion": round(self.dispersion, 4),
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "entry_price": self.entry_price,
            "blocked_reason": self.blocked_reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ContraReport:
    """Rapport du DevilAdvocate : les preuves CONTRE le trade propose."""

    contra_signals: list[str]
    contra_score: float
    recommendation: ContraRecommendation
    checks: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Representation serialisable pour l'audit trail."""
        return {
            "contra_score": round(self.contra_score, 4),
            "recommendation": self.recommendation.value,
            "contra_signals": list(self.contra_signals),
            "checks": {k: round(v, 4) for k, v in self.checks.items()},
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class TradeIntent:
    """Intention de trade soumise au risk manager."""

    asset: str
    side: OrderSide
    entry_price: float
    stop_loss: float
    target_price: float | None
    confidence: float
    regime: RegimeState
    decision: EnsembleDecision
    contra_report: ContraReport | None = None
    timestamp: datetime = field(default_factory=utc_now)

    @property
    def stop_distance_pct(self) -> float:
        """Distance du stop loss en % du prix d'entree."""
        if self.entry_price <= 0:
            return float("inf")
        return abs(self.entry_price - self.stop_loss) / self.entry_price * 100.0

    @property
    def risk_reward(self) -> float | None:
        """Ratio risk/reward de l'intention."""
        if self.target_price is None:
            return None
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0:
            return None
        return abs(self.target_price - self.entry_price) / risk


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    """Verdict du risk manager. Un rejet n'est jamais contournable."""

    decision: RiskDecision
    approved_size: float
    approved_notional: float
    reasons: list[str]
    checks: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)

    @property
    def is_approved(self) -> bool:
        """Vrai si un ordre peut etre transmis a l'execution."""
        return self.decision is not RiskDecision.REJECTED and self.approved_size > 0

    def to_dict(self) -> dict[str, Any]:
        """Representation serialisable pour l'audit trail."""
        return {
            "decision": self.decision.value,
            "approved_size": self.approved_size,
            "approved_notional": round(self.approved_notional, 4),
            "reasons": list(self.reasons),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(slots=True)
class Order:
    """Ordre soumis a l'execution."""

    asset: str
    side: OrderSide
    size: float
    order_type: OrderType
    price: float | None = None
    stop_loss: float | None = None
    target_price: float | None = None
    exchange: str = "paper"
    client_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    average_price: float = 0.0
    fees: float = 0.0
    slippage_bps: float = 0.0
    reason: str = ""
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fill_ratio(self) -> float:
        """Fraction de l'ordre effectivement remplie."""
        if self.size <= 0:
            return 0.0
        return self.filled_size / self.size

    def to_dict(self) -> dict[str, Any]:
        """Representation serialisable pour la persistence."""
        return {
            "asset": self.asset,
            "side": self.side.value,
            "size": self.size,
            "order_type": self.order_type.value,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "exchange": self.exchange,
            "client_id": self.client_id,
            "status": self.status.value,
            "filled_size": self.filled_size,
            "average_price": self.average_price,
            "fees": self.fees,
            "slippage_bps": self.slippage_bps,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(slots=True)
class Position:
    """Position ouverte, suivie par le portefeuille."""

    asset: str
    side: OrderSide
    size: float
    entry_price: float
    stop_loss: float
    target_price: float | None = None
    strategy: str = "ensemble"
    opened_at: datetime = field(default_factory=utc_now)
    fees_paid: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> int:
        """1 pour un long, -1 pour un short."""
        return 1 if self.side is OrderSide.BUY else -1

    def notional(self, price: float) -> float:
        """Valeur notionnelle de la position au prix donne."""
        return abs(self.size) * price

    def unrealized_pnl(self, price: float) -> float:
        """P&L latent au prix courant, frais d'entree deduits."""
        return (price - self.entry_price) * self.size * self.direction - self.fees_paid

    def should_stop_out(self, price: float) -> bool:
        """Vrai si le stop loss est touche au prix courant."""
        if self.direction > 0:
            return price <= self.stop_loss
        return price >= self.stop_loss

    def should_take_profit(self, price: float) -> bool:
        """Vrai si la cible de profit est atteinte au prix courant."""
        if self.target_price is None:
            return False
        if self.direction > 0:
            return price >= self.target_price
        return price <= self.target_price


@dataclass(slots=True)
class Trade:
    """Trade cloture, unite de mesure de la performance."""

    asset: str
    side: OrderSide
    size: float
    entry_price: float
    exit_price: float
    pnl: float
    fees: float
    opened_at: datetime
    closed_at: datetime
    strategy: str = "ensemble"
    regime: str = ""
    exit_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def return_pct(self) -> float:
        """Rendement du trade en % du notionnel engage."""
        notional = abs(self.size) * self.entry_price
        if notional <= 0:
            return 0.0
        return self.pnl / notional * 100.0

    @property
    def is_winner(self) -> bool:
        """Vrai si le trade est gagnant net de frais."""
        return self.pnl > 0

    def to_dict(self) -> dict[str, Any]:
        """Representation serialisable pour la persistence."""
        return {
            "asset": self.asset,
            "side": self.side.value,
            "size": self.size,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "fees": self.fees,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat(),
            "strategy": self.strategy,
            "regime": self.regime,
            "exit_reason": self.exit_reason,
        }


@dataclass(slots=True)
class MarketSnapshot:
    """Photo du marche pour un actif a un instant t.

    Contient uniquement des donnees disponibles a `timestamp` : c'est le garde-fou
    principal contre le look-ahead bias dans les strategies.
    """

    asset: str
    timestamp: datetime
    ohlcv: Any  # pandas.DataFrame indexe par timestamp
    features: Any  # pandas.DataFrame aligne sur ohlcv
    last_price: float
    bid: float | None = None
    ask: float | None = None
    order_book: dict[str, Any] = field(default_factory=dict)
    funding_rate: float | None = None
    open_interest: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def spread_pct(self) -> float | None:
        """Spread bid/ask en % du mid, si le carnet est disponible."""
        if self.bid is None or self.ask is None or self.bid <= 0 or self.ask <= 0:
            return None
        mid = (self.bid + self.ask) / 2.0
        if mid <= 0:
            return None
        return (self.ask - self.bid) / mid * 100.0

    def feature(self, name: str, default: float | None = None) -> float | None:
        """Derniere valeur d'une feature, ou `default` si absente ou NaN."""
        import math

        if self.features is None or name not in getattr(self.features, "columns", []):
            return default
        series = self.features[name].dropna()
        if series.empty:
            return default
        value = float(series.iloc[-1])
        return default if math.isnan(value) else value

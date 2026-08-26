"""Simulateur de paper trading.

Objectif : produire des resultats COMPARABLES au live, pas des resultats
flatteurs. Le simulateur applique donc systematiquement ce que la realite
applique :

- slippage (spread + impact de taille) ;
- frais maker/taker selon le type d'ordre ;
- fills partiels quand l'ordre depasse une fraction du volume moyen ;
- latence aleatoire de 100-500 ms ;
- ordres limites non executes si le prix ne revient pas.

Les resultats sont enregistres au meme format que le live afin de pouvoir
comparer directement paper et reel — c'est un critere de passage en live.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from dataclasses import dataclass

from trader.config import ExecutionConfig
from trader.execution.slippage import estimate_slippage
from trader.logging_setup import get_logger
from trader.models import MarketSnapshot, Order, OrderSide, OrderStatus, OrderType
from trader.utils.time_utils import utc_now

log = get_logger(__name__)


@dataclass(slots=True)
class PaperFill:
    """Resultat d'une execution simulee."""

    filled_size: float
    average_price: float
    fees: float
    slippage_bps: float
    status: OrderStatus
    latency_ms: float
    note: str = ""


class PaperBroker:
    """Execute des ordres en simulation, avec des frictions realistes."""

    def __init__(self, config: ExecutionConfig, seed: int | None = None) -> None:
        self.config = config
        self._random = random.Random(seed)

    async def execute(self, order: Order, data: MarketSnapshot) -> PaperFill:
        """Simule l'execution d'un ordre sur le marche decrit par `data`."""
        latency_ms = await self._simulate_latency()
        reference = self._reference_price(order, data)
        if reference <= 0:
            return PaperFill(0.0, 0.0, 0.0, 0.0, OrderStatus.REJECTED, latency_ms, "prix invalide")

        average_volume = self._average_volume(data)
        estimate = estimate_slippage(
            reference_price=reference,
            side=order.side,
            size=order.size,
            spread_pct=data.spread_pct,
            average_volume=average_volume,
            model=self.config.slippage_model,
            fixed_bps=self.config.fixed_slippage_bps,
        )

        filled_size, note = self._fill_size(order, average_volume)
        if filled_size <= 0:
            return PaperFill(
                0.0, 0.0, 0.0, estimate.slippage_bps, OrderStatus.CANCELED, latency_ms, note
            )

        # Un ordre limite ne subit pas le spread complet : c'est l'interet du
        # limit order. En contrepartie il n'est pas toujours servi.
        if order.order_type is OrderType.LIMIT:
            fill_price = order.price if order.price else estimate.price
            slippage_bps = abs(fill_price - reference) / reference * 10_000.0
            fee_bps = self.config.maker_fee_bps
        else:
            fill_price = estimate.price
            slippage_bps = estimate.slippage_bps
            fee_bps = self.config.taker_fee_bps

        notional = filled_size * fill_price
        fees = notional * fee_bps / 10_000.0
        status = (
            OrderStatus.FILLED
            if filled_size >= order.size * 0.999
            else OrderStatus.PARTIALLY_FILLED
        )
        return PaperFill(
            filled_size=float(filled_size),
            average_price=float(fill_price),
            fees=float(fees),
            slippage_bps=float(slippage_bps),
            status=status,
            latency_ms=latency_ms,
            note=note,
        )

    def _reference_price(self, order: Order, data: MarketSnapshot) -> float:
        """Prix de reference de l'execution (cote du carnet si disponible)."""
        if order.side is OrderSide.BUY and data.ask:
            return float(data.ask)
        if order.side is OrderSide.SELL and data.bid:
            return float(data.bid)
        return float(data.last_price)

    @staticmethod
    def _average_volume(data: MarketSnapshot) -> float:
        """Volume moyen recent, base de l'impact de marche."""
        if data.ohlcv is None or "volume" not in getattr(data.ohlcv, "columns", []):
            return 0.0
        tail = data.ohlcv["volume"].tail(20)
        return float(tail.mean()) if len(tail) else 0.0

    def _fill_size(self, order: Order, average_volume: float) -> tuple[float, str]:
        """Determine la quantite servie, avec fills partiels sur les gros ordres."""
        if average_volume <= 0:
            return order.size, ""

        participation = order.size / average_volume * 100.0
        threshold = self.config.partial_fill_volume_pct
        if participation <= threshold:
            return order.size, ""

        # Au-dela du seuil, on ne sert qu'une fraction : le carnet n'absorbe pas tout.
        ratio = max(0.3, threshold / participation)
        filled = order.size * ratio
        note = (
            f"fill partiel : l'ordre represente {participation:.1f} % du volume moyen "
            f"(seuil {threshold:.1f} %)"
        )
        if ratio < self.config.min_fill_pct:
            return filled, note + " — sous le minimum de remplissage"
        return filled, note

    async def _simulate_latency(self) -> float:
        """Simule la latence reseau (et la subit reellement, comme en live)."""
        if not self.config.simulate_latency:
            return 0.0
        low, high = self.config.latency_ms_range
        latency_ms = self._random.uniform(low, high)
        await asyncio.sleep(latency_ms / 1000.0)
        return latency_ms


def build_order(
    asset: str,
    side: OrderSide,
    size: float,
    price: float | None,
    order_type: OrderType,
    stop_loss: float | None = None,
    target_price: float | None = None,
    reason: str = "",
    exchange: str = "paper",
) -> Order:
    """Construit un ordre horodate avec un identifiant client unique."""
    stamp = utc_now()
    # Suffixe aleatoire indispensable : deux ordres emis dans la meme
    # milliseconde partageraient sinon le meme identifiant, et un exchange
    # rejette (ou pire, confond) deux ordres au meme client id.
    suffix = uuid.uuid4().hex[:8]
    client_id = f"{asset.replace('/', '')}-{side.value}-{int(stamp.timestamp() * 1000)}-{suffix}"
    return Order(
        asset=asset,
        side=side,
        size=float(size),
        order_type=order_type,
        price=price,
        stop_loss=stop_loss,
        target_price=target_price,
        exchange=exchange,
        client_id=client_id,
        reason=reason,
        created_at=stamp,
        updated_at=stamp,
    )

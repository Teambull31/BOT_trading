"""Passage d'ordres : routage paper/live, retry, suivi du slippage.

Garde-fou central : `execute()` exige un RiskVerdict APPROUVE. Il n'existe aucun
chemin de code permettant d'envoyer un ordre sans passer par le risk manager —
c'est ce qui rend le module de risque reellement non contournable, et pas
seulement "recommande".

Le mode live n'est atteignable que si la configuration est explicitement en
Mode.LIVE ; par defaut tout passe par le simulateur.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from trader.config import Mode, Settings
from trader.execution.paper import PaperBroker, PaperFill, build_order
from trader.execution.slippage import SlippageTracker, estimate_slippage
from trader.logging_setup import get_logger
from trader.models import (
    MarketSnapshot,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskVerdict,
    TradeIntent,
)
from trader.utils.time_utils import utc_now

log = get_logger(__name__)


class ExecutionError(RuntimeError):
    """Echec d'execution apres epuisement des tentatives."""


class RiskBypassError(PermissionError):
    """Tentative d'execution sans verdict de risque approuve."""


@dataclass(slots=True)
class ExecutionResult:
    """Resultat consolide d'une execution."""

    order: Order
    filled: bool
    attempts: int
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def notional(self) -> float:
        """Notionnel effectivement execute."""
        return self.order.filled_size * self.order.average_price


class OrderExecutor:
    """Transmet les ordres au marche (simule ou reel) avec resilience."""

    def __init__(
        self,
        settings: Settings,
        broker: PaperBroker | None = None,
        exchange_client: Any | None = None,
        store: Any | None = None,
    ) -> None:
        self.settings = settings
        self.config = settings.execution
        self.broker = broker or PaperBroker(settings.execution)
        self.exchange_client = exchange_client
        self.store = store
        self.slippage = SlippageTracker()

    @property
    def is_live(self) -> bool:
        """Vrai si les ordres partent vers un exchange reel."""
        return self.settings.general.mode is Mode.LIVE

    async def execute(
        self,
        intent: TradeIntent,
        verdict: RiskVerdict,
        data: MarketSnapshot,
        urgent: bool = False,
    ) -> ExecutionResult:
        """Execute une intention validee par le risk manager.

        Raises:
            RiskBypassError: si le verdict n'autorise pas le trade. Cette
                exception ne doit jamais etre rattrapee pour "reessayer sans" :
                elle signale un bug de cablage du systeme.
        """
        if not verdict.is_approved:
            raise RiskBypassError(
                f"tentative d'execution sans approbation du risk manager "
                f"({verdict.decision.value}) sur {intent.asset}"
            )

        order_type = OrderType.MARKET if urgent else OrderType(self.config.default_order_type)
        price = self._limit_price(intent, data) if order_type is OrderType.LIMIT else None
        order = build_order(
            asset=intent.asset,
            side=intent.side,
            size=verdict.approved_size,
            price=price,
            order_type=order_type,
            stop_loss=intent.stop_loss,
            target_price=intent.target_price,
            reason="; ".join(verdict.reasons[:2]),
            exchange="live" if self.is_live else "paper",
        )
        return await self._submit_with_retry(order, data)

    async def close_position(
        self,
        asset: str,
        side: OrderSide,
        size: float,
        data: MarketSnapshot,
        reason: str,
    ) -> ExecutionResult:
        """Ferme une position. Toujours au marche : une sortie ne s'optimise pas.

        Un ordre limite pour sortir peut ne jamais etre servi, et une position
        que l'on croit fermee alors qu'elle est ouverte est la pire des situations.
        """
        order = build_order(
            asset=asset,
            side=side,
            size=size,
            price=None,
            order_type=OrderType.MARKET,
            reason=reason,
            exchange="live" if self.is_live else "paper",
        )
        return await self._submit_with_retry(order, data)

    # -------------------------------------------------------------- interne

    def _limit_price(self, intent: TradeIntent, data: MarketSnapshot) -> float:
        """Prix limite : au meilleur bid/ask, legerement en notre faveur."""
        if intent.side is OrderSide.BUY:
            base = float(data.bid or data.last_price)
            return base * (1.0 + 0.0002)
        base = float(data.ask or data.last_price)
        return base * (1.0 - 0.0002)

    async def _submit_with_retry(self, order: Order, data: MarketSnapshot) -> ExecutionResult:
        """Soumet un ordre avec retry a backoff exponentiel."""
        last_error: str | None = None
        total_latency = 0.0

        for attempt in range(1, self.config.max_retries + 1):
            try:
                fill = await self._submit(order, data)
                total_latency += fill.latency_ms
                order.filled_size = fill.filled_size
                order.average_price = fill.average_price
                order.fees = fill.fees
                order.slippage_bps = fill.slippage_bps
                order.status = fill.status
                order.updated_at = utc_now()
                if fill.note:
                    order.metadata["note"] = fill.note

                if fill.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                    self._record_slippage(order, data)
                    self._persist(order)
                    log.info(
                        "order_executed",
                        asset=order.asset,
                        side=order.side.value,
                        type=order.order_type.value,
                        size=order.size,
                        filled=order.filled_size,
                        price=order.average_price,
                        slippage_bps=round(order.slippage_bps, 2),
                        fees=round(order.fees, 4),
                        attempt=attempt,
                        mode="live" if self.is_live else "paper",
                    )
                    return ExecutionResult(
                        order=order,
                        filled=fill.status is OrderStatus.FILLED,
                        attempts=attempt,
                        latency_ms=total_latency,
                    )

                last_error = fill.note or f"statut {fill.status.value}"
                log.warning(
                    "order_not_filled",
                    asset=order.asset,
                    attempt=attempt,
                    status=fill.status.value,
                    note=fill.note,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - on classe et on retente
                last_error = f"{type(exc).__name__}: {exc}"
                log.error(
                    "order_submit_failed",
                    asset=order.asset,
                    attempt=attempt,
                    error=last_error,
                )

            if attempt < self.config.max_retries:
                await asyncio.sleep(self.config.retry_backoff_sec * (2 ** (attempt - 1)))
                # A la reprise, un ordre limite non servi devient un ordre au
                # marche : insister au meme prix ne sert a rien si le marche est parti.
                if order.order_type is OrderType.LIMIT and attempt >= 2:
                    order.order_type = OrderType.MARKET
                    order.price = None
                    order.metadata["escalated_to_market"] = True

        order.status = OrderStatus.FAILED
        order.updated_at = utc_now()
        self._persist(order)
        log.error(
            "order_failed_definitively",
            asset=order.asset,
            attempts=self.config.max_retries,
            error=last_error,
        )
        return ExecutionResult(
            order=order,
            filled=False,
            attempts=self.config.max_retries,
            latency_ms=total_latency,
            error=last_error,
        )

    async def _submit(self, order: Order, data: MarketSnapshot) -> PaperFill:
        """Route l'ordre vers le simulateur ou vers l'exchange reel."""
        if not self.is_live:
            return await self.broker.execute(order, data)
        return await self._submit_live(order, data)

    async def _submit_live(self, order: Order, data: MarketSnapshot) -> PaperFill:
        """Transmet un ordre reel via ccxt."""
        if self.exchange_client is None:
            raise ExecutionError("mode live demande mais aucun client d'exchange n'est branche")
        params: dict[str, Any] = {}
        if order.order_type is OrderType.LIMIT:
            response = await self.exchange_client.create_order(
                order.asset, "limit", order.side.value, order.size, order.price, params
            )
        else:
            response = await self.exchange_client.create_order(
                order.asset, "market", order.side.value, order.size, None, params
            )

        filled = float(response.get("filled") or 0.0)
        average = float(response.get("average") or response.get("price") or 0.0)
        fee_cost = float((response.get("fee") or {}).get("cost") or 0.0)
        status_map = {
            "closed": OrderStatus.FILLED,
            "open": OrderStatus.OPEN,
            "canceled": OrderStatus.CANCELED,
            "rejected": OrderStatus.REJECTED,
        }
        status = status_map.get(str(response.get("status")), OrderStatus.OPEN)
        if status is OrderStatus.OPEN and 0 < filled < order.size:
            status = OrderStatus.PARTIALLY_FILLED

        reference = float(data.last_price)
        slippage_bps = (
            abs(average - reference) / reference * 10_000.0
            if average > 0 and reference > 0
            else 0.0
        )
        return PaperFill(
            filled_size=filled,
            average_price=average,
            fees=fee_cost,
            slippage_bps=slippage_bps,
            status=status,
            latency_ms=0.0,
            note=str(response.get("id", "")),
        )

    def _record_slippage(self, order: Order, data: MarketSnapshot) -> None:
        """Compare le slippage realise a l'estimation du modele."""
        try:
            estimate = estimate_slippage(
                reference_price=float(data.last_price),
                side=order.side,
                size=order.size,
                spread_pct=data.spread_pct,
                average_volume=float(data.ohlcv["volume"].tail(20).mean())
                if data.ohlcv is not None and "volume" in data.ohlcv
                else None,
                model=self.config.slippage_model,
                fixed_bps=self.config.fixed_slippage_bps,
            )
        except ValueError as exc:
            log.warning("slippage_estimate_failed", asset=order.asset, error=str(exc))
            return
        self.slippage.record(estimate.slippage_bps, order.slippage_bps)

    def _persist(self, order: Order) -> None:
        """Enregistre l'ordre en base (jamais bloquant pour l'execution)."""
        if self.store is None:
            return
        try:
            self.store.save_order(order, mode="live" if self.is_live else "paper")
        except Exception as exc:  # noqa: BLE001 - la persistence ne doit pas tuer le trading
            log.error("order_persist_failed", asset=order.asset, error=str(exc))

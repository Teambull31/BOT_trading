"""Metriques Prometheus.

`prometheus_client` est optionnel : sans lui, les metriques deviennent des
no-ops silencieux. Un systeme de trading ne doit pas refuser de trader parce
qu'une librairie d'observabilite manque — l'observabilite sert le trading, elle
ne le conditionne pas.
"""

from __future__ import annotations

from typing import Any

from trader.logging_setup import get_logger

log = get_logger(__name__)

try:  # pragma: no cover - depend de l'environnement
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False
    CollectorRegistry = object  # type: ignore[assignment,misc]


class _NoOpMetric:
    """Metrique inerte, utilisee quand prometheus_client est absent."""

    def labels(self, *args: Any, **kwargs: Any) -> _NoOpMetric:
        """Retourne la meme metrique inerte."""
        return self

    def set(self, value: float) -> None:
        """Ne fait rien."""

    def inc(self, value: float = 1.0) -> None:
        """Ne fait rien."""

    def observe(self, value: float) -> None:
        """Ne fait rien."""


class TraderMetrics:
    """Expose l'etat du systeme au format Prometheus."""

    def __init__(self, port: int = 9090, enabled: bool = True, registry: Any = None) -> None:
        self.port = port
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        self.registry = registry if registry is not None else self._default_registry()
        self._server_started = False

        if not self.enabled:
            if enabled and not PROMETHEUS_AVAILABLE:
                log.warning("prometheus_client_absent", consequence="metriques desactivees")
            self._build_noop()
            return
        self._build_metrics()

    def _default_registry(self) -> Any:
        """Registre dedie : evite les collisions entre instances (tests inclus)."""
        return CollectorRegistry() if PROMETHEUS_AVAILABLE else None

    def _build_noop(self) -> None:
        """Installe des metriques inertes."""
        for name in (
            "pnl_total",
            "pnl_daily",
            "drawdown_current",
            "drawdown_max",
            "sharpe_rolling_30d",
            "equity",
            "orders_total",
            "slippage_bps",
            "api_latency_seconds",
            "api_errors_total",
            "strategy_weight",
            "strategy_sharpe",
            "strategy_health",
            "regime_current",
            "regime_confidence",
            "devil_advocate_score",
            "exposure_total_pct",
            "positions_open",
            "circuit_breaker_triggered",
            "cycles_total",
            "cycle_errors_total",
        ):
            setattr(self, name, _NoOpMetric())

    def _build_metrics(self) -> None:
        """Declare les metriques Prometheus."""
        registry = self.registry

        # Performance.
        self.pnl_total = Gauge("trader_pnl_total", "P&L total realise", registry=registry)
        self.pnl_daily = Gauge("trader_pnl_daily", "P&L du jour", registry=registry)
        self.equity = Gauge("trader_equity", "Valeur du portefeuille", registry=registry)
        self.drawdown_current = Gauge(
            "trader_drawdown_current", "Drawdown courant en %", registry=registry
        )
        self.drawdown_max = Gauge("trader_drawdown_max", "Drawdown maximal en %", registry=registry)
        self.sharpe_rolling_30d = Gauge(
            "trader_sharpe_rolling_30d", "Sharpe glissant 30 jours", registry=registry
        )

        # Operationnel.
        self.orders_total = Counter(
            "trader_orders_total", "Ordres emis", ["side", "status"], registry=registry
        )
        self.slippage_bps = Histogram(
            "trader_slippage_bps",
            "Slippage realise en points de base",
            buckets=(1, 2, 5, 10, 20, 50, 100, 250),
            registry=registry,
        )
        self.api_latency_seconds = Histogram(
            "trader_api_latency_seconds",
            "Latence des appels exchange",
            ["exchange"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
            registry=registry,
        )
        self.api_errors_total = Counter(
            "trader_api_errors_total",
            "Erreurs d'API",
            ["exchange", "error_type"],
            registry=registry,
        )
        self.cycles_total = Counter(
            "trader_cycles_total", "Cycles de trading executes", registry=registry
        )
        self.cycle_errors_total = Counter(
            "trader_cycle_errors_total", "Erreurs rencontrees pendant un cycle", registry=registry
        )

        # Strategies et regime.
        self.strategy_weight = Gauge(
            "trader_strategy_weight", "Poids d'une strategie", ["strategy"], registry=registry
        )
        self.strategy_sharpe = Gauge(
            "trader_strategy_sharpe", "Sharpe d'une strategie", ["strategy"], registry=registry
        )
        self.strategy_health = Gauge(
            "trader_strategy_health",
            "Sante d'une strategie (1 healthy, 0.5 degrading, 0 dead/zombie)",
            ["strategy"],
            registry=registry,
        )
        self.regime_current = Gauge(
            "trader_regime_current", "Regime actif (1 = actif)", ["regime"], registry=registry
        )
        self.regime_confidence = Gauge(
            "trader_regime_confidence", "Confiance du regime detecte", registry=registry
        )
        self.devil_advocate_score = Histogram(
            "trader_devil_advocate_score",
            "Score contra du devil advocate",
            buckets=(0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=registry,
        )

        # Risque.
        self.exposure_total_pct = Gauge(
            "trader_exposure_total_pct", "Exposition totale en %", registry=registry
        )
        self.positions_open = Gauge(
            "trader_positions_open", "Positions ouvertes", registry=registry
        )
        self.circuit_breaker_triggered = Counter(
            "trader_circuit_breaker_triggered",
            "Declenchements de circuit breaker",
            ["reason"],
            registry=registry,
        )

    # ------------------------------------------------------------- serveur

    def start(self) -> None:
        """Demarre le serveur HTTP d'exposition des metriques."""
        if not self.enabled or self._server_started:
            return
        try:
            start_http_server(self.port, registry=self.registry)
            self._server_started = True
            log.info("prometheus_started", port=self.port)
        except OSError as exc:
            log.error("prometheus_start_failed", port=self.port, error=str(exc))

    # ---------------------------------------------------------- mise a jour

    def update_from_cycle(self, orchestrator: Any, report: Any, prices: dict[str, float]) -> None:
        """Met a jour toutes les metriques a partir d'un cycle termine.

        Ne leve jamais : une metrique ratee ne doit pas interrompre le trading.
        """
        try:
            self._update(orchestrator, report, prices)
        except Exception as exc:  # noqa: BLE001 - l'observabilite ne casse pas le trading
            log.error("metrics_update_failed", error=str(exc), error_type=type(exc).__name__)

    def _update(self, orchestrator: Any, report: Any, prices: dict[str, float]) -> None:
        """Corps de la mise a jour."""
        portfolio = orchestrator.portfolio
        snapshot = portfolio.snapshot(prices)

        self.cycles_total.inc()
        for _ in getattr(report, "errors", []):
            self.cycle_errors_total.inc()

        self.equity.set(snapshot["equity"])
        self.pnl_total.set(snapshot["realized_pnl"])
        self.drawdown_current.set(snapshot["drawdown_total_pct"])
        self.drawdown_max.set(max(snapshot["drawdown_total_pct"], 0.0))
        self.exposure_total_pct.set(snapshot["exposure_pct"])
        self.positions_open.set(snapshot["open_positions"])

        for name, state in orchestrator.ensemble.snapshot().items():
            self.strategy_weight.labels(strategy=name).set(state.get("weight", 0.0))
            metrics = state.get("metrics", {})
            self.strategy_sharpe.labels(strategy=name).set(metrics.get("sharpe_30d", 0.0))
            self.strategy_health.labels(strategy=name).set(
                {"healthy": 1.0, "degrading": 0.5, "dead": 0.0, "zombie": 0.0}.get(
                    state.get("health", "healthy"), 0.0
                )
            )

        for regime in getattr(report, "regime_by_asset", {}).values():
            self.regime_current.labels(regime=regime).set(1.0)
        last_state = getattr(orchestrator.detector, "last_state", None)
        if last_state is not None:
            self.regime_confidence.set(last_state.confidence)

        for exchange, latency in getattr(orchestrator.ingester, "latencies", {}).items():
            self.api_latency_seconds.labels(exchange=exchange).observe(latency)

        tracker = getattr(orchestrator.executor, "slippage", None)
        if tracker is not None and tracker.count:
            self.slippage_bps.observe(tracker.mean_realized())

    def record_order(self, side: str, status: str, slippage_bps: float = 0.0) -> None:
        """Enregistre un ordre emis."""
        self.orders_total.labels(side=side, status=status).inc()
        if slippage_bps:
            self.slippage_bps.observe(slippage_bps)

    def record_api_error(self, exchange: str, error_type: str) -> None:
        """Enregistre une erreur d'API."""
        self.api_errors_total.labels(exchange=exchange, error_type=error_type).inc()

    def record_breaker(self, reason: str) -> None:
        """Enregistre un declenchement de circuit breaker."""
        self.circuit_breaker_triggered.labels(reason=reason).inc()

    def record_contra_score(self, score: float) -> None:
        """Enregistre un score du devil advocate."""
        self.devil_advocate_score.observe(score)

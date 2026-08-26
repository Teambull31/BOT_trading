"""Chargement et validation de la configuration (Pydantic + TOML).

Regle fondamentale : les HARD LIMITS sont definis dans ce module, en dur.
La configuration peut les rendre PLUS conservateurs, jamais plus permissifs.
Toute tentative de depassement leve une erreur au chargement, pas a l'execution.
"""

from __future__ import annotations

import tomllib
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------------------
# HARD LIMITS — dans le CODE, jamais dans la config.
# Les modifier demande un changement de code source revu et commite.
# --------------------------------------------------------------------------------------
HARD_MAX_POSITION_PCT: float = 2.0
"""Exposition maximale absolue d'une seule position, en % du capital."""

HARD_MAX_DRAWDOWN_TOTAL_PCT: float = 15.0
"""Drawdown total maximal absolu : au-dela, kill switch et liquidation."""

HARD_MAX_EXPOSURE_PCT: float = 50.0
"""Exposition brute totale maximale absolue, en % du capital."""

HARD_MIN_EXPOSURE_PCT: float = 5.0
"""Plancher de configuration de l'exposition totale (evite une config absurde)."""

HARD_MAX_CONCURRENT_POSITIONS: int = 20
"""Nombre maximal absolu de positions ouvertes simultanement."""


class Mode(str, Enum):
    """Mode d'execution du systeme."""

    PAPER = "paper"
    LIVE = "live"


class GeneralConfig(BaseModel):
    """Parametres generaux du systeme."""

    model_config = ConfigDict(extra="forbid")

    mode: Mode = Mode.PAPER
    base_currency: str = "USDT"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    timezone: str = "UTC"
    initial_capital: float = Field(default=500.0, gt=0)
    loop_interval_sec: int = Field(default=60, ge=1)
    heartbeat_path: str = "/tmp/trader_heartbeat"


class ExchangeConfig(BaseModel):
    """Parametres specifiques a un exchange."""

    model_config = ConfigDict(extra="forbid")

    rate_limit_ms: int = Field(default=100, ge=0)
    sandbox: bool = True
    api_key_env: str | None = None
    api_secret_env: str | None = None


class ExchangesConfig(BaseModel):
    """Ensemble des exchanges utilises."""

    model_config = ConfigDict(extra="allow")

    primary: str = "binance"
    secondary: list[str] = Field(default_factory=list)

    def settings_for(self, name: str) -> ExchangeConfig:
        """Retourne la config d'un exchange, ou des valeurs par defaut."""
        raw = getattr(self, name, None)
        if isinstance(raw, ExchangeConfig):
            return raw
        if isinstance(raw, dict):
            return ExchangeConfig(**raw)
        return ExchangeConfig()

    @property
    def all_names(self) -> list[str]:
        """Liste ordonnee de tous les exchanges (primaire en tete)."""
        return [self.primary, *[s for s in self.secondary if s != self.primary]]


class DataConfig(BaseModel):
    """Parametres d'ingestion et de stockage des donnees."""

    model_config = ConfigDict(extra="forbid")

    timeframes: list[str] = Field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])
    primary_timeframe: str = "1h"
    history_days: int = Field(default=180, ge=1)
    order_book_depth: int = Field(default=20, ge=1)
    db_url: str = "sqlite:///data/trader.db"

    @model_validator(mode="after")
    def _primary_in_timeframes(self) -> DataConfig:
        if self.primary_timeframe not in self.timeframes:
            raise ValueError(
                f"primary_timeframe={self.primary_timeframe!r} absent de "
                f"timeframes={self.timeframes}"
            )
        return self


class UniverseConfig(BaseModel):
    """Univers d'actifs tradables."""

    model_config = ConfigDict(extra="forbid")

    assets: list[str] = Field(default_factory=lambda: ["ETH/USDT"])
    max_assets: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def _check_size(self) -> UniverseConfig:
        if not self.assets:
            raise ValueError("l'univers d'actifs ne peut pas etre vide")
        if len(self.assets) > self.max_assets:
            raise ValueError(f"{len(self.assets)} actifs > max_assets={self.max_assets}")
        return self


class RegimeConfig(BaseModel):
    """Parametres du detecteur de regime."""

    model_config = ConfigDict(extra="forbid")

    hmm_states: int = Field(default=6, ge=2, le=12)
    retrain_interval_days: int = Field(default=7, ge=1)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    min_agreement: float = Field(default=0.67, ge=0.0, le=1.0)
    lookback_days: int = Field(default=90, ge=10)
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0
    hurst_trend_threshold: float = 0.6
    hurst_revert_threshold: float = 0.4
    crisis_vol_sigma: float = 2.0


class EnsembleConfig(BaseModel):
    """Parametres du meta-modele d'ensemble."""

    model_config = ConfigDict(extra="forbid")

    min_active_strategies: int = Field(default=2, ge=2)
    max_weight_single: float = Field(default=0.40, gt=0.0, le=0.40)
    consensus_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_signal_dispersion: float = Field(default=1.0, gt=0.0)
    shadow_mode_days: int = Field(default=14, ge=1)
    perf_window_days: int = Field(default=30, ge=5)
    flip_flop_penalty: float = Field(default=0.2, ge=0.0, le=1.0)
    diversification_bonus: float = Field(default=0.2, ge=0.0, le=1.0)

    @field_validator("max_weight_single")
    @classmethod
    def _cap_weight(cls, v: float) -> float:
        """Le poids d'une seule strategie est cape a 40 % (anti mono-dependance)."""
        if v > 0.40:
            raise ValueError("max_weight_single ne peut pas depasser 0.40 (limite en dur)")
        return v


class CircuitBreakerConfig(BaseModel):
    """Seuils des circuit breakers."""

    model_config = ConfigDict(extra="forbid")

    max_spread_pct: float = Field(default=2.0, gt=0)
    max_api_latency_sec: float = Field(default=5.0, gt=0)
    max_price_move_5min_pct: float = Field(default=10.0, gt=0)
    pause_duration_min: int = Field(default=30, ge=1)
    max_execution_retries: int = Field(default=3, ge=1)


class UncertainRegimeConfig(BaseModel):
    """Comportement en regime incertain."""

    model_config = ConfigDict(extra="forbid")

    exposure_multiplier: float = Field(default=0.5, gt=0.0, le=1.0)


class CrisisRegimeConfig(BaseModel):
    """Comportement en regime de crise."""

    model_config = ConfigDict(extra="forbid")

    allow_new_positions: bool = False
    close_existing: bool = True
    close_speed: Literal["immediate", "gradual"] = "gradual"

    @field_validator("allow_new_positions")
    @classmethod
    def _no_new_positions_in_crisis(cls, v: bool) -> bool:
        """En crise, aucune nouvelle position n'est autorisee. Non configurable."""
        if v:
            raise ValueError("allow_new_positions doit rester false en regime de crise")
        return False


class RiskConfig(BaseModel):
    """Parametres du risk manager. Bornes par les HARD LIMITS du module."""

    model_config = ConfigDict(extra="forbid")

    max_position_pct: float = Field(default=2.0, gt=0)
    max_exposure_pct: float = Field(default=20.0, gt=0)
    max_drawdown_daily_pct: float = Field(default=3.0, gt=0)
    max_drawdown_weekly_pct: float = Field(default=5.0, gt=0)
    max_drawdown_total_pct: float = Field(default=15.0, gt=0)
    daily_pause_hours: int = Field(default=24, ge=1)
    weekly_pause_hours: int = Field(default=72, ge=1)
    max_orders_per_hour: int = Field(default=10, ge=1)
    max_orders_per_day: int = Field(default=50, ge=1)
    cooldown_same_asset_min: int = Field(default=5, ge=0)
    min_risk_reward: float = Field(default=1.5, gt=0)
    max_stop_distance_pct: float = Field(default=5.0, gt=0)
    max_concurrent_positions: int = Field(default=5, ge=1)
    kelly_fraction: float = Field(default=0.5, gt=0.0, le=0.5)
    circuit_breakers: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    uncertain_regime: UncertainRegimeConfig = Field(default_factory=UncertainRegimeConfig)
    crisis_regime: CrisisRegimeConfig = Field(default_factory=CrisisRegimeConfig)

    @model_validator(mode="after")
    def _enforce_hard_limits(self) -> RiskConfig:
        """Applique les limites en dur. La config ne peut que resserrer, jamais elargir."""
        if self.max_position_pct > HARD_MAX_POSITION_PCT:
            raise ValueError(
                f"max_position_pct={self.max_position_pct} depasse la limite en dur "
                f"{HARD_MAX_POSITION_PCT} % (non configurable)"
            )
        if self.max_drawdown_total_pct > HARD_MAX_DRAWDOWN_TOTAL_PCT:
            raise ValueError(
                f"max_drawdown_total_pct={self.max_drawdown_total_pct} depasse la limite en dur "
                f"{HARD_MAX_DRAWDOWN_TOTAL_PCT} % (non configurable)"
            )
        if self.max_exposure_pct > HARD_MAX_EXPOSURE_PCT:
            raise ValueError(
                f"max_exposure_pct={self.max_exposure_pct} depasse la limite en dur "
                f"{HARD_MAX_EXPOSURE_PCT} %"
            )
        if self.max_exposure_pct < HARD_MIN_EXPOSURE_PCT:
            raise ValueError(
                f"max_exposure_pct={self.max_exposure_pct} sous le plancher "
                f"{HARD_MIN_EXPOSURE_PCT} %"
            )
        if self.max_concurrent_positions > HARD_MAX_CONCURRENT_POSITIONS:
            raise ValueError(
                f"max_concurrent_positions={self.max_concurrent_positions} depasse "
                f"{HARD_MAX_CONCURRENT_POSITIONS}"
            )
        if self.max_drawdown_daily_pct > self.max_drawdown_weekly_pct:
            raise ValueError("le drawdown journalier max doit etre <= au drawdown hebdo max")
        if self.max_drawdown_weekly_pct > self.max_drawdown_total_pct:
            raise ValueError("le drawdown hebdo max doit etre <= au drawdown total max")
        if self.max_orders_per_hour > self.max_orders_per_day:
            raise ValueError("max_orders_per_hour doit etre <= max_orders_per_day")
        return self


class DevilAdvocateConfig(BaseModel):
    """Parametres du module anti-biais. Ne peut pas etre desactive."""

    model_config = ConfigDict(extra="forbid")

    abort_threshold: float = Field(default=0.7, gt=0.0, le=1.0)
    reduce_threshold: float = Field(default=0.4, gt=0.0, le=1.0)
    reduce_factor: float = Field(default=0.5, gt=0.0, le=1.0)
    enabled: bool = True

    @field_validator("enabled")
    @classmethod
    def _always_enabled(cls, v: bool) -> bool:
        """Le DevilAdvocate ne peut pas etre desactive par configuration."""
        if not v:
            raise ValueError(
                "devil_advocate.enabled=false est interdit : ce module est non desactivable"
            )
        return True

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> DevilAdvocateConfig:
        if self.reduce_threshold >= self.abort_threshold:
            raise ValueError("reduce_threshold doit etre strictement < abort_threshold")
        return self


class DecayDetectionConfig(BaseModel):
    """Parametres de detection de decay des strategies."""

    model_config = ConfigDict(extra="forbid")

    sharpe_decay_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    min_hit_rate_14d: float = Field(default=0.45, ge=0.0, le=1.0)
    min_profit_factor_14d: float = Field(default=1.0, ge=0.0)
    max_consecutive_losses_alert: int = Field(default=5, ge=1)
    max_consecutive_losses: int = Field(default=8, ge=1)
    check_interval_hours: int = Field(default=6, ge=1)
    min_trades_for_verdict: int = Field(default=10, ge=1)
    dead_period_days: int = Field(default=30, ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> DecayDetectionConfig:
        if self.max_consecutive_losses_alert > self.max_consecutive_losses:
            raise ValueError("le seuil d'alerte doit etre <= au seuil de desactivation")
        return self


class RetrainingConfig(BaseModel):
    """Parametres du retraining walk-forward."""

    model_config = ConfigDict(extra="forbid")

    train_window_days: int = Field(default=90, ge=10)
    validation_window_days: int = Field(default=14, ge=1)
    test_window_days: int = Field(default=7, ge=1)
    purge_gap_days: int = Field(default=2, ge=0)
    min_oos_ratio: float = Field(default=0.70, gt=0.0, le=1.0)
    schedule: Literal["weekly", "on_decay_trigger", "manual"] = "weekly"
    max_params_per_strategy: int = Field(default=6, ge=1)
    artifacts_dir: str = "artifacts/retraining"


class ExecutionConfig(BaseModel):
    """Parametres du moteur d'execution."""

    model_config = ConfigDict(extra="forbid")

    default_order_type: Literal["limit", "market"] = "limit"
    fill_timeout_sec: int = Field(default=60, ge=1)
    min_fill_pct: float = Field(default=0.80, gt=0.0, le=1.0)
    max_retries: int = Field(default=3, ge=1)
    retry_backoff_sec: float = Field(default=1.0, gt=0)
    slippage_model: Literal["spread_plus_impact", "fixed_bps"] = "spread_plus_impact"
    fixed_slippage_bps: float = Field(default=5.0, ge=0)
    taker_fee_bps: float = Field(default=7.5, ge=0)
    maker_fee_bps: float = Field(default=2.0, ge=0)
    simulate_latency: bool = True
    latency_ms_range: tuple[int, int] = (100, 500)
    partial_fill_volume_pct: float = Field(default=5.0, gt=0)


class MonitoringConfig(BaseModel):
    """Parametres de monitoring et d'alerting."""

    model_config = ConfigDict(extra="forbid")

    prometheus_port: int = Field(default=9090, ge=1024, le=65535)
    prometheus_enabled: bool = True
    alert_channels: list[str] = Field(default_factory=lambda: ["telegram"])
    alert_urls_env: str = "TRADER_ALERT_URLS"
    alert_rate_limit_info_sec: int = Field(default=300, ge=0)
    daily_report_hour: int = Field(default=8, ge=0, le=23)


class PaperTradingConfig(BaseModel):
    """Criteres statistiques de passage en live."""

    model_config = ConfigDict(extra="forbid")

    min_days_before_live: int = Field(default=30, ge=30)
    min_sharpe_for_live: float = Field(default=0.8, ge=0.8)
    min_trades_for_live: int = Field(default=50, ge=50)
    max_drawdown_for_live_pct: float = Field(default=10.0, gt=0, le=10.0)
    min_profit_factor_for_live: float = Field(default=1.2, ge=1.2)
    max_backtest_divergence_pct: float = Field(default=30.0, gt=0)
    max_slippage_divergence_pct: float = Field(default=50.0, gt=0)
    max_live_capital: float = Field(default=500.0, gt=0)


class KillSwitchConfig(BaseModel):
    """Parametres du kill switch (processus separe)."""

    model_config = ConfigDict(extra="forbid")

    sentinel_path: str = "/tmp/trader_kill"
    http_enabled: bool = True
    http_host: str = "127.0.0.1"
    http_port: int = Field(default=9091, ge=1024, le=65535)
    watchdog_timeout_sec: int = Field(default=60, ge=10)
    check_interval_sec: int = Field(default=30, ge=5)


class Settings(BaseModel):
    """Configuration complete du systeme."""

    model_config = ConfigDict(extra="forbid")

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    exchanges: ExchangesConfig = Field(default_factory=ExchangesConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    devil_advocate: DevilAdvocateConfig = Field(default_factory=DevilAdvocateConfig)
    decay_detection: DecayDetectionConfig = Field(default_factory=DecayDetectionConfig)
    retraining: RetrainingConfig = Field(default_factory=RetrainingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    paper_trading: PaperTradingConfig = Field(default_factory=PaperTradingConfig)
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)

    @property
    def is_live(self) -> bool:
        """Vrai si le systeme tourne avec de l'argent reel."""
        return self.general.mode is Mode.LIVE

    @model_validator(mode="after")
    def _live_guardrails(self) -> Settings:
        """Le mode live impose des garde-fous supplementaires."""
        if (
            self.general.mode is Mode.LIVE
            and self.general.initial_capital > self.paper_trading.max_live_capital
        ):
            raise ValueError(
                f"capital live {self.general.initial_capital} > plafond "
                f"{self.paper_trading.max_live_capital} (paper_trading.max_live_capital)"
            )
        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusionne recursivement deux dictionnaires (override gagne)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_toml(path: str | Path) -> dict[str, Any]:
    """Charge un fichier TOML en dictionnaire."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"fichier de configuration introuvable : {file_path}")
    with file_path.open("rb") as handle:
        return tomllib.load(handle)


def load_settings(
    default_path: str | Path = "config/default.toml",
    override_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Charge la configuration : defaut, puis override fichier, puis override dict.

    Args:
        default_path: fichier TOML de base.
        override_path: fichier TOML d'override (paper.toml / live.toml).
        overrides: dictionnaire d'override applique en dernier (CLI, tests).

    Returns:
        Un objet Settings valide, avec les hard limits verifiees.
    """
    raw = load_toml(default_path)
    if override_path is not None:
        raw = _deep_merge(raw, load_toml(override_path))
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Settings(**raw)

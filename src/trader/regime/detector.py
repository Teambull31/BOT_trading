"""Detection de regime : combinaison HMM + clustering + regles quantitatives.

Trois methodes independantes votent. On ne fait confiance a une conclusion que
si les methodes sont d'accord ET confiantes :

    confidence < min_confidence  OU  agreement < min_agreement  ->  UNCERTAIN

Un regime UNCERTAIN n'est pas une panne : c'est une information exploitable, qui
fait diviser l'exposition par deux cote risk manager. Le systeme a le droit de
dire "je ne sais pas".

La crise court-circuite le vote : si la volatilite atteint un niveau extreme,
le regime est CRISIS meme si les autres methodes voient autre chose. En cas de
doute sur un risque de ruine, on tranche toujours du cote prudent.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from trader.config import RegimeConfig
from trader.logging_setup import get_logger
from trader.models import Regime, RegimeState
from trader.regime.trend import TrendState, classify_trend
from trader.regime.volatility import VolatilityState, VolRegime, classify_volatility
from trader.utils.math_utils import EPSILON
from trader.utils.time_utils import annualization_factor, utc_now

log = get_logger(__name__)

HMM_FEATURES: tuple[str, ...] = ("log_return", "abs_return", "vol", "volume_ratio")
CLUSTER_FEATURES: tuple[str, ...] = (
    "log_return_20",
    "realized_vol_7d",
    "adx",
    "hurst",
    "bb_width",
)


class Direction(str, Enum):
    """Axe directionnel d'un regime, independamment du niveau de volatilite."""

    BULL = "bull"
    BEAR = "bear"
    RANGE = "range"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


_DIRECTION_OF: dict[Regime, Direction] = {
    Regime.BULL_LOW_VOL: Direction.BULL,
    Regime.BULL_HIGH_VOL: Direction.BULL,
    Regime.BEAR_LOW_VOL: Direction.BEAR,
    Regime.BEAR_HIGH_VOL: Direction.BEAR,
    Regime.RANGE_BOUND: Direction.RANGE,
    Regime.CRISIS: Direction.CRISIS,
    Regime.UNCERTAIN: Direction.UNKNOWN,
}


def direction_of(regime: Regime) -> Direction:
    """Axe directionnel porte par un label de regime."""
    return _DIRECTION_OF.get(regime, Direction.UNKNOWN)


def compose_regime(direction: Direction, volatility: VolatilityState) -> Regime:
    """Recompose un label de regime a partir de la direction votee et de la vol mesuree.

    La direction est CONTESTEE : les trois methodes votent. Le niveau de
    volatilite, lui, est MESURE : inutile de le faire voter, on lit la mesure.
    """
    high_vol = volatility.regime in (VolRegime.HIGH, VolRegime.EXTREME)
    if direction is Direction.BULL:
        return Regime.BULL_HIGH_VOL if high_vol else Regime.BULL_LOW_VOL
    if direction is Direction.BEAR:
        return Regime.BEAR_HIGH_VOL if high_vol else Regime.BEAR_LOW_VOL
    if direction is Direction.RANGE:
        return Regime.RANGE_BOUND
    if direction is Direction.CRISIS:
        return Regime.CRISIS
    return Regime.UNCERTAIN


def _required_votes(n_methods: int, min_agreement: float) -> int:
    """Nombre de votes concordants exiges pour valider un regime.

    On raisonne en nombre de votes plutot qu'en fraction : avec trois methodes,
    "2/3 d'accord" vaut 0.6667, ce qui echouerait bêtement contre un seuil
    configure a 0.67. La tolerance absorbe cet ecart d'arrondi, jamais un vote.
    """
    required = math.ceil(min_agreement * n_methods - 0.05)
    return max(2, min(n_methods, required))


@contextmanager
def _quiet_hmmlearn() -> Iterator[None]:
    """Silence les warnings de convergence bruyants de hmmlearn.

    Ces messages ne sont pas ignores : un modele qui ne converge pas produit une
    matrice de transition degeneree, detectee et rejetee explicitement ensuite.
    """
    hmm_logger = logging.getLogger("hmmlearn")
    previous = hmm_logger.level
    hmm_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        hmm_logger.setLevel(previous)


@dataclass(slots=True)
class MethodVote:
    """Vote d'une methode de detection."""

    method: str
    regime: Regime
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)


class RegimeDetector:
    """Classifie le regime de marche courant par vote de trois methodes.

    Methode 1 : Hidden Markov Model (hmmlearn) sur returns + volatilite.
    Methode 2 : clustering K-Means, dont les clusters ne sont PAS labelises
                a priori mais interpretes a posteriori via leurs centroides.
    Methode 3 : regles quantitatives explicites (ADX, Bollinger, Hurst, vol).

    Le HMM et le clustering sont optionnels : si `scikit-learn`/`hmmlearn` ne
    convergent pas ou manquent de donnees, le detecteur continue avec les
    methodes restantes et l'agreement_score en tient compte.
    """

    def __init__(self, config: RegimeConfig, timeframe: str = "1h") -> None:
        self.config = config
        self.timeframe = timeframe
        self.hmm_model: Any | None = None
        self.hmm_scaler: Any | None = None
        self.hmm_state_map: dict[int, Regime] = {}
        self.cluster_model: Any | None = None
        self.cluster_scaler: Any | None = None
        self.cluster_map: dict[int, Regime] = {}
        self.last_fit: datetime | None = None
        self.last_state: RegimeState | None = None
        self._fit_rows: int = 0

    # ------------------------------------------------------------ training

    def needs_retrain(self, now: datetime | None = None) -> bool:
        """Vrai si les modeles doivent etre reentraines (walk-forward periodique)."""
        if self.last_fit is None:
            return True
        reference = now or utc_now()
        return reference - self.last_fit >= timedelta(days=self.config.retrain_interval_days)

    def fit(self, features: pd.DataFrame, now: datetime | None = None) -> dict[str, Any]:
        """Entraine le HMM et le clustering sur l'historique fourni.

        Ne leve jamais : un modele qui ne converge pas est simplement desactive,
        le detecteur retombe sur les methodes restantes.
        """
        report: dict[str, Any] = {"hmm": "skipped", "cluster": "skipped", "rows": 0}
        if features is None or features.empty:
            return report

        matrix = self._hmm_matrix(features)
        report["rows"] = len(matrix)
        self._fit_rows = len(matrix)
        if len(matrix) >= 100:
            report["hmm"] = self._fit_hmm(matrix)
        cluster_matrix = self._cluster_matrix(features)
        if len(cluster_matrix) >= 60:
            report["cluster"] = self._fit_clusters(cluster_matrix)

        self.last_fit = now or utc_now()
        log.info("regime_models_fitted", **report)
        return report

    def _fit_hmm(self, matrix: pd.DataFrame) -> str:
        """Entraine un GaussianHMM et interprete ses etats a posteriori.

        Les observations sont standardisees : les returns (~1e-3) et la
        volatilite annualisee (~1) different de trois ordres de grandeur, et un
        HMM gaussien a covariance diagonale degenere sur des echelles pareilles.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            log.warning("hmmlearn_missing")
            return "unavailable"

        n_states = min(self.config.hmm_states, max(2, len(matrix) // 60))
        try:
            scaler = StandardScaler().fit(matrix.to_numpy(dtype=float))
            values = scaler.transform(matrix.to_numpy(dtype=float))
            with _quiet_hmmlearn():
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",
                    n_iter=300,
                    random_state=42,
                    tol=1e-4,
                    min_covar=1e-3,
                )
                model.fit(values)
                states = model.predict(values)
        except (ValueError, RuntimeError) as exc:
            log.warning("hmm_fit_failed", error=str(exc))
            self.hmm_model = None
            return f"failed: {exc}"

        if not np.all(np.isfinite(model.transmat_)):
            log.warning("hmm_degenerate_transmat")
            self.hmm_model = None
            return "failed: matrice de transition degeneree"

        self.hmm_model = model
        self.hmm_scaler = scaler
        self.hmm_state_map = self._label_states(matrix, states)
        return f"ok ({n_states} etats)"

    def _fit_clusters(self, matrix: pd.DataFrame) -> str:
        """Entraine un K-Means dont les clusters emergent des donnees."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            log.warning("sklearn_missing")
            return "unavailable"

        values = matrix.to_numpy(dtype=float)
        n_clusters = min(max(3, self.config.hmm_states - 2), max(2, len(values) // 30))
        try:
            scaler = StandardScaler().fit(values)
            scaled = scaler.transform(values)
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(scaled)
        except (ValueError, RuntimeError) as exc:
            log.warning("cluster_fit_failed", error=str(exc))
            self.cluster_model = None
            return f"failed: {exc}"

        self.cluster_scaler = scaler
        self.cluster_model = model
        self.cluster_map = self._label_states(matrix, model.labels_)
        return f"ok ({n_clusters} clusters)"

    def _label_states(self, matrix: pd.DataFrame, states: np.ndarray) -> dict[int, Regime]:
        """Interprete a posteriori chaque etat/cluster via son comportement moyen.

        On ne decide PAS a priori que "l'etat 3 est un bull market" : on regarde
        le rendement moyen et la volatilite moyenne des periodes classees dans
        cet etat, puis on lui colle le label qui correspond.
        """
        return_column = "log_return" if "log_return" in matrix.columns else matrix.columns[0]
        vol_column = "vol" if "vol" in matrix.columns else matrix.columns[-1]

        frame = matrix.assign(_state=states)
        grouped = frame.groupby("_state").agg(
            mean_return=(return_column, "mean"), mean_vol=(vol_column, "mean")
        )
        if grouped.empty:
            return {}

        vol_high = float(grouped["mean_vol"].median())
        vol_extreme = float(grouped["mean_vol"].quantile(0.9))
        return_scale = float(frame[return_column].std(ddof=1)) or EPSILON

        mapping: dict[int, Regime] = {}
        for state, row in grouped.iterrows():
            mean_return = float(row["mean_return"])
            mean_vol = float(row["mean_vol"])
            high_vol = mean_vol > vol_high
            directional = abs(mean_return) > 0.10 * return_scale

            if mean_vol >= vol_extreme and mean_return < 0:
                mapping[int(state)] = Regime.CRISIS
            elif not directional:
                mapping[int(state)] = Regime.RANGE_BOUND
            elif mean_return > 0:
                mapping[int(state)] = Regime.BULL_HIGH_VOL if high_vol else Regime.BULL_LOW_VOL
            else:
                mapping[int(state)] = Regime.BEAR_HIGH_VOL if high_vol else Regime.BEAR_LOW_VOL
        return mapping

    # ------------------------------------------------------------ detection

    def detect(
        self,
        features: pd.DataFrame,
        ohlcv: pd.DataFrame | None = None,
        now: datetime | None = None,
    ) -> RegimeState:
        """Determine le regime courant par vote des trois methodes."""
        timestamp = now or (features.index[-1].to_pydatetime() if len(features) else utc_now())
        if features is None or features.empty:
            return self._uncertain("aucune donnee", timestamp, {})

        volatility = self._volatility_state(features)
        votes: list[MethodVote] = []

        hmm_vote = self._hmm_vote(features)
        if hmm_vote is not None:
            votes.append(hmm_vote)
        cluster_vote = self._cluster_vote(features)
        if cluster_vote is not None:
            votes.append(cluster_vote)
        votes.append(self._rules_vote(features, volatility, ohlcv))

        transition = self._transition_probability(features, votes)
        if len(votes) < 2 and not volatility.is_crisis_level:
            return self._uncertain(
                "une seule methode disponible : aucune corroboration possible",
                timestamp,
                {vote.method: vote.regime.value for vote in votes},
                transition=transition,
            )

        # Court-circuit crise : la prudence prime sur le consensus.
        if volatility.is_crisis_level:
            state = RegimeState(
                regime=Regime.CRISIS,
                confidence=max(0.7, min(1.0, abs(volatility.zscore) / 4.0)),
                agreement_score=1.0,
                transition_probability=transition,
                timestamp=timestamp,
                method_votes={vote.method: vote.regime.value for vote in votes},
                details={
                    "trigger": "volatility_extreme",
                    "vol_zscore": round(volatility.zscore, 3),
                    "vol_ratio": round(volatility.ratio, 3),
                    "vol_shock": volatility.is_shock,
                },
            )
            self.last_state = state
            log.warning("regime_crisis", **state.to_dict())
            return state

        # L'accord se mesure sur la DIRECTION : deux methodes qui voient toutes
        # deux un marche haussier sont d'accord, meme si elles divergent sur le
        # niveau de volatilite (lequel est mesure, pas vote).
        counts = Counter(direction_of(vote.regime) for vote in votes)
        winner_direction, winner_count = counts.most_common(1)[0]
        winner = compose_regime(winner_direction, volatility)
        agreement = winner_count / len(votes)
        supporting = [
            vote.confidence for vote in votes if direction_of(vote.regime) is winner_direction
        ]
        # L'accord penalise la confiance sans l'ecraser : 2 methodes sur 3 qui
        # convergent avec une forte conviction restent une information utile.
        confidence = float(np.mean(supporting)) * (0.5 + 0.5 * agreement)

        method_votes = {vote.method: vote.regime.value for vote in votes}
        details = {
            "direction": winner_direction.value,
            "vol_regime": volatility.regime.value,
            "vol_ratio": round(volatility.ratio, 3),
            "vol_zscore": round(volatility.zscore, 3),
            "methods": len(votes),
            "raw_confidence": round(confidence, 3),
        }

        if confidence < self.config.min_confidence or winner_count < _required_votes(
            len(votes), self.config.min_agreement
        ):
            state = self._uncertain(
                f"confidence={confidence:.2f} agreement={agreement:.2f}",
                timestamp,
                method_votes,
                details | {"proposed_regime": winner.value},
                agreement=agreement,
                confidence=confidence,
                transition=transition,
            )
            self.last_state = state
            return state

        state = RegimeState(
            regime=winner,
            confidence=float(min(1.0, confidence)),
            agreement_score=agreement,
            transition_probability=transition,
            timestamp=timestamp,
            method_votes=method_votes,
            details=details,
        )
        self.last_state = state
        log.info("regime_detected", **state.to_dict())
        return state

    @property
    def smoothing_bars(self) -> int:
        """Nombre de bougies sur lesquelles lisser les votes.

        Un regime est une propriete PERSISTANTE du marche. Classer la derniere
        bougie isolement revient a classer du bruit : on agrege donc les
        posteriors sur la derniere journee de donnees.
        """
        bars_per_day = annualization_factor(self.timeframe) / 365.0
        return max(3, int(round(bars_per_day)))

    def _hmm_vote(self, features: pd.DataFrame) -> MethodVote | None:
        """Vote du HMM, moyenne sur la fenetre de lissage."""
        if self.hmm_model is None or not self.hmm_state_map:
            return None
        matrix = self._hmm_matrix(features, bounded=True)
        if matrix.empty:
            return None
        try:
            values = matrix.to_numpy(dtype=float)
            if self.hmm_scaler is not None:
                values = self.hmm_scaler.transform(values)
            with _quiet_hmmlearn():
                posteriors = self.hmm_model.predict_proba(values)
        except (ValueError, RuntimeError) as exc:
            log.warning("hmm_predict_failed", error=str(exc))
            return None
        window = posteriors[-self.smoothing_bars :]
        averaged = window.mean(axis=0)

        # On agrege les posteriors PAR REGIME : deux etats HMM distincts peuvent
        # decrire le meme regime, et leurs probabilites doivent s'additionner.
        by_regime: dict[Regime, float] = {}
        for state, probability in enumerate(averaged):
            regime = self.hmm_state_map.get(state, Regime.UNCERTAIN)
            by_regime[regime] = by_regime.get(regime, 0.0) + float(probability)
        regime, confidence = max(by_regime.items(), key=lambda item: item[1])
        dominant_state = int(np.argmax(averaged))
        return MethodVote(
            method="hmm",
            regime=regime,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            details={"state": dominant_state, "bars": len(window)},
        )

    def _cluster_vote(self, features: pd.DataFrame) -> MethodVote | None:
        """Vote du clustering, majoritaire sur la fenetre de lissage."""
        if self.cluster_model is None or self.cluster_scaler is None or not self.cluster_map:
            return None
        matrix = self._cluster_matrix(features)
        if matrix.empty:
            return None
        window = matrix.to_numpy(dtype=float)[-self.smoothing_bars :]
        try:
            scaled = self.cluster_scaler.transform(window)
            labels = self.cluster_model.predict(scaled)
        except (ValueError, RuntimeError) as exc:
            log.warning("cluster_predict_failed", error=str(exc))
            return None

        regimes = [self.cluster_map.get(int(label), Regime.UNCERTAIN) for label in labels]
        counts = Counter(regimes)
        regime, count = counts.most_common(1)[0]
        stability = count / len(regimes)
        confidence = float(np.clip(stability, 0.2, 0.95))
        return MethodVote(
            method="cluster",
            regime=regime,
            confidence=confidence,
            details={"stability": round(stability, 3), "bars": len(regimes)},
        )

    def _rules_vote(
        self,
        features: pd.DataFrame,
        volatility: VolatilityState,
        ohlcv: pd.DataFrame | None,
    ) -> MethodVote:
        """Vote des regles quantitatives explicites."""
        close = ohlcv["close"] if ohlcv is not None and "close" in ohlcv else None
        trend = classify_trend(
            features,
            adx_trend_threshold=self.config.adx_trend_threshold,
            adx_range_threshold=self.config.adx_range_threshold,
            hurst_trend_threshold=self.config.hurst_trend_threshold,
            hurst_revert_threshold=self.config.hurst_revert_threshold,
            close=close,
        )
        high_vol = volatility.regime in (VolRegime.HIGH, VolRegime.EXTREME)

        if trend.state is TrendState.UPTREND:
            regime = Regime.BULL_HIGH_VOL if high_vol else Regime.BULL_LOW_VOL
        elif trend.state is TrendState.DOWNTREND:
            regime = Regime.BEAR_HIGH_VOL if high_vol else Regime.BEAR_LOW_VOL
        elif trend.state in (TrendState.RANGE, TrendState.MEAN_REVERTING):
            regime = Regime.RANGE_BOUND
        else:
            regime = Regime.UNCERTAIN

        confidence = float(np.clip(0.35 + 0.5 * trend.strength, 0.2, 0.95))
        return MethodVote(
            method="rules",
            regime=regime,
            confidence=confidence,
            details={
                "trend": trend.state.value,
                "adx": round(trend.adx, 2),
                "hurst": round(trend.hurst, 3),
                "vol_regime": volatility.regime.value,
            },
        )

    def _transition_probability(self, features: pd.DataFrame, votes: list[MethodVote]) -> float:
        """Probabilite que le regime change bientot.

        Source principale : la matrice de transition du HMM (1 - proba de rester).
        A defaut : l'instabilite recente des votes de regles et la derive de
        volatilite, qui montent avant les ruptures.
        """
        if self.hmm_model is not None and self.hmm_state_map:
            hmm_vote = next((vote for vote in votes if vote.method == "hmm"), None)
            if hmm_vote is not None and "state" in hmm_vote.details:
                state = int(hmm_vote.details["state"])
                transmat = np.asarray(self.hmm_model.transmat_)
                if 0 <= state < transmat.shape[0]:
                    return float(np.clip(1.0 - transmat[state, state], 0.0, 1.0))

        vol_column = "vol_ratio_short_long"
        if vol_column in features.columns:
            recent = features[vol_column].dropna().tail(20)
            if len(recent) > 5:
                drift = abs(float(recent.iloc[-1]) - float(recent.mean()))
                return float(np.clip(drift, 0.0, 1.0))
        return 0.2

    def _uncertain(
        self,
        reason: str,
        timestamp: datetime,
        method_votes: dict[str, str],
        details: dict[str, Any] | None = None,
        agreement: float = 0.0,
        confidence: float = 0.0,
        transition: float = 0.5,
    ) -> RegimeState:
        """Construit un etat UNCERTAIN documente."""
        payload = {"reason": reason} | {k: v for k, v in (details or {}).items() if k != "reason"}
        state = RegimeState(
            regime=Regime.UNCERTAIN,
            confidence=confidence,
            agreement_score=agreement,
            transition_probability=transition,
            timestamp=timestamp,
            method_votes=method_votes,
            details=payload,
        )
        log.info("regime_uncertain", **{k: str(v) for k, v in payload.items()})
        return state

    # -------------------------------------------------------------- matrices

    @property
    def inference_bars(self) -> int:
        """Longueur de l'historique utilise pour INFERER (pas pour entrainer).

        Le filtre forward d'un HMM oublie exponentiellement le passe lointain :
        au-dela de quelques centaines de barres, les posteriors ne bougent plus.
        Borner la fenetre evite un cout quadratique quand le detecteur tourne a
        chaque bougie d'un backtest, sans changer la decision.
        """
        return max(200, self.smoothing_bars * 10)

    def _hmm_matrix(self, features: pd.DataFrame, bounded: bool = False) -> pd.DataFrame:
        """Matrice d'observation du HMM : returns, |returns|, volatilite, volume."""
        if "log_return" not in features.columns:
            return pd.DataFrame()
        if bounded:
            features = features.tail(self.inference_bars)
        periods = annualization_factor(self.timeframe)
        window = max(5, int(24 * (periods / 8760.0) * 7))
        frame = pd.DataFrame(index=features.index)
        frame["log_return"] = features["log_return"]
        frame["abs_return"] = features["log_return"].abs()
        vol_column = next(
            (c for c in ("realized_vol_7d", "realized_vol_30d") if c in features.columns), None
        )
        frame["vol"] = (
            features[vol_column]
            if vol_column
            else features["log_return"].rolling(window, min_periods=3).std(ddof=1)
        )
        frame["volume_ratio"] = (
            features["volume_ratio"] if "volume_ratio" in features.columns else 1.0
        )
        return frame.replace([np.inf, -np.inf], np.nan).dropna()

    def _cluster_matrix(self, features: pd.DataFrame, bounded: bool = False) -> pd.DataFrame:
        """Matrice de clustering : features de comportement, pas de prix brut."""
        available = [c for c in CLUSTER_FEATURES if c in features.columns]
        if len(available) < 3:
            return pd.DataFrame()
        if bounded:
            features = features.tail(self.inference_bars)
        return features[available].replace([np.inf, -np.inf], np.nan).dropna()

    def _volatility_state(self, features: pd.DataFrame) -> VolatilityState:
        """Etat de volatilite calcule sur les returns disponibles."""
        returns = (
            features["log_return"] if "log_return" in features.columns else pd.Series(dtype=float)
        )
        bars_per_day = annualization_factor(self.timeframe) / 365.0
        long_window = max(30, int(self.config.lookback_days * bars_per_day))
        # On ne garde que la fenetre utile : sinon le cout devient quadratique
        # quand le detecteur est appele a chaque bougie d'un backtest.
        returns = returns.tail(long_window * 2)
        return classify_volatility(
            returns,
            timeframe=self.timeframe,
            short_window=max(10, int(7 * bars_per_day)),
            long_window=long_window,
            crisis_sigma=self.config.crisis_vol_sigma,
        )

    def fit_detect(self, features: pd.DataFrame, ohlcv: pd.DataFrame | None = None) -> RegimeState:
        """Entraine si necessaire puis detecte (usage courant dans l'orchestrateur)."""
        if self.needs_retrain():
            self.fit(features)
        return self.detect(features, ohlcv)

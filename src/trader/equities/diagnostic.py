"""Diagnostic de l'etat du marche, a la derniere date disponible.

Objectif : repondre a une question simple — est-ce le moment de pousser le
curseur de risque, ou de le retirer ? Le diagnostic ne predit rien. Il constate
des faits mesurables et les traduit en niveau de prudence.

Cinq mesures, choisies parce qu'elles decrivent des choses differentes :

1. POSITION DANS LA TENDANCE : part des titres au-dessus de leur moyenne
   200 seances. C'est la mesure la plus robuste de "sommes-nous en marche
   haussier" — celle qui pilote deja les entrees du systeme.
2. DRAWDOWN EN COURS : distance au plus haut des 12 derniers mois. Un titre a
   -30 % de son sommet n'est pas dans la meme situation qu'un titre au plus haut,
   meme si les deux sont au-dessus de leur moyenne longue.
3. REGIME DE VOLATILITE : volatilite realisee 20 seances rapportee a la
   volatilite d'un an. Au-dessus de 1,5, les mouvements deviennent trop rapides
   pour des stops calibres sur le passe recent.
4. CORRELATION MOYENNE : quand tout se met a bouger ensemble, la
   diversification cesse de proteger exactement au moment ou l'on en a besoin.
5. LARGEUR DU MARCHE : la hausse est-elle portee par tous les titres ou par un
   seul ? Une hausse etroite est fragile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from trader.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SymbolDiagnostic:
    """Etat d'un titre a la derniere date disponible."""

    symbol: str
    last_close: float
    above_sma200: bool
    distance_sma200_pct: float
    drawdown_from_high_pct: float
    vol_ratio: float
    trend_slope_pct: float

    def to_row(self) -> dict:
        """Ligne de tableau lisible."""
        return {
            "titre": self.symbol,
            "cours": round(self.last_close, 2),
            "vs_SMA200_%": round(self.distance_sma200_pct, 1),
            "sous_plus_haut_%": round(self.drawdown_from_high_pct, 1),
            "vol_20j/1an": round(self.vol_ratio, 2),
            "pente_3m_%": round(self.trend_slope_pct, 1),
            "en_tendance": "oui" if self.above_sma200 else "non",
        }


@dataclass(slots=True)
class MarketDiagnostic:
    """Synthese de l'etat du marche."""

    as_of: date
    symbols: list[SymbolDiagnostic] = field(default_factory=list)
    breadth_pct: float = 0.0
    mean_correlation: float = 0.0
    mean_vol_ratio: float = 0.0
    mean_drawdown_pct: float = 0.0
    warnings: list[str] = field(default_factory=list)
    verdict: str = ""
    recommended_profile: str = "equilibre"
    caution_score: int = 0
    max_caution: int = 9
    near_threshold: str = ""
    benchmark: SymbolDiagnostic | None = None

    def table(self) -> pd.DataFrame:
        """Tableau par titre."""
        return pd.DataFrame([symbol.to_row() for symbol in self.symbols])

    def render(self) -> str:
        """Rapport lisible."""
        lines = [
            f"Etat du marche au {self.as_of}",
            "",
            self.table().to_string(index=False),
            "",
            f"  Titres en tendance haussiere : {self.breadth_pct:.0f} % de l'univers",
            f"  Correlation moyenne          : {self.mean_correlation:+.2f}",
            f"  Regime de volatilite         : {self.mean_vol_ratio:.2f} "
            f"(1.00 = volatilite normale)",
            f"  Recul moyen depuis les plus hauts : {self.mean_drawdown_pct:.1f} %",
        ]
        if self.benchmark is not None:
            lines.append(
                f"  Marche large ({self.benchmark.symbol})      : "
                f"{self.benchmark.distance_sma200_pct:+.1f} % vs SMA200, "
                f"{self.benchmark.drawdown_from_high_pct:.1f} % sous son plus haut"
            )
        lines.append(f"  Score de prudence            : {self.caution_score}/{self.max_caution}")
        if self.warnings:
            lines.append("\n  Signaux de prudence :")
            lines.extend(f"    - {warning}" for warning in self.warnings)
        else:
            lines.append("\n  Aucun signal de prudence particulier.")
        lines.append(f"\n  VERDICT : {self.verdict}")
        if self.near_threshold:
            lines.append(f"  Nuance  : {self.near_threshold}")
        lines.append(f"  Profil coherent avec cet etat : {self.recommended_profile.upper()}")
        return "\n".join(lines)


def diagnose(
    frames: dict[str, pd.DataFrame],
    lookback_days: int = 252,
    benchmark_frame: pd.DataFrame | None = None,
    benchmark_symbol: str = "SPY",
) -> MarketDiagnostic:
    """Analyse l'etat courant du marche a partir des cours disponibles.

    Args:
        benchmark_frame: cours d'un indice large. L'univers peut aller mal
            pendant que le marche va bien (probleme sectoriel) ou l'inverse
            (correction generale) : distinguer les deux change la conclusion.
    """
    if not frames:
        raise ValueError("aucune donnee a diagnostiquer")

    as_of = max(frame.index[-1].date() for frame in frames.values())
    diagnostics: list[SymbolDiagnostic] = []
    returns_by_symbol: dict[str, pd.Series] = {}

    for symbol, frame in frames.items():
        if len(frame) < 210:
            continue
        returns_by_symbol[symbol] = np.log(frame["close"] / frame["close"].shift(1)).dropna()
        diagnostics.append(_diagnose_symbol(symbol, frame, lookback_days))

    if not diagnostics:
        raise ValueError("historique insuffisant pour diagnostiquer le marche")

    breadth = sum(1 for item in diagnostics if item.above_sma200) / len(diagnostics) * 100.0
    mean_vol_ratio = float(np.mean([item.vol_ratio for item in diagnostics]))
    mean_drawdown = float(np.mean([item.drawdown_from_high_pct for item in diagnostics]))
    correlation = _mean_correlation(returns_by_symbol)

    benchmark_diagnostic = (
        _diagnose_symbol(benchmark_symbol, benchmark_frame, lookback_days)
        if benchmark_frame is not None and len(benchmark_frame) >= 210
        else None
    )

    diagnostic = MarketDiagnostic(
        as_of=as_of,
        benchmark=benchmark_diagnostic,
        symbols=sorted(diagnostics, key=lambda item: item.drawdown_from_high_pct),
        breadth_pct=breadth,
        mean_correlation=correlation,
        mean_vol_ratio=mean_vol_ratio,
        mean_drawdown_pct=mean_drawdown,
    )
    _assess(diagnostic)
    log.info(
        "market_diagnostic",
        as_of=str(as_of),
        breadth_pct=round(breadth, 1),
        vol_ratio=round(mean_vol_ratio, 2),
        verdict=diagnostic.verdict,
    )
    return diagnostic


def _diagnose_symbol(symbol: str, frame: pd.DataFrame, lookback_days: int) -> SymbolDiagnostic:
    """Calcule l'etat d'un titre a la derniere date disponible."""
    close = frame["close"]
    sma200 = close.rolling(200, min_periods=200).mean()
    last_close = float(close.iloc[-1])
    last_sma = float(sma200.iloc[-1])
    highest = float(close.tail(lookback_days).max())

    returns = np.log(close / close.shift(1)).dropna()
    vol_short = float(returns.tail(20).std(ddof=1)) if len(returns) > 25 else 0.0
    vol_long = float(returns.tail(lookback_days).std(ddof=1)) if len(returns) > 60 else 0.0
    vol_ratio = vol_short / vol_long if vol_long > 1e-12 else 1.0

    slope_window = min(63, len(close) - 1)
    slope = (last_close / float(close.iloc[-slope_window]) - 1.0) * 100.0

    return SymbolDiagnostic(
        symbol=symbol,
        last_close=last_close,
        above_sma200=last_close > last_sma,
        distance_sma200_pct=(last_close / last_sma - 1.0) * 100.0,
        drawdown_from_high_pct=(last_close / highest - 1.0) * 100.0,
        vol_ratio=vol_ratio,
        trend_slope_pct=slope,
    )


def _mean_correlation(returns_by_symbol: dict[str, pd.Series], window: int = 60) -> float:
    """Correlation moyenne des rendements recents entre titres."""
    if len(returns_by_symbol) < 2:
        return 0.0
    frame = pd.DataFrame(
        {symbol: series.tail(window) for symbol, series in returns_by_symbol.items()}
    ).dropna()
    if len(frame) < 20:
        return 0.0
    matrix = frame.corr().to_numpy()
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def _assess(diagnostic: MarketDiagnostic) -> None:
    """Traduit les mesures en signaux de prudence et en verdict."""
    warnings: list[str] = []
    caution = 0

    if diagnostic.breadth_pct < 50.0:
        caution += 2
        warnings.append(
            f"seuls {diagnostic.breadth_pct:.0f} % des titres sont en tendance haussiere : "
            "le marche n'est plus porteur pour un systeme long-only"
        )
    elif diagnostic.breadth_pct < 75.0:
        caution += 1
        warnings.append(
            f"hausse etroite : {diagnostic.breadth_pct:.0f} % des titres seulement portent "
            "le mouvement"
        )

    if diagnostic.mean_vol_ratio > 1.5:
        caution += 2
        warnings.append(
            f"volatilite recente a {diagnostic.mean_vol_ratio:.2f}x sa normale : les stops "
            "calibres sur le passe se declenchent sur du bruit"
        )
    elif diagnostic.mean_vol_ratio > 1.2:
        caution += 1
        warnings.append(f"volatilite en hausse ({diagnostic.mean_vol_ratio:.2f}x la normale)")

    if diagnostic.mean_correlation > 0.6:
        caution += 1
        warnings.append(
            f"correlation moyenne de {diagnostic.mean_correlation:.2f} : les titres bougent "
            "ensemble, la diversification protege moins"
        )

    if diagnostic.mean_drawdown_pct < -15.0:
        caution += 2
        warnings.append(
            f"l'univers est en moyenne a {diagnostic.mean_drawdown_pct:.1f} % de ses plus hauts "
            "d'un an : correction en cours"
        )
    elif diagnostic.mean_drawdown_pct < -8.0:
        caution += 1
        warnings.append(
            f"recul moyen de {diagnostic.mean_drawdown_pct:.1f} % depuis les plus hauts"
        )

    if diagnostic.benchmark is not None and not diagnostic.benchmark.above_sma200:
        caution += 2
        warnings.append(
            f"le marche large ({diagnostic.benchmark.symbol}) est sous sa moyenne 200 "
            "seances : ce n'est pas un probleme propre a l'univers choisi"
        )

    severe = [item for item in diagnostic.symbols if item.drawdown_from_high_pct < -25.0]
    if severe:
        caution += 1
        warnings.append(
            "titres en forte correction : "
            + ", ".join(f"{item.symbol} ({item.drawdown_from_high_pct:.0f} %)" for item in severe)
        )

    diagnostic.warnings = warnings
    diagnostic.caution_score = caution

    # Les seuils sont des paliers : signaler quand on en frole un evite de lire
    # un verdict tranche la ou la situation est en realite a la limite.
    if -16.0 < diagnostic.mean_drawdown_pct < -13.0:
        diagnostic.near_threshold = (
            f"le recul moyen ({diagnostic.mean_drawdown_pct:.1f} %) est proche du seuil de "
            "-15 % qui ferait basculer le verdict d'un cran vers la prudence"
        )
    elif 1.15 < diagnostic.mean_vol_ratio < 1.30:
        diagnostic.near_threshold = (
            f"la volatilite ({diagnostic.mean_vol_ratio:.2f}x) est proche du seuil de 1.20 "
            "qui ferait basculer le verdict d'un cran vers la prudence"
        )

    if caution >= 4:
        diagnostic.verdict = (
            "PASSE DIFFICILE — plusieurs signaux defavorables se cumulent. Augmenter le "
            "risque maintenant reviendrait a le faire au pire moment."
        )
        diagnostic.recommended_profile = "defensif"
    elif caution >= 2:
        diagnostic.verdict = (
            "MARCHE MITIGE — la tendance de fond tient mais se degrade. Un profil "
            "intermediaire garde l'exposition sans la pousser."
        )
        diagnostic.recommended_profile = "equilibre"
    else:
        diagnostic.verdict = (
            "MARCHE PORTEUR — les conditions restent favorables a un systeme long-only."
        )
        diagnostic.recommended_profile = "equilibre"

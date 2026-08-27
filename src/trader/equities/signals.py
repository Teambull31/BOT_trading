"""Faisceau de signaux directionnels independants, tous causaux.

L'idee testee ici : un signal isole se trompe souvent, mais quand plusieurs
signaux construits sur des logiques DIFFERENTES pointent dans la meme direction,
la probabilite que le mouvement se poursuive doit monter. C'est une hypothese,
pas un acquis — le module `probability` la mesure au lieu de la supposer.

Chaque signal renvoie -1 (baissier), 0 (neutre) ou +1 (haussier). Ils sont
choisis pour capter des choses distinctes :

- `tendance`      : position vs moyenne longue (regime de fond) ;
- `momentum`      : performance 12 mois hors dernier mois (facteur academique) ;
- `macd`          : acceleration de la tendance moyenne ;
- `rsi`           : force relative au-dessus/sous 50 (regime, pas sur-achat) ;
- `canal`         : position dans le canal Donchian (structure de prix) ;
- `volume`        : pente de l'OBV (les volumes confirment-ils le prix) ;
- `volatilite`    : compression ou expansion de l'ATR.

Aucun n'est optimise : ce sont les reglages standards. Sommer des signaux
sur-ajustes individuellement donnerait un ensemble sur-ajuste.

CAUSALITE : toute valeur en t ne depend que des barres <= t. `assert_signals_causal`
(module strategy) et les tests dedies le verifient par recalcul sur prefixe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.data.features import adx, atr, macd, obv, rsi

SIGNAL_NAMES: tuple[str, ...] = (
    "tendance",
    "momentum",
    "macd",
    "rsi",
    "canal",
    "volume",
    "volatilite",
)


def _sign(series: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Convertit une serie en -1 / 0 / +1 autour d'un seuil."""
    return pd.Series(
        np.where(series > threshold, 1.0, np.where(series < threshold, -1.0, 0.0)),
        index=series.index,
    ).where(series.notna())


def compute_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Calcule les sept signaux directionnels et leur score agrege.

    Le score est la SOMME des signaux (de -7 a +7). Une somme, pas une moyenne
    ponderee : ponderer suppose de connaitre l'importance relative de chaque
    signal, ce qui demanderait de l'estimer sur les donnees — donc de les
    sur-ajuster. A poids egal, l'ensemble n'a aucun degre de liberte cache.
    """
    close, high, low = frame["close"], frame["high"], frame["low"]
    volume = frame.get("volume", pd.Series(1.0, index=frame.index))
    out = pd.DataFrame(index=frame.index)

    out["tendance"] = _sign(close - close.rolling(200, min_periods=200).mean())

    # Momentum 12 mois hors dernier mois : le saut du dernier mois est bruite
    # (effet de retournement court terme), l'exclure est l'usage academique.
    momentum = close.shift(21) / close.shift(252) - 1.0
    out["momentum"] = _sign(momentum)

    out["macd"] = _sign(macd(close)[2])

    out["rsi"] = _sign(rsi(close, 14) - 50.0)

    # Position dans le canal 20 jours, ramenee a [-1, +1] ; les extremes sont
    # decales d'une barre pour ne pas inclure la cloture qu'on evalue.
    channel_high = high.rolling(20, min_periods=20).max().shift(1)
    channel_low = low.rolling(20, min_periods=20).min().shift(1)
    span = (channel_high - channel_low).replace(0.0, np.nan)
    position = (close - channel_low) / span
    out["canal"] = _sign(position - 0.5)

    on_balance = obv(close, volume)
    out["volume"] = _sign(on_balance - on_balance.rolling(20, min_periods=20).mean())

    # Volatilite qui se comprime = continuation plus probable ; qui explose =
    # regime instable. Signe inverse de la variation de l'ATR relatif.
    atr_pct = atr(high, low, close, 14) / close
    out["volatilite"] = -_sign(atr_pct - atr_pct.rolling(50, min_periods=50).mean())

    out["score"] = out[list(SIGNAL_NAMES)].sum(axis=1, min_count=len(SIGNAL_NAMES))
    out["accord"] = out[list(SIGNAL_NAMES)].abs().sum(axis=1, min_count=len(SIGNAL_NAMES))
    out["close"] = close
    out["atr"] = atr(high, low, close, 14)
    out["adx"] = adx(high, low, close, 14)[0]
    return out


def forward_outcome(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """Resultat observe APRES chaque barre : le cours monte-t-il sur `horizon` seances ?

    Cette serie regarde delibererement le futur — c'est sa nature : c'est la
    variable a expliquer. Elle ne doit JAMAIS servir a decider un trade en t,
    seulement a apprendre, plus tard, ce qu'un signal passe valait. Le module
    `probability` impose ce decalage ; les tests le verifient.
    """
    forward = frame["close"].shift(-horizon) / frame["close"] - 1.0
    return forward.rename(f"forward_{horizon}")

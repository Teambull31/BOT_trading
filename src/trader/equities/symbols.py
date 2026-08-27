"""Classement des tickers par classe d'actif.

Module volontairement sans dependance : il est importe par le mode
d'entrainement, qui doit pouvoir tourner sans pandas ni la chaine de backtest
(l'application hebergee n'embarque que FastAPI et httpx).
"""

from __future__ import annotations

ETF_SYMBOLS: frozenset[str] = frozenset({"SPY", "QQQ", "GLD", "IWM", "TLT", "VTI", "EFA"})
"""Tickers a interroger en tant qu'ETF et non en tant qu'action.

L'API de cotation exige la bonne classe d'actif : demander GLD en `stocks`
renvoie une erreur, pas un prix.
"""

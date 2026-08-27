"""Cours en temps reel via l'API de quotation Nasdaq.

Distinction importante pour l'entrainement : hors séance, le "dernier prix"
affiche est celui du pre-marche ou de l'after-hours, ou les volumes sont
faibles et les écarts larges. Un débutant qui croit trader ce prix apprend un
reflexe faux. Le module remonte donc explicitement `market_status` et separe
le prix courant du dernier cours de Clôture, pour que l'interface puisse le
dire au lieu de l'ignorer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from trader.equities.data import ETF_SYMBOLS
from trader.logging_setup import get_logger

log = get_logger(__name__)

QUOTE_URL = "https://api.nasdaq.com/api/quote/{symbol}/info"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}
CACHE_SECONDS: float = 15.0
"""Duree de vie d'un cours en cache. Rafraichir plus vite ne sert à rien pour
un entrainement sur des positions de plusieurs jours, et fait courir le risque
d'un blocage cote fournisseur."""


@dataclass(frozen=True, slots=True)
class Quote:
    """Cotation instantanee d'un titre."""

    symbol: str
    price: float
    change: float
    change_pct: float
    previous_close: float
    market_status: str
    is_real_time: bool
    timestamp: str
    bid: float | None = None
    ask: float | None = None
    company: str = ""
    week52_range: str = ""
    fetched_at: datetime = None  # type: ignore[assignment]

    @property
    def is_tradable_session(self) -> bool:
        """Vrai pendant la séance principale, ou les prix sont fiables."""
        return "open" in self.market_status.lower()

    @property
    def spread_pct(self) -> float | None:
        """Écart bid/ask en % du prix — coût implicite d'un aller-retour."""
        if self.bid is None or self.ask is None or self.price <= 0:
            return None
        return (self.ask - self.bid) / self.price * 100.0

    def to_dict(self) -> dict:
        """Representation serialisable pour l'interface."""
        return {
            "symbol": self.symbol,
            "company": self.company,
            "price": round(self.price, 4),
            "change": round(self.change, 4),
            "change_pct": round(self.change_pct, 2),
            "previous_close": round(self.previous_close, 4),
            "market_status": self.market_status,
            "is_real_time": self.is_real_time,
            "is_tradable_session": self.is_tradable_session,
            "timestamp": self.timestamp,
            "bid": self.bid,
            "ask": self.ask,
            "spread_pct": round(self.spread_pct, 3) if self.spread_pct is not None else None,
            "week52_range": self.week52_range,
        }


class QuoteError(RuntimeError):
    """Cours indisponible."""


_cache: dict[str, tuple[float, Quote]] = {}


def _parse_price(raw: str | None) -> float:
    """Convertit '$977.5738' ou '+4.17%' en nombre."""
    if not raw:
        return float("nan")
    cleaned = raw.replace("$", "").replace(",", "").replace("%", "").replace("+", "").strip()
    if not cleaned or cleaned.upper() in {"N/A", "NA", "--"}:
        return float("nan")
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def fetch_quote(symbol: str, *, use_cache: bool = True, timeout: float = 12.0) -> Quote:
    """Recupere la cotation courante d'un titre.

    Le cache evite de marteler l'API quand plusieurs composants de l'interface
    demandent le même titre au même rafraichissement.
    """
    symbol = symbol.upper().strip()
    now = time.monotonic()
    if use_cache and symbol in _cache:
        stamped, quote = _cache[symbol]
        if now - stamped < CACHE_SECONDS:
            return quote

    asset_class = "etf" if symbol in ETF_SYMBOLS else "stocks"
    try:
        response = httpx.get(
            QUOTE_URL.format(symbol=symbol),
            params={"assetclass": asset_class},
            headers=HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise QuoteError(f"cours indisponible pour {symbol} : {error}") from error

    data = payload.get("data") or {}
    primary = data.get("primaryData") or {}
    if not primary:
        raise QuoteError(f"aucune donnee de cotation pour {symbol}")
    secondary = data.get("secondaryData") or {}
    price = _parse_price(primary.get("lastSalePrice"))
    if price != price:  # NaN
        raise QuoteError(f"prix illisible pour {symbol}")

    # Hors séance, `secondaryData` porte la derniere clôture officielle. C'est
    # elle qui sert de reference de variation, pas le prix du pre-marche.
    close = _parse_price(secondary.get("lastSalePrice"))
    change = _parse_price(primary.get("netChange"))
    if primary.get("deltaIndicator") == "down" and change == change:
        change = -abs(change)
    change_pct = _parse_price(primary.get("percentageChange"))
    if primary.get("deltaIndicator") == "down" and change_pct == change_pct:
        change_pct = -abs(change_pct)

    bid = _parse_price(primary.get("bidPrice"))
    ask = _parse_price(primary.get("askPrice"))

    quote = Quote(
        symbol=symbol,
        price=price,
        change=change if change == change else 0.0,
        change_pct=change_pct if change_pct == change_pct else 0.0,
        previous_close=close if close == close else price - (change if change == change else 0.0),
        market_status=data.get("marketStatus") or "inconnu",
        is_real_time=bool(primary.get("isRealTime")),
        timestamp=primary.get("lastTradeTimestamp") or "",
        bid=bid if bid == bid and bid > 0 else None,
        ask=ask if ask == ask and ask > 0 else None,
        company=data.get("companyName") or "",
        # L'API renvoie parfois `null` (et non un objet vide) pour ces champs
        # selon le titre et l'heure : chaque niveau doit donc être defendu, un
        # `.get(cle, {})` ne protege pas d'une valeur explicitement nulle.
        week52_range=((data.get("keyStats") or {}).get("fiftyTwoWeekHighLow") or {}).get(
            "value", ""
        )
        or "",
        fetched_at=datetime.now(UTC),
    )
    _cache[symbol] = (now, quote)
    log.info("quote_fetched", symbol=symbol, price=price, status=quote.market_status)
    return quote


def fetch_quotes(symbols: list[str], *, use_cache: bool = True) -> dict[str, Quote]:
    """Recupere plusieurs cotations ; un titre indisponible n'interrompt rien."""
    quotes: dict[str, Quote] = {}
    for symbol in symbols:
        try:
            quotes[symbol] = fetch_quote(symbol, use_cache=use_cache)
        except QuoteError as error:
            log.warning("quote_failed", symbol=symbol, error=str(error))
    return quotes


def clear_cache() -> None:
    """Vide le cache — utile aux tests et a un rafraichissement force."""
    _cache.clear()

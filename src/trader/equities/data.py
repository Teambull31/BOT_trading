"""Recuperation des cours quotidiens actions (API historique Nasdaq).

Les donnees sont mises en cache sur disque : un backtest doit etre reproductible
a l'identique, et re-telecharger a chaque execution introduirait des differences
silencieuses entre deux runs.

Note sur les cours : l'API renvoie des prix NON ajustes des dividendes. Pour du
trend-following sur quelques mois sur des valeurs a faible rendement, l'ecart est
negligeable ; il est signale explicitement plutot que passe sous silence.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from trader.logging_setup import get_logger

log = get_logger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
CACHE_DIR = Path("data/equities")
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class MarketDataError(RuntimeError):
    """Impossible de recuperer les cours d'un titre."""


def _parse_money(value: str) -> float:
    """Convertit '$932.97' ou '19,163,180' en flottant."""
    cleaned = str(value).replace("$", "").replace(",", "").strip()
    if cleaned in ("", "N/A", "--"):
        return float("nan")
    return float(cleaned)


ETF_SYMBOLS: frozenset[str] = frozenset({"SPY", "QQQ", "GLD", "IWM", "TLT", "VTI", "EFA"})
"""Symboles servis par l'API sous la classe d'actif 'etf' et non 'stocks'."""


def fetch_history(
    symbol: str,
    start: date,
    end: date,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
    retries: int = 3,
    asset_class: str | None = None,
) -> pd.DataFrame:
    """Recupere l'historique quotidien d'un titre, avec cache disque.

    Args:
        symbol: ticker (ex. 'MU', 'ASML').
        start: premiere date souhaitee.
        end: derniere date souhaitee.
        refresh: force le re-telechargement meme si le cache couvre la periode.

    Returns:
        DataFrame OHLCV indexe par date (UTC), trie chronologiquement.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol.upper()}.csv"

    if cache_path.is_file() and not refresh:
        cached = _read_cache(cache_path)
        if not cached.empty and cached.index[0].date() <= start and cached.index[-1].date() >= end:
            return cached.loc[str(start) : str(end)]

    resolved_class = asset_class or ("etf" if symbol.upper() in ETF_SYMBOLS else "stocks")
    raw = _download(symbol, start, end, retries, resolved_class)
    frame = _to_frame(raw)
    if frame.empty:
        raise MarketDataError(f"{symbol}: aucune donnee retournee pour {start} -> {end}")

    if cache_path.is_file():
        # On fusionne avec le cache : l'API limite la profondeur par requete.
        merged = pd.concat([_read_cache(cache_path), frame])
        frame = merged[~merged.index.duplicated(keep="last")].sort_index()
    frame.to_csv(cache_path)
    log.info(
        "equity_history_fetched",
        symbol=symbol,
        rows=len(frame),
        start=str(frame.index[0].date()),
        end=str(frame.index[-1].date()),
    )
    return frame.loc[str(start) : str(end)]


def _read_cache(path: Path) -> pd.DataFrame:
    """Relit un cache CSV."""
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    return frame.sort_index()


def _download(
    symbol: str, start: date, end: date, retries: int, asset_class: str = "stocks"
) -> list[dict]:
    """Appelle l'API historique, avec backoff exponentiel sur erreur reseau."""
    url = (
        f"https://api.nasdaq.com/api/quote/{symbol.upper()}/historical"
        f"?assetclass={asset_class}&fromdate={start}&todate={end}&limit=9999"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
            table = (payload.get("data") or {}).get("tradesTable") or {}
            return table.get("rows") or []
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            log.warning(
                "equity_download_retry", symbol=symbol, attempt=attempt, error=str(exc)[:120]
            )
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise MarketDataError(f"{symbol}: telechargement impossible ({last_error})")


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    """Convertit la reponse brute en DataFrame OHLCV propre."""
    if not rows:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))
    records = []
    for row in rows:
        try:
            stamp = datetime.strptime(row["date"], "%m/%d/%Y")
        except (KeyError, ValueError):
            continue
        records.append(
            {
                "date": stamp,
                "open": _parse_money(row.get("open", "")),
                "high": _parse_money(row.get("high", "")),
                "low": _parse_money(row.get("low", "")),
                "close": _parse_money(row.get("close", "")),
                "volume": _parse_money(row.get("volume", "0")),
            }
        )
    frame = pd.DataFrame(records).set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    # Une ligne incomplete est retiree : mieux vaut un trou qu'un prix invente.
    return frame.dropna(subset=["open", "high", "low", "close"])


def load_universe(
    symbols: list[str], start: date, end: date, refresh: bool = False
) -> dict[str, pd.DataFrame]:
    """Charge l'historique de plusieurs titres."""
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frames[symbol] = fetch_history(symbol, start, end, refresh=refresh)
        except MarketDataError as exc:
            log.error("equity_history_failed", symbol=symbol, error=str(exc))
    return frames

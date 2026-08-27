"""Serveur local du mode d'entrainement.

Application volontairement LOCALE et mono-utilisateur : aucune authentification,
aucune donnee envoyee ailleurs, un fichier JSON sur disque. Ce n'est pas un
service en ligne et il ne doit pas etre expose sur un reseau public.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from trader.coach.account import InsufficientFunds, PaperAccount
from trader.coach.advisor import TradePlan, review_plan, suggest_size
from trader.coach.curriculum import evaluate_progress
from trader.coach.debrief import debrief_trade, recurring_patterns
from trader.coach.quotes import QuoteError, fetch_quote, fetch_quotes
from trader.logging_setup import get_logger

log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
WATCHLIST = ["MU", "ASML", "NVDA", "WMT", "GLD", "JPM", "KO", "XOM"]


class DepositRequest(BaseModel):
    amount: float = Field(gt=0, description="Montant fictif a crediter")
    note: str = ""


class PlanRequest(BaseModel):
    symbol: str
    shares: float = Field(gt=0)
    stop: float = Field(gt=0)
    target: float | None = None


class OpenRequest(PlanRequest):
    rationale: str = ""


class CloseRequest(BaseModel):
    position_id: str
    reason: str = "manuel"


class StopRequest(BaseModel):
    position_id: str
    stop: float = Field(gt=0)


class SizeRequest(BaseModel):
    symbol: str
    stop: float = Field(gt=0)
    risk_pct: float = Field(default=1.0, gt=0, le=100)


def create_app(store: Path | str | None = None) -> FastAPI:
    """Construit l'application. `store` permet de l'isoler dans les tests."""
    app = FastAPI(title="Coach Trading — zero to hero", docs_url="/api/docs")
    account = PaperAccount(store) if store else PaperAccount()

    def current_prices() -> dict[str, float]:
        """Cours des seules positions detenues — pas de requete inutile."""
        symbols = [position.symbol for position in account.state.positions]
        if not symbols:
            return {}
        return {symbol: quote.price for symbol, quote in fetch_quotes(symbols).items()}

    # ------------------------------------------------------------------ etat

    @app.get("/api/state")
    def get_state() -> dict:
        """Etat complet : compte, positions valorisees, progression."""
        prices = current_prices()
        account.mark(prices)
        progress = evaluate_progress(account.state)
        positions = []
        for position in account.state.positions:
            price = prices.get(position.symbol, position.entry_price)
            positions.append(
                {
                    "id": position.id,
                    "symbol": position.symbol,
                    "shares": round(position.shares, 6),
                    "entry_price": round(position.entry_price, 4),
                    "price": round(price, 4),
                    "stop": round(position.stop, 4),
                    "target": round(position.target, 4) if position.target else None,
                    "value": round(position.value(price), 2),
                    "unrealised": round(position.unrealised(price), 2),
                    "unrealised_pct": round(position.unrealised_pct(price), 2),
                    "distance_to_stop_pct": round(position.distance_to_stop_pct(price), 2),
                    "opened_at": position.opened_at,
                    "rationale": position.rationale,
                    "live": position.symbol in prices,
                }
            )
        return {
            "performance": {
                key: round(value, 2) if isinstance(value, float) else value
                for key, value in account.performance(prices).items()
            },
            "positions": positions,
            "progress": progress.to_dict(),
            "patterns": [lesson.to_dict() for lesson in recurring_patterns(account.state)],
            "has_capital": account.state.total_deposited > 0,
        }

    @app.get("/api/history")
    def get_history(limit: int = 25) -> dict:
        """Historique des trades, du plus recent au plus ancien."""
        trades = sorted(account.state.history, key=lambda t: t.closed_at, reverse=True)
        return {
            "trades": [
                {
                    "id": trade.id,
                    "symbol": trade.symbol,
                    "entry_price": round(trade.entry_price, 4),
                    "exit_price": round(trade.exit_price, 4),
                    "shares": round(trade.shares, 6),
                    "pnl": round(trade.pnl, 2),
                    "return_pct": round(trade.return_pct, 2),
                    "holding_days": trade.holding_days,
                    "closed_at": trade.closed_at,
                    "exit_reason": trade.exit_reason,
                    "is_win": trade.is_win,
                    "respected_stop": trade.respected_stop,
                    "stop_moved_against": trade.stop_moved_against,
                }
                for trade in trades[:limit]
            ]
        }

    @app.get("/api/quotes")
    def get_quotes(symbols: str = "") -> dict:
        """Cotations temps reel de la liste de suivi."""
        wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()] or WATCHLIST
        quotes = fetch_quotes(wanted)
        return {"quotes": [quote.to_dict() for quote in quotes.values()]}

    @app.get("/api/quote/{symbol}")
    def get_quote(symbol: str) -> dict:
        try:
            return fetch_quote(symbol).to_dict()
        except QuoteError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    # ------------------------------------------------------------- operations

    @app.post("/api/deposit")
    def post_deposit(request: DepositRequest) -> dict:
        """Saisie manuelle du capital fictif."""
        deposit = account.deposit(request.amount, request.note)
        return {"ok": True, "amount": deposit.amount, "cash": round(account.state.cash, 2)}

    @app.post("/api/suggest-size")
    def post_suggest_size(request: SizeRequest) -> dict:
        """Quantite telle que la perte au stop vaille `risk_pct` du capital."""
        try:
            quote = fetch_quote(request.symbol)
        except QuoteError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        prices = current_prices()
        equity = account.equity(prices) or account.state.total_deposited
        shares = suggest_size(equity, quote.price, request.stop, request.risk_pct)
        return {
            "symbol": quote.symbol,
            "price": round(quote.price, 4),
            "shares": round(shares, 6),
            "notional": round(shares * quote.price, 2),
            "risk_amount": round(shares * (quote.price - request.stop), 2),
            "equity": round(equity, 2),
        }

    @app.post("/api/review")
    def post_review(request: PlanRequest) -> dict:
        """Conseils AVANT ouverture — le coeur du mode entrainement."""
        try:
            quote = fetch_quote(request.symbol)
        except QuoteError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        plan = TradePlan(
            symbol=quote.symbol,
            shares=request.shares,
            price=quote.price,
            stop=request.stop,
            target=request.target,
        )
        review = review_plan(plan, account, quote, current_prices())
        return {**review.to_dict(), "quote": quote.to_dict()}

    @app.post("/api/open")
    def post_open(request: OpenRequest) -> dict:
        """Ouvre une position au cours courant."""
        try:
            quote = fetch_quote(request.symbol)
            position = account.open_position(
                quote.symbol,
                request.shares,
                quote.price,
                request.stop,
                target=request.target,
                rationale=request.rationale,
            )
        except QuoteError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (InsufficientFunds, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "ok": True,
            "position_id": position.id,
            "entry_price": round(position.entry_price, 4),
        }

    @app.post("/api/stop")
    def post_stop(request: StopRequest) -> dict:
        """Deplace un stop. Un elargissement est trace et rappele au debrief."""
        try:
            before = account.find_position(request.position_id).stop
            position = account.update_stop(request.position_id, request.stop)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"ok": True, "stop": position.stop, "widened": request.stop < before}

    @app.post("/api/close")
    def post_close(request: CloseRequest) -> dict:
        """Ferme une position et renvoie immediatement le debrief."""
        try:
            position = account.find_position(request.position_id)
            quote = fetch_quote(position.symbol)
            trade = account.close_position(request.position_id, quote.price, request.reason)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except QuoteError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"ok": True, "debrief": debrief_trade(trade, account.state).to_dict()}

    @app.get("/api/debrief/{trade_id}")
    def get_debrief(trade_id: str) -> dict:
        """Rejoue le debrief d'un trade passe."""
        for trade in account.state.history:
            if trade.id == trade_id:
                return debrief_trade(trade, account.state).to_dict()
        raise HTTPException(status_code=404, detail="trade introuvable")

    @app.post("/api/reset")
    def post_reset() -> dict:
        """Remet le compte a zero pour recommencer l'entrainement."""
        account.reset()
        return {"ok": True}

    # ---------------------------------------------------------------- statique

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()

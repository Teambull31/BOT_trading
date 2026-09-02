"""Serveur du mode d'entrainement, en local ou heberge.

Deux modes, un seul code :

- **local** (defaut) : un compte unique dans un fichier JSON, aucune
  authentification, rien qui sorte de la machine.
- **heberge** (`accounts_dir=`) : le serveur ne detient plus la reference. Le
  navigateur conserve son compte et l'envoie ; le serveur n'en garde qu'une
  copie de travail jetable, dans un fichier par navigateur.

Ce second mode existe parce qu'une fonction sans serveur n'a pas de disque
durable et pas d'authentification : un fichier unique partage voudrait dire un
seul compte pour tous les visiteurs, efface a chaque redemarrage. Chacun a donc
le sien, et c'est son navigateur qui le garde.

Dans les deux cas : argent fictif, aucun ordre reel, aucun courtier.
"""

from __future__ import annotations

import hashlib
import math
import re
from contextvars import ContextVar
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from trader.coach.account import ClosedTrade, InsufficientFunds, PaperAccount
from trader.coach.advisor import (
    TradePlan,
    review_plan,
    suggest_size,
    suggest_target,
)
from trader.coach.curriculum import (
    MAX_OPEN_RISK_PCT,
    MIN_PLANNED_R,
    evaluate_progress,
    planned_ratio,
)
from trader.coach.debrief import debrief_trade, recurring_patterns
from trader.coach.quotes import QuoteError, fetch_history, fetch_quote, fetch_quotes
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
    trailing_pct: float | None = Field(default=None, gt=0, lt=100)


class OpenRequest(PlanRequest):
    rationale: str = ""


class CloseRequest(BaseModel):
    position_id: str
    reason: str = "manuel"


class StopRequest(BaseModel):
    position_id: str
    stop: float = Field(gt=0)


class TrailingRequest(BaseModel):
    position_id: str
    trailing_pct: float | None = Field(default=None, gt=0, lt=100)
    """None retire le stop suiveur ; le stop deja remonte, lui, reste en place."""


class SizeRequest(BaseModel):
    symbol: str
    stop: float = Field(gt=0)
    risk_pct: float = Field(default=1.0, gt=0, le=100)


class RestoreRequest(BaseModel):
    snapshot: dict
    """Instantane complet du compte, tel que renvoye par `/api/snapshot`."""


class OrderRequest(BaseModel):
    """Ordre d'achat conditionnel : « acheter pour X EUR si le cours atteint Y »."""

    symbol: str
    trigger: float = Field(gt=0, description="Prix qui declenche l'achat")
    stop: float = Field(gt=0)
    direction: str | None = Field(default=None, description="« dip » (repli) ou « rise » (franchissement)")
    budget: float | None = Field(default=None, gt=0, description="Montant en euros a engager")
    shares: float | None = Field(default=None, gt=0, description="Quantite fixe, alternative au budget")
    target: float | None = None
    trailing_pct: float | None = Field(default=None, gt=0, lt=100)
    rationale: str = ""
    expires_in_days: int | None = Field(default=None, gt=0)


ACCOUNT_HEADER = "X-Coach-Account"
REVISION_HEADER = "X-Coach-Rev"
ACCOUNT_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
"""Un identifiant de compte devient un nom de fichier : il est verifie et non
assaini. Tout ce qui n'est pas alphanumerique, tiret ou souligne est refuse,
faute de quoi un « ../ » ferait ecrire le compte ailleurs sur le disque."""

_CURRENT: ContextVar[PaperAccount] = ContextVar("compte_de_la_requete")


class _RequestAccount:
    """Renvoie vers le compte de la requete en cours.

    Les routes manipulent `account` sans savoir s'il vient d'un fichier local
    unique ou du navigateur qui appelle : la resolution se fait une fois, dans
    l'intergiciel, et nulle part ailleurs.
    """

    def __getattr__(self, name: str) -> object:
        return getattr(_CURRENT.get(), name)


def _history_entry(trade: ClosedTrade) -> dict:
    """Un trade cloture, tel que l'historique de l'interface l'affiche.

    Le "plan" est ce que l'eleve avait mis EN FACE de son stop avant d'entrer :
    un objectif, un stop suiveur, ou rien. Le palier 5 le compte globalement ;
    l'historique doit pouvoir designer les trades concernes, sinon le reproche
    reste abstrait.

    `planned_ratio` vaut None quand le gain n'a pas de plafond chiffrable —
    stop suiveur, ou stop deja remonte au-dessus de l'entree. On ne renvoie
    jamais l'infini : `json.dumps` l'ecrirait `Infinity`, que `JSON.parse`
    refuse. `planned_ok` porte le seuil cote serveur, pour qu'il ne soit pas
    recopie dans le JavaScript et ne puisse pas diverger.
    """
    ratio = planned_ratio(trade)
    unbounded = ratio is not None and math.isinf(ratio)
    return {
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
        "plan": "aucun" if ratio is None else "suiveur" if trade.trailing_pct else "objectif",
        "trailing_pct": trade.trailing_pct,
        "planned_ratio": None if ratio is None or unbounded else round(ratio, 2),
        "planned_ok": ratio is not None and ratio >= MIN_PLANNED_R,
    }


def create_app(
    store: Path | str | None = None, *, accounts_dir: Path | str | None = None
) -> FastAPI:
    """Construit l'application.

    `store` isole le compte unique du mode local (utilise par les tests).
    `accounts_dir` bascule en mode heberge : un fichier de travail par
    navigateur, sous ce repertoire.
    """
    app = FastAPI(title="Coach Trading — zero to hero", docs_url="/api/docs")
    hosted = accounts_dir is not None
    workdir = Path(accounts_dir) if hosted else None
    shared = None if hosted else (PaperAccount(store) if store else PaperAccount())
    account = _RequestAccount()

    @app.middleware("http")
    async def bind_account(request: Request, call_next):
        """Attache le bon compte a la requete, et refuse d'operer a l'aveugle."""
        if not request.url.path.startswith("/api/"):
            return await call_next(request)
        if not hosted:
            _CURRENT.set(shared)
            return await call_next(request)

        identifier = request.headers.get(ACCOUNT_HEADER, "")
        if not ACCOUNT_ID.match(identifier):
            return JSONResponse(
                {"detail": "identifiant de compte absent ou invalide"}, status_code=400
            )
        current = PaperAccount(workdir / f"{identifier}.json")
        try:
            client_rev = int(request.headers.get(REVISION_HEADER, "0"))
        except ValueError:
            client_rev = 0
        # Le navigateur a plus recent que nous : notre copie de travail a ete
        # perdue (instance neuve, disque jetable). On ne devine pas la
        # difference, on la reclame — operer sur l'ancien etat afficherait des
        # liquidites fausses et pourrait rouvrir un trade deja clos.
        if client_rev > current.state.rev and request.url.path != "/api/restore":
            return JSONResponse(
                {"detail": "etat perime cote serveur", "code": "stale_state"}, status_code=409
            )
        _CURRENT.set(current)
        return await call_next(request)

    def current_prices() -> dict[str, float]:
        """Cours des positions detenues ET des ordres en attente — rien de plus.

        Un ordre conditionnel a besoin d'un cours pour savoir si son declencheur
        est franchi ; sans cela il resterait en attente meme une fois le seuil
        depasse.
        """
        symbols = {position.symbol for position in account.state.positions}
        symbols |= {order.symbol for order in account.state.pending}
        if not symbols:
            return {}
        return {
            symbol: quote.price
            for symbol, quote in fetch_quotes(sorted(symbols)).items()
        }

    # ------------------------------------------------------------------ etat

    @app.get("/api/state")
    def get_state() -> dict:
        """Etat complet : compte, positions valorisees, progression.

        Les stops sont evalues ICI, a chaque rafraichissement : un stop qui ne
        s'executerait pas laisserait l'utilisateur regarder son niveau etre
        franchi en esperant que ca remonte, soit exactement le reflexe que le
        parcours cherche a desapprendre.
        """
        prices = current_prices()
        account.mark(prices)
        stopped = account.check_stops(prices)
        # Les ordres conditionnels sont evalues ICI, au meme titre que les stops :
        # un declencheur franchi doit se traduire par une entree au prochain
        # rafraichissement, pas rester lettre morte.
        order_events = account.check_pending(prices)
        targets = account.targets_reached(prices)
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
                    "trailing_pct": position.trailing_pct,
                    "value": round(position.value(price), 2),
                    "unrealised": round(position.unrealised(price), 2),
                    "unrealised_pct": round(position.unrealised_pct(price), 2),
                    "distance_to_stop_pct": round(position.distance_to_stop_pct(price), 2),
                    # Ce que la position coute encore si le stop tombe, frais
                    # compris. Negatif, c'est un gain deja verrouille. Le
                    # pourcentage seul ne dit pas a l'utilisateur combien
                    # d'euros sont en jeu, et c'est ce chiffre-la qu'il decide.
                    "risk_at_stop": round(position.risk_at_stop(), 2),
                    "opened_at": position.opened_at,
                    "rationale": position.rationale,
                    "live": position.symbol in prices,
                }
            )
        performance = {
            key: round(value, 2) if isinstance(value, float) else value
            for key, value in account.performance(prices).items()
        }
        # La limite voyage avec la mesure : l'interface signale le depassement
        # sans avoir a redefinir de son cote un seuil qui vit dans le parcours.
        performance["open_risk_limit_pct"] = MAX_OPEN_RISK_PCT
        # Ce qu'un ordre en attente immobilise : l'interface doit montrer le
        # liquide REELLEMENT disponible, pas le solde brut qui laisse croire a
        # une marge de manoeuvre deja promise ailleurs.
        performance["available_cash"] = round(account.available_cash(), 2)
        performance["reserved_cash"] = round(account.state.cash - account.available_cash(), 2)
        pending = [
            {
                "id": order.id,
                "symbol": order.symbol,
                "direction": order.direction,
                "trigger": round(order.trigger, 4),
                "stop": round(order.stop, 4),
                "budget": round(order.budget, 2) if order.budget is not None else None,
                "shares": round(order.shares, 6) if order.shares is not None else None,
                "target": round(order.target, 4) if order.target else None,
                "trailing_pct": order.trailing_pct,
                "rationale": order.rationale,
                "created_at": order.created_at,
                "expires_at": order.expires_at,
                "price": round(prices.get(order.symbol, 0.0), 4),
                "reserved": round(order.reserved(), 2),
                "label": order.describe(),
                "live": order.symbol in prices,
            }
            for order in account.state.pending
        ]
        return {
            "performance": performance,
            "positions": positions,
            # Ordres conditionnels encore en attente, et ce qui vient d'arriver a
            # ceux qui ne le sont plus (executes, refuses faute de liquidites,
            # expires) — l'interface l'annonce comme elle annonce un stop touche.
            "pending": pending,
            "order_events": order_events,
            "progress": progress.to_dict(),
            "patterns": [lesson.to_dict() for lesson in recurring_patterns(account.state)],
            "has_capital": account.state.total_deposited > 0,
            # L'interface doit prevenir que le compte vit dans le navigateur :
            # une progression de trente trades effacee par un nettoyage du
            # cache, sans avertissement, serait une trahison du parcours.
            "hosted": hosted,
            # Stops declenches pendant ce rafraichissement : l'interface doit les
            # montrer immediatement, avec leur debrief. Une sortie subie sans
            # explication est une occasion d'apprendre perdue.
            "stopped": [debrief_trade(trade, account.state).to_dict() for trade in stopped],
            # Copie de reference renvoyee au navigateur, qui la conserve. En
            # mode local elle est simplement ignoree par l'interface.
            "snapshot": account.snapshot(),
            "targets_reached": [
                {
                    "id": position.id,
                    "symbol": position.symbol,
                    "target": round(position.target, 4) if position.target else None,
                    "price": round(prices.get(position.symbol, 0.0), 4),
                    "unrealised": round(
                        position.unrealised(prices.get(position.symbol, position.entry_price)), 2
                    ),
                }
                for position in targets
            ],
        }

    @app.get("/api/history")
    def get_history(limit: int = 25) -> dict:
        """Historique des trades, du plus recent au plus ancien."""
        trades = sorted(account.state.history, key=lambda t: t.closed_at, reverse=True)
        return {"trades": [_history_entry(trade) for trade in trades[:limit]]}

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

    @app.get("/api/history/{symbol}")
    def get_price_history(symbol: str, period: str = "1M") -> dict:
        """Courbe de cours d'un titre : 1D (intra-seance), 1M, 3M ou 1Y."""
        try:
            return fetch_history(symbol, period).to_dict()
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
        # L'objectif voyage avec la quantite : les deux se deduisent du meme
        # stop, et l'utilisateur les saisit dans la meme foulee.
        target = suggest_target(quote.price, request.stop)
        return {
            "symbol": quote.symbol,
            "price": round(quote.price, 4),
            "shares": round(shares, 6),
            "notional": round(shares * quote.price, 2),
            "risk_amount": round(shares * (quote.price - request.stop), 2),
            "equity": round(equity, 2),
            "suggested_target": None if target is None else round(target, 2),
            "suggested_target_ratio": MIN_PLANNED_R,
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
            trailing_pct=request.trailing_pct,
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
                trailing_pct=request.trailing_pct,
            )
        except QuoteError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (InsufficientFunds, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "ok": True,
            "position_id": position.id,
            "entry_price": round(position.entry_price, 4),
            # Le stop suiveur peut avoir resserre le stop des l'entree : c'est
            # celui-la qui fait foi, pas celui saisi dans le formulaire.
            "stop": round(position.stop, 4),
            "trailing_pct": position.trailing_pct,
        }

    @app.post("/api/order")
    def post_order(request: OrderRequest) -> dict:
        """Place un ordre d'achat conditionnel : entree au marche au declenchement.

        Le meme crible qu'avant une ouverture immediate est renvoye (`review`),
        calcule au prix du declencheur : l'utilisateur voit taille, risque et
        concentration AVANT de laisser l'ordre vivre sa vie. Les points
        reellement bloquants — stop au-dessus du declencheur, budget et quantite
        tous deux fournis (ou aucun), liquidites deja promises — sont refuses ici.
        """
        try:
            quote = fetch_quote(request.symbol)
        except QuoteError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        apercu_shares = request.shares
        if apercu_shares is None and request.budget is not None:
            apercu_shares = account.shares_for_budget(request.budget, request.trigger)
        review = None
        if apercu_shares:
            plan = TradePlan(
                symbol=quote.symbol,
                shares=apercu_shares,
                price=request.trigger,
                stop=request.stop,
                target=request.target,
                trailing_pct=request.trailing_pct,
            )
            review = review_plan(plan, account, None, current_prices()).to_dict()

        try:
            order = account.place_order(
                quote.symbol,
                request.trigger,
                request.stop,
                quote.price,
                direction=request.direction,
                budget=request.budget,
                shares=request.shares,
                target=request.target,
                rationale=request.rationale,
                trailing_pct=request.trailing_pct,
                expires_in_days=request.expires_in_days,
            )
        except (InsufficientFunds, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "ok": True,
            "order_id": order.id,
            "direction": order.direction,
            "label": order.describe(),
            "review": review,
        }

    @app.delete("/api/order/{order_id}")
    def delete_order(order_id: str) -> dict:
        """Annule un ordre en attente et libere la reserve qu'il tenait."""
        try:
            order = account.cancel_order(order_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {"ok": True, "order_id": order.id}

    @app.post("/api/stop")
    def post_stop(request: StopRequest) -> dict:
        """Deplace un stop. Un elargissement est trace et rappele au debrief.

        La reponse dit aussi si le niveau pose est DEJA touche. Un stop au-dessus
        du cours n'est pas une protection, c'est un ordre de vente : `check_stops`
        soldera la position au prochain rafraichissement, au cours observe et non
        a ce niveau. Sans ce signalement, l'utilisateur qui croit verrouiller un
        gain voit sa position disparaitre sans explication — la confusion la plus
        facile a faire sur ce qu'est un stop, et la moins facile a rattraper une
        fois le trade solde.
        """
        try:
            before = account.find_position(request.position_id).stop
            position = account.update_stop(request.position_id, request.stop)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        price = None
        try:
            price = fetch_quote(position.symbol).price
        except QuoteError:
            # Sans cotation, on n'affirme rien : mieux vaut se taire que
            # d'annoncer une sortie qui n'aura peut-etre pas lieu.
            log.warning("stop_sans_cotation", symbol=position.symbol)

        return {
            "ok": True,
            "stop": position.stop,
            "widened": request.stop < before,
            "price": None if price is None else round(price, 4),
            "triggers_now": price is not None and position.stop >= price,
        }

    @app.post("/api/trailing")
    def post_trailing(request: TrailingRequest) -> dict:
        """Active, ajuste ou retire le stop suiveur d'une position ouverte.

        Le cours du moment est recupere ici : c'est lui qui ancre un suiveur
        que l'on arme, sans quoi le stop se poserait sur un sommet passe.
        """
        try:
            position = account.find_position(request.position_id)
            price = None
            try:
                price = fetch_quote(position.symbol).price
            except QuoteError:
                # Sans cotation, on n'invente pas d'ancrage : `set_trailing`
                # retombe sur le prix d'entree, jamais sur un sommet passe.
                log.warning("trailing_sans_cotation", symbol=position.symbol)
            position = account.set_trailing(request.position_id, request.trailing_pct, price)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "ok": True,
            "trailing_pct": position.trailing_pct,
            "stop": round(position.stop, 4),
        }

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

    @app.get("/api/snapshot")
    def get_snapshot() -> dict:
        """Instantane complet du compte, a conserver cote navigateur."""
        return {"snapshot": account.snapshot()}

    @app.post("/api/restore")
    def post_restore(request: RestoreRequest) -> dict:
        """Reinjecte l'instantane detenu par le navigateur.

        Seul point d'entree exempte du controle de revision : c'est lui qui
        repare l'ecart, il ne peut donc pas exiger qu'il n'existe pas.
        """
        try:
            account.restore(request.snapshot)
        except (TypeError, ValueError) as error:
            raise HTTPException(
                status_code=400, detail=f"instantane illisible : {error}"
            ) from error
        return {"ok": True, "rev": account.state.rev}

    @app.post("/api/reset")
    def post_reset() -> dict:
        """Remet le compte a zero pour recommencer l'entrainement."""
        account.reset()
        return {"ok": True}

    # ---------------------------------------------------------------- statique

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        def _asset_tag(name: str) -> str:
            """Empreinte courte du fichier : sert de cache-buster. L'URL de la
            feuille de style ou du script change des que son contenu change, si
            bien qu'apres un deploiement le navigateur ne peut pas resservir
            l'ancienne version depuis son cache."""
            try:
                digest = hashlib.sha1((STATIC_DIR / name).read_bytes()).hexdigest()
            except OSError:
                return "0"
            return digest[:8]

        @app.get("/")
        def index() -> HTMLResponse:
            html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
            for asset in ("app.css", "app.js"):
                html = html.replace(
                    f"/static/{asset}", f"/static/{asset}?v={_asset_tag(asset)}"
                )
            # Le HTML lui-meme doit toujours etre reverifie, sinon il continue de
            # pointer vers l'ancienne empreinte.
            return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    return app


app = create_app()

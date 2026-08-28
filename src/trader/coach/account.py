"""Compte d'entrainement en argent fictif, persisté sur disque.

Le capital est saisi MANUELLEMENT par l'utilisateur : chaque dépôt est un
événement date et conservé. C'est volontaire — voir en clair "j'ai remis 500 EUR
après m'etre fait sortir" est la lecon la plus utile que puisse donner un compte
d'entrainement, et un solde qui se recharge en silence l'effacerait.

Toutes les positions sont longues et sans levier. Les mesures du dépôt montrent
que le levier au-delà de 2x détruit le capital ; l'imposer a un débutant serait
lui apprendre a se ruiner vite.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from trader.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_STORE = Path("data/coach/account.json")
COMMISSION_PCT: float = 0.10
SLIPPAGE_PCT: float = 0.05
MIN_COMMISSION: float = 1.0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass(slots=True)
class Deposit:
    """Apport de capital saisi par l'utilisateur."""

    amount: float
    at: str = field(default_factory=_now)
    note: str = ""


@dataclass(slots=True)
class Position:
    """Position ouverte."""

    id: str
    symbol: str
    shares: float
    entry_price: float
    stop: float
    opened_at: str
    entry_costs: float
    target: float | None = None
    rationale: str = ""
    highest_price: float = 0.0
    trailing_pct: float | None = None
    """Distance, en % sous le plus haut atteint, a laquelle le stop suit le cours.

    Le debrief recommande le stop suiveur des que les pertes moyennes depassent
    les gains moyens ; sans ce champ, l'application conseillait un outil qu'elle
    ne fournissait pas.

    Ce qu'un stop suiveur fait, et ne fait pas : il n'annonce rien sur la suite
    du cours et n'ameliore aucune esperance de gain — les mesures de ce depot ne
    montrent aucun pouvoir predictif. Il impose une asymetrie, en transformant
    une plus-value latente en plancher, et il retire la decision de sortie au
    moment ou elle se prend le plus mal : quand le cours vient de baisser.
    """
    trail_high: float = 0.0
    """Plus haut atteint DEPUIS l'armement du suiveur — son point d'ancrage.

    Volontairement distinct de `highest_price`, qui couvre toute la vie de la
    position. Ancrer un suiveur arme aujourd'hui sur un sommet d'il y a trois
    semaines placerait le stop au-dessus du cours et sortirait la position sur
    le champ, a un niveau que personne n'a choisi. Un suiveur commence a
    compter la ou on l'arme.
    """

    def __post_init__(self) -> None:
        if self.highest_price <= 0:
            self.highest_price = self.entry_price
        if self.trail_high <= 0:
            # Position relue d'un fichier ecrit avant l'arrivee du suiveur :
            # sans ancrage, `trailing_stop()` renverrait 0 et le suiveur serait
            # silencieusement inoperant.
            self.trail_high = self.highest_price

    def trailing_stop(self) -> float | None:
        """Niveau que le stop suiveur impose, ou None s'il est desactive."""
        if not self.trailing_pct:
            return None
        return self.trail_high * (1.0 - self.trailing_pct / 100.0)

    @property
    def cost_basis(self) -> float:
        """Montant engagé a l'ouverture."""
        return self.shares * self.entry_price

    def risk_at_stop(self) -> float:
        """Ce que la position coûterait encore si le stop tombait maintenant.

        Positif, c'est la perte qui reste exposée. Négatif, le stop est passé
        au-dessus du prix de revient : le trade ne peut plus rien coûter et le
        chiffre est un gain déjà acquis. C'est précisément ce que le stop
        suiveur du parcours cherche à produire, et l'application ne le disait
        nulle part.

        Frais des deux ordres et écart de cotation compris — le montant rendu
        est exactement l'inverse du résultat que `close_position` inscrirait à
        ce prix-là, et non la différence théorique entre l'entrée et le stop,
        qui minore la perte. Il suppose en revanche une sortie AU niveau du
        stop ; sur un écart brutal elle se fait plus bas, et `check_stops`
        solde alors au cours réellement observé.
        """
        fill = self.stop * (1.0 - SLIPPAGE_PCT / 100.0)
        proceeds = self.shares * fill
        return self.cost_basis + self.entry_costs + PaperAccount.costs_for(proceeds) - proceeds

    def value(self, price: float) -> float:
        """Valeur courante."""
        return self.shares * price

    def unrealised(self, price: float) -> float:
        """Plus ou moins-value latente."""
        return (price - self.entry_price) * self.shares - self.entry_costs

    def unrealised_pct(self, price: float) -> float:
        """Plus ou moins-value latente en % du montant engagé."""
        return self.unrealised(price) / self.cost_basis * 100.0 if self.cost_basis else 0.0

    def distance_to_stop_pct(self, price: float) -> float:
        """Marge restante avant le stop, en % du cours actuel."""
        return (price - self.stop) / price * 100.0 if price > 0 else 0.0


@dataclass(slots=True)
class ClosedTrade:
    """Trade terminé, conservé pour le debrief et la progression."""

    id: str
    symbol: str
    shares: float
    entry_price: float
    exit_price: float
    stop: float
    opened_at: str
    closed_at: str
    pnl: float
    costs: float
    exit_reason: str
    target: float | None = None
    rationale: str = ""
    highest_price: float = 0.0
    trailing_pct: float | None = None
    """Distance du stop suiveur si le trade en utilisait un, sinon None."""
    stop_moved_against: bool = False
    """Le stop a-t-il été élargi pendant la vie du trade ?

    C'est l'erreur la plus coûteuse du débutant : reculer son stop pour ne pas
    matérialiser une perte. On la trace explicitement pour pouvoir la montrer.
    """

    @property
    def return_pct(self) -> float:
        """Rendement en % du montant engagé."""
        notional = self.shares * self.entry_price
        return self.pnl / notional * 100.0 if notional else 0.0

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    @property
    def holding_days(self) -> int:
        opened = datetime.fromisoformat(self.opened_at)
        closed = datetime.fromisoformat(self.closed_at)
        return max(0, (closed - opened).days)

    @property
    def planned_risk(self) -> float:
        """Perte qui était prévue si le stop était touché — jamais négative.

        Un stop remonté AU-DESSUS du prix d'entrée ne planifie plus une perte
        mais un gain : le risque planifié vaut alors zéro. Sans ce garde-fou,
        une « enveloppe de perte » négative rendrait absurdes le ratio
        gain/risque et les messages du debrief qui en dépendent.
        """
        return max(0.0, (self.entry_price - self.stop) * self.shares)

    @property
    def stop_locks_gain(self) -> bool:
        """Le stop était-il au-dessus du prix d'entrée (gain verrouillé) ?"""
        return self.stop > self.entry_price

    @property
    def respected_stop(self) -> bool:
        """La perte est-elle restee dans l'enveloppe prévue ?

        Tolerance de 15 % : un gap d'ouverture peut faire sortir sous le stop
        sans que ce soit une faute de discipline.
        """
        if self.pnl >= 0:
            return True
        return abs(self.pnl) <= self.planned_risk * 1.15 + self.costs


@dataclass(slots=True)
class AccountState:
    """État complet du compte d'entrainement."""

    cash: float = 0.0
    deposits: list[Deposit] = field(default_factory=list)
    positions: list[Position] = field(default_factory=list)
    history: list[ClosedTrade] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    rev: int = 0
    """Numero de revision, incremente a chaque ecriture.

    Sert au mode heberge, ou le navigateur detient la copie de reference du
    compte et le serveur une copie de travail jetable : comparer les revisions
    est le seul moyen de refuser d'operer sur un etat perime, par exemple
    quand deux instances du serveur se relaient et que l'une n'a jamais vu les
    derniers trades. Un compte qui execute un ordre sur un historique tronque
    afficherait des liquidites et un risque faux.
    """

    @property
    def total_deposited(self) -> float:
        """Argent fictif injecte au total — le vrai denominateur du résultat."""
        return sum(deposit.amount for deposit in self.deposits)


class InsufficientFunds(ValueError):
    """Liquidités insuffisantes."""


class PaperAccount:
    """Compte fictif : depots manuels, achats, ventes, historique."""

    def __init__(self, store: Path | str = DEFAULT_STORE) -> None:
        self.store = Path(store)
        self.state = self._load()

    # ------------------------------------------------------------ persistance

    @staticmethod
    def state_from_dict(raw: dict) -> AccountState:
        """Reconstruit un etat a partir de sa forme serialisee."""
        return AccountState(
            cash=float(raw.get("cash", 0.0)),
            deposits=[Deposit(**item) for item in raw.get("deposits", [])],
            positions=[Position(**item) for item in raw.get("positions", [])],
            history=[ClosedTrade(**item) for item in raw.get("history", [])],
            created_at=raw.get("created_at", _now()),
            rev=int(raw.get("rev", 0)),
        )

    def snapshot(self) -> dict:
        """Etat complet serialisable — la forme ecrite sur disque, telle quelle.

        C'est aussi ce que le navigateur conserve en mode heberge : une seule
        representation, donc aucun risque qu'une copie oublie un champ que
        l'autre attend.
        """
        return {
            "cash": round(self.state.cash, 2),
            "deposits": [asdict(item) for item in self.state.deposits],
            "positions": [asdict(item) for item in self.state.positions],
            "history": [asdict(item) for item in self.state.history],
            "created_at": self.state.created_at,
            "rev": self.state.rev,
        }

    def restore(self, raw: dict) -> None:
        """Remplace l'etat courant par un instantane recu, et l'ecrit."""
        self.state = self.state_from_dict(raw)
        self.save()

    def _load(self) -> AccountState:
        if not self.store.exists():
            return AccountState()
        try:
            raw = json.loads(self.store.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            log.warning("account_load_failed", error=str(error))
            return AccountState()
        return self.state_from_dict(raw)

    def save(self) -> None:
        """Ecrit l'etat sur disque, de facon atomique, et avance la revision."""
        self.state.rev += 1
        self.store.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.store.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.snapshot(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(self.store)

    # ------------------------------------------------------------ operations

    def deposit(self, amount: float, note: str = "") -> Deposit:
        """Ajoute du capital fictif. C'est la saisie manuelle demandee."""
        if amount <= 0:
            raise ValueError("le montant doit être positif")
        entry = Deposit(amount=float(amount), note=note)
        self.state.deposits.append(entry)
        self.state.cash += float(amount)
        self.save()
        log.info("deposit", amount=amount, cash=self.state.cash)
        return entry

    @staticmethod
    def costs_for(notional: float) -> float:
        """Frais d'un ordre : commission plancher plus slippage."""
        return max(MIN_COMMISSION, notional * COMMISSION_PCT / 100.0)

    def open_position(
        self,
        symbol: str,
        shares: float,
        price: float,
        stop: float,
        *,
        target: float | None = None,
        rationale: str = "",
        trailing_pct: float | None = None,
    ) -> Position:
        """Ouvre une position longue.

        Le stop est OBLIGATOIRE et doit être sous le prix d'entrée : une
        position sans niveau de sortie défini à l'avance n'est pas un trade,
        c'est un pari dont on décidera la fin sous le coup de l'émotion.

        Si un stop suiveur est demandé et qu'il impose, dès l'entrée, un niveau
        plus haut que le stop saisi, c'est LUI qui s'applique. Le stop réellement
        en vigueur est celui que porte la position renvoyée : laisser croire à
        une marge de 5 % alors qu'un suiveur à 2 % gouverne serait afficher un
        risque que le compte ne court pas.
        """
        symbol = symbol.upper().strip()
        if shares <= 0:
            raise ValueError("quantite invalide")
        if price <= 0:
            raise ValueError("prix invalide")
        if stop <= 0 or stop >= price:
            raise ValueError("le stop doit être strictement sous le prix d'entrée")
        if trailing_pct is not None and not 0.0 < trailing_pct < 100.0:
            raise ValueError("le stop suiveur doit être une distance entre 0 et 100 %")
        if any(position.symbol == symbol for position in self.state.positions):
            raise ValueError(f"une position est déjà ouverte sur {symbol}")

        fill = price * (1.0 + SLIPPAGE_PCT / 100.0)
        notional = shares * fill
        costs = self.costs_for(notional)
        if notional + costs > self.state.cash + 1e-9:
            raise InsufficientFunds(
                f"il faut {notional + costs:,.2f} EUR, le compte en a {self.state.cash:,.2f}"
            )

        position = Position(
            id=_new_id(),
            symbol=symbol,
            shares=float(shares),
            entry_price=fill,
            stop=float(stop),
            opened_at=_now(),
            entry_costs=costs,
            target=float(target) if target else None,
            rationale=rationale,
            highest_price=fill,
            trailing_pct=float(trailing_pct) if trailing_pct else None,
            trail_high=fill,
        )
        level = position.trailing_stop()
        if level is not None and level > position.stop:
            position.stop = level
        self.state.cash -= notional + costs
        self.state.positions.append(position)
        self.save()
        log.info("position_opened", symbol=symbol, shares=shares, price=fill, stop=stop)
        return position

    def update_stop(self, position_id: str, stop: float) -> Position:
        """Deplace un stop, en tracant tout élargissement.

        Reculer un stop est autorise — l'interdire empecherait d'apprendre par
        l'erreur — mais c'est enregistre et le debrief le rappellera.
        """
        position = self.find_position(position_id)
        if stop <= 0:
            raise ValueError("stop invalide")
        if stop < position.stop:
            log.warning("stop_widened", symbol=position.symbol, old=position.stop, new=stop)
            position.rationale = (position.rationale + " [stop élargi]").strip()
        position.stop = float(stop)
        self.save()
        return position

    def set_trailing(
        self,
        position_id: str,
        trailing_pct: float | None,
        price: float | None = None,
    ) -> Position:
        """Active, ajuste ou retire le stop suiveur d'une position ouverte.

        Le retirer est permis : interdire un retour en arrière empêcherait
        d'apprendre par l'erreur. Mais le stop déjà remonté par le suiveur, lui,
        RESTE en place — le suiveur ne rend jamais le terrain qu'il a pris.

        `price` est le cours du moment. Il sert de point d'ancrage quand on
        ARME le suiveur : un suiveur commence à compter là où on le pose, pas
        au sommet que la position a touché la semaine dernière. Ancrer sur ce
        sommet-là placerait le stop au-dessus du cours et solderait la position
        immédiatement — une sortie que l'utilisateur n'a jamais demandée.
        Resserrer un suiveur déjà actif, en revanche, ne redéplace pas
        l'ancrage : le terrain déjà pris est acquis.
        """
        position = self.find_position(position_id)
        if trailing_pct is not None and not 0.0 < trailing_pct < 100.0:
            raise ValueError("le stop suiveur doit être une distance entre 0 et 100 %")
        arming = trailing_pct and not position.trailing_pct
        position.trailing_pct = float(trailing_pct) if trailing_pct else None
        if arming:
            position.trail_high = float(price) if price else position.entry_price
        level = position.trailing_stop()
        if level is not None and level > position.stop:
            position.stop = level
        self.save()
        log.info("trailing_set", symbol=position.symbol, pct=trailing_pct, stop=position.stop)
        return position

    def mark(self, prices: dict[str, float]) -> None:
        """Met à jour le plus haut atteint, puis fait suivre les stops suiveurs.

        Le plus haut sert au debrief : sans lui, impossible de dire à
        l'utilisateur qu'il avait +18 % latents avant de sortir à +3 %.

        Un stop suiveur ne descend JAMAIS. C'est ce qui le distingue d'un stop
        que l'on déplace à la main : il ne peut pas devenir le prétexte à ne pas
        matérialiser une perte, puisque le seul mouvement qu'il autorise est
        celui qui réduit le risque.
        """
        for position in self.state.positions:
            price = prices.get(position.symbol)
            if price and price > position.highest_price:
                position.highest_price = float(price)
            if price and price > position.trail_high:
                position.trail_high = float(price)
            level = position.trailing_stop()
            if level is not None and level > position.stop:
                log.info(
                    "trailing_stop_raised",
                    symbol=position.symbol,
                    old=round(position.stop, 4),
                    new=round(level, 4),
                )
                position.stop = level
        self.save()

    def check_stops(self, prices: dict[str, float]) -> list[ClosedTrade]:
        """Declenche les stops touches, au prix REELLEMENT observe.

        Sans cette execution automatique, le stop ne serait qu'une intention :
        l'utilisateur pourrait le regarder etre franchi et attendre que ca
        remonte — precisement le reflexe que le parcours cherche a desapprendre.
        Un stop qui ne s'execute pas n'enseigne rien.

        Le prix de sortie retenu est celui de la cotation observee, pas le
        niveau theorique du stop. C'est volontaire et c'est plus honnete : dans
        la realite on ne sort pas au niveau exact, et sur un ecart brutal la
        sortie se fait nettement plus bas. Le debrief chiffre cet ecart.
        """
        triggered: list[ClosedTrade] = []
        for position in list(self.state.positions):
            price = prices.get(position.symbol)
            if price is None or price > position.stop:
                continue
            trade = self.close_position(position.id, price, reason="stop_touche")
            triggered.append(trade)
            log.info(
                "stop_triggered",
                symbol=trade.symbol,
                stop=position.stop,
                fill=round(trade.exit_price, 4),
            )
        return triggered

    def targets_reached(self, prices: dict[str, float]) -> list[Position]:
        """Positions ayant atteint leur objectif — signalees, PAS fermees.

        Le stop s'execute d'office parce qu'il protege ; l'objectif, non. Le
        moment ou une position atteint sa cible est justement la decision qui
        merite d'etre travaillee : encaisser, ou remonter le stop et laisser
        courir. La fermer automatiquement priverait l'utilisateur du seul
        arbitrage qui distingue le palier 5 du parcours.
        """
        return [
            position
            for position in self.state.positions
            if position.target and prices.get(position.symbol, 0.0) >= position.target
        ]

    def close_position(self, position_id: str, price: float, reason: str = "manuel") -> ClosedTrade:
        """Ferme une position et l'archive."""
        position = self.find_position(position_id)
        if price <= 0:
            raise ValueError("prix invalide")

        fill = price * (1.0 - SLIPPAGE_PCT / 100.0)
        proceeds = position.shares * fill
        costs = self.costs_for(proceeds)
        pnl = proceeds - costs - position.cost_basis - position.entry_costs

        trade = ClosedTrade(
            id=position.id,
            symbol=position.symbol,
            shares=position.shares,
            entry_price=position.entry_price,
            exit_price=fill,
            stop=position.stop,
            opened_at=position.opened_at,
            closed_at=_now(),
            pnl=pnl,
            costs=costs + position.entry_costs,
            exit_reason=reason,
            target=position.target,
            rationale=position.rationale,
            highest_price=max(position.highest_price, fill),
            trailing_pct=position.trailing_pct,
            stop_moved_against="[stop élargi]" in position.rationale,
        )
        self.state.cash += proceeds - costs
        self.state.positions = [p for p in self.state.positions if p.id != position_id]
        self.state.history.append(trade)
        self.save()
        log.info("position_closed", symbol=trade.symbol, pnl=round(pnl, 2), reason=reason)
        return trade

    # --------------------------------------------------------------- lectures

    def find_position(self, position_id: str) -> Position:
        for position in self.state.positions:
            if position.id == position_id:
                return position
        raise KeyError(f"position introuvable : {position_id}")

    def equity(self, prices: dict[str, float]) -> float:
        """Valeur totale du compte : liquidites plus positions valorisees."""
        held = sum(
            position.value(prices.get(position.symbol, position.entry_price))
            for position in self.state.positions
        )
        return self.state.cash + held

    def exposure_pct(self, prices: dict[str, float]) -> float:
        """Part du compte engagee en positions."""
        total = self.equity(prices)
        if total <= 0:
            return 0.0
        held = total - self.state.cash
        return held / total * 100.0

    def open_risk(self) -> float:
        """Ce que le compte perdrait si TOUS les stops ouverts tombaient maintenant.

        Somme des risques de chaque position, frais des deux ordres compris. Le
        risque par trade est borne par le parcours ; ce total-la ne l'etait par
        rien, et c'est pourtant lui que subit une seance de baisse generale, ou
        les stops ne tombent pas independamment les uns des autres.

        Une position dont le stop verrouille un gain compte en negatif : par
        hypothese tous les stops tombent EN MEME TEMPS, et ce gain-la serait bien
        encaisse ce jour-la. Le total peut donc etre negatif, ce qui est la
        situation recherchee : un portefeuille qui ne peut plus rien couter.
        """
        return sum(position.risk_at_stop() for position in self.state.positions)

    def open_risk_pct(self, prices: dict[str, float]) -> float:
        """`open_risk` rapporte au capital, en %. Zero sans capital."""
        total = self.equity(prices)
        return self.open_risk() / total * 100.0 if total > 0 else 0.0

    def performance(self, prices: dict[str, float]) -> dict[str, float]:
        """Indicateurs de suivi, rapportes au capital REELLEMENT injecte."""
        deposited = self.state.total_deposited
        equity = self.equity(prices)
        wins = [trade for trade in self.state.history if trade.is_win]
        losses = [trade for trade in self.state.history if not trade.is_win]
        gross_win = sum(trade.pnl for trade in wins)
        gross_loss = abs(sum(trade.pnl for trade in losses))
        return {
            "equity": equity,
            "deposited": deposited,
            "pnl": equity - deposited,
            "pnl_pct": (equity / deposited - 1.0) * 100.0 if deposited > 0 else 0.0,
            "cash": self.state.cash,
            "exposure_pct": self.exposure_pct(prices),
            "open_risk": self.open_risk(),
            "open_risk_pct": self.open_risk_pct(prices),
            "closed_trades": len(self.state.history),
            "open_positions": len(self.state.positions),
            "hit_rate": len(wins) / len(self.state.history) if self.state.history else 0.0,
            "avg_win": gross_win / len(wins) if wins else 0.0,
            "avg_loss": gross_loss / len(losses) if losses else 0.0,
            "profit_factor": gross_win / gross_loss if gross_loss > 0 else 0.0,
            "total_costs": sum(trade.costs for trade in self.state.history),
        }

    def reset(self) -> None:
        """Repart de zero — l'entrainement doit pouvoir être recommence.

        La revision, elle, continue d'avancer : la faire reculer ferait passer
        la remise a zero pour un etat perime, et le navigateur reinjecterait
        aussitot le compte que l'utilisateur venait d'effacer.
        """
        self.state = AccountState(rev=self.state.rev)
        self.save()

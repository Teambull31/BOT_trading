"""Conseils AVANT d'ouvrir une position.

Ce que ce module refuse de faire : annoncer si le cours va monter. Les mesures
du dépôt sont sans appel — sur 12 000 observations et sept ans, un faisceau de
sept signaux tous concordants donne 57.8 % de hausse contre 56.9 % de taux de
base, soit moins d'un point. Prétendre en tirer une prévision serait mentir.

Ce qu'il fait à la place : vérifier ce qui est réellement sous contrôle avant
d'appuyer sur le bouton. La taille, le stop, le rapport gain/perte visé, la
concentration du portefeuille, le moment de la séance. Ce sont les seuls
éléments qui séparent, sur la durée, un débutant d'un praticien.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from trader.coach.account import AccountState, PaperAccount
from trader.coach.curriculum import MAX_RISK_PCT
from trader.coach.quotes import Quote

MAX_CONCENTRATION_PCT: float = 60.0
"""Part maximale du compte sur une seule position.

Bloquant, et non simplement déconseillé : au-delà, un trou de cotation — contre
lequel aucun stop ne protège — met le compte en jeu quel que soit le risque
affiche au stop.
"""

GAP_SHOCK_PCT: float = 15.0
"""Choc d'ouverture servant de référence. Ni théorique ni pessimiste : les
titres de l'univers suivi ont produit plusieurs séances de cette ampleur."""


class Severity(str, Enum):
    """Gravité d'une remarque."""

    BLOCKER = "bloquant"
    WARNING = "attention"
    INFO = "info"
    GOOD = "ok"


@dataclass(frozen=True, slots=True)
class Advice:
    """Une remarque adressée à l'utilisateur avant son trade."""

    severity: Severity
    title: str
    message: str

    def to_dict(self) -> dict:
        return {"severity": self.severity.value, "title": self.title, "message": self.message}


@dataclass(frozen=True, slots=True)
class TradePlan:
    """Projet de trade soumis à vérification."""

    symbol: str
    shares: float
    price: float
    stop: float
    target: float | None = None

    @property
    def notional(self) -> float:
        return self.shares * self.price

    @property
    def risk_amount(self) -> float:
        """Perte si le stop est touché."""
        return (self.price - self.stop) * self.shares

    @property
    def stop_distance_pct(self) -> float:
        return (self.price - self.stop) / self.price * 100.0 if self.price > 0 else 0.0

    @property
    def reward_risk(self) -> float | None:
        """Rapport entre le gain visé et la perte acceptée."""
        if self.target is None or self.risk_amount <= 0:
            return None
        return (self.target - self.price) * self.shares / self.risk_amount


@dataclass(frozen=True, slots=True)
class Review:
    """Verdict complet sur un projet de trade."""

    plan: TradePlan
    advices: list[Advice]
    risk_pct: float
    position_pct: float

    @property
    def blockers(self) -> list[Advice]:
        return [advice for advice in self.advices if advice.severity is Severity.BLOCKER]

    @property
    def can_proceed(self) -> bool:
        """Vrai si rien de bloquant n'a été relevé."""
        return not self.blockers

    def to_dict(self) -> dict:
        return {
            "can_proceed": self.can_proceed,
            "risk_pct": round(self.risk_pct, 2),
            "position_pct": round(self.position_pct, 2),
            "risk_amount": round(self.plan.risk_amount, 2),
            "stop_distance_pct": round(self.plan.stop_distance_pct, 2),
            "reward_risk": round(self.plan.reward_risk, 2) if self.plan.reward_risk else None,
            "advices": [advice.to_dict() for advice in self.advices],
        }


def suggest_size(equity: float, price: float, stop: float, risk_pct: float = 1.0) -> float:
    """Quantité telle que la perte au stop vaille `risk_pct` % du capital.

    C'est la formule de dimensionnement au risque : elle rend comparables un
    titre à 40 EUR et un titre à 900 EUR, parce qu'elle raisonne sur la perte
    encourue et non sur le montant investi.
    """
    distance = price - stop
    if distance <= 0 or price <= 0 or equity <= 0:
        return 0.0
    return max(0.0, equity * risk_pct / 100.0 / distance)


def review_plan(
    plan: TradePlan,
    account: PaperAccount,
    quote: Quote | None = None,
    prices: dict[str, float] | None = None,
) -> Review:
    """Passe un projet de trade au crible et renvoie des conseils actionnables."""
    prices = prices or {}
    state: AccountState = account.state
    equity = account.equity(prices) or state.total_deposited
    advices: list[Advice] = []

    risk_pct = plan.risk_amount / equity * 100.0 if equity > 0 else 0.0
    position_pct = plan.notional / equity * 100.0 if equity > 0 else 0.0

    # --- Verifications bloquantes ------------------------------------------
    if plan.stop >= plan.price:
        advices.append(
            Advice(
                Severity.BLOCKER,
                "Stop invalide",
                "Le stop doit être sous le prix d'entrée. Sans lui, la perte n'a pas de "
                "limite définie à l'avance.",
            )
        )
    costs = account.costs_for(plan.notional) * 2
    if plan.notional + costs > state.cash + 1e-9:
        advices.append(
            Advice(
                Severity.BLOCKER,
                "Liquidités insuffisantes",
                f"Ce trade demande {plan.notional + costs:,.2f} EUR frais compris, "
                f"le compte dispose de {state.cash:,.2f} EUR.",
            )
        )
    if any(position.symbol == plan.symbol for position in state.positions):
        advices.append(
            Advice(
                Severity.BLOCKER,
                "Position déjà ouverte",
                f"Vous détenez déjà {plan.symbol}. Renforcer une position perdante "
                "double la mise sur une décision qui se révèle fausse.",
            )
        )

    # --- Dimensionnement ----------------------------------------------------
    if risk_pct > MAX_RISK_PCT * 2:
        advices.append(
            Advice(
                Severity.BLOCKER,
                f"Risque de {risk_pct:.1f} % du capital",
                f"Au-delà de {MAX_RISK_PCT * 2:.0f} %, quelques pertes consécutives — "
                "statistiquement banales — suffisent à compromettre le compte. "
                f"Suggestion : {suggest_size(equity, plan.price, plan.stop, 1.0):.4f} titres "
                "pour un risque de 1 %.",
            )
        )
    elif risk_pct > MAX_RISK_PCT:
        advices.append(
            Advice(
                Severity.WARNING,
                f"Risque de {risk_pct:.1f} % du capital",
                f"Au-dessus de la limite de {MAX_RISK_PCT:.0f} % du parcours. Une série "
                "de cinq pertes coûte alors plus de 10 % du compte.",
            )
        )
    elif risk_pct > 0:
        advices.append(
            Advice(
                Severity.GOOD,
                f"Risque maîtrisé : {risk_pct:.2f} % du capital",
                f"Une perte au stop coûtera {plan.risk_amount:,.2f} EUR. "
                "C'est un montant que le compte encaisse sans dommage.",
            )
        )

    # Le risque "au stop" sous-estime le danger reel : un trou de cotation saute
    # le stop et frappe la position ENTIERE. Sur l'univers suivi, des séances a
    # -15 % ne sont pas theoriques (MU en a produit plusieurs en 2026). C'est la
    # taille de la position, pas la distance au stop, qui borne ce risque-la.
    gap_loss_pct = position_pct * GAP_SHOCK_PCT / 100.0
    if position_pct > MAX_CONCENTRATION_PCT:
        advices.append(
            Advice(
                Severity.BLOCKER,
                f"Concentration excessive : {position_pct:.0f} % du compte",
                f"Le risque affiche ({risk_pct:.2f} %) suppose que le stop tienne. Il ne "
                f"tient pas sur un trou de cotation : a l'ouverture, le cours peut sauter "
                f"par-dessus. Une séance a -{GAP_SHOCK_PCT:.0f} % — il y en a eu plusieurs "
                f"sur cet univers en 2026 — coûterait alors {gap_loss_pct:.1f} % du compte. "
                f"Restez sous {MAX_CONCENTRATION_PCT:.0f} % par position.",
            )
        )
    elif position_pct > 40.0:
        advices.append(
            Advice(
                Severity.WARNING,
                f"Concentration : {position_pct:.0f} % du compte",
                f"Une position qui pese plus de 40 % du compte fait dépendre le résultat "
                f"d'un seul titre. Sur un trou de cotation a -{GAP_SHOCK_PCT:.0f} %, le "
                f"stop est saute et la perte atteint {gap_loss_pct:.1f} % du compte, "
                f"quel que soit le risque affiche.",
            )
        )

    # --- Qualite du plan ----------------------------------------------------
    if plan.stop_distance_pct < 2.0:
        advices.append(
            Advice(
                Severity.WARNING,
                f"Stop très serré ({plan.stop_distance_pct:.1f} %)",
                "Une action bouge couramment de 2 à 3 % dans une séance ordinaire. "
                "Un stop aussi proche sera déclenché par le bruit, pas par un vrai "
                "retournement, et vous paierez les frais a chaque fois.",
            )
        )
    elif plan.stop_distance_pct > 25.0:
        advices.append(
            Advice(
                Severity.WARNING,
                f"Stop très large ({plan.stop_distance_pct:.1f} %)",
                "Un stop aussi éloigné laisse la perte s'installer. Il vaut mieux "
                "réduire la taille et rapprocher le stop : même risque, sortie plus nette.",
            )
        )

    ratio = plan.reward_risk
    if ratio is None:
        advices.append(
            Advice(
                Severity.INFO,
                "Aucun objectif défini",
                "Sans objectif, la sortie gagnante se décide dans l'euphorie — "
                "généralement trop tôt. Fixer une cible, même approximative, permet de "
                "juger le trade autrement que par son résultat.",
            )
        )
    elif ratio < 1.0:
        advices.append(
            Advice(
                Severity.WARNING,
                f"Gain vise plus petit que la perte acceptee ({ratio:.2f})",
                "Vous risquez plus que ce que vous espérez. Comme aucun signal ne rend "
                "la hausse plus probable qu'un tirage a pile ou face, ce rapport rend "
                "l'opération perdante en moyenne.",
            )
        )
    else:
        advices.append(
            Advice(
                Severity.GOOD,
                f"Rapport gain/perte de {ratio:.2f}",
                "Vous visez plus que ce que vous risquez : le trade reste rentable même "
                "en se trompant plus d'une fois sur deux.",
            )
        )

    # --- Contexte de marche -------------------------------------------------
    if quote is not None:
        if not quote.is_tradable_session:
            advices.append(
                Advice(
                    Severity.WARNING,
                    f"Marche : {quote.market_status}",
                    "Hors séance principale, les volumes sont faibles et les écarts "
                    "larges. Le prix affiche n'est pas celui auquel vous seriez servi "
                    "a l'ouverture.",
                )
            )
        spread = quote.spread_pct
        if spread is not None and spread > 0.5:
            advices.append(
                Advice(
                    Severity.WARNING,
                    f"Écart achat/vente de {spread:.2f} %",
                    "Cet écart est paye deux fois, à l'entrée et a la sortie. Il ampute "
                    "le gain avant même que le cours ait bouge.",
                )
            )
        if abs(quote.change_pct) > 5.0:
            direction = "bondi" if quote.change_pct > 0 else "chute"
            advices.append(
                Advice(
                    Severity.INFO,
                    f"Le titre a {direction} de {abs(quote.change_pct):.1f} % aujourd'hui",
                    "Entrer après un mouvement violent expose a la réaction inverse. "
                    "Si vous entrez quand même, élargissez le stop et reduisez la taille "
                    "en conséquence.",
                )
            )

    # --- Rappel de fond -----------------------------------------------------
    advices.append(
        Advice(
            Severity.INFO,
            "Ce que ce conseil ne dit pas",
            "Rien ici ne prévoit la direction du cours. Les mesures de ce dépôt montrent "
            "qu'un faisceau de sept signaux concordants n'apporte qu'un point de "
            "probabilité sur le taux de base. Ce qui est vérifie ci-dessus, c'est ce que "
            "vous contrôlez : la taille, le stop et le rapport gain/perte.",
        )
    )

    return Review(plan=plan, advices=advices, risk_pct=risk_pct, position_pct=position_pct)

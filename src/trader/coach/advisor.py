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
from trader.coach.curriculum import (
    MAX_OPEN_RISK_PCT,
    MAX_RISK_PCT,
    MIN_PLANNED_R,
    break_even_rate,
)
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
    trailing_pct: float | None = None
    """Distance du stop suiveur envisagé, en % sous le plus haut atteint."""

    @property
    def notional(self) -> float:
        return self.shares * self.price

    @property
    def effective_stop(self) -> float:
        """Stop RÉELLEMENT en vigueur dès l'entrée.

        Un stop suiveur plus serré que le stop saisi gouverne immédiatement.
        Tout ce qui suit — risque, dimensionnement, rapport gain/perte — se
        calcule donc sur ce niveau-là : afficher un risque que le compte ne
        court pas serait le mensonge le plus utile à corriger dans cette app.
        """
        if not self.trailing_pct:
            return self.stop
        return max(self.stop, self.price * (1.0 - self.trailing_pct / 100.0))

    @property
    def trailing_overrides_stop(self) -> bool:
        """Le suiveur est-il plus serré que le stop saisi ?"""
        return self.effective_stop > self.stop

    @property
    def risk_amount(self) -> float:
        """Perte si le stop en vigueur est touché."""
        return (self.price - self.effective_stop) * self.shares

    @property
    def stop_distance_pct(self) -> float:
        if self.price <= 0:
            return 0.0
        return (self.price - self.effective_stop) / self.price * 100.0

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
            "effective_stop": round(self.plan.effective_stop, 4),
            "trailing_overrides_stop": self.plan.trailing_overrides_stop,
            "reward_risk": (
                None if self.plan.reward_risk is None else round(self.plan.reward_risk, 2)
            ),
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


def suggest_target(price: float, stop: float, ratio: float = MIN_PLANNED_R) -> float | None:
    """Objectif qui vise `ratio` fois la perte acceptée au stop.

    Le parcours note l'utilisateur sur cette asymétrie sans jamais lui dire à
    quel PRIX elle correspond : le seuil restait une note, pas une consigne.
    Cette fonction le traduit en un nombre qu'il peut saisir — ou refuser en
    connaissance de cause.

    Ce n'est pas une prévision : rien ici ne dit que le cours ATTEINDRA ce
    niveau. C'est la contrepartie qu'il faut viser pour que le risque déjà
    accepté ait un sens, et elle se déduit du seul stop.

    `None` quand le stop est au-dessus du prix : il n'y a alors pas de perte
    planifiée dont l'objectif serait un multiple.
    """
    if price <= 0 or stop >= price:
        return None
    return price + ratio * (price - stop)


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

    # --- Stop suiveur -------------------------------------------------------
    if plan.trailing_overrides_stop:
        advices.append(
            Advice(
                Severity.WARNING,
                f"Le suiveur à {plan.trailing_pct:.1f} % remplace votre stop",
                f"Vous avez saisi {plan.stop:,.2f}, mais un suiveur à "
                f"{plan.trailing_pct:.1f} % impose {plan.effective_stop:,.2f} dès "
                f"l'entrée. Le risque affiché ici ({plan.risk_amount:,.2f} EUR) est "
                "calculé sur ce niveau-là, le seul en vigueur. Si vous vouliez la "
                "marge plus large, élargissez le suiveur plutôt que le stop.",
            )
        )
    elif plan.trailing_pct:
        advices.append(
            Advice(
                Severity.INFO,
                f"Stop suiveur à {plan.trailing_pct:.1f} %",
                "Le stop remontera avec le cours et ne redescendra jamais. Il ne rend "
                "pas la hausse plus probable — rien ici ne le fait — mais il retire la "
                "décision de sortie au moment où elle se prend le plus mal, et il "
                "impose l'asymétrie que mesure le palier 5.",
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

    # Le risque par trade ne dit rien du risque du COMPTE. Chaque position prise
    # isolement peut etre irreprochable pendant que leur somme joue une part du
    # capital que le parcours n'autorise pas a perdre — et une seance de baisse
    # generale fait tomber les stops ensemble, pas un par un.
    deja_engage = account.open_risk()
    if state.positions and plan.risk_amount > 0:
        # Le total est un plancher : la part du trade projete ne compte pas
        # encore ses frais, ceux des positions ouvertes le sont deja.
        cumule = deja_engage + plan.risk_amount
        cumule_pct = cumule / equity * 100.0 if equity > 0 else 0.0
        detail = (
            f"{len(state.positions)} position(s) déjà ouverte(s) risquent "
            f"{deja_engage:,.2f} EUR ; celle-ci en ajoute {plan.risk_amount:,.2f}."
        )
        if cumule_pct > MAX_OPEN_RISK_PCT * 2:
            advices.append(
                Advice(
                    Severity.BLOCKER,
                    f"Risque cumulé de {cumule_pct:.1f} % du compte",
                    f"{detail} Une seule séance de baisse générale les fait tomber "
                    f"ensemble : le compte perdrait {cumule:,.2f} EUR d'un coup. Soldez "
                    "ou resserrez une position existante avant d'en ouvrir une de plus.",
                )
            )
        elif cumule_pct > MAX_OPEN_RISK_PCT:
            advices.append(
                Advice(
                    Severity.WARNING,
                    f"Risque cumulé de {cumule_pct:.1f} % du compte",
                    f"{detail} Chaque trade respecte peut-être sa limite, leur somme "
                    f"dépasse les {MAX_OPEN_RISK_PCT:.0f} % du parcours. Les stops ne "
                    "tombent pas indépendamment : ce qui fait baisser un titre du "
                    "secteur fait baisser les autres le même jour.",
                )
            )
        elif cumule > 0:
            advices.append(
                Advice(
                    Severity.GOOD,
                    f"Risque cumulé maîtrisé : {cumule_pct:.2f} % du compte",
                    f"{detail} Tous stops touchés le même jour, le compte perdrait "
                    f"{cumule:,.2f} EUR — sous la limite de {MAX_OPEN_RISK_PCT:.0f} %.",
                )
            )
        else:
            advices.append(
                Advice(
                    Severity.GOOD,
                    "Les positions ouvertes ne peuvent plus rien coûter",
                    f"Leurs stops sont passés au-dessus de leur prix de revient : tous "
                    f"touchés, elles rapporteraient encore {-deja_engage:,.2f} EUR. Seul "
                    f"ce trade-ci met du capital en jeu ({plan.risk_amount:,.2f} EUR).",
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
                f"Le risque affiché ({risk_pct:.2f} %) suppose que le stop tienne. Il ne "
                f"tient pas sur un trou de cotation : à l'ouverture, le cours peut sauter "
                f"par-dessus. Une séance à -{GAP_SHOCK_PCT:.0f} % — il y en a eu plusieurs "
                f"sur cet univers en 2026 — coûterait alors {gap_loss_pct:.1f} % du compte. "
                f"Restez sous {MAX_CONCENTRATION_PCT:.0f} % par position.",
            )
        )
    elif position_pct > 40.0:
        advices.append(
            Advice(
                Severity.WARNING,
                f"Concentration : {position_pct:.0f} % du compte",
                f"Une position qui pèse plus de 40 % du compte fait dépendre le résultat "
                f"d'un seul titre. Sur un trou de cotation à -{GAP_SHOCK_PCT:.0f} %, le "
                f"stop est sauté et la perte atteint {gap_loss_pct:.1f} % du compte, "
                f"quel que soit le risque affiché.",
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
                "retournement, et vous paierez les frais à chaque fois.",
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
    if ratio is None and plan.trailing_pct:
        advices.append(
            Advice(
                Severity.GOOD,
                f"Sortie confiée au stop suiveur ({plan.trailing_pct:g} %)",
                "Pas d'objectif chiffré, mais le gain n'est pas plafonné : le suiveur "
                "remonte avec le cours et ne redescend jamais. Il n'annonce rien sur la "
                "suite du cours et n'améliore aucune espérance de gain — il impose "
                "seulement l'asymétrie, ce que le palier « couper court, laisser courir » "
                "vous demande.",
            )
        )
    elif ratio is None:
        advices.append(
            Advice(
                Severity.INFO,
                "Aucun objectif défini",
                "Sans objectif ni stop suiveur, la sortie gagnante se décide dans "
                "l'euphorie — généralement trop tôt. Fixer une cible, même approximative, "
                "permet de juger le trade autrement que par son résultat.",
            )
        )
    elif ratio < 1.0:
        advices.append(
            Advice(
                Severity.WARNING,
                f"Gain visé plus petit que la perte acceptée ({ratio:.2f})",
                "Vous risquez plus que ce que vous espérez. Comme aucun signal ne rend "
                "la hausse plus probable qu'un tirage à pile ou face, ce rapport rend "
                "l'opération perdante en moyenne.",
            )
        )
    elif ratio < MIN_PLANNED_R:
        advices.append(
            Advice(
                Severity.INFO,
                f"Rapport gain/perte trop juste ({ratio:.2f})",
                f"Il faudrait gagner {break_even_rate(ratio):.0f} % de vos trades pour "
                "seulement rentrer dans vos frais — et les commissions relèvent encore ce "
                f"seuil. Le palier « couper court, laisser courir » demande {MIN_PLANNED_R:.1f} : "
                "éloigner l'objectif ou rapprocher le stop vous laisse une marge d'erreur.",
            )
        )
    else:
        advices.append(
            Advice(
                Severity.GOOD,
                f"Rapport gain/perte de {ratio:.2f}",
                f"Vous visez plus que ce que vous risquez : {break_even_rate(ratio):.0f} % de "
                "trades gagnants suffisent à l'équilibre avant frais. C'est la seule marge "
                "d'erreur qui se décide à l'avance.",
            )
        )

    # --- Contexte de marche -------------------------------------------------
    if quote is not None:
        if not quote.is_tradable_session:
            advices.append(
                Advice(
                    Severity.WARNING,
                    f"Marché : {quote.market_status}",
                    "Hors séance principale, les volumes sont faibles et les écarts "
                    "larges. Le prix affiché n'est pas celui auquel vous seriez servi "
                    "à l'ouverture.",
                )
            )
        spread = quote.spread_pct
        if spread is not None and spread > 0.5:
            advices.append(
                Advice(
                    Severity.WARNING,
                    f"Écart achat/vente de {spread:.2f} %",
                    "Cet écart est payé deux fois, à l'entrée et à la sortie. Il ampute "
                    "le gain avant même que le cours ait bougé.",
                )
            )
        if abs(quote.change_pct) > 5.0:
            direction = "bondi" if quote.change_pct > 0 else "chute"
            advices.append(
                Advice(
                    Severity.INFO,
                    f"Le titre a {direction} de {abs(quote.change_pct):.1f} % aujourd'hui",
                    "Entrer après un mouvement violent expose à la réaction inverse. "
                    "Si vous entrez quand même, élargissez le stop et réduisez la taille "
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
            "probabilité sur le taux de base. Ce qui est vérifié ci-dessus, c'est ce que "
            "vous contrôlez : la taille, le stop et le rapport gain/perte.",
        )
    )

    return Review(plan=plan, advices=advices, risk_pct=risk_pct, position_pct=position_pct)

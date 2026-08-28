"""Progression "zero to hero" : sept paliers, chacun mesurable sur l'historique.

Choix structurant : aucun palier ne récompense le fait de GAGNER. Les mesures
du dépôt (README, "Pistes mesurees puis ecartees") montrent qu'aucun signal
teste n'a de pouvoir prédictif — sur dix trades, le résultat est surtout du
hasard. Un programme qui ferait progresser au résultat apprendrait donc a
confondre chance et compétence, exactement le biais que ce projet combat.

Ce qui est evalue est le PROCESS, qui lui depend entierement de l'utilisateur :
définir son stop avant d'entrer, dimensionner sa position, tenir sa decision,
ne pas surtrader, couper court et laisser courir. Ce sont les seuls leviers
dont on sait qu'ils changent le résultat a long terme.

Le dernier palier demande de la Régularité sur trente trades, seule échelle ou
la discipline se distingue statistiquement de la chance.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from trader.coach.account import AccountState, ClosedTrade

MAX_RISK_PCT: float = 2.0
"""Perte au stop toleree, en % du capital. Au-delà, une série de pertes
normale suffit a effacer le compte."""

MAX_OPEN_RISK_PCT: float = 6.0
"""Perte cumulee toleree, en % du capital, si TOUS les stops tombaient.

`MAX_RISK_PCT` borne un trade ; il ne borne pas le compte. Cinq positions a
2 % chacune sont cinq trades irreprochables et un compte qui joue 10 % sur une
seule seance de baisse generale — cas ou les stops tombent ensemble, et non
independamment. Trois trades pleins (3 x 2 %) est la limite au-dela de laquelle
une seule mauvaise seance coute plus que ce que le parcours autorise a perdre.
"""

OVERTRADE_PER_WEEK: int = 5
"""Nombre de trades par semaine au-delà duquel on parle de surtrading."""

MIN_PLANNED_R: float = 1.5
"""Gain visé minimal, en multiples de la perte acceptée, décidé A L'ENTREE."""

REVENGE_MULTIPLE: float = 1.5
"""Multiple de la mise habituelle au-dela duquel un trade pris APRES UNE PERTE
cesse d'etre une decision et devient une revanche.

Aucun palier ne borne cela : chaque trade peut respecter `MAX_RISK_PCT` et leur
somme `MAX_OPEN_RISK_PCT` pendant que l'utilisateur remet systematiquement plus
gros juste apres avoir perdu — pour "se refaire". C'est la faute qui vide les
comptes le plus vite, parce qu'elle augmente la mise exactement quand le
jugement est le plus mauvais, et parce qu'elle est invisible dans les moyennes :
elle ne se voit que dans l'ORDRE des trades.

1.5 fois la mise habituelle, et non deux : au-dela de la moitie en plus, le
choix ne s'explique plus par la seule difference de distance au stop.
"""


def break_even_rate(ratio: float) -> float:
    """Part de trades gagnants, en %, qu'il faut atteindre pour finir à l'équilibre.

    Gagner `ratio` fois la mise `p` fois sur cent et la perdre le reste du temps
    laisse `p * ratio - (1 - p)` : nul pour `p = 1 / (1 + ratio)`. Chiffre AVANT
    frais — commissions et écart de cotation relèvent le seuil réel.

    C'est le seul repère chiffré que cette app puisse donner honnêtement sur
    l'issue d'un trade : il ne suppose aucune prévision, seulement ce que
    l'utilisateur a placé en face de son stop. C'est aussi ce qui justifie
    `MIN_PLANNED_R` — à 1.5, il reste de la marge (40 %) ; à 1.1, presque plus.
    """
    return 100.0 / (1.0 + ratio)


def usual_risk(history: list[ClosedTrade], window: int = 10) -> float | None:
    """Mise habituelle de l'utilisateur : la MEDIANE des risques planifies.

    Mediane et non moyenne : c'est ce chiffre qui sert de reference pour
    reperer un trade surdimensionne, et une moyenne se laisserait tirer vers le
    haut par les trades memes qu'il s'agit de detecter — le repere monterait
    avec la faute, jusqu'a ne plus rien signaler.

    `None` quand il n'y a rien a comparer : sans trades passes, ou quand tous
    ont un risque planifie nul (stops deja au-dessus de l'entree), toute
    comparaison serait arbitraire, et un conseil arbitraire est pire que pas
    de conseil.
    """
    mises = sorted(t.planned_risk for t in history[-window:] if t.planned_risk > 0)
    if not mises:
        return None
    milieu = len(mises) // 2
    if len(mises) % 2:
        return mises[milieu]
    return (mises[milieu - 1] + mises[milieu]) / 2.0


def revenge_multiple(history: list[ClosedTrade], risk_amount: float) -> float | None:
    """Combien de fois la mise habituelle represente un trade pris APRES UNE PERTE.

    `None` des que la question ne se pose pas : aucun trade passe, dernier trade
    gagnant, ou mise habituelle non mesurable. La comparaison n'a de sens que
    juste apres une perte — c'est la, et seulement la, que remettre plus gros
    cesse d'etre un choix de dimensionnement pour devenir un rattrapage.

    N'utilise que des trades DEJA CLOTURES : rien ici ne suppose de connaitre
    la suite du cours.
    """
    if not history or history[-1].is_win:
        return None
    habituel = usual_risk(history)
    if habituel is None or habituel <= 0 or risk_amount <= 0:
        return None
    return risk_amount / habituel


@dataclass(frozen=True, slots=True)
class Level:
    """Un palier du parcours."""

    number: int
    key: str
    title: str
    goal: str
    why: str
    check: Callable[[AccountState], tuple[bool, str]]

    def evaluate(self, state: AccountState) -> LevelStatus:
        """Evalue le palier sur l'etat courant du compte."""
        done, detail = self.check(state)
        return LevelStatus(level=self, achieved=done, detail=detail)


@dataclass(frozen=True, slots=True)
class LevelStatus:
    """Résultat de l'evaluation d'un palier."""

    level: Level
    achieved: bool
    detail: str

    def to_dict(self) -> dict:
        return {
            "number": self.level.number,
            "key": self.level.key,
            "title": self.level.title,
            "goal": self.level.goal,
            "why": self.level.why,
            "achieved": self.achieved,
            "detail": self.detail,
        }


# ------------------------------------------------------------------ criteres


def _risk_pct(trade: ClosedTrade, capital: float) -> float:
    """Risque planifie du trade en % du capital de reference."""
    return trade.planned_risk / capital * 100.0 if capital > 0 else 0.0


def _check_first_trade(state: AccountState) -> tuple[bool, str]:
    total = len(state.history) + len(state.positions)
    if total == 0:
        return False, "Aucune position ouverte pour l'instant."
    return True, f"{total} position(s) engagée(s) avec un stop défini dès l'entrée."


def _check_stop_discipline(state: AccountState) -> tuple[bool, str]:
    trades = state.history
    if len(trades) < 5:
        return False, f"{len(trades)}/5 trades clôturés."
    widened = [trade for trade in trades if trade.stop_moved_against]
    if widened:
        symbols = ", ".join(sorted({trade.symbol for trade in widened}))
        return False, f"Stop élargi sur {len(widened)} trade(s) : {symbols}."
    breached = [trade for trade in trades if not trade.respected_stop]
    if breached:
        return False, f"{len(breached)} perte(s) au-delà de l'enveloppe prévue."
    return True, f"{len(trades)} trades sans jamais reculer un stop."


def _check_sizing(state: AccountState) -> tuple[bool, str]:
    trades = state.history
    if len(trades) < 8:
        return False, f"{len(trades)}/8 trades clôturés."
    capital = state.total_deposited
    excessive = [t for t in trades if _risk_pct(t, capital) > MAX_RISK_PCT]
    if excessive:
        worst = max(excessive, key=lambda t: _risk_pct(t, capital))
        return False, (
            f"{len(excessive)} trade(s) risquaient plus de {MAX_RISK_PCT:.0f} % du capital "
            f"(pire : {_risk_pct(worst, capital):.1f} % sur {worst.symbol})."
        )
    return True, f"Les {len(trades)} trades risquaient au plus {MAX_RISK_PCT:.0f} % du capital."


def _check_patience(state: AccountState) -> tuple[bool, str]:
    trades = state.history
    if len(trades) < 10:
        return False, f"{len(trades)}/10 trades clôturés."
    by_week: dict[tuple[int, int], int] = {}
    for trade in trades:
        opened = datetime.fromisoformat(trade.opened_at)
        key = opened.isocalendar()[:2]
        by_week[key] = by_week.get(key, 0) + 1
    busiest = max(by_week.values())
    if busiest > OVERTRADE_PER_WEEK:
        return (
            False,
            f"Jusqu'a {busiest} trades sur une même semaine (limite {OVERTRADE_PER_WEEK}).",
        )
    return True, f"Jamais plus de {busiest} trades par semaine."


def planned_ratio(trade: ClosedTrade) -> float | None:
    """Gain visé rapporté a la perte acceptée, tel qu'il etait decide A L'ENTREE.

    Public : l'historique de l'interface s'en sert pour montrer trade par trade
    ce que le palier 5 reproche globalement. Un palier qui compte des fautes
    sans dire lesquelles n'apprend rien.

    `None` quand rien n'a ete planifie en face du stop : ni objectif, ni stop
    suiveur. Ce trade-la n'a pas de sortie prevue par le haut, elle sera
    improvisee — c'est exactement ce que le palier cherche a eliminer.

    Un stop suiveur vaut l'infini : il ne pose aucun plafond au gain, c'est tout
    son interet. Un risque planifie nul — stop deja remonte au-dessus du prix
    d'entree — aussi, faute de quoi la division n'aurait pas de sens.
    """
    if trade.trailing_pct:
        return math.inf
    if trade.target is None:
        return None
    risk = trade.planned_risk
    if risk <= 0:
        return math.inf
    return (trade.target - trade.entry_price) * trade.shares / risk


def _check_asymmetry(state: AccountState) -> tuple[bool, str]:
    trades = state.history
    if len(trades) < 12:
        return False, f"{len(trades)}/12 trades clôturés."
    planned = [(trade, planned_ratio(trade)) for trade in trades]
    unplanned = [trade for trade, ratio in planned if ratio is None]
    if unplanned:
        symbols = ", ".join(sorted({trade.symbol for trade in unplanned}))
        return False, (
            f"{len(unplanned)} trade(s) sans objectif ni stop suiveur : {symbols}. "
            "Rien n'etait prevu en face du stop."
        )
    weak = [(trade, ratio) for trade, ratio in planned if ratio < MIN_PLANNED_R]
    if weak:
        worst, worst_ratio = min(weak, key=lambda pair: pair[1])
        return False, (
            f"{len(weak)} trade(s) visaient moins de {MIN_PLANNED_R:.1f} fois le risque "
            f"accepte (pire : {worst_ratio:.1f} fois sur {worst.symbol})."
        )
    trailing = sum(1 for trade in trades if trade.trailing_pct)
    return True, (
        f"Les {len(trades)} trades visaient au moins {MIN_PLANNED_R:.1f} fois la perte "
        f"acceptée, dont {trailing} sans plafond grâce au stop suiveur."
    )


def _check_drawdown(state: AccountState) -> tuple[bool, str]:
    trades = state.history
    if len(trades) < 20:
        return False, f"{len(trades)}/20 trades clôturés."
    capital = state.total_deposited
    if capital <= 0:
        return False, "Aucun capital saisi."
    equity, peak, worst = capital, capital, 0.0
    for trade in sorted(trades, key=lambda t: t.closed_at):
        equity += trade.pnl
        peak = max(peak, equity)
        worst = max(worst, (peak - equity) / peak * 100.0)
    if worst > 15.0:
        return False, f"Repli maximal de {worst:.1f} % (limite 15 %)."
    return True, f"Repli maximal contenu a {worst:.1f} %."


def _check_consistency(state: AccountState) -> tuple[bool, str]:
    trades = state.history
    if len(trades) < 30:
        return False, f"{len(trades)}/30 trades clôturés."
    widened = sum(trade.stop_moved_against for trade in trades)
    capital = state.total_deposited
    oversized = sum(1 for t in trades if _risk_pct(t, capital) > MAX_RISK_PCT)
    if widened or oversized:
        return False, (
            f"{widened} stop(s) élargi(s) et {oversized} position(s) surdimensionnée(s) "
            "sur les 30 derniers trades."
        )
    return True, (
        f"{len(trades)} trades : stops tenus, tailles maîtrisées. "
        "La discipline est devenue un automatisme."
    )


LEVELS: tuple[Level, ...] = (
    Level(
        1,
        "premier_trade",
        "Le premier trade",
        "Ouvrir une position en ayant défini son stop AVANT d'entrer.",
        "Un trade sans niveau de sortie prévu n'est pas un trade : c'est un pari "
        "dont on décidera la fin sous le coup de l'émotion, au pire moment.",
        _check_first_trade,
    ),
    Level(
        2,
        "discipline_stop",
        "Le stop est sacré",
        "Clôturer 5 trades sans jamais reculer un stop.",
        "Élargir un stop pour ne pas matérialiser une perte est l'erreur la plus "
        "coûteuse du débutant : elle transforme une petite perte prévue en grande "
        "perte subie.",
        _check_stop_discipline,
    ),
    Level(
        3,
        "dimensionnement",
        "Dimensionner sa position",
        f"8 trades ne risquant jamais plus de {MAX_RISK_PCT:.0f} % du capital.",
        "Avec 10 % de risque par trade, cinq pertes d'affilée — parfaitement "
        "banales — effacent la moitié du compte. Avec 2 %, elles coûtent 10 %.",
        _check_sizing,
    ),
    Level(
        4,
        "patience",
        "Ne pas surtrader",
        f"10 trades sans jamais dépasser {OVERTRADE_PER_WEEK} trades sur une semaine.",
        "Chaque aller-retour coûte des frais et du slippage. Multiplier les trades "
        "multiplie les coûts avec certitude, les gains seulement avec espoir.",
        _check_patience,
    ),
    Level(
        5,
        "asymetrie",
        "Couper court, laisser courir",
        f"Sur 12 trades, viser dès l'entrée au moins {MIN_PLANNED_R:.1f} fois la perte "
        "acceptée — par un objectif, ou par un stop suiveur qui ne plafonne rien.",
        "Comme aucun signal ne prédit la direction, le seul réglage qui reste est "
        "l'asymétrie : perdre peu souvent ne sert à rien si l'on perd gros. Et cette "
        "asymétrie se décide à l'entrée. Noter le gain moyen OBTENU reviendrait à "
        "noter la chance — sur douze trades sans pouvoir prédictif, il dépend surtout "
        "du hasard ; ce que vous choisissez, c'est ce que vous mettez en face du stop.",
        _check_asymmetry,
    ),
    Level(
        6,
        "resilience",
        "Encaisser sans casser",
        "20 trades avec un repli maximal du compte sous 15 %.",
        "Un compte qui recule de 50 % doit doubler pour revenir à l'équilibre. "
        "Limiter le repli vaut mieux que chercher le gros coup.",
        _check_drawdown,
    ),
    Level(
        7,
        "hero",
        "La régularité",
        "30 trades, stops tenus et tailles maîtrisées de bout en bout.",
        "Trente trades est la première échelle où la discipline se distingue "
        "statistiquement de la chance. Y arriver sans faute de process est le "
        "vrai diplôme.",
        _check_consistency,
    ),
)


@dataclass(frozen=True, slots=True)
class Progress:
    """État d'avancement complet dans le parcours."""

    levels: list[LevelStatus]

    @property
    def current(self) -> LevelStatus:
        """Premier palier non valide — l'objectif du moment."""
        for status in self.levels:
            if not status.achieved:
                return status
        return self.levels[-1]

    @property
    def completed(self) -> int:
        """Nombre de paliers valides."""
        return sum(status.achieved for status in self.levels)

    @property
    def is_hero(self) -> bool:
        return self.completed == len(self.levels)

    @property
    def rank(self) -> str:
        """Titre correspondant à l'avancement."""
        titles = (
            "Zero",
            "Novice",
            "Apprenti",
            "Praticien",
            "Confirme",
            "Aguerri",
            "Expert",
            "Hero",
        )
        return titles[min(self.completed, len(titles) - 1)]

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "completed": self.completed,
            "total": len(self.levels),
            "is_hero": self.is_hero,
            "current": self.current.to_dict(),
            "levels": [status.to_dict() for status in self.levels],
        }


def evaluate_progress(state: AccountState) -> Progress:
    """Evalue les sept paliers sur l'historique du compte."""
    return Progress(levels=[level.evaluate(state) for level in LEVELS])

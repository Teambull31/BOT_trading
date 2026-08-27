"""Debrief automatique a la clôture d'un trade.

Le principe qui gouverne ce module : juger la DÉCISION, pas le résultat. Un
trade gagnant peut être mal joué (taille excessive, stop reculé, sortie au
hasard) et un trade perdant parfaitement joué. Comme aucun signal testé dans ce
dépôt n'a de pouvoir prédictif mesurable, le résultat d'un trade isolé est
majoritairement du bruit — féliciter un gain et sermonner une perte apprendrait
donc surtout à confondre chance et compétence.

Le debrief remonte donc d'abord ce qui était sous contrôle, puis chiffre ce
qu'une décision différente aurait donné — parce qu'un conseil sans chiffre ne
change aucun comportement.
"""

from __future__ import annotations

from dataclasses import dataclass

from trader.coach.account import AccountState, ClosedTrade
from trader.coach.curriculum import MAX_RISK_PCT


@dataclass(frozen=True, slots=True)
class Lesson:
    """Un enseignement tiré du trade."""

    kind: str
    """'process' (ce qui était sous contrôle), 'chiffre' (ce qu'aurait donné
    une autre décision) ou 'contexte' (rappel utile)."""
    title: str
    message: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "title": self.title, "message": self.message}


@dataclass(frozen=True, slots=True)
class Debrief:
    """Analyse complète d'un trade terminé."""

    trade: ClosedTrade
    verdict: str
    well_played: bool
    lessons: list[Lesson]

    def to_dict(self) -> dict:
        return {
            "symbol": self.trade.symbol,
            "pnl": round(self.trade.pnl, 2),
            "return_pct": round(self.trade.return_pct, 2),
            "holding_days": self.trade.holding_days,
            "verdict": self.verdict,
            "well_played": self.well_played,
            "lessons": [lesson.to_dict() for lesson in self.lessons],
        }


def debrief_trade(trade: ClosedTrade, state: AccountState) -> Debrief:
    """Produit le debrief d'un trade qui vient d'être clôturé."""
    lessons: list[Lesson] = []
    faults = 0
    capital = state.total_deposited or 1.0
    risk_pct = trade.planned_risk / capital * 100.0

    # ------------------------------------------------- ce qui était sous contrôle
    if trade.stop_moved_against:
        faults += 1
        lessons.append(
            Lesson(
                "process",
                "Le stop a été reculé en cours de route",
                "C'est la faute la plus coûteuse du métier. Le stop initial était un "
                "engagement pris à froid ; le reculer revient à laisser la position "
                "perdante décider à votre place, au moment où vous jugez le plus mal. "
                "Si le stop vous paraît trop serré, la réponse est de réduire la taille "
                "AVANT d'entrer, jamais d'élargir après.",
            )
        )
    else:
        lessons.append(
            Lesson(
                "process",
                "Stop tenu jusqu'au bout",
                "Le niveau de sortie décidé à froid a été respecté. C'est ce que mesure "
                "le parcours, et c'est ce qui distingue un praticien sur la durée.",
            )
        )

    if risk_pct > MAX_RISK_PCT:
        faults += 1
        lessons.append(
            Lesson(
                "process",
                f"Position trop grosse : {risk_pct:.1f} % du capital en jeu",
                f"Au-delà de {MAX_RISK_PCT:.0f} %, cinq pertes consécutives — une série "
                f"parfaitement ordinaire — coûtent {risk_pct * 5:.0f} % du compte. "
                "Diviser la taille par deux ne divise pas les gains par deux sur la "
                "durée : cela évite surtout la perte qui met hors jeu.",
            )
        )

    # ------------------------------------------ qualité de la sortie : UNE leçon
    # Un seul enseignement ici. Afficher « sortie au-delà du stop prévu » ET
    # « stop exécuté comme prévu » dans le même debrief ne laisse rien
    # apprendre : les deux se contredisent, donc aucun des deux ne porte.
    gap_pct = (trade.stop - trade.exit_price) / trade.stop * 100.0 if trade.stop > 0 else 0.0
    slippage = (trade.stop - trade.exit_price) * trade.shares
    overshoot = not trade.respected_stop and not trade.stop_moved_against

    if overshoot and trade.stop_locks_gain:
        lessons.append(
            Lesson(
                "contexte",
                "Le stop protégeait un gain, la sortie est une perte",
                f"Le stop ({trade.stop:,.2f}) était au-dessus du prix d'entrée "
                f"({trade.entry_price:,.2f}) : touché à son niveau, il verrouillait un "
                f"gain. La sortie s'est faite à {trade.exit_price:,.2f}, soit "
                f"{abs(trade.pnl):,.2f} EUR de perte. Le cours est passé par-dessus le "
                "niveau sans s'y arrêter. Un stop borne une perte, il ne la garantit "
                "pas — seule la taille de la position est vraiment sous contrôle.",
            )
        )
    elif trade.exit_reason == "stop_touche" and gap_pct > 1.0:
        lessons.append(
            Lesson(
                "chiffre",
                f"Sortie {gap_pct:.1f} % sous le niveau du stop",
                f"Le stop était à {trade.stop:,.2f}, la sortie s'est faite à "
                f"{trade.exit_price:,.2f} : {slippage:,.2f} EUR de plus que prévu. "
                "Un stop n'est pas un prix garanti, c'est un ordre déclenché — entre "
                "le franchissement et l'exécution, le cours continue de bouger. "
                "C'est pour cela que la TAILLE de la position protège mieux qu'un "
                "stop rapproché : elle, on la choisit vraiment.",
            )
        )
    elif overshoot:
        lessons.append(
            Lesson(
                "contexte",
                "Sortie au-delà du stop prévu",
                f"La perte ({abs(trade.pnl):,.2f} EUR) dépasse l'enveloppe prévue "
                f"({trade.planned_risk:,.2f} EUR). Sans stop élargi, c'est le signe d'un "
                "trou de cotation : le cours a sauté par-dessus le niveau sans y passer. "
                "C'est le risque résiduel qu'aucun stop n'élimine, et la raison pour "
                "laquelle la taille compte plus que le stop.",
            )
        )
    elif trade.exit_reason == "stop_touche":
        lessons.append(
            Lesson(
                "process",
                "Stop exécuté comme prévu",
                f"Sortie à {trade.exit_price:,.2f} pour un stop à {trade.stop:,.2f}. "
                "La perte est restée dans l'enveloppe décidée avant d'entrer : "
                "c'est exactement ce à quoi sert un stop, et c'est ce qui permet "
                "de se tromper souvent sans jamais être mis hors jeu.",
            )
        )

    # ------------------------------------------ ce qu'une autre décision donne
    peak = trade.highest_price
    if peak > trade.exit_price and trade.shares > 0:
        missed = (peak - trade.exit_price) * trade.shares
        peak_gain_pct = (peak / trade.entry_price - 1.0) * 100.0
        if missed > abs(trade.pnl) * 0.25 and peak_gain_pct > 1.0:
            lessons.append(
                Lesson(
                    "chiffre",
                    f"Le titre a valu jusqu'à {peak:,.2f} ({peak_gain_pct:+.1f} %)",
                    f"Sortir au plus haut aurait rapporté {missed:,.2f} EUR de plus. "
                    "Personne ne sort au plus haut — l'objectif n'est pas là. Mais si "
                    "l'écart se répète, c'est qu'un stop suiveur, qui remonte avec le "
                    "cours au lieu d'une cible fixe, capterait davantage de ces "
                    "mouvements.",
                )
            )

    # Un risque planifie nul (stop au-dessus de l'entree) ne se juge pas au
    # rapport gain/perte : ce rapport serait infiniment favorable, pas nul.
    if trade.pnl < 0 and trade.target and trade.planned_risk > 0:
        planned_gain = (trade.target - trade.entry_price) * trade.shares
        if planned_gain > 0:
            ratio = planned_gain / trade.planned_risk
            if ratio < 1.5:
                lessons.append(
                    Lesson(
                        "chiffre",
                        f"Le rapport gain/perte visé n'était que de {ratio:.2f}",
                        f"Vous risquiez {trade.planned_risk:,.2f} EUR pour en espérer "
                        f"{planned_gain:,.2f}. A ce rapport, il faut avoir raison bien plus "
                        "d'une fois sur deux pour s'en sortir — or aucun signal mesuré dans "
                        "ce dépôt ne procure un tel avantage. Viser au moins deux fois le "
                        "risque rend le trade rentable même avec 40 % de réussite.",
                    )
                )

    if trade.holding_days == 0 and trade.pnl < 0:
        lessons.append(
            Lesson(
                "process",
                "Trade ouvert et fermé le même jour",
                "Une sortie le jour même signifie souvent un stop placé dans le bruit de "
                "la séance. Une action bouge couramment de 2 à 3 % sans que rien ne se "
                "passe : sous cette distance, c'est le hasard qui déclenche la sortie, "
                "et les frais sont payés pour rien.",
            )
        )

    costs_share = trade.costs / abs(trade.pnl) * 100.0 if trade.pnl else 0.0
    if trade.costs > 0 and abs(trade.pnl) > 0 and costs_share > 25.0:
        lessons.append(
            Lesson(
                "chiffre",
                f"Les frais représentent {costs_share:.0f} % du résultat",
                f"{trade.costs:,.2f} EUR de frais pour un résultat de {trade.pnl:,.2f} EUR. "
                "Sur de petites positions, le plancher de commission pèse énormément : "
                "moins de trades et des positions plus conséquentes valent mieux que "
                "l'inverse.",
            )
        )

    # --------------------------------------------------------------- verdict
    well_played = faults == 0
    if well_played and trade.is_win:
        verdict = (
            "Trade bien mené et gagnant. Attention toutefois : un gain isolé ne prouve "
            "rien sur la méthode, seule la répétition le fera."
        )
    elif well_played:
        verdict = (
            "Trade PERDANT mais bien mené. C'est le cas le plus important à comprendre : "
            "la perte était prévue, dimensionnée et tenue. Rien à corriger — répéter ce "
            "process est exactement ce qu'il faut faire."
        )
    elif trade.is_win:
        verdict = (
            "Trade gagnant mais MAL mené. Le résultat masque une faute de process qui, "
            "répétée, finira par coûter cher. Le gain ici doit peu à la méthode."
        )
    else:
        verdict = (
            "Trade perdant et faute de process identifiée. C'est la combinaison la plus "
            "instructive : la perte était évitable, ou du moins réductible."
        )

    return Debrief(trade=trade, verdict=verdict, well_played=well_played, lessons=lessons)


def recurring_patterns(state: AccountState, minimum: int = 5) -> list[Lesson]:
    """Detecte les erreurs qui REVIENNENT, seules vraiment corrigeables.

    Une faute isolee est du bruit ; la même faute cinq fois est une habitude.
    Ce sont ces habitudes que l'entrainement doit changer.
    """
    trades = state.history
    if len(trades) < minimum:
        return []

    lessons: list[Lesson] = []
    capital = state.total_deposited or 1.0
    recent = trades[-20:]

    widened = sum(trade.stop_moved_against for trade in recent)
    if widened >= max(2, len(recent) // 5):
        lessons.append(
            Lesson(
                "process",
                f"Stop reculé sur {widened} des {len(recent)} derniers trades",
                "Ce n'est plus un accident, c'est une habitude. Elle vient presque "
                "toujours d'un stop placé trop près au départ. Placez-le à une distance "
                "que le titre parcourt normalement en une séance, et ajustez la TAILLE "
                "pour que la perte reste supportable.",
            )
        )

    oversized = sum(1 for t in recent if t.planned_risk / capital * 100.0 > MAX_RISK_PCT)
    if oversized >= max(2, len(recent) // 4):
        lessons.append(
            Lesson(
                "process",
                f"{oversized} des {len(recent)} derniers trades étaient surdimensionnés",
                "Le dimensionnement est le seul réglage dont l'effet est certain. "
                "Fixez la perte au stop à 1 % du capital et déduisez-en la quantité, "
                "au lieu de choisir un montant à investir puis d'y coller un stop.",
            )
        )

    same_day = sum(1 for t in recent if t.holding_days == 0)
    if same_day >= max(3, len(recent) // 3):
        lessons.append(
            Lesson(
                "process",
                f"{same_day} des {len(recent)} derniers trades ont duré moins d'un jour",
                "Vos stops sont dans le bruit de la séance. Chaque aller-retour paie des "
                "frais avec certitude et n'attrape une tendance qu'avec espoir. Élargir "
                "le stop et réduire la taille change ce rapport.",
            )
        )

    losses = [abs(t.pnl) for t in trades if not t.is_win]
    wins = [t.pnl for t in trades if t.is_win]
    if len(losses) >= 3 and len(wins) >= 3:
        avg_loss, avg_win = sum(losses) / len(losses), sum(wins) / len(wins)
        if avg_loss > avg_win * 1.2:
            lessons.append(
                Lesson(
                    "chiffre",
                    f"Perte moyenne {avg_loss:,.2f} EUR contre gain moyen {avg_win:,.2f} EUR",
                    "Vos pertes sont plus grosses que vos gains : vous laissez courir les "
                    "mauvaises positions et coupez les bonnes. C'est le réflexe naturel, "
                    "et c'est exactement l'inverse de ce qu'il faut faire. Un stop suiveur "
                    "impose mécaniquement la bonne asymétrie.",
                )
            )

    return lessons

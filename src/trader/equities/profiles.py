"""Profils de risque : le curseur entre esperance de gain et drawdown supportable.

Il n'existe pas de reglage qui augmente le gain sans augmenter le risque. Tout
ce que fait un profil, c'est deplacer le curseur — et le role de ce module est
de rendre ce deplacement EXPLICITE et MESURE, au lieu de le cacher derriere un
adjectif rassurant.

Trois leviers, et trois seulement, agissent reellement sur le couple
rendement/risque de ce systeme :

1. L'EXPOSITION : quelle part du capital est investie quand un signal existe.
   C'est le levier dominant. Passer de 40 % a 100 % investi multiplie a la fois
   le gain et la perte, presque lineairement.
2. La CONCENTRATION : combien de positions se partagent cette exposition. Moins
   de positions = plus de dependance a chaque titre. On gagne plus quand on a
   raison, on perd plus quand on a tort, et la diversification qui amortit les
   accidents disparait.
3. La LONGUEUR DE LAISSE : la largeur des stops. Des stops serres coupent vite
   les pertes mais tuent aussi les tendances gagnantes par des sorties
   prematurees ; des stops larges laissent respirer la position au prix de
   pertes unitaires plus lourdes.

Aucun profil n'utilise de LEVIER. Avec 1000 euros, un appel de marge sur une
position a effet de levier peut effacer le capital plus vite que n'importe quel
stop ne peut le proteger.
"""

from __future__ import annotations

from dataclasses import dataclass

from trader.equities.backtest import RiskParams
from trader.equities.strategy import TrendParams


@dataclass(frozen=True, slots=True)
class RiskProfile:
    """Un point sur le curseur risque/rendement."""

    key: str
    label: str
    strategy: TrendParams
    risk: RiskParams
    intent: str
    expected_behaviour: str

    @property
    def max_exposure_pct(self) -> float:
        """Part maximale du capital investie simultanement."""
        return min(100.0, self.risk.max_position_pct * self.risk.max_positions)

    def describe(self) -> str:
        """Description lisible du profil."""
        return (
            f"{self.label}\n"
            f"    exposition max   : {self.max_exposure_pct:.0f} % du capital "
            f"({self.risk.max_positions} position(s) x {self.risk.max_position_pct:.0f} %)\n"
            f"    stops            : initial {self.strategy.initial_stop_atr:.1f}xATR, "
            f"suiveur {self.strategy.trailing_atr:.1f}xATR\n"
            f"    intention        : {self.intent}\n"
            f"    a quoi s'attendre: {self.expected_behaviour}"
        )


def _trend(initial_stop: float, trailing: float) -> TrendParams:
    """Strategie de tendance avec des stops de largeur donnee."""
    return TrendParams(entry_mode="trend", initial_stop_atr=initial_stop, trailing_atr=trailing)


PROFILES: dict[str, RiskProfile] = {
    "defensif": RiskProfile(
        key="defensif",
        label="DEFENSIF — preserver d'abord",
        strategy=_trend(initial_stop=3.0, trailing=4.0),
        risk=RiskParams(sizing_mode="target_weight", max_position_pct=20.0, max_positions=3),
        intent="traverser les mauvaises annees sans degats, accepter de peu gagner dans les bonnes",
        expected_behaviour=(
            "drawdown contenu, forte sous-performance en marche haussier, beaucoup de cash dormant"
        ),
    ),
    "equilibre": RiskProfile(
        key="equilibre",
        label="EQUILIBRE — investi mais diversifie",
        strategy=_trend(initial_stop=5.0, trailing=6.0),
        risk=RiskParams(sizing_mode="target_weight", max_position_pct=33.0, max_positions=3),
        intent="capter l'essentiel des tendances tout en repartissant le risque sur trois titres",
        expected_behaviour="suit le marche de loin a la hausse, amortit nettement les baisses",
    ),
    "offensif": RiskProfile(
        key="offensif",
        label="OFFENSIF — concentre sur les tendances les plus fortes",
        strategy=_trend(initial_stop=6.0, trailing=7.0),
        risk=RiskParams(sizing_mode="target_weight", max_position_pct=50.0, max_positions=2),
        intent="concentrer le capital sur les deux meilleures tendances, laisser courir",
        expected_behaviour=(
            "gains superieurs quand la tendance porte, drawdowns nettement plus profonds, "
            "resultat tres dependant de deux titres"
        ),
    ),
    "maximal": RiskProfile(
        key="maximal",
        label="MAXIMAL — tout sur la tendance la plus forte",
        strategy=_trend(initial_stop=7.0, trailing=8.0),
        risk=RiskParams(sizing_mode="target_weight", max_position_pct=100.0, max_positions=1),
        intent="maximiser l'esperance de gain, sans aucune diversification",
        expected_behaviour=(
            "le resultat depend d'un seul titre a la fois ; un accident isole "
            "(resultats decevants, choc sectoriel) frappe 100 % du capital"
        ),
    ),
    "budget_risque": RiskProfile(
        key="budget_risque",
        label="BUDGET DE RISQUE — perte bornee par trade",
        strategy=_trend(initial_stop=3.0, trailing=4.0),
        risk=RiskParams(
            sizing_mode="risk",
            risk_per_trade_pct=1.0,
            max_position_pct=35.0,
            max_positions=3,
        ),
        intent="ne jamais risquer plus de 1 % du capital sur un trade donne",
        expected_behaviour=(
            "pertes unitaires tres faibles, mais capital majoritairement en cash : "
            "capte une petite fraction des hausses"
        ),
    ),
}

ORDER: tuple[str, ...] = ("budget_risque", "defensif", "equilibre", "offensif", "maximal")
"""Profils du plus prudent au plus agressif, pour un affichage lisible."""


def get_profile(key: str) -> RiskProfile:
    """Retourne un profil par sa cle."""
    normalized = key.strip().lower()
    if normalized not in PROFILES:
        raise ValueError(f"profil inconnu : {key!r}. Choix possibles : {', '.join(ORDER)}")
    return PROFILES[normalized]

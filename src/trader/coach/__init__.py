"""Mode d'entrainement "zero to hero" : compte fictif, cours reels, coaching.

Ce paquet n'execute AUCUN ordre reel et ne se connecte a aucun courtier. Il
sert a acquerir le geste — dimensionner, placer un stop, tenir sa discipline —
sur des cours vrais, avec de l'argent qui ne l'est pas.

Parti pris assume, appuye sur les mesures du depot (voir README, section
"Pistes mesurees puis ecartees") : le faisceau de signaux n'a aucun pouvoir
predictif mesurable (+0.9 point contre le taux de base) et le levier au-dela de
2x detruit le capital. L'application n'enseigne donc pas a prevoir le marche —
personne ne sait le faire — mais a gerer le risque, ce qui se mesure et
s'ameliore reellement.
"""

"""Agent de trading autonome adaptatif.

Principes non negociables :
- Le risk management est independant et a toujours le dernier mot.
- Aucune strategie n'est eternelle : ensemble + detection de regime.
- Lutte explicite contre le biais de confirmation (DevilAdvocate).
- Paper trading d'abord, live seulement apres validation statistique.
"""

__version__ = "0.1.0"

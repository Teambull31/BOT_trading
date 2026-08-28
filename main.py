"""Point d'entree de l'application hebergee (fonction sans serveur Vercel).

Ce fichier est a la racine, et non dans un dossier `api/`, parce que les deux
emplacements ne veulent pas dire la meme chose sur Vercel. Un fichier de `api/`
est une fonction adressee par son chemin : `api/index.py` ne repond qu'a
l'adresse `/api/index`. Une application ASGI declaree a la racine recoit au
contraire TOUT le trafic, chemin d'origine compris, et c'est FastAPI qui
choisit la route -- le seul montage ou `@app.get("/")` sert bien `/`.

Deux differences avec le lancement local, toutes deux imposees par
l'hebergement :

1. Aucun disque durable. Le repertoire temporaire n'est qu'un cache de travail,
   efface a tout moment ; la reference du compte est detenue par le navigateur
   de chaque visiteur (voir `trader.webapp.server`).
2. Aucune authentification, donc plusieurs visiteurs a la fois. Chacun a son
   propre fichier de travail, personne ne voit ni ne modifie le compte d'un
   autre.

Ce que l'hebergement ne change pas : l'argent reste fictif, aucun ordre n'est
transmis nulle part et aucun courtier n'est connecte.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from trader.webapp.server import create_app  # noqa: E402

app = create_app(accounts_dir=Path(tempfile.gettempdir()) / "coach-accounts")

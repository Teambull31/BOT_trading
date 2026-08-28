"""La configuration d'hebergement, verifiee comme du code.

Une erreur de routage ne casse aucun test metier : en local le serveur est
lance directement, et tout passe au vert pendant que l'application en ligne
repond a cote sur chacune de ses adresses. C'est arrive -- six projets Vercel
successifs n'ont jamais servi autre chose que
`{"detail":"identifiant de compte absent ou invalide"}`. Ce fichier fige donc
la forme du montage, pour que la faute redevienne visible ici.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

RACINE = Path(__file__).resolve().parents[2]
ENVOI_MANUEL = RACINE / "deploy" / "vercel"


def _config(dossier: Path) -> dict:
    return json.loads((dossier / "vercel.json").read_text())


def test_le_point_d_entree_est_a_la_racine_des_deux_envois():
    """`api/<nom>.py` n'est joignable qu'a l'adresse qui porte son nom.

    Une application ASGI declaree a la racine recoit au contraire tout le
    trafic avec son chemin d'origine : c'est le seul emplacement ou FastAPI
    peut router lui-meme, donc le seul ou `@app.get("/")` sert bien `/`.
    """
    for dossier in (RACINE, ENVOI_MANUEL):
        assert (dossier / "main.py").exists(), dossier
        assert not (dossier / "api").exists(), f"{dossier}: montage `api/` mixte"
        assert set(_config(dossier)["functions"]) == {"main.py"}, dossier


def test_aucune_reecriture_attrape_tout():
    """Une reecriture REMPLACE le chemin, elle ne le conserve pas.

    `"/(.*)" -> "/api/index"` livrait donc toujours `/api/index` a
    l'application, quelle que soit l'adresse demandee. C'est la faute exacte
    qui a mis six projets hors service ; qu'elle soit tentante -- elle a l'air
    d'un simple attrape-tout -- est la raison de ce test.
    """
    for dossier in (RACINE, ENVOI_MANUEL):
        assert "rewrites" not in _config(dossier), dossier
        assert "routes" not in _config(dossier), dossier


def test_la_reecriture_expliquait_bien_l_erreur_vue_en_ligne():
    """Reproduit la panne, pour que le diagnostic ne repose pas sur la memoire.

    Le serveur exige un en-tete de compte sur `/api/...` et sur rien d'autre.
    Un chemin ecrase en `/api/index` franchissait donc ce controle sans en-tete
    et repondait 400 -- y compris quand le visiteur demandait l'accueil.
    """
    import main  # le point d'entree hebergé lui-meme

    client = TestClient(main.app)

    ecrase = client.get("/api/index")
    assert ecrase.status_code == 400
    assert ecrase.json()["detail"] == "identifiant de compte absent ou invalide"

    # Chemin conserve : l'accueil est servi, sans en-tete ni compte.
    assert client.get("/").status_code == 200


def test_l_envoi_manuel_reste_autonome():
    """Les fichiers du dossier doivent suffire tels quels.

    Ce chemin sert quand le canal d'envoi ne transporte que quelques fichiers :
    ce qui manque ici ne sera pas la-bas, et la panne n'apparaitra qu'en ligne.
    """
    presents = {p.name for p in ENVOI_MANUEL.iterdir()}
    manquants = {"main.py", "vercel.json", "requirements.txt", ".python-version"} - presents
    assert not manquants

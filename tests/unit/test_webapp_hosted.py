"""Mode heberge : le compte appartient au navigateur, pas au serveur.

Ces tests couvrent ce que l'hebergement change et rien d'autre. Ils ne touchent
jamais le reseau : les cotations sont remplacees, car une suite qui depend de
l'ouverture des marches ne dit plus rien le week-end.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from trader.coach.account import PaperAccount
from trader.coach.quotes import Quote
from trader.webapp import server as webapp


@pytest.fixture(autouse=True)
def cotations_figees(monkeypatch):
    """Cours constants : ce qui est teste ici, c'est la tenue de l'etat."""

    def quote(symbol: str) -> Quote:
        return Quote(
            symbol=symbol.upper(),
            price=100.0,
            change=0.0,
            change_pct=0.0,
            previous_close=100.0,
            market_status="Market Open",
            is_real_time=True,
            timestamp="",
        )

    monkeypatch.setattr(webapp, "fetch_quote", quote)
    monkeypatch.setattr(webapp, "fetch_quotes", lambda symbols: {s: quote(s) for s in symbols})


@pytest.fixture
def heberge(tmp_path):
    """Application en mode heberge, un repertoire de travail par test."""
    return TestClient(webapp.create_app(accounts_dir=tmp_path / "comptes"))


def _tete(identifier: str, rev: int = 0) -> dict:
    return {"X-Coach-Account": identifier, "X-Coach-Rev": str(rev)}


# ------------------------------------------------------------------- local


def test_le_mode_local_ne_demande_aucun_en_tete(tmp_path):
    """La regression a eviter : casser le lancement local en ajoutant le mode
    heberge. Sans en-tete, le compte du fichier doit repondre comme avant."""
    client = TestClient(webapp.create_app(store=tmp_path / "compte.json"))
    assert client.post("/api/deposit", json={"amount": 1000}).status_code == 200
    body = client.get("/api/state").json()
    assert body["performance"]["cash"] == pytest.approx(1000.0)


def test_l_etat_renvoie_l_instantane_a_conserver(tmp_path):
    """Le navigateur ne peut detenir le compte que si le serveur le lui rend."""
    client = TestClient(webapp.create_app(store=tmp_path / "compte.json"))
    client.post("/api/deposit", json={"amount": 500})
    snapshot = client.get("/api/state").json()["snapshot"]
    assert snapshot["cash"] == pytest.approx(500.0)
    assert snapshot["rev"] > 0


# ----------------------------------------------------------------- heberge


def test_un_appel_sans_identifiant_est_refuse(heberge):
    """Sans identifiant, on ne saurait pas quel compte servir : servir « celui
    par defaut » ferait partager un compte unique a tous les visiteurs."""
    assert heberge.get("/api/state").status_code == 400


@pytest.mark.parametrize("mauvais", ["../evasion", "court", "a" * 65, "avec espace", ""])
def test_un_identifiant_hors_format_est_refuse(heberge, mauvais):
    """L'identifiant devient un nom de fichier : il est verifie, pas assaini."""
    reponse = heberge.get("/api/state", headers=_tete(mauvais))
    assert reponse.status_code == 400


def test_deux_navigateurs_ont_deux_comptes_distincts(heberge):
    """Sans authentification, l'isolation ne tient qu'a cela."""
    heberge.post("/api/deposit", json={"amount": 1000}, headers=_tete("navigateur-un-aaaa"))
    etat_du_second = heberge.get("/api/state", headers=_tete("navigateur-deux-bbb")).json()
    assert etat_du_second["has_capital"] is False
    etat_du_premier = heberge.get("/api/state", headers=_tete("navigateur-un-aaaa")).json()
    assert etat_du_premier["performance"]["cash"] == pytest.approx(1000.0)


def test_le_serveur_refuse_d_operer_sur_un_etat_perime(heberge):
    """Copie de travail perdue (instance neuve) alors que le navigateur a plus
    recent : operer quand meme afficherait des liquidites fausses."""
    reponse = heberge.post(
        "/api/deposit", json={"amount": 1000}, headers=_tete("navigateur-un-aaaa", rev=7)
    )
    assert reponse.status_code == 409
    assert reponse.json()["code"] == "stale_state"


def test_rendre_l_instantane_repare_l_ecart(heberge, tmp_path):
    """Le chemin de reparation complet, celui que le client rejoue tout seul :
    une instance neuve ne connait pas le compte, le navigateur le lui rend."""
    identifiant = "navigateur-un-aaaa"
    heberge.post("/api/deposit", json={"amount": 1000}, headers=_tete(identifiant))
    instantane = heberge.get("/api/state", headers=_tete(identifiant)).json()["snapshot"]

    # Instance neuve : repertoire de travail vide, elle n'a jamais vu ce compte.
    frais = TestClient(webapp.create_app(accounts_dir=tmp_path / "instance-neuve"))
    tetes = _tete(identifiant, rev=instantane["rev"])
    assert frais.get("/api/state", headers=tetes).status_code == 409

    rendu = frais.post("/api/restore", json={"snapshot": instantane}, headers=tetes)
    assert rendu.status_code == 200

    etat = frais.get("/api/state", headers=_tete(identifiant, rev=rendu.json()["rev"])).json()
    assert etat["performance"]["cash"] == pytest.approx(1000.0)


def test_la_restauration_echappe_au_controle_de_revision(heberge):
    """C'est elle qui repare l'ecart : elle ne peut pas exiger qu'il n'existe pas."""
    reponse = heberge.post(
        "/api/restore",
        json={"snapshot": {"cash": 42.0, "deposits": [], "positions": [], "history": [], "rev": 9}},
        headers=_tete("navigateur-un-aaaa", rev=9),
    )
    assert reponse.status_code == 200
    assert reponse.json()["rev"] == 10


def test_un_instantane_illisible_ne_fait_pas_tomber_le_serveur(heberge):
    reponse = heberge.post(
        "/api/restore",
        json={"snapshot": {"positions": [{"nimporte": "quoi"}]}},
        headers=_tete("navigateur-un-aaaa"),
    )
    assert reponse.status_code == 400


# ------------------------------------------------------------- revisions


def test_la_revision_avance_a_chaque_ecriture(tmp_path):
    compte = PaperAccount(tmp_path / "c.json")
    assert compte.state.rev == 0
    compte.deposit(100.0)
    compte.deposit(100.0)
    assert compte.state.rev == 2


def test_la_remise_a_zero_ne_fait_pas_reculer_la_revision(tmp_path):
    """Sinon le navigateur prendrait l'effacement pour un etat perime et
    reinjecterait le compte que l'utilisateur venait de supprimer."""
    compte = PaperAccount(tmp_path / "c.json")
    compte.deposit(1000.0)
    avant = compte.state.rev
    compte.reset()
    assert compte.state.rev > avant
    assert compte.state.cash == 0.0


def test_l_instantane_fait_l_aller_retour_sans_rien_perdre(tmp_path):
    """Y compris le stop suiveur : un champ oublie ici desarmerait
    silencieusement le suiveur d'un utilisateur a chaque rechargement."""
    compte = PaperAccount(tmp_path / "c.json")
    compte.deposit(10_000.0)
    compte.open_position("AAA", 10, 100.0, 90.0, target=120.0, trailing_pct=5.0)
    instantane = compte.snapshot()

    relu = PaperAccount(tmp_path / "autre.json")
    relu.restore(instantane)
    position = relu.state.positions[0]
    assert position.trailing_pct == pytest.approx(5.0)
    assert position.trail_high == pytest.approx(compte.state.positions[0].trail_high)
    assert relu.state.cash == pytest.approx(compte.state.cash)

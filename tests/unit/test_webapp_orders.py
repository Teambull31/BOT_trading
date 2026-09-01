"""Ordres conditionnels : « acheter pour X EUR si le cours atteint Y ».

Ce que ces tests verrouillent : un ordre reste en attente tant que le cours n'a
pas franchi le declencheur, il s'execute au cours OBSERVE (pas au declencheur),
le budget est reserve des la pose et l'annulation le rend. Comme le reste de la
suite webapp, aucune cotation reelle : une suite qui depend de l'ouverture des
marches ne dit plus rien le week-end.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from trader.coach.account import SLIPPAGE_PCT, PaperAccount, PendingOrder
from trader.coach.quotes import Quote
from trader.webapp import server as webapp

PRIX = {"valeur": 300.0}


@pytest.fixture(autouse=True)
def cotations_pilotees(monkeypatch):
    """Le cours est un dial que chaque test tourne via PRIX['valeur']."""

    def quote(symbol: str) -> Quote:
        return Quote(
            symbol=symbol.upper(),
            price=PRIX["valeur"],
            change=0.0,
            change_pct=0.0,
            previous_close=PRIX["valeur"],
            market_status="Market Open",
            is_real_time=True,
            timestamp="",
        )

    monkeypatch.setattr(webapp, "fetch_quote", quote)
    monkeypatch.setattr(webapp, "fetch_quotes", lambda symbols: {s: quote(s) for s in symbols})
    PRIX["valeur"] = 300.0


@pytest.fixture
def heberge(tmp_path):
    return TestClient(webapp.create_app(accounts_dir=tmp_path / "comptes"))


def _tete(identifier: str, rev: int = 0) -> dict:
    return {"X-Coach-Account": identifier, "X-Coach-Rev": str(rev)}


def _compte(heberge, identifier="compte-conditionnel-1", depot=10000):
    heberge.post("/api/deposit", json={"amount": depot}, headers=_tete(identifier))
    return identifier


# --------------------------------------------------------------- cycle nominal


def test_l_ordre_reste_en_attente_tant_que_le_declencheur_n_est_pas_atteint(heberge):
    identifier = _compte(heberge)
    reponse = heberge.post(
        "/api/order",
        json={"symbol": "MU", "trigger": 250, "stop": 240, "budget": 2000, "direction": "dip"},
        headers=_tete(identifier),
    )
    assert reponse.status_code == 200, reponse.text
    assert reponse.json()["direction"] == "dip"

    etat = heberge.get("/api/state", headers=_tete(identifier)).json()
    assert len(etat["pending"]) == 1
    assert not etat["positions"]
    assert etat["performance"]["reserved_cash"] == pytest.approx(2000.0)
    assert etat["performance"]["available_cash"] == pytest.approx(8000.0)


def test_l_ordre_s_execute_au_cours_observe_pas_au_declencheur(heberge):
    identifier = _compte(heberge)
    heberge.post(
        "/api/order",
        json={"symbol": "MU", "trigger": 250, "stop": 240, "budget": 2000, "direction": "dip"},
        headers=_tete(identifier),
    )
    PRIX["valeur"] = 249.0

    etat = heberge.get("/api/state", headers=_tete(identifier)).json()
    assert not etat["pending"]
    evenements = etat["order_events"]
    assert evenements and evenements[0]["status"] == "exécuté"
    assert evenements[0]["fill"] == pytest.approx(249.0 * (1 + SLIPPAGE_PCT / 100.0), rel=1e-4)

    positions = etat["positions"]
    assert len(positions) == 1 and positions[0]["symbol"] == "MU"
    # Le liquide a bien ete debite du montant reserve, pas plus.
    assert etat["performance"]["cash"] == pytest.approx(8000.0)
    assert etat["performance"]["reserved_cash"] == pytest.approx(0.0)


def test_direction_rise_se_declenche_a_la_hausse(heberge):
    identifier = _compte(heberge)
    heberge.post(
        "/api/order",
        json={"symbol": "NVDA", "trigger": 320, "stop": 300, "budget": 1000, "direction": "rise"},
        headers=_tete(identifier),
    )
    etat = heberge.get("/api/state", headers=_tete(identifier)).json()
    assert len(etat["pending"]) == 1  # a 300, pas encore

    PRIX["valeur"] = 321.0
    etat = heberge.get("/api/state", headers=_tete(identifier)).json()
    assert not etat["pending"]
    assert etat["order_events"][0]["status"] == "exécuté"
    assert len(etat["positions"]) == 1


# ------------------------------------------------------------------- refus


def test_stop_au_dessus_du_declencheur_refuse(heberge):
    identifier = _compte(heberge)
    reponse = heberge.post(
        "/api/order",
        json={"symbol": "MU", "trigger": 250, "stop": 260, "budget": 1000},
        headers=_tete(identifier),
    )
    assert reponse.status_code == 400
    assert "stop" in reponse.json()["detail"].lower()


def test_declencheur_deja_franchi_refuse_l_ordre(heberge):
    identifier = _compte(heberge)
    PRIX["valeur"] = 240.0  # deja sous le declencheur de 250
    reponse = heberge.post(
        "/api/order",
        json={"symbol": "MU", "trigger": 250, "stop": 235, "budget": 1000, "direction": "dip"},
        headers=_tete(identifier),
    )
    assert reponse.status_code == 400
    assert "au marché" in reponse.json()["detail"]


def test_budget_et_quantite_ensemble_refuses(heberge):
    identifier = _compte(heberge)
    reponse = heberge.post(
        "/api/order",
        json={"symbol": "MU", "trigger": 250, "stop": 240, "budget": 1000, "shares": 4},
        headers=_tete(identifier),
    )
    assert reponse.status_code == 400


def test_ni_budget_ni_quantite_refuses(heberge):
    identifier = _compte(heberge)
    reponse = heberge.post(
        "/api/order",
        json={"symbol": "MU", "trigger": 250, "stop": 240},
        headers=_tete(identifier),
    )
    assert reponse.status_code == 400


def test_la_reserve_bloque_un_second_ordre_hors_budget(heberge):
    identifier = _compte(heberge, depot=1000)
    premier = heberge.post(
        "/api/order",
        json={"symbol": "MU", "trigger": 250, "stop": 240, "budget": 800, "direction": "dip"},
        headers=_tete(identifier),
    )
    assert premier.status_code == 200

    second = heberge.post(
        "/api/order",
        json={"symbol": "AAPL", "trigger": 150, "stop": 140, "budget": 800, "direction": "dip"},
        headers=_tete(identifier),
    )
    assert second.status_code == 400
    etat = heberge.get("/api/state", headers=_tete(identifier)).json()
    assert etat["performance"]["available_cash"] == pytest.approx(200.0)
    assert etat["performance"]["reserved_cash"] == pytest.approx(800.0)


def test_annuler_un_ordre_libere_la_reserve(heberge):
    identifier = _compte(heberge)
    reponse = heberge.post(
        "/api/order",
        json={"symbol": "MU", "trigger": 250, "stop": 240, "budget": 2000, "direction": "dip"},
        headers=_tete(identifier),
    )
    order_id = reponse.json()["order_id"]

    suppression = heberge.delete(f"/api/order/{order_id}", headers=_tete(identifier))
    assert suppression.status_code == 200

    etat = heberge.get("/api/state", headers=_tete(identifier)).json()
    assert not etat["pending"]
    assert etat["performance"]["reserved_cash"] == pytest.approx(0.0)


def test_annuler_un_ordre_inconnu_renvoie_404(heberge):
    identifier = _compte(heberge)
    assert heberge.delete("/api/order/inexistant", headers=_tete(identifier)).status_code == 404


# ------------------------------------------------------ modele : instantane


def test_l_instantane_conserve_les_ordres_en_attente(tmp_path):
    compte = PaperAccount(tmp_path / "c.json")
    compte.deposit(10000)
    compte.place_order("MU", 250.0, 240.0, current_price=300.0, budget=2000.0)

    reconstruit = PaperAccount(tmp_path / "autre.json")
    reconstruit.restore(compte.snapshot())
    assert [(o.symbol, o.direction, o.trigger, o.budget) for o in reconstruit.state.pending] == [
        ("MU", "dip", 250.0, 2000.0)
    ]
    assert reconstruit.available_cash() == pytest.approx(compte.available_cash())


def test_ordre_expire_ne_s_execute_plus(tmp_path):
    compte = PaperAccount(tmp_path / "c.json")
    compte.deposit(10000)
    order = compte.place_order(
        "MU", 250.0, 240.0, current_price=300.0, budget=2000.0, expires_in_days=1
    )
    # On repousse l'echeance dans le passe pour eprouver la branche d'expiration.
    order.expires_at = "2000-01-01T00:00:00+00:00"

    evenements = compte.check_pending({"MU": 249.0})
    assert evenements and evenements[0]["status"] == "expiré"
    assert not compte.state.pending
    assert not compte.state.positions
    assert compte.available_cash() == pytest.approx(10000.0)


def test_pendingorder_decrit_l_ordre_en_clair():
    order = PendingOrder(
        id="x", symbol="MU", direction="dip", trigger=250.0, stop=240.0,
        created_at="", budget=200.0,
    )
    texte = order.describe()
    assert "MU" in texte and "descend à" in texte and "250" in texte

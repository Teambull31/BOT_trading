"""Contrat du serveur avec l'interface : les chiffres que l'ecran affiche.

L'interface affiche le stop REELLEMENT en vigueur et le rapport gain/perte au
moment ou l'utilisateur decide. Ces deux chiffres viennent du serveur ; s'ils
disparaissent du corps de la reponse, l'ecran de decision ment en silence,
sans qu'aucun test Python ne s'en apercoive. `/api/suggest-size` propose de
meme la quantite ET l'objectif, deduits du seul stop. D'ou ces verifications-la.

Aucun acces reseau : les cotations sont remplacees, une suite qui depend de
l'ouverture des marches ne dit plus rien le week-end.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from trader.coach.account import PaperAccount
from trader.coach.curriculum import MAX_OPEN_RISK_PCT, MIN_PLANNED_R
from trader.coach.quotes import Quote
from trader.webapp import server as webapp


def _fige_le_cours(monkeypatch, prix: float) -> None:
    """Remplace les cotations par un cours constant, sans acces reseau."""

    def quote(symbol: str) -> Quote:
        return Quote(
            symbol=symbol.upper(),
            price=prix,
            change=0.0,
            change_pct=0.0,
            previous_close=100.0,
            market_status="Market Open",
            is_real_time=True,
            timestamp="",
        )

    monkeypatch.setattr(webapp, "fetch_quote", quote)
    monkeypatch.setattr(webapp, "fetch_quotes", lambda symbols: {s: quote(s) for s in symbols})


@pytest.fixture(autouse=True)
def cotations_figees(monkeypatch):
    """Cours a 100.00 par defaut ; un test qui a besoin d'un autre le repose."""
    _fige_le_cours(monkeypatch, 100.0)


@pytest.fixture
def client(tmp_path) -> TestClient:
    store = tmp_path / "compte.json"
    PaperAccount(store).deposit(1000.0)
    return TestClient(webapp.create_app(store=store))


def _review(client: TestClient, **plan) -> dict:
    reponse = client.post("/api/review", json={"symbol": "AAA", "shares": 2.0, **plan})
    assert reponse.status_code == 200, reponse.text
    return reponse.json()


def test_review_publishes_the_stop_actually_in_force(client):
    """Un suiveur plus serre gouverne des l'entree : la vignette doit le montrer."""
    review = _review(client, stop=80.0, trailing_pct=5.0)
    assert review["effective_stop"] == pytest.approx(95.0)
    assert review["trailing_overrides_stop"] is True


def test_review_publishes_the_typed_stop_when_the_trail_changes_nothing(client):
    review = _review(client, stop=98.0, trailing_pct=20.0)
    assert review["effective_stop"] == pytest.approx(98.0)
    assert review["trailing_overrides_stop"] is False


def test_review_distinguishes_a_null_ratio_from_an_absent_objective(client):
    """Objectif pose au prix d'entree : le rapport vaut 0, il n'est pas absent."""
    assert _review(client, stop=95.0, target=100.0)["reward_risk"] == 0.0
    assert _review(client, stop=95.0)["reward_risk"] is None


def test_suggest_size_also_proposes_the_objective(client):
    """La quantite et l'objectif se deduisent du meme stop : ils voyagent ensemble."""
    reponse = client.post(
        "/api/suggest-size", json={"symbol": "AAA", "stop": 95.0, "risk_pct": 1.0}
    )
    assert reponse.status_code == 200, reponse.text
    corps = reponse.json()
    # Cotation figee a 100.0, stop a 95.0 : 5.0 de perte acceptee par titre.
    assert corps["suggested_target"] == pytest.approx(100.0 + corps["suggested_target_ratio"] * 5.0)
    assert corps["suggested_target_ratio"] == pytest.approx(MIN_PLANNED_R)


def test_suggest_size_omits_the_objective_when_the_stop_is_above_the_price(client):
    reponse = client.post(
        "/api/suggest-size", json={"symbol": "AAA", "stop": 105.0, "risk_pct": 1.0}
    )
    assert reponse.status_code == 200, reponse.text
    assert reponse.json()["suggested_target"] is None


def _ouvre(client: TestClient, **plan) -> dict:
    reponse = client.post("/api/open", json={"symbol": "AAA", "shares": 2.0, "stop": 95.0, **plan})
    assert reponse.status_code == 200, reponse.text
    return client.get("/api/state").json()["positions"][0]


def test_state_publishes_the_euros_still_at_risk(client):
    """Le pourcentage de marge ne dit pas combien d'euros sont en jeu.

    C'est pourtant ce chiffre que l'utilisateur decide, et l'interface ne peut
    pas le recalculer : les frais des deux ordres n'y sont pas.
    """
    position = _ouvre(client)
    assert position["risk_at_stop"] > 0
    # Perte brute : 2 titres entres a 100.05 (slippage) sortis vers 95.
    # Les frais des deux ordres la creusent, ils ne l'allegent jamais.
    assert position["risk_at_stop"] > 2.0 * (100.0 - 95.0)


def test_state_reports_a_gain_locked_by_the_stop_as_a_negative_risk(client, monkeypatch):
    """Stop remonte au-dessus du prix de revient : le trade ne peut plus couter.

    Le cours doit monter d'abord : un stop pose au-dessus du cours est un stop
    TOUCHE, et le serveur solde la position au rafraichissement suivant.
    """
    position = _ouvre(client)
    _fige_le_cours(monkeypatch, 130.0)

    reponse = client.post("/api/stop", json={"position_id": position["id"], "stop": 120.0})
    assert reponse.status_code == 200, reponse.text

    etat = client.get("/api/state").json()
    assert etat["positions"], "la position ne devait pas etre soldee"
    assert etat["positions"][0]["risk_at_stop"] < 0


def test_moving_a_stop_above_the_price_announces_the_sale_it_really_is(client):
    """Un stop pose au-dessus du cours est deja touche : c'est une vente.

    Le serveur doit le dire pendant que la position existe encore. Le test
    verifie les deux moities de la promesse : l'annonce, puis la sortie qu'elle
    annonce — un message qui ne correspondrait pas au comportement serait pire
    que pas de message du tout.
    """
    position = _ouvre(client)
    reponse = client.post("/api/stop", json={"position_id": position["id"], "stop": 120.0})
    assert reponse.status_code == 200, reponse.text
    corps = reponse.json()
    assert corps["triggers_now"] is True
    assert corps["price"] == pytest.approx(100.0)

    assert client.get("/api/state").json()["positions"] == []


def test_moving_a_stop_below_the_price_is_not_announced_as_a_sale(client):
    """Resserrer un stop sous le cours protege : rien a signaler."""
    position = _ouvre(client)
    reponse = client.post("/api/stop", json={"position_id": position["id"], "stop": 98.0})
    assert reponse.status_code == 200, reponse.text
    assert reponse.json()["triggers_now"] is False
    assert client.get("/api/state").json()["positions"], "la position devait rester ouverte"


def test_a_widened_stop_is_still_flagged_as_such(client):
    """Le signalement d'elargissement ne doit pas avoir ete perdu en chemin."""
    position = _ouvre(client)
    reponse = client.post("/api/stop", json={"position_id": position["id"], "stop": 90.0})
    assert reponse.json() == {
        "ok": True,
        "stop": pytest.approx(90.0),
        "widened": True,
        "price": pytest.approx(100.0),
        "triggers_now": False,
    }


def test_state_publishes_what_the_whole_account_risks(client):
    """Le total engage, pas seulement le risque de chaque ligne.

    L'exposition dit ce qui est investi ; elle ne dit pas ce qui serait perdu.
    Sans ce chiffre, trois positions saines cachaient un compte qui joue plus
    que ce que le parcours autorise a perdre.
    """
    _ouvre(client, symbol="AAA")
    _ouvre(client, symbol="BBB")
    perf = client.get("/api/state").json()["performance"]

    assert perf["open_positions"] == 2
    assert perf["open_risk"] > 0
    assert perf["open_risk_pct"] == pytest.approx(
        perf["open_risk"] / perf["equity"] * 100.0, abs=0.05
    )
    # La limite voyage avec la mesure : l'interface n'a pas a la redefinir.
    assert perf["open_risk_limit_pct"] == MAX_OPEN_RISK_PCT


def test_state_reports_a_portfolio_that_can_no_longer_cost_anything(client, monkeypatch):
    """Tous les stops au-dessus du prix de revient : le total passe en negatif.

    Le cours monte d'abord : un stop pose au-dessus du cours serait un stop
    touche, et la position serait soldee au lieu de verrouiller quoi que ce soit.
    """
    position = _ouvre(client, symbol="AAA")
    _fige_le_cours(monkeypatch, 130.0)
    client.post("/api/stop", json={"position_id": position["id"], "stop": 120.0})

    perf = client.get("/api/state").json()["performance"]
    assert perf["open_positions"] == 1
    assert perf["open_risk"] < 0

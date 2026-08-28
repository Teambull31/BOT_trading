"""Contrat de `/api/review` et `/api/suggest-size` avec l'interface.

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
from trader.coach.curriculum import MIN_PLANNED_R
from trader.coach.quotes import Quote
from trader.webapp import server as webapp


@pytest.fixture(autouse=True)
def cotations_figees(monkeypatch):
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

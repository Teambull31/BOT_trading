"""L'historique doit dire, trade par trade, ce qui etait prevu en face du stop.

Le palier « couper court, laisser courir » compte les trades sans plan de
sortie. Un palier qui compte des fautes sans jamais designer lesquelles
n'apprend rien : ces tests verifient que `/api/history` porte l'information.

Aucun acces reseau : les cotations sont remplacees, sinon la suite dependrait
de l'ouverture des marches.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from trader.coach.account import PaperAccount
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
def compte(tmp_path) -> PaperAccount:
    account = PaperAccount(tmp_path / "compte.json")
    account.deposit(1000.0)
    return account


def _cloture(compte: PaperAccount, symbol: str, **kwargs) -> None:
    position = compte.open_position(symbol, 1.0, 100.0, stop=98.0, **kwargs)
    compte.close_position(position.id, 101.0)


def _reponse(tmp_path) -> str:
    """Corps brut de /api/history. L'application relit le compte sur disque,
    elle est donc construite APRES que les trades y ont ete ecrits."""
    client = TestClient(webapp.create_app(store=tmp_path / "compte.json"))
    reponse = client.get("/api/history")
    assert reponse.status_code == 200
    return reponse.text


def _trades(tmp_path) -> dict[str, dict]:
    return {trade["symbol"]: trade for trade in json.loads(_reponse(tmp_path))["trades"]}


def test_un_objectif_est_rendu_en_multiples_du_risque(compte, tmp_path):
    _cloture(compte, "AAA", target=105.0)
    trade = _trades(tmp_path)["AAA"]
    assert trade["plan"] == "objectif"
    assert trade["planned_ok"] is True
    # Mesure sur le prix REELLEMENT execute, slippage compris : c'est celui-la
    # qui fixe le risque au stop, pas la cotation affichee avant l'ordre.
    cloture = compte.state.history[0]
    attendu = (105.0 - cloture.entry_price) * cloture.shares / cloture.planned_risk
    assert trade["planned_ratio"] == pytest.approx(round(attendu, 2))


def test_un_stop_suiveur_compte_comme_un_plan(compte, tmp_path):
    _cloture(compte, "BBB", trailing_pct=8.0)
    trade = _trades(tmp_path)["BBB"]
    assert trade["plan"] == "suiveur"
    assert trade["trailing_pct"] == 8.0
    assert trade["planned_ok"] is True


def test_un_trade_sans_rien_en_face_du_stop_est_signale(compte, tmp_path):
    _cloture(compte, "CCC")
    trade = _trades(tmp_path)["CCC"]
    assert trade["plan"] == "aucun"
    assert trade["planned_ratio"] is None
    assert trade["planned_ok"] is False


def test_un_objectif_plus_petit_que_le_risque_est_signale(compte, tmp_path):
    """Viser moins que ce qu'on risque est le contraire de « couper court »."""
    _cloture(compte, "DDD", target=101.0)
    trade = _trades(tmp_path)["DDD"]
    assert trade["planned_ratio"] < 1.0
    assert trade["planned_ok"] is False


def test_un_gain_sans_plafond_ne_renvoie_jamais_l_infini(compte, tmp_path):
    """`json.dumps` ecrirait `Infinity`, que `JSON.parse` refuse : page blanche.

    Le cas se produit des qu'un stop suiveur est arme, donc en usage normal.
    """
    _cloture(compte, "EEE", trailing_pct=8.0)
    assert "Infinity" not in _reponse(tmp_path)
    assert _trades(tmp_path)["EEE"]["planned_ratio"] is None

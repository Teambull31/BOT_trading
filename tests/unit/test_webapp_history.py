"""Courbe de cours d'un titre : « montre-moi NVDA sur trois mois ».

Deux niveaux verrouilles ici : le parsing de la reponse Nasdaq (clotures
quotidiennes et cours intra-seance) dans `fetch_history`, et la route
`/api/history/{symbol}` qui l'expose. Aucune requete reelle : `httpx.get` est
remplace par un double qui rejoue une charge utile figee.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from trader.coach import quotes
from trader.coach.quotes import PriceHistory, QuoteError, fetch_history
from trader.webapp import server as webapp


class _FausseReponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:  # noqa: D401 - double de test
        return None

    def json(self) -> dict:
        return self._payload


_LIGNES_QUOTIDIENNES = {
    "data": {
        "tradesTable": {
            "rows": [
                {"date": "09/02/2026", "close": "$252.10"},
                {"date": "08/29/2026", "close": "$248.00"},
                {"date": "08/28/2026", "close": "$255.40"},
                {"date": "08/27/2026", "close": "N/A"},  # ligne trouee : ignoree
            ]
        }
    }
}

_POINTS_INTRASEANCE = {
    "data": {
        "chart": [
            {"z": {"dateTime": "2026-09-02 09:30", "value": "250.00"}},
            {"z": {"dateTime": "2026-09-02 10:00", "value": "251.20"}},
            {"z": {"dateTime": "2026-09-02 10:30", "value": "249.80"}},
        ]
    }
}


@pytest.fixture(autouse=True)
def _vide_cache():
    quotes.clear_cache()
    yield
    quotes.clear_cache()


# ------------------------------------------------------- parsing fetch_history


def test_periode_mensuelle_rend_les_clotures_triees(monkeypatch):
    monkeypatch.setattr(quotes.httpx, "get", lambda *a, **k: _FausseReponse(_LIGNES_QUOTIDIENNES))
    histo = fetch_history("NVDA", "1M")

    assert histo.symbol == "NVDA"
    assert histo.period == "1M"
    # trie du plus ancien au plus recent, la ligne "N/A" ecartee
    assert [stamp for stamp, _ in histo.points] == [
        "2026-08-28",
        "2026-08-29",
        "2026-09-02",
    ]
    assert [prix for _, prix in histo.points] == [255.4, 248.0, 252.1]


def test_to_dict_calcule_les_bornes_et_la_variation(monkeypatch):
    monkeypatch.setattr(quotes.httpx, "get", lambda *a, **k: _FausseReponse(_LIGNES_QUOTIDIENNES))
    charge = fetch_history("NVDA", "3M").to_dict()

    assert charge["period"] == "3M"
    assert charge["first"] == 255.4
    assert charge["last"] == 252.1
    assert charge["low"] == 248.0
    assert charge["high"] == 255.4
    assert charge["change_pct"] == pytest.approx(round((252.1 / 255.4 - 1) * 100, 2))
    assert charge["points"][0] == ["2026-08-28", 255.4]


def test_periode_1d_passe_par_l_endpoint_intraseance(monkeypatch):
    vus: dict[str, str] = {}

    def faux_get(url, **kwargs):
        vus["url"] = url
        return _FausseReponse(_POINTS_INTRASEANCE)

    monkeypatch.setattr(quotes.httpx, "get", faux_get)
    histo = fetch_history("NVDA", "1D")

    assert "chart" in vus["url"]
    assert len(histo.points) == 3
    assert histo.points[0] == ("2026-09-02 09:30", 250.0)


def test_periode_inconnue_retombe_sur_1m(monkeypatch):
    monkeypatch.setattr(quotes.httpx, "get", lambda *a, **k: _FausseReponse(_LIGNES_QUOTIDIENNES))
    assert fetch_history("NVDA", "10Y").period == "1M"


def test_historique_trop_court_leve_quote_error(monkeypatch):
    monkeypatch.setattr(
        quotes.httpx,
        "get",
        lambda *a, **k: _FausseReponse({"data": {"tradesTable": {"rows": []}}}),
    )
    with pytest.raises(QuoteError):
        fetch_history("ZZZZ", "1M")


def test_erreur_reseau_devient_quote_error(monkeypatch):
    def boum(*a, **k):
        raise quotes.httpx.HTTPError("connexion coupee")

    monkeypatch.setattr(quotes.httpx, "get", boum)
    with pytest.raises(QuoteError):
        fetch_history("NVDA", "1M")


def test_le_cache_evite_un_second_appel(monkeypatch):
    appels = {"n": 0}

    def compte(*a, **k):
        appels["n"] += 1
        return _FausseReponse(_LIGNES_QUOTIDIENNES)

    monkeypatch.setattr(quotes.httpx, "get", compte)
    fetch_history("NVDA", "1M")
    fetch_history("NVDA", "1M")
    assert appels["n"] == 1


# ------------------------------------------------------------------ la route


@pytest.fixture
def client(tmp_path):
    # Mode solo (store=...) : la courbe de cours ne touche pas au compte, pas
    # besoin de l'en-tete X-Coach-Account qu'impose le mode heberge.
    return TestClient(webapp.create_app(store=tmp_path / "compte.json"))


def test_la_route_sert_la_courbe(client, monkeypatch):
    points = [("2026-08-01", 100.0), ("2026-08-15", 110.0), ("2026-09-01", 121.0)]
    monkeypatch.setattr(
        webapp, "fetch_history", lambda symbol, period="1M": PriceHistory(symbol.upper(), period, points)
    )
    reponse = client.get("/api/history/nvda", params={"period": "3M"})

    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["symbol"] == "NVDA"
    assert corps["period"] == "3M"
    assert corps["points"] == [["2026-08-01", 100.0], ["2026-08-15", 110.0], ["2026-09-01", 121.0]]
    assert corps["low"] == 100.0 and corps["high"] == 121.0
    assert corps["change_pct"] == pytest.approx(21.0)


def test_la_route_renvoie_404_si_indisponible(client, monkeypatch):
    def indispo(symbol, period="1M"):
        raise QuoteError("historique indisponible")

    monkeypatch.setattr(webapp, "fetch_history", indispo)
    assert client.get("/api/history/zzzz").status_code == 404


def test_periode_par_defaut_est_1m(client, monkeypatch):
    recu: dict[str, str] = {}

    def capte(symbol, period="1M"):
        recu["period"] = period
        return PriceHistory(symbol.upper(), period, [("2026-08-01", 1.0), ("2026-08-02", 2.0)])

    monkeypatch.setattr(webapp, "fetch_history", capte)
    client.get("/api/history/nvda")
    assert recu["period"] == "1M"

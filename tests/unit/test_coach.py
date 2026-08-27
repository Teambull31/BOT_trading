"""Tests du mode d'entrainement : compte fictif, progression, conseils, debrief.

Ce qui est verifie : que le compte ne peut pas mentir (frais toujours preleves,
liquidites respectees, stop obligatoire), que la progression mesure le PROCESS
et non le resultat, et que le debrief identifie les fautes independamment du
fait que le trade ait gagne ou perdu.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from trader.coach.account import (
    ClosedTrade,
    InsufficientFunds,
    PaperAccount,
)
from trader.coach.advisor import Severity, TradePlan, review_plan, suggest_size
from trader.coach.curriculum import LEVELS, evaluate_progress
from trader.coach.debrief import debrief_trade, recurring_patterns
from trader.coach.quotes import Quote, _parse_price


@pytest.fixture
def account(tmp_path) -> PaperAccount:
    acc = PaperAccount(tmp_path / "account.json")
    acc.deposit(1000.0)
    return acc


def _trade(
    symbol: str = "AAA",
    pnl: float = 10.0,
    entry: float = 100.0,
    stop: float = 95.0,
    shares: float = 1.0,
    days_ago: int = 5,
    duration: int = 3,
    widened: bool = False,
    costs: float = 2.0,
) -> ClosedTrade:
    opened = datetime.now(UTC) - timedelta(days=days_ago)
    return ClosedTrade(
        id=f"t{days_ago}{symbol}",
        symbol=symbol,
        shares=shares,
        entry_price=entry,
        exit_price=entry + pnl / shares,
        stop=stop,
        opened_at=opened.isoformat(timespec="seconds"),
        closed_at=(opened + timedelta(days=duration)).isoformat(timespec="seconds"),
        pnl=pnl,
        costs=costs,
        exit_reason="manuel",
        highest_price=entry * 1.05,
        stop_moved_against=widened,
    )


# --------------------------------------------------------------------- cours


def test_price_parsing_handles_nasdaq_formats():
    assert _parse_price("$977.5738") == pytest.approx(977.5738)
    assert _parse_price("+4.17%") == pytest.approx(4.17)
    assert _parse_price("1,255.00") == pytest.approx(1255.0)
    assert _parse_price("N/A") != _parse_price("N/A")  # NaN
    assert _parse_price(None) != _parse_price(None)


def test_quote_flags_out_of_session():
    """Hors seance, l'interface doit pouvoir prevenir que le prix est indicatif."""
    quote = Quote(
        symbol="MU",
        price=100.0,
        change=1.0,
        change_pct=1.0,
        previous_close=99.0,
        market_status="Pre-Market",
        is_real_time=True,
        timestamp="",
        bid=99.5,
        ask=100.5,
    )
    assert not quote.is_tradable_session
    assert quote.spread_pct == pytest.approx(1.0)
    assert replace(quote, market_status="Market Open").is_tradable_session


# -------------------------------------------------------------------- compte


def test_deposit_is_recorded_and_manual(account):
    assert account.state.total_deposited == 1000.0
    account.deposit(500.0, note="recharge")
    assert account.state.total_deposited == 1500.0
    assert account.state.cash == 1500.0
    assert account.state.deposits[-1].note == "recharge"


def test_deposit_rejects_non_positive(account):
    with pytest.raises(ValueError):
        account.deposit(0.0)


def test_position_requires_a_stop_below_entry(account):
    with pytest.raises(ValueError, match="stop"):
        account.open_position("AAA", 1.0, 100.0, stop=100.0)
    with pytest.raises(ValueError, match="stop"):
        account.open_position("AAA", 1.0, 100.0, stop=105.0)


def test_cannot_spend_more_than_cash(account):
    with pytest.raises(InsufficientFunds):
        account.open_position("AAA", 100.0, 100.0, stop=95.0)


def test_costs_are_always_charged(account):
    position = account.open_position("AAA", 1.0, 100.0, stop=95.0)
    assert position.entry_costs > 0
    # Slippage : on paie au-dessus du cours affiche.
    assert position.entry_price > 100.0
    trade = account.close_position(position.id, 100.0)
    assert trade.costs > position.entry_costs
    # Aller-retour a cours constant : le resultat doit etre negatif.
    assert trade.pnl < 0


def test_no_duplicate_position_on_same_symbol(account):
    account.open_position("AAA", 1.0, 100.0, stop=95.0)
    with pytest.raises(ValueError, match="déjà ouverte"):
        account.open_position("AAA", 1.0, 100.0, stop=95.0)


def test_widening_a_stop_is_traced(account):
    position = account.open_position("AAA", 1.0, 100.0, stop=95.0)
    account.update_stop(position.id, 90.0)
    trade = account.close_position(position.id, 92.0)
    assert trade.stop_moved_against


def test_tightening_a_stop_is_not_flagged(account):
    position = account.open_position("AAA", 1.0, 100.0, stop=95.0)
    account.update_stop(position.id, 98.0)
    trade = account.close_position(position.id, 99.0)
    assert not trade.stop_moved_against


def test_mark_tracks_the_peak_for_the_debrief(account):
    position = account.open_position("AAA", 1.0, 100.0, stop=95.0)
    account.mark({"AAA": 130.0})
    account.mark({"AAA": 110.0})
    assert account.find_position(position.id).highest_price == 130.0


def test_state_survives_a_reload(account, tmp_path):
    account.open_position("AAA", 1.0, 100.0, stop=95.0)
    reloaded = PaperAccount(tmp_path / "account.json")
    assert len(reloaded.state.positions) == 1
    assert reloaded.state.total_deposited == 1000.0


def test_corrupted_store_does_not_crash(tmp_path):
    store = tmp_path / "broken.json"
    store.write_text("{ pas du json", encoding="utf-8")
    assert PaperAccount(store).state.total_deposited == 0.0


def test_performance_uses_deposited_capital_as_denominator(account):
    """Le resultat se juge sur l'argent injecte, pas sur le solde courant."""
    account.deposit(1000.0)
    metrics = account.performance({})
    assert metrics["deposited"] == 2000.0
    assert metrics["pnl_pct"] == pytest.approx(0.0)


def test_saved_file_is_valid_json(account, tmp_path):
    account.open_position("AAA", 1.0, 100.0, stop=95.0)
    payload = json.loads((tmp_path / "account.json").read_text(encoding="utf-8"))
    assert payload["deposits"][0]["amount"] == 1000.0


# ---------------------------------------------------------------- progression


def test_progression_starts_at_zero(account):
    progress = evaluate_progress(account.state)
    assert progress.completed == 0
    assert progress.rank == "Zero"
    assert progress.current.level.number == 1


def test_first_trade_unlocks_first_level(account):
    account.open_position("AAA", 1.0, 100.0, stop=95.0)
    assert evaluate_progress(account.state).levels[0].achieved


def test_progression_rewards_process_not_profit(account):
    """Cinq trades PERDANTS mais bien menes doivent valider la discipline."""
    account.state.history = [
        _trade(symbol=f"S{i}", pnl=-4.0, days_ago=30 - i * 5) for i in range(5)
    ]
    levels = {level.level.key: level for level in evaluate_progress(account.state).levels}
    assert levels["discipline_stop"].achieved, levels["discipline_stop"].detail


def test_widened_stop_blocks_discipline_level(account):
    account.state.history = [
        _trade(symbol=f"S{i}", pnl=5.0, days_ago=30 - i * 5, widened=(i == 2)) for i in range(5)
    ]
    levels = {level.level.key: level for level in evaluate_progress(account.state).levels}
    assert not levels["discipline_stop"].achieved
    assert "élargi" in levels["discipline_stop"].detail


def test_oversized_positions_block_sizing_level(account):
    # Risque planifie de 50 EUR par trade sur 1000 EUR de capital = 5 %.
    account.state.history = [
        _trade(symbol=f"S{i}", pnl=3.0, entry=100.0, stop=90.0, shares=5.0, days_ago=40 - i * 4)
        for i in range(8)
    ]
    levels = {level.level.key: level for level in evaluate_progress(account.state).levels}
    assert not levels["dimensionnement"].achieved


def test_all_levels_have_distinct_keys():
    keys = [level.key for level in LEVELS]
    assert len(keys) == len(set(keys))
    assert [level.number for level in LEVELS] == list(range(1, len(LEVELS) + 1))


# ------------------------------------------------------------------- conseils


def test_suggest_size_matches_the_requested_risk():
    shares = suggest_size(equity=1000.0, price=100.0, stop=95.0, risk_pct=1.0)
    assert shares * (100.0 - 95.0) == pytest.approx(10.0)


def test_suggest_size_is_zero_when_stop_is_above_price():
    assert suggest_size(1000.0, 100.0, 105.0) == 0.0


def test_review_blocks_an_oversized_position(account):
    plan = TradePlan(symbol="AAA", shares=9.0, price=100.0, stop=90.0)
    review = review_plan(plan, account, prices={})
    assert not review.can_proceed
    assert any("%" in advice.title for advice in review.blockers)


def test_review_accepts_a_well_sized_trade(account):
    plan = TradePlan(symbol="AAA", shares=2.0, price=100.0, stop=95.0, target=115.0)
    review = review_plan(plan, account, prices={})
    assert review.can_proceed
    assert review.risk_pct == pytest.approx(1.0)
    assert any(advice.severity is Severity.GOOD for advice in review.advices)


def test_review_warns_on_a_poor_reward_ratio(account):
    plan = TradePlan(symbol="AAA", shares=2.0, price=100.0, stop=95.0, target=102.0)
    review = review_plan(plan, account, prices={})
    assert any("gain" in advice.title.lower() for advice in review.advices)


def test_review_always_states_it_does_not_predict(account):
    """Garde-fou : le conseil ne doit jamais laisser croire a une prevision."""
    plan = TradePlan(symbol="AAA", shares=1.0, price=100.0, stop=95.0)
    review = review_plan(plan, account, prices={})
    assert any("ne dit pas" in advice.title for advice in review.advices)


def test_review_warns_on_a_stop_inside_the_noise(account):
    plan = TradePlan(symbol="AAA", shares=1.0, price=100.0, stop=99.5)
    review = review_plan(plan, account, prices={})
    assert any("serré" in advice.title for advice in review.advices)


# -------------------------------------------------------------------- debrief


def test_losing_but_well_played_trade_is_praised(account):
    """Le cas pedagogique central : perdre proprement n'est pas une faute."""
    debrief = debrief_trade(_trade(pnl=-9.0), account.state)
    assert debrief.well_played
    assert "bien mené" in debrief.verdict


def test_winning_but_badly_played_trade_is_flagged(account):
    debrief = debrief_trade(_trade(pnl=40.0, widened=True), account.state)
    assert not debrief.well_played
    assert "MAL mené" in debrief.verdict
    assert any("reculé" in lesson.title for lesson in debrief.lessons)


def test_debrief_quantifies_the_missed_move(account):
    trade = _trade(pnl=2.0, entry=100.0, shares=1.0)
    trade.highest_price = 130.0
    trade.exit_price = 102.0
    lessons = debrief_trade(trade, account.state).lessons
    assert any("130" in lesson.message or "valu" in lesson.title for lesson in lessons)


def test_debrief_flags_same_day_losing_trades(account):
    trade = _trade(pnl=-5.0, duration=0)
    lessons = debrief_trade(trade, account.state).lessons
    assert any("même jour" in lesson.title for lesson in lessons)


def test_recurring_patterns_need_repetition(account):
    account.state.history = [_trade(symbol="A", pnl=5.0, days_ago=10)]
    assert recurring_patterns(account.state) == []


def test_recurring_patterns_detect_a_habit(account):
    account.state.history = [
        _trade(symbol=f"S{i}", pnl=-3.0, days_ago=40 - i, widened=True) for i in range(6)
    ]
    patterns = recurring_patterns(account.state)
    assert any("reculé" in lesson.title for lesson in patterns)


def test_recurring_patterns_detect_bad_asymmetry(account):
    history = [_trade(symbol=f"W{i}", pnl=5.0, days_ago=40 - i) for i in range(3)]
    history += [_trade(symbol=f"L{i}", pnl=-40.0, days_ago=30 - i) for i in range(3)]
    account.state.history = history
    patterns = recurring_patterns(account.state)
    assert any("moyenne" in lesson.title for lesson in patterns)


def test_review_blocks_concentration_even_when_stop_risk_looks_small(account):
    """Le piege : un stop serre sur une position enorme parait sur, et ne l'est pas.

    Le risque au stop suppose que le stop tienne. Un trou de cotation le saute
    et frappe la position entiere : c'est la TAILLE qui borne ce risque-la.
    """
    # 0.9 titre a 1000 = 900 EUR sur un compte de 1000, stop a 1.9 % seulement.
    plan = TradePlan(symbol="AAA", shares=0.9, price=1000.0, stop=981.0)
    review = review_plan(plan, account, prices={})
    assert review.risk_pct < 2.0, "le risque au stop parait faible"
    assert review.position_pct > 60.0
    assert not review.can_proceed, "une position de 88 % du compte doit etre bloquee"
    assert any("Concentration" in advice.title for advice in review.blockers)


def test_review_allows_a_concentrated_but_reasonable_position(account):
    plan = TradePlan(symbol="AAA", shares=0.5, price=1000.0, stop=970.0, target=1100.0)
    review = review_plan(plan, account, prices={})
    assert review.position_pct == pytest.approx(50.0)
    assert review.can_proceed


def test_quote_survives_null_fields_from_the_api(monkeypatch):
    """L'API renvoie parfois `null` la ou on attend un objet : ne pas planter.

    Regression : `data.get("keyStats", {})` ne protege pas d'une valeur
    explicitement nulle, seulement d'une cle absente. Le 500 constate sur
    /api/quotes venait de la.
    """
    import httpx

    from trader.coach import quotes as module

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {
                    "symbol": "ZZZ",
                    "companyName": None,
                    "marketStatus": None,
                    "keyStats": None,
                    "secondaryData": None,
                    "primaryData": {
                        "lastSalePrice": "$10.00",
                        "netChange": "0.10",
                        "percentageChange": "1.00%",
                        "deltaIndicator": "up",
                        "lastTradeTimestamp": None,
                        "isRealTime": True,
                        "bidPrice": "",
                        "askPrice": "",
                    },
                }
            }

    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse())
    module.clear_cache()
    quote = module.fetch_quote("ZZZ", use_cache=False)
    assert quote.price == pytest.approx(10.0)
    assert quote.week52_range == ""
    assert quote.market_status == "inconnu"
    assert quote.bid is None


# ------------------------------------------------------- declenchement du stop


def test_stop_triggers_when_price_falls_below(account):
    """Sans exécution automatique, le stop ne serait qu'une intention."""
    position = account.open_position("AAA", 1.0, 100.0, stop=95.0)
    assert account.check_stops({"AAA": 96.0}) == []
    triggered = account.check_stops({"AAA": 94.0})
    assert len(triggered) == 1
    assert triggered[0].exit_reason == "stop_touche"
    assert not account.state.positions
    with pytest.raises(KeyError):
        account.find_position(position.id)


def test_stop_does_not_trigger_without_a_price(account):
    """Un cours indisponible ne doit jamais provoquer de sortie fantome."""
    account.open_position("AAA", 1.0, 100.0, stop=95.0)
    assert account.check_stops({}) == []
    assert account.check_stops({"BBB": 1.0}) == []
    assert len(account.state.positions) == 1


def test_stop_fills_at_the_observed_price_not_the_stop_level(account):
    """Sur un écart brutal, la sortie se fait bien plus bas que le stop.

    Remplir au niveau théorique du stop rendrait la simulation flatteuse et
    masquerait le risque que justement aucun stop n'élimine.
    """
    account.open_position("AAA", 1.0, 100.0, stop=95.0)
    trade = account.check_stops({"AAA": 80.0})[0]
    assert trade.exit_price < 81.0, "la sortie doit refléter le cours réel"
    assert trade.pnl < -(100.0 - 95.0), "la perte dépasse l'enveloppe prévue"
    assert not trade.respected_stop


def test_several_stops_can_trigger_at_once(account):
    account.deposit(2000.0)
    account.open_position("AAA", 1.0, 100.0, stop=95.0)
    account.open_position("BBB", 1.0, 200.0, stop=190.0)
    triggered = account.check_stops({"AAA": 90.0, "BBB": 185.0})
    assert {t.symbol for t in triggered} == {"AAA", "BBB"}
    assert not account.state.positions


def test_debrief_explains_a_clean_stop(account):
    account.open_position("AAA", 1.0, 100.0, stop=95.0)
    trade = account.check_stops({"AAA": 94.9})[0]
    lessons = debrief_trade(trade, account.state).lessons
    assert any("exécuté comme prévu" in lesson.title for lesson in lessons)


def test_debrief_quantifies_a_gap_below_the_stop(account):
    account.open_position("AAA", 1.0, 100.0, stop=95.0)
    trade = account.check_stops({"AAA": 80.0})[0]
    lessons = debrief_trade(trade, account.state).lessons
    assert any("sous le niveau du stop" in lesson.title for lesson in lessons)


def test_targets_are_signalled_but_never_closed(account):
    """L'objectif atteint est la décision à travailler : l'app ne tranche pas."""
    account.open_position("AAA", 1.0, 100.0, stop=95.0, target=120.0)
    assert account.targets_reached({"AAA": 119.0}) == []
    reached = account.targets_reached({"AAA": 121.0})
    assert len(reached) == 1
    assert len(account.state.positions) == 1, "la position doit rester ouverte"


def test_position_without_target_is_never_signalled(account):
    account.open_position("AAA", 1.0, 100.0, stop=95.0)
    assert account.targets_reached({"AAA": 999.0}) == []


# ------------------------------------------- coherence du debrief sur la sortie


def test_planned_risk_is_never_negative(account):
    """Un stop remonté au-dessus de l'entrée ne planifie plus une perte.

    Sans ce garde-fou, l'« enveloppe de perte » devenait négative et le debrief
    annonçait qu'une perte de 2 EUR « dépassait l'enveloppe prévue de -0,26 EUR »,
    une phrase qui n'enseigne rien.
    """
    account.open_position("AAA", 1.0, 100.0, stop=95.0)
    position = account.state.positions[0]
    account.update_stop(position.id, 110.0)
    trade = account.check_stops({"AAA": 90.0})[0]
    assert trade.stop_locks_gain
    assert trade.planned_risk == 0.0


def test_debrief_never_contradicts_itself_about_the_stop(account):
    """Deux leçons opposées sur la même sortie s'annulent : il n'en faut qu'une.

    Le cas reproduit ici est celui observe en conditions réelles : stop resserré
    au-dessus de l'entrée, puis cours nettement plus bas au déclenchement.
    """
    account.open_position("AAA", 1.0, 100.0, stop=95.0)
    account.update_stop(account.state.positions[0].id, 101.0)
    trade = account.check_stops({"AAA": 97.0})[0]
    titles = [lesson.title for lesson in debrief_trade(trade, account.state).lessons]
    assert sum("comme prévu" in title for title in titles) + sum(
        "au-delà du stop" in title or "protégeait un gain" in title for title in titles
    ) == 1, titles


def test_debrief_names_a_stop_that_was_protecting_a_gain(account):
    account.open_position("AAA", 1.0, 100.0, stop=95.0)
    account.update_stop(account.state.positions[0].id, 101.0)
    trade = account.check_stops({"AAA": 97.0})[0]
    lessons = debrief_trade(trade, account.state).lessons
    assert any("protégeait un gain" in lesson.title for lesson in lessons)


def test_every_displayed_lesson_is_written_in_correct_french(account):
    """Garde-fou contre les mots amputes de leur accent dans les textes affiches."""
    account.deposit(5000.0)
    account.open_position("AAA", 1.0, 100.0, stop=95.0, target=101.0)
    trade = account.check_stops({"AAA": 80.0})[0]
    debrief = debrief_trade(trade, account.state)
    texts = [debrief.verdict] + [f"{les.title} {les.message}" for les in debrief.lessons]
    fautes = ("repeter", "methode", "duree", "reglage", "reflexe", "asymetrie",
              "surdimensionnes", "ferme le", "prevu", "acceptee", "affiche n")
    for text in texts:
        for faute in fautes:
            assert faute not in text.lower(), f"{faute!r} sans accent dans : {text}"


def test_no_reward_ratio_lesson_when_nothing_was_risked(account):
    """Risquer 0 pour espérer 42 n'est pas un mauvais rapport : il n'y en a pas.

    Le rapport valait alors 0.00 et le debrief reprochait à l'utilisateur un
    trade dont le stop, une fois remonté, ne planifiait plus aucune perte.
    """
    account.deposit(3000.0)
    account.open_position("AAA", 1.0, 100.0, stop=95.0, target=145.0)
    account.update_stop(account.state.positions[0].id, 101.0)
    trade = account.check_stops({"AAA": 99.0})[0]
    assert trade.planned_risk == 0.0
    lessons = debrief_trade(trade, account.state).lessons
    assert not any("rapport gain/perte visé" in lesson.title for lesson in lessons)

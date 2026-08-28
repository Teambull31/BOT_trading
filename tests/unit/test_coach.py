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
from trader.coach.advisor import (
    Severity,
    TradePlan,
    review_plan,
    suggest_size,
    suggest_target,
)
from trader.coach.curriculum import (
    LEVELS,
    MAX_OPEN_RISK_PCT,
    MIN_PLANNED_R,
    break_even_rate,
    evaluate_progress,
)
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
    target: float | None = None,
    trailing: float | None = None,
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
        target=target,
        trailing_pct=trailing,
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


def _levels(state) -> dict:
    return {status.level.key: status for status in evaluate_progress(state).levels}


def test_asymmetry_level_grades_the_plan_not_the_outcome(account):
    """Douze trades PERDANTS mais planifies asymetriques valident le palier.

    Le gain moyen obtenu ne dit rien du process : sans pouvoir predictif, il
    depend surtout du hasard. Ce que l'eleve choisit, c'est l'objectif qu'il
    place en face de son stop.
    """
    account.state.history = [
        _trade(symbol=f"S{i}", pnl=-4.0, entry=100.0, stop=98.0, target=105.0, days_ago=90 - i * 7)
        for i in range(12)
    ]
    status = _levels(account.state)["asymetrie"]
    assert status.achieved, status.detail


def test_asymmetry_level_accepts_a_trailing_stop_as_the_plan(account):
    """Un stop suiveur ne plafonne pas le gain : c'est le « laisser courir »."""
    account.state.history = [
        _trade(symbol=f"S{i}", pnl=1.0, entry=100.0, stop=98.0, trailing=8.0, days_ago=90 - i * 7)
        for i in range(12)
    ]
    status = _levels(account.state)["asymetrie"]
    assert status.achieved, status.detail
    assert "suiveur" in status.detail


def test_asymmetry_level_blocks_a_trade_without_any_plan(account):
    account.state.history = [
        _trade(symbol=f"S{i}", pnl=20.0, entry=100.0, stop=98.0, days_ago=90 - i * 7)
        for i in range(12)
    ]
    status = _levels(account.state)["asymetrie"]
    assert not status.achieved
    assert "sans objectif ni stop suiveur" in status.detail


def test_asymmetry_level_blocks_an_objective_smaller_than_the_risk(account):
    """Viser 1 EUR pour en risquer 2 est le contraire de « couper court »."""
    account.state.history = [
        _trade(symbol=f"S{i}", pnl=1.0, entry=100.0, stop=98.0, target=101.0, days_ago=90 - i * 7)
        for i in range(12)
    ]
    status = _levels(account.state)["asymetrie"]
    assert not status.achieved
    assert "0.5 fois" in status.detail


def test_asymmetry_level_does_not_divide_by_a_null_planned_risk(account):
    """Stop deja remonte au-dessus de l'entree : le risque planifie vaut zero."""
    account.state.history = [
        _trade(symbol=f"S{i}", pnl=5.0, entry=100.0, stop=104.0, target=110.0, days_ago=90 - i * 7)
        for i in range(12)
    ]
    assert _levels(account.state)["asymetrie"].achieved


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


def test_risk_at_stop_is_exactly_the_loss_that_would_be_booked(account):
    """Le chiffre montre a l'utilisateur doit etre celui qu'il perdrait vraiment.

    Frais des deux ordres et ecart de cotation compris : une difference
    entree/stop toute nue minore la perte, et l'app afficherait un risque
    plus doux que la realite qu'elle simule elle-meme.
    """
    position = account.open_position("AAA", shares=5.0, price=100.0, stop=95.0)
    annonce = position.risk_at_stop()
    assert annonce > 0

    # Le stop est touche : on solde AU niveau du stop, l'hypothese exacte du
    # chiffre annonce. Le resultat inscrit doit en etre l'oppose, au centime.
    trade = account.close_position(position.id, 95.0, reason="stop_touche")
    assert trade.pnl == pytest.approx(-annonce)


def test_risk_at_stop_turns_into_a_locked_gain_once_the_stop_passes_the_entry(account):
    """Stop remonte au-dessus du prix de revient : le trade ne peut plus couter.

    C'est ce que le stop suiveur du parcours cherche a produire, et le signe
    negatif est la seule facon de le dire sans promettre quoi que ce soit sur
    la suite du cours.
    """
    position = account.open_position("AAA", shares=5.0, price=100.0, stop=95.0)
    assert position.risk_at_stop() > 0

    account.update_stop(position.id, 110.0)
    verrouille = account.find_position(position.id).risk_at_stop()
    assert verrouille < 0
    assert account.close_position(position.id, 110.0).pnl == pytest.approx(-verrouille)


def test_suggest_target_places_the_objective_at_the_curriculum_ratio():
    """L'objectif propose est le seuil du palier traduit en prix."""
    # 100 - 95 = 5 de perte acceptee ; 1.5 fois cette perte porte la cible a 107.5.
    cible = suggest_target(price=100.0, stop=95.0)
    assert cible == pytest.approx(100.0 + MIN_PLANNED_R * 5.0)

    plan = TradePlan(symbol="AAA", shares=1.0, price=100.0, stop=95.0, target=cible)
    assert plan.reward_risk == pytest.approx(MIN_PLANNED_R)


def test_suggest_target_honours_an_explicit_ratio():
    assert suggest_target(100.0, 90.0, ratio=3.0) == pytest.approx(130.0)


def test_suggest_target_is_absent_without_a_planned_loss():
    """Stop au-dessus (ou au niveau) du prix : aucune perte dont l'objectif soit un multiple."""
    assert suggest_target(100.0, 105.0) is None
    assert suggest_target(100.0, 100.0) is None
    assert suggest_target(0.0, -1.0) is None


def test_open_risk_is_what_the_account_loses_if_every_stop_falls(account):
    """La somme annoncee doit etre celle qui serait reellement inscrite.

    C'est le chiffre que le compte subit le jour d'une baisse generale, ou les
    stops ne tombent pas les uns apres les autres mais ensemble.
    """
    account.open_position("AAA", shares=2.0, price=100.0, stop=95.0)
    account.open_position("BBB", shares=1.0, price=100.0, stop=90.0)
    annonce = account.open_risk()
    assert annonce > 0

    subi = sum(
        account.close_position(position.id, position.stop, reason="stop_touche").pnl
        for position in list(account.state.positions)
    )
    assert subi == pytest.approx(-annonce)


def test_open_risk_goes_negative_once_every_stop_locks_a_gain(account):
    """Portefeuille qui ne peut plus rien couter : le total passe sous zero."""
    position = account.open_position("AAA", shares=2.0, price=100.0, stop=95.0)
    account.update_stop(position.id, 110.0)
    assert account.open_risk() < 0


def test_open_risk_pct_is_zero_without_positions(account):
    assert account.open_risk() == 0.0
    assert account.open_risk_pct({}) == 0.0


def test_review_warns_when_the_positions_together_risk_too_much(account):
    """Chaque trade dans sa limite, leur somme au-dessus : c'est le compte qui joue.

    Rien dans l'app ne mesurait ce cumul ; trois trades irreprochables
    pouvaient engager une part du capital que le parcours interdit de perdre.
    """
    account.open_position("AAA", shares=2.0, price=100.0, stop=80.0)
    account.open_position("BBB", shares=2.0, price=100.0, stop=80.0)
    assert account.open_risk_pct({}) > MAX_OPEN_RISK_PCT

    plan = TradePlan(symbol="CCC", shares=1.0, price=100.0, stop=95.0, target=110.0)
    review = review_plan(plan, account, prices={})
    cumul = [a for a in review.advices if "cumulé" in a.title]
    assert len(cumul) == 1
    # Le trade lui-meme reste petit : sans le cumul, il serait declare sain.
    assert review.risk_pct < 1.0
    assert cumul[0].severity is Severity.WARNING


def test_review_blocks_a_cumulative_risk_twice_over_the_limit(account):
    account.open_position("AAA", shares=2.0, price=100.0, stop=80.0)
    account.open_position("BBB", shares=2.0, price=100.0, stop=80.0)

    plan = TradePlan(symbol="CCC", shares=3.0, price=100.0, stop=70.0)
    review = review_plan(plan, account, prices={})
    assert not review.can_proceed
    assert any("cumulé" in advice.title for advice in review.blockers)


def test_review_says_nothing_about_a_cumulative_risk_on_a_first_trade(account):
    """Sans position ouverte, il n'y a pas de cumul : le conseil serait du bruit."""
    plan = TradePlan(symbol="AAA", shares=2.0, price=100.0, stop=95.0, target=115.0)
    review = review_plan(plan, account, prices={})
    assert not any("cumulé" in advice.title for advice in review.advices)


def test_review_credits_positions_whose_stops_can_no_longer_cost_anything(account):
    """Stops passes au-dessus du prix de revient : seul le nouveau trade engage du capital."""
    position = account.open_position("AAA", shares=2.0, price=100.0, stop=95.0)
    account.update_stop(position.id, 110.0)

    plan = TradePlan(symbol="BBB", shares=2.0, price=100.0, stop=95.0, target=115.0)
    review = review_plan(plan, account, prices={})
    acquis = [a for a in review.advices if "plus rien coûter" in a.title]
    assert len(acquis) == 1
    assert acquis[0].severity is Severity.GOOD
    assert review.can_proceed


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


def test_review_does_not_call_a_ratio_below_the_curriculum_threshold_good(account):
    """1.2 fois le risque n'est pas « rentable en se trompant une fois sur deux ».

    Il faut 45 % de reussite pour seulement rentrer dans ses frais. Annoncer
    l'inverse serait une promesse fausse, et contredirait le palier 5 qui exige
    MIN_PLANNED_R.
    """
    plan = TradePlan(symbol="AAA", shares=2.0, price=100.0, stop=95.0, target=106.0)
    review = review_plan(plan, account, prices={})
    ratio_advices = [a for a in review.advices if "gain/perte" in a.title]
    assert [a.severity for a in ratio_advices] == [Severity.INFO]
    assert "45 %" in ratio_advices[0].message


def test_review_states_the_break_even_rate_of_a_good_ratio(account):
    """Le seul chiffre honnete sur l'issue d'un trade : il ne prevoit rien."""
    plan = TradePlan(symbol="AAA", shares=2.0, price=100.0, stop=95.0, target=115.0)
    review = review_plan(plan, account, prices={})
    good = [a for a in review.advices if "gain/perte" in a.title]
    assert [a.severity for a in good] == [Severity.GOOD]
    assert "25 %" in good[0].message


def test_review_counts_a_trailing_stop_as_a_plan(account):
    """Le palier 5 accepte le suiveur comme plan ; le conseil doit dire pareil."""
    plan = TradePlan(symbol="AAA", shares=2.0, price=100.0, stop=95.0, trailing_pct=8.0)
    review = review_plan(plan, account, prices={})
    assert not any("Aucun objectif" in advice.title for advice in review.advices)
    assert any(
        "stop suiveur" in advice.title and advice.severity is Severity.GOOD
        for advice in review.advices
    )


def test_review_still_flags_a_trade_without_objective_nor_trail(account):
    plan = TradePlan(symbol="AAA", shares=2.0, price=100.0, stop=95.0)
    review = review_plan(plan, account, prices={})
    assert any("Aucun objectif" in advice.title for advice in review.advices)


def test_review_reports_a_null_reward_ratio_as_zero_not_as_absent(account):
    """Objectif pose au prix d'entree : le ratio vaut 0, ce n'est pas « pas d'objectif »."""
    plan = TradePlan(symbol="AAA", shares=2.0, price=100.0, stop=95.0, target=100.0)
    review = review_plan(plan, account, prices={})
    assert review.to_dict()["reward_risk"] == 0.0


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
    assert (
        sum("comme prévu" in title for title in titles)
        + sum("au-delà du stop" in title or "protégeait un gain" in title for title in titles)
        == 1
    ), titles


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
    fautes = (
        "repeter",
        "methode",
        "duree",
        "reglage",
        "reflexe",
        "asymetrie",
        "surdimensionnes",
        "ferme le",
        "prevu",
        "acceptee",
        "affiche n",
    )
    for text in texts:
        for faute in fautes:
            assert faute not in text.lower(), f"{faute!r} sans accent dans : {text}"


def test_weak_plan_lesson_fires_on_a_winning_trade_too(account):
    """Le cas ou la lecon se perd : le resultat semble donner raison au plan.

    Juger la decision et non le resultat, c'est aussi reprocher un mauvais plan
    a un trade qui a gagne. Ne le dire qu'aux perdants apprendrait a confondre
    chance et competence — exactement ce que ce module refuse.
    """
    trade = _trade(pnl=40.0, entry=100.0, stop=95.0, shares=1.0, target=102.0)
    lessons = debrief_trade(trade, account.state).lessons
    faibles = [lesson for lesson in lessons if "rapport gain/perte visé" in lesson.title]
    assert len(faibles) == 1
    assert "0.40" in faibles[0].title
    # 1 / (1 + 0.4) = 71 %, et le seuil du palier ramene a 40 %.
    assert "71 %" in faibles[0].message
    assert "40 %" in faibles[0].message
    assert "le tirage" in faibles[0].message


def test_weak_plan_lesson_uses_the_curriculum_threshold(account):
    """Un plan conforme au palier ne doit rien se voir reprocher."""
    trade = _trade(pnl=-5.0, entry=100.0, stop=95.0, shares=1.0, target=110.0)
    lessons = debrief_trade(trade, account.state).lessons
    assert not any("rapport gain/perte visé" in lesson.title for lesson in lessons)


def test_no_weak_plan_lesson_for_a_trailing_stop(account):
    """Le suiveur ne plafonne rien : il n'y a pas de rapport a reprocher."""
    trade = _trade(pnl=-5.0, entry=100.0, stop=95.0, shares=1.0, trailing=8.0)
    lessons = debrief_trade(trade, account.state).lessons
    assert not any("rapport gain/perte visé" in lesson.title for lesson in lessons)


def test_break_even_rate_is_the_inverse_of_one_plus_the_ratio():
    """La seule affirmation chiffree que l'app fasse sur l'issue d'un trade."""
    assert break_even_rate(1.0) == pytest.approx(50.0)
    assert break_even_rate(MIN_PLANNED_R) == pytest.approx(40.0)
    assert break_even_rate(3.0) == pytest.approx(25.0)


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


def _quote(price: float, symbol: str = "AAA") -> Quote:
    """Cotation minimale pour les tests d'analyse de plan."""
    return Quote(
        symbol=symbol,
        price=price,
        change=0.0,
        change_pct=0.0,
        previous_close=price,
        market_status="Market Open",
        is_real_time=True,
        timestamp="",
    )


# ------------------------------------------------------------- stop suiveur


def test_trailing_stop_only_ever_rises(account):
    """Le seul mouvement qu'un suiveur autorise est celui qui réduit le risque.

    C'est ce qui le sépare d'un stop déplacé à la main : il ne peut jamais
    devenir le prétexte à ne pas matérialiser une perte.
    """
    account.open_position("AAA", 1.0, 100.0, stop=90.0, trailing_pct=10.0)
    position = account.state.positions[0]
    assert position.stop == pytest.approx(position.entry_price * 0.9)

    account.mark({"AAA": 120.0})
    assert position.stop == pytest.approx(108.0)

    account.mark({"AAA": 95.0})
    assert position.stop == pytest.approx(108.0), "un suiveur ne redescend jamais"


def test_trailing_stop_tighter_than_typed_stop_governs_from_entry(account):
    """Le stop en vigueur à l'ouverture est le plus serré des deux."""
    position = account.open_position("AAA", 1.0, 100.0, stop=80.0, trailing_pct=5.0)
    assert position.stop == pytest.approx(position.entry_price * 0.95)
    assert position.stop > 80.0


def test_looser_trailing_stop_does_not_widen_the_typed_stop(account):
    """Armer un suiveur large ne doit jamais relâcher un stop déjà serré."""
    position = account.open_position("AAA", 1.0, 100.0, stop=98.0, trailing_pct=20.0)
    assert position.stop == pytest.approx(98.0)


def test_removing_the_trail_keeps_the_ground_it_took(account):
    """Retirer le suiveur est permis ; rendre le stop qu'il a remonté ne l'est pas."""
    account.open_position("AAA", 1.0, 100.0, stop=90.0, trailing_pct=10.0)
    position = account.state.positions[0]
    account.mark({"AAA": 130.0})
    assert position.stop == pytest.approx(117.0)

    account.set_trailing(position.id, None)
    assert position.trailing_pct is None
    assert position.stop == pytest.approx(117.0)


def test_arming_a_trail_anchors_on_the_current_price_not_a_past_peak(account):
    """Un suiveur commence à compter là où on le pose.

    Sans cet ancrage, armer un suiveur sur une position qui a reflué placerait
    le stop au-dessus du cours et solderait la position sur le champ, à un
    niveau que personne n'a choisi.
    """
    account.open_position("AAA", 1.0, 100.0, stop=90.0)
    position = account.state.positions[0]
    account.mark({"AAA": 150.0})
    account.mark({"AAA": 120.0})
    assert position.highest_price == pytest.approx(150.0)

    account.set_trailing(position.id, 5.0, price=120.0)
    assert position.stop == pytest.approx(114.0)
    assert position.stop < 120.0, "armer un suiveur ne doit pas sortir la position"
    assert not account.check_stops({"AAA": 120.0})


def test_trailing_pct_must_be_a_sane_distance(account):
    with pytest.raises(ValueError):
        account.open_position("AAA", 1.0, 100.0, stop=90.0, trailing_pct=0.0)
    with pytest.raises(ValueError):
        account.open_position("BBB", 1.0, 100.0, stop=90.0, trailing_pct=100.0)
    account.open_position("CCC", 1.0, 100.0, stop=90.0)
    with pytest.raises(ValueError):
        account.set_trailing(account.state.positions[0].id, 150.0)


def test_trail_survives_a_reload(account, tmp_path):
    """Le suiveur et son ancrage doivent traverser un redémarrage."""
    account.open_position("AAA", 1.0, 100.0, stop=90.0, trailing_pct=8.0)
    account.mark({"AAA": 140.0})
    reloaded = PaperAccount(tmp_path / "account.json")
    position = reloaded.state.positions[0]
    assert position.trailing_pct == pytest.approx(8.0)
    assert position.trail_high == pytest.approx(140.0)
    assert position.stop == pytest.approx(128.8)


def test_a_position_written_before_trailing_existed_still_loads(account, tmp_path):
    """Un fichier sans `trail_high` ne doit pas rendre le suiveur inopérant."""
    account.open_position("AAA", 1.0, 100.0, stop=90.0)
    store = tmp_path / "account.json"
    payload = json.loads(store.read_text(encoding="utf-8"))
    for position in payload["positions"]:
        position.pop("trail_high", None)
        position.pop("trailing_pct", None)
    store.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = PaperAccount(store)
    position = reloaded.state.positions[0]
    assert position.trail_high == pytest.approx(position.highest_price)
    reloaded.set_trailing(position.id, 10.0, price=100.0)
    assert position.stop == pytest.approx(90.0)


# ----------------------------------------------- le suiveur dit la vérité


def test_review_computes_the_risk_on_the_stop_actually_in_force(account):
    """Afficher un risque que le compte ne court pas serait le pire mensonge ici."""
    plan = TradePlan(symbol="AAA", shares=10.0, price=100.0, stop=80.0, trailing_pct=5.0)
    assert plan.effective_stop == pytest.approx(95.0)
    assert plan.trailing_overrides_stop
    assert plan.risk_amount == pytest.approx(50.0)
    assert plan.stop_distance_pct == pytest.approx(5.0)


def test_review_warns_when_the_trail_replaces_the_typed_stop(account):
    plan = TradePlan(symbol="AAA", shares=1.0, price=100.0, stop=80.0, trailing_pct=5.0)
    quote = _quote(100.0)
    review = review_plan(plan, account, quote, {})
    warnings = [a for a in review.advices if a.severity is Severity.WARNING]
    assert any("remplace votre stop" in advice.title for advice in warnings)


def test_review_stays_silent_about_a_trail_that_changes_nothing(account):
    """Un suiveur plus large que le stop saisi ne remplace rien : ne pas crier."""
    plan = TradePlan(symbol="AAA", shares=1.0, price=100.0, stop=98.0, trailing_pct=20.0)
    quote = _quote(100.0)
    review = review_plan(plan, account, quote, {})
    assert not any("remplace votre stop" in advice.title for advice in review.advices)
    assert any(advice.title.startswith("Stop suiveur à") for advice in review.advices)


def test_debrief_names_what_the_trail_cost_and_what_it_bought(account):
    """Un suiveur laisse toujours une part du plus haut : c'est son prix, pas son défaut."""
    account.deposit(3000.0)
    account.open_position("AAA", 1.0, 100.0, stop=90.0, trailing_pct=10.0)
    account.mark({"AAA": 130.0})
    trade = account.check_stops({"AAA": 116.0})[0]

    assert trade.trailing_pct == pytest.approx(10.0)
    assert trade.pnl > 0, "le suiveur a verrouillé un gain"
    lessons = debrief_trade(trade, account.state).lessons
    assert any("suiveur à 10.0 %" in lesson.title for lesson in lessons)

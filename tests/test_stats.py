"""Opportunity counting for each stat, on hand-crafted hands (button on seat 5:
SB=0, BB=1, UTG=2, HJ=3, CO=4, BTN=5; blinds 1/2)."""

import pytest

from poker_lab.profiling.stats import C, counters_by_player, hand_counters, raw_stats
from poker_lab.schema import ChipAward, HandHistoryStore, Street
from tests.helpers import BET as B
from tests.helpers import CALL as CA
from tests.helpers import CHECK as K
from tests.helpers import FOLD as F
from tests.helpers import PLAYERS, TABLE, seats
from tests.helpers import RAISE as R
from tests.helpers import build_scripted_hand as build


def counts(hand) -> dict[int, dict[str, int]]:
    by_pid = dict(hand_counters(hand))
    return {i: dict(zip(C, by_pid[PLAYERS[i]], strict=True)) for i in range(6)}


P, FL, T, RV = Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER


def test_open_call_cbet_folds():
    hand = build(
        {
            P: [(2, F), (3, R, 6), (4, CA), (5, F), (0, F), (1, CA)],
            FL: [(1, K), (3, B, 10), (4, F), (1, F)],
        }
    )
    c = counts(hand)
    assert all(c[i]["vpip_opp"] == 1 for i in range(6))
    assert [c[i]["vpip"] for i in range(6)] == [0, 1, 0, 1, 1, 0]
    assert [c[i]["pfr"] for i in range(6)] == [0, 0, 0, 1, 0, 0]
    # Facing exactly one raise: CO, BTN, SB, BB had a 3-bet opportunity; UTG acted before it.
    assert [c[i]["three_bet_opp"] for i in range(6)] == [1, 1, 0, 0, 1, 1]
    assert sum(c[i]["three_bet"] for i in range(6)) == 0
    assert (c[3]["cbet_opp"], c[3]["cbet"]) == (1, 1)
    # BB checked before the c-bet, then faced it; CO faced it directly. Both folded.
    assert (c[1]["fold_to_cbet_opp"], c[1]["fold_to_cbet"]) == (1, 1)
    assert (c[4]["fold_to_cbet_opp"], c[4]["fold_to_cbet"]) == (1, 1)
    assert [c[i]["saw_flop"] for i in range(6)] == [0, 1, 0, 1, 1, 0]
    assert c[3]["pf_aggr"] == 1
    assert c[3]["bet_size_x100"] == round(100 * 10 / 19)  # 10 into 1 + 6 + 6 + 6 = 19 chips
    assert c[3]["open_size_x100"] == 300  # opened to 3 big blinds
    assert sum(c[i]["wtsd"] for i in range(6)) == 0


def test_three_bet_and_fold_to_three_bet():
    hand = build({P: [(2, R, 6), (3, R, 18), (4, F), (5, F), (0, F), (1, F), (2, F)]})
    c = counts(hand)
    assert (c[3]["three_bet_opp"], c[3]["three_bet"]) == (1, 1)
    assert (c[2]["fold_to_3bet_opp"], c[2]["fold_to_3bet"]) == (1, 1)
    # Players acting after the 3-bet face two raises: no 3-bet opportunity for them.
    assert all(c[i]["three_bet_opp"] == 0 for i in (4, 5, 0, 1))
    assert hand.uncalled_returns == (ChipAward(3, 12),)


def test_walk_is_not_a_vpip_opportunity_for_the_big_blind():
    hand = build({P: [(2, F), (3, F), (4, F), (5, F), (0, F)]})
    c = counts(hand)
    assert c[1]["vpip_opp"] == 0
    assert c[0]["vpip_opp"] == 1
    assert c[1]["hands"] == 1


def test_limped_pot_has_no_cbet_and_big_blind_check_is_not_vpip():
    hand = build(
        {
            P: [(2, CA), (3, F), (4, F), (5, F), (0, CA), (1, K)],
            FL: [(0, B, 4), (1, F), (2, F)],
        }
    )
    c = counts(hand)
    assert [c[i]["vpip"] for i in (0, 1, 2)] == [1, 0, 1]
    assert sum(c[i]["cbet_opp"] for i in range(6)) == 0
    assert sum(c[i]["fold_to_cbet_opp"] for i in range(6)) == 0
    assert c[1]["fold_to_bet_opp"] == 1


def test_donk_bet_removes_the_cbet_opportunity():
    hand = build(
        {
            P: [(2, F), (3, R, 6), (4, F), (5, F), (0, F), (1, CA)],
            FL: [(1, B, 6), (3, CA)],
            T: [(1, K), (3, K)],
            RV: [(1, K), (3, K)],
        },
        showdown=(1, 3),
        winners=[3],
    )
    c = counts(hand)
    assert c[3]["cbet_opp"] == 0
    assert c[3]["pf_calls"] == 1
    assert c[1]["pf_aggr"] == 1
    assert (c[1]["saw_flop"], c[1]["wtsd"], c[1]["wsd"]) == (1, 1, 0)
    assert (c[3]["saw_flop"], c[3]["wtsd"], c[3]["wsd"]) == (1, 1, 1)


def test_raise_over_cbet_ends_fold_to_cbet_spots_behind_it():
    hand = build(
        {
            P: [(2, F), (3, R, 6), (4, CA), (5, CA), (0, F), (1, F)],
            FL: [(3, B, 10), (4, R, 30), (5, F), (3, F)],
        }
    )
    c = counts(hand)
    assert (c[4]["fold_to_cbet_opp"], c[4]["fold_to_cbet"]) == (1, 0)
    assert c[5]["fold_to_cbet_opp"] == 0  # faced a raise, not the c-bet
    assert c[5]["fold_to_bet_opp"] == 1
    assert c[3]["fold_to_bet"] == 1


def test_preflop_all_in_counts_as_seeing_the_flop_and_showdown():
    hand = build(
        {P: [(2, F), (3, R, 200), (4, F), (5, F), (0, F), (1, CA)]},
        showdown=(1, 3),
        winners=[1, 3],
    )
    c = counts(hand)
    for seat in (1, 3):
        assert (c[seat]["saw_flop"], c[seat]["wtsd"], c[seat]["wsd"]) == (1, 1, 1)


def test_raw_stats_and_aggregation():
    hands = [
        build({P: [(2, F), (3, R, 6), (4, F), (5, F), (0, F), (1, F)]}),
        build(
            {
                P: [(2, F), (3, CA), (4, F), (5, F), (0, F), (1, K)],
                FL: [(1, B, 2), (3, CA)],
                T: [(1, K), (3, B, 4), (1, F)],
            }
        ),
    ]
    totals = counters_by_player(hands)
    s = raw_stats(totals["dave"])  # seat 3
    assert s["vpip"] == 1.0
    assert s["pfr"] == 0.5
    assert s["af"] == 1.0  # one bet, one call postflop
    assert s["hands"] == 2
    assert s["cbet"] != s["cbet"]  # NaN: no opportunity


def test_rejects_in_progress_hands():
    store = HandHistoryStore()
    store.begin_hand("y", TABLE, 5, seats(5))
    with pytest.raises(ValueError, match="complete hands"):
        hand_counters(store.get("y"))

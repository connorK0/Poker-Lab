"""Decision-point extraction uses only information available before each action."""

import pytest

from poker_lab.profiling import SITUATION_FEATURES, DecisionRowExtractor, decision_points
from poker_lab.schema import Street
from tests.helpers import BET, CALL, CHECK, FOLD, PLAYERS, RAISE, build_scripted_hand

P, FL, T = Street.PREFLOP, Street.FLOP, Street.TURN
F = {name: i for i, name in enumerate(SITUATION_FEATURES)}


def test_decision_points_features_and_labels():
    hand = build_scripted_hand(
        {
            P: [(2, FOLD), (3, RAISE, 6), (4, CALL), (5, FOLD), (0, FOLD), (1, CALL)],
            FL: [(1, CHECK), (3, BET, 10), (4, RAISE, 30), (1, FOLD), (3, FOLD)],
        }
    )
    dps = decision_points(hand)
    assert len(dps) == 11  # every voluntary action, blinds excluded
    labels = [label for _, _, label in dps]
    assert labels == [0, 2, 1, 0, 0, 1, 1, 2, 2, 0, 0]
    by_order = [(pid, dict(zip(SITUATION_FEATURES, f, strict=True))) for pid, f, _ in dps]

    pid, utg = by_order[0]
    assert pid == PLAYERS[2]
    assert utg["unopened"] == 1.0
    assert utg["facing_bet"] == 1.0  # the big blind is a bet to call
    assert utg["pot_odds"] == pytest.approx(2 / 5)

    _, co = by_order[2]  # CO facing the open: not unopened, one raise on the street
    assert co["unopened"] == 0.0
    assert co["n_raises_street"] == 1.0

    _, bb_check = by_order[6]
    assert bb_check["cbet_spot"] == 0.0  # BB is not the preflop aggressor
    assert bb_check["checked_to"] == 0.0

    _, hj_flop = by_order[7]  # HJ's c-bet spot after BB checked
    assert hj_flop["is_pfa"] == 1.0
    assert hj_flop["cbet_spot"] == 1.0
    assert hj_flop["checked_to"] == 1.0
    assert hj_flop["facing_bet"] == 0.0

    _, co_flop = by_order[8]  # CO facing the c-bet: pot 19 + bet 10, 10 to call
    assert co_flop["facing_cbet"] == 1.0
    assert co_flop["pot_odds"] == pytest.approx(10 / (19 + 10 + 10))

    _, bb_flop2 = by_order[9]  # BB facing bet + raise: no longer "facing the c-bet"
    assert bb_flop2["facing_cbet"] == 0.0
    assert bb_flop2["acted_this_street"] == 1.0
    _, hj_again = by_order[10]
    assert hj_again["acted_this_street"] == 1.0
    assert hj_again["n_raises_street"] == 2.0


def test_features_do_not_depend_on_future_actions():
    script = {P: [(2, FOLD), (3, RAISE, 6), (4, FOLD), (5, FOLD), (0, FOLD), (1, CALL)]}
    a = build_scripted_hand({**script, FL: [(1, CHECK), (3, BET, 4), (1, FOLD)]})
    b = build_scripted_hand({**script, FL: [(1, CHECK), (3, CHECK)], T: [(1, CHECK), (3, CHECK)]})
    da, db = decision_points(a), decision_points(b)
    # Identical prefixes give identical features for every shared decision.
    for (pa, fa, _), (pb, fb, _) in zip(da[:7], db[:7], strict=True):
        assert pa == pb
        assert fa == fb


def test_row_extractor_filters_by_window():
    hand = build_scripted_hand({P: [(2, FOLD), (3, FOLD), (4, FOLD), (5, FOLD), (0, FOLD)]})
    extractor = DecisionRowExtractor(windows=((10, 20),))
    pop_index = {p: i for i, p in enumerate(PLAYERS)}
    assert extractor(hand, pop_index, 5) == []
    rows = extractor(hand, pop_index, 12)
    assert len(rows) == 5
    assert rows[0][0] == 2  # population index of UTG ("carol")
    assert rows[0][1] == 12
    assert len(rows[0]) == 2 + len(SITUATION_FEATURES) + 1

"""Hand-history data model: construction invariants and deep validation."""

import dataclasses

import pytest

from poker_lab.schema import (
    Action,
    ActionType,
    CardObservation,
    ChipAward,
    HandHistory,
    HandHistoryStore,
    HandHistoryValidationError,
    HandStatus,
    HoleCards,
    SeatInfo,
    ShowdownEntry,
    Street,
    TableConfig,
    compute_uncalled_returns,
    find_problems,
    position_names,
    seat_order_from_button,
    validate_hand_history,
)
from tests.helpers import TABLE, A, P, act, obs, record_reference_hand, seats


@pytest.fixture
def hand() -> HandHistory:
    h, _ = record_reference_hand(HandHistoryStore())
    assert h is not None
    return h


def test_reference_hand_is_valid_and_conserves_chips(hand):
    assert find_problems(hand) == []
    assert hand.contributions() == {0: 1, 1: 6, 2: 0, 3: 36, 4: 0, 5: 36}
    net = hand.net_results()
    assert sum(net.values()) == 0
    assert net[5] == 79 - 36
    assert hand.went_to_showdown(3)
    assert hand.went_to_showdown(5)
    assert not hand.went_to_showdown(1)
    assert hand.hole_cards_of(5).shown
    assert not hand.hole_cards_of(1).shown


def test_positions():
    assert position_names(6) == ("SB", "BB", "UTG", "HJ", "CO", "BTN")
    assert position_names(2) == ("BB", "BTN")
    assert position_names(3) == ("SB", "BB", "BTN")
    assert len(position_names(10)) == len(set(position_names(10))) == 10
    assert seat_order_from_button([0, 2, 5, 7], 5) == (7, 0, 2, 5)
    with pytest.raises(ValueError, match="not occupied"):
        seat_order_from_button([0, 1], 3)


def test_in_progress_hand_history_cannot_carry_hole_cards():
    hole = HoleCards(0, (obs("As"), obs("Kd")))
    with pytest.raises(ValueError, match="sealed"):
        HandHistory("h", TABLE, 5, seats(), hole_cards=(hole,), status=HandStatus.IN_PROGRESS)
    with pytest.raises(ValueError, match="in-progress"):
        HandHistory(
            "h", TABLE, 5, seats(), payouts=(ChipAward(0, 3),), status=HandStatus.IN_PROGRESS
        )


def test_objects_are_frozen_and_slotted(hand):
    with pytest.raises(dataclasses.FrozenInstanceError):
        hand.status = HandStatus.IN_PROGRESS  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        hand.extra = 1  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        hand.metadata["x"] = "y"  # type: ignore[index]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"card": "Zz"}, "invalid card"),
        ({"card": "As", "confidence": 1.5}, r"\[0, 1\]"),
        ({"card": "As", "confidence": float("nan")}, "finite"),
        ({"card": "As", "source": "telepathy"}, "telepathy"),
    ],
)
def test_card_observation_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CardObservation(**kwargs)


def test_action_invariants():
    with pytest.raises(ValueError, match="amount 0"):
        Action(0, Street.PREFLOP, ActionType.FOLD, amount=5)
    with pytest.raises(ValueError, match="must move chips"):
        Action(0, Street.FLOP, ActionType.BET, amount=0)
    with pytest.raises(ValueError, match="only happen preflop"):
        Action(0, Street.FLOP, ActionType.POST_BIG_BLIND, amount=2, total_bet=2)
    with pytest.raises(ValueError, match="total_bet"):
        Action(0, Street.FLOP, ActionType.BET, amount=10, total_bet=5)
    assert Action(0, "flop", "bet", 10, 10).street is Street.FLOP  # str coercion


def test_structural_checks():
    with pytest.raises(ValueError, match="duplicate seat"):
        HandHistory("h", TABLE, 0, (seats()[0], seats()[0]))
    with pytest.raises(ValueError, match="button_seat"):
        HandHistory("h", TABLE, 3, seats()[:2])
    with pytest.raises(ValueError, match="max_seats"):
        HandHistory("h", TableConfig("t", 2, 1, 2), 0, seats()[:3])
    with pytest.raises(ValueError, match="unknown position"):
        SeatInfo(0, "p", 100, "DEALER")


def test_validation_catches_broken_records(hand):
    broken = dataclasses.replace(hand, payouts=(ChipAward(5, 78),))
    with pytest.raises(HandHistoryValidationError, match="not conserved"):
        validate_hand_history(broken)

    bad_actions = list(hand.actions)
    bad_actions[-2] = act(3, Street.RIVER, A.BET, 5, 5)
    bad_actions[-1] = act(5, Street.RIVER, A.CHECK)  # check while facing a bet
    problems = find_problems(dataclasses.replace(hand, actions=tuple(bad_actions)))
    assert any("check while facing a bet" in p for p in problems)

    dup_board = dataclasses.replace(hand, board=(obs("Ac"), *hand.board[1:]))  # Ac is HJ's card
    assert any("duplicated" in p for p in find_problems(dup_board))

    to_folded = dataclasses.replace(hand, payouts=(ChipAward(1, 79),))
    assert any("folded" in p for p in find_problems(to_folded))

    wrong_pos = dataclasses.replace(hand, button_seat=0)
    assert any("position" in p for p in find_problems(wrong_pos))

    short_board = dataclasses.replace(hand, board=hand.board[:3])
    assert any("board cards" in p for p in find_problems(short_board))

    acted_after_fold = dataclasses.replace(
        hand, actions=(*hand.actions[:-1], act(2, Street.RIVER, A.CHECK))
    )
    assert any("already folded" in p for p in find_problems(acted_after_fold))


def test_uncalled_bet_accounting():
    store = HandHistoryStore()
    store.begin_hand("w", TABLE, 5, seats(5))
    for a in (
        act(0, P, A.POST_SMALL_BLIND, 1, 1),
        act(1, P, A.POST_BIG_BLIND, 2, 2),
        act(2, P, A.RAISE, 6, 6),
        act(3, P, A.FOLD),
        act(4, P, A.FOLD),
        act(5, P, A.FOLD),
        act(0, P, A.FOLD, 0, 1),
        act(1, P, A.FOLD, 0, 2),
    ):
        store.record_action("w", a)
    assert compute_uncalled_returns(store.get("w").actions) == {2: 4}
    with pytest.raises(HandHistoryValidationError, match="uncalled"):
        store.complete_hand("w", payouts=[ChipAward(2, 9)])
    assert store.is_in_progress("w")  # a failed completion leaves the hand open and sealed
    done = store.complete_hand("w", payouts=[ChipAward(2, 5)], uncalled_returns=[ChipAward(2, 4)])
    assert done.net_results()[2] == 3


def test_all_in_short_call_is_valid():
    store = HandHistoryStore()
    short = list(seats(5))
    short[1] = SeatInfo(1, "bob", 50, "BB")
    store.begin_hand("ai", TABLE, 5, short)
    for a in (
        act(0, P, A.POST_SMALL_BLIND, 1, 1),
        act(1, P, A.POST_BIG_BLIND, 2, 2),
        act(2, P, A.RAISE, 200, 200, all_in=True),
        act(3, P, A.FOLD),
        act(4, P, A.FOLD),
        act(5, P, A.FOLD),
        act(0, P, A.FOLD, 0, 1),
        act(1, P, A.CALL, 48, 50, all_in=True),
    ):
        store.record_action("ai", a)
    store.deal_board("ai", [obs(c) for c in ("2c", "3d", "4h", "5s", "9c")])
    hand = store.complete_hand(
        "ai",
        showdown=[ShowdownEntry(1, True), ShowdownEntry(2, True)],
        payouts=[ChipAward(1, 101)],
        uncalled_returns=[ChipAward(2, 150)],
    )
    assert hand.net_results() == {0: -1, 1: 51, 2: -50, 3: 0, 4: 0, 5: 0}

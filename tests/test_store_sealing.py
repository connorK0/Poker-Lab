"""Firewall: hole cards of an in-progress hand are sealed inside HandHistoryStore."""

import pytest

from poker_lab.schema import (
    CardObservation,
    ChipAward,
    HandHistory,
    HandHistoryStore,
    HandStatus,
    HandStoreError,
    HoleCards,
    LegalActions,
    SealedHoleCardsError,
    SeatKey,
    ShowdownEntry,
)
from poker_lab.schema.store import _HoleCardVault, _LiveHand
from tests.helpers import (
    BOARD,
    HOLE,
    obs,
    reachable_card_strings,
    record_reference_hand,
    walk_reachable,
)


@pytest.fixture
def live():
    store = HandHistoryStore()
    _, keys = record_reference_hand(store, "live", complete=False)
    return store, keys


def test_in_progress_record_has_no_hole_cards(live):
    store, _ = live
    hand = store.get("live")
    assert hand.status is HandStatus.IN_PROGRESS
    assert hand.hole_cards == ()
    for seat, cards in HOLE.items():
        assert not (set(cards) & reachable_card_strings(hand)), f"seat {seat} cards reachable"


def test_reveal_is_refused_until_complete(live):
    store, _ = live
    with pytest.raises(SealedHoleCardsError):
        store.reveal_hole_cards("live")


def test_seat_view_needs_the_issued_key(live):
    store, keys = live
    view = store.seat_view("live", keys[3])
    assert view.hole_cards == HOLE[3]
    forged = SeatKey("live", 5)  # same fields as the real key, but not the issued object
    with pytest.raises(SealedHoleCardsError):
        store.seat_view("live", forged)
    with pytest.raises(SealedHoleCardsError):
        store.seat_view("live", ("live", 5))  # type: ignore[arg-type]
    other = HandHistoryStore()
    _, other_keys = record_reference_hand(other, "live", complete=False)
    with pytest.raises(SealedHoleCardsError):
        store.seat_view("live", other_keys[5])  # a key from a different store/hand


def test_seat_view_reaches_only_own_cards_and_board(live):
    store, keys = live
    legal = LegalActions(can_fold=False, can_check=True, call_amount=0, can_bet_or_raise=True,
                         min_raise_to=2, max_raise_to=164)  # fmt: skip
    for seat, key in keys.items():
        view = store.seat_view("live", key, legal=legal)
        reachable = reachable_card_strings(view)
        assert reachable <= set(HOLE[seat]) | set(BOARD), f"seat {seat} sees {reachable}"
        assert set(HOLE[seat]) <= reachable
        for obj in walk_reachable(view):
            assert not isinstance(
                obj, HandHistoryStore | _HoleCardVault | _LiveHand | HoleCards | HandHistory
            )


def test_public_state_reaches_no_hole_cards(live):
    store, _ = live
    public = store.public_state("live")
    assert reachable_card_strings(public) <= set(BOARD)


def test_completion_unseals(live):
    store, keys = live
    hand = store.complete_hand(
        "live",
        showdown=[ShowdownEntry(3, True), ShowdownEntry(5, True)],
        payouts=[ChipAward(5, 79)],
    )
    revealed = store.reveal_hole_cards("live")
    assert {s: tuple(o.card for o in hc.cards) for s, hc in revealed.items()} == HOLE
    assert hand.hole_cards_of(3).shown
    assert not hand.hole_cards_of(0).shown
    with pytest.raises(HandStoreError):  # the hand is no longer live; old keys are useless
        store.seat_view("live", keys[3])


def test_dealing_rejects_duplicates_and_unknown_seats():
    store = HandHistoryStore()
    record_reference_hand(store, "x", complete=False)
    with pytest.raises(HandStoreError):
        store.deal_hole_cards("x", 0, [obs("2d"), obs("2h")])  # seat 0 already dealt
    store.begin_hand("y", store.get("x").table, 5, store.get("x").seats)
    store.deal_hole_cards("y", 0, [obs("2d"), obs("2h")])
    with pytest.raises(HandStoreError):
        store.deal_hole_cards("y", 1, [obs("2d"), obs("3h")])  # 2d already dealt
    with pytest.raises(HandStoreError):
        store.deal_board("y", [obs("2h"), obs("5c"), obs("6c")])  # 2h is a sealed hole card
    with pytest.raises(HandStoreError):
        store.deal_hole_cards("y", 9, [obs("4d"), obs("4h")])
    with pytest.raises(ValueError, match="duplicate hole card"):
        store.deal_hole_cards("y", 2, [obs("4d"), CardObservation("4d")])


def test_abandon_destroys_sealed_cards(live):
    store, keys = live
    store.abandon_hand("live")
    with pytest.raises(KeyError):
        store.get("live")
    with pytest.raises(KeyError):
        store.seat_view("live", keys[0])

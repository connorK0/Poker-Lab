"""P2.9: duplicate dealing in the no-limit engine, end to end with the paired evaluator."""

from __future__ import annotations

import pytest

from poker_lab.engine import CallingAgent, RandomAgent, SeatedPlayer, TableSimulator
from poker_lab.engine.duplicate import duplicate_match
from poker_lab.evaluation import DuplicateDealEvaluator, WinRateEvaluator, hand_results
from poker_lab.schema import FULL_DECK, TableConfig, validate_hand_history

TABLE = TableConfig("dup", 2, 1, 2)


def _cards(hand, seat):
    return tuple(c.card for c in hand.hole_cards_of(seat).cards)


def test_each_deal_is_replayed_with_seats_swapped():
    match = duplicate_match(TABLE, (("hero", RandomAgent), ("villain", RandomAgent)), 40, seed=3)
    for a, b in zip(match.first, match.second, strict=True):
        validate_hand_history(a)
        validate_hand_history(b)
        assert a.button_seat == b.button_seat
        # same cards in the same seats, held by the other player
        assert _cards(a, 0) == _cards(b, 0)
        assert _cards(a, 1) == _cards(b, 1)
        assert a.seats[0].player_id == "hero"
        assert b.seats[0].player_id == "villain"
        # boards come from the same run-out (one may stop earlier than the other)
        short, long = sorted((a.board_cards(), b.board_cards()), key=len)
        assert long[: len(short)] == short
    # the deals themselves differ from hand to hand
    assert len({_cards(h, 0) for h in match.first}) > 30


def test_duplicate_pairs_cancel_card_luck_exactly_for_identical_agents():
    """Two always-call agents: every pot is decided by the cards alone, so each pair of plays
    cancels exactly, while the naive per-hand estimate is pure noise."""
    match = duplicate_match(TABLE, (("a", CallingAgent), ("b", CallingAgent)), 200, seed=11)
    paired = match.paired_results("a")
    assert len(paired) == 400
    for x, y in zip(paired[::2], paired[1::2], strict=True):
        assert x.net_chips + y.net_chips == 0
    dup = DuplicateDealEvaluator().evaluate(paired)
    assert dup.bb_per_100 == 0.0
    assert dup.upper - dup.lower == 0.0
    naive = WinRateEvaluator().evaluate(hand_results(match.first, "a"))
    assert naive.upper - naive.lower > 10.0  # bb/100


def test_play_hand_rejects_a_bad_deck():
    sim = TableSimulator(
        TABLE, [SeatedPlayer(0, "a", CallingAgent()), SeatedPlayer(1, "b", CallingAgent())], seed=1
    )
    with pytest.raises(ValueError, match="52 cards"):
        sim.play_hand(deck=[*FULL_DECK[:51], FULL_DECK[0]])
    hand = sim.play_hand(deck=list(reversed(FULL_DECK)))
    assert len(hand.hole_cards) == 2

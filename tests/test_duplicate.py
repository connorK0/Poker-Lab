"""P2.9: duplicate dealing in the no-limit engine, end to end with the paired evaluator; n
players with seats rotated cyclically."""

from __future__ import annotations

import random

import numpy as np
import pytest

from poker_lab.engine import CallingAgent, RandomAgent, SeatedPlayer, TableSimulator
from poker_lab.engine.duplicate import duplicate_match
from poker_lab.evaluation import DuplicateDealEvaluator, WinRateEvaluator, hand_results
from poker_lab.schema import FULL_DECK, TableConfig, dumps_hand, validate_hand_history

TABLE = TableConfig("dup", 2, 1, 2)
TABLE6 = TableConfig("dup6", 6, 1, 2)


def _cards(hand, seat):
    return tuple(c.card for c in hand.hole_cards_of(seat).cards)


def _counting_factories(ids, calls):
    """A RandomAgent factory per id that logs its calls in ``calls``. The k-th agent built (over
    all factories) gets its own weights, so the order of the calls shows in the hands."""

    def factory(player_id):
        def make():
            calls.append(player_id)
            k = len(calls)
            return RandomAgent(fold_weight=0.5 * k, raise_weight=1.0 / k)

        return make

    return [(player_id, factory(player_id)) for player_id in ids]


def _heads_up_match_at_2131bb4(table, players, n_deals, seed, *, starting_stack=200):
    """``duplicate_match`` as it was for two players only (commit 2131bb4), rebuilt inline."""
    (id_a, make_a), (id_b, make_b) = players
    deck_seed, seed_1, seed_2 = (int(s) for s in np.random.SeedSequence(seed).generate_state(3))
    deck_rng = random.Random(deck_seed)
    first = TableSimulator(
        table,
        [SeatedPlayer(0, id_a, make_a()), SeatedPlayer(1, id_b, make_b())],
        seed=seed_1,
        starting_stack=starting_stack,
        button_seat=0,
        hand_id_prefix=f"{table.table_id}-dup1-",
    )
    second = TableSimulator(
        table,
        [SeatedPlayer(0, id_b, make_b()), SeatedPlayer(1, id_a, make_a())],
        seed=seed_2,
        starting_stack=starting_stack,
        button_seat=0,
        hand_id_prefix=f"{table.table_id}-dup2-",
    )
    hands_1, hands_2 = [], []
    for _ in range(n_deals):
        deck = list(FULL_DECK)
        deck_rng.shuffle(deck)
        hands_1.append(first.play_hand(deck=deck))
        hands_2.append(second.play_hand(deck=deck))
    return hands_1, hands_2


def _check_rotation(match, ids):
    """Every deal: same cards per seat and same button at all tables, player ``j`` in seat
    ``(j + r) % n`` at table ``r``, so each player holds each seat once; one shared run-out."""
    n = len(ids)
    assert len(match.tables) == n
    for i, deal in enumerate(zip(*match.tables, strict=True)):
        for r, hand in enumerate(deal):
            validate_hand_history(hand)
            assert len(hand.seats) == n
            assert hand.button_seat == i % n  # seat 0 first, then one seat per deal
            assert all(_cards(hand, s) == _cards(deal[0], s) for s in range(n))
            assert [hand.seat_of(p) for p in ids] == [(j + r) % n for j in range(n)]
        for p in ids:
            assert sorted(hand.seat_of(p) for hand in deal) == list(range(n))
        boards = sorted((hand.board_cards() for hand in deal), key=len)
        assert all(boards[-1][: len(board)] == board for board in boards)


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


def test_two_players_give_exactly_the_hands_of_the_heads_up_algorithm():
    """n = 2 reproduces commit 2131bb4 hand for hand, including the order the factories are
    called in (the Phase 3 bridge's hero factory draws a seed per call)."""
    old_calls, new_calls = [], []
    old = _heads_up_match_at_2131bb4(
        TABLE, _counting_factories(("hero", "villain"), old_calls), 60, 7, starting_stack=150
    )
    new = duplicate_match(
        TABLE, _counting_factories(("hero", "villain"), new_calls), 60, 7, starting_stack=150
    )
    assert new_calls == old_calls == ["hero", "villain", "villain", "hero"]
    assert len(new.tables) == 2
    assert new.first is new.tables[0]
    assert new.second is new.tables[1]
    for old_hands, new_hands in zip(old, new.tables, strict=True):
        assert [dumps_hand(h) for h in new_hands] == [dumps_hand(h) for h in old_hands]


def test_six_players_hold_every_seat_once_per_deal():
    ids = [f"p{j}" for j in range(6)]
    match = duplicate_match(TABLE6, [(p, RandomAgent) for p in ids], 20, seed=4)
    _check_rotation(match, ids)
    assert match.tables[3][0].hand_id == "dup6-dup4-000000"
    # the deals themselves differ from hand to hand
    assert len({_cards(h, 0) for h in match.first}) > 15


def test_rotation_cancels_card_luck_exactly_for_six_identical_agents():
    """Six always-call agents: each seat's result is the same at all six tables, and each player
    holds each seat once, so every player's results over a deal's six tables sum to zero."""
    ids = [f"p{j}" for j in range(6)]
    match = duplicate_match(TABLE6, [(p, CallingAgent) for p in ids], 100, seed=11)
    for deal in zip(*match.tables, strict=True):
        assert all(h.net_results() == deal[0].net_results() for h in deal)
    for p in ids:
        grouped = match.grouped_results(p)
        assert len(grouped) == 600
        for g in range(0, 600, 6):
            assert sum(r.net_chips for r in grouped[g : g + 6]) == 0
    dup = DuplicateDealEvaluator(group_size=6).evaluate(match.grouped_results("p0"))
    assert dup.bb_per_100 == 0.0
    assert dup.upper - dup.lower == 0.0
    assert dup.n_hands == 600
    assert match.evaluate("p0") == dup
    naive = WinRateEvaluator().evaluate(hand_results(match.first, "p0"))
    assert naive.upper - naive.lower > 10.0  # bb/100


def test_odd_chips_go_by_position_so_three_callers_cancel_exactly_with_antes():
    """Three always-call agents with an ante of 1: 9-chip pots, so a two-way split leaves an
    odd chip. It goes by position, so each seat's result is still the same at every table."""
    table = TableConfig("dup3", 6, 1, 2, ante=1)
    ids = ("a", "b", "c")
    match = duplicate_match(table, [(p, CallingAgent) for p in ids], 400, seed=8)
    odd_splits = 0
    for deal in zip(*match.tables, strict=True):
        assert all(h.net_results() == deal[0].net_results() for h in deal)
        hand = deal[0]
        winners = {award.seat for award in hand.payouts}
        pot = sum(award.amount for award in hand.payouts)
        odd_splits += len(winners) > 1 and pot % len(winners) != 0
    assert odd_splits > 0
    for p in ids:
        grouped = match.grouped_results(p)
        assert all(sum(r.net_chips for r in grouped[g : g + 3]) == 0 for g in range(0, 1200, 3))
    assert match.evaluate("a").upper - match.evaluate("a").lower == 0.0


def test_three_players_at_a_six_seat_table():
    ids = ("a", "b", "c")
    match = duplicate_match(TABLE6, [(p, RandomAgent) for p in ids], 30, seed=5)
    _check_rotation(match, ids)


def test_grouped_results_go_deal_by_deal_and_paired_results_need_two_tables():
    ids = ("a", "b", "c")
    match = duplicate_match(TABLE6, [(p, RandomAgent) for p in ids], 4, seed=9)
    grouped = match.grouped_results("b")
    assert [r.hand_id for r in grouped] == [
        f"dup6-dup{r + 1}-{i:06d}" for i in range(4) for r in range(3)
    ]
    assert [r.net_chips for r in grouped] == [
        match.tables[r][i].net_results()[match.tables[r][i].seat_of("b")]
        for i in range(4)
        for r in range(3)
    ]
    with pytest.raises(ValueError, match="did not play every hand"):
        match.grouped_results("nobody")
    with pytest.raises(ValueError, match="two tables"):
        match.paired_results("b")
    heads_up = duplicate_match(TABLE, (("a", RandomAgent), ("b", RandomAgent)), 4, seed=9)
    assert heads_up.paired_results("a") == heads_up.grouped_results("a")


def test_duplicate_match_rejects_bad_players_before_building_agents():
    calls = []
    with pytest.raises(ValueError, match="at least two players"):
        duplicate_match(TABLE, _counting_factories(("a",), calls), 1, seed=1)
    with pytest.raises(ValueError, match="3 players do not fit at a 2-seat table"):
        duplicate_match(TABLE, _counting_factories(("a", "b", "c"), calls), 1, seed=1)
    with pytest.raises(ValueError, match="distinct"):
        duplicate_match(TABLE6, _counting_factories(("a", "b", "a"), calls), 1, seed=1)
    assert calls == []


def test_play_hand_rejects_a_bad_deck():
    sim = TableSimulator(
        TABLE, [SeatedPlayer(0, "a", CallingAgent()), SeatedPlayer(1, "b", CallingAgent())], seed=1
    )
    with pytest.raises(ValueError, match="52 cards"):
        sim.play_hand(deck=[*FULL_DECK[:51], FULL_DECK[0]])
    hand = sim.play_hand(deck=list(reversed(FULL_DECK)))
    assert len(hand.hole_cards) == 2

"""All-in EV: a hand's result with the luck of the remaining cards taken out.

When two players are all in before the river, the cards still to come decide who wins, but not
how well anyone played. All-in EV replaces the actual result by its expectation over those
cards. With pot ``P`` (everything in the middle once uncalled chips are returned), your
contribution ``c`` and your equity ``e`` (win probability, ties counted half) given both hands
and the board at the moment of the all-in::

    all-in EV = e * P - c        actual = (what you won) - c

Equity is exact from the flop on (990 runouts from the flop, 44 from the turn). Preflop there
are 1,712,304 boards, so it is a seeded Monte Carlo average over ``PREFLOP_SAMPLES`` boards
(standard error at most 0.5 / sqrt(20,000) = 0.0035 of the pot, 0.14 bb in a 40 bb pot); the
seed is the hand id, so a report reproduces exactly.

Only heads-up all-ins are adjusted. Hands without an all-in, all-ins on the river and pots with
three or more players keep their actual result.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass

from poker_lab.engine.fast_hand import phe_rank
from poker_lab.schema import ActionType, HandHistory, Street
from poker_lab.schema.cards import FULL_DECK, card_to_id

PREFLOP_SAMPLES = 20_000


@dataclass(frozen=True)
class AllInResult:
    actual: int  # chips
    ev: float  # chips: all-in EV (equals ``actual`` when not adjusted)
    adjusted: bool
    equity: float | None = None
    method: str = ""  # "exact" or "monte_carlo"


def equity(
    mine: tuple[str, str], theirs: tuple[str, str], board: tuple[str, ...], seed: str
) -> tuple[float, str]:
    """Equity of ``mine`` against ``theirs`` given ``board`` (0, 3 or 4 cards)."""
    dead = {*mine, *theirs, *board}
    deck = [card_to_id(c) for c in FULL_DECK if c not in dead]
    a = [card_to_id(c) for c in mine]
    b = [card_to_id(c) for c in theirs]
    known = [card_to_id(c) for c in board]
    need = 5 - len(board)
    if need <= 2:
        runouts = itertools.combinations(deck, need)
        method = "exact"
    else:
        rng = random.Random(
            int.from_bytes(hashlib.blake2b(seed.encode(), digest_size=8).digest(), "big")
        )
        runouts = (rng.sample(deck, need) for _ in range(PREFLOP_SAMPLES))
        method = "monte_carlo"
    score = n = 0.0
    for extra in runouts:
        full = [*known, *extra]
        ra, rb = phe_rank([*a, *full]), phe_rank([*b, *full])
        score += 1.0 if ra < rb else 0.5 if ra == rb else 0.0
        n += 1
    return score / n, method


def all_in_result(hand: HandHistory, seat: int) -> AllInResult:
    """``seat``'s actual and all-in EV result in ``hand`` (a completed hand, possibly redacted:
    when a needed hand is unknown the result is not adjusted)."""
    actual = hand.net_results()[seat]
    plain = AllInResult(actual, float(actual), False)
    decisions = [a for a in hand.actions if a.action_type.is_decision]
    folded = {a.seat for a in decisions if a.action_type is ActionType.FOLD}
    live = [s.seat for s in hand.seats if s.seat not in folded]
    if seat not in live or len(live) != 2 or not any(a.is_all_in for a in hand.actions):
        return plain
    street = decisions[-1].street if decisions else Street.PREFLOP
    if street is Street.RIVER or len(hand.board) < 5:
        return plain
    other = live[0] if live[1] == seat else live[1]
    cards = {}
    for s in (seat, other):
        hc = hand.hole_cards_of(s)
        if hc is None or hc.cards is None:
            return plain
        cards[s] = tuple(o.card for o in hc.cards)
    board = hand.board_cards()[: street.board_size]
    eq, method = equity(cards[seat], cards[other], board, hand.hand_id)
    contrib = hand.contributions()
    returned = {r.seat: r.amount for r in hand.uncalled_returns}
    pot = sum(contrib.values()) - sum(returned.values()) - hand.rake
    mine = contrib[seat] - returned.get(seat, 0)
    return AllInResult(actual, eq * pot - mine, True, eq, method)

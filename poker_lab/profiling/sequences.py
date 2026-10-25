"""Raw action sequences: every hand of every player as a short sequence of tokens (P3.8).

``hand_tokens(hand)`` turns one complete hand into, for each seated player, the player's own
decisions in the order they were made, followed by one closing token for the hand. A decision
token is the public situation before the action (``SITUATION_FEATURES``, as in
``decision_points``), the action taken (fold / check / call / bet / raise) and, for a bet or a
raise, its size: the raise over the call as a fraction of the pot after calling, and the
player's total bet on the street in big blinds. The closing token carries what the stats
counters know about the hand's outcome: saw the flop, reached showdown, won at showdown. A
hand where the player made no decision (a walk) is the closing token alone, so a player's
number of hands is the number of closing tokens.

Only public information is used (no hole cards, not even shown ones): the tokens carry what the
counters of ``stats`` see, but as raw events rather than totals with hand-written opportunity
rules, which is what a sequence model is compared on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from poker_lab.profiling.situations import SITUATION_FEATURES, decision_points
from poker_lab.profiling.stats import C, hand_counters
from poker_lab.schema import ActionType, HandHistory

ACTIONS: tuple[ActionType, ...] = (
    ActionType.FOLD,
    ActionType.CHECK,
    ActionType.CALL,
    ActionType.BET,
    ActionType.RAISE,
)
OUTCOMES = ("saw_flop", "showdown", "won_showdown")
TOKEN_FEATURES: tuple[str, ...] = (
    "end",
    *SITUATION_FEATURES,
    *(a.value for a in ACTIONS),
    "log1p_raise_pot",
    "log_total_bb",
    *OUTCOMES,
)
N_TOKEN_FEATURES = len(TOKEN_FEATURES)
_ACTION_COL = {a: 1 + len(SITUATION_FEATURES) + i for i, a in enumerate(ACTIONS)}
_SIZE_COL = 1 + len(SITUATION_FEATURES) + len(ACTIONS)
_OUTCOME_COL = _SIZE_COL + 2
_OUTCOME_COUNTERS = ("saw_flop", "wtsd", "wsd")


def _sizes(hand: HandHistory) -> list[tuple[float, float]]:
    """(log1p(raise / pot after calling), log(street total / bb)) for every decision, in the
    order of ``decision_points``; (0, 0) for folds, checks and calls."""
    n = len(hand.seats)
    idx = {s.seat: i for i, s in enumerate(hand.seats)}
    bb = hand.table.big_blind
    street = None
    bets = [0] * n
    max_bet = 0
    pot = 0
    out = []
    for a in hand.actions:
        if a.street is not street:
            street = a.street
            bets = [0] * n
            max_bet = 0
        i = idx[a.seat]
        kind = a.action_type
        if kind.is_decision:
            if kind.is_aggressive:
                call = max_bet - bets[i]
                total = bets[i] + a.amount
                raise_pot = max(total - max_bet, 0) / max(pot + call, 1)
                out.append((math.log1p(raise_pot), math.log(total / bb)))
            else:
                out.append((0.0, 0.0))
        if kind is not ActionType.POST_ANTE:
            bets[i] += a.amount
            max_bet = max(max_bet, bets[i])
        pot += a.amount
    return out


def hand_tokens(hand: HandHistory) -> list[tuple[str, list[tuple[float, ...]]]]:
    """(player id, tokens) for every seated player of a complete hand, in seat order."""
    points = decision_points(hand)
    sizes = _sizes(hand)
    kinds = [a.action_type for a in hand.actions if a.action_type.is_decision]
    if not len(points) == len(sizes) == len(kinds):
        raise AssertionError("decision points and actions are out of step")
    tokens: dict[str, list[tuple[float, ...]]] = {s.player_id: [] for s in hand.seats}
    for (pid, feats, _), kind, size in zip(points, kinds, sizes, strict=True):
        row = [0.0] * N_TOKEN_FEATURES
        row[1 : 1 + len(feats)] = feats
        row[_ACTION_COL[kind]] = 1.0
        row[_SIZE_COL], row[_SIZE_COL + 1] = size
        tokens[pid].append(tuple(row))
    for pid, counters in hand_counters(hand):
        row = [0.0] * N_TOKEN_FEATURES
        row[0] = 1.0
        for j, name in enumerate(_OUTCOME_COUNTERS):
            row[_OUTCOME_COL + j] = float(counters[C[name]] > 0)
        tokens[pid].append(tuple(row))
    return list(tokens.items())


@dataclass(frozen=True)
class SequenceRowExtractor:
    """League extractor: every token of every hand, as rows ``(population index, seq,
    *token)``. A player's rows within a hand keep their order (decisions, then the closing
    token)."""

    def __call__(self, hand: HandHistory, pop_index: dict[str, int], seq: int) -> list[Any]:
        return [(pop_index[pid], seq, *t) for pid, toks in hand_tokens(hand) for t in toks]

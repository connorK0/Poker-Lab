"""Decision points: the public situation at each voluntary action, plus what was chosen.

``decision_points(hand)`` replays a complete hand's public action log and, for every decision
(fold / check / call / bet / raise), emits the situation features known *before* the action
and the label: 0 = fold, 1 = check/call, 2 = bet/raise. Only public information is used (no
hole cards, not even shown ones), so the same extractor works on real hand histories (Phase 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from poker_lab.schema import ActionType, HandHistory, Street, seat_order_from_button

SITUATION_FEATURES: tuple[str, ...] = (
    "street",
    "facing_bet",
    "pot_odds",
    "log_pot_bb",
    "spr",
    "n_active",
    "n_raises_street",
    "pos_postflop",
    "is_blind",
    "unopened",
    "limpers",
    "is_pfa",
    "cbet_spot",
    "facing_cbet",
    "checked_to",
    "acted_this_street",
)
ACTION_CLASSES = ("fold", "check_call", "bet_raise")
_LABEL = {
    ActionType.FOLD: 0,
    ActionType.CHECK: 1,
    ActionType.CALL: 1,
    ActionType.BET: 2,
    ActionType.RAISE: 2,
}


def decision_points(hand: HandHistory) -> list[tuple[str, tuple[float, ...], int]]:
    """(player_id, situation features, label) for every decision in a complete hand."""
    seats = [s.seat for s in hand.seats]
    n = len(seats)
    idx = {seat: i for i, seat in enumerate(seats)}
    player = [s.player_id for s in hand.seats]
    stacks = [s.starting_stack for s in hand.seats]
    order = seat_order_from_button(seats, hand.button_seat)
    pos = {seat: (i / (n - 1) if n > 1 else 0.0) for i, seat in enumerate(order)}
    blinds = set(order[:2]) if n > 2 else {order[0], order[1]}
    bb = hand.table.big_blind

    folded = [False] * n
    street = Street.PREFLOP
    bets = [0] * n
    max_bet = 0
    pot = 0
    aggr_street = 0
    checks_street = 0
    acted = [False] * n
    limpers = 0
    pf_raises = 0
    last_raiser: int | None = None
    flop_aggr = 0
    cbet_made = False
    pfa_acted_flop = False
    out: list[tuple[str, tuple[float, ...], int]] = []

    for a in hand.actions:
        if a.street is not street:
            street = a.street
            bets = [0] * n
            max_bet = 0
            aggr_street = 0
            checks_street = 0
            acted = [False] * n
        i = idx[a.seat]
        kind = a.action_type
        if kind.is_forced:
            if kind is not ActionType.POST_ANTE:
                bets[i] += a.amount
                max_bet = max(max_bet, bets[i])
            pot += a.amount
            stacks[i] -= a.amount
            continue

        to_call = min(max_bet - bets[i], stacks[i])
        facing = to_call > 0
        preflop = street is Street.PREFLOP
        is_pfa = (not preflop) and last_raiser == i
        cbet_spot = street is Street.FLOP and is_pfa and flop_aggr == 0 and not pfa_acted_flop
        facing_cbet = (
            street is Street.FLOP and cbet_made and flop_aggr == 1 and not is_pfa and facing
        )
        feats = (
            float(street.index),
            float(facing),
            to_call / (pot + to_call) if facing else 0.0,
            math.log(max(pot, 1) / bb),
            min(stacks[i] / max(pot, 1), 25.0),
            float(n - sum(folded)),
            float(min(aggr_street, 3)),
            pos[a.seat],
            float(a.seat in blinds),
            float(preflop and pf_raises == 0),
            float(min(limpers, 3)) if preflop else 0.0,
            float(is_pfa),
            float(cbet_spot),
            float(facing_cbet),
            float((not preflop) and (not facing) and checks_street > 0),
            float(acted[i]),
        )
        out.append((player[i], feats, _LABEL[kind]))

        # update public state with the action
        if street is Street.FLOP and last_raiser == i:
            pfa_acted_flop = True
        if kind is ActionType.FOLD:
            folded[i] = True
        elif kind is ActionType.CHECK:
            checks_street += 1
        elif kind is ActionType.CALL and preflop and pf_raises == 0:
            limpers += 1
        if kind.is_aggressive:
            aggr_street += 1
            if preflop:
                pf_raises += 1
                last_raiser = i
            elif street is Street.FLOP:
                if flop_aggr == 0 and last_raiser == i:
                    cbet_made = True
                flop_aggr += 1
        acted[i] = True
        bets[i] += a.amount
        max_bet = max(max_bet, bets[i])
        pot += a.amount
        stacks[i] -= a.amount
    return out


@dataclass(frozen=True)
class DecisionRowExtractor:
    """League extractor: decision rows for hands whose sequence number falls in ``windows``.

    Each row is ``(population index, seq, *situation features, label)``.
    """

    windows: tuple[tuple[int, int], ...]

    def __call__(self, hand: HandHistory, pop_index: dict[str, int], seq: int) -> list[Any]:
        if not any(lo <= seq < hi for lo, hi in self.windows):
            return []
        return [(pop_index[pid], seq, *feats, label) for pid, feats, label in decision_points(hand)]

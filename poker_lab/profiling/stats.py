"""Opponent statistics from hand histories, with explicit opportunity counting.

``hand_counters`` makes one pass over a complete ``HandHistory`` and returns, for every seated
player, a vector of integer counters (``COUNTERS``). Counters add up across hands, so a
player's profile after ``n`` hands is a sum, and cumulative sums give profiles at every sample
size at once. Only public information is used (actions, board, showdown participation, pot
awards); hole cards are never read.

Definitions (numerator / denominator, each counted at most once per hand unless noted):

vpip          called or raised preflop / had a preflop decision (walks excluded)
pfr           raised preflop / had a preflop decision
three_bet     re-raised when first facing exactly one raise / that spot, with chips behind
              beyond the call
fold_to_3bet  folded / opened (first raise) and then faced exactly one re-raise
cbet          bet the flop / was the last preflop raiser and made the first flop decision
              with no bet in front
fold_to_cbet  folded / faced a flop c-bet with no raise after it
af            postflop bets+raises / postflop calls (ratio; shrunk via the aggression share
              (bets+raises)/(bets+raises+calls), which is binomial)
fold_to_bet   folded / postflop decisions facing a bet (counted per decision)
wtsd          reached showdown / saw the flop
wsd           won a pot at showdown / reached showdown (after seeing the flop)
bet_size      mean postflop bet as a fraction of the pot (continuous; sum and count)
open_size     mean open-raise size in big blinds (continuous; sum and count)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from poker_lab.schema import ActionType, HandHistory, Street

COUNTERS: tuple[str, ...] = (
    "hands",
    "vpip_opp",
    "vpip",
    "pfr_opp",
    "pfr",
    "three_bet_opp",
    "three_bet",
    "fold_to_3bet_opp",
    "fold_to_3bet",
    "cbet_opp",
    "cbet",
    "fold_to_cbet_opp",
    "fold_to_cbet",
    "pf_aggr",
    "pf_calls",
    "fold_to_bet_opp",
    "fold_to_bet",
    "saw_flop",
    "wtsd",
    "wsd",
    "bet_size_x100",
    "bet_size_n",
    "open_size_x100",
    "open_size_n",
)
C = {name: i for i, name in enumerate(COUNTERS)}
N_COUNTERS = len(COUNTERS)


@dataclass(frozen=True)
class BinomialStat:
    name: str
    success: str
    trials: str
    description: str


#: Binomial stats: (successes, opportunities) counter pairs. Order is the feature order.
BINOMIAL_STATS: tuple[BinomialStat, ...] = (
    BinomialStat("vpip", "vpip", "vpip_opp", "voluntarily put money in preflop"),
    BinomialStat("pfr", "pfr", "pfr_opp", "raised preflop"),
    BinomialStat("three_bet", "three_bet", "three_bet_opp", "3-bet when facing an open"),
    BinomialStat("fold_to_3bet", "fold_to_3bet", "fold_to_3bet_opp", "opener folded to a 3-bet"),
    BinomialStat("cbet", "cbet", "cbet_opp", "continuation bet the flop"),
    BinomialStat("fold_to_cbet", "fold_to_cbet", "fold_to_cbet_opp", "folded to a flop c-bet"),
    BinomialStat("aggr_share", "pf_aggr", "_aggr_trials", "postflop bets+raises share"),
    BinomialStat("fold_to_bet", "fold_to_bet", "fold_to_bet_opp", "postflop fold facing a bet"),
    BinomialStat("wtsd", "wtsd", "saw_flop", "went to showdown after seeing the flop"),
    BinomialStat("wsd", "wsd", "wtsd", "won money at showdown"),
)
REQUIRED_STATS = ("vpip", "pfr", "three_bet", "af", "cbet", "fold_to_cbet", "wtsd", "wsd")


def hand_counters(hand: HandHistory) -> list[tuple[str, list[int]]]:
    """Per-player counter vectors for one complete hand, in seat order."""
    if not hand.is_complete:
        raise ValueError("stats are computed from complete hands only")
    seats = [s.seat for s in hand.seats]
    n = len(seats)
    idx = {seat: i for i, seat in enumerate(seats)}
    stacks = [s.starting_stack for s in hand.seats]
    counts = [[0] * N_COUNTERS for _ in range(n)]
    for row in counts:
        row[C["hands"]] = 1

    bb = hand.table.big_blind
    street = Street.PREFLOP
    street_bets = [0] * n
    max_bet = 0
    pot = 0
    fold_street: list[Street | None] = [None] * n
    pf_decided = [False] * n
    put_in_vol = [False] * n
    raised_pf = [False] * n
    n_raises = 0
    first_raiser: int | None = None
    last_raiser: int | None = None
    three_bet_seen = [False] * n
    fold3_counted = False
    flop_aggr = 0
    pfa_acted_flop = False
    cbet_made = False
    faced_cbet = [False] * n

    for a in hand.actions:
        if a.street is not street:
            street = a.street
            street_bets = [0] * n
            max_bet = 0
        i = idx[a.seat]
        kind = a.action_type
        row = counts[i]
        if kind is ActionType.POST_ANTE:
            pot += a.amount
            stacks[i] -= a.amount
            continue
        if kind.is_forced:
            street_bets[i] += a.amount
            max_bet = max(max_bet, street_bets[i])
            pot += a.amount
            stacks[i] -= a.amount
            continue
        to_call = max_bet - street_bets[i]

        if street is Street.PREFLOP:
            if not pf_decided[i]:
                pf_decided[i] = True
                row[C["vpip_opp"]] += 1
                row[C["pfr_opp"]] += 1
            if n_raises == 1 and i != first_raiser and not three_bet_seen[i]:
                three_bet_seen[i] = True
                if stacks[i] > to_call:
                    row[C["three_bet_opp"]] += 1
                    row[C["three_bet"]] += kind is ActionType.RAISE
            if n_raises == 2 and i == first_raiser and not fold3_counted:
                fold3_counted = True
                row[C["fold_to_3bet_opp"]] += 1
                row[C["fold_to_3bet"]] += kind is ActionType.FOLD
            if kind is ActionType.CALL or kind is ActionType.RAISE:
                put_in_vol[i] = True
            if kind is ActionType.RAISE:
                if n_raises == 0:
                    row[C["open_size_x100"]] += round(100 * a.total_bet / bb)
                    row[C["open_size_n"]] += 1
                    first_raiser = i
                raised_pf[i] = True
                n_raises += 1
                last_raiser = i
        else:
            if to_call > 0:
                row[C["fold_to_bet_opp"]] += 1
                row[C["fold_to_bet"]] += kind is ActionType.FOLD
            if kind is ActionType.BET or kind is ActionType.RAISE:
                row[C["pf_aggr"]] += 1
            elif kind is ActionType.CALL:
                row[C["pf_calls"]] += 1
            if kind is ActionType.BET and pot > 0:
                row[C["bet_size_x100"]] += round(100 * a.amount / pot)
                row[C["bet_size_n"]] += 1
            if street is Street.FLOP:
                if cbet_made and flop_aggr == 1 and i != last_raiser and not faced_cbet[i]:
                    faced_cbet[i] = True
                    row[C["fold_to_cbet_opp"]] += 1
                    row[C["fold_to_cbet"]] += kind is ActionType.FOLD
                if i == last_raiser and not pfa_acted_flop:
                    pfa_acted_flop = True
                    if flop_aggr == 0:
                        row[C["cbet_opp"]] += 1
                        if kind is ActionType.BET:
                            row[C["cbet"]] += 1
                            cbet_made = True
                if kind is ActionType.BET or kind is ActionType.RAISE:
                    flop_aggr += 1

        if kind is ActionType.FOLD:
            fold_street[i] = street
        street_bets[i] += a.amount
        max_bet = max(max_bet, street_bets[i])
        pot += a.amount
        stacks[i] -= a.amount

    saw_flop_board = len(hand.board) >= 3
    showdown = {e.seat for e in hand.showdown}
    winners = {p.seat for p in hand.payouts}
    out = []
    for i, info in enumerate(hand.seats):
        row = counts[i]
        row[C["vpip"]] = int(put_in_vol[i])
        row[C["pfr"]] = int(raised_pf[i])
        saw = saw_flop_board and fold_street[i] is not Street.PREFLOP
        row[C["saw_flop"]] = int(saw)
        reached = saw and info.seat in showdown
        row[C["wtsd"]] = int(reached)
        row[C["wsd"]] = int(reached and info.seat in winners)
        out.append((info.player_id, row))
    return out


def counters_by_player(hands: Iterable[HandHistory]) -> dict[str, np.ndarray]:
    """Total counters per player over many hands."""
    totals: dict[str, np.ndarray] = {}
    for hand in hands:
        for pid, row in hand_counters(hand):
            if pid in totals:
                totals[pid] += row
            else:
                totals[pid] = np.array(row, dtype=np.int64)
    return totals


def _col(counts: np.ndarray, name: str) -> np.ndarray:
    if name == "_aggr_trials":
        return counts[..., C["pf_aggr"]] + counts[..., C["pf_calls"]]
    return counts[..., C[name]]


def binomial_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(successes, trials) arrays of shape (..., len(BINOMIAL_STATS))."""
    k = np.stack([_col(counts, s.success) for s in BINOMIAL_STATS], axis=-1)
    m = np.stack([_col(counts, s.trials) for s in BINOMIAL_STATS], axis=-1)
    return k, m


def raw_stats(counts: np.ndarray | Sequence[int]) -> dict[str, float]:
    """Plain (unshrunk) stats from one counter vector; NaN where there were no opportunities."""
    c = np.asarray(counts, dtype=float)
    out: dict[str, float] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in BINOMIAL_STATS:
            out[s.name] = float(_col(c, s.success) / _col(c, s.trials))
        out["af"] = float(c[C["pf_aggr"]] / c[C["pf_calls"]]) if c[C["pf_calls"]] else float("nan")
        out["bet_size"] = float(c[C["bet_size_x100"]] / 100 / c[C["bet_size_n"]])
        out["open_size"] = float(c[C["open_size_x100"]] / 100 / c[C["open_size_n"]])
    out["hands"] = float(c[C["hands"]])
    return out

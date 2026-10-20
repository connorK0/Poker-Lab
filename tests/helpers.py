"""Shared test helpers: hand-built hands and an object-graph reachability walker."""

from __future__ import annotations

import gc
import types
from collections.abc import Iterator

from poker_lab.schema import (
    Action,
    ActionType,
    CardObservation,
    ChipAward,
    HandHistory,
    HandHistoryStore,
    SeatInfo,
    SeatKey,
    ShowdownEntry,
    Street,
    TableConfig,
    assign_positions,
    compute_uncalled_returns,
    is_valid_card,
)

TABLE = TableConfig(table_id="t1", max_seats=6, small_blind=1, big_blind=2)
PLAYERS = ("alice", "bob", "carol", "dave", "erin", "frank")
STACK = 200

# Seat -> hole cards for the reference hand. Button on seat 5: SB=0, BB=1, UTG=2, HJ=3, CO=4.
HOLE = {
    0: ("3s", "4s"),
    1: ("8c", "8s"),
    2: ("Jc", "Tc"),
    3: ("Ac", "Qd"),
    4: ("5h", "6h"),
    5: ("Kd", "Kh"),
}
BOARD = ("Ah", "7d", "2c", "Ks", "9h")


def seats(button: int = 5) -> tuple[SeatInfo, ...]:
    pos = assign_positions(range(6), button)
    return tuple(SeatInfo(i, PLAYERS[i], STACK, pos[i]) for i in range(6))


def obs(card: str, source: str = "simulator", confidence: float = 1.0) -> CardObservation:
    return CardObservation(card, source, confidence)  # type: ignore[arg-type]


def act(
    seat: int,
    street: Street,
    kind: ActionType,
    amount: int = 0,
    total: int = 0,
    all_in: bool = False,
) -> Action:
    return Action(seat, street, kind, amount, total, all_in)


P, F, T, R = Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER
A = ActionType


def record_reference_hand(
    store: HandHistoryStore, hand_id: str = "h1", *, complete: bool = True
) -> tuple[HandHistory | None, dict[int, SeatKey]]:
    """Record a full 6-max hand through the store. Returns (completed hand or None, keys).

    Preflop: blinds 1/2, UTG folds, HJ raises to 6, CO folds, BTN calls, SB folds, BB calls.
    Flop Ah7d2c: BB checks, HJ c-bets 10, BTN calls, BB folds.
    Turn Ks: HJ checks, BTN bets 20, HJ calls.  River 9h: check, check.
    Showdown: HJ shows AcQd, BTN shows KdKh (trips) and wins the 79-chip pot.
    """
    store.begin_hand(hand_id, TABLE, 5, seats(5))
    for a in (act(0, P, A.POST_SMALL_BLIND, 1, 1), act(1, P, A.POST_BIG_BLIND, 2, 2)):
        store.record_action(hand_id, a)
    keys = {s: store.deal_hole_cards(hand_id, s, [obs(c) for c in HOLE[s]]) for s in range(6)}
    for a in (
        act(2, P, A.FOLD),
        act(3, P, A.RAISE, 6, 6),
        act(4, P, A.FOLD),
        act(5, P, A.CALL, 6, 6),
        act(0, P, A.FOLD, 0, 1),
        act(1, P, A.CALL, 4, 6),
    ):
        store.record_action(hand_id, a)
    store.deal_board(hand_id, [obs(c) for c in BOARD[:3]])
    for a in (
        act(1, F, A.CHECK),
        act(3, F, A.BET, 10, 10),
        act(5, F, A.CALL, 10, 10),
        act(1, F, A.FOLD),
    ):
        store.record_action(hand_id, a)
    store.deal_board(hand_id, [obs(BOARD[3])])
    for a in (act(3, T, A.CHECK), act(5, T, A.BET, 20, 20), act(3, T, A.CALL, 20, 20)):
        store.record_action(hand_id, a)
    store.deal_board(hand_id, [obs(BOARD[4])])
    for a in (act(3, R, A.CHECK), act(5, R, A.CHECK)):
        store.record_action(hand_id, a)
    if not complete:
        return None, keys
    hand = store.complete_hand(
        hand_id,
        showdown=[ShowdownEntry(3, True), ShowdownEntry(5, True)],
        payouts=[ChipAward(5, 79)],
    )
    return hand, keys


SCRIPT_BOARD = ("2c", "7d", "Jh", "4s", "9c")
FOLD, CHECK, CALL, BET, RAISE = "fold", "check", "call", "bet", "raise"


def build_scripted_hand(
    script: dict[Street, list[tuple]], *, showdown=(), winners=None, hand_id: str = "x"
) -> HandHistory:
    """Record a 6-max hand (button seat 5, blinds 1/2, 200-chip stacks) from a compact script.

    ``script[street]`` lists ``(seat, kind, to_amount)`` with kind one of FOLD/CHECK/CALL/BET/
    RAISE; amounts and all-in flags are derived. The pot goes to ``winners`` (default: the first
    seat that didn't fold).
    """
    store = HandHistoryStore()
    store.begin_hand(hand_id, TABLE, 5, seats(5))
    store.record_action(hand_id, Action(0, Street.PREFLOP, ActionType.POST_SMALL_BLIND, 1, 1))
    store.record_action(hand_id, Action(1, Street.PREFLOP, ActionType.POST_BIG_BLIND, 2, 2))
    board_size = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}
    dealt = 0
    folded: set[int] = set()
    stacks = dict.fromkeys(range(6), STACK)
    stacks[0], stacks[1] = STACK - 1, STACK - 2
    for street, actions in script.items():
        if street in board_size:
            n = board_size[street]
            store.deal_board(hand_id, [obs(c) for c in SCRIPT_BOARD[dealt:n]])
            dealt = n
        bets = dict.fromkeys(range(6), 0)
        if street is Street.PREFLOP:
            bets[0], bets[1] = 1, 2
        for seat, kind, *to in actions:
            current, top = bets[seat], max(bets.values())
            if kind in (FOLD, CHECK):
                amount = 0
            elif kind == CALL:
                amount = top - current
            else:
                amount = to[0] - current
            bets[seat] += amount
            stacks[seat] -= amount
            if kind == FOLD:
                folded.add(seat)
            all_in = amount > 0 and stacks[seat] == 0
            store.record_action(
                hand_id, Action(seat, street, ActionType(kind), amount, bets[seat], all_in)
            )
    if dealt < 5 and showdown:
        store.deal_board(hand_id, [obs(c) for c in SCRIPT_BOARD[dealt:5]])
    live = store.get(hand_id)
    uncalled = compute_uncalled_returns(live.actions)
    pot = sum(live.contributions().values()) - sum(uncalled.values())
    if winners is None:
        winners = [s for s in range(6) if s not in folded][:1]
    share = pot // len(winners)
    payouts = [
        ChipAward(w, share + (pot - share * len(winners) if i == 0 else 0))
        for i, w in enumerate(winners)
    ]
    return store.complete_hand(
        hand_id,
        showdown=[ShowdownEntry(s, True) for s in showdown],
        payouts=payouts,
        uncalled_returns=[ChipAward(s, a) for s, a in uncalled.items()],
    )


def walk_reachable(root: object, limit: int = 200_000) -> Iterator[object]:
    """Yield every object reachable from ``root`` through references, without descending into
    types, modules or functions (which would reach global constants such as the full deck)."""
    skip = (
        type,
        types.ModuleType,
        types.FunctionType,
        types.BuiltinFunctionType,
        types.MethodType,
        types.CodeType,
    )
    seen: set[int] = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if len(seen) > limit:
            raise RuntimeError("object graph unexpectedly large")
        if isinstance(obj, skip):
            continue
        yield obj
        stack.extend(gc.get_referents(obj))


def reachable_card_strings(root: object) -> set[str]:
    return {o for o in walk_reachable(root) if isinstance(o, str) and is_valid_card(o)}

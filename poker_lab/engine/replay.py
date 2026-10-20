"""Replay a recorded ``HandHistory`` through PokerKit.

Uses:
* differential tests: hands produced with the fast evaluator are replayed through unmodified
  ``pokerkit.NoLimitTexasHoldem``, and every action must be legal there with identical payoffs;
* Phase 4: checking that a manually entered action log is a legal hand (``allow_unknown_cards``
  lets opponents' cards be unknown until showdown).
"""

from __future__ import annotations

from dataclasses import dataclass

from pokerkit import HoleCardsShowingOrMucking, Mode, NoLimitTexasHoldem, State

from poker_lab.engine.simulator import AUTOMATIONS
from poker_lab.schema import ActionType, HandHistory, seat_order_from_button


class ReplayError(ValueError):
    """The recorded hand is not a legal sequence under PokerKit's rules."""


@dataclass(frozen=True)
class ReplayResult:
    state: State
    net: dict[int, int]  # seat -> chips won/lost according to PokerKit
    showdown: dict[int, bool]  # seat -> shown (True) / mucked (False)


def replay_in_pokerkit(
    hand: HandHistory,
    *,
    game: type[NoLimitTexasHoldem] = NoLimitTexasHoldem,
    allow_unknown_cards: bool = False,
) -> ReplayResult:
    """Replay ``hand`` and return PokerKit's view of the outcome.

    Raises ``ReplayError`` on any out-of-turn or illegal action, or an amount mismatch.
    """
    t = hand.table
    order = seat_order_from_button([s.seat for s in hand.seats], hand.button_seat)
    stacks = {s.seat: s.starting_stack for s in hand.seats}
    state = game.create_state(
        AUTOMATIONS,
        True,
        t.ante,
        (t.small_blind, t.big_blind),
        t.big_blind,
        tuple(stacks[s] for s in order),
        len(order),
        mode=Mode.TOURNAMENT,
    )
    hole = {hc.seat: hc for hc in hand.hole_cards}
    while (dealee := state.hole_dealee_index) is not None:
        hc = hole.get(order[dealee])
        if hc is None or hc.cards is None:
            if not allow_unknown_cards:
                raise ReplayError(f"hole cards of seat {order[dealee]} are unknown")
            cards = "????"
        else:
            cards = "".join(o.card for o in hc.cards)
        state.deal_hole(cards, dealee)

    board = list(hand.board_cards())
    decisions = [a for a in hand.actions if not a.action_type.is_forced]
    for i, action in enumerate(decisions):
        _advance_dealing(state, board, hand.hand_id)
        actor = state.actor_index
        if actor is None or order[actor] != action.seat:
            expected = None if actor is None else order[actor]
            raise ReplayError(f"action {i} by seat {action.seat}, but seat {expected} is to act")
        try:
            kind = action.action_type
            if kind is ActionType.FOLD:
                state.fold()
            elif kind in (ActionType.CHECK, ActionType.CALL):
                op = state.check_or_call()
                if op.amount != action.amount:
                    raise ReplayError(f"action {i}: {kind} moves {op.amount}, not {action.amount}")
            else:
                state.complete_bet_or_raise_to(action.total_bet)
        except ValueError as exc:
            if isinstance(exc, ReplayError):
                raise
            raise ReplayError(f"action {i} ({action}) is illegal: {exc}") from exc
    _advance_dealing(state, board, hand.hand_id)
    if state.status:
        raise ReplayError("the recorded actions end before the hand is over")
    if board:
        raise ReplayError(f"{len(board)} recorded board cards were never needed")
    showdown = {
        order[op.player_index]: bool(op.hole_cards)
        for op in state.operations
        if isinstance(op, HoleCardsShowingOrMucking)
    }
    net = {seat: state.payoffs[i] for i, seat in enumerate(order)}
    return ReplayResult(state=state, net=net, showdown=showdown)


def _advance_dealing(state: State, board: list[str], hand_id: str) -> None:
    """Perform pending burns/board deals using the recorded board."""
    while state.status:
        if state.card_burning_status:
            state.burn_card("??")
        elif (n := state.board_dealing_count) is not None:
            if len(board) < n:
                raise ReplayError(f"{hand_id}: board ran out of recorded cards")
            state.deal_board("".join(board[:n]))
            del board[:n]
        else:
            return

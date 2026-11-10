"""Plain JSON for the page: the hand in progress (from your ``SeatView``) and completed hands
(redacted to your seat). Nothing here sees more than you could: a view holds the public state
and your own cards, and a completed hand shows other players' cards only when they were shown."""

from __future__ import annotations

from typing import Any

from poker_lab.schema import Action, ActionType, HandHistory, SeatInfo, SeatView, Street

_VERB = {
    ActionType.POST_ANTE: "posts an ante of",
    ActionType.POST_SMALL_BLIND: "posts the small blind",
    ActionType.POST_BIG_BLIND: "posts the big blind",
    ActionType.POST_STRADDLE: "straddles",
    ActionType.FOLD: "folds",
    ActionType.CHECK: "checks",
    ActionType.CALL: "calls",
    ActionType.BET: "bets",
    ActionType.RAISE: "raises to",
}


def bb(chips: float, big_blind: int) -> float:
    return round(chips / big_blind, 2)


def fmt_bb(chips: float, big_blind: int) -> str:
    return f"{chips / big_blind:g} bb"


def _name(seat: int, seats: dict[int, SeatInfo], you: int) -> str:
    info = seats[seat]
    who = "You" if seat == you else info.player_id
    return f"{who} ({info.position})"


def action_lines(
    actions: tuple[Action, ...],
    board: tuple[str, ...],
    seats: tuple[SeatInfo, ...],
    you: int,
    big_blind: int,
) -> list[str]:
    by_seat = {s.seat: s for s in seats}
    lines: list[str] = []
    street = Street.PREFLOP
    for a in actions:
        if a.street is not street:
            street = a.street
            lines.append(f"{street.value.capitalize()}: {' '.join(board[: street.board_size])}")
        verb = _VERB[a.action_type]
        if a.action_type in (ActionType.FOLD, ActionType.CHECK):
            amount = ""
        elif a.action_type is ActionType.RAISE:
            amount = f" {fmt_bb(a.total_bet, big_blind)}"
        else:
            amount = f" {fmt_bb(a.amount, big_blind)}"
        if a.action_type in (ActionType.POST_SMALL_BLIND, ActionType.POST_BIG_BLIND):
            amount = f" ({fmt_bb(a.amount, big_blind)})"
        tail = " (all-in)" if a.is_all_in else ""
        lines.append(f"{_name(a.seat, by_seat, you)} {verb}{amount}{tail}")
    return lines


def describe_view(view: SeatView, token: int, mode: str) -> dict[str, Any]:
    """The decision waiting for you."""
    pub = view.public
    legal = view.legal
    assert legal is not None
    big = pub.table.big_blind
    seats = [
        {
            "seat": s.seat,
            "player": "You" if s.seat == view.seat else s.player_id,
            "position": s.position,
            "stack": pub.stacks[i],
            "bet": pub.street_bets[i],
            "folded": pub.folded[i],
            "you": s.seat == view.seat,
            "button": s.seat == pub.button_seat,
            "actor": s.seat == pub.actor_seat,
        }
        for i, s in enumerate(pub.seats)
    ]
    if mode == "pushfold":
        if legal.can_bet_or_raise:
            buttons = [{"action": "fold"}, {"action": "allin", "label": "Push"}]
        else:
            buttons = [{"action": "fold"}, {"action": "call", "amount": legal.call_amount}]
    else:
        buttons = [{"action": "fold"}] if legal.can_fold else []
        if legal.can_check:
            buttons.append({"action": "check"})
        else:
            buttons.append({"action": "call", "amount": legal.call_amount})
        if legal.can_bet_or_raise:
            kind = "raise" if pub.max_street_bet > 0 else "bet"
            buttons.append({"action": "raise", "label": kind.capitalize()})
            buttons.append({"action": "allin"})
    return {
        "token": token,
        "hand_id": pub.hand_id,
        "street": pub.street.value,
        "board": list(pub.board),
        "pot": pub.pot,
        "big_blind": big,
        "your_cards": list(view.hole_cards),
        "seats": seats,
        "legal": {
            "can_fold": legal.can_fold,
            "can_check": legal.can_check,
            "call_amount": legal.call_amount,
            "can_raise": legal.can_bet_or_raise,
            "min_raise_to": legal.min_raise_to,
            "max_raise_to": legal.max_raise_to,
            "street_bet": view.street_bet,
            "stack": view.stack,
        },
        "buttons": buttons,
        "actions": action_lines(pub.actions, pub.board, pub.seats, view.seat, big),
    }


def describe_hand(hand: HandHistory, you: int) -> dict[str, Any]:
    """A completed hand as you saw it (``hand`` redacted to your seat)."""
    big = hand.table.big_blind
    net = hand.net_results()
    board = hand.board_cards()
    lines = action_lines(hand.actions, board, hand.seats, you, big)
    by_seat = {s.seat: s for s in hand.seats}
    last_street = hand.actions[-1].street if hand.actions else Street.PREFLOP
    if len(board) > last_street.board_size:  # all-in: the rest of the board ran out
        lines.append(f"Board runs out: {' '.join(board)}")
    seats = []
    for s in hand.seats:
        hc = hand.hole_cards_of(s.seat)
        cards = [o.card for o in hc.cards] if hc is not None and hc.cards is not None else None
        shown = bool(hc is not None and hc.shown)
        if s.seat != you and shown and cards:
            lines.append(f"{_name(s.seat, by_seat, you)} shows {' '.join(cards)}")
        seats.append(
            {
                "seat": s.seat,
                "player": "You" if s.seat == you else s.player_id,
                "position": s.position,
                "cards": cards,
                "net": net[s.seat],
            }
        )
    for award in hand.payouts:
        lines.append(f"{_name(award.seat, by_seat, you)} wins {fmt_bb(award.amount, big)}")
    mine = hand.hole_cards_of(you)
    return {
        "hand_id": hand.hand_id,
        "board": board,
        "big_blind": big,
        "your_cards": [o.card for o in mine.cards] if mine and mine.cards else None,
        "seats": seats,
        "actions": lines,
        "net": net[you],
    }

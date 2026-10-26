"""Deep consistency checks for hand histories.

``HandHistory.__post_init__`` only checks structure. These checks replay the action log and
verify that the record is internally consistent:

* positions match the button and seat order;
* the forced posts (antes, blinds, straddles) are the ones the table's rules require, given
  each seat's stack (``expected_forced_posts``);
* actions are in street order, folded or all-in seats do not act again, ``amount`` and
  ``total_bet`` agree, stacks never go negative, and ``is_all_in`` flags are right;
* checks happen only when nothing is owed, calls match the outstanding bet (or all-in),
  bets open a street, raises exceed the current bet;
* board and known hole cards are distinct, and the board covers every street with action;
* complete hands conserve chips, with uncalled bets recomputed independently, and award pots
  only to seats that did not fold.

Betting minimums (min-bet / min-raise / reopening rules) are not re-derived here. In
simulation PokerKit enforces them and the engine tests replay hands through PokerKit.
"""

from __future__ import annotations

from collections.abc import Iterable

from poker_lab.schema.cards import assert_distinct
from poker_lab.schema.hand_history import (
    Action,
    ActionType,
    HandHistory,
    HandStatus,
    SeatInfo,
    Street,
    TableConfig,
)
from poker_lab.schema.positions import assign_positions, seat_order_from_button


class HandHistoryValidationError(ValueError):
    """Raised when a hand history is internally inconsistent."""

    def __init__(self, hand_id: str, problems: list[str]) -> None:
        self.hand_id = hand_id
        self.problems = problems
        joined = "; ".join(problems)
        super().__init__(f"hand {hand_id!r} is inconsistent: {joined}")


def compute_uncalled_returns(actions: Iterable[Action]) -> dict[int, int]:
    """Uncalled bet per seat: on each street, the excess of the largest live bet over the
    second largest is returned to the seat that made it."""
    by_street: dict[Street, dict[int, int]] = {}
    for action in actions:
        if action.action_type is ActionType.POST_ANTE:
            continue
        bets = by_street.setdefault(action.street, {})
        bets[action.seat] = bets.get(action.seat, 0) + action.amount
    returns: dict[int, int] = {}
    for bets in by_street.values():
        ranked = sorted(bets.items(), key=lambda kv: kv[1], reverse=True)
        top_seat, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0
        if top > second:
            returns[top_seat] = returns.get(top_seat, 0) + (top - second)
    return returns


_BLIND_KINDS = (ActionType.POST_SMALL_BLIND, ActionType.POST_BIG_BLIND)


def expected_forced_posts(
    table: TableConfig, seats: Iterable[SeatInfo], button_seat: int
) -> list[tuple[int, ActionType, int]]:
    """(seat, kind, amount) of every forced post the table's rules require, in dealing order.

    Dealing-order position ``k`` posts ``table.blinds_or_straddles[k]`` (heads-up the button
    posts the small blind) and the ante, plus ``table.big_blind_ante`` for the big blind. A
    short stack posts the ante first and then what is left of the blind, as PokerKit does.
    """
    stacks = {s.seat: s.starting_stack for s in seats}
    order = seat_order_from_button(stacks, button_seat)
    blinds = table.blinds_or_straddles
    posts: list[tuple[int, ActionType, int]] = []
    for i, seat in enumerate(order):
        role = (1 - i) if len(order) == 2 else i  # heads-up: the BB is dealt first
        ante = min(table.ante + (table.big_blind_ante if role == 1 else 0), stacks[seat])
        if ante:
            posts.append((seat, ActionType.POST_ANTE, ante))
        blind = min(blinds[role] if role < len(blinds) else 0, stacks[seat] - ante)
        if blind:
            kind = _BLIND_KINDS[role] if role < 2 else ActionType.POST_STRADDLE
            posts.append((seat, kind, blind))
    return posts


def _check_forced_posts(hand: HandHistory, problems: list[str]) -> None:
    if len(hand.table.straddles) > len(hand.seats) - 2:
        problems.append(f"{len(hand.table.straddles)} straddles with {len(hand.seats)} players")
        return
    rules = expected_forced_posts(hand.table, hand.seats, hand.button_seat)
    expected = sorted((seat, kind.value, amount) for seat, kind, amount in rules)
    recorded = sorted(
        (a.seat, a.action_type.value, a.amount) for a in hand.actions if a.action_type.is_forced
    )
    if recorded != expected:
        problems.append(f"forced posts {recorded} != the table's rules {expected}")


def _check_actions(hand: HandHistory, problems: list[str]) -> tuple[set[int], dict[int, int]]:
    """Replay actions; return (folded seats, chips remaining per seat)."""
    stacks = {s.seat: s.starting_stack for s in hand.seats}
    folded: set[int] = set()
    all_in: set[int] = set()
    street = Street.PREFLOP
    street_bets: dict[int, int] = dict.fromkeys(stacks, 0)
    seen_decision = False
    for i, action in enumerate(hand.actions):
        where = f"action {i} ({action.seat} {action.action_type})"
        if action.seat not in stacks:
            problems.append(f"{where}: seat not in hand")
            continue
        if action.street.index < street.index:
            problems.append(f"{where}: street goes backwards")
        elif action.street is not street:
            street = action.street
            street_bets = dict.fromkeys(stacks, 0)
        kind = action.action_type
        if kind.is_forced:
            if seen_decision:
                problems.append(f"{where}: forced post after a decision")
        else:
            seen_decision = True
        if action.seat in folded:
            problems.append(f"{where}: seat already folded")
        if action.seat in all_in:
            problems.append(f"{where}: seat already all-in")
        max_bet = max(street_bets.values())
        current = street_bets[action.seat]
        if kind is ActionType.CHECK and current < max_bet:
            problems.append(f"{where}: check while facing a bet")
        if kind is ActionType.BET and max_bet > 0:
            problems.append(f"{where}: bet when a bet is already open (should be raise)")
        if kind is ActionType.RAISE:
            if max_bet == 0:
                problems.append(f"{where}: raise with no open bet (should be bet)")
            elif current + action.amount <= max_bet:
                problems.append(f"{where}: raise does not exceed the current bet")
        if kind is ActionType.CALL:
            if current >= max_bet:
                problems.append(f"{where}: call with nothing to call")
            elif current + action.amount > max_bet:
                problems.append(f"{where}: call puts in more than the outstanding bet")
            elif current + action.amount < max_bet and action.amount != stacks[action.seat]:
                problems.append(f"{where}: short call that is not all-in")
        if kind is not ActionType.POST_ANTE:
            if action.total_bet != current + action.amount:
                problems.append(
                    f"{where}: total_bet {action.total_bet} != {current} + {action.amount}"
                )
            street_bets[action.seat] = current + action.amount
        stacks[action.seat] -= action.amount
        if stacks[action.seat] < 0:
            problems.append(f"{where}: stack goes negative")
        if action.amount > 0 and action.is_all_in != (stacks[action.seat] == 0):
            problems.append(f"{where}: is_all_in flag inconsistent with remaining stack")
        if kind is ActionType.FOLD:
            folded.add(action.seat)
        elif stacks[action.seat] == 0:
            all_in.add(action.seat)
    return folded, stacks


def find_problems(hand: HandHistory, *, check_positions: bool = True) -> list[str]:
    """Return a list of human-readable consistency problems (empty if the hand is valid)."""
    problems: list[str] = []
    seats: dict[int, SeatInfo] = {s.seat: s for s in hand.seats}

    if check_positions:
        expected = assign_positions(seats, hand.button_seat)
        for seat, info in seats.items():
            if info.position != expected[seat]:
                problems.append(f"seat {seat}: position {info.position} != {expected[seat]}")
    player_ids = [s.player_id for s in hand.seats]
    if len(set(player_ids)) != len(player_ids):
        problems.append("a player occupies more than one seat")

    folded, _ = _check_actions(hand, problems)
    _check_forced_posts(hand, problems)

    board = hand.board_cards()
    if len(board) not in (0, 3, 4, 5):
        problems.append(f"board has {len(board)} cards")
    last_street = max((a.street for a in hand.actions), key=lambda s: s.index, default=None)
    if last_street is not None and len(board) < last_street.board_size:
        problems.append(f"actions on {last_street} but only {len(board)} board cards")

    known: list[str] = list(board)
    hole_seats: list[int] = []
    for hc in hand.hole_cards:
        hole_seats.append(hc.seat)
        if hc.seat not in seats:
            problems.append(f"hole cards for empty seat {hc.seat}")
        if hc.cards is not None:
            known.extend(obs.card for obs in hc.cards)
    if len(set(hole_seats)) != len(hole_seats):
        problems.append("more than one hole-card entry for a seat")
    try:
        assert_distinct(known)
    except ValueError as exc:
        problems.append(f"card duplicated across board/hole cards: {exc}")

    if hand.status is HandStatus.COMPLETE:
        _check_results(hand, seats, folded, problems)
    return problems


def _check_results(
    hand: HandHistory, seats: dict[int, SeatInfo], folded: set[int], problems: list[str]
) -> None:
    contributions = hand.contributions()
    for seat, amount in contributions.items():
        if amount > seats[seat].starting_stack:
            problems.append(f"seat {seat} contributed more than its stack")
    for name, awards in (("payout", hand.payouts), ("uncalled return", hand.uncalled_returns)):
        award_seats = [a.seat for a in awards]
        if len(set(award_seats)) != len(award_seats):
            problems.append(f"duplicate {name} entries for a seat")
        for award in awards:
            if award.seat not in seats:
                problems.append(f"{name} to empty seat {award.seat}")
            elif award.seat in folded:
                problems.append(f"{name} to seat {award.seat}, which folded")
    expected_returns = compute_uncalled_returns(hand.actions)
    recorded_returns = {a.seat: a.amount for a in hand.uncalled_returns}
    if expected_returns != recorded_returns:
        problems.append(f"uncalled returns {recorded_returns} != recomputed {expected_returns}")
    paid_in = sum(contributions.values())
    paid_out = sum(a.amount for a in hand.payouts) + sum(recorded_returns.values()) + hand.rake
    if paid_in != paid_out:
        problems.append(f"chips not conserved: in={paid_in}, out={paid_out}")
    if paid_in > sum(recorded_returns.values()) + hand.rake and not hand.payouts:
        problems.append("pot was never awarded")
    remaining = [s for s in seats if s not in folded]
    showdown_seats = {e.seat for e in hand.showdown}
    if showdown_seats - set(remaining):
        problems.append("showdown includes a seat that folded")
    if len(remaining) >= 2 and hand.showdown and showdown_seats != set(remaining):
        problems.append("showdown must list every seat that did not fold")
    if len(remaining) == 1 and hand.showdown:
        problems.append("showdown recorded although only one player remained")
    for hc in hand.hole_cards:
        if hc.shown and hc.seat not in showdown_seats:
            problems.append(f"seat {hc.seat} marked shown but did not reach showdown")


def validate_hand_history(hand: HandHistory, *, check_positions: bool = True) -> HandHistory:
    """Return ``hand`` if consistent, otherwise raise ``HandHistoryValidationError``."""
    problems = find_problems(hand, check_positions=check_positions)
    if problems:
        raise HandHistoryValidationError(hand.hand_id, problems)
    return hand

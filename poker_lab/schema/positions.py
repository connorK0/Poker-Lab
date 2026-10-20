"""Seat ordering and position names.

Dealing order starts at the first occupied seat clockwise from the button and ends with the
button. With three or more players the order is ``SB, BB, ..., BTN``. Heads-up the button
posts the small blind, so the order is ``BB, BTN``. This is also PokerKit's player-index
order, so ``seat_order_from_button(...)[i]`` is the seat of PokerKit player ``i``.
"""

from __future__ import annotations

from collections.abc import Iterable

#: Every position label the schema accepts.
POSITION_LABELS: frozenset[str] = frozenset(
    {"SB", "BB", "UTG", "UTG+1", "UTG+2", "UTG+3", "UTG+4", "HJ", "CO", "BTN"}
)


def position_names(n_players: int) -> tuple[str, ...]:
    """Position labels in dealing order for ``n_players`` (2 to 10) players."""
    if not 2 <= n_players <= 10:
        raise ValueError(f"unsupported number of players: {n_players}")
    if n_players == 2:
        return ("BB", "BTN")
    late = ("HJ", "CO", "BTN")
    k = n_players - 2  # seats that are not blinds
    if k <= len(late):
        rest = late[len(late) - k :]
    else:
        early = ("UTG", *(f"UTG+{i}" for i in range(1, k - len(late))))
        rest = early + late
    return ("SB", "BB", *rest)


def seat_order_from_button(occupied_seats: Iterable[int], button_seat: int) -> tuple[int, ...]:
    """Occupied seats in dealing order: clockwise from the seat after the button.

    Seats are numbered clockwise. ``button_seat`` must be one of the occupied seats.
    """
    seats = sorted(set(occupied_seats))
    if button_seat not in seats:
        raise ValueError(f"button seat {button_seat} is not occupied: {seats}")
    i = seats.index(button_seat)
    return tuple(seats[i + 1 :] + seats[: i + 1])


def assign_positions(occupied_seats: Iterable[int], button_seat: int) -> dict[int, str]:
    """Map each occupied seat to its position label for a hand with the given button."""
    order = seat_order_from_button(occupied_seats, button_seat)
    return dict(zip(order, position_names(len(order)), strict=True))

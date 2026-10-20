"""Firewalled views of an in-progress hand.

These are the only types through which decision-making code (bots, ``StrategyProvider``,
``ExploitEngine``, ``Advisor``) sees the hand being played:

* ``PublicHandState``: everything any spectator could know (table, seats, positions, stacks,
  live bets, pot, board, the public action log). It has no hole-card field at all.
* ``SeatView``: the public state plus exactly one seat's own hole cards, and that seat's legal
  options when it is the actor.

The constructors check exact types (subclasses are rejected), cards, and cross-field
consistency, so a full ``HandHistory``, a PokerKit ``State``, or any other container that
holds other players' cards cannot be passed in. All classes are frozen and slotted, so
nothing can be attached to an instance after construction. In-process Python code can always
defeat such measures by reflection. The firewall protects against accidental leakage and
honest-but-buggy code, not a deliberately malicious caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from poker_lab.schema.cards import validate_card
from poker_lab.schema.hand_history import Action, SeatInfo, Street, TableConfig


def _exact(value: object, cls: type, name: str) -> None:
    if type(value) is not cls:
        raise TypeError(f"{name} must be exactly {cls.__name__}, got {type(value).__name__}")


def _int_tuple(value: object, name: str, length: int) -> None:
    if type(value) is not tuple or len(value) != length:
        raise TypeError(f"{name} must be a tuple of length {length}")
    for v in value:
        if type(v) is not int:
            raise TypeError(f"{name} must contain ints")


def _bool_tuple(value: object, name: str, length: int) -> None:
    if type(value) is not tuple or len(value) != length:
        raise TypeError(f"{name} must be a tuple of length {length}")
    for v in value:
        if type(v) is not bool:
            raise TypeError(f"{name} must contain bools")


@dataclass(frozen=True, slots=True)
class LegalActions:
    """The actor's legal options. Amounts are totals for the street (``raise to``)."""

    can_fold: bool
    can_check: bool
    call_amount: int
    can_bet_or_raise: bool
    min_raise_to: int = 0
    max_raise_to: int = 0

    def __post_init__(self) -> None:
        for name in ("can_fold", "can_check", "can_bet_or_raise"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        for name in ("call_amount", "min_raise_to", "max_raise_to"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise TypeError(f"{name} must be a non-negative int")
        if self.can_check == (self.call_amount > 0):
            raise ValueError("exactly one of: check is free, or a positive call amount")
        if self.can_bet_or_raise:
            if not 0 < self.min_raise_to <= self.max_raise_to:
                raise ValueError("need 0 < min_raise_to <= max_raise_to when raising is legal")
        elif self.min_raise_to or self.max_raise_to:
            raise ValueError("raise bounds must be 0 when raising is not legal")


@dataclass(frozen=True, slots=True)
class PublicHandState:
    """Public information about a hand in progress. Per-seat tuples align with ``seats``."""

    hand_id: str
    table: TableConfig
    button_seat: int
    seats: tuple[SeatInfo, ...]
    street: Street
    board: tuple[str, ...]
    actions: tuple[Action, ...]
    stacks: tuple[int, ...]
    street_bets: tuple[int, ...]
    pot: int
    folded: tuple[bool, ...]
    actor_seat: int | None

    def __post_init__(self) -> None:
        if type(self.hand_id) is not str or not self.hand_id:
            raise TypeError("hand_id must be a non-empty str")
        _exact(self.table, TableConfig, "table")
        _exact(self.street, Street, "street")
        if type(self.seats) is not tuple or not self.seats:
            raise TypeError("seats must be a non-empty tuple")
        for s in self.seats:
            _exact(s, SeatInfo, "seat entry")
        n = len(self.seats)
        if type(self.board) is not tuple:
            raise TypeError("board must be a tuple of card strings")
        for card in self.board:
            validate_card(card)
        if len(self.board) != self.street.board_size:
            raise ValueError(f"{self.street} needs {self.street.board_size} board cards")
        if type(self.actions) is not tuple:
            raise TypeError("actions must be a tuple")
        for a in self.actions:
            _exact(a, Action, "action")
        _int_tuple(self.stacks, "stacks", n)
        _int_tuple(self.street_bets, "street_bets", n)
        _bool_tuple(self.folded, "folded", n)
        if type(self.pot) is not int or self.pot < 0:
            raise TypeError("pot must be a non-negative int")
        seat_numbers = {s.seat for s in self.seats}
        if self.button_seat not in seat_numbers:
            raise ValueError("button_seat must be occupied")
        if self.actor_seat is not None and self.actor_seat not in seat_numbers:
            raise ValueError("actor_seat must be occupied")

    def index_of(self, seat: int) -> int:
        """Position of ``seat`` in the per-seat tuples."""
        for i, s in enumerate(self.seats):
            if s.seat == seat:
                return i
        raise KeyError(seat)

    @property
    def max_street_bet(self) -> int:
        return max(self.street_bets)

    def active_seats(self) -> tuple[int, ...]:
        """Seats that have not folded, in table order."""
        return tuple(s.seat for s, f in zip(self.seats, self.folded, strict=True) if not f)


@dataclass(frozen=True, slots=True)
class SeatView:
    """What one seat may know during the hand: public state plus its own two hole cards."""

    public: PublicHandState
    seat: int
    hole_cards: tuple[str, str]
    legal: LegalActions | None = None

    def __post_init__(self) -> None:
        _exact(self.public, PublicHandState, "public")
        if type(self.seat) is not int:
            raise TypeError("seat must be an int")
        self.public.index_of(self.seat)  # raises KeyError if the seat is not in the hand
        hc = self.hole_cards
        if type(hc) is not tuple or len(hc) != 2:
            raise TypeError("hole_cards must be a tuple of exactly two card strings")
        for card in hc:
            if type(card) is not str:
                raise TypeError("hole cards must be plain str")
            validate_card(card)
        if hc[0] == hc[1] or set(hc) & set(self.public.board):
            raise ValueError("hole cards must be distinct and not on the board")
        if self.legal is not None:
            _exact(self.legal, LegalActions, "legal")
            if self.public.actor_seat != self.seat:
                raise ValueError("legal actions are only given to the seat whose turn it is")

    @property
    def is_actor(self) -> bool:
        return self.legal is not None

    @property
    def stack(self) -> int:
        return self.public.stacks[self.public.index_of(self.seat)]

    @property
    def street_bet(self) -> int:
        return self.public.street_bets[self.public.index_of(self.seat)]

    @property
    def position(self) -> str:
        return self.public.seats[self.public.index_of(self.seat)].position

    @property
    def player_id(self) -> str:
        return self.public.seats[self.public.index_of(self.seat)].player_id

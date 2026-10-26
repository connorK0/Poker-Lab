"""Canonical hand-history data model (schema version 2).

This module is the contract between all phases. The simulator (Phase 1), external imports
(Phase 3), and the camera + manual capture system (Phase 4) all emit ``HandHistory`` objects.
The profiler, evaluators, and advisor only consume them.

Conventions
-----------
* Chip amounts are integers in the table's smallest unit (``TableConfig.currency``).
* ``Action.amount`` is the number of chips the action moved from the stack into the pot.
  ``Action.total_bet`` is the seat's total live bet on the street after the action. Antes are
  dead money and never count toward ``total_bet``.
* Seats are integers in ``[0, max_seats)``, numbered clockwise.
* ``payouts`` are pot awards. ``uncalled_returns`` are bets returned because nobody matched
  them. Chip conservation: ``sum(contributions) == sum(payouts) + sum(uncalled_returns) + rake``.
* Forced bets (``TableConfig``) are listed in dealing order, as PokerKit and PHH do: the small
  blind, the big blind, then the straddles of the players after it (``straddles[0]`` is the
  first player after the big blind). Antes are either the same for every seat (``ante``) or
  one ante the big blind posts for the table (``big_blind_ante``, as in modern tournaments).
  The action log records what each seat actually posted, which is less when a stack is short.

Version 2 added ``big_blind_ante``, ``straddles`` and ``min_bet``; ``serialization`` reads
version 1 records by migrating them.

Information firewall
--------------------
A ``HandHistory`` with ``status == IN_PROGRESS`` may not carry any hole cards. That is
enforced here, at construction. While a hand is in progress its hole cards live sealed inside
``HandHistoryStore`` (see ``store.py``) and only reach a ``HandHistory`` when the hand is
marked complete.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from poker_lab.schema.cards import validate_card
from poker_lab.schema.positions import POSITION_LABELS

SCHEMA_VERSION = 2


class Street(StrEnum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"

    @property
    def index(self) -> int:
        return _STREET_INDEX[self]

    @property
    def board_size(self) -> int:
        """Number of board cards visible during this street."""
        return _BOARD_SIZE[self]


STREETS: tuple[Street, ...] = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)
_STREET_INDEX = {s: i for i, s in enumerate(STREETS)}
_BOARD_SIZE = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}


def street_for_board_size(n_cards: int) -> Street:
    """The street during which exactly ``n_cards`` board cards are visible."""
    for street, size in _BOARD_SIZE.items():
        if size == n_cards:
            return street
    raise ValueError(f"invalid board size: {n_cards}")


class ActionType(StrEnum):
    POST_ANTE = "post_ante"
    POST_SMALL_BLIND = "post_small_blind"
    POST_BIG_BLIND = "post_big_blind"
    POST_STRADDLE = "post_straddle"
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"

    @property
    def is_forced(self) -> bool:
        """Antes, blinds and straddles: chips posted without a decision."""
        return self in _FORCED

    @property
    def is_decision(self) -> bool:
        """A voluntary choice by the player (fold/check/call/bet/raise)."""
        return self not in _FORCED

    @property
    def is_aggressive(self) -> bool:
        return self is ActionType.BET or self is ActionType.RAISE


_FORCED = frozenset(
    {
        ActionType.POST_ANTE,
        ActionType.POST_SMALL_BLIND,
        ActionType.POST_BIG_BLIND,
        ActionType.POST_STRADDLE,
    }
)
_ZERO_AMOUNT = frozenset({ActionType.FOLD, ActionType.CHECK})


class HandStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class CardSource(StrEnum):
    """Where a card observation came from."""

    SIMULATOR = "simulator"  # dealt by the simulation engine (ground truth)
    MANUAL = "manual"  # typed in by a person
    CAMERA = "camera"  # detected by a vision model (Phase 4)
    IMPORTED = "imported"  # parsed from an external hand-history file


def _set(obj: object, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


def _require_int(name: str, value: object, minimum: int | None = None) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


@dataclass(frozen=True, slots=True)
class CardObservation:
    """A single observed card with provenance.

    ``confidence`` is the probability that ``card`` is correct. It is 1.0 for simulator,
    manual and imported cards and the detector's calibrated score for camera detections.
    """

    card: str
    source: CardSource = CardSource.SIMULATOR
    confidence: float = 1.0

    def __post_init__(self) -> None:
        validate_card(self.card)
        if not isinstance(self.source, CardSource):
            _set(self, "source", CardSource(self.source))
        conf = self.confidence
        if isinstance(conf, bool) or not isinstance(conf, int | float) or not math.isfinite(conf):
            raise ValueError(f"confidence must be a finite number, got {conf!r}")
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {conf}")
        _set(self, "confidence", float(conf))


@dataclass(frozen=True, slots=True)
class TableConfig:
    """A table's rules. ``min_bet`` (the smallest bet, and the smallest raise increment before
    any raise) defaults to the big blind. ``straddles`` has no trailing zeros, so equal rules
    compare equal."""

    table_id: str
    max_seats: int
    small_blind: int
    big_blind: int
    ante: int = 0
    game: str = "NLHE"
    currency: str = "chips"
    big_blind_ante: int = 0
    straddles: tuple[int, ...] = ()
    min_bet: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.table_id, str) or not self.table_id:
            raise ValueError("table_id must be a non-empty string")
        _require_int("max_seats", self.max_seats, 2)
        if self.max_seats > 10:
            raise ValueError(f"max_seats must be <= 10, got {self.max_seats}")
        _require_int("small_blind", self.small_blind, 0)
        _require_int("big_blind", self.big_blind, 1)
        _require_int("ante", self.ante, 0)
        if self.small_blind > self.big_blind:
            raise ValueError("small_blind must not exceed big_blind")
        if self.game != "NLHE":
            raise ValueError(f"unsupported game {self.game!r}; schema v2 supports 'NLHE' only")
        _require_int("big_blind_ante", self.big_blind_ante, 0)
        if self.ante and self.big_blind_ante:
            raise ValueError("a table has either an ante for every seat or a big-blind ante")
        if type(self.straddles) is not tuple:
            _set(self, "straddles", tuple(self.straddles))
        level = self.big_blind
        for k, straddle in enumerate(self.straddles):
            _require_int(f"straddles[{k}]", straddle, 0)
            if straddle and straddle <= level:
                raise ValueError(f"straddle {straddle} must exceed the bet before it ({level})")
            level = max(level, straddle)
        if self.straddles and not self.straddles[-1]:
            raise ValueError("straddles must not end with 0 (no straddle)")
        if len(self.straddles) > self.max_seats - 2:
            raise ValueError(f"{len(self.straddles)} straddles at a {self.max_seats}-seat table")
        if self.min_bet is None:
            _set(self, "min_bet", self.big_blind)
        _require_int("min_bet", self.min_bet, 1)

    @property
    def blinds_or_straddles(self) -> tuple[int, ...]:
        """Small blind, big blind and straddles, in dealing order."""
        return (self.small_blind, self.big_blind, *self.straddles)


@dataclass(frozen=True, slots=True)
class SeatInfo:
    seat: int
    player_id: str
    starting_stack: int
    position: str

    def __post_init__(self) -> None:
        _require_int("seat", self.seat, 0)
        if not isinstance(self.player_id, str) or not self.player_id:
            raise ValueError("player_id must be a non-empty string")
        _require_int("starting_stack", self.starting_stack, 1)
        if self.position not in POSITION_LABELS:
            raise ValueError(f"unknown position {self.position!r}")


@dataclass(frozen=True, slots=True)
class Action:
    """One entry in the ordered action log."""

    seat: int
    street: Street
    action_type: ActionType
    amount: int = 0
    total_bet: int = 0
    is_all_in: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.street, Street):
            _set(self, "street", Street(self.street))
        if not isinstance(self.action_type, ActionType):
            _set(self, "action_type", ActionType(self.action_type))
        _require_int("seat", self.seat, 0)
        _require_int("amount", self.amount, 0)
        _require_int("total_bet", self.total_bet, 0)
        kind = self.action_type
        if kind in _ZERO_AMOUNT and self.amount != 0:
            raise ValueError(f"{kind} must have amount 0, got {self.amount}")
        if kind not in _ZERO_AMOUNT and self.amount <= 0:
            raise ValueError(f"{kind} must move chips (amount > 0), got {self.amount}")
        if kind is ActionType.POST_ANTE:
            if self.street is not Street.PREFLOP:
                raise ValueError("antes are posted preflop")
        elif self.total_bet < self.amount:
            raise ValueError(f"total_bet ({self.total_bet}) < amount ({self.amount}) for {kind}")
        if kind.is_forced and self.street is not Street.PREFLOP:
            raise ValueError(f"{kind} can only happen preflop")
        if not isinstance(self.is_all_in, bool):
            raise TypeError("is_all_in must be a bool")


@dataclass(frozen=True, slots=True)
class HoleCards:
    """A seat's hole cards. ``cards is None`` means unknown, e.g. a mucked hand in a real game."""

    seat: int
    cards: tuple[CardObservation, ...] | None
    shown: bool = False

    def __post_init__(self) -> None:
        _require_int("seat", self.seat, 0)
        if self.cards is not None:
            cards = tuple(self.cards)
            if len(cards) != 2 or not all(isinstance(c, CardObservation) for c in cards):
                raise ValueError("hole cards must be exactly two CardObservation objects")
            if cards[0].card == cards[1].card:
                raise ValueError("duplicate hole card")
            _set(self, "cards", cards)
        elif self.shown:
            raise ValueError("cards marked shown must be known")


@dataclass(frozen=True, slots=True)
class ShowdownEntry:
    """A seat that reached showdown; ``shown`` is False if the hand was mucked."""

    seat: int
    shown: bool

    def __post_init__(self) -> None:
        _require_int("seat", self.seat, 0)


@dataclass(frozen=True, slots=True)
class ChipAward:
    seat: int
    amount: int

    def __post_init__(self) -> None:
        _require_int("seat", self.seat, 0)
        _require_int("amount", self.amount, 1)


def _as_tuple(obj: object, name: str, value: Any, item_type: type) -> None:
    items = tuple(value)
    for item in items:
        if not isinstance(item, item_type):
            raise TypeError(f"{name} must contain {item_type.__name__}, got {type(item).__name__}")
    _set(obj, name, items)


@dataclass(frozen=True, slots=True)
class HandHistory:
    """A single hand, in progress or complete. See the module docstring for conventions."""

    hand_id: str
    table: TableConfig
    button_seat: int
    seats: tuple[SeatInfo, ...]
    actions: tuple[Action, ...] = ()
    board: tuple[CardObservation, ...] = ()
    hole_cards: tuple[HoleCards, ...] = ()
    showdown: tuple[ShowdownEntry, ...] = ()
    payouts: tuple[ChipAward, ...] = ()
    uncalled_returns: tuple[ChipAward, ...] = ()
    rake: int = 0
    status: HandStatus = HandStatus.COMPLETE
    source: str = "simulator"
    started_at: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict, hash=False)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.hand_id, str) or not self.hand_id:
            raise ValueError("hand_id must be a non-empty string")
        if not isinstance(self.table, TableConfig):
            raise TypeError("table must be a TableConfig")
        if not isinstance(self.status, HandStatus):
            _set(self, "status", HandStatus(self.status))
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version}")
        _require_int("button_seat", self.button_seat, 0)
        _require_int("rake", self.rake, 0)
        _as_tuple(self, "seats", self.seats, SeatInfo)
        _as_tuple(self, "actions", self.actions, Action)
        _as_tuple(self, "board", self.board, CardObservation)
        _as_tuple(self, "hole_cards", self.hole_cards, HoleCards)
        _as_tuple(self, "showdown", self.showdown, ShowdownEntry)
        _as_tuple(self, "payouts", self.payouts, ChipAward)
        _as_tuple(self, "uncalled_returns", self.uncalled_returns, ChipAward)
        meta = dict(self.metadata)
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
            raise TypeError("metadata keys and values must be strings")
        _set(self, "metadata", MappingProxyType(meta))

        seat_numbers = [s.seat for s in self.seats]
        if len(set(seat_numbers)) != len(seat_numbers):
            raise ValueError("duplicate seat numbers")
        if len(seat_numbers) < 2:
            raise ValueError("a hand needs at least two seated players")
        if any(s >= self.table.max_seats for s in seat_numbers):
            raise ValueError("seat number outside the table's max_seats")
        if self.button_seat not in seat_numbers:
            raise ValueError("button_seat must be an occupied seat")
        if len(self.board) > 5:
            raise ValueError("board has more than 5 cards")

        if self.status is HandStatus.IN_PROGRESS:
            # Firewall invariant: hole cards of an in-progress hand never leave the sealed store.
            if self.hole_cards:
                raise ValueError(
                    "an in-progress HandHistory cannot carry hole cards; they stay sealed in "
                    "HandHistoryStore until the hand is complete"
                )
            if self.showdown or self.payouts or self.uncalled_returns or self.rake:
                raise ValueError("an in-progress hand cannot have showdown or payout results")

    # ------------------------------------------------------------------ convenience accessors
    @property
    def is_complete(self) -> bool:
        return self.status is HandStatus.COMPLETE

    def seat_info(self, seat: int) -> SeatInfo:
        for info in self.seats:
            if info.seat == seat:
                return info
        raise KeyError(seat)

    def seat_of(self, player_id: str) -> int:
        for info in self.seats:
            if info.player_id == player_id:
                return info.seat
        raise KeyError(player_id)

    def player_ids(self) -> tuple[str, ...]:
        return tuple(s.player_id for s in self.seats)

    def hole_cards_of(self, seat: int) -> HoleCards | None:
        for hc in self.hole_cards:
            if hc.seat == seat:
                return hc
        return None

    def board_cards(self) -> tuple[str, ...]:
        return tuple(obs.card for obs in self.board)

    def contributions(self) -> dict[int, int]:
        """Total chips each seat put into the pot (antes, blinds, calls, bets, raises)."""
        totals = {s.seat: 0 for s in self.seats}
        for action in self.actions:
            totals[action.seat] += action.amount
        return totals

    def net_results(self) -> dict[int, int]:
        """Chips won (positive) or lost (negative) per seat. Complete hands only."""
        if not self.is_complete:
            raise ValueError("net results are only defined for complete hands")
        net = {seat: -c for seat, c in self.contributions().items()}
        for award in self.payouts:
            net[award.seat] += award.amount
        for award in self.uncalled_returns:
            net[award.seat] += award.amount
        return net

    def went_to_showdown(self, seat: int) -> bool:
        return any(entry.seat == seat for entry in self.showdown)

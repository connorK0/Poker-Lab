"""Batch import of completed hand histories from PHH text files, and export back to PHH.

PHH ("Poker Hand History", Kim 2024) is an open, TOML-based text format from the University
of Toronto Computer Poker Student Research Group, the authors of PokerKit. Public research data
uses it, e.g. the 10,000 hands of Pluribus against professionals (Brown and Sandholm 2019) and
televised hands. A ``.phh`` file holds one hand; a ``.phhs`` file holds several, as numbered
TOML tables (``[1]``, ``[2]``, ...).

Offline and batch only: ``import_phh`` reads the files it is given, once, and returns. It never
watches a folder for new hands (that is how real-time HUDs work) and it does not read online
poker sites' own export formats (PokerKit has parsers for several; they are deliberately not
used).

Strict conversion
-----------------
Every hand is replayed through PokerKit's rules, action by action. PokerKit's own loader
"repairs" a hand history whose next action does not apply by inventing folds, checks and
showdown mucks; here such an action rejects the hand. Only two steps may be left implicit,
because neither adds information: dealing unknown hole cards and burning cards. A hand is
accepted only if its actions run to the end of the hand and the converted record passes
``find_problems``. Every known card must appear once, checked before the replay: PokerKit
only warns about a card dealt twice (not at all within one deal). A show must be of the cards
dealt: PokerKit accepts other cards and pays the pot to them. Recorded finishing stacks must
equal the replay's result less a non-negative rake taken from the pot winners. When a hand
shown at showdown is unknown the replay cannot decide the pot, and finishing stacks are then
required.

Mapping
-------
* Amounts: decimal amounts (dollars and cents) are scaled by the smallest power of ten, at
  most ``10**4``, that makes all of them integers. The currency records the unit:
  ``"USD/100"`` means hundredths of a dollar. Without a currency field the unit is ``chips``.
  The scale is per hand: a file whose split pot gives each winner half of the odd chip (some
  Pluribus hands) is imported in tenths of a chip, ``"chips/10"``.
* Seats: PHH lists players in dealing order (small blind first, button last; heads-up the big
  blind first, as in PokerKit). The ``seats`` field gives physical seat numbers (1-based).
  Without it the players sit in seats 0..n-1 and the last one has the button.
* Players: the ``players`` field, or ``"<hand id>/p<k>"`` when it is missing, so that
  anonymous players are never merged across hands.
* Cards: every card gets ``source=imported`` and confidence 1. Every seat gets a hole-card
  entry (``None`` when unknown); cards shown at showdown replace unknown dealt cards.
* Descriptive fields (event, venue, author, ...) and user-defined fields go to
  ``metadata["phh.<field>"]``, the date and time to ``started_at``.

* Forced bets: the blinds and straddles in dealing order, the minimum bet, and antes that are
  either the same for every player or posted by the big blind alone (the big-blind ante of
  modern tournaments, which PHH records without ante trimming).

Schema v2 cannot hold other antes that differ by seat (a button ante, say), a big-blind ante
with ante trimming, more than one runout, or variants other than no-limit hold'em. Such hands
are rejected as unsupported, never approximated.

The export (``hand_to_phh``) is the inverse up to card provenance. It places board deals and
showdown decisions where PokerKit's rules put them and records finishing stacks, so that a
re-import checks the result.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import re
import tomllib
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pokerkit import (
    BoardDealing,
    CheckingOrCalling,
    CompletionBettingOrRaisingTo,
    Folding,
    HoleCardsShowingOrMucking,
    HoleDealing,
    State,
)
from pokerkit import HandHistory as PHH
from pokerkit.notation import parse_action
from pokerkit.utilities import clean_values, parse_value

from poker_lab.engine.fast_hand import FastNoLimitTexasHoldem
from poker_lab.engine.simulator import forced_posts, pokerkit_rules
from poker_lab.schema import (
    Action,
    ActionType,
    CardObservation,
    CardSource,
    ChipAward,
    HandHistory,
    HandStatus,
    HoleCards,
    SeatInfo,
    ShowdownEntry,
    Street,
    TableConfig,
    assign_positions,
    compute_uncalled_returns,
    find_problems,
    seat_order_from_button,
)
from poker_lab.schema.hand_history import street_for_board_size

SUFFIXES = (".phh", ".phhs")
MAX_SCALE_DIGITS = 4

#: PHH descriptive fields kept as metadata, by type.
STR_FIELDS = (
    "author",
    "event",
    "url",
    "venue",
    "address",
    "city",
    "region",
    "postal_code",
    "country",
    "time_zone",
    "currency_symbol",
)
INT_FIELDS = ("level", "time_limit")
#: User-defined fields this module writes and reads back.
META_FIELD = "_poker_lab_metadata"
STARTED_AT_FIELD = "_poker_lab_started_at"
_RESERVED = frozenset({META_FIELD, STARTED_AT_FIELD})
_PARTIAL_ISO = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2})(?:T(\d{2}:\d{2}:\d{2}))?)?)?$")
_BENIGN_WARNINGS = ("There is no reason for this player to fold.",)  # legal in cash games


class _PHH(PHH):
    """PHH whose no-limit hold'em games use the phevaluator-backed hand type (the same rules,
    a faster showdown; see ``engine/fast_hand.py``)."""

    game_types = {**PHH.game_types, "NT": FastNoLimitTexasHoldem}  # noqa: RUF012


class RejectReason(StrEnum):
    PARSE = "parse"  # not TOML, or not a PHH hand (missing or mistyped fields)
    UNSUPPORTED = "unsupported"  # valid PHH outside schema v2 (variant, button ante, ...)
    ILLEGAL_ACTION = "illegal_action"  # PokerKit refuses an action
    INCOMPLETE = "incomplete"  # the actions stop before the hand is over
    RESULT_MISMATCH = "result_mismatch"  # recorded finishing stacks contradict the replay
    UNKNOWN_RESULT = "unknown_result"  # unknown cards shown at showdown, no finishing stacks
    INCONSISTENT = "inconsistent"  # the converted record fails the schema's checks
    DUPLICATE = "duplicate"  # a hand id already imported in this batch


class PHHImportError(ValueError):
    def __init__(self, reason: RejectReason, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}")


def _reject(reason: RejectReason, detail: str) -> PHHImportError:
    return PHHImportError(reason, detail)


# ---------------------------------------------------------------------------- amounts
def _decimal(value: Any, what: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, int | Decimal | float):
        raise _reject(RejectReason.PARSE, f"{what} is not a number: {value!r}")
    d = Decimal(str(value)) if isinstance(value, float) else Decimal(value)
    if not d.is_finite():
        raise _reject(RejectReason.UNSUPPORTED, f"{what} is not finite (unknown stack?)")
    if d < 0:
        raise _reject(RejectReason.PARSE, f"{what} is negative: {value}")
    return d


def _cbr_amount(action: str) -> str | None:
    """The amount of a ``pN cbr <amount>`` action, else None."""
    words = action.partition("#")[0].split()
    return words[2] if len(words) >= 3 and words[1] == "cbr" else None


def _map_cbr(action: str, f: Callable[[str], str]) -> str:
    """``action`` with the amount of a ``pN cbr <amount>`` action replaced by ``f(amount)``."""
    amount = _cbr_amount(action)
    if amount is None:
        return action
    comment = action.partition("#")[2].strip()
    return f"{action.split()[0]} cbr {f(amount)}" + (f" # {comment}" if comment else "")


def _as_list(value: Any, n: int, what: str) -> list[Any]:
    if isinstance(value, int | Decimal) and not isinstance(value, bool):
        return [value] * n
    if not isinstance(value, list) or len(value) > n:
        raise _reject(RejectReason.PARSE, f"{what} must be a list of at most {n} values")
    return [*value, *([0] * (n - len(value)))]


def _scaled(phh: PHH) -> tuple[PHH, int]:
    """Copy of ``phh`` with every amount an integer, and the scale applied."""
    n = len(phh.starting_stacks)
    amounts: dict[str, list[Decimal]] = {
        "antes": [_decimal(v, "ante") for v in _as_list(phh.antes, n, "antes")],
        "blinds_or_straddles": [
            _decimal(v, "blind") for v in _as_list(phh.blinds_or_straddles, n, "blinds")
        ],
        "starting_stacks": [_decimal(v, "starting stack") for v in phh.starting_stacks],
        "min_bet": [_decimal(phh.min_bet, "min_bet")],
    }
    if phh.finishing_stacks is not None:
        if len(phh.finishing_stacks) != n:
            raise _reject(RejectReason.PARSE, "finishing_stacks has the wrong length")
        amounts["finishing_stacks"] = [_decimal(v, "finishing stack") for v in phh.finishing_stacks]
    bets = [
        _decimal(_parse_amount(amount), "bet")
        for amount in map(_cbr_amount, phh.actions)
        if amount is not None
    ]
    for digits in range(MAX_SCALE_DIGITS + 1):
        scale = 10**digits
        values = [d * scale for ds in (*amounts.values(), bets) for d in ds]
        if all(v == v.to_integral_value() for v in values):
            break
    else:
        raise _reject(RejectReason.UNSUPPORTED, f"amounts finer than 1/10**{MAX_SCALE_DIGITS}")
    ints = {k: [int(d * scale) for d in v] for k, v in amounts.items()}
    actions = phh.actions
    if scale != 1:
        actions = [
            _map_cbr(a, lambda s: str(int(Decimal(_parse_amount(s)) * scale))) for a in actions
        ]
    return (
        dataclasses.replace(
            phh,
            antes=ints["antes"],
            blinds_or_straddles=ints["blinds_or_straddles"],
            starting_stacks=ints["starting_stacks"],
            min_bet=ints["min_bet"][0],
            finishing_stacks=ints.get("finishing_stacks"),
            winnings=None,
            actions=actions,
        ),
        scale,
    )


def _parse_amount(text: str) -> int | Decimal:
    try:
        return parse_value(text)
    except ArithmeticError as exc:  # decimal.InvalidOperation
        raise _reject(RejectReason.PARSE, f"amount {text!r} is not a number") from exc


def _unit(currency: str) -> tuple[str | None, int]:
    """(PHH currency code or None for plain chips, scale) for a schema currency string."""
    code, _, scale = currency.partition("/")
    if scale and not scale.isdigit():
        code, scale = currency, ""
    return (None if code == "chips" else code), int(scale or 1)


# ---------------------------------------------------------------------------- PHH -> schema
def _pending(state: State) -> str:
    if state.actor_index is not None:
        return f"p{state.actor_index + 1} to act"
    if state.showdown_index is not None:
        return f"p{state.showdown_index + 1} to show or muck"
    if state.board_dealing_count is not None:
        return "board cards to deal"
    return "more operations"


def _check_cards(actions: Sequence[str]) -> None:
    """Every known card in the deals and shows appears once. PokerKit only warns when a card
    is dealt twice (not at all within one deal) and may then fail an internal assertion, so
    this runs before the replay. A show may repeat the player's own dealt cards."""
    board: list[str] = []
    per_player: dict[str, tuple[list[str], list[str]]] = {}
    for action in actions:
        words = action.partition("#")[0].split()
        if len(words) == 3 and words[:2] == ["d", "db"]:
            board += _known(words[2])
        elif len(words) == 4 and words[:2] == ["d", "dh"]:
            per_player.setdefault(words[2], ([], []))[0].extend(_known(words[3]))
        elif len(words) == 3 and words[1] == "sm" and words[2] != "-":
            per_player.setdefault(words[0], ([], []))[1].extend(_known(words[2]))
    seen = list(board)
    for dealt, shown in per_player.values():
        for cards in (dealt, shown):
            if len(set(cards)) != len(cards):
                raise _reject(RejectReason.INCONSISTENT, f"a player holds {cards} twice")
        seen += dealt + [c for c in shown if c not in dealt]
    repeated = sorted(c for c, k in Counter(seen).items() if k > 1)
    if repeated:
        raise _reject(RejectReason.INCONSISTENT, f"card(s) {repeated} dealt twice")


def _known(cards: str) -> list[str]:
    pairs = [cards[i : i + 2] for i in range(0, len(cards), 2)]
    return [c for c in pairs if "?" not in c]


def _replay(phh: PHH) -> State:
    """Apply every action through PokerKit's rules; return the finished state."""
    _check_cards(phh.actions)
    try:
        state = phh.create_state()
    except (ValueError, TypeError) as exc:
        raise _reject(RejectReason.INCONSISTENT, f"PokerKit refused the setup: {exc}") from exc
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for k, action in enumerate(phh.actions):
            while state.can_burn_card():
                state.burn_card("??")
            if action.split()[:2] != ["d", "dh"]:
                while state.can_deal_hole():
                    state.deal_hole("??")  # an unknown hole card, left implicit
            try:
                parse_action(state, action, phh.parse_value)
            except (ValueError, IndexError) as exc:
                raise _reject(RejectReason.ILLEGAL_ACTION, f"action {k} {action!r}: {exc}") from exc
            except AssertionError as exc:  # PokerKit's own invariants, on input it accepted
                raise _reject(
                    RejectReason.INCONSISTENT, f"action {k} {action!r} broke a PokerKit invariant"
                ) from exc
        while state.can_burn_card():
            state.burn_card("??")
    for w in caught:
        message = str(w.message)
        if message not in _BENIGN_WARNINGS:
            raise _reject(RejectReason.INCONSISTENT, f"PokerKit: {message}")
    if state.status:
        raise _reject(RejectReason.INCOMPLETE, f"the actions stop with {_pending(state)}")
    return state


def _seating(phh: PHH, n: int) -> tuple[tuple[int, ...], int]:
    """(seat of each PokerKit player index, max_seats)."""
    if phh.seats is None:
        order = tuple(range(n))
        max_seats = phh.seat_count if phh.seat_count is not None else n
    else:
        if len(phh.seats) != n or not all(isinstance(s, int) and s >= 1 for s in phh.seats):
            raise _reject(RejectReason.PARSE, "seats must be one 1-based number per player")
        order = tuple(s - 1 for s in phh.seats)
        max_seats = phh.seat_count if phh.seat_count is not None else max(phh.seats)
    if len(set(order)) != n:
        raise _reject(RejectReason.INCONSISTENT, "two players share a seat")
    if max_seats > 10:
        raise _reject(RejectReason.UNSUPPORTED, f"{max_seats} seats (schema v2 holds at most 10)")
    if max(order) >= max_seats:
        raise _reject(RejectReason.INCONSISTENT, "a seat number exceeds seat_count")
    if seat_order_from_button(order, order[-1]) != order:
        raise _reject(RejectReason.INCONSISTENT, "seat numbers do not follow the dealing order")
    return order, max_seats


def _started_at(phh: PHH) -> str | None:
    custom = phh.user_defined_fields.get(STARTED_AT_FIELD)
    if custom is not None:
        return str(custom)
    if phh.year is None:
        return None
    out = f"{phh.year:04d}"
    if phh.month is not None:
        out += f"-{phh.month:02d}"
        if phh.day is not None:
            out += f"-{phh.day:02d}"
            if phh.time is not None:
                out += f"T{phh.time.isoformat()}"
    return out


def _json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _metadata(phh: PHH) -> dict[str, str]:
    meta: dict[str, str] = {}
    for name in STR_FIELDS:
        value = getattr(phh, name)
        if value is not None:
            meta[f"phh.{name}"] = str(value)
    for name in INT_FIELDS:
        value = getattr(phh, name)
        if value is not None:
            meta[f"phh.{name}"] = str(value)
    if phh.time_banks is not None:
        meta["phh.time_banks"] = _json(phh.time_banks)
    if not phh.ante_trimming_status and _uniform(phh.antes):  # implied by a big-blind ante
        meta["phh.ante_trimming_status"] = "false"
    if phh.year is not None and phh.month is None and phh.time is not None:
        meta["phh.time"] = phh.time.isoformat()  # a time without a full date
    for key, value in phh.user_defined_fields.items():
        if key not in _RESERVED:
            meta[f"phh.{key}"] = _json(value)
    ours = phh.user_defined_fields.get(META_FIELD)
    if ours is not None:
        try:
            extra = json.loads(ours)
        except (TypeError, json.JSONDecodeError) as exc:
            raise _reject(RejectReason.PARSE, f"{META_FIELD} is not JSON") from exc
        if not isinstance(extra, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in extra.items()
        ):
            raise _reject(RejectReason.PARSE, f"{META_FIELD} must map strings to strings")
        meta.update(extra)
    return meta


def _results(
    phh: PHH,
    state: State,
    order: tuple[int, ...],
    actions: Sequence[Action],
    showdown: Mapping[int, tuple[Any, ...]],
    allow_rake: bool,
) -> tuple[dict[int, int], int]:
    """(pot award per seat, rake). The replay decides the pot unless a hand shown at showdown
    is unknown; recorded finishing stacks must agree with it, up to rake from the winners."""
    contributions = dict.fromkeys(order, 0)
    for a in actions:
        contributions[a.seat] += a.amount
    uncalled = compute_uncalled_returns(actions)
    replayed = {
        seat: state.payoffs[i] + contributions[seat] - uncalled.get(seat, 0)
        for i, seat in enumerate(order)
    }
    unknown = [s for s, cards in showdown.items() if any(c.unknown_status for c in cards)]
    if phh.finishing_stacks is None:
        if unknown:
            raise _reject(
                RejectReason.UNKNOWN_RESULT,
                f"seat(s) {unknown} showed unknown cards and there are no finishing stacks",
            )
        return replayed, 0
    starting = dict(zip(order, phh.starting_stacks, strict=True))
    payouts = {
        seat: finish - starting[seat] + contributions[seat] - uncalled.get(seat, 0)
        for seat, finish in zip(order, phh.finishing_stacks, strict=True)
    }
    rake = sum(contributions.values()) - sum(payouts.values()) - sum(uncalled.values())
    if unknown:
        consistent = all(p >= 0 for p in payouts.values())
    else:
        consistent = all(0 <= payouts[s] <= replayed[s] for s in order)
    if not consistent or rake < 0 or (rake and not allow_rake):
        raise _reject(
            RejectReason.RESULT_MISMATCH,
            f"finishing stacks give pot awards {payouts}, the replay {replayed}",
        )
    return payouts, rake


def _text(cards: Sequence[Any]) -> str:
    return "".join(map(repr, cards))


def _same_cards(dealt: Sequence[Any], shown: Sequence[Any]) -> bool:
    """Whether shown cards can be the dealt ones: each side's known cards that the other side
    lacks must fit into the other side's unknown cards. (PokerKit lets a player show cards
    other than those dealt; a muck shows nothing and always fits.)"""
    if not shown:
        return True
    known_dealt = Counter(c for c in dealt if not c.unknown_status)
    known_shown = Counter(c for c in shown if not c.unknown_status)
    unknown_dealt = len(dealt) - known_dealt.total()
    unknown_shown = len(shown) - known_shown.total()
    return (known_shown - known_dealt).total() <= unknown_dealt and (
        known_dealt - known_shown
    ).total() <= unknown_shown


def _cards(cards: Sequence[Any]) -> tuple[CardObservation, CardObservation] | None:
    if len(cards) != 2 or any(c.unknown_status for c in cards):
        return None
    a, b = (CardObservation(repr(c), CardSource.IMPORTED, 1.0) for c in cards)
    return a, b


_SCALAR = (str, int)
_TYPES: dict[str, tuple[type, ...]] = {
    "variant": (str,),
    "ante_trimming_status": (bool,),
    "starting_stacks": (list,),
    "actions": (list,),
    "time": (datetime.time,),
    "day": (int,),
    "month": (int,),
    "year": (int,),
    "hand": _SCALAR,
    "table": _SCALAR,
    "seats": (list,),
    "seat_count": (int,),
    "players": (list,),
    "finishing_stacks": (list,),
    "currency": (str,),
    "time_banks": (list,),
    **dict.fromkeys(STR_FIELDS, _SCALAR),
    **dict.fromkeys(INT_FIELDS, (int,)),
}


def _check_types(phh: PHH) -> None:
    for name, types in _TYPES.items():
        value = getattr(phh, name)
        bad_bool = isinstance(value, bool) and bool not in types
        if value is not None and (bad_bool or not isinstance(value, types)):
            raise _reject(RejectReason.PARSE, f"field {name!r} has type {type(value).__name__}")
    if not all(isinstance(a, str) for a in phh.actions):
        raise _reject(RejectReason.PARSE, "actions must be strings")
    if phh.players is not None and not all(isinstance(p, _SCALAR) for p in phh.players):
        raise _reject(RejectReason.PARSE, "players must be names")


def _uniform(antes: Sequence[Any]) -> bool:
    """Whether ``antes`` are the same nonzero ante for every player."""
    return any(antes) and len(set(antes)) == 1


def _antes(phh: PHH) -> tuple[int, int]:
    """(ante for every player, big-blind ante) of a scaled hand."""
    antes = list(phh.antes)
    if not any(antes) or _uniform(antes):
        return antes[0], 0
    if any(a for i, a in enumerate(antes) if i != 1):
        raise _reject(RejectReason.UNSUPPORTED, f"antes differ by seat {antes}")
    if phh.ante_trimming_status:
        raise _reject(RejectReason.UNSUPPORTED, "a big-blind ante with ante trimming")
    return 0, antes[1]


def _as_fast(phh: PHH) -> _PHH:
    if isinstance(phh, _PHH):
        return phh
    return _PHH(**{f.name: getattr(phh, f.name) for f in dataclasses.fields(phh) if f.init})


def hand_from_phh(phh: PHH, *, hand_id: str | None = None, allow_rake: bool = True) -> HandHistory:
    """Convert one PHH hand into a complete, validated ``HandHistory`` with
    ``source="imported"``. Raises ``PHHImportError`` with the reason it was rejected.

    With ``allow_rake=False`` (rake-free data: tournaments, research matches) finishing stacks
    must equal the replay's result exactly; otherwise a shortfall taken from the pot winners
    is read as rake.
    """
    _check_types(phh)
    phh = _as_fast(phh)
    if phh.variant != "NT":
        raise _reject(
            RejectReason.UNSUPPORTED,
            f"variant {phh.variant!r} (schema v2 holds no-limit hold'em only)",
        )
    for name in PHH.required_field_names["NT"]:
        if getattr(phh, name) is None:
            raise _reject(RejectReason.PARSE, f"missing field {name!r}")
    if hand_id is None:
        if phh.hand is None:
            raise _reject(RejectReason.PARSE, "no hand id (the 'hand' field is missing)")
        hand_id = str(phh.hand)
    n = len(phh.starting_stacks)
    if not 2 <= n <= 10:
        raise _reject(RejectReason.UNSUPPORTED, f"{n} players (schema v2 holds 2 to 10)")
    phh, scale = _scaled(phh)
    ante, big_blind_ante = _antes(phh)
    small_blind, big_blind, *straddles = phh.blinds_or_straddles
    while straddles and not straddles[-1]:
        straddles.pop()
    order, max_seats = _seating(phh, n)
    if phh.players is None:
        players = [f"{hand_id}/p{i + 1}" for i in range(n)]
    else:
        players = [str(p) for p in phh.players]
        if len(players) != n or len(set(players)) != n or not all(players):
            raise _reject(RejectReason.INCONSISTENT, "players must be n distinct, non-empty names")
    currency = phh.currency or "chips"
    try:
        table = TableConfig(
            table_id=str(phh.table) if phh.table is not None else "imported",
            max_seats=max_seats,
            small_blind=small_blind,
            big_blind=big_blind,
            ante=ante,
            currency=currency if scale == 1 else f"{currency}/{scale}",
            big_blind_ante=big_blind_ante,
            straddles=tuple(straddles),
            min_bet=phh.min_bet,
        )
    except (ValueError, TypeError) as exc:
        raise _reject(RejectReason.UNSUPPORTED, str(exc)) from exc

    state = _replay(phh)
    starting = dict(zip(order, phh.starting_stacks, strict=True))
    actions = forced_posts(state.operations, order, starting)
    stacks = dict(starting)
    bets = dict.fromkeys(order, 0)
    for a in actions:
        stacks[a.seat] -= a.amount
        bets[a.seat] += 0 if a.action_type is ActionType.POST_ANTE else a.amount
    street = Street.PREFLOP
    board: list[CardObservation] = []
    dealt: dict[int, list[Any]] = {seat: [] for seat in order}
    showdown: dict[int, tuple[Any, ...]] = {}
    for op in state.operations:
        seat = order[op.player_index] if hasattr(op, "player_index") else -1
        if isinstance(op, HoleDealing):
            dealt[seat].extend(op.cards)
            continue
        if isinstance(op, BoardDealing):
            board.extend(CardObservation(repr(c), CardSource.IMPORTED, 1.0) for c in op.cards)
            if len(board) > 5:
                raise _reject(RejectReason.UNSUPPORTED, "more than one runout")
            street = street_for_board_size(len(board))
            bets = dict.fromkeys(order, 0)
            continue
        if isinstance(op, HoleCardsShowingOrMucking):
            if not _same_cards(dealt[seat], op.hole_cards):
                raise _reject(
                    RejectReason.INCONSISTENT,
                    f"seat {seat} was dealt {_text(dealt[seat])} but showed {_text(op.hole_cards)}",
                )
            showdown[seat] = tuple(op.hole_cards)
            continue
        if isinstance(op, Folding):
            action = Action(seat, street, ActionType.FOLD, 0, bets[seat])
        elif isinstance(op, CheckingOrCalling):
            kind = ActionType.CALL if op.amount else ActionType.CHECK
            all_in = bool(op.amount) and op.amount == stacks[seat]
            action = Action(seat, street, kind, op.amount, bets[seat] + op.amount, all_in)
        elif isinstance(op, CompletionBettingOrRaisingTo):
            added = op.amount - bets[seat]
            kind = ActionType.RAISE if max(bets.values()) > 0 else ActionType.BET
            action = Action(seat, street, kind, added, op.amount, added == stacks[seat])
        else:
            continue
        stacks[seat] -= action.amount
        bets[seat] = action.total_bet
        actions.append(action)

    hole_cards = []
    for seat in sorted(order):
        cards = _cards(showdown.get(seat) or dealt[seat])
        try:
            hole = HoleCards(seat, cards, shown=bool(showdown.get(seat)) and bool(cards))
        except ValueError as exc:
            raise _reject(RejectReason.INCONSISTENT, f"seat {seat}: {exc}") from exc
        hole_cards.append(hole)
    payouts, rake = _results(phh, state, order, actions, showdown, allow_rake)
    positions = assign_positions(order, order[-1])
    try:
        hand = HandHistory(
            hand_id=hand_id,
            table=table,
            button_seat=order[-1],
            seats=tuple(
                SeatInfo(seat, players[order.index(seat)], starting[seat], positions[seat])
                for seat in sorted(order)
            ),
            actions=tuple(actions),
            board=tuple(board),
            hole_cards=tuple(hole_cards),
            showdown=tuple(ShowdownEntry(s, bool(c)) for s, c in sorted(showdown.items())),
            payouts=tuple(ChipAward(s, a) for s, a in sorted(payouts.items()) if a > 0),
            uncalled_returns=tuple(
                ChipAward(s, a) for s, a in sorted(compute_uncalled_returns(actions).items())
            ),
            rake=rake,
            status=HandStatus.COMPLETE,
            source="imported",
            started_at=_started_at(phh),
            metadata=_metadata(phh),
        )
    except (ValueError, TypeError) as exc:
        raise _reject(RejectReason.INCONSISTENT, str(exc)) from exc
    problems = find_problems(hand)
    if problems:
        raise _reject(RejectReason.INCONSISTENT, "; ".join(problems[:3]))
    return hand


# ---------------------------------------------------------------------------- schema -> PHH
def as_imported(hand: HandHistory) -> HandHistory:
    """``hand`` as an import of its export reads it: imported cards (confidence 1), source
    ``"imported"``, and a hole-card entry for every seat (``None`` when unknown)."""

    def imported(obs: CardObservation) -> CardObservation:
        return CardObservation(obs.card, CardSource.IMPORTED, 1.0)

    hole = []
    for seat in sorted(s.seat for s in hand.seats):
        hc = hand.hole_cards_of(seat)
        if hc is None or hc.cards is None:
            hole.append(HoleCards(seat, None))
        else:
            hole.append(HoleCards(seat, tuple(map(imported, hc.cards)), shown=hc.shown))
    return dataclasses.replace(
        hand,
        source="imported",
        board=tuple(map(imported, hand.board)),
        hole_cards=tuple(hole),
    )


def _started_at_fields(started_at: str | None) -> dict[str, Any]:
    if started_at is None:
        return {}
    m = _PARTIAL_ISO.match(started_at)
    if m is None:
        return {"user_defined_fields": {STARTED_AT_FIELD: started_at}}
    year, month, day, time = m.groups()
    out: dict[str, Any] = {"year": int(year)}
    if month:
        out["month"] = int(month)
    if day:
        out["day"] = int(day)
    if time:
        out["time"] = datetime.time.fromisoformat(time)
    return out


def hand_to_phh(hand: HandHistory) -> PHH:
    """PHH for a complete hand: ``hand_from_phh(hand_to_phh(h))`` equals ``h`` except that
    cards become ``source=imported``, the hand ``source="imported"``, and every seat gets a
    hole-card entry. Raises ``ValueError`` for hands PokerKit's rules cannot replay."""
    if not hand.is_complete:
        raise ValueError("only complete hands can be exported")
    t = hand.table
    code, scale = _unit(t.currency)
    order = seat_order_from_button([s.seat for s in hand.seats], hand.button_seat)
    n = len(order)
    meta = dict(hand.metadata)
    trimming, raw_antes, raw_blinds, min_bet = pokerkit_rules(t, n)
    if meta.pop("phh.ante_trimming_status", "true") == "false":
        trimming = False
    antes = list(clean_values(raw_antes, n))
    blinds = list(clean_values(raw_blinds, n))
    base = _PHH(
        variant="NT",
        ante_trimming_status=trimming,
        antes=antes,
        blinds_or_straddles=blinds,
        min_bet=min_bet,
        starting_stacks=[hand.seat_info(s).starting_stack for s in order],
        actions=[],
    )
    state = base.create_state()
    actions: list[str] = []

    def emit(action: str) -> None:
        parse_action(state, action)
        actions.append(action)

    hole = {}
    for seat in order:
        hc = hand.hole_cards_of(seat)
        hole[seat] = "????" if hc is None or hc.cards is None else "".join(o.card for o in hc.cards)
    shown = {e.seat: e.shown for e in hand.showdown}
    board = list(hand.board_cards())
    decisions = iter(a for a in hand.actions if a.action_type.is_decision)
    for i, seat in enumerate(order):
        emit(f"d dh p{i + 1} {hole[seat]}")
    while state.status:
        if state.can_burn_card():
            state.burn_card("??")
        elif (count := state.board_dealing_count) is not None:
            if len(board) < count:
                raise ValueError(f"{hand.hand_id}: the recorded board is too short")
            emit("d db " + "".join(board[:count]))
            del board[:count]
        elif (i := state.showdown_index) is not None:
            seat = order[i]
            if seat not in shown:
                raise ValueError(f"{hand.hand_id}: showdown of seat {seat} is not recorded")
            emit(f"p{i + 1} sm {hole[seat]}" if shown[seat] else f"p{i + 1} sm")
        elif (i := state.actor_index) is not None:
            a = next(decisions, None)
            if a is None or a.seat != order[i]:
                raise ValueError(f"{hand.hand_id}: seat {order[i]} is to act, not {a}")
            kind = a.action_type
            if kind is ActionType.FOLD:
                emit(f"p{i + 1} f")
            elif kind in (ActionType.CHECK, ActionType.CALL):
                if state.checking_or_calling_amount != a.amount:
                    raise ValueError(f"{hand.hand_id}: {a} moves {a.amount} chips in the log")
                emit(f"p{i + 1} cc")
            else:
                emit(f"p{i + 1} cbr {a.total_bet}")
        else:
            raise ValueError(f"{hand.hand_id}: PokerKit is waiting on an unexpected operation")
    if next(decisions, None) is not None or board:
        raise ValueError(f"{hand.hand_id}: the hand ends before the recorded actions or board")

    def amount(x: int) -> int | Decimal:
        return x if scale == 1 else Decimal(x) / scale

    net = hand.net_results()
    user: dict[str, Any] = {}
    fields: dict[str, Any] = {}
    for key in list(meta):
        name = key.removeprefix("phh.")
        if name == key:
            continue
        value = meta.pop(key)
        if name in STR_FIELDS:
            fields[name] = value
        elif name in INT_FIELDS:
            fields[name] = int(value)
        elif name == "time_banks":
            fields[name] = json.loads(value)
        elif name == "time":
            fields[name] = datetime.time.fromisoformat(value)
        elif name.startswith("_"):
            user[name] = json.loads(value)
        else:
            meta[key] = value  # not a PHH field: keep it with the rest
    if meta:
        user[META_FIELD] = json.dumps(meta, sort_keys=True)
    dated = _started_at_fields(hand.started_at)
    user.update(dated.pop("user_defined_fields", {}))
    return dataclasses.replace(
        base,
        antes=[amount(a) for a in antes],
        blinds_or_straddles=[amount(b) for b in blinds],
        min_bet=amount(min_bet),
        starting_stacks=[amount(hand.seat_info(s).starting_stack) for s in order],
        actions=[_map_cbr(a, lambda s: str(amount(int(s)))) for a in actions],
        hand=hand.hand_id,
        table=t.table_id,
        seats=[s + 1 for s in order],
        seat_count=t.max_seats,
        players=[hand.seat_info(s).player_id for s in order],
        finishing_stacks=[amount(hand.seat_info(s).starting_stack + net[s]) for s in order],
        currency=code,
        user_defined_fields=user,
        **fields,
        **dated,
    )


def dumps_phh(hands: Iterable[HandHistory]) -> str:
    """The ``.phhs`` text of ``hands`` (numbered TOML tables)."""
    return PHH.dumps_all(hand_to_phh(h) for h in hands) + "\n"


def write_phh(hands: Iterable[HandHistory], path: str | Path) -> int:
    """Write ``hands`` to ``path``: one hand for ``.phh``, any number for ``.phhs``."""
    path = Path(path)
    hands = list(hands)
    if path.suffix == ".phh":
        if len(hands) != 1:
            raise ValueError("a .phh file holds exactly one hand; use .phhs")
        text = hand_to_phh(hands[0]).dumps() + "\n"
    elif path.suffix == ".phhs":
        text = dumps_phh(hands)
    else:
        raise ValueError(f"unknown PHH suffix {path.suffix!r}")
    path.write_text(text, encoding="utf-8", newline="\n")
    return len(hands)


# ---------------------------------------------------------------------------- batch import
@dataclass(frozen=True)
class ImportRejection:
    file: str
    index: int  # 0-based position of the hand in its file; -1 if the whole file is unreadable
    reason: RejectReason
    detail: str


@dataclass(frozen=True)
class ImportReport:
    hands: tuple[HandHistory, ...]
    rejected: tuple[ImportRejection, ...]
    files: int

    def counts(self) -> Counter[str]:
        """Number of rejected hands (or unreadable files) per reason."""
        return Counter(r.reason.value for r in self.rejected)

    def summary(self) -> str:
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(self.counts().items()))
        return (
            f"{len(self.hands)} hands imported from {self.files} files; "
            f"{len(self.rejected)} rejected" + (f" ({reasons})" if reasons else "")
        )


def phh_files(
    paths: Iterable[str | Path], *, root: str | Path | None = None
) -> list[tuple[Path, str]]:
    """(file, key) for every PHH file under ``paths``, sorted by key and listed once.
    Directories contribute their ``.phh``/``.phhs`` files recursively. A key is the file's
    path relative to ``root`` (default: the deepest directory containing all of ``paths``),
    without the suffix, so files with the same name in different folders get different keys."""
    paths = [Path(p).resolve() for p in paths]
    found: set[Path] = set()
    for p in paths:
        if p.is_dir():
            found.update(f for f in p.rglob("*") if f.suffix in SUFFIXES and f.is_file())
        elif p.suffix in SUFFIXES and p.is_file():
            found.add(p)
        else:
            raise ValueError(f"not a PHH file or directory: {p}")
    if not paths:
        return []
    if root is None:
        root = os.path.commonpath([p if p.is_dir() else p.parent for p in paths])
    root = Path(root).resolve()
    keyed = ((f, f.relative_to(root).with_suffix("").as_posix()) for f in found)
    return sorted(keyed, key=lambda fk: fk[1])


#: Every PHH field name; anything else in a file is a user-defined field (PokerKit's
#: settings such as ``rake`` or ``automations`` cannot be set from a file).
_PHH_FIELDS = frozenset(PHH.optional_field_names) | {
    name for names in PHH.required_field_names.values() for name in names
}


def _load(raw: Mapping[str, Any]) -> PHH:
    if not isinstance(raw, Mapping):
        raise _reject(RejectReason.PARSE, "a hand must be a TOML table")
    kwargs = {k: v for k, v in raw.items() if k in _PHH_FIELDS}
    kwargs["user_defined_fields"] = {k: v for k, v in raw.items() if k not in _PHH_FIELDS}
    try:
        return _PHH(**kwargs)
    except TypeError as exc:
        raise _reject(RejectReason.PARSE, str(exc)) from exc


def loads_phh(text: str, *, many: bool = False) -> list[PHH | PHHImportError]:
    """The hands in PHH text (``many=True`` for the numbered tables of a ``.phhs`` file), in
    order. A hand that is not valid PHH comes back as the error; text that is not TOML
    raises ``PHHImportError``."""
    try:
        raw = tomllib.loads(text, parse_float=parse_value)
    except tomllib.TOMLDecodeError as exc:
        raise _reject(RejectReason.PARSE, f"not TOML: {exc}") from exc
    out: list[PHH | PHHImportError] = []
    for table in raw.values() if many else [raw]:
        try:
            out.append(_load(table))
        except PHHImportError as exc:
            out.append(exc)
    return out


def read_phh(path: str | Path) -> list[PHH | PHHImportError]:
    """``loads_phh`` for one ``.phh`` or ``.phhs`` file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise _reject(RejectReason.PARSE, f"{path.name} is not UTF-8") from exc
    return loads_phh(text, many=path.suffix == ".phhs")


def import_phh(
    paths: Iterable[str | Path],
    *,
    ids: Literal["path", "field"] = "path",
    allow_rake: bool = True,
    root: str | Path | None = None,
) -> ImportReport:
    """Import every hand in ``paths`` (files or directories, see ``phh_files``), once.

    ``ids="path"`` names hands ``"<file key>:<hand field or index>"``, unique across files
    whose own hand numbers restart (the Pluribus files do); ``ids="field"`` uses the ``hand``
    field as is, as written by ``write_phh``. A hand id seen twice is rejected the second time.
    Accepted hands carry ``metadata["import.file"]``. ``allow_rake`` as in ``hand_from_phh``;
    ``root`` as in ``phh_files``.
    """
    files = phh_files(paths, root=root)
    hands: list[HandHistory] = []
    rejected: list[ImportRejection] = []
    seen: set[str] = set()
    for path, key in files:
        name = f"{key}{path.suffix}"
        try:
            loaded = read_phh(path)
        except PHHImportError as exc:
            rejected.append(ImportRejection(name, -1, exc.reason, exc.detail))
            continue
        for index, phh in enumerate(loaded):
            try:
                if isinstance(phh, PHHImportError):
                    raise phh
                if ids == "field":
                    hand_id = None
                else:
                    hand_id = f"{key}:{phh.hand if phh.hand is not None else index}"
                hand = hand_from_phh(phh, hand_id=hand_id, allow_rake=allow_rake)
                if hand.hand_id in seen:
                    raise _reject(RejectReason.DUPLICATE, f"hand id {hand.hand_id!r}")
            except PHHImportError as exc:
                rejected.append(ImportRejection(name, index, exc.reason, exc.detail))
                continue
            seen.add(hand.hand_id)
            hands.append(dataclasses.replace(hand, metadata={**hand.metadata, "import.file": name}))
    return ImportReport(tuple(hands), tuple(rejected), len(files))

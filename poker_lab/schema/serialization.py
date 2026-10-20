"""JSON / JSON Lines serialization for ``HandHistory``.

One hand per line, UTF-8. Paths ending in ``.gz`` are gzip-compressed transparently.
Decoding is strict: unknown or missing keys raise ``ValueError``, so a record cannot silently
drop or smuggle fields. In-progress records that contain hole cards are rejected because
``HandHistory`` itself forbids them.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import IO, Any

from poker_lab.schema.hand_history import (
    Action,
    CardObservation,
    ChipAward,
    HandHistory,
    HoleCards,
    SeatInfo,
    ShowdownEntry,
    TableConfig,
)
from poker_lab.schema.validation import validate_hand_history

_HAND_KEYS = frozenset(
    {
        "schema_version",
        "hand_id",
        "status",
        "source",
        "started_at",
        "table",
        "button_seat",
        "seats",
        "hole_cards",
        "board",
        "actions",
        "showdown",
        "payouts",
        "uncalled_returns",
        "rake",
        "metadata",
    }
)


def _check_keys(obj: Mapping[str, Any], allowed: frozenset[str], what: str) -> None:
    if not isinstance(obj, Mapping):
        raise ValueError(f"{what} must be a JSON object")
    extra = set(obj) - allowed
    if extra:
        raise ValueError(f"unknown keys in {what}: {sorted(extra)}")


def _card_to_dict(obs: CardObservation) -> dict[str, Any]:
    return {"card": obs.card, "source": obs.source.value, "confidence": obs.confidence}


_CARD_KEYS = frozenset({"card", "source", "confidence"})


def _card_from_dict(d: Mapping[str, Any]) -> CardObservation:
    _check_keys(d, _CARD_KEYS, "card observation")
    return CardObservation(card=d["card"], source=d["source"], confidence=d["confidence"])


def hand_to_dict(hand: HandHistory) -> dict[str, Any]:
    """Convert a hand to plain JSON-compatible Python objects."""
    t = hand.table
    return {
        "schema_version": hand.schema_version,
        "hand_id": hand.hand_id,
        "status": hand.status.value,
        "source": hand.source,
        "started_at": hand.started_at,
        "table": {
            "table_id": t.table_id,
            "max_seats": t.max_seats,
            "small_blind": t.small_blind,
            "big_blind": t.big_blind,
            "ante": t.ante,
            "game": t.game,
            "currency": t.currency,
        },
        "button_seat": hand.button_seat,
        "seats": [
            {
                "seat": s.seat,
                "player_id": s.player_id,
                "starting_stack": s.starting_stack,
                "position": s.position,
            }
            for s in hand.seats
        ],
        "hole_cards": [
            {
                "seat": hc.seat,
                "cards": None if hc.cards is None else [_card_to_dict(c) for c in hc.cards],
                "shown": hc.shown,
            }
            for hc in hand.hole_cards
        ],
        "board": [_card_to_dict(c) for c in hand.board],
        "actions": [
            {
                "seat": a.seat,
                "street": a.street.value,
                "type": a.action_type.value,
                "amount": a.amount,
                "total_bet": a.total_bet,
                "all_in": a.is_all_in,
            }
            for a in hand.actions
        ],
        "showdown": [{"seat": e.seat, "shown": e.shown} for e in hand.showdown],
        "payouts": [{"seat": p.seat, "amount": p.amount} for p in hand.payouts],
        "uncalled_returns": [{"seat": p.seat, "amount": p.amount} for p in hand.uncalled_returns],
        "rake": hand.rake,
        "metadata": dict(hand.metadata),
    }


_TABLE_KEYS = frozenset(
    {"table_id", "max_seats", "small_blind", "big_blind", "ante", "game", "currency"}
)
_SEAT_KEYS = frozenset({"seat", "player_id", "starting_stack", "position"})
_HOLE_KEYS = frozenset({"seat", "cards", "shown"})
_ACTION_KEYS = frozenset({"seat", "street", "type", "amount", "total_bet", "all_in"})
_SHOWDOWN_KEYS = frozenset({"seat", "shown"})
_AWARD_KEYS = frozenset({"seat", "amount"})


def hand_from_dict(data: Mapping[str, Any], *, validate: bool = False) -> HandHistory:
    """Build a ``HandHistory`` from ``hand_to_dict`` output (strict).

    With ``validate=True`` the deep consistency checks in ``validation.py`` also run.
    """
    _check_keys(data, _HAND_KEYS, "hand")
    missing = _HAND_KEYS - set(data)
    if missing:
        raise ValueError(f"missing keys in hand: {sorted(missing)}")
    _check_keys(data["table"], _TABLE_KEYS, "table")

    def awards(items: Iterable[Mapping[str, Any]], what: str) -> tuple[ChipAward, ...]:
        out = []
        for p in items:
            _check_keys(p, _AWARD_KEYS, what)
            out.append(ChipAward(seat=p["seat"], amount=p["amount"]))
        return tuple(out)

    seats = []
    for s in data["seats"]:
        _check_keys(s, _SEAT_KEYS, "seat")
        seats.append(SeatInfo(**s))
    hole_cards = []
    for hc in data["hole_cards"]:
        _check_keys(hc, _HOLE_KEYS, "hole_cards entry")
        cards = None if hc["cards"] is None else tuple(_card_from_dict(c) for c in hc["cards"])
        hole_cards.append(HoleCards(seat=hc["seat"], cards=cards, shown=hc["shown"]))
    actions = []
    for a in data["actions"]:
        _check_keys(a, _ACTION_KEYS, "action")
        actions.append(
            Action(
                seat=a["seat"],
                street=a["street"],
                action_type=a["type"],
                amount=a["amount"],
                total_bet=a["total_bet"],
                is_all_in=a["all_in"],
            )
        )
    showdown = []
    for e in data["showdown"]:
        _check_keys(e, _SHOWDOWN_KEYS, "showdown entry")
        showdown.append(ShowdownEntry(seat=e["seat"], shown=e["shown"]))

    hand = HandHistory(
        hand_id=data["hand_id"],
        table=TableConfig(**data["table"]),
        button_seat=data["button_seat"],
        seats=tuple(seats),
        actions=tuple(actions),
        board=tuple(_card_from_dict(c) for c in data["board"]),
        hole_cards=tuple(hole_cards),
        showdown=tuple(showdown),
        payouts=awards(data["payouts"], "payout"),
        uncalled_returns=awards(data["uncalled_returns"], "uncalled return"),
        rake=data["rake"],
        status=data["status"],
        source=data["source"],
        started_at=data["started_at"],
        metadata=data["metadata"],
        schema_version=data["schema_version"],
    )
    if validate:
        validate_hand_history(hand)
    return hand


def dumps_hand(hand: HandHistory) -> str:
    """Serialize one hand as a single compact JSON line (no trailing newline)."""
    return json.dumps(hand_to_dict(hand), separators=(",", ":"), ensure_ascii=False)


def loads_hand(line: str, *, validate: bool = False) -> HandHistory:
    return hand_from_dict(json.loads(line), validate=validate)


def _open(path: Path, mode: str) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")  # type: ignore[return-value]
    return path.open(mode, encoding="utf-8", newline="\n")


def write_jsonl(hands: Iterable[HandHistory], path: str | Path, *, append: bool = False) -> int:
    """Write hands to a JSON Lines file (gzip if the path ends in ``.gz``); return the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "a" if append else "w") as fh:
        for hand in hands:
            fh.write(dumps_hand(hand))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path, *, validate: bool = False) -> Iterator[HandHistory]:
    """Yield hands from a JSON Lines file. Blank lines are skipped."""
    path = Path(path)
    with _open(path, "r") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield loads_hand(line, validate=validate)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc

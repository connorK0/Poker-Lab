"""Card representation used across the schema.

Cards are two-character strings: a rank from ``RANKS`` followed by a suit from ``SUITS``
(``"As"``, ``"Td"``, ``"2c"``). Strings keep hand histories human-readable as JSON.

Each card also has an integer id ``rank_index * 4 + suit_index`` (deuce of clubs = 0, ace of
spades = 51). The id order matches phevaluator's, so fast numeric code can use the ids
directly.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

RANKS = "23456789TJQKA"
SUITS = "cdhs"

#: The 52 cards; ``FULL_DECK[card_id]`` is the card with that id.
FULL_DECK: tuple[str, ...] = tuple(rank + suit for rank in RANKS for suit in SUITS)

_CARD_TO_ID: dict[str, int] = {card: i for i, card in enumerate(FULL_DECK)}
_SPLIT_RE = re.compile(r"[\s,]+")


def is_valid_card(card: object) -> bool:
    """Return True if ``card`` is a canonical two-character card string."""
    return isinstance(card, str) and card in _CARD_TO_ID


def validate_card(card: object) -> str:
    """Return ``card`` unchanged if canonical, otherwise raise ``ValueError``."""
    if not is_valid_card(card):
        raise ValueError(
            f"invalid card {card!r}: expected a rank from {RANKS!r} followed by a suit from "
            f"{SUITS!r}, e.g. 'As'"
        )
    return card  # type: ignore[return-value]


def normalize_card(text: str) -> str:
    """Normalize a loosely written card (``"aS"``, ``"10h"``) to canonical form (``"As"``)."""
    raw = text.strip()
    if raw[:2] == "10":
        raw = "T" + raw[2:]
    if len(raw) != 2:
        raise ValueError(f"cannot parse card {text!r}")
    return validate_card(raw[0].upper() + raw[1].lower())


def parse_cards(text: str) -> tuple[str, ...]:
    """Parse ``"AsKd"``, ``"As Kd"`` or ``"As,Kd"`` into a tuple of canonical cards."""
    cards: list[str] = []
    for chunk in _SPLIT_RE.split(text.strip()):
        if not chunk:
            continue
        chunk = chunk.replace("10", "T")
        if len(chunk) % 2:
            raise ValueError(f"cannot parse cards from {text!r}")
        cards.extend(normalize_card(chunk[i : i + 2]) for i in range(0, len(chunk), 2))
    return tuple(cards)


def card_to_id(card: str) -> int:
    """Integer id of a canonical card (``rank_index * 4 + suit_index``)."""
    try:
        return _CARD_TO_ID[card]
    except KeyError:
        validate_card(card)
        raise


def id_to_card(card_id: int) -> str:
    """Canonical card string for an integer id in ``[0, 52)``."""
    if not 0 <= card_id < 52:
        raise ValueError(f"card id out of range: {card_id}")
    return FULL_DECK[card_id]


def card_rank_index(card: str) -> int:
    """Rank index of a card: 0 for a deuce up to 12 for an ace."""
    return RANKS.index(card[0])


def card_suit_index(card: str) -> int:
    """Suit index of a card: 0=c, 1=d, 2=h, 3=s."""
    return SUITS.index(card[1])


def assert_distinct(cards: Iterable[str]) -> None:
    """Raise ``ValueError`` if any card appears more than once."""
    seen: set[str] = set()
    for card in cards:
        if card in seen:
            raise ValueError(f"duplicate card {card!r}")
        seen.add(card)

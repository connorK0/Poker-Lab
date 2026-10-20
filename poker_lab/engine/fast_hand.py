"""A phevaluator-backed hand type plugged into PokerKit's ``hand_types`` extension point.

PokerKit's ``StandardHighHand.from_game`` tries every 5-card combination in pure Python and
takes ~0.75 ms per call; profiling showed ~80% of simulation time spent there.
``FastStandardHighHand`` keeps the same interface and ordering semantics, but ranks hands with
phevaluator's C perfect-hash evaluator. ``FastNoLimitTexasHoldem`` is PokerKit's
``NoLimitTexasHoldem`` with only the hand type swapped. Betting, side pots and pot splitting
are all still PokerKit's.

Equivalence with PokerKit's own evaluator is checked by differential tests
(``tests/test_engine.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pokerkit import Card, NoLimitTexasHoldem, StandardHighHand
from pokerkit.utilities import CardsLike

from poker_lab.schema.cards import FULL_DECK

try:  # direct C entry points (int arguments) avoid the wrappers' per-card conversion
    from phevaluator._pheval import evaluate_5cards as _ev5
    from phevaluator._pheval import evaluate_6cards as _ev6
    from phevaluator._pheval import evaluate_7cards as _ev7
except ImportError:  # pragma: no cover - fall back to the public API
    from phevaluator import evaluate_5cards as _ev5
    from phevaluator import evaluate_6cards as _ev6
    from phevaluator import evaluate_7cards as _ev7

_EVALUATORS: dict[int, Callable[..., int]] = {5: _ev5, 6: _ev6, 7: _ev7}
_PK_CARD_ID: dict[Card, int] = {next(iter(Card.parse(c))): i for i, c in enumerate(FULL_DECK)}


def phe_rank(card_ids: list[int] | tuple[int, ...]) -> int:
    """phevaluator rank of the best 5-card hand in 5-7 card ids (1 = royal flush; lower wins)."""
    return _EVALUATORS[len(card_ids)](*card_ids)


def _clean(cards: CardsLike) -> tuple[Card, ...]:
    """``Card.clean`` that also parses string elements inside an iterable."""
    out: list[Card] = []
    for card in Card.clean(cards):
        if isinstance(card, str):
            out.extend(Card.parse(card))
        else:
            out.append(card)
    return tuple(out)


class FastStandardHighHand(StandardHighHand):
    """``StandardHighHand`` with phevaluator ranking. Stronger hands compare greater."""

    _rank: int
    _fallback: ClassVar[type[StandardHighHand]] = StandardHighHand

    def __init__(self, cards: CardsLike) -> None:
        cards = _clean(cards)
        if len(cards) != 5 or not all(cards) or len(set(cards)) != 5:
            raise ValueError(f"The cards {cards!r} form an invalid {type(self).__qualname__}.")
        self._Hand__cards = cards  # same private slot the base class uses for ``cards``
        self._rank = phe_rank([_PK_CARD_ID[c] for c in cards])

    @classmethod
    def from_game(cls, hole_cards: CardsLike, board_cards: CardsLike = ()) -> FastStandardHighHand:
        cards = tuple(c for c in _clean(hole_cards) + _clean(board_cards) if c)
        if len(cards) > 7:  # not reachable in hold'em; keep PokerKit's general behaviour
            return super().from_game(cards)  # type: ignore[return-value]
        if len(cards) < 5 or len(set(cards)) != len(cards):
            raise ValueError("No valid hand can be formed from the hole and board cards.")
        hand = cls.__new__(cls)
        hand._Hand__cards = cards  # type: ignore[attr-defined]
        hand._rank = phe_rank([_PK_CARD_ID[c] for c in cards])
        return hand

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._rank == other._rank

    def __hash__(self) -> int:
        return hash(self._rank)

    def __lt__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._rank > other._rank  # phevaluator: lower rank = stronger hand

    @property
    def entry(self) -> Any:
        """PokerKit lookup entry (label etc.), computed with the slow evaluator on demand."""
        return self._fallback.from_game(self.cards).entry


class FastNoLimitTexasHoldem(NoLimitTexasHoldem):
    """PokerKit no-limit hold'em with the phevaluator-backed hand type."""

    hand_types = (FastStandardHighHand,)

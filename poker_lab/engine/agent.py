"""The ``Agent`` interface shared by bots, baselines and advisor-backed players.

An agent never sees engine internals. On its turn it receives a ``SeatView`` (public state, its
own hole cards, its legal options) and a seeded ``random.Random`` owned by the table, and returns
a ``Decision``. The engine translates the decision into the precise schema action (check vs.
call, bet vs. raise) and rejects illegal decisions with ``IllegalActionError``.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from poker_lab.schema import HandHistory, LegalActions, SeatView


class DecisionKind(StrEnum):
    FOLD = "fold"
    CHECK_CALL = "check_call"
    BET_RAISE = "bet_raise"


@dataclass(frozen=True, slots=True)
class Decision:
    """An agent's choice. ``to_amount`` is the total street bet after a bet/raise."""

    kind: DecisionKind
    to_amount: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DecisionKind):
            object.__setattr__(self, "kind", DecisionKind(self.kind))
        if type(self.to_amount) is not int:
            raise TypeError("to_amount must be an int")
        if self.kind is DecisionKind.BET_RAISE:
            if self.to_amount <= 0:
                raise ValueError("a bet/raise needs a positive to_amount")
        elif self.to_amount != 0:
            raise ValueError("to_amount is only used for bet/raise decisions")

    @classmethod
    def fold(cls) -> Decision:
        return cls(DecisionKind.FOLD)

    @classmethod
    def check_call(cls) -> Decision:
        return cls(DecisionKind.CHECK_CALL)

    @classmethod
    def bet_raise(cls, to_amount: int) -> Decision:
        return cls(DecisionKind.BET_RAISE, to_amount)


class IllegalActionError(ValueError):
    """An agent returned a decision that is not legal in the current state."""


def check_decision(decision: Decision, legal: LegalActions) -> None:
    """Raise ``IllegalActionError`` unless ``decision`` is legal under ``legal``."""
    if not isinstance(decision, Decision):
        raise IllegalActionError(f"agents must return a Decision, got {type(decision).__name__}")
    if decision.kind is DecisionKind.FOLD and not legal.can_fold:
        raise IllegalActionError("folding is not allowed when checking is free")
    if decision.kind is DecisionKind.BET_RAISE:
        if not legal.can_bet_or_raise:
            raise IllegalActionError("betting/raising is not allowed here")
        if not legal.min_raise_to <= decision.to_amount <= legal.max_raise_to:
            raise IllegalActionError(
                f"raise to {decision.to_amount} outside "
                f"[{legal.min_raise_to}, {legal.max_raise_to}]"
            )


def clamp_raise(legal: LegalActions, desired_to: float) -> int:
    """Round a desired bet/raise size to the nearest legal ``to`` amount."""
    return int(min(max(round(desired_to), legal.min_raise_to), legal.max_raise_to))


class Agent(ABC):
    """Base class for anything that plays a seat."""

    @abstractmethod
    def act(self, view: SeatView, rng: random.Random) -> Decision:
        """Choose an action. ``view.legal`` is always set when this is called."""

    def hand_completed(self, hand: HandHistory) -> None:  # noqa: B027 - optional hook
        """Called after each hand with the record as this seat would know it: hole cards that
        were not shown at showdown are redacted, except the agent's own. Default: no-op."""

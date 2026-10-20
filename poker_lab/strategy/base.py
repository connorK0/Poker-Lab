"""``StrategyProvider``: what to do at a decision point, as a mixed strategy.

A strategy provider maps a ``SeatView`` (public state + the hero's own cards + legal options;
see ``poker_lab.schema.views``) to an ``ActionDistribution`` over concrete legal
``Decision``s. Phase 2 will plug in providers backed by CFR / Deep CFR policies. Those are
solved on abstracted games, so the action abstraction (bet sizes) lives inside the provider,
and this interface stays the same.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from poker_lab.engine.agent import Decision, DecisionKind, check_decision
from poker_lab.schema import LegalActions, SeatView

_TOL = 1e-9


@dataclass(frozen=True, slots=True)
class ActionOption:
    decision: Decision
    probability: float


@dataclass(frozen=True, slots=True)
class ActionDistribution:
    """A probability distribution over distinct concrete decisions (probabilities sum to 1)."""

    options: tuple[ActionOption, ...]

    def __post_init__(self) -> None:
        opts = tuple(self.options)
        if not opts:
            raise ValueError("an action distribution needs at least one option")
        seen = set()
        for o in opts:
            if not isinstance(o, ActionOption) or not isinstance(o.decision, Decision):
                raise TypeError("options must be ActionOption(Decision, probability)")
            if not 0.0 <= o.probability <= 1.0 + _TOL:
                raise ValueError(f"probability out of range: {o.probability}")
            if o.decision in seen:
                raise ValueError(f"duplicate decision {o.decision}")
            seen.add(o.decision)
        total = sum(o.probability for o in opts)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probabilities sum to {total}, not 1")
        object.__setattr__(self, "options", opts)

    @classmethod
    def from_weights(cls, weights: Mapping[Decision, float] | Iterable[tuple[Decision, float]]):
        """Normalize non-negative weights; zero-weight options are dropped."""
        items = list(weights.items() if isinstance(weights, Mapping) else weights)
        total = sum(w for _, w in items)
        if total <= 0 or any(w < 0 for _, w in items):
            raise ValueError("weights must be non-negative with a positive sum")
        merged: dict[Decision, float] = {}
        for d, w in items:
            if w > 0:
                merged[d] = merged.get(d, 0.0) + w / total
        return cls(tuple(ActionOption(d, p) for d, p in merged.items()))

    @classmethod
    def pure(cls, decision: Decision) -> ActionDistribution:
        return cls((ActionOption(decision, 1.0),))

    def probability_of(self, kind: DecisionKind) -> float:
        return sum(o.probability for o in self.options if o.decision.kind is kind)

    def most_likely(self) -> Decision:
        return max(self.options, key=lambda o: o.probability).decision

    def sample(self, rng: random.Random) -> Decision:
        u, acc = rng.random(), 0.0
        for o in self.options:
            acc += o.probability
            if u < acc:
                return o.decision
        return self.options[-1].decision

    def validate_against(self, legal: LegalActions) -> None:
        """Raise ``IllegalActionError`` if any option with positive probability is illegal."""
        for o in self.options:
            if o.probability > 0:
                check_decision(o.decision, legal)

    def total_variation(self, other: ActionDistribution) -> float:
        """Total-variation distance between two distributions (0 = identical, 1 = disjoint)."""
        p = {o.decision: o.probability for o in self.options}
        q = {o.decision: o.probability for o in other.options}
        return 0.5 * sum(abs(p.get(d, 0.0) - q.get(d, 0.0)) for d in set(p) | set(q))


class StrategyProvider(ABC):
    """Maps a seat's view of the current hand to a mixed strategy over its legal decisions.

    Contract (checked by ``tests/test_strategy_exploit.py`` for every implementation):
    * the input is exactly a ``SeatView`` whose ``legal`` field is set;
    * every option with positive probability is legal under ``view.legal``;
    * the call has no side effects on the view (all inputs are immutable).
    """

    name: str = "strategy"

    @abstractmethod
    def action_distribution(self, view: SeatView) -> ActionDistribution:
        """Mixed strategy for the seat in ``view`` (``view.legal`` is not None)."""

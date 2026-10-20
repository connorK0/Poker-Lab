"""Simple agents used by tests, smoke runs and as opponents for baselines."""

from __future__ import annotations

import random
from collections.abc import Sequence

from poker_lab.engine.agent import Agent, Decision
from poker_lab.schema import SeatView


class RandomAgent(Agent):
    """Uniform over {fold (if legal), check/call, bet/raise (if legal)}; raise size uniform
    between the minimum and all-in, with extra mass on min-raises and all-ins so that edge
    cases (short all-ins, side pots) appear often."""

    def __init__(self, fold_weight: float = 1.0, raise_weight: float = 1.0) -> None:
        self.fold_weight = fold_weight
        self.raise_weight = raise_weight

    def act(self, view: SeatView, rng: random.Random) -> Decision:
        legal = view.legal
        assert legal is not None
        kinds, weights = ["call"], [1.0]
        if legal.can_fold:
            kinds.append("fold")
            weights.append(self.fold_weight)
        if legal.can_bet_or_raise:
            kinds.append("raise")
            weights.append(self.raise_weight)
        kind = rng.choices(kinds, weights)[0]
        if kind == "fold":
            return Decision.fold()
        if kind == "call":
            return Decision.check_call()
        r = rng.random()
        if r < 0.3:
            to = legal.min_raise_to
        elif r < 0.45:
            to = legal.max_raise_to
        else:
            to = rng.randint(legal.min_raise_to, legal.max_raise_to)
        return Decision.bet_raise(to)


class CallingAgent(Agent):
    """Always checks or calls."""

    def act(self, view: SeatView, rng: random.Random) -> Decision:
        return Decision.check_call()


class AllInAgent(Agent):
    """Moves all-in whenever it can, otherwise calls."""

    def act(self, view: SeatView, rng: random.Random) -> Decision:
        legal = view.legal
        assert legal is not None
        if legal.can_bet_or_raise:
            return Decision.bet_raise(legal.max_raise_to)
        return Decision.check_call()


class ScriptedAgent(Agent):
    """Replays a fixed list of decisions (for hand-crafted test scenarios)."""

    def __init__(self, script: Sequence[Decision]) -> None:
        self.script = list(script)
        self.views: list[SeatView] = []

    def act(self, view: SeatView, rng: random.Random) -> Decision:
        self.views.append(view)
        if not self.script:
            raise RuntimeError("script exhausted")
        return self.script.pop(0)

"""The firewalled advice entry point.

``Advisor.advise(view, profiles)`` is the only way strategy advice is produced, and its inputs
are restricted by type:

* ``view`` must be exactly a ``SeatView``: public information plus the hero's own two hole
  cards. ``SeatView`` has no field for anyone else's cards, rejects subclasses and lookalike
  containers, and is frozen and slotted. The ``HandHistoryStore`` builds it from sealed cards
  using the hero's ``SeatKey``.
* ``profiles`` must be exactly a ``ProfileSnapshot`` built by a ``ProfileStore``, which ingests
  completed hands only. The advisor also refuses a snapshot whose source hands include the
  hand being played.

The advisor holds no reference to any hand store, so no in-progress hole cards are reachable
from it (``tests/test_advisor_firewall.py`` walks the object graph to check).

Scope: offline research, simulation and post-session review. This project provides no
real-time assistance for real-money online play.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from poker_lab.engine.agent import Agent, Decision
from poker_lab.exploit.base import ExploitabilityEstimate, ExploitEngine
from poker_lab.exploit.engines import IdentityExploitEngine
from poker_lab.profiling.profile_store import ProfileSnapshot, ProfileStore
from poker_lab.schema import HandHistory, SeatView
from poker_lab.strategy.base import ActionDistribution, StrategyProvider


class FirewallViolation(ValueError):
    """Advisor inputs would expose information the advisor must not use."""


@dataclass(frozen=True)
class Advice:
    hand_id: str
    seat: int
    recommended: Decision
    strategy: ActionDistribution
    baseline: ActionDistribution
    exploitability: ExploitabilityEstimate
    rationale: str
    opponents_profiled: tuple[str, ...]


class Advisor:
    def __init__(
        self,
        strategy: StrategyProvider,
        exploit: ExploitEngine | None = None,
        *,
        sample: bool = False,
        seed: int = 0,
    ) -> None:
        """``sample=True`` draws the recommendation from the strategy (mixed play); otherwise
        the most likely decision is recommended."""
        if not isinstance(strategy, StrategyProvider):
            raise TypeError("strategy must be a StrategyProvider")
        if exploit is not None and not isinstance(exploit, ExploitEngine):
            raise TypeError("exploit must be an ExploitEngine")
        self.strategy = strategy
        self.exploit = exploit or IdentityExploitEngine()
        self.sample = sample
        self._rng = random.Random(seed)

    @staticmethod
    def check_inputs(view: SeatView, profiles: ProfileSnapshot) -> None:
        if type(view) is not SeatView:
            raise TypeError(
                f"Advisor accepts exactly a SeatView (public state + own cards), got "
                f"{type(view).__name__}"
            )
        if type(profiles) is not ProfileSnapshot:
            raise TypeError(f"profiles must be a ProfileSnapshot, got {type(profiles).__name__}")
        if view.legal is None:
            raise ValueError("advice is only given to the seat whose turn it is")
        if view.public.hand_id in profiles.source_hand_ids:
            raise FirewallViolation(
                f"profiles were built from hand {view.public.hand_id!r}, which is being played"
            )

    def advise(self, view: SeatView, profiles: ProfileSnapshot) -> Advice:
        self.check_inputs(view, profiles)
        legal = view.legal
        assert legal is not None
        baseline = self.strategy.action_distribution(view)
        baseline.validate_against(legal)
        result = self.exploit.adjust(view, baseline, profiles)
        result.strategy.validate_against(legal)
        recommended = (
            result.strategy.sample(self._rng) if self.sample else result.strategy.most_likely()
        )
        opponents = tuple(
            s.player_id
            for s in view.public.seats
            if s.seat != view.seat and s.player_id in profiles
        )
        return Advice(
            hand_id=view.public.hand_id,
            seat=view.seat,
            recommended=recommended,
            strategy=result.strategy,
            baseline=baseline,
            exploitability=result.exploitability,
            rationale=result.rationale,
            opponents_profiled=opponents,
        )


class AdvisorAgent(Agent):
    """Plays a seat in simulation by following an ``Advisor``. Completed (redacted) hands are
    fed to its ``ProfileStore``, so profiles grow online, from completed hands only."""

    def __init__(self, advisor: Advisor, store: ProfileStore | None = None) -> None:
        self.advisor = advisor
        self.store = store or ProfileStore()
        self.advice_log: list[Advice] = []
        self.keep_log = True
        self._snapshot: ProfileSnapshot | None = None

    def act(self, view: SeatView, rng: random.Random) -> Decision:
        if self._snapshot is None:  # profiles only change between hands
            self._snapshot = self.store.snapshot(s.player_id for s in view.public.seats)
        advice = self.advisor.advise(view, self._snapshot)
        if self.keep_log:
            self.advice_log.append(advice)
        return advice.recommended

    def hand_completed(self, hand: HandHistory) -> None:
        self.store.ingest(hand)
        self._snapshot = None

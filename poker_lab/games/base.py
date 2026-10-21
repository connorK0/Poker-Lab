"""``Game``: a finite two-player zero-sum game with chance moves, described by pure functions.

Phase 2 solves small poker games *exactly*. A ``Game`` only describes the rules: immutable
states plus functions on them. ``poker_lab.games.tree.compile_tree`` walks the game once and
produces flat NumPy arrays; every solver, best response and evaluator runs on those arrays.

Conventions:
* players are 0 and 1; chance moves are made by ``CHANCE``;
* ``utility`` is player 0's payoff at a terminal state (player 1 receives the negative);
* ``information_set`` returns a key that encodes exactly what the acting player knows (their
  own private cards, public cards and the betting so far). Two states share a key if and only
  if the acting player cannot tell them apart. Solvers only ever see these keys, so a policy
  structurally cannot condition on the opponent's private cards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable

CHANCE = -1


class Game(ABC):
    """Two-player zero-sum extensive-form game with chance and perfect recall."""

    name: str = "game"

    @abstractmethod
    def initial_state(self) -> Hashable:
        """The root state (usually a chance node that deals cards)."""

    @abstractmethod
    def is_terminal(self, state: Hashable) -> bool: ...

    @abstractmethod
    def current_player(self, state: Hashable) -> int:
        """0 or 1 at decision states, ``CHANCE`` at chance states (undefined at terminals)."""

    @abstractmethod
    def legal_actions(self, state: Hashable) -> tuple[str, ...]:
        """Action labels at a decision state, in a fixed order."""

    @abstractmethod
    def chance_outcomes(self, state: Hashable) -> tuple[tuple[str, float], ...]:
        """``(label, probability)`` pairs at a chance state; probabilities sum to 1."""

    @abstractmethod
    def next_state(self, state: Hashable, action: str) -> Hashable:
        """The state after ``action`` (a legal action label or a chance outcome label)."""

    @abstractmethod
    def utility(self, state: Hashable) -> float:
        """Player 0's payoff at a terminal state."""

    @abstractmethod
    def information_set(self, state: Hashable) -> str:
        """Key of the acting player's information set at a decision state."""

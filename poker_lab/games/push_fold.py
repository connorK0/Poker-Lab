"""Heads-up push/fold no-limit hold'em: the first hold'em game this project solves exactly (P3.1).

Rules (the standard push/fold abstraction, as in HoldemResources' heads-up Nash charts):

* two players, blinds 0.5 and 1 bb, no ante, effective stack ``S`` bb before posting blinds;
* the small blind acts first and either folds (losing its 0.5) or goes all-in;
* facing the all-in, the big blind folds (losing its 1) or calls; the board is then dealt and
  the better hand wins the pot of ``2 S`` (ties split).

The deal is one chance move over the 169 x 169 starting-hand class pairs with exact
card-removal probabilities (``preflop.Matchups``). A called all-in is worth ``S (2 e - 1)`` to
the small blind in expectation, ``e`` being the class matchup equity, so no board needs to be
dealt in the tree. Each player has 169 information sets (one per class); the tree has about
143,000 nodes. Payoffs are in big blinds.

Only the called-all-in payoffs depend on ``S``, and linearly. ``push_fold_tree`` therefore
compiles the game once (``S = 1``) and rescales those leaves, so a sweep over hundreds of stack
sizes costs a single compile.

Information-set keys: ``"AKs|"`` (small blind holding AKs, first to act) and ``"72o|p"`` (big
blind holding 72o, facing the all-in). Actions: ``f`` fold, ``p`` push, ``c`` call.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cache

import numpy as np

from poker_lab.games.base import CHANCE, Game
from poker_lab.games.policy import TabularPolicy
from poker_lab.games.preflop import (
    CLASS_INDEX,
    CLASSES,
    N_COMBOS,
    Matchups,
    combo_counts,
    load_matchups,
)
from poker_lab.games.tree import TERMINAL, GameTree, compile_tree

SB, BB = 0, 1
_TERMINAL_HISTORIES = frozenset({"f", "pf", "pc"})


@dataclass(frozen=True, slots=True)
class PushFoldState:
    deal: tuple[int, int] | None = None  # (small blind's class, big blind's class)
    history: str = ""


class PushFoldGame(Game):
    name = "push_fold"

    def __init__(self, stack: float = 1.0, matchups: Matchups | None = None) -> None:
        if stack < 1.0:
            raise ValueError("the effective stack must cover the big blind (S >= 1)")
        self.stack = float(stack)
        self.matchups = matchups or load_matchups()
        p = self.matchups.deal_prob
        n = len(CLASSES)
        self._outcomes = tuple(
            (f"{CLASSES[i]}|{CLASSES[j]}", float(p[i, j]))
            for i in range(n)
            for j in range(n)
            if p[i, j] > 0
        )

    def initial_state(self) -> PushFoldState:
        return PushFoldState()

    def is_terminal(self, state: PushFoldState) -> bool:
        return state.history in _TERMINAL_HISTORIES

    def current_player(self, state: PushFoldState) -> int:
        if state.deal is None:
            return CHANCE
        return SB if state.history == "" else BB

    def legal_actions(self, state: PushFoldState) -> tuple[str, ...]:
        return ("f", "p") if state.history == "" else ("f", "c")

    def chance_outcomes(self, state: PushFoldState) -> tuple[tuple[str, float], ...]:
        return self._outcomes

    def next_state(self, state: PushFoldState, action: str) -> PushFoldState:
        if state.deal is None:
            sb, bb = action.split("|")
            return PushFoldState((CLASS_INDEX[sb], CLASS_INDEX[bb]))
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal action {action!r} at {state}")
        return PushFoldState(state.deal, state.history + action)

    def utility(self, state: PushFoldState) -> float:
        if state.history == "f":
            return -0.5
        if state.history == "pf":
            return 1.0
        i, j = state.deal
        return self.stack * (2.0 * self.matchups.equity[i, j] - 1.0)

    def information_set(self, state: PushFoldState) -> str:
        i, j = state.deal
        return f"{CLASSES[i]}|" if state.history == "" else f"{CLASSES[j]}|p"


@cache
def _unit_tree(matchups: Matchups) -> tuple[GameTree, np.ndarray]:
    tree = compile_tree(PushFoldGame(1.0, matchups))
    called = (tree.player == TERMINAL) & np.array([lab == "c" for lab in tree.labels])
    return tree, called


def push_fold_tree(stack: float, matchups: Matchups | None = None) -> GameTree:
    """The compiled game at effective stack ``stack`` (bb), sharing one compiled structure."""
    if stack < 1.0:
        raise ValueError("the effective stack must cover the big blind (S >= 1)")
    unit, called = _unit_tree(matchups or load_matchups())
    utility = unit.utility.copy()
    utility[called] *= stack
    return dataclasses.replace(unit, game_name=f"push_fold_{stack:g}bb", utility=utility)


def infoset_key(cls: str, player: int) -> str:
    return f"{cls}|" if player == SB else f"{cls}|p"


def aggressive_probabilities(policy: TabularPolicy, player: int) -> np.ndarray:
    """Per class (``CLASSES`` order): the small blind's push or the big blind's call probability."""
    action = "p" if player == SB else "c"
    return np.array([policy.action_probabilities(infoset_key(c, player))[action] for c in CLASSES])


def range_share(probs: np.ndarray) -> float:
    """Share of the 1,326 combos played, given a probability per class."""
    return float(np.dot(probs, combo_counts()) / N_COMBOS)


def threshold_policy(
    tree: GameTree, push: np.ndarray | None = None, call: np.ndarray | None = None
) -> TabularPolicy:
    """A policy from per-class push and call probabilities (uniform where not given)."""
    mapping = {}
    for k, c in enumerate(CLASSES):
        if push is not None:
            mapping[infoset_key(c, SB)] = {"p": float(push[k]), "f": 1.0 - float(push[k])}
        if call is not None:
            mapping[infoset_key(c, BB)] = {"c": float(call[k]), "f": 1.0 - float(call[k])}
    return TabularPolicy.from_mapping(tree, mapping, default="uniform")

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

    def observer_key(self, state: PushFoldState, hidden: int) -> str:
        if state.deal is None:
            return "|"
        sb, bb = ("?" if p == hidden else CLASSES[c] for p, c in enumerate(state.deal))
        return f"{sb},{bb}|{state.history}"


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


def hand_gains(
    stack: float, push: np.ndarray, call: np.ndarray, matchups: Matchups | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """What each hand gains by playing instead of folding, in bb per time it is dealt.

    ``sb[i]``: EV(push) - EV(fold) for the small blind holding class ``i`` against the big
    blind's per-class call probabilities ``call``. ``bb[j]``: EV(call) - EV(fold) for the big
    blind holding ``j`` facing a push from the per-class push probabilities ``push`` (``nan``
    if ``j`` never faces a push). At a Nash equilibrium every hand that plays has gain >= 0 and
    every hand that folds has gain <= 0.
    """
    m = matchups or load_matchups()
    counts = m.counts.astype(float)
    called = stack * (2.0 * m.equity - 1.0)  # small blind's payoff when called, per (i, j)
    p_opp = counts / counts.sum(axis=1, keepdims=True)  # P(big blind holds j | small blind i)
    sb = (p_opp * (call[None, :] * called + (1.0 - call[None, :]))).sum(axis=1) + 0.5
    w = counts * push[:, None]  # small blind classes that push, weighted by card removal
    total = w.sum(axis=0)
    ev_call = -(w * called).sum(axis=0) / np.where(total > 0, total, 1.0)
    bb = np.where(total > 0, ev_call + 1.0, np.nan)
    return sb, bb


@cache
def _value_terms(matchups: Matchups) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(r, P, Q): row marginals of the deal, the deal probabilities, P * (2 e - 1)."""
    p = matchups.deal_prob
    return p.sum(axis=1), p, p * (2.0 * matchups.equity - 1.0)


def match_values(
    stack: float,
    hero_push: np.ndarray,
    hero_call: np.ndarray,
    opp_push: np.ndarray,
    opp_call: np.ndarray,
    matchups: Matchups | None = None,
) -> np.ndarray:
    """Hero's expected bb/hand against each opponent, averaged over both seats (closed form).

    Strategies are per-class push and call probabilities: hero arrays ``(M, 169)``, opponent
    arrays ``(K, 169)``; the result is ``(M, K)`` (1-D inputs drop their axis). It equals
    ``evaluate.head_to_head`` on the tree. With deal probabilities ``P``, ``r_i = sum_j P_ij``
    and ``D = P * (S (2 e - 1) - 1)``, the small blind's value is
    ``u(s, c) = r . (1.5 s - 0.5) + s^T D c``, and the hero's is
    ``(u(s_h, c_o) - u(s_o, c_h)) / 2``.
    """
    r, p, q = _value_terms(matchups or load_matchups())
    d = stack * q - p
    hp, hc = np.atleast_2d(hero_push), np.atleast_2d(hero_call)
    op, oc = np.atleast_2d(opp_push), np.atleast_2d(opp_call)
    as_sb = (hp * 1.5 - 0.5) @ r
    opp_as_sb = (op * 1.5 - 0.5) @ r
    out = 0.5 * (as_sb[:, None] + hp @ d @ oc.T - opp_as_sb[None, :] - (op @ d @ hc.T).T)
    if np.ndim(opp_push) == 1:
        out = out[:, 0]
    if np.ndim(hero_push) == 1:
        out = out[0]
    return out


def strategy_exploitability(
    stack: float, push: np.ndarray, call: np.ndarray, matchups: Matchups | None = None
) -> np.ndarray | float:
    """Exact exploitability (NashConv / 2, bb/hand) of per-class ranges, in closed form.

    A best-responding opponent calls the hero's push ``s`` with class ``j`` when
    ``(s^T D)_j < 0`` and pushes class ``i`` against the hero's calls ``c`` when
    ``1.5 r_i + (D c)_i > 0``. Equals ``evaluate.exploitability`` on the tree; ``push`` and
    ``call`` may be ``(M, 169)`` for M strategies at once.
    """
    r, p, q = _value_terms(matchups or load_matchups())
    d = stack * q - p
    s, c = np.atleast_2d(push), np.atleast_2d(call)
    vs_push = -(s * 1.5 - 0.5) @ r - np.minimum(0.0, s @ d).sum(axis=1)
    vs_call = -0.5 * r.sum() + np.maximum(0.0, 1.5 * r[None, :] + c @ d.T).sum(axis=1)
    out = 0.5 * (vs_push + vs_call)
    return float(out[0]) if np.ndim(push) == 1 else out


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

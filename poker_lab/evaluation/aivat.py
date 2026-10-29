"""AIVAT: unbiased, low-variance win rates for a player whose strategy is known (P2.11).

A hero whose strategy we know plays an opponent whose strategy we do not. A hand ends at a
terminal history ``z`` and pays the hero ``u(z)``. The plain estimate of the hero's expected
value, the mean of ``u`` over hands, is noisy because of card luck.

**Control variates.** Along a hand, every step taken by a *known* actor (chance, or the hero)
moves the evaluator's information from ``S_t`` to ``S_{t+1}`` with probabilities the evaluator
can compute. For any value function ``V``, the term ``V(S_{t+1}) - E[V(S_{t+1}) | S_t]`` has
expectation zero, so subtracting the sum of these terms from ``u`` leaves the mean unchanged. If
``V`` is close to the true expected value, the terms track the luck of each step and the
difference varies far less. Opponent steps get no term: their probabilities are unknown.

**Imaginary observations.** Knowing the hero's strategy, the evaluator can also average over the
hero's private cards: every card the hero could have held, weighted by how likely chance was to
deal it *and* the hero to play the observed actions with it. The opponent's reach is the same for
all of them (it cannot see the hero's cards), so these weights are the exact posterior. The
result is replaced by that average, and ``V`` at each step is the same average of a value
function over full histories. Burch et al. (2018) call the combination AIVAT.

**Information classes.** Both averages run over nodes an observer cannot tell apart:
``observer_classes(game, tree, hidden)`` groups nodes by ``Game.observer_key(state, hidden)``.

* ``hidden = hero``: the observer sees the opponent's cards, as after a match where every hand
  is shown, and averages over the hero's. Chance and hero steps have known probabilities.
* ``hidden = opponent``: the observer is the hero, as in a home game where folded cards stay
  hidden. A chance step's probabilities are then known only when they do not depend on the
  opponent's hidden cards (``known_transitions``), and the result cannot be averaged.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from poker_lab.games.base import CHANCE, Game
from poker_lab.games.tree import TERMINAL, GameTree

_TOL = 1e-12


def observer_classes(game: Game, tree: GameTree, hidden: int) -> np.ndarray:
    """Class id per node of ``tree``: nodes share a class iff their observer keys agree.

    Walks ``game`` in ``compile_tree``'s breadth-first order (checking the labels). Every class
    must hold nodes of one kind (one actor, or terminals), and their parents must share a class:
    the observer never forgets what it saw.
    """
    ids: dict[str, int] = {}
    classes = np.empty(tree.num_nodes, dtype=np.int64)
    queue = deque([(game.initial_state(), "")])
    node = 0
    while queue:
        state, label = queue.popleft()
        if node >= tree.num_nodes or tree.labels[node] != label:
            raise ValueError("the game does not match the compiled tree")
        classes[node] = ids.setdefault(game.observer_key(state, hidden), len(ids))
        node += 1
        if game.is_terminal(state):
            continue
        if game.current_player(state) == CHANCE:
            labels = [out for out, _ in game.chance_outcomes(state)]
        else:
            labels = game.legal_actions(state)
        queue.extend((game.next_state(state, a), a) for a in labels)
    if node != tree.num_nodes:
        raise ValueError("the game does not match the compiled tree")
    _check_classes(tree, classes)
    return classes


def _check_classes(tree: GameTree, classes: np.ndarray) -> None:
    k = int(classes.max()) + 1
    if np.unique(np.stack([classes, tree.player]), axis=1).shape[1] != k:
        raise ValueError("an information class mixes actors or terminals")
    child = np.arange(1, tree.num_nodes)
    pairs = np.unique(np.stack([classes[child], classes[tree.parent[child]]]), axis=1)
    if pairs.shape[1] != len(np.unique(classes[child])) or classes[0] in classes[1:]:
        raise ValueError("nodes of an information class have parents in different classes")


def class_actor(tree: GameTree, classes: np.ndarray) -> np.ndarray:
    """Who acts at each class: 0, 1, ``CHANCE`` or ``TERMINAL``."""
    _, first = np.unique(classes, return_index=True)
    return tree.player[first]


def known_transitions(tree: GameTree, classes: np.ndarray, hero: int) -> np.ndarray:
    """Per class: whether the hero, seeing only ``classes``, knows the step's probabilities.

    The hero's own decisions always qualify. A chance step does when no node of the class has an
    opponent decision above it (the opponent's reach is 1 at all of them), or when every node of
    the class leads to each child class with the same probability. Opponent steps never do.
    """
    k = int(classes.max()) + 1
    actor = class_actor(tree, classes)
    size = np.bincount(classes, minlength=k)
    acted = np.bincount(classes, weights=tree.own_seq[1 - hero] >= 0, minlength=k) > 0
    # probability of reaching each child class from each node of a chance class
    child = np.flatnonzero(tree.player[tree.parent.clip(0)] == CHANCE)
    child = child[child > 0]
    pair, at = np.unique(
        tree.parent[child].astype(np.int64) * k + classes[child], return_inverse=True
    )
    prob = np.bincount(at, weights=tree.chance_prob[child])
    group, gi = np.unique(classes[pair // k] * k + pair % k, return_inverse=True)
    count = np.bincount(gi)
    high = np.zeros(len(group))
    np.maximum.at(high, gi, prob)
    low = np.full(len(group), np.inf)
    np.minimum.at(low, gi, prob)
    same = (count == size[group // k]) & (high - low <= _TOL)
    uniform = np.ones(k, dtype=bool)
    np.logical_and.at(uniform, group // k, same)
    return (actor == hero) | ((actor == CHANCE) & (~acted | uniform))


@dataclass(frozen=True)
class ControlVariates:
    """Output of ``control_variates``, in the hero's payoff."""

    estimate: np.ndarray  # the corrected result at each terminal (nan elsewhere)
    correction: np.ndarray  # sum of the control-variate terms from the root to each node
    root_value: float  # V at the root: the value function's prediction before the deal


def _average(classes: np.ndarray, weight: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    den = np.bincount(classes, weights=weight, minlength=k)
    num = np.bincount(classes, weights=weight * x, minlength=k)
    return np.divide(num, den, out=np.full(k, np.nan), where=den > 0)


def control_variates(
    tree: GameTree,
    hero: int,
    edge: np.ndarray,
    value: np.ndarray,
    classes: np.ndarray,
    corrected: np.ndarray,
    *,
    imaginary: bool,
    value_weight: np.ndarray | None = None,
) -> ControlVariates:
    """The hero's AIVAT estimate at every terminal.

    * ``edge``: edge probabilities (``GameTree.edge_probabilities``) of a profile whose chance
      and hero parts are the real ones. The opponent's entries are read only for steps out of a
      class with ``corrected`` set, which is right only when its strategy is known as well.
    * ``value``: any value per node, in the hero's payoff. The estimate is unbiased whatever it
      is; how close it is to the truth sets how much variance goes.
    * ``classes``: information classes (``observer_classes``). Within a class, nodes are
      weighted by ``w``, chance's reach times the hero's.
    * ``corrected``: per class, whether the step out of it gets a term. The term uses the
      probabilities ``W(S') / W(S)`` (``W`` sums ``w`` over a class; the opponent's own
      probabilities at its steps), which are exact when the opponent's reach is the same at
      every node of ``S``: always for ``hidden = hero``, see ``known_transitions`` otherwise.
    * ``imaginary``: replace each result by its ``w``-weighted average over the terminal's class
      (only for ``hidden = hero``, where a class's nodes differ only in the hero's cards).
    * ``value_weight``: weights for averaging ``value`` within a class (default ``w``). The
      hero's own view puts a model of the opponent's reach here; nodes of a class the model
      never reaches fall back to ``w``.
    """
    if hero not in (0, 1):
        raise ValueError("hero must be 0 or 1")
    n, k = tree.num_nodes, int(classes.max()) + 1
    if classes.shape != (n,) or corrected.shape != (k,):
        raise ValueError("need one class per node and one corrected flag per class")
    u = (1.0 if hero == 0 else -1.0) * tree.utility
    opponent = tree.edge_player == 1 - hero
    w = tree.reach(np.where(opponent, 1.0, edge))
    big_w = np.bincount(classes, weights=w, minlength=k)
    v = _average(classes, w, value, k)
    if value_weight is not None:
        modeled = _average(classes, value_weight, value, k)
        v = np.where(np.isnan(modeled), v, modeled)
    v = np.nan_to_num(v)  # classes the hero never reaches: no term is ever taken there

    child = np.arange(1, n)
    par = tree.parent[child]
    on = corrected[classes[par]]
    child, par = child[on], par[on]
    weight = w[par] * edge[child]  # w(child) for chance and hero steps
    ev = np.bincount(classes[par], weights=weight * v[classes[child]], minlength=k)
    ev = np.divide(ev, big_w, out=np.zeros(k), where=big_w > 0)
    term = np.zeros(n)
    term[child] = v[classes[child]] - ev[classes[par]]
    total = np.zeros(n)
    for d in range(1, tree.max_depth + 1):
        lv = tree.level(d)
        total[lv] = total[tree.parent[lv]] + term[lv]

    result = np.nan_to_num(_average(classes, w, u, k))[classes] if imaginary else u
    estimate = np.where(tree.player == TERMINAL, result - total, np.nan)
    return ControlVariates(estimate, total, float(v[classes[0]]))


def terminal_moments(tree: GameTree, edge: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Mean and second moment of ``x`` (read at terminals) when hands follow ``edge``."""
    term = tree.player == TERMINAL
    r = tree.reach(edge)[term]
    return float(r @ x[term]), float(r @ x[term] ** 2)


def deal_moments(
    tree: GameTree, edge: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per complete deal: its probability, and the mean and second moment of ``x`` given it.

    A deal fixes every chance outcome: the private cards dealt at the top of the tree and the
    public card dealt later, if the game has one (a hand that ends before it is compatible with
    every public card). Duplicate dealing plays the same deal in both seats, so the deals line up
    between two profiles on the same tree. At most one later chance node per path.
    """
    top = 0
    while np.all(tree.player[tree.level(top)] == CHANCE):
        top += 1
    deals = np.arange(tree.level_start[top], tree.level_start[top + 1])
    chance_only = tree.reach(np.where(tree.edge_player == CHANCE, tree.chance_prob, 1.0))
    term = tree.player == TERMINAL
    leaf, leaf2 = np.where(term, x, 0.0), np.where(term, x, 0.0) ** 2

    later = np.flatnonzero((tree.player == CHANCE) & (tree.depth > top))
    if len(later) == 0:
        m = tree.backward(edge, leaf)[deals]
        q = tree.backward(edge, leaf2)[deals]
        return chance_only[deals], m, q

    ancestor = later.copy()
    seen_chance = np.zeros(len(later), dtype=bool)
    while np.any(tree.depth[ancestor] > top):
        up = tree.depth[ancestor] > top
        ancestor[up] = tree.parent[ancestor[up]]
        seen_chance |= up & (tree.player[ancestor] == CHANCE) & (tree.depth[ancestor] > top)
    if seen_chance.any():
        raise ValueError("deal_moments supports one later chance node per path")
    row = ancestor - deals[0]
    kids = np.flatnonzero(np.isin(tree.parent, later) & (np.arange(tree.num_nodes) > 0))
    labels = sorted({tree.labels[c] for c in kids})
    kid_label = np.array([labels.index(tree.labels[c]) for c in kids])
    kid_row = row[np.searchsorted(later, tree.parent[kids])]
    p_label = np.full((len(deals), len(labels)), np.nan)
    for r_, lab, p in zip(kid_row, kid_label, tree.chance_prob[kids], strict=True):
        if not np.isnan(p_label[r_, lab]) and abs(p_label[r_, lab] - p) > _TOL:
            raise ValueError("a deal's public card has different probabilities below it")
        p_label[r_, lab] = p
    covered = np.zeros(len(deals), dtype=bool)
    covered[row] = True
    if not covered.all():
        raise ValueError("every deal must reach the later chance node on some path")
    p_label = np.nan_to_num(p_label)

    probs, means, seconds = [], [], []
    for lab in range(len(labels)):
        edge_b = edge.copy()
        edge_b[kids] = (kid_label == lab).astype(float)
        probs.append(chance_only[deals] * p_label[:, lab])
        means.append(tree.backward(edge_b, leaf)[deals])
        seconds.append(tree.backward(edge_b, leaf2)[deals])
    p, m, q = (np.concatenate(a) for a in (probs, means, seconds))
    keep = p > 0
    return p[keep], m[keep], q[keep]

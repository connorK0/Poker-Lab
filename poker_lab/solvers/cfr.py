"""Counterfactual regret minimization (CFR) and its modern variants, vectorized over a game tree.

**Regret matching.** At each information set ``I`` keep a cumulative regret ``R(I, a)`` per
action and play actions in proportion to their positive regret:

    sigma(I, a) = R+(I, a) / sum_b R+(I, b)      (uniform if no regret is positive)

**Counterfactual regret** (Zinkevich et al. 2007). After each iteration add

    r(I, a) = q(I, a) - sum_b sigma(I, b) q(I, b),

where ``q(I, a)`` is the counterfactual value of ``a`` (the action's value weighted by the
probability that chance and the opponent reach ``I``; see ``poker_lab.games.evaluate``). CFR
minimizes regret separately at every information set, and the regrets of the whole strategy
are bounded by the sum of these local regrets. So after ``T`` iterations the *average*
strategy profile is an epsilon-equilibrium with epsilon = O(1 / sqrt(T)). The *current*
strategy need not converge (it can cycle); the average is

    avg(I, a) proportional to sum_t w_t * pi_i(I; sigma_t) * sigma_t(I, a),

where ``pi_i(I; sigma_t)`` is the player's own probability of reaching ``I`` (so iterations
in which the player would rarely be at ``I`` count less) and ``w_t`` is an iteration weight.

**Variants** (all share the code below; ``CFRVariant`` holds the knobs):

=========  ==================================================  ===========  ============
name       cumulative regret update                            weight w_t   updates
=========  ==================================================  ===========  ============
cfr        R <- R + r                                          1            simultaneous
cfr+       R <- max(R + r, 0)  (regret matching+)              t            alternating
lcfr       R <- (R + r) * t / (t + 1)  (iteration t has weight t)  t        alternating
dcfr       R <- R + r, then positive R *= t^1.5 / (t^1.5 + 1),  t^2          alternating
           negative R *= 1/2
=========  ==================================================  ===========  ============

CFR+ is due to Tammelin (2014); linear and discounted CFR (LCFR, DCFR with alpha = 1.5,
beta = 0, gamma = 2) to Brown & Sandholm (2019). *Alternating* updates update player 0's
regrets, recompute the strategy, then update player 1 against player 0's new strategy;
*simultaneous* updates use the same strategy for both, as in the original paper.

Each iteration is a few vectorized passes over the compiled tree: one backward pass for node
values, two forward passes for reach probabilities, and ``bincount``/``reduceat`` to gather
counterfactual values per sequence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from poker_lab.games.evaluate import exploitability
from poker_lab.games.policy import TabularPolicy
from poker_lab.games.tree import GameTree


@dataclass(frozen=True)
class CFRVariant:
    name: str
    floor: bool = False  # regret matching+: clip cumulative regrets at zero after each update
    alternating: bool = True
    alpha: float | None = None  # positive regrets are multiplied by t^alpha / (t^alpha + 1)
    beta: float | None = None  # negative regrets are multiplied by t^beta / (t^beta + 1)
    gamma: float = 0.0  # iteration t's average-strategy contribution has weight t^gamma


VARIANTS: dict[str, CFRVariant] = {
    "cfr": CFRVariant("cfr", alternating=False),
    "cfr+": CFRVariant("cfr+", floor=True, gamma=1.0),
    "lcfr": CFRVariant("lcfr", alpha=1.0, beta=1.0, gamma=1.0),
    "dcfr": CFRVariant("dcfr", alpha=1.5, beta=0.0, gamma=2.0),
}


class CFRSolver:
    """Full-width tabular CFR on a compiled game tree (see the module docstring)."""

    def __init__(self, tree: GameTree, variant: str | CFRVariant = "cfr+") -> None:
        self.tree = tree
        self.variant = VARIANTS[variant] if isinstance(variant, str) else variant
        self.regrets = np.zeros(tree.num_sequences)
        self.strategy_sum = np.zeros(tree.num_sequences)
        self.iteration = 0
        self.nodes_touched = 0  # work counter: tree nodes evaluated, for comparisons with MCCFR
        owner = tree.infoset_player[tree.seq_infoset]
        self._mask = (owner == 0, owner == 1)
        self._kids = tuple(np.flatnonzero(tree.edge_player == p) for p in (0, 1))

    def current_strategy(self) -> np.ndarray:
        """Regret matching on the cumulative regrets (one probability per sequence)."""
        return self.tree.normalize(np.maximum(self.regrets, 0.0))

    def current_policy(self) -> TabularPolicy:
        return TabularPolicy(self.tree, self.current_strategy())

    def average_policy(self) -> TabularPolicy:
        return TabularPolicy.from_weights(self.tree, self.strategy_sum)

    def counterfactual_regrets(self, probs: np.ndarray, player: int):
        """Instantaneous regrets ``r`` and the player's own reach of each sequence's infoset."""
        tree = self.tree
        edge = tree.edge_probabilities(probs)
        v = tree.backward(edge)
        if player == 1:
            v = -v
        mine = tree.edge_player == player
        reach_others = tree.reach(np.where(mine, 1.0, edge))
        reach_own = tree.reach(np.where(mine, edge, 1.0))
        kids = self._kids[player]
        q = np.bincount(
            tree.edge_seq[kids],
            weights=reach_others[tree.parent[kids]] * v[kids],
            minlength=tree.num_sequences,
        )
        regret = q - tree.segment_sum(probs * q)[tree.seq_infoset]
        own_reach = reach_own[tree.infoset_node][tree.seq_infoset]
        self.nodes_touched += tree.num_nodes
        return regret, own_reach

    def _update(self, player: int, probs: np.ndarray) -> None:
        t, var, mask = self.iteration, self.variant, self._mask[player]
        regret, own_reach = self.counterfactual_regrets(probs, player)
        r = self.regrets
        r[mask] += regret[mask]
        if var.floor:
            r[mask] = np.maximum(r[mask], 0.0)
        if var.alpha is not None:
            pos, neg = mask & (r > 0), mask & (r < 0)
            r[pos] *= t**var.alpha / (t**var.alpha + 1.0)
            r[neg] *= t**var.beta / (t**var.beta + 1.0)
        s = self.strategy_sum
        if var.gamma:
            s[mask] *= ((t - 1) / t) ** var.gamma
        s[mask] += (own_reach * probs)[mask]

    def iterate(self, n: int = 1) -> None:
        for _ in range(n):
            self.iteration += 1
            if self.variant.alternating:
                for player in (0, 1):
                    self._update(player, self.current_strategy())
            else:
                probs = self.current_strategy()
                for player in (0, 1):
                    self._update(player, probs)

    def run(self, checkpoints: Iterable[int]) -> list[dict[str, float]]:
        """Iterate up to each checkpoint and record the average policy's exploitability."""
        curve = []
        for target in sorted(set(checkpoints)):
            if target < self.iteration:
                continue
            self.iterate(target - self.iteration)
            curve.append(
                {
                    "iteration": self.iteration,
                    "nodes_touched": self.nodes_touched,
                    "exploitability": exploitability(self.average_policy()),
                }
            )
        return curve


def log_checkpoints(last: int, per_decade: int = 8, first: int = 1) -> list[int]:
    """Log-spaced integers from ``first`` to ``last`` inclusive, ``per_decade`` per factor of
    10 and aligned so that ``first`` times every power of 10 is included."""
    steps = int(np.floor(per_decade * np.log10(last / first) + 1e-9))
    points = np.round(first * 10.0 ** (np.arange(steps + 1) / per_decade))
    return sorted({int(p) for p in points} | {int(last)})

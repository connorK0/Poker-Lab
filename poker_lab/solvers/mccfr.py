"""External-sampling Monte Carlo CFR (Lanctot, Waugh, Zinkevich, Bowling 2009).

Full-width CFR evaluates the whole tree every iteration. External sampling instead runs one
*traversal* per player per iteration:

* at the traverser's own information sets, every action is explored;
* at chance nodes one outcome is sampled from the chance probabilities;
* at the opponent's information sets one action is sampled from the opponent's current
  (regret-matching) strategy.

The regrets accumulated along a traversal are unbiased estimates of the true counterfactual
regrets, up to a positive factor that is constant for each information set (the probability
that sampling reaches it). Regret matching is unchanged by a constant factor, so the same
convergence guarantee holds in expectation, while each iteration touches tens of nodes
instead of the whole tree.

The average strategy is accumulated at the *opponent's* information sets during a traversal
("stochastically weighted averaging"): an opponent information set is visited with
probability proportional to the opponent's own reach, which is exactly the weight the average
needs. With ``linear=True`` both regrets and average-strategy contributions of iteration
``t`` are weighted by ``t`` (as in linear CFR and Deep CFR).

The traversal is plain recursive Python over lists (faster than NumPy for 2-3 actions per
node) and is fully determined by the seed.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np

from poker_lab.games.base import CHANCE
from poker_lab.games.evaluate import exploitability
from poker_lab.games.policy import TabularPolicy
from poker_lab.games.tree import TERMINAL, GameTree


class ExternalSamplingMCCFR:
    def __init__(self, tree: GameTree, seed: int, *, linear: bool = False) -> None:
        self.tree = tree
        self.seed = seed
        self.linear = linear
        self.rng = random.Random(seed)
        self.iteration = 0
        self.nodes_touched = 0
        n_seq = tree.num_sequences
        self.regrets = [0.0] * n_seq
        self.strategy_sum = [0.0] * n_seq
        # plain lists: indexing a list is several times faster than indexing a NumPy array
        self._player = tree.player.tolist()
        self._infoset = tree.infoset.tolist()
        self._child = tree.child_start.tolist()
        self._utility = tree.utility.tolist()
        self._offset = tree.seq_offset.tolist()
        self._n_actions = tree.infoset_num_actions.tolist()
        cum = {}
        for node in np.flatnonzero(tree.player == CHANCE).tolist():
            start, k = self._child[node], int(tree.num_children[node])
            cum[node] = np.cumsum(tree.chance_prob[start : start + k]).tolist()
        self._chance_cum = cum

    def _strategy(self, start: int, n: int) -> list[float]:
        pos = [max(r, 0.0) for r in self.regrets[start : start + n]]
        total = sum(pos)
        return [p / total for p in pos] if total > 0 else [1.0 / n] * n

    def _traverse(self, node: int, player: int, weight: float) -> float:
        self.nodes_touched += 1
        actor = self._player[node]
        if actor == TERMINAL:
            u = self._utility[node]
            return u if player == 0 else -u
        if actor == CHANCE:
            u, cum = self.rng.random(), self._chance_cum[node]
            k = 0
            while k < len(cum) - 1 and u >= cum[k]:
                k += 1
            return self._traverse(self._child[node] + k, player, weight)
        i = self._infoset[node]
        start, n = self._offset[i], self._n_actions[i]
        sigma = self._strategy(start, n)
        first_child = self._child[node]
        if actor == player:
            values = [self._traverse(first_child + a, player, weight) for a in range(n)]
            v = sum(s * x for s, x in zip(sigma, values, strict=True))
            regrets = self.regrets
            for a in range(n):
                regrets[start + a] += weight * (values[a] - v)
            return v
        sums = self.strategy_sum
        for a in range(n):
            sums[start + a] += weight * sigma[a]
        u, acc, a = self.rng.random(), 0.0, 0
        for a in range(n):  # after the loop, a is the sampled action
            acc += sigma[a]
            if u < acc:
                break
        return self._traverse(first_child + a, player, weight)

    def iterate(self, n: int = 1) -> None:
        """``n`` iterations; each is one traversal for player 0 and one for player 1."""
        for _ in range(n):
            self.iteration += 1
            weight = float(self.iteration) if self.linear else 1.0
            for player in (0, 1):
                self._traverse(0, player, weight)

    def average_policy(self) -> TabularPolicy:
        return TabularPolicy.from_weights(self.tree, np.asarray(self.strategy_sum))

    def current_policy(self) -> TabularPolicy:
        return TabularPolicy.from_weights(self.tree, np.maximum(np.asarray(self.regrets), 0.0))

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
                    "traversals": 2 * self.iteration,
                    "nodes_touched": self.nodes_touched,
                    "exploitability": exploitability(self.average_policy()),
                }
            )
        return curve

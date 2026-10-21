"""``TabularPolicy``: a behavior strategy for both players of a compiled game.

One probability per sequence (information set, action); each information set's probabilities
sum to 1. A policy only ever sees information-set keys, which encode the acting player's own
cards and public information, so it cannot condition on the opponent's private cards.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from poker_lab.games.tree import GameTree


class TabularPolicy:
    __slots__ = ("probs", "tree")

    def __init__(self, tree: GameTree, probs: np.ndarray, *, atol: float = 1e-6) -> None:
        probs = np.array(probs, dtype=float)  # private copy
        if probs.shape != (tree.num_sequences,):
            raise ValueError(f"expected {tree.num_sequences} probabilities, got {probs.shape}")
        if not np.all(np.isfinite(probs)) or probs.min(initial=0.0) < -atol:
            raise ValueError("probabilities must be finite and non-negative")
        sums = tree.segment_sum(probs)
        if np.max(np.abs(sums - 1.0), initial=0.0) > atol:
            bad = int(np.argmax(np.abs(sums - 1.0)))
            raise ValueError(f"probabilities at {tree.infoset_keys[bad]!r} sum to {sums[bad]}")
        if probs.min(initial=0.0) < 0 or np.max(np.abs(sums - 1.0), initial=0.0) > 1e-12:
            probs = np.clip(probs, 0.0, None)  # repair rounding error; valid input is kept as is
            probs /= tree.segment_sum(probs)[tree.seq_infoset]
        probs.setflags(write=False)
        self.tree = tree
        self.probs = probs

    @classmethod
    def uniform(cls, tree: GameTree) -> TabularPolicy:
        return cls(tree, 1.0 / tree.infoset_num_actions[tree.seq_infoset])

    @classmethod
    def from_weights(cls, tree: GameTree, weights: np.ndarray) -> TabularPolicy:
        """Normalize non-negative weights per information set (uniform where all are 0)."""
        return cls(tree, tree.normalize(np.asarray(weights, dtype=float)))

    @classmethod
    def from_function(
        cls, tree: GameTree, fn: Callable[[str, tuple[str, ...]], Mapping[str, float]]
    ) -> TabularPolicy:
        """Build from ``fn(infoset_key, legal_actions) -> {action: probability}``."""
        probs = np.zeros(tree.num_sequences)
        for i, (key, actions) in enumerate(
            zip(tree.infoset_keys, tree.infoset_actions, strict=True)
        ):
            dist = fn(key, actions)
            unknown = set(dist) - set(actions)
            if unknown:
                raise ValueError(f"unknown actions {sorted(unknown)} at {key!r}")
            start = int(tree.seq_offset[i])
            for a, action in enumerate(actions):
                probs[start + a] = dist.get(action, 0.0)
        return cls(tree, probs)

    @classmethod
    def from_mapping(
        cls, tree: GameTree, mapping: Mapping[str, Mapping[str, float]], *, default: str = "error"
    ) -> TabularPolicy:
        """Build from ``{infoset_key: {action: probability}}``.

        Information sets missing from ``mapping`` raise (``default="error"``) or play uniformly
        (``default="uniform"``).
        """
        if default not in ("error", "uniform"):
            raise ValueError("default must be 'error' or 'uniform'")
        unknown = set(mapping) - set(tree.infoset_index)
        if unknown:
            raise ValueError(f"unknown information sets: {sorted(unknown)[:5]}")

        def lookup(key: str, actions: tuple[str, ...]) -> Mapping[str, float]:
            if key in mapping:
                return mapping[key]
            if default == "error":
                raise KeyError(f"no strategy for information set {key!r}")
            return {a: 1.0 / len(actions) for a in actions}

        return cls.from_function(tree, lookup)

    def action_probabilities(self, key: str) -> dict[str, float]:
        i = self.tree.infoset_index[key]
        start = int(self.tree.seq_offset[i])
        actions = self.tree.infoset_actions[i]
        return {a: float(self.probs[start + k]) for k, a in enumerate(actions)}

    def to_mapping(self) -> dict[str, dict[str, float]]:
        return {key: self.action_probabilities(key) for key in self.tree.infoset_keys}

    def combine(self, other: TabularPolicy, player: int) -> TabularPolicy:
        """Profile that plays ``self`` at ``player``'s information sets and ``other`` elsewhere."""
        if other.tree is not self.tree:
            raise ValueError("policies belong to different game trees")
        mine = self.tree.sequences_of(player)
        return TabularPolicy(self.tree, np.where(mine, self.probs, other.probs))

    def max_difference(self, other: TabularPolicy) -> float:
        """Largest absolute probability difference over all sequences."""
        if other.tree is not self.tree:
            raise ValueError("policies belong to different game trees")
        return float(np.max(np.abs(self.probs - other.probs)))

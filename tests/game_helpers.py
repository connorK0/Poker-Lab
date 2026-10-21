"""Helpers for the Phase 2 small-game tests."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from poker_lab.games import CHANCE, Game, TabularPolicy
from poker_lab.games.tree import GameTree


def decision_states(game: Game, state=None) -> Iterator:
    """Every decision state of ``game``, depth first."""
    state = game.initial_state() if state is None else state
    if game.is_terminal(state):
        return
    if game.current_player(state) == CHANCE:
        for label, _ in game.chance_outcomes(state):
            yield from decision_states(game, game.next_state(state, label))
        return
    yield state
    for action in game.legal_actions(state):
        yield from decision_states(game, game.next_state(state, action))


def recursive_value(game: Game, policy: TabularPolicy, state=None) -> float:
    """Player 0's expected payoff by plain recursion over the game (reference implementation)."""
    state = game.initial_state() if state is None else state
    if game.is_terminal(state):
        return game.utility(state)
    if game.current_player(state) == CHANCE:
        return sum(
            p * recursive_value(game, policy, game.next_state(state, label))
            for label, p in game.chance_outcomes(state)
        )
    probs = policy.action_probabilities(game.information_set(state))
    return sum(
        p * recursive_value(game, policy, game.next_state(state, a)) for a, p in probs.items()
    )


def random_policy(tree: GameTree, seed: int) -> TabularPolicy:
    rng = np.random.default_rng(seed)
    return TabularPolicy.from_weights(tree, rng.dirichlet(np.ones(3), tree.num_sequences)[:, 0])


def kuhn_equilibrium(tree: GameTree, alpha: float) -> TabularPolicy:
    """The known one-parameter equilibrium family of Kuhn poker, alpha in [0, 1/3]."""
    aggressive = {  # probability of the aggressive action (bet or call) at each infoset
        "J|": alpha,
        "Q|": 0.0,
        "K|": 3 * alpha,
        "J|kb": 0.0,
        "Q|kb": alpha + 1 / 3,
        "K|kb": 1.0,
        "J|k": 1 / 3,
        "Q|k": 0.0,
        "K|k": 1.0,
        "J|b": 0.0,
        "Q|b": 1 / 3,
        "K|b": 1.0,
    }
    mapping = {}
    for key, p in aggressive.items():
        passive, aggr = ("f", "c") if key.endswith("b") else ("k", "b")
        mapping[key] = {aggr: p, passive: 1 - p}
    return TabularPolicy.from_mapping(tree, mapping)

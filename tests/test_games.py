"""Phase 2 games: Kuhn and Leduc rules, the tree compiler and tabular policies."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from poker_lab.games import (
    CHANCE,
    TERMINAL,
    Game,
    KuhnPoker,
    LeducHoldem,
    LeducState,
    TabularPolicy,
    compile_tree,
    expected_value,
)
from tests.game_helpers import decision_states, random_policy, recursive_value


@pytest.fixture(scope="module")
def kuhn_tree():
    return compile_tree(KuhnPoker())


@pytest.fixture(scope="module")
def leduc_tree():
    return compile_tree(LeducHoldem())


def test_kuhn_tree_shape(kuhn_tree):
    assert kuhn_tree.counts() == {
        "nodes": 58,
        "terminals": 30,
        "chance_nodes": 4,
        "decision_nodes": 24,
        "infosets": 12,
        "infosets_player0": 6,
        "infosets_player1": 6,
        "sequences": 24,
        "max_depth": 5,
    }
    assert set(kuhn_tree.infoset_keys) == {
        f"{card}|{h}" for card in "JQK" for h in ("", "k", "b", "kb")
    }


def test_leduc_tree_shape(leduc_tree):
    # 6 private-card deals x (6 round-1 decision points + 5 continuations x 5 public cards x 6
    # round-2 decision points): 936 information sets, the standard count with suits told apart.
    counts = leduc_tree.counts()
    assert counts["infosets"] == 936
    assert counts["infosets_player0"] == counts["infosets_player1"] == 468
    assert counts["nodes"] == 9457
    assert counts["terminals"] == 5520
    assert counts["sequences"] == 2184
    assert "JsQh|kbc/b" in leduc_tree.infoset_index


@pytest.mark.parametrize(
    ("cards", "rounds", "expected"),
    [
        ((0, 2, 1), ("kk", "kk"), 1.0),  # Js vs Qs, board Jh: player 0 pairs
        ((0, 2, 5), ("brc", "bc"), -9.0),  # board Kh, Q beats J; 1 + 2 + 2 + 4 each
        ((4, 2, 5), ("brc", "brc"), 13.0),  # Ks pairs the Kh board; the largest possible pot
        ((0, 1, 2), ("kk", "kk"), 0.0),  # Js vs Jh on a Qs board: split
        ((0, 2), ("bf",), 1.0),
        ((0, 2), ("kbf",), -1.0),
        ((0, 2), ("brf",), -3.0),  # player 0 bet 2 then folded to the raise
        ((0, 2), ("kbrf",), 3.0),  # player 1 bet 2 then folded to the raise
        ((0, 2, 4), ("bc", "kbrf"), 7.0),  # player 1 put in 1 + 2 + 4 before folding
    ],
)
def test_leduc_payoffs(cards, rounds, expected):
    game = LeducHoldem()
    state = LeducState(cards, rounds)
    assert game.is_terminal(state)
    assert game.utility(state) == expected


def test_leduc_betting_rules():
    game = LeducHoldem()
    s = LeducState((0, 2), ("",))
    assert game.legal_actions(s) == ("k", "b")
    assert game.legal_actions(replace(s, rounds=("b",))) == ("f", "c", "r")
    assert game.legal_actions(replace(s, rounds=("br",))) == ("f", "c")  # two-bet cap
    assert game.current_player(replace(s, rounds=("kb",))) == 0
    assert game.current_player(replace(s, rounds=("kk",))) == CHANCE  # deal the public card
    with pytest.raises(ValueError, match="illegal"):
        game.next_state(replace(s, rounds=("br",)), "r")


def test_vectorized_values_match_recursive_evaluation(kuhn_tree, leduc_tree):
    for game, tree in ((KuhnPoker(), kuhn_tree), (LeducHoldem(), leduc_tree)):
        policy = random_policy(tree, seed=7)
        assert expected_value(policy) == pytest.approx(recursive_value(game, policy), abs=1e-12)


def test_tree_arrays_are_consistent(leduc_tree):
    t = leduc_tree
    nodes = np.arange(1, t.num_nodes)
    assert np.all(t.parent[nodes] < nodes)
    assert np.all(t.depth[nodes] == t.depth[t.parent[nodes]] + 1)
    for n in range(t.num_nodes):  # children are contiguous and point back to their parent
        kids = range(t.child_start[n], t.child_start[n] + t.num_children[n])
        assert all(t.parent[k] == n for k in kids)
        if t.player[n] >= 0:
            i = t.infoset[n]
            expected = list(range(t.seq_offset[i], t.seq_offset[i + 1]))
            assert [t.edge_seq[k] for k in kids] == expected
    # chance probabilities out of every chance node sum to one
    chance = np.flatnonzero(t.player == CHANCE)
    sums = np.bincount(t.parent[1:], weights=t.chance_prob[1:], minlength=t.num_nodes)
    np.testing.assert_allclose(sums[chance], 1.0)
    # under any profile the leaves' reach probabilities sum to one
    reach = t.reach(t.edge_probabilities(random_policy(t, seed=3).probs))
    assert reach[t.player == TERMINAL].sum() == pytest.approx(1.0)


def test_information_sets_never_reveal_the_opponents_card():
    """Small-game analogue of the information firewall: swapping the opponent's private card
    for any other unseen card never changes the acting player's information-set key."""
    game = LeducHoldem()
    checked = 0
    for state in decision_states(game):
        me = game.current_player(state)
        key = game.information_set(state)
        assert key.startswith(game.deck[state.cards[me]])
        for card in range(len(game.deck)):
            if card in state.cards:
                continue
            cards = list(state.cards)
            cards[1 - me] = card
            assert game.information_set(replace(state, cards=tuple(cards))) == key
            checked += 1
    assert checked > 3000


def test_compiler_rejects_imperfect_recall():
    class Forgetful(Game):
        """Player 0 acts twice but the second information set forgets the first action."""

        name = "forgetful"

        def initial_state(self):
            return ""

        def is_terminal(self, s):
            return len(s) == 3

        def current_player(self, s):
            return (0, 1, 0)[len(s)]

        def legal_actions(self, s):
            return ("x", "y")

        def chance_outcomes(self, s):
            return ()

        def next_state(self, s, a):
            return s + a

        def utility(self, s):
            return 1.0 if s == "xxx" else 0.0

        def information_set(self, s):
            return "first" if len(s) == 0 else ("p1" if len(s) == 1 else "second")

    with pytest.raises(ValueError, match="imperfect recall"):
        compile_tree(Forgetful())


def test_policy_construction_and_validation(kuhn_tree):
    uniform = TabularPolicy.uniform(kuhn_tree)
    assert uniform.action_probabilities("Q|kb") == {"f": 0.5, "c": 0.5}
    mapping = random_policy(kuhn_tree, seed=1).to_mapping()
    rebuilt = TabularPolicy.from_mapping(kuhn_tree, mapping)
    assert rebuilt.to_mapping() == mapping
    with pytest.raises(KeyError, match="no strategy"):
        TabularPolicy.from_mapping(kuhn_tree, {"J|": {"k": 1.0}})
    partial = TabularPolicy.from_mapping(kuhn_tree, {"J|": {"b": 1.0}}, default="uniform")
    assert partial.action_probabilities("J|") == {"k": 0.0, "b": 1.0}
    assert partial.action_probabilities("K|b") == {"f": 0.5, "c": 0.5}
    with pytest.raises(ValueError, match="sum to"):
        TabularPolicy(kuhn_tree, np.full(kuhn_tree.num_sequences, 0.4))
    with pytest.raises(ValueError, match="unknown actions"):
        TabularPolicy.from_mapping(kuhn_tree, {**mapping, "J|": {"r": 1.0}})
    # combine takes player 0's part from one policy and player 1's from the other
    mixed = partial.combine(uniform, player=0)
    assert mixed.action_probabilities("J|") == {"k": 0.0, "b": 1.0}
    assert mixed.action_probabilities("J|k") == {"k": 0.5, "b": 0.5}
    assert mixed.probs.flags.writeable is False

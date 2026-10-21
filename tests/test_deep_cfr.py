"""Phase 2: Deep CFR (needs the ``nn`` extra; skipped when PyTorch is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from poker_lab.games import (
    KuhnPoker,
    LeducHoldem,
    TabularPolicy,
    compile_tree,
    exploitability,
)
from poker_lab.solvers import DeepCFR, DeepCFRConfig, InfosetEncoder, Reservoir

UNIFORM_KUHN = 11 / 24  # exploitability of the uniform policy

SMALL = DeepCFRConfig(
    traversals=100, hidden=(16, 16), advantage_steps=60, policy_steps=200, cosine_lr=False
)


@pytest.fixture(scope="module")
def kuhn():
    game = KuhnPoker()
    tree = compile_tree(game)
    return tree, InfosetEncoder.for_game(tree, game)


def test_encoder_features_and_slots():
    game = LeducHoldem()
    tree = compile_tree(game)
    enc = InfosetEncoder.for_game(tree, game)
    assert enc.features.shape[0] == 936
    # rank-only features merge suit-isomorphic information sets: 288 distinct rows
    assert len({row.tobytes() for row in enc.features}) == 288
    np.testing.assert_array_equal(enc.legal.sum(axis=1), tree.infoset_num_actions)
    i = tree.infoset_index["JsQh|kbc/b"]
    assert enc.legal[i].tolist() == [True, True, True]  # fold, call, raise
    assert enc.legal[tree.infoset_index["Ks|"]].tolist() == [False, True, True]  # check, bet


def test_reservoir_keeps_a_uniform_sample():
    n_items, capacity, runs = 200, 50, 600
    kept = np.zeros(n_items)
    for run in range(runs):
        res = Reservoir(capacity, 1, np.random.default_rng(run))
        res.add(np.arange(n_items), 1, np.zeros((n_items, 1)))
        assert len(res) == capacity
        assert res.seen == n_items
        kept[res.infoset[:capacity]] += 1
    freq = kept / runs
    se = np.sqrt(0.25 * 0.75 / runs)  # every item should be kept with probability 50/200
    assert np.all(np.abs(freq - 0.25) < 5 * se)


def test_perfect_regressor_variant_converges_on_kuhn(kuhn):
    tree, enc = kuhn
    solver = DeepCFR(tree, enc, DeepCFRConfig(traversals=200, regressor="tabular"), seed=1)
    solver.iterate(1)
    # Alternating updates (Algorithm 1 of the paper): player 0's regressor is refit before
    # player 1 traverses, so after one iteration only player 1 has played uniformly.
    p1 = tree.sequences_of(1)
    uniform = TabularPolicy.uniform(tree).probs
    np.testing.assert_allclose(solver.exact_average_policy().probs[p1], uniform[p1])
    assert not np.allclose(solver.exact_average_policy().probs[~p1], uniform[~p1])
    solver.iterate(19)
    assert exploitability(solver.exact_average_policy()) < 0.05
    assert exploitability(solver.network_policy()) < 0.08  # = the memory average here


def test_network_deep_cfr_is_reproducible_and_learns(kuhn):
    tree, enc = kuhn
    a = DeepCFR(tree, enc, SMALL, seed=5).run([6], exploitability, policy_checkpoints=[6])
    # evaluating at extra checkpoints must not change the training trajectory
    b = DeepCFR(tree, enc, SMALL, seed=5).run([2, 4, 6], exploitability, policy_checkpoints=[3, 6])
    for key in ("exact_average", "current", "policy_network", "memory_average"):
        assert a[-1][key] == b[-1][key]
    assert a[-1]["traversals"] == 2 * 6 * SMALL.traversals
    assert a[-1]["exact_average"] < 0.75 * UNIFORM_KUHN
    c = DeepCFR(tree, enc, SMALL, seed=6).run([6], exploitability)
    assert c[-1]["exact_average"] != a[-1]["exact_average"]


def test_config_validation():
    with pytest.raises(ValueError, match="regressor"):
        DeepCFRConfig(regressor="forest")

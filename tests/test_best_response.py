"""Phase 2: exact best responses, exploitability and sequence-form LP equilibria."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from poker_lab.games import (
    KuhnPoker,
    LeducHoldem,
    TabularPolicy,
    best_response,
    compile_tree,
    expected_value,
    exploitability,
    exploitability_report,
    head_to_head,
)
from poker_lab.solvers import solve_lp
from tests.game_helpers import kuhn_equilibrium, random_policy

KUHN_VALUE = -1 / 18
LEDUC_VALUE = -0.0856064  # published value of Leduc hold'em for the first player


@pytest.fixture(scope="module")
def kuhn_tree():
    return compile_tree(KuhnPoker())


@pytest.fixture(scope="module")
def leduc_tree():
    return compile_tree(LeducHoldem())


def test_uniform_random_nash_conv_matches_known_values(kuhn_tree, leduc_tree):
    # Kuhn: BR values 1/2 and 5/12 (worked by hand in docs/BUILD_LOG.md), NashConv 11/12.
    kuhn = exploitability_report(TabularPolicy.uniform(kuhn_tree))
    assert kuhn.br_values == pytest.approx((0.5, 5 / 12))
    assert kuhn.nash_conv == pytest.approx(11 / 12)
    assert kuhn.value == pytest.approx(1 / 8)
    # Leduc: 4.747222..., the value OpenSpiel's test suite uses for the uniform policy.
    leduc = exploitability_report(TabularPolicy.uniform(leduc_tree))
    assert leduc.nash_conv == pytest.approx(4.747222222222222, abs=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 1 / 12, 1 / 6, 1 / 3])
def test_kuhn_equilibrium_family_is_unexploitable(kuhn_tree, alpha):
    policy = kuhn_equilibrium(kuhn_tree, alpha)
    report = exploitability_report(policy)
    assert report.value == pytest.approx(KUHN_VALUE, abs=1e-12)
    assert abs(report.exploitability) < 1e-12


def test_leaving_the_equilibrium_family_is_exploitable(kuhn_tree):
    mapping = kuhn_equilibrium(kuhn_tree, 1 / 6).to_mapping()
    mapping["Q|kb"] = {"f": 0.2, "c": 0.8}  # calls too often with the middle card
    assert exploitability(TabularPolicy.from_mapping(kuhn_tree, mapping)) > 0.01


@pytest.mark.parametrize("player", [0, 1])
def test_best_response_matches_brute_force_on_kuhn(kuhn_tree, player):
    opponent = random_policy(kuhn_tree, seed=11 + player)
    mine = [i for i in range(kuhn_tree.num_infosets) if kuhn_tree.infoset_player[i] == player]
    sign = 1.0 if player == 0 else -1.0
    best = -np.inf
    for choice in itertools.product((0, 1), repeat=len(mine)):  # all 64 pure strategies
        probs = opponent.probs.copy()
        for i, a in zip(mine, choice, strict=True):
            start = kuhn_tree.seq_offset[i]
            probs[start : start + 2] = (1.0 - a, float(a))
        best = max(best, sign * expected_value(TabularPolicy(kuhn_tree, probs)))
    br = best_response(opponent, player)
    assert br.value == pytest.approx(best, abs=1e-12)
    assert sign * expected_value(br.policy) == pytest.approx(br.value, abs=1e-12)


@pytest.mark.parametrize("player", [0, 1])
def test_leduc_best_response_is_achieved_and_locally_optimal(leduc_tree, player):
    tree = leduc_tree
    opponent = random_policy(tree, seed=5)
    br = best_response(opponent, player)
    sign = 1.0 if player == 0 else -1.0
    assert sign * expected_value(br.policy) == pytest.approx(br.value, abs=1e-10)
    # the opponent's part of the profile is untouched
    theirs = tree.sequences_of(1 - player)
    np.testing.assert_array_equal(br.policy.probs[theirs], opponent.probs[theirs])
    # no single-infoset deviation improves on it
    rng = np.random.default_rng(0)
    mine = np.flatnonzero(tree.infoset_player == player)
    for i in rng.choice(mine, size=60, replace=False):
        start, stop = tree.seq_offset[i], tree.seq_offset[i + 1]
        for a in range(stop - start):
            probs = br.policy.probs.copy()
            probs[start:stop] = 0.0
            probs[start + a] = 1.0
            deviation = sign * expected_value(TabularPolicy(tree, probs))
            assert deviation <= br.value + 1e-10


def test_lp_finds_exact_equilibria(kuhn_tree, leduc_tree):
    kuhn = solve_lp(kuhn_tree)
    assert kuhn.value == pytest.approx(KUHN_VALUE, abs=1e-9)
    assert exploitability(kuhn.policy) < 1e-9
    leduc = solve_lp(leduc_tree)
    assert leduc.value == pytest.approx(LEDUC_VALUE, abs=1e-6)
    report = exploitability_report(leduc.policy)
    assert report.value == pytest.approx(leduc.value, abs=1e-8)
    assert report.exploitability < 1e-7


def test_equilibrium_never_loses_head_to_head(leduc_tree):
    eq = solve_lp(leduc_tree).policy
    assert head_to_head(eq, eq) == pytest.approx(0.0, abs=1e-12)
    for seed in range(5):
        assert head_to_head(eq, random_policy(leduc_tree, seed)) > -1e-7
    # and the best response to a fixed opponent beats it by exactly its exploitability gap
    villain = random_policy(leduc_tree, seed=9)
    br = best_response(villain, 0).policy.combine(best_response(villain, 1).policy, 0)
    assert head_to_head(br, villain) == pytest.approx(exploitability(villain), abs=1e-10)

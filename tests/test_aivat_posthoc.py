"""P2.11 post hoc: the exact distributions of one hand and of a duplicate pair's mean in the
push/fold game, their moments, and the sampling error of a sample SD."""

import itertools
import math
from collections import defaultdict
from functools import cache

import numpy as np
import pytest
from scipy import stats

from experiments.phase2_aivat import duplicate_stats, single_stats
from experiments.phase2_aivat_posthoc import (
    _collapse,
    moments,
    pair_distribution,
    sd_rel_se,
    simulate_sd_ratio,
    single_distribution,
)
from experiments.phase3_bridge import policies
from poker_lab.evaluation.aivat import deal_moments
from poker_lab.games import (
    CHANCE,
    TERMINAL,
    KuhnPoker,
    LeducHoldem,
    TabularPolicy,
    compile_tree,
    head_to_head,
)

STACK = 3.0  # bb, as in the quick configs
MATCHUPS = [("nash", "nash"), ("nash", "maniac"), ("best_response", "maniac")]


@cache
def _matchup(hero_name: str, opp_name: str):
    """The policies at 3 bb and both exact distributions, computed once per matchup."""
    hero, opp = policies(STACK, hero_name, opp_name)
    return hero, opp, single_distribution(hero, opp), pair_distribution(hero, opp)


@pytest.mark.parametrize(("hero_name", "opp_name"), MATCHUPS)
def test_distributions_match_the_experiments_exact_moments(hero_name, opp_name):
    """Mean and SD agree with ``single_stats`` / ``duplicate_stats`` (whose duplicate SD is per
    hand: sqrt(2) times a pair mean's) and the mean with ``head_to_head``."""
    hero, opp, single, pair = _matchup(hero_name, opp_name)
    tree = hero.tree
    edges = {s: tree.edge_probabilities(hero.combine(opp, s).probs) for s in (0, 1)}
    u = {s: (1.0 if s == 0 else -1.0) * tree.utility for s in (0, 1)}
    exact = head_to_head(hero, opp)
    expected = (
        (single, single_stats(tree, edges, u, exact, 0.05), 1.0),
        (pair, duplicate_stats(tree, edges, u, exact, 0.05), math.sqrt(2.0)),
    )
    for (p, v), want, per_hand in expected:
        assert p.sum() == pytest.approx(1.0, abs=1e-9)
        assert (p >= 0).all()
        assert np.all(np.diff(v) > 0)  # one entry per value
        assert np.abs(v).max() <= STACK + 1e-9  # nobody wins more than the stack
        m = moments(p, v)
        assert m["mean"] == pytest.approx(exact, abs=1e-10)
        assert m["mean"] == pytest.approx(want["mean"], abs=1e-10)
        assert per_hand * m["sd"] == pytest.approx(want["sd"], rel=1e-9, abs=1e-12)


def test_nash_pairs_cancel_when_both_tables_play_alike():
    """Nash against Nash: both tables play the same profile, so a pair scores
    0.5 (u(z0) - u(z1)) for the terminals z0, z1 the two tables reach. It is exactly 0 when they
    reach the same one; the plays are independent given the deal, so the rest is symmetric about
    0 and the pair's variance is half the average variance within a deal."""
    hero, opp, single, pair = _matchup("nash", "nash")
    tree = hero.tree
    edge = tree.edge_probabilities(hero.combine(opp, 0).probs)
    np.testing.assert_array_equal(edge, tree.edge_probabilities(hero.combine(opp, 1).probs))
    p, v = pair

    # P(same terminal) = sum over deals of P(deal) * sum_z P(z | deal)^2
    actions = tree.reach(np.where(tree.edge_player == CHANCE, 1.0, edge))  # P(z | its deal)
    p_deal, p_same_given_deal, _ = deal_moments(tree, edge, actions)
    p_same = p_deal @ p_same_given_deal
    assert 0.9 < p_same < 1.0  # Nash mixes only at its threshold hands
    assert p[np.abs(v) < 1e-12].sum() >= p_same - 1e-9

    _, m, q = deal_moments(tree, edge, tree.utility)
    pair_m, single_m = moments(p, v), moments(*single)
    assert pair_m["mean"] == pytest.approx(0.0, abs=1e-12)
    assert pair_m["sd"] ** 2 == pytest.approx(0.5 * p_deal @ (q - m**2), rel=1e-9)
    assert 0.0 < pair_m["sd"] < single_m["sd"] / 10
    np.testing.assert_allclose(v, -v[::-1], atol=1e-12)
    np.testing.assert_allclose(p, p[::-1], atol=1e-12)


def _chance_labels(tree, node: int) -> tuple:
    labels = []
    while node > 0:
        if tree.edge_player[node] == CHANCE:
            labels.append(tree.labels[node])
        node = int(tree.parent[node])
    return tuple(labels[::-1])


def test_pair_distribution_matches_brute_force_duplicate_pairs_in_kuhn():
    """Every pair of terminals with the same deal, one per seat, weighted independently."""
    tree = compile_tree(KuhnPoker())
    rng = np.random.default_rng(7)
    hero, opp = (TabularPolicy.from_weights(tree, rng.random(tree.num_sequences)) for _ in "ab")
    term = np.flatnonzero(tree.player == TERMINAL)
    chance = tree.reach(np.where(tree.edge_player == CHANCE, tree.chance_prob, 1.0))
    actions = [
        tree.reach(np.where(tree.edge_player == CHANCE, 1.0, e))
        for e in (tree.edge_probabilities(hero.combine(opp, s).probs) for s in (0, 1))
    ]
    deal = {z: _chance_labels(tree, z) for z in term}
    brute: dict[float, float] = defaultdict(float)
    for z0, z1 in itertools.product(term, term):
        if deal[z0] == deal[z1]:
            value = round(0.5 * (tree.utility[z0] - tree.utility[z1]), 12)
            brute[value] += chance[z0] * actions[0][z0] * actions[1][z1]
    p, v = pair_distribution(hero, opp)
    assert sorted(brute) == pytest.approx(list(v), abs=1e-12)
    assert [brute[x] for x in sorted(brute)] == pytest.approx(list(p), abs=1e-12)


def test_pair_distribution_rejects_chance_below_the_deal():
    tree = compile_tree(LeducHoldem())  # the board card comes after the first betting round
    uniform = TabularPolicy.uniform(tree)
    with pytest.raises(ValueError, match="every chance node above the deal"):
        pair_distribution(uniform, uniform)


@pytest.mark.parametrize("n", [1, 100, 100_000])
def test_sd_rel_se_of_a_normal_sample(n):
    assert sd_rel_se(3.0, n) == pytest.approx(1.0 / math.sqrt(2 * n))


@pytest.mark.parametrize(("q", "n"), [(0.1, 1_000), (0.01, 10_000)])
def test_sd_rel_se_matches_the_simulated_spread_of_a_two_point_distribution(q, n):
    """Kurtosis (1 - 3 q (1 - q)) / (q (1 - q)): 8.1 and 98. The SD of 2,000 simulated ratios
    is itself uncertain by about 1 / sqrt(2 * 2000) = 1.6% (checked over 300 seeds with 400
    reps: spread 3.5%, mean 1.00), so 10% is about 6 of those."""
    p, v = np.array([1.0 - q, q]), np.array([0.0, 1.0])
    m = moments(p, v)
    assert m["kurtosis"] == pytest.approx((1 - 3 * q * (1 - q)) / (q * (1 - q)))
    ratios = simulate_sd_ratio(p, v, n, 2000, np.random.default_rng(11))
    assert ratios.std(ddof=1) == pytest.approx(sd_rel_se(m["kurtosis"], n), rel=0.1)


def test_simulate_sd_ratio_is_centred_on_one_and_reproducible():
    """A three-point distribution, kurtosis 1.79: relative SE 1.4% at n = 1000, so the mean of
    4,000 ratios has an SE of 0.0002 and a bias of about -(kurtosis - 1) / (8n) = -0.0001."""
    p, v = np.array([0.2, 0.5, 0.3]), np.array([-1.0, 0.0, 2.0])
    ratios = simulate_sd_ratio(p, v, 1000, 4000, np.random.default_rng(5))
    assert ratios.shape == (4000,)
    assert ratios.mean() == pytest.approx(1.0, abs=0.002)
    assert ratios.std(ddof=1) == pytest.approx(sd_rel_se(moments(p, v)["kurtosis"], 1000), rel=0.1)
    again = simulate_sd_ratio(p, v, 1000, 4000, np.random.default_rng(5))
    np.testing.assert_array_equal(ratios, again)
    other = simulate_sd_ratio(p, v, 1000, 4000, np.random.default_rng(6))
    assert not np.array_equal(ratios, other)


def test_simulate_sd_ratio_in_chunks_matches_one_draw():
    """The two-point distribution {0: 0.9, 1: 0.1} split over 100,000 values (90,000 zeros,
    10,000 ones): the same sample SDs in distribution. It is drawn in blocks of 20 reps (the
    last one short), the collapsed version in one block."""
    k = 100_000
    split_v = np.repeat([0.0, 1.0], [90_000, 10_000])
    split_p = np.where(split_v == 0.0, 0.9 / 90_000, 0.1 / 10_000)
    reps = 110
    assert 2_000_000 // k < reps  # the chunking path
    chunked = simulate_sd_ratio(split_p, split_v, 1000, reps, np.random.default_rng(1))
    p, v = _collapse(split_p, split_v)
    assert len(v) == 2
    whole = simulate_sd_ratio(p, v, 1000, 4000, np.random.default_rng(2))
    rel = sd_rel_se(moments(p, v)["kurtosis"], 1000)  # 4.2%
    assert chunked.shape == (reps,)
    assert np.all(np.abs(chunked - 1.0) < 8 * rel)  # every block filled
    assert chunked.mean() == pytest.approx(whole.mean(), abs=4.5 * rel / math.sqrt(reps))
    assert stats.ks_2samp(chunked, whole).pvalue > 1e-3


def test_collapse_merges_equal_values():
    rng = np.random.default_rng(3)
    v = rng.choice([-1.5, 0.0, 0.3, 2.0], size=200)
    v[::7] = 0.1 + 0.2  # 0.30000000000000004: merged with 0.3 after rounding
    p = rng.random(200)
    p /= p.sum()
    q, w = _collapse(p, v)
    np.testing.assert_allclose(w, [-1.5, 0.0, 0.3, 2.0])
    np.testing.assert_allclose(q, [p[np.isclose(v, x)].sum() for x in w])
    assert q.sum() == pytest.approx(1.0)
    before, after = moments(p, v), moments(q, w)
    assert after["mean"] == pytest.approx(before["mean"], abs=1e-12)
    assert after["sd"] == pytest.approx(before["sd"], rel=1e-12)
    assert after["kurtosis"] == pytest.approx(before["kurtosis"], rel=1e-10)

"""Phase 2: tabular CFR variants and external-sampling MCCFR."""

from __future__ import annotations

import numpy as np
import pytest

from poker_lab.games import (
    CHANCE,
    KuhnPoker,
    LeducHoldem,
    TabularPolicy,
    compile_tree,
    expected_value,
    exploitability,
)
from poker_lab.solvers import VARIANTS, CFRSolver, ExternalSamplingMCCFR, log_checkpoints

LEDUC_VALUE = -0.0856064


@pytest.fixture(scope="module")
def kuhn_tree():
    return compile_tree(KuhnPoker())


@pytest.fixture(scope="module")
def leduc_tree():
    return compile_tree(LeducHoldem())


def textbook_cfr(game, tree, iterations: int):
    """Vanilla CFR with simultaneous updates as the usual recursion over histories."""
    shape = {k: len(a) for k, a in zip(tree.infoset_keys, tree.infoset_actions, strict=True)}
    regrets = {k: np.zeros(n) for k, n in shape.items()}
    sums = {k: np.zeros(n) for k, n in shape.items()}

    def strategy(key):
        pos = np.maximum(regrets[key], 0.0)
        return pos / pos.sum() if pos.sum() > 0 else np.full(len(pos), 1.0 / len(pos))

    def walk(state, reach, d_regret, d_sum):  # reach = (player 0, player 1, chance)
        if game.is_terminal(state):
            return game.utility(state)  # player 0's value
        p = game.current_player(state)
        if p == CHANCE:
            return sum(
                prob
                * walk(
                    game.next_state(state, lab),
                    (reach[0], reach[1], reach[2] * prob),
                    d_regret,
                    d_sum,
                )
                for lab, prob in game.chance_outcomes(state)
            )
        key = game.information_set(state)
        sigma = strategy(key)
        values = []
        for a, action in enumerate(game.legal_actions(state)):
            child = list(reach)
            child[p] *= sigma[a]
            values.append(walk(game.next_state(state, action), tuple(child), d_regret, d_sum))
        values = np.array(values)
        v = float(sigma @ values)
        sign = 1.0 if p == 0 else -1.0
        d_regret[key] += reach[1 - p] * reach[2] * sign * (values - v)
        d_sum[key] += reach[p] * sigma
        return v

    for _ in range(iterations):
        d_regret = {k: np.zeros(n) for k, n in shape.items()}
        d_sum = {k: np.zeros(n) for k, n in shape.items()}
        walk(game.initial_state(), (1.0, 1.0, 1.0), d_regret, d_sum)
        for k in shape:
            regrets[k] += d_regret[k]
            sums[k] += d_sum[k]
    return regrets, sums


@pytest.mark.parametrize("game", [KuhnPoker(), LeducHoldem()], ids=["kuhn", "leduc"])
def test_vectorized_cfr_matches_textbook_recursion(game):
    tree = compile_tree(game)
    solver = CFRSolver(tree, "cfr")
    solver.iterate(3)
    regrets, sums = textbook_cfr(game, tree, 3)
    average = solver.average_policy()
    for i, key in enumerate(tree.infoset_keys):
        seqs = slice(tree.seq_offset[i], tree.seq_offset[i + 1])
        np.testing.assert_allclose(solver.regrets[seqs], regrets[key], atol=1e-10)
        # The recursion adds the own-reach-weighted strategy once per history in the
        # information set, a constant multiple of the vectorized sum, so compare normalized.
        np.testing.assert_allclose(average.probs[seqs], sums[key] / sums[key].sum(), atol=1e-10)


# Regression thresholds: the exploitability each variant reached when first run (2026-10-21),
# with headroom. Measured values at 1,000 iterations: Kuhn cfr 7.3e-3, cfr+ 8.7e-5, lcfr 9.4e-5,
# dcfr 1.5e-4; Leduc cfr 4.0e-2, cfr+ 2.6e-4, lcfr 5.2e-3, dcfr 1.6e-4.
THRESHOLDS = {
    "kuhn": {"cfr": 1e-2, "cfr+": 2e-4, "lcfr": 2e-4, "dcfr": 3e-4},
    "leduc": {"cfr": 5e-2, "cfr+": 5e-4, "lcfr": 1e-2, "dcfr": 4e-4},
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_cfr_variants_converge(kuhn_tree, leduc_tree, variant):
    for name, tree in (("kuhn", kuhn_tree), ("leduc", leduc_tree)):
        solver = CFRSolver(tree, variant)
        curve = solver.run([10, 1000])
        assert curve[-1]["exploitability"] < THRESHOLDS[name][variant]
        assert curve[-1]["exploitability"] < curve[0]["exploitability"]


def test_cfr_plus_agrees_with_the_lp_value(leduc_tree):
    solver = CFRSolver(leduc_tree, "cfr+")
    solver.iterate(1000)
    avg = solver.average_policy()
    assert expected_value(avg) == pytest.approx(LEDUC_VALUE, abs=1e-3)
    # CFR+'s current strategy converges too in practice (not guaranteed by theory)
    assert exploitability(solver.current_policy()) < 0.05


def test_mccfr_is_reproducible_and_converges(kuhn_tree):
    a = ExternalSamplingMCCFR(kuhn_tree, seed=3)
    b = ExternalSamplingMCCFR(kuhn_tree, seed=3)
    c = ExternalSamplingMCCFR(kuhn_tree, seed=4)
    for s in (a, b, c):
        s.iterate(500)
    assert a.regrets == b.regrets
    assert a.strategy_sum == b.strategy_sum
    assert a.regrets != c.regrets
    curve = a.run([20_000])
    assert curve[-1]["exploitability"] < 0.03
    assert curve[-1]["traversals"] == 40_000


def test_external_sampling_regrets_are_unbiased(kuhn_tree):
    """The mean sampled regret increment equals the exact counterfactual regret (uniform play)."""
    exact, _ = CFRSolver(kuhn_tree, "cfr").counterfactual_regrets(
        TabularPolicy.uniform(kuhn_tree).probs, player=0
    )
    mccfr = ExternalSamplingMCCFR(kuhn_tree, seed=0)
    zero = list(mccfr.regrets)
    n = 40_000
    deltas = np.empty((n, kuhn_tree.num_sequences))
    for k in range(n):
        mccfr.regrets = list(zero)  # keep the strategy uniform: undo every update
        mccfr._traverse(0, 0, 1.0)
        deltas[k] = mccfr.regrets
    mean, se = deltas.mean(axis=0), deltas.std(axis=0, ddof=1) / np.sqrt(n)
    mine = kuhn_tree.sequences_of(0)
    assert np.all(np.abs(mean[mine] - exact[mine]) <= 4 * se[mine] + 1e-9)
    assert np.all(deltas[:, ~mine] == 0)  # player 1's regrets are untouched


def test_log_checkpoints():
    cps = log_checkpoints(10_000, per_decade=4)
    assert cps[0] == 1
    assert cps[-1] == 10_000
    assert cps == sorted(set(cps))
    assert 15 <= len(cps) <= 18

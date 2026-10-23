"""Preflop matchups and heads-up push/fold hold'em (P3.1)."""

import json
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
import pytest

from poker_lab.bots.hand_strength import all_hand_classes, hand_class
from poker_lab.games.evaluate import exploitability
from poker_lab.games.preflop import (
    CLASS_INDEX,
    CLASSES,
    N_DEALS,
    combo_counts,
    combos,
    load_matchups,
    matchup_counts,
)
from poker_lab.games.push_fold import (
    BB,
    SB,
    PushFoldGame,
    aggressive_probabilities,
    hand_gains,
    push_fold_tree,
    range_share,
)
from poker_lab.games.tree import compile_tree
from poker_lab.schema.cards import FULL_DECK
from poker_lab.solvers.cfr import CFRSolver
from poker_lab.solvers.lp import solve_lp

REFERENCE = Path(__file__).resolve().parents[1] / "experiments" / "references" / "hune_no_ante.json"


def test_classes_and_combos_match_the_phase1_definitions():
    assert tuple(all_hand_classes()) == CLASSES
    assert combo_counts().sum() == 1326
    for cls in CLASSES:
        for a, b in combos(cls):
            assert hand_class(FULL_DECK[a], FULL_DECK[b]) == cls


def test_matchup_counts_equal_a_brute_force_count():
    counts = matchup_counts()
    assert counts.sum() == N_DEALS
    assert (counts == counts.T).all()
    by_class: dict[str, list[tuple[int, int]]] = {}
    for h in combinations(range(52), 2):
        by_class.setdefault(hand_class(FULL_DECK[h[0]], FULL_DECK[h[1]]), []).append(h)
    for a_cls, b_cls in [("AA", "AA"), ("AA", "AKs"), ("AKo", "AKo"), ("72o", "KQs")]:
        brute = sum(1 for h in by_class[a_cls] for g in by_class[b_cls] if not set(h) & set(g))
        assert counts[CLASS_INDEX[a_cls], CLASS_INDEX[b_cls]] == brute


@pytest.fixture(scope="module")
def m():
    return load_matchups()


def test_matchup_table_is_consistent(m):
    n = len(CLASSES)
    assert np.allclose(m.equity + m.equity.T, 1.0, atol=1e-12)
    assert np.allclose(np.diag(m.equity), 0.5)
    off = ~np.eye(n, dtype=bool)
    assert m.se[off].max() < 0.003
    assert (m.counts == matchup_counts()).all()


@pytest.mark.parametrize(
    ("a", "b", "published"),
    [("AA", "KK", 0.819), ("KK", "AKs", 0.661), ("AKo", "QQ", 0.432), ("AA", "72o", 0.88)],
)
def test_well_known_matchups(m, a, b, published):
    # Published all-in equities (class averages, to about half a point).
    assert m.equity[CLASS_INDEX[a], CLASS_INDEX[b]] == pytest.approx(published, abs=0.012)


def test_equity_vs_random_agrees_with_the_independent_phase1_table(m):
    path = (
        Path(__file__).resolve().parents[1] / "poker_lab" / "bots" / "data" / "preflop_equity.json"
    )
    phase1 = json.loads(path.read_text(encoding="utf-8"))["equity"]
    ours = m.equity_vs_random()
    diff = np.array([ours[k] - phase1[c] for k, c in enumerate(CLASSES)])
    assert np.abs(diff).max() < 0.01


def test_tree_shape_and_payoffs(m):
    tree = push_fold_tree(10.0, m)
    counts = tree.counts()
    assert counts["infosets_player0"] == counts["infosets_player1"] == 169
    assert counts["nodes"] == 1 + 3 * 169 * 169 + 2 * 169 * 169
    game = PushFoldGame(10.0, m)
    s = game.next_state(game.initial_state(), "AKs|QQ")
    assert game.information_set(s) == "AKs|"
    assert game.utility(game.next_state(s, "f")) == -0.5
    pushed = game.next_state(s, "p")
    assert game.information_set(pushed) == "QQ|p"
    assert game.utility(game.next_state(pushed, "f")) == 1.0
    e = m.equity[CLASS_INDEX["AKs"], CLASS_INDEX["QQ"]]
    assert game.utility(game.next_state(pushed, "c")) == pytest.approx(10.0 * (2 * e - 1))


def test_rescaled_tree_equals_a_direct_compile(m):
    direct = compile_tree(PushFoldGame(7.3, m))
    assert np.allclose(push_fold_tree(7.3, m).utility, direct.utility, atol=1e-12)


def test_range_share():
    assert range_share(np.ones(169)) == pytest.approx(1.0)
    only_aces = np.zeros(169)
    only_aces[CLASS_INDEX["AA"]] = 1.0
    assert range_share(only_aces) == pytest.approx(6 / 1326)


def test_lp_nash_is_exact_and_cfr_plus_agrees(m):
    tree = push_fold_tree(10.0, m)
    sol = solve_lp(tree)
    assert exploitability(sol.policy) < 1e-7
    solver = CFRSolver(tree, "cfr+")
    solver.iterate(300)  # the experiment's curves reach ~1e-5 by 300 iterations
    assert exploitability(solver.average_policy()) < 1e-3
    push = aggressive_probabilities(sol.policy, SB)
    call = aggressive_probabilities(sol.policy, BB)
    assert push[CLASS_INDEX["AA"]] == pytest.approx(1.0)
    assert call[CLASS_INDEX["AA"]] == pytest.approx(1.0)
    assert push[CLASS_INDEX["72o"]] == pytest.approx(0.0)


@pytest.mark.parametrize("stack", [2.5, 10.0, 18.0])
def test_lp_solution_satisfies_the_per_hand_equilibrium_conditions(m, stack):
    """Every hand that plays gains by playing, every hand that folds would lose (computed
    from the matchup table directly, independently of the tree and the LP)."""
    sol = solve_lp(push_fold_tree(stack, m))
    push = aggressive_probabilities(sol.policy, SB)
    call = aggressive_probabilities(sol.policy, BB)
    sb, bb = hand_gains(stack, push, call, m)
    assert (sb[push > 1e-6] >= -1e-7).all()
    assert (sb[push < 1 - 1e-6] <= 1e-7).all()
    faced = ~np.isnan(bb)
    assert (bb[faced & (call > 1e-6)] >= -1e-7).all()
    assert (bb[faced & (call < 1 - 1e-6)] <= 1e-7).all()


def test_ranges_tighten_as_stacks_grow(m):
    shares = []
    for stack in (2.0, 4.0, 8.0, 16.0):
        sol = solve_lp(push_fold_tree(stack, m))
        shares.append(
            (
                range_share(aggressive_probabilities(sol.policy, SB)),
                range_share(aggressive_probabilities(sol.policy, BB)),
            )
        )
    for (p0, c0), (p1, c1) in pairwise(shares):
        assert p1 < p0
        assert c1 < c0


@pytest.mark.parametrize(
    ("hand", "player", "chart"),
    [("K2o", SB, "pusher"), ("Q7o", SB, "pusher"), ("Q2o", BB, "caller"), ("A2o", BB, "caller")],
)
def test_thresholds_match_the_published_chart(m, hand, player, chart):
    """HoldemResources' no-ante chart: the hand plays up to the listed stack and not above."""
    ref = float(json.loads(REFERENCE.read_text(encoding="utf-8"))[chart][hand])
    k = CLASS_INDEX[hand]
    below = aggressive_probabilities(solve_lp(push_fold_tree(ref - 0.3, m)).policy, player)
    above = aggressive_probabilities(solve_lp(push_fold_tree(ref + 0.3, m)).policy, player)
    assert below[k] > 0.5
    assert above[k] < 0.5

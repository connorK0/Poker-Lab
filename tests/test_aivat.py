"""AIVAT control variates (P2.11): unbiased for any value function, exact zero-variance cases,
the hero's own view, and the per-deal moments behind duplicate dealing."""

import itertools

import numpy as np
import pytest

from poker_lab.evaluation import AIVATEvaluator, HandResult, WinRateEvaluator
from poker_lab.evaluation.aivat import (
    class_actor,
    control_variates,
    deal_moments,
    known_transitions,
    observer_classes,
    terminal_moments,
)
from poker_lab.games import (
    CHANCE,
    TERMINAL,
    KuhnPoker,
    LeducHoldem,
    TabularPolicy,
    compile_tree,
    head_to_head,
)
from poker_lab.games.evaluate import expected_value
from poker_lab.solvers import solve_lp


@pytest.fixture(scope="module", params=["kuhn", "leduc"])
def setup(request):
    game = KuhnPoker() if request.param == "kuhn" else LeducHoldem()
    tree = compile_tree(game)
    classes = {
        (observer, s): observer_classes(game, tree, hidden=s if observer == "full" else 1 - s)
        for observer in ("full", "view")
        for s in (0, 1)
    }
    return game, tree, solve_lp(tree).policy, classes


def _random_policy(tree, rng):
    return TabularPolicy.from_weights(tree, rng.random(tree.num_sequences))


def _estimates(tree, classes, hero, edge, value, mode, belief=None):
    if mode == "view":
        cls = classes[("view", hero)]
        corrected = known_transitions(tree, cls, hero)
        return control_variates(
            tree, hero, edge, value, cls, corrected, imaginary=False, value_weight=belief
        )
    cls = classes[("full", hero)]
    actor = class_actor(tree, cls)
    steps = {"aivat": (CHANCE, hero), "all": (CHANCE, 0, 1), "chance": (CHANCE,)}[mode]
    return control_variates(tree, hero, edge, value, cls, np.isin(actor, steps), imaginary=True)


@pytest.mark.parametrize("mode", ["aivat", "all", "chance", "view"])
def test_unbiased_for_any_value_function(setup, mode):
    _game, tree, _eq, classes = setup
    rng = np.random.default_rng(1)
    for hero in (0, 1):
        sign = 1.0 if hero == 0 else -1.0
        profile = _random_policy(tree, rng).combine(_random_policy(tree, rng), hero)
        edge = tree.edge_probabilities(profile.probs)
        value = rng.normal(scale=3.0, size=tree.num_nodes)
        belief = tree.reach(tree.edge_probabilities(_random_policy(tree, rng).probs))
        cv = _estimates(tree, classes, hero, edge, value, mode, belief)
        mean, _ = terminal_moments(tree, edge, cv.estimate)
        assert mean == pytest.approx(sign * expected_value(profile), abs=1e-10)


@pytest.mark.parametrize("mode", ["aivat", "view"])
def test_estimate_does_not_depend_on_the_opponents_strategy(setup, mode):
    _game, tree, eq, classes = setup
    rng = np.random.default_rng(2)
    value = rng.normal(size=tree.num_nodes)
    for hero in (0, 1):
        first, second = (
            _estimates(
                tree,
                classes,
                hero,
                tree.edge_probabilities(eq.combine(_random_policy(tree, rng), hero).probs),
                value,
                mode,
                tree.reach(tree.edge_probabilities(eq.probs)),
            ).estimate
            for _ in range(2)
        )
        term = tree.player == TERMINAL
        np.testing.assert_allclose(first[term], second[term], atol=1e-12)


def test_self_play_with_equilibrium_values_has_no_variance(setup):
    """At an equilibrium the opponent mixes only between actions of equal value (averaged over
    the hero's cards), so once chance and the hero are corrected nothing random is left."""
    _game, tree, eq, classes = setup
    edge = tree.edge_probabilities(eq.probs)
    for hero in (0, 1):
        sign = 1.0 if hero == 0 else -1.0
        cv = _estimates(tree, classes, hero, edge, sign * tree.backward(edge), "aivat")
        mean, second = terminal_moments(tree, edge, cv.estimate)
        assert second - mean**2 < 1e-12


def test_true_values_with_every_step_corrected_leave_no_variance(setup):
    _game, tree, eq, classes = setup
    rng = np.random.default_rng(3)
    for hero in (0, 1):
        sign = 1.0 if hero == 0 else -1.0
        edge = tree.edge_probabilities(eq.combine(_random_policy(tree, rng), hero).probs)
        cv = _estimates(tree, classes, hero, edge, sign * tree.backward(edge), "all")
        mean, second = terminal_moments(tree, edge, cv.estimate)
        assert second - mean**2 < 1e-12
        assert cv.root_value == pytest.approx(mean, abs=1e-12)


def test_hero_view_corrections_use_only_the_heros_information(setup):
    """Correction = result - estimate is the same at terminals the hero cannot tell apart."""
    _game, tree, eq, classes = setup
    rng = np.random.default_rng(4)
    term = tree.player == TERMINAL
    for hero in (0, 1):
        edge = tree.edge_probabilities(eq.combine(_random_policy(tree, rng), hero).probs)
        value = rng.normal(size=tree.num_nodes)
        cv = _estimates(tree, classes, hero, edge, value, "view")
        sign = 1.0 if hero == 0 else -1.0
        correction = sign * tree.utility - cv.estimate
        cls = classes[("view", hero)]
        for c in np.unique(cls[term]):
            np.testing.assert_allclose(correction[cls == c], correction[cls == c][0], atol=1e-12)


def test_known_transitions_in_leduc():
    game = LeducHoldem()
    tree = compile_tree(game)
    for hero in (0, 1):
        cls = observer_classes(game, tree, hidden=1 - hero)
        known = known_transitions(tree, cls, hero)
        actor = class_actor(tree, cls)
        assert known[actor == hero].all()
        assert not known[actor == 1 - hero].any()
        chance = np.flatnonzero(actor == CHANCE)
        _, first = np.unique(cls, return_index=True)
        depth = tree.depth[first[chance]]
        assert known[chance[depth < 2]].all()  # the private deals
        assert not known[chance[depth >= 2]].any()  # the board, after the opponent has acted


def test_deal_moments_match_brute_force_duplicate_pairs():
    game = KuhnPoker()
    tree = compile_tree(game)
    rng = np.random.default_rng(5)
    hero, opp = _random_policy(tree, rng), _random_policy(tree, rng)
    edges = [tree.edge_probabilities(hero.combine(opp, s).probs) for s in (0, 1)]
    x = [(1.0 if s == 0 else -1.0) * tree.utility for s in (0, 1)]
    term = np.flatnonzero(tree.player == TERMINAL)
    actions_only = [tree.reach(np.where(tree.edge_player == CHANCE, 1.0, e)) for e in edges]

    def deal_of(node):
        return tuple(tree.labels[n] for n in _path(tree, node) if tree.edge_player[n] == CHANCE)

    deals = sorted({deal_of(z) for z in term})
    brute_mean = brute_second = 0.0
    for deal in deals:
        za = [z for z in term if deal_of(z) == deal]
        p = 1.0 / len(deals)
        for z0, z1 in itertools.product(za, za):
            w = p * actions_only[0][z0] * actions_only[1][z1]
            m = 0.5 * (x[0][z0] + x[1][z1])
            brute_mean += w * m
            brute_second += w * m * m
    p0, m0, q0 = deal_moments(tree, edges[0], x[0])
    p1, m1, q1 = deal_moments(tree, edges[1], x[1])
    pair = 0.5 * (m0 + m1)
    second = p0 @ (0.25 * ((q0 - m0**2) + (q1 - m1**2)) + pair**2)
    assert p0 @ pair == pytest.approx(brute_mean, abs=1e-12)
    assert second == pytest.approx(brute_second, abs=1e-12)
    np.testing.assert_allclose(p0, p1)


def _path(tree, node):
    path = []
    while node > 0:
        path.append(node)
        node = int(tree.parent[node])
    return path[::-1]


def test_deal_moments_cover_every_deal_with_a_public_card():
    tree = compile_tree(LeducHoldem())
    edge = tree.edge_probabilities(TabularPolicy.uniform(tree).probs)
    p, m, q = deal_moments(tree, edge, tree.utility)
    assert len(p) == 6 * 5 * 4
    assert p.sum() == pytest.approx(1.0)
    assert (p @ m, p @ q) == pytest.approx(terminal_moments(tree, edge, tree.utility))


def test_observer_classes_need_the_matching_game():
    with pytest.raises(ValueError, match="does not match"):
        observer_classes(LeducHoldem(), compile_tree(KuhnPoker()), hidden=0)


def test_follow_maps_labels_to_nodes():
    tree = compile_tree(KuhnPoker())
    node = tree.follow(["K", "J", "b", "c"])
    assert tree.player[node] == TERMINAL
    assert tree.utility[node] == 2.0
    with pytest.raises(ValueError, match="no edge"):
        tree.follow(["K", "K"])


def test_aivat_evaluator_on_sampled_leduc_hands():
    """End to end: hands played out in both seats (equally often, so the seat needs no term),
    corrected through ``AIVATEvaluator``: the interval covers the exact value and is far
    narrower than the plain one (2.7 times against this random opponent)."""
    game = LeducHoldem()
    tree = compile_tree(game)
    eq = solve_lp(tree).policy
    rng = np.random.default_rng(6)
    opp = _random_policy(tree, rng)
    exact = head_to_head(eq, opp)
    terminals = np.flatnonzero(tree.player == TERMINAL)
    results, corrections = [], {}
    for s in (0, 1):
        sign = 1.0 if s == 0 else -1.0
        edge = tree.edge_probabilities(eq.combine(opp, s).probs)
        cls = observer_classes(game, tree, hidden=s)
        actor = class_actor(tree, cls)
        value = sign * tree.backward(tree.edge_probabilities(eq.probs))
        cv = control_variates(
            tree, s, edge, value, cls, np.isin(actor, (CHANCE, s)), imaginary=True
        )
        reach = tree.reach(edge)[terminals]
        for k, z in enumerate(rng.choice(terminals, size=3000, p=reach / reach.sum())):
            u = sign * tree.utility[z]
            hand_id = f"{s}-{k}"
            results.append(HandResult(hand_id, "hero", round(u), 1))
            corrections[hand_id] = u - cv.estimate[z]
    plain = WinRateEvaluator().evaluate(results)
    aivat = AIVATEvaluator(lambda r: corrections[r.hand_id]).evaluate(results)
    assert aivat.lower <= 100 * exact <= aivat.upper
    assert aivat.half_width < plain.half_width / 2

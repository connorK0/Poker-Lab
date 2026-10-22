"""P2.5 + P2.8: contracts shared by every small-game strategy source and exploit engine.

Small-game strategies are ``TabularPolicy`` objects (functions of information-set keys only,
so they cannot see the opponent's cards). Every solver must return one on the tree it was
given; every policy-level exploit engine must return a valid strategy with an *exact*
exploitability estimate. New solvers and engines join these parameter lists.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from poker_lab.exploit.small_games import (
    BestResponseEngine,
    DataBiasedEngine,
    PolicyExploitResult,
    RestrictedNashEngine,
)
from poker_lab.games import (
    KuhnPoker,
    LeducHoldem,
    TabularPolicy,
    compile_tree,
    exploitability,
)
from poker_lab.games.leduc_bots import ARCHETYPE_MEANS, BotPolicyBuilder
from poker_lab.solvers import CFRSolver, ExternalSamplingMCCFR, solve_lp
from tests.game_helpers import random_policy

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _cfr(tree, game):
    solver = CFRSolver(tree, "cfr+")
    solver.iterate(30)
    return solver.average_policy()


def _mccfr(tree, game):
    solver = ExternalSamplingMCCFR(tree, seed=4)
    solver.iterate(300)
    return solver.average_policy()


def _deep_cfr(tree, game):
    from poker_lab.solvers import DeepCFR, DeepCFRConfig, InfosetEncoder

    config = DeepCFRConfig(traversals=200, regressor="tabular")  # fast: no network training
    solver = DeepCFR(tree, InfosetEncoder.for_game(tree, game), config, seed=2)
    solver.iterate(10)
    return solver.network_policy()


SOURCES = {
    "lp": lambda tree, game: solve_lp(tree).policy,
    "cfr+": _cfr,
    "es_mccfr": _mccfr,
    "deep_cfr": pytest.param(
        _deep_cfr, marks=pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
    ),
}


@pytest.fixture(scope="module", params=["kuhn", "leduc"])
def game_and_tree(request):
    game = KuhnPoker() if request.param == "kuhn" else LeducHoldem()
    return game, compile_tree(game)


@pytest.mark.parametrize("source", list(SOURCES.values()), ids=list(SOURCES))
def test_strategy_source_contract(game_and_tree, source):
    game, tree = game_and_tree
    policy = source(tree, game)
    again = source(tree, game)
    assert type(policy) is TabularPolicy
    assert policy.tree is tree
    np.testing.assert_array_equal(policy.probs, again.probs)  # reproducible (seeded)
    expl = exploitability(policy)
    assert -1e-9 <= expl < exploitability(TabularPolicy.uniform(tree))


def test_bot_policies_meet_the_source_contract():
    game = LeducHoldem()
    tree = compile_tree(game)
    builder = BotPolicyBuilder(tree, game)
    for params in ARCHETYPE_MEANS.values():
        policy = builder.policy(params)
        assert type(policy) is TabularPolicy
        assert policy.tree is tree
        assert exploitability(policy) > 0.1  # archetypes are deliberately exploitable


ENGINES = {
    "best_response": BestResponseEngine,
    "rnr_0.5": lambda: RestrictedNashEngine(0.5),
    "dbr_array": None,  # needs the tree; built in the test
    "dbr_callable": lambda: DataBiasedEngine(lambda tree: np.full(tree.num_infosets, 0.5)),
}


@pytest.mark.parametrize("name", list(ENGINES))
def test_policy_exploit_engine_contract(game_and_tree, name):
    _, tree = game_and_tree
    engine = (
        DataBiasedEngine(np.full(tree.num_infosets, 0.5))
        if name == "dbr_array"
        else ENGINES[name]()
    )
    model = random_policy(tree, seed=21)
    baseline = solve_lp(tree).policy
    result = engine.adjust(model, baseline)
    assert isinstance(result, PolicyExploitResult)
    assert type(result.strategy) is TabularPolicy
    assert result.strategy.tree is tree
    est = result.exploitability
    assert est.method == "exact_best_response"
    assert est.unit == "chips/hand"
    assert est.value == pytest.approx(exploitability(result.strategy), abs=1e-12)
    assert est.value >= -1e-9
    if name in ("best_response", "rnr_0.5"):
        assert result.gain_vs_model >= -1e-9  # never worse than the equilibrium vs the model
    with pytest.raises(TypeError, match="TabularPolicy"):
        engine.adjust(model.to_mapping(), baseline)
    other = compile_tree(KuhnPoker() if tree.game_name == "leduc" else LeducHoldem())
    with pytest.raises(ValueError, match="different game trees"):
        engine.adjust(model, TabularPolicy.uniform(other))


def test_rnr_engine_validates_p():
    with pytest.raises(ValueError, match="p must be"):
        RestrictedNashEngine(1.5)

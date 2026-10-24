"""Push/fold opponents: bots, observation, likelihood and fitting (P3.2)."""

import numpy as np
import pytest

from poker_lab.games.preflop import CLASS_INDEX, CLASSES, combo_counts, load_matchups
from poker_lab.games.push_fold import BB, SB, aggressive_probabilities, push_fold_tree
from poker_lab.games.push_fold_bots import (
    ARCHETYPES,
    HandLog,
    PushFoldBotParams,
    PushFoldPrior,
    fit_frequency,
    fit_likelihood,
    log_likelihood,
    percentiles,
    range_shares,
    sample_params,
    simulate,
    strategy,
)
from poker_lab.solvers.lp import solve_lp


@pytest.fixture(scope="module")
def hero():
    sol = solve_lp(push_fold_tree(10.0))
    return aggressive_probabilities(sol.policy, SB), aggressive_probabilities(sol.policy, BB)


@pytest.fixture(scope="module")
def prior():
    rng = np.random.default_rng(7)
    return PushFoldPrior.fit([sample_params(a, rng) for a in ARCHETYPES for _ in range(40)])


def test_percentiles_are_a_cdf_in_each_order():
    combos = combo_counts()
    for order in ("equity", "pairs_first"):
        pct = percentiles(order)
        assert pct.min() > 0
        assert pct.max() < 1
        assert len(set(np.round(pct, 12))) == len(CLASSES)
        # Sorting by percentile and adding half-blocks rebuilds the cumulative combo count.
        k = np.argsort(pct)
        cum = np.cumsum(combos[k]) - combos[k] / 2
        assert np.allclose(pct[k] * 1326, cum)
    eq, pf = percentiles("equity"), percentiles("pairs_first")
    assert eq.argmin() == CLASS_INDEX["AA"]
    pairs = [i for i, c in enumerate(CLASSES) if len(c) == 2]
    others = [i for i, c in enumerate(CLASSES) if len(c) == 3]
    assert pf[pairs].max() < pf[others].min()


def test_a_sharp_threshold_plays_its_width():
    push, call = range_shares(PushFoldBotParams(0.4, 0.2, 0.004))
    assert push == pytest.approx(0.4, abs=0.02)
    assert call == pytest.approx(0.2, abs=0.02)


def test_simulation_reveals_cards_only_at_showdown(hero):
    rng = np.random.default_rng(0)
    bot = strategy(PushFoldBotParams(0.5, 0.3, 0.03))
    log = simulate(bot, hero, 4000, rng)
    assert (log.shown <= (log.acted & (log.action == 1))).all()
    # As big blind the bot acts only facing a push; as small blind always.
    assert log.acted[log.bot_is_sb].all()
    # Hidden classes never reach the observations: scramble them and nothing changes.
    scrambled = log.bot_class.copy()
    hidden = ~log.shown
    scrambled[hidden] = rng.integers(0, 169, hidden.sum())
    other = HandLog(log.bot_is_sb, log.hero_class, scrambled, log.acted, log.action, log.shown)
    a, b = log.observations(4000), other.observations(4000)
    for field in ("sb_shown", "sb_hidden", "bb_shown", "bb_hidden"):
        assert (getattr(a, field) == getattr(b, field)).all()


def test_simulated_push_rate_matches_the_strategy(hero):
    params = PushFoldBotParams(0.35, 0.2, 0.03)
    bot = strategy(params)
    log = simulate(bot, hero, 40_000, np.random.default_rng(1))
    obs = log.observations(40_000)
    m = load_matchups()
    expected = float(m.deal_prob.sum(axis=1) @ bot[0])  # the bot is dealt as small blind
    assert obs.pushes / obs.n_sb == pytest.approx(expected, abs=0.01)


def test_likelihood_equals_a_hand_by_hand_sum(hero):
    params = PushFoldBotParams(0.45, 0.25, 0.04)
    model = PushFoldBotParams(0.5, 0.3, 0.05)
    log = simulate(strategy(params), hero, 600, np.random.default_rng(2))
    m = load_matchups()
    cond = m.counts / m.counts.sum(axis=1, keepdims=True)
    push, call = strategy(model)
    total = 0.0
    for k in range(600):
        if not log.acted[k]:
            continue
        p = push if log.bot_is_sb[k] else call
        p_action = p if log.action[k] == 1 else 1 - p
        if log.shown[k]:
            total += np.log(p_action[log.bot_class[k]])
        else:
            total += np.log(cond[log.hero_class[k]] @ p_action)
    assert log_likelihood(model, log.observations(600)) == pytest.approx(total, rel=1e-9)


def test_zero_hands_give_the_population(prior):
    empty = HandLog(*(np.zeros(0, dtype=t) for t in (bool, int, int, bool, int, bool)))
    obs = empty.observations(0)
    fit = fit_likelihood(obs, prior)
    pop = prior.mean_params
    assert fit.push_width == pytest.approx(pop.push_width, abs=1e-4)
    assert fit.call_width == pytest.approx(pop.call_width, abs=1e-4)
    assert fit.temperature == pytest.approx(pop.temperature, rel=1e-3)
    freq = fit_frequency(obs, prior)
    assert freq.push_width == pytest.approx(pop.push_width, abs=1e-3)
    assert freq.call_width == pytest.approx(pop.call_width, abs=1e-3)


@pytest.mark.parametrize("archetype", ["nit", "lag", "calling_station"])
def test_parameters_are_recovered_from_3000_hands(hero, prior, archetype):
    rng = np.random.default_rng(3)
    truth = sample_params(archetype, rng)
    log = simulate(strategy(truth), hero, 3000, rng)
    obs = log.observations(3000)
    for fit in (fit_likelihood(obs, prior), fit_frequency(obs, prior)):
        assert fit.push_width == pytest.approx(truth.push_width, abs=0.05)
        assert fit.call_width == pytest.approx(truth.call_width, abs=0.06)

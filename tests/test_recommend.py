"""P3.4: posterior over a push/fold bot and ranked exploit recommendations with confidence."""

import numpy as np
import pytest

from poker_lab.exploit import PushFoldRecommender
from poker_lab.games.evaluate import best_response, exploitability, head_to_head
from poker_lab.games.push_fold import (
    match_values,
    push_fold_tree,
    strategy_exploitability,
    threshold_policy,
)
from poker_lab.games.push_fold_bots import (
    ARCHETYPES,
    Observations,
    PushFoldBotParams,
    PushFoldPrior,
    log_likelihood,
    log_posterior,
    posterior,
    sample_params,
    simulate,
    strategy,
    to_z,
    weighted_quantile,
)
from poker_lab.strategy import nash_ranges


@pytest.fixture(scope="module")
def prior():
    rng = np.random.default_rng(7)
    return PushFoldPrior.fit([sample_params(a, rng) for a in ARCHETYPES for _ in range(40)])


@pytest.fixture(scope="module")
def recommender(prior):
    return PushFoldRecommender(10.0, prior, draws=1000, seed=3)


def _log(params, n, seed, order="equity"):
    return simulate(strategy(params, order), nash_ranges(10.0), n, np.random.default_rng(seed))


@pytest.mark.parametrize("stack", [3.0, 10.0])
def test_closed_form_values_equal_the_tree(stack):
    rng = np.random.default_rng(int(stack))
    tree = push_fold_tree(stack)
    hero, opp = rng.random((2, 169)), rng.random((3, 2, 169))
    values = match_values(stack, hero[0], hero[1], opp[:, 0], opp[:, 1])
    assert values.shape == (3,)
    h = threshold_policy(tree, push=hero[0], call=hero[1])
    for k in range(3):
        o = threshold_policy(tree, push=opp[k, 0], call=opp[k, 1])
        assert values[k] == pytest.approx(head_to_head(h, o), abs=1e-12)
        assert strategy_exploitability(stack, opp[k, 0], opp[k, 1]) == pytest.approx(
            exploitability(o), abs=1e-12
        )
    batch = strategy_exploitability(stack, opp[:, 0], opp[:, 1])
    assert batch.shape == (3,)
    assert match_values(stack, opp[:, 0], opp[:, 1], hero[0], hero[1]).shape == (3,)


def test_vectorized_log_posterior_is_likelihood_plus_prior(prior):
    obs = _log(PushFoldBotParams(0.6, 0.3, 0.04), 200, 1).observations(200)
    for params in (PushFoldBotParams(0.6, 0.3, 0.04), PushFoldBotParams(0.2, 0.7, 0.01)):
        w = np.array([params.push_width, params.call_width])
        expected = log_likelihood(params, obs)
        expected += sum(
            a * np.log(x) + b * np.log1p(-x)
            for x, (a, b) in zip(w, (prior.push, prior.call), strict=True)
        )
        expected += -0.5 * ((np.log(params.temperature) - prior.log_t[0]) / prior.log_t[1]) ** 2
        assert log_posterior(to_z(params), obs, prior)[0] == pytest.approx(expected, rel=1e-10)


def test_weighted_quantile():
    rng = np.random.default_rng(0)
    x = rng.normal(size=101)
    w = np.full(101, 1 / 101)
    for q in (0.05, 0.5, 0.95):
        assert weighted_quantile(x, w, q) == np.quantile(x, q, method="inverted_cdf")
    assert weighted_quantile(np.array([1.0, 2.0]), np.array([0.9, 0.1]), 0.5) == 1.0
    assert weighted_quantile(np.array([1.0, 2.0]), np.array([0.04, 0.96]), 0.05) == 2.0
    rows = weighted_quantile(np.vstack([x, -x]), w, [0.1, 0.9])
    assert rows.shape == (2, 2)
    assert rows[1, 0] == pytest.approx(-rows[0, 1])


def test_empty_observations_give_the_prior(prior):
    post = posterior(Observations.empty(), prior, np.random.default_rng(1), draws=4000)
    mean = post.weights @ post.draws
    (ap, bp), (ac, bc) = prior.push, prior.call
    assert mean[0] == pytest.approx(ap / (ap + bp), abs=0.02)
    assert mean[1] == pytest.approx(ac / (ac + bc), abs=0.02)
    mu, sd = prior.log_t
    assert mean[2] == pytest.approx(np.exp(mu + sd**2 / 2), rel=0.05)
    assert post.ess > 2000


@pytest.mark.parametrize(("params", "n"), [((0.25, 0.1, 0.03), 30), ((0.85, 0.55, 0.02), 300)])
def test_importance_sampling_matches_grid_integration(prior, params, n):
    """The weighted draws reproduce the posterior mean found by brute-force integration."""
    obs = _log(PushFoldBotParams(*params), n, 11).observations(n)
    post = posterior(obs, prior, np.random.default_rng(2), draws=4000)
    z = np.column_stack(
        [np.log(post.draws[:, :2] / (1 - post.draws[:, :2])), np.log(post.draws[:, 2])]
    )
    mu = post.weights @ z
    sd = np.sqrt(post.weights @ (z - mu) ** 2)
    axes = [np.linspace(m - 7 * s, m + 7 * s, 41) for m, s in zip(mu, sd, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    lp = log_posterior(grid, obs, prior)
    w = np.exp(lp - lp.max())
    w /= w.sum()
    grid_draws = np.column_stack([1 / (1 + np.exp(-grid[:, :2])), np.exp(grid[:, 2])])
    grid_mean = w @ grid_draws
    is_mean = post.weights @ post.draws
    is_sd = np.sqrt(post.weights @ (post.draws - is_mean) ** 2)
    assert post.ess > 500
    assert np.all(np.abs(is_mean - grid_mean) < 5 * is_sd / np.sqrt(post.ess))


def test_report_is_ranked_by_the_lower_bound(recommender):
    obs = _log(PushFoldBotParams(0.75, 0.45, 0.03), 100, 5).observations(100)
    report = recommender.recommend(obs)
    lows = [r.gain.lower for r in report.ranked]
    assert lows == sorted(lows, reverse=True)
    names = [r.name for r in report.ranked]
    assert sorted(names) == sorted(
        ["equilibrium", "best_response"] + [f"restricted_nash_p{p:g}" for p in recommender.ps]
    )
    eq = report.get("equilibrium")
    assert (eq.gain.mean, eq.gain.lower, eq.gain.upper, eq.prob_gain) == (0.0, 0.0, 0.0, 0.0)
    assert eq.exploitability < 1e-7
    assert report.best.gain.lower >= 0.0
    if report.best.name != "equilibrium":
        # a positive 5% quantile means less than 5% of the posterior weight is at or below 0
        assert report.best.prob_gain > (1 + recommender.level) / 2
    assert 0.0 < report.ess <= recommender.draws


def test_best_response_to_the_posterior_mean_is_the_bayes_action(recommender):
    """Expected gain is linear in the opponent's probabilities, so the best response to the
    posterior mean has the highest posterior mean gain; along the restricted Nash family both
    the expected gain and the exploitability grow with p."""
    obs = _log(PushFoldBotParams(0.3, 0.65, 0.04), 60, 6).observations(60)
    report = recommender.recommend(obs)
    order = ["equilibrium", *(f"restricted_nash_p{p:g}" for p in recommender.ps), "best_response"]
    recs = [report.get(n) for n in order]
    means = [r.gain.mean for r in recs]
    expl = [r.exploitability for r in recs]
    assert max(means) == pytest.approx(means[-1], abs=1e-12)
    assert np.all(np.diff(means) >= -1e-9)
    assert np.all(np.diff(expl) >= -1e-9)


def test_closed_form_best_response_matches_the_tree(recommender):
    rng = np.random.default_rng(4)
    push, call = rng.random(169), rng.random(169)
    *_, (name, p, hero_push, hero_call) = recommender.candidates(push, call)
    assert (name, p) == ("best_response", 1.0)
    model = threshold_policy(recommender.tree, push=push, call=call)
    br = best_response(model, 0).policy.combine(best_response(model, 1).policy, 0)
    mine = threshold_policy(recommender.tree, push=hero_push, call=hero_call)
    assert head_to_head(mine, model) == pytest.approx(head_to_head(br, model), abs=1e-12)


def test_no_hands_give_the_population_recommendation(recommender):
    report = recommender.recommend(Observations.empty())
    best = report.best
    assert best.name == recommender.population.name
    np.testing.assert_array_equal(best.push, recommender.population.push)
    assert best.over_population.mean == pytest.approx(0.0, abs=1e-12)
    assert "No hands seen" in report.explanation()


def test_reports_are_reproducible(prior, recommender):
    obs = _log(PushFoldBotParams(0.5, 0.2, 0.03), 40, 8).observations(40)
    a, b = recommender.recommend(obs), recommender.recommend(obs)
    assert [r.gain for r in a.ranked] == [r.gain for r in b.ranked]
    other = PushFoldRecommender(10.0, prior, draws=1000, seed=4).recommend(obs)
    assert other.get("best_response").gain != a.get("best_response").gain


def test_strong_evidence_recovers_most_of_the_oracle_gain(recommender):
    truth = PushFoldBotParams(0.9, 0.6, 0.03)
    report = recommender.recommend(_log(truth, 3000, 9).observations(3000))
    bot = strategy(truth)
    eq = recommender.equilibrium

    def true_gain(push, call):
        return float(match_values(10.0, push, call, *bot) - match_values(10.0, *eq, *bot))

    oracle = recommender.candidates(*bot)[-1]
    best = report.best
    assert best.p >= 0.75
    assert true_gain(best.push, best.call) > 0.95 * true_gain(oracle[2], oracle[3])
    assert best.gain.lower <= true_gain(best.push, best.call) <= best.gain.upper
    text = report.explanation()
    assert "Seen 3,000 hands" in text
    assert best.label in text


def test_weak_evidence_explains_why_it_stays_at_equilibrium(recommender):
    obs = _log(PushFoldBotParams(0.55, 0.35, 0.03), 10, 12).observations(10)
    report = recommender.recommend(obs)
    if report.best.name == "equilibrium":
        assert "Play the equilibrium" in report.explanation()
    else:  # the evidence was already strong enough for a deviation
        assert report.best.prob_gain > 0.95


def test_recommender_validates_its_settings(prior):
    with pytest.raises(ValueError, match="level"):
        PushFoldRecommender(10.0, prior, level=1.0)
    with pytest.raises(ValueError, match="increasing"):
        PushFoldRecommender(10.0, prior, ps=(0.5, 0.25))
    with pytest.raises(ValueError, match="increasing"):
        PushFoldRecommender(10.0, prior, ps=(0.0, 0.5))

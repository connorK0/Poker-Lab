"""Empirical-Bayes shrinkage: prior recovery, posterior behavior, interval coverage."""

import numpy as np
import pytest

from poker_lab.profiling import (
    BINOMIAL_STATS,
    N_COUNTERS,
    BetaPrior,
    EmpiricalBayesShrinker,
    aggression_factor,
    fit_beta_binomial,
    fit_normal_prior,
)
from poker_lab.profiling.stats import C


def _simulate(alpha, beta, n_players, mean_opps, rng):
    p = rng.beta(alpha, beta, n_players)
    m = rng.poisson(mean_opps, n_players)
    k = rng.binomial(m, p)
    return p, k, m


def test_beta_binomial_mle_recovers_the_prior():
    rng = np.random.default_rng(0)
    _, k, m = _simulate(4.0, 12.0, 4000, 40, rng)
    prior = fit_beta_binomial(k, m)
    assert prior.mean == pytest.approx(0.25, abs=0.01)
    assert prior.concentration == pytest.approx(16.0, rel=0.2)


def test_posterior_mean_moves_from_prior_to_data():
    prior = BetaPrior(3.0, 7.0)
    assert prior.posterior_mean(np.array(0.0), np.array(0.0)) == pytest.approx(0.3)
    assert prior.posterior_mean(np.array(900.0), np.array(1000.0)) == pytest.approx(0.9, abs=0.01)
    small = prior.posterior_mean(np.array(2.0), np.array(2.0))
    assert 0.3 < small < 1.0  # between prior mean and raw rate (1.0)


def test_credible_intervals_have_nominal_coverage():
    rng = np.random.default_rng(1)
    prior = BetaPrior(2.0, 6.0)
    p, k, m = _simulate(prior.alpha, prior.beta, 5000, 25, rng)
    lo, hi = prior.posterior_interval(k, m)
    coverage = np.mean((p >= lo) & (p <= hi))
    assert coverage == pytest.approx(0.95, abs=0.015)


def test_normal_prior_kappa_from_split_halves():
    rng = np.random.default_rng(2)
    n_players, mu, tau, sigma = 3000, 0.6, 0.1, 0.3
    means = rng.normal(mu, tau, n_players)
    na, nb = rng.integers(5, 40, n_players), rng.integers(5, 40, n_players)
    sa = np.array([rng.normal(x, sigma, k).sum() for x, k in zip(means, na, strict=True)])
    sb = np.array([rng.normal(x, sigma, k).sum() for x, k in zip(means, nb, strict=True)])
    prior = fit_normal_prior(sa + sb, na + nb, (sa, na), (sb, nb))
    assert prior.mean == pytest.approx(mu, abs=0.01)
    assert prior.kappa == pytest.approx(sigma**2 / tau**2, rel=0.2)  # 9 pseudo-observations


def _counters(rng, n_players):
    c = np.zeros((n_players, N_COUNTERS))
    for opp, succ in [
        ("vpip_opp", "vpip"),
        ("pfr_opp", "pfr"),
        ("three_bet_opp", "three_bet"),
        ("fold_to_3bet_opp", "fold_to_3bet"),
        ("cbet_opp", "cbet"),
        ("fold_to_cbet_opp", "fold_to_cbet"),
        ("fold_to_bet_opp", "fold_to_bet"),
        ("saw_flop", "wtsd"),
        ("wtsd", "wsd"),
    ]:
        if not c[:, C[opp]].any():
            c[:, C[opp]] = rng.integers(0, 60, n_players)
        c[:, C[succ]] = rng.binomial(c[:, C[opp]].astype(int), 0.3)
    c[:, C["pf_aggr"]] = rng.integers(0, 30, n_players)
    c[:, C["pf_calls"]] = rng.integers(0, 30, n_players)
    c[:, C["bet_size_n"]] = rng.integers(0, 20, n_players)
    c[:, C["bet_size_x100"]] = c[:, C["bet_size_n"]] * 66
    c[:, C["open_size_n"]] = rng.integers(0, 20, n_players)
    c[:, C["open_size_x100"]] = c[:, C["open_size_n"]] * 300
    c[:, C["hands"]] = 100
    return c


def test_shrinker_end_to_end():
    rng = np.random.default_rng(3)
    totals = _counters(rng, 400)
    shrinker = EmpiricalBayesShrinker.fit(totals)
    out = shrinker.transform(totals[:5], intervals=True)
    assert out.mean.shape == (5, len(shrinker.names))
    assert np.all((out.lower <= out.mean)[:, : len(BINOMIAL_STATS)])
    assert np.all((out.mean <= out.upper)[:, : len(BINOMIAL_STATS)])
    empty = shrinker.transform(np.zeros(N_COUNTERS))
    for j, s in enumerate(BINOMIAL_STATS):
        assert empty.mean[j] == pytest.approx(shrinker.binomial[s.name].mean)
    assert shrinker.raw(np.zeros(N_COUNTERS))[0] == pytest.approx(shrinker.binomial["vpip"].mean)
    assert out.get("bet_size")[0] == pytest.approx(0.66, abs=0.01)
    again = EmpiricalBayesShrinker.from_dict(shrinker.to_dict())
    assert np.allclose(again.transform(totals).mean, shrinker.transform(totals).mean)


def test_shrinker_rejects_impossible_counts():
    rng = np.random.default_rng(4)
    shrinker = EmpiricalBayesShrinker.fit(_counters(rng, 100))
    bad = np.zeros(N_COUNTERS)
    bad[C["vpip"]], bad[C["vpip_opp"]] = 5, 3
    with pytest.raises(ValueError, match="more successes than opportunities"):
        shrinker.transform(bad)


def test_aggression_factor_from_share():
    assert aggression_factor(np.array(0.75)) == pytest.approx(3.0)
    assert np.isfinite(aggression_factor(np.array(1.0)))

"""P3.5: posterior predictive checks of the push/fold opponent model."""

import numpy as np
import pytest

from poker_lab.exploit import PushFoldRecommender
from poker_lab.games.preflop import CLASSES
from poker_lab.games.push_fold_bots import (
    ARCHETYPES,
    HandLog,
    PushFoldBotParams,
    PushFoldPrior,
    percentiles,
    posterior,
    sample_params,
    simulate,
    strategy,
)
from poker_lab.games.push_fold_checks import (
    CHECKS,
    ActionLog,
    CheckResult,
    FitCheck,
    FitChecker,
    _p_value,
    calibration_pvalues,
    check_pvalues,
    hand_bins,
    shown_hands_pvalue,
    stability_pvalue,
)
from poker_lab.strategy import nash_ranges


@pytest.fixture(scope="module")
def prior():
    rng = np.random.default_rng(7)
    return PushFoldPrior.fit([sample_params(a, rng) for a in ARCHETYPES for _ in range(40)])


def _log(params, n, seed, order="equity", hero=None):
    hero = hero or nash_ranges(10.0)
    return simulate(strategy(params, order), hero, n, np.random.default_rng(seed))


def _pvalues(prior, log, seed=0, n=None):
    actions = ActionLog.from_hand_log(log, n)
    post = posterior(actions.observations(), prior, np.random.default_rng(seed), draws=1000)
    return {k: v[0] for k, v in check_pvalues(actions, post, np.random.default_rng(seed)).items()}


def test_action_log_matches_the_hand_log():
    log = _log(PushFoldBotParams(0.6, 0.35, 0.03), 500, 1)
    for n in (0, 37, 500):
        actions = ActionLog.from_hand_log(log, n)
        a, b = actions.observations(), log.observations(n)
        assert a.n_hands == b.n_hands == n
        for key in ("sb_shown", "sb_hidden", "bb_shown", "bb_hidden"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    full = ActionLog.from_hand_log(log)
    assert len(full.action) == int(log.acted.sum())
    assert np.all(full.action[full.shown >= 0] == 1)
    with pytest.raises(ValueError, match="equal lengths"):
        ActionLog(np.array([True]), np.array([1, 0]), np.array([3]), np.array([-1]), 2)
    with pytest.raises(ValueError, match="push or a call"):
        ActionLog(np.array([True]), np.array([0]), np.array([3]), np.array([5]), 1)


def test_hand_bins_split_type_and_strength():
    bins = hand_bins()
    assert bins.shape == (169,)
    assert set(bins) == set(range(12)) - {2, 3}  # every pair is in the top half by equity
    pct = percentiles("equity")
    for k, cls in enumerate(CLASSES):
        kind = 0 if cls[0] == cls[1] else 1 if cls.endswith("s") else 2
        assert bins[k] // 4 == kind
        assert bins[k] % 4 == min(int(pct[k] * 4), 3)


def test_mid_p_counts_ties_half():
    t_obs = np.array([1.0, 1.0, 1.0, 1.0])
    t_rep = np.array([0.0, 1.0, 2.0, 1.0])
    assert _p_value(t_rep, t_obs, mid_p=False) == 0.75
    assert _p_value(t_rep, t_obs, mid_p=True) == pytest.approx(0.25 + 0.5 * 0.5)


def test_a_bot_the_model_describes_is_not_flagged(prior):
    """Under the model, p-values are spread out (conservative), not piled near zero."""
    lows = []
    for k in range(12):
        rng = np.random.default_rng(100 + k)
        params = sample_params(list(ARCHETYPES)[k % 5], rng)
        pv = _pvalues(prior, _log(params, 600, 200 + k), seed=k)
        lows.append(min(pv.values()))
    assert np.mean(np.array(lows) < 0.01) <= 1 / 12


def test_shown_hands_catch_a_wrong_hand_order(prior):
    params = PushFoldBotParams(0.45, 0.35, 0.03)
    pv = _pvalues(prior, _log(params, 1000, 3, order="connectors_first"))
    assert pv["shown_hands"] < 0.01


def test_stability_catches_a_gear_change(prior):
    loose = PushFoldBotParams(0.9, 0.6, 0.03)
    tight = PushFoldBotParams(0.2, 0.1, 0.03)
    hero = nash_ranges(10.0)
    rng = np.random.default_rng(4)
    first = simulate(strategy(loose), hero, 500, rng)
    second = simulate(strategy(tight), hero, 500, rng, first_hand=500)
    assert np.array_equal(
        np.concatenate([first.bot_is_sb, second.bot_is_sb]), np.arange(1000) % 2 == 0
    )
    switched = HandLog.concat([first, second])
    assert _pvalues(prior, switched)["stability"] < 0.01
    assert _pvalues(prior, _log(tight, 1000, 5))["stability"] > 0.05


def test_checks_do_not_depend_on_how_the_hero_plays(prior):
    """The shown-hands check conditions on the hero's own class, so a hero who calls far
    tighter than equilibrium (and so sees a different mix of the bot's hands) is no problem."""
    push, _ = nash_ranges(10.0)
    tight_hero = (push, (percentiles("equity") < 0.1).astype(float))
    params = PushFoldBotParams(0.7, 0.4, 0.03)
    log = _log(params, 1500, 6, hero=tight_hero)
    actions = ActionLog.from_hand_log(log)
    post = posterior(actions.observations(), prior, np.random.default_rng(1), draws=1000)
    p, n = shown_hands_pvalue(actions, post, np.random.default_rng(2))
    assert n > 30
    assert p > 0.05


def test_checks_without_data_return_one(prior):
    empty = ActionLog(np.zeros(0, bool), np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), 0)
    post = posterior(empty.observations(), prior, np.random.default_rng(0), draws=200)
    assert shown_hands_pvalue(empty, post, np.random.default_rng(0)) == (1.0, 0)
    assert stability_pvalue(empty, post, np.random.default_rng(0)) == (1.0, 0)


def test_calibration_and_thresholds(prior):
    hero = nash_ranges(10.0)
    pv = [
        calibration_pvalues(prior, hero, [30, 300], np.random.default_rng(k), draws=100)
        for k in range(8)
    ]
    assert set(pv[0]) == {30, 300}
    assert all(0.0 <= p[n][c] <= 1.0 for p in pv for n in (30, 300) for c in CHECKS)
    checker = FitChecker.from_pvalues(pv, level=0.5)
    for n in (30, 300):
        for c in CHECKS:
            assert checker.thresholds[n][c] == pytest.approx(
                np.quantile([p[n][c] for p in pv], 0.25)
            )
    assert checker.thresholds_for(10) == checker.thresholds[30]
    assert checker.thresholds_for(299) == checker.thresholds[30]
    assert checker.thresholds_for(5000) == checker.thresholds[300]
    with pytest.raises(ValueError, match="at least one"):
        FitChecker({})


def test_fit_check_flags_and_describes():
    ok = FitCheck(
        (CheckResult("shown_hands", 0.4, 0.05, 30), CheckResult("stability", 0.3, 0.2, 90))
    )
    assert not ok.flagged
    assert ok.describe().startswith("Model check passed")
    bad = FitCheck(
        (CheckResult("shown_hands", 0.01, 0.05, 30), CheckResult("stability", 0.3, 0.2, 90))
    )
    assert bad.flagged
    assert bad.get("shown_hands").failed
    assert "hand order" in bad.describe()
    assert not CheckResult("stability", 0.0, None, 5).failed  # uncalibrated: never fails


def test_recommender_falls_back_when_the_model_does_not_fit(prior):
    params = PushFoldBotParams(0.85, 0.6, 0.03)
    actions = ActionLog.from_hand_log(_log(params, 300, 8))
    always = FitChecker({0: dict.fromkeys(CHECKS, 1.01)}, draws=100)
    never = FitChecker({0: dict.fromkeys(CHECKS, 0.0)}, draws=100)
    flagged = PushFoldRecommender(10.0, prior, draws=500, checker=always).recommend(actions)
    passed = PushFoldRecommender(10.0, prior, draws=500, checker=never).recommend(actions)
    assert flagged.fit.flagged
    assert flagged.advised.name == "equilibrium"
    assert flagged.best.name == passed.best.name != "equilibrium"
    assert "Model check failed" in flagged.explanation()
    assert "play the equilibrium" in flagged.explanation()
    assert not passed.fit.flagged
    assert passed.advised is passed.best
    assert "Model check passed" in passed.explanation()
    plain = PushFoldRecommender(10.0, prior, draws=500).recommend(actions.observations())
    assert plain.fit is None
    assert plain.advised is plain.best

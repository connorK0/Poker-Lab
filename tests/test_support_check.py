"""SupportCheck: conformal flags for profiles unlike every training class."""

import numpy as np
import pytest
from scipy import special

from poker_lab.profiling import SupportCheck
from poker_lab.profiling.features import N_RATE_FEATURES, N_SIZE_FEATURES

SIZES = (25, 400)
CENTERS = np.array([[-2.0] * 6 + [-1.0] * 6, [0.0] * 12, [1.5] * 6 + [0.5] * 6])


def _rows(rng, n_players, windows=8, shift=0.0, centers=CENTERS):
    """Profile-feature rows: players of three classes; noise shrinks with the sample size."""
    X, y, g = [], [], []
    for p in range(n_players):
        k = p % len(centers)
        own = centers[k] + 0.3 * rng.standard_normal(12) + shift  # the player's true profile
        for n in SIZES:
            z = own + rng.standard_normal((windows, 12)) * 2.0 / np.sqrt(n)
            rates = special.expit(z[:, :N_RATE_FEATURES])
            sizes = np.exp(z[:, N_RATE_FEATURES : N_RATE_FEATURES + N_SIZE_FEATURES])
            X.append(np.column_stack([rates, sizes, np.full(windows, np.log(n))]))
            y += [k] * windows
            g += [p] * windows
    return np.concatenate(X), np.array(y), np.array(g)


@pytest.fixture(scope="module")
def check():
    X, y, g = _rows(np.random.default_rng(0), 300)
    return SupportCheck(alpha=0.05, random_state=1).fit(X, y, g)


def test_held_out_split_is_by_player(check):
    assert check.fit_players
    assert check.held_out_players
    assert not check.fit_players & check.held_out_players
    counts = check.held_out_windows()
    assert set(counts) == set(SIZES)
    assert all(c == 8 * len(check.held_out_players) for c in counts.values())


def test_false_alarm_rate_is_about_alpha_on_new_players(check):
    X, _, _ = _rows(np.random.default_rng(2), 600)
    p = check.pvalue(X)
    assert ((p > 0) & (p <= 1)).all()
    rate = check.flags(X).mean()
    assert 0.02 < rate < 0.09  # nominal 5%; windows of one player are correlated


def test_flags_players_unlike_every_class(check):
    X, _, _ = _rows(np.random.default_rng(3), 90, shift=2.5)
    big = X[:, -1] == np.log(400)
    assert check.flags(X[big]).mean() > 0.95


def test_nearest_size_in_log_hands(check):
    X, _, _ = _rows(np.random.default_rng(4), 30)
    at_400 = X[X[:, -1] == np.log(400)]
    moved = at_400.copy()
    moved[:, -1] = np.log(300)  # nearer to 400 than to 25 in log(hands)
    assert np.array_equal(check.score(moved), check.score(at_400))


def test_small_classes_use_the_pooled_covariance():
    X, y, g = _rows(np.random.default_rng(5), 30, windows=1)  # ~8 fit rows per class
    small = SupportCheck(alpha=0.2, random_state=0).fit(X, y, g)
    shifted, _, _ = _rows(np.random.default_rng(6), 30, windows=1, shift=3.0)
    assert small.flags(shifted).mean() > 0.9
    X, y, g = _rows(np.random.default_rng(5), 6, windows=1)
    with pytest.raises(ValueError, match="rows at log"):
        SupportCheck().fit(X, y, g)
    with pytest.raises(ValueError, match="alpha"):
        SupportCheck(alpha=0)
    with pytest.raises(RuntimeError):
        SupportCheck().score(X)

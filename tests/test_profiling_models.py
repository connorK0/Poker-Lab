"""Windows, features, classifier calibration, regression, metrics, intervals."""

import numpy as np
import pytest

from poker_lab.evaluation import (
    accuracy,
    bootstrap_ci,
    confusion_matrix,
    expected_calibration_error,
    log_loss,
    reliability,
    t_interval,
    wilson_interval,
)
from poker_lab.profiling import (
    N_COUNTERS,
    ArchetypeClassifier,
    ParamRegressor,
    fit_temperature,
    oos_r2,
    prefix_totals,
    to_model_space,
    window_totals,
)


def test_window_totals_match_manual_sums():
    rng = np.random.default_rng(0)
    counters = rng.integers(0, 3, size=(4, 100, N_COUNTERS))
    cum = counters.cumsum(axis=1)
    w = window_totals(cum, 25, max_windows=3)
    assert w.totals.shape == (12, N_COUNTERS)
    for row in range(12):
        p, i = w.player[row], w.index[row]
        assert np.array_equal(w.totals[row], counters[p, 25 * i : 25 * (i + 1)].sum(axis=0))
    w2 = window_totals(cum, 30, max_windows=10, offset=5)
    assert len(w2.totals) == 4 * 3  # (100 - 5) // 30 = 3 windows each
    assert np.array_equal(prefix_totals(cum, 10), counters[:, :10].sum(axis=1))
    assert not prefix_totals(cum, 0).any()
    with pytest.raises(ValueError, match="not enough hands"):
        window_totals(cum, 200, max_windows=1)


def test_model_space_transform_is_finite():
    x = np.zeros((3, 13))
    x[:, :10] = [[0.0], [0.5], [1.0]]
    x[:, 10:12] = [[0.0], [1.0], [100.0]]
    z = to_model_space(x)
    assert np.isfinite(z).all()
    assert z[1, 0] == pytest.approx(0.0)


def _blobs(rng, n_per, n_classes=5, spread=1.0, dim=13):
    centers = rng.normal(0, 2.0, size=(n_classes, dim))
    X = np.concatenate(
        [centers[c] + spread * rng.normal(size=(n_per, dim)) for c in range(n_classes)]
    )
    y = np.repeat(np.arange(n_classes), n_per)
    groups = np.arange(len(y)) // 4  # 4 rows per "player"
    X[:, :10] = 1 / (1 + np.exp(-X[:, :10]))  # rates in (0, 1)
    X[:, 10:12] = np.exp(X[:, 10:12] / 3)
    return X, y, groups


@pytest.mark.parametrize("model", ["lr", "gbm"])
def test_classifier_learns_and_calibrates(model):
    rng = np.random.default_rng(1)
    X, y, groups = _blobs(rng, 400, spread=2.5)
    Xt, yt, _ = _blobs(np.random.default_rng(1), 200, spread=2.5)  # same centers, new draws
    clf = ArchetypeClassifier(model, calibrate=True, random_state=0).fit(X, y, groups)
    proba = clf.predict_proba(Xt)
    assert proba.shape == (len(yt), 5)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert accuracy(yt, proba) > 0.6
    for mode in ("none", "by_size"):
        other = clf.predict_proba(Xt, calibration=mode)
        assert (other.argmax(1) == proba.argmax(1)).all()  # temperature never changes the argmax
    assert 0.05 <= clf.temperature <= 20


def test_per_size_temperature_corrects_size_dependent_miscalibration():
    """A model that is overconfident at one sample size and underconfident at another cannot
    be fixed by one temperature, but can be by a temperature per size."""
    rng = np.random.default_rng(7)
    n, k = 6000, 5
    logits = rng.normal(0, 1.5, (n, k))
    true_p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=p) for p in true_p])
    size = np.where(np.arange(n) % 2 == 0, 25, 1000)
    distortion = np.where(size == 25, 3.0, 0.4)[:, None]  # over- vs under-confident
    distorted = np.exp(np.log(true_p) * distortion)
    distorted /= distorted.sum(1, keepdims=True)

    class _Fixed:  # stands in for a fitted model returning the distorted probabilities
        def predict_proba(self, X):
            return distorted[X[:, 0].astype(int)]

    X = np.column_stack([np.arange(n), np.log(size)]).astype(float)
    clf = ArchetypeClassifier("lr")
    clf.model = _Fixed()
    log_p = clf._log_proba_uncalibrated(X)
    clf.temperature = fit_temperature(log_p, y)
    grid = np.log([25.0, 1000.0])
    temps = [fit_temperature(log_p[size == s], y[size == s]) for s in (25, 1000)]
    clf.size_temperatures = (grid, np.array(temps))
    assert temps[0] == pytest.approx(3.0, rel=0.2)
    assert temps[1] == pytest.approx(0.4, rel=0.2)
    ece_global = expected_calibration_error(y, clf.predict_proba(X, calibration="global"))
    ece_by_size = expected_calibration_error(y, clf.predict_proba(X, calibration="by_size"))
    assert ece_by_size < 0.5 * ece_global


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(2)
    n, k = 4000, 5
    logits = rng.normal(0, 1, (n, k))
    true_p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=p) for p in true_p])
    overconfident = np.log(true_p) * 3.0  # same ranking, temperature 1/3
    t = fit_temperature(overconfident, y)
    assert t == pytest.approx(3.0, rel=0.15)


def test_regression_recovers_signal():
    rng = np.random.default_rng(3)
    X = rng.uniform(0.05, 0.95, size=(2000, 13))
    X[:, 10:12] = rng.uniform(0.5, 2, size=(2000, 2))
    Y = np.stack([X[:, 0] * 0.8 + 0.1, X[:, 1] ** 2, rng.uniform(size=2000)], axis=1)
    for kind in ("gbm", "ridge"):
        reg = ParamRegressor(kind).fit(X[:1500], Y[:1500])
        r2 = oos_r2(Y[1500:], reg.predict(X[1500:]), reg.train_mean)
        assert r2[0] > 0.9
        assert r2[1] > 0.8
        assert r2[2] < 0.05  # pure noise is not predictable


def test_metrics():
    y = np.array([0, 1, 2, 1])
    uniform = np.full((4, 3), 1 / 3)
    assert log_loss(y, uniform) == pytest.approx(np.log(3))
    perfect = np.eye(3)[y]
    assert accuracy(y, perfect) == 1.0
    assert confusion_matrix(y, perfect.argmax(1), 3).trace() == 4
    rng = np.random.default_rng(4)
    conf = rng.uniform(0.3, 1.0, 20000)
    correct = rng.uniform(size=20000) < conf  # calibrated by construction
    proba = np.stack([conf, 1 - conf], axis=1)
    labels = np.where(correct, 0, 1)
    assert expected_calibration_error(labels, proba) < 0.02
    rel = reliability(labels, proba, n_bins=10)
    assert rel.count.sum() == 20000


def test_intervals():
    rng = np.random.default_rng(5)
    hits = 0
    for _ in range(400):
        x = rng.normal(1.0, 2.0, 8)
        ci = t_interval(x)
        hits += ci.lower <= 1.0 <= ci.upper
    assert hits / 400 == pytest.approx(0.95, abs=0.04)
    b = bootstrap_ci(rng.normal(0, 1, 500), n_boot=500, rng=1)
    assert b.lower < 0 < b.upper
    w = wilson_interval(0, 20)
    assert w.lower == 0.0
    assert 0 < w.upper < 0.2
    assert np.isnan(t_interval([1.0]).lower)

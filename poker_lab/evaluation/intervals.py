"""Confidence intervals.

* ``t_interval``: mean +/- t-quantile * SE. Used across independent replicates (seeds).
* ``bootstrap_ci``: percentile bootstrap of any statistic, resampling units (players or
  hands), never individual decisions within a unit when units are the independent draw.
* ``wilson_interval``: binomial proportion; well behaved near 0 and 1.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    n: int
    method: str
    confidence: float = 0.95

    @property
    def half_width(self) -> float:
        return (self.upper - self.lower) / 2

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "n": self.n,
            "method": self.method,
            "confidence": self.confidence,
        }


def t_interval(values: Sequence[float] | np.ndarray, confidence: float = 0.95) -> Interval:
    """Student-t interval for the mean of independent replicates (NaNs dropped)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, "t", confidence)
    mean = float(x.mean())
    if n == 1:
        return Interval(mean, float("nan"), float("nan"), 1, "t", confidence)
    se = float(x.std(ddof=1) / np.sqrt(n))
    q = float(stats.t.ppf(0.5 + confidence / 2, df=n - 1))
    return Interval(mean, mean - q * se, mean + q * se, n, "t", confidence)


def bootstrap_ci(
    data: np.ndarray | Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator | int | None = 0,
) -> Interval:
    """Percentile bootstrap over the first axis of ``data`` (the independent units)."""
    x = np.asarray(data)
    n = len(x)
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    estimate = float(statistic(x))
    if n < 2:
        return Interval(estimate, float("nan"), float("nan"), n, "bootstrap", confidence)
    idx = gen.integers(0, n, size=(n_boot, n))
    boots = np.array([statistic(x[i]) for i in idx], dtype=float)
    alpha = (1 - confidence) / 2
    lo, hi = np.nanquantile(boots, [alpha, 1 - alpha])
    return Interval(estimate, float(lo), float(hi), n, "bootstrap", confidence)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Interval:
    if trials == 0:
        return Interval(float("nan"), 0.0, 1.0, 0, "wilson", confidence)
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return Interval(p, float(center - half), float(center + half), trials, "wilson", confidence)

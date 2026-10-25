"""Empirical-Bayes shrinkage of opponent stats.

Binomial stats (VPIP, PFR, ...): each player's true rate p_i is drawn from a population prior
Beta(alpha, beta) and the player shows k_i successes in m_i opportunities. ``fit_beta_binomial``
fits the prior by maximizing the beta-binomial marginal likelihood over a *training* population.
The posterior Beta(alpha + k, beta + m - k) gives a shrunk estimate (posterior mean) and a
credible interval. With few opportunities the estimate stays near the population mean; as
evidence accumulates it moves to the player's own rate.

Continuous stats (mean bet size, open size): normal-normal shrinkage with a pseudo-count,
posterior mean = (kappa * mu + sum) / (kappa + n), where kappa = sigma^2 / tau^2 (within-player
variance over between-player variance). sigma^2 and tau^2 are estimated by method of moments
from two disjoint halves of each training player's hands; this needs only sums and counts.

The aggression factor is shrunk through the aggression share s = aggr / (aggr + calls), which
is binomial; AF = s / (1 - s).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

from poker_lab.profiling.stats import BINOMIAL_STATS, C, binomial_counts

CONTINUOUS_STATS: tuple[tuple[str, str, str, float], ...] = (
    # name, sum counter, count counter, scale applied to the sum
    ("bet_size", "bet_size_x100", "bet_size_n", 0.01),
    ("open_size", "open_size_x100", "open_size_n", 0.01),
)


@dataclass(frozen=True)
class BetaPrior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def concentration(self) -> float:
        """Prior strength in pseudo-opportunities."""
        return self.alpha + self.beta

    def posterior_mean(self, k: np.ndarray, m: np.ndarray) -> np.ndarray:
        return (self.alpha + k) / (self.alpha + self.beta + m)

    def posterior_interval(
        self, k: np.ndarray, m: np.ndarray, level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.alpha + k, self.beta + m - k
        # Beta quantiles; betaincinv equals stats.beta.ppf bit for bit, ~25x faster per call.
        lo = special.betaincinv(a, b, (1 - level) / 2)
        hi = special.betaincinv(a, b, (1 + level) / 2)
        return lo, hi


def fit_beta_binomial(k: np.ndarray, m: np.ndarray) -> BetaPrior:
    """Maximum marginal-likelihood Beta prior for success counts ``k`` out of ``m``."""
    k = np.asarray(k, dtype=float)
    m = np.asarray(m, dtype=float)
    keep = m > 0
    k, m = k[keep], m[keep]
    if len(k) == 0:
        raise ValueError("no opportunities to fit a prior on")
    pooled = np.clip(k.sum() / m.sum(), 1e-4, 1 - 1e-4)

    def nll(theta: np.ndarray) -> float:
        mu = special.expit(theta[0])
        kappa = np.exp(theta[1])
        a, b = mu * kappa, (1 - mu) * kappa
        ll = special.betaln(k + a, m - k + b) - special.betaln(a, b)
        return -float(ll.sum())

    start = np.array([special.logit(pooled), np.log(10.0)])
    res = optimize.minimize(nll, start, method="L-BFGS-B", bounds=[(-9, 9), (-3, 12)])
    mu = float(special.expit(res.x[0]))
    kappa = float(np.exp(res.x[1]))
    return BetaPrior(alpha=mu * kappa, beta=(1 - mu) * kappa)


@dataclass(frozen=True)
class NormalPrior:
    mean: float
    kappa: float  # pseudo-count: weight of the prior mean in observations

    def posterior_mean(self, total: np.ndarray, n: np.ndarray) -> np.ndarray:
        return (self.kappa * self.mean + total) / (self.kappa + n)


def fit_normal_prior(
    total: np.ndarray,
    n: np.ndarray,
    half_a: tuple[np.ndarray, np.ndarray] | None = None,
    half_b: tuple[np.ndarray, np.ndarray] | None = None,
    default_kappa: float = 5.0,
) -> NormalPrior:
    """Prior mean from pooled data; kappa = sigma^2 / tau^2 from split halves (if given)."""
    total, n = np.asarray(total, float), np.asarray(n, float)
    mean = float(total.sum() / max(n.sum(), 1.0))
    if half_a is None or half_b is None:
        return NormalPrior(mean, default_kappa)
    (sa, na), (sb, nb) = half_a, half_b
    ok = (na > 0) & (nb > 0)
    if ok.sum() < 3:  # too few players with data in both halves to estimate tau^2
        return NormalPrior(mean, default_kappa)
    xa, xb = sa[ok] / na[ok], sb[ok] / nb[ok]
    # E[(xa - xb)^2] = sigma^2 (1/na + 1/nb); Cov(xa, xb) across players = tau^2.
    sigma2 = float(np.mean((xa - xb) ** 2 / (1 / na[ok] + 1 / nb[ok])))
    tau2 = float(np.cov(xa, xb)[0, 1])
    if not np.isfinite(tau2) or tau2 <= 1e-12:
        return NormalPrior(mean, default_kappa)
    return NormalPrior(mean, float(np.clip(sigma2 / tau2, 0.1, 1e4)))


@dataclass(frozen=True)
class ShrunkStats:
    """Posterior summaries; arrays have shape ``(..., n_stats)`` in ``names`` order."""

    names: tuple[str, ...]
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    opportunities: np.ndarray

    def get(self, name: str) -> np.ndarray:
        return self.mean[..., self.names.index(name)]


class EmpiricalBayesShrinker:
    """Population priors for every profile stat, fit on training players only."""

    def __init__(self, binomial: dict[str, BetaPrior], continuous: dict[str, NormalPrior]) -> None:
        self.binomial = binomial
        self.continuous = continuous

    @classmethod
    def fit(
        cls,
        totals: np.ndarray,
        halves: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> EmpiricalBayesShrinker:
        """``totals``: per-player counter totals (players x counters). ``halves``: the same for
        two disjoint halves of each player's hands, used to fit continuous-stat pseudo-counts."""
        k, m = binomial_counts(totals)
        binomial = {
            s.name: fit_beta_binomial(k[:, j], m[:, j]) for j, s in enumerate(BINOMIAL_STATS)
        }
        continuous = {}
        for name, s_col, n_col, scale in CONTINUOUS_STATS:
            split = None
            if halves is not None:
                split = [(h[:, C[s_col]] * scale, h[:, C[n_col]]) for h in halves]
            continuous[name] = fit_normal_prior(
                totals[:, C[s_col]] * scale,
                totals[:, C[n_col]],
                *(split if split is not None else (None, None)),
            )
        return cls(binomial, continuous)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in BINOMIAL_STATS) + tuple(c[0] for c in CONTINUOUS_STATS)

    def transform(self, counts: np.ndarray, *, intervals: bool = False) -> ShrunkStats:
        """Shrunk stats for counter totals of shape (..., N_COUNTERS)."""
        counts = np.asarray(counts, dtype=float)
        k, m = binomial_counts(counts)
        if np.any(k > m) or np.any(k < 0):
            raise ValueError("counter totals have more successes than opportunities")
        means, lows, highs, opps = [], [], [], []
        for j, s in enumerate(BINOMIAL_STATS):
            prior = self.binomial[s.name]
            means.append(prior.posterior_mean(k[..., j], m[..., j]))
            if intervals:
                lo, hi = prior.posterior_interval(k[..., j], m[..., j])
            else:
                lo = hi = np.full(k[..., j].shape, np.nan)
            lows.append(lo)
            highs.append(hi)
            opps.append(m[..., j])
        for name, s_col, n_col, scale in CONTINUOUS_STATS:
            prior = self.continuous[name]
            total, n = counts[..., C[s_col]] * scale, counts[..., C[n_col]]
            means.append(prior.posterior_mean(total, n))
            lows.append(np.full(n.shape, np.nan))
            highs.append(np.full(n.shape, np.nan))
            opps.append(n)
        return ShrunkStats(
            names=self.names,
            mean=np.stack(means, axis=-1),
            lower=np.stack(lows, axis=-1),
            upper=np.stack(highs, axis=-1),
            opportunities=np.stack(opps, axis=-1),
        )

    def raw(self, counts: np.ndarray) -> np.ndarray:
        """Unshrunk rates in ``names`` order; where there is no data, the prior mean is used
        (the minimal imputation needed to have a feature at all)."""
        counts = np.asarray(counts, dtype=float)
        k, m = binomial_counts(counts)
        cols = []
        for j, s in enumerate(BINOMIAL_STATS):
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = k[..., j] / m[..., j]
            cols.append(np.where(m[..., j] > 0, rate, self.binomial[s.name].mean))
        for name, s_col, n_col, scale in CONTINUOUS_STATS:
            total, n = counts[..., C[s_col]] * scale, counts[..., C[n_col]]
            with np.errstate(invalid="ignore", divide="ignore"):
                avg = total / n
            cols.append(np.where(n > 0, avg, self.continuous[name].mean))
        return np.stack(cols, axis=-1)

    def to_dict(self) -> dict:
        return {
            "binomial": {n: {"alpha": p.alpha, "beta": p.beta} for n, p in self.binomial.items()},
            "continuous": {
                n: {"mean": p.mean, "kappa": p.kappa} for n, p in self.continuous.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> EmpiricalBayesShrinker:
        return cls(
            {n: BetaPrior(v["alpha"], v["beta"]) for n, v in d["binomial"].items()},
            {n: NormalPrior(v["mean"], v["kappa"]) for n, v in d["continuous"].items()},
        )


def aggression_factor(share: np.ndarray) -> np.ndarray:
    """AF from the aggression share s = aggr / (aggr + calls)."""
    share = np.clip(share, 1e-6, 1 - 1e-6)
    return share / (1 - share)

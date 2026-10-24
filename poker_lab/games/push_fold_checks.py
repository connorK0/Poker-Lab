"""Posterior predictive checks of the push/fold opponent model (P3.5).

The likelihood model of ``push_fold_bots`` assumes that a bot plays one stationary soft threshold
on one hand order (equity against a random hand). When that is wrong the posterior concentrates
on a wrong answer and its intervals stop meaning what they say (P3.4). These checks ask whether
the observed decisions look like data the fitted model could have produced.

**Posterior predictive p-values** (Rubin 1984; Gelman, Meng and Stern 1996). For a discrepancy
``T(y, theta)``, draw ``theta_r`` from the posterior and replicate the data ``y_r`` from the model
with ``theta_r``, holding fixed what the bot did not choose (which decisions it faced, the hero's
own class each time). Then ``p = Pr(T(y_r, theta_r) >= T(y, theta_r))``: a small ``p`` says the
observed data are more extreme than the fitted model produces.

Two discrepancies, each aimed at one assumption:

* **Shown hands** (the hand order). A shown hand is the bot's class at a showdown. Given the
  hero's own class ``h``, the bot's class ``i`` when it pushed (or called) has probability
  ``q(i | h)`` proportional to ``P(i | h) push(i)`` (or ``call(i)``): card removal times the
  model's range. The hero's own decision depends only on ``h``, so conditioning on it is exact
  and the hero's strategy is not needed. Classes are grouped into bins (pair, suited, offsuit x
  quartiles of equity: 10 non-empty bins, since every pair is in the top half) and ``T`` is
  Pearson's chi-square between observed and expected bin counts, summed over both seats
  (``statistic="chi2"``). ``statistic="loglik"`` uses the shown classes' negative log
  probability instead; a development-seed pilot found it quicker against one wrong hand order
  but blind to another, so chi-square is the default.
* **Stability** (stationarity). For each seat, the residuals ``x_t - p_t`` of the bot's decisions
  in hand order, with ``p_t`` the model's probability of an aggressive action given only the
  hero's class. ``T`` is the largest absolute partial sum (a CUSUM) divided by the square root
  of the summed Bernoulli variances, the larger of the two seats. A bot that changes gear
  leaves a long run of same-signed residuals.

**Calibration.** Posterior predictive p-values are conservative: the posterior is fitted to the
same data, so under the true model they pile up toward 0.5 instead of spreading uniformly
(Bayarri and Berger 2000), and a fixed 0.05 cut-off would rarely fire. ``FitChecker`` sets each
check's threshold at a quantile of its p-values on data simulated from the model itself: bots
drawn from the prior, played by the same hero (calibrated posterior predictive p-values; Hjort,
Dahl and Steinbakk 2006). With the two checks at ``level / 2`` each, a bot the model describes
is flagged with probability about ``level``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from poker_lab.games.preflop import CLASSES, Matchups, load_matchups
from poker_lab.games.push_fold_bots import (
    HandLog,
    Observations,
    Posterior,
    PushFoldPrior,
    opponent_class_probs,
    percentiles,
    posterior,
    simulate,
    strategy,
)

CHECKS = ("shown_hands", "stability")
N_BINS = 12


def hand_bins(matchups: Matchups | None = None) -> np.ndarray:
    """Bin of each class: 4 x type (0 pair, 1 suited, 2 offsuit) + equity quartile."""
    pct = percentiles("equity", matchups)
    quartile = np.minimum((pct * 4).astype(int), 3)
    kind = np.array([0 if c[0] == c[1] else 1 if c.endswith("s") else 2 for c in CLASSES])
    return 4 * kind + quartile


@dataclass(frozen=True, eq=False)
class ActionLog:
    """The bot's observed decisions in hand order: its seat, its action (1 push or call, 0
    fold), the hero's own class, and the bot's class when shown (-1 otherwise)."""

    is_sb: np.ndarray  # bool: small blind (push or fold); else big blind facing a push
    action: np.ndarray  # int
    hero_class: np.ndarray  # int
    shown: np.ndarray  # int: the bot's class at a showdown, -1 if not shown
    n_hands: int  # hands played, including those where the bot made no decision

    def __post_init__(self) -> None:
        sizes = {len(self.is_sb), len(self.action), len(self.hero_class), len(self.shown)}
        if len(sizes) != 1:
            raise ValueError("ActionLog columns must have equal lengths")
        if np.any((self.shown >= 0) & (self.action != 1)):
            raise ValueError("cards are shown only after a push or a call")

    @classmethod
    def from_hand_log(cls, log: HandLog, n: int | None = None) -> ActionLog:
        n = len(log.action) if n is None else n
        sel = log.acted[:n]
        shown = np.where(log.shown[:n], log.bot_class[:n], -1)[sel]
        return cls(
            log.bot_is_sb[:n][sel],
            log.action[:n][sel].astype(int),
            log.hero_class[:n][sel],
            shown,
            n,
        )

    def observations(self) -> Observations:
        out = {"sb_shown": np.zeros(169), "sb_hidden": np.zeros((2, 169))}
        out |= {"bb_shown": np.zeros(169), "bb_hidden": np.zeros((2, 169))}
        for seat, is_sb in (("sb", True), ("bb", False)):
            sel = self.is_sb == is_sb
            shown = sel & (self.shown >= 0)
            hidden = sel & (self.shown < 0)
            np.add.at(out[f"{seat}_shown"], self.shown[shown], 1)
            np.add.at(out[f"{seat}_hidden"], (self.action[hidden], self.hero_class[hidden]), 1)
        return Observations(self.n_hands, **out)


@dataclass(frozen=True)
class CheckResult:
    name: str
    p_value: float
    threshold: float | None  # flag when p_value < threshold (None: not calibrated)
    n: int  # data the check used: shown hands, or decisions

    @property
    def failed(self) -> bool:
        return self.threshold is not None and self.p_value < self.threshold


@dataclass(frozen=True)
class FitCheck:
    results: tuple[CheckResult, ...]

    @property
    def flagged(self) -> bool:
        return any(r.failed for r in self.results)

    def get(self, name: str) -> CheckResult:
        return next(r for r in self.results if r.name == name)

    def describe(self) -> str:
        what = {
            "shown_hands": "the hands it showed do not match the hand order the model assumes",
            "stability": "its push and call rates changed during the session",
        }
        failed = [r for r in self.results if r.failed]
        if not failed:
            return "Model check passed: the observed play looks like data the model produces."
        parts = [
            f"{what[r.name]} (p = {r.p_value:.3f}, threshold {r.threshold:.3f})" for r in failed
        ]
        return "Model check failed: " + "; ".join(parts) + "."


def _resample(post: Posterior, draws: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(len(post.weights), size=draws, p=post.weights)


def _p_value(t_rep: np.ndarray, t_obs: np.ndarray, mid_p: bool) -> float:
    """Share of replicates at least as extreme; with ``mid_p`` ties count half, which removes
    most of the lumpiness of discrete statistics on small samples (Lancaster 1961)."""
    if mid_p:
        return float(np.mean(t_rep > t_obs) + 0.5 * np.mean(t_rep == t_obs))
    return float(np.mean(t_rep >= t_obs))


def shown_hands_pvalue(
    log: ActionLog,
    post: Posterior,
    rng: np.random.Generator,
    *,
    draws: int = 400,
    statistic: str = "chi2",
    mid_p: bool = True,
    matchups: Matchups | None = None,
) -> tuple[float, int]:
    """Posterior predictive p-value of the shown hands, and how many there were."""
    m = matchups or load_matchups()
    cond = opponent_class_probs(m)
    idx = _resample(post, draws, rng)
    bins = hand_bins(m)
    onehot = np.eye(N_BINS)[bins]
    t_obs, t_rep, total = np.zeros(draws), np.zeros(draws), 0
    for is_sb, strat in ((True, post.push), (False, post.call)):
        sel = (log.shown >= 0) & (log.is_sb == is_sb)
        if not sel.any():
            continue
        total += int(sel.sum())
        s = strat[idx]  # (R, 169)
        heroes, counts = np.unique(log.hero_class[sel], return_counts=True)
        if statistic == "chi2":
            # q[r, g, b]: P(bin b | hero class heroes[g], draw r), as one matrix product
            weights = (cond[heroes].T[:, :, None] * onehot[:, None, :]).reshape(169, -1)
            num = (s @ weights).reshape(draws, len(heroes), N_BINS)
            q = num / num.sum(axis=2, keepdims=True)
            expected = np.maximum((counts[None, :, None] * q).sum(axis=1), 1e-12)
            observed = np.bincount(bins[log.shown[sel]], minlength=N_BINS)
            rep = np.zeros((draws, N_BINS))
            for g, c in enumerate(counts):
                rep += rng.multinomial(c, q[:, g, :])
            t_obs += ((observed - expected) ** 2 / expected).sum(axis=1)
            t_rep += ((rep - expected) ** 2 / expected).sum(axis=1)
        elif statistic == "loglik":
            for h, c in zip(heroes, counts, strict=True):
                q = s * cond[h]
                q /= q.sum(axis=1, keepdims=True)
                logq = np.log(np.maximum(q, 1e-300))
                shown = log.shown[sel & (log.hero_class == h)]
                t_obs -= logq[:, shown].sum(axis=1)
                t_rep -= (rng.multinomial(c, q) * logq).sum(axis=1)
        else:
            raise ValueError(f"unknown statistic {statistic!r}")
    if total == 0:
        return 1.0, 0
    return _p_value(t_rep, t_obs, mid_p), total


def stability_pvalue(
    log: ActionLog,
    post: Posterior,
    rng: np.random.Generator,
    *,
    draws: int = 400,
    mid_p: bool = True,
    matchups: Matchups | None = None,
) -> tuple[float, int]:
    """Posterior predictive p-value of the CUSUM of action residuals, and the decisions used."""
    m = matchups or load_matchups()
    cond_t = opponent_class_probs(m).T
    idx = _resample(post, draws, rng)
    t_obs, t_rep = np.zeros(draws), np.zeros(draws)
    for is_sb, strat in ((True, post.push), (False, post.call)):
        sel = log.is_sb == is_sb
        if not sel.any():
            continue
        p = (strat[idx] @ cond_t)[:, log.hero_class[sel]]  # (R, T)
        scale = np.sqrt(np.maximum((p * (1 - p)).sum(axis=1), 1e-12))
        x = log.action[sel][None, :]
        x_rep = rng.random(p.shape) < p
        t_obs = np.maximum(t_obs, np.abs(np.cumsum(x - p, axis=1)).max(axis=1) / scale)
        t_rep = np.maximum(t_rep, np.abs(np.cumsum(x_rep - p, axis=1)).max(axis=1) / scale)
    n = len(log.action)
    if n == 0:
        return 1.0, 0
    return _p_value(t_rep, t_obs, mid_p), n


def check_pvalues(
    log: ActionLog,
    post: Posterior,
    rng: np.random.Generator,
    *,
    draws: int = 400,
    statistic: str = "chi2",
    mid_p: bool = True,
    matchups: Matchups | None = None,
) -> dict[str, tuple[float, int]]:
    return {
        "shown_hands": shown_hands_pvalue(
            log, post, rng, draws=draws, statistic=statistic, mid_p=mid_p, matchups=matchups
        ),
        "stability": stability_pvalue(log, post, rng, draws=draws, mid_p=mid_p, matchups=matchups),
    }


def calibration_pvalues(
    prior: PushFoldPrior,
    hero: tuple[np.ndarray, np.ndarray],
    sizes: Sequence[int],
    rng: np.random.Generator,
    *,
    draws: int = 400,
    posterior_draws: int = 2000,
    statistic: str = "chi2",
    matchups: Matchups | None = None,
) -> dict[int, dict[str, float]]:
    """Check p-values for one bot drawn from the prior and played by ``hero`` (the model is
    right by construction), after each number of hands in ``sizes``."""
    m = matchups or load_matchups()
    bot = strategy(prior.sample(rng), "equity", m)
    log = simulate(bot, hero, max(sizes), rng, m)
    out = {}
    for n in sizes:
        actions = ActionLog.from_hand_log(log, n)
        post = posterior(actions.observations(), prior, rng, draws=posterior_draws, matchups=m)
        pv = check_pvalues(actions, post, rng, draws=draws, statistic=statistic, matchups=m)
        out[int(n)] = {k: v[0] for k, v in pv.items()}
    return out


class FitChecker:
    """Posterior predictive checks with calibrated thresholds per number of hands observed.

    ``thresholds[n][check]`` applies from ``n`` hands up to the next calibrated size (below the
    smallest size, the smallest size's thresholds apply).
    """

    def __init__(
        self,
        thresholds: Mapping[int, Mapping[str, float]],
        *,
        draws: int = 400,
        statistic: str = "chi2",
        seed: int | Sequence[int] = 0,
        matchups: Matchups | None = None,
    ) -> None:
        if not thresholds:
            raise ValueError("thresholds must cover at least one sample size")
        self.thresholds = {int(n): dict(t) for n, t in sorted(thresholds.items())}
        self.draws = draws
        self.statistic = statistic
        self.seed = seed
        self.matchups = matchups or load_matchups()

    @classmethod
    def from_pvalues(
        cls, pvalues: Sequence[Mapping[int, Mapping[str, float]]], *, level: float = 0.05, **kw
    ) -> FitChecker:
        """Thresholds at the ``level / len(CHECKS)`` quantile of calibration p-values (one
        mapping per calibration bot, from ``calibration_pvalues``)."""
        sizes = sorted(pvalues[0])
        q = level / len(CHECKS)
        thresholds = {
            n: {c: float(np.quantile([p[n][c] for p in pvalues], q)) for c in CHECKS} for n in sizes
        }
        return cls(thresholds, **kw)

    @classmethod
    def calibrate(
        cls,
        prior: PushFoldPrior,
        hero: tuple[np.ndarray, np.ndarray],
        sizes: Sequence[int],
        *,
        bots: int = 200,
        level: float = 0.05,
        seed: int = 0,
        **kw,
    ) -> FitChecker:
        """Simulate ``bots`` bots from the prior and calibrate on their p-values (serial; an
        experiment can call ``calibration_pvalues`` in parallel and use ``from_pvalues``)."""
        rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(bots)]
        draws = kw.get("draws", 400)
        statistic = kw.get("statistic", "chi2")
        pv = [
            calibration_pvalues(prior, hero, sizes, r, draws=draws, statistic=statistic)
            for r in rngs
        ]
        return cls.from_pvalues(pv, level=level, seed=seed, **kw)

    def thresholds_for(self, n_hands: int) -> dict[str, float]:
        sizes = [n for n in self.thresholds if n <= n_hands]
        return self.thresholds[sizes[-1] if sizes else min(self.thresholds)]

    def check(
        self, log: ActionLog, post: Posterior, *, seed: int | Sequence[int] | None = None
    ) -> FitCheck:
        rng = np.random.default_rng(np.random.SeedSequence(self.seed if seed is None else seed))
        th = self.thresholds_for(log.n_hands)
        pv = check_pvalues(
            log, post, rng, draws=self.draws, statistic=self.statistic, matchups=self.matchups
        )
        return FitCheck(tuple(CheckResult(c, pv[c][0], th.get(c), pv[c][1]) for c in CHECKS))

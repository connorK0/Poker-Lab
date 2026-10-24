"""Push/fold opponents: parametric bots, what a player sees of them, two ways to model them (P3.2).

**Bots.** A bot plays each hand class ``i`` with a logistic threshold on its strength:

    P(push i) = sigmoid((push_width - pct_i) / temperature)
    P(call j) = sigmoid((call_width - pct_j) / temperature)

``pct_i`` is the class's strength percentile under the bot's *hand order*: the share of the 1,326
combos ranked above it plus half its own. A width of 0.4 means "about the top 40%"; the
temperature softens the edge of the range. Archetypes are distributions over the widths (Beta,
concentration 30, as in Phases 1 and 2) and the temperature (log-normal).

Two hand orders: ``"equity"`` (equity against a random hand, the order both models assume) and
``"pairs_first"`` (every pair above every non-pair, then high card, kicker, suited before
offsuit: a plausible human heuristic, used to make the models misspecified on purpose).

**Observation (the firewall in miniature).** The hero plays the bot heads-up for ``n`` hands,
alternating seats, and records only what a player sees: its own class, the bot's actions, and
the bot's class *only when a push is called* (a showdown). Folds never reveal cards. The
simulation deals class pairs with the game's own probabilities, so it is consistent with the
exact evaluation in ``push_fold``.

**Models.**

* ``fit_likelihood``: posterior mode of the three parameters given the observations. A hand
  whose cards were shown contributes ``log P(action | class)``; a hand whose cards were not
  contributes ``log sum_i P(i | hero's class) P(action | i)``, card removal included. Priors:
  Beta on the widths (mode taken in logit space, as in Phase 2) and normal on log temperature,
  fitted to a separate training population.
* ``fit_frequency``: the "HUD stats" model. It uses only how often the bot pushed (as small
  blind) and called (when facing a push), shrunk toward the population with a pseudo-count, and
  finds the widths that reproduce those frequencies at the population's temperature. It never
  looks at shown cards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import astuple, dataclass
from functools import cache

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit

from poker_lab.games.preflop import CLASSES, N_COMBOS, Matchups, combo_counts, load_matchups

PARAMS = ("push_width", "call_width", "temperature")
ORDERS = ("equity", "pairs_first")
RANKS = "23456789TJQKA"
T_BOUNDS = (0.003, 0.5)


@dataclass(frozen=True)
class PushFoldBotParams:
    push_width: float
    call_width: float
    temperature: float

    def as_array(self) -> np.ndarray:
        return np.array(astuple(self), dtype=float)


def _order(order: str, m: Matchups) -> np.ndarray:
    """Class indices from strongest to weakest under ``order``."""
    if order == "equity":
        return np.argsort(-m.equity_vs_random(), kind="stable")
    if order == "pairs_first":

        def key(cls: str) -> tuple:
            pair = cls[0] == cls[1]
            suited = len(cls) == 3 and cls[2] == "s"
            return (pair, RANKS.index(cls[0]), RANKS.index(cls[1]), suited)

        return np.array(sorted(range(len(CLASSES)), key=lambda k: key(CLASSES[k]), reverse=True))
    raise ValueError(f"unknown hand order {order!r}")


def percentiles(order: str = "equity", matchups: Matchups | None = None) -> np.ndarray:
    """Per class (``CLASSES`` order): share of combos ranked above it plus half its own."""
    return _percentiles(order, matchups or load_matchups())


@cache
def _percentiles(order: str, m: Matchups) -> np.ndarray:
    combos = combo_counts()
    pct = np.empty(len(CLASSES))
    above = 0
    for k in _order(order, m):
        pct[k] = (above + combos[k] / 2) / N_COMBOS
        above += combos[k]
    pct.setflags(write=False)
    return pct


def strategy(
    params: PushFoldBotParams, order: str = "equity", matchups: Matchups | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class push and call probabilities."""
    pct = percentiles(order, matchups)
    t = params.temperature
    return expit((params.push_width - pct) / t), expit((params.call_width - pct) / t)


def range_shares(params: PushFoldBotParams, order: str = "equity") -> tuple[float, float]:
    """Share of combos pushed and called (before card removal)."""
    push, call = strategy(params, order)
    w = combo_counts() / N_COMBOS
    return float(push @ w), float(call @ w)


# ------------------------------------------------------------------------------- archetypes

ARCHETYPES: dict[str, dict[str, float]] = {
    # Mean widths; Nash at 10 bb pushes about 58% and calls about 37% of hands.
    "nit": {"push_width": 0.20, "call_width": 0.10},
    "tag": {"push_width": 0.40, "call_width": 0.25},
    "lag": {"push_width": 0.65, "call_width": 0.40},
    "calling_station": {"push_width": 0.30, "call_width": 0.70},
    "maniac": {"push_width": 0.90, "call_width": 0.55},
}
CONCENTRATION = 30.0
TEMPERATURE_MEDIAN = 0.03
TEMPERATURE_SIGMA = 0.5


def sample_params(archetype: str, rng: np.random.Generator) -> PushFoldBotParams:
    mean = ARCHETYPES[archetype]
    widths = [
        float(rng.beta(mean[k] * CONCENTRATION, (1 - mean[k]) * CONCENTRATION))
        for k in ("push_width", "call_width")
    ]
    t = float(np.exp(rng.normal(np.log(TEMPERATURE_MEDIAN), TEMPERATURE_SIGMA)))
    return PushFoldBotParams(*widths, float(np.clip(t, *T_BOUNDS)))


# ------------------------------------------------------------------------------ observation


@dataclass(frozen=True, eq=False)
class Observations:
    """Sufficient statistics of what the hero saw over ``n_hands`` (see module docstring).

    ``sb_*``: hands where the bot was the small blind; ``bb_*``: the bot was the big blind and
    faced a push. ``*_shown[i]``: the bot acted with class ``i`` shown (a called push as small
    blind, a call as big blind). ``*_hidden[a, h]``: the bot's action ``a`` (0 fold, 1 push or
    call) with its class unseen, by the hero's class ``h``.
    """

    n_hands: int
    sb_shown: np.ndarray  # (169,)
    sb_hidden: np.ndarray  # (2, 169)
    bb_shown: np.ndarray  # (169,)
    bb_hidden: np.ndarray  # (2, 169)

    @property
    def n_sb(self) -> int:
        return int(self.sb_shown.sum() + self.sb_hidden.sum())

    @property
    def n_bb(self) -> int:
        return int(self.bb_shown.sum() + self.bb_hidden.sum())

    @property
    def pushes(self) -> int:
        return int(self.sb_shown.sum() + self.sb_hidden[1].sum())

    @property
    def calls(self) -> int:
        return int(self.bb_shown.sum() + self.bb_hidden[1].sum())


@dataclass(frozen=True, eq=False)
class HandLog:
    """Per-hand record of a simulated match (prefixes give the observations after n hands)."""

    bot_is_sb: np.ndarray  # bool
    hero_class: np.ndarray  # int
    bot_class: np.ndarray  # int (never read by the models unless shown)
    acted: np.ndarray  # bool: the bot made a decision (not when the hero folded as small blind)
    action: np.ndarray  # int: 1 push or call, 0 fold
    shown: np.ndarray  # bool: the bot's class was revealed at showdown

    def observations(self, n: int) -> Observations:
        out = {"sb_shown": np.zeros(169), "sb_hidden": np.zeros((2, 169))}
        out |= {"bb_shown": np.zeros(169), "bb_hidden": np.zeros((2, 169))}
        for seat, is_sb in (("sb", True), ("bb", False)):
            sel = (self.bot_is_sb[:n] == is_sb) & self.acted[:n]
            shown = sel & self.shown[:n]
            hidden = sel & ~self.shown[:n]
            np.add.at(out[f"{seat}_shown"], self.bot_class[:n][shown], 1)
            np.add.at(
                out[f"{seat}_hidden"], (self.action[:n][hidden], self.hero_class[:n][hidden]), 1
            )
        return Observations(n, **out)


def simulate(
    bot: tuple[np.ndarray, np.ndarray],
    hero: tuple[np.ndarray, np.ndarray],
    n_hands: int,
    rng: np.random.Generator,
    matchups: Matchups | None = None,
) -> HandLog:
    """Play ``n_hands`` of push/fold, the bot in the small blind on even hands. ``bot`` and
    ``hero`` are (push, call) probabilities per class."""
    m = matchups or load_matchups()
    p = m.deal_prob.ravel()
    deals = rng.choice(p.size, size=n_hands, p=p / p.sum())
    sb_cls, bb_cls = np.divmod(deals, len(CLASSES))
    bot_is_sb = np.arange(n_hands) % 2 == 0
    bot_cls = np.where(bot_is_sb, sb_cls, bb_cls)
    hero_cls = np.where(bot_is_sb, bb_cls, sb_cls)
    u1, u2 = rng.random(n_hands), rng.random(n_hands)
    sb_push = np.where(bot_is_sb, u1 < bot[0][sb_cls], u1 < hero[0][sb_cls])
    bb_call = np.where(bot_is_sb, u2 < hero[1][bb_cls], u2 < bot[1][bb_cls])
    acted = bot_is_sb | sb_push  # as big blind the bot only acts facing a push
    action = np.where(bot_is_sb, sb_push, bb_call).astype(int)
    shown = sb_push & bb_call  # showdown
    return HandLog(bot_is_sb, hero_cls, bot_cls, acted, action, shown)


# ---------------------------------------------------------------------------------- fitting


@dataclass(frozen=True)
class PushFoldPrior:
    """Beta priors on the widths and a normal prior on log temperature."""

    push: tuple[float, float]
    call: tuple[float, float]
    log_t: tuple[float, float]  # mean, sd

    @property
    def mean_params(self) -> PushFoldBotParams:
        (ap, bp), (ac, bc) = self.push, self.call
        return PushFoldBotParams(ap / (ap + bp), ac / (ac + bc), float(np.exp(self.log_t[0])))

    @classmethod
    def fit(cls, population: Sequence[PushFoldBotParams]) -> PushFoldPrior:
        """Method of moments on a training population's true parameters."""
        arr = np.array([p.as_array() for p in population])

        def beta(x: np.ndarray) -> tuple[float, float]:
            mu, var = float(x.mean()), float(x.var(ddof=1))
            k = mu * (1 - mu) / var - 1
            return mu * k, (1 - mu) * k

        lt = np.log(arr[:, 2])
        return cls(beta(arr[:, 0]), beta(arr[:, 1]), (float(lt.mean()), float(lt.std(ddof=1))))


def _conditional(matchups: Matchups) -> np.ndarray:
    """P(opponent's class | own class): rows sum to 1 (card removal)."""
    c = matchups.counts.astype(float)
    return c / c.sum(axis=1, keepdims=True)


def log_likelihood(
    params: PushFoldBotParams, obs: Observations, matchups: Matchups | None = None
) -> float:
    m = matchups or load_matchups()
    cond = _conditional(m)
    push, call = strategy(params, "equity", m)
    eps = 1e-12
    ll = float(obs.sb_shown @ np.log(push + eps))
    ll += float(obs.sb_hidden[1] @ np.log(cond @ push + eps))
    ll += float(obs.sb_hidden[0] @ np.log(cond @ (1 - push) + eps))
    ll += float(obs.bb_shown @ np.log(call + eps))
    ll += float(obs.bb_hidden[1] @ np.log(cond @ call + eps))
    ll += float(obs.bb_hidden[0] @ np.log(cond @ (1 - call) + eps))
    return ll


def _unpack(z: np.ndarray) -> PushFoldBotParams:
    return PushFoldBotParams(float(expit(z[0])), float(expit(z[1])), float(np.exp(z[2])))


def fit_likelihood(
    obs: Observations, prior: PushFoldPrior | None, matchups: Matchups | None = None
) -> PushFoldBotParams:
    """Posterior mode in (logit width, logit width, log temperature); ``prior=None`` gives the
    maximum-likelihood estimate (from the prior-free start at the population defaults)."""
    m = matchups or load_matchups()
    start = prior.mean_params if prior else PushFoldBotParams(0.5, 0.3, TEMPERATURE_MEDIAN)
    z0 = np.array([logit(start.push_width), logit(start.call_width), np.log(start.temperature)])

    def objective(z: np.ndarray) -> float:
        p = _unpack(z)
        val = log_likelihood(p, obs, m)
        if prior is not None:
            # Beta(a, b) on w with the logit Jacobian w (1 - w): w^a (1 - w)^b.
            for w, (a, b) in ((p.push_width, prior.push), (p.call_width, prior.call)):
                val += a * np.log(w) + b * np.log1p(-w)
            mu, sd = prior.log_t
            val += -0.5 * ((z[2] - mu) / sd) ** 2
        return -val

    bounds = [(-9, 9), (-9, 9), tuple(np.log(T_BOUNDS))]
    res = minimize(objective, z0, method="L-BFGS-B", bounds=bounds)
    return _unpack(res.x)


def fit_frequency(
    obs: Observations, prior: PushFoldPrior, pseudo: float = 10.0
) -> PushFoldBotParams:
    """Widths that reproduce the bot's shrunk push and call frequencies (shown cards unused)."""
    pop = prior.mean_params
    t = pop.temperature
    pop_push, pop_call = range_shares(pop)
    push_rate = (obs.pushes + pseudo * pop_push) / (obs.n_sb + pseudo)
    call_rate = (obs.calls + pseudo * pop_call) / (obs.n_bb + pseudo)

    def width_for(rate: float, which: int) -> float:
        def gap(w: float) -> float:
            params = PushFoldBotParams(w, w, t)
            return range_shares(params)[which] - rate

        lo, hi = 1e-4, 1 - 1e-4
        if gap(lo) >= 0:
            return lo
        if gap(hi) <= 0:
            return hi
        return float(brentq(gap, lo, hi, xtol=1e-6))

    return PushFoldBotParams(width_for(push_rate, 0), width_for(call_rate, 1), t)

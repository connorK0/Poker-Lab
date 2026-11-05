"""P2.11c: how six copies' shortfall from perfect cancellation divides between the bots' randomness.

    python -m experiments.phase2_randomness --seed 2026 [--quick]
    python -m experiments.run_phase2 --seed 2026 --only randomness

Pre-registered plan (written after development on seeds 98 and 99, before the real run):

**Question.** P2.11b dealt each deal at six tables with the seats rotated. Six copies of one bot
reached a median ratio of 1.53x: by median rho (-0.069 against the -1/5 at which the ratio is
unbounded) about 35% of the way to perfect cancellation, and by this plan's metric, the mean of
1 / ratio, 33.8% (nits lowest, at 25.5%). Copies can differ only through the bots' own randomness.
How does the shortfall divide between its three sources: mixed actions, bet-size noise and Monte
Carlo equity? ``experiments/phase2_duplicate.py`` and ``poker_lab/bots`` are not edited: the
switches live in ``SwitchedBot`` here, so the simulation fingerprint and the committed results do
not move, and the baseline is the default ``ParametricBot`` itself.

**What the shortfall is, for copies.** Let Y be a seat's result in a hand. Given the cards, the n
tables are independent draws of one experiment: the same bots, each table with its own decision
RNG. A player's per-deal mean u holds each seat once, so E[u | cards] is the mean of the seats'
expected results, which is 0 because a hand is zero-sum without rake. So

    Var(u) = E[Var(u | cards)] = (1 / n) * mean over seats of E[Var(Y | cards)],

and with sigma^2 = Var(Y) the quantity 1 / ratio = n s_u^2 / sigma^2 estimates

    R = E[Var(Y | cards)] / Var(Y),

the share of a hand's variance that comes from the bots' draws and not from the cards (and the
seat, which the cards' position fixes). The rotation cancels all of the rest. So the copies' 35%
says how much of a hand's variance the cards explain between these bots, and is not a limit of the
rotation. The direct estimate is the sample variance of a seat's results across the tables of a
deal, averaged over seats and deals (``within_deal_variance``); checking it against n s_u^2 tests
the argument (A3).

**Switching a source off without changing the bots.** The first idea is to replace a draw by a
constant, but that changes the bots: with every mixed action decided by u = 0.5 a bot always takes
the same branch at a given set of probabilities, so it may never 3-bet and may always c-bet. So the
main design *shares* a source's draws between the tables instead. A seat's draw of that source is
a hash of the chunk's seed, the deal, the seat, the street, the source and a pairing key; the table
number does not enter (``SwitchedBot._digest``). Hash outputs behave as independent uniforms that do
not depend on the situation, so each table on its own plays hands with the default's distribution:
this rests on the construction, and B1 to B4 are coarse checks of it. Only the correlation between
tables changes. With a source shared the bots are the same bots.

**The pairing is a choice.** Sharing needs a rule for which decisions at two tables use one draw,
once the tables have parted. Two natural rules:

* ``decision``: the seat's k-th decision on the street (equity: the street). Tables keep the same
  draws after they part.
* ``history``: the whole public history so far, amounts included (``history_*`` conditions): tables
  use one draw only while nothing public has differed. Any difference in a bet's size then makes all
  later draws of a shared source fresh, at every seat.

Each table's distribution is the default's under both, so both are legitimate, and neither is shown
to be a bound. Both rules give the same two ends (nothing shared: the baseline; everything shared:
exact cancellation), and they differ in between. So the shortfall itself does not depend on the
rule, and its split does: the split below is a split under a pairing, and is reported under both.

**The split.** Under one pairing, Y is a function of independent inputs: the cards C and the
fields of draws M, Z and E (mixed, noise, equity). The Sobol-Hoeffding decomposition writes Var(Y)
as a sum of non-negative terms V_A, one per subset A of the inputs. Grouping terms by the sources in
A (with or without C), W_B = sum of V_A over the A whose sources are exactly B, the condition with
live set S (the rest shared) measures

    R(S) = E[Var(Y | C, the shared sources)] / Var(Y) = sum of W_B over B meeting S,

so R(S) is the group's total-effect Sobol index for this f, R(nothing live) = 0 and R(all live) is
the shortfall. Eight conditions give the seven W_B by Moebius inversion (``decompose``); they sum
to R(all), with no order of removal assumed. W_mixed = R(all) - R(all but mixed) is the part that
sharing mixed actions alone removes, under this pairing; W_{mixed,noise} the part that needs both
shared. Each source's single number, "interactions split equally", is its W plus equal parts of the
interactions it is in (its Shapley value for the game R). Sample terms can come out below zero by
sampling error.

**Conditions.** Every condition plays the same 50 lineups of P2.11b's ``clones_6`` with the same
bots and chunk seeds, so the same cards; only the draws differ:

* ``baseline``: all three live: the default ``ParametricBot``, which must reproduce the committed
  ratios exactly at seed 2026 (A1).
* ``shared_mixed``, ``shared_noise``, ``shared_equity``: one source shared, the others live.
* ``only_mixed``, ``only_noise``, ``only_equity``: one source live, the others shared.
* ``shared_all``: all shared. Exact cancellation, the library's argument: only a check (A2).
* ``history_shared_*`` and ``history_only_*``: the six partial conditions under the history key.
* ``off_mixed`` (every mixed action decided by u = 0.5), ``off_noise`` (size factor 1) and
  ``off_all`` (both, with equity shared; only a check): the sources switched off outright, as first
  proposed. ``off_mixed`` changes the bots, so its ratio is not part of the split: it is reported
  beside it to show what the plain reading would have said. There is no ``off_equity``: exact equity
  needs about a million evaluations per flop decision, and an estimate that is a function of the
  cards alone is what ``shared_equity`` already is.

**Metric and aggregates.** A lineup's remaining share is the mean over its six players of
n s_u^2 / sigma^2 (the mean of 1 / ratio; means of it add, ratios do not), each condition with its
own sigma^2. The shared conditions estimate one Var(Y), so the review of this plan proposed one
pooled denominator per lineup. Development rejected that: a rare big pot enters a condition's
numerator and denominator together and mostly cancels, and pooling breaks this (nits' cancelled
share on seed 99 went from 26.8% to 0.8%, and only_mixed over the baseline from 0.97 to 0.89). Means
over lineups, 95% percentile bootstrap CIs resampling lineups with one set of resamples for all
conditions of a split, overall and per archetype (10 lineups each); the reading of the CIs is
P2.11b's. Per condition also the committed statistic (median ratio over players and its CI), rho,
the per-hand SD, the lineup-by-lineup log change in SD, and the mean change in VPIP, PFR, 3-bet,
c-bet, postflop aggression share, fold to a bet and went-to-showdown, pooled over each lineup's
hands. A stat with no opportunities in a lineup is left out of its mean (under ``off_mixed`` the
calling station has no c-bet or fold-to-bet opportunities); ``behaviour_change_lineups`` counts the
lineups used.

**Seen in development** (not predictions; seeds 98 | 99, 2 replicates, 10 lineups, 1,500 deals, 8
workers, 1,067 | 990 s; only 1.9 GB of memory free at the start):

* Cancelled: 36.9% [32.7, 41.0] | 33.7% [30.7, 36.0]; default median ratio 1.60x | 1.54x.
* Median ratio, decision key: mixed shared 7.13x | 6.68x; noise shared 1.53x | 1.54x; equity shared
  1.64x | 1.58x; only noise live 28x | 22x; only equity live 8.8x | 8.1x; only mixed live 1.61x |
  1.57x. History key: mixed shared 1.82x | 1.70x; noise shared 1.57x | 1.54x; equity shared 1.53x |
  1.54x; only noise live 1.89x | 1.82x; only equity live 7.95x | 7.09x; only mixed live 1.61x |
  1.58x.
* Remaining share: only mixed live 63.6% | 64.4% (decision), 61.3% | 62.8% (history), of a default's
  63.1% | 66.3%. Only equity live 14.1% | 15.4%, 14.5% | 14.4%. Only noise live 4.2% | 6.5%, against
  52.8% | 56.3%: with the history key a size difference alone re-draws every later mixed action.
* Split of the shortfall, interactions shared equally. Decision key: mixed 87.4% | 83.4%; equity
  10.7% | 12.2%; noise 1.9% | 4.4%. History key: mixed 53.3% | 51.4%; noise 38.6% | 40.5%; equity
  8.1% | 8.1%.
* Terms of a hand's variance. Decision key: mixed alone 47.7% | 48.3%; mixed x equity 10.9% | 8.6%;
  all three 2.8% | 6.2%; equity alone 0.3% | 2.8%; others between -2.3% and +2.3%. History key:
  mixed x noise 40.9% | 42.2%; mixed alone 8.9% | 8.8%; all three 8.8% | 10.4%; others between -1.3%
  and +4.3%.
* By archetype (decision key): equity's split share lowest for maniacs (5.1% | 5.4%; others at least
  9%). All five archetypes' history-key splits are near the overall one.
* The bots with a source shared, 13 conditions: median log SD change at most 0.011 | 0.035, stats at
  most 0.024 | 0.027 and of no steady sign. By archetype (2 lineups each here, 10 in the full run):
  median SD factor 0.96-1.35 | 0.86-1.05, stats at most 0.12 | 0.13. Direct against n s_u^2: largest
  |log| 0.030 | 0.050. Both exact checks at 0 chips.
* Mixed actions off changes the bots: median ratio 6.86x | 9.12x; the calling station's SD falls
  about 20-fold (no postflop betting at all, always to showdown once in), nits' SD about doubles,
  c-bet rises 0.14 | 0.23 and fold to a bet falls 0.08 | 0.10. Noise off barely changes them: median
  log SD change at most 0.004, stats within 0.014, median ratio 1.61x | 1.57x.

**Full run.** 10 replicates, 5 lineups each, 2,000 deals: P2.11b's clones_6, so 600,000 hands per
condition and 10.2 million in all (17 conditions).

**Predictions** (thresholds from development; evaluated by ``PREDICTIONS``). The overall cancelled
share and nits being lowest are not predicted: A1 ties them to P2.11b's committed values, above.

* A1: the baseline repeats P2.11b's committed clones_6: every player's ratio equal.
* A2: with all three shared and with all three off, every per-deal sum is exactly 0 chips.
* A3: direct within-deal variance over n s_u^2: largest |log| at most 0.08 (all but the checks).
* B1: sharing a source, either key: median log SD change within 0.05 in each condition.
* B2: sharing a source, either key: each stat's mean within 0.03 of the default.
* B3: by archetype: median SD factor within [0.8, 1.4] in every sharing condition.
* B4: by archetype: each stat's mean within 0.15 of the default in every sharing condition.
* C1: only mixed live leaves 90-110% of the default's remaining share, under both keys.
* C2: only equity live leaves 10-20% of a hand's variance, under both keys.
* C3: sharing noise alone or equity alone: median ratio in [1.35, 1.9], under both keys.
* D1: decision key, mixed shared alone: median ratio in [4.5, 12].
* D2: decision key, split of the shortfall: mixed at least 75%.
* D3: decision key, split: equity 5-20%.
* D4: decision key, split: noise at most 8%.
* D5: decision key: mixed x equity at least 4% of a hand's variance, and above equity alone.
* D6: decision key: maniacs have the lowest equity split share of the five archetypes.
* H1: only noise live: at most 10% of a hand's variance with the decision key, at least 40% with
  the history key.
* H2: history key, mixed shared alone: median ratio at most 2.2.
* H3: history key, split: mixed 40-65%.
* H4: history key, split: noise 30-50%.
* H5: history key, split: equity at most 15%.
* H6: history key: mixed x noise at least 25% of a hand's variance.
* H7: mixed's split share is above noise's, under both keys.
* E1: mixed actions off changes the bots: the calling station's median SD falls below 0.1x and nits'
  rises above 1.5x.
* E2: mixed actions off: mean c-bet up at least 0.05 and fold to a bet down at least 0.04 (over the
  lineups with opportunities).
* E3: noise off leaves the bots: median log SD change within 0.03, each stat within 0.03, median
  ratio in [1.35, 1.9].

**What this cannot show.** The identity is for copies: different bots' play differs for reasons
other than draws, so nothing here divides the shortfall of mirror_6 or mixed_6. The split is of one
pairing's f, and it also depends on how the policy lays out its branches on [0, 1]: other pairings
would give other splits, and these two are not shown to bound them. So what can be read without
that qualification is the ends and the one-source-live levels that come out alike under both keys. A
split among these three sources of these bots says nothing about people. Sharing draws is possible
only in simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from functools import partial
from itertools import chain, combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from experiments import phase2_duplicate as pd
from experiments.common import RESULTS_ROOT, derive_seeds, load_config, run_metadata, save_json
from poker_lab.bots.archetype_bot import ParametricBot
from poker_lab.bots.archetypes import ARCHETYPES
from poker_lab.bots.params import BotParams
from poker_lab.bots.population import PlayerSpec
from poker_lab.engine.agent import Decision
from poker_lab.schema import SeatView

CONFIG = pd.CONFIG
COMMITTED = RESULTS_ROOT / "phase2" / "duplicate" / "duplicate.json"
N = 6
SOURCES = ("mixed", "noise", "equity")
SOURCE_LABELS = {
    "mixed": "Mixed actions",
    "noise": "Bet-size noise",
    "equity": "Monte Carlo equity",
}
LIVE, SHARED, OFF = "live", "shared", "off"
TAG_BOOTSTRAP = 6  # after phase2_duplicate's tags, so the streams stay apart
_NORMAL = NormalDist()


@dataclass(frozen=True)
class Modes:
    """How a bot draws each source. ``live``: from the table's RNG, as ``ParametricBot`` does.
    ``shared``: from a hash of the deal, the seat, the source and ``key``, so a seat's draws are
    the same at every table of a deal. ``off``: a constant (mixed actions and noise only).
    ``key``: ``decision`` pairs the tables by the seat's decision number on the street,
    ``history`` only when the whole public history so far is the same."""

    mixed: str = LIVE
    noise: str = LIVE
    equity: str = LIVE
    key: str = "decision"


def _shared_modes(live: Sequence[str], key: str = "decision") -> Modes:
    return Modes(**{s: LIVE if s in live else SHARED for s in SOURCES}, key=key)


# ``live`` set of each condition of the decomposition; the rest are shared.
DECOMPOSITION: dict[str, tuple[str, ...]] = {
    "baseline": ("mixed", "noise", "equity"),
    "shared_mixed": ("noise", "equity"),
    "shared_noise": ("mixed", "equity"),
    "shared_equity": ("mixed", "noise"),
    "only_mixed": ("mixed",),
    "only_noise": ("noise",),
    "only_equity": ("equity",),
    "shared_all": (),
}
# The same split with the tables paired by the exact public history. The two ends need no
# rerun: nothing is shared in the baseline, and with everything shared histories never differ.
HISTORY_PREFIX = "history_"
HISTORY = {
    HISTORY_PREFIX + name: live
    for name, live in DECOMPOSITION.items()
    if name not in ("baseline", "shared_all")
}
CONDITIONS: dict[str, Modes] = {
    **{name: _shared_modes(live) for name, live in DECOMPOSITION.items()},
    **{name: _shared_modes(live, "history") for name, live in HISTORY.items()},
    # Switched off outright: the bots play differently (see the docstring). An equity estimate
    # that is a function of the cards alone is what ``shared`` already gives, so equity has no
    # separate ``off`` run.
    "off_mixed": Modes(mixed=OFF),
    "off_noise": Modes(noise=OFF),
    "off_all": Modes(mixed=OFF, noise=OFF, equity=SHARED),
}
EXACT_CHECKS = ("shared_all", "off_all")
LABELS = {
    "baseline": "Default: all three live",
    "shared_mixed": "Mixed actions shared",
    "shared_noise": "Bet-size noise shared",
    "shared_equity": "Equity shared",
    "only_mixed": "Only mixed actions live",
    "only_noise": "Only bet-size noise live",
    "only_equity": "Only equity live",
    "shared_all": "All three shared (check)",
    "history_shared_mixed": "Mixed actions shared, history key",
    "history_shared_noise": "Bet-size noise shared, history key",
    "history_shared_equity": "Equity shared, history key",
    "history_only_mixed": "Only mixed actions live, history key",
    "history_only_noise": "Only bet-size noise live, history key",
    "history_only_equity": "Only equity live, history key",
    "off_mixed": "Mixed actions off (u = 0.5)",
    "off_noise": "Bet-size noise off (factor 1)",
    "off_all": "All three off (check)",
}
BEHAVIOUR = ("vpip", "pfr", "three_bet", "cbet", "aggr_share", "fold_to_bet", "wtsd")


class SwitchedBot(ParametricBot):
    """``ParametricBot`` with each source of randomness live, shared between tables or off. The
    policy is the parent's: only where the draws come from changes. With every source live it
    plays exactly as the parent does, draw for draw."""

    def __init__(
        self, params: BotParams, *, modes: Modes, salt: int, equity_samples: int = 150
    ) -> None:
        super().__init__(params, equity_samples=equity_samples)
        self.modes = modes
        self.salt = salt
        self._rng: random.Random | None = None
        self._view: SeatView | None = None

    def _digest(self, view: SeatView, source: str, *, per_decision: bool = True) -> int:
        """A 64-bit hash of the deal, seat, street, source and the modes' key. The deal number is
        the hand id's counter, the same at every table of a deal (a table's prefix differs). The
        decision is the seat's k-th on the street, which needs no memory in the agent."""
        pub = view.public
        if self.modes.key == "history":
            pair = ";".join(
                f"{a.seat},{a.street.name},{a.action_type.name},{a.amount}" for a in pub.actions
            )
        elif per_decision:
            pair = str(
                sum(1 for a in pub.actions if a.seat == view.seat and a.street is pub.street)
            )
        else:
            pair = ""
        where = f"{pub.hand_id.rsplit('-', 1)[1]}|{view.seat}|{pub.street.name}"
        text = f"{self.salt}|{where}|{pair}|{source}"
        return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")

    def _uniform(self, view: SeatView, source: str) -> float:
        return (self._digest(view, source) + 0.5) / 2**64

    def act(self, view: SeatView, rng: random.Random) -> Decision:
        self._rng, self._view = rng, view
        if self.modes.mixed == LIVE:
            return super().act(view, rng)
        u = 0.5 if self.modes.mixed == OFF else self._uniform(view, "mixed")
        # The policy makes at most one uniform draw per decision, so one value serves.
        return super().act(view, _FixedUniform(u))

    def _noise(self, rng: random.Random) -> float:  # type: ignore[override]
        if self.modes.noise == OFF:
            return 1.0
        if self.modes.noise == SHARED:
            return math.exp(0.12 * _NORMAL.inv_cdf(self._uniform(self._view, "noise")))
        return ParametricBot._noise(self._rng)

    def _equity_now(self, view: SeatView, rng: random.Random) -> float:
        if self.modes.equity == LIVE:
            return super()._equity_now(view, self._rng)
        # The parent's cached estimate, its samples drawn from a generator of the deal, seat and
        # street: the same hole cards and board give the same estimate at every table.
        seed = self._digest(view, "equity", per_decision=False)
        return super()._equity_now(view, random.Random(seed))


class _FixedUniform:
    """Stands in for the table RNG inside the policy: ``random`` is all the policy calls on it,
    since ``_noise`` and ``_equity_now`` are overridden. Anything else raises."""

    __slots__ = ("u",)

    def __init__(self, u: float) -> None:
        self.u = u

    def random(self) -> float:
        return self.u


# ---- tasks --------------------------------------------------------------------------------


def chunk_task(
    seats: tuple[tuple[str, PlayerSpec], ...],
    n_deals: int,
    seed: int,
    equity_samples: int,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    """One duplicate match: net chips, shape (n tables, n_deals, n players), players in lineup
    order, and the behaviour counters over all players and tables. The baseline is the default
    ``ParametricBot`` itself, as in ``phase2_duplicate.chunk_task``; the salt is the chunk's seed,
    so the conditions share their draws wherever they share a mode."""
    from poker_lab.engine.duplicate import duplicate_match
    from poker_lab.profiling.stats import N_COUNTERS, hand_counters
    from poker_lab.schema import TableConfig

    n = len(seats)
    table = TableConfig(f"dup{n}", n, pd.SMALL_BLIND, pd.BIG_BLIND)

    def factory(spec: PlayerSpec):
        if condition == "baseline":
            return partial(spec.make_agent, equity_samples=equity_samples)
        return partial(
            SwitchedBot,
            spec.params,
            modes=CONDITIONS[condition],
            salt=seed,
            equity_samples=equity_samples,
        )

    players = [(pid, factory(spec)) for pid, spec in seats]
    match = duplicate_match(
        table, players, n_deals, seed, starting_stack=pd.STACK_BB * pd.BIG_BLIND
    )
    chips = np.empty((match.n_tables, n_deals, n), dtype=np.int32)
    for j, (pid, _) in enumerate(seats):
        grouped = np.array([r.net_chips for r in match.grouped_results(pid)], dtype=np.int32)
        chips[:, :, j] = grouped.reshape(n_deals, match.n_tables).T
    counters = np.zeros(N_COUNTERS, dtype=np.int64)
    for hands in match.tables:
        for hand in hands:
            for _, row in hand_counters(hand):
                counters += row
    return chips, counters


def _task(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    return chunk_task(*args)


def tasks(lineups: Sequence[pd.Lineup], cfg: dict[str, Any], conditions: Sequence[str]) -> list:
    return [
        (line.seats, deals, seed, cfg["equity_samples"], condition)
        for condition in conditions
        for line in lineups
        for seed, deals in line.chunks
    ]


def _results(task_list: list[tuple], workers: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if workers <= 1:
        yield from map(_task, task_list)
        return
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        yield from pool.map(_task, task_list)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def plan(master_seed: int, cfg: dict[str, Any]) -> list[pd.Lineup]:
    """The same lineups, bots and chunk seeds as P2.11b's ``clones_6``: its seeds do not depend on
    the other lineup kinds."""
    cfg = {**cfg, "lineups": {"clones_6": cfg["lineups_per_replicate"]}}
    seeds = derive_seeds(master_seed, cfg["replicates"])
    return [line for r, s in enumerate(seeds) for line in pd.replicate_lineups(r, s, cfg)]


# ---- the metric ---------------------------------------------------------------------------


def within_deal_variance(x: np.ndarray) -> float:
    """The direct estimate of E[Var(a seat's result | the cards)] in bb^2: the sample variance of
    a seat's results across the n tables of a deal, averaged over seats and deals. ``x`` has shape
    (n tables, D deals, n players), players in lineup order; player j sits in seat (j + r) mod n
    at table r."""
    n = x.shape[0]
    by_seat = np.stack([np.roll(x[r], r, axis=1) for r in range(n)])  # (tables, deals, seats)
    return float(by_seat.var(axis=0, ddof=1).mean())


def lineup_record(
    line: pd.Lineup, condition: str, chips: np.ndarray, counters: np.ndarray, half_width: float
) -> dict[str, Any]:
    from poker_lab.profiling.stats import raw_stats

    x = chips / pd.BIG_BLIND
    stats = [pd.player_stats(x[:, :, j], half_width) for j in range(line.n)]
    behaviour = raw_stats(counters)
    return {
        "condition": condition,
        "replicate": line.replicate,
        "index": line.index,
        "archetype": line.archetype,
        "bot_id": line.seats[0][1].player_id,
        "remaining": float(np.mean([line.n * s["s_u"] ** 2 / s["sigma"] ** 2 for s in stats])),
        "n_su2": float(np.mean([line.n * s["s_u"] ** 2 for s in stats])),
        "sigma2": float(np.mean([s["sigma"] ** 2 for s in stats])),
        "within_deal_variance": within_deal_variance(x),
        "ratio": [s["ratio"] for s in stats],
        "rho": [s["rho"] for s in stats],
        "max_abs_deal_sum_chips": int(np.abs(chips.sum(axis=0)).max()),
        "behaviour": {k: behaviour[k] for k in BEHAVIOUR},
    }


def _subsets(items: Sequence[str]) -> Iterator[tuple[str, ...]]:
    return chain.from_iterable(combinations(items, k) for k in range(len(items) + 1))


def decompose(remaining: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The Sobol-Hoeffding split of the baseline's remaining share (see the docstring), from the
    decomposition's eight conditions; arrays broadcast, so this also serves the bootstrap.
    Sharing the sources in T removes G(T) = R(all) - R(all but T), the sum of W_B over the
    nonempty B inside T, so W_B is the Moebius inverse of G. Returns W_B under ``"+"``-joined
    keys and each source's Shapley share: W_i plus equal parts of the interactions it is in."""
    live_to_name = {frozenset(v): k for k, v in DECOMPOSITION.items()}
    base = remaining["baseline"]

    def gain(t: tuple[str, ...]) -> np.ndarray:
        return base - remaining[live_to_name[frozenset(SOURCES) - frozenset(t)]]

    out: dict[str, np.ndarray] = {}
    for b in _subsets(SOURCES):
        if b:
            out["+".join(b)] = sum((-1) ** (len(b) - len(t)) * gain(t) for t in _subsets(b))
    shapley = {
        s: sum(out["+".join(b)] / len(b) for b in _subsets(SOURCES) if s in b) for s in SOURCES
    }
    return {**out, **{f"shapley_{s}": v for s, v in shapley.items()}}


def _ci(samples: np.ndarray, point: float) -> dict[str, float]:
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"mean": float(point), "lower": float(lo), "upper": float(hi)}


def decomposition(
    records: dict[str, list[dict]], n_boot: int, entropy: Sequence[int], *, prefix: str = ""
) -> dict:
    """Means over lineups of the remaining share by condition, the split and its shares of the
    baseline, with 95% percentile bootstrap CIs resampling lineups: one set of resamples for all
    conditions, since a lineup plays the same deals in each. ``prefix`` picks the pairing."""

    def condition(name: str) -> list[dict]:
        return records[name if name in ("baseline", "shared_all") else prefix + name]

    r = {name: np.array([row["remaining"] for row in condition(name)]) for name in DECOMPOSITION}
    lineups = len(r["baseline"])
    rng = np.random.default_rng(np.random.SeedSequence(list(entropy)))
    idx = rng.integers(lineups, size=(n_boot, lineups))
    means = {k: v.mean() for k, v in r.items()}
    boots = {k: v[idx].mean(axis=1) for k, v in r.items()}
    terms, term_boots = decompose(means), decompose(boots)
    return {
        "lineups": lineups,
        "remaining": {k: _ci(boots[k], means[k]) for k in r},
        "cancelled": _ci(1.0 - boots["baseline"], 1.0 - means["baseline"]),
        "terms": {k: _ci(term_boots[k], terms[k]) for k in terms},
        "shares": {
            k: _ci(term_boots[k] / boots["baseline"], terms[k] / means["baseline"]) for k in terms
        },
    }


def _mean(values: Sequence[float]) -> float:
    """The mean of the values that are numbers: a stat with no opportunities is NaN."""
    kept = [v for v in values if not math.isnan(v)]
    return float(np.mean(kept)) if kept else math.nan


def condition_summary(rows: list[dict], base: list[dict]) -> dict[str, Any]:
    """Across a condition's lineups: the committed statistic (the median ratio over players) and
    the changes from the baseline, lineup by lineup, in the per-hand SD and the bots' stats."""
    ratios = [x for row in rows for x in row["ratio"]]
    groups = [row["ratio"] for row in rows]
    sigma_change = np.log([a["sigma2"] / b["sigma2"] for a, b in zip(rows, base, strict=True)]) / 2
    return {
        "ratio": pd._quantiles(ratios),
        "groups": groups,
        "rho_median": float(np.median([x for row in rows for x in row["rho"]])),
        "remaining_mean": float(np.mean([row["remaining"] for row in rows])),
        "sigma_median": float(np.median([math.sqrt(row["sigma2"]) for row in rows])),
        "sigma_log_change_median": float(np.median(sigma_change)),
        "sigma_log_change_max_abs": float(np.abs(sigma_change).max()),
        "direct_over_n_su2": (
            float(sum(x["within_deal_variance"] for x in rows) / sum(x["n_su2"] for x in rows))
            if sum(x["n_su2"] for x in rows) > 0
            else math.nan
        ),
        "no_variance_lineups": int(sum(row["max_abs_deal_sum_chips"] == 0 for row in rows)),
        "max_abs_deal_sum_chips": max(row["max_abs_deal_sum_chips"] for row in rows),
        "behaviour": {k: _mean([x["behaviour"][k] for x in rows]) for k in BEHAVIOUR},
        "behaviour_change": {
            k: _mean(
                [x["behaviour"][k] - y["behaviour"][k] for x, y in zip(rows, base, strict=True)]
            )
            for k in BEHAVIOUR
        },
        "behaviour_change_lineups": {
            k: sum(
                not math.isnan(x["behaviour"][k] - y["behaviour"][k])
                for x, y in zip(rows, base, strict=True)
            )
            for k in BEHAVIOUR
        },
    }


def aggregate(records: dict[str, list[dict]], n_boot: int, master_seed: int) -> dict[str, Any]:
    by_condition = {}
    for q, (name, rows) in enumerate(records.items()):
        summary = condition_summary(rows, records["baseline"])
        summary["ratio_ci"] = pd.median_ci(
            summary.pop("groups"), n_boot, [master_seed, TAG_BOOTSTRAP, 1, q]
        )
        by_condition[name] = summary
    by_archetype = {}
    for a, arch in enumerate(ARCHETYPES, start=1):
        subset = {
            name: [row for row in rows if row["archetype"] == arch.value]
            for name, rows in records.items()
        }
        if not subset["baseline"]:
            continue
        by_archetype[arch.value] = {
            "decomposition": decomposition(subset, n_boot, [master_seed, TAG_BOOTSTRAP, 2, a]),
            "decomposition_history": decomposition(
                subset, n_boot, [master_seed, TAG_BOOTSTRAP, 4, a], prefix=HISTORY_PREFIX
            ),
            "ratio_median": {
                name: float(np.median([x for row in rows for x in row["ratio"]]))
                for name, rows in subset.items()
            },
            "sigma_median": {
                name: float(np.median([math.sqrt(row["sigma2"]) for row in rows]))
                for name, rows in subset.items()
            },
            "sigma_factor": {
                name: float(
                    np.median([math.sqrt(row["sigma2"]) for row in rows])
                    / np.median([math.sqrt(row["sigma2"]) for row in subset["baseline"]])
                )
                for name, rows in subset.items()
            },
            "behaviour": {
                name: condition_summary(rows, subset["baseline"])["behaviour"]
                for name, rows in subset.items()
            },
            "behaviour_change": {
                name: condition_summary(rows, subset["baseline"])["behaviour_change"]
                for name, rows in subset.items()
            },
        }
    shared = [n for n in (*DECOMPOSITION, *HISTORY) if n != "baseline"]
    main = decomposition(records, n_boot, [master_seed, TAG_BOOTSTRAP, 3])
    history = decomposition(records, n_boot, [master_seed, TAG_BOOTSTRAP, 5], prefix=HISTORY_PREFIX)
    return {
        "by_condition": by_condition,
        "decomposition": main,
        "decomposition_history": history,
        "by_archetype": by_archetype,
        "checks": {
            "exact_max_abs_deal_sum_chips": {
                n: by_condition[n]["max_abs_deal_sum_chips"] for n in EXACT_CHECKS
            },
            "shared_sigma_median_abs_change": max(
                abs(by_condition[n]["sigma_log_change_median"]) for n in shared
            ),
            "shared_behaviour_max_abs_change": max(
                abs(v)
                for n in shared
                for v in by_condition[n]["behaviour_change"].values()
                if not math.isnan(v)
            ),
            "only_mixed_over_baseline": [
                d["remaining"]["only_mixed"]["mean"] / d["remaining"]["baseline"]["mean"]
                for d in (main, history)
            ],
            "shared_sigma_factor_by_archetype": {
                arch: [e["sigma_factor"][n] for n in shared] for arch, e in by_archetype.items()
            },
            "shared_sigma_factor_extremes": [
                min(e["sigma_factor"][n] for e in by_archetype.values() for n in shared),
                max(e["sigma_factor"][n] for e in by_archetype.values() for n in shared),
            ],
            "shared_behaviour_by_archetype_max_abs_change": max(
                abs(v)
                for e in by_archetype.values()
                for n in shared
                for v in e["behaviour_change"][n].values()
                if not math.isnan(v)
            ),
            "direct_over_n_su2_max_abs_log": max(
                abs(math.log(by_condition[n]["direct_over_n_su2"]))
                for n in CONDITIONS
                if n not in EXACT_CHECKS
            ),
        },
    }


def committed_check(seed: int, cfg: dict[str, Any], records: list[dict]) -> dict[str, Any] | None:
    """The baseline against P2.11b's committed ``clones_6``, when this run repeats it: same seed
    and sizes. The largest absolute difference in any player's ratio, which must be zero."""
    import json

    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    old = committed["meta"]["config"]
    sizes = ("replicates", "population_per_archetype", "deals", "chunk_deals", "equity_samples")
    if committed["meta"]["seed"] != seed or any(old[k] != cfg[k] for k in sizes):
        return None
    rows = {
        (row["replicate"], row["index"]): row
        for row in committed["per_lineup"]
        if row["kind"] == "clones_6"
    }
    diffs = [
        abs(a - p["ratio"])
        for rec in records
        for a, p in zip(
            rec["ratio"], rows[(rec["replicate"], rec["index"])]["players"], strict=True
        )
    ]
    return {"players": len(diffs), "max_abs_ratio_difference": max(diffs)}


# ---- predictions (written after development) --------------------------------------------------


def _cond(name: str, *path: Any) -> tuple:
    return ("aggregates", "by_condition", name, *path)


def _arch(arch: str, *path: Any) -> tuple:
    return ("aggregates", "by_archetype", arch, *path)


def _split(source: str) -> tuple:
    return ("aggregates", "decomposition", "shares", f"shapley_{source}", "mean")


def _term(key: str) -> tuple:
    return ("aggregates", "decomposition", "terms", key, "mean")


_SHARED = tuple(n for n in DECOMPOSITION if n != "baseline")

# The tests are phase2_duplicate's. ``first_lowest``: the first value is below all the others.
_HISTORY = ("aggregates", "decomposition_history")
_MAIN = ("aggregates", "decomposition")
_ALL_SHARED = (*_SHARED, *HISTORY)
_CHECK = ("aggregates", "checks")


def _split_history(source: str) -> tuple:
    return (*_HISTORY, "shares", f"shapley_{source}", "mean")


PREDICTIONS: dict[str, dict[str, Any]] = {
    "A1": {
        "claim": "The baseline repeats P2.11b's committed clones_6: largest ratio difference 0",
        "values": [("committed_check", "max_abs_ratio_difference")],
        "test": ("<=", 1e-9),
    },
    "A2": {
        "claim": "All three shared, all three off: every per-deal sum is exactly 0 chips",
        "values": [(*_CHECK, "exact_max_abs_deal_sum_chips", n) for n in EXACT_CHECKS],
        "test": ("<=", 0),
    },
    "A3": {
        "claim": "Direct within-deal variance over n s_u^2: largest |log| at most 0.08",
        "values": [(*_CHECK, "direct_over_n_su2_max_abs_log")],
        "test": ("<=", 0.08),
    },
    "B1": {
        "claim": "Sharing a source: median log SD change from the default within 0.05",
        "values": [_cond(n, "sigma_log_change_median") for n in _ALL_SHARED],
        "test": ("between", [-0.05, 0.05]),
    },
    "B2": {
        "claim": "Sharing a source: each stat's mean within 0.03 of the default (largest)",
        "values": [(*_CHECK, "shared_behaviour_max_abs_change")],
        "test": ("<=", 0.03),
    },
    "B3": {
        "claim": "Sharing a source, by archetype: median SD factor within [0.8, 1.4]",
        "values": [(*_CHECK, "shared_sigma_factor_extremes", k) for k in range(2)],
        "test": ("between", [0.8, 1.4]),
    },
    "B4": {
        "claim": "Sharing a source, by archetype: each stat's mean within 0.15 (largest)",
        "values": [(*_CHECK, "shared_behaviour_by_archetype_max_abs_change")],
        "test": ("<=", 0.15),
    },
    "C1": {
        "claim": "Only mixed live: 90-110% of the default's remaining share, both keys",
        "values": [(*_CHECK, "only_mixed_over_baseline", k) for k in range(2)],
        "test": ("between", [0.9, 1.1]),
    },
    "C2": {
        "claim": "Only equity live: 10-20% of a hand's variance, both keys",
        "values": [(*d, "remaining", "only_equity", "mean") for d in (_MAIN, _HISTORY)],
        "test": ("between", [0.10, 0.20]),
    },
    "C3": {
        "claim": "Noise shared alone, equity shared alone: median ratio in [1.35, 1.9], both keys",
        "values": [
            _cond(n, "ratio", "median")
            for n in (
                "shared_noise",
                "shared_equity",
                "history_shared_noise",
                "history_shared_equity",
            )
        ],
        "test": ("between", [1.35, 1.9]),
    },
    "D1": {
        "claim": "Decision key, mixed shared alone: median ratio in [4.5, 12]",
        "values": [_cond("shared_mixed", "ratio", "median")],
        "test": ("between", [4.5, 12]),
    },
    "D2": {
        "claim": "Decision key, split: mixed at least 75% of the shortfall",
        "values": [_split("mixed")],
        "test": (">=", 0.75),
    },
    "D3": {
        "claim": "Decision key, split: equity 5-20% of the shortfall",
        "values": [_split("equity")],
        "test": ("between", [0.05, 0.20]),
    },
    "D4": {
        "claim": "Decision key, split: noise at most 8% of the shortfall",
        "values": [_split("noise")],
        "test": ("<=", 0.08),
    },
    "D5a": {
        "claim": "Decision key: mixed x equity at least 4% of a hand's variance",
        "values": [_term("mixed+equity")],
        "test": (">=", 0.04),
    },
    "D5b": {
        "claim": "Decision key: mixed x equity is above equity alone",
        "values": [_term("mixed+equity"), _term("equity")],
        "test": ("decreasing", None),
    },
    "D6": {
        "claim": "Decision key: maniacs have the lowest equity split share of the five",
        "values": [
            _arch(a, "decomposition", "shares", "shapley_equity", "mean")
            for a in ("maniac", "nit", "tag", "lag", "calling_station")
        ],
        "test": ("first_lowest", None),
    },
    "H1a": {
        "claim": "Only noise live, decision key: at most 10% of a hand's variance",
        "values": [(*_MAIN, "remaining", "only_noise", "mean")],
        "test": ("<=", 0.10),
    },
    "H1b": {
        "claim": "Only noise live, history key: at least 40% of a hand's variance",
        "values": [(*_HISTORY, "remaining", "only_noise", "mean")],
        "test": (">=", 0.40),
    },
    "H2": {
        "claim": "History key, mixed shared alone: median ratio at most 2.2",
        "values": [_cond("history_shared_mixed", "ratio", "median")],
        "test": ("<=", 2.2),
    },
    "H3": {
        "claim": "History key, split: mixed 40-65% of the shortfall",
        "values": [_split_history("mixed")],
        "test": ("between", [0.40, 0.65]),
    },
    "H4": {
        "claim": "History key, split: noise 30-50% of the shortfall",
        "values": [_split_history("noise")],
        "test": ("between", [0.30, 0.50]),
    },
    "H5": {
        "claim": "History key, split: equity at most 15% of the shortfall",
        "values": [_split_history("equity")],
        "test": ("<=", 0.15),
    },
    "H6": {
        "claim": "History key: mixed x noise at least 25% of a hand's variance",
        "values": [(*_HISTORY, "terms", "mixed+noise", "mean")],
        "test": (">=", 0.25),
    },
    "H7a": {
        "claim": "Decision key: mixed's split share above noise's",
        "values": [_split("mixed"), _split("noise")],
        "test": ("decreasing", None),
    },
    "H7b": {
        "claim": "History key: mixed's split share above noise's",
        "values": [_split_history("mixed"), _split_history("noise")],
        "test": ("decreasing", None),
    },
    "E1a": {
        "claim": "Mixed actions off: the calling station's median SD below 0.1x the default's",
        "values": [_arch("calling_station", "sigma_factor", "off_mixed")],
        "test": ("<=", 0.1),
    },
    "E1b": {
        "claim": "Mixed actions off: nits' median SD above 1.5x the default's",
        "values": [_arch("nit", "sigma_factor", "off_mixed")],
        "test": (">=", 1.5),
    },
    "E2a": {
        "claim": "Mixed actions off: mean c-bet rate up at least 0.05 (lineups with opportunities)",
        "values": [_cond("off_mixed", "behaviour_change", "cbet")],
        "test": (">=", 0.05),
    },
    "E2b": {
        "claim": "Mixed actions off: mean fold to a bet down at least 0.04 (lineups with them)",
        "values": [_cond("off_mixed", "behaviour_change", "fold_to_bet")],
        "test": ("<=", -0.04),
    },
    "E3a": {
        "claim": "Noise off: median log SD change within 0.03",
        "values": [_cond("off_noise", "sigma_log_change_median")],
        "test": ("between", [-0.03, 0.03]),
    },
    "E3b": {
        "claim": "Noise off: each stat's mean within 0.03 of the default",
        "values": [_cond("off_noise", "behaviour_change", k) for k in BEHAVIOUR],
        "test": ("between", [-0.03, 0.03]),
    },
    "E3c": {
        "claim": "Noise off: median ratio in [1.35, 1.9]",
        "values": [_cond("off_noise", "ratio", "median")],
        "test": ("between", [1.35, 1.9]),
    },
}


# ---- the run ------------------------------------------------------------------------------


def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase2" / "randomness"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()
    lineups = plan(seed, cfg)
    workers = workers or cfg.get("workers") or 1
    conditions = list(CONDITIONS)
    records: dict[str, list[dict]] = {c: [] for c in conditions}
    times: dict[str, float] = {}
    with closing(_results(tasks(lineups, cfg, conditions), workers)) as res:
        for condition in conditions:
            started = time.perf_counter()
            for line in lineups:
                parts = [next(res) for _ in line.chunks]
                chips = np.concatenate([p[0] for p in parts], axis=1)
                counters = np.sum([p[1] for p in parts], axis=0)
                records[condition].append(
                    lineup_record(line, condition, chips, counters, cfg["half_width"])
                )
            times[condition] = round(time.perf_counter() - started, 1)
            print(f"{condition} done ({times[condition]:.0f} s)", flush=True)
    for name in EXACT_CHECKS:
        worst = max(row["max_abs_deal_sum_chips"] for row in records[name])
        if worst != 0:
            raise AssertionError(f"{name}: a per-deal sum is {worst} chips, not 0")
    out = {
        "meta": meta,
        "aggregates": aggregate(records, cfg["bootstrap"], seed),
        "committed_check": committed_check(seed, cfg, records["baseline"]),
        "per_lineup": records,
        "condition_seconds": times,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    out["predictions"] = pd.predictions(out, PREDICTIONS)
    save_json(out, out_dir / "randomness.json")
    (out_dir / "randomness_summary.md").write_text(
        markdown_summary(out), encoding="utf-8", newline="\n"
    )
    make_figures(out, out_dir)
    return out


# ---- reporting ----------------------------------------------------------------------------

TERM_LABELS = {
    "mixed": "Mixed actions alone",
    "noise": "Bet-size noise alone",
    "equity": "Monte Carlo equity alone",
    "mixed+noise": "Mixed x noise",
    "mixed+equity": "Mixed x equity",
    "noise+equity": "Noise x equity",
    "mixed+noise+equity": "All three together",
}


def _ratio_text(x: float) -> str:
    """A quantile or CI bound interpolating a player with no variance left is not a ratio."""
    return "no variance" if x >= 1e6 else pd._fmt_ratio(x)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _pct_ci(e: dict[str, float]) -> str:
    return f"{_pct(e['mean'])} [{_pct(e['lower'])}, {_pct(e['upper'])}]"


def markdown_summary(r: dict[str, Any]) -> str:
    cfg = r["meta"]["config"]
    agg = r["aggregates"]
    dec = agg["decomposition"]
    lines = [
        "# P2.11c: how six copies' shortfall divides between the bots' randomness",
        "",
        f"Seed {r['meta']['seed']}, commit {(r['meta']['git']['commit'] or '?')[:7]}.",
        "",
        f"{dec['lineups']} lineups of six copies of one bot, {cfg['deals']:,} deals each, played "
        "in every condition with the same bots and the same cards. Remaining share: "
        "n s_u^2 / sigma^2 = 1 / ratio, the part of a hand's variance the rotation does not "
        "cancel. Shared: the six tables of a deal use the same draws of that source, so each "
        "hand's distribution is the default's. Off: the source is a constant, so the bots play "
        "differently.",
        "",
        "## By condition",
        "",
        "| Condition | Median ratio [IQR] | 95% CI | Remaining share (mean) | Median rho | "
        "Median SD per hand (bb) | SD change from default (median) | VPIP | PFR | Fold to bet |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, e in agg["by_condition"].items():
        q = e["ratio"]
        b = e["behaviour"]
        lines.append(
            f"| {LABELS[name]} | {_ratio_text(q['median'])} [{_ratio_text(q['q25'])}, "
            f"{_ratio_text(q['q75'])}] | [{_ratio_text(e['ratio_ci']['lower'])}, "
            f"{_ratio_text(e['ratio_ci']['upper'])}] | "
            f"{_pct(e['remaining_mean'])} | {e['rho_median']:+.3f} | {e['sigma_median']:.2f} | "
            f"{math.expm1(e['sigma_log_change_median']):+.1%} | {b['vpip']:.3f} | {b['pfr']:.3f} | "
            f"{b['fold_to_bet']:.3f} |"
        )
    lines += [
        "",
        "## The split of the shortfall (shared conditions only)",
        "",
        f"Cancelled by the rotation: {_pct_ci(dec['cancelled'])} of a hand's variance. Remaining: "
        f"{_pct_ci(dec['remaining']['baseline'])}, which divides as follows.",
        "",
        "| Term | Share of a hand's variance | Share of the shortfall |",
        "|---|---|---|",
    ]
    for key, label in TERM_LABELS.items():
        lines.append(f"| {label} | {_pct_ci(dec['terms'][key])} | {_pct_ci(dec['shares'][key])} |")
    for s in SOURCES:
        key = f"shapley_{s}"
        lines.append(
            f"| {SOURCE_LABELS[s]}, interactions split equally | {_pct_ci(dec['terms'][key])} | "
            f"{_pct_ci(dec['shares'][key])} |"
        )
    hist = agg["decomposition_history"]
    lines += [
        "",
        "## The same split with the tables paired by the exact public history",
        "",
        "The two ends do not depend on the pairing; the terms between them do.",
        "",
        "| Term | Share of a hand's variance | Share of the shortfall |",
        "|---|---|---|",
    ]
    for key, label in TERM_LABELS.items():
        lines.append(
            f"| {label} | {_pct_ci(hist['terms'][key])} | {_pct_ci(hist['shares'][key])} |"
        )
    for s in SOURCES:
        key = f"shapley_{s}"
        lines.append(
            f"| {SOURCE_LABELS[s]}, interactions split equally | {_pct_ci(hist['terms'][key])} | "
            f"{_pct_ci(hist['shares'][key])} |"
        )
    lines += [
        "",
        "## By archetype",
        "",
        "| Archetype | Median ratio | Cancelled | Mixed (split) | Noise (split) | Equity (split) | "
        "Mixed, noise, equity (split, history key) | Median ratio, mixed off | "
        "SD, default -> mixed off (bb) | VPIP, default -> mixed off |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arch, e in agg["by_archetype"].items():
        d = e["decomposition"]
        b, off = e["behaviour"]["baseline"], e["behaviour"]["off_mixed"]
        h = e["decomposition_history"]["shares"]
        lines.append(
            f"| {pd.ARCH_LABELS[arch]} | {_ratio_text(e['ratio_median']['baseline'])} | "
            f"{_pct_ci(d['cancelled'])} | {_pct_ci(d['shares']['shapley_mixed'])} | "
            f"{_pct_ci(d['shares']['shapley_noise'])} | "
            f"{_pct_ci(d['shares']['shapley_equity'])} | "
            f"{_pct(h['shapley_mixed']['mean'])}, {_pct(h['shapley_noise']['mean'])}, "
            f"{_pct(h['shapley_equity']['mean'])} | "
            f"{_ratio_text(e['ratio_median']['off_mixed'])} | "
            f"{e['sigma_median']['baseline']:.2f} -> {e['sigma_median']['off_mixed']:.2f} | "
            f"{b['vpip']:.3f} -> {off['vpip']:.3f} |"
        )
    c = agg["checks"]
    lines += [
        "",
        "## Checks",
        "",
        "Largest |per-deal sum| in chips with all three shared or off: "
        f"{c['exact_max_abs_deal_sum_chips']}. Direct within-deal variance over n s_u^2: at most "
        f"{math.expm1(c['direct_over_n_su2_max_abs_log']):.1%} from 1. Shared conditions against "
        f"the default: SD at most {math.expm1(c['shared_sigma_median_abs_change']):.1%} (median "
        f"over lineups), stats at most {c['shared_behaviour_max_abs_change']:.4f}.",
    ]
    if r["committed_check"] is not None:
        c = r["committed_check"]
        lines.append(
            f"Default against P2.11b's committed clones_6: {c['players']} players, largest ratio "
            f"difference {c['max_abs_ratio_difference']:g}."
        )
    if r["predictions"]:
        lines += [
            "",
            "## Predictions",
            "",
            "| | Prediction | Result | Holds |",
            "|---|---|---|---|",
        ]
        for key, p in r["predictions"].items():
            result = ", ".join("missing" if v is None else f"{v:.3g}" for v in p["values"].values())
            lines.append(f"| {key} | {p['claim']} | {result} | {'yes' if p['passed'] else 'NO'} |")
    lines += ["", f"Run time {r['elapsed_seconds']:.0f} s."]
    return "\n".join(lines) + "\n"


def make_figures(r: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import legend, new_figure, save_both, style_axes, title

    agg = r["aggregates"]
    rows = [
        ("All, paired by decision", agg["decomposition"]),
        ("All, paired by history", agg["decomposition_history"]),
    ] + [(pd.ARCH_LABELS[a], e["decomposition"]) for a, e in agg["by_archetype"].items()]

    def division(theme):
        fig, ax = new_figure(theme, 1, 1, figsize=(9, 0.55 * len(rows) + 2.2))
        style_axes(ax, theme, grid_axis="x")
        s = theme["series"]
        parts = [
            ("Cancelled by the rotation", theme["axis"], lambda d: d["cancelled"]["mean"]),
            *[
                (SOURCE_LABELS[src], s[i], lambda d, k=src: d["terms"][k]["mean"])
                for i, src in enumerate(SOURCES)
            ],
            (
                "Interactions",
                s[3],
                lambda d: sum(d["terms"][k]["mean"] for k in TERM_LABELS if "+" in k),
            ),
        ]
        ys = np.arange(len(rows))
        left = np.zeros(len(rows))
        for label, color, get in parts:
            width = np.clip([get(d) for _, d in rows], 0.0, None)
            ax.barh(ys, width, left=left, color=color, height=0.62, label=label)
            left += width
        ax.set_yticks(ys)
        ax.set_yticklabels([name for name, _ in rows], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, max(1.0, float(left.max())))
        ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
        ax.set_xlabel("Share of a hand's variance")
        legend(ax, theme, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
        title(
            fig,
            theme,
            "Where the variance of a hand between six copies goes",
            "Tables share each source's draws, so the bots play as usual; archetypes by decision",
            top=0.86,
        )
        return fig

    names = [n for n in CONDITIONS if n not in EXACT_CHECKS]

    def ratios(theme):
        fig, ax = new_figure(theme, 1, 1, figsize=(9, 0.42 * len(names) + 1.7))
        style_axes(ax, theme, grid_axis="x")
        s = theme["series"]
        every = [1.0]
        for k, name in enumerate(names):
            q = agg["by_condition"][name]["ratio"]
            every += [q["q25"], q["q75"]]
            color = s[1] if name.startswith("off_") else s[0]
            ax.plot([q["q25"], q["q75"]], [k, k], color=color, linewidth=2.5,
                    solid_capstyle="round")  # fmt: skip
            ax.plot(q["median"], k, "|", color=color, markersize=14, markeredgewidth=2.5)
        pd._ratio_axis(ax, every)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels([LABELS[n] for n in names], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Times fewer hands than independent dealing (log)")
        title(
            fig,
            theme,
            "Six copies of one bot: the gain in each condition",
            "Median over players, bars the middle half; orange changes how the bots play",
            top=0.88,
        )
        return fig

    save_both(division, out_dir, "randomness_division")
    save_both(ratios, out_dir, "randomness_ratio")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--quick", action="store_true", help="use configs/phase2_quick.yaml")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    path = CONFIG.with_name("phase2_quick.yaml") if args.quick else args.config
    cfg = load_config(path)["randomness"]
    r = run(args.seed, cfg=cfg, workers=args.workers, out_dir=args.out)
    print(markdown_summary(r))


if __name__ == "__main__":
    main()

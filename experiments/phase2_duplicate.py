"""P2.11b: duplicate dealing for n players, measured at 6-max no-limit hold'em.

    python -m experiments.phase2_duplicate --seed 2026 [--quick]
    python -m experiments.run_phase2 --seed 2026 --only duplicate

Pre-registered plan (written after development on seeds 98 and 99, before the real run):

**Question.** Duplicate dealing removes card luck heads-up (Step 23; at 10 bb push/fold Step 28
measured 32,073 median hands naive against 10,733). With n players each deal is played at n
tables with the seats rotated cyclically (``poker_lab.engine.duplicate.duplicate_match``:
player j in seat (j + r) mod n at table r), so every player holds every seat's cards and
position once per deal. How much variance per hand played does that remove at 6-max no-limit
hold'em with Phase 1's archetype bots, and how does it depend on the opponents (copies of one
bot, bots of one archetype, a random mix) and on the number of players (2, 3, 6)?

**The metric.** For one player of a lineup, x[i, r] is the result of deal i at table r in big
blinds (D deals, n tables). sigma^2 is the sample variance of all n D results: the variance per
hand when every hand is dealt independently (each table alone is an ordinary table). The
duplicate unit is a deal's mean over its tables, u[i] = mean_r x[i, r], with sample variance
s_u^2 over the D deals. A deal costs n hands, so

    ratio = sigma^2 / (n s_u^2)

is how many times fewer hands duplicate dealing needs than independent dealing for an interval
of the same width. The point estimate of the win rate is the same (the mean of all hands); only
the interval width differs. With rho the mean correlation between a player's results at two
tables of one deal,

    Var(u) = (1 / n^2) sum_{r, r'} Cov(x_r, x_r') = sigma^2 (1 + (n - 1) rho) / n,
    ratio = 1 / (1 + (n - 1) rho),

so duplicate dealing pays only when rho < 0, and the ratio is unbounded as rho -> -1/(n - 1):
a player's result at one table then cancels exactly against its results at the others, as for
copies of one deterministic bot (the ``calling_6`` check). rho is measured as the mean over the
offsets k = 1..n-1 of c_k, the correlation of x[i, r] with x[i, (r + k) mod n], pooled over the
tables and deals around the player's grand mean (offsets k and n - k are the same pairs, so
k = 1..floor(n/2) are reported). With these estimates the identity is exact once the degrees
of freedom are counted: ratio = n (D - 1) / ((n D - 1) (1 + (n - 1) rho)). Hands needed for a 95%
interval of +/- ``half_width`` bb/100 (h = half_width / 100 bb per hand): naive
(1.96 sigma / h)^2, duplicate n (1.96 s_u / h)^2. A player whose per-deal means never vary
(s_u = 0) gets the ratio ``RATIO_CAP`` and the flag ``no_variance_left``; the quartiles
interpolate, so such a value would print as a huge number (no player had one in development).

**Lineups.** Per replicate (seeds from the master seed with ``derive_seeds``), a population of
``population_per_archetype`` bots of each of Phase 1's five archetypes (``make_population``;
``ParametricBot`` with ``equity_samples`` Monte Carlo samples, as in Phase 1):

* ``clones_6``: one lineup per archetype: six copies of one bot drawn from the archetype (the
  same parameters, ids ``<id>-c0`` to ``<id>-c5``). Only the bots' own randomness (mixed
  actions, Monte Carlo equity) is not mirrored between the tables: an upper reference.
* ``mirror_6``: one lineup per archetype: six different bots of the archetype.
* ``mixed_6``, ``mixed_3``, ``mixed_2``: bots drawn at random from the whole population,
  without replacement within a lineup (a bot can sit in several lineups of a replicate), at a
  table of six, three or two seats (table size = n).
* ``calling_6``: six always-call agents (``CallingAgent``): a check. Every pot is decided by
  the cards alone, so every per-deal mean must be exactly 0 chips.

The config gives the lineups per replicate; for ``clones_6`` and ``mirror_6`` the number is
how many archetypes are covered, in Phase 1's order (5: all). The seats go in the order drawn.
The rotation never changes who sits to whose left, so a lineup measures its own seating order
(recorded in the JSON as ``seating``); the aggregates average over the orders drawn.

**Simulation.** Blinds 1/2, 100 bb stacks reset every hand. A lineup's deals are played in
chunks of ``chunk_deals``, each one ``duplicate_match`` run as one worker task with its own
seed, from ``SeedSequence([replicate seed, 3, kind index, lineup, chunk])``. Each chunk starts
with the button at seat 0, which does not matter: every player holds every position once per
deal, whatever the button. The population draws
from ``[replicate seed, 1]``, a kind's lineup draws from ``[replicate seed, 2, kind]``, the
calling check from ``[master seed, 4, chunk]`` and the bootstrap from ``[master seed, 5, ...]``.
The tags keep the streams apart: SeedSequence pads entropy shorter than four words with zeros,
so ``[s]`` and ``[s, 0, 0]`` would be the same seed. Results are aggregated in task order, so
they do not depend on the number of workers.

**Aggregates.** Per lineup kind, and per kind x player archetype: the median and quartiles of
the ratio over players, and a 95% percentile bootstrap CI of the median that resamples lineups
(a lineup's players stay together), reported only from ``MIN_CI_LINEUPS`` lineups up, since a
percentile bootstrap over few clusters covers badly; rho and the correlation by offset; the
per-hand SD and the median hands needed. Mixed lineups against the number of players.

Reading the CIs: heads-up the two players' results are opposite, so their ratios are equal: a
lineup counts twice among the players but once in the bootstrap, and a heads-up ratio by
archetype describes the pair. A bot can sit in several mixed lineups of a replicate, so the CIs
are conditional on the populations and a little narrow. The quartiles and the share below 1
include each ratio's own sampling noise (clones of one bot spread about 2% in development).

The heads-up reference: Step 28's push/fold matchups (P3.3's bridge results): the median over
the matchups against the bots of each matchup's naive / duplicate hands needed, the same
statistic as here (the ratio of the two medians is a different number).

**Seen in development** (not predictions; seeds 98 | 99, 2 replicates, 1,500 deals per lineup,
5, 5, 6, 6 and 6 lineups per replicate; a short pilot on seed 97 sized those runs):

* Median ratio: six copies of one bot 1.60x | 1.54x; six bots of one archetype 1.46x | 1.45x;
  six bots, random mix 1.33x | 1.26x; three bots 1.23x | 1.35x; two bots 1.35x | 1.40x. Heads-up
  push/fold (the reference above, committed results): 2.06x over the matchups.
* Median rho: copies -0.075 | -0.070; one archetype -0.063 | -0.062; random mix -0.050 | -0.041;
  three bots -0.094 | -0.130; two bots -0.260 | -0.286. So (n - 1) rho stays about -0.2 to -0.3
  at every n, and the ratio about flat.
* Six copies of one bot gain little more: the bots randomize their actions and their Monte
  Carlo equity, so the plays of a deal part early in the hand.
* The correlation is nearly flat across the offsets: at six players the medians at k = 1, 2, 3
  differ by at most 0.013 within a kind.
* Nits gain least in the random mix (1.20x | 1.13x; the others 1.28x to 1.44x) and among copies
  (1.35x | 1.34x; the others 1.49x to 1.84x). In the one-archetype lineups only on seed 98
  (1.17x | 1.35x, with TAG at 1.34x on seed 99), so that is not predicted.
* Players with a ratio below 1: 0-2% in the six-player kinds, 0% with three, 8% with two.
* Calling check: exactly 0 chips on both seeds. About 140 s per run (378,000 hands).
* The median ratio did not drift with the deals between 250 and 1,500, and the spread between
  lineups dominates the CI, so the full configuration adds lineups, not deals: 10 replicates
  (10 lineups per archetype among copies and one-archetype lineups) and 20 heads-up lineups per
  replicate, since a heads-up lineup is one value and had the widest CI (+/-9% projected).

**Predictions** (thresholds from development; evaluated by ``predictions``):

* A1: calling check: every per-deal mean is exactly 0 chips.
* B1: six bots, random mix: median ratio in [1.15, 1.5], well below heads-up push/fold's 2x.
* B2: six copies of one bot: median ratio in [1.35, 1.9]: under 2x even with identical players,
  because the bots' own randomness is not mirrored.
* B3: the median ratio falls from copies of one bot to one archetype to the random mix.
* B4: at most 5% of the players have a ratio below 1, in each of the three six-player kinds.
* C1: nits gain least in the random mix: the lowest median ratio of the five archetypes.
* C2: nits gain least among copies too.
* D1, D2, D3: at six players the correlation barely depends on the offset: within each kind
  (copies, one archetype, random mix) the medians at k = 1, 2, 3 differ by at most 0.03.
* E1: random mixes: median ratio in [1.1, 1.6] at each of n = 2, 3, 6: the gain stays about
  flat as the table grows.
* E2: random mixes: median rho at most -0.02 at each of n = 2, 3, 6.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import (
    REPO_ROOT,
    RESULTS_ROOT,
    derive_seeds,
    load_config,
    run_metadata,
    save_json,
)
from poker_lab.bots.archetypes import ARCHETYPES
from poker_lab.bots.population import PlayerSpec, make_population
from poker_lab.evaluation.intervals import bootstrap_ci
from poker_lab.evaluation.winrate import DuplicateDealEvaluator, HandResult, WinRateEvaluator

CONFIG = REPO_ROOT / "experiments" / "configs" / "phase2.yaml"
BRIDGE = RESULTS_ROOT / "phase3" / "bridge" / "bridge.json"

KINDS = ("clones_6", "mirror_6", "mixed_6", "mixed_3", "mixed_2")
CALLING = "calling_6"
TABLE_SIZE = {"clones_6": 6, "mirror_6": 6, "mixed_6": 6, "mixed_3": 3, "mixed_2": 2, CALLING: 6}
SMALL_BLIND, BIG_BLIND, STACK_BB = 1, 2, 100
# SeedSequence tags (see the docstring)
TAG_POPULATION, TAG_DRAWS, TAG_CHUNKS, TAG_CALLING, TAG_BOOTSTRAP = 1, 2, 3, 4, 5
RATIO_CAP = 1e9  # the ratio of a player whose per-deal means never vary (s_u = 0)
ZERO_SD = 1e-9
MIN_CI_LINEUPS = 8  # fewer lineups: no bootstrap CI (see the docstring)


def _kind(kind: str, *path: Any) -> tuple:
    return ("aggregates", "by_kind", kind, *path)


def _archetypes(kind: str) -> list[tuple]:
    """Median ratios by archetype, the nit first."""
    return [_kind(kind, "by_archetype", a.value, "ratio", "median") for a in ARCHETYPES]


def _offset_medians(kind: str) -> list[tuple]:
    return [_kind(kind, "rho_by_offset", k, "median") for k in range(3)]


# Each prediction: a claim, the quantities it tests (paths of keys into the results; a list
# index for an offset) and the test on them: every quantity (">=", x), ("<=", x) or ("between",
# [lo, hi]); all together ("decreasing", None), ("first_lowest", None) or ("range_at_most", x).
PREDICTIONS: dict[str, dict[str, Any]] = {
    "A1": {
        "claim": "Calling check: every per-deal mean is exactly 0 chips",
        "values": [("calling_check", "max_abs_deal_mean_chips")],
        "test": ("<=", 0.0),
    },
    "B1": {
        "claim": "Six bots, random mix: median ratio in [1.15, 1.5]",
        "values": [_kind("mixed_6", "ratio", "median")],
        "test": ("between", [1.15, 1.5]),
    },
    "B2": {
        "claim": "Six copies of one bot: median ratio in [1.35, 1.9]",
        "values": [_kind("clones_6", "ratio", "median")],
        "test": ("between", [1.35, 1.9]),
    },
    "B3": {
        "claim": "The median ratio falls: copies, then one archetype, then the random mix",
        "values": [_kind(k, "ratio", "median") for k in ("clones_6", "mirror_6", "mixed_6")],
        "test": ("decreasing", None),
    },
    "B4": {
        "claim": "At most 5% of the players have a ratio below 1, in each six-player kind",
        "values": [_kind(k, "share_below_1") for k in ("clones_6", "mirror_6", "mixed_6")],
        "test": ("<=", 0.05),
    },
    "C1": {
        "claim": "Random mix: nits have the lowest median ratio of the five archetypes",
        "values": _archetypes("mixed_6"),
        "test": ("first_lowest", None),
    },
    "C2": {
        "claim": "Copies of one bot: nits have the lowest median ratio of the five archetypes",
        "values": _archetypes("clones_6"),
        "test": ("first_lowest", None),
    },
    "D1": {
        "claim": "Copies of one bot: correlations at offsets 1-3 within 0.03",
        "values": _offset_medians("clones_6"),
        "test": ("range_at_most", 0.03),
    },
    "D2": {
        "claim": "One archetype: correlations at offsets 1-3 within 0.03",
        "values": _offset_medians("mirror_6"),
        "test": ("range_at_most", 0.03),
    },
    "D3": {
        "claim": "Random mix: correlations at offsets 1-3 within 0.03",
        "values": _offset_medians("mixed_6"),
        "test": ("range_at_most", 0.03),
    },
    "E1": {
        "claim": "Random mixes: median ratio in [1.1, 1.6] at n = 2, 3 and 6",
        "values": [("aggregates", "mixed_by_n", n, "ratio_median") for n in ("2", "3", "6")],
        "test": ("between", [1.1, 1.6]),
    },
    "E2": {
        "claim": "Random mixes: median rho at most -0.02 at n = 2, 3 and 6",
        "values": [("aggregates", "mixed_by_n", n, "rho_median") for n in ("2", "3", "6")],
        "test": ("<=", -0.02),
    },
}


# ---- lineups and tasks --------------------------------------------------------------------


@dataclass(frozen=True)
class Lineup:
    """Players in lineup order (player j sits in seat (j + r) mod n at table r) and the
    duplicate matches that play the lineup's deals."""

    kind: str
    replicate: int  # -1 for the calling check
    index: int  # within its kind and replicate
    archetype: str | None  # clones and mirror lineups: the archetype they cover
    seats: tuple[tuple[str, PlayerSpec | None], ...]  # (player id, bot); None: a CallingAgent
    chunks: tuple[tuple[int, int], ...]  # (seed, deals) per duplicate match

    @property
    def n(self) -> int:
        return len(self.seats)

    @property
    def deals(self) -> int:
        return sum(d for _, d in self.chunks)


def _chunks(entropy: Sequence[int], deals: int, chunk_deals: int) -> tuple[tuple[int, int], ...]:
    """Split ``deals`` into chunks of ``chunk_deals`` (the last may be shorter), chunk c seeded
    from ``SeedSequence([*entropy, c])``."""
    sizes = [chunk_deals] * (deals // chunk_deals)
    if deals % chunk_deals:
        sizes.append(deals % chunk_deals)
    return tuple(
        (int(np.random.SeedSequence([*entropy, c]).generate_state(1)[0]), size)
        for c, size in enumerate(sizes)
    )


def _validate(cfg: dict[str, Any]) -> None:
    unknown = set(cfg["lineups"]) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown lineup kinds {sorted(unknown)}; known: {KINDS}")
    if cfg["deals"] < 2 or cfg["calling_deals"] < 1 or cfg["chunk_deals"] < 1:
        raise ValueError("need deals >= 2, calling_deals >= 1 and chunk_deals >= 1")
    for kind in ("clones_6", "mirror_6"):
        if not 0 <= cfg["lineups"].get(kind, 0) <= len(ARCHETYPES):
            raise ValueError(f"{kind}: one lineup per archetype, at most {len(ARCHETYPES)}")
    per = cfg["population_per_archetype"]
    if cfg["lineups"].get("mirror_6", 0) and per < 6:
        raise ValueError("mirror_6 needs at least 6 bots per archetype")
    if max(TABLE_SIZE[k] for k in KINDS if cfg["lineups"].get(k, 0)) > per * len(ARCHETYPES):
        raise ValueError("the population is smaller than a table")


def replicate_lineups(replicate: int, seed: int, cfg: dict[str, Any]) -> list[Lineup]:
    """One replicate's population and lineups, kind by kind in ``KINDS`` order."""
    population = make_population(
        cfg["population_per_archetype"],
        np.random.default_rng(np.random.SeedSequence([seed, TAG_POPULATION])),
    )
    members = {a: [p for p in population if p.archetype == a] for a in ARCHETYPES}
    out = []
    for q, kind in enumerate(KINDS):
        count = cfg["lineups"].get(kind, 0)
        rng = np.random.default_rng(np.random.SeedSequence([seed, TAG_DRAWS, q]))
        for k in range(count):
            archetype = None
            if kind == "clones_6":
                archetype = ARCHETYPES[k]
                bot = members[archetype][int(rng.integers(len(members[archetype])))]
                seats = tuple((f"{bot.player_id}-c{j}", bot) for j in range(6))
            elif kind == "mirror_6":
                archetype = ARCHETYPES[k]
                pick = rng.choice(len(members[archetype]), 6, replace=False)
                seats = tuple(
                    (members[archetype][i].player_id, members[archetype][i]) for i in pick
                )
            else:
                pick = rng.choice(len(population), TABLE_SIZE[kind], replace=False)
                seats = tuple((population[i].player_id, population[i]) for i in pick)
            chunks = _chunks([seed, TAG_CHUNKS, q, k], cfg["deals"], cfg["chunk_deals"])
            label = None if archetype is None else archetype.value
            out.append(Lineup(kind, replicate, k, label, seats, chunks))
    return out


def plan(master_seed: int, cfg: dict[str, Any]) -> tuple[list[int], Lineup, list[Lineup]]:
    """Replicate seeds, the calling check and every replicate's lineups."""
    _validate(cfg)
    seeds = derive_seeds(master_seed, cfg["replicates"])
    calling = Lineup(
        CALLING,
        -1,
        0,
        None,
        tuple((f"call{j}", None) for j in range(TABLE_SIZE[CALLING])),
        _chunks([master_seed, TAG_CALLING], cfg["calling_deals"], cfg["chunk_deals"]),
    )
    lineups = [line for r, s in enumerate(seeds) for line in replicate_lineups(r, s, cfg)]
    return seeds, calling, lineups


def chunk_task(
    seats: tuple[tuple[str, PlayerSpec | None], ...], n_deals: int, seed: int, equity_samples: int
) -> np.ndarray:
    """One duplicate match: per-hand net chips, shape (n tables, n_deals, n players), players in
    lineup order. A top-level function: worker processes (Windows spawn) import it by name."""
    from poker_lab.engine import CallingAgent
    from poker_lab.engine.duplicate import duplicate_match
    from poker_lab.schema import TableConfig

    n = len(seats)
    table = TableConfig(f"dup{n}", n, SMALL_BLIND, BIG_BLIND)
    players = [
        (
            pid,
            CallingAgent
            if spec is None
            else partial(spec.make_agent, equity_samples=equity_samples),
        )
        for pid, spec in seats
    ]
    match = duplicate_match(table, players, n_deals, seed, starting_stack=STACK_BB * BIG_BLIND)
    out = np.empty((match.n_tables, n_deals, n), dtype=np.int32)
    for j, (pid, _) in enumerate(seats):
        grouped = np.array([r.net_chips for r in match.grouped_results(pid)], dtype=np.int32)
        out[:, :, j] = grouped.reshape(n_deals, match.n_tables).T  # deal-major -> (tables, deals)
    return out


def _task(args: tuple) -> np.ndarray:
    return chunk_task(*args)


def tasks(lineups: Sequence[Lineup], equity_samples: int) -> list[tuple]:
    return [
        (line.seats, deals, seed, equity_samples) for line in lineups for seed, deals in line.chunks
    ]


def _results(task_list: list[tuple], workers: int) -> Iterator[np.ndarray]:
    """Task results in task order, in this process (``workers`` <= 1) or in a pool."""
    if workers <= 1:
        yield from map(_task, task_list)
        return
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        yield from pool.map(_task, task_list)
    finally:  # an error in the parent drops the tasks not started yet
        pool.shutdown(wait=True, cancel_futures=True)


# ---- the metric ---------------------------------------------------------------------------


def player_stats(x: np.ndarray, half_width: float) -> dict[str, Any]:
    """The metric for one player: ``x`` holds the results in big blinds, shape (n tables,
    D deals). See the module docstring."""
    x = np.asarray(x, dtype=float)
    n, _ = x.shape
    sigma2 = float(x.var(ddof=1))
    s_u2 = float(x.mean(axis=0).var(ddof=1))
    z = x - x.mean()
    s0 = float((z * z).sum())
    offsets = [
        float((z * np.roll(z, -k, axis=0)).sum() / s0) if s0 > 0 else math.nan for k in range(1, n)
    ]
    sigma, s_u = math.sqrt(sigma2), math.sqrt(s_u2)
    flat = s_u < ZERO_SD
    h = half_width / 100
    return {
        "sigma": sigma,
        "s_u": s_u,
        "ratio": RATIO_CAP if flat else min(sigma2 / (n * s_u2), RATIO_CAP),
        "no_variance_left": flat,
        "rho": float(np.mean(offsets)),
        "rho_by_offset": offsets[: n // 2],
        "hands_needed": {
            "naive": math.ceil((1.96 * sigma / h) ** 2),
            "duplicate": 0 if flat else math.ceil(n * (1.96 * s_u / h) ** 2),
        },
    }


def winrates(chips: np.ndarray, player_id: str, confidence: float = 0.95) -> dict[str, Any]:
    """Win rate in bb/100 with the naive-t interval over hands (for display: it assumes
    independent hands) and the duplicate-t interval over per-deal means, from the library's
    evaluators. ``chips``: shape (n tables, D deals)."""
    n = chips.shape[0]
    results = [HandResult("", player_id, c, BIG_BLIND) for c in chips.T.ravel().tolist()]
    naive = WinRateEvaluator(confidence).evaluate(results)
    dup = DuplicateDealEvaluator(confidence, group_size=n).evaluate(results)
    return {
        "bb_per_100": naive.bb_per_100,
        "naive": [naive.lower, naive.upper],
        "duplicate": [dup.lower, dup.upper],
    }


def lineup_row(line: Lineup, chips: np.ndarray, half_width: float) -> dict[str, Any]:
    """Per-player statistics of a lineup; ``chips`` has shape (n tables, D deals, n players)."""
    players = []
    for j, (pid, spec) in enumerate(line.seats):
        x = chips[:, :, j]
        players.append(
            {
                "player_id": pid,
                "bot_id": spec.player_id,
                "archetype": spec.archetype.value,
                "params": spec.params.to_dict(),
                **player_stats(x / BIG_BLIND, half_width),
                "winrate": winrates(x, pid),
            }
        )
    return {
        "kind": line.kind,
        "replicate": line.replicate,
        "index": line.index,
        "n": line.n,
        "archetype": line.archetype,
        "deals": line.deals,
        "seating": [pid for pid, _ in line.seats],
        "players": players,
    }


def calling_check(line: Lineup, chips: np.ndarray) -> dict[str, Any]:
    """Six always-call agents: every player's per-deal mean must be exactly 0 chips."""
    sums = chips.sum(axis=0)  # (deals, players): exact integers
    sigma = [float(chips[:, :, j].std(ddof=1)) / BIG_BLIND for j in range(line.n)]
    return {
        "deals": line.deals,
        "hands_per_player": int(chips.shape[0] * chips.shape[1]),
        "max_abs_deal_mean_chips": float(np.abs(sums).max()) / line.n,
        "all_zero": bool((sums == 0).all()),
        "sigma_median": float(np.median(sigma)),
    }


# ---- aggregation --------------------------------------------------------------------------


def _quantiles(x) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    q = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    return dict(zip(("min", "q25", "median", "q75", "max"), map(float, q), strict=True))


def median_ci(groups: Sequence[Sequence[float]], n_boot: int, entropy: Sequence[int]) -> dict:
    """95% percentile bootstrap CI of the median over all values, resampling the groups
    (lineups) with replacement; a group's values stay together. Empty groups (lineups without
    the archetype asked for) are left out. NaN bounds below ``MIN_CI_LINEUPS`` lineups."""
    arrays = [np.asarray(g, dtype=float) for g in groups if len(g)]
    if len(arrays) < MIN_CI_LINEUPS:
        return {"lower": math.nan, "upper": math.nan, "lineups": len(arrays)}

    def stat(idx: np.ndarray) -> float:
        x = np.concatenate([arrays[i] for i in idx])
        return float(np.median(x)) if x.size else math.nan

    rng = np.random.default_rng(np.random.SeedSequence(list(entropy)))
    iv = bootstrap_ci(np.arange(len(arrays)), stat, n_boot=n_boot, rng=rng)
    return {"lower": iv.lower, "upper": iv.upper, "lineups": len(arrays)}


def _summary(players: list[dict], groups: list[list[float]], n: int, n_boot: int, entropy) -> dict:
    ratios = [p["ratio"] for p in players]
    return {
        "players": len(players),
        "ratio": _quantiles(ratios),
        "ratio_ci": median_ci(groups, n_boot, entropy),
        "share_below_1": float(np.mean(np.array(ratios) < 1.0)),
        "no_variance_left": int(sum(p["no_variance_left"] for p in players)),
        "rho": _quantiles([p["rho"] for p in players]),
        "rho_by_offset": [
            _quantiles([p["rho_by_offset"][k] for p in players]) for k in range(n // 2)
        ],
        "sigma": _quantiles([p["sigma"] for p in players]),
        "s_u": _quantiles([p["s_u"] for p in players]),
        "hands_needed": {
            name: float(np.median([p["hands_needed"][name] for p in players]))
            for name in ("naive", "duplicate")
        },
    }


def aggregate(rows: list[dict[str, Any]], n_boot: int, master_seed: int) -> dict[str, Any]:
    by_kind: dict[str, Any] = {}
    for q, kind in enumerate(KINDS):
        lineups = [row for row in rows if row["kind"] == kind]
        if not lineups:
            continue
        n = TABLE_SIZE[kind]
        players = [p for row in lineups for p in row["players"]]
        groups = [[p["ratio"] for p in row["players"]] for row in lineups]
        entry = {"n": n, "lineups": len(lineups)}
        entry.update(_summary(players, groups, n, n_boot, [master_seed, TAG_BOOTSTRAP, q]))
        entry["by_archetype"] = {}
        for a, arch in enumerate(ARCHETYPES, start=1):
            mine = [p for p in players if p["archetype"] == arch.value]
            if not mine:
                continue
            groups = [
                [p["ratio"] for p in row["players"] if p["archetype"] == arch.value]
                for row in lineups
            ]
            entry["by_archetype"][arch.value] = _summary(
                mine, groups, n, n_boot, [master_seed, TAG_BOOTSTRAP, q, a]
            )
        by_kind[kind] = entry
    mixed_by_n = {
        str(e["n"]): {
            "kind": kind,
            "ratio_median": e["ratio"]["median"],
            "ratio_ci": e["ratio_ci"],
            "rho_median": e["rho"]["median"],
            "sigma_median": e["sigma"]["median"],
        }
        for kind, e in sorted(by_kind.items(), key=lambda kv: kv[1]["n"])
        if kind.startswith("mixed_")
    }
    return {"by_kind": by_kind, "mixed_by_n": mixed_by_n}


def bridge_reference() -> dict[str, Any]:
    """Step 28's heads-up push/fold duplicate dealing (P3.3's bridge): per stack, the median
    over the matchups against the bots (both heroes) of naive / duplicate hands needed."""
    matchups = [
        m
        for m in json.loads(BRIDGE.read_text(encoding="utf-8"))["matchups"]
        if m["opponent"] != "nash" and m["estimates"]["duplicate"]["hands_needed"] > 0
    ]

    def summary(ms):
        naive = [m["estimates"]["naive"]["hands_needed"] for m in ms]
        dup = [m["estimates"]["duplicate"]["hands_needed"] for m in ms]
        return {
            "matchups": len(ms),
            "naive_median": float(np.median(naive)),
            "duplicate_median": float(np.median(dup)),
            "ratio_median": float(np.median(np.divide(naive, dup))),
        }

    return {
        "source": BRIDGE.relative_to(REPO_ROOT).as_posix(),
        "all": summary(matchups),
        "by_stack": {
            f"{s:g}": summary([m for m in matchups if m["stack"] == s])
            for s in sorted({m["stack"] for m in matchups})
        },
    }


# ---- the run ------------------------------------------------------------------------------


def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase2" / "duplicate"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()
    seeds, calling, lineups = plan(seed, cfg)
    workers = workers or cfg.get("workers") or 1
    rows = []
    with closing(_results(tasks([calling, *lineups], cfg["equity_samples"]), workers)) as res:

        def chips_of(line: Lineup) -> np.ndarray:
            return np.concatenate([next(res) for _ in line.chunks], axis=1)

        check = calling_check(calling, chips_of(calling))
        if not check["all_zero"]:
            raise AssertionError(f"calling check: a per-deal mean is not 0 chips: {check}")
        for k, line in enumerate(lineups):
            rows.append(lineup_row(line, chips_of(line), cfg["half_width"]))
            if k + 1 == len(lineups) or lineups[k + 1].replicate != line.replicate:
                print(f"replicate {line.replicate + 1}/{len(seeds)} done", flush=True)
    out = {
        "meta": meta,
        "replicate_seeds": seeds,
        "calling_check": check,
        "per_lineup": rows,
        "aggregates": aggregate(rows, cfg["bootstrap"], seed),
        "bridge": bridge_reference(),
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    out["predictions"] = predictions(out)
    save_json(out, out_dir / "duplicate.json")
    (out_dir / "duplicate_summary.md").write_text(
        markdown_summary(out), encoding="utf-8", newline="\n"
    )
    make_figures(out, out_dir)
    return out


def _lookup(r: dict[str, Any], path: Sequence[Any]) -> Any:
    for key in path:
        r = r[key]
    return r


def _number(value: Any) -> float | None:
    """A prediction's quantity, or None: missing, NaN, or a path that stops at a dict or list."""
    if isinstance(value, bool) or not isinstance(value, int | float) or math.isnan(value):
        return None
    return value


def _holds(values: Sequence[float | None], test: tuple[str, Any]) -> bool:
    op, threshold = test
    if not values or any(v is None for v in values):
        return False
    if op == ">=":
        return all(v >= threshold for v in values)
    if op == "<=":
        return all(v <= threshold for v in values)
    if op == "between":
        return all(threshold[0] <= v <= threshold[1] for v in values)
    if op == "decreasing":
        return all(a > b for a, b in pairwise(values))
    if op == "first_lowest":
        return all(values[0] < v for v in values[1:])
    if op == "range_at_most":
        return max(values) - min(values) <= threshold
    raise ValueError(f"unknown test {op!r}")


def predictions(r: dict[str, Any], table: dict[str, dict] | None = None) -> dict[str, Any]:
    """The pre-registered predictions in ``PREDICTIONS`` (or ``table``), evaluated. A quantity
    that is not a number in the results fails its prediction and is stored as None."""
    out = {}
    for key, p in (PREDICTIONS if table is None else table).items():
        values = {}
        for path in p["values"]:
            try:
                value = _number(_lookup(r, path))
            except (KeyError, IndexError, TypeError):
                value = None
            values[".".join(map(str, path))] = value
        out[key] = {
            "claim": p["claim"],
            "values": values,
            "test": p["test"][0],
            "threshold": p["test"][1],
            "passed": _holds(list(values.values()), p["test"]),
        }
    return out


# ---- reporting ----------------------------------------------------------------------------

LABELS = {
    "clones_6": "Six copies of one bot",
    "mirror_6": "Six bots of one archetype",
    "mixed_6": "Six bots, random mix",
    "mixed_3": "Three bots, random mix",
    "mixed_2": "Two bots, random mix",
}
ARCH_LABELS = {
    "nit": "Nit",
    "tag": "TAG",
    "lag": "LAG",
    "calling_station": "Calling station",
    "maniac": "Maniac",
}


def _fmt_ratio(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return "no variance" if x >= RATIO_CAP else f"{x:.2f}x" if x < 10 else f"{x:,.0f}x"


def _ci(ci: dict[str, float]) -> str:
    return f"[{_fmt_ratio(ci['lower'])}, {_fmt_ratio(ci['upper'])}]"


def _offsets(e: dict[str, Any]) -> str:
    return ", ".join(f"{q['median']:+.3f}" for q in e["rho_by_offset"])


def markdown_summary(r: dict[str, Any]) -> str:
    cfg = r["meta"]["config"]
    agg = r["aggregates"]
    hw = cfg["half_width"]
    lines = [
        "# P2.11b: duplicate dealing for n players, no-limit hold'em",
        "",
        f"Seed {r['meta']['seed']}, commit {(r['meta']['git']['commit'] or '?')[:7]}.",
        "",
        f"{cfg['replicates']} replicates of {cfg['population_per_archetype']} bots per archetype; "
        f"{cfg['deals']:,} deals per lineup, each played at n tables with the seats rotated; "
        f"blinds {SMALL_BLIND}/{BIG_BLIND}, {STACK_BB} bb stacks, {cfg['equity_samples']} "
        "equity samples. Ratio = sigma^2 / (n s_u^2): how many times fewer hands duplicate "
        "dealing needs than independent dealing for the same interval (above 1: it pays). "
        "rho: mean correlation of a player's results at two tables of one deal.",
        "",
        "## By lineup kind (players pooled over lineups and replicates)",
        "",
        "| Lineup kind | Lineups played | Players | Ratio: median [IQR] | 95% CI of the median | "
        "Below 1 | rho: median | rho by offset k = 1.. | SD per hand (bb) | "
        f"Median hands for +/-{hw} bb/100: naive | duplicate |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for kind, e in agg["by_kind"].items():
        q = e["ratio"]
        lines.append(
            f"| {LABELS[kind]} | {e['lineups']} | {e['players']} | {_fmt_ratio(q['median'])} "
            f"[{_fmt_ratio(q['q25'])}, {_fmt_ratio(q['q75'])}] | {_ci(e['ratio_ci'])} | "
            f"{e['share_below_1']:.0%} | {e['rho']['median']:+.3f} | {_offsets(e)} | "
            f"{e['sigma']['median']:.2f} | {e['hands_needed']['naive']:,.0f} | "
            f"{e['hands_needed']['duplicate']:,.0f} |"
        )
    lines += [
        "",
        "## By lineup kind and the player's archetype",
        "",
        "| Lineup kind | Archetype | Players | Ratio: median [IQR] | 95% CI of the median | "
        "rho: median | rho by offset | SD per hand (bb) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for kind, e in agg["by_kind"].items():
        for arch, a in e["by_archetype"].items():
            q = a["ratio"]
            lines.append(
                f"| {LABELS[kind]} | {ARCH_LABELS[arch]} | {a['players']} | "
                f"{_fmt_ratio(q['median'])} [{_fmt_ratio(q['q25'])}, {_fmt_ratio(q['q75'])}] | "
                f"{_ci(a['ratio_ci'])} | {a['rho']['median']:+.3f} | {_offsets(a)} | "
                f"{a['sigma']['median']:.2f} |"
            )
    if agg["mixed_by_n"]:
        lines += [
            "",
            "## Mixed lineups by the number of players",
            "",
            "| Players | Median ratio | 95% CI | Median rho | Median SD per hand (bb) |",
            "|---|---|---|---|---|",
        ]
        for n, m in agg["mixed_by_n"].items():
            lines.append(
                f"| {n} | {_fmt_ratio(m['ratio_median'])} | {_ci(m['ratio_ci'])} | "
                f"{m['rho_median']:+.3f} | {m['sigma_median']:.2f} |"
            )
    c = r["calling_check"]
    lines += [
        "",
        f"Calling check: six always-call agents, {c['deals']:,} deals: largest |per-deal mean| "
        f"{c['max_abs_deal_mean_chips']:g} chips (per-hand SD {c['sigma_median']:.2f} bb).",
    ]
    per = ", ".join(
        f"{v['ratio_median']:.2f}x at {s} bb" for s, v in r["bridge"]["by_stack"].items()
    )
    lines.append(
        "Heads-up push/fold reference (Step 28, P3.3's bridge): median over the matchups against "
        f"the bots of naive / duplicate hands needed: {per}."
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


RATIO_TICKS = (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000)


def _ratio_axis(ax, values) -> None:
    """A log axis with plain tick labels (0.5x, 1x, 2x, ...) over the values' range."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    lo, hi = float(np.min(values)), float(np.max(values))
    ticks = [t for t in RATIO_TICKS if lo / 1.2 <= t <= hi * 1.2] or [1]
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def make_figures(r: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import legend, line, new_figure, save_both, style_axes, title

    agg = r["aggregates"]["by_kind"]
    kinds = [k for k in KINDS if k in agg]
    arch_index = {a.value: i for i, a in enumerate(ARCHETYPES)}

    def ratios(theme):
        fig, ax = new_figure(theme, 1, 1, figsize=(9, 0.75 * len(kinds) + 1.9))
        style_axes(ax, theme, grid_axis="x")
        s = theme["series"]
        seen, every = set(), [1.0]
        for k, kind in enumerate(kinds):
            players = [p for row in r["per_lineup"] if row["kind"] == kind for p in row["players"]]
            x = np.clip([p["ratio"] for p in players], 1e-2, 1e3)
            every += x.tolist()
            jitter = (np.arange(len(x)) % 9 - 4) * 0.05
            for a, i in arch_index.items():
                mask = np.array([p["archetype"] == a for p in players])
                if mask.any():
                    label = None if a in seen else ARCH_LABELS[a]
                    seen.add(a)
                    ax.scatter(x[mask], k + jitter[mask], s=10, color=s[i], linewidths=0,
                               alpha=0.8, label=label)  # fmt: skip
            q = agg[kind]["ratio"]
            ax.plot([q["q25"], q["q75"]], [k, k], color=theme["ink"], linewidth=2.5,
                    solid_capstyle="round")  # fmt: skip
            ax.plot(q["median"], k, "|", color=theme["ink"], markersize=16, markeredgewidth=2.5)
        ax.axvline(1.0, color=theme["axis"], linewidth=0.8)
        _ratio_axis(ax, every)
        ax.set_yticks(np.arange(len(kinds)))
        ax.set_yticklabels([LABELS[k] for k in kinds], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Times fewer hands than independent dealing (log; right of 1: it pays)")
        legend(ax, theme, loc="lower right", markerscale=1.6)
        title(
            fig,
            theme,
            "Duplicate dealing in no-limit hold'em: how many times fewer hands",
            "Dots are players, colored by archetype; bars the middle half, ticks the median",
            top=0.9 if len(kinds) > 2 else 0.82,
        )
        return fig

    six = [k for k in kinds if TABLE_SIZE[k] == 6]

    def offsets(theme):
        fig, ax = new_figure(theme, 1, 1, figsize=(7.5, 4.2))
        style_axes(ax, theme)
        s = theme["series"]
        ks = np.arange(1, 4)
        for i, kind in enumerate(six):
            e = agg[kind]
            med = [q["median"] for q in e["rho_by_offset"]]
            lo = [q["q25"] for q in e["rho_by_offset"]]
            hi = [q["q75"] for q in e["rho_by_offset"]]
            ax.fill_between(ks, lo, hi, color=s[i], alpha=0.1, linewidth=0)
            label = f"{LABELS[kind]} (rho: median {e['rho']['median']:+.2f})"
            line(ax, theme, ks, med, s[i], label)
        ax.axhline(0.0, color=theme["axis"], linewidth=0.8)
        ax.set_xticks(ks)
        ax.set_xticklabels(["1 (next seat)", "2", "3 (opposite)"])
        ax.set_xlabel("Table offset k: the player's seat k seats further round")
        ax.set_ylabel("Correlation of results, same deal")
        legend(ax, theme, loc="best")
        title(
            fig,
            theme,
            "How a player's results at two tables of one deal move together",
            "Median over players, middle half shaded; below 0 the tables cancel card luck",
            top=0.86,
        )
        return fig

    mixed = r["aggregates"]["mixed_by_n"]

    def by_n(theme):
        fig, ax = new_figure(theme, 1, 1, figsize=(6.5, 4.2))
        style_axes(ax, theme)
        s = theme["series"]
        ns = [int(n) for n in mixed]
        med = [mixed[n]["ratio_median"] for n in mixed]
        lo = [mixed[n]["ratio_ci"]["lower"] for n in mixed]
        hi = [mixed[n]["ratio_ci"]["upper"] for n in mixed]
        err = np.array([np.subtract(med, lo), np.subtract(hi, med)])
        # a percentile CI need not contain the median, and matplotlib refuses a negative error
        ax.errorbar(ns, med, yerr=np.clip(np.nan_to_num(err), 0, None), fmt="none", ecolor=s[0],
                    elinewidth=1.5, capsize=4)  # fmt: skip
        line(ax, theme, ns, med, s[0], "Random mix: median over players, 95% CI")
        ax.plot(2, r["bridge"]["all"]["ratio_median"], "D", color=s[1], markersize=7,
                markeredgecolor=theme["surface"],
                label="Heads-up push/fold (Step 28), median over matchups")  # fmt: skip
        ax.axhline(1.0, color=theme["axis"], linewidth=0.8)
        ax.set_xticks(ns)
        ax.set_xlabel("Players at the table (n)")
        ax.set_ylabel("Times fewer hands than independent dealing")
        legend(ax, theme, loc="best")
        title(
            fig,
            theme,
            "Duplicate dealing against the number of players",
            "Mixed lineups; a deal is played at n tables, so it costs n hands",
            top=0.86,
        )
        return fig

    save_both(ratios, out_dir, "duplicate_ratio")
    if six:
        save_both(offsets, out_dir, "duplicate_offsets")
    if mixed:
        save_both(by_n, out_dir, "duplicate_players")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--quick", action="store_true", help="use configs/phase2_quick.yaml")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    path = CONFIG.with_name("phase2_quick.yaml") if args.quick else args.config
    cfg = load_config(path)["duplicate"]
    r = run(args.seed, cfg=cfg, workers=args.workers, out_dir=args.out)
    print(markdown_summary(r))


if __name__ == "__main__":
    main()

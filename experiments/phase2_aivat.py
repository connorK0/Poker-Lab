"""P2.11: AIVAT, control variates that cut the variance of win rates, measured exactly.

    python -m experiments.phase2_aivat --seed 2026 [--quick]
    python -m experiments.run_phase2 --seed 2026 --only aivat

Pre-registered plan (written after development on seeds 98 and 99, before the real run):

**Question.** A live evaluation (Phase 5) cannot deal duplicate and will not reach tens of
thousands of hands (P3.3). How much variance do AIVAT's control variates remove (Burch,
Schmid, Moravcik, Morrill and Bowling 2018; ``poker_lab.evaluation.aivat``)? How good must the
value function be? And how much is left when the evaluator sees only what the hero saw, so
that folded cards stay hidden, as at a home game?

**Exact, not sampled.** In Leduc and in the push/fold game every estimator is a function of the
terminal history, so its mean and variance are sums over the game tree; duplicate dealing's
variance comes from per-deal moments (``deal_moments``). The hero's seat is a fair coin the
evaluator sees, so every control-variate estimator gets a term for it. Metrics: the *ratio*,
variance per hand of the plain result over variance per hand of the estimator (how many times
fewer hands the estimator needs for the same interval; a duplicate pair costs two hands); the
exact bias; the hands needed for a 95% interval of +/-0.1 chips (Leduc) or +/-0.05 bb
(push/fold).

**Estimators** (``CV_ESTIMATORS``):

* ``chips``: the plain result; ``duplicate``: every deal played in both seats.
* ``mivat``: terms at chance steps, values of full histories, no averaging (MIVAT, the
  chance-only method Burch et al. compare with).
* ``mivat_io``: chance terms with imaginary observations over the hero's cards.
* ``aivat``: chance and hero terms with imaginary observations. It needs the opponent's cards
  after every hand. ``aivat_duplicate``: the same on duplicate deals.
* ``aivat_all``: the opponent's steps too, as if its strategy were known (a reference).
* ``hero_view``: only the hero's information. Terms only at steps whose probabilities do not
  depend on the opponent's hidden card (the hero's own deal and decisions); unbiased.
* ``hero_view_approx``: ``hero_view`` plus terms at the public card, with its probabilities set
  as if the opponent's card were uniform over the cards the hero has not seen. Biased; the
  exact bias is reported.

**Value functions.** Self-play values of a profile: the exact equilibrium (LP), uniform play,
and CFR+ average strategies after 1, 3, 10, 30, 100, 300 and 1,000 iterations; and the oracle,
the true values of this hero against this opponent (unavailable in practice: the ceiling). The
hero's-view estimators weigh the opponent's possible cards by that profile's reach. Results
are for the equilibrium values unless stated.

**Leduc.** 10 bots per archetype (Phase 2's five archetypes; one population from the seed).
Heroes: the equilibrium, and an exploiting hero, the restricted Nash response (p = 0.9) to the
bot's true policy. Also the equilibrium in self-play, the setting of Burch et al.'s Table 1.

**Sampled check.** For the first 2 bots of each archetype and both heroes: 20,000 duplicate
deals played hand by hand, each hand recorded as its edge labels and replayed to its terminal
(``GameTree.follow``), every estimator scored on those hands. ``z`` = (sample mean - the
estimator's exact mean) / SE: it tests the exact moments (a biased estimator's exact mean
includes its bias).

**Push/fold.** P3.3's 33 matchups (5, 10 and 20 bb; Nash and best-response heroes; the five
archetypes at their mean parameters, and Nash) in the abstract game, whose plain result is
P3.3's all-in EV. Cross-check: P3.3 measured all-in EV and duplicate + all-in EV on 100,000
real deals per matchup (``experiments/results/phase3/bridge/bridge.json``).

**Seen in development** (not predictions; self-play does not depend on the seed):

* Leduc self-play, SD per hand: chips 3.497 (the paper: 3.513), chance terms 1.463 (2.327),
  chance terms + imaginary observations 1.933 (1.928), AIVAT 0 and AIVAT with the opponent
  known 0 (0.00643 and 0.00377: the paper's values were approximate), hero's view 3.400,
  approximate hero's view 2.449 (bias +0.006).
* Leduc bots, seeds 98 and 99 (medians): AIVAT 26x (equilibrium hero) and 4.4-4.5x (exploiting
  hero); oracle values 44-45x and 8.7-9.4x; duplicate on top of AIVAT +4%; hero's view
  1.04-1.05x; approximate hero's view 1.86-1.88x and 1.55-1.57x, with a largest bias of
  0.017-0.019 and 0.055-0.059 chips/hand (0.20-0.24 with oracle values); uniform-play values
  8.9x; CFR+ from 30 iterations 28-29x against 26x for the LP; the sampled check passed.
* Push/fold at 3 bb only (the quick configuration; not a stack of this run), medians over the
  bots: AIVAT 13x over all-in EV (Nash hero) and 7.5x (best-response hero); hero's view 1.1x.
  Push/fold at 5, 10 and 20 bb was not computed before this plan.
* Changes during development: the sampled ``z`` first compared every estimator with the true
  value, which flags the biased one twice; an estimator with no variance left is shown as such.

**Predictions** (thresholds from development; evaluated by ``predictions``):

* A1: every estimator but the approximate hero's view is unbiased (|bias| < 1e-9), in every
  matchup and with every value function.
* A2: sampled hands agree with the exact moments: per estimator, no |z| > 4, mean-z t-test
  p >= 0.01, median sampled/exact SD in [0.97, 1.03].
* B1: AIVAT, equilibrium hero: median ratio >= 15x and every bot >= 5x.
* B2: AIVAT, exploiting hero: median ratio between 3x and 8x.
* B3: oracle values beat equilibrium values for the equilibrium hero by >= 1.4x (ratio of the
  median ratios).
* C1: hero's view (folded cards hidden): median ratio < 1.15x for both heroes.
* C2: approximate hero's view: median ratio in [1.4x, 2.5x] for both heroes, and its bias
  reaches >= 0.02 chips/hand against some bot (exploiting hero).
* D1: uniform-play values still give AIVAT a median ratio >= 4x (equilibrium hero).
* D2: from 30 CFR+ iterations on, AIVAT's median ratio is within 20% of the equilibrium
  values' (equilibrium hero).
* E1: duplicate dealing adds < 25% on top of AIVAT (equilibrium hero).
* F1: push/fold: exact SDs of all-in EV and duplicate + all-in EV within 3% of P3.3's sampled
  SDs in every matchup.
* F2: push/fold, Nash hero: AIVAT needs >= 4x fewer hands than all-in EV at every stack
  (median over bots).
* F3: push/fold: hero's view gains < 1.3x over all-in EV at every stack, both heroes.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from experiments.common import REPO_ROOT, RESULTS_ROOT, load_config, run_metadata, save_json
from poker_lab.evaluation.aivat import (
    class_actor,
    control_variates,
    deal_moments,
    known_transitions,
    observer_classes,
    terminal_moments,
)
from poker_lab.games import (
    CHANCE,
    TERMINAL,
    LeducHoldem,
    TabularPolicy,
    compile_tree,
    exploitability,
    head_to_head,
)
from poker_lab.games.leduc_bots import BotPolicyBuilder, LeducBot, sample_population
from poker_lab.games.push_fold import PushFoldGame, push_fold_tree, threshold_policy
from poker_lab.solvers import CFRSolver, RobustResponder, solve_lp

CONFIG = REPO_ROOT / "experiments" / "configs" / "phase2.yaml"
BRIDGE = RESULTS_ROOT / "phase3" / "bridge" / "bridge.json"

# control-variate estimators: (observer, steps that get a term, imaginary observations)
CV_ESTIMATORS: dict[str, tuple[str, Any, bool]] = {
    "mivat": ("none", ("chance",), False),
    "mivat_io": ("full", ("chance",), True),
    "aivat": ("full", ("chance", "hero"), True),
    "aivat_all": ("full", ("chance", "hero", "opponent"), True),
    "hero_view": ("view", "known", False),
    "hero_view_approx": ("view", ("chance", "hero"), False),
}
DUPLICATE = {"duplicate": "chips", "aivat_duplicate": "aivat"}  # paired version: base estimator
ESTIMATORS = (
    "chips",
    "duplicate",
    "mivat",
    "mivat_io",
    "aivat",
    "aivat_duplicate",
    "aivat_all",
    "hero_view",
    "hero_view_approx",
)
LABELS = {
    "chips": "Chips (the plain result)",
    "duplicate": "Duplicate dealing",
    "mivat": "Chance terms only (MIVAT)",
    "mivat_io": "Chance terms + imaginary observations",
    "aivat": "AIVAT (every hand's cards shown)",
    "aivat_duplicate": "AIVAT + duplicate dealing",
    "aivat_all": "AIVAT, opponent's strategy known too",
    "hero_view": "Hero's view (folded cards stay hidden)",
    "hero_view_approx": "Hero's view + public-card terms (biased)",
}
# Burch et al. (2018), Table 1: Leduc self-play, standard deviation per hand (chips)
PAPER_SELFPLAY = {
    "chips": 3.513,
    "mivat": 2.327,
    "mivat_io": 1.928,
    "aivat": 0.00643,
    "aivat_all": 0.00377,
}
MAIN_VALUES = ("equilibrium", "oracle", "uniform")
CURVE_ESTIMATORS = ("aivat", "hero_view", "hero_view_approx")
ZERO_SD = 1e-9


# ---- exact moments of every estimator ------------------------------------------------------


def observers(game, tree) -> dict[int, dict[str, Any]]:
    """Per hero seat: information classes and actors for the three observers, and which steps
    the hero's own view can correct exactly."""
    out = {}
    for s in (0, 1):
        full = observer_classes(game, tree, hidden=s)
        view = observer_classes(game, tree, hidden=1 - s)
        out[s] = {
            "full": (full, class_actor(tree, full)),
            "view": (view, class_actor(tree, view)),
            "none": (np.arange(tree.num_nodes), tree.player.copy()),
            "known": known_transitions(tree, view, s),
        }
    return out


def _stats(mean: float, var: float, exact: float, half_width: float) -> dict[str, float]:
    var = max(var, 0.0)
    sd = math.sqrt(var)
    return {
        "mean": mean,
        "bias": mean - exact,
        "sd": sd,
        "variance": var,
        "hands": math.ceil((1.96 * sd / half_width) ** 2),
    }


def single_stats(tree, edges, xs, exact, half_width) -> dict[str, float]:
    """One hand per unit, the hero's seat a fair coin."""
    (m0, q0), (m1, q1) = (terminal_moments(tree, edges[s], xs[s]) for s in (0, 1))
    mean = 0.5 * (m0 + m1)
    return _stats(mean, 0.5 * (q0 + q1) - mean**2, exact, half_width)


def duplicate_stats(tree, edges, xs, exact, half_width) -> dict[str, float]:
    """Every deal played in both seats; variance per hand is twice the pair mean's."""
    p, m0, q0 = deal_moments(tree, edges[0], xs[0])
    p1, m1, q1 = deal_moments(tree, edges[1], xs[1])
    if not np.allclose(p, p1):
        raise ValueError("the two seats' deals do not line up")
    pair = 0.5 * (m0 + m1)
    within = 0.25 * ((q0 - m0**2) + (q1 - m1**2))
    mean = float(p @ pair)
    var_pair = float(p @ within + p @ pair**2 - mean**2)
    return _stats(mean, 2.0 * var_pair, exact, half_width)


def estimator_values(tree, obs, hero, opp, model, which) -> dict[str, dict[int, np.ndarray]]:
    """Per estimator and hero seat, the estimate at every terminal (0 elsewhere).

    ``model`` gives the value function: the self-play values of that profile, and for the
    hero's view its beliefs about the opponent's cards (``None``: the true profile, an oracle).
    The seat is a fair coin the evaluator sees, so it gets a term too.
    """
    term = tree.player == TERMINAL
    out: dict[str, dict[int, np.ndarray]] = {name: {} for name in which}
    roots: dict[str, dict[int, float]] = {name: {} for name in which}
    for s in (0, 1):
        sign = 1.0 if s == 0 else -1.0
        edge = tree.edge_probabilities(hero.combine(opp, s).probs)
        if model is None:
            value, belief = sign * tree.backward(edge), tree.reach(edge)
        else:
            value = sign * tree.backward(tree.edge_probabilities(model.probs))
            belief = tree.reach(tree.edge_probabilities(hero.combine(model, s).probs))
        codes = {"chance": CHANCE, "hero": s, "opponent": 1 - s}
        for name in which:
            observer, steps, imaginary = CV_ESTIMATORS[name]
            classes, actor = obs[s][observer]
            if steps == "known":
                corrected = obs[s]["known"]
            else:
                corrected = np.isin(actor, [codes[a] for a in steps])
            cv = control_variates(
                tree,
                s,
                edge,
                value,
                classes,
                corrected,
                imaginary=imaginary,
                value_weight=belief if observer == "view" else None,
            )
            out[name][s] = np.where(term, cv.estimate, 0.0)
            roots[name][s] = cv.root_value
    for name in which:
        mid = 0.5 * (roots[name][0] + roots[name][1])
        for s in (0, 1):
            out[name][s] = np.where(term, out[name][s] - (roots[name][s] - mid), 0.0)
    return out


def matchup(tree, obs, hero, opp, values, exact, half_width, which) -> tuple[dict, dict]:
    """Exact statistics of every estimator; also the per-terminal values (for sampling)."""
    edges = {s: tree.edge_probabilities(hero.combine(opp, s).probs) for s in (0, 1)}
    u = {s: (1.0 if s == 0 else -1.0) * tree.utility for s in (0, 1)}
    base = {
        "chips": single_stats(tree, edges, u, exact, half_width),
        "duplicate": duplicate_stats(tree, edges, u, exact, half_width),
    }
    out: dict[str, Any] = {"base": base, "values": {}}
    xs: dict[str, Any] = {"chips": u}
    for vname, (model, names) in values.items():
        names = tuple(n for n in names if n in which)
        cv_names = tuple(n for n in names if n in CV_ESTIMATORS)
        x = estimator_values(tree, obs, hero, opp, model, cv_names)
        row = {n: single_stats(tree, edges, x[n], exact, half_width) for n in cv_names}
        if "aivat_duplicate" in names:
            row["aivat_duplicate"] = duplicate_stats(tree, edges, x["aivat"], exact, half_width)
        out["values"][vname] = row
        if vname == "equilibrium":
            xs.update(x)
    return out, xs


# ---- Leduc: the bot population -------------------------------------------------------------

_CTX: dict[str, Any] = {}


def _leduc(cfg: dict[str, Any]) -> dict[str, Any]:
    """Per-process cache: the game, its tree, the equilibrium, CFR+ checkpoints, observers."""
    if not _CTX:
        game = LeducHoldem()
        tree = compile_tree(game)
        solver = CFRSolver(tree, "cfr+")
        cfr = {}
        for t in cfg["cfr_checkpoints"]:
            solver.iterate(t - solver.iteration)
            cfr[t] = solver.average_policy()
        _CTX.update(
            game=game,
            tree=tree,
            builder=BotPolicyBuilder(tree, game),
            eq=solve_lp(tree).policy,
            uniform=TabularPolicy.uniform(tree),
            cfr=cfr,
            responder=RobustResponder(tree),
            obs=observers(game, tree),
        )
    return _CTX


def value_functions(ctx: dict[str, Any]) -> dict[str, tuple[Any, tuple[str, ...]]]:
    """Name -> (profile, estimators computed with it). ``None`` is the oracle."""
    out = {
        "equilibrium": (ctx["eq"], ESTIMATORS),
        "oracle": (None, ESTIMATORS),
        "uniform": (ctx["uniform"], ESTIMATORS),
    }
    for t, policy in ctx["cfr"].items():
        out[f"cfr+_{t}"] = (policy, CURVE_ESTIMATORS)
    return out


def _walk(tree, edge, deal, rng) -> tuple[int, list[str]]:
    """One hand with the chance outcomes fixed to ``deal`` (in dealing order)."""
    node, k, labels = 0, 0, []
    while tree.player[node] != TERMINAL:
        start, n = int(tree.child_start[node]), int(tree.num_children[node])
        if tree.player[node] == CHANCE:
            node = start + tree.labels[start : start + n].index(deal[k])
            k += 1
        else:
            p = edge[start : start + n]
            node = start + int(rng.choice(n, p=p / p.sum()))
        labels.append(tree.labels[node])
    return node, labels


def sampled_check(tree, deck, hero, opp, xs, means, n_pairs, rng) -> dict[str, Any]:
    """Play ``n_pairs`` deals at two tables (seats swapped), record each hand as its labels,
    map the labels back to the tree, and score every estimator from those hands.

    ``z`` compares each sample mean with the estimator's exact mean (``means``), which includes
    the bias of a biased estimator: the check tests the exact moments, not unbiasedness."""
    edges = {s: tree.edge_probabilities(hero.combine(opp, s).probs) for s in (0, 1)}
    first = np.empty((n_pairs, 2), dtype=np.int64)  # (seat, terminal) at the first table
    second = np.empty((n_pairs, 2), dtype=np.int64)
    for k in range(n_pairs):
        deal = [deck[i] for i in rng.permutation(len(deck))[:3]]
        for table, seat in ((first, k % 2), (second, 1 - k % 2)):
            node, labels = _walk(tree, edges[seat], deal, rng)
            if tree.follow(labels) != node:
                raise AssertionError("a recorded hand does not replay to its terminal")
            table[k] = seat, node
    out = {}
    for name in ESTIMATORS:
        base = DUPLICATE.get(name)
        if base is None and name not in xs:
            continue
        x = xs[base or name]
        one = np.array([x[s][z] for s, z in first])
        units = 0.5 * (one + np.array([x[s][z] for s, z in second])) if base else one
        sd = float(units.std(ddof=1))
        se = sd / math.sqrt(len(units))
        mean = float(units.mean())
        out[name] = {
            "mean": mean,
            "sd_per_hand": sd * (math.sqrt(2) if base else 1.0),
            "se": se,
            "z": None if sd < ZERO_SD else (mean - means[name]) / se,
        }
    return out


def bot_task(seed: int, bot: LeducBot, cfg: dict[str, Any], sample: bool) -> dict[str, Any]:
    ctx = _leduc(cfg)
    tree = ctx["tree"]
    opp = ctx["builder"].policy(bot.params)
    heroes = {
        "equilibrium": ctx["eq"],
        "exploiter": ctx["responder"].both_seats("rnr", opp, p=cfg["rnr_p"]),
    }
    record: dict[str, Any] = {
        "bot_id": bot.bot_id,
        "archetype": bot.archetype,
        "params": bot.params.as_array(),
        "heroes": {},
    }
    for h, (name, hero) in enumerate(heroes.items()):
        exact = head_to_head(hero, opp)
        stats_, xs = matchup(
            tree, ctx["obs"], hero, opp, value_functions(ctx), exact, cfg["half_width"], ESTIMATORS
        )
        entry = {"exact": exact, **stats_}
        if sample:
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(bot.bot_id[1:]), h]))
            means = {name: _get(entry, name)["mean"] for name in ESTIMATORS}
            entry["sampled"] = sampled_check(
                tree, ctx["game"].deck, hero, opp, xs, means, cfg["sampled_pairs"], rng
            )
        record["heroes"][name] = entry
    return record


def selfplay_task(cfg: dict[str, Any]) -> dict[str, Any]:
    """The equilibrium against itself: the setting of Burch et al.'s Table 1."""
    ctx = _leduc(cfg)
    eq = ctx["eq"]
    exact = head_to_head(eq, eq)
    stats_, _ = matchup(
        ctx["tree"], ctx["obs"], eq, eq, value_functions(ctx), exact, cfg["half_width"], ESTIMATORS
    )
    return {"exact": exact, **stats_}


def value_quality(cfg: dict[str, Any]) -> dict[str, Any]:
    """Exploitability of each value function's profile (how good a solver produced it)."""
    ctx = _leduc(cfg)
    policies = {"equilibrium": ctx["eq"], "uniform": ctx["uniform"]}
    policies.update({f"cfr+_{t}": p for t, p in ctx["cfr"].items()})
    return {name: max(exploitability(p), 0.0) for name, p in policies.items()}  # LP round-off


# ---- push/fold: the P3.3 matchups, exactly ---------------------------------------------------

_PF: dict[str, Any] = {}


def pushfold_task(stack: float, hero_name: str, opp_name: str, half_width: float) -> dict:
    from experiments.phase3_bridge import policies
    from poker_lab.strategy import nash_ranges

    if not _PF:  # one structure for every stack (only the called-all-in payoffs scale)
        _PF["obs"] = observers(PushFoldGame(1.0), push_fold_tree(1.0))
    hero, opp = policies(stack, hero_name, opp_name)
    tree = hero.tree
    eq = threshold_policy(tree, *nash_ranges(stack))
    exact = head_to_head(hero, opp)
    names = ("aivat", "aivat_duplicate", "hero_view")
    values = {
        "equilibrium": (eq, names),
        "oracle": (None, names),
        "uniform": (TabularPolicy.uniform(tree), names),
    }
    which = ("chips", "duplicate", *names)
    stats_, _ = matchup(tree, _PF["obs"], hero, opp, values, exact, half_width, which)
    return {"stack": stack, "hero": hero_name, "opponent": opp_name, "exact": exact, **stats_}


def pushfold_matchups(cfg: dict[str, Any]) -> list[tuple[float, str, str]]:
    return [
        (float(s), h, o)
        for s in cfg["stacks"]
        for h in ("nash", "best_response")
        for o in ("nit", "tag", "lag", "calling_station", "maniac", "nash")
        if not (h == "best_response" and o == "nash")
    ]


def bridge_crosscheck(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """P3.3 measured the all-in-EV estimators on 100,000 real deals per matchup: their sampled
    standard deviations against the exact ones (per hand). Hands needed are for +/-0.05 bb on
    both sides (P3.3's half-width and the config's)."""
    if not BRIDGE.exists():
        return []
    bridge = {
        (m["stack"], m["hero"], m["opponent"]): m["estimates"]
        for m in json.loads(BRIDGE.read_text(encoding="utf-8"))["matchups"]
    }
    out = []
    for r in rows:
        b = bridge.get((r["stack"], r["hero"], r["opponent"]))
        if b is None:
            continue
        for ours, theirs, per_unit in (
            ("chips", "allin_ev", 1),
            ("duplicate", "duplicate_allin_ev", 2),
        ):
            exact_sd = r["base"][ours]["sd"]
            sampled_sd = b[theirs]["sd_per_unit"] * math.sqrt(per_unit)
            both_zero = exact_sd < ZERO_SD and sampled_sd < ZERO_SD
            out.append(
                {
                    "stack": r["stack"],
                    "hero": r["hero"],
                    "opponent": r["opponent"],
                    "estimator": ours,
                    "exact_sd": exact_sd,
                    "sampled_sd": sampled_sd,
                    "ratio": 1.0 if both_zero else sampled_sd / exact_sd,
                    "exact_hands": r["base"][ours]["hands"],
                    "sampled_hands": b[theirs]["hands_needed"],
                }
            )
    return out


# ---- aggregation ------------------------------------------------------------------------------


RATIO_CAP = 1e9  # an estimator with no variance left (its SD below ZERO_SD) needs "no" hands


def _ratio(chips: dict[str, float], est: dict[str, float]) -> float:
    """How many times fewer hands the estimator needs than the plain result."""
    return RATIO_CAP if est["sd"] < ZERO_SD else min(chips["variance"] / est["variance"], RATIO_CAP)


def _get(entry: dict[str, Any], name: str, value: str = "equilibrium") -> dict[str, float]:
    return entry["base"][name] if name in entry["base"] else entry["values"][value][name]


def _quantiles(x) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    q = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    return dict(zip(("min", "q25", "median", "q75", "max"), map(float, q), strict=True))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for hero in ("equilibrium", "exploiter"):
        per: dict[str, Any] = {}
        for value in MAIN_VALUES:
            per[value] = {}
            for name in ESTIMATORS:
                entries = [r["heroes"][hero] for r in rows]
                chips = [e["base"]["chips"] for e in entries]
                est = [_get(e, name, value) for e in entries]
                ratios = [_ratio(c, x) for c, x in zip(chips, est, strict=True)]
                by_arch = {}
                for arch in sorted({r["archetype"] for r in rows}):
                    sub = [q for q, r in zip(ratios, rows, strict=True) if r["archetype"] == arch]
                    by_arch[arch] = float(np.median(sub))
                per[value][name] = {
                    "ratio": _quantiles(ratios),
                    "ratio_by_archetype": by_arch,
                    "max_abs_bias": float(max(abs(x["bias"]) for x in est)),
                    "median_abs_bias": float(np.median([abs(x["bias"]) for x in est])),
                    "hands": _quantiles([x["hands"] for x in est]),
                    "share_worse_than_chips": float(np.mean(np.array(ratios) < 1.0)),
                }
        out[hero] = per
    return out


def curve(rows: list[dict[str, Any]], quality: dict[str, float]) -> dict[str, Any]:
    """Median variance ratio against each value function's exploitability."""
    out: dict[str, Any] = {}
    for hero in ("equilibrium", "exploiter"):
        out[hero] = []
        for value, expl in sorted(quality.items(), key=lambda kv: -kv[1]):
            row = {"value": value, "exploitability": expl}
            for name in CURVE_ESTIMATORS:
                entries = [r["heroes"][hero] for r in rows]
                ratios = [_ratio(e["base"]["chips"], e["values"][value][name]) for e in entries]
                row[name] = _quantiles(ratios)
            out[hero].append(row)
    return out


def sampled_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ESTIMATORS:
        z, sd_ratio = [], []
        for r in rows:
            for e in r["heroes"].values():
                s = e.get("sampled", {}).get(name)
                if s is None:
                    continue
                exact = _get(e, name)
                if s["z"] is not None:
                    z.append(s["z"])
                if exact["sd"] > ZERO_SD:
                    sd_ratio.append(s["sd_per_hand"] / exact["sd"])
        if not z:
            continue
        z_arr = np.array(z)
        out[name] = {
            "n": len(z),
            "mean_z": float(z_arr.mean()),
            "max_abs_z": float(np.abs(z_arr).max()),
            "share_above_1_96": float((np.abs(z_arr) > 1.96).mean()),
            "mean_z_p": float(stats.ttest_1samp(z_arr, 0.0).pvalue) if len(z) > 1 else None,
            "sd_ratio": _quantiles(sd_ratio),
        }
    return out


def pushfold_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stack in sorted({r["stack"] for r in rows}):
        out[str(stack)] = {}
        for hero in ("nash", "best_response"):
            bots = [r for r in rows if r["stack"] == stack and r["hero"] == hero]
            bots = [r for r in bots if r["opponent"] != "nash"]
            row = {}
            for name in ("chips", "duplicate", "aivat", "aivat_duplicate", "hero_view"):
                est = [_get(r, name) for r in bots]
                row[name] = {
                    "hands_median": float(np.median([x["hands"] for x in est])),
                    "ratio_median": float(
                        np.median(
                            [_ratio(r["base"]["chips"], x) for r, x in zip(bots, est, strict=True)]
                        )
                    ),
                    "max_abs_bias": float(max(abs(x["bias"]) for x in est)),
                }
            out[str(stack)][hero] = row
    return out


# ---- the run ----------------------------------------------------------------------------------


def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase2" / "aivat"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()
    bots = sample_population(cfg["bots_per_archetype"], np.random.default_rng(seed))
    seen: dict[str, int] = {}
    tasks = []
    for bot in bots:
        sample = seen.get(bot.archetype, 0) < cfg["sampled_bots"]
        seen[bot.archetype] = seen.get(bot.archetype, 0) + 1
        tasks.append((seed, bot, cfg, sample))
    pf = cfg["pushfold"]
    games = pushfold_matchups(pf)
    with ProcessPoolExecutor(max_workers=workers or cfg.get("workers")) as pool:
        selfplay = pool.submit(selfplay_task, cfg)
        quality = pool.submit(value_quality, cfg)
        rows = list(pool.map(bot_task, *zip(*tasks, strict=True)))
        pf_tasks = [(*g, pf["half_width"]) for g in games]
        pf_rows = list(pool.map(pushfold_task, *zip(*pf_tasks, strict=True))) if games else []
        selfplay, quality = selfplay.result(), quality.result()
    results = {
        "meta": meta,
        "selfplay": selfplay,
        "paper_selfplay_sd": PAPER_SELFPLAY,
        "value_exploitability": quality,
        "bots": rows,
        "population": aggregate(rows),
        "curve": curve(rows, quality),
        "sampled": sampled_summary(rows),
        "pushfold": pf_rows,
        "pushfold_summary": pushfold_summary(pf_rows),
        "bridge_crosscheck": bridge_crosscheck(pf_rows),
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    results["predictions"] = predictions(results)
    save_json(results, out_dir / "aivat.json")
    (out_dir / "aivat_summary.md").write_text(
        markdown_summary(results), encoding="utf-8", newline="\n"
    )
    make_figures(results, out_dir)
    return results


def _all_biases(r: dict[str, Any]) -> list[float]:
    """|bias| of every unbiased estimator in every matchup and with every value function."""
    entries = [e for row in r["bots"] for e in row["heroes"].values()]
    entries += [r["selfplay"], *r["pushfold"]]
    out = []
    for e in entries:
        out += [abs(x["bias"]) for x in e["base"].values()]
        for est in e["values"].values():
            out += [abs(x["bias"]) for k, x in est.items() if k != "hero_view_approx"]
    return out


def predictions(r: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered predictions, evaluated (see the module docstring)."""
    pop = r["population"]

    def med(hero, name, value="equilibrium"):
        return pop[hero][value][name]["ratio"]["median"]

    def item(claim, result, ok):
        return {"claim": claim, "result": result, "pass": bool(ok)}

    out = {}
    worst = max(_all_biases(r))
    out["A1"] = item(
        "Every estimator but the approximate hero's view is unbiased (|bias| < 1e-9)",
        f"largest |bias| {worst:.1e}",
        worst < 1e-9,
    )
    s = r["sampled"]
    max_z = max(x["max_abs_z"] for x in s.values())
    min_p = min(x["mean_z_p"] for x in s.values())
    sd = [x["sd_ratio"]["median"] for x in s.values()]
    out["A2"] = item(
        "Sampled hands agree with the exact moments: no |z| > 4, mean-z p >= 0.01, "
        "median sampled/exact SD in [0.97, 1.03]",
        f"max |z| {max_z:.2f}, min p {min_p:.2f}, SD ratios {min(sd):.3f}-{max(sd):.3f}",
        max_z <= 4 and min_p >= 0.01 and all(0.97 <= x <= 1.03 for x in sd),
    )
    q = pop["equilibrium"]["equilibrium"]["aivat"]["ratio"]
    out["B1"] = item(
        "AIVAT, equilibrium hero: median ratio >= 15x and every bot >= 5x",
        f"median {_fmt_ratio(q['median'])}, min {_fmt_ratio(q['min'])}",
        q["median"] >= 15 and q["min"] >= 5,
    )
    m = med("exploiter", "aivat")
    out["B2"] = item(
        "AIVAT, exploiting hero: median ratio between 3x and 8x", _fmt_ratio(m), 3 <= m <= 8
    )
    lift = med("equilibrium", "aivat", "oracle") / med("equilibrium", "aivat")
    out["B3"] = item(
        "Oracle values beat equilibrium values for the equilibrium hero by >= 1.4x",
        f"{lift:.2f}x",
        lift >= 1.4,
    )
    views = [med(h, "hero_view") for h in ("equilibrium", "exploiter")]
    out["C1"] = item(
        "Hero's view (folded cards hidden): median ratio < 1.15x for both heroes",
        " / ".join(_fmt_ratio(v) for v in views),
        max(views) < 1.15,
    )
    approx = [med(h, "hero_view_approx") for h in ("equilibrium", "exploiter")]
    bias = pop["exploiter"]["equilibrium"]["hero_view_approx"]["max_abs_bias"]
    out["C2"] = item(
        "Approximate hero's view: median ratio in [1.4x, 2.5x] for both heroes, and its bias "
        "reaches >= 0.02 chips/hand against some bot (exploiting hero)",
        " / ".join(_fmt_ratio(v) for v in approx) + f"; max |bias| {bias:.3f}",
        all(1.4 <= v <= 2.5 for v in approx) and bias >= 0.02,
    )
    m = med("equilibrium", "aivat", "uniform")
    out["D1"] = item(
        "Uniform-play values still give AIVAT a median ratio >= 4x (equilibrium hero)",
        _fmt_ratio(m),
        m >= 4,
    )
    ref = med("equilibrium", "aivat")
    late = [
        row["aivat"]["median"]
        for row in r["curve"]["equilibrium"]
        if row["value"].startswith("cfr+_") and int(row["value"][5:]) >= 30
    ]
    out["D2"] = item(
        "From 30 CFR+ iterations on, AIVAT's median ratio is within 20% of the equilibrium "
        "values' (equilibrium hero)",
        ", ".join(_fmt_ratio(x) for x in late) + f" vs {_fmt_ratio(ref)}",
        all(abs(x / ref - 1) <= 0.2 for x in late),
    )
    gain = med("equilibrium", "aivat_duplicate") / med("equilibrium", "aivat")
    out["E1"] = item(
        "Duplicate dealing adds < 25% on top of AIVAT (equilibrium hero)",
        f"{gain:.2f}x",
        gain < 1.25,
    )
    checks = r["bridge_crosscheck"]
    ratios = [c["ratio"] for c in checks]
    out["F1"] = item(
        "Push/fold: exact SDs of all-in EV and duplicate + all-in EV within 3% of P3.3's "
        "sampled SDs in every matchup",
        f"{len(ratios)} estimates, {min(ratios):.3f}-{max(ratios):.3f}" if ratios else "missing",
        bool(ratios) and all(0.97 <= x <= 1.03 for x in ratios),
    )
    pf = r["pushfold_summary"]
    nash = {s: pf[s]["nash"]["aivat"]["ratio_median"] for s in pf}
    out["F2"] = item(
        "Push/fold, Nash hero: AIVAT needs >= 4x fewer hands than all-in EV at every stack "
        "(median over bots)",
        ", ".join(f"{float(s):g} bb {_fmt_ratio(v)}" for s, v in nash.items()),
        bool(nash) and min(nash.values()) >= 4,
    )
    views = [pf[s][h]["hero_view"]["ratio_median"] for s in pf for h in pf[s]]
    out["F3"] = item(
        "Push/fold: hero's view gains < 1.3x over all-in EV at every stack, both heroes",
        f"{min(views):.2f}x-{max(views):.2f}x" if views else "missing",
        bool(views) and max(views) < 1.3,
    )
    return out


# ---- reporting --------------------------------------------------------------------------------


def _fmt_ratio(x: float) -> str:
    return "no variance" if x >= RATIO_CAP else f"{x:.2f}x" if x < 10 else f"{x:,.0f}x"


def markdown_summary(r: dict[str, Any]) -> str:
    cfg = r["meta"]["config"]
    pop = r["population"]
    lines = [
        "# P2.11: AIVAT in Leduc and push/fold, exact variances",
        "",
        f"Seed {r['meta']['seed']}, commit {(r['meta']['git']['commit'] or '?')[:7]}. "
        f"Leduc: {len(r['bots'])} bots ({cfg['bots_per_archetype']} per archetype), hero = "
        f"the equilibrium or a restricted Nash response (p = {cfg['rnr_p']}) to the bot. "
        "Ratio = variance of the plain result / variance of the estimator, per hand: how many "
        "times fewer hands the estimator needs for the same interval.",
        "",
        "## Leduc self-play (Burch et al. 2018, Table 1)",
        "",
        "| Estimator | SD per hand (chips) | Paper's SD | Bias |",
        "|---|---|---|---|",
    ]
    sp = r["selfplay"]
    for name in ESTIMATORS:
        e = _get(sp, name)
        paper = r["paper_selfplay_sd"].get(name)
        lines.append(
            f"| {LABELS[name]} | {e['sd']:.4f} | {'' if paper is None else paper} | "
            f"{e['bias']:+.1e} |"
        )
    for hero, value in (
        ("equilibrium", "equilibrium"),
        ("exploiter", "equilibrium"),
        ("equilibrium", "oracle"),
        ("exploiter", "oracle"),
    ):
        lines += [
            "",
            f"## Leduc bots, {hero} hero, {value} value function",
            "",
            f"| Estimator | Ratio: median [IQR] | min | max | worse than chips | max abs bias | "
            f"median hands for +/-{cfg['half_width']} |",
            "|---|---|---|---|---|---|---|",
        ]
        for name in ESTIMATORS:
            e = pop[hero][value][name]
            q = e["ratio"]
            lines.append(
                f"| {LABELS[name]} | {_fmt_ratio(q['median'])} [{_fmt_ratio(q['q25'])}, "
                f"{_fmt_ratio(q['q75'])}] | {_fmt_ratio(q['min'])} | {_fmt_ratio(q['max'])} | "
                f"{e['share_worse_than_chips']:.0%} | {e['max_abs_bias']:.1e} | "
                f"{e['hands']['median']:,.0f} |"
            )
    for hero in ("equilibrium", "exploiter"):
        lines += ["", f"## Value function quality, {hero} hero (median ratio over bots)", ""]
        lines += [
            "| Value function | Exploitability | "
            + " | ".join(LABELS[n] for n in CURVE_ESTIMATORS)
            + " |",
            "|---|---|" + "---|" * len(CURVE_ESTIMATORS),
        ]
        for row in r["curve"][hero]:
            lines.append(
                f"| {row['value']} | {row['exploitability']:.4f} | "
                + " | ".join(_fmt_ratio(row[n]["median"]) for n in CURVE_ESTIMATORS)
                + " |"
            )
    lines += ["", "## Sampled check", ""]
    lines += [
        "| Estimator | z-scores | mean z | max abs z | share abs z > 1.96 | t-test p | "
        "SD sampled / exact (median) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in r["sampled"].items():
        lines.append(
            f"| {LABELS[name]} | {s['n']} | {s['mean_z']:+.2f} | {s['max_abs_z']:.2f} | "
            f"{s['share_above_1_96']:.0%} | "
            f"{'' if s['mean_z_p'] is None else format(s['mean_z_p'], '.2f')} | "
            f"{s['sd_ratio']['median']:.3f} |"
        )
    lines += ["", "## Push/fold (median over the five archetype bots)", ""]
    lines += [
        "| Stack | Hero | "
        + " | ".join(
            LABELS[n] for n in ("chips", "duplicate", "aivat", "aivat_duplicate", "hero_view")
        )
        + " |",
        "|---|---|---|---|---|---|---|",
    ]
    for stack, per in r["pushfold_summary"].items():
        for hero, row in per.items():
            lines.append(
                f"| {float(stack):g} | {hero} | "
                + " | ".join(
                    f"{row[n]['hands_median']:,.0f} ({_fmt_ratio(row[n]['ratio_median'])})"
                    for n in ("chips", "duplicate", "aivat", "aivat_duplicate", "hero_view")
                )
                + " |"
            )
    if r["bridge_crosscheck"]:
        ratios = [c["ratio"] for c in r["bridge_crosscheck"]]
        lines += [
            "",
            f"P3.3 cross-check: sampled / exact SD over {len(ratios)} estimates: "
            f"{min(ratios):.3f} to {max(ratios):.3f}.",
        ]
    if r["predictions"]:
        lines += [
            "",
            "## Predictions",
            "",
            "| | Prediction | Result | Holds |",
            "|---|---|---|---|",
        ]
        for key, p in r["predictions"].items():
            lines.append(
                f"| {key} | {p['claim']} | {p['result']} | {'yes' if p['pass'] else 'NO'} |"
            )
    lines += ["", f"Run time {r['elapsed_seconds']:.0f} s."]
    return "\n".join(lines) + "\n"


FIGURE_ESTIMATORS = (
    "duplicate",
    "mivat",
    "mivat_io",
    "aivat",
    "aivat_duplicate",
    "aivat_all",
    "hero_view",
    "hero_view_approx",
)
SHORT = {
    "chips": "Plain result",
    "duplicate": "Duplicate dealing",
    "mivat": "Chance terms (MIVAT)",
    "mivat_io": "Chance terms + imaginary obs.",
    "aivat": "AIVAT, all cards shown",
    "aivat_duplicate": "AIVAT + duplicate",
    "aivat_all": "AIVAT, opponent's strategy known",
    "hero_view": "Hero's view only",
    "hero_view_approx": "Hero's view + board terms (biased)",
}
HEROES = {"equilibrium": "Equilibrium hero", "exploiter": "Exploiting hero (RNR, p = 0.9)"}


def make_figures(r: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import legend, line, new_figure, save_both, style_axes, title

    bots = r["bots"]

    def leduc(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(10.5, 4.6), sharey=True)
        s = theme["series"]
        for ax, hero in zip(axes, HEROES, strict=True):
            style_axes(ax, theme, grid_axis="x")
            y = np.arange(len(FIGURE_ESTIMATORS))
            for k, name in enumerate(FIGURE_ESTIMATORS):
                ratios = np.array(
                    [
                        _ratio(b["heroes"][hero]["base"]["chips"], _get(b["heroes"][hero], name))
                        for b in bots
                    ]
                )
                ratios = np.minimum(ratios, 1e3)
                jitter = (np.arange(len(ratios)) % 7 - 3) * 0.06
                ax.scatter(ratios, k + jitter, s=9, color=theme["neutral"], linewidths=0)
                q = np.quantile(ratios, [0.25, 0.5, 0.75])
                color = s[3] if name == "hero_view_approx" else s[0]
                ax.plot(q[[0, 2]], [k, k], color=color, linewidth=3, solid_capstyle="round")
                ax.plot(q[1], k, "o", color=color, markersize=7, markeredgecolor=theme["surface"])
            ax.axvline(1.0, color=theme["axis"], linewidth=0.8)
            ax.set_xscale("log")
            ax.set_yticks(y)
            ax.set_yticklabels([SHORT[n] for n in FIGURE_ESTIMATORS], fontsize=8.5)
            ax.invert_yaxis()
            ax.set_xlabel("Times fewer hands than the plain result (log)")
            ax.set_title(HEROES[hero], loc="left", fontsize=10, color=theme["ink"])
        title(
            fig,
            theme,
            "Leduc: how much each estimator saves",
            f"{len(bots)} bots; dots are bots, bars the middle half, circles the median "
            "(equilibrium value function; the orange row is biased)",
            top=0.86,
        )
        return fig

    def quality(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(10.5, 4.2), sharey=True)
        s = theme["series"]
        for ax, hero in zip(axes, HEROES, strict=True):
            style_axes(ax, theme)
            by_name = {row["value"]: row for row in r["curve"][hero]}
            cfr = sorted((int(v[5:]), v) for v in by_name if v.startswith("cfr+_"))
            rows = [by_name[v] for _, v in cfr] + [by_name["equilibrium"]]
            x = np.arange(len(rows))
            for k, name in enumerate(CURVE_ESTIMATORS):
                line(ax, theme, x, [row[name]["median"] for row in rows], s[k], SHORT[name])
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(
                ["LP" if row["value"] == "equilibrium" else row["value"][5:] for row in rows],
                fontsize=8.5,
            )
            ax.set_xlabel("CFR+ iterations behind the values (1 = uniform play; LP = exact)")
            ax.set_title(HEROES[hero], loc="left", fontsize=10, color=theme["ink"])
        axes[0].set_ylabel("Times fewer hands (median over bots, log)")
        legend(axes[0], theme, loc="upper left")
        title(
            fig,
            theme,
            "How good does the value function need to be?",
            "Median variance ratio against the plain result, by the solver that made the values",
            top=0.86,
        )
        return fig

    def pushfold(theme):
        fig, ax = new_figure(theme, 1, 1, figsize=(8, 4.4))
        style_axes(ax, theme)
        s = theme["series"]
        names = ("chips", "duplicate", "aivat", "aivat_duplicate", "hero_view")
        pf = r["pushfold_summary"]
        stacks = list(pf)
        x = np.arange(len(stacks))
        width = 0.16
        sampled = {}
        for c in r["bridge_crosscheck"]:
            if c["hero"] == "nash" and c["opponent"] != "nash":
                sampled.setdefault((c["stack"], c["estimator"]), []).append(c["sampled_hands"])
        for k, name in enumerate(names):
            med = [pf[st]["nash"][name]["hands_median"] for st in stacks]
            pos = x + (k - 2) * width
            label = {"chips": "All-in EV", "duplicate": "Duplicate + all-in EV"}.get(
                name, SHORT[name]
            )
            ax.bar(pos, med, width * 0.9, color=s[k], label=label)
            for p, st in zip(pos, stacks, strict=True):
                got = sampled.get((float(st), name))
                if got:
                    ax.plot(p, np.median(got), "D", color=theme["ink"], markersize=4)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{float(st):g} bb" for st in stacks])
        ax.set_ylabel(f"Hands for a +/-{r['meta']['config']['pushfold']['half_width']} bb 95% CI")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
                   labelcolor=theme["ink2"])  # fmt: skip
        title(
            fig,
            theme,
            "Push/fold: hands a win rate needs, exactly",
            "Nash hero, median over the five archetype bots; diamonds: P3.3's sampled values",
        )
        fig.subplots_adjust(bottom=0.24)
        return fig

    save_both(leduc, out_dir, "aivat_leduc")
    save_both(quality, out_dir, "aivat_value_quality")
    if r["pushfold_summary"]:
        save_both(pushfold, out_dir, "aivat_pushfold")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--quick", action="store_true", help="use configs/phase2_quick.yaml")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    path = CONFIG.with_name("phase2_quick.yaml") if args.quick else args.config
    cfg = load_config(path)["aivat"]
    r = run(args.seed, cfg=cfg, workers=args.workers, out_dir=args.out)
    print(markdown_summary(r))


if __name__ == "__main__":
    main()

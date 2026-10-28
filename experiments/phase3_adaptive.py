"""P3.9: exploiting push/fold opponents that adapt to the hero.

    python -m experiments.phase3_adaptive --seed 2026

P3.2 to P3.5 judged every hero strategy against a bot that keeps its strategy. A real opponent
answers an exploit. Here the bots adapt (``poker_lab.games.push_fold_adaptive``): every 25 hands
they move their push and call widths toward a best response to the hero's observed push and call
rates. Sessions are scored by the exact expected value of the two strategies in force, block by
block, so the luck of the cards is removed while the path the adaptation took is kept.

Pre-registered plan, written before the real run. The design (bot kinds, speeds, heroes, session
length) and the predictions were fixed on development seeds 98 and 99 (10 starting bots each),
never on the reporting seeds.

**Setup.** Heads-up push/fold at 10 bb. ``seeds`` replicates, each with the prior fit to P3.2's
training population and P3.2's standard test bots (``test_per_archetype`` per archetype) as the
starting points, so the static control is P3.4's standard population. Four kinds of bot start
from each of them:

* **static**: P3.2's bot (the control);
* **slow**: steps 10% of the way to its best response every 25 hands, rates with a half-life of
  200 hands;
* **fast**: steps 50% of the way, half-life 50 hands;
* **calls only**: like the fast bot, but only its call width moves (the player who "calls wider
  after the hero pushes often" but keeps its own pushing habits).

Six heroes, each re-deciding from the redacted log of completed hands (the bot's cards only
where shown):

* **equilibrium**: always the Nash equilibrium; every gain is measured from it;
* **fit once**: the equilibrium while watching ``observe_hands`` hands, then P3.4's advised
  response (ranked by the 90% lower bound; the equilibrium when P3.5's stability check fails,
  P3.5b's default), kept for the rest of the session;
* **refit**: the advised response recomputed every ``refit_every`` hands on all hands so far
  (from hand 0, where it is the population recommendation);
* **refit recent**: the same on the last ``window`` hands only;
* **best response**: the Bayes best response recomputed every ``refit_every`` hands on all hands
  so far, with no lower bound and no fallback;
* **clairvoyant** (reference, not a strategy anyone can play): the best response to the bot's
  current strategy, read from the bot every block. Against a static bot it is P3.2's oracle.

Sessions of ``hands`` hands; every hero meets a fresh copy of the same bot with the same random
numbers. The model-fit checks are calibrated per replicate on ``calibration_bots`` bots drawn
from the prior and played by the equilibrium hero (P3.5's method) at ``check_sizes``.

**Measures.** A session's *value* is the hero's mean exact value per hand (bb/hand); a hero's
*gain* is its value minus the equilibrium hero's against the same bot. Gains are averaged over
bots per replicate, with 95% t-intervals across replicates; "loses" is the share of bots with
a negative gain (95% Wilson interval). The *fallback rate* is the share of the refit hero's
refits in hands 1,000-1,999 whose advice is the equilibrium because the stability check failed.

**Predictions.**

* A1 (control): against static bots the refit hero gains at least 85% of the clairvoyant
  hero's gain (mean over replicates of the per-replicate ratio).
* B1: against slow and fast adapters the equilibrium hero's value is at most half its value
  against the same bots static (the 95% interval of the per-replicate ratio below 0.5).
* B2: against slow and fast adapters even the clairvoyant hero gains at most 0.01 bb/hand over
  equilibrium play (interval below 0.01).
* C1: against fast adapters the Bayes best response loses to equilibrium play (interval below
  0).
* C2: against slow adapters fitting once loses to equilibrium play (interval below 0).
* C3: against slow and fast adapters the refit hero stays within 0.005 bb/hand of equilibrium
  play (interval inside [-0.005, 0.005]).
* C4: against calls-only adapters the refit hero gains over equilibrium play (interval above 0).
* D1: the refit hero's fallback rate is at least 30% against slow and fast adapters and at most
  10% against static bots.

Reported without predictions: every hero against every kind, gains over time, exploitability of
the heroes' strategies, how far the bots move, and example trajectories.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
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
from experiments.phase3_exploit import hero_strategy, sample_populations
from poker_lab.evaluation.intervals import t_interval, wilson_interval
from poker_lab.exploit.recommend import PushFoldRecommender
from poker_lab.games.preflop import N_COMBOS, combo_counts
from poker_lab.games.push_fold import hand_gains, strategy_exploitability
from poker_lab.games.push_fold_adaptive import STATIC, Adaptation, AdaptiveBot, play_session
from poker_lab.games.push_fold_bots import HandLog, Observations, PushFoldPrior
from poker_lab.games.push_fold_checks import ActionLog, FitChecker, calibration_pvalues

CONFIG = REPO_ROOT / "experiments" / "configs" / "phase3.yaml"
HEROES = ("equilibrium", "fit_once", "refit", "refit_recent", "best_response", "clairvoyant")
HERO_LABELS = {
    "equilibrium": "Equilibrium",
    "fit_once": "Fit once, then keep",
    "refit": "Refit on all hands",
    "refit_recent": "Refit on recent hands",
    "best_response": "Bayes best response",
    "clairvoyant": "Clairvoyant (reference)",
}
KIND_LABELS = {
    "static": "Static bots",
    "slow": "Slow adapters",
    "fast": "Fast adapters",
    "calls_only": "Adapt calls only",
}
ADAPTERS = ("slow", "fast")
TOL = 1e-9
_RECOMMENDERS: dict[int, PushFoldRecommender] = {}


def adaptations(cfg: dict[str, Any]) -> dict[str, Adaptation]:
    out = {"static": STATIC}
    for kind, a in cfg["adaptive"].items():
        out[kind] = Adaptation(
            a["speed"],
            a["half_life"],
            cfg["period"],
            cfg["pseudo"],
            adapts=tuple(a.get("adapts", ("push", "call"))),
        )
    return out


def calibration_task(seed: int, k: int, prior: PushFoldPrior, cfg: dict[str, Any]):
    rng = np.random.default_rng([seed, 39, k])
    hero = hero_strategy(float(cfg["stack"]))
    return calibration_pvalues(
        prior, hero, cfg["check_sizes"], rng, draws=cfg["check_draws"], posterior_draws=cfg["draws"]
    )


def recommender(seed: int, prior: PushFoldPrior, thresholds, cfg) -> PushFoldRecommender:
    if seed not in _RECOMMENDERS:
        checker = FitChecker(thresholds, draws=cfg["check_draws"], seed=[seed, 30])
        _RECOMMENDERS[seed] = PushFoldRecommender(
            float(cfg["stack"]),
            prior,
            ps=cfg["ps"],
            level=cfg["level"],
            draws=cfg["draws"],
            seed=[seed, 20],
            checker=checker,
        )
    return _RECOMMENDERS[seed]


class RecommenderHero:
    """A hero that re-decides from the recommender every ``every`` hands from ``start`` on (the
    equilibrium before), on all completed hands or the last ``window``. ``rule`` is
    ``"advised"`` (P3.4's lower-bound pick with P3.5b's fallback) or ``"best_response"``; with
    ``once`` it decides a single time."""

    def __init__(
        self,
        rec: PushFoldRecommender,
        *,
        every: int,
        start: int = 0,
        window: int | None = None,
        rule: str = "advised",
        once: bool = False,
        seed: list[int],
    ) -> None:
        self.rec, self.every, self.start, self.window = rec, every, start, window
        self.rule, self.once, self.seed = rule, once, seed
        self.current = rec.equilibrium
        self.records: list[dict[str, Any]] = []

    def __call__(self, log: HandLog | None, n: int):
        due = n >= self.start and (n - self.start) % self.every == 0
        if not due or (self.once and self.records):
            return self.current
        if log is None:
            data: Observations | ActionLog = Observations.empty()
        else:
            lo = 0 if self.window is None else max(0, n - self.window)
            data = ActionLog.from_hand_log(log.window(lo, n))
        report = self.rec.recommend(data, seed=self.seed)
        choice = report.advised if self.rule == "advised" else report.get("best_response")
        failed = {r.name: r.failed for r in report.fit.results} if report.fit else {}
        self.records.append(
            {
                "hand": n,
                "choice": choice.name,
                "p": choice.p,
                "fallback": report.falls_back if self.rule == "advised" else False,
                "failed": failed,
            }
        )
        self.current = (choice.push, choice.call)
        return self.current


class ClairvoyantHero:
    """The best response to the bot's current strategy (a reference: it reads the bot)."""

    def __init__(self, bot: AdaptiveBot) -> None:
        self.bot = bot

    def __call__(self, log: HandLog | None, n: int):
        push, call = self.bot.strategy()
        sb, bb = hand_gains(self.bot.stack, push, call, self.bot.matchups)
        return (sb > 0).astype(float), (np.nan_to_num(bb, nan=-1.0) > 0).astype(float)


def _bins(session, edges: list[int]) -> list[float]:
    return [session.value(a, b) for a, b in pairwise(edges)]


def bot_task(
    seed: int,
    index: int,
    archetype: str,
    params,
    kind: str,
    prior: PushFoldPrior,
    thresholds,
    cfg: dict[str, Any],
    example: bool = False,
) -> dict[str, Any]:
    stack = float(cfg["stack"])
    rec = recommender(seed, prior, thresholds, cfg)
    adapt = adaptations(cfg)[kind]
    kinds = list(adaptations(cfg))
    stream = [seed, 39, kinds.index(kind), index]
    edges = list(range(0, cfg["hands"] + 1, cfg["bin"]))
    combos = combo_counts() / N_COMBOS
    out: dict[str, Any] = {"index": index, "archetype": archetype, "kind": kind, "heroes": {}}
    for name in HEROES:
        bot = AdaptiveBot(params, adapt, stack, rec.equilibrium, rec.matchups)
        if name == "equilibrium":
            hero: Any = lambda log, n: rec.equilibrium  # noqa: E731
        elif name == "clairvoyant":
            hero = ClairvoyantHero(bot)
        else:
            hero = RecommenderHero(
                rec,
                every=cfg["refit_every"],
                start=cfg["observe_hands"] if name == "fit_once" else 0,
                window=cfg["window"] if name == "refit_recent" else None,
                rule="best_response" if name == "best_response" else "advised",
                once=name == "fit_once",
                seed=[seed, 21, index],
            )
        session = play_session(hero, bot, cfg["hands"], np.random.default_rng(stream))
        expl = np.asarray(strategy_exploitability(stack, session.hero_push, session.hero_call))
        sizes = session.sizes
        row = {
            "value": session.value(),
            "bins": _bins(session, edges),
            "exploitability": float(expl @ sizes / sizes.sum()),
            "end_widths": [session.bot_params[-1].push_width, session.bot_params[-1].call_width],
            "records": getattr(hero, "records", []),
        }
        if example:
            row["trajectory"] = {
                "values": session.values.tolist(),
                "widths": [[p.push_width, p.call_width] for p in session.bot_params],
                "hero_shares": np.column_stack(
                    [session.hero_push @ combos, session.hero_call @ combos]
                ).tolist(),
            }
        out["heroes"][name] = row
    out["start_widths"] = [params.push_width, params.call_width]
    return out


def evaluate(seeds: list[int], cfg: dict[str, Any], workers: int | None):
    setups = {s: sample_populations(s, cfg) for s in seeds}
    with ProcessPoolExecutor(max_workers=workers or cfg.get("workers")) as pool:
        calib_tasks = [
            (s, k, setups[s][0], cfg) for s in seeds for k in range(cfg["calibration_bots"])
        ]
        calib = list(pool.map(calibration_task, *zip(*calib_tasks, strict=True), chunksize=4))
        thresholds = {}
        for s in seeds:
            pv = [c for c, t in zip(calib, calib_tasks, strict=True) if t[0] == s]
            thresholds[s] = FitChecker.from_pvalues(pv, level=cfg["fit_level"]).thresholds
        tasks = []
        for s in seeds:
            prior, bots = setups[s]
            for pop, k, archetype, params in bots:
                if pop != "standard":
                    continue
                example = s == seeds[0] and k % cfg["test_per_archetype"] == 0
                for kind in adaptations(cfg):
                    tasks.append(
                        (s, k, archetype, params, kind, prior, thresholds[s], cfg, example)
                    )
        records = list(pool.map(bot_task, *zip(*tasks, strict=True), chunksize=1))
    per_seed = [[r for r, t in zip(records, tasks, strict=True) if t[0] == s] for s in seeds]
    return per_seed, {str(s): {str(n): t for n, t in thresholds[s].items()} for s in seeds}


def _t(values: list[float]) -> dict[str, float]:
    ci = t_interval(np.asarray(values, dtype=float))
    return {"mean": ci.estimate, "lower": ci.lower, "upper": ci.upper}


def _share(k: int, n: int) -> dict[str, float]:
    ci = wilson_interval(int(k), int(n))
    return {"share": k / n if n else float("nan"), "lower": ci.lower, "upper": ci.upper, "n": n}


def aggregate(per_seed: list[list[dict[str, Any]]], cfg: dict[str, Any]) -> dict[str, Any]:
    edges = list(range(0, cfg["hands"] + 1, cfg["bin"]))
    half = cfg["hands"] // 2
    out: dict[str, Any] = {}
    for kind in adaptations(cfg):
        block: dict[str, Any] = {}
        for h in HEROES:
            gains_by_seed, values_by_seed, expl_by_seed, bins_by_seed = [], [], [], []
            lost = n_bots = ever = 0
            fallback_by_bin = np.zeros(len(edges) - 1)
            refits_by_bin = np.zeros(len(edges) - 1)
            late = [0, 0]  # fallbacks and refits in the second half of the session
            for recs in per_seed:
                mine = [r for r in recs if r["kind"] == kind]
                gains = [
                    r["heroes"][h]["value"] - r["heroes"]["equilibrium"]["value"] for r in mine
                ]
                gains_by_seed.append(float(np.mean(gains)))
                values_by_seed.append(float(np.mean([r["heroes"][h]["value"] for r in mine])))
                expl_by_seed.append(
                    float(np.mean([r["heroes"][h]["exploitability"] for r in mine]))
                )
                bins_by_seed.append(
                    np.mean(
                        [
                            np.subtract(r["heroes"][h]["bins"], r["heroes"]["equilibrium"]["bins"])
                            for r in mine
                        ],
                        axis=0,
                    )
                )
                lost += sum(g < -TOL for g in gains)
                n_bots += len(gains)
                for r in mine:
                    hero_recs = r["heroes"][h]["records"]
                    ever += any(x["fallback"] for x in hero_recs)
                    for x in hero_recs:
                        b = min(x["hand"] // cfg["bin"], len(edges) - 2)
                        refits_by_bin[b] += 1
                        fallback_by_bin[b] += x["fallback"]
                        if x["hand"] >= half:
                            late[0] += x["fallback"]
                            late[1] += 1
            bins = np.array(bins_by_seed)
            block[h] = {
                "value": _t(values_by_seed),
                "value_by_seed": values_by_seed,
                "gain": _t(gains_by_seed),
                "gain_by_seed": gains_by_seed,
                "loses": _share(lost, n_bots),
                "exploitability": _t(expl_by_seed),
                "gain_by_bin": [_t(bins[:, b].tolist()) for b in range(bins.shape[1])],
                "fallback_by_bin": (
                    (fallback_by_bin / np.maximum(refits_by_bin, 1)).tolist()
                    if refits_by_bin.any()
                    else None
                ),
                "fallback_second_half": _share(*late) if late[1] else None,
                "ever_fell_back": ever / n_bots,
            }
        moved = [
            float(
                np.abs(
                    np.subtract(r["heroes"]["equilibrium"]["end_widths"], r["start_widths"])
                ).sum()
            )
            for recs in per_seed
            for r in recs
            if r["kind"] == kind
        ]
        out[kind] = {"heroes": block, "width_moved_vs_equilibrium": float(np.mean(moved))}
    return out


def predictions(agg: dict[str, Any]) -> dict[str, Any]:
    static = agg["static"]["heroes"]
    ratios = [
        r / c
        for r, c in zip(
            static["refit"]["gain_by_seed"], static["clairvoyant"]["gain_by_seed"], strict=True
        )
    ]
    out: dict[str, Any] = {
        "A1": {"ratio": float(np.mean(ratios)), "holds": float(np.mean(ratios)) >= 0.85}
    }
    b1, b2, c3, d1 = {}, {}, {}, {}
    for kind in ADAPTERS:
        h = agg[kind]["heroes"]
        rel = _t(
            [
                a / s
                for a, s in zip(
                    h["equilibrium"]["value_by_seed"],
                    static["equilibrium"]["value_by_seed"],
                    strict=True,
                )
            ]
        )
        b1[kind] = {**rel, "holds": rel["upper"] < 0.5}
        b2[kind] = {**h["clairvoyant"]["gain"], "holds": h["clairvoyant"]["gain"]["upper"] < 0.01}
        g = h["refit"]["gain"]
        c3[kind] = {**g, "holds": g["lower"] >= -0.005 and g["upper"] <= 0.005}
        late = h["refit"]["fallback_second_half"]
        d1[kind] = {**late, "holds": late["share"] >= 0.30}
    late = static["refit"]["fallback_second_half"]
    d1["static"] = {**late, "holds": late["share"] <= 0.10}
    c1 = agg["fast"]["heroes"]["best_response"]["gain"]
    c2 = agg["slow"]["heroes"]["fit_once"]["gain"]
    c4 = agg["calls_only"]["heroes"]["refit"]["gain"]
    return out | {
        "B1": {"by_kind": b1, "holds": all(v["holds"] for v in b1.values())},
        "B2": {"by_kind": b2, "holds": all(v["holds"] for v in b2.values())},
        "C1": {**c1, "holds": c1["upper"] < 0},
        "C2": {**c2, "holds": c2["upper"] < 0},
        "C3": {"by_kind": c3, "holds": all(v["holds"] for v in c3.values())},
        "C4": {**c4, "holds": c4["lower"] > 0},
        "D1": {"by_kind": d1, "holds": all(v["holds"] for v in d1.values())},
    }


def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase3" / "adaptive"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()
    seeds = derive_seeds(seed, cfg["seeds"])
    per_seed, thresholds = evaluate(seeds, cfg, workers)
    agg = aggregate(per_seed, cfg)
    examples = [
        {k: r[k] for k in ("index", "archetype", "kind", "start_widths")}
        | {"heroes": {h: v["trajectory"] for h, v in r["heroes"].items()}}
        for r in per_seed[0]
        if "trajectory" in r["heroes"]["equilibrium"]
    ]
    for recs in per_seed:
        for r in recs:
            for v in r["heroes"].values():
                v.pop("trajectory", None)
    results = {
        "meta": meta,
        "seeds": seeds,
        "thresholds": thresholds,
        "aggregate": agg,
        "predictions": predictions(agg),
        "examples": examples,
        "per_seed": per_seed,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    save_json(results, out_dir / "adaptive.json")
    (out_dir / "adaptive_summary.md").write_text(
        markdown_summary(results), encoding="utf-8", newline="\n"
    )
    make_figures(results, out_dir)
    return results


# ---------------------------------------------------------------------------- report
def _verdict(holds: bool) -> str:
    return "holds" if holds else "**does not hold**"


def _ci(v: dict[str, float], digits: int = 4) -> str:
    return f"{v['mean']:+.{digits}f} [{v['lower']:+.{digits}f}, {v['upper']:+.{digits}f}]"


def markdown_summary(r: dict[str, Any]) -> str:
    m, agg, pr = r["meta"], r["aggregate"], r["predictions"]
    cfg = m["config"]
    git = m["git"]
    lines = [
        "# P3.9: exploiting push/fold opponents that adapt",
        "",
        f"Seed {m['seed']}, commit {git['commit']} (dirty: {git['dirty']}), "
        f"{r['elapsed_seconds']:.0f} s. {cfg['seeds']} replicates x "
        f"{5 * cfg['test_per_archetype']} starting bots x {len(agg)} kinds; sessions of "
        f"{cfg['hands']:,} hands at {cfg['stack']} bb.",
        "",
        "## Value and gain over equilibrium play (bb/hand, 95% t-intervals across replicates)",
        "",
    ]
    for kind, b in agg.items():
        lines += [
            f"### {KIND_LABELS[kind]} (widths moved {b['width_moved_vs_equilibrium']:.2f} "
            "against the equilibrium hero)",
            "",
            "| Hero | Value | Gain over equilibrium | Loses to equilibrium | Exploitability | "
            "Fallback rate, hands 1,000+ |",
            "|---|---|---|---|---|---|",
        ]
        for h in HEROES:
            v = b["heroes"][h]
            fb = v["fallback_second_half"]
            lines.append(
                f"| {HERO_LABELS[h]} | {_ci(v['value'])} | {_ci(v['gain'])} | "
                f"{v['loses']['share']:.0%} | {v['exploitability']['mean']:.3f} | "
                + ("" if fb is None else f"{fb['share']:.0%}")
                + " |"
            )
        lines.append("")
    a1 = pr["A1"]
    lines += [
        "## Predictions",
        "",
        f"* A1 (refit keeps at least 85% of the clairvoyant gain against static bots): "
        f"{a1['ratio']:.0%}; {_verdict(a1['holds'])}.",
        "* B1 (adaptation at least halves the equilibrium hero's value): "
        + "; ".join(
            f"{k} ratio {v['mean']:.2f} [{v['lower']:.2f}, {v['upper']:.2f}]"
            for k, v in pr["B1"]["by_kind"].items()
        )
        + f"; {_verdict(pr['B1']['holds'])}.",
        "* B2 (the clairvoyant gains at most 0.01 against adapters): "
        + "; ".join(f"{k} {_ci(v)}" for k, v in pr["B2"]["by_kind"].items())
        + f"; {_verdict(pr['B2']['holds'])}.",
        f"* C1 (the Bayes best response loses against fast adapters): {_ci(pr['C1'])}; "
        f"{_verdict(pr['C1']['holds'])}.",
        f"* C2 (fitting once loses against slow adapters): {_ci(pr['C2'])}; "
        f"{_verdict(pr['C2']['holds'])}.",
        "* C3 (refit within 0.005 of equilibrium against adapters): "
        + "; ".join(f"{k} {_ci(v)}" for k, v in pr["C3"]["by_kind"].items())
        + f"; {_verdict(pr['C3']['holds'])}.",
        f"* C4 (refit gains against calls-only adapters): {_ci(pr['C4'])}; "
        f"{_verdict(pr['C4']['holds'])}.",
        "* D1 (refit falls back at least 30% against adapters, at most 10% against static bots): "
        + "; ".join(
            f"{k} {v['share']:.0%} [{v['lower']:.0%}, {v['upper']:.0%}]"
            for k, v in pr["D1"]["by_kind"].items()
        )
        + f"; {_verdict(pr['D1']['holds'])}.",
        "",
    ]
    return "\n".join(lines)


def make_figures(r: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import band, legend, line, new_figure, save_both, style_axes, title

    agg = r["aggregate"]
    cfg = r["meta"]["config"]
    edges = np.arange(0, cfg["hands"] + 1, cfg["bin"])
    x = (edges[:-1] + edges[1:]) / 2
    shown = [h for h in HEROES if h != "equilibrium"]

    def draw_gain(theme):
        fig, axes = new_figure(theme, 1, len(agg), figsize=(4.0 * len(agg), 4.4))
        axes = np.atleast_1d(axes)
        s = theme["series"]
        for ax, (kind, b) in zip(axes, agg.items(), strict=True):
            style_axes(ax, theme)
            for k, h in enumerate(shown):
                g = b["heroes"][h]["gain_by_bin"]
                band(ax, x, [v["lower"] for v in g], [v["upper"] for v in g], s[k])
                dash = "--" if h == "clairvoyant" else "-"
                line(ax, theme, x, [v["mean"] for v in g], s[k], HERO_LABELS[h], linestyle=dash)
            ax.axhline(0, color=theme["axis"], linewidth=0.8)
            ax.set_title(KIND_LABELS[kind], loc="left", fontsize=10, color=theme["ink"])
            ax.set_xlabel("Hands into the session")
        axes[0].set_ylabel("Gain over equilibrium play (bb/hand)")
        legend(axes[0], theme, loc="lower right")
        title(
            fig,
            theme,
            "Exploiting bots that adapt to the hero",
            f"Mean gain per {cfg['bin']}-hand stretch over the equilibrium hero against the same "
            "bots. Bands: 95% t-intervals across replicates. Each panel has its own scale.",
        )
        return fig

    def draw_values(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(12.5, 4.4))
        s = theme["series"]
        # heroes keep their colors from the gain figure; equilibrium play is the neutral baseline
        hero_color = {h: s[k] for k, h in enumerate(shown)} | {"equilibrium": theme["muted"]}
        kind_color = (s[0], s[2], s[3], s[6])  # colors the heroes in the left panel do not use
        kinds = list(agg)
        xs = np.arange(len(kinds))
        ax = axes[0]
        style_axes(ax, theme)
        for k, h in enumerate(("equilibrium", "refit", "clairvoyant")):
            v = [agg[kind]["heroes"][h]["value"] for kind in kinds]
            ax.errorbar(
                xs + (k - 1) * 0.18,
                [y["mean"] for y in v],
                yerr=[[y["mean"] - y["lower"] for y in v], [y["upper"] - y["mean"] for y in v]],
                fmt="o",
                color=hero_color[h],
                markersize=7,
                markeredgecolor=theme["surface"],
                capsize=0,
                label=HERO_LABELS[h],
            )
        ax.set_xticks(xs, [KIND_LABELS[k] for k in kinds], fontsize=8.5, color=theme["ink2"])
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Hero's value (bb/hand)")
        ax.set_title("What is left to win", loc="left", fontsize=10, color=theme["ink"])
        legend(ax, theme, loc="upper right")
        ax = axes[1]
        style_axes(ax, theme)
        for k, kind in enumerate(kinds):
            fb = agg[kind]["heroes"]["refit"]["fallback_by_bin"]
            line(ax, theme, x, [100 * v for v in fb], kind_color[k], KIND_LABELS[kind])
        ax.axhline(5, color=theme["muted"], linewidth=1, linestyle=":")
        ax.set_xlabel("Hands into the session")
        ax.set_ylabel("Refits that fall back to equilibrium (%)")
        ax.set_title(
            "The stability check notices adaptation", loc="left", fontsize=10, color=theme["ink"]
        )
        legend(ax, theme, loc="upper left")
        title(
            fig,
            theme,
            "Adaptation takes the money off the table, and the checks see it",
            "Left: session value with 95% t-intervals across replicates. Right: the refit hero's "
            "refits whose advice is equilibrium play; dotted: the 5% calibrated level.",
        )
        return fig

    save_both(draw_gain, out_dir, "adaptive_gain")
    save_both(draw_values, out_dir, "adaptive_values")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--section", default="adaptive")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)[args.section]
    r = run(args.seed, cfg=cfg, workers=args.workers, out_dir=args.out)
    print(json.dumps({k: v.get("holds") for k, v in r["predictions"].items()}))


if __name__ == "__main__":
    main()

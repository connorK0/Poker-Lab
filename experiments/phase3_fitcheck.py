"""P3.5: model-fit checks before exploiting, evaluated on held-out push/fold bots.

    python -m experiments.phase3_fitcheck --seed 2026

Pre-registered plan (written before the first run; the design, i.e. the chi-square statistic,
mid-p values and 300 calibration bots, was fixed in pilots on development seeds 98 and 99,
never on the reporting seeds):

* Game, seeds, training populations and priors as in P3.2 and P3.4 (10 bb). Four held-out
  populations per seed, ``test_per_archetype`` bots per archetype each:
  - **standard**: P3.2's standard bots and hands (the model is right);
  - **pairs_first**: P3.2's misspecified bots and hands (a mild error in the hand order);
  - **connectors_first**: new bots that rank hands "connectors first" (rank correlation 0.48
    with the equity order the model assumes);
  - **gear_change**: new bots that switch to another archetype once per session. For each sample
    size ``n`` the bot plays its own session of ``n`` hands and switches at a hand drawn
    uniformly from the middle half; recommendations are judged against the second gear, the one
    it plays from then on.
* Calibration: per seed, ``calibration_bots`` bots drawn from the fitted prior and played for
  ``max(sizes)`` hands by the equilibrium hero; each check's threshold per sample size is the
  ``fit_level / 2`` quantile of its p-values. Test bots are never used.
* For every test bot and ``n``: P3.4's recommender (same settings) with both checks on the same
  posterior: p-values, flags, the ranked candidates with their 90% intervals, and every
  candidate's exact true gain.
* Rules: the **lower-bound pick** (P3.4) and the **checked pick** (the equilibrium when either
  check fails).
* Questions and predictions:
  1. *False alarms.* Share of standard bots flagged by either check. Prediction: at most 10% at
     every ``n``.
  2. *Power.* Connectors-first bots flagged: at least 60% at 100 hands and at least 90% from 300.
     Gear-change bots: at least 40% at 100 hands and at least 85% from 1,000. Pairs-first: no
     prediction.
  3. *Do flags mark untrustworthy intervals?* 90% interval coverage (non-equilibrium candidates)
     among flagged vs unflagged bots of the three misspecified populations pooled. Prediction:
     lower among flagged bots at every ``n`` >= 100.
  4. *What does acting on a flag cost or save?* True gain over equilibrium and the share of bots
     lost to, for the checked pick vs the lower-bound pick, by population and ``n``. No
     prediction.
* Shares are pooled over bots and seeds with 95% Wilson intervals; gains are means over bots per
  seed with 95% t-intervals across seeds.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
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
from experiments.phase3_exploit import hero_strategy, observation_log, sample_populations
from poker_lab.evaluation.intervals import t_interval, wilson_interval
from poker_lab.exploit.recommend import PushFoldRecommender
from poker_lab.games.push_fold import hand_gains, match_values
from poker_lab.games.push_fold_bots import (
    ARCHETYPES,
    HandLog,
    PushFoldPrior,
    sample_params,
    simulate,
    strategy,
)
from poker_lab.games.push_fold_checks import (
    CHECKS,
    ActionLog,
    FitChecker,
    calibration_pvalues,
)

CONFIG = REPO_ROOT / "experiments" / "configs" / "phase3.yaml"
POPULATIONS = ("standard", "pairs_first", "connectors_first", "gear_change")
MISSPECIFIED = POPULATIONS[1:]
P32_NAME = {"standard": "standard", "pairs_first": "misspecified"}
RULES = ("lower_bound", "checked")
TOL = 1e-9
_RECOMMENDERS: dict[int, PushFoldRecommender] = {}


def test_bots(seed: int, cfg: dict[str, Any]) -> tuple[PushFoldPrior, list[dict[str, Any]]]:
    """The prior and the held-out bots of one replicate (P3.2's two populations, then the new)."""
    prior, p32 = sample_populations(seed, cfg)
    bots = [
        {
            "population": "standard" if pop == "standard" else "pairs_first",
            "index": k,
            "archetype": a,
            "params": params,
        }
        for pop, k, a, params in p32
    ]
    archs = list(ARCHETYPES)
    rng = np.random.default_rng([seed, 35])
    k = 0
    for a in archs:
        for _ in range(cfg["test_per_archetype"]):
            bots.append(
                {
                    "population": "connectors_first",
                    "index": k,
                    "archetype": a,
                    "params": sample_params(a, rng),
                }
            )
            k += 1
    rng = np.random.default_rng([seed, 36])
    k = 0
    for a in archs:
        for _ in range(cfg["test_per_archetype"]):
            first = sample_params(a, rng)
            second_arch = [b for b in archs if b != a][int(rng.integers(4))]
            bots.append(
                {
                    "population": "gear_change",
                    "index": k,
                    "archetype": a,
                    "params": first,
                    "second_archetype": second_arch,
                    "second_params": sample_params(second_arch, rng),
                }
            )
            k += 1
    return prior, bots


def action_logs(seed: int, bot: dict[str, Any], cfg: dict[str, Any]):
    """The hero's observations of ``bot`` at each sample size, and the gear-change hands."""
    sizes, pop, k = cfg["sizes"], bot["population"], bot["index"]
    hero = hero_strategy(float(cfg["stack"]))
    switch: dict[int, int] = {}
    if pop in P32_NAME:
        log = observation_log(seed, P32_NAME[pop], k, bot["params"], cfg)
        return {n: ActionLog.from_hand_log(log, n) for n in sizes}, switch
    if pop == "connectors_first":
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2, k]))
        log = simulate(strategy(bot["params"], "connectors_first"), hero, max(sizes), rng)
        return {n: ActionLog.from_hand_log(log, n) for n in sizes}, switch
    logs = {}
    for n in sizes:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 3, k, n]))
        tau = int(rng.integers(n // 4, 3 * n // 4 + 1))
        before = simulate(strategy(bot["params"]), hero, tau, rng)
        after = simulate(strategy(bot["second_params"]), hero, n - tau, rng, first_hand=tau)
        logs[n] = ActionLog.from_hand_log(HandLog.concat([before, after]))
        switch[n] = tau
    return logs, switch


def true_strategy(bot: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pop = bot["population"]
    if pop == "gear_change":
        return strategy(bot["second_params"])
    order = {"standard": "equity", "pairs_first": "pairs_first"}.get(pop, pop)
    return strategy(bot["params"], order)


def calibration_task(seed: int, k: int, prior: PushFoldPrior, cfg: dict[str, Any]):
    rng = np.random.default_rng([seed, 40, k])
    hero = hero_strategy(float(cfg["stack"]))
    return calibration_pvalues(
        prior, hero, cfg["sizes"], rng, draws=cfg["check_draws"], posterior_draws=cfg["draws"]
    )


def recommender(seed, prior, thresholds, cfg) -> PushFoldRecommender:
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


def bot_task(
    seed: int,
    bot: dict[str, Any],
    prior: PushFoldPrior,
    thresholds: dict[int, dict[str, float]],
    cfg: dict[str, Any],
    example: bool,
) -> dict[str, Any]:
    stack = float(cfg["stack"])
    rec = recommender(seed, prior, thresholds, cfg)
    logs, switch = action_logs(seed, bot, cfg)
    truth = true_strategy(bot)
    eq_value = float(match_values(stack, *rec.equilibrium, *truth))

    def true_gain(push, call) -> float:
        return float(match_values(stack, push, call, *truth)) - eq_value

    sb, bb = hand_gains(stack, *truth)
    oracle = true_gain((sb > 0).astype(float), (np.nan_to_num(bb, nan=-1.0) > 0).astype(float))
    pop_index = POPULATIONS.index(bot["population"])
    record: dict[str, Any] = {
        "bot_id": f"{bot['population'][0]}{bot['index']:03d}",
        "population": bot["population"],
        "archetype": bot["archetype"],
        "params": bot["params"].as_array(),
        "oracle_gain": oracle,
        "sizes": {},
    }
    if bot["population"] == "gear_change":
        record["second_archetype"] = bot["second_archetype"]
        record["second_params"] = bot["second_params"].as_array()
    for n in cfg["sizes"]:
        report = rec.recommend(logs[n], seed=[seed, 50 + pop_index, bot["index"]])
        fit = report.fit
        entry: dict[str, Any] = {
            "p": {r.name: r.p_value for r in fit.results},
            "data": {r.name: r.n for r in fit.results},
            "failed": {r.name: r.failed for r in fit.results},
            "flagged": fit.flagged,
            "ranked": [r.name for r in report.ranked],
            "candidates": {
                r.name: {
                    "mean": r.gain.mean,
                    "lower": r.gain.lower,
                    "upper": r.gain.upper,
                    "exploitability": r.exploitability,
                    "true_gain": true_gain(r.push, r.call),
                }
                for r in report.ranked
            },
        }
        if n in switch:
            entry["switch"] = switch[n]
        if example:
            entry["explanation"] = report.explanation()
        record["sizes"][str(n)] = entry
    return record


# ------------------------------------------------------------------------------- aggregation


def _ci(values) -> dict[str, float]:
    iv = t_interval(np.asarray(values, dtype=float))
    return {"mean": iv.estimate, "lower": iv.lower, "upper": iv.upper, "n": iv.n}


def _share(k: int, n: int) -> dict[str, float]:
    iv = wilson_interval(int(k), int(n))
    share = k / n if n else float("nan")
    return {"share": share, "lower": iv.lower, "upper": iv.upper, "k": int(k), "n": int(n)}


def rule_score(b: dict[str, Any], n: str, rule: str) -> dict[str, float]:
    s = b["sizes"][n]
    if rule == "checked" and s["flagged"]:
        return s["candidates"]["equilibrium"]
    return s["candidates"][s["ranked"][0]]


def _coverage(bots: list[dict[str, Any]], n: str) -> dict[str, float]:
    cands = [
        c for b in bots for name, c in b["sizes"][n]["candidates"].items() if name != "equilibrium"
    ]
    inside = sum(c["lower"] - 1e-12 <= c["true_gain"] <= c["upper"] + 1e-12 for c in cands)
    return _share(inside, len(cands))


def _size_block(groups: list[list[dict[str, Any]]], n: str) -> dict[str, Any]:
    flat = [b for g in groups for b in g]

    def seed_ci(fn):
        return _ci([np.mean([fn(b) for b in g]) for g in groups])

    flagged = [b for b in flat if b["sizes"][n]["flagged"]]
    clean = [b for b in flat if not b["sizes"][n]["flagged"]]
    return {
        "flags": {
            **{c: _share(sum(b["sizes"][n]["failed"][c] for b in flat), len(flat)) for c in CHECKS},
            "either": _share(len(flagged), len(flat)),
        },
        "rules": {
            rule: {
                "gain": seed_ci(lambda b, rule=rule: rule_score(b, n, rule)["true_gain"]),
                "exploitability": seed_ci(
                    lambda b, rule=rule: rule_score(b, n, rule)["exploitability"]
                ),
                "loses": _share(
                    sum(rule_score(b, n, rule)["true_gain"] < -TOL for b in flat), len(flat)
                ),
            }
            for rule in RULES
        },
        "oracle_gain": seed_ci(lambda b: b["oracle_gain"]),
        "coverage": {
            "all": _coverage(flat, n),
            "flagged": _coverage(flagged, n),
            "unflagged": _coverage(clean, n),
        },
    }


def aggregate(per_seed: list[list[dict[str, Any]]], cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pop in POPULATIONS:
        groups = [[b for b in bots if b["population"] == pop] for bots in per_seed]
        out[pop] = {n: _size_block(groups, n) for n in map(str, cfg["sizes"])}
    mis = [b for bots in per_seed for b in bots if b["population"] in MISSPECIFIED]
    out["misspecified_pooled"] = {
        n: {
            "flagged": _coverage([b for b in mis if b["sizes"][n]["flagged"]], n),
            "unflagged": _coverage([b for b in mis if not b["sizes"][n]["flagged"]], n),
        }
        for n in map(str, cfg["sizes"])
    }
    return out


def predictions(agg: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered predictions, evaluated."""
    sizes = [str(n) for n in cfg["sizes"]]

    def either(pop: str, n: str) -> float:
        return agg[pop][n]["flags"]["either"]["share"]

    def at_least(pop: str, n: str, bound: float) -> dict[str, Any]:
        if n not in agg[pop]:
            return {"value": None, "bound": bound, "holds": None}
        return {"value": either(pop, n), "bound": bound, "holds": either(pop, n) >= bound}

    false_alarm = max(either("standard", n) for n in sizes)
    pooled = agg["misspecified_pooled"]
    coverage = {
        n: {
            "flagged": pooled[n]["flagged"]["share"],
            "unflagged": pooled[n]["unflagged"]["share"],
            "holds": pooled[n]["flagged"]["n"] > 0
            and pooled[n]["flagged"]["share"] < pooled[n]["unflagged"]["share"],
        }
        for n in sizes
        if int(n) >= 100
    }
    out = {
        "false_alarms": {"max": false_alarm, "bound": 0.10, "holds": false_alarm <= 0.10},
        "connectors_100": at_least("connectors_first", "100", 0.60),
        **{
            f"connectors_{n}": at_least("connectors_first", n, 0.90) for n in sizes if int(n) >= 300
        },
        "gear_100": at_least("gear_change", "100", 0.40),
        **{f"gear_{n}": at_least("gear_change", n, 0.85) for n in sizes if int(n) >= 1000},
        "coverage": {"by_size": coverage, "holds": all(c["holds"] for c in coverage.values())},
    }
    return out


def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase3" / "fitcheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()
    seeds = derive_seeds(seed, cfg["seeds"])
    setups = {s: test_bots(s, cfg) for s in seeds}
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
            for bot in bots:
                example = (
                    s == seeds[0]
                    and bot["population"] in ("connectors_first", "gear_change")
                    and bot["index"] % cfg["test_per_archetype"] == 0
                )
                tasks.append((s, bot, prior, thresholds[s], cfg, example))
        records = list(pool.map(bot_task, *zip(*tasks, strict=True), chunksize=1))
    per_seed = [[r for r, t in zip(records, tasks, strict=True) if t[0] == s] for s in seeds]
    calibration = {
        str(s): {
            "thresholds": {str(n): t for n, t in thresholds[s].items()},
            "median_p": {
                str(n): {
                    c: float(
                        np.median(
                            [p[n][c] for p, t in zip(calib, calib_tasks, strict=True) if t[0] == s]
                        )
                    )
                    for c in CHECKS
                }
                for n in cfg["sizes"]
            },
        }
        for s in seeds
    }
    agg = aggregate(per_seed, cfg)
    results = {
        "meta": meta,
        "seeds": seeds,
        "calibration": calibration,
        "aggregate": agg,
        "predictions": predictions(agg, cfg),
        "per_seed": per_seed,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    save_json(results, out_dir / "fitcheck.json")
    (out_dir / "fitcheck_summary.md").write_text(
        markdown_summary(results), encoding="utf-8", newline="\n"
    )
    make_figures(results, out_dir)
    return results


# ---------------------------------------------------------------------------------- reports

POP_LABELS = {
    "standard": "Standard (model right)",
    "pairs_first": "Pairs first",
    "connectors_first": "Connectors first",
    "gear_change": "Gear change",
}
RULE_LABELS = {"lower_bound": "Lower-bound pick (P3.4)", "checked": "Checked pick"}


def _fmt(d: dict[str, float]) -> str:
    return f"{d['mean']:+.3f} [{d['lower']:+.3f}, {d['upper']:+.3f}]"


def _pct(d: dict[str, float]) -> str:
    if not d["n"]:
        return "n/a"
    return f"{d['share']:.1%} [{d['lower']:.1%}, {d['upper']:.1%}] ({d['k']}/{d['n']})"


def _verdict(holds) -> str:
    return "n/a" if holds is None else ("holds" if holds else "FAILS")


def markdown_summary(r: dict[str, Any]) -> str:
    cfg = r["meta"]["config"]
    agg = r["aggregate"]
    sizes = [str(n) for n in cfg["sizes"]]
    pred = r["predictions"]
    lines = [
        "# P3.5: model-fit checks before exploiting",
        "",
        f"Stack {cfg['stack']} bb. {cfg['seeds']} seeds x {5 * cfg['test_per_archetype']} held-out "
        f"bots per population; thresholds calibrated per seed on {cfg['calibration_bots']} bots "
        f"simulated from the prior (flag level {cfg['fit_level']:.0%}, half per check); "
        f"{cfg['check_draws']} posterior predictive replicates per check. Shares pooled over bots "
        "with 95% Wilson intervals; gains in bb/hand over equilibrium (exact, against the bot's "
        "true current strategy) with 95% t-intervals across seeds.",
        "",
        "## Pre-registered predictions",
        "",
        f"* False alarms (standard bots flagged) at most 10% at every n: max "
        f"{pred['false_alarms']['max']:.1%}, {_verdict(pred['false_alarms']['holds'])}.",
    ]
    for key, label in [(k, k) for k in pred if k.startswith(("connectors_", "gear_"))]:
        p = pred[key]
        pop, n = label.rsplit("_", 1)
        value = "n/a" if p["value"] is None else f"{p['value']:.1%}"
        lines.append(
            f"* {pop.capitalize()} bots flagged at {int(n):,} hands at least {p['bound']:.0%}: "
            f"{value}, {_verdict(p['holds'])}."
        )
    cov = pred["coverage"]
    lines.append(
        "* Misspecified bots: 90% coverage lower when flagged than unflagged at every n >= 100: "
        + ", ".join(
            f"{int(n):,} hands {c['flagged']:.0%} vs {c['unflagged']:.0%}"
            for n, c in cov["by_size"].items()
        )
        + f"; {_verdict(cov['holds'])}."
    )
    lines += ["", "## Flag rates (either check)", ""]
    lines += ["| Hands | " + " | ".join(POP_LABELS[p] for p in POPULATIONS) + " |"]
    lines += ["|---|" + "---|" * len(POPULATIONS)]
    for n in sizes:
        lines.append(
            f"| {int(n):,} | "
            + " | ".join(_pct(agg[p][n]["flags"]["either"]) for p in POPULATIONS)
            + " |"
        )
    for c in CHECKS:
        lines += ["", f"Flag rates, {c.replace('_', ' ')} check alone:", ""]
        lines += ["| Hands | " + " | ".join(POP_LABELS[p] for p in POPULATIONS) + " |"]
        lines += ["|---|" + "---|" * len(POPULATIONS)]
        for n in sizes:
            lines.append(
                f"| {int(n):,} | "
                + " | ".join(f"{agg[p][n]['flags'][c]['share']:.1%}" for p in POPULATIONS)
                + " |"
            )
    lines += [
        "",
        "## Interval coverage, flagged vs unflagged (90% intervals, non-equilibrium candidates)",
        "",
        "| Hands | Population | All | Flagged | Unflagged |",
        "|---|---|---|---|---|",
    ]
    for n in sizes:
        for p in POPULATIONS:
            c = agg[p][n]["coverage"]
            lines.append(
                f"| {int(n):,} | {POP_LABELS[p]} | {_pct(c['all'])} | {_pct(c['flagged'])} | "
                f"{_pct(c['unflagged'])} |"
            )
    lines += [
        "",
        "## Acting on a flag: lower-bound pick vs checked pick",
        "",
        "| Hands | Population | Oracle gain | Lower-bound pick | Checked pick | "
        "Lower-bound loses | Checked loses |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in sizes:
        for p in POPULATIONS:
            x = agg[p][n]
            lines.append(
                f"| {int(n):,} | {POP_LABELS[p]} | {_fmt(x['oracle_gain'])} | "
                f"{_fmt(x['rules']['lower_bound']['gain'])} | "
                f"{_fmt(x['rules']['checked']['gain'])} | "
                f"{x['rules']['lower_bound']['loses']['share']:.1%} | "
                f"{x['rules']['checked']['loses']['share']:.1%} |"
            )
    lines += ["", "## Calibrated thresholds (first seed)", ""]
    first = r["calibration"][str(r["seeds"][0])]
    lines += ["| Hands | " + " | ".join(f"{c} threshold | {c} median p" for c in CHECKS) + " |"]
    lines += ["|---|" + "---|" * (2 * len(CHECKS))]
    for n in sizes:
        cells = []
        for c in CHECKS:
            cells += [f"{first['thresholds'][n][c]:.3f}", f"{first['median_p'][n][c]:.3f}"]
        lines.append(f"| {int(n):,} | " + " | ".join(cells) + " |")
    examples = [b for b in r["per_seed"][0] if "explanation" in b["sizes"][sizes[-1]]]
    if examples:
        lines += ["", "## Example reports (first seed)", ""]
        for b in examples:
            for n in ("100", "1000"):
                if n in b["sizes"]:
                    who = f"{b['population'].replace('_', ' ')} {b['archetype']}"
                    if "second_archetype" in b:
                        who += f" -> {b['second_archetype']}"
                    lines += [f"* **{who}, {int(n):,} hands.** {b['sizes'][n]['explanation']}"]
    return "\n".join(lines) + "\n"


def make_figures(r: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import band, legend, line, new_figure, save_both, style_axes, title

    cfg = r["meta"]["config"]
    sizes = list(cfg["sizes"])
    x = np.array(sizes, dtype=float)
    agg = r["aggregate"]

    def flags(theme):
        fig, ax = new_figure(theme, 1, 1, figsize=(8, 4.6))
        style_axes(ax, theme)
        s = theme["series"]
        for k, pop in enumerate(POPULATIONS):
            d = [agg[pop][str(n)]["flags"]["either"] for n in sizes]
            band(ax, x, [100 * v["lower"] for v in d], [100 * v["upper"] for v in d], s[k])
            line(ax, theme, x, [100 * v["share"] for v in d], s[k], POP_LABELS[pop])
        ax.axhline(100 * cfg["fit_level"], color=theme["muted"], linewidth=1, linestyle=":")
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Hands observed")
        ax.set_ylabel("Bots flagged by either check (%)")
        legend(ax, theme, loc="upper left")
        title(
            fig,
            theme,
            "Which opponents fail the model-fit checks?",
            "Dotted: the calibrated false-alarm level. Bands: 95% Wilson intervals.",
        )
        return fig

    def consequences(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(12.5, 4.6))
        s = theme["series"]
        ax = axes[0]
        style_axes(ax, theme)
        pooled = agg["misspecified_pooled"]
        for key, c, dashed in (("unflagged", 0, False), ("flagged", 1, False)):
            pts = [(n, pooled[str(n)][key]) for n in sizes if pooled[str(n)][key]["n"]]
            line(
                ax,
                theme,
                [p[0] for p in pts],
                [100 * p[1]["share"] for p in pts],
                s[c],
                f"{key.capitalize()} bots",
                linestyle="--" if dashed else "-",
            )
        ax.axhline(90, color=theme["muted"], linewidth=1, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlabel("Hands observed")
        ax.set_ylabel("90% interval coverage (%)")
        ax.set_title(
            "Misspecified bots: are flagged intervals worse?",
            loc="left",
            fontsize=10,
            color=theme["ink"],
        )
        legend(ax, theme, loc="lower left")
        ax = axes[1]
        style_axes(ax, theme)
        for k, pop in enumerate(POPULATIONS):
            for rule, dashed in (("lower_bound", False), ("checked", True)):
                y = [100 * agg[pop][str(n)]["rules"][rule]["loses"]["share"] for n in sizes]
                label = f"{POP_LABELS[pop]}, {'checked' if dashed else 'lower bound'}"
                line(ax, theme, x, y, s[k], label, linestyle="--" if dashed else "-")
        ax.set_xscale("log")
        ax.set_xlabel("Hands observed")
        ax.set_ylabel("Bots the pick loses to equilibrium against (%)")
        ax.set_title(
            "Solid: lower-bound pick. Dashed: equilibrium when flagged.",
            loc="left",
            fontsize=10,
            color=theme["ink"],
        )
        legend(ax, theme, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handlelength=3.5)
        title(fig, theme, "What a flag means, and what acting on it does", "")
        fig.subplots_adjust(bottom=0.32)
        return fig

    save_both(flags, out_dir, "fitcheck_flags")
    save_both(consequences, out_dir, "fitcheck_consequences")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)["fitcheck"]
    r = run(args.seed, cfg=cfg, workers=args.workers)
    print(markdown_summary(r))


if __name__ == "__main__":
    main()

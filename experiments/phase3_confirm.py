"""P3.5b: confirming P3.5's post-hoc fallback rule on fresh seeds.

    python -m experiments.phase3_confirm --seed 2026

P3.5 found, after seeing its results, that falling back to the equilibrium only when the
*stability* check fails kept most of the gain against bots with a wrong hand order and all of the
safety against bots that changed gear. A rule chosen by looking at data must be tested on new
data before it is trusted. Pre-registered plan (written before this run):

* Rules, as in ``phase3_fitcheck``: **trust** (the recommender's top pick), **any check** (the
  equilibrium when either check fails: P3.5's pre-registered fallback and the library default)
  and **stability only** (the equilibrium only when the stability check fails).
* Everything as in P3.5 (``phase3_fitcheck.evaluate``: the four populations, calibration on 300
  bots per seed drawn from the prior, the recommender, both checks, the sample sizes) except the
  seeds. They come from ``SeedSequence([master_seed, 35])`` and were never used before, so the
  training populations, priors, test bots, hands and calibration bots are all new.
* Predictions (all three must hold):
  1. *Safety against gear-changers.* Stability only loses to the equilibrium against at most
     10% of gear-change bots at 1,000 and at 3,000 hands.
  2. *Gain kept against wrong hand orders.* For pairs-first and connectors-first bots at 300,
     1,000 and 3,000 hands, stability only earns more than any check: the 95% t-interval of the
     per-seed mean difference lies above 0.
  3. *Safety against wrong hand orders.* Stability only loses to the equilibrium against at
     most 5% of pairs-first and of connectors-first bots at every sample size from 100 hands.
* Decision: if all three hold, ``PushFoldRecommender``'s default ``fallback_on`` becomes
  ``("stability",)``; otherwise it stays ``("shown_hands", "stability")``.
* Also reported, without predictions: the flag rates (a replication of P3.5's), and all three
  rules' gains and losses for every population.
"""

from __future__ import annotations

import argparse
import time
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
from experiments.phase3_fitcheck import (
    POP_LABELS,
    POPULATIONS,
    RULE_LABELS,
    RULES,
    TOL,
    aggregate,
    evaluate,
    rule_score,
)
from poker_lab.evaluation.intervals import t_interval

CONFIG = REPO_ROOT / "experiments" / "configs" / "phase3.yaml"
SALT = 35
WRONG_ORDER = ("pairs_first", "connectors_first")


def fresh_seeds(master_seed: int, n: int) -> list[int]:
    """Replicate seeds disjoint from every earlier Phase 3 experiment's."""
    seeds = [int(s) for s in np.random.SeedSequence([master_seed, SALT]).generate_state(n)]
    assert not set(seeds) & set(derive_seeds(master_seed, n)), "seed collision"
    return seeds


def _gain(b: dict[str, Any], n: str, rule: str) -> float:
    return rule_score(b, n, rule)["true_gain"]


def paired_difference(
    per_seed: list[list[dict[str, Any]]], pop: str, n: str, rule: str, baseline: str
) -> dict[str, float]:
    """Per-seed mean of ``rule``'s true gain minus ``baseline``'s, with a t-interval."""
    diffs = [
        np.mean([_gain(b, n, rule) - _gain(b, n, baseline) for b in seed if b["population"] == pop])
        for seed in per_seed
    ]
    iv = t_interval(np.asarray(diffs, dtype=float))
    return {"mean": iv.estimate, "lower": iv.lower, "upper": iv.upper, "n": iv.n}


def loss_share(per_seed: list[list[dict[str, Any]]], pop: str, n: str, rule: str) -> float:
    bots = [b for seed in per_seed for b in seed if b["population"] == pop]
    return float(np.mean([_gain(b, n, rule) < -TOL for b in bots]))


def predictions(per_seed: list[list[dict[str, Any]]], cfg: dict[str, Any]) -> dict[str, Any]:
    sizes = [str(n) for n in cfg["sizes"]]
    p1 = {
        n: {"value": loss_share(per_seed, "gear_change", n, "stability_only"), "bound": 0.10}
        for n in ("1000", "3000")
        if n in sizes
    }
    for v in p1.values():
        v["holds"] = v["value"] <= v["bound"]
    p2 = {
        pop: {
            n: paired_difference(per_seed, pop, n, "stability_only", "checked")
            for n in ("300", "1000", "3000")
            if n in sizes
        }
        for pop in WRONG_ORDER
    }
    for by_n in p2.values():
        for v in by_n.values():
            v["holds"] = v["lower"] > 0
    p3 = {
        pop: {
            n: {"value": loss_share(per_seed, pop, n, "stability_only"), "bound": 0.05}
            for n in sizes
            if int(n) >= 100
        }
        for pop in WRONG_ORDER
    }
    for by_n in p3.values():
        for v in by_n.values():
            v["holds"] = v["value"] <= v["bound"]
    out = {
        "gear_safety": {"by_size": p1, "holds": all(v["holds"] for v in p1.values())},
        "gain_kept": {
            "by_population": p2,
            "holds": all(v["holds"] for by_n in p2.values() for v in by_n.values()),
        },
        "order_safety": {
            "by_population": p3,
            "holds": all(v["holds"] for by_n in p3.values() for v in by_n.values()),
        },
    }
    # a prediction about a sample size the config does not run is not evaluated, never "holds"
    out["evaluated"] = all(n in sizes for n in ("100", "300", "1000", "3000"))
    out["all_hold"] = out["evaluated"] and all(
        out[k]["holds"] for k in ("gear_safety", "gain_kept", "order_safety")
    )
    if not out["evaluated"]:
        out["decision"] = "not evaluated: the config lacks a predicted sample size"
    elif out["all_hold"]:
        out["decision"] = 'default fallback_on becomes ("stability",)'
    else:
        out["decision"] = 'default fallback_on stays ("shown_hands", "stability")'
    return out


def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase3" / "confirm"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()
    seeds = fresh_seeds(seed, cfg["seeds"])
    per_seed, calibration = evaluate(seeds, cfg, workers)
    sizes = [str(n) for n in cfg["sizes"]]
    results = {
        "meta": meta,
        "seeds": seeds,
        "calibration": calibration,
        "aggregate": aggregate(per_seed, cfg),
        "differences": {
            pop: {
                n: {
                    base: paired_difference(per_seed, pop, n, "stability_only", base)
                    for base in ("checked", "lower_bound")
                }
                for n in sizes
            }
            for pop in POPULATIONS
        },
        "predictions": predictions(per_seed, cfg),
        "per_seed": per_seed,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    save_json(results, out_dir / "confirm.json")
    (out_dir / "confirm_summary.md").write_text(
        markdown_summary(results), encoding="utf-8", newline="\n"
    )
    make_figures(results, out_dir)
    return results


def _verdict(holds: bool) -> str:
    return "holds" if holds else "FAILS"


def markdown_summary(r: dict[str, Any]) -> str:
    cfg = r["meta"]["config"]
    sizes = [str(n) for n in cfg["sizes"]]
    agg = r["aggregate"]
    pred = r["predictions"]
    lines = [
        "# P3.5b: confirming the stability-only fallback on fresh seeds",
        "",
        f"{cfg['seeds']} fresh seeds x {5 * cfg['test_per_archetype']} bots per population, the "
        "P3.5 design otherwise unchanged. Gains in bb/hand over equilibrium against each bot's "
        "true current strategy; 95% t-intervals across seeds.",
        "",
        "## Pre-registered predictions",
        "",
        "1. Stability only loses to equilibrium against at most 10% of gear-change bots: "
        + ", ".join(
            f"{int(n):,} hands {v['value']:.1%}" for n, v in pred["gear_safety"]["by_size"].items()
        )
        + f"; {_verdict(pred['gear_safety']['holds'])}.",
        "2. Stability only earns more than any check against wrong hand orders (difference, "
        "95% interval above 0): "
        + "; ".join(
            f"{POP_LABELS[pop].lower()} "
            + ", ".join(
                f"{int(n):,} hands {v['mean']:+.3f} [{v['lower']:+.3f}, {v['upper']:+.3f}]"
                for n, v in by_n.items()
            )
            for pop, by_n in pred["gain_kept"]["by_population"].items()
        )
        + f"; {_verdict(pred['gain_kept']['holds'])}.",
        "3. Stability only loses to equilibrium against at most 5% of wrong-order bots from 100 "
        "hands: max "
        + ", ".join(
            f"{POP_LABELS[pop].lower()} {max(v['value'] for v in by_n.values()):.1%}"
            for pop, by_n in pred["order_safety"]["by_population"].items()
        )
        + f"; {_verdict(pred['order_safety']['holds'])}.",
        "",
        f"Decision: {pred['decision']}.",
        "",
        "## The three rules",
        "",
        "| Hands | Population | Oracle | "
        + " | ".join(RULE_LABELS[k] for k in RULES)
        + " | "
        + " | ".join(f"{RULE_LABELS[k]} loses" for k in RULES)
        + " |",
        "|---|---|---|" + "---|" * (2 * len(RULES)),
    ]
    for n in sizes:
        for pop in POPULATIONS:
            x = agg[pop][n]
            gains = " | ".join(f"{x['rules'][k]['gain']['mean']:+.3f}" for k in RULES)
            loses = " | ".join(f"{x['rules'][k]['loses']['share']:.1%}" for k in RULES)
            lines.append(
                f"| {int(n):,} | {POP_LABELS[pop]} | {x['oracle_gain']['mean']:+.3f} | "
                f"{gains} | {loses} |"
            )
    lines += ["", "## Flag rates (either check), a replication of P3.5", ""]
    lines += ["| Hands | " + " | ".join(POP_LABELS[p] for p in POPULATIONS) + " |"]
    lines += ["|---|" + "---|" * len(POPULATIONS)]
    for n in sizes:
        lines.append(
            f"| {int(n):,} | "
            + " | ".join(f"{agg[p][n]['flags']['either']['share']:.1%}" for p in POPULATIONS)
            + " |"
        )
    return "\n".join(lines) + "\n"


def make_figures(r: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import band, legend, line, new_figure, save_both, style_axes, title

    cfg = r["meta"]["config"]
    sizes = list(cfg["sizes"])
    x = np.array(sizes, dtype=float)
    diff = r["differences"]

    def draw(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(12.5, 4.6))
        s = theme["series"]
        ax = axes[0]
        style_axes(ax, theme)
        for k, pop in enumerate(POPULATIONS):
            d = [diff[pop][str(n)]["checked"] for n in sizes]
            band(ax, x, [v["lower"] for v in d], [v["upper"] for v in d], s[k])
            line(ax, theme, x, [v["mean"] for v in d], s[k], POP_LABELS[pop])
        ax.axhline(0, color=theme["axis"], linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Hands observed")
        ax.set_ylabel("Stability only minus any check (bb/hand)")
        ax.set_title(
            "Gain kept by not falling back on shown hands",
            loc="left",
            fontsize=10,
            color=theme["ink"],
        )
        legend(ax, theme, loc="upper left")
        ax = axes[1]
        style_axes(ax, theme)
        for k, pop in enumerate(POPULATIONS):
            y = [
                100 * r["aggregate"][pop][str(n)]["rules"]["stability_only"]["loses"]["share"]
                for n in sizes
            ]
            line(ax, theme, x, y, s[k], POP_LABELS[pop])
        for ref_value in (5, 10):
            ax.axhline(ref_value, color=theme["muted"], linewidth=1, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlabel("Hands observed")
        ax.set_ylabel("Bots lost to under stability only (%)")
        ax.set_title(
            "Safety of the stability-only rule", loc="left", fontsize=10, color=theme["ink"]
        )
        legend(ax, theme, loc="upper right")
        title(
            fig,
            theme,
            "Confirming the stability-only fallback on fresh seeds",
            "Bands: 95% t-intervals across seeds. Dotted: the 5% and 10% bounds of the "
            "predictions.",
        )
        return fig

    save_both(draw, out_dir, "confirm_rules")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)["confirm"]
    r = run(args.seed, cfg=cfg, workers=args.workers)
    print(markdown_summary(r))


if __name__ == "__main__":
    main()

"""P3.1: heads-up push/fold hold'em solved exactly and checked against published Nash charts.

    python -m experiments.phase3_pushfold --seed 2026

Analyses, fixed before the first run:

1. **Matchup table.** Monte Carlo standard errors, and agreement of each class's equity
   against a random hand with the independently computed Phase 1 table
   (``poker_lab/bots/data/preflop_equity.json``, different code and seed).
2. **Exact Nash on a stack grid.** The sequence-form LP at every stack from ``stack_min`` to
   ``stack_max`` bb (step ``stack_step``); each solution is checked by an exact best response.
   Recorded per stack: the small blind's game value, push share, call share.
3. **Per-hand charts vs HoldemResources.** For each hand, the stacks at which Nash pushes
   (calls) with probability >= 0.5, compared with the published no-ante charts
   (``experiments/references/hune_no_ante.json``): the length of the stack range on which the
   two disagree, in bb. *Post hoc*: each disagreement priced in EV (what the hand gains by
   playing at that stack against our Nash opponent), and the bb/hand a player would lose by
   following the reference wherever the two differ.
4. **CFR+ vs LP** at ``cfr_stacks``: exploitability of CFR+'s average strategy by iteration.
5. **Heuristics.** Exploitability of simple range rules: push/call the top x% of hands by
   equity against a random hand, with (a) Nash's own range sizes at each stack ("ranked") and
   (b) fixed sizes (``heuristics`` in the config). (a) isolates *which* hands from *how many*.

The solve is deterministic given the matchup table, whose own seed and sample count are
recorded in its metadata; ``--seed`` is recorded for uniformity.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import REPO_ROOT, RESULTS_ROOT, load_config, run_metadata, save_json
from poker_lab.games.evaluate import exploitability
from poker_lab.games.preflop import CLASSES, combo_counts, load_matchups
from poker_lab.games.push_fold import (
    BB,
    SB,
    aggressive_probabilities,
    hand_gains,
    push_fold_tree,
    range_share,
    threshold_policy,
)
from poker_lab.solvers.cfr import CFRSolver, log_checkpoints
from poker_lab.solvers.lp import solve_lp

CONFIG = REPO_ROOT / "experiments" / "configs" / "phase3.yaml"
REFERENCE = REPO_ROOT / "experiments" / "references" / "hune_no_ante.json"
PHASE1_TABLE = REPO_ROOT / "poker_lab" / "bots" / "data" / "preflop_equity.json"
GRID_RANKS = "AKQJT98765432"


def grid_hand(r: int, c: int) -> str:
    """13x13 chart layout: pairs on the diagonal, suited above it, offsuit below."""
    a, b = GRID_RANKS[min(r, c)], GRID_RANKS[max(r, c)]
    if r == c:
        return a + a
    return a + b + ("s" if c > r else "o")


def solve_task(stack: float) -> dict[str, Any]:
    tree = push_fold_tree(stack)
    sol = solve_lp(tree)
    push = aggressive_probabilities(sol.policy, SB)
    call = aggressive_probabilities(sol.policy, BB)
    sb_gain, bb_gain = hand_gains(stack, push, call)
    return {
        "stack": stack,
        "value": sol.value,
        "exploitability": exploitability(sol.policy),
        "push": push,
        "call": call,
        "sb_gain": sb_gain,
        "bb_gain": bb_gain,
    }


def cfr_task(stack: float, iterations: int, per_decade: int) -> dict[str, Any]:
    tree = push_fold_tree(stack)
    solver = CFRSolver(tree, "cfr+")
    t0 = time.perf_counter()
    curve = solver.run(log_checkpoints(iterations, per_decade))
    avg = solver.average_policy()
    return {
        "stack": stack,
        "curve": curve,
        "seconds": time.perf_counter() - t0,
        "push": aggressive_probabilities(avg, SB),
        "call": aggressive_probabilities(avg, BB),
    }


def ranked_probs(order: np.ndarray, share: float) -> np.ndarray:
    """Play classes in ``order`` until ``share`` of combos is covered; the last one mixed."""
    combos = combo_counts() / combo_counts().sum()
    probs = np.zeros(len(CLASSES))
    left = share
    for k in order:
        if left <= 1e-12:
            break
        take = min(1.0, left / combos[k])
        probs[k] = take
        left -= take * combos[k]
    return probs


def heuristic_task(stack: float, push_share: float, call_share: float) -> float:
    order = np.argsort(-load_matchups().equity_vs_random(), kind="stable")
    tree = push_fold_tree(stack)
    policy = threshold_policy(
        tree, push=ranked_probs(order, push_share), call=ranked_probs(order, call_share)
    )
    return exploitability(policy)


def parse_reference(value: str, top: float) -> list[tuple[float, float]]:
    """``"20+"`` -> [(0, top)]; ``"7.1 - 5.1, 2.3"`` -> [(0, 2.3), (5.1, 7.1)]; ``"4.2"``."""
    if value.endswith("+"):
        return [(0.0, top)]
    if "-" in value:
        span, low = value.split(",")
        hi, lo = (float(x) for x in span.split("-"))
        return [(0.0, float(low)), (lo, hi)]
    return [(0.0, float(value))]


def intervals(stacks: np.ndarray, on: np.ndarray) -> list[list[float]]:
    """Maximal runs of grid stacks where ``on`` is true, as [first, last] stack values."""
    runs, start = [], None
    for k, flag in enumerate(on):
        if flag and start is None:
            start = k
        if not flag and start is not None:
            runs.append([float(stacks[start]), float(stacks[k - 1])])
            start = None
    if start is not None:
        runs.append([float(stacks[start]), float(stacks[-1])])
    return runs


def compare(
    stacks: np.ndarray,
    probs: np.ndarray,
    reference: dict[str, str],
    step: float,
    gains: np.ndarray,
    weights: np.ndarray,
):
    """Per-hand stack ranges vs the reference chart. ``gains[t, k]``: what hand k gains by
    playing at stack t against our Nash opponent (bb); ``weights[t, k]``: probability per hand
    that the decision for k arises (post hoc: prices each disagreement in EV)."""
    top = float(stacks[-1])
    per_hand = {}
    cost = np.zeros(len(stacks))  # bb/hand lost by following the reference where it differs
    worst_gap = 0.0
    for k, cls in enumerate(CLASSES):
        ours = probs[:, k] >= 0.5
        ref_iv = parse_reference(reference[cls], top)
        ref = np.array([any(lo - 1e-9 <= s <= hi + 1e-9 for lo, hi in ref_iv) for s in stacks])
        # Stack ranges are compared on the grid; a hand pushed up to 7.0 bb by one chart and
        # 7.1 bb by the other disagrees on ~0.1 bb of stacks.
        differ = ours != ref
        gap = np.abs(np.nan_to_num(gains[:, k]))
        cost += np.where(differ, gap * weights[:, k], 0.0)
        max_gap = float(gap[differ].max()) if differ.any() else 0.0
        worst_gap = max(worst_gap, max_gap)
        per_hand[cls] = {
            "ours": intervals(stacks, ours),
            "reference": reference[cls],
            "disagreement_bb": round(float(np.sum(differ) * step), 4),
            "max_ev_gap_bb": max_gap,
        }
    d = np.array([v["disagreement_bb"] for v in per_hand.values()])
    summary = {
        "hands": len(d),
        "exact_on_grid": int(np.sum(d == 0)),
        "within_0_1bb": int(np.sum(d <= 0.1 + 1e-9)),
        "within_0_25bb": int(np.sum(d <= 0.25 + 1e-9)),
        "within_0_5bb": int(np.sum(d <= 0.5 + 1e-9)),
        "mean_bb": float(d.mean()),
        "max_bb": float(d.max()),
        "worst": sorted(per_hand, key=lambda c: -per_hand[c]["disagreement_bb"])[:12],
        # Post hoc: what the disagreements are worth.
        "max_ev_gap_bb": worst_gap,
        "max_cost_mbb_per_hand": float(cost.max() * 1000),
        "stack_of_max_cost": float(stacks[int(cost.argmax())]),
    }
    return {"per_hand": per_hand, "summary": summary}


def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase3" / "pushfold"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()
    m = load_matchups()

    # 1. Matchup table quality and agreement with the Phase 1 equity table.
    phase1 = json.loads(PHASE1_TABLE.read_text(encoding="utf-8"))
    p1 = phase1["equity"]
    ours = m.equity_vs_random()
    diff = np.array([ours[k] - p1[c] for k, c in enumerate(CLASSES)])
    off = ~np.eye(len(CLASSES), dtype=bool)
    matchups = {
        "meta": m.meta,
        "max_se": float(m.se[off].max()),
        "mean_se": float(m.se[off].mean()),
        "vs_phase1_max_abs": float(np.abs(diff).max()),
        "vs_phase1_mean_abs": float(np.abs(diff).mean()),
        "vs_phase1_worst": CLASSES[int(np.abs(diff).argmax())],
    }

    # 2. Exact Nash on the stack grid (and CFR+, heuristics) in worker processes.
    n = round((cfg["stack_max"] - cfg["stack_min"]) / cfg["stack_step"]) + 1
    stacks = np.round(cfg["stack_min"] + cfg["stack_step"] * np.arange(n), 4)
    workers = workers or cfg.get("workers")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        sweep = list(pool.map(solve_task, stacks.tolist(), chunksize=8))
        cfr_futures = [
            pool.submit(cfr_task, float(s), cfg["cfr_iterations"], cfg["cfr_per_decade"])
            for s in cfg["cfr_stacks"]
        ]
        nash_at = {r["stack"]: r for r in sweep}
        h_stacks = [float(s) for s in cfg["heuristic_stacks"]]
        h_jobs = {}
        for s in h_stacks:
            r = nash_at[min(nash_at, key=lambda x: abs(x - s))]
            h_jobs[("ranked", s)] = pool.submit(
                heuristic_task, s, range_share(r["push"]), range_share(r["call"])
            )
            for name, h in cfg["heuristics"].items():
                h_jobs[(name, s)] = pool.submit(heuristic_task, s, h["push"], h["call"])
        cfr = [f.result() for f in cfr_futures]
        h_vals = {k: f.result() for k, f in h_jobs.items()}

    push = np.array([r["push"] for r in sweep])
    call = np.array([r["call"] for r in sweep])
    sb_gain = np.array([r["sb_gain"] for r in sweep])
    bb_gain = np.array([r["bb_gain"] for r in sweep])
    # How often each decision arises per hand: the small blind's class is dealt; the big
    # blind's class is dealt *and* faces a push.
    step = cfg["stack_step"]
    w_sb = np.tile(m.deal_prob.sum(axis=1), (len(stacks), 1))
    w_bb = push @ m.deal_prob
    nash = {
        "stacks": stacks.tolist(),
        "value": [r["value"] for r in sweep],
        "exploitability": [r["exploitability"] for r in sweep],
        "push_share": [range_share(p) for p in push],
        "call_share": [range_share(c) for c in call],
    }
    display = {}
    for s in cfg["display_stacks"]:
        k = int(np.argmin(np.abs(stacks - s)))
        display[f"{stacks[k]:g}"] = {
            "push": [round(float(x), 4) for x in push[k]],
            "call": [round(float(x), 4) for x in call[k]],
        }

    # 3. Per-hand charts vs HoldemResources.
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    comparison = {
        "source": reference["source"],
        "retrieved": reference["retrieved"],
        "pusher": compare(stacks, push, reference["pusher"], step, sb_gain, w_sb),
        "caller": compare(stacks, call, reference["caller"], step, bb_gain, w_bb),
    }

    # 4. CFR+ vs LP.
    cfr_out = []
    for r in cfr:
        k = int(np.argmin(np.abs(stacks - r["stack"])))
        cfr_out.append(
            {
                "stack": r["stack"],
                "curve": r["curve"],
                "seconds": r["seconds"],
                "push_decisions_match_lp": int(np.sum((r["push"] >= 0.5) == (push[k] >= 0.5))),
                "call_decisions_match_lp": int(np.sum((r["call"] >= 0.5) == (call[k] >= 0.5))),
            }
        )

    # 5. Heuristics.
    heuristics = {
        name: {
            "stacks": h_stacks,
            "exploitability": [h_vals[(name, s)] for s in h_stacks],
            "description": (
                "top hands by equity vs a random hand, Nash's own push and call shares"
                if name == "ranked"
                else f"push top {cfg['heuristics'][name]['push']:.0%}, "
                f"call top {cfg['heuristics'][name]['call']:.0%} at every stack"
            ),
        }
        for name in ["ranked", *cfg["heuristics"]]
    }

    results = {
        "meta": meta,
        "classes": list(CLASSES),
        "matchups": matchups,
        "nash": nash,
        "display": display,
        "reference_comparison": comparison,
        "cfr": cfr_out,
        "heuristics": heuristics,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    save_json(results, out_dir / "pushfold.json")
    (out_dir / "pushfold_summary.md").write_text(
        markdown_summary(results), encoding="utf-8", newline="\n"
    )
    make_figures(results, out_dir)
    return results


def _at(nash: dict[str, Any], key: str, s: float) -> float:
    k = int(np.argmin(np.abs(np.array(nash["stacks"]) - s)))
    return nash[key][k]


def markdown_summary(r: dict[str, Any]) -> str:
    m, nash, cmp = r["matchups"], r["nash"], r["reference_comparison"]
    lines = [
        "# P3.1: heads-up push/fold hold'em, exact Nash",
        "",
        f"Matchup table: {m['meta']['n_samples']:,} boards per class pair (seed "
        f"{m['meta']['seed']}), max standard error {m['max_se']:.4f}. Equity vs a random hand "
        f"agrees with the Phase 1 table to {m['vs_phase1_max_abs']:.4f} at worst "
        f"({m['vs_phase1_worst']}), {m['vs_phase1_mean_abs']:.4f} on average.",
        "",
        f"Exact LP on {len(nash['stacks'])} stacks from {nash['stacks'][0]:g} to "
        f"{nash['stacks'][-1]:g} bb; worst exploitability of a solution "
        f"{max(nash['exploitability']):.1e} bb/hand.",
        "",
        "| Stack (bb) | SB value (bb/hand) | SB push share | BB call share |",
        "|---|---|---|---|",
    ]
    for s in (1.5, 2, 3, 5, 8, 10, 15, 20):
        lines.append(
            f"| {s:g} | {_at(nash, 'value', s):+.4f} | {_at(nash, 'push_share', s):.1%} | "
            f"{_at(nash, 'call_share', s):.1%} |"
        )
    lines += [
        "",
        f"Per-hand charts vs HoldemResources ({cmp['source']}, retrieved {cmp['retrieved']}):",
        "",
    ]
    lines += [
        "| Chart | Exact on the grid | Within 0.1 bb | Within 0.25 bb | Mean | Worst |",
        "|---|---|---|---|---|---|",
    ]
    for side in ("pusher", "caller"):
        sm = cmp[side]["summary"]
        n = sm["hands"]
        lines.append(
            f"| {side} | {sm['exact_on_grid']}/{n} | {sm['within_0_1bb']}/{n} | "
            f"{sm['within_0_25bb']}/{n} | {sm['mean_bb']:.3f} bb | "
            f"{sm['max_bb']:.2f} bb ({', '.join(sm['worst'][:3])}) |"
        )
    lines += [
        "",
        "CFR+ (average strategy) vs LP:",
        "",
        "| Stack | Iterations | Exploitability | Push decisions = LP | Call decisions = LP |",
        "|---|---|---|---|---|",
    ]
    for c in r["cfr"]:
        last = c["curve"][-1]
        lines.append(
            f"| {c['stack']:g} | {last['iteration']:,} | {last['exploitability']:.1e} | "
            f"{c['push_decisions_match_lp']}/169 | {c['call_decisions_match_lp']}/169 |"
        )
    lines += ["", "Exploitability of heuristic range rules (bb/hand; Nash = 0):", ""]
    hs = r["heuristics"]
    stacks = next(iter(hs.values()))["stacks"]
    lines += [
        "| Rule | " + " | ".join(f"{s:g} bb" for s in stacks) + " |",
        "|---" * (len(stacks) + 1) + "|",
    ]
    for name, h in hs.items():
        lines.append(
            f"| {name}: {h['description']} | "
            + " | ".join(f"{v:.3f}" for v in h["exploitability"])
            + " |"
        )
    return "\n".join(lines) + "\n"


def make_figures(r: dict[str, Any], out_dir: Path) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    from experiments.plotting import THEMES, legend, line, new_figure, save_both, style_axes, title

    nash = r["nash"]
    stacks = np.array(nash["stacks"])

    def ranges(theme):
        fig, (a1, a2) = new_figure(theme, 1, 2, figsize=(12, 4.2))
        s = theme["series"]
        for ax in (a1, a2):
            style_axes(ax, theme)
        line(
            a1,
            theme,
            stacks,
            np.array(nash["push_share"]) * 100,
            s[0],
            "Small blind pushes",
            marker="",
        )
        line(
            a1,
            theme,
            stacks,
            np.array(nash["call_share"]) * 100,
            s[1],
            "Big blind calls",
            marker="",
        )
        a1.set_xlabel("Effective stack (bb)")
        a1.set_ylabel("Share of hands (%)")
        a1.set_ylim(0, 100)
        legend(a1, theme, loc="upper right")
        line(a2, theme, stacks, nash["value"], s[0], "Small blind's value", marker="")
        a2.axhline(0, color=theme["axis"], linewidth=0.8)
        a2.set_xlabel("Effective stack (bb)")
        a2.set_ylabel("Small blind's expected value (bb/hand)")
        title(
            fig,
            theme,
            "Heads-up push/fold: exact Nash ranges by stack size",
            "Sequence-form LP at every 0.05 bb; no ante, blinds 0.5/1",
        )
        return fig

    def chart(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(13, 6.4))
        for ax, side in zip(axes, ("pusher", "caller"), strict=True):
            per = r["reference_comparison"][side]["per_hand"]
            grid = np.zeros((13, 13))
            text = [["" for _ in range(13)] for _ in range(13)]
            for i in range(13):
                for j in range(13):
                    h = grid_hand(i, j)
                    iv = per[h]["ours"]
                    top = max((b for a, b in iv if a <= stacks[0] + 1e-9), default=0.0)
                    grid[i, j] = top
                    text[i][j] = f"{h}\n{'20+' if top >= stacks[-1] else f'{top:.1f}'}"
            cmap = LinearSegmentedColormap.from_list("seq", theme["seq"])
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=stacks[-1])
            # The ramp darkens with value on the light theme and lightens on the dark one.
            strong_ink = "#0b0b0b" if theme is THEMES["dark"] else "#ffffff"
            for i in range(13):
                for j in range(13):
                    strong = grid[i, j] > stacks[-1] * 0.55
                    ax.text(
                        j,
                        i,
                        text[i][j],
                        ha="center",
                        va="center",
                        fontsize=6.2,
                        color=strong_ink if strong else theme["ink"],
                    )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(
                "Small blind: push up to (bb)"
                if side == "pusher"
                else "Big blind: call up to (bb)",
                loc="left",
                fontsize=10,
                color=theme["ink"],
            )
        title(
            fig,
            theme,
            "Nash push/fold charts computed here",
            "Largest effective stack (bb) at which each hand pushes / calls; "
            "pairs on the diagonal, suited above it",
        )
        return fig

    def versus(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(11, 4.8))
        ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
        for ax, side, color in zip(axes, ("pusher", "caller"), theme["series"][:2], strict=True):
            style_axes(ax, theme, grid_axis="both")
            per = r["reference_comparison"][side]["per_hand"]
            xs, ys = [], []
            for h, v in per.items():
                rv = ref[side][h]
                if "-" in rv or rv.endswith("+"):
                    continue
                top = max((b for a, b in v["ours"] if a <= stacks[0] + 1e-9), default=0.0)
                xs.append(float(rv))
                ys.append(top)
            ax.plot([0, 20], [0, 20], color=theme["muted"], linewidth=1, linestyle="--")
            ax.scatter(xs, ys, s=22, color=color, edgecolor=theme["surface"], linewidth=1, zorder=3)
            ax.set_xlabel("HoldemResources chart (bb)")
            ax.set_ylabel("This project's LP solution (bb)")
            ax.set_title(
                "Small blind pushes" if side == "pusher" else "Big blind calls",
                loc="left",
                fontsize=10,
                color=theme["ink"],
            )
        title(
            fig,
            theme,
            "Per-hand thresholds vs a published Nash chart",
            "Hands with a single threshold below 20 bb; on the dashed line the two agree",
        )
        return fig

    def heur(theme):
        fig, ax = new_figure(theme, 1, 1, figsize=(7.5, 4.4))
        style_axes(ax, theme)
        for (name, h), color in zip(r["heuristics"].items(), theme["series"], strict=False):
            line(ax, theme, h["stacks"], h["exploitability"], color, f"{name}: {h['description']}")
        ax.set_yscale("log")
        ax.set_xlabel("Effective stack (bb)")
        ax.set_ylabel("Exploitability (bb/hand)")
        legend(ax, theme, loc="lower right")
        title(
            fig, theme, "How exploitable are simple range rules?", "Nash is 0; exact best responses"
        )
        return fig

    save_both(ranges, out_dir, "pushfold_ranges")
    save_both(chart, out_dir, "pushfold_chart")
    save_both(versus, out_dir, "pushfold_vs_reference")
    save_both(heur, out_dir, "pushfold_heuristics")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)["pushfold"]
    r = run(args.seed, cfg=cfg, workers=args.workers)
    print(markdown_summary(r))


if __name__ == "__main__":
    main()

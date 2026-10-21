"""P2.4 experiment: Deep CFR on Leduc hold'em vs tabular MCCFR at matched traversals.

For each derived seed, three runs share the same traversal budget (``K`` traversals per player
per iteration, ``T`` iterations, so ``2KT`` traversals in total):

* **Deep CFR** with neural advantage networks (``poker_lab.solvers.deep_cfr``). Reported:
  the policy network ``Pi`` (Deep CFR's output, trained at a few checkpoints), the exact
  average of the strategies its advantage networks played (what Single Deep CFR plays), the
  per-infoset average of the strategy memory (``Pi``'s regression target) and the current
  strategy.
* **Deep CFR with a perfect regressor**: the same traversals and memories, with each network
  replaced by the exact minimizer of its loss (per-infoset weighted means). The gap to Deep CFR
  is the cost of function approximation.
* **Tabular external-sampling MCCFR with linear weights**, ``KT`` iterations (2 traversals each),
  the closest tabular counterpart. The gap to the perfect regressor is the cost of updating the
  strategy only once per ``K`` traversals.

Exploitability is exact (chips per hand). Seeds are summarized by geometric means with 95%
t-intervals of log10 exploitability.

    python -m experiments.phase2_deep_cfr --seed 2026 [--quick]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import RESULTS_ROOT, derive_seeds, load_config, run_metadata, save_json
from poker_lab.games import LeducHoldem, compile_tree, exploitability

# torch is imported inside the Deep CFR task only: MCCFR workers do not need it.

METHOD_LABEL = {
    "policy_network": "Deep CFR: policy network (its output)",
    "exact_average": "Deep CFR: exact average of played strategies (SD-CFR)",
    "memory_average": "Deep CFR: strategy-memory average (Pi's target)",
    "current": "Deep CFR: current strategy",
    "perfect_regressor": "Deep CFR with a perfect regressor",
    "mccfr": "Tabular ES-MCCFR, linear weights",
}


def deep_cfr_task(seed: int, regressor: str, cfg: dict[str, Any]) -> dict[str, Any]:
    from poker_lab.solvers import DeepCFR, DeepCFRConfig, InfosetEncoder

    game = LeducHoldem()
    tree = compile_tree(game)
    config = DeepCFRConfig(
        traversals=cfg["traversals"],
        hidden=tuple(cfg["hidden"]),
        advantage_steps=cfg["advantage_steps"],
        policy_steps=cfg["policy_steps"],
        regressor=regressor,
        threads=1,
    )
    solver = DeepCFR(tree, InfosetEncoder.for_game(tree, game), config, seed)
    curve = solver.run(
        cfg["checkpoints"], exploitability, policy_checkpoints=cfg["policy_checkpoints"]
    )
    return {"seed": seed, "regressor": regressor, "curve": curve, "describe": solver.describe()}


def mccfr_task(seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    from poker_lab.solvers import ExternalSamplingMCCFR

    tree = compile_tree(LeducHoldem())
    solver = ExternalSamplingMCCFR(tree, seed, linear=True)
    curve, elapsed = [], 0.0
    for t in sorted(set(cfg["checkpoints"]) | set(cfg["policy_checkpoints"])):
        t0 = time.perf_counter()
        solver.iterate(cfg["traversals"] * t - solver.iteration)
        elapsed += time.perf_counter() - t0
        curve.append(
            {
                "iteration": t,  # in Deep CFR iterations (K MCCFR iterations each)
                "traversals": 2 * solver.iteration,
                "seconds": elapsed,
                "exploitability": exploitability(solver.average_policy()),
            }
        )
    return {"seed": seed, "curve": curve}


def _log_interval(values: list[float]) -> dict[str, float]:
    from poker_lab.evaluation.intervals import t_interval

    iv = t_interval(np.log10(np.asarray(values, dtype=float)))
    return {
        "geo_mean": float(10**iv.estimate),
        "lower": float(10**iv.lower),
        "upper": float(10**iv.upper),
        "n": len(values),
    }


def aggregate(network, tabular, mccfr) -> dict[str, list[dict[str, Any]]]:
    """Per method: geometric mean and 95% interval across seeds at each checkpoint."""
    series: dict[str, list[tuple[int, int, list[float]]]] = {}

    def collect(name: str, runs: list[dict[str, Any]], key: str) -> None:
        points = []
        for k, first in enumerate(runs[0]["curve"]):
            values = [r["curve"][k].get(key) for r in runs]
            if any(v is None for v in values):
                continue
            points.append((first["iteration"], first["traversals"], values))
        series[name] = points

    for key in ("policy_network", "exact_average", "memory_average", "current"):
        collect(key, network, key)
    collect("perfect_regressor", tabular, "exact_average")
    collect("mccfr", mccfr, "exploitability")
    return {
        name: [{"iteration": it, "traversals": tr, **_log_interval(v)} for it, tr, v in points]
        for name, points in series.items()
    }


def markdown_summary(results: dict[str, Any]) -> str:
    agg = results["aggregate"]
    cfg = results["meta"]["config"]
    report_at = cfg["policy_checkpoints"]
    describe = results["network_runs"][0]["describe"]
    seconds = [
        r["curve"][-1]["seconds"] / r["curve"][-1]["iteration"] for r in results["network_runs"]
    ]
    lines = [
        "# P2.4: Deep CFR on Leduc vs tabular MCCFR at matched traversals",
        "",
        f"{len(results['seeds'])} seeds; {cfg['traversals']:,} traversals per player per "
        f"iteration; advantage networks {tuple(cfg['hidden'])} "
        f"({describe['parameters_per_network']:,} parameters, {describe['features']} input "
        f"features), retrained from scratch for {cfg['advantage_steps']} Adam steps per "
        f"iteration; torch {describe['torch']}. Exploitability in chips per hand (geometric "
        "mean over seeds, 95% t-interval on the log scale). The uniform policy is at 2.37.",
        "",
        "| Method | "
        + " | ".join(f"{t} it. ({2 * cfg['traversals'] * t:,} trav.)" for t in report_at)
        + " |",
        "|---|" + "---|" * len(report_at),
    ]
    for name in (
        "mccfr",
        "perfect_regressor",
        "exact_average",
        "memory_average",
        "policy_network",
        "current",
    ):
        by_it = {p["iteration"]: p for p in agg[name]}
        cells = []
        for t in report_at:
            p = by_it.get(t)
            cells.append(
                "" if p is None else f"{p['geo_mean']:.3f} [{p['lower']:.3f}, {p['upper']:.3f}]"
            )
        lines.append(f"| {METHOD_LABEL[name]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Deep CFR wall-clock: {np.mean(seconds):.1f} s per iteration on average (both players, "
        "single thread; machine-dependent). Every other number is reproduced exactly by the "
        "seed on the same machine and thread count.",
        "",
    ]
    return "\n".join(lines)


def make_figure(results: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import band, legend, line, new_figure, save_both, style_axes, title

    agg = results["aggregate"]

    def draw(theme):
        fig, ax = new_figure(theme, figsize=(8.6, 4.8))
        style_axes(ax, theme, grid_axis="both")
        styles = [
            ("mccfr", theme["series"][5], "--", "o"),
            ("perfect_regressor", theme["series"][6], ":", "s"),
            ("exact_average", theme["series"][6], "-", "o"),
        ]
        for name, color, style, marker in styles:
            pts = agg[name]
            x = [p["traversals"] for p in pts]
            band(ax, x, [p["lower"] for p in pts], [p["upper"] for p in pts], color)
            y = [p["geo_mean"] for p in pts]
            label = METHOD_LABEL[name]
            line(ax, theme, x, y, color, label, linestyle=style, marker=marker, markevery=3)
        pts = agg["policy_network"]
        x = [p["traversals"] for p in pts]
        y = [p["geo_mean"] for p in pts]
        err = [
            [p["geo_mean"] - p["lower"] for p in pts],
            [p["upper"] - p["geo_mean"] for p in pts],
        ]
        ax.errorbar(
            x,
            y,
            yerr=err,
            fmt="D",
            color=theme["series"][6],
            markersize=7,
            markeredgecolor=theme["surface"],
            ecolor=theme["series"][6],
            capsize=3,
            label=METHOD_LABEL["policy_network"],
            zorder=5,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Traversals (both players)")
        ax.set_ylabel("Exploitability (chips / hand)")
        legend(ax, theme, loc="lower left")
        subtitle = (
            f"Matched traversals; geometric mean over {len(results['seeds'])} seeds with 95% "
            "intervals; exact exploitability"
        )
        title(fig, theme, "Deep CFR vs tabular MCCFR on Leduc hold'em", subtitle)
        return fig

    save_both(draw, out_dir, "deep_cfr_leduc")


def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase2" / "deep_cfr"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()
    seeds = derive_seeds(seed, cfg["seeds"])
    with ProcessPoolExecutor(max_workers=max(1, min(workers or 8, 3 * len(seeds)))) as pool:
        net_f = [pool.submit(deep_cfr_task, s, "network", cfg) for s in seeds]
        tab_f = [pool.submit(deep_cfr_task, s, "tabular", cfg) for s in seeds]
        mc_f = [pool.submit(mccfr_task, s, cfg) for s in seeds]
        network = [f.result() for f in net_f]
        tabular = [f.result() for f in tab_f]
        mccfr = [f.result() for f in mc_f]
    results = {
        "meta": meta,
        "seeds": seeds,
        "aggregate": aggregate(network, tabular, mccfr),
        "network_runs": network,
        "perfect_regressor_runs": tabular,
        "mccfr_runs": mccfr,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    save_json(results, out_dir / "deep_cfr.json")
    (out_dir / "deep_cfr_summary.md").write_text(markdown_summary(results), encoding="utf-8")
    make_figure(results, out_dir)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true", help="use configs/phase2_quick.yaml")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    from experiments.common import REPO_ROOT

    name = "phase2_quick.yaml" if args.quick else "phase2.yaml"
    cfg = load_config(REPO_ROOT / "experiments" / "configs" / name)
    out = RESULTS_ROOT / cfg["name"] / "deep_cfr"
    workers = args.workers or cfg["deep_cfr"].get("workers")
    results = run(args.seed, cfg=cfg["deep_cfr"], workers=workers, out_dir=out)
    print(markdown_summary(results))
    print(f"wrote {out} in {results['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()

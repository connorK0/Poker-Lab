"""P2.3 experiment: how fast do CFR variants converge on Kuhn poker and Leduc hold'em?

* Full-width variants (CFR, CFR+, linear CFR, discounted CFR) are deterministic: each runs once
  and its curve is exact.
* External-sampling MCCFR (plain and linearly weighted) is random: it runs once per derived
  seed. Exploitability spans orders of magnitude, so seeds are summarized on the log scale:
  the geometric mean with a 95% t-interval of log10(exploitability) across seeds.
* Ablation (post hoc, added after linear CFR stalled on Leduc in the first run): the 2 x 2 x 2
  grid of discounted-CFR exponents alpha in {1, 1.5} (positive regrets), beta in {0, 1}
  (negative regrets), gamma in {1, 2} (averaging). It contains linear CFR (1, 1, 1) and DCFR
  (1.5, 0, 2).

Every exploitability is exact (best responses on the full tree), in chips per hand, for the
*average* strategy. Work is measured in iterations and in tree nodes touched, the fair unit
for comparing full-width and sampled methods.

    python -m experiments.phase2_cfr --seed 2026 [--quick]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import RESULTS_ROOT, derive_seeds, load_config, run_metadata, save_json
from poker_lab.games import KuhnPoker, LeducHoldem, compile_tree, exploitability
from poker_lab.solvers import (
    CFRSolver,
    CFRVariant,
    ExternalSamplingMCCFR,
    log_checkpoints,
    solve_lp,
)

# matplotlib and scipy.stats are imported inside the functions that need them: worker
# processes re-import this module and should stay light.

GAMES = {"kuhn": KuhnPoker, "leduc": LeducHoldem}
GAME_LABEL = {"kuhn": "Kuhn poker", "leduc": "Leduc hold'em"}
VARIANT_LABEL = {
    "cfr": "CFR (vanilla)",
    "cfr+": "CFR+",
    "lcfr": "Linear CFR",
    "dcfr": "Discounted CFR",
    "es": "External-sampling MCCFR",
    "es_linear": "External-sampling MCCFR, linear weights",
}
REPORT_ITERATIONS = (10, 100, 1000, 10000)


def _timed_curve(solver, checkpoints: list[int]) -> list[dict[str, float]]:
    """Iterate to each checkpoint; time only the iterations, then evaluate exactly."""
    curve, elapsed = [], 0.0
    for target in checkpoints:
        t0 = time.perf_counter()
        solver.iterate(target - solver.iteration)
        elapsed += time.perf_counter() - t0
        curve.append(
            {
                "iteration": solver.iteration,
                "nodes_touched": solver.nodes_touched,
                "seconds": elapsed,
                "exploitability": exploitability(solver.average_policy()),
            }
        )
    return curve


def full_width_task(game: str, variant: str, iterations: int, per_decade: int) -> dict[str, Any]:
    tree = compile_tree(GAMES[game]())
    solver = CFRSolver(tree, variant)
    curve = _timed_curve(solver, log_checkpoints(iterations, per_decade))
    return {"game": game, "variant": variant, "curve": curve}


def ablation_task(alpha: float, beta: float, gamma: float, iterations: int) -> dict[str, Any]:
    tree = compile_tree(LeducHoldem())
    name = f"dcfr({alpha:g},{beta:g},{gamma:g})"
    solver = CFRSolver(tree, CFRVariant(name, alpha=alpha, beta=beta, gamma=gamma))
    curve = _timed_curve(solver, sorted({min(1000, iterations), iterations}))
    return {"alpha": alpha, "beta": beta, "gamma": gamma, "curve": curve}


def mccfr_task(
    game: str, seed: int, linear: bool, iterations: int, per_decade: int
) -> dict[str, Any]:
    tree = compile_tree(GAMES[game]())
    solver = ExternalSamplingMCCFR(tree, seed, linear=linear)
    curve = _timed_curve(solver, log_checkpoints(iterations, per_decade))
    return {"game": game, "variant": "es_linear" if linear else "es", "seed": seed, "curve": curve}


def _log_interval(values: list[float]) -> dict[str, float]:
    from poker_lab.evaluation.intervals import t_interval

    logs = np.log10(np.asarray(values, dtype=float))
    iv = t_interval(logs)
    return {
        "geo_mean": float(10**iv.estimate),
        "lower": float(10**iv.lower),
        "upper": float(10**iv.upper),
        "n": len(values),
    }


def aggregate_mccfr(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per (game, variant): geometric mean and 95% interval across seeds at each checkpoint."""
    out: dict[str, dict[str, Any]] = {}
    keys = sorted({(r["game"], r["variant"]) for r in runs})
    for game, variant in keys:
        group = [r for r in runs if (r["game"], r["variant"]) == (game, variant)]
        points = []
        for k, first in enumerate(group[0]["curve"]):
            values = [r["curve"][k]["exploitability"] for r in group]
            points.append(
                {
                    "iteration": first["iteration"],
                    "nodes_touched_mean": float(
                        np.mean([r["curve"][k]["nodes_touched"] for r in group])
                    ),
                    "seconds_mean": float(np.mean([r["curve"][k]["seconds"] for r in group])),
                    **_log_interval(values),
                }
            )
        out.setdefault(game, {})[variant] = {"seeds": [r["seed"] for r in group], "points": points}
    return out


def _at(curve: list[dict[str, float]], iteration: int, key: str = "exploitability"):
    for point in curve:
        if point["iteration"] == iteration:
            return point[key]
    return None


def markdown_summary(results: dict[str, Any]) -> str:
    lines = [
        "# P2.3: CFR convergence on Kuhn and Leduc",
        "",
        "Exploitability (NashConv / 2, chips per hand) of the average strategy. Full-width",
        "variants are deterministic; MCCFR rows are geometric means over seeds with 95%",
        "t-intervals on the log scale. Timings (ms per iteration) are wall-clock on the",
        "recording machine and vary between runs; every other number is reproduced exactly",
        "by the seed.",
        "",
    ]
    for game in results["full_width"]:
        lp = results["lp"][game]
        lines += [
            f"## {GAME_LABEL[game]}",
            "",
            f"Exact LP game value {lp['value']:.7f} (solved in {lp['seconds']:.2f} s).",
            "",
            "| Variant | " + " | ".join(f"{t:,} it." for t in REPORT_ITERATIONS) + " | ms / it. |",
            "|---|" + "---|" * (len(REPORT_ITERATIONS) + 1),
        ]
        for variant, curve in results["full_width"][game].items():
            cells = []
            for t in REPORT_ITERATIONS:
                v = _at(curve, t)
                cells.append("" if v is None else f"{v:.2e}")
            last = curve[-1]
            ms = 1000 * last["seconds"] / last["iteration"]
            lines.append(f"| {VARIANT_LABEL[variant]} | " + " | ".join(cells) + f" | {ms:.2f} |")
        lines.append("")
        for variant, agg in results["mccfr"].get(game, {}).items():
            last = agg["points"][-1]
            ms = 1000 * last["seconds_mean"] / last["iteration"]
            lines.append(
                f"* {VARIANT_LABEL[variant]} ({len(agg['seeds'])} seeds): after "
                f"{last['iteration']:,} iterations ({last['nodes_touched_mean']:,.0f} nodes "
                f"touched, {ms:.3f} ms / it.) exploitability {last['geo_mean']:.3g} "
                f"[{last['lower']:.3g}, {last['upper']:.3g}]."
            )
        lines.append("")
    if results.get("ablation"):
        its = [p["iteration"] for p in results["ablation"][0]["curve"]]
        lines += [
            "## Ablation: discounted-CFR exponents on Leduc (post hoc)",
            "",
            "alpha discounts positive regrets, beta negative regrets, and gamma sets the",
            "averaging weight t^gamma. Linear CFR is (1, 1, 1); DCFR is (1.5, 0, 2).",
            "",
            "| alpha | beta | gamma | " + " | ".join(f"{t:,} it." for t in its) + " |",
            "|---|---|---|" + "---|" * len(its),
        ]
        for r in results["ablation"]:
            cells = " | ".join(f"{p['exploitability']:.2e}" for p in r["curve"])
            lines.append(f"| {r['alpha']:g} | {r['beta']:g} | {r['gamma']:g} | {cells} |")
        lines.append("")
    return "\n".join(lines)


def make_figures(results: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import band, legend, line, new_figure, save_both, style_axes, title

    games = list(results["full_width"])

    def convergence(theme):
        fig, axes = new_figure(theme, 1, len(games), figsize=(10.5, 4.2), squeeze=False)
        for ax, game in zip(axes[0], games, strict=True):
            style_axes(ax, theme, grid_axis="both")
            for k, (variant, curve) in enumerate(results["full_width"][game].items()):
                x = [p["iteration"] for p in curve]
                y = [p["exploitability"] for p in curve]
                line(ax, theme, x, y, theme["series"][k], VARIANT_LABEL[variant], markevery=8)
            ref = results["full_width"][game]["cfr"]
            x0 = next(p for p in ref if p["iteration"] >= 10)
            xs = np.array([x0["iteration"], ref[-1]["iteration"]], dtype=float)
            for power, text in ((0.5, "1/sqrt(T)"), (1.0, "1/T")):
                ys = x0["exploitability"] * (xs / xs[0]) ** -power
                ax.plot(xs, ys, color=theme["muted"], linewidth=1, linestyle="--")
                xm = xs[0] * (xs[1] / xs[0]) ** 0.35  # label a third of the way along
                ym = x0["exploitability"] * (xm / xs[0]) ** -power
                ax.annotate(
                    text,
                    (xm, ym),
                    xytext=(6, 6),
                    textcoords="offset points",
                    color=theme["ink2"],
                    fontsize=8,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Exploitability (chips / hand)")
            ax.set_title(GAME_LABEL[game], loc="left", fontsize=10, color=theme["ink"])
            legend(ax, theme, loc="lower left")
        title(
            fig,
            theme,
            "CFR variants: exploitability of the average strategy",
            "Exact best-response evaluation; dashed lines show 1/sqrt(T) and 1/T rates",
        )
        return fig

    save_both(convergence, out_dir, "cfr_convergence")

    def sampled(theme):
        fig, axes = new_figure(theme, 1, len(games), figsize=(10.5, 4.2), squeeze=False)
        for ax, game in zip(axes[0], games, strict=True):
            style_axes(ax, theme, grid_axis="both")
            ref = results["full_width"][game]["cfr+"]
            x = [p["nodes_touched"] for p in ref]
            y = [p["exploitability"] for p in ref]
            line(ax, theme, x, y, theme["series"][1], "CFR+ (full width)", markevery=8)
            for k, (variant, agg) in enumerate(results["mccfr"].get(game, {}).items()):
                color = theme["series"][4 + k]
                x = [p["nodes_touched_mean"] for p in agg["points"]]
                lo = [p["lower"] for p in agg["points"]]
                hi = [p["upper"] for p in agg["points"]]
                band(ax, x, lo, hi, color)
                label = f"{VARIANT_LABEL[variant]} ({agg['points'][0]['n']} seeds)"
                y = [p["geo_mean"] for p in agg["points"]]
                style = "--" if variant == "es_linear" else "-"
                line(ax, theme, x, y, color, label, markevery=4, linestyle=style)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Tree nodes touched")
            ax.set_ylabel("Exploitability (chips / hand)")
            ax.set_title(GAME_LABEL[game], loc="left", fontsize=10, color=theme["ink"])
            legend(ax, theme, loc="lower left")
        title(
            fig,
            theme,
            "Sampled vs full-width CFR at equal work",
            "MCCFR: geometric mean over seeds with 95% interval; full-width CFR+ is exact",
        )
        return fig

    save_both(sampled, out_dir, "mccfr_vs_cfr")


def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase2" / "cfr"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()

    lp = {}
    for game, cls in GAMES.items():
        t0 = time.perf_counter()
        sol = solve_lp(compile_tree(cls()))
        lp[game] = {
            "value": sol.value,
            "exploitability": exploitability(sol.policy),
            "seconds": time.perf_counter() - t0,
        }

    seeds = derive_seeds(seed, cfg["mccfr_seeds"])
    full_tasks = [
        (game, variant, cfg["iterations"][game], cfg["per_decade"])
        for game in GAMES
        for variant in cfg["variants"]
    ]
    mc_tasks = [
        (game, s, linear, cfg["mccfr_iterations"][game], cfg["mccfr_per_decade"])
        for game in GAMES
        for linear in (False, True)
        for s in seeds
    ]
    ab_tasks = [
        (alpha, beta, gamma, cfg["ablation_iterations"])
        for alpha in (1.0, 1.5)
        for beta in (0.0, 1.0)
        for gamma in (1.0, 2.0)
    ]
    n_tasks = len(full_tasks) + len(mc_tasks) + len(ab_tasks)
    with ProcessPoolExecutor(max_workers=max(1, min(workers or 8, n_tasks))) as pool:
        full_futures = [pool.submit(full_width_task, *t) for t in full_tasks]
        mc_futures = [pool.submit(mccfr_task, *t) for t in mc_tasks]
        ab_futures = [pool.submit(ablation_task, *t) for t in ab_tasks]
        full = [f.result() for f in full_futures]
        mc = [f.result() for f in mc_futures]
        ablation = [f.result() for f in ab_futures]

    full_width: dict[str, dict[str, Any]] = {}
    for r in full:
        full_width.setdefault(r["game"], {})[r["variant"]] = r["curve"]
    results = {
        "meta": meta,
        "mccfr_seeds": seeds,
        "lp": lp,
        "full_width": full_width,
        "mccfr": aggregate_mccfr(mc),
        "mccfr_runs": mc,
        "ablation": ablation,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    save_json(results, out_dir / "cfr_convergence.json")
    (out_dir / "cfr_summary.md").write_text(markdown_summary(results), encoding="utf-8")
    make_figures(results, out_dir)
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
    out = RESULTS_ROOT / cfg["name"] / "cfr"
    results = run(args.seed, cfg=cfg["cfr"], workers=args.workers, out_dir=out)
    print(markdown_summary(results))
    print(f"wrote {out} in {results['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()

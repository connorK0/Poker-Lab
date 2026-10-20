"""Reproduce every Phase 1 experiment from one master seed.

    python -m experiments.run_phase1 --seed 2026              # full run (~25 min, 16 cores)
    python -m experiments.run_phase1 --seed 2026 --quick      # smoke test (~3 min)

Steps:
1. M3 sanity check: archetype bots vs intended tendencies -> ``<out>/bot_sanity/``.
2. For each of ``n_seeds`` seeds derived from the master seed: simulate disjoint train/test
   populations (cached in ``experiments/cache``), then run every analysis
   (``phase1_analysis``): hands-to-identify curves, calibration, shrinkage ablation, confusion,
   parameter regression, action prediction, oracle ceiling, permutation control.
3. Aggregate across seeds (95% t-intervals) and write ``phase1_results.json`` (with seed,
   config, git commit, package versions), ``phase1_summary.md`` and figures.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments.common import (
    REPO_ROOT,
    RESULTS_ROOT,
    derive_seeds,
    load_config,
    run_metadata,
    save_json,
)

CONFIG_DIR = REPO_ROOT / "experiments" / "configs"


def run(
    master_seed: int,
    cfg: dict,
    *,
    workers: int | None = None,
    use_cache: bool = True,
    skip_sanity: bool = False,
    out_dir: Path | None = None,
) -> dict:
    # Heavy imports live here: simulation workers re-import this module (Windows spawn) and
    # must not each load scikit-learn and matplotlib.
    from experiments import bot_sanity
    from experiments.phase1_analysis import analyze
    from experiments.phase1_data import generate
    from experiments.phase1_report import aggregate, make_figures, markdown_summary

    out_dir = out_dir or RESULTS_ROOT / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(master_seed, cfg)  # provenance of the code as the run starts
    if meta["git"]["dirty"]:
        print("WARNING: uncommitted changes to tracked files; results are not reproducible "
              "from the recorded commit", flush=True)  # fmt: skip
    t_start = time.perf_counter()
    if not skip_sanity:
        print("[1/3] bot sanity check", flush=True)
        report = bot_sanity.run(
            master_seed, cfg=cfg["sanity"], workers=workers, out_dir=out_dir / "bot_sanity"
        )
        s = report["summary"]
        print(
            f"      {s['cells_in_intended_range']}/{s['cells_with_intended_range']} cells in "
            f"range, {s['orderings_holding']}/{s['orderings_total']} orderings hold",
            flush=True,
        )

    seeds = derive_seeds(master_seed, cfg["n_seeds"])
    per_seed = []
    for i, seed in enumerate(seeds):
        t0 = time.perf_counter()
        print(f"[2/3] seed {i + 1}/{len(seeds)} ({seed}): simulating", flush=True)
        data = generate(seed, cfg, workers=workers, use_cache=use_cache)
        print(
            f"      analyzing ({data.train.hands_simulated + data.test.hands_simulated} hands)",
            flush=True,
        )
        result = analyze(data, cfg)
        per_seed.append(result)
        save_json(result, out_dir / "per_seed" / f"seed_{seed}.json")
        primary = cfg["primary_model"]
        acc = result["classification"][primary][str(cfg["sample_sizes"][-1])]["accuracy"]
        print(
            f"      done in {time.perf_counter() - t0:.0f}s; {primary} accuracy at "
            f"{cfg['sample_sizes'][-1]} hands = {acc:.3f}",
            flush=True,
        )

    print("[3/3] aggregating", flush=True)
    agg = aggregate(per_seed, cfg)
    results = {
        "meta": meta,
        "derived_seeds": seeds,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
        "aggregate": agg,
        "per_seed": per_seed,
    }
    save_json(results, out_dir / "phase1_results.json")
    (out_dir / "phase1_summary.md").write_text(markdown_summary(agg), encoding="utf-8")
    make_figures(per_seed, agg, out_dir, cfg)
    print(f"wrote {out_dir} in {results['elapsed_seconds']:.0f}s", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce all Phase 1 experiments from a seed.")
    ap.add_argument("--seed", type=int, default=2026, help="master seed")
    ap.add_argument("--config", type=Path, default=None, help="YAML config (default: phase1.yaml)")
    ap.add_argument("--quick", action="store_true", help="use configs/phase1_quick.yaml")
    ap.add_argument("--workers", type=int, default=None, help="simulation processes")
    ap.add_argument("--no-cache", action="store_true", help="re-simulate even if cached")
    ap.add_argument("--skip-sanity", action="store_true", help="skip the M3 sanity check")
    args = ap.parse_args()
    path = args.config or CONFIG_DIR / ("phase1_quick.yaml" if args.quick else "phase1.yaml")
    cfg = load_config(path)
    run(
        args.seed,
        cfg,
        workers=args.workers,
        use_cache=not args.no_cache,
        skip_sanity=args.skip_sanity,
    )


if __name__ == "__main__":
    main()

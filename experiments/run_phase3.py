"""Reproduce every Phase 3 experiment from one master seed.

    python -m experiments.run_phase3 --seed 2026

Steps (each writes JSON with seed, config, git commit and package versions, a markdown summary
and light/dark figures under ``experiments/results/phase3/<step>/``, then the results explorer
is rebuilt):

1. ``pushfold``: heads-up push/fold hold'em solved exactly on a stack grid, compared with
   published Nash charts, CFR+ vs LP, and the exploitability of simple range rules (P3.1).
2. ``exploit``: push/fold bots modeled from what a player sees (likelihood vs frequency
   models), exploited with best, restricted Nash and data-biased responses, with a
   misspecified population as a control (P3.2).
3. ``bridge``: the push/fold provider played through the real engine (PokerKit) behind the
   advisor firewall, checked against the exact abstract-game values, with four win-rate
   estimators (naive, duplicate, all-in EV, both) (P3.3).
4. ``recommend``: ranked exploit recommendations with posterior intervals for the same bots as
   ``exploit``, checked against each bot's true strategy: do responses ranked by a lower
   confidence bound really win, and are the intervals honest? (P3.4).
5. ``fitcheck``: posterior predictive checks of the opponent model, calibrated on simulated
   bots, against well-specified, mildly and badly misspecified, and gear-changing bots: false
   alarms, power, and what falling back to equilibrium on a flag costs or saves (P3.5).
6. ``confirm``: P3.5's post-hoc rule (fall back only when the stability check fails) on fresh
   seeds, with pre-registered predictions that decide the recommender's default (P3.5b).
7. ``import``: completed hand histories imported from PHH files: round trips at scale, fault
   injection with and without recorded finishing stacks, and the public PHH dataset's
   Pluribus, WSOP and historical hands at a pinned commit (P3.6).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments.common import REPO_ROOT, RESULTS_ROOT, load_config

CONFIG_DIR = REPO_ROOT / "experiments" / "configs"
STEPS = ("pushfold", "exploit", "bridge", "recommend", "fitcheck", "confirm", "import")


def run(
    master_seed: int,
    cfg: dict,
    *,
    steps: tuple[str, ...] = STEPS,
    workers: int | None = None,
    out_dir: Path | None = None,
) -> dict[str, float]:
    # Imported here: worker processes re-import this module (Windows spawn).
    from experiments import (
        phase3_bridge,
        phase3_confirm,
        phase3_exploit,
        phase3_fitcheck,
        phase3_import,
        phase3_pushfold,
        phase3_recommend,
    )

    out_dir = out_dir or RESULTS_ROOT / cfg["name"]
    modules = {
        "pushfold": phase3_pushfold,
        "exploit": phase3_exploit,
        "bridge": phase3_bridge,
        "recommend": phase3_recommend,
        "fitcheck": phase3_fitcheck,
        "confirm": phase3_confirm,
        "import": phase3_import,
    }
    elapsed = {}
    for k, step in enumerate(steps, start=1):
        print(f"[{k}/{len(steps)}] {step}", flush=True)
        t0 = time.perf_counter()
        section = cfg[step]
        modules[step].run(
            master_seed,
            cfg=section,
            workers=workers or section.get("workers"),
            out_dir=out_dir / step,
        )
        elapsed[step] = round(time.perf_counter() - t0, 1)
        print(f"      done in {elapsed[step]:.0f}s", flush=True)
    return elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce all Phase 3 experiments from a seed.")
    ap.add_argument("--seed", type=int, default=2026, help="master seed")
    ap.add_argument("--config", type=Path, default=None, help="YAML config (default: phase3.yaml)")
    ap.add_argument("--quick", action="store_true", help="use configs/phase3_quick.yaml")
    ap.add_argument("--workers", type=int, default=None, help="worker processes")
    ap.add_argument("--only", choices=STEPS, nargs="+", default=None, help="run some steps only")
    args = ap.parse_args()
    path = args.config or CONFIG_DIR / ("phase3_quick.yaml" if args.quick else "phase3.yaml")
    cfg = load_config(path)
    steps = tuple(args.only) if args.only else STEPS
    elapsed = run(args.seed, cfg, steps=steps, workers=args.workers)
    timing = ", ".join(f"{k} {v:.0f}s" for k, v in elapsed.items())
    print(f"wrote {RESULTS_ROOT / cfg['name']} ({timing})")
    from experiments.explorer.build import refresh

    refresh(RESULTS_ROOT / cfg["name"])


if __name__ == "__main__":
    main()

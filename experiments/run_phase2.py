"""Reproduce every Phase 2 experiment from one master seed.

    python -m experiments.run_phase2 --seed 2026            # full run (~15 min, 16 cores)
    python -m experiments.run_phase2 --seed 2026 --quick    # smoke test (~30 s)

Steps (each writes JSON with seed, config, git commit and package versions, a markdown
summary and light/dark figures under ``experiments/results/<name>/``):

1. ``cfr``: CFR, CFR+, linear and discounted CFR on Kuhn and Leduc (exact curves), external-
   sampling MCCFR over seeds, and the discounted-CFR exponent ablation (P2.3).
2. ``deep_cfr``: Deep CFR vs tabular MCCFR at matched traversals on Leduc, with the
   perfect-regressor ablation (P2.4). Needs the ``nn`` extra (PyTorch); skipped otherwise.
3. ``exploit``: modeling Leduc archetype bots from showdown-only observations and exploiting
   them with best, restricted Nash and data-biased responses, evaluated exactly (P2.6-P2.7).
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

from experiments.common import REPO_ROOT, RESULTS_ROOT, load_config

CONFIG_DIR = REPO_ROOT / "experiments" / "configs"
STEPS = ("cfr", "deep_cfr", "exploit")


def run(
    master_seed: int,
    cfg: dict,
    *,
    steps: tuple[str, ...] = STEPS,
    workers: int | None = None,
    out_dir: Path | None = None,
) -> dict[str, float]:
    # Experiment modules are imported here: worker processes re-import this module (Windows
    # spawn) and should not each load matplotlib, scikit-learn or torch.
    from experiments import phase2_cfr, phase2_deep_cfr, phase2_exploit

    out_dir = out_dir or RESULTS_ROOT / cfg["name"]
    modules = {"cfr": phase2_cfr, "deep_cfr": phase2_deep_cfr, "exploit": phase2_exploit}
    elapsed = {}
    for k, step in enumerate(steps, start=1):
        if step == "deep_cfr" and importlib.util.find_spec("torch") is None:
            print(f"[{k}/{len(steps)}] deep_cfr: skipped (install the nn extra for PyTorch)")
            continue
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
    ap = argparse.ArgumentParser(description="Reproduce all Phase 2 experiments from a seed.")
    ap.add_argument("--seed", type=int, default=2026, help="master seed")
    ap.add_argument("--config", type=Path, default=None, help="YAML config (default: phase2.yaml)")
    ap.add_argument("--quick", action="store_true", help="use configs/phase2_quick.yaml")
    ap.add_argument("--workers", type=int, default=None, help="worker processes")
    ap.add_argument("--only", choices=STEPS, nargs="+", default=None, help="run some steps only")
    args = ap.parse_args()
    path = args.config or CONFIG_DIR / ("phase2_quick.yaml" if args.quick else "phase2.yaml")
    cfg = load_config(path)
    steps = tuple(args.only) if args.only else STEPS
    elapsed = run(args.seed, cfg, steps=steps, workers=args.workers)
    timing = ", ".join(f"{k} {v:.0f}s" for k, v in elapsed.items())
    print(f"wrote {RESULTS_ROOT / cfg['name']} ({timing})")


if __name__ == "__main__":
    main()

"""P2.11, post hoc: is prediction F1's one miss sampling noise?

    python -m experiments.phase2_aivat_posthoc --seed 2026

Written after the real run (seed 2026 at 695cb26), so nothing here is pre-registered.

**The miss.** F1 predicted that P3.3's sampled standard deviations of all-in EV and duplicate +
all-in EV, from 100,000 real deals per matchup, are within 3% of the exact ones in every
matchup. 65 of 66 were; the exception is the Nash hero against Nash at 5 bb, duplicate + all-in
EV: sampled / exact 0.957.

**Why a fixed tolerance can fail.** With n independent units of variance sigma^2 and kurtosis
kappa (E[(X - mu)^4] / sigma^4), the sample variance has variance about sigma^4 (kappa - 1) / n,
so the sample SD has a relative standard error of about sqrt((kappa - 1) / (4n)). A normal
variable has kappa = 3: 0.2% at n = 100,000. An estimate that is almost always zero is
different: if a pair's two plays of a deal differ only on rare deals, kappa is large and the
sampled SD is noisy.

**What this computes.** For every one of F1's 66 estimates, the exact distribution of P3.3's
unit (one hand with the hero's seat a fair coin, or a duplicate pair's mean, the two plays
independent given the deal as in ``duplicate_match``) in the abstract push/fold game; its exact
SD (checked against ``aivat.json``, whose duplicate SDs are per hand: sqrt(2) times a pair's),
kurtosis, the sample SD's relative SE at P3.3's sample
size, and z = (sampled / exact - 1) / relative SE. For the two Nash-against-Nash duplicate
estimates with variance (5 and 10 bb), the sampling distribution of the ratio is also
simulated from the exact distribution.

**A simplification.** P3.3's hero alternates seats hand by hand rather than tossing a coin, so
its all-in EV sample is stratified by seat; the independent-draws standard error slightly
overstates the sampling error of those estimates. Duplicate pairs play both seats and are
unaffected.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import RESULTS_ROOT, run_metadata, save_json
from poker_lab.games import CHANCE, TERMINAL

AIVAT_JSON = RESULTS_ROOT / "phase2" / "aivat" / "aivat.json"
BRIDGE_JSON = RESULTS_ROOT / "phase3" / "bridge" / "bridge.json"
CONFIG = {"reps": 4000, "band": [0.97, 1.03], "zero_sd": 1e-9}


def _collapse(p: np.ndarray, v: np.ndarray, decimals: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Merge equal values (rounded) so the distribution has one entry per value."""
    keys, inverse = np.unique(np.round(v, decimals), return_inverse=True)
    return np.bincount(inverse, weights=p), keys


def single_distribution(hero, opp) -> tuple[np.ndarray, np.ndarray]:
    """(probability, value) of the hero's result in one hand, its seat a fair coin."""
    tree = hero.tree
    term = np.flatnonzero(tree.player == TERMINAL)
    ps, vs = [], []
    for s in (0, 1):
        edge = tree.edge_probabilities(hero.combine(opp, s).probs)
        ps.append(0.5 * tree.reach(edge)[term])
        vs.append((1.0 if s == 0 else -1.0) * tree.utility[term])
    return _collapse(np.concatenate(ps), np.concatenate(vs))


def pair_distribution(hero, opp) -> tuple[np.ndarray, np.ndarray]:
    """(probability, value) of a duplicate pair's mean: each deal played once in each seat, the
    two plays independent given the deal. Games without chance below the deal only."""
    tree = hero.tree
    top = 0
    while np.all(tree.player[tree.level(top)] == CHANCE):
        top += 1
    if np.any((tree.player == CHANCE) & (tree.depth > top)):
        raise ValueError("pair_distribution needs every chance node above the deal")
    deals = np.arange(tree.level_start[top], tree.level_start[top + 1])
    term = np.flatnonzero(tree.player == TERMINAL)
    deal_of = term.copy()
    while np.any(tree.depth[deal_of] > top):
        up = tree.depth[deal_of] > top
        deal_of[up] = tree.parent[deal_of[up]]
    chance = tree.reach(np.where(tree.edge_player == CHANCE, tree.chance_prob, 1.0))[deals]
    seat = {}
    for s in (0, 1):
        edge = tree.edge_probabilities(hero.combine(opp, s).probs)
        seat[s] = (tree.reach(edge)[term], (1.0 if s == 0 else -1.0) * tree.utility[term])
    order = np.argsort(deal_of, kind="stable")
    lo = np.searchsorted(deal_of[order], deals)
    hi = np.searchsorted(deal_of[order], deals, side="right")
    probs, values = [], []
    for pd, a, b in zip(chance, lo, hi, strict=True):
        if pd <= 0:
            continue
        idx = order[a:b]
        (r0, x0), (r1, x1) = ((seat[s][0][idx] / pd, seat[s][1][idx]) for s in (0, 1))
        k0, k1 = r0 > 0, r1 > 0
        probs.append((pd * np.outer(r0[k0], r1[k1])).ravel())
        values.append((0.5 * (x0[k0][:, None] + x1[k1][None, :])).ravel())
    return _collapse(np.concatenate(probs), np.concatenate(values))


def moments(p: np.ndarray, v: np.ndarray) -> dict[str, float]:
    mean = float(p @ v)
    var = float(p @ (v - mean) ** 2)
    kurt = float(p @ (v - mean) ** 4) / var**2 if var > 0 else math.nan
    return {"mean": mean, "sd": math.sqrt(var), "kurtosis": kurt}


def sd_rel_se(kurtosis: float, n: int) -> float:
    """Large-sample relative standard error of a sample SD: sqrt((kappa - 1) / (4n))."""
    return math.sqrt((kurtosis - 1.0) / (4.0 * n))


def simulate_sd_ratio(
    p: np.ndarray, v: np.ndarray, n: int, reps: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample SD of ``n`` independent draws / the exact SD, ``reps`` times (draws as
    multinomial counts over the distinct values: the same distribution, much faster)."""
    m = moments(p, v)
    out = np.empty(reps)
    step = max(1, 2_000_000 // len(p))  # keep each block of counts to a few tens of MB
    for a in range(0, reps, step):
        counts = rng.multinomial(n, p / p.sum(), size=min(step, reps - a)).astype(float)
        mean = counts @ v / n
        ss = counts @ v**2 - n * mean**2
        out[a : a + len(counts)] = np.sqrt(np.maximum(ss, 0.0) / (n - 1)) / m["sd"]
    return out


def run(seed: int, *, out_dir: Path | None = None) -> dict[str, Any]:
    from experiments.phase3_bridge import policies

    out_dir = out_dir or AIVAT_JSON.parent
    t_start = time.perf_counter()
    aivat = json.loads(AIVAT_JSON.read_text(encoding="utf-8"))
    n = json.loads(BRIDGE_JSON.read_text(encoding="utf-8"))["meta"]["config"]["n_deals"]
    lo, hi = CONFIG["band"]
    rows, simulated = [], []
    for k, c in enumerate(aivat["bridge_crosscheck"]):
        hero, opp = policies(c["stack"], c["hero"], c["opponent"])
        dist = pair_distribution if c["estimator"] == "duplicate" else single_distribution
        p, v = dist(hero, opp)
        m = moments(p, v)
        per_hand = m["sd"] * (math.sqrt(2.0) if c["estimator"] == "duplicate" else 1.0)
        if abs(per_hand - c["exact_sd"]) > 1e-9 * max(1.0, c["exact_sd"]):
            raise ValueError(f"exact SD {per_hand} does not match aivat.json's {c['exact_sd']}")
        has_var = m["sd"] >= CONFIG["zero_sd"]
        rel = sd_rel_se(m["kurtosis"], n) if has_var else None
        row = {
            key: c[key] for key in ("stack", "hero", "opponent", "estimator", "ratio", "exact_sd")
        }
        row.update(
            kurtosis=m["kurtosis"] if has_var else None,
            nonzero_share=float(p[np.abs(v - m["mean"]) > 1e-12].sum()),
            rel_se=rel,
            z=(c["ratio"] - 1.0) / rel if has_var else None,
        )
        rows.append(row)
        nash_pair = (c["hero"], c["opponent"], c["estimator"]) == ("nash", "nash", "duplicate")
        if has_var and nash_pair:
            rng = np.random.default_rng(np.random.SeedSequence([seed, k]))
            ratios = simulate_sd_ratio(p, v, n, CONFIG["reps"], rng)
            simulated.append(
                {
                    "stack": c["stack"],
                    "estimator": c["estimator"],
                    "observed": c["ratio"],
                    "rel_se": rel,
                    "quantiles": {
                        q: float(np.quantile(ratios, float(q))) for q in ("0.025", "0.5", "0.975")
                    },
                    "share_at_or_below_observed": float(np.mean(ratios <= c["ratio"])),
                    "share_at_or_above_observed": float(np.mean(ratios >= c["ratio"])),
                    "share_outside_band": float(np.mean((ratios < lo) | (ratios > hi))),
                    "sd_of_ratio": float(ratios.std(ddof=1)),
                }
            )
    zs = np.array([r["z"] for r in rows if r["z"] is not None])
    rel = np.array([r["rel_se"] for r in rows if r["rel_se"] is not None])
    results = {
        "meta": run_metadata(seed, {**CONFIG, "n_units": n}),
        "estimates": rows,
        "simulated": simulated,
        "summary": {
            "n": len(rows),
            "with_variance": len(zs),
            "outside_band": int(sum(not lo <= r["ratio"] <= hi for r in rows)),
            "max_abs_z": float(np.abs(zs).max()),
            "share_abs_z_above_1_96": float(np.mean(np.abs(zs) > 1.96)),
            "mean_z": float(zs.mean()),
            "sd_z": float(zs.std(ddof=1)),
            "rel_se_median": float(np.median(rel)),
            "rel_se_max": float(rel.max()),
            # estimates whose 95% sampling interval (+/-1.96 SE) is wider than F1's band
            "interval_wider_than_band": int(np.sum(1.96 * rel > (hi - lo) / 2)),
        },
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    save_json(results, out_dir / "aivat_posthoc.json")
    (out_dir / "aivat_posthoc_summary.md").write_text(
        markdown_summary(results), encoding="utf-8", newline="\n"
    )
    return results


def _who(r: dict[str, Any]) -> str:
    hero = {"nash": "Nash", "best_response": "best-response"}[r["hero"]]
    opp = {"nash": "Nash", "tag": "TAG", "lag": "LAG"}.get(r["opponent"], r["opponent"])
    opp = opp.replace("_", " ")
    est = {"chips": "all-in EV", "duplicate": "duplicate + all-in EV"}[r["estimator"]]
    return f"{r['stack']:g} bb, {hero} hero vs {opp}, {est}"


def markdown_summary(r: dict[str, Any]) -> str:
    s, cfg = r["summary"], r["meta"]["config"]
    lines = [
        "# P2.11, post hoc: F1's sampled SDs against their sampling error",
        "",
        f"Seed {r['meta']['seed']}, commit {(r['meta']['git']['commit'] or '?')[:7]}. Not "
        f"pre-registered. Sample SDs of {cfg['n_units']:,} units (P3.3) have a relative SE of "
        "about sqrt((kurtosis - 1) / (4n)); z = (sampled / exact - 1) / that SE.",
        "",
        f"{s['outside_band']} of {s['n']} estimates outside F1's band "
        f"[{cfg['band'][0]}, {cfg['band'][1]}]. Over the {s['with_variance']} with variance: max "
        f"|z| {s['max_abs_z']:.2f}, share |z| > 1.96 {s['share_abs_z_above_1_96']:.0%}, mean z "
        f"{s['mean_z']:+.2f}, SD of z {s['sd_z']:.2f}; relative SE median "
        f"{s['rel_se_median']:.4f}, max {s['rel_se_max']:.4f}; "
        f"{s['interval_wider_than_band']} estimates have a 95% sampling interval wider than the "
        "band.",
        "",
        "## Largest |z|",
        "",
        "| Estimate | Sampled / exact | Kurtosis | Relative SE | z |",
        "|---|---|---|---|---|",
    ]
    ranked = sorted((x for x in r["estimates"] if x["z"] is not None), key=lambda x: -abs(x["z"]))
    for x in ranked[:8]:
        lines.append(
            f"| {_who(x)} | {x['ratio']:.3f} | {x['kurtosis']:,.1f} | {x['rel_se']:.4f} | "
            f"{x['z']:+.2f} |"
        )
    lines += [
        "",
        f"## Simulated sampling distribution ({cfg['reps']:,} samples of {cfg['n_units']:,})",
        "",
        "| Estimate | Observed | 2.5% | median | 97.5% | P(<= observed) | P(outside band) |",
        "|---|---|---|---|---|---|---|",
    ]
    for x in r["simulated"]:
        q = x["quantiles"]
        who = _who({**x, "hero": "nash", "opponent": "nash"})
        lines.append(
            f"| {who} | {x['observed']:.3f} | {q['0.025']:.3f} | {q['0.5']:.3f} | "
            f"{q['0.975']:.3f} | {x['share_at_or_below_observed']:.3f} | "
            f"{x['share_outside_band']:.3f} |"
        )
    lines += ["", f"Run time {r['elapsed_seconds']:.0f} s."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    print(markdown_summary(run(args.seed, out_dir=args.out)))


if __name__ == "__main__":
    main()

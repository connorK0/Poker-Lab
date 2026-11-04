"""P2.11b, post hoc: reading the duplicate step's results, computed from its committed JSON.

    python -m experiments.phase2_duplicate_posthoc

Written after the real run (seed 2026 at 0fb6e53), so nothing here is pre-registered. It plays no
hands: every number comes from ``experiments/results/phase2/duplicate/duplicate.json``.

**The share of perfect cancellation.** A variance cannot be negative, so the mean correlation
between a player's results at the n tables of a deal is at least -1/(n - 1). The share
s = -(n - 1) rho is 0 for no cancellation and 1 at the bound, and it gives the ratio 1 / (1 - s) at
any n, so it compares table sizes where rho cannot.

**How far to trust a percentile bootstrap over lineups.** Normal theory for the mean of G values:
the bootstrap distribution has variance (G - 1) / G * s^2 / G, so read as normal its 95% interval
has half-width 1.96 sqrt((G - 1) / G) s / sqrt(G), where the exact interval has the t quantile.
Its coverage is P(|t_(G-1)| < 1.96 sqrt((G - 1) / G)). And empirically: take one lineup kind's
lineups as the truth, draw G of them with replacement, compute the interval as the step does, and
count how often it covers the kind's median. The same run's lineups are both the data and the
truth, so this shows the size of the effect, not the interval's exact coverage on new lineups.

**Why lineups, not players.** The interval from resampling players, for comparison.

**Other readings.** Heads-up lineups by archetype pair; the rank correlation of ratio and per-hand
SD over players; the spread of the offsets' correlations over players; the spread of ratios among
six copies of one bot, which differ only by sampling noise; how many distinct bots the mixed lineups
of a replicate seat.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from experiments import phase2_duplicate as pd
from experiments.common import RESULTS_ROOT, run_metadata, save_json

SOURCE = RESULTS_ROOT / "phase2" / "duplicate" / "duplicate.json"
CONFIG = {
    "normal_clusters": [5, 8, 10, 20, 50, 100],
    "lineup_clusters": [8, 10, 20, 50],
    "trials": 2000,
    "resamples": 500,
    "rng_seed": 20261103,
    "bootstrap": 2000,
}


def normal_coverage(clusters: int) -> float:
    """The coverage of a nominal 95% percentile bootstrap of a mean, from ``clusters`` normal
    values, reading the bootstrap distribution as normal."""
    q = stats.norm.ppf(0.975) * np.sqrt((clusters - 1) / clusters)
    return float(2 * stats.t.cdf(q, clusters - 1) - 1)


def lineup_coverage(
    ratios: np.ndarray, clusters: int, trials: int, resamples: int, rng: np.random.Generator
) -> float:
    """How often the percentile interval from ``clusters`` lineups covers the median of all of
    them. ``ratios``: lineups x players."""
    truth = np.median(ratios)
    hits = 0
    for _ in range(trials):
        sample = ratios[rng.integers(0, len(ratios), clusters)]
        drawn = rng.integers(0, clusters, (resamples, clusters))
        medians = np.median(sample[drawn].reshape(resamples, -1), axis=1)
        lower, upper = np.quantile(medians, [0.025, 0.975])
        hits += lower <= truth <= upper
    return hits / trials


def shares(by_kind: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for kind in pd.KINDS:
        n = pd.TABLE_SIZE[kind]
        s = [-(n - 1) * p["rho"] for row in rows if row["kind"] == kind for p in row["players"]]
        q25, median, q75 = np.quantile(s, [0.25, 0.5, 0.75])
        out[kind] = {
            "n": n,
            "bound": -1 / (n - 1),
            "rho_median": by_kind[kind]["rho"]["median"],
            "share": {"median": float(median), "q25": float(q25), "q75": float(q75)},
            "implied_ratio": float(1 / (1 - median)),
            "measured_ratio": by_kind[kind]["ratio"]["median"],
        }
    return out


def heads_up_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Heads-up the two players' ratios are equal, so a lineup is one value."""
    pairs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["kind"] == "mixed_2":
            key = tuple(sorted(p["archetype"] for p in row["players"]))
            pairs[key].append(row["players"][0]["ratio"])
    same = [v for k, vs in pairs.items() if k[0] == k[1] for v in vs]
    different = [v for k, vs in pairs.items() if k[0] != k[1] for v in vs]
    top = max(
        (row for row in rows if row["kind"] == "mixed_2"),
        key=lambda row: row["players"][0]["ratio"],
    )
    every = sorted(v for vs in pairs.values() for v in vs)
    return {
        "same_archetype": {"lineups": len(same), "median": float(np.median(same))},
        "different_archetypes": {"lineups": len(different), "median": float(np.median(different))},
        "pairs": {
            "|".join(k): {"lineups": len(v), "median": float(np.median(v)), "max": max(v)}
            for k, v in sorted(pairs.items())
        },
        "largest": {
            "ratio": top["players"][0]["ratio"],
            "archetypes": [p["archetype"] for p in top["players"]],
            "rho": top["players"][0]["rho"],
            "sigma": top["players"][0]["sigma"],
        },
        "next_largest_ratio": every[-2],
    }


def offset_spread(rows: list[dict[str, Any]], kind: str) -> list[dict[str, float]]:
    """The correlation at each reported offset over a kind's players: mean, median, quartiles and
    skewness. Medians of the offsets need not average to the median of rho."""
    n = pd.TABLE_SIZE[kind]
    out = []
    for k in range(n // 2):
        x = np.array(
            [p["rho_by_offset"][k] for row in rows if row["kind"] == kind for p in row["players"]]
        )
        q25, median, q75 = np.quantile(x, [0.25, 0.5, 0.75])
        out.append(
            {
                "mean": float(x.mean()),
                "median": float(median),
                "q25": float(q25),
                "q75": float(q75),
                "skewness": float(stats.skew(x)),
            }
        )
    return out


def run(seed_path: Path = SOURCE, *, out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = out_dir or seed_path.parent
    t0 = time.perf_counter()
    r = json.loads(seed_path.read_text(encoding="utf-8"))
    rows, by_kind = r["per_lineup"], r["aggregates"]["by_kind"]
    master = r["meta"]["seed"]
    rng = np.random.default_rng(CONFIG["rng_seed"])
    groups = {
        kind: np.array(
            [[p["ratio"] for p in row["players"]] for row in rows if row["kind"] == kind]
        )
        for kind in pd.KINDS
    }

    coverage = {
        kind: {
            str(g): lineup_coverage(groups[kind], g, CONFIG["trials"], CONFIG["resamples"], rng)
            for g in CONFIG["lineup_clusters"]
        }
        for kind in pd.KINDS
    }
    resampling = {}
    for kind in pd.KINDS:
        ratios = groups[kind]
        entropy = [master, pd.TAG_BOOTSTRAP, 99]
        lineup = pd.median_ci(ratios.tolist(), CONFIG["bootstrap"], entropy)
        player = pd.median_ci([[v] for v in ratios.ravel()], CONFIG["bootstrap"], entropy)
        resampling[kind] = {
            "lineups": lineup,
            "players": player,
            "width_ratio": (lineup["upper"] - lineup["lower"])
            / (player["upper"] - player["lower"]),
        }
    gaps = []
    for row in rows:
        n, d = row["n"], row["deals"]
        for p in row["players"]:
            identity = n * (d - 1) / ((n * d - 1) * (1 + (n - 1) * p["rho"]))
            gaps.append(abs(identity / p["ratio"] - 1))
    copies = groups["clones_6"]
    medians = np.median(copies, axis=1)
    lowest, highest = (
        next(
            row
            for row in rows
            if row["kind"] == "clones_6" and np.median([p["ratio"] for p in row["players"]]) == m
        )["archetype"]
        for m in (medians.min(), medians.max())
    )
    half_widths = {}
    for kind in pd.KINDS:
        players = [p for row in rows if row["kind"] == kind for p in row["players"]]
        half_widths[kind] = {
            name: float(
                np.median([(p["winrate"][name][1] - p["winrate"][name][0]) / 2 for p in players])
            )
            for name in ("naive", "duplicate")
        }
    intervals = [(e["ratio_ci"], e["ratio"]["median"]) for e in by_kind.values()] + [
        (a["ratio_ci"], a["ratio"]["median"])
        for e in by_kind.values()
        for a in e["by_archetype"].values()
    ]
    out = {
        "meta": {
            **run_metadata(master, CONFIG),
            "source": seed_path.name,
            "source_commit": r["meta"]["git"]["commit"],
        },
        "shares": shares(by_kind, rows),
        "coverage_normal": {str(g): normal_coverage(g) for g in CONFIG["normal_clusters"]},
        "coverage_lineups": coverage,
        "resampling": resampling,
        "heads_up": heads_up_pairs(rows),
        "rank_correlation_ratio_sd": {
            kind: float(
                stats.spearmanr(
                    [p["ratio"] for row in rows if row["kind"] == kind for p in row["players"]],
                    [p["sigma"] for row in rows if row["kind"] == kind for p in row["players"]],
                ).statistic
            )
            for kind in pd.KINDS
        },
        "offsets_random_mix": offset_spread(rows, "mixed_6"),
        "copies": {
            "within_lineup_relative_sd_median": float(
                np.median(copies.std(axis=1, ddof=1) / copies.mean(axis=1))
            ),
            "lineup_median_lowest": {"ratio": float(medians.min()), "archetype": lowest},
            "lineup_median_highest": {"ratio": float(medians.max()), "archetype": highest},
        },
        "mixed_6_distinct_bots": [
            len(
                {
                    p["bot_id"]
                    for row in rows
                    if row["kind"] == "mixed_6" and row["replicate"] == k
                    for p in row["players"]
                }
            )
            for k in range(r["meta"]["config"]["replicates"])
        ],
        "half_width_bb_per_100": half_widths,
        "identity_largest_relative_gap": max(gaps),
        "every_ci_contains_its_median": all(
            ci["lower"] <= median <= ci["upper"] for ci, median in intervals
        ),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    save_json(out, out_dir / "duplicate_posthoc.json")
    (out_dir / "duplicate_posthoc_summary.md").write_text(
        markdown_summary(out), encoding="utf-8", newline="\n"
    )
    return out


def markdown_summary(r: dict[str, Any]) -> str:
    lines = [
        "# P2.11b, post hoc: reading the duplicate step's results",
        "",
        f"Seed {r['meta']['seed']}, commit {(r['meta']['git']['commit'] or '?')[:7]}; from "
        f"{r['meta']['source']} at {(r['meta']['source_commit'] or '?')[:7]}. Nothing here is "
        "pre-registered.",
        "",
        "## The share of perfect cancellation, -(n - 1) rho",
        "",
        "| Lineup kind | n | Median rho | Bound | Share: median [IQR] | 1 / (1 - s) | "
        "Measured ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind, e in r["shares"].items():
        q = e["share"]
        lines.append(
            f"| {pd.LABELS[kind]} | {e['n']} | {e['rho_median']:+.3f} | {e['bound']:.2f} | "
            f"{q['median']:.1%} [{q['q25']:.1%}, {q['q75']:.1%}] | {e['implied_ratio']:.3f} | "
            f"{e['measured_ratio']:.3f} |"
        )
    clusters = list(next(iter(r["coverage_lineups"].values())))
    lines += [
        "",
        "## Coverage of a nominal 95% percentile bootstrap",
        "",
        "Normal theory, mean of G values: "
        + ", ".join(f"G = {g}: {c:.1%}" for g, c in r["coverage_normal"].items())
        + ".",
        "",
        "| Lineup kind | " + " | ".join(f"G = {g}" for g in clusters) + " |",
        "|---|" + "---|" * len(clusters),
    ]
    for kind, c in r["coverage_lineups"].items():
        lines.append(f"| {pd.LABELS[kind]} | " + " | ".join(f"{c[g]:.1%}" for g in clusters) + " |")
    lines += [
        "",
        "## Resampling lineups against resampling players",
        "",
        "| Lineup kind | Lineups | Players | Width, lineups / players |",
        "|---|---|---|---|",
    ]
    for kind, e in r["resampling"].items():
        a, b = e["lineups"], e["players"]
        lines.append(
            f"| {pd.LABELS[kind]} | [{a['lower']:.3f}, {a['upper']:.3f}] | "
            f"[{b['lower']:.3f}, {b['upper']:.3f}] | {e['width_ratio']:.2f} |"
        )
    h = r["heads_up"]
    c = r["copies"]
    lines += [
        "",
        "## Other readings",
        "",
        f"* Heads-up: same archetype {h['same_archetype']['median']:.2f}x "
        f"({h['same_archetype']['lineups']} lineups), different archetypes "
        f"{h['different_archetypes']['median']:.2f}x ({h['different_archetypes']['lineups']}); "
        f"largest {h['largest']['ratio']:.2f}x ({' against '.join(h['largest']['archetypes'])}, "
        f"rho {h['largest']['rho']:+.3f}), next {h['next_largest_ratio']:.2f}x.",
        "* Rank correlation of ratio and per-hand SD over players: "
        + ", ".join(f"{pd.LABELS[k]} {v:.2f}" for k, v in r["rank_correlation_ratio_sd"].items())
        + ".",
        f"* Six copies of one bot: median relative SD of the ratios within a lineup "
        f"{c['within_lineup_relative_sd_median']:.1%}; lineup medians from "
        f"{c['lineup_median_lowest']['ratio']:.2f}x ({c['lineup_median_lowest']['archetype']}) to "
        f"{c['lineup_median_highest']['ratio']:.2f}x ({c['lineup_median_highest']['archetype']}).",
        f"* Distinct bots in the 60 seats of a replicate's mixed_6 lineups: "
        f"{min(r['mixed_6_distinct_bots'])} to {max(r['mixed_6_distinct_bots'])}.",
        f"* The identity from rho: largest relative gap {r['identity_largest_relative_gap']:.1e}. "
        f"Every reported CI contains its median: {r['every_ci_contains_its_median']}.",
        "",
        f"Run time {r['elapsed_seconds']:.0f} s.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    print(markdown_summary(run(args.source, out_dir=args.out)))


if __name__ == "__main__":
    main()

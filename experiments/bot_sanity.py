"""M3 sanity check: do the archetype bots produce their intended tendencies?

Simulates a large league, measures every player's stats over all their hands, and reports per
archetype: mean of per-player stats with a 95% t-interval across players, the intended range
fixed in advance (``INTENDED_STATS``), the intended orderings, and Spearman correlations
between each bot parameter and the stat it is meant to drive.

    python -m experiments.bot_sanity --seed 2026 [--quick]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.common import RESULTS_ROOT, run_metadata, save_json
from poker_lab.bots import (
    ARCHETYPES,
    INTENDED_ORDERINGS,
    INTENDED_STATS,
    PARAM_NAMES,
    GroundTruth,
    make_population,
)
from poker_lab.bots.league import LeagueConfig, run_league
from poker_lab.profiling.stats import raw_stats

# scipy and matplotlib are imported inside the functions that use them: simulation workers
# re-import this module when it is run as a script and should stay light.

REPORT_STATS = (
    "vpip",
    "pfr",
    "three_bet",
    "fold_to_3bet",
    "af",
    "cbet",
    "fold_to_cbet",
    "fold_to_bet",
    "wtsd",
    "wsd",
    "bet_size",
    "open_size",
)
PERCENT = {s for s in REPORT_STATS if s not in ("af", "bet_size", "open_size")}
LABELS = {
    "vpip": "VPIP %",
    "pfr": "PFR %",
    "three_bet": "3-bet %",
    "fold_to_3bet": "Fold to 3-bet %",
    "af": "AF",
    "cbet": "C-bet %",
    "fold_to_cbet": "Fold to c-bet %",
    "fold_to_bet": "Fold to bet %",
    "wtsd": "WTSD %",
    "wsd": "W$SD %",
    "bet_size": "Bet size (x pot)",
    "open_size": "Open size (bb)",
}
ARCH_LABEL = {
    "nit": "Nit",
    "tag": "TAG",
    "lag": "LAG",
    "calling_station": "Station",
    "maniac": "Maniac",
}
# Parameter -> the stat it is designed to move most directly.
DRIVERS = (
    ("range_width", "vpip"),
    ("raise_share", "pfr_over_vpip"),
    ("three_bet_freq", "three_bet"),
    ("cbet_freq", "cbet"),
    ("fold_to_aggression", "fold_to_bet"),
    ("value_aggression", "aggr_share"),
    ("bluff_freq", "aggr_share"),
    ("bet_size_pot", "bet_size"),
)

DEFAULTS = {"n_per_archetype": 60, "hands_per_player": 3000, "block_hands": 50}
QUICK = {"n_per_archetype": 12, "hands_per_player": 300, "block_hands": 50}


def player_stats(counters: np.ndarray) -> list[dict[str, float]]:
    totals = counters.astype(np.int64).sum(axis=1)
    out = []
    for row in totals:
        s = raw_stats(row)
        s["pfr_over_vpip"] = s["pfr"] / s["vpip"] if s["vpip"] else float("nan")
        out.append(s)
    return out


def summarize(per_player: list[dict[str, float]], labels: list[str]) -> dict:
    from poker_lab.evaluation.intervals import t_interval

    table: dict[str, dict] = {}
    for a in ARCHETYPES:
        rows = [s for s, lab in zip(per_player, labels, strict=True) if lab == a.value]
        entry = {}
        for stat in REPORT_STATS:
            scale = 100.0 if stat in PERCENT else 1.0
            ci = t_interval([r[stat] * scale for r in rows])
            intended = INTENDED_STATS[a].get(stat)
            entry[stat] = {
                **ci.as_dict(),
                "player_sd": float(np.nanstd([r[stat] * scale for r in rows], ddof=1)),
                "intended": intended,
                "in_range": None if intended is None else intended[0] <= ci.estimate <= intended[1],
            }
        table[a.value] = entry
    return table


def check_orderings(table: dict) -> list[dict]:
    out = []
    for stat, hi, lo in INTENDED_ORDERINGS:
        a, b = table[hi.value][stat]["estimate"], table[lo.value][stat]["estimate"]
        out.append(
            {"stat": stat, "higher": hi.value, "lower": lo.value, "values": [a, b], "holds": a > b}
        )
    return out


def driver_correlations(per_player, params: np.ndarray, labels: list[str]) -> list[dict]:
    from scipy import stats as sps

    out = []
    for pname, stat in DRIVERS:
        x = params[:, PARAM_NAMES.index(pname)]
        y = np.array([s[stat] for s in per_player])
        ok = ~np.isnan(y)
        rho_all = sps.spearmanr(x[ok], y[ok]).statistic
        within = {}
        for a in ARCHETYPES:
            m = ok & (np.array(labels) == a.value)
            within[a.value] = float(sps.spearmanr(x[m], y[m]).statistic) if m.sum() > 3 else None
        out.append(
            {
                "param": pname,
                "stat": stat,
                "spearman_all": float(rho_all),
                "spearman_within_archetype": within,
            }
        )
    return out


def markdown_table(table: dict) -> str:
    stats = ("vpip", "pfr", "three_bet", "af", "cbet", "fold_to_cbet", "wtsd", "wsd")
    head = "| Archetype | " + " | ".join(LABELS[s] for s in stats) + " |"
    lines = [head, "|---" * (len(stats) + 1) + "|"]
    for a in ARCHETYPES:
        cells = []
        for s in stats:
            e = table[a.value][s]
            val = f"{e['estimate']:.1f}" if s != "af" else f"{e['estimate']:.2f}"
            if e["intended"] is not None:
                lo, hi = e["intended"]
                mark = "" if e["in_range"] else " ✗"
                val += f" ({lo:g}–{hi:g}){mark}"
            cells.append(val)
        lines.append(f"| {ARCH_LABEL[a.value]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot(per_player, labels, table, out_dir: Path) -> None:
    from experiments.plotting import band, new_figure, save_both, style_axes, title

    stats = ("vpip", "pfr", "three_bet", "af", "cbet", "fold_to_cbet", "wtsd", "wsd")
    rng = np.random.default_rng(0)

    def draw(theme):
        fig, axes = new_figure(theme, 2, 4, figsize=(13, 6.4))
        color = theme["series"][0]
        for ax, stat in zip(axes.flat, stats, strict=True):
            style_axes(ax, theme)
            scale = 100.0 if stat in PERCENT else 1.0
            for i, a in enumerate(ARCHETYPES):
                vals = np.array(
                    [
                        s[stat] * scale
                        for s, lab in zip(per_player, labels, strict=True)
                        if lab == a.value
                    ]
                )
                vals = vals[~np.isnan(vals)]
                intended = INTENDED_STATS[a].get(stat)
                if intended is not None:
                    band(
                        ax,
                        [i - 0.38, i + 0.38],
                        intended[0],
                        intended[1],
                        theme["wash"],
                        alpha=0.75,
                    )
                jitter = rng.uniform(-0.22, 0.22, len(vals))
                ax.scatter(i + jitter, vals, s=8, color=color, alpha=0.55, linewidths=0)
                m = table[a.value][stat]["estimate"]
                ax.plot(
                    [i - 0.3, i + 0.3],
                    [m, m],
                    color=theme["ink"],
                    linewidth=2,
                    solid_capstyle="round",
                )
            ax.set_xticks(range(len(ARCHETYPES)))
            ax.set_xticklabels([ARCH_LABEL[a.value] for a in ARCHETYPES], fontsize=8)
            ax.set_title(LABELS[stat], fontsize=10, loc="left", color=theme["ink"])
        title(
            fig,
            theme,
            "Measured tendencies per archetype",
            "Dots: individual bots; bar: archetype mean; shaded: intended range "
            "fixed before simulation",
            top=0.94,
        )
        return fig

    save_both(draw, out_dir, "bot_sanity_stats")


def run(
    seed: int,
    *,
    quick: bool = False,
    workers: int | None = None,
    out_dir: Path | None = None,
    cfg: dict | None = None,
) -> dict:
    cfg = cfg or (QUICK if quick else DEFAULTS)
    out_dir = out_dir or RESULTS_ROOT / ("bot_sanity_quick" if quick else "bot_sanity")
    rng = np.random.default_rng(seed)
    pop = make_population(cfg["n_per_archetype"], rng)
    truth = GroundTruth(pop)
    league_cfg = LeagueConfig(
        hands_per_player=cfg["hands_per_player"], block_hands=cfg["block_hands"]
    )
    t0 = time.perf_counter()
    result = run_league(pop, league_cfg, seed=seed, workers=workers)
    elapsed = time.perf_counter() - t0
    labels = [truth.archetype_of(p).value for p in result.player_ids]
    per_player = player_stats(result.counters)
    table = summarize(per_player, labels)
    orderings = check_orderings(table)
    drivers = driver_correlations(per_player, truth.param_matrix(result.player_ids), labels)
    n_cells = sum(e["in_range"] is not None for a in table.values() for e in a.values())
    n_ok = sum(bool(e["in_range"]) for a in table.values() for e in a.values())
    report = {
        "meta": run_metadata(seed, {**cfg, "league": league_cfg}),
        "hands_simulated": result.hands_simulated,
        "seconds": round(elapsed, 1),
        "summary": {
            "cells_in_intended_range": n_ok,
            "cells_with_intended_range": n_cells,
            "orderings_holding": sum(o["holds"] for o in orderings),
            "orderings_total": len(orderings),
        },
        "per_archetype": table,
        "orderings": orderings,
        "driver_correlations": drivers,
    }
    save_json(report, out_dir / "bot_sanity.json")
    (out_dir / "bot_sanity_table.md").write_text(markdown_table(table), encoding="utf-8")
    truth.to_json(out_dir / "ground_truth.json")
    plot(per_player, labels, table, out_dir)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    report = run(args.seed, quick=args.quick, workers=args.workers)
    s = report["summary"]
    print(
        f"{report['hands_simulated']} hands in {report['seconds']}s; "
        f"{s['cells_in_intended_range']}/{s['cells_with_intended_range']} cells in range; "
        f"{s['orderings_holding']}/{s['orderings_total']} orderings hold"
    )


if __name__ == "__main__":
    main()

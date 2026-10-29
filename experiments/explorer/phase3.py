"""Explorer sections for Phase 3: from hand histories to exploit recommendations."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.explorer.phase2 import ARCH_OPTIONS, RESPONSE_COLOR, RESPONSE_LABEL
from experiments.explorer.spec import (
    ARCHETYPE_LABELS,
    axis,
    ci,
    control,
    dot_series,
    dots_chart,
    heatmap,
    line_chart,
    provenance,
    ref,
    scatter_chart,
    scatter_group,
    section,
    series,
    tile,
    view,
)

PUSHFOLD = "experiments/results/phase3/pushfold/pushfold.json"
EXPLOIT = "experiments/results/phase3/exploit/exploit.json"
BRIDGE = "experiments/results/phase3/bridge/bridge.json"
RECOMMEND = "experiments/results/phase3/recommend/recommend.json"
FITCHECK = "experiments/results/phase3/fitcheck/fitcheck.json"
CONFIRM = "experiments/results/phase3/confirm/confirm.json"
IMPORT = "experiments/results/phase3/import/import.json"
PRIORS = "experiments/results/phase3/priors/priors.json"
SEQUENCE = "experiments/results/phase3/sequence/sequence.json"
SCHEMA = "experiments/results/phase3/schema/schema.json"
ADAPTIVE = "experiments/results/phase3/adaptive/adaptive.json"
SOURCES = (
    PUSHFOLD,
    EXPLOIT,
    BRIDGE,
    RECOMMEND,
    FITCHECK,
    CONFIRM,
    IMPORT,
    PRIORS,
    SEQUENCE,
    SCHEMA,
    ADAPTIVE,
)
GRID_RANKS = "AKQJT98765432"
SIDES = (("pusher", "Small blind: push"), ("caller", "Big blind: call"))
STACK = axis("Effective stack (bb)", fmt="num")


def grid_hand(r: int, c: int) -> str:
    """13 x 13 chart layout: pairs on the diagonal, suited above it, offsuit below."""
    a, b = GRID_RANKS[min(r, c)], GRID_RANKS[max(r, c)]
    return a + a if r == c else a + b + ("s" if c > r else "o")


GRID = [[grid_hand(r, c) for c in range(13)] for r in range(13)]
RANK_LABELS = list(GRID_RANKS)


def _low_top(intervals: list[list[float]], start: float) -> float:
    """Top of the stack range that starts at the bottom of the grid (0 if none)."""
    return max((hi for lo, hi in intervals if lo <= start + 1e-9), default=0.0)


def _grid(values: dict[str, float]) -> list[list[float]]:
    return [[values[h] for h in row] for row in GRID]


def overview(
    r: dict[str, Any],
    ex: dict[str, Any],
    br: dict[str, Any],
    rc: dict[str, Any],
    fc: dict[str, Any],
    im: dict[str, Any],
    pr: dict[str, Any],
    sq: dict[str, Any],
    sc: dict[str, Any],
    ad: dict[str, Any],
) -> dict[str, Any]:
    cmp = r["reference_comparison"]
    p, c = cmp["pusher"]["summary"], cmp["caller"]["summary"]
    nash = r["nash"]
    k10 = int(np.argmin(np.abs(np.array(nash["stacks"]) - 10)))
    tight = r["heuristics"]["tight"]["exploitability"]
    std = ex["aggregate"]["standard"]
    oracle = std["oracle"]["br"]["gain"]["mean"]
    br30 = std["by_size"]["30"]["likelihood"]["br"]["gain"]["mean"]
    checks = br["checks"]
    max_z = max(checks[k]["max_abs_z"] for k in ESTIMATORS)
    ten = [m for m in _at_stack(br, "10") if m["opponent"] != "nash"]
    naive, best = (
        float(np.median([m["estimates"][k]["hands_needed"] for m in ten]))
        for k in ("naive", "duplicate_allin_ev")
    )
    rstd = rc["aggregate"]["standard"]
    rsizes = _rec_sizes(rc)
    rec_loss = max(rstd[n]["rules"]["lower_bound"]["loses"]["share"] for n in rsizes)
    rec_cov = [rstd[n]["coverage"]["share"] for n in rsizes]
    fagg = fc["aggregate"]
    fsizes = [str(n) for n in fc["meta"]["config"]["sizes"]]
    false_alarm = max(fagg["standard"][n]["flags"]["either"]["share"] for n in fsizes)
    harsh = fagg["connectors_first"]["300"]["flags"]["either"]["share"]
    gear = fagg["gear_change"]["1000"]["flags"]["either"]["share"]
    rt, pub = im["round_trip"], im["public_data"]
    est = pr["populations"]["humans"]["estimation"]["25"]["rmse_vs_truth"]
    sup = pr["populations"]["humans"]["classifier"]["support"]
    flagged_from = next(int(n) for n, v in sup.items() if v["humans_flagged"] >= 0.999)
    seq_gain = [v["log_loss_gain"]["mean"] for v in sq["a"]["paired"]["seq"].values()]
    big_gain = [v["log_loss_gain"]["mean"] for v in sq["c"]["paired"]["bag"].values()]
    agree = sq["b"]["humans"]["agreement"]
    srt, wsop = sc["round_trip"], sc["public"]["by_subset"]["wsop"]
    adg = ad["aggregate"]
    return {
        "tiles": [
            tile(
                "Call thresholds within 0.1 bb of a published Nash chart",
                f"{c['within_0_1bb']}/{c['hands']}",
                "HoldemResources, no ante",
            ),
            tile(
                "Push thresholds within 0.1 bb",
                f"{p['within_0_1bb']}/{p['hands']}",
                f"the rest are near-indifferent: at most {p['max_cost_mbb_per_hand']:.2f} mbb/hand",
            ),
            tile(
                "Nash at 10 bb",
                f"{nash['push_share'][k10]:.0%} / {nash['call_share'][k10]:.0%}",
                "of hands the small blind pushes / the big blind calls",
            ),
            tile(
                "A fixed tight range rule loses",
                f"{min(tight):.2f}-{max(tight):.2f} bb/hand",
                "to a best response (push top 20%, call top 10%)",
            ),
            tile(
                "Exploiting a bot after 30 hands of watching",
                f"{br30 / oracle:.0%} of the oracle's gain",
                f"{br30:+.3f} of {oracle:+.3f} bb/hand at 10 bb (likelihood model)",
            ),
            tile(
                "Exploiting the average bot instead",
                f"{std['population_model']['br']['gain']['mean']:+.3f} bb/hand",
                "a best response to the population model loses to equilibrium play",
            ),
            tile(
                "Real hands vs the exact game",
                f"largest |z| {max_z:.1f}",
                f"{sum(checks[k]['n'] for k in ESTIMATORS)} checks of the push/fold provider "
                "played through PokerKit"
                + ("" if checks["pass"] else "; the pre-registered check FAILED"),
            ),
            tile(
                "Hands for a ±0.05 bb/hand interval at 10 bb",
                f"{naive:,.0f} → {best:,.0f}",
                "naive vs duplicate dealing + all-in EV (median over bots)",
            ),
            tile(
                "Recommendations that lose to equilibrium play",
                f"{rec_loss:.1%} of bots",
                "at worst over sample sizes; ranked by a 90% posterior lower bound",
            ),
            tile(
                "90% intervals that contain the true gain",
                f"{min(rec_cov):.0%}-{max(rec_cov):.0%}",
                "of candidates, by sample size (standard bots)",
            ),
            tile(
                "Model-fit checks: false alarms",
                f"at most {false_alarm:.0%}",
                "of bots the model describes are flagged (calibrated level 5%)",
            ),
            tile(
                "Misfits caught",
                f"{harsh:.0%} / {gear:.0%}",
                "of connectors-first bots by 300 hands / gear-changers by 1,000",
            ),
            tile(
                "Simulated hands exported to PHH and imported back",
                f"{rt['identical']:,}/{rt['checked']:,}",
                "identical (each hand and its redacted copy)",
            ),
            tile(
                "Public research hands imported",
                f"{pub['accepted']:,}/{pub['files']:,}",
                f"all {pub['by_subset']['pluribus']['accepted']:,} Pluribus hands; mixed games "
                "and big-blind antes rejected by name",
            ),
            tile(
                "A new professional's stats after 25 hands",
                f"{100 * np.mean(est['human']):.1f} vs {100 * np.mean(est['uniform']):.1f} pp",
                "error with a prior fit to the other professionals vs the uniform default",
            ),
            tile(
                "Professionals unlike every training archetype",
                f"100% from {flagged_from:,} hands",
                f"of profiles flagged (held-out bots: at most "
                f"{max(v['bots_flagged'] for v in sup.values()):.0%})",
            ),
            tile(
                "A network on every decision vs Phase 1's stats",
                f"{min(seq_gain):+.3f} to {max(seq_gain):+.3f} nats",
                "log-loss gain at 25-2,000 hands; with five times the training bots "
                f"{min(big_gain):+.3f} to {max(big_gain):+.3f}",
            ),
            tile(
                "Two bot-trained classifiers agree on a professional",
                f"{min(v['players'] for v in agree.values()):.0%}-"
                f"{max(v['players'] for v in agree.values()):.0%}",
                f"of windows (on held-out bots {min(v['bots'] for v in agree.values()):.0%}-"
                f"{max(v['bots'] for v in agree.values()):.0%})",
            ),
            tile(
                "Checks passed with big-blind antes, straddles and other minimum bets",
                f"{srt['passed']:,}/{srt['checked']:,}",
                f"{srt['hands']:,} hands; the {srt['checked'] - srt['passed']} misses are "
                "split cents (schema v2)",
            ),
            tile(
                "WSOP final-table hold'em hands imported",
                f"{wsop['imported']}/{wsop['nlhe']}",
                "all with a big-blind ante, which schema v1 rejected",
            ),
            tile(
                "Equilibrium play's winnings once the bots adapt",
                f"{adg['fast']['heroes']['equilibrium']['value']['mean']:+.3f} bb/hand",
                f"from {adg['static']['heroes']['equilibrium']['value']['mean']:+.3f} against "
                "the same bots static: they fix most of their leaks, whoever they face",
            ),
            tile(
                "Exploiting a fast adapter with a best response",
                f"{adg['fast']['heroes']['best_response']['gain']['mean']:+.4f} bb/hand",
                f"over equilibrium play; loses to "
                f"{adg['fast']['heroes']['best_response']['loses']['share']:.0%} of them",
            ),
        ],
        "summary": (
            "Phase 3 turns hand histories into ranked exploit recommendations. It starts with a "
            "bridge to hold'em: heads-up push/fold no-limit hold'em (blinds 0.5/1, no ante), "
            "solved exactly with the Phase 2 tools on the 169 starting-hand classes with exact "
            "card-removal weights and checked against a published Nash solution (P3.1). Then "
            "push/fold bots are modeled from what a player sees and exploited (P3.2), the "
            "solved strategy plays real hands through the game engine behind the advisor "
            "firewall, checked against the exact values (P3.3), and exploit recommendations "
            "are ranked by how confidently they beat equilibrium play, with posterior intervals "
            "checked against the truth (P3.4). Before the model is trusted, posterior predictive "
            "checks ask whether the opponent plays the way it assumes (P3.5), and a fallback rule "
            "found in that analysis was confirmed on fresh seeds (P3.5b). Completed hand "
            "histories can now come from text files in the open PHH format: each hand is "
            "replayed through PokerKit's rules and rejected with a reason if it does not "
            "replay exactly (P3.6). Priors fit to the 13 professionals of the Pluribus hands "
            "estimate a new professional's statistics far better than the defaults and much "
            "worse for anyone else, and the bot-trained classifier and action models do not "
            "transfer to these players (P3.7). A PyTorch model that reads every decision of "
            "every hand does not beat Phase 1's hand-written statistics with Phase 1's "
            "training bots, edges ahead only with five times as many, and is even more "
            "overconfident on the professionals (P3.8). Schema v2 holds big-blind antes, "
            "straddles and other minimum bets, reads version 1 files by migrating them, and "
            "imports the hold'em hands of a modern tournament final table (P3.10). Bots that "
            "adapt to the hero take most of the money off the table, for equilibrium play too; "
            "what survives is exploiting the leaks they do not fix (P3.9). One command reruns "
            "every step: at commit 98449f0 eight of the eleven reproduce exactly, and the other "
            "three differ only through later changes described in their sections (they "
            "reproduce exactly at the commits their results record)."
        ),
    }


def charts(r: dict[str, Any]) -> dict[str, Any]:
    cmp = r["reference_comparison"]
    start = r["nash"]["stacks"][0]
    top = r["nash"]["stacks"][-1]
    views = {}
    for side, label in SIDES:
        per = cmp[side]["per_hand"]
        ours = {h: _low_top(v["ours"], start) for h, v in per.items()}
        gap = {h: v["disagreement_bb"] for h, v in per.items()}
        single = [
            (float(v["reference"]), ours[h], h)
            for h, v in per.items()
            if "-" not in v["reference"] and not v["reference"].endswith("+")
        ]
        views[side] = view(
            heatmap(
                f"{label} up to this stack (bb)",
                RANK_LABELS,
                RANK_LABELS,
                _grid(ours),
                labels=GRID,
                fmt="num",
                vmax=top,
                subtitle="Largest effective stack at which the hand plays (20 = at least 20). "
                "Pairs on the diagonal, suited hands above it, offsuit below.",
            ),
            scatter_chart(
                "This solution vs HoldemResources",
                axis("HoldemResources chart (bb)", lo=0, hi=20),
                axis("This project's LP solution (bb)", lo=0, hi=20),
                [scatter_group("Hands with one threshold below 20 bb", "c1", single)],
                diagonal=True,
                subtitle="On the diagonal the two agree. Hover for the hand.",
            ),
            heatmap(
                "Where the two charts disagree (bb of stack range)",
                RANK_LABELS,
                RANK_LABELS,
                _grid(gap),
                labels=GRID,
                fmt="num",
                vmax=max(1.0, max(gap.values())),
                subtitle="Length of the stack range on which the decisions differ; 0.05 to 0.1 bb "
                "is the charts' resolution.",
            ),
        )
    p, c = cmp["pusher"]["summary"], cmp["caller"]["summary"]
    return section(
        "pushfold-charts",
        "Push/fold Nash charts, checked against a published solution",
        "The classic chart format: each hand's largest stack at which it plays. Switch between "
        "the small blind's pushes and the big blind's calls.",
        sources=[PUSHFOLD],
        controls=[control("side", "Chart", SIDES)],
        views=views,
        takeaways=[
            f"Calls: {c['within_0_1bb']} of {c['hands']} hands within 0.1 bb of HoldemResources' "
            "chart, which is the chart's own resolution.",
            f"Pushes: {p['within_0_1bb']} within 0.1 bb and {p['within_0_25bb']} within 0.25 bb. "
            f"The rest ({', '.join(p['worst'][:4])}, ...) are hands that are nearly indifferent "
            f"between pushing and folding: at any stack where the charts differ, the hand gains "
            f"at most {p['max_ev_gap_bb']:.3f} bb by either choice.",
            f"Following the published chart wherever it differs would cost at most "
            f"{p['max_cost_mbb_per_hand']:.2f} mbb/hand (pushes) and "
            f"{c['max_cost_mbb_per_hand']:.2f} mbb/hand (calls): the two solutions are "
            "practically the same. Differences of this size are expected from Monte Carlo "
            "matchup equities with a standard error of about 0.002.",
            "Small suited connectors (63s, 43s) push at small and medium stacks but not in "
            "between, as in the published chart.",
        ],
        method=(
            "Exact LP equilibrium at every 0.05 bb from 1 to 20 bb; a hand plays at a stack when "
            "its probability is at least 0.5. Reference: HoldemResources' heads-up Nash charts "
            "(no ante), stored in experiments/references/hune_no_ante.json with five cells "
            "corrected after a second read of the page. The disagreement pricing was added after "
            "the first run (post hoc)."
        ),
        post_hoc=True,
    )


def ranges(r: dict[str, Any]) -> dict[str, Any]:
    classes = r["classes"]
    views = {}
    stacks = list(r["display"])
    for s in stacks:
        d = r["display"][s]
        push = dict(zip(classes, d["push"], strict=True))
        call = dict(zip(classes, d["call"], strict=True))
        views[s] = view(
            heatmap(
                f"Small blind: probability of pushing at {s} bb",
                RANK_LABELS,
                RANK_LABELS,
                _grid(push),
                labels=GRID,
                fmt="pct",
                vmax=1.0,
            ),
            heatmap(
                f"Big blind: probability of calling at {s} bb",
                RANK_LABELS,
                RANK_LABELS,
                _grid(call),
                labels=GRID,
                fmt="pct",
                vmax=1.0,
            ),
        )
    mixed = max(
        sum(1 for x in d["push"] + d["call"] if 0.01 < x < 0.99) for d in r["display"].values()
    )
    nash = r["nash"]
    tighter = next(
        s
        for s, p, c in zip(nash["stacks"], nash["push_share"], nash["call_share"], strict=True)
        if c < p
    )
    return section(
        "pushfold-ranges",
        "Nash ranges at a given stack",
        "The full equilibrium strategy at one stack size: darker cells play more often.",
        sources=[PUSHFOLD],
        controls=[control("stack", "Effective stack (bb)", [(s, s) for s in stacks])],
        views=views,
        defaults={"stack": "10"},
        takeaways=[
            f"The equilibrium is almost pure: at most {mixed} of the 338 decisions mix at any "
            "of these stacks; mixing happens only where a hand is exactly indifferent.",
            f"From {tighter:g} bb up the big blind calls with fewer hands than the small blind "
            "pushes (the caller has no fold equity); below that the big blind, already "
            "committed by its blind, calls wider.",
        ],
    )


def sizes(r: dict[str, Any]) -> dict[str, Any]:
    nash = r["nash"]
    stacks = nash["stacks"]
    shares = line_chart(
        "Range sizes by stack",
        STACK,
        axis("Share of hands", fmt="pct", lo=0, hi=1),
        [
            series(
                "Small blind pushes",
                "c1",
                list(zip(stacks, nash["push_share"], strict=True)),
                marker=False,
            ),
            series(
                "Big blind calls",
                "c2",
                list(zip(stacks, nash["call_share"], strict=True)),
                marker=False,
            ),
        ],
        subtitle="Exact LP at every 0.05 bb.",
    )
    value = line_chart(
        "The small blind's value",
        STACK,
        axis("Expected value per hand as small blind (bb)", fmt="signed"),
        [series("Small blind", "c1", list(zip(stacks, nash["value"], strict=True)), marker=False)],
        refs=[ref(0.0, "Break-even")],
        subtitle="Positive: the small blind profits from moving first at short stacks.",
    )
    k5 = int(np.argmax(nash["value"]))
    zero = next(s for s, v in zip(stacks, nash["value"], strict=True) if s > 3 and v < 0)
    push = np.array(nash["push_share"])
    rises = np.diff(push)
    n_up, max_up = int(np.sum(rises > 1e-12)), float(rises.max())
    return section(
        "pushfold-sizes",
        "How Nash ranges change with stack size",
        "Deeper stacks mean more is risked per push, so both ranges tighten.",
        sources=[PUSHFOLD],
        views={"": view(shares, value)},
        takeaways=[
            f"The small blind's value peaks at {nash['value'][k5]:+.3f} bb/hand around "
            f"{stacks[k5]:g} bb and turns negative near {zero:g} bb: in this game deeper "
            f"stacks favor the big blind (the small blind is at {nash['value'][-1]:+.3f} "
            f"bb/hand at {stacks[-1]:g} bb).",
            f"Both ranges shrink as stacks grow, but not strictly: the push share rises at "
            f"{n_up} of {len(rises)} steps of 0.05 bb (by at most {max_up:.1%}) as "
            "near-indifferent hands flip, and some small suited hands re-enter the range.",
        ],
    )


def solvers(r: dict[str, Any]) -> dict[str, Any]:
    cfr = line_chart(
        "CFR+ converging to the LP equilibrium",
        axis("Iterations", scale="log", fmt="int"),
        axis("Exploitability (bb/hand)", scale="log", fmt="sci"),
        [
            series(
                f"{c['stack']:g} bb",
                f"c{i}",
                [(p["iteration"], p["exploitability"]) for p in c["curve"]],
            )
            for i, c in enumerate(r["cfr"], start=1)
        ],
        subtitle="Average strategy, exact best-response evaluation.",
    )
    hs = r["heuristics"]
    heur = line_chart(
        "Exploitability of simple range rules",
        STACK,
        # Linear: the ranked rule is exactly 0 (= Nash) at some stacks.
        axis("Exploitability (bb/hand)", fmt="num", lo=0),
        [
            series(
                f"{name}: {h['description']}",
                f"c{i}",
                list(zip(h["stacks"], h["exploitability"], strict=True)),
            )
            for i, (name, h) in enumerate(hs.items(), start=1)
        ],
        subtitle="How much a best response wins against each rule (Nash = 0).",
    )
    ranked = hs["ranked"]["exploitability"]
    cfr_stacks = ", ".join(f"{c['stack']:g}" for c in r["cfr"])
    agree = all(
        c["push_decisions_match_lp"] == 169 and c["call_decisions_match_lp"] == 169
        for c in r["cfr"]
    )
    if not agree:
        raise ValueError("CFR+ and LP decisions differ: update the takeaway text")
    fixed = {k: v for k, v in hs.items() if k != "ranked"}
    return section(
        "pushfold-rules",
        "Solvers agree; simple rules are costly",
        "CFR+ reaches the LP's equilibrium; fixed range rules leak a lot against a best response.",
        sources=[PUSHFOLD],
        views={"": view(heur, cfr)},
        takeaways=[
            f"Ranking hands by equity against a random hand is almost right: with Nash's own "
            f"range sizes it is exploitable for at most {max(ranked):.3f} bb/hand. What matters "
            "is how many hands to play.",
            "Fixed range sizes are exploitable for "
            + ", ".join(
                f"{k} {min(v['exploitability']):.2f}-{max(v['exploitability']):.2f}"
                for k, v in fixed.items()
            )
            + " bb/hand across 2-20 bb stacks; each is least bad at the stack its sizes happen "
            "to fit.",
            f"CFR+ makes the same push and call decision as the LP for all 169 hands at "
            f"{cfr_stacks} bb.",
        ],
        method=(
            "Rules are scored by exact exploitability: a best response's average gain over both "
            "seats. The rule sizes were fixed before the first run."
        ),
    )


# ------------------------------------------------------------- P3.2: exploiting push/fold bots

POP_OPTIONS = [("standard", "Standard bots"), ("misspecified", "Pairs-first bots (misspecified)")]
MODEL_OPTIONS = [
    ("likelihood", "Likelihood (uses shown cards)"),
    ("frequency", "Frequency (HUD stats)"),
]
HANDS = axis("Hands observed (cards seen only when a push is called)", scale="log", fmt="int")


def _responses(ex: dict[str, Any]) -> list[str]:
    return ["br", *(f"rnr_{p:g}" for p in ex["meta"]["config"]["ps"]), "dbr"]


def _bots(ex: dict[str, Any], pop: str, arch: str) -> list[list[dict[str, Any]]]:
    return [
        [b for b in seed if b["population"] == pop and arch in ("all", b["archetype"])]
        for seed in ex["per_seed"]
    ]


def _seed_ci(groups: list[list[dict[str, Any]]], fn) -> tuple:
    """Mean over bots within each seed, then a t-interval across seeds."""
    return ci([float(np.mean([fn(b) for b in bots])) for bots in groups if bots])


def _metric(score: dict[str, float], metric: str) -> float:
    if metric == "loses":
        return float(score["gain"] < 0)
    return score[metric]


def exploit_hands(ex: dict[str, Any]) -> dict[str, Any]:
    sizes = [str(n) for n in ex["meta"]["config"]["sizes"]]
    responses = _responses(ex)
    metrics = {
        "gain": ("Gain over equilibrium play", axis("bb/hand above equilibrium", fmt="signed")),
        "exploitability": (
            "Risk: exploitability of the hero's response",
            axis("Exploitability (bb/hand)", lo=0),
        ),
        "loses": (
            "Bots against which the response does worse than equilibrium",
            axis("Share of bots", fmt="pct", lo=0),
        ),
    }
    views = {}
    for metric, (title, yax) in metrics.items():
        for model, mlabel in MODEL_OPTIONS:
            for pop, plabel in POP_OPTIONS:
                for arch, alabel in ARCH_OPTIONS:
                    groups = _bots(ex, pop, arch)
                    lines = []
                    for resp in responses:
                        pts = []
                        for n in sizes:
                            mean, lo, hi = _seed_ci(
                                groups,
                                lambda b, n=n, r=resp, mo=model, me=metric: _metric(
                                    b["sizes"][n][mo]["scores"][r], me
                                ),
                            )
                            if metric == "loses" and lo is not None:
                                lo, hi = max(lo, 0.0), min(hi, 1.0)
                            pts.append((int(n), mean, lo, hi))
                        lines.append(series(RESPONSE_LABEL[resp], RESPONSE_COLOR[resp], pts))
                    refs = []
                    if metric == "gain":
                        oracle = _seed_ci(groups, lambda b: b["oracle"]["br"]["gain"])[0]
                        pop_br = _seed_ci(groups, lambda b: b["population_model"]["br"]["gain"])[0]
                        refs = [
                            ref(oracle, "Oracle best response (true parameters)"),
                            ref(pop_br, "Best response to the population model (0 hands)"),
                            ref(0.0, "Equilibrium play"),
                        ]
                    who = alabel.lower() if arch == "all" else f"{alabel} bots"
                    views[f"{metric}|{model}|{pop}|{arch}"] = view(
                        line_chart(
                            f"{title} ({who})",
                            HANDS,
                            yax,
                            lines,
                            refs=refs,
                            subtitle=f"{plabel}; {mlabel.lower()} model. 5 seeds x 30 "
                            "held-out bots at 10 bb; exact values; bands are 95% t-intervals "
                            "across seeds.",
                        )
                    )
    agg = ex["aggregate"]
    std = agg["standard"]
    oracle = std["oracle"]["br"]["gain"]["mean"]
    br = {n: std["by_size"][n]["likelihood"]["br"] for n in sizes}
    rnr9 = std["by_size"]["10"]["likelihood"].get("rnr_0.9")
    mis = agg["misspecified"]
    mis_share = (
        mis["by_size"][sizes[-1]]["likelihood"]["br"]["gain"]["mean"]
        / mis["oracle"]["br"]["gain"]["mean"]
    )
    takeaways = [
        f"Watching the individual is what pays: a best response to the population model gains "
        f"{std['population_model']['br']['gain']['mean']:+.3f} bb/hand (it loses to equilibrium "
        "play), because the archetypes straddle the equilibrium and their average is close to it.",
        f"With the likelihood model the best response reaches "
        f"{br['10']['gain']['mean'] / oracle:.0%} of the oracle's gain after 10 hands, "
        f"{br['30']['gain']['mean'] / oracle:.0%} after 30 and "
        f"{br['100']['gain']['mean'] / oracle:.0%} after 100.",
        f"After 10 hands the best response loses to equilibrium play against "
        f"{br['10']['loses']['mean']:.0%} of bots.",
    ]
    if rnr9:
        takeaways.append(
            f"Restricted Nash with p = 0.9 gains as much after 10 hands "
            f"({rnr9['gain']['mean']:+.3f} vs {br['10']['gain']['mean']:+.3f}) at "
            f"{rnr9['exploitability']['mean'] / br['10']['exploitability']['mean']:.0%} of the "
            "exploitability: hedging pays while the model is uncertain."
        )
    takeaways.append(
        f"Against pairs-first bots both models are wrong about which hands a bot plays, yet the "
        f"best response still gets {mis_share:.0%} of the oracle's gain by 3,000 hands. This "
        "control turned out mild: the two hand orders are close (rank correlation 0.95), so "
        "getting the range *size* right is most of the job, as in P3.1."
    )
    return section(
        "pushfold-exploit",
        "Exploiting modeled push/fold bots",
        "The hero watches a push/fold bot (10 bb) for n hands, fits a model and responds. Choose "
        "the metric, the model, the population and the bots to average over.",
        sources=[EXPLOIT],
        controls=[
            control(
                "metric",
                "Metric",
                [("gain", "Gain"), ("exploitability", "Risk"), ("loses", "Bots lost to")],
            ),
            control("model", "Model", MODEL_OPTIONS),
            control("pop", "Bots", POP_OPTIONS),
            control("arch", "Archetype", ARCH_OPTIONS),
        ],
        views=views,
        takeaways=takeaways,
        method=(
            "Pre-registered in experiments/phase3_exploit.py. Bots push (call) hand class i with "
            "probability sigmoid((width - percentile_i) / temperature); the hero plays the exact "
            "equilibrium while observing and sees a bot's cards only at a called push. Gain is "
            "exact: the response's value against the true bot minus the equilibrium's, averaged "
            "over both seats."
        ),
    )


def model_fit(ex: dict[str, Any]) -> dict[str, Any]:
    sizes = [str(n) for n in ex["meta"]["config"]["sizes"]]
    views = {}
    for pop, plabel in POP_OPTIONS:
        for arch, alabel in ARCH_OPTIONS:
            groups = _bots(ex, pop, arch)
            err = []
            for k, (p, color) in enumerate((("push width", "c1"), ("call width", "c2"))):
                for model, dashed in (("likelihood", False), ("frequency", True)):
                    pts = [
                        (
                            int(n),
                            *_seed_ci(
                                groups,
                                lambda b, n=n, k=k, model=model: b["sizes"][n][model][
                                    "param_abs_error"
                                ][k],
                            ),
                        )
                        for n in sizes
                    ]
                    err.append(series(f"{p}, {model}", color, pts, dash=dashed))
            gain = [
                series(
                    f"Best response, {model} model",
                    "c1",
                    [
                        (
                            int(n),
                            *_seed_ci(
                                groups,
                                lambda b, n=n, model=model: b["sizes"][n][model]["scores"]["br"][
                                    "gain"
                                ],
                            ),
                        )
                        for n in sizes
                    ],
                    dash=model == "frequency",
                )
                for model, _ in MODEL_OPTIONS
            ]
            oracle = _seed_ci(groups, lambda b: b["oracle"]["br"]["gain"])[0]
            who = alabel.lower() if arch == "all" else f"{alabel} bots"
            views[f"{pop}|{arch}"] = view(
                line_chart(
                    f"Range-width error vs hands ({who})",
                    HANDS,
                    axis("Mean absolute error of the fitted width", scale="log", fmt="num"),
                    err,
                    subtitle=f"{plabel}. Solid: likelihood model; dashed: frequency model."
                    + (
                        " For pairs-first bots the error is against widths under the other order."
                        if pop == "misspecified"
                        else ""
                    ),
                ),
                line_chart(
                    f"What the shown cards are worth ({who})",
                    HANDS,
                    axis("Best response's gain over equilibrium (bb/hand)", fmt="signed"),
                    gain,
                    refs=[ref(oracle, "Oracle best response"), ref(0.0, "Equilibrium play")],
                ),
            )
    std = ex["aggregate"]["standard"]["by_size"]
    lik, freq = std["10"]["likelihood"], std["10"]["frequency"]
    conv = next(
        n
        for n in sizes
        if abs(
            std[n]["likelihood"]["br"]["gain"]["mean"] - std[n]["frequency"]["br"]["gain"]["mean"]
        )
        < 0.003
    )
    return section(
        "pushfold-models",
        "Shown cards vs frequencies: two ways to model a bot",
        "The likelihood model uses every hand the bot showed; the frequency model only counts how "
        "often it pushed and called, like a HUD.",
        sources=[EXPLOIT],
        controls=[control("pop", "Bots", POP_OPTIONS), control("arch", "Archetype", ARCH_OPTIONS)],
        views=views,
        takeaways=[
            "Shown cards help most early: after 10 hands the likelihood model's push-width "
            f"error is {lik['param_mae']['push_width']['mean']:.3f} vs "
            f"{freq['param_mae']['push_width']['mean']:.3f}, "
            f"and its best response gains {lik['br']['gain']['mean']:+.3f} vs "
            f"{freq['br']['gain']['mean']:+.3f} bb/hand.",
            f"From about {int(conv):,} hands the two best responses gain the same (within "
            "0.003 bb/hand): frequencies alone pin down the range sizes, which is what matters.",
            "The frequency model keeps the population's temperature, so it never learns how sharp "
            "a bot's range edge is; this costs almost nothing in gain.",
        ],
    )


# ------------------------------------------------------------------- P3.3: the bridge check

ESTIMATOR_OPTIONS = [
    ("duplicate_allin_ev", "Duplicate + all-in EV"),
    ("duplicate", "Duplicate"),
    ("allin_ev", "All-in EV"),
    ("naive", "Naive"),
]
ESTIMATOR_LABEL = dict(ESTIMATOR_OPTIONS)
ESTIMATOR_COLOR = {"naive": "c1", "duplicate": "c2", "allin_ev": "c3", "duplicate_allin_ev": "c4"}
ESTIMATORS = ("naive", "duplicate", "allin_ev", "duplicate_allin_ev")


def _archetype_in_text(arch: str) -> str:
    """Archetype name inside a sentence: "calling station", but acronyms stay "TAG"."""
    label = ARCHETYPE_LABELS[arch]
    return label if label.isupper() else label.lower()


def _matchup_label(m: dict[str, Any]) -> str:
    hero = {"nash": "Nash", "best_response": "Best response"}[m["hero"]]
    opp = "Nash" if m["opponent"] == "nash" else _archetype_in_text(m["opponent"])
    return f"{hero} vs {opp}"


def _stack_options(br: dict[str, Any]) -> list[tuple[str, str]]:
    return [(f"{s:g}", f"{s:g} bb") for s in sorted({m["stack"] for m in br["matchups"]})]


def _at_stack(br: dict[str, Any], st: str) -> list[dict[str, Any]]:
    return [m for m in br["matchups"] if f"{m['stack']:g}" == st]


def bridge_check(br: dict[str, Any]) -> dict[str, Any]:
    cfg = br["meta"]["config"]
    checks = br["checks"]
    views = {}
    for st, _ in _stack_options(br):
        sub = _at_stack(br, st)
        labels = [_matchup_label(m) for m in sub]
        z = dots_chart(
            "z-scores: (simulated - exact) / standard error",
            labels,
            axis("z-score", fmt="num"),
            [
                dot_series(
                    ESTIMATOR_LABEL[k], ESTIMATOR_COLOR[k], [(m["estimates"][k]["z"],) for m in sub]
                )
                for k in ESTIMATORS
            ],
            subtitle="If the bridge is right, z behaves like a standard normal draw: about 1 in 20 "
            "beyond ±1.96 and none beyond ±4. No dot: the estimate has no variance (see notes).",
            refs=[ref(-1.96, "-1.96", axis_name="x"), ref(1.96, "+1.96", axis_name="x")],
        )
        for est, elabel in ESTIMATOR_OPTIONS:
            win = dots_chart(
                f"Hero's win rate at {st} bb: real hands vs the exact game",
                labels,
                axis("Win rate (bb/hand)", fmt="signed"),
                [
                    dot_series("Exact (abstract game)", "c5", [(m["exact"],) for m in sub]),
                    dot_series(
                        f"Simulated ({elabel}), 95% CI",
                        ESTIMATOR_COLOR[est],
                        [
                            (e["mean"], e["lower"], e["upper"])
                            for e in (m["estimates"][est] for m in sub)
                        ],
                    ),
                ],
                subtitle=f"{cfg['n_deals']:,} deals per matchup, each played twice with seats "
                "swapped; both seats averaged.",
            )
            zero = [
                f"{_matchup_label(m)}, {ESTIMATOR_LABEL[k].lower()}: no variance (every deal "
                "pair cancels exactly), estimate "
                + ("equal to" if abs(m["estimates"][k]["diff"]) < 1e-9 else "NOT equal to")
                + " the exact value."
                for m in sub
                for k in ESTIMATORS
                if m["estimates"][k]["zero_variance"]
            ]
            views[f"{st}|{est}"] = view(win, z, notes=zero)
    n_checks = sum(checks[k]["n"] for k in ESTIMATORS)
    n_zero = sum(checks[k]["zero_variance"] for k in ESTIMATORS)
    above = sum(checks[k]["share_above_1_96"] * checks[k]["n"] for k in ESTIMATORS)
    max_z = max(checks[k]["max_abs_z"] for k in ESTIMATORS)
    hands = sum(2 * m["deals"] for m in br["matchups"])
    verdict = "passes" if checks["pass"] else "FAILS"
    mde = [
        m["estimates"]["duplicate_allin_ev"]["mde"]
        for m in br["matchups"]
        if m["opponent"] != "nash"
    ]
    return section(
        "pushfold-bridge",
        "The bridge check: push/fold in real hands vs the exact game",
        "The push/fold provider plays real heads-up hands through PokerKit, behind the advisor "
        "firewall. If the mapping from real hands to the abstract game is right, its win rate "
        "matches the abstract game's exact value.",
        sources=[BRIDGE],
        controls=[
            control("stack", "Stack", _stack_options(br)),
            control("est", "Estimator", ESTIMATOR_OPTIONS),
        ],
        defaults={"stack": "10"},
        views=views,
        takeaways=[
            f"All {hands:,} hands ended preflop, with a fold or a called all-in: the provider "
            "never left the tree."
            if checks["off_tree_hands"] == 0
            else f"{checks['off_tree_hands']} hands left the tree (a postflop decision).",
            f"Across {len(br['matchups'])} matchups and 4 estimators ({n_checks} z-scores), the "
            f"largest |z| is {max_z:.1f} and {above / n_checks:.0%} exceed 1.96 (about 5% "
            f"expected). The pre-registered check {verdict}."
            + (
                f" Estimates with no variance at all (Nash against Nash under duplicate dealing "
                f"where the equilibrium is pure): {n_zero}. Each must equal the exact value: "
                + (
                    "they do."
                    if all(checks[k]["zero_variance_exact"] for k in ESTIMATORS)
                    else "they DO NOT."
                )
                if n_zero
                else ""
            ),
            f"Sensitivity: with duplicate + all-in EV, each matchup against a bot would detect a "
            f"discrepancy of {min(mde):.3f} to {max(mde):.3f} bb/hand (80% power, 5% level).",
            "The all-in EV estimators score called all-ins with the same matchup-equity table "
            "that defines the exact game, so they test the mapping and the chip accounting. The "
            "naive and duplicate estimators use the boards that actually came, so they also test "
            "the table against real dealing, with wider intervals.",
        ],
        method="Hero: PushFoldStrategy through AdvisorAgent (equilibrium, or the exact best "
        "response to the opponent's true ranges). Opponents: the P3.2 archetypes at their mean "
        "parameters, and the equilibrium. Exact value: head-to-head in the abstract game. "
        "All-in EV replaces a called all-in's result by c(2e - 1), with e the class matchup "
        "equity and c the chips each player put in; it is unbiased because e averages the class "
        "pair's combos, which are equally likely given the classes.",
    )


def bridge_variance(br: dict[str, Any]) -> dict[str, Any]:
    hw = br["meta"]["config"]["half_width"]
    views = {}
    for st, _ in _stack_options(br):
        sub = _at_stack(br, st)
        views[st] = view(
            dots_chart(
                f"Hands for a ±{hw:g} bb/hand 95% interval at {st} bb",
                [_matchup_label(m) for m in sub],
                axis("Hands needed (log scale)", scale="log", fmt="int"),
                [
                    dot_series(
                        ESTIMATOR_LABEL[k],
                        ESTIMATOR_COLOR[k],
                        # 0 hands: the two plays of every deal cancel exactly (no log-scale dot)
                        [(m["estimates"][k]["hands_needed"] or None,) for m in sub],
                    )
                    for k in ESTIMATORS
                ],
                subtitle="From each estimator's standard deviation; a duplicate pair counts as "
                "two hands.",
            ),
            notes=[
                "No dot: the estimator has no variance here (the two plays of every deal cancel "
                "exactly), so any number of hands gives the exact value."
            ]
            if any(m["estimates"][k]["hands_needed"] == 0 for m in sub for k in ESTIMATORS)
            else [],
        )

    def median(st: str, k: str) -> float:
        bots = [m for m in _at_stack(br, st) if m["opponent"] != "nash"]
        return float(np.median([m["estimates"][k]["hands_needed"] for m in bots]))

    stacks = [s for s, _ in _stack_options(br)]
    mid = "10" if "10" in stacks else stacks[len(stacks) // 2]
    n, b = median(mid, "naive"), median(mid, "duplicate_allin_ev")
    factor = {
        m["opponent"]: m["estimates"]["naive"]["hands_needed"]
        / m["estimates"]["duplicate"]["hands_needed"]
        for m in _at_stack(br, mid)
        if m["hero"] == "nash" and m["opponent"] != "nash"
    }
    hi, lo = max(factor, key=factor.get), min(factor, key=factor.get)
    return section(
        "pushfold-variance",
        "How many hands a win rate needs",
        "Card luck hides skill. Duplicate dealing and all-in EV remove two kinds of luck; here "
        "is how many hands each estimator needs for the same precision.",
        sources=[BRIDGE],
        controls=[control("stack", "Stack", _stack_options(br))],
        defaults={"stack": "10"},
        views=views,
        takeaways=[
            f"At {mid} bb against the archetype bots (median over matchups): naive "
            f"{n:,.0f} hands, duplicate {median(mid, 'duplicate'):,.0f}, all-in EV "
            f"{median(mid, 'allin_ev'):,.0f}, both together {b:,.0f}: {n / b:.0f}× fewer than "
            "naive.",
            "Needs grow with the stack, since a called all-in risks more: naive "
            + ", ".join(f"{median(s, 'naive'):,.0f} at {s} bb" for s in stacks)
            + ".",
            f"Duplicate dealing's help depends on the opponent: at {mid} bb Nash needs "
            f"{factor[hi]:.1f}× fewer hands against the {_archetype_in_text(hi)} but only "
            f"{factor[lo]:.1f}× fewer against the {_archetype_in_text(lo)}. The closer the "
            "opponent's play to the hero's, the more the two plays of a deal mirror each other; "
            "Nash against Nash is the limit, where the luck of the deal cancels almost exactly.",
            "All-in EV works here because a called preflop all-in has a known equity. Deeper "
            "play needs a value function for every chance node (AIVAT), planned for Phase 5.",
        ],
    )


# ------------------------------------------------------------------------ recommendations (P3.4)

RULE_OPTIONS = [
    ("lower_bound", "Lower-bound pick (the recommendation)"),
    ("bayes_br", "Bayes best response"),
    ("mode_br", "Mode best response (P3.2)"),
    ("rnr_0.5", "Restricted Nash, p = 0.5"),
    ("hindsight", "Best candidate in hindsight"),
]
RULE_LABEL = dict(RULE_OPTIONS)
RULE_COLOR = {
    "lower_bound": "c1",
    "bayes_br": "c2",
    "mode_br": "c3",
    "rnr_0.5": "c5",
    "hindsight": "muted",
}
HANDS_SEEN = axis("Hands observed (cards seen only when a push is called)", scale="log", fmt="int")


def _rec_sizes(rc: dict[str, Any], *, zero: bool = False) -> list[str]:
    return [str(n) for n in rc["meta"]["config"]["sizes"] if zero or n > 0]


def _rule_metric(b: dict[str, Any], n: str, rule: str, metric: str) -> float:
    from experiments.phase3_recommend import TOL, rule_score

    score = rule_score(b, n, rule)
    if metric == "loses":
        return float(score["true_gain"] < -TOL)
    return score["exploitability" if metric == "exploitability" else "true_gain"]


def recommend_rules(rc: dict[str, Any]) -> dict[str, Any]:
    sizes = _rec_sizes(rc)
    metrics = {
        "gain": (
            "True gain over equilibrium play",
            axis("bb/hand above equilibrium", fmt="signed"),
        ),
        "loses": (
            "Bots against which the choice does worse than equilibrium",
            axis("Share of bots", fmt="pct", lo=0),
        ),
        "exploitability": (
            "Risk: exploitability of the choice",
            axis("Exploitability (bb/hand)", lo=0),
        ),
    }
    views = {}
    for metric, (title, yax) in metrics.items():
        for pop, plabel in POP_OPTIONS:
            for arch, alabel in ARCH_OPTIONS:
                groups = _bots(rc, pop, arch)
                lines = []
                for rule, rlabel in RULE_OPTIONS:
                    pts = []
                    for n in sizes:
                        mean, lo, hi = _seed_ci(
                            groups, lambda b, n=n, r=rule, m=metric: _rule_metric(b, n, r, m)
                        )
                        if metric == "loses" and lo is not None:
                            lo, hi = max(lo, 0.0), min(hi, 1.0)
                        pts.append((int(n), mean, lo, hi))
                    lines.append(series(rlabel, RULE_COLOR[rule], pts, dash=rule == "hindsight"))
                refs = []
                if metric == "gain":
                    refs = [
                        ref(
                            _seed_ci(groups, lambda b: b["oracle"]["true_gain"])[0],
                            "Oracle best response (true strategy)",
                        ),
                        ref(
                            _seed_ci(groups, lambda b: b["population_true_gain"])[0],
                            "Population recommendation (0 hands)",
                        ),
                        ref(0.0, "Equilibrium play"),
                    ]
                elif metric == "loses":
                    refs = [ref(0.05, "5%: what a 95% lower bound allows")]
                who = alabel.lower() if arch == "all" else f"{alabel} bots"
                views[f"{metric}|{pop}|{arch}"] = view(
                    line_chart(
                        f"{title} ({who})",
                        HANDS_SEEN,
                        yax,
                        lines,
                        refs=refs,
                        subtitle=f"{plabel}. 5 seeds x 30 held-out bots at 10 bb (the P3.2 bots "
                        "and hands); exact values; bands are 95% t-intervals across seeds.",
                    )
                )
    agg = rc["aggregate"]
    std = agg["standard"]
    first, last = sizes[0], sizes[-1]
    oracle = std[last]["rules"]["oracle"]["gain"]["mean"]

    def loss(pop: str, n: str, rule: str = "lower_bound") -> dict[str, Any]:
        return agg[pop][n]["rules"][rule]["loses"]

    def share_of_oracle(n: str, rule: str) -> str:
        return f"{std[n]['rules'][rule]['gain']['mean'] / oracle:.0%}"

    worst_n = max(sizes, key=lambda n: loss("standard", n)["share"])
    worst = loss("standard", worst_n)
    bots = [b for seed in rc["per_seed"] for b in seed if b["population"] == "standard"]
    deepest = min(
        rule_metric
        for b in bots
        for n in sizes
        if (rule_metric := _rule_metric(b, n, "lower_bound", "gain")) < 0
    )
    gain_gap = max(sizes, key=lambda n: std[n]["rules"]["bayes_br"]["gain"]["mean"])
    diff = {n: std[n]["bayes_minus_mode"] for n in _rec_sizes(rc, zero=True)}
    tiny = next(
        n for k, n in enumerate(sizes) if all(abs(diff[m]["mean"]) < 1e-3 for m in sizes[k:])
    )
    mis_worst = max(loss("misspecified", n)["share"] for n in sizes)
    takeaways = [
        f"Ranked by the lower end of a 90% posterior interval, the recommendation does worse than "
        f"equilibrium play against at most {worst['share']:.1%} of standard bots ({worst['k']} of "
        f"{worst['n']} after {int(worst_n):,} hands; the largest such loss is "
        f"{-deepest:.3f} bb/hand). The pre-registered prediction was at most 5%. The Bayes best "
        f"response, which maximizes expected gain, loses to "
        f"{loss('standard', first, 'bayes_br')['share']:.0%} after {int(first):,} hands, and "
        f"P3.2's best response to the posterior mode to "
        f"{loss('standard', first, 'mode_br')['share']:.0%}.",
        f"The price of that safety is gain while evidence is thin: after {int(first):,} hands the "
        f"recommendation earns {share_of_oracle(first, 'lower_bound')} of the oracle's gain vs "
        f"{share_of_oracle(first, 'bayes_br')} for the Bayes best response; after 30 hands "
        f"{share_of_oracle('30', 'lower_bound')} vs {share_of_oracle('30', 'bayes_br')}; after "
        f"100 hands {share_of_oracle('100', 'lower_bound')} vs "
        f"{share_of_oracle('100', 'bayes_br')}. Neither rule is better in general: the "
        "recommendation fits a player who cannot afford a losing adjustment, the Bayes best "
        "response one who only cares about the average.",
        f"Responding to the posterior mean beats responding to the mode while evidence is thin: "
        f"{diff['0']['mean']:+.3f} bb/hand with no hands (the best response to the population "
        f"mode loses to equilibrium play, as in P3.2) and {diff[first]['mean']:+.3f} "
        f"[{diff[first]['lower']:+.3f}, {diff[first]['upper']:+.3f}] after {int(first):,}; "
        f"after {int(sizes[1]):,} the interval includes zero "
        f"({diff[sizes[1]]['mean']:+.3f} [{diff[sizes[1]]['lower']:+.3f}, "
        f"{diff[sizes[1]]['upper']:+.3f}]), and from {int(tiny):,} hands on the difference is "
        "below 0.001 bb/hand.",
        f"Pairs-first bots, whose hand order the model gets wrong: the recommendation loses to "
        f"equilibrium against at most {mis_worst:.1%} of them after any number of hands "
        f"({loss('misspecified', '0')['share']:.1%} for the population recommendation at 0 "
        "hands).",
    ]
    if gain_gap != last:  # keep the claim honest if the Bayes curve ever turns down
        takeaways.append(
            f"The Bayes best response's gain peaks at {int(gain_gap):,} hands in this run."
        )
    return section(
        "recommend-rules",
        "Ranked recommendations: do they win?",
        "After watching a bot for n hands, the recommender ranks the equilibrium, restricted Nash "
        "responses and the best response by the lower end of a 90% posterior interval for their "
        "gain over equilibrium, and recommends the top one. Here is what each way of choosing "
        "really earns against the bot.",
        sources=[RECOMMEND],
        controls=[
            control(
                "metric",
                "Metric",
                [("gain", "Gain"), ("loses", "Bots lost to"), ("exploitability", "Risk")],
            ),
            control("pop", "Bots", POP_OPTIONS),
            control("arch", "Archetype", ARCH_OPTIONS),
        ],
        views=views,
        takeaways=takeaways,
        method=(
            "Pre-registered in experiments/phase3_recommend.py. Posterior over (push width, call "
            "width, temperature) by importance sampling from a Laplace-fitted Student-t; every "
            "candidate responds to the posterior mean strategy and is scored on every draw "
            "exactly, so its gain has a posterior distribution. The equilibrium's gain is 0, so "
            "the recommendation deviates only when some response gains with posterior "
            "probability of at least 95%. The best candidate in hindsight uses the bot's true "
            "strategy and is not achievable."
        ),
    )


def recommend_calibration(rc: dict[str, Any]) -> dict[str, Any]:
    from experiments.phase3_recommend import candidate_label, candidate_names

    cfg = rc["meta"]["config"]
    sizes = _rec_sizes(rc)
    all_sizes = _rec_sizes(rc, zero=True)
    names = candidate_names(cfg)
    views = {}
    for pop, plabel in POP_OPTIONS:
        agg = rc["aggregate"][pop]

        def pts(key, agg=agg):
            return [
                (int(n), agg[n][key]["share"], agg[n][key]["lower"], agg[n][key]["upper"])
                for n in sizes
                if agg[n][key]["n"]
            ]

        cover = line_chart(
            "Do the intervals contain the true gain?",
            HANDS_SEEN,
            axis("Coverage", fmt="pct", hi=1),
            [
                series("90% interval, every candidate", "c1", pts("coverage")),
                series("Lower end (one-sided 95%), every candidate", "c2", pts("one_sided")),
                series("90% interval, the top pick when it deviates", "c3", pts("pick_coverage")),
            ],
            refs=[ref(0.9, "Nominal 90%"), ref(0.95, "Nominal 95%")],
            subtitle=f"{plabel}. Pooled over bots and seeds (Wilson 95% intervals; candidates of "
            "the same bot are correlated, so these are optimistic).",
        )
        picks = heatmap(
            "Which response the evidence picks",
            [candidate_label(k) for k in names],
            [f"{int(n):,}" for n in all_sizes],
            [[agg[n]["picks"][k]["share"] for n in all_sizes] for k in names],
            subtitle="Share of bots for which each candidate is the top recommendation.",
            row_label="Top pick",
            col_label="Hands observed",
            fmt="pct",
            vmax=1.0,
        )
        views[pop] = view(cover, picks)
    std = rc["aggregate"]["standard"]
    mis = rc["aggregate"]["misspecified"]

    def span(block: dict[str, Any], key: str, ns: list[str]) -> str:
        vals = [block[n][key]["share"] for n in ns]
        return f"{min(vals):.0%} to {max(vals):.0%}"

    eq_share = [std[n]["picks"]["equilibrium"]["share"] for n in all_sizes]
    br_share = [std[n]["picks"]["best_response"]["share"] for n in all_sizes]
    first = sizes[0]
    low_ess = [
        (b["population"], n, b["sizes"][n]["ess"])
        for seed in rc["per_seed"]
        for b in seed
        for n in all_sizes
        if b["sizes"][n]["ess"] < 100
    ]
    fits = sum(len(seed) for seed in rc["per_seed"]) * len(all_sizes)
    draws = rc["meta"]["config"]["draws"]
    takeaways = [
        f"Standard bots: from {int(first):,} hands on, the 90% intervals contain the true gain "
        f"{span(std, 'coverage', sizes)} of the time, inside the pre-registered 85-95% band; with "
        f"no hands (prior only) {std['0']['coverage']['share']:.0%}, slightly conservative, "
        f"which misses the band. The true gain is above the interval's lower end "
        f"{span(std, 'one_sided', sizes)} of the time (nominal 95%).",
        f"The top pick alone is less well covered: after {int(first):,} hands its 90% interval "
        f"holds the true gain {std[first]['pick_coverage']['share']:.0%} of the time and its "
        f"lower end {std[first]['pick_one_sided']['share']:.0%} (nominal 95%). Choosing the "
        "largest lower bound favors candidates whose bound came out too high, a winner's curse "
        "that an exactly right prior would remove (a posterior statement is conditional on the "
        "data, and the choice is a function of the data). The prior here is only approximately "
        "right: independent Betas fitted to a mixture of five archetypes. The deviations still "
        f"beat equilibrium play {std[first]['deviation_wins']['share']:.0%} of the time.",
        f"Pairs-first bots: 90% coverage falls as evidence grows, from "
        f"{mis['0']['coverage']['share']:.0%} with no hands to "
        f"{mis[sizes[-1]]['coverage']['share']:.0%} after {int(sizes[-1]):,}. The model ranks "
        "hands by equity, the bots pairs first, so the posterior concentrates on a slightly "
        "wrong answer. Decisions stay safe because the errors are small next to the gains: the "
        f"top pick still beats equilibrium {mis[sizes[-1]]['deviation_wins']['share']:.0%} of "
        "the time. Checking the model's fit before trusting its intervals is the next step "
        "(P3.5).",
        f"The evidence sets p: the equilibrium is the top pick for {eq_share[1]:.0%} of bots "
        f"after {int(all_sizes[1]):,} hands and {eq_share[-1]:.0%} after "
        f"{int(all_sizes[-1]):,}; the full best response for {br_share[1]:.0%} and "
        f"{br_share[-1]:.0%}.",
        f"Importance sampling: {len(low_ess)} of {fits:,} posterior fits had an effective sample "
        f"size below 100 of {draws:,} draws"
        + (
            f" (the lowest {min(e for *_, e in low_ess):.0f}), all against "
            + (
                "pairs-first bots"
                if all(pop == "misspecified" for pop, *_ in low_ess)
                else "standard and pairs-first bots"
            )
            + "."
            if low_ess
            else "."
        ),
    ]
    return section(
        "recommend-calibration",
        "Are the confidence statements honest?",
        "A recommendation is only as good as its interval. Since the bots are simulated, the "
        "true gain of every candidate is known and can be checked against the posterior "
        "interval the recommender reported.",
        sources=[RECOMMEND],
        controls=[control("pop", "Bots", POP_OPTIONS)],
        views=views,
        takeaways=takeaways,
    )


def recommend_examples(rc: dict[str, Any]) -> dict[str, Any]:
    from experiments.phase3_recommend import candidate_label

    sizes = _rec_sizes(rc, zero=True)
    examples = {
        b["archetype"]: b for b in rc["per_seed"][0] if "explanation" in b["sizes"][sizes[0]]
    }
    archs = [a for a in ARCHETYPE_LABELS if a in examples]
    views = {}
    for arch in archs:
        b = examples[arch]
        for n in sizes:
            s = b["sizes"][n]
            ranked = s["ranked"]
            cands = [s["candidates"][k] for k in ranked]
            rows = [f"{i}. {candidate_label(k)}" for i, k in enumerate(ranked, start=1)]
            chart = dots_chart(
                f"{ARCHETYPE_LABELS[arch]} bot after {int(n):,} hands: the ranked candidates",
                rows,
                axis("Gain over equilibrium (bb/hand)", fmt="signed"),
                [
                    dot_series(
                        "Posterior mean, 90% interval",
                        "c1",
                        [(c["mean"], c["lower"], c["upper"]) for c in cands],
                    ),
                    dot_series(
                        "True gain (known only in simulation)",
                        "c2",
                        [(c["true_gain"],) for c in cands],
                    ),
                ],
                subtitle="Ranked by the interval's lower end; ties go to the smaller p.",
                refs=[ref(0.0, "Equilibrium", axis_name="x")],
            )
            risk = dots_chart(
                "Risk: exploitability of each candidate",
                rows,
                axis("Exploitability (bb/hand)", lo=0),
                [dot_series("Exploitability", "c7", [(c["exploitability"],) for c in cands])],
            )
            top = cands[0]
            views[f"{arch}|{n}"] = view(
                chart,
                risk,
                notes=[
                    s["explanation"],
                    f"True gain of the top pick: {top['true_gain']:+.3f} bb/hand (the oracle "
                    f"best response gains {b['oracle']['true_gain']:+.3f}).",
                ],
            )

    def top(arch: str, n: str) -> dict[str, Any]:
        sz = examples[arch]["sizes"][n]
        return {"name": sz["ranked"][0], **sz["candidates"][sz["ranked"][0]]}

    early, late = sizes[1], sizes[-1]
    cautious = [
        a for a in archs if top(a, early)["name"] in ("equilibrium", "restricted_nash_p0.1")
    ]
    inside = [
        a
        for a in archs
        if top(a, late)["lower"] <= top(a, late)["true_gain"] <= top(a, late)["upper"]
    ]
    risk_early = max(top(a, early)["exploitability"] for a in archs)
    takeaways = [
        f"After {int(early):,} hands {len(cautious)} of the {len(archs)} example bots get the "
        "equilibrium or the mildest restricted Nash response (p = 0.1): a handful of hands is "
        "too little to be 95% sure of more. "
        + (
            f"These picks are practically unexploitable (at most {risk_early:.0e} bb/hand)."
            if risk_early < 1e-3
            else f"These picks are exploitable for at most {risk_early:.3f} bb/hand."
        ),
        f"After {int(late):,} hands the top pick's 90% interval contains its true gain for "
        f"{len(inside)} of the {len(archs)} bots.",
        "The explanation names the hands that change: a response is a list of hands to start "
        "or stop pushing and calling, which is what a player can act on.",
    ]
    return section(
        "recommend-examples",
        "What a recommendation looks like",
        "The recommender's report for one held-out bot of each archetype (first seed): the "
        "ranked candidates with their posterior intervals and the explanation it writes. The "
        "true gain is shown because these are simulated bots.",
        sources=[RECOMMEND],
        controls=[
            control("arch", "Bot", [(a, ARCHETYPE_LABELS[a]) for a in archs]),
            control("n", "Hands observed", [(n, f"{int(n):,}") for n in sizes]),
        ],
        defaults={"arch": "lag" if "lag" in archs else archs[0], "n": "100"},
        views=views,
        takeaways=takeaways,
        method="The explanation text is generated by RecommendationReport.explanation() from the "
        "observations, the posterior and the top recommendation.",
    )


# --------------------------------------------------------------------- model-fit checks (P3.5)

FIT_POPS = [
    ("standard", "Standard (the model is right)"),
    ("pairs_first", "Pairs first (mild)"),
    ("connectors_first", "Connectors first (harsh)"),
    ("gear_change", "Gear change"),
]
FIT_POP_LABEL = dict(FIT_POPS)
FIT_POP_COLOR = {
    "standard": "c1",
    "pairs_first": "c2",
    "connectors_first": "c3",
    "gear_change": "c4",
}
FIT_CHECKS = [
    ("either", "Either check (the flag)"),
    ("shown_hands", "Shown hands"),
    ("stability", "Stability"),
]


def _fit_sizes(fc: dict[str, Any]) -> list[str]:
    return [str(n) for n in fc["meta"]["config"]["sizes"]]


def _share_pts(blocks: list[tuple[str, dict[str, Any]]]) -> list[tuple]:
    return [(int(n), d["share"], d["lower"], d["upper"]) for n, d in blocks if d["n"]]


def fitcheck_flags(fc: dict[str, Any]) -> dict[str, Any]:
    cfg = fc["meta"]["config"]
    sizes = _fit_sizes(fc)
    agg = fc["aggregate"]
    level = cfg["fit_level"]
    views = {}
    for check, clabel in FIT_CHECKS:
        rates = line_chart(
            f"Bots flagged: {clabel.lower()}",
            HANDS_SEEN,
            axis("Share of bots flagged", fmt="pct", lo=0, hi=1),
            [
                series(
                    plabel,
                    FIT_POP_COLOR[pop],
                    _share_pts([(n, agg[pop][n]["flags"][check]) for n in sizes]),
                )
                for pop, plabel in FIT_POPS
            ],
            refs=[ref(level if check == "either" else level / 2, "Calibrated false-alarm level")],
            subtitle="5 seeds x 30 held-out bots per population; 95% Wilson intervals.",
        )
        cal_checks = ["shown_hands", "stability"] if check == "either" else [check]
        cal = line_chart(
            "Calibration: why the threshold is not 0.05",
            HANDS_SEEN,
            axis("p-value", lo=0, hi=1),
            [
                series(
                    f"{name.replace('_', ' ').capitalize()}: {what}",
                    "c1" if name == "shown_hands" else "c2",
                    [
                        (
                            int(n),
                            float(np.mean([c[key][n][name] for c in fc["calibration"].values()])),
                        )
                        for n in sizes
                    ],
                    dash=key == "median_p",
                )
                for name in cal_checks
                for key, what in (("thresholds", "threshold"), ("median_p", "median p-value"))
            ],
            subtitle="On bots simulated from the model itself (300 per seed, averaged over seeds): "
            "posterior predictive p-values pile up toward 0.5, so the threshold that flags 2.5% "
            "of them sits well above 0.025.",
        )
        views[check] = view(rates, cal)

    def either(pop: str, n: str) -> float:
        return agg[pop][n]["flags"]["either"]["share"]

    worst_n = max(sizes, key=lambda n: either("standard", n))
    takeaways = [
        f"False alarms: {min(either('standard', n) for n in sizes):.0%} to "
        f"{either('standard', worst_n):.0%} of the bots the model describes are flagged, the most "
        f"after {int(worst_n):,} hands: above the calibrated {level:.0%}, within the "
        "pre-registered 10%.",
        "Connectors-first bots (a hand order far from equity): flagged "
        + ", ".join(f"{either('connectors_first', n):.0%} after {int(n):,}" for n in sizes[1:4])
        + " hands, by the shown-hands check.",
        "Gear-change bots: flagged "
        + ", ".join(f"{either('gear_change', n):.0%} after {int(n):,}" for n in sizes[2:])
        + " hands, mostly by the stability check.",
        "Pairs-first bots (a hand order close to equity, rank correlation 0.95) are the slow "
        "case: flagged "
        + ", ".join(f"{either('pairs_first', n):.0%} after {int(n):,}" for n in sizes[2:])
        + " hands, almost all by the shown-hands check.",
    ]
    return section(
        "fitcheck-flags",
        "Model-fit checks: who gets flagged?",
        "Before trusting the model, two posterior predictive checks ask whether the observed play "
        "looks like data the fitted model produces: do the hands the bot showed match the model's "
        "hand order, and did its push and call rates stay put? Thresholds are calibrated on bots "
        "simulated from the model, so a bot the model describes is flagged about 5% of the time.",
        sources=[FITCHECK],
        controls=[control("check", "Check", FIT_CHECKS)],
        views=views,
        takeaways=takeaways,
        method=(
            "Pre-registered in experiments/phase3_fitcheck.py. Shown hands: Pearson chi-square of "
            "the bot's shown classes over 10 bins (pair, suited, offsuit x equity quartile), "
            "conditional on the hero's own class each time. Stability: the largest absolute "
            "partial sum of the residuals of the bot's decisions in hand order (a CUSUM), "
            "standardized. Each compared with 400 replicates from the posterior (mid-p). Gear-"
            "change bots switch archetype at a random hand in the middle half of each session."
        ),
    )


FIT_RULES = [
    ("lower_bound", "Trust the ranking (P3.4)", "c1"),
    ("checked", "Equilibrium when either check fails", "c3"),
    ("stability_only", "Equilibrium only when stability fails (post hoc)", "c5"),
]


def _fit_rule_gain(b: dict[str, Any], n: str, rule: str) -> float:
    """True gain of a rule's choice: the top pick, or the equilibrium (gain 0) on a flag."""
    s = b["sizes"][n]
    fallback = (rule == "checked" and s["flagged"]) or (
        rule == "stability_only" and s["failed"]["stability"]
    )
    return s["candidates"]["equilibrium" if fallback else s["ranked"][0]]["true_gain"]


def fitcheck_consequences(fc: dict[str, Any]) -> dict[str, Any]:
    from experiments.phase3_fitcheck import TOL

    sizes = _fit_sizes(fc)
    agg = fc["aggregate"]
    views = {}
    for pop, plabel in FIT_POPS:
        a = agg[pop]
        groups = [[b for b in seed if b["population"] == pop] for seed in fc["per_seed"]]
        flat = [b for g in groups for b in g]
        cover = line_chart(
            "Do flags mark untrustworthy intervals?",
            HANDS_SEEN,
            axis("90% interval coverage", fmt="pct", hi=1),
            [
                series(lab, color, _share_pts([(n, a[n]["coverage"][key]) for n in sizes]))
                for key, lab, color in (
                    ("unflagged", "Unflagged bots", "c1"),
                    ("flagged", "Flagged bots", "c2"),
                    ("all", "All bots", "muted"),
                )
            ],
            refs=[ref(0.9, "Nominal 90%")],
            subtitle=f"{plabel}. Every non-equilibrium candidate's interval vs its true gain.",
        )
        gains = line_chart(
            "True gain: trust the ranking, or fall back when flagged?",
            HANDS_SEEN,
            axis("bb/hand above equilibrium", fmt="signed"),
            [
                series(
                    label,
                    color,
                    [
                        (int(n), *_seed_ci(groups, lambda b, n=n, r=rule: _fit_rule_gain(b, n, r)))
                        for n in sizes
                    ],
                    dash=rule == "stability_only",
                )
                for rule, label, color in FIT_RULES
            ],
            refs=[
                ref(a[sizes[-1]]["oracle_gain"]["mean"], "Oracle best response"),
                ref(0.0, "Equilibrium"),
            ],
            subtitle=f"{plabel}. Against the bot's true current strategy; 95% t-intervals "
            "across seeds.",
        )
        losses = line_chart(
            "Bots the choice does worse than equilibrium against",
            HANDS_SEEN,
            axis("Share of bots", fmt="pct", lo=0),
            [
                series(
                    label,
                    color,
                    [
                        (int(n), float(np.mean([_fit_rule_gain(b, n, rule) < -TOL for b in flat])))
                        for n in sizes
                    ],
                    dash=rule == "stability_only",
                )
                for rule, label, color in FIT_RULES
            ],
            refs=[ref(0.05, "5%: what P3.4's ranking promises")],
            subtitle=f"{plabel}. Pooled over 150 bots.",
        )
        views[pop] = view(cover, gains, losses)

    def mean_gain(pop: str, n: str, rule: str) -> float:
        groups = [[b for b in seed if b["population"] == pop] for seed in fc["per_seed"]]
        return float(np.mean([np.mean([_fit_rule_gain(b, n, rule) for b in g]) for g in groups]))

    def lose(pop: str, n: str, rule: str) -> float:
        flat = [b for seed in fc["per_seed"] for b in seed if b["population"] == pop]
        return float(np.mean([_fit_rule_gain(b, n, rule) < -TOL for b in flat]))

    gear_n = [n for n in sizes if int(n) >= 300]
    oracle_c = agg["connectors_first"]["300"]["oracle_gain"]["mean"]
    pooled = agg["misspecified_pooled"]
    big = [n for n in sizes if int(n) >= 100 and pooled[n]["flagged"]["n"]]
    takeaways = [
        "Gear-changers break the ranking's promise: trusting the model loses to equilibrium "
        "against "
        + ", ".join(f"{lose('gear_change', n, 'lower_bound'):.0%}" for n in gear_n)
        + f" of them after {', '.join(f'{int(n):,}' for n in gear_n)} hands (P3.4 promised at most "
        "5%). Falling back on a flag cuts that to "
        + ", ".join(f"{lose('gear_change', n, 'checked'):.0%}" for n in gear_n)
        + ", for a lower average gain ("
        + ", ".join(
            f"{mean_gain('gear_change', n, 'checked'):+.3f} vs "
            f"{mean_gain('gear_change', n, 'lower_bound'):+.3f}"
            for n in gear_n
        )
        + " bb/hand): trusting the model wins big against some gear-changers and loses against "
        "others.",
        f"A wrong hand order is a different story. Against connectors-first bots the model's "
        f"recommendation still earns {mean_gain('connectors_first', '300', 'lower_bound'):+.3f} "
        f"bb/hand after 300 hands (the oracle {oracle_c:+.3f}) and loses to "
        f"{lose('connectors_first', '300', 'lower_bound'):.0%} of them, even though its "
        f"intervals cover only {agg['connectors_first']['300']['coverage']['all']['share']:.0%}. "
        f"The flag is right about the intervals, but falling back throws the gain away "
        f"({mean_gain('connectors_first', '300', 'checked'):+.3f}). Pairs-first bots: "
        f"{mean_gain('pairs_first', '1000', 'lower_bound'):+.3f} vs "
        f"{mean_gain('pairs_first', '1000', 'checked'):+.3f} after 1,000 hands.",
        "Post hoc: falling back only when the stability check fails keeps "
        f"{mean_gain('standard', '1000', 'stability_only'):+.3f} (standard), "
        f"{mean_gain('pairs_first', '1000', 'stability_only'):+.3f} (pairs first) and "
        f"{mean_gain('connectors_first', '1000', 'stability_only'):+.3f} (connectors first) "
        "bb/hand after 1,000 hands, while matching the full fallback against gear-changers "
        f"({lose('gear_change', '1000', 'stability_only'):.0%} lost to). This rule was chosen "
        "after seeing these results, so it was tested on fresh seeds before becoming the "
        "default (next section).",
        "Pre-registered: pooled over the three misspecified populations, flagged bots have "
        "lower coverage than unflagged ones ("
        + ", ".join(
            f"{pooled[n]['flagged']['share']:.0%} vs {pooled[n]['unflagged']['share']:.0%} at "
            f"{int(n):,}"
            for n in big
        )
        + "). Within one population the gap is small or mixed (for pairs-first bots after "
        "1,000 hands it even reverses), so most of it comes from the flags picking out the "
        "populations whose intervals are worst.",
    ]
    return section(
        "fitcheck-consequences",
        "What a flag means, and what acting on it does",
        "A flag says the model's intervals should not be trusted. Here is whether flagged bots "
        "really have worse intervals, and what playing the equilibrium instead of the top-ranked "
        "response earns or loses against each kind of bot.",
        sources=[FITCHECK],
        controls=[control("pop", "Bots", FIT_POPS)],
        defaults={"pop": "gear_change"},
        views=views,
        takeaways=takeaways,
        method="The two pre-registered rules are the top pick and the top pick replaced by the "
        "equilibrium whenever either check fails. The third rule (equilibrium only when the "
        "stability check fails) was added after the results were seen and is computed from the "
        "same per-bot records.",
    )


def fitcheck_examples(fc: dict[str, Any]) -> dict[str, Any]:
    from experiments.phase3_recommend import candidate_label

    sizes = _fit_sizes(fc)
    examples = [b for b in fc["per_seed"][0] if "explanation" in b["sizes"][sizes[-1]]]
    options = []
    views = {}
    for b in examples:
        key = b["bot_id"]
        who = f"{FIT_POP_LABEL[b['population']].split(' (')[0]}: {ARCHETYPE_LABELS[b['archetype']]}"
        if "second_archetype" in b:
            who += f" → {ARCHETYPE_LABELS[b['second_archetype']]}"
        options.append((key, who))
        for n in sizes:
            s = b["sizes"][n]
            ranked = s["ranked"]
            cands = [s["candidates"][k] for k in ranked]
            rows = [f"{i}. {candidate_label(k)}" for i, k in enumerate(ranked, start=1)]
            chart = dots_chart(
                f"{who} after {int(n):,} hands: the ranked candidates",
                rows,
                axis("Gain over equilibrium (bb/hand)", fmt="signed"),
                [
                    dot_series(
                        "Posterior mean, 90% interval",
                        "c1",
                        [(c["mean"], c["lower"], c["upper"]) for c in cands],
                    ),
                    dot_series("True gain", "c2", [(c["true_gain"],) for c in cands]),
                ],
                subtitle="Flagged: "
                + ("yes" if s["flagged"] else "no")
                + ". p-values: "
                + ", ".join(f"{k.replace('_', ' ')} {v:.3f}" for k, v in s["p"].items())
                + (f". The bot changed gear at hand {s['switch']:,}." if "switch" in s else "."),
                refs=[ref(0.0, "Equilibrium", axis_name="x")],
            )
            views[f"{key}|{n}"] = view(chart, notes=[s["explanation"]])
    return section(
        "fitcheck-examples",
        "Reports with a model-fit check",
        "The recommender's report for one connectors-first and one gear-change bot of each "
        "archetype (first seed), with the fit check's verdict in the explanation.",
        sources=[FITCHECK],
        controls=[
            control("bot", "Bot", options),
            control("n", "Hands observed", [(n, f"{int(n):,}") for n in sizes]),
        ],
        defaults={"n": "300"},
        views=views,
        takeaways=[
            "When a check fails, the explanation says which assumption broke (the hand order or "
            "the stability of the bot's play) and advises the equilibrium, while still showing "
            "what the model would have recommended.",
        ],
    )


def _short(pop: str) -> str:
    """Population name inside a sentence: "connectors first"."""
    return FIT_POP_LABEL[pop].split(" (")[0].lower()


# in the confirmation the stability-only rule is the pre-registered hypothesis, not post hoc
UNDER_TEST = "Equilibrium only when stability fails (under test)"
CONFIRM_RULES = [
    (rule, UNDER_TEST if rule == "stability_only" else lab, c) for rule, lab, c in FIT_RULES
]


def confirm_fallback(cf: dict[str, Any]) -> dict[str, Any]:
    sizes = _fit_sizes(cf)
    agg = cf["aggregate"]
    pred = cf["predictions"]
    views = {}
    for pop, plabel in FIT_POPS:
        a = agg[pop]
        gains = line_chart(
            "True gain of the three rules",
            HANDS_SEEN,
            axis("bb/hand above equilibrium", fmt="signed"),
            [
                series(
                    label,
                    color,
                    [
                        (int(n), x["mean"], x["lower"], x["upper"])
                        for n in sizes
                        for x in [a[n]["rules"][rule]["gain"]]
                    ],
                    dash=rule == "stability_only",
                )
                for rule, label, color in CONFIRM_RULES
            ],
            refs=[
                ref(a[sizes[-1]]["oracle_gain"]["mean"], "Oracle best response"),
                ref(0.0, "Equilibrium"),
            ],
            subtitle=f"{plabel}. Fresh seeds; 95% t-intervals across seeds.",
        )
        losses = line_chart(
            "Bots the rule does worse than equilibrium against",
            HANDS_SEEN,
            axis("Share of bots", fmt="pct", lo=0),
            [
                series(
                    label,
                    color,
                    _share_pts([(n, a[n]["rules"][rule]["loses"]) for n in sizes]),
                    dash=rule == "stability_only",
                )
                for rule, label, color in CONFIRM_RULES
            ],
            refs=[ref(0.05, "5%"), ref(0.10, "10%")],
            subtitle=f"{plabel}. Pooled over 150 bots; 95% Wilson intervals.",
        )
        diff = line_chart(
            "Stability only minus any check",
            HANDS_SEEN,
            axis("Difference in true gain (bb/hand)", fmt="signed"),
            [
                series(
                    "Paired difference, 95% interval",
                    "c5",
                    [
                        (int(n), d["mean"], d["lower"], d["upper"])
                        for n in sizes
                        for d in [cf["differences"][pop][n]["checked"]]
                    ],
                )
            ],
            refs=[ref(0.0, "No difference")],
            subtitle=f"{plabel}. Per-seed mean difference over the same bots.",
        )
        views[pop] = view(gains, losses, diff)

    def verdict(holds: bool) -> str:
        return "holds" if holds else "does NOT hold"

    gear = pred["gear_safety"]["by_size"]
    kept = pred["gain_kept"]["by_population"]
    safe = pred["order_safety"]["by_population"]
    takeaways = [
        "Prediction 1 (safety against gear-changers, at most 10% lost to): "
        + ", ".join(f"{v['value']:.1%} after {int(n):,} hands" for n, v in gear.items())
        + f"; {verdict(pred['gear_safety']['holds'])}.",
        "Prediction 2 (gain kept against wrong hand orders): stability only minus any check "
        + "; ".join(
            _short(pop)
            + " "
            + ", ".join(
                f"{v['mean']:+.3f} [{v['lower']:+.3f}, {v['upper']:+.3f}] at {int(n):,}"
                for n, v in by_n.items()
            )
            for pop, by_n in kept.items()
        )
        + f"; {verdict(pred['gain_kept']['holds'])}.",
        "Prediction 3 (at most 5% of wrong-order bots lost to, from 100 hands): at most "
        + ", ".join(
            f"{max(v['value'] for v in by_n.values()):.1%} ({_short(pop)})"
            for pop, by_n in safe.items()
        )
        + f"; {verdict(pred['order_safety']['holds'])}.",
        "What the rule still costs: against connectors-first bots after 1,000 hands it earns "
        f"{agg['connectors_first']['1000']['rules']['stability_only']['gain']['mean']:+.3f} "
        "bb/hand, against "
        f"{agg['connectors_first']['1000']['rules']['lower_bound']['gain']['mean']:+.3f} for "
        "trusting the model, because the stability check fires on some of them too; and after "
        "100 hands it still loses to "
        f"{agg['gear_change']['100']['rules']['stability_only']['loses']['share']:.0%} of "
        "gear-changers, before the stability check has the data to see the change.",
        f"Pre-registered decision: {pred['decision']}.",
    ]
    return section(
        "confirm-fallback",
        "Confirming the fallback rule on fresh seeds",
        "P3.5 found after the fact that falling back to equilibrium only when the stability check "
        "fails kept the gain against wrong hand orders and the safety against gear-changers. A "
        "rule found by looking at data is tested on new data: the same experiment on seeds never "
        "used before, with predictions written first.",
        sources=[CONFIRM],
        controls=[control("pop", "Bots", FIT_POPS)],
        defaults={"pop": "connectors_first"},
        views=views,
        takeaways=takeaways,
        method="Pre-registered in experiments/phase3_confirm.py. Fresh replicate seeds from "
        "SeedSequence([2026, 35]): new training populations, priors, test bots, hands and "
        "calibration bots; everything else as in P3.5.",
    )


IMPORT_OPS = (
    ("drop", "Drop a decision"),
    ("swap", "Swap two decisions"),
    ("truncate", "Cut the actions short"),
    ("wrong_player", "Decision by another player"),
    ("delimiter", "Delete a TOML delimiter"),
    ("amount", "Change a bet amount"),
    ("card", "Change a card"),
    ("stack", "Change a starting stack"),
    ("finishing", "Move chips between finishing stacks"),
)
IMPORT_STRUCTURAL = ("drop", "swap", "truncate", "wrong_player", "delimiter")
IMPORT_SOURCES = (("simulated", "Simulated hands"), ("pluribus", "Pluribus hands"))
IMPORT_CONDITIONS = (
    ("with_finishing", "With finishing stacks", "c1"),
    ("without_finishing", "Without finishing stacks", "c2"),
)
IMPORT_SUBSETS = (
    ("pluribus", "Pluribus vs professionals"),
    ("wsop", "WSOP 2023 Event #43 final table"),
    ("historical", "Historical hands"),
)
IMPORT_STATS = (
    ("vpip", "VPIP"),
    ("pfr", "PFR"),
    ("three_bet", "3-bet"),
    ("cbet", "C-bet"),
    ("fold_to_cbet", "Fold to c-bet"),
    ("wtsd", "WTSD"),
    ("wsd", "W$SD"),
)


def _wilson_point(block: dict[str, Any], outcome: str) -> tuple | None:
    s = block[f"{outcome}_share"]
    return None if s is None else (s["value"], s["lower"], s["upper"])


def import_faults(im: dict[str, Any]) -> dict[str, Any]:
    rows = [label for _, label in IMPORT_OPS]
    rt, fp = im["round_trip"], im["fault_predictions"]
    outcomes = (
        (
            "rejected",
            "Rejected",
            "Mutations the importer rejects",
            "one random mutation per operator, each imported with its finishing stacks and "
            "without them. 95% Wilson intervals; mutations that change nothing are left out.",
        ),
        (
            "result_changed",
            "Silent errors",
            "Accepted with some player's result changed (silent errors)",
            "the import succeeds but a player's net result differs from the true one. With "
            "finishing stacks recorded, the pre-registered prediction was none.",
        ),
    )
    views = {}
    for source, slabel in IMPORT_SOURCES:
        f = im["faults"][source]
        n = im["fault_hands"][source]
        for outcome, _, title, subtitle in outcomes:
            views[f"{source}|{outcome}"] = view(
                dots_chart(
                    title,
                    rows,
                    axis("Share of effective mutations", fmt="pct", lo=0, hi=1),
                    [
                        dot_series(
                            clabel,
                            color,
                            [_wilson_point(f[cond][op], outcome) for op, _ in IMPORT_OPS],
                        )
                        for cond, clabel, color in IMPORT_CONDITIONS
                    ],
                    subtitle=f"{slabel} ({n:,}): {subtitle}",
                ),
                notes=[
                    "The rest of each row was accepted with a changed record and every result "
                    "intact: the size of a bet nobody called, a folded player's card, a stack "
                    "deeper than anything bet. No check can see those from the file alone. The "
                    "last operator only exists when finishing stacks are recorded.",
                ],
            )
    silent = [
        (f["without_finishing"][op]["result_changed_share"]["value"], src, op)
        for src, f in im["faults"].items()
        for op, _ in IMPORT_OPS
        if f["without_finishing"][op]["result_changed_share"] is not None
    ]
    worst, worst_src, worst_op = max(silent)
    feats = rt["features"]
    takeaways = [
        f"Round trip: {rt['identical']:,} of {rt['checked']:,} simulated hands and redacted "
        f"copies came back identical after export to PHH and import ({rt['hands']:,} hands at "
        f"{rt['tables']} tables of 2 to 9 players; {feats.get('all_in', 0):,} with an all-in, "
        f"{feats.get('side_pot', 0):,} with side pots, {feats.get('cents', 0):,} in cents).",
        (
            "Structural damage was always caught: every dropped, swapped, misattributed or "
            "cut-short action and every broken delimiter was rejected"
            if fp["structural_rejected"]["minimum"] == 1
            else "Structural damage was caught in at least "
            f"{fp['structural_rejected']['minimum']:.1%} of cases"
        )
        + " (over operators, sources and conditions; pre-registered bound 99%).",
        "Finishing stacks act as a checksum. With them, "
        f"{fp['no_silent_with_finishing']['total']} mutations slipped through with a changed "
        "result; without them up to "
        f"{worst:.0%} did ({dict(IMPORT_OPS)[worst_op]}, {dict(IMPORT_SOURCES)[worst_src]}).",
        f"No import crashed in this run ({fp['no_crashes']['crashes']} errors other than typed "
        "rejections). The first pre-registered run did crash, see the method note.",
    ]
    return section(
        "import-faults",
        "What the hand-history importer catches",
        "Hands imported from text files are replayed through PokerKit's rules and rejected, "
        "with a reason, unless they replay exactly. To measure what that catches, valid files "
        "were damaged on purpose: one random mutation per hand, nine kinds, and each damaged "
        "hand imported with and without the finishing stacks the file records.",
        sources=[IMPORT],
        controls=[
            control("source", "Hands", IMPORT_SOURCES),
            control("outcome", "Outcome", [(o, label) for o, label, _, _ in outcomes]),
        ],
        views=views,
        takeaways=takeaways,
        method="Pre-registered in experiments/phase3_import.py (commit 582ac7c). Its first run "
        "falsified the no-crash prediction: 40 of 36,000 mutated imports crashed, all from "
        "cards dealt twice, and diagnosing them showed that a repeated hole card could be "
        "'repaired' by a show. The importer was fixed (commit 3c1f25c) and the experiment "
        "rerun with the same seed; these are the rerun's numbers. Rake not allowed.",
    )


def import_public_data(im: dict[str, Any]) -> dict[str, Any]:
    pd = im["public_data"]
    groups = (
        ("accepted", "Imported"),
        ("variant", "Not no-limit hold'em"),
        ("bb_ante", "Big-blind ante"),
        ("other", "Other reason"),
    )

    def group(kind: str) -> str:
        if kind.startswith("variant"):
            return "variant"
        return "bb_ante" if kind.startswith("antes differ") else "other"

    counts = {sub: dict.fromkeys([g for g, _ in groups], 0) for sub, _ in IMPORT_SUBSETS}
    for sub, _ in IMPORT_SUBSETS:
        counts[sub]["accepted"] = pd["by_subset"][sub]["accepted"]
    for k in pd["rejection_kinds"]:
        counts[k["subset"]][group(k["kind"])] += k["count"]
    used = [(g, label) for g, label in groups if any(counts[s][g] for s, _ in IMPORT_SUBSETS)]
    files = {sub: pd["by_subset"][sub]["files"] for sub, _ in IMPORT_SUBSETS}
    chart = heatmap(
        "Where the public hands go",
        [f"{label} ({files[sub]:,} files)" for sub, label in IMPORT_SUBSETS],
        [label for _, label in used],
        [[counts[sub][g] / files[sub] for g, _ in used] for sub, _ in IMPORT_SUBSETS],
        counts=[[counts[sub][g] for g, _ in used] for sub, _ in IMPORT_SUBSETS],
        fmt="pct",
        vmax=1.0,
        subtitle="Share of each subset's files. Schema v1 holds no-limit hold'em with uniform "
        "antes; anything else is rejected by name, never approximated.",
    )
    cc, fx, kinds = pd["cross_check"], pd["fixed_point"], pd["disagreement_kinds"]
    by = pd["by_subset"]
    takeaways = [
        f"{pd['accepted']:,} of {pd['files']:,} files imported: all "
        f"{by['pluribus']['accepted']:,} Pluribus hands, {by['historical']['accepted']} of "
        f"{by['historical']['files']} historical hands and {by['wsop']['accepted']} of the "
        f"{by['wsop']['files']} WSOP hands. The WSOP event is a mixed-game championship and "
        "its hold'em hands use a big-blind ante, which schema v1 could not hold (schema v2 "
        "imports them: see the schema v2 section).",
        f"Pre-registered: PokerKit's own loader replays every imported hand to the same result. "
        f"It does for {cc['agree']:,} of {cc['of']:,}, so the prediction does not hold."
        if not cc["holds"]
        else f"PokerKit's own loader replays all {cc['of']:,} imported hands to the same result.",
        f"Post hoc: the {kinds['odd_chip']} disagreements are all split pots with an odd chip. "
        "The Pluribus files give each winner half a chip; the importer keeps that (those "
        "hands are in tenths of a chip) and PokerKit's whole-chip replay cannot."
        if kinds["odd_chip"] and not kinds["other"]
        else f"Disagreements: {kinds['odd_chip']} odd-chip splits, {kinds['other']} other.",
        f"Exported and imported again, {fx['same']:,} of {fx['of']:,} hands come back "
        "identical (pre-registered: all).",
    ]
    return section(
        "import-public-data",
        "Importing public research hands",
        "The open PHH dataset collects hand histories from research and broadcasts. Its Pluribus "
        "subset is the 10,000 hands that the Pluribus AI played against professionals, "
        "published with the paper. The subset scraped from online sites is never fetched.",
        sources=[IMPORT],
        views={"": view(chart)},
        takeaways=takeaways,
        method=f"Dataset {im['dataset']['url']} at commit {im['dataset']['commit'][:7]} "
        f"({im['dataset']['files']:,} files, SHA-256 {im['dataset']['content_sha256'][:12]}...), "
        "fetched by a sparse clone (experiments/datasets.py). Imported without rake; the "
        "cross-check reads each file with PokerKit's own loader and standard evaluator.",
    )


def import_pluribus(im: dict[str, Any]) -> dict[str, Any]:
    stats = im["pluribus_stats"]
    bots = im["phase1_bots"]["per_archetype"]
    players = list(stats)
    rows = [f"{pid} ({stats[pid]['hands']:,} hands)" for pid in players]
    views = {}
    for name, label in IMPORT_STATS:
        means = [bots[a][name]["estimate"] / 100 for a in bots]
        tag = bots["tag"][name]["estimate"] / 100
        views[name] = view(
            dots_chart(
                f"{label} of each player in the Pluribus hands",
                rows,
                axis(label, fmt="pct", lo=0),
                [
                    dot_series(
                        "Imported hands, 95% Wilson interval",
                        "c1",
                        [
                            (v["value"] / 100, v["lower"] / 100, v["upper"] / 100)
                            for v in (stats[pid][name] for pid in players)
                        ],
                    )
                ],
                ranges=[(min(means), max(means))] * len(rows),
                range_label="Phase 1 bots: lowest to highest archetype average",
                refs=[ref(tag, "Phase 1 TAG bots", axis_name="x")],
                subtitle="From public information only (hands redacted to what a participant "
                "sees), with the Phase 1 stats code. Player names are the dataset's pseudonyms.",
            )
        )
    humans, pluribus = stats["All humans"], stats["Pluribus"]

    def span(stat: str) -> tuple[float, float]:
        values = [bots[a][stat]["estimate"] for a in bots]
        return min(values), max(values)

    lo, hi = span("wtsd")
    pre = ("vpip", "pfr", "three_bet")
    tag, lag = bots["tag"], bots["lag"]
    between = all(
        min(tag[k]["estimate"], lag[k]["estimate"])
        <= humans[k]["value"]
        <= max(tag[k]["estimate"], lag[k]["estimate"])
        for k in pre
    )
    takeaways = [
        f"The professionals went to showdown after seeing the flop {humans['wtsd']['value']:.0f}% "
        f"of the time and Pluribus {pluribus['wtsd']['value']:.0f}%; the Phase 1 archetypes "
        f"average {lo:.0f}% to {hi:.0f}%. Phase 1 logged that its bots reach showdown too "
        "often; real data now measure the gap.",
        f"Preflop the professionals' VPIP / PFR / 3-bet is {humans['vpip']['value']:.0f}% / "
        f"{humans['pfr']['value']:.0f}% / {humans['three_bet']['value']:.1f}%, "
        + ("between the TAG bots' " if between else "against the TAG bots' ")
        + f"{tag['vpip']['estimate']:.0f}% / {tag['pfr']['estimate']:.0f}% / "
        f"{tag['three_bet']['estimate']:.1f}% and the LAG bots' "
        f"{lag['vpip']['estimate']:.0f}% / {lag['pfr']['estimate']:.0f}% / "
        f"{lag['three_bet']['estimate']:.1f}%.",
        f"After the flop they bet less: c-bet {humans['cbet']['value']:.0f}% against "
        f"{bots['tag']['cbet']['estimate']:.0f}% for TAG and "
        f"{bots['lag']['cbet']['estimate']:.0f}% for LAG bots.",
        "Descriptive only. Fitting population priors to real data is the next item (P3.7).",
    ]
    return section(
        "import-pluribus",
        "Real hands next to the Phase 1 bots",
        "Imported hands go straight into the Phase 1 profiling code. These are the statistics "
        f"of Pluribus and the {len(players) - 2} professionals it played (by their dataset "
        "pseudonyms), against the range of the bots the classifier was trained on.",
        sources=[IMPORT],
        controls=[control("stat", "Statistic", IMPORT_STATS)],
        defaults={"stat": "wtsd"},
        views=views,
        takeaways=takeaways,
        method="Every imported Pluribus hand, redacted to public information; statistics as "
        "defined in poker_lab/profiling/stats.py (Phase 1). Bot values: the Phase 1 sanity "
        "check (experiments/results/phase1/bot_sanity), mean over bots per archetype.",
    )


PRIOR_STATS = (
    ("vpip", "VPIP"),
    ("pfr", "PFR"),
    ("three_bet", "3-bet"),
    ("fold_to_3bet", "Fold to 3-bet"),
    ("cbet", "C-bet"),
    ("fold_to_cbet", "Fold to c-bet"),
    ("aggr_share", "Aggression share"),
    ("fold_to_bet", "Fold to a bet"),
    ("wtsd", "WTSD"),
    ("wsd", "W$SD"),
)
PRIOR_STAT_LABEL = dict(PRIOR_STATS)
PRIOR_KINDS = (
    ("human", "Prior fit to the other players", "c1"),
    ("synthetic", "Prior fit to the Phase 1 bots", "c4"),
    ("uniform", "Uniform Beta(1, 1) (the default)", "c2"),
)
PRIOR_POPS = (
    ("humans", "The professionals, one left out at a time"),
    ("reverse", "The Phase 1 held-out bots"),
    ("control", "Control: bots in the professionals' seats"),
)
ACTION_MODELS = (
    ("syn_sit", "Trained on bots: situation only", "c4", False),
    ("syn_prof", "Trained on bots: + profile", "c2", True),
    ("hum_sit", "Trained on the other players: situation only", "c1", False),
    ("hum_prof", "Trained on the other players: + profile", "c3", True),
)
WINDOW_HANDS = axis("Hands observed", scale="log", fmt="int")


def _verdict(holds: bool) -> str:
    return "holds" if holds else "does not hold"


def _between_sd(mu: Any, kappa: Any) -> float:
    mu, kappa = np.asarray(mu, dtype=float), np.asarray(kappa, dtype=float)
    return float(np.mean(np.sqrt(mu * (1 - mu) / (kappa + 1))))


def priors_fit(pr: dict[str, Any]) -> dict[str, Any]:
    hp, syn = pr["populations"]["humans"]["priors"], pr["synthetic_priors"]
    human = hp["human"]
    rows = [label for _, label in PRIOR_STATS]
    bot = hp.get("bot") or {}
    centers = [
        dot_series(
            "Fit to the 13 professionals (95% bootstrap over players)",
            "c1",
            [(human[s]["mean"], *human[s]["mean_ci"]) for s, _ in PRIOR_STATS],
        ),
        dot_series(
            "Fit to the Phase 1 training bots (mean of 5 replicates)",
            "c4",
            [(float(np.mean(syn[s]["mean"])),) for s, _ in PRIOR_STATS],
        ),
    ]
    if bot:
        centers.append(
            dot_series(
                "Pluribus's own rate (not used in any fit)",
                "c7",
                [(bot[s],) if bot.get(s) is not None else None for s, _ in PRIOR_STATS],
            )
        )
    mean_view = view(
        dots_chart(
            "Where each population is centered: prior means",
            rows,
            axis("Prior mean", fmt="pct", lo=0, hi=1),
            centers,
            subtitle="Beta-binomial priors fit by maximum marginal likelihood (empirical "
            "Bayes, Step 10).",
        )
    )
    spread_view = view(
        dots_chart(
            "How different the players are: between-player spread",
            rows,
            axis("Between-player SD of the true rate", fmt="pct", lo=0),
            [
                dot_series(
                    "Professionals (95% bootstrap over players)",
                    "c1",
                    [(human[s]["between_sd"], *human[s]["between_sd_ci"]) for s, _ in PRIOR_STATS],
                ),
                dot_series(
                    "Phase 1 training bots (mean of 5 replicates)",
                    "c4",
                    [
                        (_between_sd(syn[s]["mean"], syn[s]["concentration"]),)
                        for s, _ in PRIOR_STATS
                    ],
                ),
            ],
            subtitle="For a prior with mean mu and concentration kappa = alpha + beta, "
            "SD = sqrt(mu (1 - mu) / (kappa + 1)): how far a typical player's true rate sits "
            "from the population's center.",
        )
    )
    ratio = {
        s: _between_sd(syn[s]["mean"], syn[s]["concentration"]) / human[s]["between_sd"]
        for s, _ in PRIOR_STATS
    }
    lo, hi = min(ratio, key=ratio.get), max(ratio, key=ratio.get)
    gap = max(
        (s for s, _ in PRIOR_STATS),
        key=lambda s: abs(human[s]["mean"] - float(np.mean(syn[s]["mean"]))),
    )
    takeaways = [
        f"The professionals are far more alike than the bots: the bots differ from each other "
        f"{ratio[lo]:.1f} times as much as the professionals do ({PRIOR_STAT_LABEL[lo]}) to "
        f"{ratio[hi]:.1f} times as much ({PRIOR_STAT_LABEL[hi]}). The professionals' WTSD varies "
        f"by about {100 * human['wtsd']['between_sd']:.1f} points from player to player.",
        f"The centers differ most for {PRIOR_STAT_LABEL[gap]}: {human[gap]['mean']:.0%} for "
        f"the professionals against {float(np.mean(syn[gap]['mean'])):.0%} for the bots. "
        f"Bet sizes too: they bet {human['bet_size']['mean']:.2f} of the pot after the flop "
        f"and open to {human['open_size']['mean']:.1f} big blinds; the bots "
        f"{float(np.mean(syn['bet_size']['mean'])):.2f} and "
        f"{float(np.mean(syn['open_size']['mean'])):.1f}.",
        "Pluribus sits at the professionals' center on almost every statistic.",
    ]
    return section(
        "priors-fit",
        "Priors fit to real players",
        "Empirical-Bayes priors for every binomial statistic, fit to the 13 professionals of "
        "the Pluribus hands, next to the priors Phase 1 fit to its bots.",
        sources=[PRIORS],
        controls=[control("measure", "Show", [("spread", "Spread"), ("mean", "Center")])],
        views={"mean": mean_view, "spread": spread_view},
        takeaways=takeaways,
        method="EmpiricalBayesShrinker.fit on each population's counter totals (public "
        "information only); the professionals' bet-size priors use split halves (hand_half). "
        "Intervals: 1,000 refits on players resampled with replacement.",
    )


def _rmse(block: dict[str, Any], prior: str, stat: str) -> float:
    v = np.asarray(block["rmse_vs_truth"][prior], dtype=float)
    if stat == "all":
        return float(np.nanmean(v))
    return float(v[[s for s, _ in PRIOR_STATS].index(stat)])


def priors_estimation(pr: dict[str, Any]) -> dict[str, Any]:
    pops = {"humans": pr["populations"]["humans"]["estimation"], "reverse": pr["reverse"]}
    if "control" in pr["populations"]:
        pops["control"] = pr["populations"]["control"]["estimation"]
    pop_opts = [(k, label) for k, label in PRIOR_POPS if k in pops]
    stat_opts = [("all", "Mean of the 10 statistics"), *PRIOR_STATS]
    views = {}
    for pop, pop_label in pop_opts:
        for stat, stat_label in stat_opts:
            views[f"{pop}|{stat}"] = view(
                line_chart(
                    f"{stat_label}: estimation error. {pop_label}",
                    WINDOW_HANDS,
                    axis("RMSE against the true rate (percentage points)", fmt="num", lo=0),
                    [
                        series(
                            label,
                            color,
                            [(int(n), 100 * _rmse(b, key, stat)) for n, b in pops[pop].items()],
                        )
                        for key, label, color in PRIOR_KINDS
                    ],
                    subtitle="The posterior mean after n hands against the player's rate in all "
                    "their other hands, corrected for that rate's own binomial noise.",
                )
            )
    h, rev, pred = pops["humans"], pr["reverse"], pr["predictions"]
    n0 = next(iter(h))
    r100 = "100" if "100" in rev else next(iter(rev))
    worse = {n: len(v) for n, v in pred["B3"]["worse_than_uniform"].items()}
    b1 = pred["B1"]["failures"]
    takeaways = [
        f"After {int(n0):,} hands of a new professional, a prior fit to the other twelve is off "
        f"by {100 * _rmse(h[n0], 'human', 'all'):.1f} points on average, against "
        f"{100 * _rmse(h[n0], 'uniform', 'all'):.1f} with the uniform default and "
        f"{100 * _rmse(h[n0], 'synthetic', 'all'):.1f} with the bots' prior.",
        "Pre-registered B1, the professionals' prior beats uniform for every statistic up to "
        f"200 hands: {_verdict(pred['B1']['holds'])}"
        + (
            f"; it does at 25, 50 and 100 hands, and at 200 hands {len(b1)} intervals reach "
            f"below 0 ({', '.join(PRIOR_STAT_LABEL[f.split(' at ')[0]] for f in b1)})."
            if b1
            else "."
        )
        + " B2, it beats the bots' prior: "
        + _verdict(pred["B2"]["holds"])
        + f" ({len(pred['B2']['failures'])} of 40 statistic-size pairs are not clearly better).",
        f"Turned on the bots, the professionals' prior is off by "
        f"{100 * _rmse(rev[r100], 'human', 'all'):.1f} points after {int(r100):,} hands, against "
        f"{100 * _rmse(rev[r100], 'uniform', 'all'):.1f} for uniform; it is worse than the "
        "default for "
        + " and ".join(f"{v} of 10 statistics at {int(n):,} hands" for n, v in worse.items())
        + f" (B3 {_verdict(pred['B3']['holds'])}). A confident prior from one population is a "
        "liability on another.",
    ]
    return section(
        "priors-estimation",
        "Which prior estimates a new player best?",
        "Leave one professional out, fit the prior to the other twelve, watch n of the left-out "
        "player's hands and estimate their statistics; score against everything else they "
        "played. The same priors are also turned on the Phase 1 bots.",
        sources=[PRIORS],
        controls=[
            control("pop", "Players", pop_opts),
            control("stat", "Statistic", stat_opts),
        ],
        views=views,
        takeaways=takeaways,
        method="Disjoint windows of n hands (at most 20 per player; a player needs 2n hands). "
        "Differences in squared error against the held-out rate are unbiased for differences "
        "against the true rate, because the held-out rate's noise is independent of the "
        "window; the RMSE shown subtracts an unbiased estimate of that noise. Bots: windows "
        "from the first half of each bot's 2,000 hands, the target from the second half.",
    )


def priors_refit_size(pr: dict[str, Any]) -> dict[str, Any]:
    mp = pr["min_players"]
    palette = ("c1", "c2", "c3", "c4", "c5", "c6")
    opts = [("human", "Professionals"), ("synthetic", "Phase 1 bots")]
    views = {}
    for who, label in opts:
        d = mp[who]
        ns = sorted({int(n) for k in d for n in d[k]})
        views[who] = view(
            line_chart(
                f"{label}: a prior refit on k players against the uniform default",
                axis("Players the prior is fit to", scale="log", fmt="int"),
                axis("Mean squared error saved, x 10^4 (mean of 10 statistics)", fmt="num"),
                [
                    series(
                        f"{n:,} hands observed",
                        palette[j % len(palette)],
                        [
                            (int(k), 1e4 * float(np.nanmean(np.asarray(d[k][str(n)], float))))
                            for k in d
                            if str(n) in d[k]
                        ],
                    )
                    for j, n in enumerate(ns)
                ],
                refs=[ref(0, "no better than the default")],
                subtitle="Positive: the refit beats the uniform prior. Mean over 20 random "
                "subsets of k players (one at the largest k).",
            )
        )
    takeaways = [
        f"The pre-registered rule set the default min_players of ProfileStore.refit_priors to "
        f"{mp['default']} (was 30): the smallest k from which a refit is no worse than uniform "
        f"on average at every sample size is {mp['k_human']} for the professionals and "
        f"{mp['k_synthetic']} for the bots.",
        "The professionals are so alike that a prior fit to two of them already helps. The bots "
        "are varied, and two or three of them make a prior that is confidently wrong about the "
        "rest.",
        "Even with all 300 training bots, the c-bet prior is worse than uniform: the fit weighs "
        "players by their opportunities, and the bots that c-bet least get the fewest chances.",
    ]
    return section(
        "priors-refit-size",
        "How many players does a refit need?",
        "ProfileStore.refit_priors replaces the uniform default with priors fit to the players "
        "it has stored. Fit to too few, those priors are worse than the default.",
        sources=[PRIORS],
        controls=[control("pop", "Players", opts)],
        views=views,
        takeaways=takeaways,
        method="Priors fit to k players (professionals: subsets of the other twelve; bots: "
        "subsets of each replicate's 300 training bots), scored as in the previous section.",
    )


def priors_support(pr: dict[str, Any]) -> dict[str, Any]:
    hc = pr["populations"]["humans"]["classifier"]
    cc = pr["populations"].get("control", {}).get("classifier")
    sup = hc["support"]
    ns = list(sup)
    groups = [
        ("Phase 1 held-out bots", "c4", [sup[n]["bots_flagged"] for n in ns]),
        ("Professionals", "c1", [sup[n]["humans_flagged"] for n in ns]),
    ]
    if cc:
        control_rates = [cc["support"][n]["humans_flagged"] for n in ns]
        groups.append(("Control: bots in the professionals' seats", "c3", control_rates))
    if all("bot_flagged" in sup[n] for n in ns):
        groups.append(("Pluribus", "c7", [sup[n]["bot_flagged"] for n in ns]))
    alpha = pr["config"]["support_alpha"]
    trend = view(
        line_chart(
            "Profiles flagged as unlike every training archetype",
            axis("Hands in the window", scale="log", fmt="int"),
            axis("Windows flagged", fmt="pct", lo=0, hi=1),
            [
                series(label, color, [(int(n), v) for n, v in zip(ns, vals, strict=True)])
                for label, color, vals in groups
            ],
            refs=[ref(alpha, f"alpha = {alpha:.0%}")],
            subtitle="Conformal p-value of the smallest Mahalanobis distance to an archetype, "
            "against held-out training bots of the same sample size.",
        )
    )
    n_show = "100" if "100" in sup else ns[0]
    by_player = sup[n_show]["humans_by_player"]
    players = sorted(by_player, key=lambda p: (-by_player[p], p))
    per_player = view(
        dots_chart(
            f"Each professional at {int(n_show):,} hands",
            players,
            axis("Windows flagged", fmt="pct", lo=0, hi=1),
            [dot_series("Share of windows flagged", "c1", [(by_player[p],) for p in players])],
            refs=[ref(alpha, "alpha", axis_name="x")],
            subtitle="Each player's windows of this size (at most 20), averaged over the five "
            "classifier replicates.",
        )
    )
    pred = pr["predictions"]
    all_from = next((n for n in ns if (sup[n]["humans_flagged"] or 0) >= 0.999), None)
    takeaways = [
        f"Held-out bots are flagged at most {max(groups[0][2]):.1%} of the time "
        f"(calibrated to {alpha:.0%}); the professionals {sup[ns[0]]['humans_flagged']:.0%} "
        f"after {int(ns[0]):,} hands"
        + (f" and every window from {int(all_from):,} hands" if all_from else "")
        + f". Pre-registered: {_verdict(pred['C1a']['holds'])} (bots) and "
        f"{_verdict(pred['C1b']['holds'])} (professionals, at least 80% from 500 hands).",
        "A flagged profile is one the classifier was never trained to label: its archetype "
        "probabilities are not calibrated for that player, however confident they look.",
    ]
    if cc:
        cp = pr["control_predictions"]["C1"]
        small = [v for n, v in zip(ns, control_rates, strict=True) if int(n) <= 200]
        takeaways.append(
            f"The control, Phase 1 bots replaying the professionals' seats and hand counts, is "
            f"flagged {min(small):.0%} to {max(small):.0%} of the time up to 200 hands but "
            f"{control_rates[-1]:.0%} at {int(ns[-1]):,}: the pre-registered bound of 25% "
            f"{_verdict(cp['holds'])}. Statistics depend on the table, and these players meet "
            "the same few opponents for thousands of hands, while Phase 1's league redraws its "
            "tables every 50 hands. At small samples the flags come from the players; at large "
            "ones the table matters too."
        )
    return section(
        "priors-support",
        "Do the professionals look like any training archetype?",
        "The archetype classifier returns probabilities for any input. A support check asks "
        "first whether a profile resembles any archetype it was trained on.",
        sources=[PRIORS],
        controls=[control("view", "Show", [("trend", "By sample size"), ("players", "By player")])],
        views={"trend": trend, "players": per_player},
        takeaways=takeaways,
        method="poker_lab.profiling.SupportCheck, fit per replicate to Phase 1's training "
        "windows (one Gaussian per archetype in the classifier's logit/log feature space; "
        "thresholds from 20% of the training players held out). Players weigh equally; shares "
        "are averaged over the five replicates.",
    )


def priors_classifier(pr: dict[str, Any]) -> dict[str, Any]:
    hc = pr["populations"]["humans"]["classifier"]
    cc = pr["populations"].get("control", {}).get("classifier")
    cons = hc["consistency"]
    ns = list(cons)

    def pts(block: dict[str, Any]) -> list[tuple]:
        return [
            (
                int(n),
                block[n]["difference"]["value"],
                block[n]["difference"]["lower"],
                block[n]["difference"]["upper"],
            )
            for n in ns
            if n in block
        ]

    ser = [series("Professionals", "c1", pts(cons))]
    if cc:
        ser.append(
            series("Control: bots in the professionals' seats", "c3", pts(cc["consistency"]))
        )
    ref_n = pr["config"]["consistency_reference"]
    consistency = view(
        line_chart(
            "Overconfidence compared with bots of the same label",
            axis("Hands in the window", scale="log", fmt="int"),
            axis("Gap minus the matched bots' gap", fmt="signed"),
            ser,
            refs=[ref(0.10, "+0.10 margin"), ref(-0.10, "-0.10 margin"), ref(0, "as for bots")],
            subtitle=f"Gap = mean top probability - share of windows whose label matches the "
            f"label of a later {ref_n:,}-hand block of the same player. Bands: 95% intervals "
            "over players.",
        )
    )
    labels = hc["labels"]
    sizes = [n for n in labels if labels[n]["humans"] is not None]
    names = list(ARCHETYPE_LABELS.values())
    palette = ("c1", "c2", "c3", "c4", "c5")
    label_view = view(
        line_chart(
            "What the classifier calls the professionals",
            axis("Hands in the window", scale="log", fmt="int"),
            axis("Share of windows", fmt="pct", lo=0, hi=1),
            [
                series(name, palette[k], [(int(n), labels[n]["humans"][k]) for n in sizes])
                for k, name in enumerate(names)
            ],
            subtitle="Predicted archetype of each window (players weigh equally). Mean top "
            "probability "
            + ", ".join(
                f"{labels[n]['humans_top_prob']:.2f} at {int(n):,}" for n in (sizes[0], sizes[-1])
            )
            + " hands; on held-out bots "
            + ", ".join(f"{hc['support'][n]['bots_top_prob']:.2f}" for n in (sizes[0], sizes[-1]))
            + ".",
        )
    )
    pred = pr["predictions"]["C3"]
    cp = pr.get("control_predictions", {}).get("C3")
    top = labels[sizes[-1]]["humans"]
    order = np.argsort(top)[::-1]
    lead = [n for n in ns if cons[n]["difference"]["lower"] > 0]
    takeaways = [
        f"Pre-set verdict: calibration {pred['verdict']} on the professionals (predicted: "
        "fails)"
        + (f"; {cp['verdict']} on the control." if cp else ".")
        + (
            f" After {', '.join(f'{int(n):,}' for n in lead)} hands the classifier is "
            f"{min(cons[n]['difference']['value'] for n in lead):.2f} to "
            f"{max(cons[n]['difference']['value'] for n in lead):.2f} more confident than its "
            "agreement with a longer look justifies, compared with bots it labels the same way."
            if lead
            else ""
        ),
        f"It calls them {names[order[0]]} ({top[order[0]]:.0%} of windows at "
        f"{int(sizes[-1]):,} hands) or {names[order[1]]} ({top[order[1]]:.0%}), never calling "
        f"station or maniac after 50 hands, with lower confidence than on bots "
        f"({labels[sizes[-1]]['humans_top_prob']:.2f} against "
        f"{hc['support'][sizes[-1]]['bots_top_prob']:.2f}).",
        "Together with the support check: on these players the archetype probabilities are "
        "neither in their training range nor calibrated. Their profile statistics, with real "
        "priors, are what can be used.",
    ]
    return section(
        "priors-classifier",
        "Does the classifier's confidence mean the same on real players?",
        "Real players have no true archetype, but a calibrated classifier makes a testable "
        "promise: its confidence after n hands should match how often a longer look at the "
        "same player gives the same label.",
        sources=[PRIORS],
        controls=[
            control("view", "Show", [("consistency", "Consistency check"), ("labels", "Labels")])
        ],
        views={"consistency": consistency, "labels": label_view},
        takeaways=takeaways,
        method="Phase 1's primary classifier (logistic, shrunk stats, one temperature), "
        f"retrained per replicate. Windows from the first half of each {2 * ref_n:,}-hand "
        "chunk, the reference label from the second half (8 professionals have a chunk). Each "
        "player is compared with held-out bots whose own reference block got the same label. "
        "Verdict: transfers if every 95% interval lies within +-0.10, fails if one excludes 0, "
        "otherwise inconclusive.",
    )


def priors_action(pr: dict[str, Any]) -> dict[str, Any]:
    pops = [
        (k, label)
        for k, label in (("humans", "Professionals"), ("control", "Control: bots in their seats"))
        if k in pr["populations"]
    ]
    metrics = (("log_loss", "Log-loss per decision (nats)"), ("ece", "Calibration error"))
    views = {}
    for pop, label in pops:
        by = pr["populations"][pop]["action"]["by_checkpoint"]
        for metric, metric_label in metrics:
            views[f"{pop}|{metric}"] = view(
                line_chart(
                    f"{label}: predicting the next action",
                    axis("Hands observed before the predicted 100", fmt="int"),
                    axis(metric_label, fmt="num"),
                    [
                        series(name, color, [(int(c), by[c][metric][key]) for c in by], dash=dash)
                        for key, name, color, dash in ACTION_MODELS
                    ],
                    subtitle="Fold / check-call / bet-raise. Every model predicts players it "
                    "never saw. Calibration error: top label, 10 bins.",
                )
            )
    by = pr["populations"]["humans"]["action"]["by_checkpoint"]
    pred = pr["predictions"]
    c0 = next(iter(by))
    d1 = [v["syn_minus_hum_log_loss"]["value"] for v in by.values()]
    red = pred["D2"]["reductions"]
    ece = [(v["ece"]["syn_sit"], v["ece"]["hum_sit"]) for v in by.values()]
    takeaways = [
        f"A model trained on bots predicts the professionals' decisions worse than one trained "
        f"on the other professionals, by {min(d1):.2f} to {max(d1):.2f} nats per decision "
        f"({by[c0]['log_loss']['syn_sit']:.3f} against {by[c0]['log_loss']['hum_sit']:.3f} "
        f"with no history; D1 {_verdict(pred['D1']['holds'])}).",
        "Knowing the player barely helps. Adding the profile changes the log-loss (+ = worse) by "
        + ", ".join(f"{-v['human']:+.1%}" for v in red.values())
        + " for the model trained on professionals and "
        + ", ".join(f"{-v['synthetic']:+.1%}" for v in red.values())
        + f" for the bot-trained one (D2, under 3%: {_verdict(pred['D2']['holds'])}, by the "
        "bot-trained model at 1,500 hands). On bots the profile cut the log-loss by 11-13%.",
        f"The bot-trained model is also worse calibrated (error {min(e[0] for e in ece):.3f} "
        f"to {max(e[0] for e in ece):.3f} against {min(e[1] for e in ece):.3f} to "
        f"{max(e[1] for e in ece):.3f}); D3 {_verdict(pred['D3']['holds'])}: the interval "
        "excludes 0 only with no or little history.",
        "For these players the population model is what matters, and it has to come from real "
        "players. This is the metric Phase 5 will use on live hands.",
    ]
    return section(
        "priors-action",
        "Predicting real players' next action",
        "The metric Phase 5 will use on live data, run now on the professionals: a model trained "
        "on bots against one trained on the other professionals, each with and without the "
        "player's profile.",
        sources=[PRIORS],
        controls=[control("pop", "Players", pops), control("metric", "Metric", list(metrics))],
        views=views,
        takeaways=takeaways,
        method="Phase 1's next-action learner. Bot-trained: per replicate on its training bots' "
        "decisions, as in Phase 1. Trained on the other players: leave one player out, on all "
        "their decisions, profiles from the hands before each 100-hand block, with their "
        "leave-one-out prior. Means weigh players equally; intervals: bootstrap over players.",
    )


SEQ_MODELS = (
    ("lr", "Logistic, shrunk stats (Phase 1)", "c1"),
    ("gbm", "Boosted trees, shrunk stats", "c6"),
    ("seq", "Sequence model (GRU)", "c2"),
    ("bag", "Sequence model, order dropped", "c3"),
    ("counters", "Same network on the stats counters", "c4"),
)
SEQ_LABEL = {k: label for k, label, _ in SEQ_MODELS}


def _seq_points(block: dict[str, Any], key: str | None = None) -> list[tuple]:
    """(n, mean, lower, upper) from a {n: ci} or {n: {key: ci}} block."""
    out = []
    for n, v in block.items():
        d = v if key is None else v[key]
        out.append((int(n), d["mean"], d["lower"], d["upper"]))
    return out


def _span(values: list[float], fmt: str = "+.3f") -> str:
    lo, hi = min(values), max(values)
    return f"{lo:{fmt}}" if f"{lo:{fmt}}" == f"{hi:{fmt}}" else f"{lo:{fmt}} to {hi:{fmt}}"


def sequence_hands(sq: dict[str, Any]) -> dict[str, Any]:
    a = sq["a"]
    sizes = [str(n) for n in a["sizes"]]
    metrics = (
        ("accuracy", "Accuracy", "pct"),
        ("log_loss", "Log-loss (nats), one temperature", "num"),
        ("log_loss_by_size", "Log-loss (nats), one temperature per size", "num"),
        ("ece", "Calibration error (10 bins)", "num"),
    )
    views = {}
    for metric, label, fmt in metrics:
        views[metric] = view(
            line_chart(
                "Identifying a bot's archetype, by model",
                WINDOW_HANDS,
                axis(label, fmt=fmt, lo=0, hi=1 if metric == "accuracy" else None),
                [
                    series(
                        name,
                        color,
                        [
                            (
                                int(n),
                                *(
                                    a["metrics"][m][n][metric][k]
                                    for k in ("mean", "lower", "upper")
                                ),
                            )
                            for n in sizes
                        ],
                        dash=m in ("gbm", "counters"),
                    )
                    for m, name, color in SEQ_MODELS
                    if m in a["metrics"]
                ],
                subtitle="Held-out bots, Phase 1's windows (up to 10 per bot and size); bands: "
                "95% t-intervals over the five replicates.",
            )
        )
    met = a["metrics"]
    first, last = sizes[0], sizes[-1]
    h90 = {m: a["hands_to"][m]["0.9"] for m in a["hands_to"]}

    def ll(m: str, n: str) -> float:
        return met[m][n]["log_loss"]["mean"]

    takeaways = [
        "With Phase 1's 300 training bots, reading every decision does not beat ten "
        f"hand-written rates and two sizes. The sequence model ties the logistic model at 50 "
        f"to 100 hands (log-loss {ll('seq', '100'):.3f} against {ll('lr', '100'):.3f}) and "
        f"falls behind with more hands: {met['seq'][last]['accuracy']['mean']:.1%} accurate "
        f"at {int(last):,} hands against {met['lr'][last]['accuracy']['mean']:.1%}.",
        f"At {int(first)} hands it is a little more accurate "
        f"({met['seq'][first]['accuracy']['mean']:.1%} against "
        f"{met['lr'][first]['accuracy']['mean']:.1%}) but less well calibrated (error "
        f"{met['seq'][first]['ece']['mean']:.3f} against {met['lr'][first]['ece']['mean']:.3f}), "
        f"so its log-loss is worse ({ll('seq', first):.3f} against {ll('lr', first):.3f}).",
        "Hands to 90% accuracy (geometric mean over replicates): logistic "
        f"{h90['lr']['geometric_mean']:,.0f}, sequence model "
        f"{h90['seq']['geometric_mean']:,.0f} ({h90['seq']['reached']} of 5 replicates reach "
        f"it), order dropped {h90['bag']['geometric_mean']:,.0f}, counters network "
        f"{h90['counters']['geometric_mean']:,.0f}. Phase 1's boosted trees have the highest "
        "log-loss from 50 to 500 hands.",
    ]
    return section(
        "sequence-hands",
        "Hand-written stats or raw action sequences?",
        "Phase 1 identifies a player's archetype from ten hand-written rates and two sizes. A "
        "PyTorch model that reads every decision of every hand instead, on the same held-out "
        "bots, windows and metrics.",
        sources=[SEQUENCE],
        controls=[control("metric", "Metric", [(m, lab) for m, lab, _ in metrics])],
        views=views,
        takeaways=takeaways,
        method="Each hand becomes tokens: every decision of the player (the public situation, "
        "fold/check/call/bet/raise, the bet size) and a closing token (saw the flop, showdown, "
        "won at showdown). A GRU reads a hand's tokens in order; each hand adds soft successes "
        "and failures to 16 learned rates, which a window turns into shrunk rates under a "
        "learned Beta prior (Phase 1's recipe, with learned stats); an MLP gives the "
        "probabilities. Ablations: the order within a hand dropped, and the hand's stats "
        "counters as input. Fit to Phase 1's 300 training bots per replicate (the same five "
        "replicates, re-simulated and checked identical); 20% of them pick the epoch and the "
        "temperature.",
    )


def sequence_gain(sq: dict[str, Any]) -> dict[str, Any]:
    a = sq["a"]
    others = [(m, name, color) for m, name, color in SEQ_MODELS if m in a["paired"]]
    views = {
        key: view(
            line_chart(
                "Log-loss gain over Phase 1's logistic model, same windows",
                WINDOW_HANDS,
                axis("Gain (nats; above 0: better than logistic)", fmt="signed"),
                [
                    series(name, color, _seq_points(a["paired"][m], key))
                    for m, name, color in others
                ],
                refs=[ref(0, "Phase 1's classifier")],
                subtitle=f"{label}. Mean over windows of the logistic model's log-loss minus "
                "the model's, paired; bands: 95% t-intervals over five replicates.",
            )
        )
        for key, label in (
            ("log_loss_gain", "One temperature"),
            ("log_loss_gain_by_size", "One temperature per sample size"),
        )
    }
    diffs = a["differences"]
    views["ablations"] = view(
        line_chart(
            "What the order and the tokens add",
            WINDOW_HANDS,
            axis("Log-loss difference (nats)", fmt="signed"),
            [
                series("Sequence model - order dropped", "c2", _seq_points(diffs["seq-bag"])),
                series(
                    "Stats counters - tokens (order dropped)",
                    "c4",
                    _seq_points(diffs["counters-bag"]),
                ),
            ],
            refs=[ref(0.02, "+0.02"), ref(-0.02, "-0.02"), ref(0, "no difference")],
            subtitle="Above 0: the first model has the higher (worse) log-loss. Bands: 95% "
            "t-intervals over five replicates.",
        )
    )
    pred = sq["predictions"]
    g = [v["log_loss_gain"]["mean"] for v in a["paired"]["seq"].values()]
    below = [n for n, v in a["paired"]["seq"].items() if v["log_loss_gain"]["upper"] < 0]
    seq_bag = [v["mean"] for v in diffs["seq-bag"].values()]
    cnt_bag = [v["mean"] for v in diffs["counters-bag"].values()]
    takeaways = [
        f"Paired window by window, the sequence model's gain over the logistic model is "
        f"{_span(g)} nats: never above 0, and clearly below at "
        + ", ".join(f"{int(n):,}" for n in below)
        + f" hands (A1 {_verdict(pred['A1']['holds'])}). Per-size temperatures narrow the "
        "small-sample gap but do not reverse it.",
        f"Reading a hand's decisions in order changes the log-loss by {_span(seq_bag)} nats "
        f"against summing them (A2, under 0.02: {_verdict(pred['A2']['holds'])}), probably "
        "because each token's situation already says where in the hand the decision was made.",
        f"Raw tokens against the hand's stats counters, same network: {_span(cnt_bag)} nats "
        f"(A3 {_verdict(pred['A3']['holds'])}). The raw tokens add little the counters do not "
        "carry, so the gap to the logistic model comes from the learner, not the input.",
    ]
    return section(
        "sequence-gain",
        "Paired against Phase 1's classifier",
        "Every model scored on exactly the windows the logistic model was, so the difference is "
        "measured window by window, not between two noisy averages.",
        sources=[SEQUENCE],
        controls=[
            control(
                "show",
                "Show",
                [
                    ("log_loss_gain", "Gain, one temperature"),
                    ("log_loss_gain_by_size", "Gain, temperature per size"),
                    ("ablations", "Order and input ablations"),
                ],
            )
        ],
        views=views,
        takeaways=takeaways,
        method="Per replicate, the mean over held-out windows of the per-window log-loss "
        "difference; 95% t-intervals over the five replicates. Predictions A1-A3 were "
        "pre-registered (commit c4578ea) after development on two other seeds.",
    )


def sequence_scale(sq: dict[str, Any]) -> dict[str, Any]:
    c = sq["c"]
    models = [(m, name, color) for m, name, color in SEQ_MODELS if m in c["metrics"]]
    base = c["training_players"] - 5 * c["extra_per_archetype"]
    views = {
        "improvement": view(
            line_chart(
                f"What {c['training_players']:,} training bots buy over {base:,}",
                WINDOW_HANDS,
                axis("Log-loss improvement (nats)", fmt="signed"),
                [
                    series(name, color, _seq_points(c["improvement"][m]))
                    for m, name, color in models
                ],
                refs=[ref(0, "no change")],
                subtitle="Log-loss with Phase 1's training bots minus with the extra ones, same "
                "held-out windows; bands: 95% t-intervals over five replicates.",
            )
        ),
        "log_loss": view(
            line_chart(
                f"Log-loss with {c['training_players']:,} training bots",
                WINDOW_HANDS,
                axis("Log-loss (nats)", fmt="num", lo=0),
                [
                    series(
                        name,
                        color,
                        [
                            (
                                int(n),
                                *(
                                    c["metrics"][m][n]["log_loss"][k]
                                    for k in ("mean", "lower", "upper")
                                ),
                            )
                            for n in c["metrics"][m]
                        ],
                    )
                    for m, name, color in models
                ],
                subtitle="Held-out bots, Phase 1's windows; bands: 95% t-intervals.",
            )
        ),
    }
    pred = sq["predictions"]
    imp = c["improvement"]
    gain = c["paired"]

    def span(block: dict[str, Any]) -> str:
        return _span([v["mean"] for v in block.values()])

    above = [n for n, v in gain["bag"].items() if v["log_loss_gain"]["lower"] > 0]
    gbm_above = [n for n, v in gain["gbm"].items() if v["log_loss_gain"]["lower"] > 0]
    takeaways = [
        f"The logistic model barely moves ({span(imp['lr'])} nats; C1, under 0.02: "
        f"{_verdict(pred['C1']['holds'])}): its twelve features, not the number of players, "
        f"limit it. The token network improves by {span(imp['bag'])} (C2 "
        f"{_verdict(pred['C2']['holds'])}), the counters network by {span(imp['counters'])}, "
        f"boosted trees by {span(imp['gbm'])}.",
        "With the extra bots the flexible models overtake it. The token network is ahead by "
        + _span([v["log_loss_gain"]["mean"] for v in gain["bag"].values()])
        + " nats, clearly at "
        + ", ".join(f"{int(n):,}" for n in above)
        + " hands; boosted trees lead from 500 hands (clearly at "
        + ", ".join(f"{int(n):,}" for n in gbm_above)
        + f"). C3 predicted the token network within 0.03 of the logistic model: "
        f"{_verdict(pred['C3']['holds'])}, because the upper ends of its intervals reach "
        f"{max(v['log_loss_gain']['upper'] for v in gain['bag'].values()):+.3f}.",
        "So at Phase 1's scale the hand-written stats are the better model, and the networks' "
        "deficit was a shortage of training players. Bots are cheap to simulate, but the "
        "gain they buy is small, and part B shows a bot-trained model of either kind does not "
        "carry over to real players.",
    ]
    return section(
        "sequence-scale",
        "Is it the method or the amount of training data?",
        f"The same models refit with five times the training bots ({c['training_players']:,} "
        f"instead of {base:,}), scored on the same held-out windows.",
        sources=[SEQUENCE],
        controls=[
            control("show", "Show", [("improvement", "Improvement"), ("log_loss", "Log-loss")])
        ],
        views=views,
        takeaways=takeaways,
        method=f"{c['extra_per_archetype']} extra bots per archetype per replicate, drawn and "
        "simulated like Phase 1's training pool in a league of their own. The order-dropped "
        "network stands in for the sequence model (order adds nothing, part A). Pre-registered "
        "predictions C1-C3.",
    )


def sequence_transfer(sq: dict[str, Any]) -> dict[str, Any]:
    pops = [("humans", "Professionals"), ("control", "Control: bots in their seats")]
    models = [(m, name, color) for m, name, color in SEQ_MODELS if m in ("lr", "seq")]
    views = {}
    for pop, label in pops:
        b = sq["b"][pop]
        views[f"{pop}|consistency"] = view(
            line_chart(
                f"{label}: overconfidence compared with bots of the same label",
                axis("Hands in the window", scale="log", fmt="int"),
                axis("Gap minus the matched bots' gap", fmt="signed"),
                [
                    series(
                        name,
                        color,
                        [
                            (
                                int(n),
                                v["difference"]["value"],
                                v["difference"]["lower"],
                                v["difference"]["upper"],
                            )
                            for n, v in b["consistency"][m].items()
                        ],
                    )
                    for m, name, color in models
                ],
                refs=[ref(0.10, "+0.10"), ref(-0.10, "-0.10"), ref(0, "as for bots")],
                subtitle="Gap = mean top probability - share of windows whose label matches a "
                "later 1,000-hand block of the same player. Bands: 95% intervals over players.",
            )
        )
        views[f"{pop}|confidence"] = view(
            line_chart(
                f"{label}: how sure each model is",
                axis("Hands in the window", scale="log", fmt="int"),
                axis("Mean top probability", fmt="pct", lo=0, hi=1),
                [
                    series(
                        name, color, [(int(n), v[m]["top_prob"]) for n, v in b["labels"].items()]
                    )
                    for m, name, color in models
                ]
                + [
                    series(
                        "The two models agree (share of windows)",
                        "muted",
                        [(int(n), v["players"]) for n, v in b["agreement"].items()],
                        dash=True,
                    )
                ],
                subtitle="Players weigh equally.",
            )
        )
    pred = sq["predictions"]
    hum, ctl = sq["b"]["humans"], sq["b"]["control"]
    early = ("25", "50", "100")
    lr_gap = [hum["consistency"]["lr"][n]["difference"]["value"] for n in early]
    seq_gap = [hum["consistency"]["seq"][n]["difference"]["value"] for n in early]
    agree = [v["players"] for v in hum["agreement"].values()]
    agree_bots = [v["bots"] for v in hum["agreement"].values()]
    takeaways = [
        f"On the professionals the sequence model is even more overconfident than the logistic "
        f"model: its confidence exceeds its agreement with a later 1,000-hand look by "
        f"{_span(seq_gap, '.2f')} more than for matched bots at 25-100 hands, against "
        f"{_span(lr_gap, '.2f')} (B1 {_verdict(pred['B1']['holds'])}). Verdicts: logistic "
        f"*{hum['verdicts']['lr']}*, sequence model *{hum['verdicts']['seq']}*.",
        f"The two bot-trained models give a professional the same archetype in only "
        f"{min(agree):.0%}-{max(agree):.0%} of windows, against {min(agree_bots):.0%}-"
        f"{max(agree_bots):.0%} on held-out bots: on real players they do not even agree with "
        "each other.",
        "On the control (bots in the professionals' seats) neither model is overconfident at "
        f"small samples (B2 {_verdict(pred['B2']['holds'])}). The sequence model's verdict "
        f"there is *{ctl['verdicts']['seq']}* because it is underconfident at 500-1,000 hands, "
        "the false alarm the plan warned of with 13 players and six sizes.",
    ]
    return section(
        "sequence-transfer",
        "Does the sequence model transfer to real players any better?",
        "P3.7 found the bot-trained logistic classifier overconfident on the 13 Pluribus "
        "professionals. The same label-free check for the sequence model, on the same windows.",
        sources=[SEQUENCE],
        controls=[
            control("pop", "Players", pops),
            control(
                "show", "Show", [("consistency", "Consistency check"), ("confidence", "Confidence")]
            ),
        ],
        views=views,
        takeaways=takeaways,
        method="Each replicate's logistic and sequence models on P3.7's windows of each "
        "professional's hands; P3.7's consistency check (each player compared with held-out "
        "bots whose own reference block got the same label; verdict on 95% intervals over "
        "players). Control: the real seating replayed by Phase 1 bots.",
    )


def build(load) -> dict[str, Any]:
    r = load(PUSHFOLD)
    ex = load(EXPLOIT)
    br = load(BRIDGE)
    rc = load(RECOMMEND)
    fc = load(FITCHECK)
    cf = load(CONFIRM)
    im = load(IMPORT)
    pr = load(PRIORS)
    sq = load(SEQUENCE)
    sc = load(SCHEMA)
    ad = load(ADAPTIVE)
    head = overview(r, ex, br, rc, fc, im, pr, sq, sc, ad)
    return {
        "id": "phase3",
        "label": "Phase 3",
        "title": "General framework: hand histories to exploit recommendations",
        "status": "done",
        "summary": head["summary"],
        "tiles": head["tiles"],
        "provenance": [
            provenance(PUSHFOLD, r),
            provenance(EXPLOIT, ex),
            provenance(BRIDGE, br),
            provenance(RECOMMEND, rc),
            provenance(FITCHECK, fc),
            provenance(CONFIRM, cf),
            provenance(IMPORT, im),
            provenance(PRIORS, pr),
            provenance(SEQUENCE, sq),
            provenance(SCHEMA, sc),
            provenance(ADAPTIVE, ad),
        ],
        "sections": [
            charts(r),
            ranges(r),
            sizes(r),
            solvers(r),
            exploit_hands(ex),
            model_fit(ex),
            bridge_check(br),
            bridge_variance(br),
            recommend_rules(rc),
            recommend_calibration(rc),
            recommend_examples(rc),
            fitcheck_flags(fc),
            fitcheck_consequences(fc),
            fitcheck_examples(fc),
            confirm_fallback(cf),
            import_faults(im),
            import_public_data(im),
            import_pluribus(im),
            priors_fit(pr),
            priors_estimation(pr),
            priors_refit_size(pr),
            priors_support(pr),
            priors_classifier(pr),
            priors_action(pr),
            sequence_hands(sq),
            sequence_gain(sq),
            sequence_scale(sq),
            sequence_transfer(sq),
            schema_v2(sc),
            adaptive_gain(ad),
            adaptive_left(ad),
            adaptive_example(ad),
        ],
    }


SCHEMA_STRUCTURES = (
    ("blinds", "Blinds only"),
    ("ante", "Ante for every seat"),
    ("bb_ante", "Big-blind ante"),
    ("straddle", "Straddle (2 bb)"),
    ("double_straddle", "Double straddle (2 and 4 bb)"),
    ("button_straddle", "Straddle on the button"),
    ("min_bet_up", "Minimum bet 2 bb"),
    ("min_bet_down", "Minimum bet 0.5 bb"),
)
SCHEMA_CHECKS = (
    ("full/rules", "Forced posts follow the rules"),
    ("full/replay", "Unmodified PokerKit replays it"),
    ("full/json", "JSON round trip"),
    ("full/phh", "PHH export and import"),
    ("full/loader", "PokerKit's own PHH loader agrees"),
)


def schema_v2(sc: dict[str, Any]) -> dict[str, Any]:
    rt, v1, pub, pred = sc["round_trip"], sc["v1_files"], sc["public"], sc["predictions"]
    by = rt["by_structure"]
    cols = [label for _, label in SCHEMA_CHECKS] + ["Redacted copy (rules, JSON, PHH)"]

    def cells(st: str) -> list[tuple[int, int]]:
        checks = by[st]["checks"]
        out = [(checks[k]["passed"], checks[k]["checked"]) for k, _ in SCHEMA_CHECKS]
        red = [v for k, v in checks.items() if k.startswith("redacted/")]
        out.append((sum(v["passed"] for v in red), sum(v["checked"] for v in red)))
        return out

    grid = [cells(st) for st, _ in SCHEMA_STRUCTURES]
    per = rt["tables"] // len(SCHEMA_STRUCTURES)
    chart = heatmap(
        "Hands at schema v2 tables, checked five ways",
        [f"{label} ({by[st]['hands']:,} hands)" for st, label in SCHEMA_STRUCTURES],
        cols,
        [[p / c if c else 0.0 for p, c in row] for row in grid],
        counts=[[p for p, _ in row] for row in grid],
        fmt="pct",
        vmax=1.0,
        subtitle=f"Share of hands that pass each check. {rt['tables']} random tables, {per} per "
        f"structure, {rt['hands'] // rt['tables']} hands each.",
    )
    odd = sum(v["checks"]["full/loader"]["odd_chip"] for v in by.values())
    missed = rt["checked"] - rt["passed"]
    whole = rt.get("whole_units", {})
    agrees = sum(v for k, v in whole.items() if k.endswith("/agrees"))
    s, i = v1["simulated"], v1["imported"]
    c1, c2, c3 = pred["C1"], pred["C2"], pred["C3"]
    wsop = pub["by_subset"]["wsop"]
    antes = sorted({h["big_blind_ante_bb"] for h in pub["hands"] if h["subset"] == "wsop"})
    variants = ", ".join(f"{v} {n}" for v, n in sorted(pub["rejected_variants"].items(), key=str))
    takeaways = [
        f"{rt['passed']:,} of {rt['checked']:,} checks pass on {rt['hands']:,} hands with "
        "big-blind antes, straddles and minimum bets above and below the big blind, and on their "
        f"redacted copies. The engine, unmodified PokerKit, JSON and PHH agree on every hand; "
        f"A1 {_verdict(pred['A1']['holds'])} because of {missed} hands where PokerKit's own PHH "
        "loader shares a split pot differently.",
        f"Post hoc: all {odd + missed} hands where the loader differs at all are pots split in "
        "cents. PokerKit gives a split pot's whole remainder to the first winner; reading cents "
        "as decimals, the loader splits them exactly. Written in whole units, the loader agrees "
        f"on {agrees} of {sum(whole.values())}.",
        f"Version 1 files read back exactly: the schema v1 code's own output ({s['records']:,} "
        f"simulated hands from P3.6's tables and {i['records']:,} Pluribus hands, written at "
        f"commit {v1['commit'][:7]}) migrates to {s['identical'] + i['identical']:,} of "
        f"{s['records'] + i['records']:,} hands identical to what the current code makes "
        f"(B1 {_verdict(pred['B1']['holds'])}). Phase 1 re-simulated with schema v2 reproduces "
        "its published numbers exactly.",
        f"The WSOP final table's {wsop['imported']} of {wsop['nlhe']} no-limit hold'em hands now "
        f"import (big-blind ante {', '.join(f'{a:g}' for a in antes)} bb; C1 "
        f"{_verdict(c1['holds'])}). PokerKit's own loader agrees on {c2['agree']} of "
        f"{c2['of']} imported public hands (C2 {_verdict(c2['holds'])}) and {c3['same']} of "
        f"{c3['of']} survive export and import (C3 {_verdict(c3['holds'])}). What is still "
        f"rejected is other games: {variants}.",
    ]
    return section(
        "schema-v2",
        "Schema v2: big-blind antes, straddles and minimum bets",
        "Schema v1 held blinds and an ante for every seat. Modern tournaments use a big-blind "
        "ante (one player antes for the whole table) and live games use straddles. Schema v2 "
        "records both, and a minimum bet other than the big blind, and checks every forced "
        "post against the table's rules.",
        sources=[SCHEMA],
        views={"": view(chart)},
        takeaways=takeaways,
        method="Part A: random tables in the style of P3.6, each hand checked by find_problems, "
        "a replay through unmodified PokerKit, JSON and PHH round trips, and PokerKit's own PHH "
        "loader. Part B: the v1 code run in a temporary git worktree. Part C: the PHH dataset "
        f"at commit {sc['dataset']['commit'][:7]}. Pre-registered predictions A1, B1, C1-C3; "
        "the whole-unit re-check is post hoc.",
        post_hoc=True,
    )


AD_KINDS = (
    ("static", "Static bots"),
    ("slow", "Slow adapters"),
    ("fast", "Fast adapters"),
    ("calls_only", "Adapt calls only"),
)
AD_KIND_LABEL = dict(AD_KINDS)
AD_HEROES = (
    ("fit_once", "Fit once, then keep", "c1"),
    ("refit", "Refit on all hands", "c2"),
    ("refit_recent", "Refit on recent hands", "c3"),
    ("best_response", "Bayes best response", "c4"),
    ("clairvoyant", "Clairvoyant (reference)", "c5"),
)
AD_HERO_LABEL = {h: label for h, label, _ in AD_HEROES} | {"equilibrium": "Equilibrium"}
AD_HANDS = axis("Hands into the session", fmt="int")


def _ad_bins(ad: dict[str, Any]) -> list[float]:
    cfg = ad["meta"]["config"]
    edges = range(0, cfg["hands"] + 1, cfg["bin"])
    return [(a + b) / 2 for a, b in zip(list(edges)[:-1], list(edges)[1:], strict=True)]


def _gain(v: dict[str, float]) -> str:
    return f"{v['mean']:+.4f} [{v['lower']:+.4f}, {v['upper']:+.4f}]"


def _paired(a: list[float], b: list[float]) -> dict[str, float]:
    """Mean difference a - b over replicates, with its 95% t-interval."""
    mean, lower, upper = ci(np.subtract(a, b))
    return {"mean": mean, "lower": lower, "upper": upper}


def adaptive_gain(ad: dict[str, Any]) -> dict[str, Any]:
    agg, pred, cfg = ad["aggregate"], ad["predictions"], ad["meta"]["config"]
    xs = _ad_bins(ad)
    views = {}
    for kind, label in AD_KINDS:
        heroes = agg[kind]["heroes"]
        views[kind] = view(
            line_chart(
                f"{label}: gain over equilibrium play",
                AD_HANDS,
                axis("Gain over equilibrium play (bb/hand)", fmt="signed"),
                [
                    series(
                        name,
                        color,
                        [
                            (x, g["mean"], g["lower"], g["upper"])
                            for x, g in zip(xs, heroes[h]["gain_by_bin"], strict=True)
                        ],
                        dash=h == "clairvoyant",
                    )
                    for h, name, color in AD_HEROES
                ],
                refs=[ref(0, "equilibrium play")],
                subtitle=f"Mean over each {cfg['bin']}-hand stretch, against the same bots the "
                "equilibrium hero met. Bands: 95% t-intervals across replicates.",
            )
        )
    st, slow, fast, calls = (agg[k]["heroes"] for k in ("static", "slow", "fast", "calls_only"))
    recent = {
        k: _paired(h["refit_recent"]["gain_by_seed"], h["refit"]["gain_by_seed"])
        for k, h in (("slow", slow), ("fast", fast), ("calls_only", calls))
    }
    takeaways = [
        f"Against static bots, refitting keeps {pred['A1']['ratio']:.0%} of the clairvoyant "
        f"hero's gain ({st['refit']['gain']['mean']:+.3f} of "
        f"{st['clairvoyant']['gain']['mean']:+.3f} bb/hand; A1 {_verdict(pred['A1']['holds'])}): "
        "P3.4's result, replayed hand by hand.",
        f"Against fast adapters the Bayes best response loses to equilibrium play: "
        f"{_gain(fast['best_response']['gain'])} bb/hand, worse than equilibrium against "
        f"{fast['best_response']['loses']['share']:.0%} of the bots (C1 "
        f"{_verdict(pred['C1']['holds'])}). Its strategies are the most exploitable of the "
        f"heroes anyone can play ({fast['best_response']['exploitability']['mean']:.3f} bb/hand "
        f"on average, against {fast['refit']['exploitability']['mean']:.3f} for refitting), and "
        "the bots best-respond to what they see the hero do.",
        f"An exploit fitted once and kept is the costliest mistake against slow adapters: "
        f"{_gain(slow['fit_once']['gain'])} bb/hand (C2 {_verdict(pred['C2']['holds'])}). The "
        "bot drifts toward a best response against it while the hero never looks again.",
        f"The recommender that refits with P3.5's stability check stays close to equilibrium "
        f"play: {_gain(slow['refit']['gain'])} against slow and {_gain(fast['refit']['gain'])} "
        f"against fast adapters (C3 {_verdict(pred['C3']['holds'])}).",
        f"Against bots that adapt only their calls, refitting still gains "
        f"{_gain(calls['refit']['gain'])} bb/hand (C4 {_verdict(pred['C4']['holds'])}): the "
        "leaks an opponent does not fix stay worth exploiting.",
        f"Post hoc: refitting on the last {cfg['window']} hands instead of all of them changes "
        f"the gain by {_gain(recent['slow'])} bb/hand against slow adapters, "
        f"{_gain(recent['fast'])} against fast and {_gain(recent['calls_only'])} against "
        "calls-only adapters (differences paired by replicate). On the shorter window the "
        "stability check fails less often: in the second half of a session it falls back on "
        f"{slow['refit_recent']['fallback_second_half']['share']:.0%} of refits against slow "
        f"adapters, not {slow['refit']['fallback_second_half']['share']:.0%}.",
    ]
    return section(
        "adaptive-gain",
        "Exploiting opponents who adapt",
        "Until now every bot kept its strategy whatever the hero did. Here bots answer: every 25 "
        "hands they move toward a best response to how often the hero pushes and calls. Each "
        "hero plays the same bots from the same cards, and every stretch of a session is scored "
        "by the exact value of the two strategies in force.",
        sources=[ADAPTIVE],
        controls=[control("kind", "Bots", AD_KINDS)],
        views=views,
        takeaways=takeaways,
        method=f"{cfg['seeds']} replicates x {5 * cfg['test_per_archetype']} of P3.2's standard "
        f"test bots as starting points, each as four kinds of bot; sessions of {cfg['hands']:,} "
        f"hands at {cfg['stack']} bb. Refitting heroes decide every {cfg['refit_every']} hands "
        f"(the recent one on the last {cfg['window']}); fit once decides after "
        f"{cfg['observe_hands']}. Pre-registered predictions A1-D1; the comparison of the two "
        "refitting heroes is post hoc.",
        post_hoc=True,
    )


def adaptive_left(ad: dict[str, Any]) -> dict[str, Any]:
    agg, pred, cfg = ad["aggregate"], ad["predictions"], ad["meta"]["config"]
    rows = [label for _, label in AD_KINDS]
    shown = (("equilibrium", "c1"), ("refit", "c2"), ("best_response", "c4"), ("clairvoyant", "c5"))
    values = dots_chart(
        "What is left to win",
        rows,
        axis("Hero's value (bb/hand)", fmt="signed"),
        [
            dot_series(
                AD_HERO_LABEL[h],
                color,
                [
                    (lambda v: [v["mean"], v["lower"], v["upper"]])(agg[k]["heroes"][h]["value"])
                    for k, _ in AD_KINDS
                ],
            )
            for h, color in shown
        ],
        refs=[ref(0, "break even")],
        subtitle="Session value, 95% t-intervals across replicates.",
    )
    xs = _ad_bins(ad)
    colors = ("c1", "c2", "c3", "c4")
    fallback = line_chart(
        "How often the refitting hero falls back to equilibrium play",
        AD_HANDS,
        axis("Refits that fall back", fmt="pct", lo=0, hi=1),
        [
            series(
                label,
                color,
                list(zip(xs, agg[k]["heroes"]["refit"]["fallback_by_bin"], strict=True)),
            )
            for (k, label), color in zip(AD_KINDS, colors, strict=True)
        ],
        refs=[ref(0.05, "5% (calibrated level)")],
        subtitle="Share of refits whose advice is the equilibrium because P3.5's stability check "
        "failed.",
    )
    b1, d1 = pred["B1"]["by_kind"], pred["D1"]["by_kind"]
    st = agg["static"]["heroes"]
    takeaways = [
        f"Adaptation takes the money off the table for everyone: the equilibrium hero wins "
        f"{agg['slow']['heroes']['equilibrium']['value']['mean']:+.3f} and "
        f"{agg['fast']['heroes']['equilibrium']['value']['mean']:+.3f} bb/hand from slow and fast "
        f"adapters, {b1['slow']['mean']:.0%} and {b1['fast']['mean']:.0%} of the "
        f"{st['equilibrium']['value']['mean']:+.3f} it wins from the same bots static (B1 "
        f"{_verdict(pred['B1']['holds'])}). The bots best-respond to whatever the hero does, "
        "which moves them toward equilibrium themselves.",
        f"Even a clairvoyant hero gains only {_gain(pred['B2']['by_kind']['slow'])} and "
        f"{_gain(pred['B2']['by_kind']['fast'])} bb/hand over equilibrium play against them (B2 "
        f"{_verdict(pred['B2']['holds'])}).",
        f"The stability check notices: in the second half of a session the refitting hero falls "
        f"back on {d1['slow']['share']:.0%} of refits against slow and {d1['fast']['share']:.0%} "
        f"against fast adapters, and on {d1['static']['share']:.0%} against static bots (D1 "
        f"{_verdict(pred['D1']['holds'])}).",
    ]
    return section(
        "adaptive-left",
        "What adaptation leaves to win, and whether the checks notice",
        "A bot that best-responds to the hero's frequencies fixes most of its leaks against "
        "every hero, equilibrium play included. And a model that assumes the bot keeps still "
        "should fail its stability check when the bot moves.",
        sources=[ADAPTIVE],
        controls=[control("show", "Show", [("value", "Value"), ("fallback", "Fallbacks")])],
        views={"value": view(values), "fallback": view(fallback)},
        takeaways=takeaways,
        method=f"Checks calibrated per replicate on {cfg['calibration_bots']} bots drawn from the "
        "prior and played by the equilibrium hero, at the 5% level for both checks together "
        "(P3.5); P3.5b's rule falls back on the stability check alone.",
    )


def adaptive_example(ad: dict[str, Any]) -> dict[str, Any]:
    cfg = ad["meta"]["config"]
    block = cfg["period"]
    examples = {(e["archetype"], e["kind"]): e for e in ad["examples"]}
    archs = [
        (a, ARCHETYPE_LABELS.get(a, a))
        for a in dict.fromkeys(e["archetype"] for e in ad["examples"])
    ]
    heroes = (
        ("equilibrium", "c1"),
        ("refit", "c2"),
        ("best_response", "c4"),
        ("clairvoyant", "c5"),
    )
    views = {}
    for arch, alabel in archs:
        for kind, klabel in AD_KINDS:
            e = examples[(arch, kind)]
            xs = [block * (k + 0.5) for k in range(len(e["heroes"]["equilibrium"]["values"]))]
            widths = line_chart(
                f"{alabel}, {klabel.lower()}: the bot's ranges",
                AD_HANDS,
                axis("Width (share of hands)", fmt="pct", lo=0, hi=1),
                [
                    series(
                        f"{'Push' if j == 0 else 'Call'} width vs {AD_HERO_LABEL[h].lower()}",
                        color,
                        [(x, w[j]) for x, w in zip(xs, e["heroes"][h]["widths"], strict=True)],
                        dash=j == 1,
                        marker=False,
                    )
                    for h, color in heroes[:3]
                    for j in (0, 1)
                ],
                subtitle="Solid: push width; dashed: call width. Nash at 10 bb pushes about 58% "
                "and calls about 37% of hands.",
            )
            value = line_chart(
                f"{alabel}, {klabel.lower()}: the hero's value",
                AD_HANDS,
                axis("Exact value (bb/hand)", fmt="signed"),
                [
                    series(
                        AD_HERO_LABEL[h],
                        color,
                        list(zip(xs, e["heroes"][h]["values"], strict=True)),
                        dash=h == "clairvoyant",
                        marker=False,
                    )
                    for h, color in heroes
                ],
                refs=[ref(0, "break even")],
                subtitle=f"Each {block}-hand block scored by the exact value of both strategies.",
            )
            views[f"{arch}|{kind}|widths"] = view(widths)
            views[f"{arch}|{kind}|value"] = view(value)
    # post hoc descriptions of the trajectories, from hand 500 on where it says "late"
    late = 500 // block
    adapters = [e for e in ad["examples"] if e["kind"] != "static"]
    eq_share = np.array(adapters[0]["heroes"]["equilibrium"]["hero_shares"][0])

    def off_eq(w: list[float]) -> float:
        return float(np.abs(np.subtract(w, eq_share)).sum())

    drift = {}
    for kind in ("slow", "fast"):
        bots = [r for recs in ad["per_seed"] for r in recs if r["kind"] == kind]
        start = np.array([off_eq(r["start_widths"]) for r in bots])
        end = np.array([off_eq(r["heroes"]["equilibrium"]["end_widths"]) for r in bots])
        drift[kind] = (start.mean(), end.mean(), float(np.mean(end < start)), len(bots))
    wander = [
        w[0]
        for e in adapters
        if e["kind"] == "fast"
        for h in ("equilibrium", "best_response")
        for w in e["heroes"][h]["widths"][late:]
    ]
    spread = [
        np.std(e["heroes"]["best_response"]["values"][late:])
        / np.std(e["heroes"]["equilibrium"]["values"][late:])
        for e in adapters
    ]
    below = [min(e["heroes"]["best_response"]["values"]) for e in adapters]
    eq_low = min(min(e["heroes"]["equilibrium"]["values"]) for e in ad["examples"])
    nit = examples[("nit", "slow")]["heroes"]["best_response"]
    v, w, s = (np.array(nit[k]) for k in ("values", "widths", "hero_shares"))
    low = int(v.argmin())
    back = low + int(v[low:].argmax())
    takeaways = [
        f"Post hoc: against equilibrium play the adapters drift toward the equilibrium's "
        f"ranges. Over all {drift['slow'][3]} starting bots, the total distance of their push "
        f"and call widths from the equilibrium's falls from {drift['slow'][0]:.2f} to "
        f"{drift['slow'][1]:.2f} for slow and {drift['fast'][1]:.2f} for fast adapters, and "
        f"{drift['slow'][2]:.0%} and {drift['fast'][2]:.0%} of them end closer. Fast adapters "
        f"never settle, though: from hand 500 on, the five fast examples' push widths still "
        f"range from {min(wander):.0%} to {max(wander):.0%}, against equilibrium play and a "
        "best response alike.",
        f"Post hoc: what a best response changes is the hero's value. From hand 500 on it "
        f"varies {np.median(spread):.1f} times as much from block to block as equilibrium "
        f"play's (median over these {len(adapters)} adapters; more in "
        f"{sum(x > 1 for x in spread)}), and in {sum(x < 0 for x in below)} of them it drops "
        f"below zero, to as low as {min(below):+.2f} bb/hand. Equilibrium play cannot lose to "
        f"any bot over both seats, and here its value never falls below {eq_low:+.3f}.",
        f"Post hoc: the slow nit shows a chase. The best response starts pushing every hand at "
        f"hand {block * int(np.argmax(s[:, 0] > 0.995)):,}; the bot answers by calling "
        f"{w[low, 1]:.0%} of hands (from {w[0, 1]:.0%}), and by hand {block * low:,} the "
        f"hero's value has fallen to {v[low]:+.3f} bb/hand. The bot keeps widening its pushes, "
        f"to {w[:, 0].max():.0%}, the hero calls {s[back, 1]:.0%} in turn, and its value "
        f"swings back to {v[back]:+.3f} before both settle near "
        f"{v[1000 // block :].mean():+.3f} from hand 1,000.",
    ]
    return section(
        "adaptive-example",
        "One bot, four ways of reacting",
        "The same starting bot, from the first replicate, as a static bot and as three kinds of "
        "adapter, against different heroes: how its ranges move and what each hero earns.",
        sources=[ADAPTIVE],
        controls=[
            control("bot", "Starts as", archs),
            control("kind", "Bots", AD_KINDS),
            control("show", "Show", [("widths", "Its ranges"), ("value", "Hero's value")]),
        ],
        views=views,
        takeaways=takeaways,
        method="Example bots: the first bot of each archetype in the first replicate (the "
        "same random numbers for every hero). The takeaways describe the trajectories after the "
        "run (post hoc); the drift toward equilibrium is measured on every starting bot, from "
        "its start and end widths.",
        post_hoc=True,
    )

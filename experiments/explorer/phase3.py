"""Explorer sections for Phase 3: from hand histories to exploit recommendations (in progress)."""

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
SOURCES = (PUSHFOLD, EXPLOIT, BRIDGE, RECOMMEND, FITCHECK)
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
            "checks ask whether the opponent plays the way it assumes (P3.5)."
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
        "after seeing these results, so it needs fresh seeds before it becomes the default.",
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


def build(load) -> dict[str, Any]:
    r = load(PUSHFOLD)
    ex = load(EXPLOIT)
    br = load(BRIDGE)
    rc = load(RECOMMEND)
    fc = load(FITCHECK)
    head = overview(r, ex, br, rc, fc)
    return {
        "id": "phase3",
        "label": "Phase 3",
        "title": "General framework: hand histories to exploit recommendations",
        "status": "active",
        "summary": head["summary"],
        "tiles": head["tiles"],
        "provenance": [
            provenance(PUSHFOLD, r),
            provenance(EXPLOIT, ex),
            provenance(BRIDGE, br),
            provenance(RECOMMEND, rc),
            provenance(FITCHECK, fc),
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
        ],
    }

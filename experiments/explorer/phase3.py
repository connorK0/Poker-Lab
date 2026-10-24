"""Explorer sections for Phase 3: from hand histories to exploit recommendations (in progress)."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.explorer.phase2 import ARCH_OPTIONS, RESPONSE_COLOR, RESPONSE_LABEL
from experiments.explorer.spec import (
    axis,
    ci,
    control,
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
SOURCES = (PUSHFOLD, EXPLOIT)
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


def overview(r: dict[str, Any], ex: dict[str, Any]) -> dict[str, Any]:
    cmp = r["reference_comparison"]
    p, c = cmp["pusher"]["summary"], cmp["caller"]["summary"]
    nash = r["nash"]
    k10 = int(np.argmin(np.abs(np.array(nash["stacks"]) - 10)))
    tight = r["heuristics"]["tight"]["exploitability"]
    std = ex["aggregate"]["standard"]
    oracle = std["oracle"]["br"]["gain"]["mean"]
    br30 = std["by_size"]["30"]["likelihood"]["br"]["gain"]["mean"]
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
        ],
        "summary": (
            "Phase 3 turns hand histories into ranked exploit recommendations. It starts with a "
            "bridge to hold'em: heads-up push/fold no-limit hold'em (blinds 0.5/1, no ante), "
            "solved exactly with the Phase 2 tools on the 169 starting-hand classes with exact "
            "card-removal weights and checked against a published Nash solution (P3.1). Then "
            "push/fold bots are modeled from what a player sees and exploited (P3.2)."
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


def build(load) -> dict[str, Any]:
    r = load(PUSHFOLD)
    ex = load(EXPLOIT)
    head = overview(r, ex)
    return {
        "id": "phase3",
        "label": "Phase 3",
        "title": "General framework: hand histories to exploit recommendations",
        "status": "active",
        "summary": head["summary"],
        "tiles": head["tiles"],
        "provenance": [provenance(PUSHFOLD, r), provenance(EXPLOIT, ex)],
        "sections": [
            charts(r),
            ranges(r),
            sizes(r),
            solvers(r),
            exploit_hands(ex),
            model_fit(ex),
        ],
    }

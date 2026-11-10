"""Explorer sections for Phase 2: equilibrium solvers and exploitation in Kuhn and Leduc."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr

from experiments.explorer.spec import (
    ARCHETYPE_COLORS,
    ARCHETYPE_LABELS,
    ARCHETYPE_SHAPES,
    ARCHETYPES,
    axis,
    ci,
    control,
    dot_series,
    dots_chart,
    line_chart,
    provenance,
    ref,
    scatter_chart,
    scatter_group,
    section,
    series,
    sig,
    table_group,
    tile,
    view,
)

CFR = "experiments/results/phase2/cfr/cfr_convergence.json"
DEEP = "experiments/results/phase2/deep_cfr/deep_cfr.json"
EXPLOIT = "experiments/results/phase2/exploit/exploit.json"
AIVAT = "experiments/results/phase2/aivat/aivat.json"
AIVAT_POSTHOC = "experiments/results/phase2/aivat/aivat_posthoc.json"
DUPLICATE = "experiments/results/phase2/duplicate/duplicate.json"
DUPLICATE_POSTHOC = "experiments/results/phase2/duplicate/duplicate_posthoc.json"
RANDOMNESS = "experiments/results/phase2/randomness/randomness.json"
MISSPEC = "experiments/results/phase2/misspec/misspec.json"
SOURCES = (
    CFR,
    DEEP,
    EXPLOIT,
    AIVAT,
    AIVAT_POSTHOC,
    DUPLICATE,
    DUPLICATE_POSTHOC,
    RANDOMNESS,
    MISSPEC,
)

GAMES = (("kuhn", "Kuhn poker"), ("leduc", "Leduc hold'em"))
VARIANTS = (
    ("cfr", "CFR (vanilla)", "c1"),
    ("cfr+", "CFR+", "c2"),
    ("lcfr", "Linear CFR", "c3"),
    ("dcfr", "Discounted CFR", "c4"),
)
MCCFR = (("es", "External-sampling MCCFR", "c5"), ("es_linear", "ES-MCCFR, linear weights", "c6"))
EXPL = axis("Exploitability (chips/hand)", scale="log", fmt="sci")

RESPONSES = ("br", "rnr_0.9", "rnr_0.75", "rnr_0.5", "rnr_0.25", "dbr")
RESPONSE_LABEL = {
    "br": "Best response",
    "rnr_0.9": "Restricted Nash, p = 0.9",
    "rnr_0.75": "Restricted Nash, p = 0.75",
    "rnr_0.5": "Restricted Nash, p = 0.5",
    "rnr_0.25": "Restricted Nash, p = 0.25",
    "dbr": "Data-biased response",
}
# Same slots as the Phase 2 PNG figures (BR blue, p = 0.5 orange, p = 0.9 aqua, DBR yellow).
RESPONSE_COLOR = {
    "br": "c1",
    "rnr_0.9": "c3",
    "rnr_0.75": "c7",
    "rnr_0.5": "c2",
    "rnr_0.25": "c5",
    "dbr": "c4",
}
FRONTIER = ("rnr_0.25", "rnr_0.5", "rnr_0.75", "rnr_0.9", "br")
LEDUC_PARAM_LABEL = {
    "bluff": "Bluff: bet/raise with a weak hand",
    "medium_aggression": "Bet/raise with a medium hand",
    "value": "Bet/raise with a strong hand",
    "fold_weak": "Fold a weak hand to a bet",
    "fold_medium": "Fold a medium hand to a bet",
}
LEDUC_PARAMS = tuple(LEDUC_PARAM_LABEL)
HANDS = axis("Hands observed (bot cards seen only at showdown)", scale="log", fmt="int")
ARCH_OPTIONS = [("all", "All bots"), *[(a, ARCHETYPE_LABELS[a]) for a in ARCHETYPES]]


def overview(
    cfr: dict[str, Any],
    deep: dict[str, Any],
    ex: dict[str, Any],
    av: dict[str, Any],
    dup: dict[str, Any],
    rnd: dict[str, Any],
    mis: dict[str, Any],
) -> dict[str, Any]:
    leduc = cfr["full_width"]["leduc"]
    cfrp = leduc["cfr+"][-1]
    vanilla = leduc["cfr"][-1]
    agg = deep["aggregate"]
    net = agg["policy_network"][-1]["geo_mean"]
    tab = agg["mccfr"][-1]["geo_mean"]
    by = ex["aggregate"]["by_size"]
    pop = ex["aggregate"]["population_model"]["br"]["gain"]["mean"]
    return {
        "tiles": [
            tile(
                "Leduc exploitability, CFR+ after 10,000 iterations",
                f"{cfrp['exploitability']:.1e}",
                f"vanilla CFR: {vanilla['exploitability']:.1e} chips/hand",
            ),
            tile(
                "Deep CFR vs a table at 200k traversals",
                f"{net / tab:.1f}× worse",
                f"{net:.3f} vs {tab:.3f} chips/hand (a negative result)",
            ),
            tile(
                "Best-response gain after 10 hands",
                f"+{by['10']['br']['gain']['mean']:.3f}",
                "chips/hand above equilibrium play",
            ),
            tile(
                "...of which the population prior alone",
                f"+{pop:.3f}",
                "a best response with zero hands observed (post hoc control)",
            ),
            tile(
                "AIVAT: hands needed for a Leduc win rate",
                f"{av['population']['equilibrium']['equilibrium']['aivat']['ratio']['median']:.0f}"
                "× fewer",
                "equilibrium hero, every hand's cards shown; "
                f"{av['population']['equilibrium']['equilibrium']['hero_view']['ratio']['median']:.2f}"
                "× with folded cards hidden",
            ),
            tile(
                "Duplicate dealing at six seats, random mix of bots",
                f"{dup['aggregates']['by_kind']['mixed_6']['ratio']['median']:.2f}× fewer",
                "hands than independent dealing; heads-up push/fold: "
                f"{dup['bridge']['all']['ratio_median']:.2f}×",
            ),
            tile(
                "Six copies of one bot: the shortfall from mixed actions",
                f"{rnd['aggregates']['decomposition']['shares']['shapley_mixed']['mean']:.0%}",
                "tables paired by decision; "
                f"{rnd['aggregates']['decomposition_history']['shares']['shapley_mixed']['mean']:.0%}"
                " paired by exact history",
            ),
            tile(
                "Best response against bots outside the model's family",
                "{:.0%}-{:.0%}".format(*_off_family_capture(mis)),
                "of the clairvoyant's gain after 3,000 hands; bots that adapt freely make it lose "
                f"{-mis['sessions']['tabular_fast']['heroes']['refit_br']['gain']['mean']:.2f} "
                "chips/hand to equilibrium play",
            ),
        ],
        "summary": (
            "Kuhn poker (12 information sets) and Leduc hold'em (936) are small enough to "
            "evaluate every strategy exactly: exploitability is a full best response, never a "
            "sampled estimate. Exploitability is NashConv / 2 in chips per hand; 0 is an exact "
            "equilibrium, uniform random play scores 0.46 (Kuhn) and 2.37 (Leduc)."
        ),
    }


def cfr_convergence(cfr: dict[str, Any]) -> dict[str, Any]:
    xs = {
        "iteration": axis("Iterations", scale="log", fmt="int"),
        "nodes_touched": axis("Tree nodes touched (work)", scale="log", fmt="int"),
        "seconds": axis("Wall-clock seconds (recording machine)", scale="log", fmt="num"),
    }
    views = {}
    for game, gname in GAMES:
        fw = cfr["full_width"][game]
        lp = cfr["lp"][game]
        for xkey, xax in xs.items():
            lines = [
                series(label, color, [(p[xkey], p["exploitability"]) for p in fw[v]], marker=False)
                for v, label, color in VARIANTS
            ]
            if xkey == "iteration":
                at10 = next(p["exploitability"] for p in fw["cfr"] if p["iteration"] >= 10)
                tmax = fw["cfr"][-1]["iteration"]
                lines.append(
                    series(
                        "1/√T rate",
                        "muted",
                        [(10, at10), (tmax, at10 * (10 / tmax) ** 0.5)],
                        dash=True,
                        marker=False,
                    )
                )
                lines.append(
                    series(
                        "1/T rate",
                        "muted",
                        [(10, at10), (tmax, at10 * 10 / tmax)],
                        dash=True,
                        marker=False,
                    )
                )
            views[f"{game}|{xkey}"] = view(
                line_chart(
                    f"{gname}: exploitability of the average strategy",
                    xax,
                    EXPL,
                    lines,
                    subtitle=f"Exact best-response evaluation. LP game value {lp['value']:.7f} "
                    f"(solved in {lp['seconds']:.2f} s). Full-width runs are deterministic.",
                )
            )
    leduc = cfr["full_width"]["leduc"]
    final = {v: leduc[v][-1]["exploitability"] for v, _, _ in VARIANTS}
    return section(
        "cfr",
        "CFR variants: how fast they approach equilibrium",
        "Counterfactual regret minimization and three modern variants, run for 10,000 "
        "iterations. Switch the x-axis to compare per iteration, per unit of work, or per second.",
        sources=[CFR],
        controls=[
            control("game", "Game", GAMES),
            control(
                "x",
                "X-axis",
                [
                    ("iteration", "Iterations"),
                    ("nodes_touched", "Nodes touched"),
                    ("seconds", "Seconds"),
                ],
            ),
        ],
        views=views,
        defaults={"game": "leduc"},
        takeaways=[
            "Vanilla CFR tracks its 1/√T rate; CFR+ and discounted CFR run faster than 1/T.",
            f"On Leduc after 10,000 iterations: CFR {final['cfr']:.1e}, CFR+ {final['cfr+']:.1e}, "
            f"linear CFR {final['lcfr']:.1e}, discounted CFR {final['dcfr']:.1e} chips/hand.",
            "Linear CFR stalls on Leduc; the exponent ablation (next sections) traces it to "
            "keeping negative regrets.",
        ],
    )


def sampling(cfr: dict[str, Any]) -> dict[str, Any]:
    xs = {
        "nodes_touched": axis("Tree nodes touched (work)", scale="log", fmt="int"),
        "seconds": axis("Wall-clock seconds (recording machine)", scale="log", fmt="num"),
    }
    views = {}
    for game, gname in GAMES:
        fw = cfr["full_width"][game]["cfr+"]
        for xkey, xax in xs.items():
            base = series(
                "CFR+ (full width)",
                "c2",
                [(p[xkey], p["exploitability"]) for p in fw],
                marker=False,
            )
            mean_lines = [base]
            seed_lines = [base]
            for v, label, color in MCCFR:
                m = cfr["mccfr"][game][v]
                mean_lines.append(
                    series(
                        f"{label} (mean of {len(m['seeds'])} seeds)",
                        color,
                        [
                            (p[f"{xkey}_mean"], p["geo_mean"], p["lower"], p["upper"])
                            for p in m["points"]
                        ],
                        marker=False,
                    )
                )
                runs = [r for r in cfr["mccfr_runs"] if r["game"] == game and r["variant"] == v]
                for i, r in enumerate(runs, start=1):
                    s = series(
                        label,
                        color,
                        [(p[xkey], p["exploitability"]) for p in r["curve"]],
                        marker=False,
                        note=f"seed {i} ({r['seed']})",
                    )
                    s["group"] = label
                    seed_lines.append(s)
            sub = "Geometric mean over seeds; bands are 95% t-intervals on the log scale."
            views[f"{game}|{xkey}|mean"] = view(
                line_chart(
                    f"{gname}: sampled vs full-width CFR", xax, EXPL, mean_lines, subtitle=sub
                )
            )
            views[f"{game}|{xkey}|seeds"] = view(
                line_chart(
                    f"{gname}: every MCCFR seed",
                    xax,
                    EXPL,
                    seed_lines,
                    subtitle="One line per seed: the spread is the run-to-run variance of "
                    "sampling.",
                )
            )
    final = cfr["mccfr"]["leduc"]["es"]["points"][-1]
    return section(
        "mccfr",
        "Sampling vs full-width updates at equal work",
        "External-sampling MCCFR updates one sampled trajectory per opponent action instead of "
        "the whole tree. Compare at equal work (nodes touched) or time.",
        sources=[CFR],
        controls=[
            control("game", "Game", GAMES),
            control("x", "X-axis", [("nodes_touched", "Nodes touched"), ("seconds", "Seconds")]),
            control("show", "Show", [("mean", "Mean and 95% band"), ("seeds", "Individual seeds")]),
        ],
        views=views,
        defaults={"game": "leduc"},
        takeaways=[
            f"On Leduc, MCCFR reaches {final['geo_mean']:.3f} [{final['lower']:.3f}, "
            f"{final['upper']:.3f}] after {final['nodes_touched_mean'] / 1e6:.1f}M nodes, about "
            "80× above full-width CFR+ at the same work.",
            "Sampling only helps early. In games this small there is no reason to sample; MCCFR "
            "is here as the baseline Deep CFR is compared with.",
            "Linear weighting does not help MCCFR.",
        ],
    )


def ablation(cfr: dict[str, Any]) -> dict[str, Any]:
    rows, by_iter = [], {}
    named = {(1.0, 1.0, 1.0): " (linear CFR)", (1.5, 0.0, 2.0): " (DCFR default)"}
    for run in cfr["ablation"]:
        key = (run["alpha"], run["beta"], run["gamma"])
        rows.append(f"α={key[0]:g}, β={key[1]:g}, γ={key[2]:g}{named.get(key, '')}")
        for p in run["curve"]:
            by_iter.setdefault(p["iteration"], []).append(p["exploitability"])
    iters = sorted(by_iter)
    chart = dots_chart(
        "Discounted-CFR exponents on Leduc",
        rows,
        axis("Exploitability (chips/hand, log scale)", scale="log", fmt="sci"),
        [
            dot_series(f"After {it:,} iterations", f"c{i}", [(v,) for v in by_iter[it]])
            for i, it in enumerate(iters, start=1)
        ],
        subtitle="α discounts positive regrets, β negative regrets, γ sets the averaging "
        "weight t^γ.",
    )
    return section(
        "ablation",
        "Why linear CFR stalls: the exponent ablation",
        "Discounted CFR has three exponents. This grid varies each one to find which matters.",
        sources=[CFR],
        views={"": view(chart)},
        takeaways=[
            "β (negative regrets) is the culprit: forgetting negative regret immediately (β = 0, "
            "as CFR+ does) is worth 4-10× at α = 1 and 41-109× at α = 1.5.",
            "With β = 1 no setting of α or γ gets below about 1.8e-3.",
        ],
        post_hoc=True,
    )


def _deep_seconds(runs: list[dict[str, Any]], key: str = "seconds") -> dict[int, float]:
    out: dict[int, list[float]] = {}
    for r in runs:
        for p in r["curve"]:
            out.setdefault(p["iteration"], []).append(p[key])
    return {k: float(np.mean(v)) for k, v in out.items()}


def deep_cfr(deep: dict[str, Any]) -> dict[str, Any]:
    agg = deep["aggregate"]
    t_net = _deep_seconds(deep["network_runs"])
    t_tab = _deep_seconds(deep["perfect_regressor_runs"])
    t_mc = _deep_seconds(deep["mccfr_runs"])
    policy_extra = {}
    for r in deep["network_runs"]:
        for p in r["curve"]:
            if "policy_seconds" in p:
                policy_extra.setdefault(p["iteration"], []).append(p["policy_seconds"])
    t_pol = {k: t_net[k] + float(np.mean(v)) for k, v in policy_extra.items()}
    spec = (
        ("mccfr", "Tabular ES-MCCFR, linear weights", "c6", t_mc, {}),
        (
            "perfect_regressor",
            "Deep CFR with a perfect regressor (ablation)",
            "c7",
            t_tab,
            {"dash": True},
        ),
        ("exact_average", "Deep CFR: exact average strategy (SD-CFR)", "c1", t_net, {}),
        ("policy_network", "Deep CFR: policy network (its output)", "c2", t_pol, {}),
        ("memory_average", "Deep CFR: strategy-memory average", "c3", t_net, {"hidden": True}),
        ("current", "Deep CFR: current strategy (not averaged)", "c4", t_net, {"hidden": True}),
    )
    xs = {
        "traversals": axis("Traversals (both players)", scale="log", fmt="int"),
        "seconds": axis("Wall-clock training seconds (single thread)", scale="log", fmt="num"),
    }
    views = {}
    for xkey, xax in xs.items():
        mean_lines = []
        for key, label, color, tmap, kw in spec:
            pts = []
            for p in agg[key]:
                x = p["traversals"] if xkey == "traversals" else tmap[p["iteration"]]
                pts.append((x, p["geo_mean"], p["lower"], p["upper"]))
            mean_lines.append(series(label, color, pts, **kw))
        seed_lines = []
        for runs, key, label, color, kw in (
            (deep["mccfr_runs"], "exploitability", spec[0][1], "c6", {}),
            (deep["perfect_regressor_runs"], "exact_average", spec[1][1], "c7", {"dash": True}),
            (deep["network_runs"], "exact_average", spec[2][1], "c1", {}),
        ):
            for i, r in enumerate(runs, start=1):
                s = series(
                    label,
                    color,
                    [(p[xkey], p[key]) for p in r["curve"]],
                    marker=False,
                    note=f"seed {i} ({r['seed']})",
                    **kw,
                )
                s["group"] = label
                seed_lines.append(s)
        sub = "Geometric mean over 5 seeds; bands are 95% t-intervals on the log scale."
        views[f"{xkey}|mean"] = view(
            line_chart(
                "Deep CFR vs tabular MCCFR on Leduc",
                xax,
                axis("Exploitability (chips/hand)", scale="log", fmt="num"),
                mean_lines,
                subtitle=sub,
                refs=[ref(2.37, "Uniform random play")],
            )
        )
        views[f"{xkey}|seeds"] = view(
            line_chart(
                "Every seed",
                xax,
                axis("Exploitability (chips/hand)", scale="log", fmt="num"),
                seed_lines,
                subtitle="Exact average strategy per seed (policy-network points need the "
                "mean view).",
            )
        )
    net, perf, tab = (
        agg[k][-1]["geo_mean"] for k in ("policy_network", "perfect_regressor", "mccfr")
    )
    cfg = deep["network_runs"][0]["describe"]
    return section(
        "deep-cfr",
        "Deep CFR vs a table: what neural networks cost here",
        "Deep CFR replaces the regret table with neural networks retrained every iteration. "
        "Switch the x-axis to training time to see the cost of training the networks.",
        sources=[DEEP],
        controls=[
            control("x", "X-axis", [("traversals", "Traversals"), ("seconds", "Training time")]),
            control("show", "Show", [("mean", "Mean and 95% band"), ("seeds", "Individual seeds")]),
        ],
        views=views,
        takeaways=[
            f"At 200k traversals Deep CFR ends {net / tab:.1f}× more exploitable than the table "
            f"({net:.3f} vs {tab:.3f}), and its curve flattens after about 50 iterations.",
            f"The perfect-regressor ablation splits the gap: sampling and batching cost "
            f"×{perf / tab:.2f}, function approximation ×{net / perf:.2f}.",
            f"Per wall-clock second the gap is far larger: each Deep CFR iteration retrains "
            f"{cfg['parameters_per_network']:,}-parameter networks from scratch.",
            "Expected for a game with 288 distinct situations, where a table is exact. Deep CFR's "
            "case is games too large for tables, which this experiment does not test.",
        ],
        method=(
            f"Advantage networks: MLP {cfg['config']['hidden']}, {cfg['features']} input features, "
            f"{cfg['config']['advantage_steps']} Adam steps per iteration; "
            f"{cfg['config']['traversals']:,} traversals per player per iteration. "
            "Hyperparameters were tuned on a separate development seed."
        ),
    )


# ---------------------------------------------------------------- exploitation (per-bot data)


def _bots(ex: dict[str, Any]) -> list[list[dict[str, Any]]]:
    return ex["per_seed"]


def _subset(bots: list[dict[str, Any]], arch: str) -> list[dict[str, Any]]:
    return bots if arch == "all" else [b for b in bots if b["archetype"] == arch]


def _scores(b: dict[str, Any], model: str, n: str) -> dict[str, Any] | None:
    if n == "oracle":
        return b["oracle"]
    if n == "0":
        return b["population_model"]
    if model == "card_blind":
        return b["card_blind"].get(n)
    return b["sizes"][n]["scores"]


def _metric(b: dict[str, Any], sc: dict[str, Any], resp: str, metric: str) -> float | None:
    if sc is None or resp not in sc:
        return None
    gain = sc[resp]["gain"]
    if metric == "gain":
        return gain
    if metric == "winnings":
        return b["eq_value"] + gain
    if metric == "exploitability":
        return sc[resp]["exploitability"]
    if metric == "loses":
        return float(gain < 0)
    raise ValueError(metric)


def _seed_ci(per_seed, arch: str, fn) -> tuple[float | None, float | None, float | None]:
    vals = []
    for bots in per_seed:
        xs = [v for v in (fn(b) for b in _subset(bots, arch)) if v is not None]
        if xs:
            vals.append(float(np.mean(xs)))
    if not vals:
        return None, None, None
    return ci(vals)


def _curve(per_seed, sizes, arch: str, model: str, resp: str, metric: str) -> list[tuple]:
    """``(hands, mean, lower, upper)`` of a per-bot metric: mean over bots within each seed,
    then a t-interval across seeds. ``share`` is the ratio of mean gain to mean oracle gain."""
    pts = []
    for n in sizes:
        if metric == "share":
            vals = []
            for bots in per_seed:
                sub = _subset(bots, arch)
                gain = np.mean([_metric(b, _scores(b, model, n), resp, "gain") for b in sub])
                vals.append(float(gain / np.mean([b["oracle"]["br"]["gain"] for b in sub])))
            pts.append((int(n), *ci(vals)))
        else:

            def value(b, n=n):
                return _metric(b, _scores(b, model, n), resp, metric)

            mean, lo, hi = _seed_ci(per_seed, arch, value)
            if metric == "loses" and mean is not None:  # a share: clip the interval to [0, 1]
                lo, hi = (None, None) if lo is None else (max(lo, 0.0), min(hi, 1.0))
            pts.append((int(n), mean, lo, hi))
    return pts


def exploit_by_hands(ex: dict[str, Any]) -> dict[str, Any]:
    per_seed = _bots(ex)
    sizes = [str(n) for n in ex["meta"]["config"]["sizes"]]
    metrics = {
        "gain": ("Gain over equilibrium play", axis("Chips/hand above equilibrium", fmt="signed")),
        "winnings": ("Winnings against the bot", axis("Chips/hand won from the bot", fmt="signed")),
        "exploitability": (
            "Risk: exploitability of the hero's response",
            axis("Exploitability (chips/hand)", lo=0),
        ),
        "share": (
            "Share of the oracle's best-response gain",
            axis("Share of oracle gain", fmt="pct"),
        ),
    }
    views = {}
    for metric, (title, yax) in metrics.items():
        for model in ("fitted", "card_blind"):
            responses = [r for r in RESPONSES if not (model == "card_blind" and r == "dbr")]
            dash = model == "card_blind"
            model_txt = (
                "Model fit by likelihood from showdown-only observations"
                if model == "fitted"
                else "Card-blind model (ignores hand strength; misspecified on purpose; post hoc)"
            )
            for arch, alabel in ARCH_OPTIONS:
                lines = [
                    series(
                        RESPONSE_LABEL[r],
                        RESPONSE_COLOR[r],
                        _curve(per_seed, sizes, arch, model, r, metric),
                        dash=dash,
                    )
                    for r in responses
                ]
                refs = []
                if metric in ("gain", "share"):
                    oracle = _seed_ci(per_seed, arch, lambda b: b["oracle"]["br"]["gain"])[0]
                    pop = _seed_ci(per_seed, arch, lambda b: b["population_model"]["br"]["gain"])[0]
                    den = oracle if metric == "share" else 1.0
                    refs = [
                        ref(oracle / den, "Oracle best response (true parameters)"),
                        ref(pop / den, "Best response to the population model (0 hands)"),
                        ref(0.0, "Equilibrium play"),
                    ]
                elif metric == "winnings":
                    eq = _seed_ci(per_seed, arch, lambda b: b["eq_value"])[0]
                    refs = [ref(eq, "Equilibrium strategy's winnings")]
                loses = line_chart(
                    "Bots against which the response does worse than equilibrium play",
                    HANDS,
                    axis("Share of bots", fmt="pct", lo=0),
                    [
                        series(
                            RESPONSE_LABEL[r],
                            RESPONSE_COLOR[r],
                            _curve(per_seed, sizes, arch, model, r, "loses"),
                            dash=dash,
                        )
                        for r in responses
                    ],
                    subtitle="Mean over bots per seed; 95% t-intervals across seeds.",
                )
                which = alabel.lower() if arch == "all" else alabel + " bots"
                views[f"{metric}|{model}|{arch}"] = view(
                    line_chart(
                        f"{title} ({which})",
                        HANDS,
                        yax,
                        lines,
                        subtitle=f"{model_txt}. 5 seeds × 30 held-out bots; exact values; "
                        "bands are 95% t-intervals across seeds.",
                        refs=refs,
                    ),
                    loses,
                )
    obs = ex["aggregate"]["observation_value"]
    share = ex["aggregate"]["card_blind_br_loses_share"]
    return section(
        "exploit",
        "Exploiting modeled bots: gain vs hands observed",
        "The hero watches a Leduc bot for n hands (seeing its card only at showdown), fits a "
        "model, and responds. Choose the metric, the model, and which bots to average over.",
        sources=[EXPLOIT],
        controls=[
            control(
                "metric",
                "Metric",
                [
                    ("gain", "Gain vs equilibrium"),
                    ("winnings", "Winnings"),
                    ("exploitability", "Risk (exploitability)"),
                    ("share", "Share of oracle gain"),
                ],
            ),
            control(
                "model",
                "Opponent model",
                [("fitted", "Fitted model"), ("card_blind", "Card-blind model")],
            ),
            control("arch", "Bots", ARCH_OPTIONS),
        ],
        views=views,
        takeaways=[
            f"Most of the gain needs no observation of this bot: a best response to the population "
            f"model already gains "
            f"+{ex['aggregate']['population_model']['br']['gain']['mean']:.3f}. "
            f"Watching adds +{obs['10']['mean']:.3f} after 10 hands and "
            f"+{obs['3000']['mean']:.3f} after 3,000.",
            "Restricted Nash with p = 0.9 keeps 96-98% of the best response's gain at about 45% "
            "of its exploitability.",
            "The data-biased response trusts the model only where it has data, so it starts at "
            "equilibrium and catches up slowly.",
            f"With the misspecified card-blind model, the best response loses to equilibrium play "
            f"against {min(v['mean'] for v in share.values()):.0%}-"
            f"{max(v['mean'] for v in share.values()):.0%} of bots.",
        ],
        method=(
            "Values are exact expectations (no sampling noise per bot): the response is evaluated "
            "against the bot's true strategy in both seats. Gain = response's value minus the "
            "equilibrium strategy's value against the same bot."
        ),
    )


def frontier(ex: dict[str, Any]) -> dict[str, Any]:
    per_seed = _bots(ex)
    sizes = [str(n) for n in ex["meta"]["config"]["sizes"]]
    options = [
        ("0", "0 (population model)"),
        *[(n, f"{int(n):,}") for n in sizes],
        ("oracle", "Oracle"),
    ]

    def curve(model: str, n: str, arch: str) -> list[list[Any]]:
        pts: list[list[Any]] = [[0.0, 0.0, "Equilibrium"]]
        for resp in FRONTIER:
            x = _seed_ci(
                per_seed,
                arch,
                lambda b, r=resp: _metric(b, _scores(b, model, n), r, "exploitability"),
            )[0]
            y = _seed_ci(
                per_seed, arch, lambda b, r=resp: _metric(b, _scores(b, model, n), r, "gain")
            )[0]
            if x is not None:
                pts.append([x, y, RESPONSE_LABEL[resp]])
        return pts

    views = {}
    for n, nlabel in options:
        for arch, alabel in ARCH_OPTIONS:
            groups = [
                scatter_group(
                    "Oracle (true parameters)",
                    "muted",
                    curve("fitted", "oracle", arch),
                    connect=True,
                    dash=True,
                )
            ]
            if n != "oracle":
                groups.insert(
                    0,
                    scatter_group(
                        "Fitted model" if n != "0" else "Population model",
                        "c1",
                        curve("fitted", n, arch),
                        connect=True,
                    ),
                )
            if n not in ("0", "oracle"):
                groups.insert(
                    1,
                    scatter_group(
                        "Card-blind model (post hoc)",
                        "c5",
                        curve("card_blind", n, arch),
                        connect=True,
                        dash=True,
                    ),
                )
                dbr = [
                    _seed_ci(
                        per_seed,
                        arch,
                        lambda b, k=k, n=n: _metric(b, _scores(b, "fitted", n), "dbr", k),
                    )[0]
                    for k in ("exploitability", "gain")
                ]
                groups.append(
                    scatter_group("Data-biased response", "c4", [[*dbr, "Data-biased response"]])
                )
            views[f"{n}|{arch}"] = view(
                scatter_chart(
                    f"Gain vs risk after {nlabel} hands "
                    f"({alabel.lower() if arch == 'all' else alabel})",
                    axis("Exploitability of the response (chips/hand)", lo=0),
                    axis("Gain over equilibrium vs the true bot (chips/hand)", fmt="signed"),
                    groups,
                    subtitle="Each curve runs from equilibrium (0, 0) through restricted Nash "
                    "responses p = 0.25, 0.5, 0.75, 0.9 to the best response. Hover for labels.",
                    refs=[ref(0.0, "Equilibrium gain")],
                )
            )
    return section(
        "frontier",
        "The gain-risk frontier",
        "A restricted Nash response with weight p trades exploitation for safety. Up and to the "
        "left is better: more gain for less exploitability.",
        sources=[EXPLOIT],
        controls=[
            control("hands", "Hands observed", options),
            control("arch", "Bots", ARCH_OPTIONS),
        ],
        views=views,
        defaults={"hands": "300"},
        takeaways=[
            "The frontier is steep then flat: p = 0.5 already captures roughly two-thirds of the "
            "gain at under a tenth of the best response's exploitability.",
            "With the misspecified card-blind model the curve bends down: the most exploitative "
            "responses gain less than moderate ones.",
        ],
    )


def per_bot(ex: dict[str, Any]) -> dict[str, Any]:
    per_seed = _bots(ex)
    sizes = [str(n) for n in ex["meta"]["config"]["sizes"]]
    bots = [b for seed in per_seed for b in seed]
    table: dict[str, list[Any]] = {
        "archetype": [b["archetype"] for b in bots],
        "label": [
            f"{ARCHETYPE_LABELS[b['archetype']]} bot {b['bot_id']} (seed {s + 1})"
            for s, seed in enumerate(per_seed)
            for b in seed
        ],
        "bot_expl": [sig(b["bot_exploitability"], 3) for b in bots],
    }
    for n in sizes:
        for resp in RESPONSES:
            table[f"f_{resp}_{n}"] = [sig(b["sizes"][n]["scores"][resp]["gain"], 3) for b in bots]
            if resp != "dbr":
                table[f"c_{resp}_{n}"] = [sig(b["card_blind"][n][resp]["gain"], 3) for b in bots]
    views = {}
    xax = axis("How exploitable the bot is (chips/hand)", lo=0)
    yax = axis("Gain over equilibrium vs this bot (chips/hand)", fmt="signed")
    for resp in RESPONSES:
        for n in sizes:
            charts = []
            for prefix, mlabel in (("f", "fitted model"), ("c", "card-blind model (post hoc)")):
                col = f"{prefix}_{resp}_{n}"
                if col not in table:
                    continue
                charts.append(
                    scatter_chart(
                        f"{RESPONSE_LABEL[resp]} to the {mlabel}, {int(n):,} hands",
                        xax,
                        yax,
                        [
                            table_group(
                                ARCHETYPE_LABELS[a],
                                ARCHETYPE_COLORS[a],
                                "bots",
                                "bot_expl",
                                col,
                                where=("archetype", a),
                                shape=ARCHETYPE_SHAPES[a],
                            )
                            for a in ARCHETYPES
                        ],
                        subtitle=f"{len(bots)} held-out bots (5 seeds × 30). Below zero, the "
                        "response "
                        "does worse than equilibrium play against that bot.",
                        refs=[ref(0.0, "Equilibrium play")],
                    )
                )
            views[f"{resp}|{n}"] = view(*charts)

    def lose_share(resp: str, n: str, arch: str | None = None) -> float:
        sel = [b for b in bots if arch is None or b["archetype"] == arch]
        return float(np.mean([b["card_blind"][n][resp]["gain"] < 0 for b in sel]))

    big = [n for n in sizes if int(n) >= 300]
    rng = {
        (r, a): (min(lose_share(r, n, a) for n in big), max(lose_share(r, n, a) for n in big))
        for r in ("br", "rnr_0.5", "rnr_0.25")
        for a in ARCHETYPES
    }
    fitted_losses = max(
        float(np.mean([b["sizes"][n]["scores"]["br"]["gain"] < 0 for b in bots])) for n in sizes
    )
    rho = float(
        spearmanr(
            [b["bot_exploitability"] for b in bots],
            [b["sizes"][sizes[-1]]["scores"]["br"]["gain"] for b in bots],
        )[0]
    )
    fitted_txt = (
        "never does worse than equilibrium play against any bot, at any sample size"
        if fitted_losses == 0
        else f"does worse than equilibrium play against at most {fitted_losses:.0%} of bots"
    )

    def pct_range(lo_hi: tuple[float, float]) -> str:
        lo, hi = lo_hi
        return f"{lo:.0%}" if abs(hi - lo) < 0.005 else f"{lo:.0%}-{hi:.0%}"

    return section(
        "per-bot",
        "Bot by bot: who gets exploited, and who punishes it",
        "Averages hide the bots a response loses against. Each dot is one held-out bot; hover "
        "for its archetype and id.",
        sources=[EXPLOIT],
        controls=[
            control("resp", "Response", [(r, RESPONSE_LABEL[r]) for r in RESPONSES]),
            control("hands", "Hands observed", [(n, f"{int(n):,}") for n in sizes]),
        ],
        views=views,
        tables={"bots": table},
        takeaways=[
            f"More exploitable bots offer more gain (Spearman ρ = {rho:.2f} between a bot's "
            "exploitability and the best response's gain after 3,000 hands); maniacs offer "
            "the most.",
            f"With the fitted model the best response {fitted_txt}.",
            f"With the card-blind model (300+ hands), the best response loses against "
            f"{pct_range(rng[('br', 'tag')])} of TAGs and {pct_range(rng[('br', 'lag')])} of LAGs "
            f"but {pct_range(rng[('br', 'calling_station')])} of calling stations.",
            f"Restricted Nash with p = 0.5 on the same wrong model loses only against some nits "
            f"({pct_range(rng[('rnr_0.5', 'nit')])}); with p = 0.25 against "
            f"{pct_range(rng[('rnr_0.25', 'nit')])} of nits and no other bot.",
        ],
    )


def model_accuracy(ex: dict[str, Any]) -> dict[str, Any]:
    per_seed = _bots(ex)
    sizes = [str(n) for n in ex["meta"]["config"]["sizes"]]
    views = {}
    for arch, alabel in ARCH_OPTIONS:
        err = line_chart(
            f"Parameter error vs hands ({alabel.lower() if arch == 'all' else alabel})",
            HANDS,
            axis("Mean absolute error of the fitted parameter", lo=0),
            [
                series(
                    LEDUC_PARAM_LABEL[p],
                    f"c{i}",
                    [
                        (
                            int(n),
                            *_seed_ci(
                                per_seed,
                                arch,
                                lambda b, n=n, i=i: b["sizes"][n]["param_abs_error"][i - 1],
                            ),
                        )
                        for n in sizes
                    ],
                )
                for i, p in enumerate(LEDUC_PARAMS, start=1)
            ],
            subtitle="Posterior mode with a population prior; parameters are probabilities "
            "in [0, 1].",
        )
        watch = line_chart(
            "Value of watching this bot",
            HANDS,
            axis("Gain above the population-model response (chips/hand)", fmt="signed"),
            [
                series(
                    RESPONSE_LABEL[r],
                    RESPONSE_COLOR[r],
                    [
                        (
                            int(n),
                            *_seed_ci(
                                per_seed,
                                arch,
                                lambda b, n=n, r=r: (
                                    b["sizes"][n]["scores"][r]["gain"]
                                    - b["population_model"][r]["gain"]
                                ),
                            ),
                        )
                        for n in sizes
                    ],
                )
                for r in ("br", "rnr_0.9", "rnr_0.5")
            ],
            subtitle="Same response type fit to zero hands (the population prior) as the baseline.",
            refs=[ref(0.0, "Population model")],
        )
        views[arch] = view(err, watch)
    mae = ex["aggregate"]["by_size"]
    all_bots = [b for seed in per_seed for b in seed]
    final = np.mean([b["sizes"][sizes[-1]]["param_abs_error"] for b in all_bots], axis=0)
    order = np.argsort(final)
    slow = [LEDUC_PARAMS[i] for i in order[-2:][::-1]]
    return section(
        "model-fit",
        "How well the bot's strategy is recovered",
        "The model has five interpretable parameters per bot. How fast do they converge, and "
        "what is the extra information worth?",
        sources=[EXPLOIT],
        controls=[control("arch", "Bots", ARCH_OPTIONS)],
        views=views,
        takeaways=[
            f"Mean absolute parameter error falls from {mae['10']['param_mae']['mean']:.3f} after "
            f"10 hands to {mae['3000']['param_mae']['mean']:.3f} after 3,000.",
            f"Slowest to learn after 3,000 hands: {LEDUC_PARAM_LABEL[slow[0]].lower()} "
            f"({final[LEDUC_PARAMS.index(slow[0])]:.3f}) and {LEDUC_PARAM_LABEL[slow[1]].lower()} "
            f"({final[LEDUC_PARAMS.index(slow[1])]:.3f}), vs {final[order[0]]:.3f} for the best. "
            "A fold ends the hand before showdown, so the folded card is never seen and the "
            "likelihood has to average over it.",
            "The right chart is a post hoc control: the same response fit to zero hands (the "
            "population prior) is the baseline, so the curve is what observing this bot adds.",
        ],
    )


AV_ROWS = (
    "duplicate",
    "mivat",
    "mivat_io",
    "aivat",
    "aivat_duplicate",
    "aivat_all",
    "hero_view",
    "hero_view_approx",
)
AV_LABEL = {
    "chips": "Plain result",
    "duplicate": "Duplicate dealing",
    "mivat": "Chance terms only (MIVAT)",
    "mivat_io": "Chance terms + imaginary observations",
    "aivat": "AIVAT, every hand's cards shown",
    "aivat_duplicate": "AIVAT + duplicate dealing",
    "aivat_all": "AIVAT, the opponent's strategy known too",
    "hero_view": "Hero's view (folded cards hidden)",
    "hero_view_approx": "Hero's view + board terms (biased)",
}
# push/fold: the plain result is P3.3's all-in EV, so its colors follow the bridge section's
AV_PF_LABEL = {
    **AV_LABEL,
    "chips": "All-in EV",
    "duplicate": "Duplicate + all-in EV",
}
AV_COLOR = {
    "chips": "c3",
    "duplicate": "c4",
    "aivat": "c1",
    "aivat_duplicate": "c7",
    "hero_view": "c5",
    "hero_view_approx": "c2",
}
AV_HEROES = (("equilibrium", "Equilibrium hero", "c1"), ("exploiter", "Exploiting hero", "c3"))
AV_VALUES = (
    ("equilibrium", "Equilibrium values (LP)"),
    ("oracle", "Oracle values (the true ones)"),
    ("uniform", "Uniform-play values"),
)
AV_VALUE_TITLE = {
    "equilibrium": "the exact equilibrium's values",
    "oracle": "oracle values",
    "uniform": "uniform-play values",
}
AV_PF_HEROES = (("nash", "Nash hero"), ("best_response", "Best-response hero"))
AV_PF_HERO_TEXT = {"nash": "Nash", "best_response": "best-response"}
AV_CAP = 1e9  # the experiment's ratio for an estimator with no variance left
AV_ZERO_SD = 1e-9  # the experiment's threshold for "no variance"
RATIO_X = axis("Times fewer hands than the plain result (log)", scale="log", fmt="x")


def _times(x: float) -> str:
    return "no variance left" if x >= AV_CAP else f"{x:.2f}×" if x < 10 else f"{x:,.0f}×"


def _holds(p: dict[str, Any]) -> str:
    return "holds" if p["pass"] else "does not hold"


def _av_ratio(q: dict[str, float]) -> list[float | None] | None:
    """[median, q25, q75] of a ratio, or None when the estimator has no variance left."""
    if q["median"] >= AV_CAP:
        return None
    return [q["median"], *(None if q[k] >= AV_CAP else q[k] for k in ("q25", "q75"))]


def _lower_first(label: str) -> str:
    return label[0].lower() + label[1:]


def _av_opponent(opp: str) -> str:
    return "Nash" if opp == "nash" else ARCHETYPE_LABELS[opp]


def _av_get(entry: dict[str, Any], name: str) -> dict[str, float]:
    return entry["base"][name] if name in entry["base"] else entry["values"]["equilibrium"][name]


def aivat_leduc(r: dict[str, Any]) -> dict[str, Any]:
    pop, pred, cfg = r["population"], r["predictions"], r["meta"]["config"]
    sp, paper = r["selfplay"], r["paper_selfplay_sd"]
    selfplay = dots_chart(
        "Leduc self-play: standard deviation per hand",
        [AV_LABEL[n] for n in ("chips", *AV_ROWS)],
        axis("Standard deviation per hand (chips)", lo=0),
        [
            dot_series(
                "This run (exact)",
                "c1",
                [(_av_get(sp, n)["sd"],) for n in ("chips", *AV_ROWS)],
            ),
            dot_series(
                "Burch et al. (2018), Table 1",
                "muted",
                [
                    (r["paper_selfplay_sd"][n],) if n in r["paper_selfplay_sd"] else None
                    for n in ("chips", *AV_ROWS)
                ],
            ),
        ],
        subtitle="The equilibrium against itself, with its own values, whichever value "
        "function is selected above. The paper sampled hands and used approximate values; here "
        f"every number is exact. Chance terms only differ from the paper's "
        f"({_av_get(sp, 'mivat')['sd']:.2f} against {paper['mivat']:.2f}), for a reason not "
        "established.",
    )
    views = {}
    for value, _ in AV_VALUES:
        notes = []
        for n in AV_ROWS:
            capped = [h for h, _, _ in AV_HEROES if pop[h][value][n]["ratio"]["median"] >= AV_CAP]
            if not capped:
                continue
            every = all(pop[h][value][n]["ratio"]["min"] >= AV_CAP for h in capped)
            heroes = " and ".join(lab.lower() for h, lab, _ in AV_HEROES if h in capped)
            notes.append(
                f"No dot for {AV_LABEL[n]} ({heroes}): no variance left "
                + ("against any bot." if every else "for the median bot.")
            )
        views[value] = view(
            dots_chart(
                f"How many times fewer hands each estimator needs, {AV_VALUE_TITLE[value]}",
                [AV_LABEL[n] for n in AV_ROWS],
                RATIO_X,
                [
                    dot_series(
                        hlabel, color, [_av_ratio(pop[h][value][n]["ratio"]) for n in AV_ROWS]
                    )
                    for h, hlabel, color in AV_HEROES
                ],
                subtitle=f"Median over the {len(r['bots'])} bots; whiskers: the middle half. "
                "Variance of the plain result over the estimator's, per hand (a duplicate pair "
                "counts as two hands).",
                refs=[ref(1.0, "plain", axis_name="x")],
            ),
            selfplay,
            notes=notes,
        )
    eq, ex = pop["equilibrium"], pop["exploiter"]
    q = eq["equilibrium"]["aivat"]["ratio"]
    bias = {h: pop[h]["equilibrium"]["hero_view_approx"]["max_abs_bias"] for h, _, _ in AV_HEROES}
    takeaways = [
        f"With every hand's cards shown, AIVAT needs {_times(q['median'])} fewer hands than "
        f"the plain result for the equilibrium hero (middle half of the bots "
        f"{_times(q['q25'])}-{_times(q['q75'])}, every bot at least {_times(q['min'])}; B1 "
        f"{_holds(pred['B1'])}) and {_times(ex['equilibrium']['aivat']['ratio']['median'])} for "
        f"the exploiting hero (B2 {_holds(pred['B2'])}).",
        f"Oracle values (the true values of this hero against this bot, unavailable in "
        f"practice) raise the equilibrium hero's savings to "
        f"{_times(eq['oracle']['aivat']['ratio']['median'])}, "
        f"{eq['oracle']['aivat']['ratio']['median'] / q['median']:.2f} times as many (B3 "
        f"{_holds(pred['B3'])}). The exploiting hero reaches "
        f"{_times(ex['oracle']['aivat']['ratio']['median'])}: what is left comes from the bot's "
        "decisions, which AIVAT cannot correct without the bot's strategy. With it known too, "
        "oracle values leave no variance.",
        f"Dealing every deal twice adds little on top: "
        f"{_times(eq['equilibrium']['aivat_duplicate']['ratio']['median'])} against "
        f"{_times(q['median'])} (E1 {_holds(pred['E1'])}). Duplicate dealing alone gives "
        f"{_times(eq['equilibrium']['duplicate']['ratio']['median'])}.",
        f"Hide the folded cards and almost none of the savings are left: the hero's own view "
        f"needs "
        f"{_times(eq['equilibrium']['hero_view']['ratio']['median'])} and "
        f"{_times(ex['equilibrium']['hero_view']['ratio']['median'])} fewer hands (C1 "
        f"{_holds(pred['C1'])}). Also correcting the board, with the opponent's card taken as "
        f"uniform over the cards the hero has not seen, gives "
        f"{_times(eq['equilibrium']['hero_view_approx']['ratio']['median'])} and "
        f"{_times(ex['equilibrium']['hero_view_approx']['ratio']['median'])}, but it is biased: "
        f"by up to {bias['exploiter']:.3f} chips/hand against the bots for the exploiting hero, "
        f"more than half the ±{cfg['half_width']} the intervals aim for, and "
        f"{bias['equilibrium']:.3f} for the equilibrium hero (C2 {_holds(pred['C2'])}).",
        f"In self-play AIVAT leaves no variance (SD {_av_get(sp, 'aivat')['sd']:.0e}, rounding "
        f"error; the paper: {paper['aivat']} chips/hand with approximate values): an "
        "equilibrium mixes only between actions of equal value, so with its own values the "
        "opponent's choices, the steps AIVAT does not correct, never change the value. The "
        f"plain result's SD {sp['base']['chips']['sd']:.3f} matches the paper's "
        f"{paper['chips']}.",
        f"Every estimator but the biased one is exactly unbiased ({pred['A1']['result']}; A1 "
        f"{_holds(pred['A1'])}), and {cfg['sampled_pairs']:,} sampled duplicate deals for each "
        f"of {cfg['sampled_bots'] * len(ARCHETYPES)} bots ({cfg['sampled_bots']} per archetype) "
        f"and both heroes agree with the exact moments ({pred['A2']['result']}; A2 "
        f"{_holds(pred['A2'])}).",
    ]
    return section(
        "aivat-leduc",
        "Win rates from fewer hands: AIVAT in Leduc",
        "A live home game cannot deal every hand twice and will not reach tens of thousands of "
        "hands. AIVAT (Burch, Schmid, Moravčík, Morrill and Bowling, 2018) subtracts, at every "
        "step whose probabilities are known, the change in a value function minus its expected "
        "change: luck the evaluator can price. Each such term has mean zero, so the estimate "
        "stays unbiased whatever the values.",
        sources=[AIVAT],
        controls=[control("values", "Value function", AV_VALUES)],
        views=views,
        takeaways=takeaways,
        method=f"{len(r['bots'])} Leduc bots ({cfg['bots_per_archetype']} per archetype). "
        f"Heroes: the equilibrium, and a restricted Nash response (p = {cfg['rnr_p']}) to the "
        "bot's true strategy. Every estimator's mean and variance are sums over the game tree, "
        "so every number is exact; duplicate dealing's variance comes from per-deal moments. "
        "Pre-registered predictions A1-E1 (seed 2026).",
    )


def aivat_values(r: dict[str, Any]) -> dict[str, Any]:
    pred = r["predictions"]
    names = ("aivat", "hero_view", "hero_view_approx")
    views = {}
    for hero, hlabel, _ in AV_HEROES:
        rows = [row for row in r["curve"][hero] if row["value"].startswith("cfr+_")]
        lp = next(row for row in r["curve"][hero] if row["value"] == "equilibrium")
        views[hero] = view(
            line_chart(
                f"{hlabel}: savings by the quality of the values",
                axis(
                    "CFR+ iterations behind the values (1 = uniform play)", scale="log", fmt="int"
                ),
                axis("Times fewer hands than the plain result (log)", scale="log", fmt="x"),
                [
                    series(
                        AV_LABEL[n],
                        AV_COLOR[n],
                        [
                            (int(row["value"][5:]), row[n]["median"], row[n]["q25"], row[n]["q75"])
                            for row in rows
                        ],
                    )
                    for n in names
                ],
                subtitle="Self-play values of CFR+'s average strategy; median over the bots, "
                "bands: the middle half.",
                refs=[ref(lp["aivat"]["median"], "AIVAT with the exact equilibrium's values (LP)")],
            )
        )
    curve = {h: {row["value"]: row for row in r["curve"][h]} for h, _, _ in AV_HEROES}
    eq, ex = curve["equilibrium"], curve["exploiter"]
    best = max((row for row in r["curve"]["exploiter"]), key=lambda row: row["aivat"]["median"])
    late = [
        row["aivat"]["median"]
        for v, row in eq.items()
        if v.startswith("cfr+_") and int(v[5:]) >= 30
    ]
    view_medians = [row["hero_view"]["median"] for c in curve.values() for row in c.values()]
    view_oracle = max(
        r["population"][h]["oracle"]["hero_view"]["ratio"]["median"] for h, _, _ in AV_HEROES
    )
    takeaways = [
        f"Values from uniform play, which know nothing about strategy, already give AIVAT "
        f"{_times(eq['uniform']['aivat']['median'])} for the equilibrium hero, "
        f"{eq['uniform']['aivat']['median'] / eq['equilibrium']['aivat']['median']:.0%} of the "
        f"exact equilibrium's {_times(eq['equilibrium']['aivat']['median'])} (D1 "
        f"{_holds(pred['D1'])}).",
        f"From 30 CFR+ iterations on (exploitability {eq['cfr+_30']['exploitability']:.3f} "
        f"chips/hand and below) the savings are at least as good as with the exact "
        f"equilibrium's values: {_times(min(late))} to {_times(max(late))} against "
        f"{_times(eq['equilibrium']['aivat']['median'])} (D2 {_holds(pred['D2'])}).",
        f"For the exploiting hero the savings are not monotone in the values' quality: they "
        f"peak at {int(best['value'][5:]) if best['value'].startswith('cfr+_') else best['value']}"
        f" iterations ({_times(best['aivat']['median'])}) and fall to "
        f"{_times(ex['equilibrium']['aivat']['median'])} with the exact equilibrium's values "
        "(not pre-registered).",
        f"The hero's own view stays between {min(view_medians):.2f}× and "
        f"{max(view_medians):.2f}× at every checkpoint, and reaches only {view_oracle:.2f}× "
        "with oracle values: better values cannot replace the cards it never sees.",
    ]
    return section(
        "aivat-values",
        "How good must the value function be?",
        "AIVAT stays unbiased with any value function; the values change only how much "
        "variance it removes. Here the values come from CFR+ stopped after 1 to 1,000 "
        "iterations.",
        sources=[AIVAT],
        controls=[control("hero", "Hero", [(h, lab) for h, lab, _ in AV_HEROES])],
        views=views,
        takeaways=takeaways,
        method="One CFR+ run on Leduc; the values at each checkpoint are its average "
        "strategy's self-play values, and the hero's-view estimators weigh the opponent's cards "
        "by that profile's reach. Pre-registered predictions D1 and D2.",
    )


def aivat_pushfold(r: dict[str, Any]) -> dict[str, Any]:
    pred, cfg = r["predictions"], r["meta"]["config"]["pushfold"]
    pf, rows = r["pushfold_summary"], r["pushfold"]
    stacks = list(pf)
    names = ("chips", "duplicate", "aivat", "aivat_duplicate", "hero_view")
    views = {}
    for hero, hlabel in AV_PF_HEROES:

        def spread(st: str, name: str, hero: str = hero) -> list[float | None]:
            hands = [
                _av_get(x, name)["hands"]
                for x in rows
                if str(x["stack"]) == st and x["hero"] == hero and x["opponent"] != "nash"
            ]
            lo, hi = min(hands), max(hands)
            return [pf[st][hero][name]["hands_median"], lo if lo > 0 else None, hi]

        hands = dots_chart(
            f"{hlabel}: hands for a ±{cfg['half_width']:g} bb/hand 95% interval",
            [f"{float(st):g} bb" for st in stacks],
            axis("Hands needed (log scale)", scale="log", fmt="int"),
            [
                dot_series(AV_PF_LABEL[n], AV_COLOR[n], [spread(st, n) for st in stacks])
                for n in names
            ],
            subtitle="Median over the five archetype bots at their mean parameters; whiskers: "
            "the five bots' range. Exact, in the abstract push/fold game.",
        )
        checks = [c for c in r["bridge_crosscheck"] if c["hero"] == hero]
        dropped = [c for c in checks if c["exact_sd"] < AV_ZERO_SD]
        cross = scatter_chart(
            f"{hlabel}: P3.3's sampled standard deviations against the exact ones",
            axis("Exact standard deviation per hand (bb, log)", scale="log", fmt="num"),
            axis("P3.3's sampled SD / exact SD"),
            [
                scatter_group(
                    AV_PF_LABEL[est],
                    AV_COLOR[est],
                    [
                        (
                            c["exact_sd"],
                            c["ratio"],
                            f"{c['stack']:g} bb, against {_av_opponent(c['opponent'])}",
                        )
                        for c in checks
                        if c["estimator"] == est and c["exact_sd"] >= AV_ZERO_SD
                    ],
                )
                for est in ("chips", "duplicate")
            ],
            subtitle="P3.3 played 100,000 real deals per matchup; F1 predicted every ratio "
            "within 3% of 1.",
            refs=[ref(0.97, "F1's band"), ref(1.03, "F1's band")],
        )
        views[hero] = view(
            hands,
            cross,
            notes=[
                f"Not shown: {_lower_first(AV_PF_LABEL[c['estimator']])} at {c['stack']:g} bb "
                f"against {_av_opponent(c['opponent'])}, which has no variance exactly "
                + (
                    "and none in P3.3's sample."
                    if c["sampled_sd"] < AV_ZERO_SD
                    else f"(P3.3's sampled SD: {c['sampled_sd']:.2g})."
                )
                for c in dropped
            ],
        )
    nash = {st: pf[st]["nash"] for st in stacks}
    view_ratios = [pf[st][h]["hero_view"]["ratio_median"] for st in stacks for h in pf[st]]
    misses = [c for c in r["bridge_crosscheck"] if not 0.97 <= c["ratio"] <= 1.03]
    miss_txt = "; ".join(
        f"{m['stack']:g} bb, {AV_PF_HERO_TEXT[m['hero']]} hero against "
        f"{_av_opponent(m['opponent'])}, {_lower_first(AV_PF_LABEL[m['estimator']])}: "
        f"{m['ratio']:.3f} (exact SD {m['exact_sd']:.3f} bb/hand, {m['exact_hands']:,} hands)"
        for m in misses
    )
    takeaways = [
        "With every hand's cards shown, the Nash hero's win rate needs (median over the five "
        "archetype bots) "
        + ", ".join(
            f"{nash[st]['aivat']['hands_median']:,.0f} hands at {float(st):g} bb (all-in EV: "
            f"{nash[st]['chips']['hands_median']:,.0f})"
            for st in stacks
        )
        + f" for ±{cfg['half_width']:g} bb/hand (F2 {_holds(pred['F2'])}).",
        "The best-response hero gains less: "
        + ", ".join(
            f"{_times(pf[st]['best_response']['aivat']['ratio_median'])} at {float(st):g} bb"
            for st in stacks
        )
        + ".",
        f"With folded cards hidden the hero's own view gains {_times(min(view_ratios))} to "
        f"{_times(max(view_ratios))} over "
        f"all-in EV (F3 {_holds(pred['F3'])}).",
        f"F1 {_holds(pred['F1'])}: {len(r['bridge_crosscheck']) - len(misses)} of "
        f"{len(r['bridge_crosscheck'])} sampled SDs from P3.3 are within 3% of the exact ones; "
        f"the exception is {miss_txt}.",
    ]
    return section(
        "aivat-pushfold",
        "AIVAT in push/fold, and a check against P3.3's real hands",
        "Five of the estimators in P3.3's 33 push/fold matchups, where the plain result is already "
        "the all-in EV (the luck of the board is gone). P3.3 measured all-in EV and duplicate + "
        "all-in EV on real deals, so its sampled spreads check the exact ones.",
        sources=[AIVAT],
        controls=[control("hero", "Hero", AV_PF_HEROES)],
        views=views,
        takeaways=takeaways,
        method="The abstract push/fold game at 5, 10 and 20 bb with P3.3's heroes and opponents; "
        "values from the stack's Nash equilibrium. Pre-registered predictions F1-F3.",
    )


def _f1_row(e: dict[str, Any]) -> str:
    return f"{_av_opponent(e['opponent'])}, {_lower_first(AV_PF_LABEL[e['estimator']])}"


def aivat_f1(ph: dict[str, Any]) -> dict[str, Any]:
    cfg, s = ph["meta"]["config"], ph["summary"]
    lo, hi = cfg["band"]
    views = {}
    for hero, hlabel in AV_PF_HEROES:
        for st in sorted({e["stack"] for e in ph["estimates"]}):
            rows = [e for e in ph["estimates"] if e["hero"] == hero and e["stack"] == st]
            views[f"{hero}|{st:g}"] = view(
                dots_chart(
                    f"{hlabel} at {st:g} bb: P3.3's sampled SD against the exact one",
                    [_f1_row(e) for e in rows],
                    axis("P3.3's sampled SD / exact SD"),
                    [
                        dot_series(
                            "Sampled / exact, with its 95% sampling interval",
                            "c1",
                            [
                                None
                                if e["rel_se"] is None
                                else [
                                    e["ratio"],
                                    e["ratio"] - 1.96 * e["rel_se"],
                                    e["ratio"] + 1.96 * e["rel_se"],
                                ]
                                for e in rows
                            ],
                        )
                    ],
                    ranges=[[lo, hi] for _ in rows],
                    range_label="F1's pre-registered band",
                    refs=[ref(1.0, "exact", axis_name="x")],
                    subtitle=f"Whiskers: ±1.96 × the sample SD's standard error at "
                    f"{cfg['n_units']:,} units, sqrt((kurtosis - 1) / (4n)).",
                ),
                notes=[
                    f"No dot for {_f1_row(e)}: no variance." for e in rows if e["rel_se"] is None
                ],
            )
    worst = next(e for e in ph["estimates"] if not lo <= e["ratio"] <= hi)
    sims = {x["stack"]: x for x in ph["simulated"]}
    sim = sims[worst["stack"]]
    others = [x for x in ph["simulated"] if x["stack"] != worst["stack"]]
    q = sim["quantiles"]
    takeaways = [
        f"A sample SD is much noisier than a sample mean when the data have heavy tails: its "
        f"relative standard error is about sqrt((kurtosis - 1) / (4n)). F1's ±{hi - 1:.0%} band "
        "ignored this.",
        f"F1's one miss ({worst['ratio']:.3f}, the Nash hero against Nash at "
        f"{worst['stack']:g} bb, duplicate + all-in EV) has kurtosis {worst['kurtosis']:,.0f}: "
        "with the same strategy on both sides, the two tables of a pair cancel exactly unless a "
        f"mixed decision goes differently at the two, which happens in "
        f"{worst['nonzero_share']:.2%} of pairs. Its relative SE is "
        f"{worst['rel_se']:.1%}, so the miss is z = {worst['z']:+.2f}.",
        f"Drawing P3.3's sample {cfg['reps']:,} times from the exact distribution, 95% of "
        f"ratios fall in [{q['0.025']:.3f}, {q['0.975']:.3f}] and "
        f"{sim['share_at_or_below_observed']:.0%} are at or below {worst['ratio']:.3f}; the "
        f"band is missed {sim['share_outside_band']:.0%} of the time even with exact agreement"
        + "".join(
            f" ({x['share_outside_band']:.0%} at {x['stack']:g} bb, where the observed "
            f"{x['observed']:.3f} passed)"
            for x in others
        )
        + ".",
        f"Over all {s['with_variance']} estimates with variance the z-scores look standard "
        f"normal: largest |z| {s['max_abs_z']:.2f}, {s['share_abs_z_above_1_96']:.0%} beyond "
        f"1.96, SD {s['sd_z']:.2f}, mean {s['mean_z']:+.2f} (estimates from the same matchup "
        "share deals, so they are not independent). The exact moments agree with P3.3's real "
        f"hands; only {s['interval_wider_than_band']} estimates had a sampling interval wider "
        "than F1's band.",
    ]
    return section(
        "aivat-f1",
        "Post hoc: was F1's miss sampling noise?",
        "F1 predicted that P3.3's sampled spreads would match the exact ones within 3% in every "
        "matchup, and one of 66 did not. This check, chosen after seeing the miss, asks how "
        "precise each sampled spread could be.",
        sources=[AIVAT_POSTHOC],
        controls=[
            control("hero", "Hero", AV_PF_HEROES),
            control(
                "stack",
                "Stack",
                [(f"{st:g}", f"{st:g} bb") for st in sorted({e["stack"] for e in ph["estimates"]})],
            ),
        ],
        views=views,
        takeaways=takeaways,
        method="For each of F1's estimates, the exact distribution of P3.3's unit (one hand, or "
        "a duplicate pair's mean with the two plays independent given the deal) in the "
        "abstract push/fold game; its exact SD matches aivat.json. The two Nash-against-Nash "
        "duplicate estimates with variance are also simulated from that distribution.",
        post_hoc=True,
    )


DUP_KINDS = (
    ("mixed_6", "Six bots, random mix"),
    ("mirror_6", "Six bots of one archetype"),
    ("clones_6", "Six copies of one bot"),
    ("mixed_3", "Three bots, random mix"),
    ("mixed_2", "Two bots, random mix"),
)
DUP_RATIO = axis("Times fewer hands than independent dealing (log)", scale="log", fmt="x")


def duplicate_dealing(r: dict[str, Any]) -> dict[str, Any]:
    by, pred, cfg = r["aggregates"]["by_kind"], r["predictions"], r["meta"]["config"]
    bridge = r["bridge"]["all"]["ratio_median"]

    def quartiles(q: dict[str, float]) -> list[float]:
        return [q["median"], q["q25"], q["q75"]]

    def held(key: str) -> str:
        return f"{key} {'holds' if pred[key]['passed'] else 'does not hold'}"

    def cancelled(kind: str) -> float:
        """The share of the possible cancellation: -(n - 1) rho, 1 at perfect cancellation."""
        return -(by[kind]["n"] - 1) * by[kind]["rho"]["median"]

    labels = [lab for _, lab in DUP_KINDS]
    overall = dots_chart(
        "How many times fewer hands duplicate dealing needs",
        [*labels, "Heads-up push/fold (Step 28)"],
        DUP_RATIO,
        [
            dot_series(
                "Bots at 100 bb: median over players, whiskers the middle half",
                "c1",
                [*[quartiles(by[k]["ratio"]) for k, _ in DUP_KINDS], None],
            ),
            dot_series(
                "Median over the matchups against the bots",
                "c2",
                [*[None] * len(DUP_KINDS), [bridge]],
            ),
        ],
        subtitle="Per hand played: a deal at n tables costs n hands. Variance per hand with "
        "independent deals over n times the variance of a deal's mean.",
        refs=[ref(1.0, "no gain", axis_name="x")],
    )
    share = dots_chart(
        "How much of the possible cancellation happens (post hoc)",
        labels,
        axis("Share of perfect cancellation: -(n - 1) x correlation", lo=0, hi=1, fmt="pct"),
        [dot_series("Median over players", "c3", [[cancelled(k)] for k, _ in DUP_KINDS])],
        subtitle="A player's results at two tables of one deal have correlation rho, and the "
        "ratio is 1 / (1 + (n - 1) rho). At -(n - 1) rho = 100% the deal's mean never varies, "
        "as for the always-call check.",
    )
    views = {}
    for kind, label in DUP_KINDS:
        archetypes = by[kind]["by_archetype"]
        rows = [a for a in ARCHETYPES if a in archetypes]
        note = (
            [
                "Heads-up the two players' results are opposite, so their ratios are equal: a "
                "ratio by archetype describes the pair, not the player."
            ]
            if kind == "mixed_2"
            else []
        )
        views[kind] = view(
            overall,
            dots_chart(
                f"By the player's archetype: {label.lower()}",
                [ARCHETYPE_LABELS[a] for a in rows],
                DUP_RATIO,
                [
                    dot_series(
                        "Median over players, whiskers the middle half",
                        "c1",
                        [quartiles(archetypes[a]["ratio"]) for a in rows],
                    )
                ],
                refs=[ref(1.0, "no gain", axis_name="x")],
            ),
            share,
            notes=note,
        )
    mix, mirror, clones = by["mixed_6"], by["mirror_6"], by["clones_6"]
    mixed_n = r["aggregates"]["mixed_by_n"]
    needed = mix["hands_needed"]
    below = [by[k]["share_below_1"] for k in ("clones_6", "mirror_6", "mixed_6")]
    takeaways = [
        f"With six bots in a random mix, duplicate dealing needs {_times(mix['ratio']['median'])} "
        f"fewer hands than independent dealing (middle half of the players "
        f"{_times(mix['ratio']['q25'])}-{_times(mix['ratio']['q75'])}, 95% CI of the median "
        f"{_times(mix['ratio_ci']['lower'])}-{_times(mix['ratio_ci']['upper'])}; {held('B1')}). "
        f"Heads-up push/fold, where every hand is an all-in or a fold, saved "
        f"{_times(bridge)} (Step 28).",
        f"Identical players do not rescue it: six copies of one bot save "
        f"{_times(clones['ratio']['median'])} ({held('B2')}), six bots of one archetype "
        f"{_times(mirror['ratio']['median'])}, the random mix {_times(mix['ratio']['median'])} "
        f"({held('B3')}). Copies can differ only through the bots' own randomness, since each "
        "table draws its own random numbers. The section on where the copies' shortfall comes "
        "from divides it between mixed actions, bet sizes and Monte Carlo equity.",
        "The gain stays about flat with the number of players: "
        + ", ".join(
            f"{_times(mixed_n[n]['ratio_median'])} at n = {n} (rho {mixed_n[n]['rho_median']:+.3f})"
            for n in ("6", "3", "2")
        )
        + f" ({held('E1')}; {held('E2')}). A smaller rho at six seats is not less cancellation: "
        "the ratio depends on (n - 1) rho, which comes to "
        + ", ".join(f"{cancelled(k):.0%}" for k in ("mixed_6", "mixed_3", "mixed_2"))
        + " of perfect cancellation at each size (post hoc).",
        f"Nits gain least: {_times(mix['by_archetype']['nit']['ratio']['median'])} in the random "
        f"mix ({held('C1')}) and {_times(clones['by_archetype']['nit']['ratio']['median'])} among "
        f"copies ({held('C2')}). Which seat the other table's player holds barely matters: the "
        f"correlations at offsets 1, 2 and 3 stay within 0.03 of each other ({held('D1')}; "
        f"{held('D2')}; {held('D3')}).",
        f"Almost every player gains: {max(below):.1%} or fewer have a ratio below 1 in any "
        f"six-player kind ({held('B4')}). Six always-call agents cancel exactly: every "
        f"per-deal mean over {r['calling_check']['deals']} deals is 0 chips ({held('A1')}).",
        f"It does not make 6-max win rates cheap: for +/-{cfg['half_width']} bb/100 with a random "
        f"mix the median player needs {needed['naive']:,.0f} independent hands and "
        f"{needed['duplicate']:,.0f} duplicate ones. And it stays a simulation tool: a live table "
        "cannot deal the same cards twice.",
    ]
    return section(
        "duplicate-n",
        "Duplicate dealing at more than two seats",
        "Heads-up, duplicate dealing plays each deal twice with the seats swapped. With n players "
        "it plays each deal at n tables with the seats rotated, so every player holds every "
        "seat's cards and position once per deal, and card luck can cancel in the mean of those n "
        "results. The same cards are dealt, but each table has its own decision randomness.",
        sources=[DUPLICATE],
        controls=[control("lineups", "Archetypes for", DUP_KINDS)],
        views=views,
        takeaways=takeaways,
        method=f"{cfg['replicates']} replicates of {cfg['population_per_archetype']} bots per "
        f"archetype (Phase 1's bots, {cfg['equity_samples']} equity samples), blinds 1/2, 100 bb "
        f"stacks, {cfg['deals']:,} deals per lineup; {sum(by[k]['lineups'] for k, _ in DUP_KINDS)} "
        "lineups. CIs resample lineups; a bot can sit in several mixed lineups of a replicate, so "
        "they are a little narrow. The rotation keeps who sits to whose left, so a lineup measures "
        "its seating order; the aggregates average over the orders drawn. Pre-registered "
        "predictions A1-E2 (seed 2026), thresholds from development on seeds 98 and 99.",
    )


def duplicate_posthoc(ph: dict[str, Any]) -> dict[str, Any]:
    normal, by_kind = ph["coverage_normal"], ph["coverage_lineups"]
    colors = dict(zip((k for k, _ in DUP_KINDS), ("c1", "c2", "c3", "c4", "c5"), strict=True))
    coverage = line_chart(
        "How often a nominal 95% interval covers, against the number of lineups",
        axis("Lineups resampled", scale="log", fmt="int"),
        axis("Coverage", fmt="pct"),
        [
            *[
                series(label, colors[kind], [(int(g), c) for g, c in by_kind[kind].items()])
                for kind, label in DUP_KINDS
            ],
            series(
                "Normal theory, mean of the values",
                "muted",
                [(int(g), c) for g, c in normal.items()],
                dash=True,
            ),
        ],
        subtitle="One kind's lineups taken as the truth, drawn with replacement, the step's "
        "interval computed from them. They are both the data and the truth, so this shows the "
        "size of the effect.",
        refs=[ref(0.95, "nominal 95%")],
    )
    hu = ph["heads_up"]
    pairs = sorted(hu["pairs"].items(), key=lambda kv: kv[1]["median"])

    def pair_label(key: str) -> str:
        a, b = key.split("|")
        return f"{ARCHETYPE_LABELS[a]} and {ARCHETYPE_LABELS[b]} ({hu['pairs'][key]['lineups']})"

    heads_up = dots_chart(
        "Heads-up lineups by the pair of archetypes (number of lineups in brackets)",
        [pair_label(k) for k, _ in pairs],
        DUP_RATIO,
        [dot_series("Median ratio", "c1", [[v["median"]] for _, v in pairs])],
        subtitle="The two players' ratios are equal, so a lineup is one value.",
        refs=[ref(1.0, "no gain", axis_name="x")],
    )
    copies, mix = ph["copies"], ph["resampling"]["mixed_6"]
    ten = by_kind["mixed_6"]["10"]
    takeaways = [
        f"A percentile bootstrap over few lineups is too narrow: at 10 lineups a nominal 95% "
        f"interval covered about {ten:.0%} of the time in the random mix "
        f"({normal['10']:.1%} in normal theory), at 50 about {by_kind['mixed_6']['50']:.0%}. The "
        "per-archetype CIs of the copies and one-archetype lineups rest on 10 lineups each.",
        f"Resampling players instead of lineups would give narrower intervals: "
        f"[{mix['players']['lower']:.3f}, {mix['players']['upper']:.3f}] against "
        f"[{mix['lineups']['lower']:.3f}, {mix['lineups']['upper']:.3f}] in the random mix of six, "
        f"which is {mix['width_ratio']:.2f} times as wide. Players at a table share cards and "
        "opponents, so they are not independent draws.",
        f"Heads-up, who plays alike tends to cancel more, with exceptions: lineups of one "
        f"archetype have a median of {hu['same_archetype']['median']:.2f}×, mixed pairs "
        f"{hu['different_archetypes']['median']:.2f}×, but the two highest pairs are "
        f"{pair_label(pairs[-1][0])} and {pair_label(pairs[-2][0])}. These groups were chosen "
        "after seeing the results.",
        f"A single player's ratio is noisy: six copies of one bot differ only by sampling noise, "
        f"and within a lineup their ratios have a median relative SD of "
        f"{copies['within_lineup_relative_sd_median']:.1%}. Between bots the spread is wider: "
        f"lineup medians among copies run from {copies['lineup_median_lowest']['ratio']:.2f}× to "
        f"{copies['lineup_median_highest']['ratio']:.2f}×.",
    ]
    return section(
        "duplicate-posthoc",
        "Post hoc: how far to trust the duplicate step's intervals",
        "Computed after the run from its committed results, playing no hands. Nothing here is "
        "pre-registered.",
        sources=[DUPLICATE_POSTHOC],
        views={"": view(coverage, heads_up)},
        takeaways=takeaways,
        method=f"{ph['meta']['config']['trials']:,} draws of G lineups, "
        f"{ph['meta']['config']['resamples']} resamples each. Source: "
        f"{ph['meta']['source']} at {ph['meta']['source_commit'][:7]}.",
        post_hoc=True,
    )


RAND_SOURCES = (
    ("mixed", "Mixed actions", "c1"),
    ("noise", "Bet-size noise", "c2"),
    ("equity", "Monte Carlo equity", "c3"),
)
RAND_TERMS = (
    ("mixed", "Mixed actions alone"),
    ("noise", "Bet-size noise alone"),
    ("equity", "Monte Carlo equity alone"),
    ("mixed+noise", "Mixed x noise"),
    ("mixed+equity", "Mixed x equity"),
    ("noise+equity", "Noise x equity"),
    ("mixed+noise+equity", "All three together"),
)
RAND_LEVELS = (
    ("baseline", "Default: all three live"),
    ("shared_mixed", "Mixed actions shared"),
    ("shared_noise", "Bet-size noise shared"),
    ("shared_equity", "Equity shared"),
    ("only_mixed", "Only mixed actions live"),
    ("only_noise", "Only bet-size noise live"),
    ("only_equity", "Only equity live"),
    ("shared_all", "All three shared"),
)
RAND_PAIRINGS = (
    ("decision", "The seat's k-th decision on the street"),
    ("history", "The exact public history"),
)


def randomness(r: dict[str, Any]) -> dict[str, Any]:
    agg, pred, cfg = r["aggregates"], r["predictions"], r["meta"]["config"]
    by, checks = agg["by_condition"], agg["checks"]
    dec, hist = agg["decomposition"], agg["decomposition_history"]
    arch = agg["by_archetype"]

    def held(*keys: str) -> str:
        return "; ".join(f"{k} {'holds' if pred[k]['passed'] else 'does not hold'}" for k in keys)

    def triple(q: dict[str, float]) -> list[float]:
        return [q["mean"], q["lower"], q["upper"]]

    def with_ci(q: dict[str, float]) -> str:
        return f"{q['mean']:.1%} [{q['lower']:.1%}, {q['upper']:.1%}]"

    def ratio(name: str) -> str:
        return _times(by[name]["ratio"]["median"])

    def share(d: dict[str, Any], source: str) -> float:
        return d["shares"][f"shapley_{source}"]["mean"]

    labels = [ARCHETYPE_LABELS[a] for a in ARCHETYPES]
    spread = dots_chart(
        "What the spread per hand does when mixed actions are switched off",
        labels,
        axis("Median SD per hand (bb, log)", scale="log", fmt="num"),
        [
            dot_series(name, color, [[arch[a]["sigma_median"][cond]] for a in ARCHETYPES])
            for name, color, cond in (
                ("Default", "c1", "baseline"),
                ("Mixed actions off (u = 0.5)", "c2", "off_mixed"),
            )
        ],
        subtitle="This changes the bots, so it is not part of the split: six calling stations' "
        "results barely vary any more, and nits' vary more.",
    )
    views = {}
    for key, _ in RAND_PAIRINGS:
        name = "decomposition" if key == "decision" else "decomposition_history"
        d = agg[name]
        division = dots_chart(
            "Share of the shortfall by source, interactions split equally",
            [f"All {d['lineups']} lineups", *labels],
            axis("Share of the shortfall", fmt="pct"),
            [
                dot_series(
                    label,
                    color,
                    [
                        triple(d["shares"][f"shapley_{source}"]),
                        *[triple(arch[a][name]["shares"][f"shapley_{source}"]) for a in ARCHETYPES],
                    ],
                )
                for source, label, color in RAND_SOURCES
            ],
            subtitle="Shortfall: the part of a hand's variance the rotation leaves. Means over "
            "lineups, 95% bootstrap CIs. An archetype's CIs rest on 10 lineups, too few to reach "
            "nominal coverage (see the post hoc section above).",
        )
        terms = dots_chart(
            "The seven terms before any are split",
            [lab for _, lab in RAND_TERMS],
            axis("Share of a hand's variance", fmt="pct"),
            [
                dot_series(
                    "Mean over lineups, 95% CI",
                    "c1",
                    [triple(d["terms"][t]) for t, _ in RAND_TERMS],
                )
            ],
            subtitle="A term is the variance that needs exactly those sources to differ between "
            "tables. They sum to the shortfall.",
        )
        levels = dots_chart(
            "What is left in each condition",
            [lab for _, lab in RAND_LEVELS],
            axis("Share of a hand's variance the rotation leaves", fmt="pct"),
            [
                dot_series(
                    "Mean over lineups, 95% CI",
                    "c4",
                    [triple(d["remaining"][c]) for c, _ in RAND_LEVELS],
                )
            ],
            subtitle="The default and all-shared rows do not depend on the pairing; the rows "
            "between them do.",
        )
        views[key] = view(division, terms, levels, spread)
    change = by["off_mixed"]["behaviour_change"]
    noise_change = by["off_noise"]["behaviour_change"]
    station = arch["calling_station"]["sigma_median"]
    nit = arch["nit"]["sigma_median"]
    equity = {a: arch[a]["decomposition"]["shares"]["shapley_equity"]["mean"] for a in ARCHETYPES}
    others = [equity[a] for a in ARCHETYPES if a != "maniac"]
    hands = len(by) * dec["lineups"] * cfg["deals"] * 6
    takeaways = [
        f"Six copies of one bot should cancel perfectly and don't: the rotation cancels "
        f"{with_ci(dec['cancelled'])} of a hand's variance (median ratio {ratio('baseline')}, the "
        f"same as the earlier step to the last digit; {held('A1')}). With all three sources "
        "shared or off, every deal of every lineup sums to exactly 0 chips "
        f"({held('A2')}), so the whole shortfall of {with_ci(dec['remaining']['baseline'])} comes "
        "from the copies drawing different random numbers.",
        f"Pairing the tables by each seat's k-th decision on a street, mixed actions account for "
        f"{with_ci(dec['shares']['shapley_mixed'])} of the shortfall, Monte Carlo equity "
        f"{with_ci(dec['shares']['shapley_equity'])} and bet-size noise "
        f"{with_ci(dec['shares']['shapley_noise'])}, with interactions split equally "
        f"({held('D2', 'D3', 'D4')}).",
        f"That split depends on the pairing. Pairing by the exact public history, including bet "
        f"sizes, gives {share(hist, 'mixed'):.1%}, {share(hist, 'equity'):.1%} and "
        f"{share(hist, 'noise'):.1%} ({held('H3', 'H4', 'H5')}): two tables whose bets differ at "
        f"all in size draw fresh mixed actions from then on, so the mixed x noise term goes from "
        f"{dec['terms']['mixed+noise']['mean']:.1%} to {hist['terms']['mixed+noise']['mean']:.1%} "
        f"of a hand's variance ({held('H6')}). Mixed actions stay above noise under both "
        f"({held('H7a', 'H7b')}), but the {share(dec, 'mixed'):.0%} is not a property of the bots "
        "alone.",
        "What barely depends on the pairing: with only mixed actions live, "
        f"{dec['remaining']['only_mixed']['mean'] / dec['remaining']['baseline']['mean']:.0%} and "
        f"{hist['remaining']['only_mixed']['mean'] / hist['remaining']['baseline']['mean']:.0%} of "
        f"the default's shortfall stays ({held('C1')}); sharing noise or equity alone leaves the "
        f"ratio at {ratio('shared_noise')} and {ratio('shared_equity')} ({held('C3')}). So mixed "
        "actions alone are enough to produce about all of it. With only equity live, "
        f"{dec['remaining']['only_equity']['mean']:.1%} and "
        f"{hist['remaining']['only_equity']['mean']:.1%} of a hand's variance is left "
        f"({held('C2')}). Noise's level is what the pairing moves most: only noise live leaves "
        f"{dec['remaining']['only_noise']['mean']:.1%} or "
        f"{hist['remaining']['only_noise']['mean']:.1%} ({held('H1a', 'H1b')}).",
        f"Under the decision pairing, equity works mostly through mixed actions: mixed x equity "
        f"is {dec['terms']['mixed+equity']['mean']:.1%} of a hand's variance against "
        f"{dec['terms']['equity']['mean']:.1%} for equity alone ({held('D5a', 'D5b')}). Maniacs "
        f"have the lowest equity share, {equity['maniac']:.1%} against {min(others):.1%}-"
        f"{max(others):.1%} for the others ({held('D6')}).",
        f"Switching mixed actions off does not fix it for free. The ratio rises to "
        f"{ratio('off_mixed')}, but these are different bots: the calling station's SD falls from "
        f"{station['baseline']:.1f} to {station['off_mixed']:.2f} bb per hand, nits' rises from "
        f"{nit['baseline']:.2f} to {nit['off_mixed']:.2f}, the c-bet rate moves "
        f"{change['cbet']:+.3f} and fold to a bet {change['fold_to_bet']:+.3f} "
        f"({held('E1a', 'E1b', 'E2a', 'E2b')}). Noise off changes almost nothing: the ratio stays "
        f"{ratio('off_noise')}, and no stat moves more than "
        f"{max(abs(v) for v in noise_change.values()):.4f} ({held('E3a', 'E3b', 'E3c')}).",
        "This divides the shortfall of copies only. Different bots play differently for reasons "
        "other than their draws, so nothing here says how much of the shortfall in a mix of bots "
        "comes from randomness. And it stays a simulation result: a live table can neither deal "
        "the same cards twice nor share a player's random numbers.",
    ]
    return section(
        "duplicate-randomness",
        "Where six copies' shortfall comes from",
        "Six copies of one bot, with the same cards and rotated seats, would cancel perfectly if "
        "they played alike. To see what stops them, every lineup is played again with each source "
        "of the bots' randomness shared across a deal's tables: the same draws at each table, so "
        "each table plays as it did and only the agreement between tables changes.",
        sources=[RANDOMNESS],
        controls=[control("pairing", "Tables share a draw when they match on", RAND_PAIRINGS)],
        views=views,
        takeaways=takeaways,
        method=f"{dec['lineups']} lineups ({cfg['replicates']} replicates of "
        f"{cfg['lineups_per_replicate']}), {cfg['deals']:,} deals each, "
        f"{cfg['equity_samples']} equity samples, {len(by)} conditions on the same bots and cards: "
        f"{hands:,} hands. Remaining share: 1 / ratio, each condition over its own per-hand "
        "variance. Split: the Sobol-Hoeffding terms from the eight shared conditions, by Moebius "
        "inversion; interactions split equally is the Shapley value. CIs resample lineups. "
        f"Sharing keeps each table's play: median SD within "
        f"{checks['shared_sigma_median_abs_change']:.1%} of the default, stats within "
        f"{checks['shared_behaviour_max_abs_change']:.4f} ({held('B1', 'B2')}), and direct "
        f"within-deal variance matches the formula to "
        f"{checks['direct_over_n_su2_max_abs_log']:.1%} ({held('A3')}). All {len(pred)} "
        "pre-registered predictions (seed 2026) hold, but their thresholds came from development "
        "on seeds 98 and 99: that shows these numbers repeat on a third seed, not that they were "
        "surprising.",
    )


MIS_KINDS = (
    ("family", "Family bot (control)"),
    ("slight_street", "A quarter of the street rule"),
    ("board_blind", "Ignores the board"),
    ("honest_raiser", "Raises only when strong"),
    ("noisy", "Noise at every decision"),
    ("near_eq", "75% equilibrium"),
    ("positional", "Aggression depends on seat"),
    ("street", "Passive, then aggressive"),
    ("equilibrium", "The equilibrium"),
)
MIS_OFF = ("board_blind", "street", "positional", "honest_raiser", "noisy", "near_eq")
MIS_SESSIONS = (
    ("static", "Fixed family bot (control)"),
    ("family_slow", "Adapts inside the family, slowly"),
    ("family_fast", "Adapts inside the family, fast"),
    ("tabular_slow", "Adapts freely, slowly"),
    ("tabular_fast", "Adapts freely, fast"),
)
MIS_VIEWS = (("fixed", "Opponents that do not adapt"), *MIS_SESSIONS)
MIS_HEROES = (
    ("equilibrium", "Equilibrium", ""),
    ("fit_once_br", "Fit once at 300 hands, best response", "c5"),
    ("fit_once_rnr", "Fit once, restricted Nash p = 0.9", "c6"),
    ("refit_br", "Refit every 100 hands, best response", "c1"),
    ("refit_rnr", "Refit, restricted Nash p = 0.9", "c3"),
    ("refit_rnr_low", "Refit, restricted Nash p = 0.5", "c2"),
    ("refit_recent_br", "Refit on the last 300 hands, best response", "c4"),
    ("clairvoyant", "Clairvoyant (reads the bot's strategy)", "c7"),
)
MIS_METHODS = (
    ("br", "Best response to the fitted model", "c1"),
    ("rnr_0.9", "Restricted Nash, p = 0.9", "c3"),
    ("rnr_0.5", "Restricted Nash, p = 0.5", "c2"),
    ("dbr", "Data-biased response", "c4"),
    ("blind_br", "Best response, card-blind model", "c5"),
)


def _off_family_capture(mis: dict[str, Any]) -> tuple[float, float]:
    top = str(max(mis["meta"]["config"]["sizes"]))
    caps = [mis["static"][k]["sizes"][top]["br"]["capture"]["mean"] for k in MIS_OFF]
    return min(caps), max(caps)


def _triple(q: dict[str, float] | None) -> list[float] | None:
    return None if q is None else [q["mean"], q["lower"], q["upper"]]


def misspec(r: dict[str, Any]) -> dict[str, Any]:
    st, ses, pred, cfg = r["static"], r["sessions"], r["predictions"], r["meta"]["config"]
    top = str(max(cfg["sizes"]))
    labels = [lab for _, lab in MIS_KINDS]
    checks = sum(k.startswith("A") for k in pred)
    checks_ok = sum(p["passed"] for k, p in pred.items() if k.startswith("A"))
    passed = sum(p["passed"] for k, p in pred.items() if not k.startswith("A"))

    def held(*keys: str) -> str:
        return "; ".join(f"{k} {'holds' if pred[k]['passed'] else 'does not hold'}" for k in keys)

    def cap(kind: str, m: str = "br", n: str = top) -> float:
        return st[kind]["sizes"][n][m]["capture"]["mean"]

    def gain(kind: str, hero: str, key: str = "gain") -> float:
        return ses[kind]["heroes"][hero][key]["mean"]

    distance = dots_chart(
        "How far each opponent is from the family the hero fits",
        labels,
        axis("KL divergence to the closest family bot (nats per hand)", fmt="num"),
        [dot_series("Mean over bots, 95% CI", "c1", [_triple(st[k]["kl"]) for k, _ in MIS_KINDS])],
        subtitle="Measured from the hero's view (the bot's cards only at showdown) before any "
        "hero plays. Hands to 20:1 evidence against the family at the mean KL: "
        + ", ".join(
            f"{lab.lower()} {st[k]['hands_to_evidence']:.0f}"
            for k, lab in MIS_KINDS
            if k != "family"
        )
        + ". That count is a lower bound: it assumes the alternative is known.",
    )
    capture = dots_chart(
        f"Share of the clairvoyant's gain kept after {int(top):,} hands",
        labels,
        axis("Mean gain / mean oracle gain", fmt="pct"),
        [
            dot_series(
                name,
                color,
                [None if k == "equilibrium" else [cap(k, m)] for k, _ in MIS_KINDS],
            )
            for m, name, color in MIS_METHODS[:3]
        ]
        + [
            dot_series(
                "Best response to the closest family bot (infinite data)",
                "c7",
                [
                    None if k == "equilibrium" else [st[k]["limit"]["br"]["capture"]["mean"]]
                    for k, _ in MIS_KINDS
                ],
            )
        ],
        subtitle="Against the equilibrium nothing can be gained, so there is no share.",
    )
    gains = dots_chart(
        f"Gain over equilibrium play after {int(top):,} hands",
        labels,
        axis("Chips per hand, mean over 5 seeds, 95% CI", fmt="signed"),
        [
            dot_series(name, color, [_triple(st[k]["sizes"][top][m]["gain"]) for k, _ in MIS_KINDS])
            for m, name, color in MIS_METHODS
        ]
        + [
            dot_series(
                "Clairvoyant", "c7", [_triple(st[k]["oracle"]["br"]["gain"]) for k, _ in MIS_KINDS]
            )
        ],
        refs=[ref(0.0, "equilibrium", axis_name="x")],
    )
    sizes = [str(n) for n in cfg["sizes"]]
    loses = dots_chart(
        "How often best response does worse than equilibrium",
        labels,
        axis("Share of 150 bots (Wilson 95% CI)", fmt="pct"),
        [
            dot_series(
                f"{int(n):,} hands",
                color,
                [
                    [
                        st[k]["sizes"][n]["br"]["loses"]["share"],
                        st[k]["sizes"][n]["br"]["loses"]["lower"],
                        st[k]["sizes"][n]["br"]["loses"]["upper"],
                    ]
                    for k, _ in MIS_KINDS
                ],
            )
            for n, color in ((sizes[0], "c2"), (sizes[2], "c4"), (top, "c1"))
        ],
    )
    expl = dots_chart(
        f"Exploitability of the hero's strategy after {int(top):,} hands",
        labels,
        axis("Chips per hand a best response to the hero wins", fmt="num"),
        [
            dot_series(
                name,
                color,
                [_triple(st[k]["sizes"][top][m]["exploitability"]) for k, _ in MIS_KINDS],
            )
            for m, name, color in MIS_METHODS[:3]
        ],
        subtitle="0 for equilibrium play.",
    )
    views = {"fixed": view(distance, capture, gains, loses, expl)}
    hands = cfg["session"]["hands"]
    period = cfg["session"]["period"]
    for kind, title in MIS_SESSIONS:
        e = ses[kind]
        rows = [lab for _, lab, _ in MIS_HEROES]
        overall = dots_chart(
            f"{title}: gain over the equilibrium hero",
            rows,
            axis("Chips per hand, mean over 5 seeds, 95% CI", fmt="signed"),
            [
                dot_series(
                    f"All {hands:,} hands",
                    "c1",
                    [_triple(e["heroes"][h]["gain"]) for h, _, _ in MIS_HEROES],
                ),
                dot_series(
                    "Second half only",
                    "c2",
                    [_triple(e["heroes"][h]["gain_second_half"]) for h, _, _ in MIS_HEROES],
                ),
            ],
            refs=[ref(0.0, "equilibrium", axis_name="x")],
            subtitle=f"The bot moved {e['bot_moved']['mean']:.2f} in total variation per "
            f"decision and ended {e['bot_kl_end']['mean']:.2f} nats per hand from the family. "
            f"Equilibrium's own value fell to {e['eq_value_ratio']['mean']:.0%} of its value "
            "against the fixed bots.",
        )
        curve = line_chart(
            "Gain over the equilibrium hero, block by block",
            axis(f"Hand (end of each {period}-hand block)", fmt="int"),
            axis("Chips per hand, mean over bots and seeds", fmt="signed"),
            [
                series(
                    lab,
                    color,
                    [[period * (i + 1), v] for i, v in enumerate(e["heroes"][h]["curve"])],
                    marker=False,
                )
                for h, lab, color in MIS_HEROES
                if h != "equilibrium"
            ],
            refs=[ref(0.0, "equilibrium")],
        )
        risk = dots_chart(
            "How often each hero ends below equilibrium, and how exploitable it is",
            rows[1:],
            axis("Share of 150 bots (Wilson 95% CI)", fmt="pct"),
            [
                dot_series(
                    "Does worse than equilibrium over the session",
                    "c6",
                    [
                        [
                            e["heroes"][h]["loses"]["share"],
                            e["heroes"][h]["loses"]["lower"],
                            e["heroes"][h]["loses"]["upper"],
                        ]
                        for h, _, _ in MIS_HEROES[1:]
                    ],
                )
            ],
            subtitle="Mean exploitability over the session: "
            + ", ".join(
                f"{lab.split(',')[0].lower()} {e['heroes'][h]['exploitability']['mean']:.2f}"
                for h, lab, _ in MIS_HEROES[1:]
            )
            + " chips per hand.",
        )
        views[kind] = view(overall, curve, risk)
    rule_caps = [cap(k) for k in MIS_OFF]
    limit_gap = max(abs(st[k]["limit"]["br"]["capture"]["mean"] - cap(k)) for k in MIS_OFF)
    vs = {k: st[k]["versus_br"] for k in st if k != "kl_shortfall_spearman"}
    rnr_keep = [vs[k]["rnr_0.9"]["gain_ratio"]["mean"] for k in MIS_OFF]
    rnr_expl = [vs[k]["rnr_0.9"]["exploitability_ratio"]["mean"] for k in MIS_OFF]
    blind = min(st[k]["fit_minus_blind"][top]["mean"] for k in MIS_OFF)
    fast = ses["tabular_fast"]
    d5 = fast["differences"]["rnr_minus_br"]
    lose_fast = [fast["heroes"][h]["loses"]["share"] for h in ("refit_br", "refit_rnr")]
    takeaways = [
        f"The control repeats P2.7's committed numbers exactly on all "
        f"{r['committed_check']['bots']} bots ({held('A1', 'A2')}). Against it, best response "
        f"after {int(top):,} hands keeps {cap('family'):.0%} of what a clairvoyant hero gains.",
        f"Against the six opponents outside the family, best response keeps "
        f"{min(rule_caps):.0%} to {max(rule_caps):.0%} ({held('C1')}). More data would not close "
        "the gap: best response to the closest family bot, the limit of infinite data, keeps the "
        f"same share to within {limit_gap * 100:.1f} points. The cost is the model's shape, not "
        "the sample.",
        f"It still beats equilibrium play against every one of them ({held('C3')}), and does "
        f"worse than equilibrium on at most "
        f"{max(st[k]['sizes'][top]['br']['loses']['share'] for k in MIS_OFF):.0%} of bots "
        f"({held('C4')}). These bots are very exploitable, so a wrong model still finds most "
        "of the gain.",
        f"How far an opponent is from the family predicts the cost only loosely: the rank "
        f"correlation of KL with the shortfall is {st['kl_shortfall_spearman']:.2f} "
        f"({held('C6')}). A quarter of the street rule costs {1 - cap('slight_street'):.0%} "
        f"({held('C2')}).",
        f"The Phase 2 orderings survive: restricted Nash at p = 0.9 keeps "
        f"{min(rnr_keep):.0%} to {max(rnr_keep):.0%} of best response's gain with "
        f"{min(rnr_expl):.0%} to {max(rnr_expl):.0%} of its exploitability ({held('C7', 'C8')}); "
        f"the data-biased response gains less ({held('C9')}); the hand-reading model beats the "
        f"card-blind one against every kind, by {blind:+.3f} on average where it is closest "
        f"({held('C11')}). With 10 hands, the population "
        f"model beats the fit against the near-equilibrium and board-blind bots ({held('C10')}).",
        f"Against the exact equilibrium, every fitted hero loses a little: best response "
        f"{st['equilibrium']['sizes'][top]['br']['gain']['mean']:+.3f} chips per hand after "
        f"{int(top):,} hands ({held('C5a', 'C5b')}).",
        f"Bots that adapt inside the family barely hurt a hero that refits: refit best response "
        f"keeps {ses['family_slow']['capture']['refit_br']['mean']:.0%} and "
        f"{ses['family_fast']['capture']['refit_br']['mean']:.0%} of the clairvoyant's gain "
        f"({held('D1', 'D2')}).",
        f"Bots that adapt freely beat refit best response and restricted Nash p = 0.9 in the "
        f"second half ({held('D3')}). Against the fast one they do worse than equilibrium on "
        f"{min(lose_fast):.0%} or more of bots ({held('D4')}), ending "
        f"{gain('tabular_fast', 'refit_br'):+.3f} and {gain('tabular_fast', 'refit_rnr'):+.3f}. "
        f"Restricted Nash at p = 0.9 does not protect, but the prediction as written fails: its "
        f"gain over best response is {d5['mean']:+.3f} [{d5['lower']:+.3f}, {d5['upper']:+.3f}], "
        f"and the upper end passes the pre-registered +0.02 ({held('D5')}). At p = 0.5 it ends "
        f"near zero, {gain('tabular_fast', 'refit_rnr_low'):+.3f} ({held('D6')}). The gain was "
        f"there: the clairvoyant takes {gain('tabular_fast', 'clairvoyant'):+.2f} "
        f"({held('D7')}).",
        f"Adapting also costs the equilibrium hero: its value against the free adapters falls to "
        f"{ses['tabular_slow']['eq_value_ratio']['mean']:.0%} and "
        f"{fast['eq_value_ratio']['mean']:.0%} of its value against fixed bots ({held('D8')}). "
        f"Against the slow one, refitting on the last 300 hands beats refitting on all "
        f"({held('D9')}).",
        "What this does not show: other games, other adaptation rules, a hero with a more "
        "flexible model or a stop-loss, or anything about people.",
    ]
    return section(
        "misspecification",
        "Exploitation when the model is wrong",
        "Every exploitation result above fits opponents from the five-parameter family the "
        "model assumes. Here the opponents break that assumption: rule bots that use the board, "
        "the street or the seat in ways the family can't express, and bots that model the hero "
        "and adapt, inside the family or freely.",
        sources=[MISSPEC],
        controls=[control("opponents", "Opponents", MIS_VIEWS)],
        views=views,
        takeaways=takeaways,
        method=f"{cfg['seeds']} seeds, {cfg['test_per_archetype'] * 5 * cfg['seeds']} bots per "
        f"kind, the same bots as P2.7; every value exact. Sessions: {hands:,} hands scored by "
        f"exact values per {period}-hand block; random streams shared across heroes per block. "
        f"Misspecification: maximum expected log-likelihood projection onto the family. "
        f"{checks_ok} of {checks} checks hold, and {passed} of {len(pred) - checks} "
        "pre-registered predictions (seed 2026); thresholds came from development on seeds 98 "
        "and 99.",
    )


def build(load) -> dict[str, Any]:
    cfr = load(CFR)
    deep = load(DEEP)
    ex = load(EXPLOIT)
    av = load(AIVAT)
    ph = load(AIVAT_POSTHOC)
    dup = load(DUPLICATE)
    dup_ph = load(DUPLICATE_POSTHOC)
    rnd = load(RANDOMNESS)
    mis = load(MISSPEC)
    head = overview(cfr, deep, ex, av, dup, rnd, mis)
    return {
        "id": "phase2",
        "label": "Phase 2",
        "title": "Equilibrium baselines and exploitation in small games",
        "status": "done",
        "summary": head["summary"],
        "tiles": head["tiles"],
        "provenance": [
            provenance(p, d)
            for p, d in (
                (CFR, cfr),
                (DEEP, deep),
                (EXPLOIT, ex),
                (AIVAT, av),
                (AIVAT_POSTHOC, ph),
                (DUPLICATE, dup),
                (DUPLICATE_POSTHOC, dup_ph),
                (RANDOMNESS, rnd),
                (MISSPEC, mis),
            )
        ],
        "sections": [
            cfr_convergence(cfr),
            sampling(cfr),
            ablation(cfr),
            deep_cfr(deep),
            exploit_by_hands(ex),
            frontier(ex),
            per_bot(ex),
            model_accuracy(ex),
            aivat_leduc(av),
            aivat_values(av),
            aivat_pushfold(av),
            aivat_f1(ph),
            duplicate_dealing(dup),
            duplicate_posthoc(dup_ph),
            randomness(rnd),
            misspec(mis),
        ],
    }

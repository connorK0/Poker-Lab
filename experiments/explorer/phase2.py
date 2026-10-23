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
SOURCES = (CFR, DEEP, EXPLOIT)

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


def overview(cfr: dict[str, Any], deep: dict[str, Any], ex: dict[str, Any]) -> dict[str, Any]:
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


def build(load) -> dict[str, Any]:
    cfr = load(CFR)
    deep = load(DEEP)
    ex = load(EXPLOIT)
    head = overview(cfr, deep, ex)
    return {
        "id": "phase2",
        "label": "Phase 2",
        "title": "Equilibrium baselines and exploitation in small games",
        "status": "done",
        "summary": head["summary"],
        "tiles": head["tiles"],
        "provenance": [provenance(p, d) for p, d in ((CFR, cfr), (DEEP, deep), (EXPLOIT, ex))],
        "sections": [
            cfr_convergence(cfr),
            sampling(cfr),
            ablation(cfr),
            deep_cfr(deep),
            exploit_by_hands(ex),
            frontier(ex),
            per_bot(ex),
            model_accuracy(ex),
        ],
    }

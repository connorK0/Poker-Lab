"""Explorer sections for Phase 1: profiling synthetic no-limit hold'em opponents."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.bot_sanity import LABELS as STAT_LABEL
from experiments.bot_sanity import PERCENT as PERCENT_STATS
from experiments.bot_sanity import REPORT_STATS
from experiments.explorer.spec import (
    ARCHETYPE_COLORS,
    ARCHETYPE_LABELS,
    ARCHETYPE_SHAPES,
    ARCHETYPES,
    axis,
    control,
    dot_series,
    dots_chart,
    heatmap,
    line_chart,
    provenance,
    ref,
    scatter_chart,
    section,
    series,
    sig,
    table_group,
    tile,
    view,
)
from experiments.phase1_report import ACTION_LABEL, CALIBRATION_LABEL, MODEL_LABEL

RESULTS = "experiments/results/phase1/phase1_results.json"
SANITY = "experiments/results/phase1/bot_sanity/bot_sanity.json"
TRUTH = "experiments/results/phase1/bot_sanity/ground_truth.json"
SOURCES = (RESULTS, SANITY, TRUTH)

MODELS = ("lr_shrunk", "gbm_shrunk", "lr_raw", "gbm_raw", "lr_shrunk_nosizing")
MODEL_COLOR = {m: f"c{i}" for i, m in enumerate(MODELS, start=1)}
CAL_MODES = ("none", "global", "by_size")
CAL_COLOR = {"none": "c2", "global": "c1", "by_size": "c3"}
ACTIONS = ("marginal", "situation", "profile", "type_oracle", "param_oracle")
ACTION_COLOR = {
    "marginal": "c4",
    "situation": "c2",
    "profile": "c1",
    "type_oracle": "c3",
    "param_oracle": "c7",
}
REG_METHODS = (
    ("ridge", "Ridge regression"),
    ("gbm", "Boosted trees"),
    ("type_mean", "Classify, then archetype mean"),
)
REG_COLOR = {"ridge": "c1", "gbm": "c2", "type_mean": "c3"}
PARAM_LABEL = {
    "range_width": "Range width",
    "raise_share": "Raise share",
    "three_bet_freq": "3-bet frequency",
    "continue_vs_raise": "Continue vs raise",
    "cbet_freq": "C-bet frequency",
    "fold_to_aggression": "Fold to aggression",
    "bluff_freq": "Bluff frequency",
    "value_aggression": "Value aggression",
    "bet_size_pot": "Bet size (x pot)",
}
HANDS = axis("Hands observed per player", scale="log", fmt="int")
CHANCE_LOG_LOSS = float(np.log(5))


def _mlu(d: dict[str, Any], scale: float = 1.0, clip: tuple[float, float] | None = None):
    lo, hi = d["lower"], d["upper"]
    if clip is not None:
        lo, hi = max(lo, clip[0]), min(hi, clip[1])
    return d["mean"] * scale, lo * scale, hi * scale


def _pooled_reliability(per_seed, mode: str, model: str, n: int):
    rel = [r["reliability"][mode][model][str(n)] for r in per_seed]
    conf = np.array([x["confidence"] for x in rel], dtype=float)
    acc = np.array([x["accuracy"] for x in rel], dtype=float)
    cnt = np.array([x["count"] for x in rel], dtype=float)
    w = cnt.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.nansum(conf * cnt, 0) / w
        a = np.nansum(acc * cnt, 0) / w
    return [(c[i], a[i], int(w[i])) for i in range(len(w)) if w[i] > 0]


def overview(res: dict[str, Any]) -> dict[str, Any]:
    agg = res["aggregate"]
    h2a = agg["hands_to_accuracy"]["lr_shrunk"]
    acc = agg["classification"]["lr_shrunk"]["2000"]["accuracy"]["mean"]
    ap = agg["action_prediction"]["25"]["relative_improvement"]["mean"]
    oracle = agg["oracle"]["lr"]["accuracy"]["mean"]
    hands = res["per_seed"][0]["simulation"]["hands_simulated"]
    return {
        "tiles": [
            tile(
                "Hands to identify a type at 80%",
                f"{h2a['0.8']['geometric_mean']:.0f}",
                f"{h2a['0.8']['min']:.0f}-{h2a['0.8']['max']:.0f} across seeds",
            ),
            tile(
                "Hands to 90%",
                f"{h2a['0.9']['geometric_mean']:.0f}",
                f"{h2a['0.9']['min']:.0f}-{h2a['0.9']['max']:.0f} across seeds",
            ),
            tile(
                "Accuracy after 2,000 hands",
                f"{acc:.1%}",
                f"ceiling {oracle:.1%} (true parameters)",
            ),
            tile(
                "Next-action log-loss cut after 25 hands",
                f"{ap:.1%}",
                "vs a population model that ignores who is acting",
            ),
        ],
        "summary": (
            "Simulated no-limit hold'em bots from five archetypes (nit, TAG, LAG, calling station, "
            "maniac) with known ground truth are profiled from hand histories alone. Every number "
            f"is measured on held-out bots over {agg['n_seeds']} independent seeds "
            f"({hands:,} hands each), with 95% t-intervals across seeds."
        ),
    }


def identify(res: dict[str, Any]) -> dict[str, Any]:
    agg = res["aggregate"]
    sizes = agg["sample_sizes"]
    cls = agg["classification"]
    oracle = agg["oracle"]["lr"]
    metrics = {
        "accuracy": ("Accuracy", axis("Accuracy", fmt="pct", hi=1), (0.0, 1.0)),
        "log_loss": ("Log-loss (nats, lower is better)", axis("Log-loss", lo=0), (0.0, 9.0)),
        "brier": ("Brier score (lower is better)", axis("Brier score", lo=0), (0.0, 2.0)),
    }
    chance = {
        "accuracy": "Chance is 20% and the shuffled-label control scores "
        f"{agg['permutation_control']['mean']:.1%}.",
        "log_loss": f"Chance (uniform over five types) is {CHANCE_LOG_LOSS:.3f}.",
        "brier": "",
    }
    h2a = agg["hands_to_accuracy"]
    thresholds = ("0.8", "0.9", "0.95")
    rows = [m for m in MODELS if m in h2a]
    reach = dots_chart(
        "Hands needed to reach a target accuracy",
        [MODEL_LABEL[m] for m in rows],
        axis("Hands observed", scale="log", fmt="int"),
        [
            dot_series(
                f"{float(t):.0%} accuracy",
                f"c{i}",
                [
                    (h2a[m][t]["geometric_mean"], h2a[m][t]["min"], h2a[m][t]["max"])
                    if h2a[m][t]["reached_in_seeds"]
                    else None
                    for m in rows
                ],
            )
            for i, t in enumerate(thresholds, start=1)
        ],
        subtitle="Geometric mean over seeds; whiskers span the fastest and slowest seed. "
        "Interpolated on log(hands); a seed that never reaches the target within 2,000 hands "
        "is left out.",
    )
    views = {}
    for key, (title, yax, clip) in metrics.items():
        refs = []
        if key in ("accuracy", "log_loss"):
            refs = [ref(oracle[key]["mean"], "Oracle (true parameters)")]
        chart = line_chart(
            title,
            HANDS,
            yax,
            [
                series(
                    MODEL_LABEL[m],
                    MODEL_COLOR[m],
                    [(n, *_mlu(cls[m][str(n)][key], clip=clip)) for n in sizes],
                    dash=m == "lr_shrunk_nosizing",
                )
                for m in MODELS
            ],
            subtitle=f"Held-out bots; bands are 95% t-intervals across seeds. {chance[key]}",
            refs=refs,
        )
        views[key] = view(chart, reach)
    h = h2a["lr_shrunk"]
    nos = h2a["lr_shrunk_nosizing"]
    return section(
        "identify",
        "How many hands to identify a player's type",
        "A classifier sees a bot's stats after n hands and guesses its archetype. "
        "Pick a metric; click legend entries to hide or show models.",
        sources=[RESULTS],
        controls=[
            control(
                "metric",
                "Metric",
                [("accuracy", "Accuracy"), ("log_loss", "Log-loss"), ("brier", "Brier")],
            )
        ],
        views=views,
        takeaways=[
            f"The primary model (logistic regression on shrunk stats) reaches 80% after about "
            f"{h['0.8']['geometric_mean']:.0f} hands and 90% after about "
            f"{h['0.9']['geometric_mean']:.0f}.",
            f"Without bet-sizing features (post hoc), 80% takes about "
            f"{nos['0.8']['geometric_mean']:.0f} hands and 90% about "
            f"{nos['0.9']['geometric_mean']:.0f}: bot sizing is a strong tell at small samples.",
            f"Shuffling the labels drops accuracy to "
            f"{agg['permutation_control']['mean']:.1%}, i.e. chance (20%): no leakage.",
            "Boosted trees are never better than logistic regression here and have worse "
            "log-loss at every sample size.",
        ],
        method=(
            "Training and test bots play in separate leagues. Each test bot's first n hands "
            "give one feature vector (stats with empirical-Bayes shrinkage). Accuracy and "
            "log-loss average over test bots, then a t-interval is taken across the seeds."
        ),
    )


def calibration(res: dict[str, Any]) -> dict[str, Any]:
    agg = res["aggregate"]
    sizes = agg["sample_sizes"]
    per_seed = res["per_seed"]
    shown = sorted(int(k) for k in per_seed[0]["reliability"]["none"]["lr_shrunk"])
    ece_key = {"none": "ece_uncal", "global": "ece", "by_size": "ece_by_size"}
    views = {}
    for m in MODELS:
        cls = agg["classification"][m]
        ece = line_chart(
            "Expected calibration error vs hands",
            HANDS,
            axis("ECE (lower is better)", lo=0),
            [
                series(
                    CALIBRATION_LABEL[mode],
                    CAL_COLOR[mode],
                    [(n, *_mlu(cls[str(n)][ece_key[mode]], clip=(0, 1))) for n in sizes],
                )
                for mode in CAL_MODES
            ],
            subtitle=MODEL_LABEL[m],
        )
        temps = agg["temperatures_by_size"][m]
        temp = line_chart(
            "Fitted temperature per sample size",
            HANDS,
            axis("Temperature T (probabilities ∝ p^(1/T))"),
            [series("Per-size temperature", "c3", [(n, *_mlu(temps[str(n)])) for n in sizes])],
            subtitle="T > 1 softens overconfident probabilities; "
            "T < 1 sharpens underconfident ones.",
            refs=[
                ref(agg["temperatures"][m]["mean"], "One global temperature"),
                ref(1.0, "No correction"),
            ],
        )
        for n in shown:
            rel_series = [
                series(
                    CALIBRATION_LABEL[mode],
                    CAL_COLOR[mode],
                    [(c, a) for c, a, _ in _pooled_reliability(per_seed, mode, m, n)],
                )
                for mode in CAL_MODES
            ]
            rel_series.append(
                series("Perfect calibration", "muted", [(0, 0), (1, 1)], dash=True, marker=False)
            )
            rel = line_chart(
                f"Reliability diagram after {n:,} hands",
                axis("Predicted probability of the top class", fmt="pct", lo=0, hi=1),
                axis("Observed accuracy", fmt="pct", lo=0, hi=1),
                rel_series,
                subtitle="Bins of 0.1, pooled over seeds. Above the diagonal = underconfident.",
            )
            views[f"{m}|{n}"] = view(rel, ece, temp)
    return section(
        "calibration",
        "Are the type probabilities trustworthy?",
        "A calibrated classifier that says 80% is right 80% of the time. Compare no correction, "
        "one global temperature, and a temperature per sample size.",
        sources=[RESULTS],
        controls=[
            control("model", "Model", [(m, MODEL_LABEL[m]) for m in MODELS]),
            control("hands", "Hands observed", [(str(n), f"{n:,}") for n in shown]),
        ],
        views=views,
        takeaways=[
            "The logistic model needs no global correction (T close to 1), but its "
            "miscalibration depends on sample size: overconfident at 25 hands, underconfident "
            "from 200 hands up.",
            "A temperature per sample size (post hoc) roughly halves the calibration error at "
            "200+ hands. Boosted trees are overconfident and gain most from scaling.",
        ],
        method="Temperatures are fit on training bots by minimizing log-loss, never on test bots.",
        post_hoc=True,
    )


def shrinkage(res: dict[str, Any]) -> dict[str, Any]:
    agg = res["aggregate"]
    sizes = agg["sample_sizes"]
    abl = agg["shrinkage_ablation"]
    names = {"lr": ("Logistic regression", "c1"), "gbm": ("Boosted trees", "c2")}
    views = {}
    for key, (title, yax, better) in {
        "delta_log_loss": (
            "Log-loss: shrunk minus raw",
            axis("Δ log-loss (negative = shrinkage helps)", fmt="signed"),
            "below",
        ),
        "delta_accuracy": (
            "Accuracy: shrunk minus raw",
            axis("Δ accuracy (positive = shrinkage helps)", fmt="pct"),
            "above",
        ),
    }.items():
        views[key] = view(
            line_chart(
                title,
                HANDS,
                yax,
                [
                    series(
                        names[k][0], names[k][1], [(n, *_mlu(abl[k][str(n)][key])) for n in sizes]
                    )
                    for k in ("lr", "gbm")
                ],
                subtitle="Paired within seed (same players, same split). "
                f"Shrinkage helps {better} 0.",
                refs=[ref(0.0, "No effect")],
            )
        )
    lr25 = abl["lr"]["25"]["delta_log_loss"]["mean"]
    lr2k = abl["lr"]["2000"]["delta_log_loss"]["mean"]
    return section(
        "shrinkage",
        "Shrinkage ablation: empirical-Bayes stats vs raw stats",
        "Raw stats from few hands are noisy (a 0-for-2 c-bet stat says 0%). Empirical-Bayes "
        "shrinkage pulls each stat toward the population prior by an amount set by its "
        "sample size.",
        sources=[RESULTS],
        controls=[
            control(
                "metric",
                "Metric",
                [("delta_log_loss", "Δ log-loss"), ("delta_accuracy", "Δ accuracy")],
            )
        ],
        views=views,
        takeaways=[
            f"Shrinkage clearly helps the linear model: log-loss falls by {-lr25:.3f} at 25 hands "
            f"and {-lr2k:.3f} at 2,000.",
            "For boosted trees the effect is indistinguishable from zero almost everywhere.",
            "The prediction that shrinkage matters most at small samples held only for log-loss at "
            "25 hands; for the linear model the benefit grows with hands.",
        ],
    )


def confusion(res: dict[str, Any]) -> dict[str, Any]:
    agg = res["aggregate"]
    sizes = agg["sample_sizes"]
    labels = [ARCHETYPE_LABELS[a] for a in ARCHETYPES]
    views = {}
    for m in MODELS:
        cls = agg["classification"][m]
        recall = line_chart(
            "Recall per archetype vs hands",
            HANDS,
            axis("Recall (share of that type identified)", fmt="pct", lo=0, hi=1),
            [
                series(
                    ARCHETYPE_LABELS[a],
                    ARCHETYPE_COLORS[a],
                    [(n, *_mlu(cls[str(n)]["recall"][a], clip=(0, 1))) for n in sizes],
                    shape=ARCHETYPE_SHAPES[a],
                )
                for a in ARCHETYPES
            ],
            subtitle=MODEL_LABEL[m],
        )
        for n in sizes:
            counts = np.array(cls[str(n)]["confusion_pooled"], dtype=float)
            rates = counts / counts.sum(axis=1, keepdims=True)
            hm = heatmap(
                f"Confusion matrix after {n:,} hands",
                labels,
                labels,
                rates,
                counts=counts,
                row_label="True type",
                col_label="Predicted type",
                vmax=1.0,
                subtitle="Row-normalized: each row shows where bots of that true type were "
                "classified (pooled over seeds).",
            )
            views[f"{m}|{n}"] = view(hm, recall)
    return section(
        "confusion",
        "Which types get confused, and until when",
        "Each cell is the share of bots of the row's true type that the model labeled as the "
        "column's type. Hover a cell for the count.",
        sources=[RESULTS],
        controls=[
            control("model", "Model", [(m, MODEL_LABEL[m]) for m in MODELS]),
            control("hands", "Hands observed", [(str(n), f"{n:,}") for n in sizes]),
        ],
        views=views,
        takeaways=[
            "Errors are between neighbors on the tight-loose axis: nit-TAG, TAG-LAG, LAG-maniac.",
            "Calling stations are recognized almost immediately (98% after 25 hands).",
            "At 2,000 hands the remaining errors are where the archetype distributions overlap "
            "by design.",
        ],
    )


def regression(res: dict[str, Any]) -> dict[str, Any]:
    agg = res["aggregate"]
    sizes = agg["sample_sizes"]
    reg = agg["regression"]
    params = reg["param_names"]
    baselines = {
        "": ("population", "R² vs predicting the population mean"),
        "_within_type": ("within", "R² within archetype (vs knowing the true type)"),
    }
    views = {}
    for suffix, (bkey, blabel) in baselines.items():
        trend = line_chart(
            "Mean R² over the nine tendencies",
            HANDS,
            axis(blabel, fmt="num"),
            [
                series(
                    label,
                    REG_COLOR[k],
                    [(n, *_mlu(reg[k + suffix][str(n)]["_mean_over_params"])) for n in sizes],
                )
                for k, label in REG_METHODS
            ],
            refs=[ref(0.0, "No better than the baseline")],
        )
        for n in sizes:
            per_param = dots_chart(
                f"Out-of-sample R² per tendency after {n:,} hands",
                [PARAM_LABEL.get(p, p) for p in params],
                axis(blabel),
                [
                    dot_series(
                        label, REG_COLOR[k], [_mlu(reg[k + suffix][str(n)][p]) for p in params]
                    )
                    for k, label in REG_METHODS
                ],
                refs=[ref(0.0, "Baseline", axis_name="x")],
                subtitle="Whiskers: 95% t-interval across seeds.",
            )
            views[f"{bkey}|{n}"] = view(per_param, trend)
    within = reg["gbm_within_type"]["2000"]
    return section(
        "regression",
        "Recovering the continuous tendencies (a weak result)",
        "Beyond the archetype label, can the profile recover each bot's nine underlying "
        "tendencies? Switch the baseline to see how much is just knowing the type.",
        sources=[RESULTS],
        controls=[
            control(
                "baseline",
                "Baseline",
                [("population", "Population mean"), ("within", "True archetype (post hoc)")],
            ),
            control("hands", "Hands observed", [(str(n), f"{n:,}") for n in sizes]),
        ],
        views=views,
        takeaways=[
            "Against the population mean, regression looks strong, but most of that is which "
            "archetype a bot belongs to.",
            f"Within archetype, after 2,000 hands: range width "
            f"{within['range_width']['mean']:.2f} and bet size "
            f"{within['bet_size_pot']['mean']:.2f} "
            f"are recovered, while bluff frequency ({within['bluff_freq']['mean']:.2f}) and "
            f"fold to aggression ({within['fold_to_aggression']['mean']:.2f}) are not.",
            "Those poorly recovered tendencies are the ones an exploit would care about most.",
        ],
        post_hoc=True,
    )


def action(res: dict[str, Any]) -> dict[str, Any]:
    agg = res["aggregate"]
    ap = agg["action_prediction"]
    checkpoints = sorted(ap, key=int)
    cats = [f"{int(c):,}" for c in checkpoints]
    xax = {
        **axis("Hands of history before the 100-hand window"),
        "scale": "point",
        "categories": cats,
    }
    views = {}
    for key, title, yax in (
        ("log_loss", "Next-action log-loss", axis("Log-loss (lower is better)")),
        ("accuracy", "Next-action accuracy", axis("Accuracy", fmt="pct")),
    ):
        main = line_chart(
            title,
            xax,
            yax,
            [
                series(
                    ACTION_LABEL[a],
                    ACTION_COLOR[a],
                    [(i, *_mlu(ap[c][key][a])) for i, c in enumerate(checkpoints)],
                    dash=a.endswith("oracle"),
                )
                for a in ACTIONS
            ],
            subtitle="Fold / check-call / bet-raise for every decision of held-out bots.",
        )
        gap = line_chart(
            "How much of the gap to the oracle the profile closes",
            xax,
            axis("Share", fmt="pct"),
            [
                series(
                    "Share of gap to the true-parameter oracle closed",
                    "c1",
                    [(i, *_mlu(ap[c]["oracle_gap_closed"])) for i, c in enumerate(checkpoints)],
                ),
                series(
                    "Log-loss reduction vs population baseline",
                    "c2",
                    [(i, *_mlu(ap[c]["relative_improvement"])) for i, c in enumerate(checkpoints)],
                ),
            ],
        )
        views[key] = view(main, gap)
    a25 = ap["25"]
    return section(
        "action",
        "Predicting a player's next action",
        "A model predicts each decision from the situation alone (population baseline) or the "
        "situation plus the player's profile built only from earlier hands.",
        sources=[RESULTS],
        controls=[
            control("metric", "Metric", [("log_loss", "Log-loss"), ("accuracy", "Accuracy")])
        ],
        views=views,
        takeaways=[
            f"25 hands of history cut log-loss by {a25['relative_improvement']['mean']:.1%} and "
            f"close {a25['oracle_gap_closed']['mean']:.0%} of the gap to an oracle that knows the "
            "bot's true parameters.",
            "From 100 hands on, the profile model beats an oracle that knows only the true "
            "archetype.",
            "With zero hands the profile carries no information, yet the model is 0.3% better than "
            "the baseline: a small modelling artifact (two separately fitted models), reported.",
        ],
        method="This is the metric Phase 5 reuses on real home-game hands.",
    )


def bots(sanity: dict[str, Any]) -> dict[str, Any]:
    per = sanity["per_archetype"]
    rows = [ARCHETYPE_LABELS[a] for a in ARCHETYPES]
    n_bots = sum(per[a]["vpip"]["n"] for a in ARCHETYPES)
    views = {}
    for stat in REPORT_STATS:
        cells = [per[a][stat] for a in ARCHETYPES]
        ranges = [c["intended"] for c in cells]
        views[stat] = view(
            dots_chart(
                STAT_LABEL[stat],
                rows,
                axis(STAT_LABEL[stat], fmt="num"),
                [
                    dot_series(
                        "Measured (mean over bots)",
                        "c1",
                        [(c["estimate"], c["lower"], c["upper"]) for c in cells],
                    )
                ],
                ranges=ranges if any(ranges) else None,
                range_label="Intended range (written before any simulation)",
                subtitle=f"{n_bots} bots in 6-max games, {sanity['hands_simulated']:,} hands "
                f"(about {sanity['hands_simulated'] * 6 // n_bots:,} per bot); whiskers: 95% "
                "t-interval over bots.",
            )
        )
    drivers = sanity["driver_correlations"]
    stat_name = {**STAT_LABEL, "pfr_over_vpip": "PFR / VPIP", "aggr_share": "Aggressive share"}
    driver_rows = [
        f"{PARAM_LABEL.get(d['param'], d['param'])} → {stat_name.get(d['stat'], d['stat'])}"
        for d in drivers
    ]
    driver_chart = dots_chart(
        "Does each tendency move the stat it should?",
        driver_rows,
        axis("Spearman correlation across bots", lo=-1, hi=1),
        [
            dot_series("All bots", "c1", [(d["spearman_all"],) for d in drivers]),
            dot_series(
                "Within archetype (mean of five)",
                "c2",
                [(float(np.mean(list(d["spearman_within_archetype"].values()))),) for d in drivers],
            ),
        ],
        refs=[ref(0.0, "None", axis_name="x")],
    )
    for v in views.values():
        v["charts"].append(driver_chart)
    s = sanity["summary"]
    return section(
        "bots",
        "Do the bots play like their archetypes?",
        "Before profiling anything, the simulated bots must actually play like the archetypes "
        "they are named after. Pick a HUD stat to compare each archetype with its intended range.",
        sources=[SANITY],
        controls=[control("stat", "Stat", [(st, STAT_LABEL[st]) for st in REPORT_STATS])],
        views=views,
        takeaways=[
            f"{s['cells_in_intended_range']} of {s['cells_with_intended_range']} cells land in "
            f"their pre-registered range and {s['orderings_holding']} of {s['orderings_total']} "
            "intended orderings hold.",
            "The one miss: nits fold to c-bets slightly less often than intended.",
            "WTSD is far above typical human values (44-70% vs 25-35%), a known realism gap.",
            f"Percent stats ({len(PERCENT_STATS)} of {len(REPORT_STATS)}) count per opportunity, "
            "e.g. c-bet % per flop where the bot was the preflop aggressor.",
        ],
    )


def population(truth: dict[str, Any]) -> dict[str, Any]:
    params = truth["param_names"]
    players = truth["players"]
    table: dict[str, list[Any]] = {"archetype": [p["archetype"] for p in players]}
    table["label"] = [f"{ARCHETYPE_LABELS[p['archetype']]} {p['player_id']}" for p in players]
    for p in params:
        table[p] = [sig(pl[p], 3) for pl in players]
    pairs = [(x, y) for x in params for y in params]
    views = {}
    for x, y in pairs:
        views[f"{x}|{y}"] = view(
            scatter_chart(
                f"{PARAM_LABEL[y]} vs {PARAM_LABEL[x]}",
                axis(PARAM_LABEL[x]),
                axis(PARAM_LABEL[y]),
                [
                    table_group(
                        ARCHETYPE_LABELS[a],
                        ARCHETYPE_COLORS[a],
                        "bots",
                        x,
                        y,
                        where=("archetype", a),
                        shape=ARCHETYPE_SHAPES[a],
                    )
                    for a in ARCHETYPES
                ],
                subtitle=f"{len(players)} bots drawn from the five archetype distributions.",
            )
        )
    opts = [(p, PARAM_LABEL[p]) for p in params]
    return section(
        "population",
        "The bot population: archetypes as distributions",
        "Each archetype is a distribution over nine tendencies, so bots vary within a type and "
        "neighboring types overlap. Choose two tendencies to plot.",
        sources=[TRUTH],
        controls=[control("x", "Horizontal", opts), control("y", "Vertical", opts)],
        views=views,
        defaults={"y": "bluff_freq"},
        tables={"bots": table},
        takeaways=[
            "Overlap between neighboring archetypes is deliberate: it is why even a perfect "
            "classifier on the true parameters tops out near 99%, not 100%.",
        ],
    )


def build(load) -> dict[str, Any]:
    res = load(RESULTS)
    sanity = load(SANITY)
    truth = load(TRUTH)
    head = overview(res)
    sections = [
        identify(res),
        calibration(res),
        shrinkage(res),
        confusion(res),
        regression(res),
        action(res),
        bots(sanity),
        population(truth),
    ]
    for s in sections:
        s["takeaways"] = [t for t in s["takeaways"] if t]
    return {
        "id": "phase1",
        "label": "Phase 1",
        "title": "Opponent modeling on synthetic players",
        "status": "done",
        "summary": head["summary"],
        "tiles": head["tiles"],
        "provenance": [provenance(RESULTS, res), provenance(SANITY, sanity)],
        "sections": sections,
    }

"""Aggregate Phase 1 replicates (seeds): across-seed means with 95% t-intervals, figures, and
a markdown summary. Per-seed values are kept in the JSON for full transparency.

Intervals are Student-t across seeds (each seed is an independent replicate of the whole
pipeline). For bounded metrics (accuracy, recall, ECE, log-loss >= 0) the *displayed* interval
is clipped to the natural range; the JSON keeps the unclipped values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiments.plotting import band, legend, line, new_figure, save_both, style_axes, title
from poker_lab.bots import ARCHETYPES, PARAM_NAMES
from poker_lab.evaluation.intervals import t_interval

MODEL_LABEL = {
    "lr_shrunk": "Logistic, shrunk stats",
    "gbm_shrunk": "Boosted trees, shrunk stats",
    "lr_raw": "Logistic, raw stats",
    "gbm_raw": "Boosted trees, raw stats",
}
ARCH_LABEL = {
    "nit": "Nit",
    "tag": "TAG",
    "lag": "LAG",
    "calling_station": "Station",
    "maniac": "Maniac",
}
ACTION_LABEL = {
    "marginal": "Population marginal",
    "situation": "Population (situation only)",
    "profile": "Situation + profile",
    "type_oracle": "Oracle: true archetype",
    "param_oracle": "Oracle: true parameters",
}
THRESHOLDS = (0.8, 0.9, 0.95)


def ci(values) -> dict[str, Any]:
    vals = [float(v) for v in values]
    iv = t_interval(vals)
    return {"mean": iv.estimate, "lower": iv.lower, "upper": iv.upper, "per_seed": vals}


def crossing(sizes: list[int], acc: list[float], target: float) -> float:
    """Hands needed for accuracy to reach ``target`` (log-linear interpolation; NaN if never)."""
    for i, a in enumerate(acc):
        if a >= target:
            if i == 0:
                return float(sizes[0])
            x0, x1 = np.log(sizes[i - 1]), np.log(sizes[i])
            frac = (target - acc[i - 1]) / (a - acc[i - 1])
            return float(np.exp(x0 + frac * (x1 - x0)))
    return float("nan")


def aggregate(per_seed: list[dict[str, Any]], cfg: dict) -> dict[str, Any]:
    first = per_seed[0]
    primary = cfg["primary_model"]
    sizes = [int(n) for n in first["classification"][primary]]
    models = list(first["classification"])
    agg: dict[str, Any] = {
        "sample_sizes": sizes,
        "n_seeds": len(per_seed),
        "primary_model": primary,
    }

    cls: dict[str, Any] = {}
    for m in models:
        cls[m] = {}
        for n in sizes:
            rows = [r["classification"][m][str(n)] for r in per_seed]
            cls[m][str(n)] = {
                k: ci([row[k] for row in rows])
                for k in ("accuracy", "log_loss", "log_loss_uncal", "ece", "ece_uncal", "brier")
            }
            cm = np.sum([row["confusion"] for row in rows], axis=0)
            cls[m][str(n)]["confusion_pooled"] = cm.tolist()
            cls[m][str(n)]["recall"] = {
                ARCHETYPES[k].value: ci([row["recall"][k] for row in rows])
                for k in range(len(ARCHETYPES))
            }
    agg["classification"] = cls
    agg["oracle"] = {
        m: {k: ci([r["oracle"][m][k] for r in per_seed]) for k in ("accuracy", "log_loss")}
        for m in ("lr", "gbm")
    }
    agg["permutation_control"] = ci([r["permutation_control"]["accuracy"] for r in per_seed])
    agg["temperatures"] = {m: ci([r["temperatures"][m] for r in per_seed]) for m in models}

    # Shrinkage ablation: paired per seed (same data, same split), shrunk minus raw.
    ablation = {}
    for model in ("lr", "gbm"):
        ablation[model] = {}
        for n in sizes:
            s = [r["classification"][f"{model}_shrunk"][str(n)] for r in per_seed]
            w = [r["classification"][f"{model}_raw"][str(n)] for r in per_seed]
            pairs = list(zip(s, w, strict=True))
            ablation[model][str(n)] = {
                "delta_accuracy": ci([a["accuracy"] - b["accuracy"] for a, b in pairs]),
                "delta_log_loss": ci([a["log_loss"] - b["log_loss"] for a, b in pairs]),
            }
    agg["shrinkage_ablation"] = ablation

    hands_needed = {}
    for m in ("lr_shrunk", "gbm_shrunk"):
        hands_needed[m] = {}
        for target in THRESHOLDS:
            per = [
                crossing(sizes, [r["classification"][m][str(n)]["accuracy"] for n in sizes], target)
                for r in per_seed
            ]
            finite = [p for p in per if np.isfinite(p)]
            hands_needed[m][str(target)] = {
                "per_seed": per,
                "reached_in_seeds": len(finite),
                "geometric_mean": float(np.exp(np.mean(np.log(finite)))) if finite else None,
                "min": min(finite) if finite else None,
                "max": max(finite) if finite else None,
            }
    agg["hands_to_accuracy"] = hands_needed

    reg: dict[str, Any] = {"param_names": list(PARAM_NAMES)}
    for k in ("gbm", "ridge", "type_mean"):
        reg[k] = {}
        for n in sizes:
            arr = np.array([r["regression"][k][str(n)] for r in per_seed])  # seeds x params
            reg[k][str(n)] = {p: ci(arr[:, j]) for j, p in enumerate(PARAM_NAMES)}
            reg[k][str(n)]["_mean_over_params"] = ci(arr.mean(axis=1))
    agg["regression"] = reg

    act: dict[str, Any] = {}
    for c in first["action"]["checkpoints"]:
        rows = [r["action"]["checkpoints"][c] for r in per_seed]
        act[c] = {
            "log_loss": {k: ci([row["log_loss"][k] for row in rows]) for k in ACTION_LABEL},
            "accuracy": {k: ci([row["accuracy"][k] for row in rows]) for k in ACTION_LABEL},
            "delta_vs_situation": ci([row["delta_vs_situation"] for row in rows]),
            "relative_improvement": ci(
                [row["delta_vs_situation"] / row["log_loss"]["situation"] for row in rows]
            ),
            "oracle_gap_closed": ci(
                [
                    (row["log_loss"]["situation"] - row["log_loss"]["profile"])
                    / (row["log_loss"]["situation"] - row["log_loss"]["param_oracle"])
                    for row in rows
                ]
            ),
            "n_decisions_per_seed": [row["n_decisions"] for row in rows],
        }
    agg["action_prediction"] = act
    return agg


# ------------------------------------------------------------------ figures
def _sizes_axis(ax, sizes):
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Hands observed per player")


def _end_label(ax, theme, x, y, text):
    ax.annotate(
        text,
        (x, y),
        xytext=(7, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=theme["ink2"],
    )


def _bounds(d: list[dict], lo: float | None = None, hi: float | None = None, scale: float = 1.0):
    lower = np.array([x["lower"] for x in d], dtype=float) * scale
    upper = np.array([x["upper"] for x in d], dtype=float) * scale
    if lo is not None:
        lower, upper = np.maximum(lower, lo), np.maximum(upper, lo)
    if hi is not None:
        lower, upper = np.minimum(lower, hi), np.minimum(upper, hi)
    return lower, upper


def _reference(ax, theme, y, text, x):
    ax.axhline(y, color=theme["muted"], linewidth=1)
    ax.annotate(
        text, (x, y), xytext=(0, 4), textcoords="offset points", fontsize=8, color=theme["ink2"]
    )


def plot_hands_to_identify(agg, out_dir: Path) -> None:
    sizes = agg["sample_sizes"]
    primary = agg["primary_model"]
    other = "gbm_shrunk" if primary == "lr_shrunk" else "lr_shrunk"
    series = [(primary, 0), (other, 1)]
    oracle_model = primary.split("_")[0]

    def draw(theme):
        fig, (ax1, ax2) = new_figure(theme, 1, 2, figsize=(11.5, 4.4))
        for ax, metric, ylabel, lim in (
            (ax1, "accuracy", "Accuracy (5 archetypes)", (0, 1)),
            (ax2, "log_loss", "Log-loss (nats)", (0, None)),
        ):
            style_axes(ax, theme)
            for m, slot in series:
                d = [agg["classification"][m][str(n)][metric] for n in sizes]
                band(ax, sizes, *_bounds(d, *lim), theme["series"][slot])
                line(
                    ax,
                    theme,
                    sizes,
                    [x["mean"] for x in d],
                    theme["series"][slot],
                    label=MODEL_LABEL[m],
                )
            oracle = agg["oracle"][oracle_model][metric]["mean"]
            _reference(ax, theme, oracle, f"Oracle (true parameters): {oracle:.2f}", sizes[0])
            chance = 0.2 if metric == "accuracy" else float(np.log(5))
            _reference(ax, theme, chance, f"Chance: {chance:.2f}", sizes[0])
            _sizes_axis(ax, sizes)
            ax.set_ylabel(ylabel)
        ax1.set_ylim(0, 1.04)
        ax2.set_ylim(bottom=0)
        legend(ax1, theme, loc="lower right")
        title(
            fig,
            theme,
            "Hands needed to identify a player's archetype",
            f"Held-out players; mean over {agg['n_seeds']} seeds, shaded 95% t-interval "
            "across seeds",
        )
        return fig

    save_both(draw, out_dir, "hands_to_identify")


def plot_shrinkage_ablation(agg, out_dir: Path) -> None:
    sizes = agg["sample_sizes"]

    def draw(theme):
        fig, (ax1, ax2) = new_figure(theme, 1, 2, figsize=(11.5, 4.2))
        for ax, metric, ylabel, scale in (
            (ax1, "delta_accuracy", "Accuracy points: shrunk − raw", 100.0),
            (ax2, "delta_log_loss", "Log-loss: shrunk − raw (below 0 = shrinkage helps)", 1.0),
        ):
            style_axes(ax, theme)
            ax.axhline(0, color=theme["muted"], linewidth=1)
            for slot, model in enumerate(("lr", "gbm")):
                d = [agg["shrinkage_ablation"][model][str(n)][metric] for n in sizes]
                band(ax, sizes, *_bounds(d, scale=scale), theme["series"][slot])
                line(
                    ax,
                    theme,
                    sizes,
                    [scale * x["mean"] for x in d],
                    theme["series"][slot],
                    label="Logistic" if model == "lr" else "Boosted trees",
                )
            _sizes_axis(ax, sizes)
            ax.set_ylabel(ylabel)
        legend(ax1, theme, loc="upper right")
        title(
            fig,
            theme,
            "Shrinkage ablation: empirical-Bayes stats vs raw stats",
            "Paired within seed (same players and windows); 95% t-interval across seeds",
        )
        return fig

    save_both(draw, out_dir, "shrinkage_ablation")


def plot_calibration(per_seed, agg, out_dir: Path, sizes_shown) -> None:
    sizes = agg["sample_sizes"]
    model = agg["primary_model"]

    def pooled(kind, n):
        conf = np.array([r[kind][model][str(n)]["confidence"] for r in per_seed], dtype=float)
        acc = np.array([r[kind][model][str(n)]["accuracy"] for r in per_seed], dtype=float)
        cnt = np.array([r[kind][model][str(n)]["count"] for r in per_seed], dtype=float)
        w = cnt.sum(axis=0)
        with np.errstate(invalid="ignore"):
            return np.nansum(conf * cnt, 0) / w, np.nansum(acc * cnt, 0) / w, w

    def draw(theme):
        fig, axes = new_figure(theme, 1, len(sizes_shown) + 1, figsize=(14, 3.9))
        for ax, n in zip(axes[:-1], sizes_shown, strict=True):
            style_axes(ax, theme, grid_axis="both")
            ax.plot([0, 1], [0, 1], color=theme["muted"], linewidth=1)
            for slot, kind, label in (
                (1, "reliability_uncal", "Uncalibrated"),
                (0, "reliability", "Temperature-scaled"),
            ):
                c, a, w = pooled(kind, n)
                keep = w > 20
                line(ax, theme, c[keep], a[keep], theme["series"][slot], label=label)
            ax.set_xlim(0.15, 1.0)
            ax.set_ylim(0.0, 1.02)
            ax.set_title(f"{n} hands", fontsize=10, loc="left", color=theme["ink"])
            ax.set_xlabel("Predicted confidence (top class)")
        axes[0].set_ylabel("Observed accuracy")
        legend(axes[0], theme, loc="upper left")
        ax = axes[-1]
        style_axes(ax, theme)
        for slot, key, label in (
            (1, "ece_uncal", "Uncalibrated"),
            (0, "ece", "Temperature-scaled"),
        ):
            d = [agg["classification"][model][str(n)][key] for n in sizes]
            band(ax, sizes, *_bounds(d, lo=0.0), theme["series"][slot])
            line(ax, theme, sizes, [x["mean"] for x in d], theme["series"][slot], label=label)
        _sizes_axis(ax, sizes)
        ax.set_title("Expected calibration error", fontsize=10, loc="left", color=theme["ink"])
        ax.set_ylim(bottom=0)
        title(
            fig,
            theme,
            f"Calibration of archetype probabilities ({MODEL_LABEL[model].lower()})",
            "Reliability diagrams pool all seeds (bins with >20 windows); diagonal = "
            "perfect calibration",
            top=0.88,
        )
        return fig

    save_both(draw, out_dir, "calibration")


def plot_confusion(agg, out_dir: Path, sizes_shown) -> None:
    names = [ARCH_LABEL[a.value] for a in ARCHETYPES]
    model = agg["primary_model"]

    def draw(theme):
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("seq", [theme["surface"], *theme["seq"][2:]])
        fig, axes = new_figure(theme, 1, len(sizes_shown), figsize=(4.0 * len(sizes_shown), 4.2))
        for ax, n in zip(axes, sizes_shown, strict=True):
            cm = np.array(agg["classification"][model][str(n)]["confusion_pooled"], float)
            norm = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
            for i in range(len(names)):
                for j in range(len(names)):
                    v = norm[i, j]
                    ink = theme["surface"] if v > 0.55 else theme["ink2"]
                    ax.text(
                        j,
                        i,
                        f"{v:.0%}" if v >= 0.005 else "",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=ink,
                    )
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, fontsize=8, rotation=30, ha="right")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names if ax is axes[0] else [], fontsize=8)
            ax.tick_params(colors=theme["muted"], labelcolor=theme["ink2"], length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{n} hands", fontsize=10, loc="left", color=theme["ink"])
            ax.set_xlabel("Predicted")
            ax.xaxis.label.set_color(theme["ink2"])
        axes[0].set_ylabel("True archetype")
        axes[0].yaxis.label.set_color(theme["ink2"])
        title(
            fig,
            theme,
            "Which archetypes get confused, and until when",
            "Row-normalized confusion matrices, held-out players pooled over seeds "
            f"({MODEL_LABEL[model].lower()})",
            top=0.86,
        )
        return fig

    save_both(draw, out_dir, "confusion")


def plot_recall(agg, out_dir: Path) -> None:
    sizes = agg["sample_sizes"]
    model = agg["primary_model"]

    def draw(theme):
        fig, ax = new_figure(theme, 1, 1, figsize=(7.5, 4.4))
        style_axes(ax, theme)
        for slot, a in enumerate(ARCHETYPES):
            d = [agg["classification"][model][str(n)]["recall"][a.value] for n in sizes]
            mean = [x["mean"] for x in d]
            line(ax, theme, sizes, mean, theme["series"][slot], label=ARCH_LABEL[a.value])
            _end_label(ax, theme, sizes[-1], mean[-1], ARCH_LABEL[a.value])
        _sizes_axis(ax, sizes)
        ax.set_ylim(0, 1.04)
        ax.set_ylabel("Recall (share of that archetype identified)")
        legend(ax, theme, loc="lower right")
        title(
            fig,
            theme,
            "Per-archetype recall vs hands observed",
            f"{MODEL_LABEL[model]}; mean over seeds",
            top=0.88,
        )
        return fig

    save_both(draw, out_dir, "per_class_recall")


def plot_regression(agg, out_dir: Path) -> None:
    sizes = agg["sample_sizes"]
    params = list(PARAM_NAMES)

    def draw(theme):
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("seq", [theme["surface"], *theme["seq"][2:]])
        fig, (ax1, ax2) = new_figure(
            theme, 1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1]}
        )
        r2 = np.array(
            [[agg["regression"]["gbm"][str(n)][p]["mean"] for n in sizes] for p in params]
        )
        ax1.imshow(np.clip(r2, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(params)):
            for j in range(len(sizes)):
                v = r2[i, j]
                ax1.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=theme["surface"] if v > 0.6 else theme["ink2"],
                )
        ax1.set_xticks(range(len(sizes)))
        ax1.set_xticklabels([str(n) for n in sizes], fontsize=8)
        ax1.set_yticks(range(len(params)))
        ax1.set_yticklabels(params, fontsize=8)
        ax1.tick_params(colors=theme["muted"], labelcolor=theme["ink2"], length=0)
        for s in ax1.spines.values():
            s.set_visible(False)
        ax1.set_xlabel("Hands observed per player")
        ax1.xaxis.label.set_color(theme["ink2"])
        ax1.set_title(
            "Out-of-sample R², boosted trees", fontsize=10, loc="left", color=theme["ink"]
        )

        style_axes(ax2, theme)
        for slot, (k, label) in enumerate(
            (
                ("gbm", "Boosted-tree regression"),
                ("ridge", "Ridge regression"),
                ("type_mean", "Classify, then archetype mean"),
            )
        ):
            d = [agg["regression"][k][str(n)]["_mean_over_params"] for n in sizes]
            band(ax2, sizes, *_bounds(d, hi=1.0), theme["series"][slot])
            line(ax2, theme, sizes, [x["mean"] for x in d], theme["series"][slot], label=label)
        ax2.axhline(0, color=theme["muted"], linewidth=1)
        _sizes_axis(ax2, sizes)
        ax2.set_ylabel("Mean R² over the 9 parameters")
        legend(ax2, theme, loc="lower right")
        title(
            fig,
            theme,
            "Recovering the continuous bot parameters",
            "R² relative to predicting the training-population mean; held-out players, "
            f"{agg['n_seeds']} seeds",
            top=0.88,
        )
        return fig

    save_both(draw, out_dir, "param_regression")


def plot_action(agg, out_dir: Path) -> None:
    cps = list(agg["action_prediction"])
    xs = list(range(len(cps)))

    def draw(theme):
        fig, (ax1, ax2) = new_figure(theme, 1, 2, figsize=(11.5, 4.3))
        style_axes(ax1, theme)
        for slot, k in enumerate(("situation", "profile", "param_oracle")):
            d = [agg["action_prediction"][c]["log_loss"][k] for c in cps]
            band(ax1, xs, *_bounds(d, lo=0.0), theme["series"][slot])
            line(
                ax1, theme, xs, [x["mean"] for x in d], theme["series"][slot], label=ACTION_LABEL[k]
            )
        ax1.set_ylabel("Log-loss of next action (nats)")
        style_axes(ax2, theme)
        ax2.axhline(0, color=theme["muted"], linewidth=1)
        d = [agg["action_prediction"][c]["relative_improvement"] for c in cps]
        band(ax2, xs, *_bounds(d, scale=100.0), theme["series"][1])
        line(ax2, theme, xs, [100 * x["mean"] for x in d], theme["series"][1])
        ax2.set_ylabel("Log-loss reduction vs population (%)")
        for ax in (ax1, ax2):
            ax.set_xticks(xs)
            ax.set_xticklabels(cps)
            ax.set_xlabel("Hands observed before the predicted window")
        legend(ax1, theme, loc="upper right")
        title(
            fig,
            theme,
            "Predicting each player's next action from their profile",
            "Fold / check-call / bet-raise on held-out players; 95% t-interval across seeds",
            top=0.88,
        )
        return fig

    save_both(draw, out_dir, "action_prediction")


def make_figures(per_seed, agg, out_dir: Path, cfg: dict) -> None:
    sizes = agg["sample_sizes"]
    rel_sizes = [n for n in cfg["reliability_sizes"] if n in sizes]
    conf_sizes = sorted({sizes[0], sizes[2], sizes[-3], sizes[-1]})
    plot_hands_to_identify(agg, out_dir)
    plot_shrinkage_ablation(agg, out_dir)
    plot_calibration(per_seed, agg, out_dir, rel_sizes)
    plot_confusion(agg, out_dir, conf_sizes)
    plot_recall(agg, out_dir)
    plot_regression(agg, out_dir)
    plot_action(agg, out_dir)


# ------------------------------------------------------------------ markdown summary
def _fmt(
    d: dict, pct: bool = False, digits: int = 3, lo: float | None = None, hi: float | None = None
) -> str:
    lower, upper = d["lower"], d["upper"]
    if lo is not None:
        lower = max(lower, lo)
    if hi is not None:
        upper = min(upper, hi)
    if pct:
        return f"{100 * d['mean']:.1f}% [{100 * lower:.1f}, {100 * upper:.1f}]"
    return f"{d['mean']:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def _acc(d: dict) -> str:
    return _fmt(d, pct=True, lo=0.0, hi=1.0)


def markdown_summary(agg: dict) -> str:
    sizes = agg["sample_sizes"]
    primary = agg["primary_model"]
    order = [primary] + [m for m in MODEL_LABEL if m != primary]
    lines = [
        f"Mean over {agg['n_seeds']} seeds; brackets are 95% t-intervals across seeds "
        "(clipped to the metric's natural range for display). Primary model: "
        f"{MODEL_LABEL[primary].lower()}.\n"
    ]
    lines.append("### Hands-to-identify: accuracy on held-out players\n")
    lines.append("| Hands | " + " | ".join(MODEL_LABEL[m] for m in order) + " |")
    lines.append("|---" * (len(order) + 1) + "|")
    for n in sizes:
        cells = [_acc(agg["classification"][m][str(n)]["accuracy"]) for m in order]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines.append("\nLog-loss (nats; chance = 1.609):\n")
    lines.append("| Hands | " + " | ".join(MODEL_LABEL[m] for m in order) + " |")
    lines.append("|---" * (len(order) + 1) + "|")
    for n in sizes:
        cells = [_fmt(agg["classification"][m][str(n)]["log_loss"], lo=0.0) for m in order]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    o = agg["oracle"]
    lines.append(
        f"\nOracle ceiling (classifier on the true parameters, i.e. infinite hands): accuracy "
        f"{_acc(o['lr']['accuracy'])} (logistic), {_acc(o['gbm']['accuracy'])} (boosted trees). "
        f"Permutation control (labels shuffled across training players): accuracy "
        f"{_acc(agg['permutation_control'])} vs 20% chance.\n"
    )
    lines.append(f"### Hands needed to reach an accuracy ({MODEL_LABEL[primary].lower()})\n")
    lines.append("| Target | Seeds reaching it | Geometric mean | Range across seeds |")
    lines.append("|---|---|---|---|")
    for t, d in agg["hands_to_accuracy"][primary].items():
        gm = "n/a" if d["geometric_mean"] is None else f"{d['geometric_mean']:.0f}"
        rng = "n/a" if d["min"] is None else f"{d['min']:.0f}–{d['max']:.0f}"
        lines.append(
            f"| {float(t):.0%} | {d['reached_in_seeds']}/{agg['n_seeds']} | {gm} | {rng} |"
        )
    lines.append("\n### Calibration: expected calibration error (10 bins)\n")
    lines.append(
        "| Hands | Logistic uncal. | Logistic temp.-scaled | Trees uncal. | Trees temp.-scaled |"
    )
    lines.append("|---|---|---|---|---|")
    for n in sizes:
        a = agg["classification"]["lr_shrunk"][str(n)]
        b = agg["classification"]["gbm_shrunk"][str(n)]
        lines.append(
            f"| {n} | {_fmt(a['ece_uncal'], lo=0.0)} | {_fmt(a['ece'], lo=0.0)} | "
            f"{_fmt(b['ece_uncal'], lo=0.0)} | {_fmt(b['ece'], lo=0.0)} |"
        )
    t = agg["temperatures"]
    lines.append(
        f"\nFitted temperatures (held-out training players): logistic "
        f"{_fmt(t['lr_shrunk'], digits=2)}, boosted trees {_fmt(t['gbm_shrunk'], digits=2)} "
        "(T > 1 means the raw model was overconfident).\n"
    )
    lines.append("### Shrinkage ablation (shrunk − raw; negative Δlog-loss = shrinkage helps)\n")
    lines.append(
        "| Hands | Logistic Δacc (points) | Logistic Δlog-loss | Trees Δacc (points) | "
        "Trees Δlog-loss |"
    )
    lines.append("|---|---|---|---|---|")
    for n in sizes:
        a, b = agg["shrinkage_ablation"]["lr"][str(n)], agg["shrinkage_ablation"]["gbm"][str(n)]
        lines.append(
            f"| {n} | {_fmt(a['delta_accuracy'], pct=True)} | {_fmt(a['delta_log_loss'])} | "
            f"{_fmt(b['delta_accuracy'], pct=True)} | {_fmt(b['delta_log_loss'])} |"
        )
    lines.append("\n### Parameter regression: out-of-sample R² (boosted trees)\n")
    show = sorted({sizes[0], sizes[2], sizes[-3], sizes[-1]})
    lines.append("| Parameter | " + " | ".join(f"{n} hands" for n in show) + " |")
    lines.append("|---" * (len(show) + 1) + "|")
    for p in PARAM_NAMES:
        cells = [_fmt(agg["regression"]["gbm"][str(n)][p], digits=2, hi=1.0) for n in show]
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    for k, label in (
        ("gbm", "boosted trees"),
        ("ridge", "ridge"),
        ("type_mean", "classify-then-archetype-mean"),
    ):
        cells = ", ".join(
            f"{n}: {_fmt(agg['regression'][k][str(n)]['_mean_over_params'], digits=2, hi=1.0)}"
            for n in show
        )
        lines.append(f"\nMean R² over parameters, {label}: {cells}")
    lines.append("\n\n### Next-action prediction (held-out players, log-loss in nats)\n")
    lines.append(
        "| Hands observed | Population (situation only) | + profile | Oracle (true params) "
        "| Reduction vs population | Share of oracle gap closed |"
    )
    lines.append("|---|---|---|---|---|---|")
    for c, d in agg["action_prediction"].items():
        lines.append(
            f"| {c} | {_fmt(d['log_loss']['situation'], lo=0.0)} | "
            f"{_fmt(d['log_loss']['profile'], lo=0.0)} | "
            f"{_fmt(d['log_loss']['param_oracle'], lo=0.0)} | "
            f"{_fmt(d['relative_improvement'], pct=True)} | "
            f"{_fmt(d['oracle_gap_closed'], pct=True)} |"
        )
    return "\n".join(lines) + "\n"

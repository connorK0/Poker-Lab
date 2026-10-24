"""P3.4: ranked exploit recommendations with confidence, evaluated on held-out push/fold bots.

    python -m experiments.phase3_recommend --seed 2026

Pre-registered plan (written before the first run):

* Game, populations and observations as in P3.2 (``phase3_exploit``), and for the same master
  seed the very same bots and hands: heads-up push/fold at ``stack`` bb; per seed a training
  population (used only to fit the prior) and held-out *standard* bots (hands ranked by equity,
  as the model assumes) and *misspecified* bots (pairs first); the hero plays the equilibrium
  for ``max(sizes)`` hands against each, alternating seats, and sees cards only at showdowns.
* Recommender (``poker_lab.exploit.recommend``): posterior by importance sampling (``draws``
  draws; one random stream per bot, the same at every ``n``); candidates are the equilibrium,
  restricted Nash responses for ``ps`` and the best response, all against the posterior mean;
  ranked by the lower end of the central ``level`` credible interval for the gain over the
  equilibrium. ``n = 0`` (no hands) gives the population recommendation.
* For every bot, ``n`` and candidate: the exact *true* gain over the equilibrium against the
  bot's real strategy (bb/hand, both seats averaged) and the exact exploitability.
* Rules, compared by true gain:
  - **lower bound**: the recommender's top pick (the rule under test);
  - **Bayes best response**: always the best response to the posterior mean (the highest
    posterior expected gain);
  - **mode best response**: the best response to the posterior mode (P3.2's likelihood model);
  - **restricted Nash p = 0.5**, always;
  - **hindsight**: the candidate with the highest true gain for this bot (not achievable; the
    ceiling for any ranking of these candidates);
  - **oracle**: the best response to the bot's true strategy.
* Questions and predictions:
  1. *Does ranking by the lower bound pick responses that really win?* The share of bots against
     which the top pick does worse than the equilibrium (true gain < 0), and among picks that
     deviate, the share with true gain > 0. Prediction (standard population): at most 5% of
     bots lose at every ``n``, which the posterior promises if its prior matches the population.
  2. *Are the intervals honest?* Coverage of the 90% interval (true gain inside it) over all
     non-equilibrium candidates and bots, and of the one-sided bound (true gain >= lower end,
     nominal 95%); also for the top pick alone, when it deviates. Prediction (standard
     population): 90% coverage between 85% and 95% at every ``n``. The prior is a product of
     independent Betas and a log-normal, fitted by moments to a mixture of five archetypes, so
     it is not the population's distribution; this measures how much that matters.
  3. *Cost of caution:* true gain of the lower-bound rule vs the Bayes and mode best responses,
     by ``n``, and its regret against hindsight.
  4. *Bayes vs mode:* the paired difference between the two best responses, by ``n``.
  5. *What p the evidence sets:* the share of bots per top pick, by ``n``.
  6. *Worth of watching:* the top pick's posterior mean gain over the population
     recommendation vs its true value.
  The misspecified population is reported with the same metrics and no prediction.
* Checks: the mode best response reproduces P3.2's per-bot gains when P3.2's results for the
  same configuration are present (max |difference| < 1e-9); the importance-sampling effective
  sample size (minimum and median) for every ``n``.
* Intervals: shares are pooled over bots and seeds with 95% Wilson intervals (coverage pools
  correlated candidates of the same bot, so its interval is optimistic); gains are means over
  bots per seed with 95% t-intervals across seeds.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import astuple
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import (
    REPO_ROOT,
    RESULTS_ROOT,
    derive_seeds,
    load_config,
    run_metadata,
    save_json,
)
from experiments.phase3_exploit import POPULATIONS, observation_log, sample_populations
from poker_lab.evaluation.intervals import t_interval, wilson_interval
from poker_lab.exploit.recommend import PushFoldRecommender
from poker_lab.games.push_fold import hand_gains, match_values, range_share, strategy_exploitability
from poker_lab.games.push_fold_bots import PushFoldBotParams, PushFoldPrior, strategy

CONFIG = REPO_ROOT / "experiments" / "configs" / "phase3.yaml"
RULES = ("lower_bound", "bayes_br", "mode_br", "rnr_0.5", "hindsight", "oracle")
TOL = 1e-9  # true gains within this of zero are ties with the equilibrium
_RECOMMENDERS: dict[int, PushFoldRecommender] = {}


def recommender(seed: int, prior: PushFoldPrior, cfg: dict[str, Any]) -> PushFoldRecommender:
    """Per-process cache: one recommender (and population recommendation) per seed."""
    if seed not in _RECOMMENDERS:
        _RECOMMENDERS[seed] = PushFoldRecommender(
            float(cfg["stack"]),
            prior,
            ps=cfg["ps"],
            level=cfg["level"],
            draws=cfg["draws"],
            seed=[seed, 20],
        )
    return _RECOMMENDERS[seed]


def best_response(stack: float, push: np.ndarray, call: np.ndarray):
    sb, bb = hand_gains(stack, push, call)
    return (sb > 0).astype(float), (np.nan_to_num(bb, nan=-1.0) > 0).astype(float)


def bot_task(
    seed: int,
    population: str,
    bot_index: int,
    archetype: str,
    params: PushFoldBotParams,
    prior: PushFoldPrior,
    cfg: dict[str, Any],
    example: bool,
) -> dict[str, Any]:
    stack = float(cfg["stack"])
    rec = recommender(seed, prior, cfg)
    log = observation_log(seed, population, bot_index, params, cfg)
    bot = strategy(params, POPULATIONS[population])
    eq_value = float(match_values(stack, *rec.equilibrium, *bot))

    def true_gain(push, call) -> float:
        return float(match_values(stack, push, call, *bot)) - eq_value

    def scored(push, call) -> dict[str, float]:
        return {
            "true_gain": true_gain(push, call),
            "exploitability": float(strategy_exploitability(stack, push, call)),
        }

    pop = rec.population
    record: dict[str, Any] = {
        "bot_id": f"{population[0]}{bot_index:03d}",
        "archetype": archetype,
        "population": population,
        "params": params.as_array(),
        "true_shares": [range_share(bot[0]), range_share(bot[1])],
        "eq_value": eq_value,
        "oracle": scored(*best_response(stack, *bot)),
        "population_pick": pop.name,
        "population_true_gain": true_gain(pop.push, pop.call),
        "sizes": {},
    }
    stream = 10 + (population != "standard")
    for n in cfg["sizes"]:
        report = rec.recommend(log.observations(n), seed=[seed, stream, bot_index])
        entry: dict[str, Any] = {
            "ranked": [r.name for r in report.ranked],
            "ess": report.ess,
            "mode": report.mode.as_array(),
            "push_share": list(astuple(report.push_share)),
            "call_share": list(astuple(report.call_share)),
            "mode_br": scored(*best_response(stack, *strategy(report.mode))),
            "candidates": {},
        }
        for r in report.ranked:
            c = {
                "mean": r.gain.mean,
                "lower": r.gain.lower,
                "upper": r.gain.upper,
                "prob_gain": r.prob_gain,
                "over_mean": r.over_population.mean,
                **scored(r.push, r.call),
            }
            if example:  # extra detail for the explorer's example reports
                c |= {
                    "over_lower": r.over_population.lower,
                    "over_upper": r.over_population.upper,
                    "push_share": range_share(r.push),
                    "call_share": range_share(r.call),
                }
            entry["candidates"][r.name] = c
        if example:
            entry["explanation"] = report.explanation()
        record["sizes"][str(n)] = entry
    return record


# ------------------------------------------------------------------------------- aggregation


def _ci(values) -> dict[str, float]:
    iv = t_interval(np.asarray(values, dtype=float))
    return {"mean": iv.estimate, "lower": iv.lower, "upper": iv.upper, "n": iv.n}


def _share(k: int, n: int) -> dict[str, float]:
    iv = wilson_interval(int(k), int(n))
    share = k / n if n else float("nan")
    return {"share": share, "lower": iv.lower, "upper": iv.upper, "k": int(k), "n": int(n)}


def candidate_names(cfg: dict[str, Any]) -> list[str]:
    return ["equilibrium", *(f"restricted_nash_p{p:g}" for p in cfg["ps"]), "best_response"]


def rule_score(b: dict[str, Any], n: str, rule: str) -> dict[str, float]:
    """True gain and exploitability of ``rule``'s choice for bot ``b`` after ``n`` hands."""
    if rule == "oracle":
        return b["oracle"]
    s = b["sizes"][n]
    cands = s["candidates"]
    if rule == "lower_bound":
        return cands[s["ranked"][0]]
    if rule == "bayes_br":
        return cands["best_response"]
    if rule == "mode_br":
        return s["mode_br"]
    if rule == "rnr_0.5":
        return cands["restricted_nash_p0.5"]
    if rule == "hindsight":
        return max(cands.values(), key=lambda c: c["true_gain"])
    raise ValueError(rule)


def _size_block(groups: list[list[dict[str, Any]]], n: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Every metric for one population after ``n`` hands; ``groups`` holds the bots per seed."""
    flat = [b for g in groups for b in g]

    def seed_ci(fn):
        return _ci([np.mean([fn(b) for b in g]) for g in groups])

    def gain(b, rule):
        return rule_score(b, n, rule)["true_gain"]

    def covered(c):
        return c["lower"] - 1e-12 <= c["true_gain"] <= c["upper"] + 1e-12

    def above_lower(c):
        return c["true_gain"] >= c["lower"] - 1e-12

    rules = {
        rule: {
            "gain": seed_ci(lambda b, rule=rule: gain(b, rule)),
            "exploitability": seed_ci(
                lambda b, rule=rule: rule_score(b, n, rule)["exploitability"]
            ),
            "loses": _share(sum(gain(b, rule) < -TOL for b in flat), len(flat)),
        }
        for rule in RULES
    }
    picks = [b["sizes"][n]["ranked"][0] for b in flat]
    top = [b["sizes"][n]["candidates"][name] for b, name in zip(flat, picks, strict=True)]
    deviating = [c for c, name in zip(top, picks, strict=True) if name != "equilibrium"]
    others = [
        c for b in flat for name, c in b["sizes"][n]["candidates"].items() if name != "equilibrium"
    ]
    ess = [b["sizes"][n]["ess"] for b in flat]
    return {
        "rules": rules,
        "deviates": _share(len(deviating), len(flat)),
        "deviation_wins": _share(sum(c["true_gain"] > TOL for c in deviating), len(deviating)),
        "coverage": _share(sum(map(covered, others)), len(others)),
        "one_sided": _share(sum(map(above_lower, others)), len(others)),
        "pick_coverage": _share(sum(map(covered, deviating)), len(deviating)),
        "pick_one_sided": _share(sum(map(above_lower, deviating)), len(deviating)),
        "picks": {name: _share(picks.count(name), len(picks)) for name in candidate_names(cfg)},
        "bayes_minus_mode": seed_ci(lambda b: gain(b, "bayes_br") - gain(b, "mode_br")),
        "regret": seed_ci(lambda b: gain(b, "hindsight") - gain(b, "lower_bound")),
        "over_population_posterior": seed_ci(
            lambda b: b["sizes"][n]["candidates"][b["sizes"][n]["ranked"][0]]["over_mean"]
        ),
        "over_population_true": seed_ci(
            lambda b: gain(b, "lower_bound") - b["population_true_gain"]
        ),
        "ess": {"min": float(np.min(ess)), "median": float(np.median(ess))},
    }


def aggregate(per_seed: list[list[dict[str, Any]]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Per population and sample size: means over bots per seed, then across seeds."""
    out: dict[str, Any] = {}
    for pop in POPULATIONS:
        groups = [[b for b in bots if b["population"] == pop] for bots in per_seed]
        out[pop] = {n: _size_block(groups, n, cfg) for n in map(str, cfg["sizes"])}
    return out


def check_against_p32(per_seed, seeds, cfg: dict[str, Any], path: Path) -> dict[str, Any]:
    """Mode best response vs P3.2's likelihood-model best response, bot by bot."""
    if not path.exists():
        return {"compared": 0, "note": f"{path.name} not found"}
    p32 = json.loads(path.read_text(encoding="utf-8"))
    keys = ("stack", "seeds", "train_per_archetype", "test_per_archetype")
    other = p32["meta"]["config"]
    same = all(other[k] == cfg[k] for k in keys) and max(other["sizes"]) == max(cfg["sizes"])
    if not same or p32["seeds"] != seeds:
        return {"compared": 0, "note": "P3.2 results come from a different configuration or seed"}
    diffs = []
    for mine, theirs in zip(per_seed, p32["per_seed"], strict=True):
        by_id = {(b["population"], b["bot_id"]): b for b in theirs}
        for b in mine:
            ref = by_id[(b["population"], b["bot_id"])]
            diffs.append(abs(b["eq_value"] - ref["eq_value"]))
            for n in map(str, other["sizes"]):
                if n in b["sizes"]:
                    got = b["sizes"][n]["mode_br"]["true_gain"]
                    diffs.append(abs(got - ref["sizes"][n]["likelihood"]["scores"]["br"]["gain"]))
    worst = float(max(diffs))
    return {"compared": len(diffs), "max_abs_diff": worst, "passed": worst < 1e-9}


def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase3" / "recommend"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()
    seeds = derive_seeds(seed, cfg["seeds"])
    tasks = []
    for s in seeds:
        prior, bots = sample_populations(s, cfg)
        for pop, k, arch, params in bots:
            example = s == seeds[0] and pop == "standard" and k % cfg["test_per_archetype"] == 0
            tasks.append((s, pop, k, arch, params, prior, cfg, example))
    with ProcessPoolExecutor(max_workers=workers or cfg.get("workers")) as pool:
        records = list(pool.map(bot_task, *zip(*tasks, strict=True), chunksize=1))
    per_seed = [[r for r, t in zip(records, tasks, strict=True) if t[0] == s] for s in seeds]
    results = {
        "meta": meta,
        "seeds": seeds,
        "aggregate": aggregate(per_seed, cfg),
        "check_p32": check_against_p32(
            per_seed, seeds, cfg, out_dir.parent / "exploit" / "exploit.json"
        ),
        "per_seed": per_seed,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    save_json(results, out_dir / "recommend.json")
    (out_dir / "recommend_summary.md").write_text(
        markdown_summary(results), encoding="utf-8", newline="\n"
    )
    make_figures(results, out_dir)
    return results


# ---------------------------------------------------------------------------------- reports

RULE_LABELS = {
    "lower_bound": "Lower-bound pick",
    "bayes_br": "Bayes best response",
    "mode_br": "Mode best response (P3.2)",
    "rnr_0.5": "Restricted Nash, p = 0.5",
    "hindsight": "Best candidate in hindsight",
    "oracle": "Oracle best response",
}


def candidate_label(name: str) -> str:
    if name == "equilibrium":
        return "Equilibrium"
    if name == "best_response":
        return "Best response"
    return f"Restricted Nash, p = {name.removeprefix('restricted_nash_p')}"


def _fmt(d: dict[str, float], signed: bool = True) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(d['mean'])} [{f.format(d['lower'])}, {f.format(d['upper'])}]"


def _pct(d: dict[str, float]) -> str:
    if not d["n"]:
        return "n/a"
    return f"{d['share']:.1%} [{d['lower']:.1%}, {d['upper']:.1%}] ({d['k']}/{d['n']})"


def markdown_summary(r: dict[str, Any]) -> str:
    cfg = r["meta"]["config"]
    agg = r["aggregate"]
    sizes = [str(n) for n in cfg["sizes"]]
    chk = r["check_p32"]
    lines = [
        "# P3.4: ranked exploit recommendations with confidence",
        "",
        f"Stack {cfg['stack']} bb. {cfg['seeds']} seeds x {5 * cfg['test_per_archetype']} held-out "
        f"bots per population; {cfg['draws']} posterior draws; {cfg['level']:.0%} intervals; "
        "candidates ranked by the interval's lower end. Gains in bb/hand over the equilibrium, "
        "exact, against each bot's true strategy (95% t-intervals across seeds). Shares pooled "
        "over bots with 95% Wilson intervals.",
        "",
        "Check against P3.2 (mode best response, bot by bot): "
        + (
            f"{chk['compared']} values, max |difference| {chk['max_abs_diff']:.1e} "
            f"({'pass' if chk['passed'] else 'FAIL'})."
            if chk["compared"]
            else chk["note"] + "."
        ),
    ]
    for pop in POPULATIONS:
        a = agg[pop]
        lines += ["", f"## {pop.capitalize()} population", "", "True gain by rule:", ""]
        lines += ["| Hands | " + " | ".join(RULE_LABELS[k] for k in RULES) + " |"]
        lines += ["|---|" + "---|" * len(RULES)]
        for n in sizes:
            cells = " | ".join(_fmt(a[n]["rules"][k]["gain"]) for k in RULES)
            lines.append(f"| {int(n):,} | {cells} |")
        lines += ["", "Share of bots against which the rule does worse than the equilibrium:", ""]
        lines += ["| Hands | " + " | ".join(RULE_LABELS[k] for k in RULES[:5]) + " |"]
        lines += ["|---|" + "---|" * 5]
        for n in sizes:
            lines.append(
                f"| {int(n):,} | "
                + " | ".join(f"{a[n]['rules'][k]['loses']['share']:.1%}" for k in RULES[:5])
                + " |"
            )
        lines += [
            "",
            "Does the lower-bound pick win, and are the intervals honest?",
            "",
            "| Hands | Deviates | Deviation wins | Pick loses | 90% coverage (all) | "
            "One-sided (all) | 90% coverage (pick) | ESS min / median |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for n in sizes:
            x = a[n]
            lines.append(
                f"| {int(n):,} | {_pct(x['deviates'])} | {_pct(x['deviation_wins'])} | "
                f"{_pct(x['rules']['lower_bound']['loses'])} | {_pct(x['coverage'])} | "
                f"{_pct(x['one_sided'])} | {_pct(x['pick_coverage'])} | "
                f"{x['ess']['min']:.0f} / {x['ess']['median']:.0f} |"
            )
        names = candidate_names(cfg)
        lines += ["", "Top pick by hands observed (share of bots):", ""]
        lines += ["| Hands | " + " | ".join(candidate_label(k) for k in names) + " |"]
        lines += ["|---|" + "---|" * len(names)]
        for n in sizes:
            lines.append(
                f"| {int(n):,} | "
                + " | ".join(f"{a[n]['picks'][k]['share']:.0%}" for k in names)
                + " |"
            )
        lines += [
            "",
            "| Hands | Bayes minus mode BR | Regret vs hindsight | Over population (posterior) | "
            "Over population (true) |",
            "|---|---|---|---|---|",
        ]
        for n in sizes:
            x = a[n]
            lines.append(
                f"| {int(n):,} | {_fmt(x['bayes_minus_mode'])} | {_fmt(x['regret'])} | "
                f"{_fmt(x['over_population_posterior'])} | {_fmt(x['over_population_true'])} |"
            )
    examples = [b for b in r["per_seed"][0] if "explanation" in b["sizes"][sizes[-1]]]
    if examples:
        lines += ["", "## Example reports (first seed, one standard bot per archetype)", ""]
        for b in examples:
            for n in ("10", "100", "1000"):
                if n in b["sizes"]:
                    text = b["sizes"][n]["explanation"]
                    lines += [f"* **{b['archetype']}, {int(n):,} hands.** {text}"]
    return "\n".join(lines) + "\n"


def make_figures(r: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import band, legend, line, new_figure, save_both, style_axes, title

    cfg = r["meta"]["config"]
    sizes = [n for n in cfg["sizes"] if n > 0]
    x = np.array(sizes, dtype=float)
    agg = r["aggregate"]
    shown = (("lower_bound", 0), ("bayes_br", 1), ("mode_br", 2), ("hindsight", 4))

    def gains(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(12.5, 4.6), sharey=True)
        s = theme["series"]
        for ax, pop in zip(axes, POPULATIONS, strict=True):
            style_axes(ax, theme)
            a = agg[pop]
            for rule, c in shown:
                d = [a[str(n)]["rules"][rule]["gain"] for n in sizes]
                if rule == "lower_bound":
                    band(ax, x, [v["lower"] for v in d], [v["upper"] for v in d], s[c])
                line(ax, theme, x, [v["mean"] for v in d], s[c], RULE_LABELS[rule])
            ax.axhline(
                a[str(sizes[-1])]["rules"]["oracle"]["gain"]["mean"],
                color=theme["muted"],
                linewidth=1,
                linestyle="--",
            )
            ax.axhline(0, color=theme["axis"], linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xlabel("Hands observed")
            ax.set_title(f"{pop.capitalize()} bots", loc="left", fontsize=10, color=theme["ink"])
        axes[0].set_ylabel("True gain over equilibrium (bb/hand)")
        legend(axes[0], theme, loc="lower right")
        title(
            fig,
            theme,
            "Recommendations ranked by a posterior lower bound (10 bb push/fold)",
            "Dashed: oracle best response. Band: 95% interval across seeds, lower-bound pick.",
        )
        return fig

    def safety(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(12.5, 4.6))
        s = theme["series"]
        ax = axes[0]
        style_axes(ax, theme)
        for rule, c in shown[:3]:
            for pop, dashed in (("standard", False), ("misspecified", True)):
                y = [100 * agg[pop][str(n)]["rules"][rule]["loses"]["share"] for n in sizes]
                label = f"{RULE_LABELS[rule]}, {pop}"
                line(ax, theme, x, y, s[c], label, linestyle="--" if dashed else "-")
        ax.axhline(5, color=theme["muted"], linewidth=1, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlabel("Hands observed")
        ax.set_ylabel("Bots where the rule loses to equilibrium (%)")
        legend(ax, theme, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handlelength=3.5)
        ax = axes[1]
        style_axes(ax, theme)
        for (pop, key), c, dashed in (
            (("standard", "coverage"), 0, False),
            (("misspecified", "coverage"), 0, True),
            (("standard", "one_sided"), 1, False),
            (("misspecified", "one_sided"), 1, True),
        ):
            d = [agg[pop][str(n)][key] for n in sizes]
            name = "90% interval" if key == "coverage" else "One-sided 95% bound"
            line(
                ax,
                theme,
                x,
                [100 * v["share"] for v in d],
                s[c],
                f"{name}, {pop}",
                linestyle="--" if dashed else "-",
            )
        for ref in (90, 95):
            ax.axhline(ref, color=theme["muted"], linewidth=1, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlabel("Hands observed")
        ax.set_ylabel("Coverage of the true gain (%)")
        legend(ax, theme, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handlelength=3.5)
        title(
            fig,
            theme,
            "Do the recommendations keep their promises?",
            "Left: losses vs equilibrium (dotted: 5%). Right: interval coverage over all "
            "non-equilibrium candidates (dotted: nominal 90% and 95%).",
        )
        fig.subplots_adjust(bottom=0.3)
        return fig

    save_both(gains, out_dir, "recommend_gain")
    save_both(safety, out_dir, "recommend_safety")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)["recommend"]
    r = run(args.seed, cfg=cfg, workers=args.workers)
    print(markdown_summary(r))


if __name__ == "__main__":
    main()

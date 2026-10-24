"""P3.3: does the push/fold provider play the solved game in real hands? (the bridge check)

    python -m experiments.phase3_bridge --seed 2026

Pre-registered plan (written before the first run):

* Real hands: heads-up no-limit hold'em through ``TableSimulator`` (PokerKit rules, phevaluator
  showdowns, the sealed hand-history store), blinds 50/100, no ante, both stacks reset to ``S``
  bb every hand.
* Hero: ``PushFoldStrategy`` behind the advisor firewall (``AdvisorAgent``), playing either the
  exact equilibrium (``nash``, the provider's default) or the exact best response to the
  opponent's true ranges (``best_response``, an oracle).
* Opponents: ``StrategyAgent``s playing the five P3.2 archetypes at their mean parameters (the same
  ranges at every stack), plus the equilibrium itself (against the ``nash`` hero only).
* Each matchup: ``n_deals`` duplicate deals (each deck played twice with seats swapped, P2.9), in
  independently seeded chunks.
* Exact value: ``head_to_head`` of the two policies in the abstract game at ``S`` (bb/hand, both
  seats averaged).
* Four estimates of the hero's win rate from the same hands:
  - **naive**: the first table only (an ordinary match), t-interval over hands;
  - **duplicate**: both tables, t-interval over the per-deal means of the two plays;
  - **all-in EV**: the first table, with each called all-in scored by the class matchup equity
    ``c (2 e - 1)`` instead of the board that came (``c`` = chips each player put in). Unbiased:
    the class equity is the average over the class pair's combos, which are equally likely
    given the classes;
  - **duplicate + all-in EV**: both.
* Check: ``z = (estimate - exact) / SE`` per matchup and estimator. If the bridge is right,
  ``z ~ N(0, 1)``: about 5% of ``|z|`` above 1.96, none above 4, mean ``z`` near 0. Any
  ``|z| > 4``, or a mean ``z`` across matchups that a t-test puts at p < 0.01, is a failure to
  investigate before reporting. Every hand must also end preflop (a fold, or a called all-in
  with a five-card runout): any postflop decision means a spot left the tree.
* Secondary: hands needed for a 95% interval of half-width ``half_width`` bb/hand under each
  estimator (variance reduction, for planning Phase 5's live evaluation), and the smallest
  discrepancy each matchup would detect (``2.8 SE``: 80% power, two-sided 5% test).

Amendment (after the first run crashed in aggregation, before any result was seen): an estimate
can have no sampling variance at all. Nash against Nash under duplicate dealing at a stack where
the equilibrium is pure (20 bb) mirrors every deal exactly, so each pair sums to zero (up to float
rounding, ~1e-16). Its ``z`` is undefined; an estimate whose SD is below ``ZERO_SD`` passes only
if it equals the exact value, and it is left out of the ``z`` statistics.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from experiments.common import REPO_ROOT, RESULTS_ROOT, load_config, run_metadata, save_json
from poker_lab.advisor import Advisor, AdvisorAgent
from poker_lab.bots.hand_strength import hand_class
from poker_lab.engine.duplicate import duplicate_match
from poker_lab.games.evaluate import best_response, head_to_head
from poker_lab.games.preflop import CLASS_INDEX, load_matchups
from poker_lab.games.push_fold import (
    BB,
    SB,
    aggressive_probabilities,
    push_fold_tree,
    threshold_policy,
)
from poker_lab.games.push_fold_bots import (
    ARCHETYPES,
    TEMPERATURE_MEDIAN,
    PushFoldBotParams,
    strategy,
)
from poker_lab.schema import HandHistory, Street, TableConfig
from poker_lab.strategy import PushFoldStrategy, StrategyAgent, fixed_ranges, nash_ranges

CONFIG = REPO_ROOT / "experiments" / "configs" / "phase3.yaml"
TABLE = TableConfig("bridge", 2, 50, 100)
ESTIMATORS = ("naive", "duplicate", "allin_ev", "duplicate_allin_ev")
ESTIMATOR_LABELS = {
    "naive": "Naive",
    "duplicate": "Duplicate",
    "allin_ev": "All-in EV",
    "duplicate_allin_ev": "Duplicate + all-in EV",
}
HANDS_PER_UNIT = {"naive": 1, "duplicate": 2, "allin_ev": 1, "duplicate_allin_ev": 2}
ZERO_SD = 1e-9  # bb per unit: below this an estimator has no sampling variance


def opponent_ranges(name: str):
    if name == "nash":
        return nash_ranges
    mean = ARCHETYPES[name]
    params = PushFoldBotParams(mean["push_width"], mean["call_width"], TEMPERATURE_MEDIAN)
    return fixed_ranges(*strategy(params))


@cache
def policies(stack: float, hero: str, opponent: str):
    """(hero policy, opponent policy) in the abstract game at ``stack`` bb."""
    tree = push_fold_tree(stack)
    opp = threshold_policy(tree, *opponent_ranges(opponent)(stack))
    if hero == "nash":
        return threshold_policy(tree, *nash_ranges(stack)), opp
    br = best_response(opp, 0).policy.combine(best_response(opp, 1).policy, 0)
    return br, opp


def hero_ranges(stack: float, hero: str, opponent: str):
    if hero == "nash":
        return nash_ranges  # the provider's default: what the bridge check is about
    br, _ = policies(stack, hero, opponent)
    return fixed_ranges(aggressive_probabilities(br, SB), aggressive_probabilities(br, BB))


def hand_values(hand: HandHistory, seat: int, equity: np.ndarray) -> tuple[float, float, bool]:
    """(realized net, all-in EV net) for ``seat`` in bb, and whether the hand left the tree."""
    bb = hand.table.big_blind
    net = hand.net_results()[seat] / bb
    off_tree = any(a.street is not Street.PREFLOP for a in hand.actions)
    if len(hand.board_cards()) < 5:
        return net, net, off_tree  # folded preflop: nothing left to chance
    other = next(s.seat for s in hand.seats if s.seat != seat)
    cls = [
        CLASS_INDEX[hand_class(*(o.card for o in hand.hole_cards_of(s).cards))]
        for s in (seat, other)
    ]
    matched = min(hand.contributions().values()) / bb
    return net, matched * (2.0 * equity[cls[0], cls[1]] - 1.0), off_tree


def chunk_task(stack: float, hero: str, opponent: str, seed: int, n_deals: int) -> dict[str, Any]:
    ranges = hero_ranges(stack, hero, opponent)
    advisor_seeds = itertools.count(seed)

    def make_hero():
        advisor = Advisor(PushFoldStrategy(ranges), sample=True, seed=next(advisor_seeds))
        agent = AdvisorAgent(advisor)
        agent.keep_log = False
        return agent

    def make_villain():
        return StrategyAgent(PushFoldStrategy(opponent_ranges(opponent), name=opponent))

    match = duplicate_match(
        TABLE,
        (("hero", make_hero), ("villain", make_villain)),
        n_deals,
        seed,
        starting_stack=round(stack * TABLE.big_blind),
    )
    equity = load_matchups().equity
    out = {k: np.empty(n_deals) for k in ("net1", "net2", "ev1", "ev2")}
    off_tree = allins = 0
    for t, hands in ((1, match.first), (2, match.second)):
        for i, hand in enumerate(hands):
            net, ev, off = hand_values(hand, hand.seat_of("hero"), equity)
            out[f"net{t}"][i], out[f"ev{t}"][i] = net, ev
            off_tree += off
            allins += len(hand.board_cards()) == 5
    return {**out, "off_tree": off_tree, "allins": allins}


def summarize(units: np.ndarray, exact: float, per_unit: int, half_width: float) -> dict:
    n = len(units)
    mean, sd = float(units.mean()), float(units.std(ddof=1))
    se = sd / math.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    zero_variance = sd < ZERO_SD  # every pair cancels; only float rounding is left
    return {
        "mean": mean,
        "lower": mean - t * se,
        "upper": mean + t * se,
        "se": se,
        "sd_per_unit": sd,
        "units": n,
        "hands": n * per_unit,
        "diff": mean - exact,
        "z": None if zero_variance else (mean - exact) / se,
        "zero_variance": zero_variance,
        "mde": 2.8 * se,
        "hands_needed": 0 if zero_variance else math.ceil(per_unit * (1.96 * sd / half_width) ** 2),
    }


def matchups(cfg: dict[str, Any]) -> list[tuple[float, str, str]]:
    return [
        (float(s), h, o)
        for s in cfg["stacks"]
        for h in cfg["heroes"]
        for o in cfg["opponents"]
        if not (h == "best_response" and o == "nash")
    ]


def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase3" / "bridge"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()
    games = matchups(cfg)
    n_chunks = math.ceil(cfg["n_deals"] / cfg["chunk_deals"])
    tasks = []
    for g, (stack, hero, opp) in enumerate(games):
        for c in range(n_chunks):
            chunk_seed = int(np.random.SeedSequence([seed, g, c]).generate_state(1)[0])
            size = min(cfg["chunk_deals"], cfg["n_deals"] - c * cfg["chunk_deals"])
            tasks.append((stack, hero, opp, chunk_seed, size))
    with ProcessPoolExecutor(max_workers=workers or cfg.get("workers")) as pool:
        chunks = list(pool.map(chunk_task, *zip(*tasks, strict=True)))
    rows = []
    for g, (stack, hero, opp) in enumerate(games):
        parts = chunks[g * n_chunks : (g + 1) * n_chunks]
        arr = {k: np.concatenate([p[k] for p in parts]) for k in ("net1", "net2", "ev1", "ev2")}
        exact = head_to_head(*policies(stack, hero, opp))
        units = {
            "naive": arr["net1"],
            "duplicate": 0.5 * (arr["net1"] + arr["net2"]),
            "allin_ev": arr["ev1"],
            "duplicate_allin_ev": 0.5 * (arr["ev1"] + arr["ev2"]),
        }
        rows.append(
            {
                "stack": stack,
                "hero": hero,
                "opponent": opp,
                "exact": exact,
                "deals": len(arr["net1"]),
                "off_tree_hands": sum(p["off_tree"] for p in parts),
                "allin_share": sum(p["allins"] for p in parts) / (2 * len(arr["net1"])),
                "estimates": {
                    k: summarize(v, exact, HANDS_PER_UNIT[k], cfg["half_width"])
                    for k, v in units.items()
                },
            }
        )
    results = {
        "meta": meta,
        "matchups": rows,
        "checks": checks(rows),
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    save_json(results, out_dir / "bridge.json")
    (out_dir / "bridge_summary.md").write_text(
        markdown_summary(results), encoding="utf-8", newline="\n"
    )
    make_figures(results, out_dir)
    return results


def checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The pre-registered pass/fail criteria, per estimator."""
    out: dict[str, Any] = {"off_tree_hands": sum(r["off_tree_hands"] for r in rows)}
    for k in ESTIMATORS:
        est = [r["estimates"][k] for r in rows]
        z = np.array([e["z"] for e in est if not e["zero_variance"]])
        exact_ok = all(abs(e["diff"]) < 1e-9 for e in est if e["zero_variance"])
        test = stats.ttest_1samp(z, 0.0)
        out[k] = {
            "n": len(z),
            "zero_variance": len(est) - len(z),
            "zero_variance_exact": exact_ok,
            "mean_z": float(z.mean()),
            "sd_z": float(z.std(ddof=1)),
            "max_abs_z": float(np.abs(z).max()),
            "share_above_1_96": float((np.abs(z) > 1.96).mean()),
            "mean_z_p": float(test.pvalue),
            "pass": bool(np.abs(z).max() <= 4 and test.pvalue >= 0.01 and exact_ok),
        }
    out["pass"] = out["off_tree_hands"] == 0 and all(out[k]["pass"] for k in ESTIMATORS)
    return out


def label(hero: str, opponent: str) -> str:
    h = {"nash": "Nash", "best_response": "Best response"}[hero]
    o = "Nash" if opponent == "nash" else opponent.replace("_", " ")
    return f"{h} vs {o}"


def markdown_summary(r: dict[str, Any]) -> str:
    cfg = r["meta"]["config"]
    c = r["checks"]
    lines = [
        "# P3.3: the push/fold provider in real hands vs the exact game",
        "",
        f"{cfg['n_deals']:,} duplicate deals per matchup ({2 * cfg['n_deals']:,} hands), "
        f"heads-up through PokerKit, blinds 50/100. Exact = abstract-game value (bb/hand). "
        f"Hands that left the tree: {c['off_tree_hands']}. Overall: "
        f"{'PASS' if c['pass'] else 'FAIL'}.",
        "",
        "| Estimator | z-scores | mean z | sd z | max abs z | share abs z > 1.96 | t-test p "
        "| zero-variance (exact?) | pass |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in ESTIMATORS:
        e = c[k]
        lines.append(
            f"| {ESTIMATOR_LABELS[k]} | {e['n']} | {e['mean_z']:+.2f} | {e['sd_z']:.2f} | "
            f"{e['max_abs_z']:.2f} | {e['share_above_1_96']:.0%} | {e['mean_z_p']:.2f} | "
            f"{e['zero_variance']} ({'yes' if e['zero_variance_exact'] else 'NO'}) | "
            f"{'yes' if e['pass'] else 'NO'} |"
        )
    lines += [
        "",
        "| Stack | Matchup | Exact | Duplicate + all-in EV [95% CI] | z | Hands needed "
        f"(+/-{cfg['half_width']} bb): naive / dup / EV / dup+EV |",
        "|---|---|---|---|---|---|",
    ]
    for m in r["matchups"]:
        e = m["estimates"]
        best = e["duplicate_allin_ev"]
        lines.append(
            f"| {m['stack']:g} | {label(m['hero'], m['opponent'])} | {m['exact']:+.4f} | "
            f"{best['mean']:+.4f} [{best['lower']:+.4f}, {best['upper']:+.4f}] | "
            f"{'exact' if best['z'] is None else format(best['z'], '+.2f')} | "
            + " / ".join(f"{e[k]['hands_needed']:,}" for k in ESTIMATORS)
            + " |"
        )
    return "\n".join(lines) + "\n"


def make_figures(r: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import new_figure, save_both, style_axes, title

    rows = r["matchups"]
    stacks = sorted({m["stack"] for m in rows})

    def check(theme):
        fig, axes = new_figure(theme, 1, len(stacks), figsize=(4.2 * len(stacks), 4.8), sharey=True)
        axes = np.atleast_1d(axes)
        s = theme["series"]
        for ax, stack in zip(axes, stacks, strict=True):
            style_axes(ax, theme, grid_axis="x")
            sub = [m for m in rows if m["stack"] == stack]
            y = np.arange(len(sub))
            for k, (est, color, dy) in enumerate(
                (("naive", s[0], -0.2), ("duplicate_allin_ev", s[1], 0.2))
            ):
                e = [m["estimates"][est] for m in sub]
                ax.errorbar(
                    [x["diff"] for x in e],
                    y + dy,
                    xerr=[[x["mean"] - x["lower"] for x in e], [x["upper"] - x["mean"] for x in e]],
                    fmt="o",
                    color=color,
                    markersize=4,
                    elinewidth=1.2,
                    capsize=0,
                    label=ESTIMATOR_LABELS[est] if k < 2 else None,
                )
            ax.axvline(0, color=theme["axis"], linewidth=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels([label(m["hero"], m["opponent"]) for m in sub], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Simulated - exact (bb/hand)")
            ax.set_title(f"{stack:g} bb", loc="left", fontsize=10, color=theme["ink"])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9,
                   labelcolor=theme["ink2"])  # fmt: skip
        title(
            fig,
            theme,
            "Push/fold in real hands vs the exact game",
            "Difference between the simulated win rate and the abstract-game value, 95% CI",
            top=0.88,
        )
        fig.subplots_adjust(bottom=0.2, left=0.2)
        return fig

    def variance(theme):
        fig, ax = new_figure(theme, 1, 1, figsize=(7, 4.2))
        style_axes(ax, theme)
        s = theme["series"]
        x = np.arange(len(stacks))
        width = 0.2
        for k, est in enumerate(ESTIMATORS):
            med = [
                np.median(
                    [
                        m["estimates"][est]["hands_needed"]
                        for m in rows
                        if m["stack"] == st and m["opponent"] != "nash"
                    ]
                )
                for st in stacks
            ]
            ax.bar(x + (k - 1.5) * width, med, width * 0.9, color=s[k], label=ESTIMATOR_LABELS[est])
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{st:g} bb" for st in stacks])
        ax.set_ylabel("Hands for a +/-0.05 bb/hand 95% CI")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=theme["ink2"])
        title(fig, theme, "How many hands a win rate needs", "Median over the archetype bots")
        return fig

    save_both(check, out_dir, "bridge_check")
    save_both(variance, out_dir, "bridge_variance")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)["bridge"]
    r = run(args.seed, cfg=cfg, workers=args.workers)
    print(markdown_summary(r))


if __name__ == "__main__":
    main()

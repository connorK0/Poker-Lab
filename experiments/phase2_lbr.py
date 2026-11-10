"""P2.8b: local best response, a lower bound on the exploitability of hold'em strategies.

    python -m experiments.phase2_lbr --seed 2026

Exploitability is what a best response wins against a strategy. In heads-up no-limit hold'em the
best response is out of reach, so this step measures a *lower bound* on it: what local best
response (LBR; Lisy and Bowling, 2017) wins. LBR is the yardstick later exploit-engine steps
report against. Its numbers bound exploitability from below; they are never the exploitability.

Design (code in ``poker_lab/exploit/lbr``):

* **Game.** Heads-up no-limit, blinds 50/100 chips, no ante, both stacks reset every hand (the
  simulator's cash game), at 100 bb ("deep") and 10 bb ("shallow"). Heads-up first because LBR's
  range and look-ahead are defined against one opponent.
* **LBR's actions.** Fold, check/call, a pot-size bet or raise (call, then raise by the pot), and
  all-in (FCPA), the small set the LBR paper uses for no-limit. At 10 bb LBR also runs with fold
  and all-in only (FA), where all-in facing a shove means calling it: that is exactly the
  push/fold game of P3.1, whose exploitability is known in closed form.
* **Range tracking.** Weights over the 1,326 combos, uniform over those not blocked by LBR's
  cards and the board, multiplied after every opponent action by the strategy's probability of
  it (``RangeModel``). The opponent's past views and its view after a bet LBR considers are
  rebuilt from the public log (``state.py``, checked view by view against the simulator).
  Strategies that only sample a decision (the archetype ``ParametricBot``, and
  ``PotOddsStrategy``, whose equity is a Monte Carlo estimate) get vectorized models that average
  over their noise (normal approximation to the estimate; bet-size noise at its median). A model
  error can only weaken LBR, never inflate it: LBR is scored by playing the real opponent.
* **Equity.** Preflop, P3.1's class-against-class table. From the flop on, one table of every
  combo's rank on every runout of the board: on the flop a sample of 300 of the 1,176 runouts
  (seeded by the board), on the turn all 48, on the river the one board. Win probability against
  a range, and every combo's equity against a random hand (what the bots estimate), come from
  that table by sorting, with no 1,326 x 1,326 comparison.
* **Sampling.** Duplicate deals (each deal twice, seats swapped), an all-in before the river
  scored at its expectation over the remaining board (exact from the flop, 1,000 boards seeded
  by the cards preflop), the deal mean as the unit, and a normal 95% interval. Rows against the
  same opponent share their deals, so ablations are paired.
* **Cost.** About 0.2 s per flop table and 0.1 s for equity against a random hand on it (once
  per board, shared by the two plays of a deal), 40 ms on the turn, 2 ms on the river; one
  worker holds about 150 MB. Four workers by default.

Rows: at 100 bb with FCPA, always check/call, check/fold, uniform random, pot odds (the play
page's advisor baseline) and the five archetype bots at their mean parameters; at 10 bb with
FA, push/fold Nash and three threshold strategies with known exact values; at 10 bb with FCPA,
push/fold Nash with a pot-odds fallback off its tree. Two paired ablations against pot odds and
the TAG bot: no range tracking, and all 1,176 flop runouts.

Development (seeds 98 and 99, 2,000 deep and 4,000 shallow deals per row): check/fold 750 and
750; push/fold Nash -0 [-2, 1] and +1 [-1, 2]; the three threshold strategies 484, 127, 180 and
465, 141, 179 against exact 473, 151, 197 (z from -1.8 to +0.8; a 30,000-deal rerun of the
15/60 row gave 202.7 +- 8.4 and a direct simulation of LBR's decisions with real cards 197.3 +-
5.7, so the low values were sampling noise, not a bias); always call 16.8k and 16.6k; uniform
random 6.0k and 5.9k; pot odds 6.9k and 7.1k; bots: nit 0.9k, TAG 1.3k and 1.4k, LAG 1.75k and
1.7k, maniac 5.4k, calling station 6.0k and 5.7k; push/fold Nash with fallback, LBR with four
actions 335 and 271. No range tracking: -2.7k and -3.7k (pot odds), -1.9k and -1.6k (TAG); all
flop runouts: +42 and -1 (pot odds), +199 and +56 (TAG). Seed 98 took 1,136 s on 4 workers (seed
99's wall time includes the laptop sleeping and is not a timing).

Pre-registered: the checks C1-C4 test the code; the predictions P1-P10 (``PREDICTIONS``) use
thresholds set from the two development seeds, so the real run on seed 2026 is a replication
rather than a strong test. Everything is mbb/hand; every LBR value is a lower bound.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments import phase2_duplicate as pd
from experiments.common import REPO_ROOT, RESULTS_ROOT, load_config, run_metadata, save_json

DEEP, SHALLOW = "deep", "shallow"
FCPA = ("fold", "call", "pot", "allin")
FA = ("fold", "allin")
BOTS = ("nit", "tag", "lag", "calling_station", "maniac")
#: push/fold ranges by equity against a random hand: (push share, call share) of the 1,326 combos
PF_RANGES = {
    "pf_any_two": (1.0, 1.0),
    "pf_push50_call20": (0.5, 0.2),
    "pf_push15_call60": (0.15, 0.6),
}


def rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Every measured row: opponent, depth, LBR's action set and options, deals."""
    deep = cfg["deep_deals"]
    out = [
        {"key": k, "opponent": k, "depth": DEEP, "actions": FCPA, "deals": deep}
        for k in ("check_call", "check_fold", "uniform_random", "pot_odds")
    ]
    out += [
        {"key": f"bot_{b}", "opponent": f"bot_{b}", "depth": DEEP, "actions": FCPA, "deals": deep}
        for b in BOTS
    ]
    shallow = cfg["shallow_deals"]
    out += [
        {"key": k, "opponent": k, "depth": SHALLOW, "actions": FA, "deals": shallow}
        for k in ("pf_nash", *PF_RANGES)
    ]
    out.append(
        {
            "key": "pf_nash_fallback_fcpa",
            "opponent": "pf_nash_fallback",
            "depth": SHALLOW,
            "actions": FCPA,
            "deals": shallow,
        }
    )
    for opp in cfg["ablation_opponents"]:
        n = cfg["ablation_deals"]
        out.append(
            {
                "key": f"norange_{opp}",
                "opponent": opp,
                "depth": DEEP,
                "actions": FCPA,
                "deals": n,
                "track_range": False,
            }
        )
        out.append(
            {
                "key": f"exactflop_{opp}",
                "opponent": opp,
                "depth": DEEP,
                "actions": FCPA,
                "deals": n,
                "flop_sample": None,
            }
        )
    return out


def _top(share: float) -> np.ndarray:
    from poker_lab.games.preflop import combo_counts, load_matchups

    m = load_matchups()
    order = np.argsort(-m.equity_vs_random(), kind="stable")
    cum = np.cumsum(combo_counts()[order]) / 1326
    p = np.zeros(169)
    p[order[cum <= share + 1e-9]] = 1.0
    return p


def push_fold_ranges(name: str):
    from poker_lab.strategy.push_fold import fixed_ranges, nash_ranges

    if name in ("pf_nash", "pf_nash_fallback"):
        return nash_ranges
    push, call = PF_RANGES[name]
    return fixed_ranges(_top(push), _top(call))


def opponent(name: str):
    """(agent factory, range model) for an opponent name; built inside the worker."""
    from poker_lab.bots import ParametricBot
    from poker_lab.bots.archetypes import Archetype, mean_params
    from poker_lab.exploit.lbr.models import (
        ParametricBotModel,
        PotOddsModel,
        PushFoldModel,
        StrategyRangeModel,
    )
    from poker_lab.strategy import (
        CheckCallStrategy,
        CheckFoldStrategy,
        PotOddsStrategy,
        PushFoldStrategy,
        StrategyAgent,
        UniformRandomStrategy,
    )

    simple = {
        "check_call": CheckCallStrategy,
        "check_fold": CheckFoldStrategy,
        "uniform_random": UniformRandomStrategy,
    }
    if name in simple:
        cls = simple[name]
        return (lambda: StrategyAgent(cls())), StrategyRangeModel(cls(), card_blind=True)
    if name == "pot_odds":
        return (lambda: StrategyAgent(PotOddsStrategy(seed=1))), PotOddsModel()
    if name.startswith("bot_"):
        params = mean_params(Archetype(name[4:]))
        return (lambda: ParametricBot(params)), ParametricBotModel(params, name=name)
    ranges = push_fold_ranges(name)
    if name == "pf_nash_fallback":

        def make():
            return StrategyAgent(PushFoldStrategy(ranges, fallback=PotOddsStrategy(seed=1)))

        return make, PushFoldModel(PushFoldStrategy(ranges), fallback=PotOddsModel())
    return (lambda: StrategyAgent(PushFoldStrategy(ranges))), PushFoldModel(
        PushFoldStrategy(ranges)
    )


def exact_value(name: str, stack: float) -> float | None:
    """Exact exploitability (mbb/hand) of a push/fold row in the push/fold game (P3.1)."""
    if name not in ("pf_nash", *PF_RANGES):
        return None
    from poker_lab.games.push_fold import strategy_exploitability

    push, call = push_fold_ranges(name)(stack)
    return 1000.0 * float(strategy_exploitability(stack, np.asarray(push), np.asarray(call)))


def chunk_seed(master: int, opponent_name: str, chunk: int) -> int:
    """Rows against the same opponent share their deals chunk by chunk (paired ablations)."""
    word = int.from_bytes(opponent_name.encode(), "little") % (2**31)
    return int(np.random.SeedSequence([master, word, chunk]).generate_state(1)[0])


def task(row: dict[str, Any], chunk: int, n_deals: int, master: int, cfg) -> dict[str, Any]:
    from poker_lab.exploit.lbr.agent import LBRConfig, LocalBestResponse
    from poker_lab.exploit.lbr.match import lbr_match

    make_opp, model = opponent(row["opponent"])
    lbr_cfg = LBRConfig(
        actions=tuple(row["actions"]),
        flop_sample=row.get("flop_sample", cfg["flop_sample"]),
        track_range=row.get("track_range", True),
    )
    stack = cfg["deep_stack"] if row["depth"] == DEEP else cfg["shallow_stack"]
    r = lbr_match(
        make_opp,
        lambda: LocalBestResponse(model, lbr_cfg),
        n_deals=n_deals,
        seed=chunk_seed(master, row["opponent"], chunk),
        stack_bb=stack,
    )
    return {
        "key": row["key"],
        "chunk": chunk,
        "adjusted": (r.per_deal * 1000.0 / r.big_blind).tolist(),
        "raw": (r.per_deal_raw * 1000.0 / r.big_blind).tolist(),
        "seconds": r.seconds,
        "stats": r.stats,
    }


def _summary(x: np.ndarray) -> dict[str, float]:
    n = len(x)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    return {"mean": mean, "se": se, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se, "n": n,
            "sd": float(x.std(ddof=1))}  # fmt: skip


def aggregate(results: list[dict[str, Any]], cfg) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_key.setdefault(r["key"], []).append(r)
    out = {}
    for row in rows(cfg):
        parts = sorted(by_key[row["key"]], key=lambda r: r["chunk"])
        adj = np.concatenate([p["adjusted"] for p in parts])
        raw = np.concatenate([p["raw"] for p in parts])
        stack = cfg["deep_stack"] if row["depth"] == DEEP else cfg["shallow_stack"]
        stats = {k: sum(p["stats"][k] for p in parts) for k in parts[0]["stats"]}
        seconds = sum(p["seconds"] for p in parts)
        entry = {
            **{k: row[k] for k in ("opponent", "depth", "actions", "deals")},
            "stack_bb": stack,
            "track_range": row.get("track_range", True),
            "flop_sample": row.get("flop_sample", cfg["flop_sample"]),
            "lbr": _summary(adj),
            "raw": _summary(raw),
            "variance_ratio": float(raw.var(ddof=1) / adj.var(ddof=1)) if adj.var() > 0 else None,
            "exact": exact_value(row["opponent"], stack),
            "seconds": round(seconds, 1),
            "ms_per_hand": 1000.0 * seconds / (2 * len(adj)),
            "stats": stats,
            "surprise_share": stats["surprises"] / max(1, stats["opponent_actions"]),
            "_adj": adj,
        }
        if entry["exact"] is not None:
            entry["lbr_minus_exact"] = entry["lbr"]["mean"] - entry["exact"]
            entry["lbr_minus_exact_z"] = entry["lbr_minus_exact"] / entry["lbr"]["se"]
        out[row["key"]] = entry
    for row in rows(cfg):  # paired ablations: the same deals as the main row's first chunks
        if row["key"].startswith(("norange_", "exactflop_")):
            main = out[row["opponent"]]["_adj"]
            mine = out[row["key"]]["_adj"]
            d = mine - main[: len(mine)]
            out[row["key"]]["paired_difference"] = _summary(d)
    for entry in out.values():
        del entry["_adj"]
    return out


def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase2" / "lbr"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()
    size = cfg["chunk_deals"]
    jobs = []
    for row in rows(cfg):
        for c in range(0, row["deals"], size):
            jobs.append((row, c // size, min(size, row["deals"] - c), seed, cfg))
    # Slow rows first, so the pool does not end on one long job.
    jobs.sort(key=lambda j: (j[0]["depth"] != DEEP, j[0].get("flop_sample", 0) is not None))
    with ProcessPoolExecutor(max_workers=max(1, workers or 4)) as pool:
        futures = [pool.submit(task, *job) for job in jobs]
        results = []
        for k, f in enumerate(futures, 1):
            results.append(f.result())
            if k % 10 == 0 or k == len(futures):
                print(f"{k}/{len(futures)} chunks ({time.perf_counter() - t_start:.0f} s)",
                      flush=True)  # fmt: skip
    out = {
        "meta": meta,
        "rows": aggregate(results, cfg),
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    out["predictions"] = pd.predictions(out, PREDICTIONS)
    save_json(out, out_dir / "lbr.json")
    (out_dir / "lbr_summary.md").write_text(markdown_summary(out), encoding="utf-8", newline="\n")
    make_figures(out, out_dir)
    return out


# ---- predictions --------------------------------------------------------------------------


def _p(claim: str, values: list[tuple], test: tuple) -> dict[str, Any]:
    return {"claim": claim, "values": values, "test": test}


_DEEP_MAIN = ("check_call", "uniform_random", "pot_odds", *(f"bot_{b}" for b in BOTS))
_PF_FA = ("pf_nash", *PF_RANGES)


def _rows(keys, *path) -> list[tuple]:
    return [("rows", k, *path) for k in keys]


PREDICTIONS: dict[str, dict[str, Any]] = {
    # ---- checks: known in advance, they test the code ----
    "C1": _p(
        "Check: against check/fold LBR wins exactly the blinds, 750 mbb/hand (the most any "
        "strategy can win from it)",
        _rows(["check_fold"], "lbr", "mean"),
        ("between", [749.999, 750.001]),
    ),
    "C2": _p(
        "Check: in the push/fold game (LBR folds or goes all-in) each row is within 3 standard "
        "errors of P3.1's exact exploitability",
        _rows(_PF_FA, "lbr_minus_exact_z"),
        ("between", [-3.0, 3.0]),
    ),
    "C3": _p(
        "Check: against push/fold Nash LBR wins between -10 and +10 mbb/hand",
        _rows(["pf_nash"], "lbr", "mean"),
        ("between", [-10.0, 10.0]),
    ),
    "C4": _p(
        "Check: the range models never give probability 0 to an action the opponent took",
        _rows([*_DEEP_MAIN, "pf_nash_fallback_fcpa"], "surprise_share"),
        ("<=", 0.0),
    ),
    # ---- predictions, thresholds from seeds 98 and 99 ----
    "P1": _p(
        "Every deep strategy other than check/fold loses at least 500 mbb/hand to LBR (lower "
        "end of the 95% interval)",
        _rows(_DEEP_MAIN, "lbr", "lo"),
        (">=", 500.0),
    ),
    "P2": _p(
        "Always check/call is the most exploitable: LBR wins at least 14,000 mbb/hand",
        _rows(["check_call"], "lbr", "mean"),
        (">=", 14_000.0),
    ),
    "P3": _p(
        "The play page's advisor baseline (pot odds) loses 5,500 to 8,500 mbb/hand to LBR",
        _rows(["pot_odds"], "lbr", "mean"),
        ("between", [5_500.0, 8_500.0]),
    ),
    "P4": _p(
        "Uniform random loses 4,500 to 7,500 mbb/hand to LBR",
        _rows(["uniform_random"], "lbr", "mean"),
        ("between", [4_500.0, 7_500.0]),
    ),
    "P5": _p(
        "The bots order maniac > loose-aggressive > tight-aggressive > nit",
        _rows(["bot_maniac", "bot_lag", "bot_tag", "bot_nit"], "lbr", "mean"),
        ("decreasing", None),
    ),
    "P6": _p(
        "The loose, passive and wild bots (calling station, maniac) lose at least 4,000 "
        "mbb/hand; the nit, TAG and LAG at most 2,500",
        _rows(["bot_calling_station", "bot_maniac"], "lbr", "mean"),
        (">=", 4_000.0),
    ),
    "P6b": _p(
        "(second half of P6)",
        _rows(["bot_nit", "bot_tag", "bot_lag"], "lbr", "mean"),
        ("<=", 2_500.0),
    ),
    "P7": _p(
        "Without range tracking LBR wins less, on the same deals (upper end of the paired "
        "interval below 0)",
        _rows([f"norange_{o}" for o in ("pot_odds", "bot_tag")], "paired_difference", "hi"),
        ("<=", 0.0),
    ),
    "P8": _p(
        "Using all 1,176 flop runouts instead of 300 changes LBR's value by less than 800 "
        "mbb/hand on the same deals",
        _rows([f"exactflop_{o}" for o in ("pot_odds", "bot_tag")], "paired_difference", "mean"),
        ("between", [-800.0, 800.0]),
    ),
    "P9": _p(
        "Push/fold Nash with a pot-odds fallback is exploitable at 10 bb once LBR may limp or "
        "bet the pot: at least 150 mbb/hand (lower end of the interval)",
        _rows(["pf_nash_fallback_fcpa"], "lbr", "lo"),
        (">=", 150.0),
    ),
    "P10": _p(
        "Scoring all-ins at their expectation never increases the variance of a deep row",
        _rows(_DEEP_MAIN, "variance_ratio"),
        (">=", 1.0),
    ),
}


# ---- reporting ----------------------------------------------------------------------------

LABELS = {
    "check_call": "Always check or call",
    "check_fold": "Check or fold",
    "uniform_random": "Uniform random",
    "pot_odds": "Pot odds (play page's advisor baseline)",
    "bot_nit": "Bot: nit",
    "bot_tag": "Bot: tight-aggressive",
    "bot_lag": "Bot: loose-aggressive",
    "bot_calling_station": "Bot: calling station",
    "bot_maniac": "Bot: maniac",
    "pf_nash": "Push/fold Nash",
    "pf_any_two": "Push any two, call any two",
    "pf_push50_call20": "Push top 50%, call top 20%",
    "pf_push15_call60": "Push top 15%, call top 60%",
    "pf_nash_fallback_fcpa": "Push/fold Nash + pot-odds fallback (LBR with all four actions)",
}


def _fmt(s: dict[str, float]) -> str:
    return f"{s['mean']:,.0f} [{s['lo']:,.0f}, {s['hi']:,.0f}]"


def markdown_summary(r: dict[str, Any]) -> str:
    rws = r["rows"]
    lines = [
        "# P2.8b: local best response (lower bounds on exploitability)",
        "",
        f"Seed {r['meta']['seed']}, {r['elapsed_seconds']:.0f} s. Values in mbb/hand "
        "(1,000 = one big blind per hand), mean over duplicate deals with a 95% interval. LBR is "
        "a lower bound: the exploitability is at least this large, not equal to it.",
        "",
        "| Opponent | Stack | LBR actions | LBR value | Exact | Raw (no all-in adj.) | ms/hand |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, e in rws.items():
        if key.startswith(("norange_", "exactflop_")):
            continue
        exact = "" if e["exact"] is None else f"{e['exact']:,.0f}"
        lines.append(
            f"| {LABELS.get(key, key)} | {e['stack_bb']:g} bb | {'/'.join(e['actions'])} | "
            f"{_fmt(e['lbr'])} | {exact} | {_fmt(e['raw'])} | {e['ms_per_hand']:.0f} |"
        )
    lines += ["", "## Ablations (paired with the main row's deals)", "",
              "| Row | LBR value | Difference from the main row |", "|---|---|---|"]  # fmt: skip
    for key, e in rws.items():
        if key.startswith(("norange_", "exactflop_")):
            lines.append(f"| {key} | {_fmt(e['lbr'])} | {_fmt(e['paired_difference'])} |")
    if r.get("predictions"):
        lines += ["", "## Pre-registered predictions", "", "| | Claim | Result |", "|---|---|---|"]
        for k, p in r["predictions"].items():
            lines.append(f"| {k} | {p['claim']} | {'holds' if p['passed'] else 'FAILS'} |")
    return "\n".join(lines) + "\n"


def make_figures(r: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import new_figure, save_both, style_axes, title

    rws = r["rows"]
    deep = [k for k, e in rws.items() if e["depth"] == DEEP and not k.startswith(("norange_",
            "exactflop_"))]  # fmt: skip
    shallow = [k for k in ("pf_nash", *PF_RANGES) if k in rws]

    def values(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(11, 4.8), width_ratios=[1.5, 1])
        s = theme["series"]
        ax = axes[0]
        style_axes(ax, theme, grid_axis="x")
        ys = np.arange(len(deep))[::-1]
        for y, k in zip(ys, deep, strict=True):
            e = rws[k]["lbr"]
            err = [[(e["mean"] - e["lo"]) / 1000], [(e["hi"] - e["mean"]) / 1000]]
            ax.errorbar(e["mean"] / 1000, y, xerr=err, fmt="o", color=s[0], capsize=3)
        ax.set_yticks(ys, [LABELS.get(k, k) for k in deep], fontsize=8.5, color=theme["ink2"])
        ax.axvline(0, color=theme["axis"], linewidth=1)
        ax.set_xlabel("LBR's win, big blinds per hand (lower bound on exploitability)")
        ax.set_title("100 bb deep, LBR folds, calls, bets the pot or goes all-in",
                     color=theme["ink"], fontsize=10)  # fmt: skip
        ax = axes[1]
        style_axes(ax, theme, grid_axis="y")
        xs = np.arange(len(shallow))
        for x, k in zip(xs, shallow, strict=True):
            e = rws[k]
            ax.errorbar(x - 0.08, e["lbr"]["mean"], yerr=[[e["lbr"]["mean"] - e["lbr"]["lo"]],
                        [e["lbr"]["hi"] - e["lbr"]["mean"]]], fmt="o", color=s[0], capsize=3,
                        label="LBR" if x == 0 else None)  # fmt: skip
            ax.plot(x + 0.08, e["exact"], marker="D", color=s[1], linestyle="none",
                    label="Exact (P3.1)" if x == 0 else None)  # fmt: skip
        ax.set_xticks(xs, ["Nash", "Any two", "Push 50 /\ncall 20", "Push 15 /\ncall 60"],
                      fontsize=8.5, color=theme["ink2"])  # fmt: skip
        ax.set_ylabel("mbb/hand")
        ax.set_title("10 bb push/fold, LBR folds or goes all-in", color=theme["ink"], fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=theme["ink2"])
        title(
            fig,
            theme,
            "Local best response against hold'em strategies",
            "Mean over duplicate deals with 95% intervals; at 10 bb the exact value is known",
        )
        return fig

    save_both(values, out_dir, "lbr_values")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true", help="use configs/phase2_quick.yaml")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    name = "phase2_quick.yaml" if args.quick else "phase2.yaml"
    cfg = load_config(REPO_ROOT / "experiments" / "configs" / name)
    out = args.out or RESULTS_ROOT / cfg["name"] / "lbr"
    workers = args.workers or cfg["lbr"].get("workers")
    results = run(args.seed, cfg=cfg["lbr"], workers=workers, out_dir=out)
    print(f"wrote {out} in {results['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()

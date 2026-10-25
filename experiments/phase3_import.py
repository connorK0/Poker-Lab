"""P3.6: importing completed hand histories from PHH text files.

    python -m experiments.phase3_import --seed 2026

The importer (``poker_lab/importers/phh.py``) replays every hand through PokerKit's rules and
rejects, with a reason, anything that does not replay exactly. This experiment measures that in
four parts. Pre-registered plan (written before this run). Parts A and B and the two checks in
part C have not been run before. The import counts of part C were seen while the importer was
developed (a smoke run over the same files) and are reported as observed, not predicted.

A. **Round trip at scale.** ``round_trip_tables`` tables with random setups (2 to 9 players,
   gaps between seat numbers, blinds and antes, stacks from 1 to 400 big blinds, amounts in
   chips or in cents) play ``round_trip_hands`` hands each between random, calling, all-in
   and Phase 1 archetype bots. Every hand, and its copy redacted to what a participant could
   know, is exported to PHH text and imported again. Prediction: all of them come back
   identical up to card provenance (``as_imported``). Any difference is a bug and is listed.
B. **Fault injection.** ``mutation_hands`` hands from A and as many Pluribus hands from C get
   one random mutation per operator. Each mutated hand is imported twice: with its finishing
   stacks, and with them removed (rake not allowed: neither source has rake).
   *Structural* operators: drop a decision, swap two adjacent decisions of different players,
   cut the action list short, give a decision to another player, delete a TOML delimiter.
   *Value* operators: change a bet amount, a card, a starting stack, or (with finishing stacks
   only) move chips between two finishing stacks.
   Outcomes: rejected (with its reason); unchanged (the mutation had no effect; left out of
   the rates); record changed (accepted with a different record, but every player's result
   the same); result changed (accepted with some player's result different: a silent error).
   Predictions: (1) no import raises anything but a typed rejection; (2) every structural
   operator is rejected in at least 99% of its effective mutations, in both conditions and
   for both sources; (3) with finishing stacks, no accepted mutation changes any player's
   result. Reported without a prediction: how much is caught without finishing stacks.
C. **Public research data.** The PHH dataset at a pinned commit (``experiments/datasets.py``):
   Pluribus (10,000 hands), the WSOP 2023 Event #43 final table (83), five historical hands.
   Its HandHQ subset, scraped from online sites, is excluded. Imported without rake.
   Predictions: (1) PokerKit's own loader, with its standard hand evaluator and its repair
   logic, replays every accepted hand to the same result for every player; (2) exporting and
   re-importing reproduces every accepted hand. Reported: counts by subset and reason.
D. **What the imported hands show.** Per-player statistics of the Pluribus hands (redacted to
   public information; the Phase 1 stats code; 95% Wilson intervals) next to the Phase 1
   bots' measured statistics: a first look at how far the synthetic population is from strong
   human players. Descriptive only. P3.7 fits priors to real data.

Post hoc, after the first run (commit 582ac7c): that run falsified B1 (40 crashes, all from
card mutations that dealt a card twice; the importer was fixed and B rerun) and C1 (8 of
10,001 hands disagree). The disagreements are classified here: ``odd_chip`` when the two
results differ only in who gets a split pot's odd chip (the Pluribus files record half chips;
PokerKit's own loader gives the whole chip to one winner).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import re
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from pokerkit import HandHistory as PHH

from experiments.common import REPO_ROOT, RESULTS_ROOT, load_config, run_metadata, save_json
from experiments.datasets import PHH_DATASET_URL, content_sha256, phh_dataset
from poker_lab.bots import Archetype, ParametricBot, sample_params
from poker_lab.engine import AllInAgent, CallingAgent, RandomAgent, SeatedPlayer, TableSimulator
from poker_lab.evaluation.intervals import wilson_interval
from poker_lab.importers import (
    PHHImportError,
    as_imported,
    hand_from_phh,
    hand_to_phh,
    import_phh,
    loads_phh,
    phh_files,
)
from poker_lab.profiling.stats import C, counters_by_player
from poker_lab.schema import (
    FULL_DECK,
    HandHistory,
    TableConfig,
    redact_hole_cards,
    seat_order_from_button,
)

CONFIG = REPO_ROOT / "experiments" / "configs" / "phase3.yaml"
BOT_SANITY = RESULTS_ROOT / "phase1" / "bot_sanity" / "bot_sanity.json"
AGENT_KINDS = (
    "random",
    "aggressive",
    "calling",
    "all_in",
    *(a.value for a in Archetype),
)
STRUCTURAL = ("drop", "swap", "truncate", "wrong_player", "delimiter")
VALUE = ("amount", "card", "stack", "finishing")
OPERATORS = STRUCTURAL + VALUE
OPERATOR_LABELS = {
    "drop": "Drop a decision",
    "swap": "Swap two decisions",
    "truncate": "Cut the actions short",
    "wrong_player": "Decision by another player",
    "delimiter": "Delete a TOML delimiter",
    "amount": "Change a bet amount",
    "card": "Change a card",
    "stack": "Change a starting stack",
    "finishing": "Move chips between finishing stacks",
}
CONDITIONS = ("with_finishing", "without_finishing")
CONDITION_LABELS = {
    "with_finishing": "With finishing stacks",
    "without_finishing": "Without finishing stacks",
}
SOURCES = ("simulated", "pluribus")
SOURCE_LABELS = {"simulated": "Simulated hands (part A)", "pluribus": "Pluribus hands"}
OUTCOMES = ("rejected", "record_changed", "result_changed", "unchanged", "not_applicable", "crash")
#: Binomial stats compared with the Phase 1 bots, as (name, successes, trials).
D_STATS = (
    ("vpip", "vpip", "vpip_opp"),
    ("pfr", "pfr", "pfr_opp"),
    ("three_bet", "three_bet", "three_bet_opp"),
    ("cbet", "cbet", "cbet_opp"),
    ("fold_to_cbet", "fold_to_cbet", "fold_to_cbet_opp"),
    ("wtsd", "wtsd", "saw_flop"),
    ("wsd", "wsd", "wtsd"),
)
STAT_LABELS = {
    "vpip": "VPIP",
    "pfr": "PFR",
    "three_bet": "3-bet",
    "cbet": "C-bet",
    "fold_to_cbet": "Fold to c-bet",
    "wtsd": "WTSD",
    "wsd": "W$SD",
    "af": "AF",
}
DECISION = re.compile(r"^p(\d+) (f|cc|cbr)\b")
CARD = re.compile(r"[2-9TJQKA][cdhs]")


# ---------------------------------------------------------------------------- part A
def table_setup(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    n = rng.randint(2, 9)
    max_seats = rng.randint(n, 10)
    seats = sorted(rng.sample(range(max_seats), n))
    bb = rng.choice([2, 4, 10, 100])
    sb = bb if rng.random() < 0.1 else bb // 2
    ante = rng.choice([0, 0, 0, max(1, bb // 4), bb // 2])
    stacks = {s: max(1, round(bb * math.exp(rng.uniform(0, math.log(400))))) for s in seats}
    return {
        "seed": seed,
        "seats": seats,
        "max_seats": max_seats,
        "small_blind": sb,
        "big_blind": bb,
        "ante": ante,
        "stacks": stacks,
        "currency": rng.choice(["chips", "chips", "USD/100"]),
        "agents": [rng.choice(AGENT_KINDS) for _ in seats],
    }


def _agent(kind: str, rng: np.random.Generator):
    if kind == "random":
        return RandomAgent()
    if kind == "aggressive":
        return RandomAgent(0.3, 2.0)
    if kind == "calling":
        return CallingAgent()
    if kind == "all_in":
        return AllInAgent()
    return ParametricBot(sample_params(Archetype(kind), rng), equity_samples=50)


def play_table(setup: dict[str, Any], n_hands: int) -> list[HandHistory]:
    table = TableConfig(
        f"t{setup['seed']}",
        setup["max_seats"],
        setup["small_blind"],
        setup["big_blind"],
        setup["ante"],
        currency=setup["currency"],
    )
    rng = np.random.default_rng(setup["seed"])
    players = [
        SeatedPlayer(s, f"p{s}", _agent(kind, rng))
        for s, kind in zip(setup["seats"], setup["agents"], strict=True)
    ]
    sim = TableSimulator(table, players, seed=setup["seed"], starting_stack=setup["stacks"])
    return list(sim.play(n_hands))


def _first_difference(a: HandHistory, b: HandHistory) -> str:
    for f in dataclasses.fields(a):
        if getattr(a, f.name) != getattr(b, f.name):
            return f.name
    return "?"


def features(hand: HandHistory) -> list[str]:
    out = []
    if len(hand.seats) == 2:
        out.append("heads_up")
    if hand.table.ante:
        out.append("antes")
    if "/" in hand.table.currency:
        out.append("cents")
    all_in = {a.seat for a in hand.actions if a.is_all_in}
    if all_in:
        out.append("all_in")
    contributions = hand.contributions()
    if len({contributions[s] for s in all_in}) >= 2 and len(hand.payouts) >= 1:
        out.append("side_pot")
    if hand.showdown:
        out.append("showdown")
    return out


def round_trip_task(seed: int, n_hands: int) -> dict[str, Any]:
    setup = table_setup(seed)
    hands = play_table(setup, n_hands)
    out: dict[str, Any] = {
        "players": len(setup["seats"]),
        "checked": 0,
        "identical": 0,
        "features": Counter(),
        "failures": [],
        "texts": [],
        "seconds": 0.0,
    }
    for hand in hands:
        out["features"].update(features(hand))
        for redacted, h in ((False, hand), (True, redact_hole_cards(hand))):
            t0 = time.perf_counter()
            try:
                text = hand_to_phh(h).dumps()
                (phh,) = loads_phh(text)
                back = hand_from_phh(phh, allow_rake=False)
                problem = None if back == as_imported(h) else _first_difference(back, h)
            except Exception as exc:  # anything here is a bug: record it, keep going
                problem = f"{type(exc).__name__}: {exc}"
            out["seconds"] += time.perf_counter() - t0
            out["checked"] += 1
            if problem is None:
                out["identical"] += 1
            else:
                out["failures"].append({"hand": h.hand_id, "redacted": redacted, "why": problem})
        out["texts"].append((hand.hand_id, hand_to_phh(hand).dumps()))
    return out


def summarize_round_trip(parts: list[dict[str, Any]]) -> dict[str, Any]:
    by_players: dict[int, list[int]] = {}
    feats: Counter[str] = Counter()
    for p in parts:
        row = by_players.setdefault(p["players"], [0, 0])
        row[0] += p["checked"]
        row[1] += p["identical"]
        feats.update(p["features"])
    checked = sum(p["checked"] for p in parts)
    identical = sum(p["identical"] for p in parts)
    failures = [f for p in parts for f in p["failures"]]
    return {
        "tables": len(parts),
        "hands": checked // 2,
        "checked": checked,
        "identical": identical,
        "share": identical / checked if checked else None,
        "by_players": {
            str(k): {"checked": v[0], "identical": v[1]} for k, v in sorted(by_players.items())
        },
        "features": dict(feats),
        "failures": failures[:50],
        "n_failures": len(failures),
        "ms_per_round_trip": 1000 * sum(p["seconds"] for p in parts) / checked if checked else None,
        "holds": identical == checked,
    }


# ---------------------------------------------------------------------------- part B
def _player(action: str) -> int:
    return int(DECISION.match(action).group(1))


def _perturb(value: int | Decimal, factor: float) -> int | Decimal:
    """``value`` times ``factor`` at the same precision, at least one unit, never unchanged."""
    if isinstance(value, int):
        new = max(1, round(value * factor))
        return new if new != value else value + 1
    unit = Decimal(1).scaleb(min(0, value.as_tuple().exponent))
    new = max(unit, (value * Decimal(str(factor))).quantize(unit))
    return new if new != value else value + unit


def mutate(phh: PHH, op: str, rng: random.Random) -> PHH | None:
    """One random mutation of ``phh`` by ``op`` (not ``delimiter``), or None if it cannot
    apply."""
    actions = list(phh.actions)
    decisions = [i for i, a in enumerate(actions) if DECISION.match(a)]
    n = len(phh.starting_stacks)
    if op == "drop":
        if not decisions:
            return None
        del actions[rng.choice(decisions)]
    elif op == "swap":
        pairs = [
            i
            for i in decisions
            if i + 1 < len(actions)
            and DECISION.match(actions[i + 1])
            and _player(actions[i]) != _player(actions[i + 1])
        ]
        if not pairs:
            return None
        i = rng.choice(pairs)
        actions[i], actions[i + 1] = actions[i + 1], actions[i]
    elif op == "truncate":
        if len(actions) < 2:
            return None
        actions = actions[: rng.randrange(1, len(actions))]
    elif op == "wrong_player":
        if not decisions:
            return None
        i = rng.choice(decisions)
        p = _player(actions[i])
        other = rng.choice([k for k in range(1, n + 1) if k != p])
        actions[i] = f"p{other}" + actions[i][len(f"p{p}") :]
    elif op == "amount":
        bets = [i for i in decisions if actions[i].split()[1] == "cbr"]
        if not bets:
            return None
        i = rng.choice(bets)
        player, _, amount = actions[i].split()[:3]
        value = int(amount) if amount.isdigit() else Decimal(amount)
        actions[i] = f"{player} cbr {_perturb(value, rng.uniform(0.5, 2.0))}"
    elif op == "card":
        spots = [  # cards are the last word of a deal
            (i, m.start())
            for i, a in enumerate(actions)
            if a.startswith(("d dh ", "d db "))
            for m in CARD.finditer(a, a.rindex(" ") + 1)
        ]
        if not spots:
            return None
        i, at = rng.choice(spots)
        old = actions[i][at : at + 2]
        new = rng.choice([c for c in FULL_DECK if c != old])
        actions[i] = actions[i][:at] + new + actions[i][at + 2 :]
    elif op == "stack":
        stacks = list(phh.starting_stacks)
        j = rng.randrange(n)
        stacks[j] = _perturb(stacks[j], rng.uniform(0.5, 1.5))
        return dataclasses.replace(phh, starting_stacks=stacks)
    elif op == "finishing":
        if phh.finishing_stacks is None:
            return None
        fin = list(phh.finishing_stacks)
        donors = [i for i in range(n) if fin[i] > 0]
        if not donors:
            return None
        i = rng.choice(donors)
        j = rng.choice([k for k in range(n) if k != i])
        moved = min(fin[i], _perturb(fin[i], 0.1))
        fin[i] -= moved
        fin[j] += moved
        return dataclasses.replace(phh, finishing_stacks=fin)
    else:
        raise ValueError(op)
    return dataclasses.replace(phh, actions=actions)


def delete_delimiter(text: str, rng: random.Random) -> str | None:
    spots = [m.start() for m in re.finditer(r"[=\[\]]", text)]
    if not spots:
        return None
    at = rng.choice(spots)
    return text[:at] + text[at + 1 :]


def _without_finishing_text(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith("finishing_stacks"))


def _import(obj: PHH | str, hand_id: str) -> tuple[str, HandHistory | str]:
    """("accepted", hand) | ("rejected", reason) | ("crash", message)."""
    try:
        if isinstance(obj, str):
            (obj,) = loads_phh(obj)
            if isinstance(obj, PHHImportError):
                raise obj
        return "accepted", hand_from_phh(obj, hand_id=hand_id, allow_rake=False)
    except PHHImportError as exc:
        return "rejected", exc.reason.value
    except Exception as exc:  # a crash is a bug: record it
        return "crash", f"{type(exc).__name__}: {exc}"


def fault_task(
    source: str, index: int, hand_id: str, text: str, seed: int
) -> list[tuple[str, str, str, str]]:
    """(operator, condition, outcome, reason or detail) for every operator and condition."""
    out = []
    (original,) = loads_phh(text)
    references = {
        "with_finishing": _import(original, hand_id),
        "without_finishing": _import(dataclasses.replace(original, finishing_stacks=None), hand_id),
    }
    for kind, ref in references.values():
        if kind != "accepted":
            raise RuntimeError(f"{hand_id}: the unmutated hand does not import ({ref})")
    for k, op in enumerate(OPERATORS):
        case_seed = int(
            np.random.SeedSequence([seed, SOURCES.index(source), index, k]).generate_state(1)[0]
        )
        for condition in CONDITIONS:
            rng = random.Random(case_seed)  # the same mutation in both conditions
            if op == "delimiter":
                base = text if condition == "with_finishing" else _without_finishing_text(text)
                mutated: PHH | str | None = delete_delimiter(base, rng)
            else:
                mutated = mutate(original, op, rng)
                if mutated is not None and condition == "without_finishing":
                    if op == "finishing":
                        mutated = None  # nothing left to mutate
                    else:
                        mutated = dataclasses.replace(mutated, finishing_stacks=None)
            if mutated is None:
                out.append((op, condition, "not_applicable", ""))
                continue
            kind, result = _import(mutated, hand_id)
            ref = references[condition][1]
            if kind != "accepted":
                out.append((op, condition, "rejected" if kind == "rejected" else "crash", result))
            elif result == ref:
                out.append((op, condition, "unchanged", ""))
            elif result.net_results() == ref.net_results():
                out.append((op, condition, "record_changed", _first_difference(result, ref)))
            else:
                out.append((op, condition, "result_changed", _first_difference(result, ref)))
    return out


def summarize_faults(rows: list[tuple[str, str, str, str, str]]) -> dict[str, Any]:
    """rows: (source, operator, condition, outcome, reason)."""
    out: dict[str, Any] = {}
    for source in SOURCES:
        out[source] = {}
        for condition in CONDITIONS:
            out[source][condition] = {}
            for op in OPERATORS:
                mine = [r for r in rows if r[:3] == (source, op, condition)]
                counts = Counter(r[3] for r in mine)
                effective = sum(
                    counts[o] for o in ("rejected", "record_changed", "result_changed", "crash")
                )
                block: dict[str, Any] = {o: counts[o] for o in OUTCOMES}
                block["effective"] = effective
                block["reasons"] = dict(Counter(r[4] for r in mine if r[3] == "rejected"))
                block["changed_fields"] = dict(
                    Counter(r[4] for r in mine if r[3] in ("record_changed", "result_changed"))
                )
                for o in ("rejected", "record_changed", "result_changed"):
                    iv = wilson_interval(counts[o], effective) if effective else None
                    block[f"{o}_share"] = (
                        {"value": iv.estimate, "lower": iv.lower, "upper": iv.upper}
                        if iv is not None
                        else None
                    )
                out[source][condition][op] = block
    return out


def fault_predictions(faults: dict[str, Any], crashes: int) -> dict[str, Any]:
    structural = {
        f"{source}/{condition}/{op}": faults[source][condition][op]["rejected_share"]["value"]
        for source in SOURCES
        for condition in CONDITIONS
        for op in STRUCTURAL
        if faults[source][condition][op]["effective"]
    }
    silent = {
        f"{source}/{op}": faults[source]["with_finishing"][op]["result_changed"]
        for source in SOURCES
        for op in OPERATORS
    }
    return {
        "no_crashes": {"crashes": crashes, "holds": crashes == 0},
        "structural_rejected": {
            "bound": 0.99,
            "values": structural,
            "minimum": min(structural.values()) if structural else None,
            "holds": bool(structural) and min(structural.values()) >= 0.99,
        },
        "no_silent_with_finishing": {
            "values": silent,
            "total": sum(silent.values()),
            "holds": sum(silent.values()) == 0,
        },
    }


# ---------------------------------------------------------------------------- part C
def subset_of(key: str) -> str:
    head = key.split("/", 1)[0]
    return head if "/" in key else "historical"


def _lenient_net(text: str, scale: int) -> list[int]:
    """Every player's result by PokerKit's own loader (standard evaluator, repairs)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        *_, final = iter(PHH.loads(text))
    return [int(Decimal(p) * scale) for p in final.payoffs]


def _odd_chip(ours: list[int], theirs: list[int], scale: int) -> bool:
    """Post hoc: the results differ only in how a split pot's odd chip is shared (at most one
    chip per player, between players who won something, summing to zero)."""
    diffs = [a - b for a, b in zip(ours, theirs, strict=True)]
    return (
        sum(diffs) == 0
        and all(abs(d) <= scale for d in diffs)
        and all(d == 0 or o > 0 for d, o in zip(diffs, ours, strict=True))
    )


def dataset_task(path: str, root: str) -> dict[str, Any]:
    report = import_phh([path], root=root, allow_rake=False)
    agree = fixed = 0
    disagreements = []
    for hand in report.hands:
        text = (Path(root) / hand.metadata["import.file"]).read_text(encoding="utf-8")
        _, _, scale = hand.table.currency.partition("/")
        order = seat_order_from_button([s.seat for s in hand.seats], hand.button_seat)
        net = hand.net_results()
        ours = [net[s] for s in order]
        theirs = _lenient_net(text, int(scale or 1))
        if ours == theirs:
            agree += 1
        else:
            disagreements.append(
                {
                    "hand": hand.hand_id,
                    "importer": ours,
                    "pokerkit": theirs,
                    "odd_chip": _odd_chip(ours, theirs, int(scale or 1)),
                }
            )
        (again,) = loads_phh(hand_to_phh(hand).dumps())
        fixed += hand_from_phh(again, allow_rake=False) == hand
    pluribus = [h for h in report.hands if subset_of(h.metadata["import.file"]) == "pluribus"]
    fractional = sum("/" in h.table.currency for h in pluribus)
    counters = counters_by_player(redact_hole_cards(h) for h in pluribus)
    return {
        "accepted": [(h.hand_id, h.metadata["import.file"]) for h in report.hands],
        "players": [len(h.seats) for h in report.hands],
        "rejected": [dataclasses.asdict(r) for r in report.rejected],
        "files": report.files,
        "agree": agree,
        "fixed": fixed,
        "disagreements": disagreements,
        "fractional": fractional,
        "counters": {k: v.tolist() for k, v in counters.items()},
    }


def summarize_dataset(parts: list[dict[str, Any]], files: list[tuple[Path, str]]) -> dict[str, Any]:
    accepted = [a for p in parts for a in p["accepted"]]
    rejected = [r for p in parts for r in p["rejected"]]
    by_subset: dict[str, dict[str, Any]] = {}
    for _, key in files:
        by_subset.setdefault(subset_of(key), {"files": 0, "accepted": 0, "rejected": Counter()})
        by_subset[subset_of(key)]["files"] += 1
    for _, file in accepted:
        by_subset[subset_of(file)]["accepted"] += 1
    for r in rejected:
        by_subset[subset_of(r["file"])]["rejected"][r["reason"]] += 1
    details = Counter(
        (subset_of(r["file"]), r["reason"], _detail_kind(r["detail"])) for r in rejected
    )
    agree = sum(p["agree"] for p in parts)
    fixed = sum(p["fixed"] for p in parts)
    return {
        "files": len(files),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "by_subset": {
            k: {**v, "rejected": dict(v["rejected"])} for k, v in sorted(by_subset.items())
        },
        "rejection_kinds": [
            {"subset": s, "reason": r, "kind": d, "count": c}
            for (s, r, d), c in details.most_common()
        ],
        "players": dict(Counter(n for p in parts for n in p["players"])),
        "cross_check": {"agree": agree, "of": len(accepted), "holds": agree == len(accepted)},
        "fixed_point": {"same": fixed, "of": len(accepted), "holds": fixed == len(accepted)},
        "disagreements": [d for p in parts for d in p["disagreements"]][:20],
        "disagreement_kinds": {
            "post_hoc": True,
            "odd_chip": sum(d["odd_chip"] for p in parts for d in p["disagreements"]),
            "other": sum(not d["odd_chip"] for p in parts for d in p["disagreements"]),
        },
        "fractional_chip_hands": sum(p["fractional"] for p in parts),
    }


def _detail_kind(detail: str) -> str:
    """A rejection detail without its numbers, for grouping."""
    if detail.startswith("variant"):
        return detail.split(" (")[0]
    if detail.startswith("antes differ"):
        return "antes differ by seat (big-blind ante)"
    return re.sub(r"\d+", "#", detail)[:80]


# ---------------------------------------------------------------------------- part D
def player_stats(counters: dict[str, list[int]]) -> dict[str, Any]:
    rows = {}
    pooled = np.zeros(len(next(iter(counters.values()))), dtype=np.int64)
    for pid, c in counters.items():
        if pid != "Pluribus":
            pooled += np.asarray(c)
    named = {**counters, "All humans": pooled.tolist()}
    for pid, c in sorted(named.items(), key=lambda kv: -kv[1][C["hands"]]):
        c = np.asarray(c)
        row: dict[str, Any] = {"hands": int(c[C["hands"]])}
        for name, k, m in D_STATS:
            iv = wilson_interval(int(c[C[k]]), int(c[C[m]]))
            row[name] = {
                "value": 100 * iv.estimate,
                "lower": 100 * iv.lower,
                "upper": 100 * iv.upper,
                "n": int(c[C[m]]),
            }
        calls = int(c[C["pf_calls"]])
        row["af"] = {"value": c[C["pf_aggr"]] / calls if calls else None}
        rows[pid] = row
    return rows


def phase1_bots() -> dict[str, Any]:
    sanity = json.loads(BOT_SANITY.read_text(encoding="utf-8"))
    keep = [name for name, _, _ in D_STATS] + ["af"]
    return {
        "source": BOT_SANITY.relative_to(REPO_ROOT).as_posix(),
        "git": sanity["meta"]["git"],
        "per_archetype": {
            arch: {
                s: {k: v[k] for k in ("estimate", "lower", "upper")}
                for s, v in stats.items()
                if s in keep
            }
            for arch, stats in sanity["per_archetype"].items()
        },
    }


# ---------------------------------------------------------------------------- run
def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase3" / "import"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()
    seeds = np.random.SeedSequence([seed, 36]).generate_state(3)
    table_seeds = [
        int(s)
        for s in np.random.SeedSequence(int(seeds[0])).generate_state(cfg["round_trip_tables"])
    ]
    data = phh_dataset(cfg["dataset_commit"])
    paths = [data / p for p in cfg["dataset_paths"]]
    files = phh_files(paths, root=data)
    tasks = []
    for p in paths:
        if p.is_dir() and (p / "100").is_dir():  # one task per Pluribus folder
            tasks += sorted(str(q) for q in p.iterdir() if q.is_dir())
        else:
            tasks.append(str(p))
    with ProcessPoolExecutor(max_workers=workers or cfg.get("workers")) as pool:
        trips = list(
            pool.map(round_trip_task, table_seeds, [cfg["round_trip_hands"]] * len(table_seeds))
        )
        t_a = time.perf_counter()
        parts = list(pool.map(dataset_task, tasks, [str(data)] * len(tasks)))
        t_c = time.perf_counter()
        rng = random.Random(int(seeds[1]))
        simulated = [t for p in trips for t in p["texts"]]
        pluribus = sorted(a for p in parts for a in p["accepted"] if subset_of(a[1]) == "pluribus")
        k = cfg["mutation_hands"]
        chosen = {
            "simulated": rng.sample(simulated, min(k, len(simulated))),
            "pluribus": [
                (hid, (data / f).read_text(encoding="utf-8"))
                for hid, f in rng.sample(pluribus, min(k, len(pluribus)))
            ],
        }
        jobs = [
            (source, i, hid, text, int(seeds[2]))
            for source in SOURCES
            for i, (hid, text) in enumerate(chosen[source])
        ]
        results = list(pool.map(fault_task, *zip(*jobs, strict=True), chunksize=8))
        t_b = time.perf_counter()
    rows = [(job[0], *r) for job, res in zip(jobs, results, strict=True) for r in res]
    crashes = [r for r in rows if r[3] == "crash"]
    faults = summarize_faults(rows)
    counters: dict[str, list[int]] = {}
    for p in parts:
        for pid, c in p["counters"].items():
            counters[pid] = (np.asarray(counters.get(pid, np.zeros_like(c))) + c).tolist()
    dataset = summarize_dataset(parts, files)
    results_out = {
        "meta": meta,
        "dataset": {
            "url": PHH_DATASET_URL,
            "commit": cfg["dataset_commit"],
            "paths": cfg["dataset_paths"],
            "files": len(files),
            "content_sha256": content_sha256(files),
        },
        "round_trip": summarize_round_trip(trips),
        "faults": faults,
        "fault_hands": {s: len(chosen[s]) for s in SOURCES},
        "crashes": crashes[:20],
        "fault_predictions": fault_predictions(faults, len(crashes)),
        "public_data": dataset,
        "pluribus_stats": player_stats(counters) if counters else {},
        "phase1_bots": phase1_bots(),
        "seconds": {
            "round_trip": round(t_a - t_start, 1),
            "public_data": round(t_c - t_a, 1),
            "faults": round(t_b - t_c, 1),
        },
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    save_json(results_out, out_dir / "import.json")
    (out_dir / "import_summary.md").write_text(
        markdown_summary(results_out), encoding="utf-8", newline="\n"
    )
    make_figures(results_out, out_dir)
    return results_out


def _verdict(holds: bool) -> str:
    return "holds" if holds else "**does not hold**"


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def markdown_summary(r: dict[str, Any]) -> str:
    rt, fp, pd = r["round_trip"], r["fault_predictions"], r["public_data"]
    meta = r["meta"]
    lines = [
        "# P3.6: importing PHH hand histories",
        "",
        f"Seed {meta['seed']}, commit {meta['git']['commit']} (dirty: {meta['git']['dirty']}), "
        f"{r['elapsed_seconds']:.0f} s. Dataset {r['dataset']['url']} at "
        f"{r['dataset']['commit'][:7]}, {r['dataset']['files']} files, content SHA-256 "
        f"{r['dataset']['content_sha256'][:16]}...",
        "",
        "## A. Round trip at scale",
        "",
        f"{rt['hands']:,} hands at {rt['tables']} tables, each also redacted: "
        f"{rt['identical']:,} of {rt['checked']:,} round trips identical "
        f"({_pct(rt['share'], 2)}); {rt['ms_per_round_trip']:.1f} ms each. Prediction (all "
        f"identical): {_verdict(rt['holds'])}.",
        "",
        "Hands with: " + ", ".join(f"{k} {v:,}" for k, v in sorted(rt["features"].items())) + ".",
        "",
    ]
    if rt["failures"]:
        lines += ["Failures (first 50):", ""]
        lines += [f"* {f['hand']} (redacted: {f['redacted']}): {f['why']}" for f in rt["failures"]]
        lines.append("")
    lines += [
        "## B. Fault injection",
        "",
        f"{r['fault_hands']['simulated']:,} simulated and {r['fault_hands']['pluribus']:,} "
        "Pluribus hands, one mutation per operator. Rejected share of effective mutations, "
        "with and without finishing stacks; silent = accepted with a changed result.",
        "",
        "| Source | Operator | Rejected (with) | Record changed (with) | Silent (with) | "
        "Rejected (without) | Record changed (without) | Silent (without) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for source in SOURCES:
        for op in OPERATORS:
            w = r["faults"][source]["with_finishing"][op]
            wo = r["faults"][source]["without_finishing"][op]

            def share(block: dict[str, Any], outcome: str) -> str:
                s = block[f"{outcome}_share"]
                return (
                    "n/a"
                    if s is None
                    else f"{_pct(s['value'])} ({block[outcome]}/{block['effective']})"
                )

            lines.append(
                f"| {SOURCE_LABELS[source]} | {OPERATOR_LABELS[op]} | {share(w, 'rejected')} | "
                f"{share(w, 'record_changed')} | {w['result_changed']} | {share(wo, 'rejected')} | "
                f"{share(wo, 'record_changed')} | {share(wo, 'result_changed')} |"
            )
    lines += [
        "",
        f"* Prediction 1 (no crashes): {fp['no_crashes']['crashes']} crashes; "
        f"{_verdict(fp['no_crashes']['holds'])}.",
        f"* Prediction 2 (structural rejected >= 99%): minimum "
        f"{_pct(fp['structural_rejected']['minimum'])}; "
        f"{_verdict(fp['structural_rejected']['holds'])}.",
        f"* Prediction 3 (no silent result change with finishing stacks): "
        f"{fp['no_silent_with_finishing']['total']} silent; "
        f"{_verdict(fp['no_silent_with_finishing']['holds'])}.",
        "",
        "## C. Public research data",
        "",
        f"{pd['accepted']:,} of {pd['files']:,} files' hands accepted, {pd['rejected']} rejected.",
        "",
        "| Subset | Files | Accepted | Rejected (by reason) |",
        "|---|---|---|---|",
    ]
    for subset, v in pd["by_subset"].items():
        reasons = ", ".join(f"{k} {n}" for k, n in sorted(v["rejected"].items())) or "none"
        lines.append(f"| {subset} | {v['files']:,} | {v['accepted']:,} | {reasons} |")
    lines += ["", "Rejections by kind:", ""]
    lines += [
        f"* {k['subset']}, {k['reason']}: {k['kind']} ({k['count']})" for k in pd["rejection_kinds"]
    ]
    lines += [
        "",
        "* Prediction 1 (PokerKit's own loader agrees on every result): "
        f"{pd['cross_check']['agree']:,} of {pd['cross_check']['of']:,}; "
        f"{_verdict(pd['cross_check']['holds'])}. Post hoc: "
        f"{pd['disagreement_kinds']['odd_chip']} of the disagreements differ only in who gets "
        f"a split pot's odd chip, {pd['disagreement_kinds']['other']} otherwise; "
        f"{pd['fractional_chip_hands']} Pluribus hands record half chips.",
        "* Prediction 2 (export then import reproduces every hand): "
        f"{pd['fixed_point']['same']:,} of {pd['fixed_point']['of']:,}; "
        f"{_verdict(pd['fixed_point']['holds'])}.",
        "",
        "## D. Pluribus hands: per-player statistics (descriptive)",
        "",
        "| Player | Hands | " + " | ".join(STAT_LABELS[n] for n, _, _ in D_STATS) + " | AF |",
        "|---|---|" + "---|" * (len(D_STATS) + 1),
    ]
    for pid, row in r["pluribus_stats"].items():
        cells = [f"{row[n]['value']:.1f}" for n, _, _ in D_STATS]
        af = row["af"]["value"]
        lines.append(
            f"| {pid} | {row['hands']:,} | "
            + " | ".join(cells)
            + f" | {'n/a' if af is None else f'{af:.2f}'} |"
        )
    bots = r["phase1_bots"]["per_archetype"]
    lines += ["", "Phase 1 bots (measured, mean over bots):", ""]
    lines.append("| Archetype | " + " | ".join(STAT_LABELS[n] for n, _, _ in D_STATS) + " | AF |")
    lines.append("|---|" + "---|" * (len(D_STATS) + 1))
    for arch, stats in bots.items():
        cells = [f"{stats[n]['estimate']:.1f}" if n in stats else "n/a" for n, _, _ in D_STATS]
        lines.append(f"| {arch} | " + " | ".join(cells) + f" | {stats['af']['estimate']:.2f} |")
    return "\n".join(lines) + "\n"


def make_figures(r: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import legend, new_figure, save_both, style_axes, title

    def draw_faults(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(12.5, 5.2), sharey=True)
        s = theme["series"]
        colors = {"rejected": s[0], "record_changed": s[3], "result_changed": s[1]}
        labels = {
            "rejected": "Rejected",
            "record_changed": "Accepted, record changed",
            "result_changed": "Accepted, a result changed",
        }
        y = np.arange(len(OPERATORS))
        for ax, source in zip(axes, SOURCES, strict=True):
            style_axes(ax, theme, grid_axis="x")
            for offset, condition, alpha in (
                (-0.2, "with_finishing", 1.0),
                (0.2, "without_finishing", 0.55),
            ):
                left = np.zeros(len(OPERATORS))
                for outcome in ("rejected", "record_changed", "result_changed"):
                    widths = []
                    for op in OPERATORS:
                        b = r["faults"][source][condition][op]
                        widths.append(100 * b[outcome] / b["effective"] if b["effective"] else 0)
                    ax.barh(
                        y + offset,
                        widths,
                        left=left,
                        height=0.36,
                        color=colors[outcome],
                        alpha=alpha,
                        label=f"{labels[outcome]} ({CONDITION_LABELS[condition].lower()})",
                    )
                    left += np.array(widths)
            ax.set_yticks(y, [OPERATOR_LABELS[op] for op in OPERATORS])
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.set_xlabel("Share of effective mutations (%)")
            ax.set_title(SOURCE_LABELS[source], loc="left", fontsize=10, color=theme["ink"])
        legend(axes[1], theme, loc="lower right")
        title(
            fig,
            theme,
            "What the importer catches",
            "Upper bar of each pair: finishing stacks recorded; lower, lighter bar: removed.",
        )
        return fig

    save_both(draw_faults, out_dir, "import_faults")

    stats = r["pluribus_stats"]
    if not stats:
        return
    bots = r["phase1_bots"]["per_archetype"]
    shown = ("vpip", "pfr", "three_bet", "wtsd")

    def draw_stats(theme):
        fig, axes = new_figure(theme, 1, len(shown), figsize=(13, 5.2), sharey=True)
        s = theme["series"]
        players = list(stats)
        y = np.arange(len(players))
        for ax, name in zip(axes, shown, strict=True):
            style_axes(ax, theme, grid_axis="x")
            for k, arch in enumerate(("tag", "lag")):
                b = bots[arch][name]
                ax.axvspan(
                    b["lower"],
                    b["upper"],
                    color=(s[2], s[6])[k],
                    alpha=0.25,
                    label=f"Phase 1 {arch.upper()} bots",
                )
            v = [stats[p][name] for p in players]
            ax.errorbar(
                [x["value"] for x in v],
                y,
                xerr=[[x["value"] - x["lower"] for x in v], [x["upper"] - x["value"] for x in v]],
                fmt="o",
                color=s[0],
                markersize=4,
                elinewidth=1.2,
                label="Pluribus hands (95% Wilson)",
            )
            ax.set_yticks(y, [f"{p} ({stats[p]['hands']:,})" for p in players])
            ax.invert_yaxis()
            ax.set_xlabel(f"{STAT_LABELS[name]} (%)")
        legend(axes[0], theme, loc="lower right")
        title(
            fig,
            theme,
            "Pluribus and the professionals it played, against the Phase 1 bots",
            "Per-player statistics from the imported hands (public information only); "
            "bands: Phase 1 bots' 95% intervals.",
        )
        return fig

    save_both(draw_stats, out_dir, "import_pluribus")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)["import"]
    r = run(args.seed, cfg=cfg, workers=args.workers)
    print(markdown_summary(r))


if __name__ == "__main__":
    main()

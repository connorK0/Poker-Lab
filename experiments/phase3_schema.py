"""P3.10: schema v2 (big-blind antes, straddles, other minimum bets) and the migration of
version 1 files.

    python -m experiments.phase3_schema --seed 2026

Schema v2 adds three table rules: an ante the big blind posts for the whole table (modern
tournaments), straddles, and a minimum bet other than the big blind. Version 1 files are read by
migrating them. Pre-registered plan, written before this run. Nothing here has been run before
except the unit tests (``tests/test_schema_v2.py``: 120 hands at each of eight v2 tables, and
the 40 hands of a fixture written by the v1 code; ``tests/test_phh_import.py``).

A. **Round trips at v2 tables.** ``tables`` random tables in the style of P3.6 part A (2 to 9
   players, gaps between seats, blinds 1:2 or 1:1, stacks from 1 to 400 big blinds, chips or
   cents, random, aggressive, calling, all-in and archetype bots). Each gets one forced-bet
   structure from ``STRUCTURES``, in turn, so each structure gets the same number of tables:
   blinds only; an ante for every seat; a big-blind ante (half, one or one and a half big
   blinds); a straddle of two big blinds; a double straddle (two and four big blinds); a
   straddle on the button (four or more players); a minimum bet of two big blinds; one of half
   a big blind. ``hands`` hands each. Checks, on every hand (*full*) and on its copy redacted to
   what a participant could know (*redacted*):

   1. *rules*: ``find_problems`` finds nothing, including the new check that every forced post
      is the one the table's rules require (full, redacted);
   2. *PokerKit replay*: unmodified PokerKit, with its own hand evaluator, replays every action
      with the same results and showdown decisions (full);
   3. *JSON*: writing and reading back gives the same hand (full, redacted);
   4. *PHH*: export, then import, gives the same hand up to card provenance (full, redacted);
   5. *PokerKit loader*: PokerKit's own PHH loader, with its repairs, gives every player the
      same result from the exported text (full). A split pot's odd unit is the exception
      (*odd chip*): with amounts in cents the loader splits the pot exactly in decimals, while
      the engine plays in whole units and gives the odd one to one winner. A difference of less
      than one unit for every player counts as agreement and is reported separately. (Found in
      the development run of the quick configuration; P3.6 saw the same thing in the Pluribus
      files.)

   Prediction A1: every check passes on every hand, for every structure.
B. **Version 1 files.** The schema v1 code (commit ``v1_commit``, run in a temporary git
   worktree) writes JSON Lines of what it makes in P3.6: the 10,000 hands of P3.6 part A's 400
   tables (the same seeds) and the 10,000 Pluribus hands as it imports them. The current code
   reads those files, migrating every record, and makes the same hands itself. Prediction B1:
   every migrated record passes validation and equals the hand the current code makes (20,000
   of 20,000), and neither version rejects a Pluribus hand.
C. **The WSOP hold'em hands.** The 2023 WSOP Event #43 final table (83 hands) and the five
   historical hands at the pinned dataset commit. P3.6 rejected 11 WSOP hands for their
   big-blind ante; the other 72 WSOP hands and 4 historical hands are other variants.
   Predictions: C1 every no-limit hold'em hand imports (all 11 WSOP hands and the historical
   one), and every rejection left is a variant other than no-limit hold'em; C2 PokerKit's own
   loader gives every player the same result in every imported hand (up to an odd chip, as in
   check 5); C3 exporting and
   re-importing reproduces every imported hand. Reported: each WSOP hand's players, big-blind
   ante and pot in big blinds.

A failed prediction is reported as such, and every failing hand is listed. There are no figures:
every result is a count.

Post hoc, after the run at commit 4c4e373: A1 failed on 5 of 80,000 checks, all of them check 5
(the loader) on pots split three ways in cents, where the difference reaches 1.33 units.
PokerKit gives a split pot's whole remainder to the first winner in dealing order, so with
three winners and two odd cents one winner gets both; the loader splits decimal amounts
exactly. The odd-chip rule above assumed a difference below one unit. To test that this is the
only cause, every hand on which the loader disagreed (odd chips and failures) is checked again
with the same hand in whole units (currency ``chips``, so the loader splits in whole units
too): ``whole_units`` in the results. The pre-registered verdict is unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
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
from experiments.phase3_import import AGENT_KINDS, _agent, features, play_table, subset_of
from experiments.phase3_import import table_setup as v1_table_setup
from poker_lab.engine import SeatedPlayer, TableSimulator, replay_in_pokerkit
from poker_lab.importers import as_imported, hand_from_phh, hand_to_phh, import_phh, loads_phh
from poker_lab.importers.phh import phh_files
from poker_lab.schema import (
    HandHistory,
    TableConfig,
    dumps_hand,
    find_problems,
    loads_hand,
    read_jsonl,
    redact_hole_cards,
    seat_order_from_button,
)

CONFIG = REPO_ROOT / "experiments" / "configs" / "phase3.yaml"
STRUCTURES = (
    "blinds",
    "ante",
    "bb_ante",
    "straddle",
    "double_straddle",
    "button_straddle",
    "min_bet_up",
    "min_bet_down",
)
STRUCTURE_LABELS = {
    "blinds": "Blinds only",
    "ante": "Ante for every seat",
    "bb_ante": "Big-blind ante",
    "straddle": "Straddle",
    "double_straddle": "Double straddle",
    "button_straddle": "Straddle on the button",
    "min_bet_up": "Minimum bet 2 bb",
    "min_bet_down": "Minimum bet 0.5 bb",
}
MIN_PLAYERS = {"straddle": 3, "double_straddle": 4, "button_straddle": 4}
CHECKS = ("rules", "replay", "json", "phh", "loader")
CHECK_LABELS = {
    "rules": "Forced posts and consistency (find_problems)",
    "replay": "Unmodified PokerKit replays it",
    "json": "JSON round trip",
    "phh": "PHH export and import",
    "loader": "PokerKit's own PHH loader agrees",
}
VIEWS = {"full": CHECKS, "redacted": ("rules", "json", "phh")}
ODD_CHIP = "odd chip"


# ---------------------------------------------------------------------------- part A
def table_setup(seed: int, structure: str) -> dict[str, Any]:
    """A random table in the style of P3.6 part A, with ``structure``'s forced bets."""
    rng = random.Random(seed)
    n = rng.randint(MIN_PLAYERS.get(structure, 2), 9)
    max_seats = rng.randint(n, 10)
    seats = sorted(rng.sample(range(max_seats), n))
    bb = rng.choice([2, 4, 10, 100])
    sb = bb if rng.random() < 0.1 else bb // 2
    rules: dict[str, Any] = {}
    if structure == "ante":
        rules["ante"] = rng.choice([max(1, bb // 4), bb // 2])
    elif structure == "bb_ante":
        rules["big_blind_ante"] = rng.choice([bb // 2, bb, 3 * bb // 2])
    elif structure == "straddle":
        rules["straddles"] = [2 * bb]
    elif structure == "double_straddle":
        rules["straddles"] = [2 * bb, 4 * bb]
    elif structure == "button_straddle":
        rules["straddles"] = [0] * (n - 3) + [2 * bb]
    elif structure == "min_bet_up":
        rules["min_bet"] = 2 * bb
    elif structure == "min_bet_down":
        rules["min_bet"] = bb // 2
    stacks = {s: max(1, round(bb * float(np.exp(rng.uniform(0, np.log(400)))))) for s in seats}
    return {
        "seed": seed,
        "structure": structure,
        "seats": seats,
        "max_seats": max_seats,
        "small_blind": sb,
        "big_blind": bb,
        "rules": rules,
        "stacks": stacks,
        "currency": rng.choice(["chips", "chips", "USD/100"]),
        "agents": [rng.choice(AGENT_KINDS) for _ in seats],
    }


def play_v2_table(setup: dict[str, Any], n_hands: int) -> list[HandHistory]:
    rules = {**setup["rules"]}
    if "straddles" in rules:
        rules["straddles"] = tuple(rules["straddles"])
    table = TableConfig(
        f"t{setup['seed']}",
        setup["max_seats"],
        setup["small_blind"],
        setup["big_blind"],
        currency=setup["currency"],
        **rules,
    )
    rng = np.random.default_rng(setup["seed"])
    players = [
        SeatedPlayer(s, f"p{s}", _agent(kind, rng))
        for s, kind in zip(setup["seats"], setup["agents"], strict=True)
    ]
    sim = TableSimulator(table, players, seed=setup["seed"], starting_stack=setup["stacks"])
    return list(sim.play(n_hands))


def _ours(hand: HandHistory) -> list[int]:
    order = seat_order_from_button([s.seat for s in hand.seats], hand.button_seat)
    net = hand.net_results()
    return [net[s] for s in order]


def loader_check(text: str, hand: HandHistory) -> str | None:
    """None if PokerKit's own loader (standard evaluator, repairs) gives every player the same
    result from ``text``; ``ODD_CHIP`` if every difference is less than one unit (a decimal pot
    split exactly); else the two results."""
    _, _, scale = hand.table.currency.partition("/")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        *_, final = iter(PHH.loads(text))
    theirs = [Decimal(p) * int(scale or 1) for p in final.payoffs]
    ours = _ours(hand)
    if theirs == ours:
        return None
    if all(abs(o - t) < 1 for o, t in zip(ours, theirs, strict=True)):
        return ODD_CHIP
    return f"loader {[str(t) for t in theirs]} != {ours}"


def check(name: str, hand: HandHistory) -> str | None:
    """None if ``hand`` passes check ``name``, else what went wrong."""
    try:
        if name == "rules":
            problems = find_problems(hand)
            return "; ".join(problems[:2]) if problems else None
        if name == "replay":
            result = replay_in_pokerkit(hand)
            if result.net != hand.net_results():
                return f"PokerKit's results {result.net} != {hand.net_results()}"
            if result.showdown != {e.seat: e.shown for e in hand.showdown}:
                return "showdown decisions differ"
            return None
        if name == "json":
            return None if loads_hand(dumps_hand(hand), validate=True) == hand else "differs"
        text = hand_to_phh(hand).dumps()
        if name == "phh":
            (phh,) = loads_phh(text)
            back = hand_from_phh(phh, allow_rake=False)
            return None if back == as_imported(hand) else "differs"
        if name == "loader":
            return loader_check(text, hand)
    except Exception as exc:  # anything here is a bug: record it, keep going
        return f"{type(exc).__name__}: {exc}"
    raise ValueError(f"unknown check {name!r}")


def whole_units(hand: HandHistory, problem: str) -> str:
    """Post hoc: does the loader agree when the same hand is written in whole units? The kind
    of difference (odd chip or larger) and the answer, e.g. ``"odd chip/agrees"``."""
    kind = "odd chip" if problem == ODD_CHIP else "larger"
    if "/" not in hand.table.currency:
        return f"{kind}/already whole units"
    whole = dataclasses.replace(hand, table=dataclasses.replace(hand.table, currency="chips"))
    again = loader_check(hand_to_phh(whole).dumps(), whole)
    return f"{kind}/{'agrees' if again is None else 'differs'}"


def v2_features(hand: HandHistory) -> list[str]:
    out = features(hand)
    forced = [a for a in hand.actions if a.action_type.is_forced]
    if any(a.is_all_in for a in forced):
        out.append("all_in_post")
    if any(a.action_type.value == "post_straddle" for a in forced):
        out.append("straddle_posted")
    if hand.table.big_blind_ante and any(a.action_type.value == "post_ante" for a in forced):
        out.append("bb_ante_posted")
    return out


def round_trip_task(seed: int, structure: str, n_hands: int) -> dict[str, Any]:
    setup = table_setup(seed, structure)
    out: dict[str, Any] = {
        "structure": structure,
        "players": len(setup["seats"]),
        "hands": 0,
        "checks": Counter(),
        "passed": Counter(),
        "odd_chip": Counter(),
        "features": Counter(),
        "failures": [],
        "whole_units": Counter(),
        "seconds": 0.0,
    }
    t0 = time.perf_counter()
    try:
        hands = play_v2_table(setup, n_hands)
    except Exception as exc:  # the simulator refusing a v2 table is a failure of A1
        out["failures"].append({"table": seed, "why": f"{type(exc).__name__}: {exc}"})
        return out
    for hand in hands:
        out["hands"] += 1
        out["features"].update(v2_features(hand))
        for view_name, h in (("full", hand), ("redacted", redact_hole_cards(hand))):
            for name in VIEWS[view_name]:
                key = f"{view_name}/{name}"
                out["checks"][key] += 1
                problem = check(name, h)
                if name == "loader" and problem is not None:
                    out["whole_units"][whole_units(h, problem)] += 1
                if problem == ODD_CHIP:
                    out["odd_chip"][key] += 1
                if problem in (None, ODD_CHIP):
                    out["passed"][key] += 1
                else:
                    out["failures"].append({"hand": h.hand_id, "check": key, "why": problem})
    out["seconds"] = time.perf_counter() - t0
    return out


def summarize_round_trips(parts: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, dict[str, Any]] = {}
    for structure in STRUCTURES:
        mine = [p for p in parts if p["structure"] == structure]
        checks: Counter[str] = Counter()
        passed: Counter[str] = Counter()
        odd: Counter[str] = Counter()
        feats: Counter[str] = Counter()
        for p in mine:
            checks.update(p["checks"])
            passed.update(p["passed"])
            odd.update(p["odd_chip"])
            feats.update(p["features"])
        by[structure] = {
            "tables": len(mine),
            "hands": sum(p["hands"] for p in mine),
            "players": dict(Counter(p["players"] for p in mine)),
            "checks": {
                k: {"checked": checks[k], "passed": passed[k], "odd_chip": odd[k]}
                for k in sorted(checks)
            },
            "features": dict(feats),
            "failures": sum(len(p["failures"]) for p in mine),
        }
    checked = sum(p["checks"].total() for p in parts)
    passed = sum(p["passed"].total() for p in parts)
    failures = [f for p in parts for f in p["failures"]]
    return {
        "tables": len(parts),
        "hands": sum(p["hands"] for p in parts),
        "checked": checked,
        "passed": passed,
        "by_structure": by,
        "failures": failures[:50],
        "n_failures": len(failures),
        "ms_per_hand": 1000
        * sum(p["seconds"] for p in parts)
        / max(1, sum(p["hands"] for p in parts)),
        "holds": not failures and checked == passed,
        "whole_units": dict(sum((p["whole_units"] for p in parts), Counter())),
    }


# ---------------------------------------------------------------------------- part B
V1_SCRIPT = '''"""Runs in a worktree at the schema v1 commit and writes what that code makes."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def simulate(args):
    seed, n_hands, out = args
    from experiments.phase3_import import play_table, table_setup
    from poker_lab.schema import write_jsonl

    return write_jsonl(play_table(table_setup(seed), n_hands), Path(out) / f"sim-{seed}.jsonl.gz")


def load(args):
    k, files, root, out = args
    from poker_lab.importers import import_phh
    from poker_lab.schema import write_jsonl

    report = import_phh(files, root=root, allow_rake=False)
    write_jsonl(report.hands, Path(out) / f"phh-{k}.jsonl.gz")
    return [len(report.hands), len(report.rejected)]


if __name__ == "__main__":
    import poker_lab
    from poker_lab.schema import SCHEMA_VERSION

    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    here = Path(poker_lab.__file__).resolve()
    assert SCHEMA_VERSION == 1, SCHEMA_VERSION
    assert here.is_relative_to(Path.cwd().resolve()), here
    with ProcessPoolExecutor(job["workers"]) as pool:
        sims = list(pool.map(simulate, [(s, job["hands"], job["out"]) for s in job["seeds"]]))
        chunks = [(k, f, job["root"], job["out"]) for k, f in enumerate(job["chunks"])]
        loaded = list(pool.map(load, chunks))
    done = {"schema_version": SCHEMA_VERSION, "simulated": sims, "imported": loaded}
    Path(job["out"], "done.json").write_text(json.dumps(done), encoding="utf-8")
'''


def p36_table_seeds(seed: int, n: int) -> list[int]:
    """The table seeds of P3.6 part A (``phase3_import.run``)."""
    first = np.random.SeedSequence([seed, 36]).generate_state(3)[0]
    return [int(s) for s in np.random.SeedSequence(int(first)).generate_state(n)]


def _git(*args: str) -> None:
    subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True)


def write_v1_files(
    commit: str, seeds: list[int], n_hands: int, chunks: list[list[str]], root: Path, workers: int
) -> tuple[Path, dict[str, Any]]:
    """Run the v1 code at ``commit`` in a temporary worktree; return (output folder, summary).
    The caller deletes the folder's parent."""
    tmp = Path(tempfile.mkdtemp(prefix="poker_lab_v1_"))
    tree, out = tmp / "tree", tmp / "out"
    out.mkdir()
    _git("worktree", "add", "--detach", str(tree), commit)
    try:
        job = {
            "seeds": seeds,
            "hands": n_hands,
            "chunks": chunks,
            "root": str(root),
            "out": str(out),
            "workers": workers,
        }
        (tmp / "job.json").write_text(json.dumps(job), encoding="utf-8")
        (tmp / "write_v1.py").write_text(V1_SCRIPT, encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(tree)}
        subprocess.run(
            [sys.executable, str(tmp / "write_v1.py"), str(tmp / "job.json")],
            cwd=tree,
            env=env,
            check=True,
        )
    finally:
        _git("worktree", "remove", "--force", str(tree))
    done = json.loads((out / "done.json").read_text(encoding="utf-8"))
    return out, done


def _records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _compare(theirs_path: Path, ours: list[HandHistory]) -> dict[str, Any]:
    raw = _records(theirs_path)
    digest = hashlib.sha256(gzip.decompress(theirs_path.read_bytes())).hexdigest()  # no mtime
    try:
        theirs = list(read_jsonl(theirs_path, validate=True))
    except ValueError as exc:
        return {"records": len(raw), "identical": 0, "failures": [str(exc)[:200]], "sha": digest}
    failures = []
    if len(theirs) != len(ours):
        failures.append(f"{theirs_path.name}: {len(theirs)} v1 records, {len(ours)} hands now")
    identical = 0
    for a, b in zip(theirs, ours, strict=False):
        if a == b:
            identical += 1
        else:
            failures.append(f"{a.hand_id}: differs")
    return {
        "records": len(raw),
        "v1_records": sum(r["schema_version"] == 1 for r in raw),
        "identical": identical,
        "failures": failures[:10],
        "sha": digest,
    }


def compare_simulated(seed: int, n_hands: int, folder: str) -> dict[str, Any]:
    hands = play_table(v1_table_setup(seed), n_hands)
    out = _compare(Path(folder) / f"sim-{seed}.jsonl.gz", hands)
    out["features"] = Counter(f for h in hands for f in features(h))
    return out


def compare_imported(k: int, files: list[str], root: str, folder: str) -> dict[str, Any]:
    report = import_phh(files, root=root, allow_rake=False)
    out = _compare(Path(folder) / f"phh-{k}.jsonl.gz", list(report.hands))
    out["rejected_now"] = len(report.rejected)
    return out


def summarize_v1(sim: list[dict[str, Any]], imp: list[dict[str, Any]], done: dict) -> dict:
    def block(parts: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "records": sum(p["records"] for p in parts),
            "v1_records": sum(p.get("v1_records", 0) for p in parts),
            "identical": sum(p["identical"] for p in parts),
            "failures": [f for p in parts for f in p["failures"]][:20],
        }

    s, i = block(sim), block(imp)
    s["features"] = dict(sum((p["features"] for p in sim), Counter()))
    rejected_v1 = sum(r for _, r in done["imported"])
    rejected_now = sum(p["rejected_now"] for p in imp)
    files = hashlib.sha256("".join(p["sha"] for p in sim + imp).encode()).hexdigest()
    holds = (
        s["identical"] == s["records"] == s["v1_records"]
        and i["identical"] == i["records"] == i["v1_records"]
        and rejected_v1 == rejected_now == 0
    )
    return {
        "simulated": s,
        "imported": {**i, "rejected_by_v1": rejected_v1, "rejected_now": rejected_now},
        "v1_files_sha256": files,
        "holds": holds,
    }


# ---------------------------------------------------------------------------- part C
def _variant(path: Path) -> str | None:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8")).get("variant")
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return None


def public_hands(data: Path, public_paths: list[str]) -> dict[str, Any]:
    paths = [data / p for p in public_paths]
    files = phh_files(paths, root=data)
    variants = {key: _variant(path) for path, key in files}
    report = import_phh(paths, root=data, allow_rake=False)
    rows = []
    for hand in report.hands:
        key = hand.metadata["import.file"].removesuffix(".phh")
        text = (data / hand.metadata["import.file"]).read_text(encoding="utf-8")
        loader = loader_check(text, hand)
        (again,) = loads_phh(hand_to_phh(hand).dumps())
        bb = hand.table.big_blind
        rows.append(
            {
                "file": key,
                "subset": subset_of(key),
                "players": len(hand.seats),
                "big_blind_ante_bb": hand.table.big_blind_ante / bb,
                "ante_bb": hand.table.ante / bb,
                "pot_bb": sum(hand.contributions().values()) / bb,
                "showdown": bool(hand.showdown),
                "loader_agrees": loader in (None, ODD_CHIP),
                "odd_chip": loader == ODD_CHIP,
                "fixed_point": hand_from_phh(again, allow_rake=False) == hand,
            }
        )
    rejected = [
        {
            "file": r.file.removesuffix(".phh"),
            "subset": subset_of(r.file.removesuffix(".phh")),
            "reason": r.reason.value,
            "detail": r.detail,
            "variant": variants.get(r.file.removesuffix(".phh")),
        }
        for r in report.rejected
    ]
    nt = {k for k, v in variants.items() if v == "NT"}
    imported = {r["file"] for r in rows}
    by_subset: dict[str, dict[str, Any]] = {}
    for key, variant in variants.items():
        b = by_subset.setdefault(subset_of(key), {"files": 0, "nlhe": 0, "imported": 0})
        b["files"] += 1
        b["nlhe"] += variant == "NT"
        b["imported"] += key in imported
    variant_rejections = all(
        r["reason"] == "unsupported" and r["detail"].startswith("variant") for r in rejected
    )
    return {
        "files": len(files),
        "content_sha256": content_sha256(files),
        "by_subset": by_subset,
        "hands": rows,
        "rejected": rejected,
        "rejected_variants": dict(Counter(r["variant"] for r in rejected)),
        "c1": {
            "nlhe": len(nt),
            "imported": len(imported & nt),
            "only_variants_rejected": variant_rejections,
            "holds": imported == nt and variant_rejections,
        },
        "c2": {
            "agree": sum(r["loader_agrees"] for r in rows),
            "of": len(rows),
            "holds": all(r["loader_agrees"] for r in rows),
        },
        "c3": {
            "same": sum(r["fixed_point"] for r in rows),
            "of": len(rows),
            "holds": all(r["fixed_point"] for r in rows),
        },
    }


# ---------------------------------------------------------------------------- run
def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase3" / "schema"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    workers = workers or cfg.get("workers")
    t_start = time.perf_counter()
    state = np.random.SeedSequence([seed, 310]).generate_state(cfg["tables"])
    jobs = [(int(s), STRUCTURES[k % len(STRUCTURES)]) for k, s in enumerate(state)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        trips = list(
            pool.map(
                round_trip_task,
                [s for s, _ in jobs],
                [st for _, st in jobs],
                [cfg["hands"]] * len(jobs),
            )
        )
    t_a = time.perf_counter()

    data = phh_dataset(cfg["dataset_commit"])
    pluribus = phh_files([data / p for p in cfg["v1_pluribus_paths"]], root=data)
    keys = [str(path) for path, _ in pluribus]
    size = -(-len(keys) // cfg["v1_chunks"])
    chunks = [keys[i : i + size] for i in range(0, len(keys), size)]
    seeds = p36_table_seeds(seed, cfg["v1_tables"])
    folder, done = write_v1_files(cfg["v1_commit"], seeds, cfg["v1_hands"], chunks, data, workers)
    t_b1 = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            sim = list(
                pool.map(
                    compare_simulated,
                    seeds,
                    [cfg["v1_hands"]] * len(seeds),
                    [str(folder)] * len(seeds),
                )
            )
            imp = list(
                pool.map(
                    compare_imported,
                    range(len(chunks)),
                    chunks,
                    [str(data)] * len(chunks),
                    [str(folder)] * len(chunks),
                )
            )
    finally:
        shutil.rmtree(folder.parent, ignore_errors=True)
    v1 = summarize_v1(sim, imp, done)
    p36 = RESULTS_ROOT / "phase3" / "import" / "import.json"
    if p36.exists():  # the same tables as P3.6 part A: the same hands, so the same features
        published = json.loads(p36.read_text(encoding="utf-8"))
        same_run = published["meta"]["seed"] == seed and (
            published["meta"]["config"]["round_trip_tables"],
            published["meta"]["config"]["round_trip_hands"],
        ) == (cfg["v1_tables"], cfg["v1_hands"])
        if same_run:
            match = published["round_trip"]["features"] == v1["simulated"]["features"]
            v1["p36_features_match"] = match
    t_b = time.perf_counter()

    public = public_hands(data, cfg["public_paths"])
    t_c = time.perf_counter()
    rt = summarize_round_trips(trips)
    results = {
        "meta": meta,
        "dataset": {"url": PHH_DATASET_URL, "commit": cfg["dataset_commit"]},
        "structures": list(STRUCTURES),
        "round_trip": rt,
        "v1_files": {
            **v1,
            "commit": cfg["v1_commit"],
            "tables": len(seeds),
            "pluribus_files": len(keys),
        },
        "public": public,
        "predictions": {
            "A1": {"holds": rt["holds"]},
            "B1": {"holds": v1["holds"]},
            "C1": public["c1"],
            "C2": public["c2"],
            "C3": public["c3"],
        },
        "seconds": {
            "round_trips": round(t_a - t_start, 1),
            "v1_code": round(t_b1 - t_a, 1),
            "v1_compare": round(t_b - t_b1, 1),
            "public": round(t_c - t_b, 1),
        },
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    save_json(results, out_dir / "schema.json")
    (out_dir / "schema_summary.md").write_text(
        markdown_summary(results), encoding="utf-8", newline="\n"
    )
    return results


def _verdict(holds: bool) -> str:
    return "holds" if holds else "**does not hold**"


def markdown_summary(r: dict[str, Any]) -> str:
    m, rt, v1, pub, pr = r["meta"], r["round_trip"], r["v1_files"], r["public"], r["predictions"]
    git = m["git"]
    lines = [
        "# P3.10: schema v2 and the migration of version 1 files",
        "",
        f"Seed {m['seed']}, commit {git['commit']} (dirty: {git['dirty']}), "
        f"{r['elapsed_seconds']:.0f} s. Dataset {r['dataset']['url']} at "
        f"{r['dataset']['commit'][:7]}.",
        "",
        "## A. Round trips at v2 tables",
        "",
        f"{rt['hands']:,} hands at {rt['tables']} tables; {rt['passed']:,} of {rt['checked']:,} "
        f"checks passed; {rt['ms_per_hand']:.1f} ms per hand for all checks.",
        "",
        "| Structure | Tables | Hands | "
        + " | ".join(CHECK_LABELS[c] for c in CHECKS)
        + " | Redacted copies (rules, JSON, PHH) |",
        "|---|---|---|" + "---|" * (len(CHECKS) + 1),
    ]
    for st in r["structures"]:
        b = rt["by_structure"][st]
        cells = []
        for c in CHECKS:
            x = b["checks"].get(f"full/{c}", {"checked": 0, "passed": 0, "odd_chip": 0})
            odd = f" ({x['odd_chip']} odd chip)" if x["odd_chip"] else ""
            cells.append(f"{x['passed']:,}/{x['checked']:,}{odd}")
        red = [
            b["checks"].get(f"redacted/{c}", {"checked": 0, "passed": 0}) for c in VIEWS["redacted"]
        ]
        red_cell = f"{sum(x['passed'] for x in red):,}/{sum(x['checked'] for x in red):,}"
        lines.append(
            f"| {STRUCTURE_LABELS[st]} | {b['tables']} | {b['hands']:,} | "
            + " | ".join(cells)
            + f" | {red_cell} |"
        )
    feats = Counter()
    for b in rt["by_structure"].values():
        feats.update(b["features"])
    lines += [
        "",
        "Hands with: " + ", ".join(f"{k} {v:,}" for k, v in sorted(feats.items())) + ".",
        "",
        f"* Prediction A1 (every check passes): {_verdict(pr['A1']['holds'])}.",
    ]
    if rt["failures"]:
        lines += ["", "Failures (first 50):", ""]
        lines += [f"* {f}" for f in rt["failures"]]
    wu = rt.get("whole_units", {})
    if wu:
        lines += [
            "",
            "Post hoc: every hand on which PokerKit's loader disagreed, checked again with the "
            "same hand in whole units: "
            + ", ".join(f"{k} {v}" for k, v in sorted(wu.items()))
            + ".",
        ]
    s, i = v1["simulated"], v1["imported"]
    lines += [
        "",
        "## B. Version 1 files",
        "",
        f"Written by the schema v1 code at {v1['commit'][:7]} (SHA-256 of their contents "
        f"{v1['v1_files_sha256'][:12]}...).",
        "",
        "| Source | v1 records | Identical to the current code |",
        "|---|---|---|",
        f"| P3.6 part A tables ({v1['tables']}) | {s['records']:,} | {s['identical']:,} |",
        f"| Pluribus hands ({v1['pluribus_files']:,} files) | {i['records']:,} | "
        f"{i['identical']:,} |",
        "",
        f"Pluribus hands rejected: {i['rejected_by_v1']} by the v1 code, {i['rejected_now']} now. "
        + (
            "The simulated hands have the same feature counts as P3.6 part A's: "
            + ("yes" if v1["p36_features_match"] else "**no**")
            + "."
            if "p36_features_match" in v1
            else ""
        ),
        "",
        f"* Prediction B1 (every record migrates to the same hand): {_verdict(pr['B1']['holds'])}.",
    ]
    for f in s["failures"] + i["failures"]:
        lines.append(f"  * {f}")
    lines += [
        "",
        "## C. The WSOP hold'em hands",
        "",
        "| Subset | Files | No-limit hold'em | Imported |",
        "|---|---|---|---|",
    ]
    for sub, b in sorted(pub["by_subset"].items()):
        lines.append(f"| {sub} | {b['files']} | {b['nlhe']} | {b['imported']} |")
    lines += [
        "",
        "Rejected, by variant: "
        + ", ".join(f"{v} {n}" for v, n in sorted(pub["rejected_variants"].items(), key=str))
        + ".",
        "",
        "| Hand | Players | Big-blind ante (bb) | Pot (bb) | Showdown | Loader agrees | "
        "Export, import |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in pub["hands"]:
        lines.append(
            f"| {h['file']} | {h['players']} | {h['big_blind_ante_bb']:g} | {h['pot_bb']:.1f} | "
            f"{'yes' if h['showdown'] else 'no'} | {'yes' if h['loader_agrees'] else 'no'} | "
            f"{'same' if h['fixed_point'] else 'differs'} |"
        )
    c1, c2, c3 = pr["C1"], pr["C2"], pr["C3"]
    lines += [
        "",
        f"* Prediction C1 (every no-limit hold'em hand imports, only variants rejected): "
        f"{c1['imported']} of {c1['nlhe']}; {_verdict(c1['holds'])}.",
        f"* Prediction C2 (PokerKit's own loader agrees): {c2['agree']} of {c2['of']}; "
        f"{_verdict(c2['holds'])}.",
        f"* Prediction C3 (export then import reproduces every hand): {c3['same']} of {c3['of']}; "
        f"{_verdict(c3['holds'])}.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true", help="use configs/phase3_quick.yaml")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    path = CONFIG.with_name("phase3_quick.yaml") if args.quick else CONFIG
    cfg = load_config(path)["schema"]
    r = run(args.seed, cfg=cfg, workers=args.workers, out_dir=args.out)
    print(json.dumps(r["predictions"], indent=1))


if __name__ == "__main__":
    main()

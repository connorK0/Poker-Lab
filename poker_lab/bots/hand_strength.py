"""Hand-strength measures used by the archetype bots.

Preflop: each of the 169 starting-hand classes (``"AA"``, ``"AKs"``, ``"72o"``, ...) is ranked by
its all-in equity against one uniformly random hand. ``preflop_percentile`` maps a hand to the
fraction of all 1,326 combos that are at least as strong. A bot with range width ``w`` plays a
hand iff its percentile is ``<= w``. The equity table is generated once by
``compute_preflop_table`` (Monte Carlo, fixed seed) and shipped as ``data/preflop_equity.json``.

Postflop: ``equity_vs_random`` estimates equity against one random hand by Monte Carlo
rollouts of the remaining board. It accounts for draws, unlike made-hand strength.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from functools import cache
from pathlib import Path

from poker_lab.engine.fast_hand import phe_rank
from poker_lab.schema.cards import RANKS, card_to_id

DATA_PATH = Path(__file__).with_name("data") / "preflop_equity.json"
_N_COMBOS = 1326


def hand_class(c1: str, c2: str) -> str:
    """Starting-hand class of two hole cards, e.g. ``("Ah", "Kh") -> "AKs"``."""
    r1, r2 = RANKS.index(c1[0]), RANKS.index(c2[0])
    hi, lo = (c1, c2) if r1 >= r2 else (c2, c1)
    if hi[0] == lo[0]:
        return hi[0] + lo[0]
    return hi[0] + lo[0] + ("s" if hi[1] == lo[1] else "o")


def all_hand_classes() -> list[str]:
    classes = []
    for i, hi in enumerate(reversed(RANKS)):
        for lo in reversed(RANKS[: len(RANKS) - i]):
            if hi == lo:
                classes.append(hi + lo)
            else:
                classes.extend([hi + lo + "s", hi + lo + "o"])
    return classes


def class_combos(cls: str) -> int:
    return 6 if len(cls) == 2 else (4 if cls[2] == "s" else 12)


def _example_cards(cls: str) -> tuple[str, str]:
    if len(cls) == 2:
        return cls[0] + "c", cls[1] + "d"
    if cls[2] == "s":
        return cls[0] + "c", cls[1] + "c"
    return cls[0] + "c", cls[1] + "d"


def compute_preflop_table(n_samples: int = 40_000, seed: int = 20261020) -> dict[str, float]:
    """Monte Carlo equity of every class against one random hand (ties count half)."""
    rng = random.Random(seed)
    table = {}
    for cls in all_hand_classes():
        a, b = _example_cards(cls)
        hole = (card_to_id(a), card_to_id(b))
        table[cls] = _mc_equity(hole, (), n_samples, rng)
    return table


def write_preflop_table(path: Path = DATA_PATH, n_samples: int = 40_000, seed: int = 20261020):
    table = compute_preflop_table(n_samples, seed)
    payload = {
        "description": "All-in equity of each starting-hand class vs one uniformly random hand",
        "method": "Monte Carlo",
        "n_samples_per_class": n_samples,
        "seed": seed,
        "equity": table,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8", newline="\n")


@cache
def preflop_equity_table() -> dict[str, float]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))["equity"]


@cache
def _percentiles() -> dict[str, float]:
    """Class -> fraction of the 1,326 combos whose class is at least as strong."""
    eq = preflop_equity_table()
    ranked = sorted(eq, key=lambda c: (-eq[c], c))
    out, cum = {}, 0
    for cls in ranked:
        cum += class_combos(cls)
        out[cls] = cum / _N_COMBOS
    return out


def preflop_percentile(c1: str, c2: str) -> float:
    """Strength percentile of a starting hand: 0.0045 for aces (top 6/1326 combos), 1.0 for
    the weakest class. A range of width ``w`` contains the hands with percentile ``<= w``."""
    return _percentiles()[hand_class(c1, c2)]


def _mc_equity(
    hole: Sequence[int], board: Sequence[int], n_samples: int, rng: random.Random
) -> float:
    dead = set(hole) | set(board)
    deck = [c for c in range(52) if c not in dead]
    need = 5 - len(board)
    h0, h1 = hole
    board = list(board)
    sample = rng.sample
    score = 0.0
    for _ in range(n_samples):
        draw = sample(deck, 2 + need)
        full = board + draw[2:]
        ours = phe_rank([h0, h1, *full])
        theirs = phe_rank([draw[0], draw[1], *full])
        if ours < theirs:
            score += 1.0
        elif ours == theirs:
            score += 0.5
    return score / n_samples


def equity_vs_random(
    hole: Sequence[str], board: Sequence[str], rng: random.Random, n_samples: int = 150
) -> float:
    """Monte Carlo equity of ``hole`` against one random hand, given the board so far."""
    return _mc_equity([card_to_id(c) for c in hole], [card_to_id(c) for c in board], n_samples, rng)


if __name__ == "__main__":  # regenerate the shipped table: python -m poker_lab.bots.hand_strength
    write_preflop_table()
    print(f"wrote {DATA_PATH}")

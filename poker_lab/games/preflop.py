"""Heads-up preflop all-in matchups between the 169 starting-hand classes.

For every ordered pair of classes ``(i, j)`` the module provides:

* ``counts[i, j]``: the number of (combo of ``i``, combo of ``j``) pairs that share no card.
  Exact. Dealing two random hands gives class pair ``(i, j)`` with probability
  ``counts[i, j] / (1326 * 1225)``, which is how card removal enters: AA vs AA has 6
  compatible combo pairs, AA vs 72o has 72.
* ``equity[i, j]``: the all-in equity of ``i`` against ``j`` (ties count half), averaged over
  the compatible combo pairs, which are equally likely given the classes. Estimated by Monte
  Carlo over boards, stratified: every compatible combo pair gets the same number of boards,
  so the combo mix adds no variance. ``equity[j, i] = 1 - equity[i, j]`` and the diagonal is
  exactly 0.5 by symmetry.
* ``se[i, j]``: the standard error of that estimate.

``compute_matchups`` takes a few minutes on 16 cores; its output ships as
``data/preflop_matchups.npz`` (with the seed and sample count) and ``load_matchups`` reads it.
Regenerate with ``python -m poker_lab.games.preflop``.

This module does not import ``poker_lab.bots``: worker processes re-import it and must stay
light. ``CLASSES`` is checked against ``poker_lab.bots.hand_strength.all_hand_classes`` in the
tests.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache
from importlib import metadata
from itertools import combinations
from pathlib import Path

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DATA_PATH = Path(__file__).with_name("data") / "preflop_matchups.npz"
N_COMBOS = 1326
N_DEALS = 1326 * 1225  # ordered pairs of disjoint hands


def _classes() -> tuple[str, ...]:
    out = []
    for i, hi in enumerate(reversed(RANKS)):
        for lo in reversed(RANKS[: len(RANKS) - i]):
            out.extend([hi + lo] if hi == lo else [hi + lo + "s", hi + lo + "o"])
    return tuple(out)


CLASSES: tuple[str, ...] = _classes()
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


def card_id(rank: str, suit: str) -> int:
    """Card id in this repo's deck order (``poker_lab.schema.cards.FULL_DECK``)."""
    return RANKS.index(rank) * 4 + SUITS.index(suit)


@cache
def combos(cls: str) -> tuple[tuple[int, int], ...]:
    """All card-id pairs of a class: 6 for a pair, 4 suited, 12 offsuit."""
    hi, lo = cls[0], cls[1]
    if hi == lo:
        return tuple((card_id(hi, a), card_id(hi, b)) for a, b in combinations(SUITS, 2))
    if cls[2] == "s":
        return tuple((card_id(hi, s), card_id(lo, s)) for s in SUITS)
    return tuple((card_id(hi, a), card_id(lo, b)) for a in SUITS for b in SUITS if a != b)


def combo_counts() -> np.ndarray:
    return np.array([len(combos(c)) for c in CLASSES])


def _compatible(i: int, j: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(a, b) for a in combos(CLASSES[i]) for b in combos(CLASSES[j]) if not set(a) & set(b)]


def matchup_counts() -> np.ndarray:
    """``counts[i, j]``: disjoint (combo of i, combo of j) pairs. Sums to 1326 * 1225."""
    n = len(CLASSES)
    out = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i, n):
            out[i, j] = out[j, i] = len(_compatible(i, j))
    return out


def _matchup_task(
    args: tuple[int, int, int, np.random.SeedSequence],
) -> tuple[int, int, float, float]:
    """Stratified Monte Carlo equity of class i vs class j (worker function)."""
    from phevaluator import evaluate_7cards as ev7

    i, j, n_samples, seed = args
    rng = np.random.default_rng(seed)
    pairs = _compatible(i, j)
    per = math.ceil(n_samples / len(pairs))
    means, variances = [], []
    for a, b in pairs:
        rest = np.array([c for c in range(52) if c not in a and c not in b])
        boards = rest[np.argsort(rng.random((per, len(rest))), axis=1)[:, :5]].tolist()
        score = np.empty(per)
        for k, bd in enumerate(boards):
            ra, rb = ev7(a[0], a[1], *bd), ev7(b[0], b[1], *bd)  # lower rank wins
            score[k] = 1.0 if ra < rb else (0.5 if ra == rb else 0.0)
        means.append(score.mean())
        variances.append(score.var(ddof=1) / per)
    k = len(pairs)
    return i, j, float(np.mean(means)), float(math.sqrt(sum(variances)) / k)


@dataclass(frozen=True, eq=False)
class Matchups:
    classes: tuple[str, ...]
    counts: np.ndarray  # (169, 169) int
    equity: np.ndarray  # (169, 169) float: equity of the row class against the column class
    se: np.ndarray  # (169, 169) float: Monte Carlo standard error
    meta: dict

    @property
    def deal_prob(self) -> np.ndarray:
        """Probability that a random deal gives (row class, column class)."""
        return self.counts / N_DEALS

    def equity_vs_random(self) -> np.ndarray:
        """Each class's equity against one uniformly random hand (card removal included)."""
        return (self.counts * self.equity).sum(axis=1) / self.counts.sum(axis=1)


def compute_matchups(
    n_samples: int = 50_000, seed: int = 20261023, workers: int | None = None
) -> Matchups:
    n = len(CLASSES)
    tasks = [(i, j) for i in range(n) for j in range(i + 1, n)]
    seeds = np.random.SeedSequence(seed).spawn(len(tasks))  # per task: worker-count independent
    equity = np.full((n, n), 0.5)
    se = np.zeros((n, n))
    args = [(i, j, n_samples, s) for (i, j), s in zip(tasks, seeds, strict=True)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, j, e, s in pool.map(_matchup_task, args, chunksize=64):
            equity[i, j], equity[j, i] = e, 1.0 - e
            se[i, j] = se[j, i] = s
    meta = {
        "n_samples": n_samples,
        "seed": seed,
        "method": "stratified Monte Carlo over compatible combo pairs; phevaluator 7-card ranks",
        "phevaluator": metadata.version("phevaluator"),
    }
    return Matchups(CLASSES, matchup_counts(), equity, se, meta)


def save_matchups(m: Matchups, path: Path = DATA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        classes=np.array(m.classes),
        counts=m.counts.astype(np.int32),
        equity=m.equity,
        se=m.se.astype(np.float32),
        meta=np.array(json.dumps(m.meta, sort_keys=True)),
    )
    return path


@cache
def load_matchups(path: Path = DATA_PATH) -> Matchups:
    with np.load(path) as z:
        classes = tuple(str(c) for c in z["classes"])
        if classes != CLASSES:
            raise ValueError(f"{path} was built with a different class order")
        counts = z["counts"].astype(np.int64)
        equity, se = z["equity"].astype(float), z["se"].astype(float)
        meta = json.loads(str(z["meta"]))
    for a in (counts, equity, se):
        a.setflags(write=False)
    return Matchups(classes, counts, equity, se, meta)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute the 169x169 preflop all-in matchup table.")
    ap.add_argument("--samples", type=int, default=50_000, help="boards per class matchup")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DATA_PATH)
    args = ap.parse_args()
    m = compute_matchups(args.samples, args.seed, args.workers)
    path = save_matchups(m, args.out)
    print(f"wrote {path}; max standard error {m.se.max():.4f}")


if __name__ == "__main__":
    main()

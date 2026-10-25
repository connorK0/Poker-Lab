"""P3.8: an archetype classifier on raw action sequences vs the Phase 1 classifier.

    python -m experiments.phase3_sequence --seed 2026

Pre-registered plan (written before any neural model was fit to the five Phase 1 replicates or
applied to the real hands).

Question. Phase 1's classifier sees a player through ten hand-written rates and two sizes. Does
a model that reads the raw hands (each decision with its public situation, action and size;
``profiling.sequences``) identify archetypes faster or more reliably, on the same splits and
metrics? Does the order of a hand's decisions matter? Is it the raw tokens or the network
that matters? And does the sequence model behave any better than the logistic model did on the
Pluribus professionals (P3.7)?

Data
* Synthetic: Phase 1's five replicates (master seed 2026), re-simulated with every decision
  recorded as a token. Each replicate must reproduce Phase 1's players, labels, parameters,
  per-hand counters and decision rows exactly (checked; the run stops otherwise), and the
  refit logistic model must reproduce Phase 1's reported accuracy and log-loss. Per replicate:
  300 training and 150 held-out bots, 2,000 hands each.
* Real: the 13 professionals of the Pluribus hands, loaded as P3.7 loads them (same order,
  seating, redaction); P3.7's control: that seating replayed by Phase 1 bots (a new draw).
* Extra training bots (part C): 240 more per archetype for each replicate (1,200 players),
  drawn and simulated like Phase 1's training pool in a league of their own (seeds from
  (replicate seed, 7)).
* Development used seeds 98 and 99 (full-size replicates with new populations; their test
  windows were looked at) and the quick configuration only. What it changed: mean pooling
  over hands plateaued at about 92% accuracy at 2,000 hands (logistic: 95%), so the pooling
  became learned shrunk rates (``SequenceClassifier`` docstring); a cosine learning-rate
  schedule and gradient clipping (validation loss swung by +-0.15 between epochs); 8 players
  per batch; 16 rates; per-size temperatures for the neural models (Phase 1 has them for its
  classifier). Settings were compared on seed 98 (differences of 0.01-0.02 nats, within the
  noise) and frozen before seed 99, where with the frozen settings every neural model was
  worse than the logistic model at every size (by 0.02-0.07 nats). Part C was added after a
  check on seed 99: with 1,500 training bots the logistic model did not change, the token
  network improved by 0.02-0.06 nats and matched it. The real hands were not loaded.

Models (fit to the training bots of each replicate only)
* ``lr``, ``gbm``: Phase 1's logistic regression and boosted trees on shrunk stats, refit
  exactly as Phase 1 did (one global temperature; per-size temperatures reported too).
* ``seq``: the sequence model: a GRU over each hand's tokens, learned shrunk rates over the
  window's hands, an MLP head (settings: ``model`` in the config).
* ``bag``: the same network with the order of a hand's tokens dropped (summed).
* ``counters``: the same order-free network reading each hand's stats counters instead of
  its tokens: the stats' information with a learned summary.
The held-out 20% of training bots (the same players as the logistic model's temperature split)
picks the epoch and fits the temperatures.

A. Hands to identify, on Phase 1's held-out windows (up to 10 disjoint windows per held-out bot
   and sample size, 25 to 2,000 hands). Phase 1's metrics: accuracy, log-loss, ECE (10 bins),
   Brier, means over replicates with 95% t-intervals; hands needed for 80/90/95% accuracy.
   Paired against ``lr`` on the same windows: per replicate the mean over windows of
   log-loss(lr) - log-loss(model) (the gain) and the accuracy difference, t-intervals over
   the five replicates.
   A1. The sequence model beats the logistic model at no sample size: no interval of its
       log-loss gain (one temperature) or of its accuracy gain lies above 0.
   A2. The order of a hand's decisions changes log-loss by less than 0.02 nats at every size:
       |mean log-loss(seq) - log-loss(bag)| < 0.02.
   A3. The tokens beat the stats counters by less than 0.02 nats at every size: mean
       log-loss(counters) - log-loss(bag) < 0.02. (Development was mixed: on seed 98 a
       32-rate counters network trailed the tokens by 0.015-0.045; on seed 99, with the
       frozen settings, they were within 0.015.)
B. The professionals. Each replicate's ``lr`` and ``seq`` on P3.7's windows of the
   professionals' hands: labels, top probability, how often the two models agree (and on
   held-out bots), and P3.7's label-free consistency check: gap = mean top probability -
   agreement with the label of a later 1,000-hand block, compared with held-out bots of the
   same reference label; 95% intervals over players (bootstrap); P3.7's verdict rule
   (transfers / fails / inconclusive). The same on the control.
   B1. On the professionals the sequence model is overconfident at 25, 50 and 100 hands
       (each interval of gap - matched bots' gap above 0), as the logistic model was in P3.7.
   B2. On the control neither model has an interval above 0 at 25, 50 or 100 hands.
   (On seed 99's simulated stand-in players the verdict rule said *fails* for the sequence
   model: with 13 players and six sizes it raises false alarms, so the verdicts are reported,
   not predicted.)
C. More training players: part A's ``lr``, ``gbm``, ``bag`` and ``counters`` refit with the
   extra 1,200 training bots per replicate (1,500 in all), on the same held-out windows.
   The improvement is log-loss with Phase 1's 300 minus log-loss with 1,500, paired by
   replicate. (``bag`` stands in for the sequence model: A2 predicts order does not matter,
   and the GRU takes five times as long.)
   C1. The logistic model's log-loss changes by less than 0.02 nats at every size (it is
       already limited by its features, not by the number of players).
   C2. The token network improves by at least 0.02 nats at every size, every interval above 0.
   C3. With 1,500 bots the token network and the logistic model are within 0.03 nats at every
       size: every interval of the gain lies inside [-0.03, 0.03].
"""

from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import REPO_ROOT, RESULTS_ROOT, load_config
from experiments.phase1_data import CACHE_DIR, generate, simulation_fingerprint
from poker_lab.bots import GroundTruth, make_population
from poker_lab.bots.league import LeagueConfig, run_league

CONFIG = REPO_ROOT / "experiments" / "configs" / "phase3.yaml"
SEQUENCE_SOURCES = ("poker_lab/profiling/sequences.py",)


# ---------------------------------------------------------------------------- data
@dataclass
class SeqPool:
    """One simulated population with every hand as tokens (see ``profiling.sequences``)."""

    player_ids: list[str]
    labels: np.ndarray
    params: np.ndarray
    counters: np.ndarray  # (players, hands, N_COUNTERS) int16, as in Phase 1
    tokens: np.ndarray  # HandSequences arrays
    hand_ptr: np.ndarray
    player_ptr: np.ndarray

    @property
    def cumulative(self) -> np.ndarray:
        return self.counters.astype(np.int32).cumsum(axis=1)

    def sequences(self):
        from poker_lab.profiling.sequence_model import HandSequences

        return HandSequences(self.tokens, self.hand_ptr, self.player_ptr)


@dataclass
class SeqSeedData:
    seed: int
    train: SeqPool
    test: SeqPool


def _fingerprint() -> str:
    digest = hashlib.sha256(simulation_fingerprint().encode())
    for rel in SEQUENCE_SOURCES:
        digest.update((REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:12]


def _cache_path(seed: int, p1: dict) -> Path:
    key = (
        f"s{seed}_tr{p1['n_train_per_archetype']}_te{p1['n_test_per_archetype']}"
        f"_h{p1['hands_per_player']}_b{p1['block_hands']}_eq{p1.get('equity_samples', 150)}"
        f"_fp{_fingerprint()}"
    )
    return CACHE_DIR / f"phase3_sequence_{key}.npz"


def _simulate_pool(pop, p1: dict, seed: int, workers) -> SeqPool:
    from poker_lab.profiling.sequence_model import HandSequences
    from poker_lab.profiling.sequences import SequenceRowExtractor

    truth = GroundTruth(pop)
    league = LeagueConfig(
        hands_per_player=p1["hands_per_player"],
        block_hands=p1["block_hands"],
        equity_samples=p1.get("equity_samples", 150),
    )
    result = run_league(pop, league, seed=seed, workers=workers, extractor=SequenceRowExtractor())
    seqs = HandSequences.from_rows(result.extra, len(pop))
    return SeqPool(
        player_ids=result.player_ids,
        labels=truth.labels(result.player_ids),
        params=truth.param_matrix(result.player_ids),
        counters=result.counters,
        tokens=seqs.tokens,
        hand_ptr=seqs.hand_ptr,
        player_ptr=seqs.player_ptr,
    )


_POOL_ARRAYS = ("labels", "params", "counters", "tokens", "hand_ptr", "player_ptr")


def _save_pools(path: Path, pools: dict[str, SeqPool]) -> None:
    arrays = {}
    for name, pool in pools.items():
        arrays[f"{name}_ids"] = np.array(pool.player_ids)
        for k in _POOL_ARRAYS:
            arrays[f"{name}_{k}"] = getattr(pool, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_pools(path: Path, names: tuple[str, ...]) -> dict[str, SeqPool]:
    z = np.load(path)
    return {
        name: SeqPool(
            player_ids=[str(x) for x in z[f"{name}_ids"]],
            **{k: z[f"{name}_{k}"] for k in _POOL_ARRAYS},
        )
        for name in names
    }


def generate_sequences(seed: int, p1: dict, *, workers=None, use_cache=True) -> SeqSeedData:
    """Phase 1's replicate ``seed`` (same populations, seeds and hands as
    ``phase1_data.generate``), with every hand's tokens."""
    path = _cache_path(seed, p1)
    if use_cache and path.exists():
        pools = _load_pools(path, ("train", "test"))
        return SeqSeedData(seed, pools["train"], pools["test"])
    rng = np.random.default_rng(seed)  # as in phase1_data.generate
    train_pop = make_population(p1["n_train_per_archetype"], rng, id_prefix="a")
    test_pop = make_population(p1["n_test_per_archetype"], rng, id_prefix="b")
    sub = np.random.SeedSequence(seed).generate_state(2)
    data = SeqSeedData(
        seed,
        _simulate_pool(train_pop, p1, int(sub[0]), workers),
        _simulate_pool(test_pop, p1, int(sub[1]), workers),
    )
    if use_cache:
        _save_pools(path, {"train": data.train, "test": data.test})
    return data


def generate_extra(seed: int, p1: dict, per_archetype: int, *, workers=None, use_cache=True):
    """Extra training bots for replicate ``seed``: ``per_archetype`` more players of each
    archetype, drawn and simulated as Phase 1's training pool is, in a league of their own,
    from seeds derived from (seed, 7). Disjoint from the replicate's players."""
    path = _cache_path(seed, p1).with_name(
        _cache_path(seed, p1).name.replace(
            "phase3_sequence_", f"phase3_sequence_extra{per_archetype}_"
        )
    )
    if use_cache and path.exists():
        return _load_pools(path, ("extra",))["extra"]
    ss = np.random.SeedSequence([seed, 7])
    pop = make_population(per_archetype, np.random.default_rng(ss), id_prefix="c")
    pool = _simulate_pool(pop, p1, int(ss.generate_state(2)[1]), workers)
    if use_cache:
        _save_pools(path, {"extra": pool})
    return pool


def combine(a: SeqPool, b: SeqPool) -> SeqPool:
    """Two pools' players in one pool (a's first)."""
    hands_a = int(a.player_ptr[-1])
    return SeqPool(
        player_ids=[*a.player_ids, *b.player_ids],
        labels=np.concatenate([a.labels, b.labels]),
        params=np.concatenate([a.params, b.params]),
        counters=np.concatenate([a.counters, b.counters]),
        tokens=np.concatenate([a.tokens, b.tokens]),
        hand_ptr=np.concatenate([a.hand_ptr, b.hand_ptr[1:] + a.hand_ptr[-1]]),
        player_ptr=np.concatenate([a.player_ptr, b.player_ptr[1:] + hands_a]),
    )


def check_against_phase1(data: SeqSeedData, p1: dict, workers=None) -> dict[str, Any]:
    """The re-simulated hands are Phase 1's: identical players, labels, parameters and
    counters, and the situation columns of the tokens equal Phase 1's decision rows."""
    from poker_lab.profiling import SITUATION_FEATURES

    ref = generate(data.seed, p1, workers=workers)
    out: dict[str, Any] = {}
    for name in ("train", "test"):
        a, b = getattr(data, name), getattr(ref, name)
        same = {
            "ids": a.player_ids == b.player_ids,
            "labels": bool(np.array_equal(a.labels, b.labels)),
            "params": bool(np.array_equal(a.params, b.params)),
            "counters": bool(np.array_equal(a.counters, b.counters)),
        }
        # Phase 1's decision rows: (player, seq, situation..., label) inside the action windows
        dec = b.decisions
        hands = np.diff(a.player_ptr)
        seq_of_hand = np.concatenate([np.arange(h) for h in hands])
        player_of_hand = np.repeat(np.arange(len(hands)), hands)
        lengths = np.diff(a.hand_ptr)
        tok_player = np.repeat(player_of_hand, lengths)
        tok_seq = np.repeat(seq_of_hand, lengths)
        decision = a.tokens[:, 0] == 0
        windows = np.zeros((len(hands), hands.max()), dtype=bool)
        windows[dec[:, 0].astype(int), dec[:, 1].astype(int)] = True
        keep = decision & windows[tok_player, tok_seq]
        sit = slice(1, 1 + len(SITUATION_FEATURES))
        mine = np.column_stack([tok_player[keep], tok_seq[keep], a.tokens[keep, sit]])
        theirs = dec[:, : 2 + len(SITUATION_FEATURES)]
        order_a = np.lexsort((mine[:, 1], mine[:, 0]))
        order_b = np.lexsort((theirs[:, 1], theirs[:, 0]))
        same["decisions"] = bool(
            len(mine) == len(theirs)
            and np.array_equal(mine[order_a].astype(np.float32), theirs[order_b])
        )
        same["hands_per_player"] = bool(np.all(hands == a.counters.shape[1]))
        out[name] = same
    out["identical"] = all(all(v.values()) for v in (out["train"], out["test"]))
    return out


# ---------------------------------------------------------------------------- real players
@dataclass
class NamedPool:
    """Named players' hands in order (the Pluribus professionals, or a simulated control):
    per-hand counters and tokens, and the seating (P3.7's ``assemble``)."""

    ids: list[str]
    counters: list[np.ndarray]  # per player: (hands, N_COUNTERS)
    tokens: np.ndarray
    hand_ptr: np.ndarray
    player_ptr: np.ndarray
    seating: list[tuple[tuple[str, ...], int]]
    source: dict[str, Any]

    def sequences(self):
        from poker_lab.profiling.sequence_model import HandSequences

        return HandSequences(self.tokens, self.hand_ptr, self.player_ptr)

    @property
    def hands(self) -> np.ndarray:
        return np.array([len(c) for c in self.counters])


def token_rows(hand) -> tuple[list, list]:
    """Counters and tokens of one hand, from its public record only."""
    from poker_lab.profiling import hand_counters
    from poker_lab.profiling.sequences import hand_tokens
    from poker_lab.schema import redact_hole_cards

    public = redact_hole_cards(hand)
    return hand_counters(public), hand_tokens(public)


def assemble(ordered: list[tuple[Any, list, list]], source: dict[str, Any]) -> NamedPool:
    """From (block key, counters, tokens) per hand in order; the seating as in P3.7."""
    from poker_lab.profiling.sequence_model import HandSequences

    counters: dict[str, list] = {}
    rows: list[tuple[str, int, tuple]] = []
    seating: list[list] = []
    last = None
    for key, hand_counters_, hand_tokens_ in ordered:
        seated = tuple(pid for pid, _ in hand_counters_)
        if (key, sorted(seated)) != last:
            seating.append([seated, 0])
            last = (key, sorted(seated))
        seating[-1][1] += 1
        for pid, row in hand_counters_:
            counters.setdefault(pid, []).append(row)
        for pid, toks in hand_tokens_:
            seq = len(counters[pid]) - 1
            rows += [(pid, seq, t) for t in toks]
    ids = sorted(counters)
    index = {pid: i for i, pid in enumerate(ids)}
    table = np.array([(index[p], s, *t) for p, s, t in rows], dtype=np.float32)
    seqs = HandSequences.from_rows(table, len(ids))
    return NamedPool(
        ids=ids,
        counters=[np.array(counters[p], dtype=np.int32) for p in ids],
        tokens=seqs.tokens,
        hand_ptr=seqs.hand_ptr,
        player_ptr=seqs.player_ptr,
        seating=[(s, n) for s, n in seating],
        source=source,
    )


def _import_folder(args: tuple[str, str]) -> list:
    from experiments.phase3_priors import session_key
    from poker_lab.importers import import_phh

    folder, root = args
    report = import_phh([folder], root=root, allow_rake=False)
    if report.rejected:
        raise ValueError(f"{len(report.rejected)} hands of {folder} rejected (P3.6: none)")
    return [(session_key(h.metadata["import.file"]), *token_rows(h)) for h in report.hands]


def load_pluribus(cfg: dict[str, Any], workers) -> NamedPool:
    """The Pluribus hands as P3.7 loads them (same order, seating and redaction), with tokens."""
    from experiments.datasets import PHH_DATASET_URL, content_sha256, phh_dataset
    from poker_lab.importers import phh_files

    root = phh_dataset(cfg["dataset_commit"])
    base = root / cfg["dataset_path"]
    folders = sorted(str(p) for p in base.iterdir() if p.is_dir())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_import_folder, [(f, str(root)) for f in folders]))
    rows = sorted((r for part in parts for r in part), key=lambda r: r[0])
    files = phh_files([base], root=root)
    source = {
        "kind": "pluribus",
        "dataset": PHH_DATASET_URL,
        "commit": cfg["dataset_commit"],
        "path": cfg["dataset_path"],
        "files": len(files),
        "sha256": content_sha256(files),
        "hands": len(rows),
    }
    return assemble([(key[:2], *rest) for key, *rest in rows], source)


def _standin_block(args) -> list:
    """P3.7's ``_standin_block`` (kept unchanged there), with tokens."""
    from poker_lab.bots import Archetype, ParametricBot, sample_params
    from poker_lab.engine import SeatedPlayer, TableSimulator
    from poker_lab.schema import TableConfig

    block, seed, seated, n_hands, prefix = args
    players = [
        SeatedPlayer(
            seat, pid, ParametricBot(sample_params(Archetype(arch), np.random.default_rng(ps)))
        )
        for seat, (pid, arch, ps) in enumerate(seated)
    ]
    sim = TableSimulator(
        TableConfig(f"{prefix}{block}", 6, 50, 100),
        players,
        seed=seed,
        starting_stack=10_000,
        hand_id_prefix=f"{prefix}/{block}/",
    )
    return [(block, *token_rows(h)) for h in sim.play(n_hands)]


def simulate_standin(seating, bot_id: str, seed: int, prefix: str, workers) -> NamedPool:
    """P3.7's control: the given seating replayed by Phase 1 bots (archetypes in turn,
    parameters drawn as for training bots), ``bot_id`` by a TAG bot; blinds 50/100, 100 bb."""
    from poker_lab.bots import ARCHETYPES, Archetype

    rng = np.random.default_rng(seed)
    others = sorted({p for seated, _ in seating for p in seated} - {bot_id})
    bots = {
        pid: (f"s{i:02d}", ARCHETYPES[i % len(ARCHETYPES)].value, int(rng.integers(2**31)))
        for i, pid in enumerate(others)
    }
    bots[bot_id] = (bot_id, Archetype.TAG.value, int(rng.integers(2**31)))
    tasks = [
        (b, int(rng.integers(2**31)), [bots[p] for p in seated], n, prefix)
        for b, (seated, n) in enumerate(seating)
    ]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        blocks = list(ex.map(_standin_block, tasks))
    source = {
        "kind": "standin",
        "prefix": prefix,
        "hands": sum(map(len, blocks)),
        "blocks": len(seating),
        "archetypes": {sid: arch for sid, arch, _ in bots.values()},
    }
    return assemble([r for part in blocks for r in part], source)


# ---------------------------------------------------------------------------- models
def held_out_windows(pool: SeqPool, n: int, max_windows: int) -> np.ndarray:
    """Phase 1's test windows of ``n`` hands as rows (player, start, n), in the row order of
    ``window_totals``."""
    hands = pool.counters.shape[1]
    k = min(max_windows, hands // n)
    players = np.repeat(np.arange(len(pool.player_ids)), k)
    starts = n * np.tile(np.arange(k), len(pool.player_ids))
    return np.column_stack([players, starts, np.full(len(players), n)])


MODES = ("none", "global", "by_size")  # calibration: none, one temperature, one per size


def phase1_classifiers(data: SeqSeedData, p1: dict, sizes: list[int]):
    """Phase 1's logistic and boosted-tree classifiers on shrunk stats, fit exactly as in
    ``phase1_analysis.analyze``, and their test probabilities per calibration mode and sample
    size."""
    from poker_lab.profiling import (
        ArchetypeClassifier,
        EmpiricalBayesShrinker,
        profile_features,
        window_totals,
    )

    tr, te = data.train, data.test
    tr_cum, te_cum = tr.cumulative, te.cumulative
    half = tr.counters.shape[1] // 2
    halves = (tr_cum[:, half - 1], tr_cum[:, -1] - tr_cum[:, half - 1])
    shrinker = EmpiricalBayesShrinker.fit(tr_cum[:, -1], halves)
    X, y, g = [], [], []
    for n in sizes:
        w = window_totals(tr_cum, n, p1["train_windows_max"])
        X.append(profile_features(shrinker, w.totals))
        y.append(tr.labels[w.player])
        g.append(w.player)
    X, y, g = np.concatenate(X), np.concatenate(y), np.concatenate(g)
    clfs = {
        m: ArchetypeClassifier(m, calibrate=True, random_state=data.seed).fit(X, y, g)
        for m in ("lr", "gbm")
    }
    probs = {m: {mode: {} for mode in MODES} for m in clfs}
    for n in sizes:
        w = window_totals(te_cum, n, p1["test_windows_max"])
        Xt = profile_features(shrinker, w.totals)
        for m, clf in clfs.items():
            for mode in MODES:
                probs[m][mode][n] = clf.predict_proba(Xt, calibration=mode)
    return shrinker, clfs, probs


def counter_vectors(counters) -> list[np.ndarray]:
    """Per-hand stats counters as float vectors (sizes back in pot fractions and big blinds),
    one (hands, N_COUNTERS) array per player."""
    from poker_lab.profiling.stats import C

    out = []
    for c in counters:
        x = np.asarray(c, dtype=np.float32).copy()
        for name in ("bet_size_x100", "open_size_x100"):
            x[:, C[name]] /= 100.0
        out.append(x)
    return out


#: The neural variants: (encoder, input). "seq" is the sequence model; "bag" drops the order of
#: a hand's decisions; "counters" reads each hand's stats counters instead of its tokens.
NEURAL = {"seq": ("gru", "tokens"), "bag": ("bag", "tokens"), "counters": ("bag", "counters")}


def neural_inputs(pool, kind: str):
    from poker_lab.profiling.sequence_model import HandSequences

    if kind == "tokens":
        return pool.sequences()
    return HandSequences.from_vectors(counter_vectors(pool.counters))


def fit_neural(data: SeqSeedData, variant: str, sizes, p1: dict, cfg: dict):
    from poker_lab.profiling.sequence_model import SequenceClassifier

    encoder, kind = NEURAL[variant]
    clf = SequenceClassifier(encoder, random_state=data.seed, **cfg["model"])
    return clf.fit(
        neural_inputs(data.train, kind), data.train.labels, sizes, p1["train_windows_max"]
    )


def _metrics(y: np.ndarray, probs: dict[str, np.ndarray]) -> dict:
    """Phase 1's metrics with one temperature, and log-loss and ECE for the other modes."""
    from poker_lab.evaluation import accuracy, brier, log_loss, reliability

    proba = probs["global"]
    return {
        "accuracy": accuracy(y, proba),
        "log_loss": log_loss(y, proba),
        "ece": reliability(y, proba).ece,
        "brier": brier(y, proba),
        "top_prob": float(proba.max(axis=1).mean()),
        "log_loss_uncal": log_loss(y, probs["none"]),
        "ece_uncal": reliability(y, probs["none"]).ece,
        "log_loss_by_size": log_loss(y, probs["by_size"]),
        "ece_by_size": reliability(y, probs["by_size"]).ece,
    }


def part_a(
    data: SeqSeedData, p1: dict, cfg: dict, variants=None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hands-to-identify on Phase 1's test windows: Phase 1's classifiers and the neural
    variants, with paired per-window differences against the logistic model. Returns the
    results and the fitted models (for part B)."""
    from poker_lab.evaluation.metrics import per_row_log_loss

    hands = data.train.counters.shape[1]
    sizes = [n for n in p1["sample_sizes"] if n <= hands]
    shrinker, clfs, probs = phase1_classifiers(data, p1, sizes)
    models: dict[str, Any] = {"shrinker": shrinker, **clfs}
    training = {}
    for variant in cfg["variants"] if variants is None else variants:
        clf = fit_neural(data, variant, sizes, p1, cfg)
        models[variant] = clf
        inputs = neural_inputs(data.test, NEURAL[variant][1])
        probs[variant] = {mode: {} for mode in MODES}
        for n in sizes:
            w = held_out_windows(data.test, n, p1["test_windows_max"])
            for mode in MODES:
                probs[variant][mode][n] = clf.predict_proba(inputs, w, calibration=mode)
        h = clf.history
        training[variant] = {
            "epochs": len(h.val_loss),
            "best_epoch": h.best_epoch,
            "train_loss": h.train_loss,
            "val_loss": h.val_loss,
            "temperature": clf.temperature,
            "size_temperatures": None
            if clf.size_temperatures is None
            else {
                str(round(float(np.exp(x)))): float(t)
                for x, t in zip(*clf.size_temperatures, strict=True)
            },
            "seconds": h.seconds,
            "parameters": sum(p.numel() for p in clf.net.parameters()),
        }
        print(f"      {variant}: {h.seconds:.0f}s, {len(h.val_loss)} epochs", flush=True)
    out: dict[str, Any] = {"sizes": sizes, "metrics": {}, "paired": {}, "training": training}
    for m in probs:
        out["metrics"][m] = {}
        for n in sizes:
            y = data.test.labels[held_out_windows(data.test, n, p1["test_windows_max"])[:, 0]]
            out["metrics"][m][str(n)] = _metrics(y, {k: probs[m][k][n] for k in MODES})
    for m in probs:
        if m == "lr":
            continue
        out["paired"][m] = {}
        for n in sizes:
            y = data.test.labels[held_out_windows(data.test, n, p1["test_windows_max"])[:, 0]]
            mine, base = probs[m]["global"][n], probs["lr"]["global"][n]
            ll = per_row_log_loss(y, base) - per_row_log_loss(y, mine)
            ll_size = per_row_log_loss(y, probs["lr"]["by_size"][n]) - per_row_log_loss(
                y, probs[m]["by_size"][n]
            )
            acc = (mine.argmax(1) == y).astype(float) - (base.argmax(1) == y)
            out["paired"][m][str(n)] = {
                "log_loss_gain": float(ll.mean()),
                "log_loss_gain_by_size": float(ll_size.mean()),
                "accuracy_gain": float(acc.mean()),
            }
    out["temperatures"] = {"lr": clfs["lr"].temperature, "gbm": clfs["gbm"].temperature}
    return out, models


# ---------------------------------------------------------------------------- part B
class PoolView:
    """A pool's per-player cumulative counters and neural inputs, built once."""

    def __init__(self, pool) -> None:
        self.pool = pool
        self.cum = []
        for c in pool.counters:
            cum = np.zeros((len(c) + 1, c.shape[1]), dtype=np.int64)
            np.cumsum(c, axis=0, out=cum[1:])
            self.cum.append(cum)
        self.hands = np.array([len(c) for c in pool.counters])
        self._inputs: dict[str, Any] = {}

    def inputs(self, kind: str):
        if kind not in self._inputs:
            self._inputs[kind] = neural_inputs(self.pool, kind)
        return self._inputs[kind]

    def proba(self, name: str, models: dict[str, Any], windows: np.ndarray) -> np.ndarray:
        """Calibrated probabilities of model ``name`` for windows (player, start, n)."""
        from poker_lab.profiling import profile_features

        windows = np.asarray(windows, dtype=np.int64).reshape(-1, 3)
        if name in ("lr", "gbm"):
            totals = np.stack([self.cum[p][s + n] - self.cum[p][s] for p, s, n in windows])
            return models[name].predict_proba(profile_features(models["shrinker"], totals))
        return models[name].predict_proba(self.inputs(NEURAL[name][1]), windows)


def _spread_windows(view: PoolView, players, n: int, cap: int) -> np.ndarray:
    """P3.7's windows: up to ``cap`` disjoint n-hand windows spread evenly over each player's
    hands."""
    from experiments.phase3_priors import _starts

    rows = [(p, s, n) for p in players for s in _starts(int(view.hands[p]), n, cap, need=n)]
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


def _chunk_windows(view: PoolView, players, n: int, ref: int, cap: int, max_chunks=None):
    """P3.7's consistency layout: chunks of ``2 * ref`` hands; windows of n hands (at most
    ``cap``) from a chunk's first half, its second half the reference block. Returns
    (chunk rows (player, chunk), windows with their chunk index, reference windows)."""
    chunks, windows, refs = [], [], []
    for p in players:
        k = int(view.hands[p]) // (2 * ref)
        for c in range(k if max_chunks is None else min(k, max_chunks)):
            base = 2 * ref * c
            j = len(chunks)
            chunks.append((p, c))
            refs.append((p, base + ref, ref))
            windows += [(p, base + s, n, j) for s in n * np.arange(min(cap, ref // n))]
    return (
        np.array(chunks, dtype=np.int64).reshape(-1, 2),
        np.array(windows, dtype=np.int64).reshape(-1, 4),
        np.array(refs, dtype=np.int64).reshape(-1, 3),
    )


def transfer_rep(models, view: PoolView, players, bots: PoolView, sizes, cfg) -> dict[str, Any]:
    """One replicate's models on a population's players (labels, confidence, consistency)
    and, for the consistency check, on the replicate's held-out bots."""
    from poker_lab.bots import ARCHETYPES

    k_arch = len(ARCHETYPES)
    cap, ref = cfg["windows_max"], cfg["consistency_reference"]
    out: dict[str, Any] = {"labels": {}, "agreement": {}, "consistency": {}}
    for n in sizes:
        w = _spread_windows(view, players, n, cap)
        if not len(w):
            continue
        pred = {}
        out["labels"][n] = {}
        for name in cfg["transfer_models"]:
            proba = view.proba(name, models, w)
            pred[name] = proba.argmax(axis=1)
            out["labels"][n][name] = {
                int(p): {
                    "labels": np.bincount(pred[name][w[:, 0] == p], minlength=k_arch)
                    / (w[:, 0] == p).sum(),
                    "top_prob": float(proba[w[:, 0] == p].max(axis=1).mean()),
                }
                for p in np.unique(w[:, 0])
            }
        a, b = cfg["transfer_models"][:2]
        same = (pred[a] == pred[b]).astype(float)
        bw = held_out_windows(bots.pool, n, cap)
        out["agreement"][n] = {
            "players": float(np.mean([same[w[:, 0] == p].mean() for p in np.unique(w[:, 0])])),
            "bots": float(
                (bots.proba(a, models, bw).argmax(1) == bots.proba(b, models, bw).argmax(1)).mean()
            ),
        }
    for n in (n for n in sizes if n <= ref):
        out["consistency"][n] = {}
        for name in cfg["transfer_models"]:
            entry = {}
            for who, v, ps, max_chunks in (
                ("players", view, players, None),
                ("bots", bots, range(len(bots.hands)), 1),
            ):
                chunks, win, refs = _chunk_windows(v, ps, n, ref, cap, max_chunks)
                if not len(chunks):
                    continue
                ref_label = v.proba(name, models, refs).argmax(axis=1)
                proba = v.proba(name, models, win[:, :3])
                j = win[:, 3]
                conf = np.bincount(j, proba.max(axis=1)) / np.bincount(j)
                agree = np.bincount(j, proba.argmax(axis=1) == ref_label[j]) / np.bincount(j)
                entry[who] = {
                    "player": chunks[:, 0],
                    "confidence": conf,
                    "agreement": agree,
                    "reference": ref_label,
                }
            out["consistency"][n][name] = entry
    return out


def summarize_transfer(reps: list[dict], ids: list[str], cfg, rng) -> dict[str, Any]:
    """Across replicates: label shares and confidence (each player weighs the same), label
    agreement between the models, and P3.7's label-matched consistency check per model."""
    from experiments.phase3_priors import _player_bootstrap, consistency_verdict
    from poker_lab.bots import ARCHETYPES

    out: dict[str, Any] = {"labels": {}, "agreement": {}, "consistency": {}, "verdicts": {}}
    sizes = sorted({n for r in reps for n in r["labels"]})
    for n in sizes:
        out["labels"][str(n)] = {}
        for name in cfg["transfer_models"]:
            per = [r["labels"][n][name] for r in reps if n in r["labels"]]
            ps = sorted(per[0])
            out["labels"][str(n)][name] = {
                "shares": np.mean([[x[p]["labels"] for p in ps] for x in per], axis=(0, 1)),
                "top_prob": float(np.mean([[x[p]["top_prob"] for p in ps] for x in per])),
                "players": len(ps),
            }
        out["agreement"][str(n)] = {
            k: float(np.mean([r["agreement"][n][k] for r in reps if n in r["agreement"]]))
            for k in ("players", "bots")
        }
    for name in cfg["transfer_models"]:
        cons = {}
        for n in sorted({n for r in reps for n in r["consistency"]}):
            entries = [r["consistency"][n][name] for r in reps]
            if not all("players" in e and "bots" in e for e in entries):
                continue
            pids = np.unique(entries[0]["players"]["player"])
            gap, expected, bot_gap = [], [], []
            for e in entries:
                b = e["bots"]
                g = b["confidence"] - b["agreement"]
                by_ref = np.array(
                    [g[b["reference"] == k].mean() if (b["reference"] == k).any() else 0.0
                     for k in range(len(ARCHETYPES))]
                )  # fmt: skip
                h = e["players"]
                hg = h["confidence"] - h["agreement"]
                gap.append([hg[h["player"] == p].mean() for p in pids])
                expected.append([by_ref[h["reference"][h["player"] == p]].mean() for p in pids])
                bot_gap.append(float(g.mean()))
            gap, expected = np.mean(gap, axis=0), np.mean(expected, axis=0)
            diff = _player_bootstrap((gap - expected)[:, None], rng, cfg["bootstrap"])
            cons[str(n)] = {
                "players_gap": float(gap.mean()),
                "expected_gap": float(expected.mean()),
                "bots_gap": float(np.mean(bot_gap)),
                "difference": {k: float(np.ravel(v)[0]) for k, v in diff.items()},
                "by_player": {ids[p]: float(x) for p, x in zip(pids, gap - expected, strict=True)},
            }
        out["consistency"][name] = cons
        out["verdicts"][name] = consistency_verdict(cons)["verdict"]
    return out


# ---------------------------------------------------------------------------- run
METRICS = ("accuracy", "log_loss", "ece", "brier", "top_prob")
THRESHOLDS = (0.8, 0.9, 0.95)


def phase1_reference(p1: dict, p1_seeds: list[int], sizes) -> dict[int, dict] | None:
    """Phase 1's reported logistic accuracy and log-loss per seed, if its results file was
    made with this configuration and holds these seeds."""
    import json

    path = RESULTS_ROOT / "phase1" / "phase1_results.json"
    if not path.exists():
        return None
    results = json.loads(path.read_text(encoding="utf-8"))
    per_seed = {r["seed"]: r for r in results["per_seed"]}
    if results["meta"]["config"] != p1 or not set(p1_seeds) <= set(per_seed):
        return None
    return {
        s: {
            n: {
                k: per_seed[s]["classification"]["lr_shrunk"][str(n)][k]
                for k in ("accuracy", "log_loss")
            }
            for n in sizes
        }
        for s in p1_seeds
    }


def aggregate_a(reps: list[dict]) -> dict[str, Any]:
    """Means over replicates with 95% t-intervals (Phase 1's convention)."""
    from experiments.phase1_report import ci, crossing

    sizes = reps[0]["sizes"]
    models = list(reps[0]["metrics"])
    out: dict[str, Any] = {"sizes": sizes, "metrics": {}, "paired": {}, "hands_to": {}}
    for m in models:
        out["metrics"][m] = {
            str(n): {
                k: ci([r["metrics"][m][str(n)][k] for r in reps])
                for k in reps[0]["metrics"][m][str(n)]
            }
            for n in sizes
        }
        out["hands_to"][m] = {}
        for t in THRESHOLDS:
            x = [
                crossing(sizes, [r["metrics"][m][str(n)]["accuracy"] for n in sizes], t)
                for r in reps
            ]
            ok = [v for v in x if np.isfinite(v)]
            out["hands_to"][m][str(t)] = {
                "reached": len(ok),
                "geometric_mean": float(np.exp(np.mean(np.log(ok)))) if ok else None,
                "range": [min(ok), max(ok)] if ok else None,
            }
    for m in reps[0]["paired"]:
        out["paired"][m] = {
            str(n): {
                k: ci([r["paired"][m][str(n)][k] for r in reps])
                for k in ("log_loss_gain", "log_loss_gain_by_size", "accuracy_gain")
            }
            for n in sizes
        }
    out["differences"] = {
        f"{x}-{y}": {
            str(n): ci(
                [
                    r["metrics"][x][str(n)]["log_loss"] - r["metrics"][y][str(n)]["log_loss"]
                    for r in reps
                ]
            )
            for n in sizes
        }
        for x, y in DIFFERENCES
        if x in models and y in models
    }
    out["training"] = {
        m: {
            k: ci([r["training"][m][k] for r in reps])
            for k in ("epochs", "best_epoch", "temperature", "seconds")
        }
        | {"parameters": reps[0]["training"][m]["parameters"]}
        for m in reps[0]["training"]
    }
    return out


#: Log-loss differences between models reported in part A (first minus second).
DIFFERENCES = (("seq", "bag"), ("counters", "bag"))


def improvement(small: list[dict], large: list[dict]) -> dict[str, Any]:
    """Part C: log-loss with Phase 1's training bots minus log-loss with the extra ones, per
    model and sample size (positive: more players helped), paired by replicate."""
    from experiments.phase1_report import ci

    sizes = small[0]["sizes"]
    return {
        m: {
            str(n): ci(
                [
                    a["metrics"][m][str(n)]["log_loss"] - b["metrics"][m][str(n)]["log_loss"]
                    for a, b in zip(small, large, strict=True)
                ]
            )
            for n in sizes
        }
        for m in large[0]["metrics"]
    }


def predictions(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any] | None) -> dict:
    """The pre-registered predictions (see the module docstring)."""
    sizes = [str(n) for n in a["sizes"]]
    out: dict[str, Any] = {}

    def fmt(d: dict[str, Any], key: str = "mean") -> str:
        return f"{d[key]:+.3f}"

    gain = a["paired"]["seq"]
    above = [n for n in sizes if gain[n]["log_loss_gain"]["lower"] > 0]
    acc_above = [n for n in sizes if gain[n]["accuracy_gain"]["lower"] > 0]
    out["A1"] = {
        "claim": "the sequence model beats the logistic model at no sample size (no interval "
        "of its log-loss or accuracy gain above 0)",
        "result": "log-loss gains "
        + ", ".join(f"{int(n):,}: {fmt(gain[n]['log_loss_gain'])}" for n in sizes)
        + (f"; intervals above 0 at {', '.join(above + acc_above)}" if above or acc_above else ""),
        "holds": not above and not acc_above,
    }
    d = a["differences"]["seq-bag"]
    out["A2"] = {
        "claim": "the order of a hand's decisions changes log-loss by less than 0.02 nats at every "
        "sample size (|sequence model - order dropped|)",
        "result": ", ".join(f"{int(n):,}: {fmt(d[n])}" for n in sizes),
        "holds": all(abs(d[n]["mean"]) < 0.02 for n in sizes),
    }
    d = a["differences"]["counters-bag"]
    out["A3"] = {
        "claim": "tokens beat the stats counters by less than 0.02 nats at every sample size "
        "(counters network - order-dropped token network)",
        "result": ", ".join(f"{int(n):,}: {fmt(d[n])}" for n in sizes),
        "holds": all(d[n]["mean"] < 0.02 for n in sizes),
    }
    hum = b["humans"]["consistency"]["seq"]
    early = [n for n in ("25", "50", "100") if n in hum]
    out["B1"] = {
        "claim": "on the professionals the sequence model is overconfident against matched bots "
        "at 25, 50 and 100 hands (each interval above 0)",
        "result": ", ".join(f"{int(n):,}: {fmt(hum[n]['difference'], 'value')}" for n in early),
        "holds": len(early) == 3 and all(hum[n]["difference"]["lower"] > 0 for n in early),
    }
    lead = {
        m: [n for n in ("25", "50", "100") if n in cons and cons[n]["difference"]["lower"] > 0]
        for m, cons in b["control"]["consistency"].items()
    }
    out["B2"] = {
        "claim": "on the control neither model is overconfident at 25, 50 or 100 hands (no "
        "interval above 0)",
        "result": "; ".join(
            f"{MODEL_LABELS[m]}: " + (", ".join(v) if v else "none above 0")
            for m, v in lead.items()
        ),
        "holds": not any(lead.values()),
    }
    if c is not None:
        imp = c["improvement"]
        out["C1"] = {
            "claim": "with the extra training bots the logistic model's log-loss changes by less "
            "than 0.02 nats at every sample size",
            "result": ", ".join(f"{int(n):,}: {fmt(imp['lr'][n])}" for n in sizes),
            "holds": all(abs(imp["lr"][n]["mean"]) < 0.02 for n in sizes),
        }
        out["C2"] = {
            "claim": "the order-dropped token network improves by at least 0.02 nats at every "
            "sample size (mean), with every interval above 0",
            "result": ", ".join(f"{int(n):,}: {fmt(imp['bag'][n])}" for n in sizes),
            "holds": all(
                imp["bag"][n]["mean"] >= 0.02 and imp["bag"][n]["lower"] > 0 for n in sizes
            ),
        }
        g = c["paired"]["bag"]
        out["C3"] = {
            "claim": "with the extra bots the token network and the logistic model are within "
            "0.03 nats at every sample size (every interval of the gain inside +-0.03)",
            "result": ", ".join(
                f"{int(n):,}: " + _ci(g[n]["log_loss_gain"], "+.3f") for n in sizes
            ),
            "holds": all(
                g[n]["log_loss_gain"]["lower"] >= -0.03 and g[n]["log_loss_gain"]["upper"] <= 0.03
                for n in sizes
            ),
        }
    return out


def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    import torch

    from experiments.common import derive_seeds, run_metadata, save_json
    from experiments.phase3_priors import random_seating

    out_dir = out_dir or RESULTS_ROOT / "phase3" / "sequence"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg["torch_threads"])
    meta = run_metadata(seed, cfg)
    meta["torch_threads"] = torch.get_num_threads()
    meta["sequence_fingerprint"] = _fingerprint()
    t_start = time.perf_counter()
    ss = [int(x) for x in np.random.SeedSequence([seed, 38]).generate_state(4)]
    p1 = load_config(REPO_ROOT / cfg["phase1_config"])
    p1_seeds = cfg.get("replicate_seeds") or derive_seeds(cfg["phase1_master_seed"], p1["n_seeds"])
    if cfg["real_source"] == "pluribus":
        humans = load_pluribus(cfg, workers)
    else:
        plan = random_seating(cfg, np.random.default_rng(ss[0]))
        humans = simulate_standin(plan, cfg["bot_player"], ss[1], "standin", workers)
    pops = {
        "humans": humans,
        "control": simulate_standin(humans.seating, cfg["bot_player"], ss[2], "control", workers),
    }
    views = {k: PoolView(p) for k, p in pops.items()}
    players = {
        k: [i for i, pid in enumerate(p.ids) if pid != cfg["bot_player"]] for k, p in pops.items()
    }
    for k in pops:
        print(
            f"{k}: {len(players[k])} players, {int(views[k].hands[players[k]].sum())} player-hands"
        )

    reps_a, reps_c, checks = [], [], []
    reps_b: dict[str, list] = {k: [] for k in pops}
    for rep, rep_seed in enumerate(p1_seeds):
        t0 = time.perf_counter()
        print(f"replicate {rep + 1}/{len(p1_seeds)} ({rep_seed})", flush=True)
        data = generate_sequences(rep_seed, p1, workers=workers)
        checks.append({"seed": rep_seed, **check_against_phase1(data, p1, workers)})
        if not checks[-1]["identical"]:
            raise AssertionError(f"replicate {rep_seed} differs from Phase 1's: {checks[-1]}")
        a, models = part_a(data, p1, cfg)
        reps_a.append(a)
        bots = PoolView(data.test)
        for k in pops:
            reps_b[k].append(transfer_rep(models, views[k], players[k], bots, a["sizes"], cfg))
        if cfg.get("extra_per_archetype"):
            extra = generate_extra(rep_seed, p1, cfg["extra_per_archetype"], workers=workers)
            big = SeqSeedData(rep_seed, combine(data.train, extra), data.test)
            reps_c.append(part_a(big, p1, cfg, variants=cfg["scale_variants"])[0])
            n_train_c = len(big.train.player_ids)
        print(f"      {time.perf_counter() - t0:.0f}s", flush=True)

    sizes = reps_a[0]["sizes"]
    ref = phase1_reference(p1, p1_seeds, sizes)
    rng = np.random.default_rng(ss[3])
    r: dict[str, Any] = {
        "meta": meta,
        "phase1": {"config": cfg["phase1_config"], "seeds": p1_seeds},
        "replicate_checks": checks,
        "phase1_match": None
        if ref is None
        else max(
            abs(a["metrics"]["lr"][str(n)][k] - ref[s][n][k])
            for a, s in zip(reps_a, p1_seeds, strict=True)
            for n in sizes
            for k in ("accuracy", "log_loss")
        ),
        "a": aggregate_a(reps_a),
        "per_replicate": reps_a,
        "b": {
            k: {
                "data": {
                    "source": pops[k].source,
                    "players": len(players[k]),
                    "hands": {pops[k].ids[i]: int(views[k].hands[i]) for i in players[k]},
                },
                **summarize_transfer(reps_b[k], pops[k].ids, cfg, rng),
            }
            for k in pops
        },
    }
    if reps_c:
        r["c"] = {
            "extra_per_archetype": cfg["extra_per_archetype"],
            "training_players": n_train_c,
            **aggregate_a(reps_c),
            "improvement": improvement(reps_a, reps_c),
        }
        r["per_replicate_c"] = reps_c
    r["predictions"] = predictions(r["a"], r["b"], r.get("c"))
    r["seconds"] = time.perf_counter() - t_start
    save_json(r, out_dir / "sequence.json")
    (out_dir / "sequence_summary.md").write_text(
        markdown_summary(r), encoding="utf-8", newline="\n"
    )
    make_figures(r, out_dir)
    return r


# ---------------------------------------------------------------------------- report
MODEL_LABELS = {
    "lr": "Logistic, shrunk stats (Phase 1)",
    "gbm": "Boosted trees, shrunk stats",
    "seq": "Sequence model (GRU)",
    "bag": "Sequence model, order dropped",
    "counters": "Same network on the stats counters",
}
POP_LABELS = {"humans": "Professionals", "control": "Control: bots in the professionals' seats"}


def _ci(d: dict[str, Any], fmt: str = ".3f", scale: float = 1.0) -> str:
    v, lo, hi = (np.nan if d[k] is None else d[k] * scale for k in ("mean", "lower", "upper"))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{v:{fmt}}"
    return f"{v:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def markdown_summary(r: dict[str, Any]) -> str:
    a, b = r["a"], r["b"]
    sizes = a["sizes"]
    models = list(a["metrics"])
    lines = [
        "# P3.8: an archetype classifier on raw action sequences",
        "",
        f"{len(r['phase1']['seeds'])} Phase 1 replicates (re-simulated with every decision; "
        f"identical to Phase 1: {all(c['identical'] for c in r['replicate_checks'])}; largest "
        "difference from Phase 1's reported logistic accuracy and log-loss: "
        + ("n/a" if r["phase1_match"] is None else f"{r['phase1_match']:.1e}")
        + "). Brackets: 95% t-intervals across replicates unless stated.",
        "",
        "## Pre-registered predictions",
        "",
    ]
    for key, p in r["predictions"].items():
        verdict = "**holds**" if p["holds"] else "**does not hold**"
        lines.append(f"- {key} {p['claim']}: {p['result']}; {verdict}")
    lines += ["", "## A. Hands to identify (held-out bots, Phase 1's windows)", ""]
    for metric, name, fmt, scale in (
        ("accuracy", "Accuracy", ".1f", 100.0),
        ("log_loss", "Log-loss (nats; one temperature)", ".3f", 1.0),
        ("log_loss_by_size", "Log-loss (one temperature per sample size)", ".3f", 1.0),
        ("ece", "Expected calibration error (10 bins; one temperature)", ".3f", 1.0),
    ):
        lines += [f"{name}:", ""]
        lines += _table(
            ["Hands", *(MODEL_LABELS[m] for m in models)],
            [
                [f"{n:,}", *(_ci(a["metrics"][m][str(n)][metric], fmt, scale) for m in models)]
                for n in sizes
            ],
        )
        lines.append("")
    lines += ["Log-loss gain over the logistic model (positive: lower log-loss), paired:", ""]
    paired = list(a["paired"])
    lines += _table(
        ["Hands", *(f"{MODEL_LABELS[m]}: one T / T per size" for m in paired)],
        [
            [
                f"{n:,}",
                *(
                    f"{_ci(a['paired'][m][str(n)]['log_loss_gain'], '+.3f')} / "
                    f"{_ci(a['paired'][m][str(n)]['log_loss_gain_by_size'], '+.3f')}"
                    for m in paired
                ),
            ]
            for n in sizes
        ],
    )
    lines += ["", "Hands needed to reach an accuracy (geometric mean over replicates):", ""]
    lines += _table(
        ["Model", *(f"{t:.0%}" for t in THRESHOLDS)],
        [
            [
                MODEL_LABELS[m],
                *(
                    "never"
                    if (h := a["hands_to"][m][str(t)])["geometric_mean"] is None
                    else f"{h['geometric_mean']:,.0f} ({h['reached']}/{len(r['phase1']['seeds'])})"
                    for t in THRESHOLDS
                ),
            ]
            for m in models
        ],
    )
    lines += ["", "Training (held-out training bots pick the epoch and the temperature):", ""]
    lines += _table(
        ["Model", "Parameters", "Best epoch", "Temperature", "Seconds"],
        [
            [
                MODEL_LABELS[m],
                f"{t['parameters']:,}",
                _ci(t["best_epoch"], ".1f"),
                _ci(t["temperature"], ".2f"),
                _ci(t["seconds"], ".0f"),
            ]
            for m, t in a["training"].items()
        ],
    )
    lines += ["", "Log-loss differences between the neural models (first minus second):", ""]
    lines += _table(
        ["Hands", *(f"{MODEL_LABELS[x]} - {MODEL_LABELS[y]}" for x, y in DIFFERENCES)],
        [
            [f"{n:,}", *(_ci(a["differences"][f"{x}-{y}"][str(n)], "+.3f") for x, y in DIFFERENCES)]
            for n in sizes
        ],
    )
    if "c" in r:
        c = r["c"]
        cm = list(c["metrics"])
        lines += [
            "",
            f"## C. More training players ({c['training_players']:,} training bots instead of "
            f"Phase 1's {c['training_players'] - 5 * c['extra_per_archetype']:,})",
            "",
            "Log-loss (one temperature):",
            "",
        ]
        lines += _table(
            ["Hands", *(MODEL_LABELS[m] for m in cm)],
            [[f"{n:,}", *(_ci(c["metrics"][m][str(n)]["log_loss"]) for m in cm)] for n in sizes],
        )
        lines += ["", "Improvement from the extra players (log-loss before minus after):", ""]
        lines += _table(
            ["Hands", *(MODEL_LABELS[m] for m in cm)],
            [[f"{n:,}", *(_ci(c["improvement"][m][str(n)], "+.3f") for m in cm)] for n in sizes],
        )
        lines += ["", "Log-loss gain over the logistic model, both with the extra players:", ""]
        lines += _table(
            ["Hands", *(MODEL_LABELS[m] for m in c["paired"])],
            [
                [
                    f"{n:,}",
                    *(_ci(c["paired"][m][str(n)]["log_loss_gain"], "+.3f") for m in c["paired"]),
                ]
                for n in sizes
            ],
        )
    lines += ["", "## B. On the professionals (and the control)", ""]
    for k, pop in b.items():
        names = [MODEL_LABELS[m] for m in pop["verdicts"]]
        lines += [
            f"### {POP_LABELS[k]}",
            "",
            f"{pop['data']['players']} players, {sum(pop['data']['hands'].values()):,} "
            "player-hands. Consistency verdicts: "
            + ", ".join(f"{n} *{v}*" for n, v in zip(names, pop["verdicts"].values(), strict=True))
            + ".",
            "",
        ]
        lines += _table(
            ["Hands", *(f"{x}: gap - matched bots' gap" for x in names)],
            [
                [
                    f"{int(n):,}",
                    *(
                        "n/a"
                        if n not in pop["consistency"][m]
                        else "{value:+.3f} [{lower:+.3f}, {upper:+.3f}]".format(
                            **pop["consistency"][m][n]["difference"]
                        )
                        for m in pop["verdicts"]
                    ),
                ]
                for n in sorted({n for c in pop["consistency"].values() for n in c}, key=int)
            ],
        )
        lines += [""]
        lines += _table(
            ["Hands", *(f"{x}: top probability" for x in names), "Same label (players / bots)"],
            [
                [
                    f"{int(n):,}",
                    *(f"{pop['labels'][n][m]['top_prob']:.2f}" for m in pop["verdicts"]),
                    f"{pop['agreement'][n]['players']:.0%} / {pop['agreement'][n]['bots']:.0%}",
                ]
                for n in pop["labels"]
            ],
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def make_figures(r: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import band, legend, line, new_figure, save_both, style_axes, title

    a = r["a"]
    sizes = a["sizes"]
    reps = f"{len(r['phase1']['seeds'])} replicates"

    def colors(theme):
        s = theme["series"]
        return {"lr": s[0], "gbm": s[5], "seq": s[1], "bag": s[2], "counters": s[3]}

    def axis(ax, theme):
        style_axes(ax, theme)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{n:,}" for n in sizes])
        ax.minorticks_off()
        ax.set_xlabel("Hands observed")

    def draw_hands(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(12, 4.6))
        for ax, metric, label, hi in (
            (axes[0], "accuracy", "Accuracy", 1.0),
            (axes[1], "log_loss", "Log-loss (nats)", np.inf),
        ):
            axis(ax, theme)
            for m, color in colors(theme).items():
                if m not in a["metrics"]:
                    continue
                d = [a["metrics"][m][str(n)][metric] for n in sizes]
                lo = np.clip([x["lower"] for x in d], 0, hi)
                band(ax, sizes, lo, np.clip([x["upper"] for x in d], 0, hi), color)
                line(ax, theme, sizes, [x["mean"] for x in d], color, MODEL_LABELS[m])
            ax.set_ylabel(label)
        legend(axes[1], theme, loc="upper right")
        title(
            fig,
            theme,
            "Identifying a bot's archetype: hand-written stats vs raw action sequences",
            f"Held-out bots, Phase 1's windows; mean over {reps}, bands: 95% t-intervals.",
        )
        return fig

    save_both(draw_hands, out_dir, "sequence_hands")

    def draw_gain(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(12, 4.6), sharey=True)
        for ax, key, name in (
            (axes[0], "log_loss_gain", "One temperature"),
            (axes[1], "log_loss_gain_by_size", "One temperature per sample size"),
        ):
            axis(ax, theme)
            for m, color in colors(theme).items():
                if m not in a["paired"]:
                    continue
                d = [a["paired"][m][str(n)][key] for n in sizes]
                band(ax, sizes, [x["lower"] for x in d], [x["upper"] for x in d], color)
                line(ax, theme, sizes, [x["mean"] for x in d], color, MODEL_LABELS[m])
            ax.axhline(0, color=theme["axis"], linewidth=1)
            ax.set_title(name, loc="left", fontsize=10, color=theme["ink"])
        axes[0].set_ylabel("Log-loss gain over the logistic model (nats)")
        legend(axes[1], theme, loc="lower right")
        title(
            fig,
            theme,
            "How much better than Phase 1's classifier, on the same windows?",
            "Mean over windows of the logistic model's log-loss minus the model's (above 0: "
            f"better); bands: 95% t-intervals over {reps}.",
        )
        return fig

    save_both(draw_gain, out_dir, "sequence_gain")

    def draw_transfer(theme):
        pops = list(r["b"])
        fig, axes = new_figure(theme, 1, len(pops), figsize=(6 * len(pops), 4.6), squeeze=False)
        for ax, k in zip(axes[0], pops, strict=True):
            style_axes(ax, theme)
            cons = r["b"][k]["consistency"]
            for m, color in colors(theme).items():
                if m not in cons or not cons[m]:
                    continue
                ns = [int(n) for n in cons[m]]
                d = [cons[m][str(n)]["difference"] for n in ns]
                band(ax, ns, [x["lower"] for x in d], [x["upper"] for x in d], color)
                line(ax, theme, ns, [x["value"] for x in d], color, MODEL_LABELS[m])
                ax.set_xscale("log")
                ax.set_xticks(ns, [f"{n:,}" for n in ns])
            for y in (-0.10, 0.10):
                ax.axhline(y, color=theme["muted"], linewidth=1, linestyle="--")
            ax.axhline(0, color=theme["axis"], linewidth=1)
            ax.minorticks_off()
            ax.set_xlabel("Hands in the window")
            ax.set_title(POP_LABELS[k], loc="left", fontsize=10, color=theme["ink"])
        axes[0][0].set_ylabel("Gap minus the matched bots' gap")
        legend(axes[0][-1], theme, loc="upper right")
        title(
            fig,
            theme,
            "Does either model's confidence hold up on real players?",
            "Gap = top probability - agreement with the label of a later 1,000-hand block, "
            "relative to held-out bots of the same label; bands: 95% intervals over players.",
        )
        return fig

    save_both(draw_transfer, out_dir, "sequence_transfer")

    if "c" not in r:
        return
    c = r["c"]

    def draw_scale(theme):
        models = [m for m in ("lr", "gbm", "bag", "counters") if m in c["metrics"]]
        fig, axes = new_figure(theme, 1, len(models), figsize=(3.4 * len(models), 4.2), sharey=True)
        for ax, m in zip(axes, models, strict=True):
            axis(ax, theme)
            color = colors(theme)[m]
            for block, style, label in (
                (a, "--", "Phase 1's training bots"),
                (c, "-", "with the extra bots"),
            ):
                d = [block["metrics"][m][str(n)]["log_loss"] for n in sizes]
                band(ax, sizes, [x["lower"] for x in d], [x["upper"] for x in d], color)
                line(ax, theme, sizes, [x["mean"] for x in d], color, label, linestyle=style)
            ax.set_title(MODEL_LABELS[m], loc="left", fontsize=9, color=theme["ink"])
        axes[0].set_ylabel("Log-loss (nats)")
        legend(axes[-1], theme, loc="upper right")
        title(
            fig,
            theme,
            f"More training bots: {c['training_players']:,} instead of "
            f"{c['training_players'] - 5 * c['extra_per_archetype']:,}",
            "Same held-out bots and windows; dashed: Phase 1's training bots, solid: with the "
            f"extra ones. Bands: 95% t-intervals over {reps}.",
        )
        return fig

    save_both(draw_scale, out_dir, "sequence_scale")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--quick", action="store_true", help="use configs/phase3_quick.yaml")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    path = CONFIG.with_name("phase3_quick.yaml") if args.quick else args.config
    cfg = load_config(path)["sequence"]
    r = run(args.seed, cfg=cfg, workers=args.workers or cfg.get("workers"), out_dir=args.out)
    print(f"done in {r['seconds']:.0f}s")


if __name__ == "__main__":
    main()

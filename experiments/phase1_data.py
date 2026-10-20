"""Phase 1 data generation: one independent replicate per seed.

For each seed we draw two disjoint bot populations, train and test, and simulate each in its
own league. Test players never sit at a table with training players, never appear in any
training row, and are never used to fit priors or calibration. The league returns per-hand stat
counters for every player plus decision rows (public situation + action) for the hands inside
the action-prediction windows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.common import REPO_ROOT
from poker_lab.bots import GroundTruth, make_population
from poker_lab.bots.league import LeagueConfig, run_league
from poker_lab.profiling import DecisionRowExtractor

CACHE_DIR = REPO_ROOT / "experiments" / "cache"


@dataclass
class PoolData:
    """One simulated population."""

    player_ids: list[str]
    labels: np.ndarray  # archetype index per player
    params: np.ndarray  # true BotParams per player (players x 9)
    counters: np.ndarray  # (players, hands, counters) int16
    decisions: np.ndarray  # rows: pop index, seq, situation features..., label
    hands_simulated: int
    seconds: float

    @property
    def cumulative(self) -> np.ndarray:
        return self.counters.astype(np.int32).cumsum(axis=1)


@dataclass
class SeedData:
    seed: int
    train: PoolData
    test: PoolData


def decision_windows(checkpoints: list[int], window: int) -> tuple[tuple[int, int], ...]:
    return tuple((c, c + window) for c in checkpoints)


def _simulate_pool(pop, cfg: dict, seed: int, workers: int | None) -> PoolData:
    truth = GroundTruth(pop)
    league = LeagueConfig(
        hands_per_player=cfg["hands_per_player"],
        block_hands=cfg["block_hands"],
        equity_samples=cfg.get("equity_samples", 150),
    )
    extractor = DecisionRowExtractor(
        decision_windows(cfg["action_checkpoints"], cfg["action_window"])
    )
    t0 = time.perf_counter()
    result = run_league(pop, league, seed=seed, workers=workers, extractor=extractor)
    return PoolData(
        player_ids=result.player_ids,
        labels=truth.labels(result.player_ids),
        params=truth.param_matrix(result.player_ids),
        counters=result.counters,
        decisions=result.extra,
        hands_simulated=result.hands_simulated,
        seconds=time.perf_counter() - t0,
    )


def _cache_path(seed: int, cfg: dict) -> Path:
    key = (
        f"s{seed}_tr{cfg['n_train_per_archetype']}_te{cfg['n_test_per_archetype']}"
        f"_h{cfg['hands_per_player']}_b{cfg['block_hands']}"
        f"_cp{'-'.join(map(str, cfg['action_checkpoints']))}_w{cfg['action_window']}"
    )
    return CACHE_DIR / f"phase1_{key}.npz"


def generate(
    seed: int, cfg: dict, *, workers: int | None = None, use_cache: bool = True
) -> SeedData:
    """Simulate (or load from ``experiments/cache``) the train and test pools for ``seed``."""
    path = _cache_path(seed, cfg)
    if use_cache and path.exists():
        return load(path, seed)
    rng = np.random.default_rng(seed)
    train_pop = make_population(cfg["n_train_per_archetype"], rng, id_prefix="a")
    test_pop = make_population(cfg["n_test_per_archetype"], rng, id_prefix="b")
    train_ids = {p.player_id for p in train_pop}
    if train_ids & {p.player_id for p in test_pop}:
        raise AssertionError("train and test populations share a player id")
    sub = np.random.SeedSequence(seed).generate_state(2)
    data = SeedData(
        seed=seed,
        train=_simulate_pool(train_pop, cfg, int(sub[0]), workers),
        test=_simulate_pool(test_pop, cfg, int(sub[1]), workers),
    )
    if use_cache:
        save(data, path)
    return data


def save(data: SeedData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name in ("train", "test"):
        pool: PoolData = getattr(data, name)
        arrays.update(
            {
                f"{name}_ids": np.array(pool.player_ids),
                f"{name}_labels": pool.labels,
                f"{name}_params": pool.params,
                f"{name}_counters": pool.counters,
                f"{name}_decisions": pool.decisions,
                f"{name}_meta": np.array([pool.hands_simulated, pool.seconds]),
            }
        )
    np.savez_compressed(path, **arrays)


def load(path: Path, seed: int) -> SeedData:
    z = np.load(path)
    pools = {}
    for name in ("train", "test"):
        meta = z[f"{name}_meta"]
        pools[name] = PoolData(
            player_ids=[str(x) for x in z[f"{name}_ids"]],
            labels=z[f"{name}_labels"],
            params=z[f"{name}_params"],
            counters=z[f"{name}_counters"],
            decisions=z[f"{name}_decisions"],
            hands_simulated=int(meta[0]),
            seconds=float(meta[1]),
        )
    return SeedData(seed=seed, train=pools["train"], test=pools["test"])

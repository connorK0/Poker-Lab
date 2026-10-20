"""League simulation: a bot population playing many hands at randomly re-drawn tables.

Each *round* the population is shuffled into tables and every table plays ``block_hands``
hands, so every player plays exactly ``rounds * block_hands`` hands against a changing mix of
opponents. Tables in a round are independent tasks and run in parallel worker processes. Each
task's seed is derived from (seed, round, table) with ``numpy.random.SeedSequence``, so results
do not depend on the number of workers.

Workers return compact arrays rather than hand objects:

* ``counters``: one row per (player, hand) with the stats counters of ``profiling.stats``,
  computed from the hand redacted to public information (what a profiler could see);
* optional extractor rows (e.g. decision points for action prediction), via a picklable
  top-level function ``extractor(hand, pop_index, seq) -> list[tuple]``. ``pop_index`` maps
  player id to population index, and ``seq`` is the hand's position in every seated player's
  own history (all players at a table play every hand);
* optionally a gzipped JSON Lines sample of the raw hand histories.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from poker_lab.bots.population import PlayerSpec
from poker_lab.engine import SeatedPlayer, TableSimulator
from poker_lab.profiling.stats import N_COUNTERS, hand_counters
from poker_lab.schema import HandHistory, TableConfig, redact_hole_cards, write_jsonl

Extractor = Callable[[HandHistory, dict[str, int], int], list[tuple[Any, ...]]]


@dataclass(frozen=True)
class LeagueConfig:
    hands_per_player: int
    block_hands: int = 50
    table_size: int = 6
    small_blind: int = 1
    big_blind: int = 2
    starting_stack_bb: int = 100
    equity_samples: int = 150

    def __post_init__(self) -> None:
        if self.hands_per_player % self.block_hands:
            raise ValueError("hands_per_player must be a multiple of block_hands")

    @property
    def rounds(self) -> int:
        return self.hands_per_player // self.block_hands


@dataclass
class LeagueResult:
    player_ids: list[str]
    counters: np.ndarray  # (n_players, hands_per_player, N_COUNTERS) int16, in play order
    extra: list[tuple[Any, ...]] = field(default_factory=list)
    hands_simulated: int = 0


@dataclass(frozen=True)
class _Task:
    round_index: int
    table_index: int
    seed: int
    players: tuple[tuple[int, PlayerSpec], ...]  # (population index, spec) per seat
    config: LeagueConfig
    extractor: Extractor | None
    save_path: str | None


def _run_task(task: _Task) -> tuple[_Task, np.ndarray, list[tuple[Any, ...]], int]:
    cfg = task.config
    table = TableConfig(
        f"r{task.round_index}t{task.table_index}", cfg.table_size, cfg.small_blind, cfg.big_blind
    )
    seated = [
        SeatedPlayer(seat, spec.player_id, spec.make_agent(equity_samples=cfg.equity_samples))
        for seat, (_, spec) in enumerate(task.players)
    ]
    sim = TableSimulator(
        table,
        seated,
        seed=task.seed,
        starting_stack=cfg.starting_stack_bb * cfg.big_blind,
        hand_id_prefix=f"{table.table_id}-s{task.seed}-",
    )
    row_of = {spec.player_id: i for i, (_, spec) in enumerate(task.players)}
    pop_index = {spec.player_id: p for p, spec in task.players}
    out = np.zeros((len(task.players), cfg.block_hands, N_COUNTERS), dtype=np.int16)
    extra: list[tuple[Any, ...]] = []
    hands: list[HandHistory] = []
    seq_base = task.round_index * cfg.block_hands
    for h in range(cfg.block_hands):
        hand = redact_hole_cards(sim.play_hand())  # what any observer could know
        for pid, row in hand_counters(hand):
            out[row_of[pid], h] = row
        if task.extractor is not None:
            extra.extend(task.extractor(hand, pop_index, seq_base + h))
        if task.save_path is not None:
            hands.append(hand)
    if task.save_path is not None:
        write_jsonl(hands, task.save_path)
    return task, out, extra, cfg.block_hands


def run_league(
    players: Sequence[PlayerSpec],
    config: LeagueConfig,
    *,
    seed: int,
    workers: int | None = None,
    extractor: Extractor | None = None,
    save_hands_dir: str | Path | None = None,
    save_rounds: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> LeagueResult:
    """Simulate the league. ``save_hands_dir`` stores the first ``save_rounds`` rounds' hands
    (redacted) as gzipped JSON Lines, one file per table."""
    n = len(players)
    if n % config.table_size:
        raise ValueError(f"population size {n} is not a multiple of {config.table_size}")
    tasks = []
    for r in range(config.rounds):
        round_seq = np.random.SeedSequence([seed, r])
        order = np.random.default_rng(round_seq).permutation(n)
        for t in range(n // config.table_size):
            idx = order[t * config.table_size : (t + 1) * config.table_size]
            task_seed = int(np.random.SeedSequence([seed, r, t]).generate_state(1)[0])
            save = None
            if save_hands_dir is not None and r < save_rounds:
                save = str(Path(save_hands_dir) / f"round{r:03d}_table{t:03d}.jsonl.gz")
            tasks.append(
                _Task(r, t, task_seed, tuple((int(i), players[int(i)]) for i in idx), config,
                      extractor, save)
            )  # fmt: skip

    counters = np.zeros((n, config.hands_per_player, N_COUNTERS), dtype=np.int16)
    extra: list[tuple[Any, ...]] = []
    total = 0
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)

    def collect(task: _Task, block: np.ndarray, rows: list[tuple[Any, ...]], n_hands: int):
        nonlocal total
        start = task.round_index * config.block_hands
        for seat, (pop_i, _) in enumerate(task.players):
            counters[pop_i, start : start + config.block_hands] = block[seat]
        extra.extend(rows)
        total += n_hands
        if progress is not None:
            progress(total, config.rounds * (n // config.table_size) * config.block_hands)

    if workers <= 1:
        for task in tasks:
            collect(*_run_task(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_run_task, tasks, chunksize=4):
                collect(*result)
    return LeagueResult(
        player_ids=[p.player_id for p in players],
        counters=counters,
        extra=extra,
        hands_simulated=total,
    )

"""Duplicate dealing (P2.9): every deal is played at n tables, the players' seats rotated.

Card luck dominates short poker samples. In a duplicate match between n players, deal ``i`` is
played at n tables with the same deck order and the same button seat (seat 0 for the first
deal; the button then moves the same way at every table). The players sit in seats 0 to n - 1,
rotated cyclically: at table ``r`` player ``j`` sits in seat ``(j + r) mod n``. Cards are dealt
by position, so across the n tables of a deal every player holds every seat's cards and
position exactly once, and the players keep their order around the table. Heads-up this is a
seat swap: the second table seats the two players the other way round. The rotation never
changes who sits to whose left: with three or more different players a match measures the
seating order given in ``players`` (the other cyclic orders are not played), so play matches
with the players listed in other orders when seating matters.

Why the rotation: the simulator takes no rake, so the seats' results in a hand sum to zero, and
each player holds each seat once per deal. For n copies of one deterministic agent that
ignores player ids (its play depends only on what its seat sees), a seat's result is the same
at every table (the odd chip of a split pot goes by position, not by player), so each player's
mean over the n tables of a deal is the mean of the seats' results: exactly zero. For
different agents the plays differ from table to table, and the rotation removes part of the
card luck. The per-deal mean is the unit of
``DuplicateDealEvaluator(group_size=n)``, which takes the results in the order of
``DuplicateMatch.grouped_results``.

Agents at different tables are *separate instances* built by factories, so an agent cannot
remember how an earlier play of a deal went. Each table also has its own decision RNG, so
agents' own randomness is not mirrored between the plays (only the cards are).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from poker_lab.engine.agent import Agent
from poker_lab.engine.simulator import SeatedPlayer, TableSimulator
from poker_lab.evaluation.winrate import (
    DuplicateDealEvaluator,
    HandResult,
    WinRateEstimate,
    hand_results,
)
from poker_lab.schema import FULL_DECK, HandHistory, TableConfig


@dataclass(frozen=True)
class DuplicateMatch:
    """Hands from the n tables of a match; ``tables[r][i]`` is deal ``i`` at table ``r``."""

    tables: tuple[tuple[HandHistory, ...], ...]

    @property
    def n_tables(self) -> int:
        """The number of tables, which is the number of players."""
        return len(self.tables)

    @property
    def first(self) -> tuple[HandHistory, ...]:
        """The hands at table 0."""
        return self.tables[0]

    @property
    def second(self) -> tuple[HandHistory, ...]:
        """The hands at table 1; ``first[i]`` and ``second[i]`` share deal ``i``."""
        return self.tables[1]

    def grouped_results(self, player_id: str) -> list[HandResult]:
        """``player_id``'s results deal by deal, tables in order within a deal: [d0 t0, d0 t1,
        ..., d0 t(n-1), d1 t0, ...], for ``DuplicateDealEvaluator(group_size=n)`` with n the
        number of tables (``evaluate`` does both)."""
        per_table = [hand_results(hands, player_id) for hands in self.tables]
        if any(len(results) != len(self.first) for results in per_table):
            raise ValueError(f"{player_id!r} did not play every hand of the match")
        return [r for deal in zip(*per_table, strict=True) for r in deal]

    def paired_results(self, player_id: str) -> list[HandResult]:
        """``player_id``'s results ordered [a0, b0, a1, b1, ...] for the paired evaluator: the
        heads-up name of ``grouped_results``. It refuses a match with any number of tables but
        two, whose results the default ``DuplicateDealEvaluator()`` would average two tables at
        a time."""
        if len(self.tables) != 2:
            raise ValueError(
                f"paired results need two tables, not {len(self.tables)}; use grouped_results"
            )
        return self.grouped_results(player_id)

    def evaluate(self, player_id: str, confidence: float = 0.95) -> WinRateEstimate:
        """``player_id``'s win rate with the interval over per-deal means, grouped by this
        match's number of tables."""
        evaluator = DuplicateDealEvaluator(confidence, group_size=self.n_tables)
        return evaluator.evaluate(self.grouped_results(player_id))


def duplicate_match(
    table: TableConfig,
    players: Sequence[tuple[str, Callable[[], Agent]]],
    n_deals: int,
    seed: int,
    *,
    starting_stack: int = 200,
) -> DuplicateMatch:
    """Play ``n_deals`` deals between n players (id, agent factory), each deal at n tables,
    with player ``j`` in seat ``(j + r) mod n`` at table ``r``: n * n_deals hands in all.

    The deck RNG uses the first word of ``np.random.SeedSequence(seed).generate_state(n + 1)``
    and table ``r``'s simulator the word ``r + 1``; table ``r``'s hand ids start with
    ``{table_id}-dup{r + 1}-`` (1-based, as heads-up). The factories are called table by table
    and within a table in seat order, once per agent; each deal is played at the tables in
    order.
    """
    players = tuple(players)
    n = len(players)
    if n < 2:
        raise ValueError("a duplicate match needs at least two players")
    if n > table.max_seats:
        raise ValueError(f"{n} players do not fit at a {table.max_seats}-seat table")
    if len({player_id for player_id, _ in players}) != n:
        raise ValueError("player ids must be distinct")
    deck_seed, *table_seeds = (int(s) for s in np.random.SeedSequence(seed).generate_state(n + 1))
    deck_rng = random.Random(deck_seed)
    sims = []
    for r, table_seed in enumerate(table_seeds):
        seated = []
        for seat in range(n):
            player_id, make_agent = players[(seat - r) % n]
            seated.append(SeatedPlayer(seat, player_id, make_agent()))
        sims.append(
            TableSimulator(
                table,
                seated,
                seed=table_seed,
                starting_stack=starting_stack,
                button_seat=0,
                hand_id_prefix=f"{table.table_id}-dup{r + 1}-",
            )
        )
    hands: list[list[HandHistory]] = [[] for _ in sims]
    for _ in range(n_deals):
        deck = list(FULL_DECK)
        deck_rng.shuffle(deck)
        for sim, played in zip(sims, hands, strict=True):
            played.append(sim.play_hand(deck=deck))
    return DuplicateMatch(tuple(tuple(played) for played in hands))

"""Duplicate dealing for heads-up matches (P2.9): every deal is played twice, seats swapped.

Card luck dominates short poker samples. In a duplicate match, deal ``i`` is played at two
tables with the same deck order and the same button seat, and the players swap seats at the
second table. Cards are dealt by position, so each player holds each side of the deal once. If
a player's result is roughly "skill + luck" at one table and "skill - luck" at the other, the
mean of the pair is mostly skill: the paired estimator (``DuplicateDealEvaluator``) removes the
shared luck from the variance.

Agents at the two tables are *separate instances* built by factories, so an agent cannot
remember how the first play of a deal went. Each table also has its own decision RNG, so
agents' own randomness is not mirrored between the two plays (only the cards are).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from poker_lab.engine.agent import Agent
from poker_lab.engine.simulator import SeatedPlayer, TableSimulator
from poker_lab.evaluation.winrate import HandResult, hand_results
from poker_lab.schema import FULL_DECK, HandHistory, TableConfig


@dataclass(frozen=True)
class DuplicateMatch:
    """Hands from the two tables; ``first[i]`` and ``second[i]`` share deal ``i``."""

    first: tuple[HandHistory, ...]
    second: tuple[HandHistory, ...]

    def paired_results(self, player_id: str) -> list[HandResult]:
        """``player_id``'s results ordered [a0, b0, a1, b1, ...] for the paired evaluator."""
        a = hand_results(self.first, player_id)
        b = hand_results(self.second, player_id)
        if len(a) != len(b) or len(a) != len(self.first):
            raise ValueError(f"{player_id!r} did not play every hand of the match")
        return [r for pair in zip(a, b, strict=True) for r in pair]


def duplicate_match(
    table: TableConfig,
    players: tuple[tuple[str, Callable[[], Agent]], tuple[str, Callable[[], Agent]]],
    n_deals: int,
    seed: int,
    *,
    starting_stack: int = 200,
) -> DuplicateMatch:
    """Play ``n_deals`` deals twice each between two players (id, agent factory)."""
    (id_a, make_a), (id_b, make_b) = players
    deck_seed, seed_1, seed_2 = (int(s) for s in np.random.SeedSequence(seed).generate_state(3))
    deck_rng = random.Random(deck_seed)
    first = TableSimulator(
        table,
        [SeatedPlayer(0, id_a, make_a()), SeatedPlayer(1, id_b, make_b())],
        seed=seed_1,
        starting_stack=starting_stack,
        button_seat=0,
        hand_id_prefix=f"{table.table_id}-dup1-",
    )
    second = TableSimulator(
        table,
        [SeatedPlayer(0, id_b, make_b()), SeatedPlayer(1, id_a, make_a())],
        seed=seed_2,
        starting_stack=starting_stack,
        button_seat=0,
        hand_id_prefix=f"{table.table_id}-dup2-",
    )
    hands_1, hands_2 = [], []
    for _ in range(n_deals):
        deck = list(FULL_DECK)
        deck_rng.shuffle(deck)
        hands_1.append(first.play_hand(deck=deck))
        hands_2.append(second.play_hand(deck=deck))
    return DuplicateMatch(tuple(hands_1), tuple(hands_2))

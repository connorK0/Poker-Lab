"""The hands in ``tests/data/schema_v1_hands.jsonl.gz``.

``v1_fixture_hands`` uses only what schema v1 could express. The file was written by this same
function at commit 0cf0c4f, the last commit with schema v1, so ``test_schema_v2.py`` can check
that reading a v1 file gives exactly the hands the current code produces from the same seeds.
"""

from __future__ import annotations

import random

from poker_lab.bots import Archetype, ParametricBot, sample_params
from poker_lab.engine import AllInAgent, CallingAgent, RandomAgent, SeatedPlayer, TableSimulator
from poker_lab.importers import hand_from_phh
from poker_lab.schema import HandHistory, TableConfig, redact_hole_cards

FIXTURE = "tests/data/schema_v1_hands.jsonl.gz"


def _play(table, seats, agents, stacks, seed, n) -> list[HandHistory]:
    players = [
        SeatedPlayer(s, f"{table.table_id}-p{s}", a) for s, a in zip(seats, agents, strict=True)
    ]
    sim = TableSimulator(table, players, seed=seed, starting_stack=stacks)
    return list(sim.play(n))


def v1_fixture_hands() -> list[HandHistory]:
    import numpy as np
    from pokerkit import HandHistory as PHH

    from tests.test_phh_import import FOREIGN, SHOWDOWN

    rng = np.random.default_rng(5)
    bots = [ParametricBot(sample_params(a, rng)) for a in Archetype]
    hands = _play(TableConfig("arch", 6, 1, 2), range(5), bots, 200, 11, 8)
    hands += _play(TableConfig("hu", 2, 1, 2), range(2), [RandomAgent(), AllInAgent()], 60, 12, 8)
    stacks = {0: 40, 1: 200, 2: 15, 3: 120, 4: 200, 5: 9, 6: 75, 7: 200, 8: 33}
    agents = [RandomAgent() if i % 3 else AllInAgent() for i in range(9)]
    hands += _play(TableConfig("ante", 9, 1, 2, ante=1), range(9), agents, stacks, 13, 8)
    gaps = [CallingAgent(), RandomAgent(), AllInAgent()]
    hands += _play(TableConfig("gaps", 9, 5, 10), (1, 4, 7), gaps, {1: 300, 4: 120, 7: 1000}, 14, 8)
    hands += [hand_from_phh(PHH.loads(FOREIGN)), hand_from_phh(PHH.loads(SHOWDOWN))]
    pick = random.Random(15).sample(range(len(hands)), 6)
    hands += [redact_hole_cards(hands[i], viewer_seat=hands[i].seats[0].seat) for i in pick]
    return hands

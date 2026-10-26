"""The P3.10 experiment's building blocks (``experiments/phase3_schema.py``)."""

from collections import Counter

import pytest

from experiments.phase3_schema import (
    CHECKS,
    MIN_PLAYERS,
    ODD_CHIP,
    STRUCTURES,
    loader_check,
    play_v2_table,
    round_trip_task,
    table_setup,
)
from poker_lab.importers import hand_to_phh


@pytest.mark.parametrize("structure", STRUCTURES)
def test_tables_have_their_structure(structure):
    for seed in range(20):
        setup = table_setup(seed, structure)
        assert len(setup["seats"]) >= MIN_PLAYERS.get(structure, 2)
        table = play_v2_table(setup, 1)[0].table
        bb = table.big_blind
        assert {
            "blinds": not (table.ante or table.big_blind_ante or table.straddles),
            "ante": table.ante > 0,
            "bb_ante": table.big_blind_ante in (bb // 2, bb, 3 * bb // 2),
            "straddle": table.straddles == (2 * bb,),
            "double_straddle": table.straddles == (2 * bb, 4 * bb),
            "button_straddle": len(table.straddles) == len(setup["seats"]) - 2,
            "min_bet_up": table.min_bet == 2 * bb,
            "min_bet_down": table.min_bet == bb // 2,
        }[structure]
        assert (structure.startswith("min_bet")) == (table.min_bet != bb)


def test_round_trip_task_runs_every_check():
    out = round_trip_task(3, "double_straddle", 6)
    assert out["failures"] == []
    assert out["hands"] == 6
    expected = {f"full/{c}": 6 for c in CHECKS}
    expected |= {f"redacted/{c}": 6 for c in ("rules", "json", "phh")}
    assert out["checks"] == Counter(expected)
    assert out["passed"] == out["checks"]


def test_the_loader_check_tells_an_odd_chip_from_a_difference():
    hands = play_v2_table(table_setup(274391908, "double_straddle"), 2)  # a cents table
    text = hand_to_phh(hands[1]).dumps()
    assert hands[1].table.currency == "USD/100"
    assert loader_check(text, hands[1]) == ODD_CHIP  # two winners split a 5.21 dollar pot
    assert loader_check(hand_to_phh(hands[0]).dumps(), hands[0]) is None
    assert loader_check(text, hands[0]).startswith("loader")

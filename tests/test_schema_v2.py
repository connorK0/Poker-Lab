"""Schema v2 (P3.10): big-blind antes, straddles and other minimum bets, and the migration of
version 1 files."""

import dataclasses
import gzip
import json

import pytest

from poker_lab.engine import (
    AllInAgent,
    CallingAgent,
    RandomAgent,
    SeatedPlayer,
    TableSimulator,
    pokerkit_rules,
    replay_in_pokerkit,
)
from poker_lab.profiling.stats import C, hand_counters
from poker_lab.schema import (
    SCHEMA_VERSION,
    ActionType,
    HandHistoryStore,
    SeatInfo,
    TableConfig,
    assign_positions,
    expected_forced_posts,
    find_problems,
    hand_from_dict,
    hand_to_dict,
    migrate_jsonl,
    migrate_record,
    read_jsonl,
)
from tests.helpers import TABLE, record_reference_hand
from tests.schema_fixture import FIXTURE, v1_fixture_hands

ANTE, SB, BB, STRADDLE = (
    ActionType.POST_ANTE,
    ActionType.POST_SMALL_BLIND,
    ActionType.POST_BIG_BLIND,
    ActionType.POST_STRADDLE,
)


def _v1_lines():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_v1_files_read_as_the_hands_the_current_code_makes():
    """The fixture was written by schema v1 code (commit 0cf0c4f) from the same seeds: reading
    it migrates every record, and the hands equal what the current engine and importer make."""
    assert {d["schema_version"] for d in _v1_lines()} == {1}
    assert list(read_jsonl(FIXTURE, validate=True)) == v1_fixture_hands()


def test_migrate_jsonl_rewrites_in_the_current_version(tmp_path):
    out = tmp_path / "v2.jsonl"
    assert migrate_jsonl(FIXTURE, out) == 40
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {d["schema_version"] for d in lines} == {SCHEMA_VERSION} == {2}
    assert {d["table"]["min_bet"] == d["table"]["big_blind"] for d in lines} == {True}
    assert list(read_jsonl(out)) == list(read_jsonl(FIXTURE))
    with pytest.raises(ValueError, match="differ"):
        migrate_jsonl(out, out)


def test_each_version_is_checked_strictly():
    v1 = _v1_lines()[0]
    migrated = migrate_record(v1)
    assert migrated["table"] == {**v1["table"], "big_blind_ante": 0, "straddles": [], "min_bet": 2}
    assert v1["schema_version"] == 1  # the input is not modified
    assert migrate_record(migrated) == migrated
    smuggled = {**v1, "table": {**v1["table"], "min_bet": 4}}
    with pytest.raises(ValueError, match="unknown keys in version 1 table"):
        migrate_record(smuggled)
    short = {**migrated, "table": {k: v for k, v in migrated["table"].items() if k != "min_bet"}}
    with pytest.raises(ValueError, match="missing keys in table"):
        hand_from_dict(short)
    with pytest.raises(ValueError, match="schema_version 3"):
        migrate_record({**v1, "schema_version": 3})


def test_table_rules_are_canonical_and_checked():
    assert TableConfig("t", 6, 1, 2).min_bet == 2
    assert TableConfig("t", 6, 1, 2) == TableConfig("t", 6, 1, 2, min_bet=2)
    straddled = TableConfig("t", 6, 1, 2, straddles=[0, 4])
    assert straddled.straddles == (0, 4)
    assert straddled.blinds_or_straddles == (1, 2, 0, 4)
    for kwargs, match in [
        ({"straddles": (4, 0)}, "end with 0"),
        ({"straddles": (2,)}, "exceed"),
        ({"straddles": (8, 6)}, "exceed"),
        ({"straddles": (4, 8, 16, 32, 64)}, "straddles at a 6-seat"),
        ({"ante": 1, "big_blind_ante": 2}, "either"),
        ({"big_blind_ante": -1}, ">= 0"),
        ({"min_bet": 0}, ">= 1"),
    ]:
        with pytest.raises(ValueError, match=match):
            TableConfig("t", 6, 1, 2, **kwargs)


def _seats(stacks, button):
    pos = assign_positions(stacks, button)
    return [SeatInfo(s, f"p{s}", stack, pos[s]) for s, stack in stacks.items()]


def test_expected_forced_posts():
    hu = TableConfig("hu", 2, 1, 2, big_blind_ante=3)
    # heads-up the button (seat 0) posts the small blind and the other seat the big blind
    assert expected_forced_posts(hu, _seats({0: 100, 1: 100}, 0), 0) == [
        (1, ANTE, 3),
        (1, BB, 2),
        (0, SB, 1),
    ]
    # a short big blind posts the ante first, then what is left of the blind
    assert expected_forced_posts(hu, _seats({0: 100, 1: 4}, 0), 0)[:2] == [(1, ANTE, 3), (1, BB, 1)]
    assert expected_forced_posts(hu, _seats({0: 100, 1: 3}, 0), 0)[:1] == [(1, ANTE, 3)]
    four = TableConfig("b", 4, 1, 2, straddles=(0, 4))  # the button straddles four-handed
    assert expected_forced_posts(four, _seats(dict.fromkeys(range(4), 50), 3), 3) == [
        (0, SB, 1),
        (1, BB, 2),
        (3, STRADDLE, 4),
    ]


def test_forced_posts_must_follow_the_table_rules():
    hand, _ = record_reference_hand(HandHistoryStore())
    assert find_problems(hand) == []
    for table in (
        dataclasses.replace(TABLE, big_blind_ante=2),
        dataclasses.replace(TABLE, ante=1),
        dataclasses.replace(TABLE, straddles=(4,)),
        dataclasses.replace(TABLE, small_blind=2),
    ):
        problems = find_problems(dataclasses.replace(hand, table=table))
        assert any("forced posts" in p for p in problems), table


V2_TABLES = [
    (TableConfig("bba", 8, 1, 2, big_blind_ante=2), 8),
    (TableConfig("bba-big", 6, 50, 100, big_blind_ante=150), 6),  # an ante bigger than the BB
    (TableConfig("hu-bba", 2, 1, 2, big_blind_ante=2), 2),
    (TableConfig("straddle", 6, 1, 2, straddles=(4,)), 6),
    (TableConfig("double", 9, 1, 2, straddles=(4, 8)), 9),
    (TableConfig("button", 4, 1, 2, straddles=(0, 4)), 4),
    (TableConfig("minbet-up", 6, 1, 2, min_bet=6), 6),
    (TableConfig("minbet-down", 3, 2, 4, min_bet=1), 3),
]


@pytest.mark.parametrize(("table", "n"), V2_TABLES, ids=[t.table_id for t, _ in V2_TABLES])
def test_v2_hands_replay_through_unmodified_pokerkit(table, n):
    """Hands at v2 tables are legal in vanilla PokerKit with the same results; their forced
    posts follow the rules (``find_problems``) even with short stacks."""
    agents = [RandomAgent() if i % 3 else AllInAgent() for i in range(n)]
    stacks = {i: (5, 200, 40, 9, 300, 120, 3, 200, 60)[i] for i in range(n)}
    players = [SeatedPlayer(i, f"p{i}", a) for i, a in enumerate(agents)]
    sim = TableSimulator(table, players, seed=n, starting_stack=stacks)
    posts = set()
    for hand in sim.play(120):
        assert find_problems(hand) == []
        result = replay_in_pokerkit(hand)
        assert result.net == hand.net_results(), hand.hand_id
        assert result.showdown == {e.seat: e.shown for e in hand.showdown}
        posts |= {a.action_type for a in hand.actions if a.action_type.is_forced}
    assert (ANTE in posts) == bool(table.big_blind_ante)
    assert (STRADDLE in posts) == bool(table.straddles)


def test_straddles_need_enough_players():
    table = TableConfig("t", 6, 1, 2, straddles=(4, 8))
    players = [SeatedPlayer(i, f"p{i}", CallingAgent()) for i in range(3)]
    with pytest.raises(ValueError, match="straddles need more than 3 players"):
        TableSimulator(table, players, seed=0)
    with pytest.raises(ValueError, match="straddles"):
        pokerkit_rules(table, 2)


def test_a_straddle_is_forced_not_voluntary():
    """Everyone limps for the straddle and the straddler checks its option: the limpers
    entered the pot voluntarily, the straddler did not."""
    table = TableConfig("t", 4, 1, 2, straddles=(4,))
    players = [SeatedPlayer(i, f"p{i}", CallingAgent()) for i in range(4)]
    hand = TableSimulator(table, players, seed=1, button_seat=3).play_hand()
    straddler = next(a.seat for a in hand.actions if a.action_type is STRADDLE)
    first = next(a for a in hand.actions if a.action_type.is_decision)
    assert first.seat == (straddler + 1) % 4  # the player after the straddle acts first
    counters = dict(hand_counters(hand))
    for seat in range(4):
        vpip = counters[f"p{seat}"][C["vpip"]]
        assert vpip == (0 if seat == straddler else 1), seat


def test_v2_records_serialize_their_rules():
    table = TableConfig("t", 6, 1, 2, straddles=(4,), min_bet=3)
    players = [SeatedPlayer(i, f"p{i}", RandomAgent()) for i in range(6)]
    hand = next(TableSimulator(table, players, seed=2).play(1))
    record = hand_to_dict(hand)
    assert record["schema_version"] == 2
    assert {k: record["table"][k] for k in ("big_blind_ante", "straddles", "min_bet")} == {
        "big_blind_ante": 0,
        "straddles": [4],
        "min_bet": 3,
    }
    assert hand_from_dict(record) == hand

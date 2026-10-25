"""P3.6: batch import of completed PHH hand histories (strict), export, and round trips."""

import dataclasses
import json
import random

import pytest
from pokerkit import HandHistory as PHH

from poker_lab.engine import AllInAgent, CallingAgent, RandomAgent, SeatedPlayer, TableSimulator
from poker_lab.importers import (
    PHHImportError,
    RejectReason,
    as_imported,
    hand_from_phh,
    hand_to_phh,
    import_phh,
    write_phh,
)
from poker_lab.importers.__main__ import main
from poker_lab.profiling.push_fold import push_fold_observations
from poker_lab.schema import (
    CardSource,
    HandHistoryStore,
    HandStatus,
    HoleCards,
    TableConfig,
    find_problems,
    read_jsonl,
    redact_hole_cards,
)
from poker_lab.strategy import PushFoldStrategy, StrategyAgent
from tests.helpers import obs, record_reference_hand
from tests.test_push_fold_profile import _bot_agent, _play

# A hand as someone else might publish it: dollars and cents, physical seats 2/5/7 of 9, a PHH
# comment inside an action, TOML comments, a user-defined field. Ann (small blind) 3-bets
# Cy (button), bets the flop, checks the turn and bets the river; Cy folds.
FOREIGN = """
variant = "NT"
antes = [0, 0, 0]
blinds_or_straddles = [0.25, 0.5, 0]
min_bet = 0.5
starting_stacks = [50, 62.75, 40]
actions = [
  "d dh p1 AsKs",  # Ann
  "d dh p2 ????",
  "d dh p3 QhQd",
  "p3 cbr 1.5",
  "p1 cbr 5 # three-bet",
  "p2 f",
  "p3 cc",
  "d db Ah7c2d",
  "p1 cbr 4.25",
  "p3 cc",
  "d db 9s",
  "p1 cc",
  "p3 cc",
  "d db 3c",
  "p1 cbr 10",
  "p3 f",
]
event = "Thursday home game"
city = "Toronto"
day = 3
month = 9
year = 2026
time = 21:15:00
seats = [2, 5, 7]
seat_count = 9
table = 4
hand = 17
players = ["Ann", "Ben", "Cy"]
currency = "USD"
finishing_stacks = [59.75, 62.25, 30.75]
_dealer_note = "first orbit"
"""

# Heads-up check-down to a river bet and call; the big blind's cards are unknown until shown.
SHOWDOWN = """
variant = "NT"
antes = [0, 0]
blinds_or_straddles = [1, 2]
min_bet = 2
starting_stacks = [100, 100]
actions = ["d dh p1 ????", "d dh p2 KcKd", "p2 cc", "p1 cc", "d db 2c3d4h", "p1 cc", "p2 cc",
           "d db 7s", "p1 cc", "p2 cc", "d db Ts", "p1 cbr 10", "p2 cc", "p1 sm 9h9s",
           "p2 sm KcKd"]
hand = "hu-1"
"""


def _phh(text, **subs):
    for old, new in subs.items():
        assert old in text, old
        text = text.replace(old, new)
    return PHH.loads(text)


def _sim(table, seats, agents, stacks, seed, n):
    players = [SeatedPlayer(s, f"pl{s}", a) for s, a in zip(seats, agents, strict=True)]
    return list(TableSimulator(table, players, seed=seed, starting_stack=stacks).play(n))


@pytest.fixture(scope="module")
def simulated():
    """Hands covering heads-up, antes, gaps between seats, short stacks and all-ins."""
    rng = random.Random(7)
    hands = []
    configs = [
        (TableConfig("hu", 2, 1, 2), [0, 1], 2),
        (TableConfig("six", 6, 1, 2, ante=1), list(range(6)), 6),
        (TableConfig("nine", 10, 2, 4, ante=2), [0, 1, 3, 4, 5, 7, 8, 9, 2], 9),
        (TableConfig("gaps", 9, 1, 2), [1, 4, 6, 8], 4),
    ]
    for k, (table, seats, _) in enumerate(configs):
        agents = [
            rng.choice([RandomAgent(), RandomAgent(0.3, 2.0), CallingAgent(), AllInAgent()])
            for _ in seats
        ]
        stacks = {s: rng.randint(3, 300) for s in seats}
        hands += _sim(table, sorted(seats), agents, stacks, seed=k, n=25)
    return hands


def test_round_trip_reproduces_simulated_hands(simulated):
    for hand in simulated:
        for h in (hand, redact_hole_cards(hand)):
            back = hand_from_phh(PHH.loads(hand_to_phh(h).dumps()))
            assert back == as_imported(h)
            assert back.source == "imported"
            assert all(o.source is CardSource.IMPORTED for o in back.board)
    assert {len(h.seats) for h in simulated} == {2, 4, 6, 9}
    assert any(a.is_all_in for h in simulated for a in h.actions)


def test_round_trip_through_files(simulated, tmp_path):
    hands = simulated[:40]
    assert write_phh(hands, tmp_path / "batch.phhs") == 40
    write_phh(hands[:1], tmp_path / "one.phh")
    report = import_phh([tmp_path / "batch.phhs"], ids="field")
    assert report.rejected == ()
    assert [h.metadata["import.file"] for h in report.hands] == ["batch.phhs"] * 40
    for got, want in zip(report.hands, hands, strict=True):
        meta = {k: v for k, v in got.metadata.items() if k != "import.file"}
        assert dataclasses.replace(got, metadata=meta) == as_imported(want)
    both = import_phh([tmp_path])  # path ids: no clash between the two files
    assert len(both.hands) == 41
    assert both.hands[0].hand_id == f"batch:{hands[0].hand_id}"
    with pytest.raises(ValueError, match="exactly one hand"):
        write_phh(hands[:2], tmp_path / "two.phh")


def test_capture_style_hand_round_trips():
    """Camera and manual cards, a mucked unknown hand, a start time and metadata."""
    hand, _ = record_reference_hand(HandHistoryStore(), "h2")
    hole = list(hand.hole_cards)
    hole[3] = HoleCards(3, (obs("Ac", "camera", 0.91), obs("Qd", "manual")), shown=True)
    hole[1] = HoleCards(1, None)
    hand = dataclasses.replace(
        hand,
        hole_cards=tuple(hole),
        source="capture",
        started_at="2026-10-20T19:30:00",
        metadata={"session": "home-01", "dealer": "seat 4"},
    )
    phh = hand_to_phh(hand)
    assert (phh.year, phh.month, phh.day, str(phh.time)) == (2026, 10, 20, "19:30:00")
    assert hand_from_phh(PHH.loads(phh.dumps())) == as_imported(hand)
    odd = dataclasses.replace(hand, started_at="2026-10-20 19:30+02:00")
    assert hand_from_phh(PHH.loads(hand_to_phh(odd).dumps())).started_at == odd.started_at


def test_foreign_file_mapping_and_fixed_point():
    hand = hand_from_phh(PHH.loads(FOREIGN))
    assert hand.hand_id == "17"
    assert hand.table == TableConfig("4", 9, 25, 50, 0, currency="USD/100")
    assert [(s.seat, s.player_id, s.starting_stack, s.position) for s in hand.seats] == [
        (1, "Ann", 5000, "SB"),
        (4, "Ben", 6275, "BB"),
        (6, "Cy", 4000, "BTN"),
    ]
    assert hand.button_seat == 6
    assert hand.started_at == "2026-09-03T21:15:00"
    assert hand.metadata == {
        "phh.event": "Thursday home game",
        "phh.city": "Toronto",
        "phh._dealer_note": json.dumps("first orbit"),
    }
    assert [(a.seat, a.action_type.value, a.amount) for a in hand.actions[-2:]] == [
        (1, "bet", 1000),
        (6, "fold", 0),
    ]
    assert [(p.seat, p.amount) for p in hand.payouts] == [(1, 1900)]
    assert [(p.seat, p.amount) for p in hand.uncalled_returns] == [(1, 1000)]
    assert hand.hole_cards_of(4).cards is None
    assert not hand.hole_cards_of(1).shown  # nobody showed: the hand ended with a fold
    assert find_problems(hand) == []
    again = hand_from_phh(PHH.loads(hand_to_phh(hand).dumps()))
    assert again == hand
    assert hand_to_phh(again).dumps() == hand_to_phh(hand).dumps()


def test_shown_cards_replace_unknown_dealt_cards():
    hand = hand_from_phh(PHH.loads(SHOWDOWN))
    shown = hand.hole_cards_of(0)
    assert shown.shown
    assert [o.card for o in shown.cards] == ["9h", "9s"]
    assert [(p.seat, p.amount) for p in hand.payouts] == [(1, 24)]
    assert [(e.seat, e.shown) for e in hand.showdown] == [(0, True), (1, True)]


def test_shown_cards_must_be_the_dealt_cards():
    """PokerKit accepts a show of other cards than were dealt (and pays the pot to them)."""
    phh = _phh(SHOWDOWN, **{'"d dh p1 ????"': '"d dh p1 9h9s"', '"p1 sm 9h9s"': '"p1 sm AhAs"'})
    assert _reason(phh) == RejectReason.INCONSISTENT
    partial = _phh(SHOWDOWN, **{'"d dh p1 ????"': '"d dh p1 9h??"'})
    assert [o.card for o in hand_from_phh(partial).hole_cards_of(0).cards] == ["9h", "9s"]


def test_unknown_hole_cards_may_be_left_implicit():
    text = FOREIGN.replace('  "d dh p2 ????",\n', "")
    assert hand_from_phh(PHH.loads(text)) == hand_from_phh(PHH.loads(FOREIGN))


def _reason(phh, **kw):
    with pytest.raises(PHHImportError) as info:
        hand_from_phh(phh, **kw)
    return info.value.reason


@pytest.mark.parametrize(
    ("subs", "reason"),
    [
        ({'variant = "NT"': 'variant = "PO"'}, RejectReason.UNSUPPORTED),
        ({"[0.25, 0.5, 0]": "[0.25, 0.5, 1]"}, RejectReason.UNSUPPORTED),  # straddle
        ({"antes = [0, 0, 0]": "antes = [0, 0.5, 0]"}, RejectReason.UNSUPPORTED),  # BB ante
        ({"min_bet = 0.5": "min_bet = 1"}, RejectReason.UNSUPPORTED),
        ({"[50, 62.75, 40]": "[50, inf, 40]"}, RejectReason.UNSUPPORTED),
        ({"[50, 62.75, 40]": "[50, 62.750001, 40]"}, RejectReason.UNSUPPORTED),
        ({'"p3 cbr 1.5",': '"p1 cbr 1.5",'}, RejectReason.ILLEGAL_ACTION),  # out of turn
        ({'"p1 cbr 5 # three-bet"': '"p1 cbr 1.6"'}, RejectReason.ILLEGAL_ACTION),  # min raise
        ({'  "p3 f",\n': ""}, RejectReason.INCOMPLETE),
        ({'"d db Ah7c2d"': '"d db As7c2d"'}, RejectReason.INCONSISTENT),  # As dealt twice
        ({"seats = [2, 5, 7]": "seats = [7, 5, 2]"}, RejectReason.INCONSISTENT),
        ({'players = ["Ann", "Ben", "Cy"]': 'players = ["Ann", "Ann", "Cy"]'}, "inconsistent"),
        ({"[59.75, 62.25, 30.75]": "[30.75, 62.25, 59.75]"}, RejectReason.RESULT_MISMATCH),
        ({"[59.75, 62.25, 30.75]": "[59.75, 62.25, 31.75]"}, RejectReason.RESULT_MISMATCH),
        ({"players = [": "players = 5\n_x = ["}, RejectReason.PARSE),
        ({"min_bet = 0.5\n": ""}, RejectReason.PARSE),
    ],
)
def test_invalid_hands_are_rejected_with_a_reason(subs, reason):
    assert _reason(_phh(FOREIGN, **subs)) == reason


def test_missing_actions_are_not_repaired():
    """PokerKit's own loader fills in a missing fold; the importer refuses the hand."""
    phh = _phh(FOREIGN, **{'  "p2 f",\n': ""})
    list(phh)  # PokerKit "repairs" it by folding Ben automatically
    assert _reason(phh) == RejectReason.ILLEGAL_ACTION


def test_missing_showdown_is_incomplete():
    phh = _phh(SHOWDOWN, **{', "p1 sm 9h9s",\n           "p2 sm KcKd"': ""})
    assert _reason(phh) == RejectReason.INCOMPLETE


def test_rake_and_unknown_results():
    raked = _phh(FOREIGN, **{"[59.75, 62.25, 30.75]": "[59.25, 62.25, 30.75]"})
    assert hand_from_phh(raked).rake == 50
    assert _reason(raked, allow_rake=False) == RejectReason.RESULT_MISMATCH
    assert _reason(_phh(SHOWDOWN, **{'"p1 sm 9h9s"': '"p1 sm -"'})) == "illegal_action"
    unknown = _phh(SHOWDOWN, **{'"p1 sm 9h9s"': '"p1 sm ????"'})  # PokerKit accepts this
    assert _reason(unknown) == RejectReason.UNKNOWN_RESULT
    recorded = SHOWDOWN.replace('"p1 sm 9h9s"', '"p1 sm ????"') + "finishing_stacks = [88, 112]\n"
    hand = hand_from_phh(PHH.loads(recorded))
    assert [(p.seat, p.amount) for p in hand.payouts] == [(1, 24)]
    assert hand.hole_cards_of(0).cards is None


def test_batch_reports_every_rejection(tmp_path):
    (tmp_path / "a.phh").write_text(FOREIGN, encoding="utf-8")
    (tmp_path / "bad.phh").write_text("variant = [", encoding="utf-8")
    sub = tmp_path / "more"
    sub.mkdir()
    (sub / "a.phh").write_text(FOREIGN, encoding="utf-8")  # same name, other folder
    razz = FOREIGN.replace('variant = "NT"', 'variant = "FR"')
    (sub / "pair.phhs").write_text(f"[1]\n{FOREIGN}\n[2]\n{razz}", encoding="utf-8")
    report = import_phh([tmp_path])
    assert sorted(h.hand_id for h in report.hands) == ["a:17", "more/a:17", "more/pair:17"]
    assert [(r.file, r.index, r.reason) for r in report.rejected] == [
        ("bad.phh", -1, RejectReason.PARSE),
        ("more/pair.phhs", 1, RejectReason.UNSUPPORTED),
    ]
    by_field = import_phh([tmp_path], ids="field")
    assert [r.reason for r in by_field.rejected].count(RejectReason.DUPLICATE) == 2
    assert all(h.status is HandStatus.COMPLETE for h in report.hands)


def test_cli_writes_json_lines(tmp_path, capsys):
    (tmp_path / "a.phh").write_text(FOREIGN, encoding="utf-8")
    (tmp_path / "b.phh").write_text(FOREIGN.replace("min_bet = 0.5", "min_bet = 1"), "utf-8")
    out, rej = tmp_path / "hands.jsonl.gz", tmp_path / "rejected.jsonl"
    assert main([str(tmp_path), "--out", str(out), "--rejections", str(rej), "--no-rake"]) == 0
    assert "1 hands imported from 2 files; 1 rejected (unsupported 1)" in capsys.readouterr().out
    (hand,) = read_jsonl(out, validate=True)
    assert hand.hand_id == "a:17"
    assert json.loads(rej.read_text(encoding="utf-8"))["reason"] == "unsupported"


def test_imported_hands_feed_the_push_fold_profiler():
    """Exported and re-imported push/fold hands give the same opponent observations; the
    importer only ever produces complete hands, which the profiler redacts to the hero's view."""
    hands = _play([StrategyAgent(PushFoldStrategy()), _bot_agent()], 120, seed=11)
    imported = [hand_from_phh(PHH.loads(hand_to_phh(h).dumps())) for h in hands]
    a = push_fold_observations(hands, "hero", "villain", stack=10)
    b = push_fold_observations(imported, "hero", "villain", stack=10)
    assert a.hand_ids == b.hand_ids
    assert a.skipped == b.skipped
    for field in dataclasses.fields(a.log):
        x, y = getattr(a.log, field.name), getattr(b.log, field.name)
        assert (x == y).all() if hasattr(x, "all") else x == y

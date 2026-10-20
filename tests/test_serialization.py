"""JSON Lines round-trips and strict decoding."""

import dataclasses
import json

import pytest

from poker_lab.schema import (
    CardSource,
    HandHistoryStore,
    HandStatus,
    HoleCards,
    dumps_hand,
    hand_from_dict,
    hand_to_dict,
    loads_hand,
    read_jsonl,
    write_jsonl,
)
from tests.helpers import obs, record_reference_hand


@pytest.fixture
def hands():
    store = HandHistoryStore()
    h1, _ = record_reference_hand(store, "h1")
    # A Phase-4 style hand: camera detections with confidences, one hand unknown (mucked).
    h2, _ = record_reference_hand(store, "h2")
    hole = list(h2.hole_cards)
    hole[3] = HoleCards(3, (obs("Ac", "camera", 0.91), obs("Qd", "manual")), shown=True)
    hole[1] = HoleCards(1, None)
    board = (obs("Ah", "camera", 0.73), *h2.board[1:])
    h2 = dataclasses.replace(
        h2,
        hole_cards=tuple(hole),
        board=board,
        source="capture",
        started_at="2026-10-20T19:30:00",
        metadata={"session": "home-01"},
    )
    return [h1, h2]


def test_dict_round_trip(hands):
    for hand in hands:
        assert hand_from_dict(hand_to_dict(hand), validate=True) == hand
        assert loads_hand(dumps_hand(hand)) == hand


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
def test_jsonl_round_trip(tmp_path, hands, suffix):
    path = tmp_path / f"hands{suffix}"
    assert write_jsonl(hands, path) == 2
    loaded = list(read_jsonl(path, validate=True))
    assert loaded == hands
    camera = loaded[1].hole_cards_of(3).cards[0]
    assert camera.source is CardSource.CAMERA
    assert camera.confidence == pytest.approx(0.91)
    assert loaded[1].hole_cards_of(1).cards is None
    write_jsonl(hands[:1], path, append=True)
    assert len(list(read_jsonl(path))) == 3


def test_in_progress_round_trip_has_no_hole_cards(tmp_path):
    store = HandHistoryStore()
    record_reference_hand(store, "live", complete=False)
    live = store.get("live")
    assert live.status is HandStatus.IN_PROGRESS
    assert live.hole_cards == ()
    path = tmp_path / "live.jsonl"
    write_jsonl([live], path)
    assert list(read_jsonl(path)) == [live]


def test_decoder_rejects_in_progress_record_with_hole_cards(hands):
    d = hand_to_dict(hands[0])
    d["status"] = "in_progress"
    d["showdown"], d["payouts"], d["uncalled_returns"] = [], [], []
    with pytest.raises(ValueError, match="sealed"):
        hand_from_dict(d)


def test_decoder_is_strict(hands, tmp_path):
    d = hand_to_dict(hands[0])
    d["opponent_cards"] = ["As", "Ks"]
    with pytest.raises(ValueError, match="unknown keys"):
        hand_from_dict(d)
    d = hand_to_dict(hands[0])
    d["actions"][0]["note"] = "x"
    with pytest.raises(ValueError, match="unknown keys"):
        hand_from_dict(d)
    d = hand_to_dict(hands[0])
    del d["board"]
    with pytest.raises(ValueError, match="missing"):
        hand_from_dict(d)
    d = hand_to_dict(hands[0])
    d["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        hand_from_dict(d)
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(hand_to_dict(hands[0])) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        list(read_jsonl(path))

"""CaptureIngestor: camera detections + manual action logs -> hand histories with conflicts.

Scenario (3-handed home game, blinds 1/2, stacks 100, button seat 2 -> SB seat 0, BB seat 1):
preflop BTN raises to 6, SB folds, BB calls; flop 2h7d9s BB checks, BTN bets 6, BB calls;
turn Jc check/check; river 3h BB bets 10, BTN calls. BTN (AsAd) beats BB (KcQc), pot 45.
"""

import dataclasses

import pytest

from poker_lab.capture import (
    CaptureIngestor,
    CardDetection,
    ConflictKind,
    ManualAction,
    ManualCardEntry,
    ManualHandLog,
)
from poker_lab.schema import ActionType, CardSource, HandStatus, Street, TableConfig

P, F, T, R = Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER
A = ActionType
TABLE = TableConfig("home", 6, 1, 2, currency="cents")
BOARD = ("2h", "7d", "9s", "Jc", "3h")
ACTIONS = (
    ManualAction(0, P, A.POST_SMALL_BLIND, 1),
    ManualAction(1, P, A.POST_BIG_BLIND, 2),
    ManualAction(2, P, A.RAISE, 6),
    ManualAction(0, P, A.FOLD),
    ManualAction(1, P, A.CALL),
    ManualAction(1, F, A.CHECK),
    ManualAction(2, F, A.BET, 6),
    ManualAction(1, F, A.CALL),
    ManualAction(1, T, A.CHECK),
    ManualAction(2, T, A.CHECK),
    ManualAction(1, R, A.BET, 10),
    ManualAction(2, R, A.CALL),
)


def log(**overrides) -> ManualHandLog:
    base = ManualHandLog(
        hand_id="home-001",
        table=TABLE,
        button_seat=2,
        players={0: ("ann", 100), 1: ("ben", 100), 2: ("cat", 100)},
        actions=ACTIONS,
        complete=True,
        winners=(2,),
    )
    return dataclasses.replace(base, **overrides)


def detections(**changes) -> list[CardDetection]:
    reads = {f"board:{i}": (c, 0.98) for i, c in enumerate(BOARD)}
    reads |= {"seat:2:0": ("As", 0.96), "seat:2:1": ("Ad", 0.95)}
    reads |= {"seat:1:0": ("Kc", 0.97), "seat:1:1": ("Qc", 0.94)}
    reads |= changes
    return [
        CardDetection(loc, card, conf, frame=i)
        for i, (loc, (card, conf)) in enumerate(reads.items())
    ]


def test_clean_hand_ingests_without_conflicts():
    result = CaptureIngestor().ingest(log(), detections())
    assert result.conflicts == ()
    hand = result.hand
    assert hand.status is HandStatus.COMPLETE
    assert hand.board_cards() == BOARD
    assert hand.board[0].source is CardSource.CAMERA
    assert hand.board[0].confidence == pytest.approx(0.98)
    assert hand.hole_cards_of(2).shown
    assert hand.net_results() == {0: -1, 1: -22, 2: 23}
    assert hand.source == "capture"


def test_manual_entry_overrides_camera_but_disagreement_is_flagged():
    result = CaptureIngestor().ingest(log(), detections(), [ManualCardEntry("board:3", "Jh")])
    assert ConflictKind.CARD_DISAGREEMENT in result.kinds()
    assert result.hand.board[3].card == "Jh"
    assert result.hand.board[3].source is CardSource.MANUAL


def test_low_confidence_and_competing_reads_are_flagged():
    reads = detections(**{"board:4": ("3h", 0.62)})
    reads.append(CardDetection("board:1", "7h", 0.7, frame=99))  # competing read
    result = CaptureIngestor(min_confidence=0.9).ingest(log(), reads)
    assert {ConflictKind.LOW_CONFIDENCE, ConflictKind.CARD_DISAGREEMENT} <= result.kinds()
    assert result.hand is not None  # still usable, but needs review
    assert result.needs_review


def test_duplicate_cards_and_bad_locations_are_flagged():
    reads = detections(**{"seat:1:1": ("As", 0.9)})  # As also read in seat 2
    reads.append(CardDetection("seat:7:0", "5c", 0.99))  # nobody sits in seat 7
    reads.append(CardDetection("table:center", "5d", 0.99))
    result = CaptureIngestor().ingest(log(), reads)
    assert {ConflictKind.DUPLICATE_CARD, ConflictKind.BAD_LOCATION} <= result.kinds()


def test_missing_board_card_is_flagged():
    reads = [d for d in detections() if d.location != "board:4"]
    result = CaptureIngestor().ingest(log(), reads)
    assert ConflictKind.BOARD_INCOMPLETE in result.kinds()


def test_partial_hole_cards_are_flagged():
    reads = [d for d in detections() if d.location != "seat:1:1"]
    result = CaptureIngestor().ingest(log(), reads)
    assert ConflictKind.PARTIAL_HOLE_CARDS in result.kinds()
    assert result.hand.hole_cards_of(1) is None


def test_inconsistent_amounts_are_flagged():
    bad = list(ACTIONS)
    bad[2] = ManualAction(2, P, A.RAISE, 2)  # "raise" to the current bet
    result = CaptureIngestor().ingest(log(actions=tuple(bad)), detections())
    assert ConflictKind.ACTION_INCONSISTENT in result.kinds()


def test_out_of_turn_actions_are_caught_by_pokerkit_replay():
    swapped = list(ACTIONS)
    swapped[2], swapped[3] = swapped[3], swapped[2]  # SB folds before the BTN acts
    result = CaptureIngestor().ingest(log(actions=tuple(swapped)), detections())
    assert ConflictKind.ILLEGAL_SEQUENCE in result.kinds()


def test_payout_contradicting_the_known_showdown_is_flagged():
    result = CaptureIngestor().ingest(log(winners=(1,)), detections())  # BB "won" with K-high
    assert ConflictKind.PAYOUT_MISMATCH in result.kinds()


def test_in_progress_log_keeps_hole_cards_out_of_the_record():
    partial = log(actions=ACTIONS[:8], complete=False)
    reads = [
        d for d in detections() if not d.location.startswith("board:") or d.location < "board:3"
    ]
    result = CaptureIngestor().ingest(partial, reads)
    assert result.hand.status is HandStatus.IN_PROGRESS
    assert result.hand.hole_cards == ()  # camera saw them, but they stay out until complete
    assert result.hand.board_cards() == BOARD[:3]

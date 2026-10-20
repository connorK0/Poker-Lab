"""Phase 4 ingestion: camera card detections + manually entered action logs -> ``HandHistory``.

Inputs
    ``CardDetection``      a vision-model read of one card location with a confidence;
    ``ManualCardEntry``    a card typed in by a person (treated as ground truth, confidence 1);
    ``ManualHandLog``      seats, stacks, button, and the action log as entered at the table.

Locations are ``"board:<i>"`` (i = 0..4) or ``"seat:<n>:<k>"`` (hole card k = 0/1 of seat n).

Every inconsistency becomes a typed ``Conflict`` for manual review, never a silent fix:
camera vs. manual disagreements, low-confidence reads, the same card in two places, partial
hole cards, missing board cards, action amounts that don't add up, sequences PokerKit rejects,
and payouts that contradict a fully known showdown.

In-progress logs produce the public record only. Their hole cards are not placed in the
``HandHistory`` (the schema forbids it); live capture should record hole cards through
``HandHistoryStore.deal_hole_cards`` so they stay sealed until the hand completes.
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from poker_lab.engine.replay import ReplayError, replay_in_pokerkit
from poker_lab.schema import (
    Action,
    ActionType,
    CardObservation,
    CardSource,
    ChipAward,
    HandHistory,
    HandStatus,
    HoleCards,
    SeatInfo,
    ShowdownEntry,
    Street,
    TableConfig,
    assign_positions,
    compute_uncalled_returns,
    find_problems,
    is_valid_card,
)

_LOCATION = re.compile(r"^(board):([0-4])$|^(seat):(\d+):([01])$")


@dataclass(frozen=True)
class CardDetection:
    location: str
    card: str
    confidence: float
    frame: int | None = None


@dataclass(frozen=True)
class ManualCardEntry:
    location: str
    card: str


@dataclass(frozen=True)
class ManualAction:
    """One entered action. ``to_amount`` is the seat's total street bet after a bet, raise or
    blind post (the ante amount for antes); it is ignored for fold/check/call."""

    seat: int
    street: Street
    kind: ActionType
    to_amount: int = 0


@dataclass(frozen=True)
class ManualHandLog:
    hand_id: str
    table: TableConfig
    button_seat: int
    players: Mapping[int, tuple[str, int]]  # seat -> (player id, starting stack)
    actions: tuple[ManualAction, ...]
    complete: bool
    payouts: Mapping[int, int] | None = None  # seat -> pot amount won (excl. uncalled bets)
    winners: tuple[int, ...] = ()  # used when payouts is None: split the pot equally
    showdown: Mapping[int, bool] | None = None  # seat -> showed (True) / mucked (False)
    started_at: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


class ConflictKind(StrEnum):
    CARD_DISAGREEMENT = "card_disagreement"
    LOW_CONFIDENCE = "low_confidence"
    DUPLICATE_CARD = "duplicate_card"
    BAD_LOCATION = "bad_location"
    PARTIAL_HOLE_CARDS = "partial_hole_cards"
    BOARD_INCOMPLETE = "board_incomplete"
    ACTION_INCONSISTENT = "action_inconsistent"
    ILLEGAL_SEQUENCE = "illegal_sequence"
    PAYOUT_MISMATCH = "payout_mismatch"


@dataclass(frozen=True)
class Conflict:
    kind: ConflictKind
    detail: str
    location: str | None = None


@dataclass(frozen=True)
class IngestResult:
    hand: HandHistory | None
    conflicts: tuple[Conflict, ...]

    @property
    def needs_review(self) -> bool:
        return bool(self.conflicts)

    def kinds(self) -> set[ConflictKind]:
        return {c.kind for c in self.conflicts}


class CaptureIngestor:
    def __init__(self, *, min_confidence: float = 0.9, disagreement_confidence: float = 0.5):
        """Camera reads below ``min_confidence`` are flagged. A competing read (a different
        card at the same location) at or above ``disagreement_confidence`` is a conflict."""
        self.min_confidence = min_confidence
        self.disagreement_confidence = disagreement_confidence

    # ------------------------------------------------------------------ cards
    def fuse_cards(
        self,
        log: ManualHandLog,
        detections: Sequence[CardDetection],
        manual: Sequence[ManualCardEntry],
    ) -> tuple[dict[str, CardObservation], list[Conflict]]:
        conflicts: list[Conflict] = []
        cam: dict[str, dict[str, float]] = defaultdict(dict)
        man: dict[str, str] = {}
        valid_locations = set()
        for loc in {d.location for d in detections} | {m.location for m in manual}:
            m = _LOCATION.match(loc)
            if not m or (m.group(3) and int(m.group(4)) not in log.players):
                conflicts.append(Conflict(ConflictKind.BAD_LOCATION, "unknown location", loc))
            else:
                valid_locations.add(loc)
        for d in detections:
            if d.location in valid_locations and is_valid_card(d.card):
                prev = cam[d.location].get(d.card, 0.0)
                cam[d.location][d.card] = max(prev, float(d.confidence))
        for e in manual:
            if e.location not in valid_locations or not is_valid_card(e.card):
                continue
            if e.location in man and man[e.location] != e.card:
                conflicts.append(
                    Conflict(
                        ConflictKind.CARD_DISAGREEMENT,
                        f"manual entries {man[e.location]} and {e.card}",
                        e.location,
                    )
                )
            man[e.location] = e.card

        chosen: dict[str, CardObservation] = {}
        for loc in sorted(valid_locations):
            reads = sorted(cam.get(loc, {}).items(), key=lambda kv: -kv[1])
            if loc in man:
                card = man[loc]
                chosen[loc] = CardObservation(card, CardSource.MANUAL, 1.0)
                rivals = [c for c, p in reads if c != card and p >= self.disagreement_confidence]
                if rivals:
                    conflicts.append(
                        Conflict(
                            ConflictKind.CARD_DISAGREEMENT,
                            f"manual {card} vs camera {rivals[0]}",
                            loc,
                        )
                    )
            elif reads:
                card, conf = reads[0]
                chosen[loc] = CardObservation(card, CardSource.CAMERA, conf)
                if len(reads) > 1 and reads[1][1] >= self.disagreement_confidence:
                    conflicts.append(
                        Conflict(
                            ConflictKind.CARD_DISAGREEMENT,
                            f"camera reads {card} ({conf:.2f}) and {reads[1][0]} "
                            f"({reads[1][1]:.2f})",
                            loc,
                        )
                    )
                if conf < self.min_confidence:
                    conflicts.append(
                        Conflict(ConflictKind.LOW_CONFIDENCE, f"{card} at {conf:.2f}", loc)
                    )
        where: dict[str, list[str]] = defaultdict(list)
        for loc, obs in chosen.items():
            where[obs.card].append(loc)
        for card, locs in where.items():
            if len(locs) > 1:
                conflicts.append(
                    Conflict(ConflictKind.DUPLICATE_CARD, f"{card} at {', '.join(sorted(locs))}")
                )
        return chosen, conflicts

    # ------------------------------------------------------------------ actions
    @staticmethod
    def build_actions(log: ManualHandLog) -> tuple[tuple[Action, ...], list[Conflict]]:
        stacks = {seat: stack for seat, (_, stack) in log.players.items()}
        street, bets = Street.PREFLOP, dict.fromkeys(stacks, 0)
        actions: list[Action] = []
        for i, m in enumerate(log.actions):
            if m.seat not in stacks:
                return (), [Conflict(ConflictKind.ACTION_INCONSISTENT, f"action {i}: no seat")]
            if m.street is not street:
                street, bets = m.street, dict.fromkeys(stacks, 0)
            current, top = bets[m.seat], max(bets.values())
            kind = m.kind
            if kind is ActionType.POST_ANTE:
                amount, total = m.to_amount, 0
            elif kind in (ActionType.FOLD, ActionType.CHECK):
                amount, total = 0, current
            elif kind is ActionType.CALL:
                amount = min(top - current, stacks[m.seat])
                total = current + amount
            else:
                amount, total = m.to_amount - current, m.to_amount
            try:
                action = Action(
                    m.seat, street, kind, amount, total, amount > 0 and amount == stacks[m.seat]
                )
            except (ValueError, TypeError) as exc:
                return (), [Conflict(ConflictKind.ACTION_INCONSISTENT, f"action {i}: {exc}")]
            if kind is not ActionType.POST_ANTE:
                bets[m.seat] = total
            stacks[m.seat] -= amount
            actions.append(action)
        return tuple(actions), []

    # ------------------------------------------------------------------ hand
    def ingest(
        self,
        log: ManualHandLog,
        detections: Sequence[CardDetection] = (),
        manual_cards: Sequence[ManualCardEntry] = (),
    ) -> IngestResult:
        cards, conflicts = self.fuse_cards(log, detections, manual_cards)
        actions, action_conflicts = self.build_actions(log)
        conflicts += action_conflicts
        if action_conflicts:
            return IngestResult(None, tuple(conflicts))

        board: list[CardObservation] = []
        for i in range(5):
            obs = cards.get(f"board:{i}")
            if obs is None:
                if any(f"board:{j}" in cards for j in range(i + 1, 5)):
                    conflicts.append(Conflict(ConflictKind.BOARD_INCOMPLETE, f"gap at board:{i}"))
                break
            board.append(obs)
        if len(board) not in (0, 3, 4, 5):
            conflicts.append(Conflict(ConflictKind.BOARD_INCOMPLETE, f"{len(board)} board cards"))
            board = board[: max(s for s in (0, 3, 4) if s <= len(board))]
        last = max((a.street for a in actions), key=lambda s: s.index, default=Street.PREFLOP)
        if len(board) < last.board_size:
            conflicts.append(
                Conflict(
                    ConflictKind.BOARD_INCOMPLETE,
                    f"actions on {last.value} but only {len(board)} board cards",
                )
            )

        folded = {a.seat for a in actions if a.action_type is ActionType.FOLD}
        remaining = [s for s in sorted(log.players) if s not in folded]
        hole: list[HoleCards] = []
        known: set[int] = set()
        for seat in sorted(log.players):
            pair = (cards.get(f"seat:{seat}:0"), cards.get(f"seat:{seat}:1"))
            if all(pair):
                hole.append(HoleCards(seat, pair))  # type: ignore[arg-type]
                known.add(seat)
            elif any(pair):
                conflicts.append(
                    Conflict(ConflictKind.PARTIAL_HOLE_CARDS, "one hole card known", f"seat:{seat}")
                )

        positions = assign_positions(log.players, log.button_seat)
        seats = tuple(
            SeatInfo(seat, pid, stack, positions[seat])
            for seat, (pid, stack) in sorted(log.players.items())
        )
        common = {
            "hand_id": log.hand_id,
            "table": log.table,
            "button_seat": log.button_seat,
            "seats": seats,
            "actions": actions,
            "board": tuple(board),
            "source": "capture",
            "started_at": log.started_at,
            "metadata": dict(log.metadata),
        }
        if not log.complete:
            hand = HandHistory(**common, status=HandStatus.IN_PROGRESS)  # hole cards stay out
            for problem in find_problems(hand):
                conflicts.append(Conflict(ConflictKind.ACTION_INCONSISTENT, problem))
            return IngestResult(hand, tuple(conflicts))

        if log.showdown is not None:
            showdown_map = dict(log.showdown)
        else:
            showdown_map = {s: s in known for s in remaining} if len(remaining) > 1 else {}
        shown = {s for s, showed in showdown_map.items() if showed}
        hole = [HoleCards(h.seat, h.cards, shown=h.seat in shown) for h in hole]
        uncalled = compute_uncalled_returns(actions)
        pot = sum(a.amount for a in actions) - sum(uncalled.values())
        if log.payouts is not None:
            payouts = {s: a for s, a in log.payouts.items() if a > 0}
        else:
            winners = list(log.winners) or remaining[:1]
            share, extra = divmod(pot, len(winners))
            payouts = {w: share + (extra if i == 0 else 0) for i, w in enumerate(winners)}
        try:
            hand = HandHistory(
                **common,
                hole_cards=tuple(hole),
                showdown=tuple(ShowdownEntry(s, b) for s, b in sorted(showdown_map.items())),
                payouts=tuple(ChipAward(s, a) for s, a in sorted(payouts.items())),
                uncalled_returns=tuple(ChipAward(s, a) for s, a in sorted(uncalled.items())),
            )
        except (ValueError, TypeError) as exc:
            conflicts.append(Conflict(ConflictKind.ACTION_INCONSISTENT, str(exc)))
            return IngestResult(None, tuple(conflicts))
        for problem in find_problems(hand):
            conflicts.append(Conflict(ConflictKind.ACTION_INCONSISTENT, problem))
        conflicts += self._replay_check(hand, showdown_map, known)
        return IngestResult(hand, tuple(conflicts))

    @staticmethod
    def _replay_check(hand: HandHistory, showdown: Mapping[int, bool], known: set[int]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # unknown cards make PokerKit warn at showdown
            try:
                result = replay_in_pokerkit(hand, allow_unknown_cards=True)
            except ReplayError as exc:
                return [Conflict(ConflictKind.ILLEGAL_SEQUENCE, str(exc))]
            except ValueError as exc:  # PokerKit refused the setup itself
                return [Conflict(ConflictKind.ILLEGAL_SEQUENCE, f"replay failed: {exc}")]
        if showdown and set(showdown) <= known and result.net != hand.net_results():
            return [
                Conflict(
                    ConflictKind.PAYOUT_MISMATCH,
                    f"recorded {hand.net_results()} vs showdown result {result.net}",
                )
            ]
        return []

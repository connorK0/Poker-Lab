"""Hand-history store with sealed hole cards for in-progress hands.

The dealer (the simulation engine in Phase 1, the capture system in Phase 4) records a hand as
it happens: ``begin_hand``, ``deal_hole_cards``, ``record_action``, ``deal_board``, then
``complete_hand``. Hole cards dealt during the hand go into a private vault:

* ``get(hand_id)`` on an in-progress hand returns a ``HandHistory`` with no hole cards
  (``HandHistory`` forbids them while in progress);
* ``reveal_hole_cards(hand_id)`` raises ``SealedHoleCardsError`` until the hand is complete;
* the only read path during the hand is ``seat_view(hand_id, key)``, which needs the
  ``SeatKey`` issued for that seat at deal time and returns a ``SeatView`` holding that seat's
  cards only;
* ``complete_hand`` is the single point where sealed cards are released into the record.

``SeatKey`` objects compare by identity. An object built with the same fields is a different
key, so a seat's key cannot be forged from public information.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from poker_lab.schema.hand_history import (
    Action,
    ActionType,
    CardObservation,
    ChipAward,
    HandHistory,
    HandStatus,
    HoleCards,
    SeatInfo,
    ShowdownEntry,
    Street,
    TableConfig,
    street_for_board_size,
)
from poker_lab.schema.validation import validate_hand_history
from poker_lab.schema.views import LegalActions, PublicHandState, SeatView


class SealedHoleCardsError(PermissionError):
    """Raised on any attempt to read sealed hole cards of an in-progress hand."""


class HandStoreError(ValueError):
    """Raised when a recording call is inconsistent with the hand's current state."""


@dataclass(frozen=True, slots=True, eq=False)
class SeatKey:
    """Capability to read one seat's own hole cards while its hand is in progress."""

    hand_id: str
    seat: int


class _HoleCardVault:
    """Private storage for sealed hole cards, keyed by hand and seat."""

    __slots__ = ("_sealed",)

    def __init__(self) -> None:
        self._sealed: dict[str, dict[int, tuple[SeatKey, tuple[CardObservation, ...]]]] = {}

    def seal(self, hand_id: str, seat: int, cards: tuple[CardObservation, ...]) -> SeatKey:
        hand = self._sealed.setdefault(hand_id, {})
        if seat in hand:
            raise HandStoreError(f"seat {seat} already has hole cards in hand {hand_id!r}")
        key = SeatKey(hand_id, seat)
        hand[seat] = (key, cards)
        return key

    def sealed_cards(self, hand_id: str) -> set[str]:
        return {obs.card for _, cards in self._sealed.get(hand_id, {}).values() for obs in cards}

    def read_own(self, key: SeatKey) -> tuple[CardObservation, ...]:
        entry = self._sealed.get(key.hand_id, {}).get(key.seat)
        if entry is None or entry[0] is not key:
            raise SealedHoleCardsError("invalid seat key for this hand")
        return entry[1]

    def contents_for_completion(self, hand_id: str) -> dict[int, tuple[CardObservation, ...]]:
        """All sealed cards of a hand. Only ``HandHistoryStore.complete_hand`` calls this."""
        return {seat: cards for seat, (_, cards) in self._sealed.get(hand_id, {}).items()}

    def discard(self, hand_id: str) -> None:
        self._sealed.pop(hand_id, None)


class _LiveHand:
    """Mutable record of a hand in progress (public information only)."""

    __slots__ = (
        "actions",
        "board",
        "button_seat",
        "folded",
        "hand_id",
        "index",
        "metadata",
        "pot",
        "seats",
        "source",
        "stacks",
        "started_at",
        "street",
        "street_bets",
        "table",
    )

    def __init__(
        self,
        hand_id: str,
        table: TableConfig,
        button_seat: int,
        seats: tuple[SeatInfo, ...],
        source: str,
        started_at: str | None,
        metadata: Mapping[str, str],
    ) -> None:
        self.hand_id = hand_id
        self.table = table
        self.button_seat = button_seat
        self.seats = seats
        self.index = {s.seat: i for i, s in enumerate(seats)}
        self.source = source
        self.started_at = started_at
        self.metadata = dict(metadata)
        self.actions: list[Action] = []
        self.board: list[CardObservation] = []
        self.street = Street.PREFLOP
        self.stacks = [s.starting_stack for s in seats]
        self.street_bets = [0] * len(seats)
        self.folded = [False] * len(seats)
        self.pot = 0


class HandHistoryStore:
    """Completed hands plus in-progress hands whose hole cards are sealed."""

    def __init__(self) -> None:
        self._completed: dict[str, HandHistory] = {}
        self._live: dict[str, _LiveHand] = {}
        self._vault = _HoleCardVault()

    # ------------------------------------------------------------------ recording a live hand
    def begin_hand(
        self,
        hand_id: str,
        table: TableConfig,
        button_seat: int,
        seats: Sequence[SeatInfo],
        *,
        source: str = "simulator",
        started_at: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        if hand_id in self._live or hand_id in self._completed:
            raise HandStoreError(f"hand {hand_id!r} already exists")
        seats = tuple(seats)
        # Construct (and discard) an in-progress record to reuse the schema's structural checks.
        HandHistory(
            hand_id=hand_id,
            table=table,
            button_seat=button_seat,
            seats=seats,
            status=HandStatus.IN_PROGRESS,
            source=source,
            started_at=started_at,
            metadata=metadata or {},
        )
        self._live[hand_id] = _LiveHand(
            hand_id, table, button_seat, seats, source, started_at, metadata or {}
        )

    def _live_hand(self, hand_id: str) -> _LiveHand:
        try:
            return self._live[hand_id]
        except KeyError:
            if hand_id in self._completed:
                raise HandStoreError(f"hand {hand_id!r} is already complete") from None
            raise KeyError(hand_id) from None

    def deal_hole_cards(self, hand_id: str, seat: int, cards: Sequence[CardObservation]) -> SeatKey:
        """Seal a seat's hole cards and return the key that lets that seat read them."""
        live = self._live_hand(hand_id)
        if seat not in live.index:
            raise HandStoreError(f"seat {seat} is not in hand {hand_id!r}")
        cards = tuple(cards)
        HoleCards(seat=seat, cards=cards)  # structural validation
        visible = {obs.card for obs in live.board} | self._vault.sealed_cards(hand_id)
        if any(obs.card in visible for obs in cards):
            raise HandStoreError("hole card already dealt in this hand")
        return self._vault.seal(hand_id, seat, cards)

    def record_action(self, hand_id: str, action: Action) -> None:
        live = self._live_hand(hand_id)
        if action.seat not in live.index:
            raise HandStoreError(f"seat {action.seat} is not in hand {hand_id!r}")
        if action.street is not live.street:
            raise HandStoreError(f"action on {action.street} but the hand is on {live.street}")
        i = live.index[action.seat]
        if live.folded[i]:
            raise HandStoreError(f"seat {action.seat} has folded")
        if action.amount > live.stacks[i]:
            raise HandStoreError(f"seat {action.seat} cannot put in more than its stack")
        if action.action_type is not ActionType.POST_ANTE:
            if action.total_bet != live.street_bets[i] + action.amount:
                raise HandStoreError(
                    f"total_bet {action.total_bet} != live bet {live.street_bets[i]} "
                    f"+ amount {action.amount}"
                )
            live.street_bets[i] += action.amount
        live.stacks[i] -= action.amount
        live.pot += action.amount
        if action.action_type is ActionType.FOLD:
            live.folded[i] = True
        live.actions.append(action)

    def deal_board(self, hand_id: str, cards: Sequence[CardObservation]) -> Street:
        """Add board cards and advance the street; returns the new street."""
        live = self._live_hand(hand_id)
        cards = tuple(cards)
        if not cards or not all(isinstance(c, CardObservation) for c in cards):
            raise HandStoreError("deal_board needs one or more CardObservation objects")
        new_size = len(live.board) + len(cards)
        try:
            street = street_for_board_size(new_size)
        except ValueError:
            raise HandStoreError(f"a board cannot have {new_size} cards") from None
        visible = {obs.card for obs in live.board} | self._vault.sealed_cards(hand_id)
        names = [obs.card for obs in cards]
        if len(set(names)) != len(names) or visible & set(names):
            raise HandStoreError("board card already dealt in this hand")
        live.board.extend(cards)
        live.street = street
        live.street_bets = [0] * len(live.seats)
        return street

    def abandon_hand(self, hand_id: str) -> None:
        """Discard an in-progress hand (e.g. a misdeal); its sealed cards are destroyed."""
        self._live.pop(hand_id)
        self._vault.discard(hand_id)

    # ------------------------------------------------------------------ firewalled read paths
    def public_state(self, hand_id: str, *, actor_seat: int | None = None) -> PublicHandState:
        live = self._live_hand(hand_id)
        return PublicHandState(
            hand_id=live.hand_id,
            table=live.table,
            button_seat=live.button_seat,
            seats=live.seats,
            street=live.street,
            board=tuple(obs.card for obs in live.board),
            actions=tuple(live.actions),
            stacks=tuple(live.stacks),
            street_bets=tuple(live.street_bets),
            pot=live.pot,
            folded=tuple(live.folded),
            actor_seat=actor_seat,
        )

    def seat_view(
        self,
        hand_id: str,
        key: SeatKey,
        *,
        legal: LegalActions | None = None,
    ) -> SeatView:
        """The key holder's view: public state plus its own hole cards (and legal options when
        ``legal`` is given, which makes the key's seat the actor)."""
        self._live_hand(hand_id)  # completed or unknown hands have no seat views
        if not isinstance(key, SeatKey) or key.hand_id != hand_id:
            raise SealedHoleCardsError("a SeatKey issued for this hand is required")
        cards = self._vault.read_own(key)
        public = self.public_state(hand_id, actor_seat=key.seat if legal is not None else None)
        return SeatView(
            public=public,
            seat=key.seat,
            hole_cards=(cards[0].card, cards[1].card),
            legal=legal,
        )

    def get(self, hand_id: str) -> HandHistory:
        """The hand's record. In-progress hands come back sealed (no hole cards)."""
        if hand_id in self._completed:
            return self._completed[hand_id]
        live = self._live_hand(hand_id)
        return HandHistory(
            hand_id=live.hand_id,
            table=live.table,
            button_seat=live.button_seat,
            seats=live.seats,
            actions=tuple(live.actions),
            board=tuple(live.board),
            status=HandStatus.IN_PROGRESS,
            source=live.source,
            started_at=live.started_at,
            metadata=live.metadata,
        )

    def reveal_hole_cards(self, hand_id: str) -> dict[int, HoleCards]:
        """Hole cards of a completed hand. Raises ``SealedHoleCardsError`` while in progress."""
        if hand_id in self._live:
            raise SealedHoleCardsError(
                f"hand {hand_id!r} is in progress; hole cards are sealed until it is complete"
            )
        hand = self._completed[hand_id]
        return {hc.seat: hc for hc in hand.hole_cards}

    def is_in_progress(self, hand_id: str) -> bool:
        return hand_id in self._live

    # ------------------------------------------------------------------ completion
    def complete_hand(
        self,
        hand_id: str,
        *,
        showdown: Iterable[ShowdownEntry] = (),
        payouts: Iterable[ChipAward] = (),
        uncalled_returns: Iterable[ChipAward] = (),
        rake: int = 0,
        retain: bool = True,
        validate: bool = True,
    ) -> HandHistory:
        """Mark the hand complete: unseal hole cards into the record and validate it.

        A seat's hole cards are flagged ``shown`` if its showdown entry says it showed. With
        ``retain=False`` the completed hand is returned but not kept (used by long simulations).
        """
        live = self._live_hand(hand_id)
        showdown = tuple(showdown)
        shown = {e.seat for e in showdown if e.shown}
        sealed = self._vault.contents_for_completion(hand_id)
        hole_cards = tuple(
            HoleCards(seat=seat, cards=sealed[seat], shown=seat in shown)
            for seat in (s.seat for s in live.seats)
            if seat in sealed
        )
        hand = HandHistory(
            hand_id=live.hand_id,
            table=live.table,
            button_seat=live.button_seat,
            seats=live.seats,
            actions=tuple(live.actions),
            board=tuple(live.board),
            hole_cards=hole_cards,
            showdown=showdown,
            payouts=tuple(payouts),
            uncalled_returns=tuple(uncalled_returns),
            rake=rake,
            status=HandStatus.COMPLETE,
            source=live.source,
            started_at=live.started_at,
            metadata=live.metadata,
        )
        if validate:
            validate_hand_history(hand)  # on failure the hand stays in progress and sealed
        del self._live[hand_id]
        self._vault.discard(hand_id)
        if retain:
            self._completed[hand_id] = hand
        return hand

    def add_completed(self, hand: HandHistory, *, validate: bool = True) -> None:
        """Add an already-complete hand (e.g. loaded from JSON Lines)."""
        if hand.status is not HandStatus.COMPLETE:
            raise HandStoreError("only complete hands can be added directly")
        if hand.hand_id in self._completed or hand.hand_id in self._live:
            raise HandStoreError(f"hand {hand.hand_id!r} already exists")
        if validate:
            validate_hand_history(hand)
        self._completed[hand.hand_id] = hand

    def completed_hands(self) -> Iterator[HandHistory]:
        return iter(tuple(self._completed.values()))

    def __len__(self) -> int:
        return len(self._completed)

"""No-limit hold'em table simulation: PokerKit state machine + sealed hand store.

``TableSimulator`` plays hands between ``Agent`` objects:

1. Cards come from the table's own seeded ``random.Random`` and are dealt to PokerKit
   explicitly, so a seed reproduces every hand exactly.
2. PokerKit enforces the rules (betting order, min-raise and reopening rules, all-ins, side
   pots, showdown order) and pays out pots.
3. Every public event is recorded into a ``HandHistoryStore``. Hole cards are sealed there, and
   each agent's ``SeatView`` is built by the store from the agent's own ``SeatKey``. The engine
   never hands agents anything else.
4. On completion the store emits a validated ``HandHistory``; the engine additionally checks
   that every seat's net result equals PokerKit's payoff.

Stacks reset to the configured starting stack every hand (a cash game with auto top-up) and the
button moves one occupied seat clockwise per hand.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from pokerkit import (
    Automation,
    BlindOrStraddlePosting,
    CheckingOrCalling,
    CompletionBettingOrRaisingTo,
    HoleCardsShowingOrMucking,
    Mode,
    Operation,
    State,
)
from pokerkit.state import AntePosting

from poker_lab.engine.agent import Agent, Decision, DecisionKind, check_decision
from poker_lab.engine.fast_hand import FastNoLimitTexasHoldem
from poker_lab.schema import (
    FULL_DECK,
    Action,
    ActionType,
    CardObservation,
    ChipAward,
    HandHistory,
    HandHistoryStore,
    LegalActions,
    SeatInfo,
    SeatView,
    ShowdownEntry,
    TableConfig,
    compute_uncalled_returns,
    position_names,
    redact_hole_cards,
    seat_order_from_button,
)

AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.RUNOUT_COUNT_SELECTION,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)

_OBS = {card: CardObservation(card) for card in FULL_DECK}  # immutable, safe to share
_POST_ORDER = {
    ActionType.POST_ANTE: 0,
    ActionType.POST_SMALL_BLIND: 1,
    ActionType.POST_BIG_BLIND: 2,
    ActionType.POST_STRADDLE: 3,
}


class EngineError(RuntimeError):
    """Internal inconsistency between PokerKit and the recorded hand (a bug, not bad input)."""


def forced_posts(
    operations: Iterable[Operation], order: Sequence[int], stacks: Mapping[int, int]
) -> list[Action]:
    """Schema actions for PokerKit's ante and blind posts, in conventional order (antes, SB,
    BB, straddles). PokerKit posts in player-index order, which heads-up puts the big blind
    first. ``order`` maps player index to seat; ``stacks`` are the stacks before the posts.
    Other operations are ignored."""
    heads_up = len(order) == 2
    stacks = {s: stacks[s] for s in order}
    posts: list[Action] = []
    for op in operations:
        if isinstance(op, AntePosting):
            seat = order[op.player_index]
            stacks[seat] -= op.amount
            action = Action(seat, "preflop", ActionType.POST_ANTE, op.amount, 0, stacks[seat] == 0)
        elif isinstance(op, BlindOrStraddlePosting):
            i = op.player_index
            if heads_up:
                kind = ActionType.POST_BIG_BLIND if i == 0 else ActionType.POST_SMALL_BLIND
            else:
                kind = (
                    (ActionType.POST_SMALL_BLIND, ActionType.POST_BIG_BLIND)[i]
                    if i < 2
                    else ActionType.POST_STRADDLE
                )
            seat = order[i]
            stacks[seat] -= op.amount
            action = Action(seat, "preflop", kind, op.amount, op.amount, stacks[seat] == 0)
        else:
            continue
        posts.append(action)
    posts.sort(key=lambda a: _POST_ORDER[a.action_type])
    return posts


@dataclass(frozen=True, slots=True)
class SeatedPlayer:
    seat: int
    player_id: str
    agent: Agent


def legal_actions_from_state(state: State) -> LegalActions:
    """The current actor's legal options, read from PokerKit."""
    call_amount = state.checking_or_calling_amount or 0
    can_raise = state.can_complete_bet_or_raise_to()
    return LegalActions(
        can_fold=call_amount > 0,
        can_check=call_amount == 0,
        call_amount=call_amount,
        can_bet_or_raise=can_raise,
        min_raise_to=state.min_completion_betting_or_raising_to_amount if can_raise else 0,
        max_raise_to=state.max_completion_betting_or_raising_to_amount if can_raise else 0,
    )


class TableSimulator:
    """Plays hands at one table. Reproducible: same seed and agents give the same hands."""

    def __init__(
        self,
        table: TableConfig,
        players: Sequence[SeatedPlayer],
        *,
        seed: int,
        starting_stack: int | Mapping[int, int] = 200,
        button_seat: int | None = None,
        hand_id_prefix: str | None = None,
        validate: bool = True,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        if len(players) < 2:
            raise ValueError("a table needs at least two players")
        by_seat = {p.seat: p for p in players}
        if len(by_seat) != len(players):
            raise ValueError("two players in the same seat")
        if any(not 0 <= s < table.max_seats for s in by_seat):
            raise ValueError("seat outside the table")
        ids = [p.player_id for p in players]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate player ids at one table")
        self.table = table
        self.players = dict(sorted(by_seat.items()))
        if isinstance(starting_stack, int):
            self.stacks = dict.fromkeys(self.players, starting_stack)
        else:
            self.stacks = {s: starting_stack[s] for s in self.players}
        self.rng = random.Random(seed)
        self.seed = seed
        seats = list(self.players)
        self.button = button_seat if button_seat is not None else self.rng.choice(seats)
        if self.button not in self.players:
            raise ValueError("button_seat must be occupied")
        self.prefix = hand_id_prefix if hand_id_prefix is not None else f"{table.table_id}-"
        self.validate = validate
        self.metadata = dict(metadata or {})
        self.hands_played = 0
        self.store = HandHistoryStore()
        self._notify = {
            s: p.agent
            for s, p in self.players.items()
            if type(p.agent).hand_completed is not Agent.hand_completed
        }

    # ------------------------------------------------------------------ public API
    def play(self, n_hands: int) -> Iterator[HandHistory]:
        for _ in range(n_hands):
            yield self.play_hand()

    def play_hand(self, deck: Sequence[str] | None = None) -> HandHistory:
        """Play one hand. ``deck`` (a full 52-card order) replaces the table's own shuffle, so
        the same deal can be replayed at another table (duplicate dealing); the table's RNG
        then drives only the agents' decisions."""
        if deck is not None:
            deck = list(deck)
            if sorted(deck) != sorted(FULL_DECK):
                raise ValueError("deck must be an ordering of all 52 cards")
        hand_id = f"{self.prefix}{self.hands_played:06d}"
        hand = self._play(hand_id, self.button, deck)
        self.hands_played += 1
        seats = list(self.players)
        self.button = seats[(seats.index(self.button) + 1) % len(seats)]
        for seat, agent in self._notify.items():
            agent.hand_completed(redact_hole_cards(hand, seat))
        return hand

    # ------------------------------------------------------------------ one hand
    def _play(self, hand_id: str, button: int, deck: list[str] | None = None) -> HandHistory:
        rng = self.rng
        order = seat_order_from_button(self.players, button)  # PokerKit index -> seat
        positions = dict(zip(order, position_names(len(order)), strict=True))
        seat_infos = tuple(
            SeatInfo(s, p.player_id, self.stacks[s], positions[s]) for s, p in self.players.items()
        )
        store = self.store
        store.begin_hand(
            hand_id,
            self.table,
            button,
            seat_infos,
            source="simulator",
            metadata={
                **self.metadata,
                "seed": str(self.seed),
                "hand_number": str(self.hands_played),
            },
        )
        if deck is None:
            deck = list(FULL_DECK)
            rng.shuffle(deck)
        next_card = 0

        t = self.table
        state = FastNoLimitTexasHoldem.create_state(
            AUTOMATIONS,
            True,
            t.ante,
            (t.small_blind, t.big_blind),
            t.big_blind,
            tuple(self.stacks[s] for s in order),
            len(order),
            mode=Mode.TOURNAMENT,
        )
        self._record_forced_posts(state, order, hand_id)

        keys = {}
        while (dealee := state.hole_dealee_index) is not None:
            cards = deck[next_card : next_card + 2]
            next_card += 2
            state.deal_hole("".join(cards), dealee)
            seat = order[dealee]
            keys[seat] = store.deal_hole_cards(hand_id, seat, (_OBS[cards[0]], _OBS[cards[1]]))

        while state.status:
            if state.card_burning_status:
                state.burn_card("??")
            elif (n_board := state.board_dealing_count) is not None:
                cards = deck[next_card : next_card + n_board]
                next_card += n_board
                state.deal_board("".join(cards))
                store.deal_board(hand_id, [_OBS[c] for c in cards])
            elif (actor := state.actor_index) is not None:
                seat = order[actor]
                legal = legal_actions_from_state(state)
                view = store.seat_view(hand_id, keys[seat], legal=legal)
                decision = self.players[seat].agent.act(view, rng)
                check_decision(decision, legal)
                store.record_action(hand_id, self._apply(state, view, decision))
            else:
                raise EngineError(f"PokerKit is waiting on an unexpected operation in {hand_id}")

        return self._complete(state, order, hand_id)

    def _record_forced_posts(self, state: State, order: tuple[int, ...], hand_id: str) -> None:
        for action in forced_posts(state.operations, order, self.stacks):
            self.store.record_action(hand_id, action)

    @staticmethod
    def _apply(state: State, view: SeatView, decision: Decision) -> Action:
        """Apply a (legal) decision to PokerKit and return the matching schema action.

        All-in status comes from the stack *before* the action: when an action ends the hand,
        PokerKit's automations pay the pot out inside the same call.
        """
        street = view.public.street
        before = view.street_bet
        stack = view.stack
        if decision.kind is DecisionKind.FOLD:
            state.fold()
            return Action(view.seat, street, ActionType.FOLD, 0, before)
        if decision.kind is DecisionKind.CHECK_CALL:
            op = state.check_or_call()
            assert isinstance(op, CheckingOrCalling)
            if op.amount == 0:
                return Action(view.seat, street, ActionType.CHECK, 0, before)
            return Action(
                view.seat,
                street,
                ActionType.CALL,
                op.amount,
                before + op.amount,
                op.amount == stack,
            )
        op = state.complete_bet_or_raise_to(decision.to_amount)
        assert isinstance(op, CompletionBettingOrRaisingTo)
        kind = ActionType.RAISE if view.public.max_street_bet > 0 else ActionType.BET
        added = op.amount - before
        return Action(view.seat, street, kind, added, op.amount, added == stack)

    def _complete(self, state: State, order: tuple[int, ...], hand_id: str) -> HandHistory:
        live = self.store.get(hand_id)
        shown: dict[int, bool] = {}
        for op in state.operations:
            if isinstance(op, HoleCardsShowingOrMucking):
                shown[order[op.player_index]] = bool(op.hole_cards)
        contributions = live.contributions()
        uncalled = compute_uncalled_returns(live.actions)
        payouts = []
        for i, seat in enumerate(order):
            received = state.stacks[i] - (self.stacks[seat] - contributions[seat])
            won = received - uncalled.get(seat, 0)
            if won < 0:
                raise EngineError(f"{hand_id}: negative payout for seat {seat}")
            if won:
                payouts.append(ChipAward(seat, won))
        hand = self.store.complete_hand(
            hand_id,
            showdown=[ShowdownEntry(s, shown[s]) for s in sorted(shown)],
            payouts=sorted(payouts, key=lambda a: a.seat),
            uncalled_returns=[ChipAward(s, a) for s, a in sorted(uncalled.items())],
            retain=False,
            validate=self.validate,
        )
        net = hand.net_results()
        for i, seat in enumerate(order):
            if net[seat] != state.payoffs[i]:
                raise EngineError(f"{hand_id}: seat {seat} net {net[seat]} != {state.payoffs[i]}")
        return hand

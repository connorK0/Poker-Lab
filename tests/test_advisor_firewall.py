"""Firewall: the advisor can only ever see public state, its own cards, and profiles built
from completed hands. These tests try to violate that in every way we could think of."""

import dataclasses
from types import MappingProxyType

import pytest
from pokerkit import NoLimitTexasHoldem

from poker_lab.advisor import Advisor, AdvisorAgent, FirewallViolation
from poker_lab.bots import Archetype, ParametricBot, mean_params
from poker_lab.engine import SeatedPlayer, TableSimulator
from poker_lab.profiling.profile_store import (
    IncompleteHandError,
    PlayerProfile,
    ProfileSnapshot,
    ProfileStore,
)
from poker_lab.schema import (
    HandHistory,
    HandHistoryStore,
    HoleCards,
    LegalActions,
    SeatView,
    TableConfig,
)
from poker_lab.schema.store import _HoleCardVault
from poker_lab.strategy import StrategyProvider, UniformRandomStrategy
from tests.helpers import BOARD, HOLE, reachable_card_strings, record_reference_hand, walk_reachable

LEGAL = LegalActions(
    can_fold=False, can_check=True, call_amount=0, can_bet_or_raise=True, min_raise_to=2,
    max_raise_to=164,
)  # fmt: skip


@pytest.fixture
def live():
    """A hand in progress (river, before showdown) with every hole card sealed in the store,
    plus profiles built from earlier completed hands."""
    store = HandHistoryStore()
    _, keys = record_reference_hand(store, "live", complete=False)
    profiles = ProfileStore()
    old = HandHistoryStore()
    for i in range(3):
        hand, _ = record_reference_hand(old, f"old{i}")
        profiles.ingest(hand)
    return store, keys, profiles


def _view(store, keys, seat=3):
    return store.seat_view("live", keys[seat], legal=LEGAL)


def test_advisor_works_with_legitimate_inputs(live):
    store, keys, profiles = live
    advice = Advisor(UniformRandomStrategy()).advise(_view(store, keys), profiles.snapshot())
    assert advice.hand_id == "live"
    assert advice.exploitability.method == "not_computed"
    assert "frank" in advice.opponents_profiled


def test_full_hand_history_is_rejected(live):
    *_, profiles = live
    old = HandHistoryStore()
    complete, _ = record_reference_hand(old, "x")  # has every hole card
    with pytest.raises(TypeError, match="SeatView"):
        Advisor(UniformRandomStrategy()).advise(complete, profiles.snapshot())


def test_lookalike_and_subclass_views_are_rejected(live):
    store, keys, profiles = live
    view = _view(store, keys)
    advisor = Advisor(UniformRandomStrategy())

    @dataclasses.dataclass(frozen=True)
    class Leaky:
        public: object
        seat: int
        hole_cards: tuple
        legal: object
        opponent_cards: tuple

    with pytest.raises(TypeError):
        advisor.advise(Leaky(view.public, 3, view.hole_cards, LEGAL, HOLE[5]), profiles.snapshot())
    with pytest.raises(TypeError):
        advisor.advise({"public": view.public, "hole_cards": HOLE[5]}, profiles.snapshot())


def test_seat_view_has_no_room_for_other_cards(live):
    store, keys, _ = live
    view = _view(store, keys)
    with pytest.raises(TypeError):
        SeatView(view.public, 3, view.hole_cards, LEGAL, opponent_cards=HOLE[5])  # type: ignore[call-arg]
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(view, "opponent_cards", HOLE[5])  # slotted: no __dict__
    with pytest.raises(TypeError):
        SeatView(view.public, 3, (*view.hole_cards, *HOLE[5]), LEGAL)  # 4 "hole cards"

    class SmugglingStr(str):
        pass

    smuggled = SmugglingStr(view.hole_cards[0])
    smuggled.extra = HOLE[5]  # str subclasses can carry attributes; exact type check stops it
    with pytest.raises(TypeError, match="plain str"):
        SeatView(view.public, 3, (smuggled, view.hole_cards[1]), LEGAL)

    class SmugglingTuple(tuple):
        pass

    with pytest.raises(TypeError):
        SeatView(view.public, 3, SmugglingTuple(view.hole_cards), LEGAL)


def test_engine_state_cannot_stand_in_for_public_state(live):
    store, keys, _ = live
    state = NoLimitTexasHoldem.create_state((), True, 0, (1, 2), 2, 200, 2)
    with pytest.raises(TypeError, match="PublicHandState"):
        SeatView(state, 0, ("As", "Kd"), None)  # type: ignore[arg-type]
    view = _view(store, keys)
    with pytest.raises(TypeError):
        dataclasses.replace(view.public, seats=(*view.public.seats, HoleCards(5, None)))


def test_profiles_must_be_a_snapshot_without_the_current_hand(live):
    store, keys, profiles = live
    advisor = Advisor(UniformRandomStrategy())
    view = _view(store, keys)
    with pytest.raises(TypeError, match="ProfileSnapshot"):
        advisor.advise(view, {"frank": {"vpip": 0.2}})  # type: ignore[arg-type]
    tainted = ProfileSnapshot(
        {
            "frank": PlayerProfile(
                "frank", 1, MappingProxyType({}), MappingProxyType({}), MappingProxyType({}), ()
            )
        },
        frozenset({"live"}),  # claims to include the hand being played
    )
    with pytest.raises(FirewallViolation, match="being played"):
        advisor.advise(view, tainted)
    with pytest.raises(IncompleteHandError):
        profiles.ingest(store.get("live"))  # an in-progress hand can never become a profile


def test_advice_only_for_the_actor(live):
    store, keys, profiles = live
    passive_view = store.seat_view("live", keys[3])  # no legal actions: not the actor
    with pytest.raises(ValueError, match="turn"):
        Advisor(UniformRandomStrategy()).advise(passive_view, profiles.snapshot())


def test_nothing_reachable_from_advisor_inputs_reveals_opponent_cards(live):
    store, keys, profiles = live
    advisor = Advisor(UniformRandomStrategy())
    view = _view(store, keys)
    snapshot = profiles.snapshot()
    for root in (advisor, view, snapshot):
        assert reachable_card_strings(root) <= set(HOLE[3]) | set(BOARD)
        for obj in walk_reachable(root):
            assert not isinstance(obj, HandHistoryStore | _HoleCardVault | HoleCards | HandHistory)


class _SpyStrategy(StrategyProvider):
    def __init__(self):
        self.seen: list[object] = []

    def action_distribution(self, view):
        self.seen.append(view)
        return UniformRandomStrategy().action_distribution(view)


def test_advisor_agent_end_to_end_firewall():
    spy = _SpyStrategy()
    hero = AdvisorAgent(Advisor(spy, sample=True, seed=1))
    bots = [ParametricBot(mean_params(a), equity_samples=50) for a in Archetype]
    players = [SeatedPlayer(0, "hero", hero)] + [
        SeatedPlayer(i + 1, f"bot{i}", bot) for i, bot in enumerate(bots)
    ]
    sim = TableSimulator(TableConfig("e", 6, 1, 2), players, seed=12)
    hands = list(sim.play(60))
    assert spy.seen
    by_id = {h.hand_id: h for h in hands}
    for view in spy.seen:
        assert type(view) is SeatView
        hand = by_id[view.public.hand_id]
        own = tuple(o.card for o in hand.hole_cards_of(0).cards)
        assert view.hole_cards == own
        assert reachable_card_strings(view) <= set(own) | set(hand.board_cards())
    # The advisor raises FirewallViolation if a snapshot ever contains the hand being played,
    # so reaching this point means every decision used profiles from earlier hands only.
    assert len(hero.advice_log) == len(spy.seen)
    assert hero.store.hands_ingested == 60  # each hand was profiled only after it completed
    seen_ids = [a.hand_id for a in hero.advice_log]
    assert seen_ids == sorted(seen_ids)  # advice follows play order

"""Simulation engine: rules correctness, chip conservation, reproducibility, firewall."""

import random

import pytest
from pokerkit import StandardHighHand

from poker_lab.engine import (
    Agent,
    AllInAgent,
    CallingAgent,
    Decision,
    FastStandardHighHand,
    IllegalActionError,
    RandomAgent,
    ScriptedAgent,
    SeatedPlayer,
    TableSimulator,
    replay_in_pokerkit,
)
from poker_lab.schema import (
    FULL_DECK,
    ActionType,
    HandHistory,
    SeatView,
    Street,
    TableConfig,
    dumps_hand,
    find_problems,
)
from tests.helpers import reachable_card_strings

TABLE6 = TableConfig("t6", 6, 1, 2)


def make_sim(agents, *, seed=0, table=TABLE6, stacks=200, **kw) -> TableSimulator:
    players = [SeatedPlayer(i, f"p{i}", a) for i, a in enumerate(agents)]
    return TableSimulator(table, players, seed=seed, starting_stack=stacks, **kw)


# ---------------------------------------------------------------- fast evaluator equivalence
def _cmp(a, b) -> int:
    return (a > b) - (a < b)


def test_fast_hand_ordering_matches_pokerkit():
    rng = random.Random(7)
    for i in range(1500):
        n_board = (3, 4, 5)[i % 3]
        cards = rng.sample(FULL_DECK, 4 + n_board)
        board, h1, h2 = "".join(cards[4:]), "".join(cards[:2]), "".join(cards[2:4])
        fast = _cmp(
            FastStandardHighHand.from_game(h1, board), FastStandardHighHand.from_game(h2, board)
        )
        slow = _cmp(StandardHighHand.from_game(h1, board), StandardHighHand.from_game(h2, board))
        assert fast == slow, (h1, h2, board)


@pytest.mark.parametrize(
    ("better", "worse", "board"),
    [
        ("6c7d", "AdKh", "2s3h4c5d9s"),  # 7-high straight beats the wheel
        ("Ah2h", "KcQd", "3s4h5c9dJs"),  # wheel beats king high
        ("2c2d", "AhKh", "2h2s7cQh3h"),  # quads beat a flush
        ("9h8h", "AsAd", "7h6h5hKcKd"),  # straight flush beats a full house
        ("KcQs", "KdJs", "Ks7h7d2c3s"),  # kicker matters
    ],
)
def test_fast_hand_known_orderings(better, worse, board):
    assert FastStandardHighHand.from_game(better, board) > FastStandardHighHand.from_game(
        worse, board
    )


def test_fast_hand_ties_and_board_play():
    board = "AsKsQsJsTs"  # royal flush on board: everyone chops
    assert FastStandardHighHand.from_game("2c3d", board) == FastStandardHighHand.from_game(
        "4h5h", board
    )
    with pytest.raises(ValueError, match="No valid hand"):
        FastStandardHighHand.from_game("AsKs", "2c3d")


# ---------------------------------------------------------------- differential replay
@pytest.mark.parametrize(
    ("n_players", "stacks", "ante"),
    [
        (6, 200, 0),
        (2, 200, 0),
        (3, (60, 200, 140), 0),
        (9, 100, 1),
        (6, (30, 200, 80, 200, 15, 200), 0),
    ],
)
def test_replay_through_unmodified_pokerkit(n_players, stacks, ante):
    """Every recorded action must be legal in vanilla PokerKit (with its own evaluator), and
    PokerKit's payoffs must equal the recorded results."""
    table = TableConfig(f"r{n_players}", max(n_players, 2), 1, 2, ante=ante)
    stack_map = stacks if isinstance(stacks, int) else dict(enumerate(stacks))
    agents = [RandomAgent() if i % 3 else AllInAgent() for i in range(n_players)]
    sim = make_sim(agents, seed=n_players, table=table, stacks=stack_map)
    for hand in sim.play(150):
        result = replay_in_pokerkit(hand)
        assert result.net == hand.net_results(), hand.hand_id
        assert result.showdown == {e.seat: e.shown for e in hand.showdown}, hand.hand_id


# ---------------------------------------------------------------- conservation & validity
@pytest.mark.parametrize("n_players", [2, 3, 6, 9])
def test_hands_are_valid_and_conserve_chips(n_players):
    table = TableConfig("c", 9, 1, 2)
    sim = make_sim([RandomAgent() for _ in range(n_players)], seed=100 + n_players, table=table)
    side_pot_hands = 0
    for hand in sim.play(300):
        assert find_problems(hand) == []
        assert sum(hand.net_results().values()) == 0
        assert hand.is_complete
        assert len(hand.hole_cards) == n_players
        all_in = sum(a.is_all_in for a in hand.actions)
        side_pot_hands += all_in >= 2
    assert side_pot_hands > 0  # the random agent exercises multi-way all-ins


def test_board_and_streets_are_consistent():
    sim = make_sim([CallingAgent() for _ in range(6)], seed=3)
    hand = sim.play_hand()
    # Calling stations never fold, so every hand goes to a 5-card showdown.
    assert len(hand.board) == 5
    assert {a.street for a in hand.actions} == {
        Street.PREFLOP,
        Street.FLOP,
        Street.TURN,
        Street.RIVER,
    }
    assert {e.seat for e in hand.showdown} == set(range(6))


# ---------------------------------------------------------------- legality enforcement
class _Illegal(Agent):
    def __init__(self, decision):
        self.decision = decision

    def act(self, view, rng):
        return self.decision


@pytest.mark.parametrize(
    "decision",
    [
        Decision.bet_raise(1),  # below the minimum raise
        Decision.bet_raise(10_000),  # above the stack
        "fold",  # not a Decision
    ],
)
def test_illegal_decisions_are_rejected(decision):
    sim = make_sim([_Illegal(decision)] * 6, seed=1)
    with pytest.raises(IllegalActionError):
        sim.play_hand()


def test_fold_when_check_is_free_is_rejected():
    # Heads-up: SB/button limps, BB may check for free; folding there is illegal.
    table = TableConfig("hu", 2, 1, 2)
    bb = ScriptedAgent([Decision.fold()])
    btn = ScriptedAgent([Decision.check_call()])
    sim = TableSimulator(
        table, [SeatedPlayer(0, "bb", bb), SeatedPlayer(1, "btn", btn)], seed=0, button_seat=1
    )
    with pytest.raises(IllegalActionError, match="checking is free"):
        sim.play_hand()


def test_heads_up_order_and_blinds():
    table = TableConfig("hu", 2, 1, 2)
    bb = ScriptedAgent([Decision.check_call(), Decision.check_call()])  # check pre, check flop
    btn = ScriptedAgent([Decision.check_call(), Decision.bet_raise(4)])  # limp, bet flop
    bb.script.append(Decision.fold())
    sim = TableSimulator(
        table, [SeatedPlayer(0, "bb", bb), SeatedPlayer(1, "btn", btn)], seed=0, button_seat=1
    )
    hand = sim.play_hand()
    kinds = [(a.seat, a.action_type) for a in hand.actions]
    assert kinds[:2] == [(1, ActionType.POST_SMALL_BLIND), (0, ActionType.POST_BIG_BLIND)]
    assert kinds[2] == (1, ActionType.CALL)  # button acts first preflop
    assert kinds[3] == (0, ActionType.CHECK)
    assert kinds[4] == (0, ActionType.CHECK)  # big blind acts first postflop
    assert kinds[5] == (1, ActionType.BET)
    assert kinds[6] == (0, ActionType.FOLD)
    assert hand.seat_info(1).position == "BTN"
    assert hand.uncalled_returns[0].seat == 1
    assert hand.net_results() == {0: -2, 1: 2}


# ---------------------------------------------------------------- reproducibility & rotation
def test_same_seed_same_hands():
    a = [dumps_hand(h) for h in make_sim([RandomAgent() for _ in range(6)], seed=42).play(50)]
    b = [dumps_hand(h) for h in make_sim([RandomAgent() for _ in range(6)], seed=42).play(50)]
    c = [dumps_hand(h) for h in make_sim([RandomAgent() for _ in range(6)], seed=43).play(50)]
    assert a == b
    assert a != c


def test_button_rotates_clockwise():
    sim = make_sim([CallingAgent() for _ in range(6)], seed=5, button_seat=2)
    buttons = [h.button_seat for h in sim.play(8)]
    assert buttons == [2, 3, 4, 5, 0, 1, 2, 3]


# ---------------------------------------------------------------- firewall at the agent boundary
class _Spy(Agent):
    def __init__(self):
        self.views: list[SeatView] = []
        self.completed: list[HandHistory] = []

    def act(self, view, rng):
        self.views.append(view)
        return Decision.check_call()

    def hand_completed(self, hand):
        self.completed.append(hand)


def test_agents_only_see_their_own_cards():
    spies = [_Spy() for _ in range(6)]
    sim = make_sim(spies, seed=9)
    hands = list(sim.play(20))
    for seat, spy in enumerate(spies):
        assert spy.views
        for view in spy.views:
            assert type(view) is SeatView
            assert view.seat == seat
            assert view.public.actor_seat == seat
            hand = next(h for h in hands if h.hand_id == view.public.hand_id)
            own = tuple(o.card for o in hand.hole_cards_of(seat).cards)
            assert view.hole_cards == own
            assert reachable_card_strings(view) <= set(own) | set(hand.board_cards())


def test_completed_hands_given_to_agents_are_redacted():
    spies = [_Spy() for _ in range(6)]
    sim = make_sim(spies[:2] + [RandomAgent() for _ in range(4)], seed=11)
    list(sim.play(30))
    for seat, spy in enumerate(spies[:2]):
        assert len(spy.completed) == 30
        for hand in spy.completed:
            for hc in hand.hole_cards:
                known = hc.cards is not None
                assert known == (hc.shown or hc.seat == seat)

"""Local best response (P2.8b): equities, rebuilt views, range models and known answers."""

import random
from dataclasses import replace
from itertools import combinations, pairwise

import numpy as np
import pytest
from phevaluator import evaluate_7cards as ev7

from poker_lab.bots import ParametricBot
from poker_lab.bots.archetypes import Archetype, mean_params
from poker_lab.engine.agent import Agent, Decision, DecisionKind
from poker_lab.engine.basic_agents import AllInAgent, CallingAgent
from poker_lab.engine.simulator import SeatedPlayer, TableSimulator
from poker_lab.exploit.lbr.agent import LBRConfig, LocalBestResponse
from poker_lab.exploit.lbr.equity import (
    EquityCache,
    RankTable,
    cards_to_ids,
    combo_cards,
    combo_class_index,
    combo_index,
    preflop_equity_vs_combos,
)
from poker_lab.exploit.lbr.match import lbr_match
from poker_lab.exploit.lbr.models import (
    ParametricBotModel,
    PotOddsModel,
    PushFoldModel,
    Situation,
    StrategyRangeModel,
    likelihood,
)
from poker_lab.exploit.lbr.state import HeadsUpTracker, after_decision
from poker_lab.games.preflop import CLASSES, combos, load_matchups
from poker_lab.games.push_fold import hand_gains, strategy_exploitability
from poker_lab.schema import TableConfig
from poker_lab.strategy import (
    CheckCallStrategy,
    PotOddsStrategy,
    PushFoldStrategy,
    StrategyAgent,
    UniformRandomStrategy,
    fixed_ranges,
)

TABLE = TableConfig("t", 2, small_blind=50, big_blind=100)


# ---------------------------------------------------------------- equities
def _brute_vs_random(hole, board):
    rest = [c for c in range(52) if c not in board and c not in hole]
    num = den = 0.0
    for d in combinations(rest, 2):
        others = [c for c in rest if c not in d]
        for extra in combinations(others, 5 - len(board)):
            full = (*board, *extra)
            a, b = ev7(*hole, *full), ev7(*d, *full)
            num += 1.0 if a < b else (0.5 if a == b else 0.0)
            den += 1
    return num / den


def test_equities_match_brute_force_on_the_turn_and_river():
    for board in [(3, 17, 40, 22), (3, 17, 40, 22, 30)]:
        table = RankTable(board)
        vs_random = table.equity_vs_random(board)
        for hole in [(0, 9), (50, 51), (4, 8)]:
            assert vs_random[combo_index(*hole)] == pytest.approx(_brute_vs_random(hole, board))
        hero, villain = (0, 9), (50, 51)
        vs = table.equity_vs_combos(combo_index(*hero), board)
        rest = [c for c in range(52) if c not in (*board, *hero, *villain)]
        scores = []
        for extra in combinations(rest, 5 - len(board)):
            a, b = ev7(*hero, *board, *extra), ev7(*villain, *board, *extra)
            scores.append(1.0 if a < b else (0.5 if a == b else 0.0))
        assert vs[combo_index(*villain)] == pytest.approx(np.mean(scores))
        assert vs[combo_index(0, 1)] == 0.0  # shares a card with the hero


def test_flop_sample_is_seeded_by_the_board():
    a = RankTable((3, 17, 40), sample=50)
    b = RankTable((3, 17, 40), sample=50)
    assert a.runouts == b.runouts
    assert len(a.runouts) == 50
    assert np.array_equal(a.ranks, b.ranks)


def test_preflop_equity_comes_from_the_class_table():
    m = load_matchups()
    aa = combo_index(48, 49)  # Ac Ad
    eq = preflop_equity_vs_combos(aa)
    k72 = combo_index(0, 21)  # 2c 7d
    i, j = combo_class_index()[aa], combo_class_index()[k72]
    assert (CLASSES[i], CLASSES[j]) == ("AA", "72o")
    assert eq[k72] == pytest.approx(m.equity[i, j])


# ---------------------------------------------------------------- rebuilt views
class _Recorder(Agent):
    def __init__(self, log):
        self.log = log

    def act(self, view, rng):
        self.log.append(view)
        legal = view.legal
        u = rng.random()
        if legal.can_bet_or_raise and u < 0.45:
            lo, hi = legal.min_raise_to, legal.max_raise_to
            return Decision.bet_raise(rng.choice([lo, hi, rng.randint(lo, hi)]))
        if legal.can_fold and u > 0.85:
            return Decision.fold()
        return Decision.check_call()


def test_tracker_rebuilds_every_view_the_simulator_hands_out():
    rng = random.Random(1)
    checked = hypothetical = 0
    for trial in range(80):
        log = []
        stacks = {0: rng.choice([120, 250, 1000, 10000]), 1: rng.choice([150, 999, 10000])}
        players = [SeatedPlayer(0, "a", _Recorder(log)), SeatedPlayer(1, "b", _Recorder(log))]
        sim = TableSimulator(TABLE, players, seed=trial, starting_stack=stacks)
        for _ in range(4):
            log.clear()
            sim.play_hand()
            if not log:
                continue
            tracker = HeadsUpTracker(log[-1].public)
            points = tracker.replay()
            assert len(points) == len(log) - 1
            for view, point in zip(log, points, strict=False):
                assert view.public == point.public
                assert view.legal == point.legal
                checked += 1
            for view, nxt in pairwise(log):
                new = nxt.public.actions[len(view.public.actions) :]
                if (
                    len(new) == 1
                    and new[0].action_type.is_aggressive
                    and nxt.public.street is view.public.street
                ):
                    pub, legal = after_decision(
                        view.public, view.seat, view.legal, Decision.bet_raise(new[0].total_bet)
                    )
                    assert pub == nxt.public
                    assert legal == nxt.legal
                    hypothetical += 1
    assert checked > 500
    assert hypothetical > 100


# ---------------------------------------------------------------- range models
def _views(agent_factory, n_hands=30, seed=3, stack=10_000):
    log = []
    players = [SeatedPlayer(0, "a", _Recorder(log)), SeatedPlayer(1, "b", agent_factory())]
    sim = TableSimulator(TABLE, players, seed=seed, starting_stack=stack)
    for _ in range(n_hands):
        sim.play_hand()
    return log


def _kind_freq(draws):
    out = dict.fromkeys(DecisionKind, 0.0)
    for d in draws:
        out[d.kind] += 1.0 / len(draws)
    return out


def test_strategy_model_matches_the_provider():
    views = _views(CallingAgent, n_hands=10)
    model = StrategyRangeModel(UniformRandomStrategy(), card_blind=True)
    for view in views[:10]:
        k = combo_index(*cards_to_ids(view.hole_cards))
        live = np.zeros(1326, dtype=bool)
        live[k] = True
        table = dict(
            model.table(Situation(view.public, view.seat, view.legal, live, EquityCache()))
        )
        dist = UniformRandomStrategy().action_distribution(view)
        for o in dist.options:
            assert table[o.decision][k] == pytest.approx(o.probability)


def _model_kinds(model, view, cache):
    k = combo_index(*cards_to_ids(view.hole_cards))
    live = np.zeros(1326, dtype=bool)
    live[k] = True
    s = Situation(view.public, view.seat, view.legal, live, cache)
    out = dict.fromkeys(DecisionKind, 0.0)
    for d, v in model.table(s):
        out[d.kind] += v[k]
    return out


@pytest.mark.parametrize("archetype", [Archetype.TAG, Archetype.MANIAC])
def test_bot_model_matches_the_bot_on_average(archetype):
    params = mean_params(archetype)
    views = _views(CallingAgent, n_hands=40, seed=5)
    model = ParametricBotModel(params)
    cache = EquityCache()
    rng = random.Random(0)
    preflop = [v for v in views if not v.public.board][:6]
    postflop = [v for v in views if v.public.board][:6]
    for view, n, tol in [(v, 1500, 0.04) for v in preflop] + [(v, 300, 0.12) for v in postflop]:
        draws = []
        for _ in range(n):
            bot = ParametricBot(params)  # a fresh bot: its equity estimate is cached per street
            draws.append(bot.act(view, rng))
        freq = _kind_freq(draws)
        expected = _model_kinds(model, view, cache)
        for kind in DecisionKind:
            assert freq[kind] == pytest.approx(expected[kind], abs=tol), (view.public.street, kind)


def test_pot_odds_model_is_exact_preflop_and_close_after():
    views = _views(CallingAgent, n_hands=40, seed=6)
    model, cache = PotOddsModel(), EquityCache()
    for view in [v for v in views if not v.public.board][:8]:
        dist = PotOddsStrategy(seed=1).action_distribution(view)
        expected = _model_kinds(model, view, cache)
        for kind in DecisionKind:
            assert expected[kind] == pytest.approx(dist.probability_of(kind), abs=1e-9)
    for view in [v for v in views if v.public.board][:5]:
        draws = [
            PotOddsStrategy(seed=s, equity_samples=200)
            .action_distribution(view)
            .sample(random.Random(s))
            for s in range(300)
        ]
        freq, expected = _kind_freq(draws), _model_kinds(model, view, cache)
        for kind in DecisionKind:
            assert freq[kind] == pytest.approx(expected[kind], abs=0.12)


def test_likelihood_matches_bets_by_nearest_size():
    a, b = np.full(1326, 0.25), np.full(1326, 0.75)
    table = [(Decision.bet_raise(300), a), (Decision.bet_raise(1000), b)]
    assert likelihood(table, Decision.bet_raise(350)) is a
    assert likelihood(table, Decision.bet_raise(900)) is b
    assert likelihood(table, Decision.fold()) is None


# ---------------------------------------------------------------- known answers
def _first_views(stack_bb=10):
    """The small blind's first view and the big blind's view facing a push."""
    sb_log, bb_log = [], []
    stack = stack_bb * 100
    sim = TableSimulator(
        TABLE,
        [SeatedPlayer(0, "sb", _Recorder(sb_log)), SeatedPlayer(1, "bb", CallingAgent())],
        seed=1,
        starting_stack=stack,
        button_seat=0,
    )
    sim.play_hand()
    sim = TableSimulator(
        TABLE,
        [SeatedPlayer(0, "sb", AllInAgent()), SeatedPlayer(1, "bb", _Recorder(bb_log))],
        seed=1,
        starting_stack=stack,
        button_seat=0,
    )
    sim.play_hand()
    return sb_log[0], bb_log[0]


def test_lbr_push_fold_values_are_the_best_response_gains():
    stack = 10.0
    m = load_matchups()
    order = np.argsort(-m.equity_vs_random())
    push = np.zeros(169)
    push[order[:60]] = 1.0
    call = np.zeros(169)
    call[order[:35]] = 1.0
    ranges = fixed_ranges(push, call)
    sb_gain, bb_gain = hand_gains(stack, push, call)
    model = PushFoldModel(PushFoldStrategy(ranges))
    sb_view, bb_view = _first_views()
    for view, gains, aggressive in [
        (sb_view, sb_gain, DecisionKind.BET_RAISE),
        (bb_view, bb_gain, DecisionKind.CHECK_CALL),
    ]:
        for i, cls in enumerate(CLASSES):
            cards = tuple(combo_cards(combo_index(*combos(cls)[0])))
            v = replace(view, hole_cards=cards)
            lbr = LocalBestResponse(model, LBRConfig(actions=("fold", "allin")))
            lbr._sync(v)
            scores = dict(lbr.utilities(v))
            play = next(d for d in scores if d.kind is aggressive)
            # The look-ahead value of playing is the hand's gain over folding, in big blinds
            # (combo-level card removal differs slightly from the class table).
            assert scores[play] / 100 == pytest.approx(gains[i], abs=0.06), cls
            if abs(gains[i]) > 0.06:
                assert (lbr.act(v, random.Random(0)).kind is aggressive) == (gains[i] > 0), cls


def test_lbr_is_near_the_exact_value_in_the_push_fold_game():
    stack = 10.0
    push = np.ones(169)
    call = np.ones(169)
    exact = strategy_exploitability(stack, push, call) * 1000  # mbb/hand
    ranges = fixed_ranges(push, call)
    model = PushFoldModel(PushFoldStrategy(ranges))
    result = lbr_match(
        lambda: StrategyAgent(PushFoldStrategy(ranges)),
        lambda: LocalBestResponse(model, LBRConfig(actions=("fold", "allin"))),
        n_deals=400,
        seed=98,
        stack_bb=stack,
    )
    s = result.summary()
    assert s["lo"] - 3 * s["se"] < exact < s["hi"] + 3 * s["se"]
    assert result.estimate().method == "local_best_response"
    assert not result.estimate().is_upper_bound


def test_lbr_beats_a_calling_station_deep():
    result = lbr_match(
        lambda: StrategyAgent(CheckCallStrategy()),
        lambda: LocalBestResponse(
            StrategyRangeModel(CheckCallStrategy(), card_blind=True), LBRConfig(flop_sample=60)
        ),
        n_deals=25,
        seed=99,
    )
    assert result.summary()["mbb_per_hand"] > 2000
    assert result.stats["surprises"] == 0

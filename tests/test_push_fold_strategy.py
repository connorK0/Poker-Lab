"""P3.3: the push/fold StrategyProvider maps real hold'em hands onto the solved abstract game."""

import numpy as np
import pytest

from poker_lab.bots.hand_strength import hand_class
from poker_lab.engine import Decision, DecisionKind, ScriptedAgent, SeatedPlayer, TableSimulator
from poker_lab.games.evaluate import exploitability
from poker_lab.games.preflop import CLASS_INDEX
from poker_lab.games.push_fold import BB, SB, push_fold_tree, range_share, threshold_policy
from poker_lab.schema import ActionType, Street, TableConfig
from poker_lab.strategy import (
    CheckCallStrategy,
    OffTree,
    PushFoldStrategy,
    StrategyAgent,
    fixed_ranges,
    nash_ranges,
    push_fold_spot,
)

HU = TableConfig("pf", 2, 50, 100)
CHECK = Decision.check_call()


class Collector(StrategyAgent):
    def __init__(self, strategy):
        super().__init__(strategy)
        self.views = []

    def act(self, view, rng):
        self.views.append(view)
        return super().act(view, rng)


def _play(table, agents, n, stacks, seed=0, button=0):
    players = [SeatedPlayer(s, f"p{s}", a) for s, a in enumerate(agents)]
    sim = TableSimulator(table, players, seed=seed, starting_stack=stacks, button_seat=button)
    return list(sim.play(n))


def _sb_seat(view):
    return next(a.seat for a in view.public.actions if a.action_type is ActionType.POST_SMALL_BLIND)


def test_nash_ranges_are_an_exact_equilibrium():
    push, call = nash_ranges(10.0)
    assert nash_ranges(10.0) is nash_ranges(10.0)  # solved once per stack
    assert not push.flags.writeable
    assert not call.flags.writeable
    tree = push_fold_tree(10.0)
    assert exploitability(threshold_policy(tree, push, call)) < 1e-7
    assert range_share(push) == pytest.approx(0.58, abs=0.01)  # P3.1: about 58% pushed
    assert range_share(call) == pytest.approx(0.37, abs=0.01)  # and about 37% called


def test_heads_up_spots_map_to_the_solved_game():
    hero = Collector(PushFoldStrategy())
    _play(HU, [hero, StrategyAgent(PushFoldStrategy())], 300, 1000, seed=1)
    push, call = nash_ranges(10.0)
    roles = set()
    for view in hero.views:
        spot = push_fold_spot(view)
        assert (spot.stack, spot.heads_up) == (10.0, True)
        assert spot.hand_class == hand_class(*view.hole_cards)
        role = SB if _sb_seat(view) == view.seat else BB
        assert spot.player == role
        roles.add(role)
        dist = hero.strategy.action_distribution(view)
        k = CLASS_INDEX[spot.hand_class]
        if role == SB:
            assert dist.probability_of(DecisionKind.BET_RAISE) == pytest.approx(push[k])
            for o in dist.options:  # a push is all-in: a raise to the hero's whole stack
                assert o.decision.kind is DecisionKind.FOLD or o.decision.to_amount == 1000
        else:
            assert dist.probability_of(DecisionKind.CHECK_CALL) == pytest.approx(call[k])
    assert roles == {SB, BB}


@pytest.mark.parametrize("stacks", [{0: 600, 1: 2000}, {0: 2000, 1: 600}])
def test_uneven_stacks_use_the_effective_stack(stacks):
    hero = Collector(PushFoldStrategy())
    villain = Collector(PushFoldStrategy())
    _play(HU, [hero, villain], 60, stacks, seed=2)
    spots = [push_fold_spot(v) for v in hero.views + villain.views]
    assert {s.stack for s in spots} == {6.0}
    assert {s.player for s in spots} == {SB, BB}


def test_fixed_ranges_play_the_same_at_every_stack():
    ranges = fixed_ranges(np.ones(169), np.zeros(169))  # push everything, never call
    hero = Collector(PushFoldStrategy(ranges, max_stack=20.0, name="always_push"))
    _play(HU, [hero, StrategyAgent(PushFoldStrategy(ranges))], 40, 1500, seed=3)
    kinds = {
        push_fold_spot(v).player: hero.strategy.action_distribution(v).most_likely().kind
        for v in hero.views
    }
    assert kinds == {SB: DecisionKind.BET_RAISE, BB: DecisionKind.FOLD}


def _off_tree_views(table, villain_script, stacks, *, seats=2, villain_seat=0, n=1):
    hero = Collector(PushFoldStrategy(fallback=CheckCallStrategy()))
    agents = [None] * seats
    agents[villain_seat] = ScriptedAgent(villain_script)
    for s in range(seats):
        if agents[s] is None:
            hero_seat = s == (villain_seat + 1) % seats
            agents[s] = hero if hero_seat else StrategyAgent(CheckCallStrategy())
    _play(table, agents, n, stacks, seed=4)
    return hero


def test_off_tree_spots_fall_back_or_raise():
    # The small blind (seat 0 on the button heads-up) limps: the big blind's spot is off the tree.
    hero = _off_tree_views(HU, [CHECK] * 4, 1000)
    first = hero.views[0]
    with pytest.raises(OffTree, match="facing a push"):
        push_fold_spot(first)
    with pytest.raises(OffTree):
        PushFoldStrategy().action_distribution(first)  # no fallback: refuse
    fallback = CheckCallStrategy().action_distribution(first)
    assert hero.strategy.action_distribution(first) == fallback
    with pytest.raises(OffTree, match="not preflop"):
        push_fold_spot(hero.views[-1])
    # A min-raise is not read as a push.
    hero = _off_tree_views(HU, [Decision.bet_raise(200)] + [CHECK] * 3, 1000)
    with pytest.raises(OffTree, match="facing a push"):
        push_fold_spot(hero.views[0])


@pytest.mark.parametrize(
    ("table", "stacks", "reason"),
    [
        (HU, 2500, "outside"),  # 25 bb: deeper than the solved range
        (HU, 100, "outside"),  # 1 bb: the big blind is all-in from posting
        (TableConfig("ante", 2, 50, 100, ante=10), 1000, "no ante"),
        (TableConfig("odd", 2, 50, 150), 1500, "1:2"),
    ],
)
def test_unsupported_tables_and_stacks_are_off_the_tree(table, stacks, reason):
    hero = Collector(PushFoldStrategy(fallback=CheckCallStrategy()))
    _play(table, [hero, StrategyAgent(CheckCallStrategy())], 2, stacks, seed=5)
    preflop = [v for v in hero.views if v.public.street is Street.PREFLOP]
    assert preflop
    for view in preflop:
        with pytest.raises(OffTree, match=reason):
            push_fold_spot(view)


def test_blind_against_blind_after_folds():
    three = TableConfig("bvb", 3, 50, 100)
    # Button on seat 0 acts first preflop three-handed; the hero (seat 1) is the small blind.
    hero = _off_tree_views(three, [Decision.fold()], 1000, seats=3)
    spot = push_fold_spot(hero.views[0])
    assert (spot.player, spot.stack, spot.heads_up) == (SB, 10.0, False)
    hero = _off_tree_views(three, [CHECK] * 4, 1000, seats=3)  # the button limps in
    with pytest.raises(OffTree, match="entered the pot"):
        push_fold_spot(hero.views[0])

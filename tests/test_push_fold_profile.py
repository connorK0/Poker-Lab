"""P3.4: push/fold observations of an opponent, built from completed hand histories only."""

import dataclasses

import numpy as np
import pytest

from poker_lab.advisor import Advisor, AdvisorAgent
from poker_lab.bots.hand_strength import hand_class
from poker_lab.engine import Decision, ScriptedAgent, SeatedPlayer, TableSimulator
from poker_lab.exploit import PushFoldRecommender
from poker_lab.games.preflop import CLASS_INDEX
from poker_lab.games.push_fold_bots import (
    ARCHETYPES,
    PushFoldBotParams,
    PushFoldPrior,
    sample_params,
    strategy,
)
from poker_lab.profiling.profile_store import IncompleteHandError
from poker_lab.profiling.push_fold import push_fold_observations
from poker_lab.schema import ActionType, CardObservation, HandStatus, HoleCards, TableConfig
from poker_lab.strategy import (
    ActionDistribution,
    PushFoldStrategy,
    StrategyAgent,
    StrategyProvider,
    fixed_ranges,
    require_actor,
)

HU = TableConfig("pf", 2, 50, 100)
BOT = PushFoldBotParams(0.7, 0.5, 0.03)


def _bot_agent(params=BOT):
    return StrategyAgent(PushFoldStrategy(fixed_ranges(*strategy(params)), name="bot"))


def _play(agents, n, *, table=HU, stack=1000, seed=3, prefix=None, ids=("hero", "villain", "x")):
    players = [SeatedPlayer(s, ids[s], a) for s, a in enumerate(agents)]
    sim = TableSimulator(table, players, seed=seed, starting_stack=stack, hand_id_prefix=prefix)
    return list(sim.play(n))


@pytest.fixture(scope="module")
def hands():
    return _play([StrategyAgent(PushFoldStrategy()), _bot_agent()], 400)


def _sb(hand):
    return next(a.seat for a in hand.actions if a.action_type is ActionType.POST_SMALL_BLIND)


def _cls(hand, seat):
    return CLASS_INDEX[hand_class(*(c.card for c in hand.hole_cards_of(seat).cards))]


def test_observations_match_the_records(hands):
    """An independent count straight from the full (unredacted) simulator records."""
    obs = push_fold_observations(hands, "hero", "villain", stack=10).observations
    expected = {"sb_shown": np.zeros(169), "sb_hidden": np.zeros((2, 169))}
    expected |= {"bb_shown": np.zeros(169), "bb_hidden": np.zeros((2, 169))}
    for hand in hands:
        hero, villain = hand.seat_of("hero"), hand.seat_of("villain")
        moves = [a for a in hand.actions if a.action_type.is_decision]
        showdown = len(hand.board_cards()) == 5
        if _sb(hand) == villain:
            key, action = "sb", int(moves[0].action_type is ActionType.RAISE)
        elif moves[0].action_type is ActionType.RAISE:
            key, action = "bb", int(moves[1].action_type is ActionType.CALL)
        else:
            continue
        if showdown:
            expected[f"{key}_shown"][_cls(hand, villain)] += 1
        else:
            expected[f"{key}_hidden"][action, _cls(hand, hero)] += 1
    assert obs.n_hands == len(hands)
    for key, value in expected.items():
        np.testing.assert_array_equal(getattr(obs, key), value)
    assert obs.n_sb == 200
    assert 0.6 < obs.pushes / obs.n_sb < 0.85  # the bot pushes about 70% of hands


def test_unshown_opponent_cards_never_reach_the_observations(hands):
    """Forging every card the opponent did not show changes nothing: the hands are redacted to
    the hero's view before anything is counted."""

    def forge(hand):
        villain = hand.seat_of("villain")
        hc = hand.hole_cards_of(villain)
        if hc.shown:
            return hand
        used = {c.card for h in hand.hole_cards for c in h.cards} | set(hand.board_cards())
        fake = [c for c in ("2c", "7d", "3h", "8s", "4c", "9d") if c not in used][:2]
        cards = tuple(CardObservation(c) for c in fake)
        hole = tuple(HoleCards(h.seat, cards) if h.seat == villain else h for h in hand.hole_cards)
        return dataclasses.replace(hand, hole_cards=hole)

    forged = [forge(h) for h in hands]
    assert any(f is not h for f, h in zip(forged, hands, strict=True))
    a = push_fold_observations(hands, "hero", "villain", stack=10).observations
    b = push_fold_observations(forged, "hero", "villain", stack=10).observations
    for key in ("sb_shown", "sb_hidden", "bb_shown", "bb_hidden"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))


def test_in_progress_hands_are_refused(hands):
    live = dataclasses.replace(
        hands[0],
        status=HandStatus.IN_PROGRESS,
        hole_cards=(),
        showdown=(),
        payouts=(),
        uncalled_returns=(),
    )
    with pytest.raises(IncompleteHandError):
        push_fold_observations([*hands[:5], live], "hero", "villain", stack=10)
    with pytest.raises(TypeError):
        push_fold_observations([{"hand_id": "x"}], "hero", "villain", stack=10)


def test_hands_off_the_tree_are_skipped_and_counted(hands):
    limp = Decision.check_call()
    limped = _play(
        [ScriptedAgent([limp] * 4), ScriptedAgent([limp] * 4)], 1, seed=1, prefix="limp-"
    )
    deeper = _play(
        [StrategyAgent(PushFoldStrategy()), _bot_agent()], 20, stack=1500, seed=2, prefix="deep-"
    )
    history = push_fold_observations(
        [*hands[:50], *hands[:10], *limped, *deeper], "hero", "villain", stack=10
    )
    assert history.observations.n_hands == 50
    assert history.skipped == {"duplicate": 10, "off_tree": 1, "stack": 20}
    assert len(history.hand_ids) == 50
    other = push_fold_observations(hands[:30], "hero", "nobody", stack=10)
    assert other.observations.n_hands == 0
    assert other.skipped == {"not_the_blinds": 30}


class _Fold(StrategyProvider):
    """Folds (used off the push/fold tree, i.e. on the button of a three-handed table)."""

    def action_distribution(self, view):
        require_actor(view)
        return ActionDistribution.from_weights({Decision.fold(): 1.0})


def test_blind_against_blind_after_a_fold_counts_and_other_pairs_do_not():
    table = TableConfig("pf3", 3, 50, 100)
    agents = [
        StrategyAgent(PushFoldStrategy(fallback=_Fold())),
        StrategyAgent(PushFoldStrategy(fixed_ranges(*strategy(BOT)), fallback=_Fold())),
        StrategyAgent(PushFoldStrategy(fallback=_Fold())),
    ]
    played = _play(agents, 30, table=table)
    history = push_fold_observations(played, "hero", "villain", stack=10)
    # in each orbit of three hands, hero and villain are the blinds once
    assert history.observations.n_hands == 10
    assert history.skipped == {"not_the_blinds": 20}


def test_end_to_end_from_hands_to_advice(hands):
    """Completed hands -> observations -> ranked recommendations -> advised play in real hands."""
    rng = np.random.default_rng(5)
    prior = PushFoldPrior.fit([sample_params(a, rng) for a in ARCHETYPES for _ in range(20)])
    obs = push_fold_observations(hands, "hero", "villain", stack=10).observations
    report = PushFoldRecommender(10.0, prior, draws=500, seed=1).recommend(obs)
    best = report.best
    assert best.name != "equilibrium"  # 400 hands of a loose bot: some deviation is safe
    assert "Seen 400 hands" in report.explanation()
    advisor = Advisor(PushFoldStrategy(fixed_ranges(best.push, best.call)), sample=True, seed=2)
    played = _play([AdvisorAgent(advisor), _bot_agent()], 60, seed=9)
    assert len(played) == 60
    assert all(len(h.actions) <= 4 for h in played)  # blinds, push or fold, call or fold

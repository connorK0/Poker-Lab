"""Push/fold bots that adapt to the hero, and sessions scored by exact values (P3.9)."""

import numpy as np
import pytest

from poker_lab.games.push_fold import (
    BB,
    SB,
    aggressive_probabilities,
    match_values,
    push_fold_tree,
    range_share,
)
from poker_lab.games.push_fold_adaptive import (
    STATIC,
    Adaptation,
    AdaptiveBot,
    best_response_shares,
    play_session,
)
from poker_lab.games.push_fold_bots import PushFoldBotParams
from poker_lab.solvers.lp import solve_lp

_NASH = solve_lp(push_fold_tree(10.0)).policy
EQ = (aggressive_probabilities(_NASH, SB), aggressive_probabilities(_NASH, BB))  # at 10 bb

NIT = PushFoldBotParams(0.2, 0.1, 0.03)
ALWAYS = (np.ones(169), np.zeros(169))  # push everything, never call
FAST = Adaptation(0.5, 50.0)


def _fixed(strategy):
    return lambda log, n: strategy


def test_a_static_bot_never_changes():
    bot = AdaptiveBot(NIT, STATIC, 10.0, EQ)
    session = play_session(_fixed(ALWAYS), bot, 300, np.random.default_rng(0))
    assert set(session.bot_params) == {NIT}
    assert bot.params == NIT


def test_rates_follow_what_the_hero_does():
    bot = AdaptiveBot(NIT, FAST, 10.0, EQ)
    start = bot.rates
    assert start == pytest.approx((range_share(EQ[0]), range_share(EQ[1])))
    play_session(_fixed(ALWAYS), bot, 400, np.random.default_rng(1))
    push_rate, call_rate = bot.rates
    assert push_rate > 0.99  # the hero pushed every time it was the small blind
    assert call_rate < 0.01  # and never called


def test_bots_move_toward_the_best_response():
    """Against a hero that pushes everything and never calls, the best response calls wider
    and pushes everything; a calls-only bot moves its call width alone."""
    bot = AdaptiveBot(NIT, FAST, 10.0, EQ)
    play_session(_fixed(ALWAYS), bot, 500, np.random.default_rng(2))
    assert bot.params.call_width > 0.3
    assert bot.params.push_width > 0.9
    calls = AdaptiveBot(NIT, Adaptation(0.5, 50.0, adapts=("call",)), 10.0, EQ)
    play_session(_fixed(ALWAYS), calls, 500, np.random.default_rng(2))
    assert calls.params.push_width == NIT.push_width
    assert calls.params.call_width == pytest.approx(bot.params.call_width, abs=0.05)
    with pytest.raises(ValueError, match="adapts"):
        Adaptation(0.5, adapts=("raise",))


def test_the_best_response_to_the_equilibrium_plays_its_shares():
    push, call = best_response_shares(10.0, *EQ)
    assert push == pytest.approx(range_share(EQ[0]), abs=0.02)
    assert call == pytest.approx(range_share(EQ[1]), abs=0.02)


def test_sessions_score_exact_values_and_hide_unshown_cards():
    seen = []

    def spy(log, n):
        if log is not None:
            assert len(log.action) == n
            assert np.all((log.bot_class == -1) == ~log.shown)  # only shown classes
            seen.append(n)
        return EQ

    bot = AdaptiveBot(NIT, FAST, 10.0, EQ)
    session = play_session(spy, bot, 110, np.random.default_rng(3))
    assert seen == [25, 50, 75, 100]
    assert session.sizes.tolist() == [25, 25, 25, 25, 10]
    for k in range(len(session.values)):
        exact = match_values(
            10.0, EQ[0], EQ[1], session.bot_push[k], session.bot_call[k], bot.matchups
        )
        assert session.values[k] == pytest.approx(float(exact))
    weights = session.sizes / 110
    assert session.value() == pytest.approx(float(session.values @ weights))
    assert session.value(0, 50) == pytest.approx(session.values[:2].mean())


def test_heroes_facing_the_same_bot_get_the_same_cards():
    a = play_session(_fixed(EQ), AdaptiveBot(NIT, FAST, 10.0, EQ), 200, np.random.default_rng(4))
    b = play_session(
        _fixed(ALWAYS), AdaptiveBot(NIT, FAST, 10.0, EQ), 200, np.random.default_rng(4)
    )
    for name in ("bot_is_sb", "hero_class", "bot_class"):
        np.testing.assert_array_equal(getattr(a.log, name), getattr(b.log, name))
    assert not np.array_equal(a.log.action, b.log.action)

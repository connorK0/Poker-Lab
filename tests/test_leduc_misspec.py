"""Leduc opponents outside the five-parameter family and adaptive bots (P2.12)."""

import numpy as np
import pytest

from poker_lab.games import LeducHoldem, compile_tree, exploitability, head_to_head
from poker_lab.games.leduc_bots import (
    ARCHETYPE_MEANS,
    BetaPrior,
    BotPolicyBuilder,
    LeducBotParams,
    ObservationIndex,
    fit_params,
    simulate_hands,
)
from poker_lab.games.leduc_misspec import (
    RULES,
    STATIC,
    Adaptation,
    AdaptiveLeducBot,
    FamilyProjection,
    Misspecification,
    equilibrium_prior,
    family_rates,
    full_best_response,
    infoset_features,
    mix_plans,
    play_session,
    rates_policy,
    rule_policy,
    simulate_views,
)
from poker_lab.solvers import solve_lp

GAME = LeducHoldem()
TREE = compile_tree(GAME)
BUILDER = BotPolicyBuilder(TREE, GAME)
EQ = solve_lp(TREE).policy
FEATURES = infoset_features(BUILDER)
TAG = ARCHETYPE_MEANS["tag"]
MANIAC = ARCHETYPE_MEANS["maniac"]


@pytest.fixture(scope="module")
def projection():
    return FamilyProjection(BUILDER, EQ)


def _rule(rule, params=TAG, rng=None):
    rng = rng if rng is not None else np.random.default_rng(1)
    return rule_policy(rule, BUILDER, params, features=FEATURES, rng=rng, equilibrium=EQ)


def _at(policy, key, action):
    return policy.action_probabilities(key)[action]


def test_family_rates_rebuild_the_family_policy_bit_for_bit():
    theta = TAG.as_array()
    aggr = theta[[0, 1, 2]][BUILDER.infoset_bucket]
    fold = np.array([theta[3], theta[4], 0.0])[BUILDER.infoset_bucket]
    assert np.array_equal(rates_policy(BUILDER, aggr, fold).probs, BUILDER.policy(TAG).probs)


def test_features_read_the_keys():
    i = TREE.infoset_index["JsKh|bc/kb"]
    assert FEATURES.seat[i] == 0
    assert FEATURES.second_street[i]
    assert FEATURES.facing[i]
    assert FEATURES.raises[i] == 1
    assert FEATURES.rank_bucket[i] == 0  # a jack
    j = TREE.infoset_index["Qs|b"]
    assert FEATURES.seat[j] == 1
    assert not FEATURES.second_street[j]
    assert FEATURES.facing[j]


@pytest.mark.parametrize("rule", RULES)
def test_every_rule_is_a_valid_strategy_away_from_its_family_bot(rule):
    policy = _rule(rule)
    assert policy.probs.shape == BUILDER.policy(TAG).probs.shape
    assert policy.max_difference(BUILDER.policy(TAG)) > 0.01


def test_the_rules_do_what_they_say():
    a, b, m, v = TAG, TAG.bluff, TAG.medium_aggression, TAG.value
    blind = _rule("board_blind")
    # a pair of jacks plays as a weak hand; an unpaired king as a strong one
    assert _at(blind, "JsJh|kk/", "b") == pytest.approx(b)
    assert _at(blind, "KsJh|kk/", "b") == pytest.approx(v)
    assert _at(blind, "KsJh|kk/b", "f") == 0.0
    assert _at(blind, "JsJh|kk/b", "f") == pytest.approx(a.fold_weak)

    street = _rule("street")
    assert _at(street, "Qs|", "b") == pytest.approx(m / 4)
    assert _at(street, "Qs|b", "f") == pytest.approx(a.fold_medium / 2)
    assert _at(street, "KsJh|kk/", "b") == pytest.approx(1 - (1 - m) ** 2)  # medium: top card
    assert _at(street, "KsJh|kk/b", "f") == pytest.approx(1 - (1 - a.fold_medium) ** 2)

    pos = _rule("positional")
    assert _at(pos, "Js|", "b") == pytest.approx(b / 2)  # seat 0
    assert _at(pos, "Js|k", "b") == pytest.approx(1 - (1 - b) ** 3)  # seat 1, checked to
    assert _at(pos, "Js|b", "f") == pytest.approx(a.fold_weak)  # facing a bet: unchanged

    honest = _rule("honest_raiser")
    assert _at(honest, "Js|", "b") == pytest.approx(b)  # bets as usual
    assert _at(honest, "Js|b", "r") == 0.0  # no raise without a strong hand
    assert _at(honest, "Ks|b", "r") == pytest.approx(v)
    assert _at(honest, "Qs|kbr", "f") == pytest.approx(1 - (1 - a.fold_medium) ** 3)
    assert _at(honest, "Ks|kbr", "f") == 0.0


def test_the_slight_rule_is_a_quarter_of_the_full_one():
    want = 0.25 * _rule("street").probs + 0.75 * BUILDER.policy(TAG).probs
    assert np.allclose(_rule("slight_street").probs, want)


def test_near_equilibrium_is_a_mix_and_equilibrium_is_exact():
    assert _rule("equilibrium").max_difference(EQ) == 0.0
    near = _rule("near_eq")
    assert exploitability(near) < exploitability(BUILDER.policy(TAG))
    assert np.allclose(near.probs, 0.25 * BUILDER.policy(TAG).probs + 0.75 * EQ.probs)


def test_noisy_draws_from_its_generator_only():
    one = _rule("noisy", rng=np.random.default_rng(5))
    assert np.array_equal(one.probs, _rule("noisy", rng=np.random.default_rng(5)).probs)
    assert not np.array_equal(one.probs, _rule("noisy", rng=np.random.default_rng(6)).probs)


def test_rules_that_need_inputs_say_so():
    with pytest.raises(ValueError, match="unknown"):
        rule_policy("nope", BUILDER, TAG)
    with pytest.raises(ValueError, match="generator"):
        rule_policy("noisy", BUILDER, TAG)
    with pytest.raises(ValueError, match="equilibrium"):
        rule_policy("near_eq", BUILDER, TAG)


# ---- the projection ------------------------------------------------------------------------


def test_outcome_probabilities_sum_to_one(projection):
    assert projection._class_weights(BUILDER.policy(TAG)).sum() == pytest.approx(1.0)
    assert projection._class_weights(_rule("street")).sum() == pytest.approx(1.0)


def test_exact_likelihood_matches_the_sampled_one(projection):
    """A sample of hands has about the exact expected log-likelihood per hand."""
    policy = _rule("street")
    obs = simulate_hands(TREE, GAME, EQ, policy, 30000, np.random.default_rng(3))
    candidate = BUILDER.probs(MANIAC.as_array())
    sampled = ObservationIndex(TREE, GAME, obs).log_likelihood(candidate) / len(obs)
    exact = projection._class_weights(policy) @ projection._class_loglik(candidate)
    assert sampled == pytest.approx(exact, abs=0.03)


def test_a_family_bot_is_its_own_closest_member(projection):
    m = projection.measure(BUILDER.policy(TAG))
    assert m.kl < 1e-6
    assert m.tv < 0.01
    assert np.abs(m.theta.as_array() - TAG.as_array()).max() < 0.01
    assert m.hands_to_evidence > 1e6


@pytest.mark.parametrize("rule", ["board_blind", "street", "positional", "honest_raiser"])
def test_rule_bots_are_outside_the_family(projection, rule):
    m = projection.measure(_rule(rule))
    assert m.kl > 1e-3
    assert m.tv > 1e-3


def test_the_fit_converges_to_the_closest_member(projection):
    """Plain maximum likelihood on many hands of an off-family bot lands on ``theta*``."""
    policy = _rule("street")
    obs = simulate_hands(TREE, GAME, EQ, policy, 30000, np.random.default_rng(4))
    fit = fit_params(BUILDER, ObservationIndex(TREE, GAME, obs), None)
    closest = projection.closest(policy)
    assert np.abs(fit.as_array() - closest.as_array()).max() < 0.03


def test_total_variation(projection):
    a, b = BUILDER.policy(TAG), BUILDER.policy(MANIAC)
    assert projection.total_variation(a, a) == 0.0
    assert 0.0 < projection.total_variation(a, b) <= 1.0


def test_infinite_evidence_when_nothing_separates():
    assert Misspecification(TAG, 0.0, 0.0).hands_to_evidence == float("inf")
    assert Misspecification(TAG, np.log(20), 0.0).hands_to_evidence == pytest.approx(1.0)


# ---- views ---------------------------------------------------------------------------------


def test_the_hero_view_is_simulate_hands():
    bot = _rule("street")
    hero_view, _ = simulate_views(TREE, GAME, EQ, bot, 300, np.random.default_rng(8))
    assert hero_view == simulate_hands(TREE, GAME, EQ, bot, 300, np.random.default_rng(8))


def test_each_view_hides_the_other_players_card_unless_shown():
    bot = _rule("positional")
    hero_view, bot_view = simulate_views(TREE, GAME, EQ, bot, 400, np.random.default_rng(9))
    shown = 0
    for h, b in zip(hero_view, bot_view, strict=True):
        assert h.actions == b.actions
        assert h.board == b.board
        assert h.bot_seat == 1 - b.bot_seat
        assert h.bot_payoff == -b.bot_payoff
        showdown = h.board is not None and h.actions[-1] != "f"
        assert (h.bot_card is not None) == showdown == (b.bot_card is not None)
        assert bool(h.bot_sequences) == bool(h.bot_card is not None and h.bot_sequences)
        if showdown:
            shown += 1
            assert h.bot_card == b.hero_card
            assert b.bot_card == h.hero_card
        else:
            assert h.bot_sequences == ()
            assert b.bot_sequences == ()
    assert shown > 0


def test_seats_alternate_by_hand_number():
    views, _ = simulate_views(TREE, GAME, EQ, EQ, 4, np.random.default_rng(1), first_hand=50)
    assert [o.bot_seat for o in views] == [1, 0, 1, 0]
    views, _ = simulate_views(TREE, GAME, EQ, EQ, 3, np.random.default_rng(1), first_hand=51)
    assert [o.bot_seat for o in views] == [0, 1, 0]


# ---- adaptation ----------------------------------------------------------------------------


def _prior(projection):
    return equilibrium_prior(projection.closest(EQ), 10.0)


def test_family_rates_read_a_family_bot_back():
    bot = BUILDER.policy(TAG)
    rates = family_rates(BUILDER, bot, EQ, np.full(5, 0.5))
    assert rates == pytest.approx(TAG.as_array(), abs=1e-9)


def test_family_rates_keep_the_fallback_without_opportunities():
    always_fold_hero = BUILDER.policy(LeducBotParams(0.0, 0.0, 0.0, 1.0, 1.0))
    never_bets = BUILDER.policy(LeducBotParams(0.0, 0.0, 0.0, 0.5, 0.5))
    rates = family_rates(BUILDER, always_fold_hero, never_bets, np.array([0, 0, 0, 0.3, 0.4]))
    assert rates[3:] == pytest.approx([0.3, 0.4])


def test_mixing_plans_is_mixing_strategies():
    """Kuhn's theorem: the mix's value against anything is the mix of the two values."""
    a, b = BUILDER.policy(TAG), BUILDER.policy(MANIAC)
    mixed = mix_plans(a, b, 0.3)
    opponent = _rule("street")
    want = 0.7 * head_to_head(a, opponent) + 0.3 * head_to_head(b, opponent)
    assert head_to_head(mixed, opponent) == pytest.approx(want, abs=1e-12)


def test_mixing_with_speed_zero_changes_nothing():
    a, b = BUILDER.policy(TAG), full_best_response(BUILDER.policy(MANIAC))
    assert mix_plans(a, b, 0.0).max_difference(a) < 1e-12


@pytest.mark.parametrize(
    "bad",
    [
        {"speed": 1.5},
        {"speed": 0.5, "mode": "other"},
        {"speed": 0.5, "window": 0},
        {"speed": 0.5, "period": 3},
        {"speed": 0.5, "pseudo": 0.0},
    ],
)
def test_adaptation_checks_its_settings(bad):
    with pytest.raises(ValueError, match="must be"):
        Adaptation(**bad)


def _rng(seed):
    return lambda block: np.random.default_rng([seed, block])


def test_a_static_bot_never_moves(projection):
    bot = AdaptiveLeducBot(BUILDER, TAG, STATIC, _prior(projection))
    start = bot.policy
    session = play_session(lambda seen, hand: EQ, bot, 200, _rng(0))
    assert bot.policy is start
    assert session.bot_final is start
    assert session.values == pytest.approx(head_to_head(EQ, start))


def test_a_family_adapter_stays_in_the_family_and_moves(projection):
    a = Adaptation(0.5, window=100, period=50)
    bot = AdaptiveLeducBot(BUILDER, TAG, a, _prior(projection))
    play_session(lambda seen, hand: EQ, bot, 300, _rng(1))
    assert bot.policy.max_difference(BUILDER.policy(bot.theta)) == 0.0
    assert not np.array_equal(bot.theta, TAG.as_array())


def test_a_tabular_adapter_leaves_the_family(projection):
    a = Adaptation(0.5, window=100, period=50, mode="tabular")
    bot = AdaptiveLeducBot(BUILDER, TAG, a, _prior(projection))
    play_session(lambda seen, hand: EQ, bot, 300, _rng(1))
    assert projection.measure(bot.policy).kl > 1e-3


@pytest.mark.parametrize("mode", ["family", "tabular"])
def test_adapters_punish_a_hero_that_stands_still(projection, mode):
    """Against a fixed best response to a maniac, an adapter wins back much of what it gives up."""
    hero = full_best_response(BUILDER.policy(MANIAC))
    a = Adaptation(0.5, window=200, period=50, mode=mode)
    bot = AdaptiveLeducBot(BUILDER, MANIAC, a, _prior(projection))
    session = play_session(lambda seen, hand: hero, bot, 1000, _rng(2))
    assert session.value(800) < session.value(0, 50) - 0.1


@pytest.mark.parametrize("mode", ["family", "tabular"])
def test_equilibrium_is_never_beaten_while_a_bot_adapts(projection, mode):
    a = Adaptation(0.5, window=100, period=50, mode=mode)
    bot = AdaptiveLeducBot(BUILDER, TAG, a, _prior(projection))
    session = play_session(lambda seen, hand: EQ, bot, 500, _rng(3))
    assert session.values.min() >= -1e-6


def test_the_hero_sees_only_its_redacted_view(projection):
    calls = []

    def hero(seen, hand):
        calls.append(hand)
        assert len(seen) == hand
        for k, o in enumerate(seen):
            assert o.bot_seat == 1 - k % 2
            if o.board is None or o.actions[-1] == "f":
                assert o.bot_card is None
                assert o.bot_sequences == ()
        return EQ

    bot = AdaptiveLeducBot(BUILDER, TAG, Adaptation(0.5, period=50), _prior(projection))
    play_session(hero, bot, 200, _rng(4))
    assert calls == [0, 50, 100, 150]


def test_a_session_repeats_and_scores_exact_values(projection):
    def play():
        bot = AdaptiveLeducBot(BUILDER, TAG, Adaptation(0.5, period=50), _prior(projection))
        return play_session(lambda seen, hand: EQ, bot, 200, _rng(5))

    a, b = play(), play()
    assert np.array_equal(a.values, b.values)
    assert a.value() == pytest.approx(a.values.mean())
    assert a.value(100) == pytest.approx(a.values[2:].mean())
    assert a.hero_exploitability == pytest.approx(exploitability(EQ))


def test_a_session_must_be_whole_blocks(projection):
    bot = AdaptiveLeducBot(BUILDER, TAG, STATIC, _prior(projection))
    with pytest.raises(ValueError, match="whole number"):
        play_session(lambda seen, hand: EQ, bot, 75, _rng(0))


def test_a_prior_centres_on_its_member():
    prior = equilibrium_prior(TAG, 10.0)
    assert isinstance(prior, BetaPrior)
    assert prior.mean == pytest.approx(TAG.as_array())
    assert (prior.a + prior.b) == pytest.approx(10.0)

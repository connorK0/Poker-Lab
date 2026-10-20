"""Archetype bots: hand strength, parameters, populations, ground-truth separation, behavior."""

import dataclasses
import json
import random

import numpy as np
import pytest

from poker_lab.bots import (
    ARCHETYPES,
    PARAM_BOUNDS,
    PARAM_NAMES,
    Archetype,
    BotParams,
    GroundTruth,
    ParametricBot,
    equity_bucket,
    equity_vs_random,
    hand_class,
    make_population,
    mean_params,
    preflop_equity_table,
    preflop_percentile,
    sample_params,
)
from poker_lab.bots.league import LeagueConfig, run_league
from poker_lab.engine import SeatedPlayer, TableSimulator
from poker_lab.profiling.stats import C, counters_by_player, raw_stats
from poker_lab.schema import TableConfig, dumps_hand

TABLE = TableConfig("b", 6, 1, 2)


# ---------------------------------------------------------------- hand strength
def test_preflop_table_matches_known_equities():
    eq = preflop_equity_table()
    assert len(eq) == 169
    assert max(eq, key=eq.get) == "AA"
    assert min(eq, key=eq.get) == "32o"
    assert eq["AA"] == pytest.approx(0.852, abs=0.006)
    assert eq["AKs"] == pytest.approx(0.670, abs=0.006)
    assert eq["72o"] == pytest.approx(0.346, abs=0.006)
    assert eq["AKs"] > eq["AKo"]


def test_hand_class_and_percentiles():
    assert hand_class("Kh", "Ah") == "AKs"
    assert hand_class("7c", "2d") == "72o"
    assert hand_class("9s", "9d") == "99"
    assert preflop_percentile("As", "Ad") == pytest.approx(6 / 1326)
    assert preflop_percentile("3c", "2d") == 1.0
    assert preflop_percentile("Ah", "Kh") < preflop_percentile("Ah", "Kd")


def test_postflop_equity_and_buckets():
    rng = random.Random(0)
    assert equity_vs_random(("As", "Ks"), ("Qs", "Js", "Ts"), rng, 200) == 1.0  # royal flush
    assert equity_vs_random(("7c", "2d"), ("As", "Ks", "Qs"), rng, 400) < 0.25
    assert equity_vs_random(("Ac", "Ad"), (), rng, 2000) == pytest.approx(0.852, abs=0.03)
    assert [equity_bucket(e) for e in (0.3, 0.5, 0.7, 0.9)] == [0, 1, 2, 3]
    assert equity_bucket(0.6) == 1
    assert equity_bucket(0.6, facing_bet=True) == 1
    assert equity_bucket(0.5, facing_bet=True) == 0


# ---------------------------------------------------------------- params & populations
def test_params_validation_and_round_trip():
    p = mean_params(Archetype.TAG)
    assert BotParams.from_dict(p.to_dict()) == p
    assert list(p.to_dict()) == list(PARAM_NAMES)
    with pytest.raises(ValueError, match="outside"):
        dataclasses.replace(p, cbet_freq=1.5)
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(p, bluff_freq=float("nan"))


def test_sampled_params_stay_in_bounds_and_vary():
    rng = np.random.default_rng(1)
    for a in ARCHETYPES:
        draws = np.stack([sample_params(a, rng).to_array() for _ in range(200)])
        lo = np.array([PARAM_BOUNDS[n][0] for n in PARAM_NAMES])
        hi = np.array([PARAM_BOUNDS[n][1] for n in PARAM_NAMES])
        assert (draws >= lo).all()
        assert (draws <= hi).all()
        assert (draws.std(axis=0) > 0.01).all()  # a population, not five fixed bots
        mean = mean_params(a).to_array()
        assert np.abs(draws.mean(axis=0) - mean).max() < 0.08


def test_population_ids_are_opaque_and_shuffled(tmp_path):
    pop = make_population(10, np.random.default_rng(2))
    ids = [p.player_id for p in pop]
    assert len(set(ids)) == 50
    for pid in ids:
        assert not any(a.value in pid for a in ARCHETYPES)
    labels = [p.archetype for p in pop]
    assert labels != sorted(labels)  # not grouped by archetype
    truth = GroundTruth(pop)
    truth.to_json(tmp_path / "truth.json")
    again = GroundTruth.from_json(tmp_path / "truth.json")
    assert again.archetype_of(ids[0]) == pop[0].archetype
    assert again.params_of(ids[3]) == pop[3].params
    assert (again.labels(ids) == truth.labels(ids)).all()
    assert again.param_matrix(ids).shape == (50, len(PARAM_NAMES))


def test_ground_truth_never_appears_in_hand_histories():
    pop = make_population(2, np.random.default_rng(3))[:6]
    sim = TableSimulator(
        TABLE, [SeatedPlayer(i, p.player_id, p.make_agent()) for i, p in enumerate(pop)], seed=3
    )
    text = "\n".join(dumps_hand(h) for h in sim.play(40))
    for p in pop:
        assert p.archetype.value not in text
        for name in PARAM_NAMES:
            assert name not in text
            assert json.dumps(getattr(p.params, name)) not in text


# ---------------------------------------------------------------- behavior
def test_bots_of_every_archetype_only_make_legal_moves():
    rng = np.random.default_rng(4)
    for k in range(3):
        archetypes = [ARCHETYPES[(k + i) % len(ARCHETYPES)] for i in range(6)]
        bots = [ParametricBot(sample_params(a, rng), equity_samples=60) for a in archetypes]
        sim = TableSimulator(
            TABLE,
            [SeatedPlayer(i, f"p{i}", bot) for i, bot in enumerate(bots)],
            seed=k,
            starting_stack={0: 40, 1: 200, 2: 120, 3: 200, 4: 60, 5: 200},
        )
        assert len(list(sim.play(150))) == 150  # IllegalActionError would propagate


def _play_counts(params: BotParams, n_hands: int = 300, seed: int = 0) -> dict[str, float]:
    opponents = [ParametricBot(mean_params(a)) for a in ARCHETYPES]
    players = [SeatedPlayer(0, "hero", ParametricBot(params))] + [
        SeatedPlayer(i + 1, f"o{i}", bot) for i, bot in enumerate(opponents)
    ]
    sim = TableSimulator(TABLE, players, seed=seed)
    return raw_stats(counters_by_player(sim.play(n_hands))["hero"])


def test_parameters_control_the_intended_tendencies():
    base = mean_params(Archetype.TAG)
    tight = _play_counts(dataclasses.replace(base, range_width=0.03))
    loose = _play_counts(dataclasses.replace(base, range_width=0.9, continue_vs_raise=1.0))
    assert tight["vpip"] < 0.1 < 0.5 < loose["vpip"]
    limper = _play_counts(
        dataclasses.replace(base, raise_share=0.0, three_bet_freq=0.0, bluff_freq=0.0,
                            range_width=0.4)
    )  # fmt: skip
    assert limper["pfr"] < 0.05 < limper["vpip"]
    always = _play_counts(dataclasses.replace(base, cbet_freq=1.0, range_width=0.5))
    never = _play_counts(
        dataclasses.replace(base, cbet_freq=0.0, value_aggression=0.0, range_width=0.5)
    )
    assert always["cbet"] == 1.0
    assert never["cbet"] == 0.0


def test_league_is_reproducible_and_independent_of_worker_count():
    pop = make_population(6, np.random.default_rng(5))
    cfg = LeagueConfig(hands_per_player=20, block_hands=10, equity_samples=40)
    a = run_league(pop, cfg, seed=7, workers=1)
    b = run_league(pop, cfg, seed=7, workers=2)
    c = run_league(pop, cfg, seed=8, workers=1)
    assert a.counters.shape == (30, 20, len(C))
    assert (a.counters[:, :, C["hands"]] == 1).all()
    assert np.array_equal(a.counters, b.counters)
    assert not np.array_equal(a.counters, c.counters)
    assert a.hands_simulated == 30 * 20 // 6

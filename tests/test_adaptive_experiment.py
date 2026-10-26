"""The P3.9 experiment's heroes and tasks (``experiments/phase3_adaptive.py``)."""

import numpy as np
import pytest

from experiments.phase3_adaptive import (
    HEROES,
    ClairvoyantHero,
    RecommenderHero,
    adaptations,
    aggregate,
    bot_task,
    predictions,
)
from experiments.phase3_exploit import sample_populations
from poker_lab.exploit.recommend import PushFoldRecommender
from poker_lab.games.push_fold_adaptive import AdaptiveBot, play_session
from poker_lab.games.push_fold_bots import PushFoldBotParams

CFG = {
    "stack": 10,
    "seeds": 2,
    "train_per_archetype": 20,
    "test_per_archetype": 1,
    "hands": 100,
    "bin": 50,
    "period": 25,
    "pseudo": 10,
    "adaptive": {
        "slow": {"speed": 0.1, "half_life": 200},
        "fast": {"speed": 0.5, "half_life": 50},
        "calls_only": {"speed": 0.5, "half_life": 50, "adapts": ["call"]},
    },
    "observe_hands": 50,
    "refit_every": 50,
    "window": 50,
    "check_sizes": [50],
    "check_draws": 50,
    "ps": [0.5],
    "level": 0.9,
    "draws": 200,
}
THRESHOLDS = {50: {"shown_hands": 0.01, "stability": 0.01}}


@pytest.fixture(scope="module")
def setup():
    prior, bots = sample_populations(5, CFG)
    rec = PushFoldRecommender(10.0, prior, ps=[0.5], draws=200, seed=0)
    return prior, bots, rec


def test_refitting_heroes_decide_on_their_schedule(setup):
    _, _, rec = setup
    bot = AdaptiveBot(
        PushFoldBotParams(0.2, 0.1, 0.03), adaptations(CFG)["fast"], 10.0, rec.equilibrium
    )
    refit = RecommenderHero(rec, every=50, seed=[1])
    play_session(refit, bot, 160, np.random.default_rng(0))
    assert [r["hand"] for r in refit.records] == [0, 50, 100, 150]
    once = RecommenderHero(rec, every=50, start=50, once=True, seed=[1])
    session = play_session(
        once,
        AdaptiveBot(bot.history[0], bot.adaptation, 10.0, rec.equilibrium),
        160,
        np.random.default_rng(0),
    )
    assert [r["hand"] for r in once.records] == [50]
    np.testing.assert_array_equal(session.hero_push[0], rec.equilibrium[0])  # watching
    assert all(np.array_equal(p, session.hero_push[2]) for p in session.hero_push[2:])


def test_a_window_limits_the_hands_used(setup):
    _, _, rec = setup
    seen = []

    class Spy:
        def recommend(self, data, seed=None):
            seen.append(data.n_hands)
            return rec.recommend(data, seed=seed)

        equilibrium = rec.equilibrium

    hero = RecommenderHero(Spy(), every=50, window=60, seed=[2])
    bot = AdaptiveBot(
        PushFoldBotParams(0.5, 0.3, 0.03), adaptations(CFG)["slow"], 10.0, rec.equilibrium
    )
    play_session(hero, bot, 200, np.random.default_rng(1))
    assert seen == [0, 50, 60, 60]


def test_the_clairvoyant_reads_the_bot(setup):
    _, _, rec = setup
    bot = AdaptiveBot(
        PushFoldBotParams(0.2, 0.1, 0.03), adaptations(CFG)["static"], 10.0, rec.equilibrium
    )
    push, _ = ClairvoyantHero(bot)(None, 0)
    assert push.mean() > 0.9  # a nit's calls are rare: push almost everything


def test_tasks_aggregate_into_predictions(setup):
    prior, bots, _ = setup
    per_seed = []
    for seed in (5, 6):
        recs = []
        for _, k, archetype, params in [b for b in bots if b[0] == "standard"][:2]:
            for kind in adaptations(CFG):
                recs.append(bot_task(seed, k, archetype, params, kind, prior, THRESHOLDS, CFG))
        per_seed.append(recs)
    r = per_seed[0][0]
    assert set(r["heroes"]) == set(HEROES)
    assert r["heroes"]["equilibrium"]["exploitability"] == pytest.approx(0.0, abs=1e-6)
    agg = aggregate(per_seed, CFG)
    assert agg["static"]["heroes"]["equilibrium"]["gain"]["mean"] == 0.0
    assert agg["static"]["width_moved_vs_equilibrium"] == 0.0
    pred = predictions(agg)
    assert set(pred) == {"A1", "B1", "B2", "C1", "C2", "C3", "C4", "D1"}
    assert all(isinstance(v["holds"], bool) for v in pred.values())

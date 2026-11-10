"""P2.12's experiment: the control repeats P2.7, heroes respect the firewall, the summary works."""

import json

import numpy as np
import pytest

from experiments import phase2_exploit, phase2_misspec
from experiments.common import load_config
from poker_lab.games.leduc_bots import (
    ARCHETYPE_MEANS,
    BetaPrior,
    LeducBot,
    Observation,
    sample_population,
)
from poker_lab.games.leduc_misspec import RULES, AdaptiveLeducBot, simulate_views

SMALL = {
    "seeds": 2,
    "train_per_archetype": 4,
    "test_per_archetype": 1,
    "sizes": [10, 40],
    "ps": [0.5, 0.9],
    "dbr_pseudocount": 10,
    "session": {
        "hands": 200,
        "period": 50,
        "pseudo": 10,
        "refit_every": 100,
        "fit_once_at": 100,
        "recent_window": 100,
        "rnr_p": 0.9,
        "rnr_p_low": 0.5,
    },
    "adaptive": {
        "family_fast": {"speed": 0.5, "window": 100, "mode": "family"},
        "tabular_fast": {"speed": 0.5, "window": 100, "mode": "tabular"},
    },
}
BOT = LeducBot("b000", "tag", ARCHETYPE_MEANS["tag"])


@pytest.fixture(scope="module")
def prior():
    rng = np.random.default_rng(1)
    train = sample_population(4, rng, prefix="t")
    return BetaPrior.fit(np.array([b.params.as_array() for b in train]))


def test_the_control_repeats_p27s_bot_task_exactly(prior):
    cfg = {"sizes": [10, 40], "ps": [0.5, 0.9], "dbr_pseudocount": 10}
    new = phase2_misspec.static_task(77, BOT, "family", prior, cfg, 3)
    old = phase2_exploit.bot_task(77, BOT, prior, cfg, 3)
    assert new["eq_value"] == old["eq_value"]
    for n in ("10", "40"):
        for m in ("br", "rnr_0.5", "rnr_0.9", "dbr"):
            assert new["sizes"][n][m] == old["sizes"][n]["scores"][m]
    assert new["oracle"]["br"] == old["oracle"]["br"]
    assert new["population"]["rnr_0.9"] == old["population_model"]["rnr_0.9"]
    assert new["kl"] < 1e-6


@pytest.mark.parametrize("kind", ["street", "noisy"])
def test_a_rule_opponent_has_a_gap_the_control_lacks(prior, kind):
    cfg = {"sizes": [40], "ps": [0.9], "dbr_pseudocount": 10}
    r = phase2_misspec.static_task(77, BOT, kind, prior, cfg, 0)
    assert r["kl"] > 1e-3
    assert r["oracle"]["br"]["gain"] >= r["limit"]["br"]["gain"] - 1e-9


def test_nothing_beats_the_equilibrium_opponent(prior):
    cfg = {"sizes": [40], "ps": [0.9], "dbr_pseudocount": 10}
    r = phase2_misspec.static_task(77, BOT, "equilibrium", prior, cfg, 0)
    assert r["eq_value"] == pytest.approx(0.0, abs=1e-6)
    assert r["oracle"]["br"]["gain"] == pytest.approx(0.0, abs=1e-6)
    assert r["sizes"]["40"]["br"]["gain"] <= 1e-6


def test_opponents_depend_only_on_their_seed_and_index():
    a = phase2_misspec.opponent_policy("noisy", BOT, 5, 2)
    assert np.array_equal(a.probs, phase2_misspec.opponent_policy("noisy", BOT, 5, 2).probs)
    assert not np.array_equal(a.probs, phase2_misspec.opponent_policy("noisy", BOT, 5, 3).probs)
    assert set(RULES) < set(phase2_misspec.STATIC_KINDS)


def _ctx():
    return phase2_misspec._context()


def _hands(n, seed=0):
    ctx = _ctx()
    view, _ = simulate_views(
        ctx["tree"], ctx["game"], ctx["eq"], ctx["eq"], n, np.random.default_rng(seed)
    )
    return view


def test_fit_once_decides_once_at_its_hand(prior):
    hero = phase2_misspec.FitHero(prior, None, every=1, start=100, once=True)
    eq = _ctx()["eq"]
    seen = _hands(150)
    assert hero([], 0) is eq
    assert hero(seen[:50], 50) is eq
    fitted = hero(seen[:100], 100)
    assert fitted is not eq
    assert hero(seen, 150) is fitted


def test_refit_decides_on_its_schedule_from_the_first_hand(prior):
    hero = phase2_misspec.FitHero(prior, 0.9, every=100)
    first = hero([], 0)
    assert first is not _ctx()["eq"]  # no hands yet: the population model's response
    seen = _hands(100)
    assert hero(seen[:50], 50) is first
    assert hero(seen, 100) is not first


def test_a_recent_window_ignores_older_hands(prior):
    seen = _hands(200)
    a = phase2_misspec.FitHero(prior, None, every=100, window=100)(seen, 200)
    b = phase2_misspec.FitHero(prior, None, every=100)(seen[100:], 200)
    assert np.array_equal(a.probs, b.probs)


def test_only_the_clairvoyant_reads_the_bot(prior):
    ctx = _ctx()
    cfg = SMALL["session"]
    adaptation = phase2_misspec.adaptations(SMALL)["family_fast"]
    bot = AdaptiveLeducBot(ctx["builder"], BOT.params, adaptation, prior)
    for name in phase2_misspec.SESSION_HEROES:
        hero = phase2_misspec.make_hero(name, prior, cfg, bot)
        if name not in ("equilibrium", "clairvoyant"):
            assert isinstance(hero, phase2_misspec.FitHero)
    seen: list[Observation] = []
    clairvoyant = phase2_misspec.make_hero("clairvoyant", prior, cfg, bot)
    assert clairvoyant(seen, 0).max_difference(
        phase2_misspec.full_best_response(bot.policy)
    ) == pytest.approx(0.0)


def test_the_share_is_a_wilson_interval_with_its_count():
    q = phase2_misspec._share([True, False, False, False])
    assert q["share"] == pytest.approx(0.25)
    assert q["n"] == 4
    assert q["lower"] < 0.25 < q["upper"]


def test_spearman():
    x = np.arange(5.0)
    assert phase2_misspec._spearman(x, x**3) == pytest.approx(1.0)
    assert phase2_misspec._spearman(x, -x) == pytest.approx(-1.0)


@pytest.mark.long
def test_a_small_run_end_to_end(tmp_path):
    results = phase2_misspec.run(98, cfg=SMALL, workers=2, out_dir=tmp_path)
    saved = json.loads((tmp_path / "misspec.json").read_text(encoding="utf-8"))
    assert set(saved["static"]) == {*phase2_misspec.STATIC_KINDS, "kl_shortfall_spearman"}
    assert set(saved["sessions"]) == {"static", "family_fast", "tabular_fast"}
    assert saved["committed_check"] is None  # only the committed seed can be compared
    assert (tmp_path / "misspec_summary.md").read_text(encoding="utf-8").startswith("# P2.12")
    assert (tmp_path / "misspec_by_hands.png").exists()
    assert (tmp_path / "misspec_sessions-dark.png").exists()
    # the equilibrium hero's gain over itself is exactly zero, in every kind
    for entry in results["sessions"].values():
        assert entry["heroes"]["equilibrium"]["gain"]["mean"] == 0.0
    assert len(saved["predictions"]) == len(phase2_misspec.PREDICTIONS)


def test_the_full_config_is_the_p27_population():
    cfg = load_config(phase2_misspec.CONFIG)
    assert cfg["misspec"]["seeds"] == cfg["exploit"]["seeds"]
    for key in ("train_per_archetype", "test_per_archetype", "sizes", "dbr_pseudocount"):
        assert cfg["misspec"][key] == cfg["exploit"][key]

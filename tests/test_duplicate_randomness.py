"""P2.11c: the switches on the bots' randomness, the variance split and a tiny run. With every
source live a ``SwitchedBot`` must be the default bot draw for draw."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments import phase2_duplicate as pd
from experiments import phase2_randomness as pr
from experiments.common import derive_seeds
from poker_lab.bots.archetypes import ARCHETYPES
from poker_lab.bots.population import make_population

TINY = {
    "replicates": 1,
    "population_per_archetype": 6,
    "lineups_per_replicate": 2,
    "deals": 12,
    "chunk_deals": 6,
    "equity_samples": 10,
    "half_width": 5,
    "bootstrap": 50,
    "workers": 1,
}


def _lineups(cfg=TINY, seed=98):
    return pr.plan(seed, cfg)


def _seats(arch_index: int = 1):
    bot = next(
        p
        for p in make_population(6, np.random.default_rng(1))
        if p.archetype == ARCHETYPES[arch_index]
    )
    return tuple((f"{bot.player_id}-c{j}", bot) for j in range(pr.N))


def test_plan_matches_the_committed_experiments_clones():
    cfg = {**TINY, "lineups": {"clones_6": TINY["lineups_per_replicate"]}}
    seed = derive_seeds(98, 1)[0]
    assert _lineups() == pd.replicate_lineups(0, seed, cfg)


def test_baseline_is_the_default_bot():
    seats = _seats()
    chips, _ = pr.chunk_task(seats, 12, 5, 10, "baseline")
    np.testing.assert_array_equal(chips, pd.chunk_task(seats, 12, 5, 10))


def test_all_live_switched_bot_is_the_default_bot_draw_for_draw(monkeypatch):
    seats = _seats()
    expected = pd.chunk_task(seats, 12, 5, 10)
    monkeypatch.setitem(pr.CONDITIONS, "all_live", pr.Modes())
    chips, _ = pr.chunk_task(seats, 12, 5, 10, "all_live")
    np.testing.assert_array_equal(chips, expected)


@pytest.mark.parametrize("condition", pr.EXACT_CHECKS)
@pytest.mark.parametrize("arch_index", range(len(ARCHETYPES)))
def test_no_randomness_between_tables_cancels_exactly(condition, arch_index):
    chips, _ = pr.chunk_task(_seats(arch_index), 40, 7, 10, condition)
    assert (chips.sum(axis=0) == 0).all()
    assert chips.any()  # the hands did move chips


def test_all_shared_with_the_history_key_also_cancels_exactly(monkeypatch):
    modes = pr._shared_modes((), "history")
    monkeypatch.setitem(pr.CONDITIONS, "history_all", modes)
    chips, _ = pr.chunk_task(_seats(), 40, 7, 10, "history_all")
    assert (chips.sum(axis=0) == 0).all()
    assert chips.any()


def test_history_key_pairs_tables_only_on_the_same_history():
    from poker_lab.schema import Action, ActionType, Street

    bot = pr.SwitchedBot(
        _seats()[0][1].params, modes=pr.Modes(key="history"), salt=1, equity_samples=10
    )

    class Public:
        hand_id = "x-000001"
        street = Street.FLOP

        def __init__(self, amount):
            self.actions = (Action(0, Street.PREFLOP, ActionType.RAISE, amount, amount),)

    class View:
        seat = 2

        def __init__(self, amount):
            self.public = Public(amount)

    assert bot._uniform(View(10), "mixed") == bot._uniform(View(10), "mixed")
    assert bot._uniform(View(10), "mixed") != bot._uniform(View(11), "mixed")


@pytest.mark.parametrize("condition", ["only_mixed", "only_noise", "only_equity", "off_mixed"])
def test_any_source_left_live_breaks_exact_cancellation(condition):
    per_deal = pr.chunk_task(_seats(1), 80, 3, 10, condition)[0].sum(axis=0)
    assert (per_deal != 0).any()


def test_shared_draws_depend_on_deal_seat_decision_and_source_not_table():
    bot = pr.SwitchedBot(_seats()[0][1].params, modes=pr.Modes(), salt=1, equity_samples=10)

    class Public:
        def __init__(self, hand_id, street, actions=()):
            self.hand_id, self.street, self.actions = hand_id, street, actions

    class View:
        def __init__(self, public, seat):
            self.public, self.seat = public, seat

    from poker_lab.schema import Street

    u = bot._uniform(View(Public("dup6-dup1-000007", Street.FLOP), 2), "mixed")
    assert u == bot._uniform(View(Public("dup6-dup5-000007", Street.FLOP), 2), "mixed")
    others = [
        bot._uniform(View(Public("dup6-dup1-000008", Street.FLOP), 2), "mixed"),
        bot._uniform(View(Public("dup6-dup1-000007", Street.TURN), 2), "mixed"),
        bot._uniform(View(Public("dup6-dup1-000007", Street.FLOP), 3), "mixed"),
        bot._uniform(View(Public("dup6-dup1-000007", Street.FLOP), 2), "noise"),
    ]
    assert u not in others
    assert all(0.0 < v < 1.0 for v in [u, *others])


def test_shared_uniforms_are_uniform():
    bot = pr.SwitchedBot(_seats()[0][1].params, modes=pr.Modes(), salt=2, equity_samples=10)
    from poker_lab.schema import Street

    class Public:
        street = Street.PREFLOP
        actions = ()

        def __init__(self, i):
            self.hand_id = f"x-{i:06d}"

    class View:
        seat = 0

        def __init__(self, i):
            self.public = Public(i)

    u = np.array([bot._uniform(View(i), "mixed") for i in range(20_000)])
    assert u.mean() == pytest.approx(0.5, abs=0.01)
    assert np.mean(u < 0.25) == pytest.approx(0.25, abs=0.015)


def test_sharing_a_source_leaves_the_bots_play_as_it_was():
    """Each table's distribution is the default's: the stats match within sampling error."""
    seats = _seats()
    from poker_lab.profiling.stats import raw_stats

    def stats(condition):
        return raw_stats(sum(pr.chunk_task(seats, 150, s, 10, condition)[1] for s in (11, 12)))

    base, shared = stats("baseline"), stats("shared_mixed")
    for key in ("vpip", "pfr", "wtsd"):
        assert shared[key] == pytest.approx(base[key], abs=0.04)


def test_within_deal_variance_reads_seats_not_players():
    """A seat's results are the same at every table when each player's rotate with its seat."""
    y = np.random.default_rng(0).normal(size=(5, 6))  # (deals, seats)
    x = np.stack([np.stack([y[:, (j + r) % 6] for j in range(6)], axis=1) for r in range(6)])
    assert pr.within_deal_variance(x) == pytest.approx(0.0, abs=1e-20)
    noisy = x + np.random.default_rng(1).normal(size=x.shape)
    assert pr.within_deal_variance(noisy) == pytest.approx(1.0, abs=0.3)


def _additive(mixed, noise, equity, mx=0.0, all3=0.0):
    """Remaining shares built from known terms: R(S) sums the terms meeting the live set S."""
    terms = {
        frozenset({"mixed"}): mixed,
        frozenset({"noise"}): noise,
        frozenset({"equity"}): equity,
        frozenset({"mixed", "equity"}): mx,
        frozenset({"mixed", "noise", "equity"}): all3,
    }
    return {
        name: np.float64(sum(v for b, v in terms.items() if b & set(live)))
        for name, live in pr.DECOMPOSITION.items()
    }


def test_decompose_recovers_known_terms():
    t = pr.decompose(_additive(0.4, 0.02, 0.1, mx=0.12, all3=0.03))
    assert t["mixed"] == pytest.approx(0.4)
    assert t["noise"] == pytest.approx(0.02)
    assert t["equity"] == pytest.approx(0.1)
    assert t["mixed+equity"] == pytest.approx(0.12)
    assert t["mixed+noise"] == pytest.approx(0.0, abs=1e-12)
    assert t["noise+equity"] == pytest.approx(0.0, abs=1e-12)
    assert t["mixed+noise+equity"] == pytest.approx(0.03)
    assert t["shapley_mixed"] == pytest.approx(0.4 + 0.06 + 0.01)
    assert t["shapley_equity"] == pytest.approx(0.1 + 0.06 + 0.01)
    assert t["shapley_noise"] == pytest.approx(0.02 + 0.01)


def test_decompose_terms_and_shapley_each_sum_to_the_baseline():
    rng = np.random.default_rng(3)
    remaining = {name: rng.random(4) for name in pr.DECOMPOSITION}
    remaining["shared_all"] = np.zeros(4)
    t = pr.decompose(remaining)
    terms = sum(t[k] for k in pr.TERM_LABELS)
    shapley = sum(t[f"shapley_{s}"] for s in pr.SOURCES)
    np.testing.assert_allclose(terms, remaining["baseline"])
    np.testing.assert_allclose(shapley, remaining["baseline"])


@pytest.mark.long
def test_tiny_run_end_to_end_same_with_one_worker_and_two(tmp_path):
    one = pr.run(98, cfg=TINY, workers=1, out_dir=tmp_path / "one")
    two = pr.run(98, cfg=TINY, workers=2, out_dir=tmp_path / "two")
    # repr: a stat with no opportunities in so few hands is NaN, which != itself
    assert repr(one["per_lineup"]) == repr(two["per_lineup"])
    assert repr(one["aggregates"]) == repr(two["aggregates"])
    assert one["committed_check"] is None  # another seed and size than P2.11b's run
    for condition in pr.EXACT_CHECKS:
        assert one["aggregates"]["by_condition"][condition]["max_abs_deal_sum_chips"] == 0
    assert (tmp_path / "one" / "randomness_summary.md").exists()
    assert (tmp_path / "one" / "randomness.json").exists()
    assert math.isfinite(one["aggregates"]["decomposition"]["cancelled"]["mean"])
    # every prediction's path exists; only the comparison with the committed run has no value here
    missing = [
        key for key, p in one["predictions"].items() if any(v is None for v in p["values"].values())
    ]
    assert "A1" in missing
    # two lineups cover two archetypes, and a stat may have no opportunities in so few hands
    assert not set(missing) - {"A1", "D6", "E1a", "E1b", "E2a", "E2b", "E3b"}

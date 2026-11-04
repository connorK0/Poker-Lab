"""P2.11b's experiment: duplicate dealing for n players at no-limit hold'em. The metric on
synthetic arrays, the lineups and their seeds, the worker task against the library, and a tiny
run end to end (the same results with one worker and two)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from experiments import phase2_duplicate as pd
from experiments.common import _jsonable
from experiments.compare_runs import compare_step
from poker_lab.bots.archetypes import ARCHETYPES
from poker_lab.engine.duplicate import duplicate_match
from poker_lab.evaluation import WinRateEvaluator, hand_results
from poker_lab.schema import TableConfig

#: Small enough to run in seconds: n = 6, 3 and 2, two replicates, two chunks per lineup.
TINY = {
    "replicates": 2,
    "population_per_archetype": 6,
    "lineups": {"clones_6": 2, "mirror_6": 1, "mixed_6": 1, "mixed_3": 2, "mixed_2": 2},
    "deals": 8,
    "chunk_deals": 4,
    "calling_deals": 6,
    "equity_samples": 10,
    "half_width": 5,
    "bootstrap": 50,
    "workers": 2,
}


# ---------------------------------------------------------------------------- the metric


def _identity(s: dict, n: int, d: int) -> float:
    """The ratio from rho: n (D - 1) / ((n D - 1) (1 + (n - 1) rho))."""
    return n * (d - 1) / ((n * d - 1) * (1 + (n - 1) * s["rho"]))


@pytest.mark.parametrize("n", [2, 3, 6])
def test_identical_results_at_every_table_give_rho_1_and_ratio_1_over_n(n):
    d = 4000
    y = np.random.default_rng(n).normal(size=d)
    s = pd.player_stats(np.tile(y, (n, 1)), half_width=5)
    assert s["rho"] == pytest.approx(1.0)
    assert s["rho_by_offset"] == pytest.approx([1.0] * (n // 2))
    assert s["ratio"] == pytest.approx((d - 1) / (n * d - 1))  # 1/n up to degrees of freedom
    assert s["ratio"] == pytest.approx(1 / n, rel=1e-3)
    assert s["s_u"] == pytest.approx(s["sigma"], rel=1e-3)
    assert not s["no_variance_left"]


@pytest.mark.parametrize("n", [2, 3, 6])
def test_perfectly_cancelling_results_leave_no_variance(n):
    """Each deal's mean is 0 but every hand is not: s_u = 0, the ratio is RATIO_CAP with the
    flag, rho = -1/(n - 1) and duplicate dealing needs no hands."""
    x = np.random.default_rng(n).normal(size=(n, 500))
    x -= x.mean(axis=0)
    s = pd.player_stats(x, half_width=5)
    assert s["s_u"] < pd.ZERO_SD
    assert s["no_variance_left"]
    assert s["ratio"] == pd.RATIO_CAP
    assert s["rho"] == pytest.approx(-1 / (n - 1))
    assert s["hands_needed"]["duplicate"] == 0
    assert s["hands_needed"]["naive"] > 0
    json.dumps(s)  # JSON-safe: no infinities


def test_independent_results_give_a_ratio_near_1():
    x = np.random.default_rng(0).normal(scale=10.0, size=(6, 20000))
    s = pd.player_stats(x, half_width=5)
    assert 0.95 < s["ratio"] < 1.05
    assert abs(s["rho"]) < 0.01
    assert s["sigma"] == pytest.approx(10.0, rel=0.01)
    assert s["ratio"] == pytest.approx(_identity(s, 6, 20000))


def test_correlation_by_offset_on_a_constructed_example():
    """x_r = e_r + e_(r+1): neighbours share one term (corr 0.5), tables two or three seats
    apart share none, so rho = (0.5 + 0.5) / 5 = 0.2 and the ratio is 1 / (1 + 5 rho) = 0.5.
    Each offset is the pooled Pearson correlation of the pairs (x[r, i], x[(r + k) % n, i])."""
    n, d = 6, 40000
    e = np.random.default_rng(1).normal(size=(n, d))
    x = e + np.roll(e, -1, axis=0)
    s = pd.player_stats(x, half_width=5)
    assert len(s["rho_by_offset"]) == 3
    assert s["rho_by_offset"] == pytest.approx([0.5, 0.0, 0.0], abs=0.015)
    assert s["rho"] == pytest.approx(0.2, abs=0.01)
    assert s["ratio"] == pytest.approx(0.5, rel=0.05)
    assert s["ratio"] == pytest.approx(_identity(s, n, d))
    for k in (1, 2, 3):
        pooled = np.corrcoef(x.ravel(), np.roll(x, -k, axis=0).ravel())[0, 1]
        assert s["rho_by_offset"][k - 1] == pytest.approx(pooled, abs=1e-12)
    # an exact pattern: x[r] = +a at even tables, -a at odd ones
    a = np.arange(1.0, 6.0)
    alt = np.stack([a if r % 2 == 0 else -a for r in range(4)])
    s = pd.player_stats(alt, half_width=5)
    assert s["rho_by_offset"] == pytest.approx([-1.0, 1.0])
    assert s["rho"] == pytest.approx(-1 / 3)
    assert s["no_variance_left"]


def test_hands_needed_follow_the_formulas():
    x = np.random.default_rng(2).normal(scale=3.0, size=(3, 1000))
    s = pd.player_stats(x, half_width=4)
    h = 4 / 100
    assert s["hands_needed"]["naive"] == math.ceil((1.96 * s["sigma"] / h) ** 2)
    assert s["hands_needed"]["duplicate"] == math.ceil(3 * (1.96 * s["s_u"] / h) ** 2)
    assert s["hands_needed"]["naive"] / s["hands_needed"]["duplicate"] == pytest.approx(
        s["ratio"], rel=1e-3
    )


def test_median_ci_resamples_lineups_and_drops_empty_ones():
    values = [float(v) for v in range(1, 11)]
    groups = [[v, 1.1 * v] for v in values] + [[]]
    ci = pd.median_ci(groups, 200, [1, 2])
    assert ci["lineups"] == 10
    assert 1.0 <= ci["lower"] <= ci["upper"] <= 11.0
    assert ci == pd.median_ci(groups, 200, [1, 2])  # seeded
    # a lineup's players stay together: a pair of equal values counts like one value, as in a
    # heads-up lineup, whose two players have the same ratio
    pairs = pd.median_ci([[v, v] for v in values], 500, [1])
    assert pairs == pd.median_ci([[v] for v in values], 500, [1])


def test_median_ci_needs_enough_lineups():
    few = [[float(v), float(v)] for v in range(pd.MIN_CI_LINEUPS - 1)]
    ci = pd.median_ci([*few, []], 200, [1])
    assert math.isnan(ci["lower"])
    assert math.isnan(ci["upper"])
    assert ci["lineups"] == pd.MIN_CI_LINEUPS - 1
    assert not math.isnan(pd.median_ci([*few, [9.0]], 200, [1])["lower"])


def test_predictions_follow_the_table():
    r = {"aggregates": {"a": {"x": 2.0, "y": [0.5, -0.1]}}}
    table = {
        "P1": {"claim": "x", "values": [("aggregates", "a", "x")], "test": (">=", 1.5)},
        "P2": {
            "claim": "y",
            "values": [("aggregates", "a", "y", 0), ("aggregates", "a", "y", 1)],
            "test": ("between", [-0.2, 0.6]),
        },
        "P3": {"claim": "fails", "values": [("aggregates", "a", "x")], "test": ("<=", 1.0)},
        "P4": {"claim": "missing", "values": [("aggregates", "b")], "test": (">=", 0)},
        "P5": {"claim": "a dict", "values": [("aggregates", "a")], "test": (">=", 0)},
    }
    out = pd.predictions(r, table)
    passed = [out[k]["passed"] for k in ("P1", "P2", "P3", "P4", "P5")]
    assert passed == [True, True, False, False, False]
    assert out["P5"]["values"] == {"aggregates.a": None}  # a path that stops short, not a crash
    assert out["P2"]["values"] == {"aggregates.a.y.0": 0.5, "aggregates.a.y.1": -0.1}
    assert out["P4"]["values"] == {"aggregates.b": None}
    assert out["P1"]["test"] == ">="
    assert out["P1"]["threshold"] == 1.5


@pytest.mark.parametrize(
    ("values", "test", "holds"),
    [
        ([3.0, 2.0, 1.0], ("decreasing", None), True),
        ([3.0, 3.0, 1.0], ("decreasing", None), False),
        ([1.0, 2.0, 3.0], ("first_lowest", None), True),
        ([2.0, 1.0, 3.0], ("first_lowest", None), False),
        ([1.0, 1.0, 3.0], ("first_lowest", None), False),
        ([-0.05, -0.03, -0.04], ("range_at_most", 0.03), True),
        ([-0.08, -0.03, -0.04], ("range_at_most", 0.03), False),
        ([1.0, None, 3.0], ("decreasing", None), False),
        ([1.0, None], ("range_at_most", 10.0), False),
        ([], (">=", 0.0), False),
    ],
)
def test_predictions_on_several_quantities_together(values, test, holds):
    assert pd._holds(values, test) is holds


def test_an_unknown_prediction_test_is_refused():
    with pytest.raises(ValueError, match="unknown test"):
        pd._holds([1.0], ("sideways", None))


def test_every_prediction_reads_numbers_from_the_results():
    """Run on made-up results of the shape the real run gives: no path may stop short or miss,
    which would only show after the whole simulation."""
    cfg = {
        **TINY,
        "lineups": {"clones_6": 5, "mirror_6": 5, "mixed_6": 10, "mixed_3": 3, "mixed_2": 3},
    }
    _, calling, lineups = pd.plan(98, cfg)
    rng = np.random.default_rng(0)
    rows = [
        pd.lineup_row(line, rng.integers(-50, 50, size=(line.n, 20, line.n), dtype=np.int32), 5)
        for line in lineups
    ]
    r = {
        "aggregates": pd.aggregate(rows, 20, 98),
        "calling_check": pd.calling_check(calling, np.zeros((6, 4, 6), dtype=np.int32)),
    }
    out = pd.predictions(r)
    assert set(out) == set(pd.PREDICTIONS)
    for key, p in out.items():
        assert None not in p["values"].values(), key
    assert out["A1"]["passed"]


# ---------------------------------------------------------------------- lineups and seeds


def test_lineups_are_drawn_as_specified():
    cfg = {
        **TINY,
        "lineups": {"clones_6": 5, "mirror_6": 5, "mixed_6": 2, "mixed_3": 2, "mixed_2": 3},
    }
    seeds, calling, lineups = pd.plan(98, cfg)
    assert len(seeds) == 2
    assert calling.kind == "calling_6"
    assert all(spec is None for _, spec in calling.seats)
    for r in range(2):
        mine = [x for x in lineups if x.replicate == r]
        assert [x.kind for x in mine] == [k for k in pd.KINDS for _ in range(cfg["lineups"][k])]
        for x in mine:
            ids = [pid for pid, _ in x.seats]
            assert len(set(ids)) == x.n == pd.TABLE_SIZE[x.kind]
            assert x.deals == cfg["deals"]
            if x.kind == "clones_6":
                (bot,) = {spec for _, spec in x.seats}
                assert ids == [f"{bot.player_id}-c{j}" for j in range(6)]
                assert x.archetype == ARCHETYPES[x.index].value == bot.archetype.value
            elif x.kind == "mirror_6":
                assert len({spec.player_id for _, spec in x.seats}) == 6
                assert {spec.archetype.value for _, spec in x.seats} == {x.archetype}
                assert x.archetype == ARCHETYPES[x.index].value
            else:
                assert x.archetype is None
                assert ids == [spec.player_id for _, spec in x.seats]
    # the two replicates have different populations
    bots = [
        {spec.player_id for x in lineups if x.replicate == r for _, spec in x.seats}
        for r in range(2)
    ]
    assert not bots[0] & bots[1]


def test_chunks_get_distinct_seeds_and_the_plan_is_reproducible():
    cfg = {**TINY, "deals": 10, "chunk_deals": 4}
    seeds, calling, lineups = pd.plan(99, cfg)
    assert [d for _, d in lineups[0].chunks] == [4, 4, 2]
    every = [s for x in [calling, *lineups] for s, _ in x.chunks]
    assert len(set(every)) == len(every)
    again = pd.plan(99, cfg)
    assert again == (seeds, calling, lineups)
    assert pd.plan(98, cfg)[0] != seeds
    # a kind's lineups do not depend on how many lineups the other kinds have
    fewer = pd.plan(99, {**cfg, "lineups": {**cfg["lineups"], "clones_6": 0, "mixed_3": 1}})[2]
    for kind in ("mirror_6", "mixed_6", "mixed_2"):
        assert [x for x in fewer if x.kind == kind] == [x for x in lineups if x.kind == kind]


def test_bad_configs_are_refused():
    with pytest.raises(ValueError, match="unknown lineup kinds"):
        pd.plan(1, {**TINY, "lineups": {"mixed_4": 1}})
    with pytest.raises(ValueError, match="at least 6 bots"):
        pd.plan(1, {**TINY, "population_per_archetype": 5})
    with pytest.raises(ValueError, match="one lineup per archetype"):
        pd.plan(1, {**TINY, "lineups": {"clones_6": 6}})


def test_a_chunk_is_a_duplicate_match_in_lineup_order():
    """The task's array holds every hand's net chips (table r, deal i, player j); the win rates
    built from it match the library's ``DuplicateMatch.evaluate`` and the naive evaluator; the
    same seed reproduces the bytes."""
    _, _, lineups = pd.plan(5, TINY)
    line = next(x for x in lineups if x.kind == "mixed_3")
    seed, deals = line.chunks[0]
    chips = pd.chunk_task(line.seats, deals, seed, 10)
    assert chips.shape == (3, deals, 3)
    assert chips.dtype == np.int32
    assert pd.chunk_task(line.seats, deals, seed, 10).tobytes() == chips.tobytes()
    players = [
        (pid, lambda spec=spec: spec.make_agent(equity_samples=10)) for pid, spec in line.seats
    ]
    match = duplicate_match(TableConfig("dup3", 3, 1, 2), players, deals, seed)
    assert (chips.sum(axis=2) == 0).all()  # no rake: every hand sums to zero
    for r, table in enumerate(match.tables):
        for i, hand in enumerate(table):
            for j, (pid, _) in enumerate(line.seats):
                assert chips[r, i, j] == hand.net_results()[hand.seat_of(pid)]
    for j, (pid, _) in enumerate(line.seats):
        w = pd.winrates(chips[:, :, j], pid)
        dup = match.evaluate(pid)
        assert w["duplicate"] == [dup.lower, dup.upper]
        naive = WinRateEvaluator().evaluate([h for t in match.tables for h in hand_results(t, pid)])
        assert w["naive"] == pytest.approx([naive.lower, naive.upper], abs=1e-9)
        assert w["bb_per_100"] == pytest.approx(dup.bb_per_100, abs=1e-9)


def test_calling_agents_cancel_exactly():
    _, calling, _ = pd.plan(7, TINY)
    chips = np.concatenate([pd.chunk_task(calling.seats, d, s, 10) for s, d in calling.chunks], 1)
    check = pd.calling_check(calling, chips)
    assert check["all_zero"]
    assert check["max_abs_deal_mean_chips"] == 0.0
    assert check["sigma_median"] > 0  # the single hands are not zero
    broken = chips.copy()
    broken[0, 0, 0] += 6
    assert not pd.calling_check(calling, broken)["all_zero"]


# ---------------------------------------------------------------------------- the wiring


def test_run_phase2_runs_duplicate_last(monkeypatch, tmp_path):
    from experiments import run_phase2

    assert run_phase2.STEPS == ("cfr", "deep_cfr", "exploit", "aivat", "duplicate")
    seen = {}

    def fake(seed, *, cfg, workers, out_dir):
        seen.update(seed=seed, cfg=cfg, workers=workers, out_dir=out_dir)

    monkeypatch.setattr(pd, "run", fake)
    run_phase2.run(
        98, {"name": "x", "duplicate": {"workers": 3}}, steps=("duplicate",), out_dir=tmp_path
    )
    assert seen == {
        "seed": 98,
        "cfg": {"workers": 3},
        "workers": 3,
        "out_dir": tmp_path / "duplicate",
    }


def test_the_configs_have_a_duplicate_section():
    from experiments.common import load_config

    for name in ("phase2.yaml", "phase2_quick.yaml"):
        cfg = load_config(pd.CONFIG.with_name(name))["duplicate"]
        pd.plan(2026, cfg)  # valid
        assert set(cfg["lineups"]) == set(pd.KINDS)


# ---------------------------------------------------------------------------- end to end


@pytest.mark.slow
def test_a_tiny_run_writes_everything_and_does_not_depend_on_the_workers(tmp_path):
    one = pd.run(98, cfg={**TINY, "workers": 1}, out_dir=tmp_path / "one")
    two = pd.run(98, cfg={**TINY, "workers": 1}, workers=2, out_dir=tmp_path / "two")
    names = {p.name for p in (tmp_path / "one").iterdir()}
    assert {"duplicate.json", "duplicate_summary.md"} <= names
    for fig in ("duplicate_ratio", "duplicate_offsets", "duplicate_players"):
        assert {f"{fig}.png", f"{fig}-dark.png"} <= names
    assert compare_step(tmp_path / "one", tmp_path / "two") == []
    saved = json.loads((tmp_path / "one" / "duplicate.json").read_text(encoding="utf-8"))
    assert saved["calling_check"]["all_zero"]
    assert saved["calling_check"]["max_abs_deal_mean_chips"] == 0.0
    rows = saved["per_lineup"]
    assert len(rows) == 2 * sum(TINY["lineups"].values())
    assert {row["n"] for row in rows} == {2, 3, 6}
    for row in rows:
        assert row["deals"] == TINY["deals"]
        assert row["seating"] == [p["player_id"] for p in row["players"]]
        for p in row["players"]:
            assert len(p["rho_by_offset"]) == row["n"] // 2
            assert p["archetype"] in {a.value for a in ARCHETYPES}
            lo, hi = p["winrate"]["duplicate"]
            assert lo <= p["winrate"]["bb_per_100"] <= hi
    agg = saved["aggregates"]
    assert list(agg["by_kind"]) == list(pd.KINDS)
    assert list(agg["mixed_by_n"]) == ["2", "3", "6"]
    for e in agg["by_kind"].values():
        assert e["players"] == e["lineups"] * e["n"]
        assert sum(a["players"] for a in e["by_archetype"].values()) == e["players"]
    assert set(saved["predictions"]) == set(pd.PREDICTIONS)
    assert json.dumps(_jsonable(one)["aggregates"]) == json.dumps(_jsonable(two)["aggregates"])
    summary = (tmp_path / "one" / "duplicate_summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# P2.11b")
    assert "Calling check" in summary

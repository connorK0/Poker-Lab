"""P2.11b post hoc: the coverage arithmetic, the share of cancellation and the heads-up pairs, on
results made up for the test."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import stats

from experiments import phase2_duplicate as pd
from experiments import phase2_duplicate_posthoc as ph


def test_normal_coverage_rises_to_the_nominal_level():
    values = [ph.normal_coverage(g) for g in (5, 10, 100, 100000)]
    assert values == sorted(values)
    assert values[0] == pytest.approx(0.846, abs=0.001)
    assert values[-1] == pytest.approx(0.95, abs=1e-3)


def test_normal_coverage_matches_a_simulation():
    """The formula against the simple variance of a mean of G normal values, read as normal."""
    g = 10
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200000, g))
    half = stats.norm.ppf(0.975) * np.sqrt((g - 1) / g) * x.std(axis=1, ddof=1) / np.sqrt(g)
    assert np.mean(np.abs(x.mean(axis=1)) < half) == pytest.approx(ph.normal_coverage(g), abs=0.005)


def test_lineup_coverage_is_low_with_few_lineups():
    ratios = np.random.default_rng(1).lognormal(0.3, 0.2, (100, 6))
    rng = np.random.default_rng(2)
    few = ph.lineup_coverage(ratios, 5, 400, 200, rng)
    many = ph.lineup_coverage(ratios, 50, 400, 200, rng)
    assert few < many
    assert 0.7 < few < 0.95
    assert many > 0.9


def _rows():
    def player(ratio, rho, sigma, archetype):
        return {
            "ratio": ratio,
            "rho": rho,
            "sigma": sigma,
            "archetype": archetype,
            "bot_id": archetype,
            "rho_by_offset": [rho],
            "winrate": {"naive": [-2.0, 2.0], "duplicate": [-1.0, 1.0]},
        }

    def lineup(ratio, rho, sigma, a, b):
        players = [player(ratio, rho, sigma, a), player(ratio, rho, sigma, b)]
        return {"kind": "mixed_2", "n": 2, "players": players}

    return [
        lineup(1.5, -0.3, 5.0, "nit", "nit"),
        lineup(2.0, -0.5, 6.0, "lag", "nit"),
        lineup(1.0, 0.0, 7.0, "lag", "tag"),
    ]


def test_heads_up_pairs_count_a_lineup_once():
    h = ph.heads_up_pairs(_rows())
    assert h["same_archetype"] == {"lineups": 1, "median": 1.5}
    assert h["different_archetypes"] == {"lineups": 2, "median": 1.5}
    assert h["pairs"]["lag|nit"] == {"lineups": 1, "median": 2.0, "max": 2.0}
    assert h["largest"]["ratio"] == 2.0
    assert h["next_largest_ratio"] == 1.5


def test_the_share_gives_the_ratio_at_any_table_size():
    for n, rho in ((2, -0.25), (3, -0.1), (6, -0.05)):
        s = -(n - 1) * rho
        assert 1 / (1 - s) == pytest.approx(1 / (1 + (n - 1) * rho))


def test_a_run_on_the_committed_results_reads_them(tmp_path, monkeypatch):
    """The real results, with few trials so the test stays fast; the output is JSON-safe."""
    monkeypatch.setitem(ph.CONFIG, "trials", 5)
    monkeypatch.setitem(ph.CONFIG, "resamples", 20)
    monkeypatch.setitem(ph.CONFIG, "bootstrap", 20)
    r = ph.run(ph.SOURCE, out_dir=tmp_path)
    saved = json.loads((tmp_path / "duplicate_posthoc.json").read_text(encoding="utf-8"))
    assert saved["meta"]["source_commit"]
    assert set(saved["shares"]) == set(pd.KINDS)
    for e in saved["shares"].values():
        assert e["share"]["q25"] <= e["share"]["median"] <= e["share"]["q75"]
        assert e["implied_ratio"] == pytest.approx(e["measured_ratio"], rel=0.01)
    assert saved["identity_largest_relative_gap"] < 1e-12
    assert (
        (tmp_path / "duplicate_posthoc_summary.md")
        .read_text(encoding="utf-8")
        .startswith("# P2.11b")
    )
    assert r["heads_up"]["largest"]["ratio"] >= r["heads_up"]["next_largest_ratio"]

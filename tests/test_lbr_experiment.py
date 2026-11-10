"""P2.8b experiment bookkeeping: rows, exact values, paired seeds and aggregation."""

import numpy as np
import pytest

from experiments import phase2_lbr as ex
from experiments.common import REPO_ROOT, load_config

CFG = load_config(REPO_ROOT / "experiments" / "configs" / "phase2_quick.yaml")["lbr"]


def test_rows_cover_every_opponent_once_per_role():
    rows = ex.rows(CFG)
    keys = [r["key"] for r in rows]
    assert len(keys) == len(set(keys))
    assert {"check_call", "check_fold", "pot_odds", "pf_nash", "pf_nash_fallback_fcpa"} <= set(keys)
    assert all(f"bot_{b}" in keys for b in ex.BOTS)
    for r in rows:
        assert set(r["actions"]) <= {"fold", "call", "pot", "allin"}
        ex.opponent(r["opponent"])  # every opponent can be built


def test_exact_values_come_from_the_push_fold_game():
    assert ex.exact_value("pf_nash", 10.0) == pytest.approx(0.0, abs=1e-6)
    assert ex.exact_value("pf_any_two", 10.0) > 100
    assert ex.exact_value("pot_odds", 100.0) is None


def test_rows_against_one_opponent_share_their_deals():
    assert ex.chunk_seed(98, "pot_odds", 0) == ex.chunk_seed(98, "pot_odds", 0)
    assert ex.chunk_seed(98, "pot_odds", 0) != ex.chunk_seed(98, "pot_odds", 1)
    assert ex.chunk_seed(98, "pot_odds", 0) != ex.chunk_seed(98, "bot_tag", 0)
    assert ex.chunk_seed(98, "pot_odds", 0) != ex.chunk_seed(99, "pot_odds", 0)


def test_aggregate_pairs_ablations_with_the_main_rows(tmp_path):
    rng = np.random.default_rng(0)
    results = []
    for row in ex.rows(CFG):
        size = CFG["chunk_deals"]
        for c in range(0, row["deals"], size):
            n = min(size, row["deals"] - c)
            x = rng.normal(1000.0, 50.0, n)
            results.append({
                "key": row["key"], "chunk": c // size, "adjusted": x.tolist(),
                "raw": (x + rng.normal(0, 100, n)).tolist(), "seconds": 1.0,
                "stats": {"decisions": 1, "opponent_actions": 1, "surprises": 0, "model_calls": 1},
            })  # fmt: skip
    agg = ex.aggregate(results, CFG)
    assert agg["pf_nash"]["exact"] == pytest.approx(0.0, abs=1e-6)
    assert agg["pot_odds"]["exact"] is None
    diff = agg["norange_pot_odds"]["paired_difference"]
    assert diff["n"] == CFG["ablation_deals"]
    out = {"meta": {"seed": 1}, "rows": agg, "elapsed_seconds": 1.0, "predictions": {}}
    text = ex.markdown_summary(out)
    assert "lower bound" in text
    ex.make_figures(out, tmp_path)
    assert (tmp_path / "lbr_values.png").exists()
    assert (tmp_path / "lbr_values-dark.png").exists()

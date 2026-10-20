"""Win-rate evaluators and variance-reduction hooks."""

import numpy as np
import pytest

from poker_lab.engine import RandomAgent, SeatedPlayer, TableSimulator
from poker_lab.evaluation import (
    AIVATEvaluator,
    DuplicateDealEvaluator,
    HandResult,
    WinRateEvaluator,
    hand_results,
)
from poker_lab.schema import TableConfig


def _synthetic(mean_bb, sd_bb, n, seed=0):
    rng = np.random.default_rng(seed)
    chips = np.round(rng.normal(mean_bb, sd_bb, n) * 2).astype(int)  # bb = 2 chips
    return [HandResult(f"h{i}", "hero", int(c), 2) for i, c in enumerate(chips)]


def test_naive_win_rate_interval_covers_the_truth():
    covered = 0
    for seed in range(200):
        est = WinRateEvaluator().evaluate(_synthetic(0.05, 10.0, 2000, seed))
        covered += est.lower <= 5.0 <= est.upper  # 0.05 bb/hand = 5 bb/100
    assert covered / 200 == pytest.approx(0.95, abs=0.04)


def test_hands_needed_for_a_given_precision():
    # SD 10 bb/hand, +/- 5 bb/100 at 95% needs (10 / (0.05/1.96))^2 ~ 153,664 hands
    assert WinRateEvaluator.hands_for_half_width(10.0, 5.0) == pytest.approx(153_664, rel=0.01)


def test_hand_results_match_net_results():
    players = [SeatedPlayer(i, f"p{i}", RandomAgent()) for i in range(6)]
    hands = list(TableSimulator(TableConfig("w", 6, 1, 2), players, seed=8).play(50))
    results = hand_results(hands, "p0")
    assert len(results) == 50
    assert sum(r.net_chips for r in results) == sum(h.net_results()[0] for h in hands)
    totals = sum(sum(r.net_chips for r in hand_results(hands, f"p{i}")) for i in range(6))
    assert totals == 0  # zero-sum without rake
    est = WinRateEvaluator().evaluate(results)
    assert est.n_hands == 50
    assert est.lower < est.bb_per_100 < est.upper


def test_duplicate_pairs_cancel_shared_luck():
    rng = np.random.default_rng(1)
    luck = rng.normal(0, 10, 3000)
    a = [HandResult(f"a{i}", "hero", round(2 * (0.1 + x)), 2) for i, x in enumerate(luck)]
    b = [HandResult(f"b{i}", "hero", round(2 * (0.1 - x)), 2) for i, x in enumerate(luck)]
    interleaved = [r for pair in zip(a, b, strict=True) for r in pair]
    dup = DuplicateDealEvaluator().evaluate(interleaved)
    naive = WinRateEvaluator().evaluate(interleaved)
    assert dup.half_width < 0.2 * naive.half_width
    assert dup.lower <= 10.0 <= dup.upper
    with pytest.raises(ValueError, match="pairs"):
        DuplicateDealEvaluator().evaluate(interleaved[:-1])


def test_aivat_hook_refuses_without_value_function_and_reduces_variance_with_one():
    rng = np.random.default_rng(2)
    luck = rng.normal(0, 10, 1000)  # zero-mean by construction, as a control variate must be
    results = [HandResult(f"h{i}", "hero", round(2 * (0.05 + x)), 2) for i, x in enumerate(luck)]
    with pytest.raises(NotImplementedError, match="TODO"):
        AIVATEvaluator().evaluate(results)
    by_id = {r.hand_id: x for r, x in zip(results, luck, strict=True)}
    est = AIVATEvaluator(lambda r: by_id[r.hand_id]).evaluate(results)
    naive = WinRateEvaluator().evaluate(results)
    assert est.half_width < 0.2 * naive.half_width
    assert est.lower <= 5.0 <= est.upper

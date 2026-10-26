"""P3.8's experiment plumbing: windows, pools and the stand-in loader."""

import numpy as np
import pytest

from experiments.phase3_sequence import (
    NamedPool,
    PoolView,
    SeqPool,
    _chunk_windows,
    _spread_windows,
    held_out_windows,
    simulate_standin,
)
from poker_lab.profiling import window_totals
from poker_lab.profiling.sequences import TOKEN_FEATURES
from poker_lab.profiling.stats import N_COUNTERS, C

COL = {name: i for i, name in enumerate(TOKEN_FEATURES)}


def _pool(players=4, hands=60, seed=0) -> SeqPool:
    rng = np.random.default_rng(seed)
    counters = rng.integers(0, 3, size=(players, hands, N_COUNTERS)).astype(np.int16)
    return SeqPool(
        player_ids=[f"p{i}" for i in range(players)],
        labels=np.arange(players) % 5,
        params=np.zeros((players, 9)),
        counters=counters,
        tokens=np.zeros((players * hands, len(TOKEN_FEATURES)), dtype=np.float32),
        hand_ptr=np.arange(players * hands + 1),
        player_ptr=np.arange(0, players * hands + 1, hands),
    )


def test_held_out_windows_are_phase1_windows_in_the_same_order():
    pool = _pool()
    view = PoolView(pool)
    for n in (10, 25):
        w = held_out_windows(pool, n, max_windows=4)
        ref = window_totals(pool.cumulative, n, 4)
        np.testing.assert_array_equal(w[:, 0], ref.player)
        totals = np.stack([view.cum[p][s + k] - view.cum[p][s] for p, s, k in w])
        np.testing.assert_array_equal(totals, ref.totals)


def test_consistency_chunks_put_windows_before_their_reference_block():
    view = PoolView(_pool(players=3, hands=95))
    chunks, win, refs = _chunk_windows(view, [0, 2], n=10, ref=20, cap=3)
    assert chunks.tolist() == [[0, 0], [0, 1], [2, 0], [2, 1]]  # 95 // 40 = 2 chunks each
    for j, (p, c) in enumerate(chunks):
        base = 40 * c
        assert refs[j].tolist() == [p, base + 20, 20]
        mine = win[win[:, 3] == j]
        assert len(mine) == 2  # min(cap, ref // n)
        assert (mine[:, 0] == p).all()
        assert (mine[:, 1] >= base).all()
        assert (mine[:, 1] + mine[:, 2] <= base + 20).all()
    first, _, _ = _chunk_windows(view, [1], n=10, ref=20, cap=3, max_chunks=1)
    assert first.tolist() == [[1, 0]]


def test_spread_windows_need_n_hands():
    view = PoolView(_pool(players=2, hands=30))
    w = _spread_windows(view, [0, 1], n=10, cap=20)
    assert len(w) == 6
    assert len(_spread_windows(view, [0], n=40, cap=20)) == 0


@pytest.fixture(scope="module")
def standin() -> NamedPool:
    pytest.importorskip("torch")  # the loader builds HandSequences
    seating = [
        (("a", "b", "c", "d", "e", "Pluribus"), 30),
        (("a", "c", "f", "g", "h", "Pluribus"), 20),
    ]
    return simulate_standin(seating, "Pluribus", seed=5, prefix="t", workers=2)


def test_standin_keeps_counters_and_tokens_in_step(standin):
    seqs = standin.sequences()
    assert seqs.n_players == len(standin.ids)
    np.testing.assert_array_equal(seqs.hands(), standin.hands)
    assert standin.hands[standin.ids.index("s00")] == 50  # "a" played both blocks
    for p, counters in enumerate(standin.counters):
        h0, h1 = seqs.player_ptr[p], seqs.player_ptr[p + 1]
        ends = seqs.hand_ptr[h0 + 1 : h1 + 1] - 1  # each hand's closing token
        np.testing.assert_array_equal(seqs.tokens[ends, COL["end"]], 1.0)
        np.testing.assert_array_equal(
            seqs.tokens[ends, COL["saw_flop"]], (counters[:, C["saw_flop"]] > 0).astype(np.float32)
        )
    assert [n for _, n in standin.seating] == [30, 20]

"""P3.8's experiment plumbing: windows, pools, the stand-in loader, and the replicates (one
after another or several at once, with the same results)."""

import sys
from concurrent.futures import wait

import numpy as np
import pytest
import yaml

from experiments import phase3_sequence as ps
from experiments.common import _jsonable
from experiments.compare_runs import _walk, compare_step
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


# ---------------------------------------------------------------------------- replicates
#: A Phase 1 configuration small enough to simulate in seconds (leagues need multiples of 6).
TINY_P1 = {
    "n_seeds": 2,
    "n_train_per_archetype": 6,
    "n_test_per_archetype": 6,
    "hands_per_player": 100,
    "block_hands": 50,
    "equity_samples": 10,
    "sample_sizes": [25, 50, 100],
    "train_windows_max": 2,
    "test_windows_max": 2,
    "action_checkpoints": [0, 25],
    "action_window": 50,
}
#: The step's settings, as small (the test that runs the step adds ``phase1_config``).
TINY = {
    "phase1_master_seed": 2026,
    "variants": ["seq", "bag", "counters"],
    "model": {"n_rates": 4, "hidden": 8, "epochs": 2, "patience": 2, "players_per_batch": 8},
    "torch_threads": 2,
    "real_source": "standin",
    "standin": {"players": 7, "sessions": 6, "session_hands": 40},
    "bot_player": "Pluribus",
    "transfer_models": ["lr", "seq"],
    "extra_per_archetype": 6,
    "scale_variants": ["bag", "counters"],
    "windows_max": 2,
    "consistency_reference": 25,
    "bootstrap": 20,
}


def _differences(x, y) -> list:
    """Every difference between two results as ``compare_runs`` sees them (floats exactly;
    wall-clock keys skipped)."""
    found: list = []
    _walk(_jsonable(x), _jsonable(y), "", found)
    return found


def test_sequence_jobs_reach_the_sequence_step_only(monkeypatch, tmp_path):
    from experiments import phase3_schema, run_phase3

    seen = {}

    def fake(step):
        def run(seed, *, cfg, workers, out_dir, **options):
            seen[step] = options

        return run

    monkeypatch.setattr(ps, "run", fake("sequence"))
    monkeypatch.setattr(phase3_schema, "run", fake("schema"))
    monkeypatch.setattr("experiments.explorer.build.refresh", lambda out_dir: None)
    argv = ["run_phase3", "--only", "sequence", "schema", "--out", str(tmp_path)]
    for extra, jobs in (([], 1), (["--sequence-jobs", "2"], 2), (["--sequence-jobs", "3"], 3)):
        monkeypatch.setattr(sys, "argv", argv + extra)
        run_phase3.main()
        assert seen == {"sequence": {"replicate_jobs": jobs}, "schema": {}}
    monkeypatch.setattr(sys, "argv", [*argv, "--sequence-jobs", "0"])
    with pytest.raises(SystemExit):
        run_phase3.main()


def test_one_job_runs_the_replicates_here_one_after_another(monkeypatch):
    """N = 1 (or a single replicate) is the old loop, in this process; N above the number of
    replicates trains every replicate at once."""
    order = []

    def here(seed, *args, **kwargs):
        order.append(seed)
        return {"seed": seed}

    monkeypatch.setattr(ps, "replicate", here)
    monkeypatch.setattr(ps, "_in_parallel", lambda *a, **k: pytest.fail("one job: no processes"))
    common = ({}, {}, {}, {}, {})
    assert ps.run_replicates([5, 6, 7], *common, jobs=1) == [{"seed": s} for s in (5, 6, 7)]
    assert ps.run_replicates([8], *common, jobs=2) == [{"seed": 8}]
    assert order == [5, 6, 7, 8]
    monkeypatch.setattr(ps, "_in_parallel", lambda seeds, *a, jobs, **k: ("processes", jobs))
    assert ps.run_replicates([5, 6, 7], *common, jobs=2) == ("processes", 2)
    assert ps.run_replicates([5, 6], *common, jobs=4) == ("processes", 2)


def test_several_jobs_warn_about_threads_and_memory(monkeypatch):
    """Warnings, not errors: more torch threads in all than logical processors, or less free
    memory than about 5 GB per replicate (Windows only); one job never warns."""
    monkeypatch.setattr(ps.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(ps, "_free_memory_gb", lambda: 7.0)
    assert ps.resource_warnings(1, 32) == []
    threads, memory = ps.resource_warnings(3, 8)
    assert "24 threads on 16 logical processors" in threads
    assert "7.0 GB" in memory
    assert "--sequence-jobs 1" in memory
    monkeypatch.setattr(ps, "_free_memory_gb", lambda: 12.0)
    assert ps.resource_warnings(2, 8) == []
    monkeypatch.setattr(ps, "_free_memory_gb", lambda: None)  # not Windows
    assert ps.resource_warnings(2, 8) == []
    monkeypatch.undo()
    free = ps._free_memory_gb()
    assert free is None if sys.platform != "win32" else free > 0


@pytest.fixture(scope="module")
def tiny_replicate():
    pytest.importorskip("torch")
    return ps.generate_sequences(5, TINY_P1, workers=1, use_cache=False)


@pytest.mark.slow
def test_transfer_reuses_part_a_embeddings_of_the_held_out_bots(tiny_replicate, standin):
    """Part A predicts the held-out bots through the view that transfer then reads, so the
    sequence model encodes none of their hands again; every number equals the old wiring's
    (part A's own inputs, a new view for transfer), float for float."""
    import torch

    torch.set_num_threads(2)
    data, view = tiny_replicate, PoolView(standin)
    players = [i for i, pid in enumerate(standin.ids) if pid != "Pluribus"]

    def transfer(test_view):
        a, models = ps.part_a(data, TINY_P1, TINY, variants=["seq"], test_view=test_view)
        bots = PoolView(data.test) if test_view is None else test_view
        encoded = []
        embed = models["seq"]._embed

        def counted(seqs, block):
            encoded.append(id(seqs))
            return embed(seqs, block)

        models["seq"]._embed = counted
        out = ps.transfer_rep(models, view, players, bots, a["sizes"], TINY)
        return a, out, encoded.count(id(bots.inputs("tokens")))

    a_old, old, encoded_old = transfer(None)
    a_new, new, encoded_new = transfer(PoolView(data.test))
    assert _differences(a_old, a_new) == []
    assert _differences(old, new) == []
    assert len(old["consistency"][25]["seq"]["bots"]["confidence"]) == 30  # the bots were read
    assert (encoded_old, encoded_new) == (2, 0)  # 30 bots: two blocks of 16, once each


@pytest.mark.slow
def test_two_replicate_jobs_write_what_one_writes(tmp_path, monkeypatch, capsys):
    """The step with two replicates trained at once (one process each, with the configured
    torch threads; the last replicate's results delivered first) writes the same JSON, summary
    and figures as one replicate after another, apart from provenance and timing; ``meta``
    records the jobs. Each run starts from an empty cache, so the parallel path simulates the
    replicates itself."""
    pytest.importorskip("torch")
    from experiments import phase1_data

    def cache(name):
        monkeypatch.setattr(phase1_data, "CACHE_DIR", tmp_path / name)
        monkeypatch.setattr(ps, "CACHE_DIR", tmp_path / name)

    p1_path = tmp_path / "phase1_tiny.yaml"
    p1_path.write_text(yaml.safe_dump(TINY_P1), encoding="utf-8")
    cfg = {**TINY, "phase1_config": str(p1_path), "replicate_seeds": [98, 99]}
    cache("c1")
    one = ps.run(2026, cfg=cfg, workers=1, out_dir=tmp_path / "one", replicate_jobs=1)

    delivered, threads = [], []

    def last_first(futures, return_when):
        done, pending = wait(futures)  # every running replicate
        delivered.extend(sorted((futures[f] for f in done), reverse=True))
        return sorted(done, key=lambda f: -futures[f]), pending

    def spy(*args, **kwargs):
        outs = run_replicates(*args, **kwargs)
        threads.extend(o["torch_threads"] for o in outs)
        return outs

    run_replicates = ps.run_replicates
    monkeypatch.setattr(ps, "wait", last_first)
    monkeypatch.setattr(ps, "run_replicates", spy)
    cache("c2")
    capsys.readouterr()
    two = ps.run(2026, cfg=cfg, workers=1, out_dir=tmp_path / "two", replicate_jobs=3)
    log = capsys.readouterr().out
    assert sorted(p.name for p in (tmp_path / "c2").iterdir()) == sorted(
        p.name for p in (tmp_path / "c1").iterdir()
    )  # the parallel path simulated what the sequential one did
    assert "replicate 2/2 (99): hands ready" in log
    assert delivered == [1, 0]
    first = log.index("replicate 2/2 (99): done (1/2 done)")
    assert log.index("replicate 1/2 (98): training starts") < first
    assert first < log.index("replicate 1/2 (98): done (2/2 done)")
    assert threads == [cfg["torch_threads"]] * 2
    assert (one["meta"]["replicate_jobs"], two["meta"]["replicate_jobs"]) == (1, 2)
    assert [c["seed"] for c in two["replicate_checks"]] == [98, 99]
    assert compare_step(tmp_path / "one", tmp_path / "two") == []


@pytest.mark.slow
def test_a_failing_replicate_reaches_the_parent_and_starts_no_more(tmp_path, monkeypatch, capsys):
    """A worker's error is raised in the parent, with the worker's output as a note. At most
    ``jobs`` replicates are handed out at a time, and none after a failure."""
    pytest.importorskip("torch")
    missing = tmp_path / "missing.npz"
    monkeypatch.setattr(ps, "cache_replicate", lambda *a: {"sequence": missing, "phase1": missing})
    with pytest.raises(FileNotFoundError) as info:
        ps._in_parallel([5, 6, 7], {}, {"torch_threads": 1}, {}, {}, workers=1, jobs=2)
    log = capsys.readouterr().out
    assert "replicate 1/3 (5) failed: FileNotFoundError" in log
    assert "replicate 2/3 (6) failed: FileNotFoundError" in log
    assert "replicate 3/3 (7): training starts" not in log
    assert any("output before the error" in note for note in info.value.__notes__)

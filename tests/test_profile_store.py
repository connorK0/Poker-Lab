"""ProfileStore: completed hands only, idempotent, persistent across sessions."""

import json

import numpy as np
import pytest

from poker_lab.engine import RandomAgent, SeatedPlayer, TableSimulator
from poker_lab.profiling import counters_by_player, raw_stats
from poker_lab.profiling.profile_store import (
    IncompleteHandError,
    ProfileSnapshot,
    ProfileStore,
    hand_half,
)
from poker_lab.profiling.stats import COUNTERS
from poker_lab.schema import HandHistoryStore, TableConfig
from tests.helpers import record_reference_hand


def _session(seed, ids):
    players = [SeatedPlayer(i, pid, RandomAgent()) for i, pid in enumerate(ids)]
    return list(TableSimulator(TableConfig(f"t{seed}", 6, 1, 2), players, seed=seed).play(60))


def test_ingests_only_completed_hands():
    live_store = HandHistoryStore()
    record_reference_hand(live_store, "live", complete=False)
    store = ProfileStore()
    with pytest.raises(IncompleteHandError):
        store.ingest(live_store.get("live"))
    with pytest.raises(TypeError):
        store.ingest({"hand_id": "x"})  # type: ignore[arg-type]
    assert store.hands_ingested == 0


def test_accumulates_across_sessions_and_is_idempotent():
    s1 = _session(1, ["ann", "ben", "cat", "dan", "eve", "fay"])
    s2 = _session(2, ["ann", "gus", "hal", "ida", "jo", "kim"])
    store = ProfileStore()
    assert store.ingest_many(s1) == 60
    assert store.ingest_many(s1) == 0  # re-importing a session does not double count
    assert store.duplicates_skipped == 60
    store.ingest_many(s2)
    assert store.profile("ann").hands == 120
    assert store.profile("ben").hands == 60
    assert len(store) == 11
    expected = counters_by_player(s1 + s2)["ann"]
    assert np.array_equal(np.array(store.profile("ann").counters), expected)


def test_profile_shrinks_and_matches_raw_rates_with_data():
    hands = _session(3, ["a", "b", "c", "d", "e", "f"]) * 1
    store = ProfileStore()
    store.ingest_many(hands)
    prof = store.profile("a")
    raw = raw_stats(np.array(prof.counters))
    lo, hi = prof.intervals["vpip"]
    assert lo <= prof.stats["vpip"] <= hi
    # flat Beta(1,1) prior: posterior mean = (k+1)/(m+2), close to the raw rate
    assert prof.stats["vpip"] == pytest.approx(raw["vpip"], abs=0.05)
    unknown = store.profile("nobody")
    assert unknown.hands == 0
    assert unknown.stats["vpip"] == pytest.approx(0.5)  # the prior mean


def test_snapshot_is_immutable_and_records_sources():
    store = ProfileStore()
    hands = _session(4, ["a", "b", "c", "d", "e", "f"])
    store.ingest_many(hands)
    snap = store.snapshot(["a", "b"])
    assert isinstance(snap, ProfileSnapshot)
    assert set(snap.profiles) == {"a", "b"}
    assert snap.source_hand_ids == {h.hand_id for h in hands}
    with pytest.raises(TypeError):
        snap.profiles["z"] = None  # type: ignore[index]
    store.ingest_many(_session(5, ["a", "b", "c", "d", "e", "f"]))
    assert snap.get("a").hands == 60  # an existing snapshot does not change


def test_save_load_round_trip_and_refit(tmp_path):
    store = ProfileStore()
    for seed in range(6, 12):
        store.ingest_many(_session(seed, [f"p{seed}{i}" for i in range(6)]))
    path = tmp_path / "profiles.json"
    store.save(path)
    again = ProfileStore.load(path)
    assert again.players() == store.players()
    assert again.profile("p60").stats == store.profile("p60").stats
    assert again.hands_ingested == store.hands_ingested
    assert again.refit_priors(min_players=30)
    assert not ProfileStore().refit_priors(min_players=30)


def test_halves_split_by_hand_id_and_sum_to_totals():
    assert {hand_half(f"h{i}") for i in range(50)} == {0, 1}
    assert hand_half("abc") == hand_half("abc")  # stable, not Python's salted hash()
    hands = _session(13, ["a", "b", "c", "d", "e", "f"])
    store = ProfileStore()
    store.ingest_many(hands)
    halves = store._counters["a"]
    assert halves.shape == (2, len(COUNTERS))
    assert np.array_equal(halves.sum(axis=0), counters_by_player(hands)["a"])
    in_first = [h for h in hands if hand_half(h.hand_id) == 0]
    assert np.array_equal(halves[0], counters_by_player(in_first)["a"])


def test_refit_fits_size_priors_from_halves_and_keeps_them_for_format_1(tmp_path):
    store = ProfileStore()
    for seed in range(14, 20):
        store.ingest_many(_session(seed, [f"q{seed}{i}" for i in range(6)]))
    default = store.shrinker.continuous
    assert store.refit_priors(min_players=30)
    assert store.shrinker.continuous != default  # refit from the split halves
    # a format 1 file kept totals only: the players load without halves
    path = tmp_path / "v1.json"
    payload = {
        "format": "poker_lab.profile_store/1",
        "counter_names": list(COUNTERS),
        "players": {pid: c.sum(axis=0).tolist() for pid, c in store._counters.items()},
        "hand_ids": sorted(store._hand_ids),
        "priors": ProfileStore().shrinker.to_dict(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    old = ProfileStore.load(path)
    assert old.profile("q140").counters == store.profile("q140").counters
    assert old.refit_priors(min_players=30)
    assert old.shrinker.continuous == default  # no halves: size priors kept
    assert old.shrinker.binomial == store.shrinker.binomial
    old.save(tmp_path / "v2.json")
    assert ProfileStore.load(tmp_path / "v2.json")._unsplit == set(old.players())

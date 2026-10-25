"""``ProfileStore``: opponent profiles accumulated per named player across sessions.

* Ingests **completed** hands only. An in-progress hand is rejected, so a hand still being
  played can never contribute to any profile.
* Ingestion is idempotent per ``hand_id`` (re-importing a session does not double count).
* Only public information is used: hands are redacted to what a participant could know before
  counting (unshown hole cards removed), and the store keeps counters, never hands.
* ``snapshot()`` returns an immutable ``ProfileSnapshot`` (shrunk stats + credible intervals
  per player) that records which hand ids it was built from. The ``Advisor`` uses that to
  verify the current hand is not among them.
* Profiles persist to JSON across sessions (``save`` / ``load``).

Priors default to uniform Beta(1, 1); once enough players are stored, ``refit_priors()``
fits empirical-Bayes priors to the stored population itself (needs >= ``min_players``).
Each player's counters are kept in two halves (a stable hash of the hand id picks the half),
so a refit can also fit the bet-size priors, whose pseudo-counts need two disjoint samples per
player (``fit_normal_prior``). Stores saved in format 1 kept totals only; their players have
no halves, and a refit keeps the size priors as they are.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np

from poker_lab.profiling.shrinkage import (
    BetaPrior,
    EmpiricalBayesShrinker,
    NormalPrior,
    aggression_factor,
)
from poker_lab.profiling.stats import BINOMIAL_STATS, COUNTERS, N_COUNTERS, hand_counters
from poker_lab.schema import HandHistory, HandStatus, redact_hole_cards

FORMAT = "poker_lab.profile_store/2"
_FORMAT_V1 = "poker_lab.profile_store/1"


class IncompleteHandError(ValueError):
    """Raised when an in-progress hand is offered to the profile store."""


def hand_half(hand_id: str) -> int:
    """Which of a player's two halves (0 or 1) a hand counts toward: a stable hash of its id,
    so the split is the same in every session and independent of what happened in the hand."""
    return hashlib.blake2b(hand_id.encode("utf-8"), digest_size=1).digest()[0] & 1


def uniform_shrinker() -> EmpiricalBayesShrinker:
    return EmpiricalBayesShrinker(
        {s.name: BetaPrior(1.0, 1.0) for s in BINOMIAL_STATS},
        {"bet_size": NormalPrior(0.6, 3.0), "open_size": NormalPrior(3.0, 3.0)},
    )


@dataclass(frozen=True)
class PlayerProfile:
    player_id: str
    hands: int
    stats: Mapping[str, float]  # shrunk posterior means (+ derived "af")
    intervals: Mapping[str, tuple[float, float]]  # 95% credible intervals (binomial stats)
    opportunities: Mapping[str, int]
    counters: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True)
class ProfileSnapshot:
    """Immutable profiles at a point in time. Safe to pass to an ``Advisor``."""

    profiles: Mapping[str, PlayerProfile]
    source_hand_ids: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
        object.__setattr__(self, "source_hand_ids", frozenset(self.source_hand_ids))

    def get(self, player_id: str) -> PlayerProfile | None:
        return self.profiles.get(player_id)

    def __contains__(self, player_id: object) -> bool:
        return player_id in self.profiles


class ProfileStore:
    def __init__(self, shrinker: EmpiricalBayesShrinker | None = None) -> None:
        self.shrinker = shrinker or uniform_shrinker()
        self._counters: dict[str, np.ndarray] = {}  # player -> (2 halves, N_COUNTERS)
        self._unsplit: set[str] = set()  # players loaded from format 1 (halves unknown)
        self._hand_ids: set[str] = set()
        self.duplicates_skipped = 0

    # ------------------------------------------------------------------ ingestion
    def ingest(self, hand: HandHistory) -> bool:
        """Add one completed hand. Returns False if the hand id was already ingested."""
        if not isinstance(hand, HandHistory):
            raise TypeError("ProfileStore ingests HandHistory objects")
        if hand.status is not HandStatus.COMPLETE:
            raise IncompleteHandError(
                f"hand {hand.hand_id!r} is {hand.status.value}; only completed hands are ingested"
            )
        if hand.hand_id in self._hand_ids:
            self.duplicates_skipped += 1
            return False
        public = redact_hole_cards(hand)
        half = hand_half(hand.hand_id)
        for player_id, row in hand_counters(public):
            if player_id not in self._counters:
                self._counters[player_id] = np.zeros((2, N_COUNTERS), dtype=np.int64)
            self._counters[player_id][half] += row
        self._hand_ids.add(hand.hand_id)
        return True

    def ingest_many(self, hands: Iterable[HandHistory]) -> int:
        return sum(self.ingest(h) for h in hands)

    # ------------------------------------------------------------------ queries
    def players(self) -> list[str]:
        return sorted(self._counters)

    def __len__(self) -> int:
        return len(self._counters)

    @property
    def hands_ingested(self) -> int:
        return len(self._hand_ids)

    def profile(self, player_id: str) -> PlayerProfile:
        halves = self._counters.get(player_id)
        counts = np.zeros(N_COUNTERS, dtype=np.int64) if halves is None else halves.sum(axis=0)
        shrunk = self.shrinker.transform(counts, intervals=True)
        stats = {name: float(v) for name, v in zip(shrunk.names, shrunk.mean, strict=True)}
        stats["af"] = float(aggression_factor(np.array(stats["aggr_share"])))
        intervals = {
            s.name: (float(shrunk.lower[j]), float(shrunk.upper[j]))
            for j, s in enumerate(BINOMIAL_STATS)
        }
        opps = {name: int(m) for name, m in zip(shrunk.names, shrunk.opportunities, strict=True)}
        return PlayerProfile(
            player_id=player_id,
            hands=int(counts[COUNTERS.index("hands")]),
            stats=MappingProxyType(stats),
            intervals=MappingProxyType(intervals),
            opportunities=MappingProxyType(opps),
            counters=tuple(int(c) for c in counts),
        )

    def snapshot(self, player_ids: Iterable[str] | None = None) -> ProfileSnapshot:
        ids = self.players() if player_ids is None else list(player_ids)
        return ProfileSnapshot(
            profiles={pid: self.profile(pid) for pid in ids},
            source_hand_ids=frozenset(self._hand_ids),
        )

    def refit_priors(self, min_players: int = 30) -> bool:
        """Fit empirical-Bayes priors to the stored players (if there are enough of them).

        The size priors are refit from the split halves; if any player has none (loaded from
        format 1), they are kept as they are."""
        if len(self._counters) < min_players:
            return False
        stacked = np.stack(list(self._counters.values()))  # players x halves x counters
        halves = None if self._unsplit else (stacked[:, 0], stacked[:, 1])
        fitted = EmpiricalBayesShrinker.fit(stacked.sum(axis=1), halves)
        continuous = self.shrinker.continuous if halves is None else fitted.continuous
        self.shrinker = EmpiricalBayesShrinker(fitted.binomial, continuous)
        return True

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        payload = {
            "format": FORMAT,
            "counter_names": list(COUNTERS),
            "players": {pid: c.tolist() for pid, c in sorted(self._counters.items())},
            "unsplit": sorted(self._unsplit),
            "hand_ids": sorted(self._hand_ids),
            "priors": self.shrinker.to_dict(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> ProfileStore:
        """Load a store saved in format 2, or in format 1 (totals only: kept in the first half,
        and the player is marked as having no halves)."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("format")
        if version not in (FORMAT, _FORMAT_V1):
            raise ValueError("not a profile store file")
        if payload["counter_names"] != list(COUNTERS):
            raise ValueError("counter layout changed since this store was saved")
        store = cls(EmpiricalBayesShrinker.from_dict(payload["priors"]))
        if version == _FORMAT_V1:
            store._counters = {
                pid: np.stack([np.array(c, dtype=np.int64), np.zeros(N_COUNTERS, np.int64)])
                for pid, c in payload["players"].items()
            }
            store._unsplit = set(store._counters)
        else:
            store._counters = {
                pid: np.array(c, dtype=np.int64) for pid, c in payload["players"].items()
            }
            store._unsplit = set(payload["unsplit"])
        store._hand_ids = set(payload["hand_ids"])
        return store

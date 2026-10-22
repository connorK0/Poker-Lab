"""Bot populations and their ground truth, kept separate from hand histories.

Player ids are random hex tokens and the population order is shuffled, so neither the id nor
the seating order says anything about a player's archetype. The ground truth (archetype label
and true ``BotParams``) lives only in ``GroundTruth``, which is saved to its own JSON file and
never written into hand histories.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poker_lab.bots.archetype_bot import ParametricBot
from poker_lab.bots.archetypes import ARCHETYPES, Archetype, sample_params
from poker_lab.bots.params import PARAM_NAMES, BotParams


@dataclass(frozen=True)
class PlayerSpec:
    player_id: str
    archetype: Archetype
    params: BotParams

    def make_agent(self, *, equity_samples: int = 150) -> ParametricBot:
        return ParametricBot(self.params, equity_samples=equity_samples)


def make_population(
    n_per_archetype: int | Mapping[Archetype, int],
    rng: np.random.Generator,
    *,
    id_prefix: str = "p",
) -> list[PlayerSpec]:
    """Sample a shuffled population with opaque, unique player ids."""
    counts = (
        dict.fromkeys(ARCHETYPES, n_per_archetype)
        if isinstance(n_per_archetype, int)
        else dict(n_per_archetype)
    )
    players: list[PlayerSpec] = []
    used: set[str] = set()
    for archetype in ARCHETYPES:
        for _ in range(counts.get(archetype, 0)):
            while True:
                pid = f"{id_prefix}{int(rng.integers(16**8)):08x}"
                if pid not in used:
                    used.add(pid)
                    break
            players.append(PlayerSpec(pid, archetype, sample_params(archetype, rng)))
    order = rng.permutation(len(players))
    return [players[i] for i in order]


class GroundTruth:
    """Player id -> (archetype, true parameters). Never part of a hand history."""

    def __init__(self, players: Iterable[PlayerSpec]) -> None:
        self._players = {p.player_id: p for p in players}

    def __contains__(self, player_id: str) -> bool:
        return player_id in self._players

    def __len__(self) -> int:
        return len(self._players)

    def player_ids(self) -> list[str]:
        return list(self._players)

    def archetype_of(self, player_id: str) -> Archetype:
        return self._players[player_id].archetype

    def params_of(self, player_id: str) -> BotParams:
        return self._players[player_id].params

    def labels(self, player_ids: Sequence[str]) -> np.ndarray:
        """Integer archetype labels (index into ``ARCHETYPES``)."""
        index = {a: i for i, a in enumerate(ARCHETYPES)}
        return np.array([index[self._players[p].archetype] for p in player_ids], dtype=int)

    def param_matrix(self, player_ids: Sequence[str]) -> np.ndarray:
        return np.stack([self._players[p].params.to_array() for p in player_ids])

    def to_json(self, path: str | Path) -> None:
        payload = {
            "param_names": list(PARAM_NAMES),
            "players": [
                {"player_id": p.player_id, "archetype": p.archetype.value, **p.params.to_dict()}
                for p in self._players.values()
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="\n")

    @classmethod
    def from_json(cls, path: str | Path) -> GroundTruth:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            PlayerSpec(
                d["player_id"],
                Archetype(d["archetype"]),
                BotParams.from_dict({n: d[n] for n in PARAM_NAMES}),
            )
            for d in payload["players"]
        )

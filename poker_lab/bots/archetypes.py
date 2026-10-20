"""Archetype definitions: parameter distributions and intended (HUD-style) tendencies.

Each archetype is a distribution over ``BotParams``, not a single bot, so a sampled population
varies within types and adjacent types overlap. Probability parameters are drawn from
Beta(mean * k, (1 - mean) * k) with concentration ``k``; bet size is log-normal.

``INTENDED_STATS`` were written down before any simulation (typical HUD reads for each player
type, in percent; AF is a ratio). The M3 sanity check compares measured stats against them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from poker_lab.bots.params import PARAM_BOUNDS, PARAM_NAMES, BotParams


class Archetype(StrEnum):
    NIT = "nit"
    TAG = "tag"
    LAG = "lag"
    CALLING_STATION = "calling_station"
    MANIAC = "maniac"


ARCHETYPES: tuple[Archetype, ...] = tuple(Archetype)


@dataclass(frozen=True)
class BetaDist:
    mean: float
    concentration: float

    def sample(self, rng: np.random.Generator) -> float:
        k = self.concentration
        return float(rng.beta(self.mean * k, (1.0 - self.mean) * k))

    @property
    def sd(self) -> float:
        return float(np.sqrt(self.mean * (1 - self.mean) / (self.concentration + 1)))


@dataclass(frozen=True)
class LogNormalDist:
    median: float
    sigma: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(self.median * np.exp(self.sigma * rng.standard_normal()))

    @property
    def mean(self) -> float:
        return self.median


K = 30.0  # default concentration: SD ~ 0.09 at mean 0.5, ~0.06 at mean 0.12


def _spec(**means: float | LogNormalDist) -> dict[str, BetaDist | LogNormalDist]:
    out: dict[str, BetaDist | LogNormalDist] = {}
    for name in PARAM_NAMES:
        v = means[name]
        out[name] = v if isinstance(v, LogNormalDist) else BetaDist(float(v), K)
    return out


PARAM_DISTRIBUTIONS: dict[Archetype, dict[str, BetaDist | LogNormalDist]] = {
    Archetype.NIT: _spec(
        range_width=0.15,
        raise_share=0.70,
        three_bet_freq=0.20,
        continue_vs_raise=0.50,
        cbet_freq=0.55,
        fold_to_aggression=0.82,
        bluff_freq=0.05,
        value_aggression=0.55,
        bet_size_pot=LogNormalDist(0.50, 0.2),
    ),
    Archetype.TAG: _spec(
        range_width=0.26,
        raise_share=0.85,
        three_bet_freq=0.28,
        continue_vs_raise=0.50,
        cbet_freq=0.68,
        fold_to_aggression=0.50,
        bluff_freq=0.18,
        value_aggression=0.75,
        bet_size_pot=LogNormalDist(0.66, 0.2),
    ),
    Archetype.LAG: _spec(
        range_width=0.36,
        raise_share=0.85,
        three_bet_freq=0.42,
        continue_vs_raise=0.60,
        cbet_freq=0.78,
        fold_to_aggression=0.33,
        bluff_freq=0.38,
        value_aggression=0.85,
        bet_size_pot=LogNormalDist(0.75, 0.2),
    ),
    Archetype.CALLING_STATION: _spec(
        range_width=0.48,
        raise_share=0.18,
        three_bet_freq=0.04,
        continue_vs_raise=0.90,
        cbet_freq=0.30,
        fold_to_aggression=0.10,
        bluff_freq=0.06,
        value_aggression=0.35,
        bet_size_pot=LogNormalDist(0.45, 0.2),
    ),
    Archetype.MANIAC: _spec(
        range_width=0.62,
        raise_share=0.80,
        three_bet_freq=0.45,
        continue_vs_raise=0.80,
        cbet_freq=0.85,
        fold_to_aggression=0.18,
        bluff_freq=0.65,
        value_aggression=0.95,
        bet_size_pot=LogNormalDist(1.00, 0.2),
    ),
}

# Intended tendencies (written before simulation). Percent, except AF (a ratio).
INTENDED_STATS: dict[Archetype, dict[str, tuple[float, float]]] = {
    Archetype.NIT: {
        "vpip": (8, 16), "pfr": (6, 13), "three_bet": (1.5, 5), "af": (1.5, 3.5),
        "cbet": (45, 65), "fold_to_cbet": (55, 80),
    },
    Archetype.TAG: {
        "vpip": (17, 26), "pfr": (13, 22), "three_bet": (4, 9), "af": (2, 4),
        "cbet": (60, 78), "fold_to_cbet": (40, 60),
    },
    Archetype.LAG: {
        "vpip": (26, 38), "pfr": (20, 32), "three_bet": (7, 14), "af": (2.5, 5),
        "cbet": (68, 85), "fold_to_cbet": (30, 50),
    },
    Archetype.CALLING_STATION: {
        "vpip": (35, 60), "pfr": (3, 12), "three_bet": (0.5, 4), "af": (0.3, 1.2),
        "cbet": (20, 45), "fold_to_cbet": (5, 25),
    },
    Archetype.MANIAC: {
        "vpip": (45, 75), "pfr": (32, 60), "three_bet": (12, 30), "af": (3, 8),
        "cbet": (75, 95), "fold_to_cbet": (10, 35),
    },
}  # fmt: skip

# Orderings that should hold for the stats we do not give ranges for.
INTENDED_ORDERINGS: tuple[tuple[str, Archetype, Archetype], ...] = (
    ("wtsd", Archetype.CALLING_STATION, Archetype.NIT),  # stations go to showdown more
    ("wtsd", Archetype.CALLING_STATION, Archetype.TAG),
    ("wsd", Archetype.NIT, Archetype.CALLING_STATION),  # nits win more often when they do
    ("wsd", Archetype.TAG, Archetype.MANIAC),
)


def _clip(name: str, value: float) -> float:
    lo, hi = PARAM_BOUNDS[name]
    return float(min(max(value, lo), hi))


def sample_params(archetype: Archetype, rng: np.random.Generator) -> BotParams:
    """Draw one bot's parameters from the archetype's distribution."""
    dists = PARAM_DISTRIBUTIONS[archetype]
    return BotParams(**{n: _clip(n, dists[n].sample(rng)) for n in PARAM_NAMES})


def mean_params(archetype: Archetype) -> BotParams:
    """The archetype's central parameters (Beta means, log-normal medians)."""
    dists = PARAM_DISTRIBUTIONS[archetype]
    return BotParams(**{n: _clip(n, dists[n].mean) for n in PARAM_NAMES})

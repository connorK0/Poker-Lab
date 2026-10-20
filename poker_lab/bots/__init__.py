"""Parameterized archetype bots with ground truth (Phase 1)."""

from poker_lab.bots.archetype_bot import (
    BUCKET_THRESHOLDS,
    ParametricBot,
    equity_bucket,
)
from poker_lab.bots.archetypes import (
    ARCHETYPES,
    INTENDED_ORDERINGS,
    INTENDED_STATS,
    PARAM_DISTRIBUTIONS,
    Archetype,
    mean_params,
    sample_params,
)
from poker_lab.bots.hand_strength import (
    equity_vs_random,
    hand_class,
    preflop_equity_table,
    preflop_percentile,
)
from poker_lab.bots.params import PARAM_BOUNDS, PARAM_NAMES, BotParams
from poker_lab.bots.population import GroundTruth, PlayerSpec, make_population

__all__ = [
    "ARCHETYPES",
    "BUCKET_THRESHOLDS",
    "INTENDED_ORDERINGS",
    "INTENDED_STATS",
    "PARAM_BOUNDS",
    "PARAM_DISTRIBUTIONS",
    "PARAM_NAMES",
    "Archetype",
    "BotParams",
    "GroundTruth",
    "ParametricBot",
    "PlayerSpec",
    "equity_bucket",
    "equity_vs_random",
    "hand_class",
    "make_population",
    "mean_params",
    "preflop_equity_table",
    "preflop_percentile",
    "sample_params",
]

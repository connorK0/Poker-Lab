"""Opponent profiling (Phase 1).

* ``stats``: counters with explicit opportunity counting, from hand histories.
* ``shrinkage``: empirical-Bayes priors (beta-binomial; normal for sizes) and posteriors.
* ``features``: profile feature vectors and sample-size windows.
* ``classifier``: calibrated archetype classifier.
* ``regression``: continuous bot-parameter regression.
* ``situations`` / ``action_model``: next-action prediction from public situations + profile.
* ``profile_store``: per-player profiles accumulated across sessions from completed hands.

Names are re-exported lazily (PEP 562): simulation workers that only need ``stats`` do not
pay for importing scikit-learn.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "poker_lab.profiling.action_model": ("ActionPredictor", "marginal_baseline"),
    "poker_lab.profiling.classifier": ("ArchetypeClassifier", "fit_temperature"),
    "poker_lab.profiling.features": (
        "Windows",
        "feature_names",
        "prefix_totals",
        "profile_features",
        "to_model_space",
        "window_totals",
    ),
    "poker_lab.profiling.regression": ("ParamRegressor", "oos_r2"),
    "poker_lab.profiling.shrinkage": (
        "BetaPrior",
        "EmpiricalBayesShrinker",
        "NormalPrior",
        "ShrunkStats",
        "aggression_factor",
        "fit_beta_binomial",
        "fit_normal_prior",
    ),
    "poker_lab.profiling.situations": (
        "ACTION_CLASSES",
        "SITUATION_FEATURES",
        "DecisionRowExtractor",
        "decision_points",
    ),
    "poker_lab.profiling.stats": (
        "BINOMIAL_STATS",
        "COUNTERS",
        "N_COUNTERS",
        "REQUIRED_STATS",
        "counters_by_player",
        "hand_counters",
        "raw_stats",
    ),
    "poker_lab.profiling.profile_store": ("PlayerProfile", "ProfileSnapshot", "ProfileStore"),
}
_WHERE = {name: module for module, names in _EXPORTS.items() for name in names}
__all__ = sorted(_WHERE)


def __getattr__(name: str) -> Any:
    if name in _WHERE:
        return getattr(import_module(_WHERE[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__

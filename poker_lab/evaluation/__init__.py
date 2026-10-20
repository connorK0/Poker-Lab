"""Evaluation: metrics, confidence intervals, win-rate evaluation with variance-reduction hooks.

Names are re-exported lazily (PEP 562) to keep simulation workers light.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "poker_lab.evaluation.intervals": ("Interval", "bootstrap_ci", "t_interval", "wilson_interval"),
    "poker_lab.evaluation.metrics": (
        "Reliability",
        "accuracy",
        "brier",
        "confusion_matrix",
        "expected_calibration_error",
        "log_loss",
        "per_row_log_loss",
        "reliability",
    ),
    "poker_lab.evaluation.winrate": (
        "AIVATEvaluator",
        "DuplicateDealEvaluator",
        "Evaluator",
        "HandResult",
        "WinRateEstimate",
        "WinRateEvaluator",
        "hand_results",
    ),
}
_WHERE = {name: module for module, names in _EXPORTS.items() for name in names}
__all__ = sorted(_WHERE)


def __getattr__(name: str) -> Any:
    if name in _WHERE:
        return getattr(import_module(_WHERE[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__

"""Small two-player zero-sum poker games (Kuhn, Leduc) for exact solving and evaluation (Phase 2).

``Game`` describes rules; ``compile_tree`` flattens a game into NumPy arrays (``GameTree``);
``TabularPolicy`` is a behavior strategy over its information sets; ``best_response`` and
``exploitability`` evaluate policies exactly. Names are re-exported lazily (PEP 562).
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "poker_lab.games.base": ("CHANCE", "Game"),
    "poker_lab.games.kuhn": ("KuhnPoker", "KuhnState"),
    "poker_lab.games.leduc": ("LeducHoldem", "LeducState"),
    "poker_lab.games.tree": ("TERMINAL", "GameTree", "compile_tree"),
    "poker_lab.games.policy": ("TabularPolicy",),
    "poker_lab.games.evaluate": (
        "BestResponse",
        "ExploitabilityReport",
        "best_response",
        "expected_value",
        "exploitability",
        "exploitability_report",
        "head_to_head",
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

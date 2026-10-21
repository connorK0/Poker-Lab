"""Equilibrium solvers for the small games in ``poker_lab.games`` (Phase 2).

``solve_lp`` computes an exact equilibrium with the sequence-form linear program. Names are
re-exported lazily (PEP 562).
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "poker_lab.solvers.lp": ("LPSolution", "SequenceForm", "sequence_form", "solve_lp"),
}
_WHERE = {name: module for module, names in _EXPORTS.items() for name in names}
__all__ = sorted(_WHERE)


def __getattr__(name: str) -> Any:
    if name in _WHERE:
        return getattr(import_module(_WHERE[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__

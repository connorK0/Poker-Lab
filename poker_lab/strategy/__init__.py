"""Strategy providers: a ``SeatView`` of a real hold'em hand -> a mixed strategy.

Baselines (``baselines``) and the push/fold provider over the solved abstract game
(``push_fold``, P3.3). The small games of Phase 2 keep their own ``TabularPolicy`` objects; see
docs/BUILD_LOG.md, Step 23. Names are re-exported lazily (PEP 562), so importing a baseline does
not load the solvers.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "poker_lab.strategy.base": (
        "ActionDistribution",
        "ActionOption",
        "StrategyAgent",
        "StrategyProvider",
        "require_actor",
    ),
    "poker_lab.strategy.baselines": (
        "CheckCallStrategy",
        "PotOddsStrategy",
        "UniformRandomStrategy",
    ),
    "poker_lab.strategy.push_fold": (
        "BlindPot",
        "OffTree",
        "PushFoldSpot",
        "PushFoldStrategy",
        "blind_pot",
        "fixed_ranges",
        "nash_ranges",
        "push_fold_spot",
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

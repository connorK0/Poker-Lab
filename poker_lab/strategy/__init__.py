"""Strategy providers. Phase 2 (GTO baseline: CFR / Deep CFR) plugs in here.

TODO(Phase 2): ``CFRStrategy`` for Kuhn/Leduc (tabular CFR+), ``DeepCFRStrategy`` (PyTorch),
and an NLHE provider over an abstracted action space with action translation.
"""

from poker_lab.strategy.base import ActionDistribution, ActionOption, StrategyProvider
from poker_lab.strategy.baselines import CheckCallStrategy, PotOddsStrategy, UniformRandomStrategy

__all__ = [
    "ActionDistribution",
    "ActionOption",
    "CheckCallStrategy",
    "PotOddsStrategy",
    "StrategyProvider",
    "UniformRandomStrategy",
]

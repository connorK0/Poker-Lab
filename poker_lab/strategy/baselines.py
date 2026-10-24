"""Baseline strategy providers that let the pipeline run end to end before Phase 2.

* ``CheckCallStrategy``: always check or call (trivial baseline).
* ``UniformRandomStrategy``: uniform over fold (when legal), check/call, min-raise.
* ``PotOddsStrategy``: a simple equity heuristic. Continue when Monte Carlo equity against one
  random hand beats the pot odds, raise strong hands. It is a reference point, not a model of
  good play.
"""

from __future__ import annotations

import random

from poker_lab.bots.hand_strength import equity_vs_random, preflop_percentile
from poker_lab.engine.agent import Decision, clamp_raise
from poker_lab.schema import SeatView, Street
from poker_lab.strategy.base import ActionDistribution, StrategyProvider, require_actor


class CheckCallStrategy(StrategyProvider):
    name = "check_call"

    def action_distribution(self, view: SeatView) -> ActionDistribution:
        require_actor(view)
        return ActionDistribution.pure(Decision.check_call())


class UniformRandomStrategy(StrategyProvider):
    name = "uniform_random"

    def action_distribution(self, view: SeatView) -> ActionDistribution:
        legal = require_actor(view)
        options = [Decision.check_call()]
        if legal.can_fold:
            options.append(Decision.fold())
        if legal.can_bet_or_raise:
            options.append(Decision.bet_raise(legal.min_raise_to))
        return ActionDistribution.from_weights(dict.fromkeys(options, 1.0))


class PotOddsStrategy(StrategyProvider):
    """Equity-vs-pot-odds heuristic with a private seeded RNG for its Monte Carlo estimates."""

    name = "pot_odds"

    def __init__(self, *, seed: int = 0, equity_samples: int = 200, raise_equity: float = 0.75):
        self._rng = random.Random(seed)
        self.equity_samples = equity_samples
        self.raise_equity = raise_equity

    def _equity(self, view: SeatView) -> float:
        if view.public.street is Street.PREFLOP:
            return 1.0 - 0.6 * preflop_percentile(*view.hole_cards)  # rank-based proxy
        return equity_vs_random(view.hole_cards, view.public.board, self._rng, self.equity_samples)

    def action_distribution(self, view: SeatView) -> ActionDistribution:
        legal = require_actor(view)
        equity = self._equity(view)
        pot = view.public.pot
        if legal.can_bet_or_raise and equity >= self.raise_equity:
            to = clamp_raise(legal, view.public.max_street_bet + 0.75 * (pot + legal.call_amount))
            return ActionDistribution.from_weights(
                {Decision.bet_raise(to): 0.8, Decision.check_call(): 0.2}
            )
        if legal.can_check:
            return ActionDistribution.pure(Decision.check_call())
        price = legal.call_amount / (pot + legal.call_amount)
        if equity > price + 0.05:
            return ActionDistribution.pure(Decision.check_call())
        return ActionDistribution.pure(Decision.fold())

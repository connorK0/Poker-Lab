"""Win-rate evaluation with confidence intervals, plus variance-reduction hooks.

Poker results are extremely noisy (per-hand SD ~ 8-12 big blinds in no-limit), so naive win
rates need tens of thousands of hands for tight intervals. Phase 5 will rely on variance
reduction:

* duplicate dealing: play each deal at n tables with the n players' seats rotated, so every
  player holds every seat's cards once per deal and card luck largely cancels in the per-deal
  mean (exactly for copies of one deterministic agent that ignores player ids, since the
  seats' results sum to zero). ``poker_lab.engine.duplicate`` deals the matches;
  ``DuplicateDealEvaluator`` implements the estimator over the per-deal means. Simulation
  only: a live table cannot deal the same cards twice.
* AIVAT (Burch et al., 2018): subtract control variates built from a value function and the
  known strategies of chance and of the hero. ``AIVATEvaluator`` applies them;
  ``poker_lab.evaluation.aivat`` builds them for games compiled to a tree (P2.11).
  TODO(Phase 5): hold'em needs a value function over its states, and full AIVAT needs the
  opponents' cards from every hand (P2.11 measures the hero's own view without them).
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from poker_lab.evaluation.intervals import t_interval
from poker_lab.schema import HandHistory


@dataclass(frozen=True)
class HandResult:
    hand_id: str
    player_id: str
    net_chips: int
    big_blind: int

    @property
    def net_bb(self) -> float:
        return self.net_chips / self.big_blind


def hand_results(hands: Iterable[HandHistory], player_id: str) -> list[HandResult]:
    """Per-hand net results of ``player_id`` in the completed hands where they were seated."""
    out = []
    for hand in hands:
        if not hand.is_complete:
            continue
        try:
            seat = hand.seat_of(player_id)
        except KeyError:
            continue
        out.append(
            HandResult(hand.hand_id, player_id, hand.net_results()[seat], hand.table.big_blind)
        )
    return out


@dataclass(frozen=True)
class WinRateEstimate:
    bb_per_100: float
    lower: float
    upper: float
    n_hands: int
    sd_bb_per_hand: float
    method: str
    confidence: float = 0.95

    @property
    def half_width(self) -> float:
        return (self.upper - self.lower) / 2


class Evaluator(ABC):
    """Turns per-hand results into a win-rate estimate with a confidence interval."""

    @abstractmethod
    def evaluate(self, results: Sequence[HandResult]) -> WinRateEstimate: ...


class WinRateEvaluator(Evaluator):
    """Mean big blinds per 100 hands with a Student-t interval over hands.

    Hands at a table are treated as independent draws (stacks reset every hand in our
    simulation, so there is no carry-over between hands).
    """

    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = confidence

    def evaluate(self, results: Sequence[HandResult]) -> WinRateEstimate:
        bb = np.array([r.net_bb for r in results], dtype=float)
        if len(bb) < 2:
            raise ValueError("need at least two hands")
        iv = t_interval(bb, self.confidence)
        return WinRateEstimate(
            bb_per_100=100 * iv.estimate,
            lower=100 * iv.lower,
            upper=100 * iv.upper,
            n_hands=len(bb),
            sd_bb_per_hand=float(bb.std(ddof=1)),
            method="naive_t",
            confidence=self.confidence,
        )

    @staticmethod
    def hands_for_half_width(sd_bb_per_hand: float, half_width_bb_per_100: float) -> int:
        """Hands needed for a 95% interval of the given half-width (normal approximation)."""
        se_per_hand = half_width_bb_per_100 / 100 / 1.96
        return int(np.ceil((sd_bb_per_hand / se_per_hand) ** 2))


class DuplicateDealEvaluator(Evaluator):
    """Grouped estimator for duplicate matches, a Student-t interval over per-deal means.

    Each group of ``group_size`` consecutive results holds the hero's results on the same cards
    from the seatings of one deal, so the per-deal mean removes most card luck. ``evaluate``
    expects results ordered deal by deal, as ``DuplicateMatch.grouped_results`` gives them:
    [a0, b0, a1, b1, ...] heads-up (``group_size=2``, the paired estimator).
    ``sd_bb_per_hand`` is the SD of the raw per-hand results, as for ``WinRateEvaluator``, not
    the spread of the per-deal means behind the interval.
    """

    def __init__(self, confidence: float = 0.95, group_size: int = 2) -> None:
        group_size = operator.index(group_size)  # an integer, not 2.0
        if group_size < 2:
            raise ValueError(f"a duplicate deal needs group_size >= 2 seatings, not {group_size}")
        self.confidence = confidence
        self.group_size = group_size

    def evaluate(self, results: Sequence[HandResult]) -> WinRateEstimate:
        k = self.group_size
        if len(results) % k:
            if k == 2:
                raise ValueError("duplicate results must come in pairs [a0, b0, a1, b1, ...]")
            raise ValueError(f"duplicate results must come in groups of {k}, one per deal")
        bb = np.array([r.net_bb for r in results], dtype=float).reshape(-1, k)
        per_deal = bb.mean(axis=1)
        iv = t_interval(per_deal, self.confidence)
        return WinRateEstimate(
            bb_per_100=100 * iv.estimate,
            lower=100 * iv.lower,
            upper=100 * iv.upper,
            n_hands=len(results),
            sd_bb_per_hand=float(bb.std(ddof=1)),
            method="duplicate_paired_t" if k == 2 else "duplicate_grouped_t",
            confidence=self.confidence,
        )


class AIVATEvaluator(Evaluator):
    """AIVAT-style variance reduction.

    ``correction(hand_result) -> float`` must return the sum of control-variate terms (in big
    blinds) for that hand: the result minus the AIVAT estimate. AIVAT builds them from
    chance-node and hero-decision values under a value function; each term has zero expected
    value, so the estimate stays unbiased. For games compiled to a tree,
    ``poker_lab.evaluation.aivat.control_variates`` gives the estimate at every terminal.
    """

    def __init__(self, correction: Callable[[HandResult], float] | None = None) -> None:
        self.correction = correction

    def evaluate(self, results: Sequence[HandResult]) -> WinRateEstimate:
        if self.correction is None:
            raise NotImplementedError(
                "TODO(Phase 5): AIVAT for hold'em needs a value function over its states; "
                "small games get theirs from poker_lab.evaluation.aivat."
            )
        adjusted = np.array([r.net_bb - self.correction(r) for r in results], dtype=float)
        iv = t_interval(adjusted)
        return WinRateEstimate(
            bb_per_100=100 * iv.estimate,
            lower=100 * iv.lower,
            upper=100 * iv.upper,
            n_hands=len(results),
            sd_bb_per_hand=float(adjusted.std(ddof=1)),
            method="aivat_control_variates",
        )

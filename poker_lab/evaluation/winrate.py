"""Win-rate evaluation with confidence intervals, plus variance-reduction hooks.

Poker results are extremely noisy (per-hand SD ~ 8-12 big blinds in no-limit), so naive win
rates need tens of thousands of hands for tight intervals. Phase 5 will rely on variance
reduction:

* duplicate dealing: replay the same cards with seats swapped, so card luck cancels in the
  paired difference. ``DuplicateDealEvaluator`` implements the paired estimator.
  TODO(Phase 5): engine hook to replay a recorded deck sequence with permuted seats.
* AIVAT (Burch et al., 2018): subtract control variates built from a value function and the
  known strategies of chance and of the hero. ``AIVATEvaluator`` is the interface.
  TODO(Phase 2/5): needs Phase 2 value estimates; refuses to run until they exist.
"""

from __future__ import annotations

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
    """Paired estimator for duplicate matches.

    ``pairs[i]`` holds the hero's results on the same cards from the two seatings, so the
    per-pair mean removes most card luck. ``evaluate`` expects results ordered as
    [a0, b0, a1, b1, ...].
    """

    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = confidence

    def evaluate(self, results: Sequence[HandResult]) -> WinRateEstimate:
        if len(results) % 2:
            raise ValueError("duplicate results must come in pairs [a0, b0, a1, b1, ...]")
        bb = np.array([r.net_bb for r in results], dtype=float).reshape(-1, 2)
        per_pair = bb.mean(axis=1)
        iv = t_interval(per_pair, self.confidence)
        return WinRateEstimate(
            bb_per_100=100 * iv.estimate,
            lower=100 * iv.lower,
            upper=100 * iv.upper,
            n_hands=len(results),
            sd_bb_per_hand=float(bb.std(ddof=1)),
            method="duplicate_paired_t",
            confidence=self.confidence,
        )


class AIVATEvaluator(Evaluator):
    """AIVAT-style variance reduction (interface only).

    ``correction(hand_result) -> float`` must return the sum of control-variate terms (in big
    blinds) for that hand. AIVAT builds them from chance-node and hero-decision values under
    a value function; each term has zero expected value, so the estimate stays unbiased.
    """

    def __init__(self, correction: Callable[[HandResult], float] | None = None) -> None:
        self.correction = correction

    def evaluate(self, results: Sequence[HandResult]) -> WinRateEstimate:
        if self.correction is None:
            raise NotImplementedError(
                "TODO(Phase 2/5): AIVAT needs a value function over public states (Phase 2) "
                "to build zero-mean control variates for chance and hero decisions."
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

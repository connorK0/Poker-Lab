"""Profile feature vectors and sample-size windows.

A profile feature vector is the shrunk (or, for the ablation, raw) stat vector of
``EmpiricalBayesShrinker.names`` plus ``log(hands observed)``. Tree models use the features
as-is. The linear model uses ``to_model_space``: logit for rates, log for sizes.

Windows: for the hands-to-identify experiments a player's hand sequence is cut into disjoint
consecutive windows of ``n`` hands. Bots are stationary, so each window is an independent
sample of ``n`` hands from that player.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

from poker_lab.profiling.shrinkage import CONTINUOUS_STATS, EmpiricalBayesShrinker
from poker_lab.profiling.stats import BINOMIAL_STATS, C

N_RATE_FEATURES = len(BINOMIAL_STATS)
N_SIZE_FEATURES = len(CONTINUOUS_STATS)


def feature_names(shrinker: EmpiricalBayesShrinker) -> tuple[str, ...]:
    return (*shrinker.names, "log_hands")


def profile_features(
    shrinker: EmpiricalBayesShrinker, counts: np.ndarray, *, shrink: bool = True
) -> np.ndarray:
    """Feature matrix (..., n_features) for counter totals (..., N_COUNTERS)."""
    counts = np.asarray(counts)
    stats = shrinker.transform(counts).mean if shrink else shrinker.raw(counts)
    log_n = np.log(np.maximum(counts[..., C["hands"]], 1.0))[..., None]
    return np.concatenate([stats, log_n], axis=-1)


def to_model_space(features: np.ndarray) -> np.ndarray:
    """Logit for rate columns, log for size columns, unchanged log-hands column."""
    x = np.array(features, dtype=float, copy=True)
    r = N_RATE_FEATURES
    x[..., :r] = special.logit(np.clip(x[..., :r], 0.005, 0.995))
    x[..., r : r + N_SIZE_FEATURES] = np.log(np.clip(x[..., r : r + N_SIZE_FEATURES], 0.05, 50))
    return x


@dataclass(frozen=True)
class Windows:
    """Disjoint windows of ``n`` hands: ``player[i]`` is the player index (into the counters
    array) of window ``i`` and ``totals[i]`` its summed counters."""

    n: int
    player: np.ndarray
    index: np.ndarray
    totals: np.ndarray


def window_totals(cumulative: np.ndarray, n: int, max_windows: int, offset: int = 0) -> Windows:
    """Disjoint windows ``[offset + w*n, offset + (w+1)*n)`` for every player.

    ``cumulative`` is ``counters.cumsum(axis=1)`` with shape (players, hands, counters).
    """
    n_players, n_hands, _ = cumulative.shape
    k = min(max_windows, (n_hands - offset) // n)
    if k < 1:
        raise ValueError(f"not enough hands for a window of {n} (have {n_hands - offset})")
    zero = np.zeros((n_players, 1, cumulative.shape[2]), dtype=cumulative.dtype)
    padded = np.concatenate([zero, cumulative], axis=1)  # padded[:, j] = sum of first j hands
    starts = offset + n * np.arange(k)
    totals = padded[:, starts + n] - padded[:, starts]  # (players, k, counters)
    return Windows(
        n=n,
        player=np.repeat(np.arange(n_players), k),
        index=np.tile(np.arange(k), n_players),
        totals=totals.reshape(n_players * k, -1),
    )


def prefix_totals(cumulative: np.ndarray, n: int) -> np.ndarray:
    """Counter totals over each player's first ``n`` hands (zeros for n = 0)."""
    if n == 0:
        return np.zeros((cumulative.shape[0], cumulative.shape[2]), dtype=cumulative.dtype)
    return cumulative[:, n - 1]

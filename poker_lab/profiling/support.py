"""Support check: does a profile look like any archetype the classifier was trained on?

A classifier trained on synthetic archetypes returns probabilities over those archetypes for
any input, including a player who resembles none of them. Its calibration was measured on the
training population, so it says nothing about such a player. ``SupportCheck`` flags them.

Per training sample size: one Gaussian per archetype in the classifier's model space (logit
rates, log sizes; ``to_model_space``), fit on training windows of that size (a class with
fewer rows than dimensions + 2 at a size uses the pooled within-class covariance). A profile's
score is its smallest squared Mahalanobis distance to an archetype,

    d^2(x) = min_k (x - mu_k)' S_k^{-1} (x - mu_k).

The threshold is split-conformal. A grouped share of the training players is held out (all of a
player's windows on one side, as for the classifier's temperatures), and a profile's p-value is

    p(x) = (1 + #{held-out windows of its size with d^2 >= d^2(x)}) / (m + 1),

with ``m`` held-out windows of that size. A profile is flagged when ``p <= alpha``. A new window
from the training population is then flagged with probability at most ``alpha``, whatever the
shape of the archetypes' real distributions (the Gaussians only order the profiles; the
guarantee needs the new window and the held-out ones to be exchangeable). With fewer than
``1 / alpha - 1`` held-out windows at a size nothing can be flagged there. A profile whose number
of hands lies between training sizes uses the nearest size in log(hands).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from poker_lab.profiling.features import to_model_space


def _mahalanobis_min(Z: np.ndarray, components: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    d2 = [np.einsum("ij,jk,ik->i", Z - mu, prec, Z - mu) for mu, prec in components]
    return np.min(d2, axis=0)


class SupportCheck:
    """Flags profile feature rows (``profile_features`` layout) unlike every training class."""

    def __init__(
        self,
        alpha: float = 0.05,
        *,
        holdout_fraction: float = 0.2,
        ridge: float = 1e-6,
        random_state: int = 0,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.holdout_fraction = holdout_fraction
        self.ridge = ridge
        self.random_state = random_state
        self.sizes = np.empty(0)  # log(hands) of each training size
        self._components: list[list[tuple[np.ndarray, np.ndarray]]] = []
        self._held_out: list[np.ndarray] = []  # sorted held-out scores per size
        self.fit_players: set = set()
        self.held_out_players: set = set()

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> SupportCheck:
        """``X``: profile feature rows (last column ``log_hands``); ``y``: class of each row;
        ``groups``: player of each row (all of a player's rows land on one side of the split)."""
        X, y, groups = np.asarray(X, float), np.asarray(y), np.asarray(groups)
        split = GroupShuffleSplit(
            n_splits=1, test_size=self.holdout_fraction, random_state=self.random_state
        )
        fit_idx, cal_idx = next(split.split(X, y, groups))
        self.fit_players = set(groups[fit_idx].tolist())
        self.held_out_players = set(groups[cal_idx].tolist())
        Z = to_model_space(X)[:, :-1]
        log_n = np.round(X[:, -1], 6)
        self.sizes = np.unique(log_n)
        self._components, self._held_out = [], []
        dim = Z.shape[1]
        for size in self.sizes:
            fit_rows = fit_idx[log_n[fit_idx] == size]
            groups_k = [Z[fit_rows[y[fit_rows] == k]] for k in np.unique(y[fit_rows])]
            centered = np.concatenate([rows - rows.mean(axis=0) for rows in groups_k])
            if len(centered) < dim + len(groups_k) + 1:
                raise ValueError(
                    f"{len(centered)} rows at log(hands) = {size}; need {dim + len(groups_k) + 1}"
                )
            pooled = centered.T @ centered / (len(centered) - len(groups_k))
            components = []
            for rows in groups_k:
                # a class with too few rows for its own covariance uses the pooled one
                cov = np.cov(rows, rowvar=False) if len(rows) >= dim + 2 else pooled.copy()
                cov += (self.ridge * np.trace(cov) / dim + 1e-12) * np.eye(dim)
                components.append((rows.mean(axis=0), np.linalg.inv(cov)))
            self._components.append(components)
            held = cal_idx[log_n[cal_idx] == size]
            self._held_out.append(np.sort(_mahalanobis_min(Z[held], components)))
        return self

    def _size_index(self, X: np.ndarray) -> np.ndarray:
        if not len(self.sizes):
            raise RuntimeError("SupportCheck is not fitted")
        return np.abs(X[:, -1][:, None] - self.sizes[None, :]).argmin(axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Smallest squared Mahalanobis distance to a class, at the nearest training size."""
        X = np.atleast_2d(np.asarray(X, float))
        Z = to_model_space(X)[:, :-1]
        which = self._size_index(X)
        out = np.empty(len(X))
        for i in np.unique(which):
            rows = which == i
            out[rows] = _mahalanobis_min(Z[rows], self._components[i])
        return out

    def pvalue(self, X: np.ndarray) -> np.ndarray:
        """Conformal p-value of each row against the held-out windows of its size."""
        X = np.atleast_2d(np.asarray(X, float))
        scores, which = self.score(X), self._size_index(X)
        out = np.empty(len(X))
        for i in np.unique(which):
            rows = which == i
            held = self._held_out[i]
            at_least = len(held) - np.searchsorted(held, scores[rows], side="left")
            out[rows] = (1 + at_least) / (len(held) + 1)
        return out

    def flags(self, X: np.ndarray) -> np.ndarray:
        """True where a profile is unlike every training class (p-value <= alpha)."""
        return self.pvalue(X) <= self.alpha

    def held_out_windows(self) -> dict[int, int]:
        """Held-out windows per training size (hands -> count)."""
        return {
            round(float(np.exp(s))): len(h) for s, h in zip(self.sizes, self._held_out, strict=True)
        }

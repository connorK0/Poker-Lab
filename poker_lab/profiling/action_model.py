"""Next-action prediction: P(fold, check/call, bet/raise | public situation [, profile]).

The same learner is trained on different inputs:

* situation only: the population-average baseline (what you predict knowing nothing about
  the player);
* situation + profile features built from the player's *earlier* hands: the opponent model;
* situation + true parameters: an oracle ceiling, available only in simulation.

The metric is multiclass log-loss on held-out players' decisions. Phase 5 reuses it on real
hand histories.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from poker_lab.profiling.situations import ACTION_CLASSES


class ActionPredictor:
    def __init__(self, *, random_state: int = 0, max_iter: int = 250) -> None:
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=max_iter,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> ActionPredictor:
        self.model.fit(X, y)
        if list(self.model.classes_) != list(range(len(ACTION_CLASSES))):
            raise ValueError("training data must contain every action class")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def marginal_baseline(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    """Unconditional class frequencies of the training population, repeated per row."""
    freq = np.bincount(y_train, minlength=len(ACTION_CLASSES)) / len(y_train)
    return np.tile(freq, (n_rows, 1))

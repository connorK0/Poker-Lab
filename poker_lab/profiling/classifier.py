"""Archetype classifier with calibrated probabilities.

``ArchetypeClassifier`` wraps a multinomial logistic regression (on standardized logit/log
features) or a histogram gradient-boosting classifier. With ``calibrate=True`` a fraction of
the *training players* (grouped split, so all of a player's windows land on one side) is held
out to fit a single temperature ``T`` that rescales the log-probabilities:
``p = softmax(log p_model / T)``. Temperature scaling changes confidence but never the argmax,
so accuracy is unaffected while log-loss and calibration error can improve.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from poker_lab.profiling.features import to_model_space

MODELS = ("lr", "gbm")


def _make_model(kind: str, random_state: int, feature_space: str = "profile"):
    if kind == "lr":
        transform = to_model_space if feature_space == "profile" else None
        return make_pipeline(
            FunctionTransformer(transform),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=5000),
        )
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=random_state,
        )
    raise ValueError(f"unknown model {kind!r}; choose from {MODELS}")


def fit_temperature(log_proba: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimizing the negative log-likelihood of ``softmax(log_proba / T)``."""

    def nll(log_t: float) -> float:
        z = special.log_softmax(log_proba / np.exp(log_t), axis=1)
        return -float(z[np.arange(len(y)), y].mean())

    res = optimize.minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


class ArchetypeClassifier:
    """Predicts archetype probabilities from profile features."""

    def __init__(
        self,
        model: str = "lr",
        *,
        calibrate: bool = True,
        calibration_fraction: float = 0.2,
        random_state: int = 0,
        feature_space: str = "profile",
    ) -> None:
        """``feature_space="profile"`` expects ``profile_features`` columns (the linear model then
        uses logit/log transforms); ``"identity"`` uses the columns as given (e.g. true params)."""
        if feature_space not in ("profile", "identity"):
            raise ValueError("feature_space must be 'profile' or 'identity'")
        self.kind = model
        self.feature_space = feature_space
        self.calibrate = calibrate
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state
        self.temperature = 1.0
        self.model = None
        self.n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> ArchetypeClassifier:
        """Fit on feature rows ``X`` with labels ``y``; ``groups`` = player of each row."""
        self.n_classes = int(y.max()) + 1
        fit_idx = np.arange(len(y))
        cal_idx = None
        if self.calibrate:
            split = GroupShuffleSplit(
                n_splits=1, test_size=self.calibration_fraction, random_state=self.random_state
            )
            fit_idx, cal_idx = next(split.split(X, y, groups))
        self.model = _make_model(self.kind, self.random_state, self.feature_space).fit(
            X[fit_idx], y[fit_idx]
        )
        if cal_idx is not None:
            self.temperature = fit_temperature(self._log_proba_uncalibrated(X[cal_idx]), y[cal_idx])
        return self

    def _log_proba_uncalibrated(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(X)
        return np.log(np.clip(proba, 1e-12, 1.0))

    def predict_proba(self, X: np.ndarray, *, calibrated: bool = True) -> np.ndarray:
        log_p = self._log_proba_uncalibrated(X)
        t = self.temperature if calibrated else 1.0
        return special.softmax(log_p / t, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

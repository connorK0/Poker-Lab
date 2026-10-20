"""Regression of the continuous bot parameters from profile features.

Gradient boosting (one regressor per parameter) or ridge regression (multi-output, on
standardized logit/log features). ``oos_r2`` scores predictions against the *training-mean*
predictor: 0 means no better than guessing the population average, 1 is perfect.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from poker_lab.profiling.features import to_model_space


class ParamRegressor:
    def __init__(self, model: str = "gbm", *, random_state: int = 0) -> None:
        if model not in ("gbm", "ridge"):
            raise ValueError("model must be 'gbm' or 'ridge'")
        self.kind = model
        self.random_state = random_state
        self.models: list = []
        self.train_mean: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> ParamRegressor:
        self.train_mean = Y.mean(axis=0)
        if self.kind == "ridge":
            m = make_pipeline(FunctionTransformer(to_model_space), StandardScaler(), Ridge(1.0))
            self.models = [m.fit(X, Y)]
        else:
            self.models = [
                HistGradientBoostingRegressor(
                    learning_rate=0.06,
                    max_iter=250,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=self.random_state,
                ).fit(X, Y[:, j])
                for j in range(Y.shape[1])
            ]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "ridge":
            return self.models[0].predict(X)
        return np.stack([m.predict(X) for m in self.models], axis=1)


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray, train_mean: np.ndarray) -> np.ndarray:
    """Out-of-sample R^2 per column, relative to predicting the training mean."""
    sse = ((y_true - y_pred) ** 2).sum(axis=0)
    sst = ((y_true - train_mean) ** 2).sum(axis=0)
    return 1.0 - sse / sst

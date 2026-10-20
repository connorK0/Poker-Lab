"""Classification and calibration metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def log_loss(y: np.ndarray, proba: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood (natural log) of the true classes."""
    p = np.clip(proba[np.arange(len(y)), y], eps, 1.0)
    return float(-np.log(p).mean())


def per_row_log_loss(y: np.ndarray, proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -np.log(np.clip(proba[np.arange(len(y)), y], eps, 1.0))


def accuracy(y: np.ndarray, proba: np.ndarray) -> float:
    return float((proba.argmax(axis=1) == y).mean())


def brier(y: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.eye(proba.shape[1])[y]
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def confusion_matrix(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


@dataclass(frozen=True)
class Reliability:
    """Top-label reliability diagram: per confidence bin, mean confidence vs accuracy."""

    bin_edges: np.ndarray
    confidence: np.ndarray  # mean predicted top-label probability per bin (NaN if empty)
    accuracy: np.ndarray  # empirical accuracy per bin
    count: np.ndarray
    ece: float


def reliability(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> Reliability:
    """Equal-width bins on top-label confidence; ECE = sum_b (n_b / N) |acc_b - conf_b|."""
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    count = np.bincount(which, minlength=n_bins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.bincount(which, weights=conf, minlength=n_bins) / count
        acc = np.bincount(which, weights=correct, minlength=n_bins) / count
    ece = float(np.nansum(count / len(y) * np.abs(acc - mean_conf)))
    return Reliability(edges, mean_conf, acc, count.astype(int), ece)


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    return reliability(y, proba, n_bins).ece

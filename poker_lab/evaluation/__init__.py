"""Evaluation: metrics, confidence intervals, win-rate evaluation with variance-reduction hooks."""

from poker_lab.evaluation.intervals import Interval, bootstrap_ci, t_interval, wilson_interval

__all__ = ["Interval", "bootstrap_ci", "t_interval", "wilson_interval"]

"""poker_lab: offline research framework for poker opponent modeling and exploitation.

Subpackages
-----------
schema      Canonical hand-history data model, sealed hand store, firewalled views (the contract
            between all phases).
engine      No-limit hold'em simulation wrapper around PokerKit; the ``Agent`` interface.
bots        Parameterized archetype bots with ground truth (Phase 1).
profiling   Stats with opportunity counting, empirical-Bayes shrinkage, type classifier,
            parameter regression, action prediction, ``ProfileStore`` (Phase 1).
strategy    ``StrategyProvider`` interface; GTO/CFR goes here (Phase 2).
exploit     ``ExploitEngine`` interface (Phase 2/3).
evaluation  Metrics, confidence intervals, win-rate ``Evaluator`` with variance-reduction hooks.
advisor     Firewalled advice entry point (Phase 3/4).
capture     Camera + manual action-log ingestion interface (Phase 4).

This project is for offline research only: it has no integration with online poker sites and
provides no real-time assistance for real-money online play.
"""

__version__ = "0.1.0"

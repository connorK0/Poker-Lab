"""Push/fold mode: the model learns your ranges from completed hands and exploits them (P3.4-P3.5).

You and the model play heads-up push/fold at a fixed effective stack of 5-20 bb (blinds 50/100,
stacks reset every hand). The model is the P3.4 pipeline with you as its opponent:

* **Prior.** ``PushFoldPrior.fit`` on 20 push/fold bots per archetype (P3.2's population, fixed
  seed): before it has seen you, the model expects you to play like a bot from that population.
* **Evidence.** Your decisions from completed hands only, each hand redacted to what the model's
  seat could see (``push_fold_observations``): your cards count only when you showed them.
* **Recommendation.** ``PushFoldRecommender`` ranks the equilibrium, restricted Nash responses
  and the best response by the lower end of a 90% interval on their gain over the equilibrium
  against you. With the P3.5 model-fit checks calibrated for this stack, a failed stability check
  makes it play the equilibrium (P3.5b's default fallback).
* **Refit schedule.** After every completed hand, on all your hands at this stack (earlier
  sessions too, unless the session starts a fresh model). P3.9's *refit* hero kept 94% of the
  clairvoyant's gain against static bots and stayed within 0.002 bb/hand of equilibrium play
  against bots that adapt; a refit costs about 0.15 s. *Refit on recent hands* is not used: in
  P3.9 it changed the gain by about 0.001 bb/hand either way, in a short session it discards most
  of the evidence, and the stability check already falls back to the equilibrium when your play
  changes. The checks then run after every hand against
  thresholds calibrated for a single look, so false fallbacks are more frequent than 5%.

The model plays the advised strategy, sampled with the table's random numbers. Its numbers
assume you play like the model's family (a range of hands ordered by equity with some noise);
the fit checks are the only test of that, and the report says so.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from poker_lab.engine.agent import Agent, Decision
from poker_lab.exploit.recommend import PushFoldRecommender, RecommendationReport
from poker_lab.games.preflop import CLASSES
from poker_lab.games.push_fold import match_values, range_share, strategy_exploitability
from poker_lab.games.push_fold_bots import ARCHETYPES, PushFoldPrior, sample_params
from poker_lab.games.push_fold_checks import FitChecker, calibration_pvalues
from poker_lab.profiling.push_fold import PushFoldHistory, push_fold_observations
from poker_lab.schema import HandHistory, SeatView, TableConfig, redact_hole_cards
from poker_lab.strategy.push_fold import PushFoldStrategy, fixed_ranges, nash_ranges

SMALL_BLIND, BIG_BLIND = 50, 100
MIN_STACK, MAX_STACK = 5, 20
MODEL_ID, HUMAN_ID = "model", "you"
PRIOR_SEED = 2026_11_09
PRIOR_PER_ARCHETYPE = 20


def table() -> TableConfig:
    return TableConfig("pf", 2, SMALL_BLIND, BIG_BLIND)


@dataclass(frozen=True)
class CalibrationConfig:
    """Model-fit check calibration (P3.5's method, fewer bots than an experiment)."""

    bots: int = 200
    sizes: tuple[int, ...] = (10, 25, 50, 100, 200, 400, 800)
    draws: int = 400  # posterior predictive replicates per check
    posterior_draws: int = 2000
    level: float = 0.05
    workers: int = 4
    seed: int = 39

    def cache_name(self, stack: int) -> str:
        sizes = "-".join(map(str, self.sizes))
        return (
            f"pf-thresholds-{stack}bb-b{self.bots}-s{sizes}-d{self.draws}-"
            f"p{self.posterior_draws}-l{self.level:g}-r{self.seed}.json"
        )


@cache
def population_prior() -> PushFoldPrior:
    rng = np.random.default_rng(PRIOR_SEED)
    return PushFoldPrior.fit(
        [sample_params(a, rng) for a in ARCHETYPES for _ in range(PRIOR_PER_ARCHETYPE)]
    )


def _calibrate_one(stack: int, k: int, cfg: CalibrationConfig) -> dict[int, dict[str, float]]:
    rng = np.random.default_rng([cfg.seed, stack, k])
    return calibration_pvalues(
        population_prior(),
        nash_ranges(float(stack)),
        cfg.sizes,
        rng,
        draws=cfg.draws,
        posterior_draws=cfg.posterior_draws,
    )


def calibrate_thresholds(
    stack: int, cfg: CalibrationConfig, progress: Callable[[int, int], None] | None = None
) -> dict[int, dict[str, float]]:
    """Fit-check thresholds at ``stack``: p-values of ``cfg.bots`` bots drawn from the prior and
    played by the equilibrium (the model is right by construction), then the ``level`` quantile.
    The results do not depend on the number of workers."""
    workers = min(cfg.workers, os.cpu_count() or 1)
    pvalues: list[dict[int, dict[str, float]]] = []
    if workers <= 1:
        for k in range(cfg.bots):
            pvalues.append(_calibrate_one(stack, k, cfg))
            if progress:
                progress(k + 1, cfg.bots)
    else:
        pool = ProcessPoolExecutor(workers)
        try:
            futures = [pool.submit(_calibrate_one, stack, k, cfg) for k in range(cfg.bots)]
            for k, f in enumerate(futures):
                pvalues.append(f.result())
                if progress:
                    progress(k + 1, cfg.bots)  # may raise to stop: queued work is cancelled
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    return FitChecker.from_pvalues(pvalues, level=cfg.level).thresholds


def _interval(iv) -> dict[str, float]:
    return {"mean": iv.mean, "lower": iv.lower, "upper": iv.upper}


def _probs(x: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in x]


class PushFoldModel:
    """The model's view of you at one stack: redacted history, posterior, recommendation."""

    def __init__(
        self,
        stack: int,
        thresholds: dict[int, dict[str, float]] | None,
        *,
        history: Sequence[HandHistory] = (),
        seed: int = 0,
        draws: int = 2000,
        check_draws: int = 400,
    ) -> None:
        if not MIN_STACK <= stack <= MAX_STACK:
            raise ValueError(f"stack must be {MIN_STACK}-{MAX_STACK} bb")
        self.stack = stack
        checker = FitChecker(thresholds, draws=check_draws, seed=[seed, 30]) if thresholds else None
        self.recommender = PushFoldRecommender(
            float(stack), population_prior(), draws=draws, seed=[seed, 20], checker=checker
        )
        self.history: list[HandHistory] = []
        self.earlier = 0  # hands from earlier sessions
        self.timeline: list[tuple[int, str, bool]] = []  # (decisions seen, advised, fell back)
        for hand in history:
            self._add(hand)
        self.earlier = len(self.history)
        self.refit()

    def _add(self, hand: HandHistory) -> None:
        """Keep ``hand`` as the model's seat saw it: your cards only if you showed them."""
        self.history.append(redact_hole_cards(hand, hand.seat_of(MODEL_ID)))

    def observe(self, hand: HandHistory) -> None:
        """A completed hand (``TableSimulator`` passes it already redacted to the model's seat;
        it is redacted again here, so a full record can never reach the model)."""
        self._add(hand)
        self.refit()

    def refit(self) -> None:
        rec = self.recommender
        self.observed: PushFoldHistory = push_fold_observations(
            self.history, MODEL_ID, HUMAN_ID, stack=float(self.stack)
        )
        log = self.observed.log
        obs = log.observations()
        # recommend(log) in two steps, to keep the posterior for the read (same numbers).
        use_log = log if len(log.action) and rec.checker is not None else None
        self.posterior, fit = rec._evidence(obs, rec.seed, use_log)
        self.report: RecommendationReport = rec._rank(obs, self.posterior, fit, rec.population)
        advised = self.report.advised
        self.ranges = (advised.push, advised.call)
        self.strategy = PushFoldStrategy(fixed_ranges(*self.ranges), name="play_model")
        self.timeline.append((len(log.action), advised.name, self.report.falls_back))

    def read(self) -> dict[str, Any]:
        """Everything the page shows of the model's read of you (plain numbers, JSON-ready)."""
        r = self.report
        obs = r.observations
        adv = r.advised
        names = [x.name for x in r.ranked]
        your_push, your_call = self.posterior.mean_strategy()
        eq_push, eq_call = r.equilibrium
        value = float(match_values(self.stack, adv.push, adv.call, your_push, your_call))
        eq_value = float(match_values(self.stack, eq_push, eq_call, your_push, your_call))
        fit = None
        if r.fit is not None:
            fit = [
                {
                    "name": c.name,
                    "p_value": c.p_value,
                    "threshold": c.threshold,
                    "n": c.n,
                    "failed": c.failed,
                }
                for c in r.fit.results
            ]
        return {
            "stack": self.stack,
            "hands": obs.n_hands,
            "earlier_hands": self.earlier,
            "sb_decisions": obs.n_sb,
            "pushes": obs.pushes,
            "bb_decisions": obs.n_bb,
            "calls": obs.calls,
            "skipped": dict(self.observed.skipped),
            "push_share": _interval(r.push_share),
            "call_share": _interval(r.call_share),
            "eq_push_share": range_share(eq_push),
            "eq_call_share": range_share(eq_call),
            "your_push": _probs(your_push),
            "your_call": _probs(your_call),
            "eq_push": _probs(eq_push),
            "eq_call": _probs(eq_call),
            "model_push": _probs(adv.push),
            "model_call": _probs(adv.call),
            "level": r.level,
            "advised": {
                "name": adv.name,
                "label": adv.label,
                "rank": names.index(adv.name) + 1,
                "p": adv.p,
                "gain": _interval(adv.gain),
                "prob_gain": adv.prob_gain,
                "exploitability": adv.exploitability,
            },
            "ranking": [
                {
                    "name": x.name,
                    "label": x.label,
                    "gain": _interval(x.gain),
                    "prob_gain": x.prob_gain,
                    "exploitability": x.exploitability,
                }
                for x in r.ranked
            ],
            "fit": fit,
            "checks_calibrated": self.recommender.checker is not None,
            "falls_back": r.falls_back,
            "value_vs_you": value,
            "eq_value_vs_you": eq_value,
            "your_exploitability": float(strategy_exploitability(self.stack, your_push, your_call)),
            "explanation": r.explanation(),
            "refits": len(self.timeline),
        }


class ModelAgent(Agent):
    """The model's seat: plays the advised ranges, learns after each completed hand."""

    def __init__(self, model: PushFoldModel) -> None:
        self.model = model

    def act(self, view: SeatView, rng: random.Random) -> Decision:
        return self.model.strategy.action_distribution(view).sample(rng)

    def hand_completed(self, hand: HandHistory) -> None:
        self.model.observe(hand)


def classes() -> list[str]:
    return list(CLASSES)

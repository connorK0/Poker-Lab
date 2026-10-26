"""Push/fold opponents that adapt to the hero, and sessions that score the adaptation (P3.9).

P3.2-P3.5 judged a hero's strategy against a bot that keeps its strategy, so every value was an
exact head-to-head number. A real opponent answers an exploit: push much wider against someone
who folds too often, and they start calling wider. This module makes bots that do that and plays
sessions in which both sides change.

**Adaptive bots.** A bot starts from its P3.2 parameters and watches the hero the way a player
with a HUD would: how often the hero pushes as the small blind and how often it calls a push as
the big blind. It never sees the hero's cards. Every ``period`` hands:

1. *Rates.* Exponentially weighted averages with half-life ``half_life`` hands: each hand, the old
   counts decay by ``0.5 ** (1 / half_life)`` and the new event is added. They start from
   ``pseudo`` hands at the equilibrium's rates, so the first estimates are not wild.
2. *A model of the hero.* Soft thresholds in the equity order at the population's typical
   temperature that cover those shares of hands (``width_for_share``, the same inversion as the
   "HUD stats" model of P3.2, applied to the hero).
3. *Best response.* Per hand class, push when pushing beats folding against the modeled hero's
   calls, call when calling beats folding against its pushes (``push_fold.hand_gains``). The
   target widths are the widths, at the bot's own temperature, that cover the same shares.
4. *A step.* ``w <- w + speed * (target - w)`` for the widths in ``adapts`` (both by default;
   ``("call",)`` is the player who only calls wider after the hero pushes often, keeping its
   own pushing habits); the temperature stays.

``speed = 0`` is a static bot. The bot stays inside the family the hero's model assumes at every
moment: only its widths move, so what the hero's model gets wrong is that it moves at all.

**Sessions.** ``play_session`` plays a bot and a hero for ``n_hands`` in blocks of the bot's
``period``. Before each block the hero chooses its per-class push and call probabilities from a
*redacted* log of the completed hands (the bot's class only where it was shown at a showdown:
the information firewall in miniature); the block is played (``push_fold_bots.simulate``); then
the bot observes the hero's actions and adapts. The random numbers depend only on the session's
generator and the block sizes, so two heroes facing the same bot from the same seed see the same
deals and the same uniform draws (common random numbers).

**Values.** Each block is scored by the exact expected value of the two strategies in force,
both seats averaged (``push_fold.match_values``, bb/hand), not by the chips that changed hands.
That removes the luck of the cards while keeping the path the adaptation took, which depends on
the actions sampled. A session's value is the mean over its hands.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from poker_lab.games.preflop import Matchups, load_matchups
from poker_lab.games.push_fold import hand_gains, match_values, range_share
from poker_lab.games.push_fold_bots import (
    TEMPERATURE_MEDIAN,
    HandLog,
    PushFoldBotParams,
    simulate,
    strategy,
    width_for_share,
)

Strategy = tuple[np.ndarray, np.ndarray]  # per-class push and call probabilities
#: A hero: (redacted log of the completed hands or None, hands played) -> its strategy.
Hero = Callable[[HandLog | None, int], Strategy]


@dataclass(frozen=True)
class Adaptation:
    """How a bot adapts: ``speed`` is the step toward its best response per update (0 static,
    1 jumps to it), ``half_life`` the memory of its rate estimates in hands, ``adapts`` the
    widths it moves."""

    speed: float
    half_life: float = 100.0
    period: int = 25
    pseudo: float = 10.0
    hero_temperature: float = TEMPERATURE_MEDIAN
    adapts: tuple[str, ...] = ("push", "call")

    def __post_init__(self) -> None:
        if not 0.0 <= self.speed <= 1.0:
            raise ValueError("speed must be in [0, 1]")
        if not self.adapts or not set(self.adapts) <= {"push", "call"}:
            raise ValueError("adapts must name 'push', 'call' or both")
        if self.half_life <= 0 or self.period < 1 or self.pseudo <= 0:
            raise ValueError("half_life and pseudo must be positive, period at least 1")


STATIC = Adaptation(0.0)


def best_response_shares(
    stack: float, opp_push: np.ndarray, opp_call: np.ndarray, matchups: Matchups | None = None
) -> tuple[float, float]:
    """Shares of combos a best response pushes and calls against per-class ranges."""
    sb, bb = hand_gains(stack, opp_push, opp_call, matchups)
    return range_share((sb > 0).astype(float)), range_share((np.nan_to_num(bb) > 0).astype(float))


class AdaptiveBot:
    """A push/fold bot that adapts its widths to the hero's observed frequencies."""

    def __init__(
        self,
        params: PushFoldBotParams,
        adaptation: Adaptation,
        stack: float,
        equilibrium: Strategy,
        matchups: Matchups | None = None,
    ) -> None:
        self.params = params
        self.adaptation = adaptation
        self.stack = float(stack)
        self.matchups = matchups or load_matchups()
        a = adaptation
        # [weighted events, weighted opportunities] for the hero's pushes and calls
        self._push = np.array([a.pseudo * range_share(equilibrium[0]), a.pseudo])
        self._call = np.array([a.pseudo * range_share(equilibrium[1]), a.pseudo])
        self._decay = 0.5 ** (1.0 / a.half_life)
        self.history = [params]

    def strategy(self) -> Strategy:
        return strategy(self.params, "equity", self.matchups)

    @property
    def rates(self) -> tuple[float, float]:
        """The bot's current estimates of the hero's push and call rates."""
        return float(self._push[0] / self._push[1]), float(self._call[0] / self._call[1])

    def observe(self, log: HandLog) -> None:
        """Update the rate estimates from a block of hands (the hero's actions only)."""
        for k in range(len(log.action)):
            self._push *= self._decay
            self._call *= self._decay
            if not log.bot_is_sb[k]:  # the hero was the small blind; the bot acted iff it pushed
                self._push += (float(log.acted[k]), 1.0)
            elif log.action[k] == 1:  # the bot pushed; the hero called iff there was a showdown
                self._call += (float(log.shown[k]), 1.0)

    def adapt(self) -> None:
        """One step toward the best response to the modeled hero."""
        a = self.adaptation
        if a.speed == 0.0:
            self.history.append(self.params)
            return
        push_rate, call_rate = self.rates
        t = a.hero_temperature
        hero = strategy(
            PushFoldBotParams(width_for_share(push_rate, t), width_for_share(call_rate, t), t),
            "equity",
            self.matchups,
        )
        push_share, call_share = best_response_shares(self.stack, *hero, self.matchups)
        p = self.params
        step = {"push": 0.0, "call": 0.0} | dict.fromkeys(a.adapts, a.speed)
        target = (
            width_for_share(push_share, p.temperature),
            width_for_share(call_share, p.temperature),
        )
        self.params = replace(
            p,
            push_width=p.push_width + step["push"] * (target[0] - p.push_width),
            call_width=p.call_width + step["call"] * (target[1] - p.call_width),
        )
        self.history.append(self.params)


@dataclass(frozen=True, eq=False)
class Session:
    """A played session: the hands, both sides' strategies per block and the exact values."""

    log: HandLog  # the full log (ground truth; the hero only ever saw ``log.redacted()``)
    block: int
    hero_push: np.ndarray  # (blocks, 169)
    hero_call: np.ndarray
    bot_push: np.ndarray
    bot_call: np.ndarray
    bot_params: tuple[PushFoldBotParams, ...]  # in force during each block
    values: np.ndarray  # (blocks,) the hero's exact expected bb/hand during each block

    @property
    def sizes(self) -> np.ndarray:
        n = len(self.log.action)
        return np.diff(np.minimum(np.arange(0, n + self.block, self.block), n))[: len(self.values)]

    def value(self, start: int = 0, stop: int | None = None) -> float:
        """Mean exact value per hand over hands ``start`` to ``stop - 1`` (block boundaries)."""
        edges = np.arange(len(self.values)) * self.block
        stop = len(self.log.action) if stop is None else stop
        sel = (edges >= start) & (edges < stop)
        w = self.sizes[sel]
        return float(self.values[sel] @ w / w.sum())


def play_session(
    hero: Hero,
    bot: AdaptiveBot,
    n_hands: int,
    rng: np.random.Generator,
) -> Session:
    """Play ``n_hands`` between ``hero`` and ``bot`` (see the module docstring)."""
    block = bot.adaptation.period
    logs: list[HandLog] = []
    rows: dict[str, list[np.ndarray]] = {k: [] for k in ("hp", "hc", "bp", "bc")}
    params, values = [], []
    for start in range(0, n_hands, block):
        size = min(block, n_hands - start)
        seen = HandLog.concat(logs).redacted() if logs else None
        hp, hc = (np.asarray(x, dtype=float) for x in hero(seen, start))
        bp, bc = bot.strategy()
        log = simulate((bp, bc), (hp, hc), size, rng, bot.matchups, first_hand=start)
        for k, x in zip(("hp", "hc", "bp", "bc"), (hp, hc, bp, bc), strict=True):
            rows[k].append(x)
        params.append(bot.params)
        values.append(float(match_values(bot.stack, hp, hc, bp, bc, bot.matchups)))
        logs.append(log)
        bot.observe(log)
        bot.adapt()
    return Session(
        HandLog.concat(logs),
        block,
        np.array(rows["hp"]),
        np.array(rows["hc"]),
        np.array(rows["bp"]),
        np.array(rows["bc"]),
        tuple(params),
        np.array(values),
    )

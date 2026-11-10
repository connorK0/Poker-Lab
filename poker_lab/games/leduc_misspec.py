"""Leduc opponents the five-parameter model cannot describe, and how far outside they are (P2.12).

P2.6-P2.7 fit and exploited bots from ``poker_lab.games.leduc_bots``: the same family the model
assumes, so the model could always be right. This module makes opponents for which it cannot.

**Rule bots (static, outside the family).** A family bot gives every information set in a bucket
the same aggression ``a`` and fold rate ``f``: it ignores its seat, the street and how many bets
it faces, and it reads hand strength through three buckets. A rule bot starts from a family bot's
five parameters and gives each information set its own ``a`` and ``f`` (``rates_policy``):

================  ==================================================================
``board_blind``   buckets by its own rank on both streets, ignoring the board: a pair of
                  jacks plays as a weak hand, an unpaired king as a strong one
``street``        passive and sticky on street 1 (``a / 4``, ``f / 2``), aggressive and
                  honest on street 2 (``1 - (1 - a)^2``, ``1 - (1 - f)^2``)
``positional``    with no bet to face, opens tight as seat 0 (``a / 2``) and stabs as seat 1
                  (``1 - (1 - a)^3``)
``honest_raiser`` bets as usual, raises only with a strong hand, and folds more to a raise
                  (``1 - (1 - f)^3``)
================  ==================================================================

``slight_street`` is a quarter of ``street`` mixed with the family bot at every information set:
the same kind of error at a lower dose. Three more leave the ``a``, ``f`` form altogether:
``noisy`` draws each information set's action probabilities from a Dirichlet distribution around
the family bot's, ``near_eq`` mixes the family bot with the equilibrium at every information set,
and ``equilibrium`` is the equilibrium itself, against which nothing can be gained (the extreme
case: no hero can beat it).

**How far outside (``FamilyProjection``).** The model is fit on what the hero sees, so "the family
member the fit converges to" is defined by that view: the parameters ``theta*`` maximizing the
*expected* log-likelihood of the hero's observations (the bot's card only at showdown), with the
hero playing a fixed strategy. Every outcome of the game is enumerated, so this needs no sampling
and no hands. Two numbers come from it, neither using any hero's result:

* ``kl``: the Kullback-Leibler divergence, in nats per hand, between the observations the bot
  produces and the ones ``theta*`` would. ``log(20) / kl`` is how many hands a likelihood-ratio
  test against the closest family member needs, on average, to reach 20 to 1.
* ``tv``: the mean total-variation distance between the action probabilities of the bot and of
  ``theta*``, per decision the bot actually reaches.

A family bot has both at zero, up to the optimizer's tolerance.

**Adaptive bots.** ``AdaptiveLeducBot`` watches the hero the way the hero watches it: the
actions, and the hero's card only at showdown (``simulate_views`` keeps both views of a hand
apart). Every ``period`` hands it fits the five-parameter model to the hero's last ``window``
hands, with a prior centred on the equilibrium's closest family member worth ``pseudo`` hands,
finds the best response to that model, and takes a step of ``speed`` towards it:

* mode ``family``: towards the best response's own rates (``family_rates``), so the bot stays
  inside the family at every moment, and all the model gets wrong is that it moves;
* mode ``tabular``: towards the best response itself, by mixing realization plans (fictitious
  play), so the bot leaves the family as it moves.

**Sessions.** ``play_session`` is the Leduc version of ``push_fold_adaptive.play_session``. Each
block is scored by the exact head-to-head value of the two strategies in force, which removes the
luck of the cards and keeps the path the adaptation took. Heroes see only their redacted view.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from poker_lab.games.evaluate import best_response, exploitability, head_to_head
from poker_lab.games.leduc import LeducHoldem
from poker_lab.games.leduc_bots import (
    MEDIUM,
    STRONG,
    WEAK,
    BetaPrior,
    BotPolicyBuilder,
    LeducBotParams,
    Observation,
    ObservationIndex,
    _walk_hand,
    fit_params,
)
from poker_lab.games.policy import TabularPolicy
from poker_lab.games.tree import TERMINAL, GameTree
from poker_lab.solvers.robust import realization_plan

RULES = (
    "board_blind",
    "street",
    "slight_street",
    "positional",
    "honest_raiser",
    "noisy",
    "near_eq",
    "equilibrium",
)
SLIGHT_STREET_WEIGHT = 0.25  # slight_street: street's share at every information set
NOISE_CONCENTRATION = 5.0  # noisy: Dirichlet(concentration * family probabilities)
NEAR_EQ_FAMILY_WEIGHT = 0.25  # near_eq: the family bot's share at every information set
EVIDENCE = 20.0  # the odds ``hands_to_evidence`` asks for


@dataclass(frozen=True)
class InfosetFeatures:
    """What a rule may look at, per information set."""

    seat: np.ndarray  # 0 or 1
    second_street: np.ndarray  # bool
    facing: np.ndarray  # bool: a bet or raise to answer
    raises: np.ndarray  # bets and raises so far on this street
    bucket: np.ndarray  # the family's bucket
    rank_bucket: np.ndarray  # the family's bucket from the own card's rank alone


def infoset_features(builder: BotPolicyBuilder) -> InfosetFeatures:
    tree, game = builder.tree, builder.game
    n = tree.num_infosets
    second, facing = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    raises, rank_bucket = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    top = len(game.ranks) - 1
    for i, key in enumerate(tree.infoset_keys):
        cards, history = key.split("|", 1)
        streets = history.split("/")
        current = streets[-1]
        second[i] = len(streets) == 2
        facing[i] = bool(current) and current[-1] in "br"
        raises[i] = sum(a in "br" for a in current)
        own = game.ranks.index(cards[0])
        rank_bucket[i] = WEAK if own == 0 else STRONG if own == top else MEDIUM
    return InfosetFeatures(
        np.asarray(tree.infoset_player, dtype=np.int64),
        second,
        facing,
        raises,
        builder.infoset_bucket,
        rank_bucket,
    )


def rates_policy(builder: BotPolicyBuilder, aggr: np.ndarray, fold: np.ndarray) -> TabularPolicy:
    """A policy with the family's form and each information set's own ``a`` and ``f``."""
    a, f = aggr[builder.tree.seq_infoset], fold[builder.tree.seq_infoset]
    k = builder.kind
    probs = np.select(
        [k == 0, k == 1, k == 2, k == 3, k == 4, k == 5],
        [1 - a, a, f, (1 - f) * (1 - a), 1 - f, (1 - f) * a],
    )
    return TabularPolicy(builder.tree, probs)


def rule_policy(
    rule: str,
    builder: BotPolicyBuilder,
    params: LeducBotParams,
    *,
    features: InfosetFeatures | None = None,
    rng: np.random.Generator | None = None,
    equilibrium: TabularPolicy | None = None,
) -> TabularPolicy:
    """The rule bot ``rule`` built from a family bot (the module docstring gives the rules).

    ``noisy`` needs ``rng``; ``near_eq`` needs ``equilibrium``."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}")
    tree = builder.tree
    theta = params.as_array()
    base = builder.probs(theta)
    if rule == "noisy":
        if rng is None:
            raise ValueError("noisy needs a generator")
        alpha = np.maximum(NOISE_CONCENTRATION * base, 1e-3)
        return TabularPolicy.from_weights(tree, rng.standard_gamma(alpha))
    if rule in ("near_eq", "equilibrium"):
        if equilibrium is None:
            raise ValueError(f"{rule} needs the equilibrium")
        w = NEAR_EQ_FAMILY_WEIGHT if rule == "near_eq" else 0.0
        return TabularPolicy(tree, w * base + (1 - w) * equilibrium.probs)
    x = features or infoset_features(builder)
    aggr_by_bucket = theta[[0, 1, 2]]
    fold_by_bucket = np.array([theta[3], theta[4], 0.0])
    a, f = aggr_by_bucket[x.bucket], fold_by_bucket[x.bucket]
    if rule == "board_blind":
        a, f = aggr_by_bucket[x.rank_bucket], fold_by_bucket[x.rank_bucket]
    elif rule in ("street", "slight_street"):
        a = np.where(x.second_street, 1 - (1 - a) ** 2, a / 4)
        f = np.where(x.second_street, 1 - (1 - f) ** 2, f / 2)
        if rule == "slight_street":
            w = SLIGHT_STREET_WEIGHT
            return TabularPolicy(tree, w * rates_policy(builder, a, f).probs + (1 - w) * base)
    elif rule == "positional":
        opened = np.where(x.seat == 0, a / 2, 1 - (1 - a) ** 3)
        a = np.where(x.facing, a, opened)
    elif rule == "honest_raiser":
        a = np.where(x.facing & (x.bucket != STRONG), 0.0, a)
        f = np.where(x.raises >= 2, 1 - (1 - f) ** 3, f)
    return rates_policy(builder, a, f)


# ---- views of a hand --------------------------------------------------------------------------


def _view(tree: GameTree, game: LeducHoldem, path: Sequence[int], modeled_seat: int) -> Observation:
    """One hand as seen by the player *not* in ``modeled_seat``: the modeled player's card only
    at showdown (the same record as ``leduc_bots.simulate_hands``)."""
    decisions = [n for n in path[1:] if tree.edge_seq[n] >= 0]
    cards = [game.deck.index(tree.labels[n]) for n in path[1:] if tree.edge_seq[n] < 0]
    leaf = path[-1]
    showdown = len(cards) == 3 and not tree.labels[leaf].endswith("f")
    seqs = tuple(int(tree.edge_seq[n]) for n in decisions if tree.edge_player[n] == modeled_seat)
    return Observation(
        bot_seat=modeled_seat,
        hero_card=cards[1 - modeled_seat],
        board=cards[2] if len(cards) == 3 else None,
        actions=tuple(tree.labels[n] for n in decisions),
        bot_card=cards[modeled_seat] if showdown else None,
        bot_sequences=seqs if showdown else (),
        bot_payoff=float(tree.utility[leaf] * (1 if modeled_seat == 0 else -1)),
    )


def simulate_views(
    tree: GameTree,
    game: LeducHoldem,
    hero: TabularPolicy,
    bot: TabularPolicy,
    n_hands: int,
    rng: np.random.Generator,
    first_hand: int = 0,
) -> tuple[list[Observation], list[Observation]]:
    """``n_hands`` hands with the seats alternating by hand number, as the hero saw them (the
    bot's card only at showdown) and as the bot saw them (the hero's card only at showdown). With
    ``first_hand`` 0 the hero's view is what ``simulate_hands`` gives from the same generator."""
    edges = [
        tree.edge_probabilities(hero.combine(bot, 0).probs),  # the bot in seat 1
        tree.edge_probabilities(bot.combine(hero, 0).probs),  # the bot in seat 0
    ]
    hero_view, bot_view = [], []
    for h in range(first_hand, first_hand + n_hands):
        bot_seat = 1 - h % 2
        path = _walk_hand(tree, edges[1 - bot_seat], rng)
        hero_view.append(_view(tree, game, path, bot_seat))
        bot_view.append(_view(tree, game, path, 1 - bot_seat))
    return hero_view, bot_view


# ---- how far outside the family -------------------------------------------------------------


@dataclass(frozen=True)
class Misspecification:
    theta: LeducBotParams  # the closest family member, as the fit sees it
    kl: float  # nats per hand between the observations of the bot and of theta
    tv: float  # mean total variation per decision the bot reaches

    @property
    def hands_to_evidence(self) -> float:
        """Hands for a likelihood-ratio test against ``theta`` to reach 20 to 1, on average."""
        return float(np.log(EVIDENCE) / self.kl) if self.kl > 0 else float("inf")


class FamilyProjection:
    """Exact projection of a strategy onto the family, as seen by a player using ``observer``.

    Every outcome of the game is enumerated once (per seat of the modeled player); a strategy
    only changes the outcomes' probabilities."""

    def __init__(self, builder: BotPolicyBuilder, observer: TabularPolicy) -> None:
        self.builder, self.observer = builder, observer
        tree, game = builder.tree, builder.game
        leaves = np.flatnonzero(tree.player == TERMINAL)
        self.leaves = leaves
        classes: dict[tuple, int] = {}
        observations: list[Observation] = []
        self.leaf_class = np.zeros((2, leaves.size), dtype=np.int64)
        for seat in (0, 1):
            for j, leaf in enumerate(leaves):
                path = [int(leaf)]
                while tree.parent[path[-1]] >= 0:
                    path.append(int(tree.parent[path[-1]]))
                obs = _view(tree, game, path[::-1], seat)
                key = (seat, obs.hero_card, obs.board, obs.actions, obs.bot_card)
                if key not in classes:
                    classes[key] = len(observations)
                    observations.append(obs)
                self.leaf_class[seat, j] = classes[key]
        self.index = ObservationIndex(tree, game, observations)
        self.n_classes = len(observations)

    def _reach(self, policy: TabularPolicy, seat: int) -> np.ndarray:
        """Reach of every node with ``policy`` in ``seat`` against the observer."""
        tree = self.builder.tree
        profile = policy.combine(self.observer, seat)
        return tree.reach(tree.edge_probabilities(profile.probs))

    def _class_weights(self, policy: TabularPolicy) -> np.ndarray:
        w = np.zeros(self.n_classes)
        for seat in (0, 1):
            np.add.at(w, self.leaf_class[seat], 0.5 * self._reach(policy, seat)[self.leaves])
        return w

    def _class_loglik(self, seq_probs: np.ndarray) -> np.ndarray:
        """log P(the modeled player's actions), per observation class."""
        idx = self.index
        logp = np.log(np.clip(seq_probs[idx.seqs], 1e-300, None))
        per_row = np.bincount(idx.rows, weights=logp, minlength=idx.n_rows) + idx.row_logprior
        peak = np.maximum.reduceat(per_row, idx.hand_start)
        total = np.add.reduceat(np.exp(per_row - peak[idx.row_hand]), idx.hand_start)
        return peak + np.log(total)

    def closest(
        self, policy: TabularPolicy, starts: Sequence[float] = (0.5, 0.2)
    ) -> LeducBotParams:
        """The family member maximizing the expected log-likelihood of the observations."""
        weights = self._class_weights(policy)
        probs = self.builder.probs

        def objective(z: np.ndarray) -> float:
            return -float(weights @ self._class_loglik(probs(expit(z))))

        best = None
        for s in starts:
            res = minimize(
                objective,
                np.full(5, logit(s)),
                method="L-BFGS-B",
                bounds=[(-9.0, 9.0)] * 5,
                options={"ftol": 1e-12, "gtol": 1e-7},
            )
            if best is None or res.fun < best.fun:
                best = res
        return LeducBotParams.from_array(np.clip(expit(best.x), 0.0, 1.0))

    def total_variation(self, a: TabularPolicy, b: TabularPolicy) -> float:
        """Mean total variation between ``a`` and ``b`` per decision ``a`` reaches (both seats)."""
        tree = self.builder.tree
        weights = np.zeros(tree.num_infosets)
        for seat in (0, 1):
            reach = self._reach(a, seat)
            nodes = np.flatnonzero(tree.player == seat)
            weights += np.bincount(
                tree.infoset[nodes], weights=reach[nodes], minlength=weights.size
            )
        gaps = 0.5 * tree.segment_sum(np.abs(a.probs - b.probs))
        return float(weights @ gaps / weights.sum())

    def measure(self, policy: TabularPolicy) -> Misspecification:
        theta = self.closest(policy)
        closest = self.builder.policy(theta)
        weights = self._class_weights(policy)
        kl = weights @ (self._class_loglik(policy.probs) - self._class_loglik(closest.probs))
        return Misspecification(theta, max(float(kl), 0.0), self.total_variation(policy, closest))


# ---- adaptive bots -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Adaptation:
    """``speed``: step towards the best response per update (0 static). ``window``: hands of the
    hero it fits. ``period``: hands between updates. ``pseudo``: the prior's weight in hands."""

    speed: float
    window: int = 400
    period: int = 50
    pseudo: float = 10.0
    mode: str = "family"

    def __post_init__(self) -> None:
        if not 0.0 <= self.speed <= 1.0:
            raise ValueError("speed must be in [0, 1]")
        if self.mode not in ("family", "tabular"):
            raise ValueError("mode must be 'family' or 'tabular'")
        if self.window < 1 or self.period < 2 or self.period % 2 or self.pseudo <= 0:
            raise ValueError("window and pseudo must be positive, period positive and even")


STATIC = Adaptation(0.0)


def equilibrium_prior(theta: LeducBotParams, pseudo: float) -> BetaPrior:
    """The prior an adaptive bot puts on the hero: centred on ``theta``, worth ``pseudo`` hands."""
    mean = theta.as_array()
    return BetaPrior(pseudo * mean, pseudo * (1 - mean))


def family_rates(
    builder: BotPolicyBuilder,
    response: TabularPolicy,
    opponent: TabularPolicy,
    fallback: np.ndarray,
) -> np.ndarray:
    """The five parameters of a strategy: its bucket frequencies where it meets ``opponent``, both
    seats counted. A parameter with no opportunity keeps its value from ``fallback``."""
    tree = builder.tree
    weights = np.zeros(tree.num_infosets)
    for seat in (0, 1):
        reach = tree.reach(tree.edge_probabilities(response.combine(opponent, seat).probs))
        nodes = np.flatnonzero(tree.player == seat)
        weights += np.bincount(tree.infoset[nodes], weights=reach[nodes], minlength=weights.size)
    mass = weights[tree.seq_infoset] * response.probs
    k, b = builder.kind, builder.seq_bucket
    theta = np.array(fallback, dtype=float)
    for bucket, slot in ((WEAK, 0), (MEDIUM, 1), (STRONG, 2)):
        here = b == bucket
        opportunities = mass[here & np.isin(k, (0, 1, 3, 5))].sum()
        if opportunities > 0:
            theta[slot] = mass[here & np.isin(k, (1, 5))].sum() / opportunities
    for bucket, slot in ((WEAK, 3), (MEDIUM, 4)):
        here = b == bucket
        opportunities = mass[here & np.isin(k, (2, 3, 4, 5))].sum()
        if opportunities > 0:
            theta[slot] = mass[here & (k == 2)].sum() / opportunities
    return np.clip(theta, 0.0, 1.0)


def full_best_response(model: TabularPolicy) -> TabularPolicy:
    return best_response(model, 0).policy.combine(best_response(model, 1).policy, 0)


def mix_plans(current: TabularPolicy, target: TabularPolicy, speed: float) -> TabularPolicy:
    """``(1 - speed) current + speed target``, mixed as whole strategies (realization plans)."""
    tree = current.tree
    plan = sum(
        (1 - speed) * realization_plan(tree, current, seat)
        + speed * realization_plan(tree, target, seat)
        for seat in (0, 1)
    )
    # where neither strategy reaches an information set, the current probabilities stay
    reached = tree.segment_sum(plan) > 0
    return TabularPolicy(
        tree, np.where(reached[tree.seq_infoset], tree.normalize(plan), current.probs)
    )


class AdaptiveLeducBot:
    """A bot that moves from ``start`` towards its best response to its model of the hero."""

    def __init__(
        self,
        builder: BotPolicyBuilder,
        start: LeducBotParams,
        adaptation: Adaptation,
        prior: BetaPrior,
    ) -> None:
        self.builder, self.adaptation, self.prior = builder, adaptation, prior
        self.theta = start.as_array()
        self.policy = builder.policy(self.theta)
        self.seen: list[Observation] = []

    def observe(self, view: Sequence[Observation]) -> None:
        """A block of hands as the bot saw them."""
        self.seen.extend(view)

    def hero_model(self) -> TabularPolicy:
        b = self.builder
        recent = self.seen[-self.adaptation.window :]
        return b.policy(fit_params(b, ObservationIndex(b.tree, b.game, recent), self.prior))

    def adapt(self) -> None:
        a = self.adaptation
        if a.speed == 0.0:
            return
        model = self.hero_model()
        target = full_best_response(model)
        if a.mode == "tabular":
            self.policy = mix_plans(self.policy, target, a.speed)
            return
        goal = family_rates(self.builder, target, model, self.theta)
        self.theta = self.theta + a.speed * (goal - self.theta)
        self.policy = self.builder.policy(self.theta)


#: A hero: (its redacted view of the completed hands, hands played) -> its strategy.
Hero = Callable[[list[Observation], int], TabularPolicy]


@dataclass(frozen=True, eq=False)
class Session:
    """Per block: the hero's exact value and exploitability, and the bot's exploitability."""

    block: int
    values: np.ndarray
    hero_exploitability: np.ndarray
    bot_exploitability: np.ndarray
    bot_final: TabularPolicy
    bot_start: TabularPolicy

    def value(self, start: int = 0, stop: int | None = None) -> float:
        """Mean exact value over the blocks starting in hands ``start`` to ``stop - 1``."""
        first = np.arange(self.values.size) * self.block
        stop = int(first[-1]) + 1 if stop is None else stop
        return float(self.values[(first >= start) & (first < stop)].mean())


def play_session(
    hero: Hero,
    bot: AdaptiveLeducBot,
    n_hands: int,
    block_rng: Callable[[int], np.random.Generator],
) -> Session:
    """``n_hands`` hands in blocks of the bot's period. Each block's generator comes from
    ``block_rng(block)``, so heroes facing the same bot draw from the same streams. A hero sees
    only its redacted view; the bot sees only its own."""
    block = bot.adaptation.period
    if n_hands % block:
        raise ValueError("a session is a whole number of blocks")
    builder = bot.builder
    tree, game = builder.tree, builder.game
    seen: list[Observation] = []
    start_policy = bot.policy
    values, hero_exp, bot_exp = [], [], []
    last_hero, last_hero_exp = None, 0.0
    last_bot, last_bot_exp = None, 0.0
    for b, start in enumerate(range(0, n_hands, block)):
        strategy = hero(seen, start)
        if strategy is not last_hero:
            last_hero, last_hero_exp = strategy, exploitability(strategy)
        if bot.policy is not last_bot:
            last_bot, last_bot_exp = bot.policy, exploitability(bot.policy)
        values.append(head_to_head(strategy, bot.policy))
        hero_exp.append(last_hero_exp)
        bot_exp.append(last_bot_exp)
        hero_view, bot_view = simulate_views(
            tree, game, strategy, bot.policy, block, block_rng(b), first_hand=start
        )
        seen = seen + hero_view
        bot.observe(bot_view)
        bot.adapt()
    return Session(
        block,
        np.array(values),
        np.array(hero_exp),
        np.array(bot_exp),
        bot.policy,
        start_policy,
    )

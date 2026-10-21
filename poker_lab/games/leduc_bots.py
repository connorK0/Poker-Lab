"""Leduc versions of the Phase 1 archetypes, observation of their play, and likelihood fitting.

**Bots.** In a small game a bot is just a behavior strategy, so each bot is a
``TabularPolicy`` (both seats) built from five interpretable parameters:

========================  =====================================================
``bluff``                 P(bet or raise) with a weak hand
``medium_aggression``     P(bet or raise) with a medium hand
``value``                 P(bet or raise) with a strong hand
``fold_weak``             P(fold to a bet) with a weak hand
``fold_medium``           P(fold to a bet) with a medium hand (strong never folds)
========================  =====================================================

Hand strength is bucketed by showdown equity against a random card: in round 1 by rank
(J weak, Q medium, K strong); in round 2 a pair is strong (it wins against everything),
an unpaired card higher than the third rank is medium (62.5% against a random card), anything
else is weak (12.5%). Not facing a bet: bet with probability ``a``. Facing a bet: fold with
probability ``f``; otherwise raise with probability ``a`` (when raising is allowed) or call.
Here ``a`` and ``f`` are the bucket's aggression and fold parameters.

Archetypes are distributions over the parameters (Beta, concentration 30, as in Phase 1),
so populations vary within types and neighbouring types overlap.

**Observation (the information firewall).** ``simulate_hands`` plays the hero against a bot
and records what the hero would see: its own card, the board, every action, and the bot's
card *only if the hand reaches showdown*. ``Observation.bot_card`` is ``None`` otherwise.

**Likelihood fitting.** ``fit_params`` finds the posterior mode given observations: the
likelihood of a hand is the product of the bot's action probabilities, averaged over the bot's
possible cards when the card was not shown (the bot's card is uniform over the cards the hero
has not seen, independent of the hero's own actions). The prior is an independent Beta per
parameter (``BetaPrior``), fit by moments to a separate training population. The mode is taken
in the *logit* parameterization, which adds the Jacobian ``log theta(1 - theta)`` and is the
same as the mode of ``Beta(a + 1, b + 1)``: it sits at the prior mean ``a / (a + b)``, so
little data shrinks toward the population mean. (The mode in theta itself is pulled onto 0 or
1 whenever ``a`` or ``b`` is below 1, as it is for this diffuse population prior.)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import astuple, dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from poker_lab.games.leduc import LeducHoldem
from poker_lab.games.policy import TabularPolicy
from poker_lab.games.tree import TERMINAL, GameTree

PARAMS = ("bluff", "medium_aggression", "value", "fold_weak", "fold_medium")
WEAK, MEDIUM, STRONG = 0, 1, 2


@dataclass(frozen=True)
class LeducBotParams:
    bluff: float
    medium_aggression: float
    value: float
    fold_weak: float
    fold_medium: float

    def __post_init__(self) -> None:
        for name, v in zip(PARAMS, astuple(self), strict=True):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")

    def as_array(self) -> np.ndarray:
        return np.array(astuple(self), dtype=float)

    @classmethod
    def from_array(cls, x) -> LeducBotParams:
        return cls(*(float(v) for v in x))


ARCHETYPE_MEANS: dict[str, LeducBotParams] = {
    "nit": LeducBotParams(0.05, 0.15, 0.60, 0.90, 0.60),
    "tag": LeducBotParams(0.20, 0.45, 0.85, 0.75, 0.30),
    "lag": LeducBotParams(0.40, 0.65, 0.90, 0.55, 0.20),
    "calling_station": LeducBotParams(0.05, 0.10, 0.35, 0.15, 0.05),
    "maniac": LeducBotParams(0.65, 0.80, 0.95, 0.35, 0.15),
}
CONCENTRATION = 30.0


def sample_params(archetype: str, rng: np.random.Generator) -> LeducBotParams:
    mean = ARCHETYPE_MEANS[archetype].as_array()
    return LeducBotParams.from_array(rng.beta(mean * CONCENTRATION, (1 - mean) * CONCENTRATION))


@dataclass(frozen=True)
class LeducBot:
    bot_id: str
    archetype: str
    params: LeducBotParams


def sample_population(
    per_archetype: int, rng: np.random.Generator, prefix: str = "b"
) -> list[LeducBot]:
    """``per_archetype`` bots of every archetype, shuffled, with opaque ids."""
    bots = [
        LeducBot("", arch, sample_params(arch, rng))
        for arch in ARCHETYPE_MEANS
        for _ in range(per_archetype)
    ]
    order = rng.permutation(len(bots))
    return [
        LeducBot(f"{prefix}{k:03d}", bots[i].archetype, bots[i].params) for k, i in enumerate(order)
    ]


# ---- the policy as a vectorized function of the parameters ----------------------------------


class BotPolicyBuilder:
    """Per-sequence lookup tables so that ``policy(theta)`` is a few array operations."""

    def __init__(self, tree: GameTree, game: LeducHoldem) -> None:
        if not isinstance(game, LeducHoldem):
            raise TypeError("Leduc bots need a LeducHoldem game")
        self.tree, self.game = tree, game
        bucket = np.zeros(tree.num_infosets, dtype=np.int64)
        for i, key in enumerate(tree.infoset_keys):
            bucket[i] = self._bucket(key)
        self.infoset_bucket = bucket
        seq_bucket = bucket[tree.seq_infoset]
        pairs = zip(tree.seq_infoset, tree.seq_action, strict=True)
        labels = np.array([tree.infoset_actions[i][a] for i, a in pairs])
        can_raise = np.array(["r" in tree.infoset_actions[i] for i in tree.seq_infoset])
        self.seq_bucket = seq_bucket
        self.kind = np.select(
            [
                labels == "k",
                labels == "b",
                labels == "f",
                (labels == "c") & can_raise,
                (labels == "c") & ~can_raise,
                labels == "r",
            ],
            [0, 1, 2, 3, 4, 5],
        )

    def _bucket(self, key: str) -> int:
        game = self.game
        cards = key.split("|", 1)[0]
        own = game.ranks.index(cards[0])
        if len(cards) == 2:  # round 1: rank decides
            return WEAK if own == 0 else STRONG if own == len(game.ranks) - 1 else MEDIUM
        board = game.ranks.index(cards[2])
        if own == board:
            return STRONG
        others = [r for r in range(len(game.ranks)) if r not in (own, board)]
        return MEDIUM if all(own > r for r in others) else WEAK

    def probs(self, theta: np.ndarray) -> np.ndarray:
        """Per-sequence probabilities for parameter vector(s) ``theta`` (..., 5)."""
        theta = np.asarray(theta, dtype=float)
        aggr = theta[..., [0, 1, 2]][..., self.seq_bucket]
        fold = np.concatenate([theta[..., [3, 4]], np.zeros((*theta.shape[:-1], 1))], axis=-1)
        fold = fold[..., self.seq_bucket]
        k = self.kind
        return np.select(
            [k == 0, k == 1, k == 2, k == 3, k == 4, k == 5],
            [1 - aggr, aggr, fold, (1 - fold) * (1 - aggr), 1 - fold, (1 - fold) * aggr],
        )

    def policy(self, params: LeducBotParams | np.ndarray) -> TabularPolicy:
        theta = params.as_array() if isinstance(params, LeducBotParams) else params
        return TabularPolicy(self.tree, self.probs(theta))


# ---- observation -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """One completed hand as the hero saw it."""

    bot_seat: int
    hero_card: int
    board: int | None  # None if the hand ended before the public card
    actions: tuple[str, ...]  # the decision-edge labels in order (both players)
    bot_card: int | None  # revealed only at showdown
    bot_sequences: tuple[int, ...]  # filled when bot_card is known (else empty)
    bot_payoff: float


def _walk_hand(tree: GameTree, edge_probs: list[np.ndarray], rng: np.random.Generator):
    """Sample one hand; returns the visited node path."""
    node, path = 0, [0]
    while tree.player[node] != TERMINAL:
        start, n = int(tree.child_start[node]), int(tree.num_children[node])
        p = edge_probs[start : start + n]
        node = start + int(rng.choice(n, p=p / p.sum()))
        path.append(node)
    return path


def simulate_hands(
    tree: GameTree,
    game: LeducHoldem,
    hero: TabularPolicy,
    bot: TabularPolicy,
    n_hands: int,
    rng: np.random.Generator,
) -> list[Observation]:
    """``n_hands`` hands, bot seat alternating (seat 1 first), recorded with the hero's
    information only: the bot's card is kept only when the hand reaches showdown."""
    profiles = [hero.combine(bot, 0), bot.combine(hero, 0)]  # bot in seat 1, then seat 0
    edges = [tree.edge_probabilities(p.probs) for p in profiles]
    out = []
    for h in range(n_hands):
        bot_seat = 1 - h % 2
        path = _walk_hand(tree, edges[1 - bot_seat], rng)
        decisions = [n for n in path[1:] if tree.edge_seq[n] >= 0]
        # chance edges deal seat 0's card, seat 1's card and (if reached) the board
        cards = [game.deck.index(tree.labels[n]) for n in path[1:] if tree.edge_seq[n] < 0]
        leaf = path[-1]
        showdown = len(cards) == 3 and not tree.labels[leaf].endswith("f")
        bot_seqs = tuple(
            int(tree.edge_seq[n]) for n in decisions if tree.edge_player[n] == bot_seat
        )
        out.append(
            Observation(
                bot_seat=bot_seat,
                hero_card=cards[1 - bot_seat],
                board=cards[2] if len(cards) == 3 else None,
                actions=tuple(tree.labels[n] for n in decisions),
                bot_card=cards[bot_seat] if showdown else None,
                bot_sequences=bot_seqs if showdown else (),
                bot_payoff=float(tree.utility[leaf] * (1 if bot_seat == 0 else -1)),
            )
        )
    return out


# ---- likelihood and fitting --------------------------------------------------------------------


class ObservationIndex:
    """Precomputed sequence indices of the bot's actions for every candidate bot card."""

    def __init__(self, tree: GameTree, game: LeducHoldem, observations: Sequence[Observation]):
        rows: list[int] = []  # which (hand, candidate) row each bot decision belongs to
        seqs: list[int] = []
        row_hand: list[int] = []
        row_logprior: list[float] = []
        n_rows = 0
        for h, obs in enumerate(observations):
            if obs.bot_card is not None:
                candidates = [obs.bot_card]
            else:
                seen = {obs.hero_card} | ({obs.board} if obs.board is not None else set())
                candidates = [c for c in range(len(game.deck)) if c not in seen]
            for card in candidates:
                for s in _bot_sequences(tree, game, obs, card):
                    rows.append(n_rows)
                    seqs.append(s)
                row_hand.append(h)
                row_logprior.append(-np.log(len(candidates)))
                n_rows += 1
        self.rows = np.asarray(rows, dtype=np.int64)
        self.seqs = np.asarray(seqs, dtype=np.int64)
        self.n_rows = n_rows
        self.row_hand = np.asarray(row_hand, dtype=np.int64)
        self.row_logprior = np.asarray(row_logprior)
        self.n_hands = len(observations)
        self.hand_start = np.flatnonzero(np.r_[True, np.diff(self.row_hand) != 0])

    def log_likelihood(self, seq_probs: np.ndarray) -> float:
        """Sum over hands of log P(bot's observed actions | per-sequence probabilities).

        Rows (one per candidate bot card) are grouped by hand in order, so the per-hand
        log-sum-exp over candidates is two ``reduceat`` calls.
        """
        if self.n_hands == 0:
            return 0.0
        logp = np.log(np.clip(seq_probs[self.seqs], 1e-300, None))
        per_row = np.bincount(self.rows, weights=logp, minlength=self.n_rows) + self.row_logprior
        starts = self.hand_start
        peak = np.maximum.reduceat(per_row, starts)
        total = np.add.reduceat(np.exp(per_row - peak[self.row_hand]), starts)
        return float(np.sum(peak + np.log(total)))


def _bot_sequences(tree: GameTree, game: LeducHoldem, obs: Observation, bot_card: int):
    """Sequence indices of the bot's decisions in ``obs`` if its card were ``bot_card``."""
    rounds: list[str] = [""]
    out = []
    own = game.deck[bot_card]
    board = game.deck[obs.board] if obs.board is not None else ""
    for action in obs.actions:
        history = rounds[-1]
        actor = len(history) % 2
        if actor == obs.bot_seat:
            public = board if len(rounds) == 2 else ""
            key = f"{own}{public}|{'/'.join(rounds)}"
            out.append(tree.sequence(key, action))
        rounds[-1] = history + action
        closed = rounds[-1].endswith("c") or rounds[-1] == "kk"
        if closed and len(rounds) == 1 and not rounds[-1].endswith("f"):
            rounds.append("")
    return out


@dataclass(frozen=True)
class BetaPrior:
    """Independent Beta(a_j, b_j) prior per parameter."""

    a: np.ndarray
    b: np.ndarray

    @classmethod
    def fit(cls, samples: np.ndarray) -> BetaPrior:
        """Method of moments on a (n, 5) array of parameter vectors."""
        mean = samples.mean(axis=0)
        var = samples.var(axis=0, ddof=1)
        common = mean * (1 - mean) / np.maximum(var, 1e-9) - 1
        return cls(mean * common, (1 - mean) * common)

    @classmethod
    def flat(cls) -> BetaPrior:
        return cls(np.ones(5), np.ones(5))

    def log_density(self, theta: np.ndarray) -> float:
        t = np.clip(theta, 1e-9, 1 - 1e-9)
        return float(np.sum((self.a - 1) * np.log(t) + (self.b - 1) * np.log1p(-t)))

    @property
    def mean(self) -> np.ndarray:
        return self.a / (self.a + self.b)


def fit_params(
    builder: BotPolicyBuilder,
    index: ObservationIndex,
    prior: BetaPrior | None,
    start: np.ndarray | None = None,
) -> LeducBotParams:
    """Posterior mode in logit space (log-likelihood + log prior + log Jacobian), or the plain
    maximum-likelihood estimate when ``prior`` is None."""
    center = np.full(5, 0.5) if prior is None else prior.mean
    x0 = logit(np.clip(center if start is None else start, 0.02, 0.98))

    def objective(z: np.ndarray) -> float:
        theta = expit(z)
        value = index.log_likelihood(builder.probs(theta))
        if prior is not None:
            value += prior.log_density(theta) + np.sum(np.log(theta) + np.log1p(-theta))
        return -value

    res = minimize(objective, x0, method="L-BFGS-B", bounds=[(-9.0, 9.0)] * 5)
    return LeducBotParams.from_array(np.clip(expit(res.x), 0.0, 1.0))


def public_visit_counts(
    tree: GameTree, game: LeducHoldem, observations: Sequence[Observation]
) -> np.ndarray:
    """For every information set, how often the bot was seen acting at its *public* state
    (betting so far and board, any private card). Data-biased responses trust the model more
    where this count is high. Information sets of a seat count only hands the bot played there.
    """
    counts: dict[str, int] = {}
    for obs in observations:
        rounds = [""]
        for action in obs.actions:
            if len(rounds[-1]) % 2 == obs.bot_seat:
                board = game.deck[obs.board] if len(rounds) == 2 else ""
                public = f"{board}|{'/'.join(rounds)}"
                counts[public] = counts.get(public, 0) + 1
            rounds[-1] += action
            if len(rounds) == 1 and (rounds[0].endswith("c") or rounds[0] == "kk"):
                rounds.append("")
    width = len(game.deck[0])
    return np.array([counts.get(key[width:], 0) for key in tree.infoset_keys], dtype=float)


def card_blind_policy(
    builder: BotPolicyBuilder, game: LeducHoldem, observations: Sequence[Observation]
) -> TabularPolicy:
    """A deliberately misspecified, HUD-style model: action frequencies that ignore the cards.

    Three Laplace-smoothed frequencies from public information only: bet when not facing a bet,
    fold when facing one, and raise (when allowed) given not folding. The same frequencies are
    used at every information set, whatever the bot's hand. Comparing exploitation based on
    this model with the correctly specified five-parameter model measures what hand-reading
    structure is worth.
    """
    counts = np.ones(6)  # Laplace: k, b, f, c (raise allowed), c (capped), r
    for obs in observations:
        rounds = [""]
        for action in obs.actions:
            history = rounds[-1]
            if len(history) % 2 == obs.bot_seat:
                facing = bool(history) and history[-1] in "br"
                raises = sum(a in "br" for a in history)
                if not facing:
                    counts[{"k": 0, "b": 1}[action]] += 1
                elif action == "f":
                    counts[2] += 1
                elif action == "r":
                    counts[5] += 1
                else:
                    counts[3 if raises < game.max_raises else 4] += 1
            rounds[-1] = history + action
            if len(rounds) == 1 and (rounds[0].endswith("c") or rounds[0] == "kk"):
                rounds.append("")
    bet = counts[1] / (counts[0] + counts[1])
    fold = counts[2] / (counts[2] + counts[3] + counts[4] + counts[5])
    raise_ = counts[5] / (counts[3] + counts[5])
    k = builder.kind
    probs = np.select(
        [k == 0, k == 1, k == 2, k == 3, k == 4, k == 5],
        [1 - bet, bet, fold, (1 - fold) * (1 - raise_), 1 - fold, (1 - fold) * raise_],
    )
    return TabularPolicy(builder.tree, probs)

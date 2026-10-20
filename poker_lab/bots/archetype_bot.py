"""``ParametricBot``: an agent whose tendencies are set by ``BotParams``.

Preflop
    Hands are ranked by ``preflop_percentile`` (0 = aces). The bot plays the top
    ``range_width`` (adjusted for position) of hands in an unraised pot, entering with a raise
    with probability ``raise_share``. Facing one raise it continues with the top
    ``range_width * continue_vs_raise`` and 3-bets those with probability ``three_bet_freq``.
    Facing a 3-bet the continuing range shrinks again; facing a 4-bet only premiums continue.

Postflop
    Monte Carlo equity against a random hand puts the hand in one of four buckets: AIR, WEAK
    (weak made hands and draws), MEDIUM, STRONG. The cut points are higher when facing a bet.
    On the flop the preflop aggressor c-bets with probability ``cbet_freq`` whatever its
    bucket (strong hands bet at least ``value_aggression``). Otherwise:

    * not facing a bet: strong hands bet with probability ``value_aggression`` (medium 0.6x),
      weak hands and air bluff with probability proportional to ``bluff_freq`` (less often
      multiway);
    * facing a bet: weak hands fold with probability ``fold_to_aggression`` (medium 0.5x,
      air ``2f + 0.15``; all scaled by the price), strong hands raise with probability
      ``0.6 * value_aggression``, medium hands ``0.5 * value_aggression**2``, and weak hands
      and air bluff-raise in proportion to ``bluff_freq``; otherwise call.

    Sizes are ``bet_size_pot`` times the pot with small multiplicative noise; bets above half
    the remaining stack become all-ins.

The coefficients were tuned (not the intended ranges, which were fixed in advance) so that each
archetype's measured stats land inside ``INTENDED_STATS``; see PROGRESS.md.
All randomness comes from the table RNG passed to ``act``.
"""

from __future__ import annotations

import math
import random

from poker_lab.bots.hand_strength import equity_vs_random, preflop_percentile
from poker_lab.bots.params import BotParams
from poker_lab.engine.agent import Agent, Decision, clamp_raise
from poker_lab.schema import ActionType, LegalActions, SeatView, Street

AIR, WEAK, MEDIUM, STRONG = 0, 1, 2, 3
# Equity cut points between consecutive buckets. Facing a bet the bar is higher: equity against
# a random hand overstates a hand's strength against a range that chose to bet.
BUCKET_THRESHOLDS = (0.45, 0.62, 0.80)
BUCKET_THRESHOLDS_FACING_BET = (0.55, 0.70, 0.85)

# Position multipliers on range_width when opening (they average 1.0 over 6-max positions).
POSITION_WIDTH = {"UTG": 0.7, "HJ": 0.85, "CO": 1.05, "BTN": 1.4, "SB": 1.0, "BB": 1.0}
BB_DEFENSE = 1.3  # the big blind continues wider against a single raise (better price)
COMMIT_FRACTION = 0.5  # if a bet would use more than this share of the stack, move all-in


def equity_bucket(equity: float, facing_bet: bool = False) -> int:
    cuts = BUCKET_THRESHOLDS_FACING_BET if facing_bet else BUCKET_THRESHOLDS
    for bucket, cut in enumerate(cuts):
        if equity < cut:
            return bucket
    return STRONG


class ParametricBot(Agent):
    """A bot with interpretable tendencies. See the module docstring for the policy."""

    def __init__(self, params: BotParams, *, equity_samples: int = 150) -> None:
        self.params = params
        self.equity_samples = equity_samples
        self._equity_key: tuple[str, Street] | None = None
        self._equity = 0.0

    # ------------------------------------------------------------------ helpers
    def _bet_to(self, view: SeatView, legal: LegalActions, target: float) -> Decision:
        """Bet/raise to roughly ``target`` (street total); go all-in past the commit point."""
        if not legal.can_bet_or_raise:
            return Decision.check_call()
        if target >= COMMIT_FRACTION * (view.stack + view.street_bet):
            return Decision.bet_raise(legal.max_raise_to)
        return Decision.bet_raise(clamp_raise(legal, target))

    def _equity_now(self, view: SeatView, rng: random.Random) -> float:
        key = (view.public.hand_id, view.public.street)
        if key != self._equity_key:
            self._equity = equity_vs_random(
                view.hole_cards, view.public.board, rng, self.equity_samples
            )
            self._equity_key = key
        return self._equity

    @staticmethod
    def _noise(rng: random.Random) -> float:
        return math.exp(0.12 * rng.gauss(0.0, 1.0))

    # ------------------------------------------------------------------ policy
    def act(self, view: SeatView, rng: random.Random) -> Decision:
        legal = view.legal
        assert legal is not None
        if view.public.street is Street.PREFLOP:
            return self._preflop(view, legal, rng)
        return self._postflop(view, legal, rng)

    def _preflop(self, view: SeatView, legal: LegalActions, rng: random.Random) -> Decision:
        p = self.params
        pub = view.public
        bb = pub.table.big_blind
        raises = [a for a in pub.actions if a.action_type is ActionType.RAISE]
        n_raises = len(raises)
        h = preflop_percentile(*view.hole_cards)
        max_bet = pub.max_street_bet
        position = view.position

        if n_raises == 0:
            width = p.range_width * POSITION_WIDTH.get(position, 1.0)
            if h <= width:
                if rng.random() < p.raise_share:
                    limpers = sum(1 for a in pub.actions if a.action_type is ActionType.CALL)
                    size = (2.0 + 2.0 * p.bet_size_pot) * bb + limpers * bb
                    return self._bet_to(view, legal, size * self._noise(rng))
                return Decision.check_call()  # limp, or check in the big blind
            return Decision.check_call() if legal.can_check else Decision.fold()

        stack_left = view.stack
        committed = legal.call_amount >= COMMIT_FRACTION * stack_left
        if n_raises == 1:
            cont = p.range_width * p.continue_vs_raise * (BB_DEFENSE if position == "BB" else 1.0)
            if h <= cont:
                if rng.random() < p.three_bet_freq or (committed and legal.can_bet_or_raise):
                    return self._bet_to(view, legal, 3.0 * max_bet * self._noise(rng))
                return Decision.check_call()
            if legal.can_bet_or_raise and rng.random() < 0.08 * p.bluff_freq:
                return self._bet_to(view, legal, 3.0 * max_bet * self._noise(rng))
            return Decision.fold()

        if n_raises == 2:
            opener = raises[0].seat == view.seat
            cont = p.range_width * p.continue_vs_raise * (0.5 if opener else 0.4)
            if h <= cont:
                four_bet = max(p.three_bet_freq * 0.5, 0.5 if h <= 0.02 else 0.0)
                if rng.random() < four_bet or committed:
                    return self._bet_to(view, legal, 2.3 * max_bet * self._noise(rng))
                return Decision.check_call()
            if legal.can_bet_or_raise and rng.random() < 0.03 * p.bluff_freq:
                return self._bet_to(view, legal, 2.3 * max_bet * self._noise(rng))
            return Decision.fold()

        if h <= 0.02 + 0.03 * p.continue_vs_raise:  # facing a 4-bet or more: premiums only
            if legal.can_bet_or_raise and rng.random() < 0.5 + 0.5 * p.three_bet_freq:
                return Decision.bet_raise(legal.max_raise_to)
            return Decision.check_call()
        return Decision.fold()

    def _postflop(self, view: SeatView, legal: LegalActions, rng: random.Random) -> Decision:
        p = self.params
        pub = view.public
        bucket = equity_bucket(self._equity_now(view, rng), facing_bet=legal.call_amount > 0)
        me = view.seat
        street = pub.street
        preflop_raisers = [
            a.seat
            for a in pub.actions
            if a.street is Street.PREFLOP and a.action_type is ActionType.RAISE
        ]
        is_pfa = bool(preflop_raisers) and preflop_raisers[-1] == me
        acted_this_street = any(a.seat == me and a.street is street for a in pub.actions)
        n_opponents = sum(1 for f in pub.folded if not f) - 1
        multiway = max(1, n_opponents - 1)
        pot = pub.pot

        if legal.call_amount == 0:
            cbet_spot = (
                street is Street.FLOP
                and is_pfa
                and pub.max_street_bet == 0
                and not acted_this_street
            )
            if cbet_spot:
                p_bet = max(p.cbet_freq, p.value_aggression) if bucket == STRONG else p.cbet_freq
            else:
                p_bet = (
                    p.bluff_freq * 0.5 / multiway,
                    p.bluff_freq * 0.7 / multiway,
                    p.value_aggression * 0.6,
                    p.value_aggression,
                )[bucket]
            if legal.can_bet_or_raise and rng.random() < p_bet:
                return self._bet_to(view, legal, p.bet_size_pot * pot * self._noise(rng))
            return Decision.check_call()

        call = legal.call_amount
        price = call / (pot + call)
        price_factor = min(max(0.6 + 1.2 * price, 0.5), 1.4)
        fold_p = (
            min(1.0, 2.0 * p.fold_to_aggression + 0.15),
            p.fold_to_aggression,
            p.fold_to_aggression * 0.5,
            0.0,
        )[bucket] * price_factor
        raise_p = (
            p.bluff_freq * 0.5 / multiway,
            p.bluff_freq * 0.55 / multiway,
            0.5 * p.value_aggression**2,
            p.value_aggression * 0.6,
        )[bucket]
        if not legal.can_bet_or_raise:
            raise_p = 0.0
        u = rng.random()
        if u < fold_p:
            return Decision.fold()
        if u < fold_p + raise_p:
            target = pub.max_street_bet + p.bet_size_pot * (pot + call) * self._noise(rng)
            return self._bet_to(view, legal, target)
        return Decision.check_call()

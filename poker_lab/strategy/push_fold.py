"""Push/fold advice in real hold'em hands (P3.3): a ``StrategyProvider`` over the solved game.

``poker_lab.games.push_fold`` solves heads-up push/fold on 169 hand classes. This provider plays
that solution at a real table. For each ``SeatView`` it decides whether the abstract game
describes the spot and, if it does, which information set the hero is in:

* preflop, blinds in the game's 1 : 2 ratio, no antes or straddles;
* every seat other than the two blinds has folded (heads-up there are no others);
* the hero is the small blind making its first decision, or the big blind facing a small-blind
  raise that puts the big blind all-in (a push), with no other voluntary action in between;
* the effective stack ``S`` (the smaller of the two blinds' stacks at the start of the hand) is
  more than 1 bb and at most ``max_stack`` bb (default 20, the range solved and checked in P3.1).

The information set is the hero's role (small or big blind) and the class of its two cards. The
policy's push or call probability for it becomes the distribution. A push is a raise to the most
the hero can bet: all-in, with any chips the big blind cannot match returned uncalled. A call is
a call.

Anything else is off the tree: a limp, a min-raise, a stack above ``max_stack``. The provider then
hands the view to its ``fallback`` provider, or raises ``OffTree`` when it has none. It does not
map off-tree actions onto the tree (a min-raise is not read as a push), because the abstraction
has no action such a mapping could be checked against.

Heads-up tables are exact, up to the Monte Carlo error of the matchup-equity table. Blind against
blind after folds is an approximation: the abstract game deals from a deck that still holds the
folded players' cards.

The decision depends on the view alone. The provider keeps per-stack policies and no reference to
a hand store; ``tests/test_advisor_firewall.py`` checks that the opponent's sealed cards cannot
change its output.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from poker_lab.bots.hand_strength import hand_class
from poker_lab.engine.agent import Decision
from poker_lab.games.preflop import CLASS_INDEX
from poker_lab.games.push_fold import BB, SB, aggressive_probabilities, infoset_key, push_fold_tree
from poker_lab.schema import Action, ActionType, SeatInfo, SeatView, Street, TableConfig
from poker_lab.solvers.lp import solve_lp
from poker_lab.strategy.base import ActionDistribution, StrategyProvider, require_actor

MAX_STACK = 20.0

#: stack in bb -> (push probability, call probability), one entry per class in ``CLASSES`` order
Ranges = Callable[[float], tuple[np.ndarray, np.ndarray]]


class OffTree(ValueError):
    """The spot is not one the push/fold game describes."""


@dataclass(frozen=True, slots=True)
class PushFoldSpot:
    player: int  # SB or BB, the abstract game's player index
    hand_class: str
    stack: float  # effective stack in big blinds
    heads_up: bool  # False: blind against blind after the other seats folded (approximate)

    @property
    def infoset(self) -> str:
        return infoset_key(self.hand_class, self.player)


@dataclass(frozen=True, slots=True)
class BlindPot:
    """A preflop pot only the two blinds have entered (``blind_pot``)."""

    small_blind: int  # seat
    big_blind: int  # seat
    effective: int  # chips: the smaller of the two blinds' starting stacks
    stack: float  # the same in big blinds
    decisions: tuple[Action, ...]  # the blinds' voluntary actions, in order

    def is_push(self, action: Action) -> bool:
        """A small-blind raise that puts the big blind all-in."""
        return (
            action.seat == self.small_blind
            and action.action_type is ActionType.RAISE
            and action.total_bet >= self.effective
        )


def blind_pot(
    table: TableConfig,
    seats: Sequence[SeatInfo],
    actions: Sequence[Action],
    *,
    max_stack: float = MAX_STACK,
) -> BlindPot:
    """The table rules and forced bets of a push/fold spot, from a hand's actions so far (or all
    of them); raises ``OffTree`` when the abstract game cannot describe the hand."""
    if table.ante or 2 * table.small_blind != table.big_blind:
        raise OffTree("the game needs blinds in a 1:2 ratio and no ante")
    forced = [a for a in actions if a.action_type.is_forced]
    posters = {a.action_type: a.seat for a in forced}
    if len(forced) != 2 or set(posters) != {ActionType.POST_SMALL_BLIND, ActionType.POST_BIG_BLIND}:
        raise OffTree("needs exactly one small blind and one big blind, and nothing else posted")
    sb, bb = posters[ActionType.POST_SMALL_BLIND], posters[ActionType.POST_BIG_BLIND]
    decisions = [a for a in actions if a.action_type.is_decision]
    if any(a.action_type is not ActionType.FOLD for a in decisions if a.seat not in (sb, bb)):
        raise OffTree("a player other than the blinds entered the pot")
    start = {s.seat: s.starting_stack for s in seats}
    effective = min(start[sb], start[bb])
    stack = effective / table.big_blind
    if not 1.0 < stack <= max_stack:
        raise OffTree(f"effective stack {stack:g} bb is outside (1, {max_stack:g}]")
    blinds = tuple(a for a in decisions if a.seat in (sb, bb))
    return BlindPot(sb, bb, effective, stack, blinds)


def push_fold_spot(view: SeatView, *, max_stack: float = MAX_STACK) -> PushFoldSpot:
    """The abstract information set of ``view``; raises ``OffTree`` if there is none."""
    if type(view) is not SeatView:
        raise TypeError("push_fold_spot takes exactly a SeatView")
    pub = view.public
    if pub.street is not Street.PREFLOP:
        raise OffTree("not preflop")
    pot = blind_pot(pub.table, pub.seats, pub.actions, max_stack=max_stack)
    sb, bb, blinds = pot.small_blind, pot.big_blind, pot.decisions
    if view.seat == sb and not blinds:
        player = SB
    elif view.seat == bb and len(blinds) == 1 and pot.is_push(blinds[0]):
        player = BB
    else:
        raise OffTree("not the small blind's first decision or the big blind facing a push")
    others_out = all(
        folded for s, folded in zip(pub.seats, pub.folded, strict=True) if s.seat not in (sb, bb)
    )
    if not others_out:  # cannot happen preflop once the small blind acts; kept as a guard
        raise OffTree("a player other than the blinds has not folded")
    return PushFoldSpot(player, hand_class(*view.hole_cards), pot.stack, len(pub.seats) == 2)


def _clean(p: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Snap LP round-off (1e-12, 1 - 1e-12) to 0 and 1, so pure choices stay pure."""
    out = np.where(p < tol, 0.0, np.where(p > 1.0 - tol, 1.0, p))
    out.setflags(write=False)
    return out


@lru_cache(maxsize=512)
def nash_ranges(stack: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact equilibrium push and call probabilities per class at ``stack`` bb (LP, ~0.1 s)."""
    policy = solve_lp(push_fold_tree(stack)).policy
    push, call = aggressive_probabilities(policy, SB), aggressive_probabilities(policy, BB)
    return _clean(push), _clean(call)


def fixed_ranges(push: np.ndarray, call: np.ndarray) -> Ranges:
    """The same per-class ranges at every stack (e.g. a P3.2 bot's)."""
    push, call = _clean(np.asarray(push, float)), _clean(np.asarray(call, float))
    return lambda stack: (push, call)


class PushFoldStrategy(StrategyProvider):
    """Plays per-class push/fold ranges in the spots the abstract game describes."""

    def __init__(
        self,
        ranges: Ranges = nash_ranges,
        *,
        fallback: StrategyProvider | None = None,
        max_stack: float = MAX_STACK,
        name: str = "push_fold_nash",
    ) -> None:
        if fallback is not None and not isinstance(fallback, StrategyProvider):
            raise TypeError("fallback must be a StrategyProvider")
        self.ranges = ranges
        self.fallback = fallback
        self.max_stack = max_stack
        self.name = name

    def covers(self, view: SeatView) -> bool:
        try:
            push_fold_spot(view, max_stack=self.max_stack)
        except OffTree:
            return False
        return True

    def action_distribution(self, view: SeatView) -> ActionDistribution:
        legal = require_actor(view)
        try:
            spot = push_fold_spot(view, max_stack=self.max_stack)
        except OffTree:
            if self.fallback is None:
                raise
            return self.fallback.action_distribution(view)
        push, call = self.ranges(spot.stack)
        k = CLASS_INDEX[spot.hand_class]
        if spot.player == SB:
            p, aggressive = float(push[k]), Decision.bet_raise(legal.max_raise_to)
        else:
            p, aggressive = float(call[k]), Decision.check_call()
        return ActionDistribution.from_weights({aggressive: p, Decision.fold(): 1.0 - p})

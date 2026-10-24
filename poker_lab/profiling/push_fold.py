"""Push/fold observations of one opponent, from completed hand histories (P3.4).

``push_fold_observations`` turns completed hands into the sufficient statistics that the push/fold
opponent models fit (``games.push_fold_bots.Observations``): the opponent's push or fold as small
blind, its call or fold facing the hero's push as big blind, its hand class when its cards were
shown, and otherwise the hero's own class (which sets the card-removal term of the likelihood).

Firewall. Only completed hands are accepted (an in-progress hand raises
``IncompleteHandError``), and every hand is first redacted to what the hero could know
(``redact_hole_cards`` with the hero's seat). An opponent's unshown cards therefore cannot reach
the model, even when the caller passes simulator records that still hold them.

A hand counts only if the push/fold game describes it exactly (``strategy.push_fold.blind_pot``):
the hero and the opponent are the two blinds, everyone else folded, blinds 1 : 2 and no ante, the
effective stack equals ``stack``, and the blinds played a fold, or a push followed by a fold or a
call. Other hands are skipped and counted by reason, so the caller sees how much of the history
the model ignores.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from poker_lab.bots.hand_strength import hand_class
from poker_lab.games.preflop import CLASS_INDEX
from poker_lab.games.push_fold_bots import Observations
from poker_lab.profiling.profile_store import IncompleteHandError
from poker_lab.schema import ActionType, HandHistory, Street, redact_hole_cards
from poker_lab.strategy.push_fold import MAX_STACK, OffTree, blind_pot

SKIP_REASONS = ("duplicate", "not_the_blinds", "off_tree", "stack", "hero_cards_unknown")


@dataclass(frozen=True, eq=False)
class PushFoldHistory:
    observations: Observations
    skipped: Mapping[str, int]  # hands left out, by reason (``SKIP_REASONS``)
    hand_ids: frozenset[str]  # the hands the observations come from

    def __post_init__(self) -> None:
        object.__setattr__(self, "skipped", MappingProxyType(dict(self.skipped)))


def _class_of(hand: HandHistory, seat: int, *, shown_only: bool) -> int | None:
    hc = hand.hole_cards_of(seat)
    if hc is None or hc.cards is None or (shown_only and not hc.shown):
        return None
    return CLASS_INDEX[hand_class(*(c.card for c in hc.cards))]


def push_fold_observations(
    hands: Iterable[HandHistory],
    hero_id: str,
    opponent_id: str,
    *,
    stack: float,
    max_stack: float = MAX_STACK,
) -> PushFoldHistory:
    """What ``hero_id`` saw of ``opponent_id``'s push/fold play at ``stack`` bb."""
    out = {
        "sb_shown": np.zeros(169),
        "sb_hidden": np.zeros((2, 169)),
        "bb_shown": np.zeros(169),
        "bb_hidden": np.zeros((2, 169)),
    }
    skipped: Counter[str] = Counter()
    used: set[str] = set()
    for raw in hands:
        if not isinstance(raw, HandHistory):
            raise TypeError("hands must be HandHistory records")
        if not raw.is_complete:
            raise IncompleteHandError(f"hand {raw.hand_id} is not complete")
        if raw.hand_id in used:
            skipped["duplicate"] += 1
            continue
        try:
            hero, opp = raw.seat_of(hero_id), raw.seat_of(opponent_id)
        except KeyError:
            skipped["not_the_blinds"] += 1
            continue
        hand = redact_hole_cards(raw, viewer_seat=hero)
        try:
            pot = blind_pot(hand.table, hand.seats, hand.actions, max_stack=max_stack)
        except OffTree:
            skipped["off_tree"] += 1
            continue
        if {pot.small_blind, pot.big_blind} != {hero, opp}:
            skipped["not_the_blinds"] += 1
            continue
        moves = pot.decisions
        folded = len(moves) == 1 and moves[0].action_type is ActionType.FOLD
        pushed = (
            len(moves) == 2
            and pot.is_push(moves[0])
            and moves[1].seat == pot.big_blind
            and moves[1].action_type in (ActionType.FOLD, ActionType.CALL)
        )
        if any(a.street is not Street.PREFLOP for a in hand.actions) or not (folded or pushed):
            skipped["off_tree"] += 1
            continue
        if abs(pot.stack - stack) > 1e-9:
            skipped["stack"] += 1
            continue
        if opp == pot.small_blind:
            seat_key, action = "sb", int(pushed)
        elif pushed:
            seat_key, action = "bb", int(moves[1].action_type is ActionType.CALL)
        else:  # the hero folded its small blind: the opponent made no decision
            used.add(hand.hand_id)
            continue
        shown = _class_of(hand, opp, shown_only=True)
        if shown is not None:
            out[f"{seat_key}_shown"][shown] += 1
        else:
            mine = _class_of(hand, hero, shown_only=False)
            if mine is None:
                skipped["hero_cards_unknown"] += 1
                continue
            out[f"{seat_key}_hidden"][action, mine] += 1
        used.add(hand.hand_id)
    return PushFoldHistory(Observations(len(used), **out), skipped, frozenset(used))

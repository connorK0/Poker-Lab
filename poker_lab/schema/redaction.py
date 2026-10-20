"""Reduce a complete hand to what a participant could know.

In a real game nobody sees mucked or folded hole cards. Simulated hand histories keep every
hole card as ground truth, so anything that models opponents (the profiler, a ``ProfileStore``,
an agent learning online) should see the redacted record. That way simulated and live data
carry the same information.
"""

from __future__ import annotations

import dataclasses

from poker_lab.schema.hand_history import HandHistory, HoleCards


def redact_hole_cards(hand: HandHistory, viewer_seat: int | None = None) -> HandHistory:
    """Copy of ``hand`` where hole cards not shown at showdown become unknown.

    The viewer's own hole cards (if ``viewer_seat`` is given) are kept. In-progress hands carry
    no hole cards and are returned unchanged.
    """
    if not hand.hole_cards:
        return hand
    redacted = tuple(
        hc if (hc.shown or hc.seat == viewer_seat or hc.cards is None) else HoleCards(hc.seat, None)
        for hc in hand.hole_cards
    )
    if redacted == hand.hole_cards:
        return hand
    return dataclasses.replace(hand, hole_cards=redacted)

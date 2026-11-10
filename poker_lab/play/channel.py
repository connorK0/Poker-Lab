"""How a person's clicks become ``Decision``s while the engine waits.

``TableSimulator`` asks each agent for a decision with a plain function call, ``agent.act(view,
rng)``, and the hand cannot go on until it returns. A bot answers at once; a person answers
seconds later, through another thread (the web server's). ``DecisionChannel`` joins the two with
one condition variable:

* the engine thread calls ``request(view)``: it stores the ``SeatView``, bumps a turn token and
  sleeps on the condition until an answer arrives (or the channel is closed);
* a server thread reads ``pending()`` to show the view, then calls ``submit(token, decision)``.
  The token must be the current one, so a double click or a stale page cannot answer a later
  decision, and the decision must be legal under ``view.legal`` (``check_decision``), so the
  engine never sees an illegal move from the page.

``close()`` wakes a waiting engine thread with ``SessionClosed``, which unwinds the hand being
played; the session then discards it (an abandoned hand is never saved or learned from).
"""

from __future__ import annotations

import random
import secrets
import threading
from collections.abc import Callable

from poker_lab.engine.agent import Agent, Decision, check_decision
from poker_lab.schema import HandHistory, SeatView


class SessionClosed(RuntimeError):
    """The session ended while the engine was waiting for the person."""


class StaleDecision(RuntimeError):
    """A decision for a turn that is not the current one."""


class DecisionChannel:
    def __init__(self) -> None:
        self.cond = threading.Condition()
        self._view: SeatView | None = None
        self._answer: Decision | None = None
        # Random start: a stale page from an earlier session cannot match this session's turns.
        self._token = secrets.randbits(30)
        self.closed = False

    def request(self, view: SeatView) -> Decision:
        """Engine thread: publish ``view`` and wait for the person's decision."""
        with self.cond:
            if self.closed:
                raise SessionClosed("session closed")
            self._token += 1
            self._view, self._answer = view, None
            self.cond.notify_all()
            while self._answer is None and not self.closed:
                self.cond.wait()
            answer, self._view, self._answer = self._answer, None, None
            self.cond.notify_all()
            if answer is None:
                raise SessionClosed("session closed")
            return answer

    def pending(self) -> tuple[int, SeatView] | None:
        """The turn token and view waiting for an answer, if any."""
        with self.cond:
            if self._view is None or self._answer is not None:  # answered, engine not yet awake
                return None
            return self._token, self._view

    def submit(
        self,
        token: int,
        decision: Decision,
        extra_check: Callable[[SeatView, Decision], None] | None = None,
    ) -> None:
        """Server thread: answer turn ``token``. Raises ``StaleDecision`` for any other turn and
        ``IllegalActionError`` for a move the rules (or ``extra_check``, the mode's own
        restrictions) do not allow; nothing is sent to the engine in either case."""
        with self.cond:
            if self._view is None or token != self._token or self._answer is not None:
                raise StaleDecision("that decision is no longer pending")
            assert self._view.legal is not None
            check_decision(decision, self._view.legal)
            if extra_check is not None:
                extra_check(self._view, decision)
            self._answer = decision
            self.cond.notify_all()

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.cond.notify_all()


class HumanAgent(Agent):
    """A seat played by a person through a ``DecisionChannel``. It keeps the completed hands it
    is shown (redacted to its own seat) for the page's hand log."""

    def __init__(self, channel: DecisionChannel, on_hand: Callable[[HandHistory], None]) -> None:
        self.channel = channel
        self.on_hand = on_hand

    def act(self, view: SeatView, rng: random.Random) -> Decision:
        return self.channel.request(view)

    def hand_completed(self, hand: HandHistory) -> None:
        self.on_hand(hand)

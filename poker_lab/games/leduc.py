"""Leduc hold'em (Southey et al. 2005): two betting rounds, small enough to solve exactly.

Six cards: J, Q, K in two suits. Each player antes 1 and receives one private card. In round 1
player 0 acts first, bets and raises are 2 chips, and there are at most two bets/raises per
round (a bet counts as the first). Then one public card is dealt and round 2 is played the same
way with bets of 4. At showdown, a private card that pairs the public card wins; otherwise the
higher private card wins, and equal ranks split the pot.

Actions: ``k`` check, ``b`` bet, ``f`` fold, ``c`` call, ``r`` raise. Information-set keys look
like ``"JsQh|kbc/b"``: own card, public card (once dealt), round-1 betting, ``/``, round-2
betting. With suits told apart there are 936 information sets.

The constructor takes the deck, ante, bet sizes and raise cap, so larger variants (for example
four ranks) can be built to test how solvers scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from poker_lab.games.base import CHANCE, Game


@dataclass(frozen=True, slots=True)
class LeducState:
    cards: tuple[int, ...] = ()  # player 0's card, player 1's card, then the public card
    rounds: tuple[str, ...] = ("",)  # betting in each round reached so far


def _round_closed(history: str) -> bool:
    return history.endswith("c") or history == "kk"


class LeducHoldem(Game):
    def __init__(
        self,
        ranks: str = "JQK",
        suits: str = "sh",
        ante: float = 1.0,
        raise_sizes: tuple[float, float] = (2.0, 4.0),
        max_raises: int = 2,
    ) -> None:
        if len(raise_sizes) != 2 or min(raise_sizes) <= 0 or ante <= 0 or max_raises < 1:
            raise ValueError("need two positive raise sizes, a positive ante and max_raises >= 1")
        if len(set(ranks)) != len(ranks) or len(set(suits)) != len(suits) or len(suits) < 1:
            raise ValueError("ranks and suits must be distinct characters")
        if len(ranks) * len(suits) < 3:
            raise ValueError("the deck needs at least three cards")
        self.ranks, self.suits = ranks, suits
        self.ante = float(ante)
        self.raise_sizes = (float(raise_sizes[0]), float(raise_sizes[1]))
        self.max_raises = max_raises
        self.deck = tuple(r + s for r in ranks for s in suits)  # id = rank * len(suits) + suit
        standard = (ranks, suits, self.ante, self.raise_sizes, max_raises) == (
            "JQK",
            "sh",
            1.0,
            (2.0, 4.0),
            2,
        )
        self.name = "leduc" if standard else f"leduc_{len(ranks)}x{len(suits)}"

    def rank(self, card: int) -> int:
        return card // len(self.suits)

    def initial_state(self) -> LeducState:
        return LeducState()

    def is_terminal(self, state: LeducState) -> bool:
        history = state.rounds[-1]
        return history.endswith("f") or (len(state.rounds) == 2 and _round_closed(history))

    def current_player(self, state: LeducState) -> int:
        if len(state.cards) < 2 or (len(state.rounds) == 1 and _round_closed(state.rounds[0])):
            return CHANCE
        return len(state.rounds[-1]) % 2

    def legal_actions(self, state: LeducState) -> tuple[str, ...]:
        history = state.rounds[-1]
        if history and history[-1] in "br":
            raises = sum(a in "br" for a in history)
            return ("f", "c", "r") if raises < self.max_raises else ("f", "c")
        return ("k", "b")

    def chance_outcomes(self, state: LeducState) -> tuple[tuple[str, float], ...]:
        remaining = [c for c in range(len(self.deck)) if c not in state.cards]
        return tuple((self.deck[c], 1.0 / len(remaining)) for c in remaining)

    def next_state(self, state: LeducState, action: str) -> LeducState:
        if self.current_player(state) == CHANCE:
            cards = (*state.cards, self.deck.index(action))
            rounds = (*state.rounds, "") if len(cards) == 3 else state.rounds
            return LeducState(cards, rounds)
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal action {action!r} at {state}")
        return LeducState(state.cards, (*state.rounds[:-1], state.rounds[-1] + action))

    def contributions(self, state: LeducState) -> tuple[float, float]:
        """Chips each player has put in so far (ante plus bets)."""
        contrib = [self.ante, self.ante]
        for rnd, history in enumerate(state.rounds):
            level = max(contrib)
            for i, action in enumerate(history):
                player = i % 2
                if action in "br":
                    level += self.raise_sizes[rnd]
                    contrib[player] = level
                elif action == "c":
                    contrib[player] = level
        return contrib[0], contrib[1]

    def utility(self, state: LeducState) -> float:
        c0, c1 = self.contributions(state)
        history = state.rounds[-1]
        if history.endswith("f"):
            folder = (len(history) - 1) % 2
            return c1 if folder == 1 else -c0
        public = self.rank(state.cards[2])
        s0 = (self.rank(state.cards[0]) == public, self.rank(state.cards[0]))
        s1 = (self.rank(state.cards[1]) == public, self.rank(state.cards[1]))
        return c0 if s0 > s1 else -c0 if s0 < s1 else 0.0

    def information_set(self, state: LeducState) -> str:
        player = self.current_player(state)
        public = self.deck[state.cards[2]] if len(state.cards) > 2 else ""
        return f"{self.deck[state.cards[player]]}{public}|{'/'.join(state.rounds)}"

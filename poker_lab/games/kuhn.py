"""Kuhn poker (Kuhn 1950): the smallest interesting poker game.

Three cards (J < Q < K); each player antes 1 and receives one card. Player 0 checks or bets 1.
After a check, player 1 checks (showdown for the antes) or bets 1. Facing a bet, a player
folds or calls; a call goes to showdown, where the higher card wins the pot.

Actions: ``k`` check, ``b`` bet, ``f`` fold, ``c`` call. Information-set keys look like
``"Q|kb"``: the acting player's card, then the betting so far. The game has 12 information
sets (6 per player) and value -1/18 for player 0.
"""

from __future__ import annotations

from dataclasses import dataclass

from poker_lab.games.base import CHANCE, Game

RANKS = "JQK"
_TERMINAL_HISTORIES = frozenset({"kk", "bf", "bc", "kbf", "kbc"})


@dataclass(frozen=True, slots=True)
class KuhnState:
    cards: tuple[int, ...] = ()  # cards[i] is player i's card (0 = J, 1 = Q, 2 = K)
    history: str = ""


class KuhnPoker(Game):
    name = "kuhn"

    def initial_state(self) -> KuhnState:
        return KuhnState()

    def is_terminal(self, state: KuhnState) -> bool:
        return state.history in _TERMINAL_HISTORIES

    def current_player(self, state: KuhnState) -> int:
        if len(state.cards) < 2:
            return CHANCE
        return len(state.history) % 2

    def legal_actions(self, state: KuhnState) -> tuple[str, ...]:
        return ("f", "c") if state.history.endswith("b") else ("k", "b")

    def chance_outcomes(self, state: KuhnState) -> tuple[tuple[str, float], ...]:
        remaining = [c for c in range(len(RANKS)) if c not in state.cards]
        return tuple((RANKS[c], 1.0 / len(remaining)) for c in remaining)

    def next_state(self, state: KuhnState, action: str) -> KuhnState:
        if self.current_player(state) == CHANCE:
            return KuhnState((*state.cards, RANKS.index(action)), state.history)
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal action {action!r} at {state}")
        return KuhnState(state.cards, state.history + action)

    def utility(self, state: KuhnState) -> float:
        h = state.history
        if h.endswith("f"):
            folder = (len(h) - 1) % 2  # the folder loses their ante
            return 1.0 if folder == 1 else -1.0
        stake = 2.0 if "b" in h else 1.0
        return stake if state.cards[0] > state.cards[1] else -stake

    def information_set(self, state: KuhnState) -> str:
        player = self.current_player(state)
        return f"{RANKS[state.cards[player]]}|{state.history}"

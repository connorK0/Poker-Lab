"""Compile a ``Game`` into flat NumPy arrays: the representation every Phase 2 solver runs on.

``compile_tree`` walks the game once, breadth first. That numbering gives three properties the
vectorized algorithms rely on:

1. nodes are sorted by depth, so each depth level is a contiguous slice;
2. a node's children are contiguous;
3. sweeping levels bottom-up visits children before parents (backward induction), and
   top-down visits parents before children (reach probabilities).

A *sequence* is an (information set, action) pair, the unit of the sequence form. Sequences
are numbered so that each information set's actions are contiguous. A behavior strategy for
both players is then a single vector with one probability per sequence, and one CFR iteration
is a handful of gathers and scatters over these arrays rather than a recursive tree walk.

The compiler also checks the assumptions the solvers make: every information set belongs to one
player, has one action list, sits at a single depth, and is reached by a single sequence of the
player's own earlier actions (perfect recall).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from poker_lab.games.base import CHANCE, Game

TERMINAL = -2


@dataclass(frozen=True, eq=False)
class GameTree:
    game_name: str
    # Per node (breadth-first order; node 0 is the root).
    parent: np.ndarray  # int32, -1 at the root
    depth: np.ndarray  # int32
    player: np.ndarray  # int8: 0 or 1 (decision), CHANCE, or TERMINAL
    infoset: np.ndarray  # int32 information set of a decision node, else -1
    edge_seq: np.ndarray  # int32 sequence of the edge parent->node if the parent decides, else -1
    edge_player: np.ndarray  # int8: who chose the edge into the node (0, 1 or CHANCE; root: CHANCE)
    chance_prob: np.ndarray  # float64 probability of a chance edge (1.0 for other edges)
    utility: np.ndarray  # float64 player 0's payoff at terminals, 0 elsewhere
    child_start: np.ndarray  # int32 index of the first child
    num_children: np.ndarray  # int32
    own_seq: np.ndarray  # int32 (2, n): each player's latest own sequence on the path, -1 if none
    labels: tuple[str, ...]  # label of the edge into each node ("" at the root)
    level_start: np.ndarray  # int64: level d is nodes [level_start[d], level_start[d + 1])
    # Per information set.
    infoset_keys: tuple[str, ...]
    infoset_player: np.ndarray  # int8
    infoset_depth: np.ndarray  # int32
    infoset_actions: tuple[tuple[str, ...], ...]
    infoset_parent_seq: np.ndarray  # int32 the owner's sequence leading to it (-1 = empty)
    infoset_node: np.ndarray  # int32 one node of the information set (for own-reach lookups)
    seq_offset: np.ndarray  # int64 (num_infosets + 1): sequences of I are [offset[I], offset[I+1])
    # Per sequence.
    seq_infoset: np.ndarray  # int32
    seq_action: np.ndarray  # int32 action index within its information set
    infoset_index: dict[str, int] = field(repr=False)

    @property
    def num_nodes(self) -> int:
        return len(self.parent)

    @property
    def num_infosets(self) -> int:
        return len(self.infoset_keys)

    @property
    def num_sequences(self) -> int:
        return len(self.seq_infoset)

    @property
    def max_depth(self) -> int:
        return len(self.level_start) - 2

    @property
    def infoset_num_actions(self) -> np.ndarray:
        return np.diff(self.seq_offset)

    def level(self, d: int) -> slice:
        return slice(int(self.level_start[d]), int(self.level_start[d + 1]))

    def sequence(self, key: str, action: str) -> int:
        """Index of the sequence (information set ``key``, ``action``)."""
        i = self.infoset_index[key]
        return int(self.seq_offset[i]) + self.infoset_actions[i].index(action)

    def follow(self, labels: Iterable[str]) -> int:
        """The node reached from the root along edge ``labels`` (chance outcomes and actions)."""
        node = 0
        for label in labels:
            start = int(self.child_start[node])
            children = self.labels[start : start + int(self.num_children[node])]
            if label not in children:
                raise ValueError(f"no edge {label!r} below node {node}")
            node = start + children.index(label)
        return node

    def sequences_of(self, player: int) -> np.ndarray:
        """Boolean mask over sequences owned by ``player``."""
        return self.infoset_player[self.seq_infoset] == player

    def counts(self) -> dict[str, int]:
        is_decision = self.player >= 0
        return {
            "nodes": self.num_nodes,
            "terminals": int(np.sum(self.player == TERMINAL)),
            "chance_nodes": int(np.sum(self.player == CHANCE)),
            "decision_nodes": int(np.sum(is_decision)),
            "infosets": self.num_infosets,
            "infosets_player0": int(np.sum(self.infoset_player == 0)),
            "infosets_player1": int(np.sum(self.infoset_player == 1)),
            "sequences": self.num_sequences,
            "max_depth": self.max_depth,
        }

    # ---- vectorized primitives shared by solvers and evaluators -------------------------------

    def segment_sum(self, x: np.ndarray) -> np.ndarray:
        """Sum of a per-sequence array over each information set's sequences."""
        return np.add.reduceat(x, self.seq_offset[:-1])

    def normalize(self, weights: np.ndarray) -> np.ndarray:
        """Per-information-set normalization of non-negative weights; uniform where all are 0."""
        total = self.segment_sum(weights)[self.seq_infoset]
        uniform = 1.0 / self.infoset_num_actions[self.seq_infoset]
        safe = np.where(total > 0, total, 1.0)
        return np.where(total > 0, weights / safe, uniform)

    def edge_probabilities(self, probs: np.ndarray) -> np.ndarray:
        """Probability of the edge into each node under a strategy profile ``probs``."""
        edge = self.chance_prob.copy()
        decided = self.edge_seq >= 0
        edge[decided] = probs[self.edge_seq[decided]]
        return edge

    def reach(self, edge: np.ndarray) -> np.ndarray:
        """Product of edge probabilities from the root to each node."""
        reach = np.empty(self.num_nodes)
        reach[0] = edge[0]
        for d in range(1, self.max_depth + 1):
            lv = self.level(d)
            reach[lv] = reach[self.parent[lv]] * edge[lv]
        return reach

    def backward(self, edge: np.ndarray, leaf: np.ndarray | None = None) -> np.ndarray:
        """Expected value of every node (player 0's payoff unless ``leaf`` is given)."""
        v = (self.utility if leaf is None else leaf).astype(float, copy=True)
        for d in range(self.max_depth, 0, -1):
            lv, up = self.level(d), self.level(d - 1)
            v[up] += np.bincount(
                self.parent[lv] - up.start, weights=edge[lv] * v[lv], minlength=up.stop - up.start
            )
        return v


def compile_tree(game: Game, *, max_nodes: int = 2_000_000) -> GameTree:
    """Walk ``game`` breadth first and return its flat representation (see module docstring)."""
    parent: list[int] = []
    depth: list[int] = []
    player: list[int] = []
    infoset: list[int] = []
    edge_seq: list[int] = []
    edge_player: list[int] = []
    chance_prob: list[float] = []
    utility: list[float] = []
    child_start: list[int] = []
    num_children: list[int] = []
    own: list[tuple[int, int]] = []
    labels: list[str] = []

    index: dict[str, int] = {}
    keys: list[str] = []
    i_player: list[int] = []
    i_depth: list[int] = []
    i_actions: list[tuple[str, ...]] = []
    i_parent_seq: list[int] = []
    i_node: list[int] = []
    seq_offset = [0]

    # queue entries: state, parent node, edge label, chance probability, edge sequence
    queue = deque([(game.initial_state(), -1, "", 1.0, -1)])
    enqueued = 1
    while queue:
        state, par, label, prob, seq = queue.popleft()
        node = len(parent)
        parent.append(par)
        labels.append(label)
        chance_prob.append(prob)
        edge_seq.append(seq)
        if par < 0:
            depth.append(0)
            edge_player.append(CHANCE)
            own.append((-1, -1))
        else:
            depth.append(depth[par] + 1)
            edge_player.append(player[par])
            o0, o1 = own[par]
            if seq >= 0:
                o0, o1 = (seq, o1) if player[par] == 0 else (o0, seq)
            own.append((o0, o1))
        child_start.append(enqueued)

        if game.is_terminal(state):
            player.append(TERMINAL)
            infoset.append(-1)
            utility.append(float(game.utility(state)))
            num_children.append(0)
            continue

        utility.append(0.0)
        actor = game.current_player(state)
        if actor == CHANCE:
            player.append(CHANCE)
            infoset.append(-1)
            outcomes = game.chance_outcomes(state)
            total = sum(p for _, p in outcomes)
            if not outcomes or abs(total - 1.0) > 1e-9 or min(p for _, p in outcomes) <= 0:
                raise ValueError(f"chance outcomes must be positive and sum to 1 at {state}")
            for out_label, out_prob in outcomes:
                queue.append((game.next_state(state, out_label), node, out_label, out_prob, -1))
            num_children.append(len(outcomes))
            enqueued += len(outcomes)
        else:
            if actor not in (0, 1):
                raise ValueError(f"current_player must be 0, 1 or CHANCE, got {actor}")
            key = game.information_set(state)
            actions = tuple(game.legal_actions(state))
            if not actions or len(set(actions)) != len(actions):
                raise ValueError(f"need distinct legal actions at {state}")
            own_seq = own[node][actor]
            i = index.get(key)
            if i is None:
                i = len(keys)
                index[key] = i
                keys.append(key)
                i_player.append(actor)
                i_depth.append(depth[node])
                i_actions.append(actions)
                i_parent_seq.append(own_seq)
                i_node.append(node)
                seq_offset.append(seq_offset[-1] + len(actions))
            elif i_player[i] != actor or i_actions[i] != actions:
                raise ValueError(f"information set {key!r} mixes players or action lists")
            elif i_depth[i] != depth[node]:
                raise ValueError(f"information set {key!r} spans depths (unsupported)")
            elif i_parent_seq[i] != own_seq:
                raise ValueError(f"imperfect recall: {key!r} is reached by different own actions")
            player.append(actor)
            infoset.append(i)
            for a, action in enumerate(actions):
                queue.append((game.next_state(state, action), node, action, 1.0, seq_offset[i] + a))
            num_children.append(len(actions))
            enqueued += len(actions)
        if enqueued > max_nodes:
            raise ValueError(f"game tree exceeds max_nodes={max_nodes}")

    depth_arr = np.asarray(depth, dtype=np.int32)
    level_start = np.searchsorted(depth_arr, np.arange(depth_arr[-1] + 2), side="left")
    offsets = np.asarray(seq_offset, dtype=np.int64)
    n_actions = np.diff(offsets)
    return GameTree(
        game_name=game.name,
        parent=np.asarray(parent, dtype=np.int32),
        depth=depth_arr,
        player=np.asarray(player, dtype=np.int8),
        infoset=np.asarray(infoset, dtype=np.int32),
        edge_seq=np.asarray(edge_seq, dtype=np.int32),
        edge_player=np.asarray(edge_player, dtype=np.int8),
        chance_prob=np.asarray(chance_prob, dtype=float),
        utility=np.asarray(utility, dtype=float),
        child_start=np.asarray(child_start, dtype=np.int32),
        num_children=np.asarray(num_children, dtype=np.int32),
        own_seq=np.asarray(own, dtype=np.int32).T.copy(),
        labels=tuple(labels),
        level_start=level_start.astype(np.int64),
        infoset_keys=tuple(keys),
        infoset_player=np.asarray(i_player, dtype=np.int8),
        infoset_depth=np.asarray(i_depth, dtype=np.int32),
        infoset_actions=tuple(i_actions),
        infoset_parent_seq=np.asarray(i_parent_seq, dtype=np.int32),
        infoset_node=np.asarray(i_node, dtype=np.int32),
        seq_offset=offsets,
        seq_infoset=np.repeat(np.arange(len(keys), dtype=np.int32), n_actions),
        seq_action=(np.arange(offsets[-1]) - np.repeat(offsets[:-1], n_actions)).astype(np.int32),
        infoset_index=index,
    )

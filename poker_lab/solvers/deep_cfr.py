"""Deep CFR (Brown, Lerer, Gross, Sandholm 2019) for small games, in PyTorch.

Tabular CFR stores a regret for every (information set, action). Deep CFR replaces the table
with a neural network that *generalizes* across similar information sets, which is what makes
CFR usable in games far too large to enumerate. Each iteration ``t`` and each player ``p``:

1. **Traverse.** Run ``K`` external-sampling traversals (as in MCCFR). At ``p``'s information
   sets the strategy is regret matching on the *predicted advantages* ``V_p(I)``; if none is
   positive, the action with the largest prediction is played (as Deep CFR does; uniform play
   is the textbook alternative, ``argmax_if_no_positive=False``). Before ``V_p`` is first
   trained, play is uniform. Each visited ``p`` information set yields a sample ``(I, t, r)``
   with the sampled regrets ``r(a) = v(a) - sum_b sigma(b) v(b)``, stored in ``p``'s
   advantage memory. Each opponent information set yields ``(I, t, sigma(I))`` for the
   strategy memory.
2. **Fit.** Re-initialize ``V_p`` and train it from scratch to minimize the ``t``-weighted
   squared error ``sum_i t_i ||V_p(I_i) - r_i||^2`` over the memory (legal actions only). The
   minimizer at each ``I`` is the weighted average of its sampled regrets, which is
   proportional to linear CFR's cumulative regret, and regret matching ignores the scale.

At the end a policy network ``Pi`` is trained on the strategy memory (weights ``t``,
cross-entropy with the stored strategies as soft targets). Its minimizer at each ``I`` is the
``t``-weighted average strategy, i.e. linear CFR's average.

Memories are reservoir samples (Vitter's algorithm R): when full, the ``n``-th sample replaces
a random slot with probability ``capacity / n``, so the memory is always a uniform sample of
everything generated.

**Small-game shortcut (exact, not an approximation).** Within an iteration the networks are
fixed, so the strategy they induce is a fixed function of the information set. Leduc's 936
information sets can be enumerated, so each iteration evaluates the network once on all of
them and traverses with that table. Samples store information-set *indices*; features come
from a precomputed table. Because everything is enumerable, these policies can be evaluated
*exactly*:

* ``policy_network``: the trained ``Pi`` (Deep CFR's actual output);
* ``exact_average``: the exact ``t``- and reach-weighted average of the strategies the
  advantage networks produced. This is what the strategy memory samples, and the policy that
  Single Deep CFR (Steinberger 2019) plays. Comparing it with ``policy_network`` isolates the
  error added by fitting ``Pi``;
* ``current``: the last iteration's strategy (not expected to converge).

**Ablation: a perfect regressor.** With ``regressor="tabular"`` the networks are replaced by
the exact minimizers of the same losses: the weighted mean of each information set's samples in
memory. Everything else (traversals, memories, weights) is unchanged, so the gap between the
two isolates the cost of function approximation.

**Reproducibility.** Every network is initialized from a seed derived from (seed, iteration,
role), and every minibatch stream likewise, so evaluating at extra checkpoints never changes
the training trajectory. Results are bit-identical for a fixed seed and thread count.

Features (``InfosetEncoder``): one-hot ranks of the private and public card (suits are ignored,
so suit-isomorphic information sets share features) and one-hot actions per (round, position).
Output slots are fixed by action type: fold, check/call, bet/raise.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from poker_lab.games.base import CHANCE
from poker_lab.games.policy import TabularPolicy
from poker_lab.games.tree import TERMINAL, GameTree

SLOTS = {"f": 0, "k": 1, "c": 1, "b": 2, "r": 2}
N_SLOTS = 3
HISTORY_ACTIONS = "kbcr"  # actions that can appear in a non-terminal history
_POLICY_ROLE = 2  # seed salt for the policy network (advantage networks use the player id)


@dataclass(frozen=True)
class DeepCFRConfig:
    traversals: int = 1000  # per player per iteration
    hidden: tuple[int, ...] = (64, 64, 64)
    # Training budget chosen on development seed 99 (never a reporting seed) by how well the
    # network fits a realistic memory: 600 Adam steps with lr 3e-3 and cosine decay reach
    # R^2 0.94 against the per-infoset targets in ~2 s; 1,000 steps only reach 0.95.
    advantage_steps: int = 600
    advantage_batch: int = 1024
    advantage_lr: float = 3e-3
    policy_steps: int = 2000
    policy_batch: int = 2048
    policy_lr: float = 3e-3
    cosine_lr: bool = True
    memory_capacity: int = 1_000_000
    grad_clip: float = 1.0
    argmax_if_no_positive: bool = True
    regressor: str = "network"  # or "tabular": the perfect-regressor ablation
    threads: int = 1

    def __post_init__(self) -> None:
        if self.regressor not in ("network", "tabular"):
            raise ValueError("regressor must be 'network' or 'tabular'")


class InfosetEncoder:
    """Feature table and action-slot map for every information set of a small poker game.

    Information-set keys look like ``"<own card><public card>|<round 1>/<round 2>"``, with
    card labels whose first character is the rank (Kuhn ``"Q|kb"``, Leduc ``"JsQh|kbc/b"``).
    """

    def __init__(self, tree: GameTree, ranks: str, card_length: int) -> None:
        keys = tree.infoset_keys
        histories = [k.split("|", 1)[1].split("/") for k in keys]
        self.rounds = max(len(h) for h in histories)
        self.positions = max(len(r) for h in histories for r in h) or 1
        n_ranks = len(ranks)
        width = 2 * n_ranks + 1 + self.rounds * self.positions * len(HISTORY_ACTIONS)
        feats = np.zeros((len(keys), width), dtype=np.float32)
        legal = np.zeros((len(keys), N_SLOTS), dtype=bool)
        slot_of_seq = np.zeros(tree.num_sequences, dtype=np.int64)
        for i, key in enumerate(keys):
            cards, history = key.split("|", 1)
            own, public = cards[:card_length], cards[card_length:]
            feats[i, ranks.index(own[0])] = 1.0
            if public:
                feats[i, n_ranks + ranks.index(public[0])] = 1.0
                feats[i, 2 * n_ranks] = 1.0
            base = 2 * n_ranks + 1
            for r, rnd in enumerate(history.split("/")):
                for pos, action in enumerate(rnd):
                    offset = (r * self.positions + pos) * len(HISTORY_ACTIONS)
                    feats[i, base + offset + HISTORY_ACTIONS.index(action)] = 1.0
            for a, action in enumerate(tree.infoset_actions[i]):
                slot = SLOTS[action]
                if legal[i, slot]:
                    raise ValueError(f"two actions share slot {slot} at {key!r}")
                legal[i, slot] = True
                slot_of_seq[tree.seq_offset[i] + a] = slot
        self.features = feats
        self.legal = legal
        self.slot_of_seq = slot_of_seq

    @classmethod
    def for_game(cls, tree: GameTree, game) -> InfosetEncoder:
        ranks = getattr(game, "ranks", "JQK")
        card_length = 2 if hasattr(game, "suits") else 1
        return cls(tree, ranks, card_length)


class Reservoir:
    """Uniform reservoir sample of (infoset, iteration, target vector) records."""

    def __init__(self, capacity: int, width: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.infoset = np.zeros(capacity, dtype=np.int64)
        self.iteration = np.zeros(capacity, dtype=np.float32)
        self.target = np.zeros((capacity, width), dtype=np.float32)
        self.seen = 0
        self._rng = rng

    def __len__(self) -> int:
        return min(self.seen, self.capacity)

    def add(self, infoset: np.ndarray, iteration: int, target: np.ndarray) -> None:
        for k in range(len(infoset)):
            if self.seen < self.capacity:
                slot = self.seen
            else:
                slot = int(self._rng.integers(0, self.seen + 1))
                if slot >= self.capacity:
                    self.seen += 1
                    continue
            self.infoset[slot] = infoset[k]
            self.iteration[slot] = iteration
            self.target[slot] = target[k]
            self.seen += 1

    def weighted_means(self, num_infosets: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-infoset ``iteration``-weighted mean target and total weight."""
        n = len(self)
        sums = np.zeros((num_infosets, self.target.shape[1]))
        weights = np.zeros(num_infosets)
        np.add.at(sums, self.infoset[:n], self.target[:n] * self.iteration[:n, None])
        np.add.at(weights, self.infoset[:n], self.iteration[:n])
        return sums / np.where(weights > 0, weights, 1.0)[:, None], weights


def _mlp(n_in: int, hidden: tuple[int, ...], n_out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = n_in
    for h in hidden:
        layers += [nn.Linear(width, h), nn.ReLU()]
        width = h
    layers.append(nn.Linear(width, n_out))
    return nn.Sequential(*layers)


class DeepCFR:
    """Deep CFR on a compiled small game (see the module docstring)."""

    def __init__(self, tree: GameTree, encoder: InfosetEncoder, config: DeepCFRConfig, seed: int):
        self.tree, self.encoder, self.config, self.seed = tree, encoder, config, seed
        torch.set_num_threads(config.threads)
        self.rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        self.iteration = 0
        self.traversals = 0
        self.features = torch.from_numpy(encoder.features)
        self.legal = torch.from_numpy(encoder.legal)
        self.advantage_nets: list[nn.Sequential | None] = [None, None]  # None: play uniformly
        self.advantage_memory = [Reservoir(config.memory_capacity, N_SLOTS, np_rng) for _ in (0, 1)]
        self.strategy_memory = Reservoir(config.memory_capacity, N_SLOTS, np_rng)
        self.policy_net: nn.Sequential | None = None
        # exact linear-weighted, reach-weighted average of the strategies actually played
        self._avg_sum = np.zeros(tree.num_sequences)
        self._current = TabularPolicy.uniform(tree).probs.copy()
        # plain lists for the recursive traversal (list indexing beats NumPy for tiny arrays)
        self._player = tree.player.tolist()
        self._infoset = tree.infoset.tolist()
        self._child = tree.child_start.tolist()
        self._utility = tree.utility.tolist()
        self._slots = [
            encoder.slot_of_seq[tree.seq_offset[i] : tree.seq_offset[i + 1]].tolist()
            for i in range(tree.num_infosets)
        ]
        self._chance_cum = {
            int(n): np.cumsum(
                tree.chance_prob[tree.child_start[n] : tree.child_start[n] + tree.num_children[n]]
            ).tolist()
            for n in np.flatnonzero(tree.player == CHANCE)
        }
        self._owner_mask = [tree.infoset_player[tree.seq_infoset] == p for p in (0, 1)]
        self._edge_owner = [tree.edge_player == p for p in (0, 1)]

    def _derived_seed(self, role: int) -> int:
        state = np.random.SeedSequence([self.seed, self.iteration, role]).generate_state(1)
        return int(state[0])

    def _new_net(self, role: int) -> nn.Sequential:
        with torch.random.fork_rng():
            torch.manual_seed(self._derived_seed(role))
            return _mlp(self.features.shape[1], self.config.hidden, N_SLOTS)

    # ---- strategies ---------------------------------------------------------------------------

    @torch.no_grad()
    def predicted_advantages(self, player: int) -> np.ndarray | None:
        """``(num_infosets, 3)`` advantages for ``player`` (None while untrained)."""
        if self.config.regressor == "tabular":
            if len(self.advantage_memory[player]) == 0:
                return None
            return self.advantage_memory[player].weighted_means(self.tree.num_infosets)[0]
        net = self.advantage_nets[player]
        return None if net is None else net(self.features).numpy().astype(float)

    def _strategy_table(self, player: int) -> np.ndarray:
        """Per-sequence probabilities for ``player``'s infosets (regret matching)."""
        adv = self.predicted_advantages(player)
        if adv is None:
            uniform = 1.0 / self.tree.infoset_num_actions[self.tree.seq_infoset]
            return np.where(self._owner_mask[player], uniform, 0.0)
        legal = self.encoder.legal
        adv = np.where(legal, adv, -np.inf)
        pos = np.where(legal, np.maximum(adv, 0.0), 0.0)
        total = pos.sum(axis=1, keepdims=True)
        if self.config.argmax_if_no_positive:
            fallback = np.zeros_like(pos)
            fallback[np.arange(len(adv)), np.argmax(adv, axis=1)] = 1.0
        else:
            fallback = legal / legal.sum(axis=1, keepdims=True)
        probs_slots = np.where(total > 0, pos / np.where(total > 0, total, 1.0), fallback)
        per_seq = probs_slots[self.tree.seq_infoset, self.encoder.slot_of_seq]
        return np.where(self._owner_mask[player], per_seq, 0.0)

    def _accumulate_average(self, player: int, table: np.ndarray, weight: float) -> None:
        """Add ``weight * own reach * strategy`` for ``player`` (exact sequence-form average)."""
        tree = self.tree
        edge = np.where(self._edge_owner[player], tree.edge_probabilities(table), 1.0)
        own_reach = tree.reach(edge)[tree.infoset_node][tree.seq_infoset]
        mask = self._owner_mask[player]
        self._avg_sum[mask] += weight * (own_reach * table)[mask]

    # ---- traversal ----------------------------------------------------------------------------

    def _traverse(self, node: int, player: int, tables, adv_out: list, strat_out: list) -> float:
        actor = self._player[node]
        if actor == TERMINAL:
            u = self._utility[node]
            return u if player == 0 else -u
        if actor == CHANCE:
            u, cum = self.rng.random(), self._chance_cum[node]
            k = 0
            while k < len(cum) - 1 and u >= cum[k]:
                k += 1
            return self._traverse(self._child[node] + k, player, tables, adv_out, strat_out)
        i = self._infoset[node]
        sigma = tables[actor][i]
        first = self._child[node]
        if actor == player:
            values = [
                self._traverse(first + a, player, tables, adv_out, strat_out)
                for a in range(len(sigma))
            ]
            v = sum(s * x for s, x in zip(sigma, values, strict=True))
            regrets = [0.0] * N_SLOTS
            for a, slot in enumerate(self._slots[i]):
                regrets[slot] = values[a] - v
            adv_out.append((i, regrets))
            return v
        probs = [0.0] * N_SLOTS
        for a, slot in enumerate(self._slots[i]):
            probs[slot] = sigma[a]
        strat_out.append((i, probs))
        u, acc, a = self.rng.random(), 0.0, 0
        for a in range(len(sigma)):  # after the loop, a is the sampled action
            acc += sigma[a]
            if u < acc:
                break
        return self._traverse(first + a, player, tables, adv_out, strat_out)

    # ---- training -----------------------------------------------------------------------------

    def _fit(self, net, memory: Reservoir, steps, batch, lr, loss_fn, role: int) -> float:
        n = len(memory)
        if n == 0:
            return float("nan")
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        schedule = (
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
            if self.config.cosine_lr
            else None
        )
        infoset = torch.from_numpy(memory.infoset[:n])
        weight = torch.from_numpy(memory.iteration[:n])
        weight = weight / weight.mean()
        target = torch.from_numpy(memory.target[:n])
        gen = torch.Generator().manual_seed(self._derived_seed(10 + role))
        loss = torch.tensor(float("nan"))
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch, n),), generator=gen)
            ids = infoset[idx]
            out = net(self.features[ids])
            loss = loss_fn(out, target[idx], self.legal[ids], weight[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), self.config.grad_clip)
            opt.step()
            if schedule is not None:
                schedule.step()
        return float(loss.detach())

    @staticmethod
    def _advantage_loss(out, target, legal, weight):
        sq = ((out - target) ** 2) * legal
        return (weight * sq.sum(dim=1)).mean()

    @staticmethod
    def _policy_loss(out, target, legal, weight):
        logits = out.masked_fill(~legal, -1e9)
        logp = torch.log_softmax(logits, dim=1)
        return (weight * -(target * logp).sum(dim=1)).mean()

    def iterate(self, n: int = 1) -> None:
        cfg = self.config
        offsets = self.tree.seq_offset
        for _ in range(n):
            self.iteration += 1
            t = self.iteration
            for player in (0, 1):
                tables = [self._strategy_table(p) for p in (0, 1)]
                lists = [
                    [
                        tables[p][offsets[i] : offsets[i + 1]].tolist()
                        for i in range(len(offsets) - 1)
                    ]
                    for p in (0, 1)
                ]
                adv_out: list = []
                strat_out: list = []
                for _ in range(cfg.traversals):
                    self._traverse(0, player, lists, adv_out, strat_out)
                self.traversals += cfg.traversals
                opponent = 1 - player
                self._accumulate_average(opponent, tables[opponent], float(t))
                mask = self._owner_mask[opponent]
                self._current[mask] = tables[opponent][mask]
                if adv_out:
                    ids, regrets = zip(*adv_out, strict=True)
                    self.advantage_memory[player].add(np.array(ids), t, np.array(regrets))
                if strat_out:
                    ids, probs = zip(*strat_out, strict=True)
                    self.strategy_memory.add(np.array(ids), t, np.array(probs))
                if cfg.regressor == "network":
                    net = self._new_net(player)
                    self._fit(
                        net,
                        self.advantage_memory[player],
                        cfg.advantage_steps,
                        cfg.advantage_batch,
                        cfg.advantage_lr,
                        self._advantage_loss,
                        role=player,
                    )
                    self.advantage_nets[player] = net

    def train_policy_network(self) -> nn.Sequential:
        cfg = self.config
        net = self._new_net(_POLICY_ROLE)
        self._fit(
            net,
            self.strategy_memory,
            cfg.policy_steps,
            cfg.policy_batch,
            cfg.policy_lr,
            self._policy_loss,
            role=_POLICY_ROLE,
        )
        self.policy_net = net
        return net

    # ---- policies -----------------------------------------------------------------------------

    @torch.no_grad()
    def network_policy(self, net: nn.Sequential | None = None) -> TabularPolicy:
        """Deep CFR's output: the policy network (the memory average for the tabular ablation)."""
        if self.config.regressor == "tabular":
            return self.memory_average_policy()
        net = net or self.policy_net or self.train_policy_network()
        logits = net(self.features).masked_fill(~self.legal, -1e9)
        probs = torch.softmax(logits, dim=1).numpy().astype(float)
        per_seq = probs[self.tree.seq_infoset, self.encoder.slot_of_seq]
        return TabularPolicy.from_weights(self.tree, per_seq)

    def exact_average_policy(self) -> TabularPolicy:
        return TabularPolicy.from_weights(self.tree, self._avg_sum)

    def current_policy(self) -> TabularPolicy:
        return TabularPolicy.from_weights(self.tree, self._current)

    def memory_average_policy(self) -> TabularPolicy:
        """Weighted average of the stored strategy samples per infoset (what ``Pi`` fits)."""
        means, _ = self.strategy_memory.weighted_means(self.tree.num_infosets)
        per_seq = means[self.tree.seq_infoset, self.encoder.slot_of_seq]
        return TabularPolicy.from_weights(self.tree, per_seq)

    def run(
        self,
        checkpoints: Iterable[int],
        evaluate: Callable[[TabularPolicy], float],
        *,
        policy_checkpoints: Iterable[int] = (),
    ) -> list[dict[str, float]]:
        """Iterate to each checkpoint and evaluate the policies (``evaluate`` maps a policy to
        a number, e.g. exact exploitability). The policy network is trained only at
        ``policy_checkpoints`` (it is the expensive part)."""
        policy_at = set(policy_checkpoints)
        curve = []
        elapsed = 0.0
        for target in sorted(set(checkpoints) | policy_at):
            t0 = time.perf_counter()
            self.iterate(target - self.iteration)
            elapsed += time.perf_counter() - t0
            point = {
                "iteration": self.iteration,
                "traversals": self.traversals,
                "seconds": elapsed,
                "exact_average": evaluate(self.exact_average_policy()),
                "current": evaluate(self.current_policy()),
            }
            if target in policy_at:
                t0 = time.perf_counter()
                if self.config.regressor == "network":
                    self.train_policy_network()  # a fresh Pi on the memory so far
                point["policy_network"] = evaluate(self.network_policy())
                point["policy_seconds"] = time.perf_counter() - t0
                point["memory_average"] = evaluate(self.memory_average_policy())
            curve.append(point)
        return curve

    def describe(self) -> dict[str, object]:
        net = _mlp(self.features.shape[1], self.config.hidden, N_SLOTS)
        return {
            "config": asdict(self.config),
            "seed": self.seed,
            "features": int(self.features.shape[1]),
            "parameters_per_network": sum(p.numel() for p in net.parameters()),
            "torch": torch.__version__,
        }

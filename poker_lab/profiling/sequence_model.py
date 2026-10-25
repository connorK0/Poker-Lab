"""Archetype classifier on raw action sequences, in PyTorch (P3.8).

The Phase 1 classifier sees a player through hand-written statistics (VPIP, c-bet, ...). This
one reads the player's hands as ``sequences`` tokens and learns its own summary:

1. **Hand encoder.** Each token goes through a small MLP. With ``encoder="gru"`` a GRU reads
   the player's tokens of one hand in order and its last state is the hand's embedding, so a
   line (raise preflop, bet the flop, check the turn) can mean more than its parts. With
   ``encoder="bag"`` the token embeddings of a hand are summed instead (order ignored), which
   isolates what the order adds. A hand of hand-level vectors (one token per hand, e.g. the
   stats counters) makes ``bag`` an MLP on that vector.
2. **Pooling over hands.** The bots are stationary, so a player's hands are exchangeable and
   their order carries nothing; the window summary is a symmetric function of its hands (a
   "deep set", Zaheer et al. 2017). With ``pool="rates"`` (the default) each hand gives, for
   ``n_rates`` learned "stats", a soft opportunity ``c`` in [0, 1] and the share ``t`` of it
   taken: soft successes ``c * t`` and failures ``c * (1 - t)``. A window sums them (S, F)
   and reads each stat as the logit of its posterior mean under a learned Beta(a, b) prior,
   ``log(S + a) - log(F + b)``: a shrunk rate, as Phase 1's hand-written stats are, so a short
   window is pulled toward the prior and a long one converges. With ``pool="mean"`` the window
   is the mean of the hand embeddings instead (development found this plateaus well below the
   logistic model at large samples).
3. **Head.** An MLP reads the pooled features and ``log n`` and gives archetype logits.

Training uses windows of every requested size at random starts in each training player's
hands; all windows of a batch of players come from one pass of the encoder over those players'
hands (window sums are differences of cumulative sums), so a hand is encoded once per epoch
however many windows contain it. Adam with a cosine-decayed learning rate and clipped
gradients. A held-out fraction of the training players (the same grouped split as
``ArchetypeClassifier``) picks the epoch (its log-loss over disjoint windows) and then fits the
temperatures: one for all sizes (``"global"``) and one per training size (``"by_size"``), as
the Phase 1 classifier does.

Reproducibility: weights, window draws and batch order come from ``random_state``; results are
identical for a fixed seed and thread count (``torch.set_num_threads``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import special
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from poker_lab.profiling.classifier import CALIBRATIONS, MIN_ROWS_PER_SIZE, fit_temperature

ENCODERS = ("gru", "bag")
POOLS = ("rates", "mean")


@dataclass(frozen=True)
class HandSequences:
    """Players' hands as token sequences, stored compressed-row style: player ``p``'s hands
    are ``player_ptr[p]:player_ptr[p + 1]``, hand ``h``'s tokens are rows
    ``hand_ptr[h]:hand_ptr[h + 1]`` of ``tokens``."""

    tokens: np.ndarray  # (n_tokens, n_features) float32
    hand_ptr: np.ndarray  # (n_hands + 1,) int64
    player_ptr: np.ndarray  # (n_players + 1,) int64

    @classmethod
    def from_rows(cls, rows: np.ndarray, n_players: int, *, end_column: int = 0) -> HandSequences:
        """From rows ``(player, seq, *token)`` in any order, where a token with a nonzero
        ``end_column`` closes its hand. Rows of the same (player, seq) keep their order."""
        order = np.lexsort((rows[:, 1], rows[:, 0]))  # stable: ties keep their order
        rows = rows[order]
        player = rows[:, 0].astype(np.int64)
        tokens = np.ascontiguousarray(rows[:, 2:], dtype=np.float32)
        ends = np.flatnonzero(tokens[:, end_column] != 0)
        if len(ends) == 0 or ends[-1] != len(tokens) - 1:
            raise ValueError("every hand must end with a closing token")
        hand_ptr = np.concatenate([[0], ends + 1]).astype(np.int64)
        hand_player = player[ends]
        if np.any(np.diff(hand_player) < 0) or np.any(player[hand_ptr[:-1]] != hand_player):
            raise ValueError("a hand's tokens belong to more than one player")
        hands_per_player = np.bincount(hand_player, minlength=n_players)
        player_ptr = np.concatenate([[0], np.cumsum(hands_per_player)]).astype(np.int64)
        return cls(tokens, hand_ptr, player_ptr)

    @classmethod
    def from_vectors(cls, vectors: list[np.ndarray] | np.ndarray) -> HandSequences:
        """One token per hand: ``vectors[p]`` is (hands, n_features) for player ``p``."""
        hands = [len(v) for v in vectors]
        tokens = np.concatenate([np.asarray(v, dtype=np.float32) for v in vectors])
        return cls(
            tokens,
            np.arange(len(tokens) + 1, dtype=np.int64),
            np.concatenate([[0], np.cumsum(hands)]).astype(np.int64),
        )

    @property
    def n_players(self) -> int:
        return len(self.player_ptr) - 1

    @property
    def n_features(self) -> int:
        return self.tokens.shape[1]

    def hands(self, p: int | None = None) -> np.ndarray | int:
        counts = np.diff(self.player_ptr)
        return counts if p is None else int(counts[p])

    def player_tokens(self, players: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tokens of every hand of ``players`` (in that order), the token count of each hand,
        and each player's offset into the returned hands."""
        tok, lengths, offsets = [], [], [0]
        for p in players:
            h0, h1 = self.player_ptr[p], self.player_ptr[p + 1]
            tok.append(self.tokens[self.hand_ptr[h0] : self.hand_ptr[h1]])
            lengths.append(np.diff(self.hand_ptr[h0 : h1 + 1]))
            offsets.append(offsets[-1] + h1 - h0)
        return np.concatenate(tok), np.concatenate(lengths), np.array(offsets[:-1])


class _Net(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int,
        encoder: str,
        n_classes: int,
        dropout: float,
        pool: str = "mean",
        n_rates: int = 32,
    ):
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.n_rates = n_rates
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("scale", torch.ones(n_features))
        self.embed = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        if encoder == "gru":
            self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        else:
            self.post = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        width = hidden
        if pool == "rates":
            self.counts = nn.Linear(hidden, 2 * n_rates)
            self.log_prior = nn.Parameter(torch.zeros(2, n_rates))  # log pseudo-counts
            width = n_rates
        self.head = nn.Sequential(
            nn.Linear(width + 1, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def hands(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Per-hand outputs for hands given as concatenated tokens: the embedding (``mean``
        pooling) or soft success and failure counts of each learned rate (``rates``)."""
        x = self.embed((tokens - self.mean) / self.scale)
        hand_of = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
        if self.encoder == "bag":
            h = self.post(x.new_zeros(len(lengths), x.shape[1]).index_add_(0, hand_of, x))
        else:
            starts = torch.cumsum(lengths, 0) - lengths
            pos = torch.arange(len(x)) - starts[hand_of]
            padded = x.new_zeros(len(lengths), int(lengths.max()), x.shape[1])
            padded[hand_of, pos] = x
            packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
            h = self.rnn(packed)[1][0]
        if self.pool == "mean":
            return h
        z = self.counts(h)
        chance = torch.sigmoid(z[:, : self.n_rates])  # an opportunity in this hand (0-1)
        taken = torch.sigmoid(z[:, self.n_rates :])  # ... and the "success" share of it
        return torch.cat([chance * taken, chance * (1 - taken)], dim=1)

    def logits(self, emb: torch.Tensor, starts: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """Logits of windows ``[starts, starts + n)`` of the per-hand outputs ``emb``."""
        cum = torch.cat([emb.new_zeros(1, emb.shape[1]), emb]).double().cumsum(0)
        total = (cum[starts + n] - cum[starts]).float()
        if self.pool == "mean":
            x = total / n[:, None]
        else:  # logit of each rate's posterior mean under a learned Beta(a, b) prior
            a, b = torch.exp(self.log_prior)
            x = torch.log(total[:, : self.n_rates] + a) - torch.log(total[:, self.n_rates :] + b)
        return self.head(torch.cat([x, torch.log(n.float())[:, None]], dim=1))


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    seconds: float = 0.0


class SequenceClassifier:
    """Archetype probabilities from windows of a player's hands (see the module docstring)."""

    def __init__(
        self,
        encoder: str = "gru",
        *,
        pool: str = "rates",
        n_rates: int = 32,
        hidden: int = 64,
        epochs: int = 40,
        patience: int = 6,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        players_per_batch: int = 8,
        schedule: str = "cosine",
        clip_norm: float | None = 1.0,
        calibration_fraction: float = 0.2,
        random_state: int = 0,
    ) -> None:
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder!r}; choose from {ENCODERS}")
        if pool not in POOLS:
            raise ValueError(f"unknown pool {pool!r}; choose from {POOLS}")
        if schedule not in ("constant", "cosine"):
            raise ValueError("schedule must be 'constant' or 'cosine'")
        self.encoder = encoder
        self.pool = pool
        self.n_rates = n_rates
        self.hidden = hidden
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.players_per_batch = players_per_batch
        self.schedule = schedule
        self.clip_norm = clip_norm
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state
        self.temperature = 1.0
        self.size_temperatures: tuple[np.ndarray, np.ndarray] | None = None  # (log n, T)
        self.history = TrainingHistory()
        self.net: _Net | None = None
        self.val_players: np.ndarray | None = None

    # ------------------------------------------------------------------ helpers
    def _embed(self, seqs: HandSequences, players: np.ndarray):
        tok, lengths, offsets = seqs.player_tokens(players)
        emb = self.net.hands(torch.from_numpy(tok), torch.from_numpy(lengths))
        return emb, offsets

    @staticmethod
    def _disjoint(seqs: HandSequences, players, sizes, max_windows):
        """Phase 1's windows: ``[w * n, (w + 1) * n)`` for w < min(max_windows, hands // n)."""
        rows = []
        for n in sizes:
            for p in players:
                k = min(max_windows, seqs.hands(p) // n)
                rows += [(p, w * n, n) for w in range(k)]
        return np.array(rows, dtype=np.int64).reshape(-1, 3)

    def _log_proba(self, seqs: HandSequences, windows: np.ndarray, chunk: int = 16) -> np.ndarray:
        """Uncalibrated log-probabilities for windows (player, start, n), in their order."""
        out = np.zeros((len(windows), self.n_classes))
        self.net.eval()
        with torch.no_grad():
            players = np.unique(windows[:, 0])
            for c in range(0, len(players), chunk):
                block = players[c : c + chunk]
                emb, offsets = self._embed(seqs, block)
                where = {int(p): o for p, o in zip(block, offsets, strict=True)}
                mask = np.isin(windows[:, 0], block)
                w = windows[mask]
                starts = torch.from_numpy(np.array([where[int(p)] for p in w[:, 0]]) + w[:, 1])
                logits = self.net.logits(emb, starts, torch.from_numpy(w[:, 2]))
                out[mask] = torch.log_softmax(logits, dim=1).numpy()
        return out

    # ------------------------------------------------------------------ fitting
    def fit(
        self, seqs: HandSequences, labels: np.ndarray, sizes: list[int], max_windows: int = 8
    ) -> SequenceClassifier:
        """Fit on every player of ``seqs`` (``labels[p]`` is player ``p``'s archetype), with
        windows of each size in ``sizes``: per player and size, ``min(max_windows, hands //
        n)`` windows at random starts each epoch."""
        import time

        t0 = time.perf_counter()
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        labels = np.asarray(labels, dtype=np.int64)
        self.n_classes = int(labels.max()) + 1
        players = np.arange(seqs.n_players)
        split = GroupShuffleSplit(
            n_splits=1, test_size=self.calibration_fraction, random_state=self.random_state
        )
        fit_players, val_players = next(split.split(players, labels, players))
        self.val_players = val_players
        self.net = _Net(
            seqs.n_features,
            self.hidden,
            self.encoder,
            self.n_classes,
            self.dropout,
            self.pool,
            self.n_rates,
        )
        fit_tok = np.concatenate([seqs.player_tokens([p])[0] for p in fit_players])
        self.net.mean.copy_(torch.from_numpy(fit_tok.mean(axis=0)))
        self.net.scale.copy_(torch.from_numpy(np.maximum(fit_tok.std(axis=0), 1e-3)))
        del fit_tok
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps = self.epochs * -(-len(fit_players) // self.players_per_batch)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            (lambda i: 0.5 * (1 + np.cos(np.pi * i / steps)))
            if self.schedule == "cosine"
            else (lambda i: 1.0),
        )
        val_windows = self._disjoint(seqs, val_players, sizes, max_windows)
        best, best_state, stale = np.inf, None, 0
        for epoch in range(self.epochs):
            self.net.train()
            order = rng.permutation(fit_players)
            losses = []
            for b in range(0, len(order), self.players_per_batch):
                block = order[b : b + self.players_per_batch]
                emb, offsets = self._embed(seqs, block)
                starts, ns, ys = [], [], []
                for p, off in zip(block, offsets, strict=True):
                    h = seqs.hands(p)
                    for n in sizes:
                        k = min(max_windows, h // n)
                        if k:
                            starts.append(off + rng.integers(0, h - n + 1, size=k))
                            ns.append(np.full(k, n))
                            ys.append(np.full(k, labels[p]))
                logits = self.net.logits(
                    emb,
                    torch.from_numpy(np.concatenate(starts)),
                    torch.from_numpy(np.concatenate(ns)),
                )
                loss = nn.functional.cross_entropy(logits, torch.from_numpy(np.concatenate(ys)))
                opt.zero_grad()
                loss.backward()
                if self.clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_norm)
                opt.step()
                sched.step()
                losses.append(loss.item())
            log_p = self._log_proba(seqs, val_windows)
            val = float(-log_p[np.arange(len(log_p)), labels[val_windows[:, 0]]].mean())
            self.history.train_loss.append(float(np.mean(losses)))
            self.history.val_loss.append(val)
            if val < best - 1e-4:
                best, best_state, stale = val, copy.deepcopy(self.net.state_dict()), 0
                self.history.best_epoch = epoch
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        log_p = self._log_proba(seqs, val_windows)
        y_val = labels[val_windows[:, 0]]
        self.temperature = fit_temperature(log_p, y_val)
        grid, temps = [], []
        for n in sizes:
            m = val_windows[:, 2] == n
            if m.sum() >= MIN_ROWS_PER_SIZE:
                grid.append(np.log(n))
                temps.append(fit_temperature(log_p[m], y_val[m]))
        if grid:
            self.size_temperatures = (np.array(grid), np.array(temps))
        self.history.seconds = time.perf_counter() - t0
        return self

    # ------------------------------------------------------------------ prediction
    def predict_proba(
        self, seqs: HandSequences, windows: np.ndarray, *, calibration: str = "global"
    ) -> np.ndarray:
        """Probabilities for windows given as rows (player, start, n) into ``seqs``.
        ``calibration``: ``"global"`` (one temperature), ``"by_size"`` (one per training size,
        interpolated in log n, as ``ArchetypeClassifier``) or ``"none"``."""
        if calibration not in CALIBRATIONS:
            raise ValueError(f"calibration must be one of {CALIBRATIONS}")
        windows = np.asarray(windows, dtype=np.int64).reshape(-1, 3)
        log_p = self._log_proba(seqs, windows)
        t = np.ones(len(windows))
        if calibration == "by_size" and self.size_temperatures is not None:
            t = np.interp(np.log(windows[:, 2]), *self.size_temperatures)
        elif calibration != "none":
            t[:] = self.temperature
        return special.softmax(log_p / t[:, None], axis=1)

    def predict(self, seqs: HandSequences, windows: np.ndarray) -> np.ndarray:
        return self.predict_proba(seqs, windows).argmax(axis=1)

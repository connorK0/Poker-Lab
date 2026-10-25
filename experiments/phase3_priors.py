"""P3.7: population priors fit to real players, and checks that the synthetic-trained models
transfer to them.

    python -m experiments.phase3_priors --seed 2026

Pre-registered plan (written before any quantity of parts B-D was computed on the real hands).

Data
* Real: the 10,000 Pluribus hands of the PHH dataset at the pinned commit (all import strictly,
  P3.6): 6-max, blinds 50/100, 100 bb stacks reset every hand, 13 human professionals and
  Pluribus. A player's hands are ordered by session folder (number, then letter) and hand
  number. Every hand is redacted to public information (unshown hole cards removed) before any
  counter or decision row is computed, as in ``ProfileStore``. Pluribus is never used to fit or
  score anything; it appears as an extra row where stated. "Humans" are the 13 professionals.
* Synthetic: the five Phase 1 replicates (``phase1_master_seed``; the ``experiments/cache``
  files, re-simulated under the current code and identical bit for bit): per replicate 300
  training and 150 held-out bots, 2,000 hands each, 6-max, 100 bb, stacks reset every hand.
  Everything synthetic is fit to training bots only, as in Phase 1.
* Seen before this plan: P3.6 reported the humans' per-player and pooled VPIP, PFR, 3-bet,
  c-bet, fold to c-bet, WTSD and W$SD. Part A therefore has no predictions, and the predictions
  below that depend on how far the humans sit from the bots are not blind.
* Development, on simulated stand-ins only (``real_source: standin``: the quick config, and
  full-size runs on development seeds 98 and 99 with 13 bots in random sessions). Those runs
  set: the support check's alpha to 5% (at 1% nothing can be flagged at 2,000 hands, where 60
  held-out training bots give 60 windows and the smallest p-value is 1/61); C3's comparison
  with bots of the same reference label and its verdict on intervals; B4's rule on the mean
  over the ten stats (even with all 300 training bots the fitted c-bet prior is worse than
  uniform on held-out bots: the marginal likelihood weighs players by their opportunities, and
  the bots that c-bet least also get the fewest chances); the control's C1 and D2 bounds. The
  real hands were loaded only to check that the loader reproduces P3.6's per-player hand
  counts.

A. Priors fit to the humans. Beta-binomial priors (mean mu, concentration kappa = alpha + beta,
   in pseudo-opportunities) for the ten binomial stats by maximum marginal likelihood, and
   normal priors for the two size stats, with pseudo-counts from split halves (``hand_half``).
   95% intervals: bootstrap over the 13 players. Compared with the five synthetic training
   priors. Descriptive.
B. Do they estimate a new human's stats better? Leave one player out: for each human, priors
   fit to the other 12. Disjoint windows of n hands (n in ``estimate_sizes``; at most
   ``windows_max`` per player and n, spread evenly over the player's hands; a player needs at
   least 2n hands). Target: the player's rate over all their hands outside the window.
   Estimators: the posterior mean under the uniform Beta(1, 1) prior (the ``ProfileStore``
   default), under each synthetic prior (the mean error over the five), and under the
   leave-one-out human prior. Score: squared error against the target, averaged over windows
   within a player, then over players (each player weighs the same). The target's noise is
   independent of the window, so differences between estimators are unbiased for differences
   in mean squared error against the player's true rate. Intervals: bootstrap over players.
   B1. Prediction: the human prior beats uniform for every binomial stat at n = 25, 50, 100
       and 200 (95% interval of MSE(uniform) - MSE(human) above 0).
   B2. Prediction: the human prior beats the synthetic priors, same stats, sizes and criterion.
   B3. The reverse: the human prior (fit to all 13) on the synthetic held-out bots (windows
       from the first half of each bot's hands; target: the second half). Prediction: worse
       than uniform for at least 5 of the 10 binomial stats at n = 100 and at n = 500 (95%
       t-interval over the five replicates of MSE(human) - MSE(uniform) above 0).
   B4. How many players does a refit need? Priors fit to k players, k in ``min_players_grid``:
       for humans, ``subset_draws`` random subsets of the other 12 per held-out human (one
       draw at k = 12), scored as in B; for bots, ``subset_draws`` random subsets of k training
       bots per replicate (one draw when k is all of them), scored as in B3. Measured:
       MSE(uniform) - MSE(refit), averaged over draws. Decision rule for the default
       ``min_players`` of ``ProfileStore.refit_priors`` (now 30): for each population, the
       smallest k of its grid from which the refit is no worse than uniform (mean difference
       over the ten stats >= 0) at every n, at that k and every larger k of the grid. The new
       default is the larger of the two populations' values; if either has none, the default
       stays 30. Stats on which a refit is worse are reported.
C. Does the synthetic-trained classifier transfer? Phase 1's primary classifier (logistic
   regression on shrunk stats, one global temperature), trained as in Phase 1 for each
   replicate, applied to disjoint windows of the humans' hands at Phase 1's sample sizes
   (features shrunk with the replicate's synthetic prior, as in training).
   C1. Support check (``SupportCheck``, alpha = ``support_alpha`` = 5%, fit to the same
       training windows). Predictions: on the synthetic held-out bots at most 7% of windows are
       flagged at every n (pooled over replicates); on humans at least 80% of windows are
       flagged at n = 500, 1000 and 2000 (each player weighs the same; mean over replicates).
   C2. What the classifier calls the humans: the share of windows per predicted archetype and
       the mean top probability, by n. Descriptive.
   C3. Consistency, a label-free calibration check. If the classifier is calibrated where it
       is used, its top probability for an n-hand window should match how often that label
       agrees with the label it gives a later, disjoint block of ``consistency_reference``
       hands of the same player. Chunks of twice that many hands: windows of n hands (at most
       ``windows_max``) from the first half; the reference block is the second half. Gap =
       mean top probability - agreement rate, by n, for humans with at least one chunk and for
       the synthetic held-out bots (one chunk each). Among bots the gap depends on the label
       (near 0 for calling stations, about -0.1 for TAG bots at 1,000 hands), so each human
       is compared with bots that got the same reference label: a human's expected gap is the
       mean, over their chunks, of the bots' gap for that chunk's reference label. Pre-set
       verdict on the 95% interval of the mean over humans of gap - expected gap (bootstrap
       over players), at every n up to the reference: *transfers* if every interval lies
       within [-0.10, 0.10]; *fails* if some interval excludes 0; otherwise *inconclusive*.
       Prediction: fails.
D. The Phase 5 metric on human decisions: next-action log-loss (fold / check-call / bet-raise),
   a profile from a player's first c hands predicting their decisions in hands [c, c + window),
   c in ``action_checkpoints``. Models (Phase 1's learner): synthetic, trained on each
   replicate's training bots as in Phase 1 (situation only; situation + profile with the
   synthetic prior); human reference, leave one player out, trained on all the other 12 humans'
   decisions (the profile from the hands before each decision's ``window``-hand block; the
   leave-one-out human prior). Synthetic metrics are averaged over replicates. Means weigh
   players equally; intervals: bootstrap over players.
   D1. Prediction: the synthetic situation-only model has a higher log-loss on human decisions
       than the human situation-only model at every c (interval above 0).
   D2. Prediction: adding the profile reduces log-loss by less than 3% at every c >= 25, for
       the synthetic models and for the human models (point estimates; Phase 1's bots: 11-13%).
   D3. Prediction: the top-label calibration error (10 bins) of the synthetic situation-only
       model is higher than the human one at every c (interval above 0).
Control. Parts A-D (without B3 and B4) on a simulated stand-in: the Pluribus seating (which
   players sat in which session, for how many hands) replayed with each human replaced by a
   Phase 1 bot (archetypes in turn, parameters drawn as for training bots) and Pluribus by a
   TAG bot. The stand-in comes from the synthetic family with the real data's size and
   structure, so the transfer checks should pass on it; a check that fails on humans but passes
   here fails because of the humans, not the pipeline or the amount of data. Predictions: C1 at
   most 25% of stand-in windows flagged at every n (13 players: one flagged player moves the
   rate by about 8 points); C3 not *fails*; D1 the synthetic situation-only model is not worse
   (every interval's lower end at or below 0); D2 the profile reduces the synthetic models'
   log-loss by at least 3% at every c >= 25 (the line D2 predicts humans stay under).
"""

from __future__ import annotations

import argparse
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import (
    REPO_ROOT,
    RESULTS_ROOT,
    derive_seeds,
    load_config,
    run_metadata,
    save_json,
)
from experiments.datasets import PHH_DATASET_URL, content_sha256, phh_dataset
from experiments.phase1_data import SeedData, generate
from poker_lab.bots import ARCHETYPES, Archetype, ParametricBot, sample_params
from poker_lab.engine import SeatedPlayer, TableSimulator
from poker_lab.evaluation.intervals import t_interval
from poker_lab.evaluation.metrics import per_row_log_loss, reliability
from poker_lab.importers import import_phh, phh_files
from poker_lab.profiling import (
    SITUATION_FEATURES,
    ActionPredictor,
    ArchetypeClassifier,
    EmpiricalBayesShrinker,
    SupportCheck,
    decision_points,
    hand_counters,
    hand_half,
    prefix_totals,
    profile_features,
    window_totals,
)
from poker_lab.profiling.shrinkage import CONTINUOUS_STATS, fit_beta_binomial
from poker_lab.profiling.stats import BINOMIAL_STATS, N_COUNTERS, binomial_counts
from poker_lab.schema import HandHistory, TableConfig, redact_hole_cards

CONFIG = REPO_ROOT / "experiments" / "configs" / "phase3.yaml"
STATS = tuple(s.name for s in BINOMIAL_STATS)
N_STATS = len(STATS)
N_SIT = len(SITUATION_FEATURES)
LOG_KAPPA_BOUND = 12.0  # fit_beta_binomial's upper bound on log(kappa)
UNIFORM = np.ones((2, N_STATS))  # Beta(1, 1) for every stat: (alpha, beta) rows
ACTION_MODELS = ("syn_sit", "syn_prof", "hum_sit", "hum_prof", "syn_marginal", "hum_marginal")


# ---------------------------------------------------------------------------- players
def _padded_cumulative(counters: np.ndarray) -> np.ndarray:
    """(hands + 1, N_COUNTERS): row j sums the first j hands."""
    out = np.zeros((len(counters) + 1, counters.shape[1]), dtype=np.int64)
    np.cumsum(counters, axis=0, out=out[1:])
    return out


@dataclass
class Players:
    """Named players' hands in order. ``counters[i]`` is (hands, N_COUNTERS) for ``ids[i]``,
    ``halves[i]`` the ``hand_half`` of each of those hands; decision rows are (player index,
    seq, situation features..., label), ``seq`` being the hand's place in that player's own
    sequence."""

    ids: list[str]
    counters: list[np.ndarray]
    halves: list[np.ndarray]
    decisions: np.ndarray
    source: dict[str, Any]
    cum: list[np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.cum = [_padded_cumulative(c) for c in self.counters]

    @property
    def hands(self) -> np.ndarray:
        return np.array([len(c) for c in self.counters])

    def totals(self, idx) -> np.ndarray:
        return np.stack([self.cum[i][-1] for i in idx])

    def split_totals(self, idx) -> tuple[np.ndarray, np.ndarray]:
        first = np.stack([self.counters[i][self.halves[i] == 0].sum(axis=0) for i in idx])
        return first, self.totals(idx) - first


def hand_rows(hand: HandHistory) -> tuple[list, list, int]:
    """Counters and decision rows of one hand, from its public record only."""
    public = redact_hole_cards(hand)
    return hand_counters(public), decision_points(public), hand_half(hand.hand_id)


def assemble(
    ordered: list[tuple[Any, list, list, int]], source: dict[str, Any]
) -> tuple[Players, list[tuple[tuple[str, ...], int]]]:
    """Players from (block key, counters, decisions, half) rows in order, and the seating:
    (players in the seat order of the block's first hand, hands) for each run of hands with
    the same key and the same set of players."""
    counters: dict[str, list] = {}
    halves: dict[str, list] = {}
    decisions = []
    seating: list[list] = []
    last = None
    for key, rows, decs, half in ordered:
        seated = tuple(pid for pid, _ in rows)
        if (key, sorted(seated)) != last:
            seating.append([seated, 0])
            last = (key, sorted(seated))
        seating[-1][1] += 1
        seq = {}
        for pid, row in rows:
            counters.setdefault(pid, []).append(row)
            halves.setdefault(pid, []).append(half)
            seq[pid] = len(counters[pid]) - 1
        decisions += [(pid, seq[pid], feats, label) for pid, feats, label in decs]
    ids = sorted(counters)
    index = {pid: i for i, pid in enumerate(ids)}
    table = np.array(
        [(index[pid], s, *f, label) for pid, s, f, label in decisions], dtype=np.float32
    )
    players = Players(
        ids=ids,
        counters=[np.array(counters[p], dtype=np.int32) for p in ids],
        halves=[np.array(halves[p], dtype=np.int8) for p in ids],
        decisions=table,
        source=source,
    )
    return players, [(seated, n) for seated, n in seating]


_SESSION = re.compile(r"(\d+)([a-z]*)")


def session_key(file: str) -> tuple[int, str, int]:
    """``pluribus/100b/7.phh`` -> (100, "b", 7): session number, table letter, hand number."""
    *_, folder, name = Path(file).with_suffix("").parts
    m = _SESSION.fullmatch(folder)
    if m is None:
        raise ValueError(f"unexpected Pluribus folder {folder!r}")
    return int(m.group(1)), m.group(2), int(name)


def _import_folder(args: tuple[str, str]) -> tuple[list, list[dict]]:
    folder, root = args
    report = import_phh([folder], root=root, allow_rake=False)
    rows = [(session_key(h.metadata["import.file"]), *hand_rows(h)) for h in report.hands]
    return rows, [{"file": r.file, "reason": str(r.reason)} for r in report.rejected]


def load_pluribus(cfg: dict[str, Any], workers: int | None):
    root = phh_dataset(cfg["dataset_commit"])
    base = root / cfg["dataset_path"]
    folders = sorted(str(p) for p in base.iterdir() if p.is_dir())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_import_folder, [(f, str(root)) for f in folders]))
    rows = sorted((r for part, _ in parts for r in part), key=lambda r: r[0])
    rejected = [x for _, rej in parts for x in rej]
    files = phh_files([base], root=root)
    source = {
        "kind": "pluribus",
        "dataset": PHH_DATASET_URL,
        "commit": cfg["dataset_commit"],
        "path": cfg["dataset_path"],
        "files": len(files),
        "sha256": content_sha256(files),
        "hands": len(rows),
        "rejected": rejected,
    }
    return assemble([(key[:2], *rest) for key, *rest in rows], source)


def random_seating(cfg: dict[str, Any], rng) -> list[tuple[tuple[str, ...], int]]:
    """Sessions of five of ``standin.players`` players plus the bot (the quick config)."""
    s = cfg["standin"]
    pool = [f"h{i:02d}" for i in range(s["players"])]
    return [
        ((*rng.choice(pool, 5, replace=False).tolist(), cfg["bot_player"]), s["session_hands"])
        for _ in range(s["sessions"])
    ]


def _standin_block(args: tuple[int, int, list[tuple[str, str, int]], int, str]) -> list:
    block, seed, seated, n_hands, prefix = args
    players = [
        SeatedPlayer(
            seat, pid, ParametricBot(sample_params(Archetype(arch), np.random.default_rng(ps)))
        )
        for seat, (pid, arch, ps) in enumerate(seated)
    ]
    sim = TableSimulator(
        TableConfig(f"{prefix}{block}", 6, 50, 100),
        players,
        seed=seed,
        starting_stack=10_000,
        hand_id_prefix=f"{prefix}/{block}/",
    )
    return [(block, *hand_rows(h)) for h in sim.play(n_hands)]


def simulate_standin(
    seating: list[tuple[tuple[str, ...], int]], bot_id: str, seed: int, prefix: str, workers
):
    """A simulated stand-in with the given seating, blinds 50/100 and 100 bb stacks reset every
    hand: every other player is replaced by a Phase 1 bot (archetypes in turn, parameters drawn
    as for training bots) named ``s00``, ``s01``, ..., and ``bot_id`` by a TAG bot."""
    rng = np.random.default_rng(seed)
    others = sorted({p for seated, _ in seating for p in seated} - {bot_id})
    bots = {
        pid: (f"s{i:02d}", ARCHETYPES[i % len(ARCHETYPES)].value, int(rng.integers(2**31)))
        for i, pid in enumerate(others)
    }
    bots[bot_id] = (bot_id, Archetype.TAG.value, int(rng.integers(2**31)))
    tasks = [
        (b, int(rng.integers(2**31)), [bots[p] for p in seated], n, prefix)
        for b, (seated, n) in enumerate(seating)
    ]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        blocks = list(ex.map(_standin_block, tasks))
    source = {
        "kind": "standin",
        "prefix": prefix,
        "hands": sum(map(len, blocks)),
        "blocks": len(seating),
        "archetypes": {sid: arch for sid, arch, _ in bots.values()},
    }
    return assemble([r for part in blocks for r in part], source)


# ---------------------------------------------------------------------------- helpers
def _nanmean(x: np.ndarray, axis=None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


def _ab(shrinker: EmpiricalBayesShrinker) -> np.ndarray:
    """(2, N_STATS): alpha and beta of each binomial prior."""
    return np.array(
        [[shrinker.binomial[s].alpha for s in STATS], [shrinker.binomial[s].beta for s in STATS]]
    )


def fit_binomial(totals: np.ndarray) -> np.ndarray:
    k, m = binomial_counts(totals)
    fits = [fit_beta_binomial(k[:, j], m[:, j]) for j in range(N_STATS)]
    return np.array([[p.alpha for p in fits], [p.beta for p in fits]])


def _interval(boot: np.ndarray, value) -> dict[str, Any]:
    lo, hi = np.nanquantile(boot, [0.025, 0.975], axis=0)
    return {"value": value, "lower": lo, "upper": hi}


def _player_bootstrap(per_player: np.ndarray, rng, n_boot: int) -> dict[str, Any]:
    """Mean over players (rows; NaN rows dropped) with a 95% bootstrap interval."""
    x = per_player[~np.all(np.isnan(per_player.reshape(len(per_player), -1)), axis=1)]
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return _interval(_nanmean(x[idx], axis=1), _nanmean(x, axis=0))


def _starts(hands: int, n: int, cap: int, need: int) -> np.ndarray:
    """Starts of up to ``cap`` disjoint n-hand windows spread evenly over ``hands`` hands
    (none if there are fewer than ``need`` hands)."""
    if hands < need or hands < n:
        return np.empty(0, dtype=int)
    slots = hands // n
    k = min(cap, slots)
    return n * np.unique(np.linspace(0, slots - 1, k).round().astype(int))


def _sq_errors(est, k_out, m_out) -> tuple[np.ndarray, np.ndarray]:
    """Squared error against the held-out rate, and an unbiased estimate of that rate's binomial
    noise variance r(1 - r) / (m - 1); NaN where undefined."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = k_out / m_out
        err = np.where(m_out > 0, (est - r) ** 2, np.nan)
        noise = np.where(m_out > 1, r * (1 - r) / (m_out - 1), np.nan)
    return err, noise


def player_errors(
    players: Players, i: int, n: int, cap: int, priors: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Mean squared error per prior (P, N_STATS) of player ``i``'s n-hand windows against the
    rest of their hands, and the mean noise term (N_STATS,). ``priors``: (P, 2, N_STATS)."""
    starts = _starts(int(players.hands[i]), n, cap, need=2 * n)
    if not len(starts):
        return None
    cum = players.cum[i]
    win = cum[starts + n] - cum[starts]
    k, m = binomial_counts(win)
    ko, mo = binomial_counts(cum[-1] - win)
    a, b = priors[:, 0, None, :], priors[:, 1, None, :]
    err, noise = _sq_errors((a + k) / (a + b + m), ko, mo)
    return _nanmean(err, axis=1), _nanmean(noise, axis=0)


def bot_errors(cum: np.ndarray, n: int, cap: int, priors: np.ndarray):
    """As ``player_errors`` for simulated bots (``cum``: bots x hands x counters): windows from
    the first half of each bot's hands, target the second half; mean over bots."""
    hands = cum.shape[1]
    half = hands // 2
    padded = np.concatenate([np.zeros_like(cum[:, :1]), cum], axis=1)
    starts = n * np.arange(min(cap, half // n))
    if not len(starts):
        return None
    win = padded[:, starts + n] - padded[:, starts]  # bots x windows x counters
    k, m = binomial_counts(win)
    ko, mo = binomial_counts(cum[:, -1] - cum[:, half - 1])
    ko, mo = ko[:, None, :], mo[:, None, :]
    a, b = priors[:, 0, None, None, :], priors[:, 1, None, None, :]
    err, noise = _sq_errors((a + k) / (a + b + m), ko, mo)
    per_bot = _nanmean(err, axis=2)  # priors x bots x stats
    return _nanmean(per_bot, axis=1), _nanmean(_nanmean(noise, axis=1), axis=0)


def _syn_shrinker(data: SeedData) -> EmpiricalBayesShrinker:
    """The Phase 1 prior: training bots' full histories, sizes from their two halves."""
    cum = data.train.cumulative
    half = cum.shape[1] // 2
    halves = (cum[:, half - 1], cum[:, -1] - cum[:, half - 1])
    return EmpiricalBayesShrinker.fit(cum[:, -1], halves)


def _prior_table(shrinker: EmpiricalBayesShrinker) -> dict[str, Any]:
    out = {
        s: {"mean": p.mean, "concentration": p.concentration} for s, p in shrinker.binomial.items()
    }
    for name, p in shrinker.continuous.items():
        out[name] = {"mean": p.mean, "kappa": p.kappa}
    return out


# ---------------------------------------------------------------------------- part A
def part_a(players: Players, humans: list[int], bot: int | None, rng, cfg) -> dict[str, Any]:
    totals = players.totals(humans)
    first, second = players.split_totals(humans)
    fitted = EmpiricalBayesShrinker.fit(totals, (first, second))
    boots = []
    for _ in range(cfg["prior_bootstrap"]):
        idx = rng.integers(0, len(humans), len(humans))
        boots.append(EmpiricalBayesShrinker.fit(totals[idx], (first[idx], second[idx])))
    k, m = binomial_counts(totals.sum(axis=0))
    human = {}
    for j, s in enumerate(STATS):
        mu = np.array([b.binomial[s].mean for b in boots])
        kappa = np.array([b.binomial[s].concentration for b in boots])
        sd = np.sqrt(mu * (1 - mu) / (kappa + 1))
        p = fitted.binomial[s]
        human[s] = {
            "mean": p.mean,
            "mean_ci": list(np.quantile(mu, [0.025, 0.975])),
            "concentration": p.concentration,
            "concentration_ci": list(np.quantile(kappa, [0.025, 0.975])),
            "at_bound": float(np.mean(np.log(kappa) >= LOG_KAPPA_BOUND - 0.01)),
            "between_sd": float(np.sqrt(p.mean * (1 - p.mean) / (p.concentration + 1))),
            "between_sd_ci": list(np.quantile(sd, [0.025, 0.975])),
            "pooled": float(k[j] / m[j]),
        }
    for name, *_ in CONTINUOUS_STATS:
        q = fitted.continuous[name]
        kap = np.array([b.continuous[name].kappa for b in boots])
        human[name] = {
            "mean": q.mean,
            "kappa": q.kappa,
            "kappa_ci": list(np.quantile(kap, [0.025, 0.975])),
        }
    out: dict[str, Any] = {"human": human, "players": len(humans)}
    if bot is not None:
        kb, mb = binomial_counts(players.totals([bot])[0])
        out["bot"] = {s: float(kb[j] / mb[j]) if mb[j] else None for j, s in enumerate(STATS)}
    return out


# ---------------------------------------------------------------------------- part B
def human_estimation(
    players: Players, humans: list[int], lopo: dict[int, np.ndarray], syn: list[np.ndarray], cfg
) -> dict[int, dict[str, Any]]:
    """Per n: per-player MSE (players x priors x stats) for uniform, human LOPO, synthetic."""
    out = {}
    for n in cfg["estimate_sizes"]:
        rows, noise, who = [], [], []
        for i in humans:
            priors = np.stack([UNIFORM, lopo[i], *syn])
            res = player_errors(players, i, n, cfg["windows_max"], priors)
            if res is None:
                continue
            err, nz = res
            rows.append(np.stack([err[0], err[2:].mean(axis=0), err[1]]))  # uniform, syn, human
            noise.append(nz)
            who.append(players.ids[i])
        out[n] = {"mse": np.array(rows), "noise": np.array(noise), "players": who}
    return out


def summarize_estimation(est: dict[int, dict], rng, n_boot: int) -> dict[str, Any]:
    out = {}
    for n, e in est.items():
        mse = e["mse"]  # players x (uniform, synthetic, human) x stats
        if not len(mse):
            continue
        noise = _nanmean(e["noise"], axis=0)
        mean = _nanmean(mse, axis=0)
        out[str(n)] = {
            "players": e["players"],
            "mse": {p: mean[j] for j, p in enumerate(("uniform", "synthetic", "human"))},
            "rmse_vs_truth": {
                p: np.sqrt(np.clip(mean[j] - noise, 0, None))
                for j, p in enumerate(("uniform", "synthetic", "human"))
            },
            "noise": noise,
            "uniform_minus_human": _player_bootstrap(mse[:, 0] - mse[:, 2], rng, n_boot),
            "synthetic_minus_human": _player_bootstrap(mse[:, 1] - mse[:, 2], rng, n_boot),
        }
    return out


def reverse_estimation(data: SeedData, syn: np.ndarray, human: np.ndarray, cfg) -> dict:
    """Synthetic held-out bots: MSE per prior (uniform, synthetic, human) per n."""
    cum = data.test.cumulative
    priors = np.stack([UNIFORM, syn, human])
    out = {}
    for n in cfg["estimate_sizes"]:
        res = bot_errors(cum, n, cfg["windows_max"], priors)
        if res is not None:
            out[n] = {"mse": res[0], "noise": res[1]}
    return out


def summarize_reverse(per_rep: list[dict]) -> dict[str, Any]:
    out = {}
    for n in per_rep[0]:
        mse = np.stack([r[n]["mse"] for r in per_rep])  # reps x priors x stats
        diff = mse[:, 2] - mse[:, 0]
        ivs = [t_interval(diff[:, j]) for j in range(N_STATS)]
        noise = np.mean([r[n]["noise"] for r in per_rep], axis=0)
        out[str(n)] = {
            "mse": {
                p: mse[:, j].mean(axis=0) for j, p in enumerate(("uniform", "synthetic", "human"))
            },
            "rmse_vs_truth": {
                p: np.sqrt(np.clip(mse[:, j].mean(axis=0) - noise, 0, None))
                for j, p in enumerate(("uniform", "synthetic", "human"))
            },
            "human_minus_uniform": {
                "value": [iv.estimate for iv in ivs],
                "lower": [iv.lower for iv in ivs],
                "upper": [iv.upper for iv in ivs],
            },
        }
    return out


def human_min_players(players: Players, humans: list[int], grid, rng, cfg) -> dict[int, dict]:
    """k -> n -> mean over players and draws of MSE(uniform) - MSE(refit on k others)."""
    ks = [k for k in grid if k <= len(humans) - 1]
    acc = {k: {n: [] for n in cfg["estimate_sizes"]} for k in ks}
    for i in humans:
        others = [h for h in humans if h != i]
        totals = players.totals(others)
        for k in ks:
            draws = 1 if k == len(others) else cfg["subset_draws"]
            priors = [
                fit_binomial(totals[rng.choice(len(others), k, replace=False)])
                for _ in range(draws)
            ]
            stack = np.stack([UNIFORM, *priors])
            for n in cfg["estimate_sizes"]:
                res = player_errors(players, i, n, cfg["windows_max"], stack)
                if res is not None:
                    acc[k][n].append(res[0][0] - res[0][1:].mean(axis=0))
    return {
        k: {n: _nanmean(np.array(v), axis=0) for n, v in d.items() if v} for k, d in acc.items()
    }


def bot_min_players(data: SeedData, grid, rng, cfg) -> dict[int, dict]:
    totals = data.train.cumulative[:, -1]
    cum = data.test.cumulative
    out = {}
    for k in grid:
        if k > len(totals):
            continue
        draws = 1 if k == len(totals) else cfg["subset_draws"]
        priors = [
            fit_binomial(totals[rng.choice(len(totals), k, replace=False)]) for _ in range(draws)
        ]
        stack = np.stack([UNIFORM, *priors])
        out[k] = {}
        for n in cfg["estimate_sizes"]:
            res = bot_errors(cum, n, cfg["windows_max"], stack)
            if res is not None:
                out[k][n] = res[0][0] - res[0][1:].mean(axis=0)
    return out


def smallest_k(diffs: dict[int, dict[int, np.ndarray]]) -> int | None:
    """Smallest k from which every larger k of the grid is no worse than uniform at every n,
    on average over the binomial stats."""
    ks = sorted(diffs)
    ok = [all(np.nanmean(d) >= 0 for d in diffs[k].values()) for k in ks]
    best = None
    for k, good in zip(reversed(ks), reversed(ok), strict=True):
        if not good:
            break
        best = k
    return best


# ---------------------------------------------------------------------------- part C
def _player_windows(players: Players, i: int, n: int, cap: int) -> np.ndarray:
    starts = _starts(int(players.hands[i]), n, cap, need=n)
    cum = players.cum[i]
    return cum[starts + n] - cum[starts]


@dataclass
class SyntheticModels:
    """One replicate's Phase 1 models, fit to its training bots only: the primary classifier,
    the support check and the two action models, with their results on the held-out bots."""

    shrinker: EmpiricalBayesShrinker
    sizes: list[int]
    clf: ArchetypeClassifier
    support: SupportCheck
    bots: dict[int, dict[str, Any]]
    bots_consistency: dict[int, dict[str, np.ndarray]]
    sit_model: ActionPredictor
    prof_model: ActionPredictor
    freq: np.ndarray


def train_synthetic(data: SeedData, shrinker, p1, cfg, rng) -> SyntheticModels:
    tr, te = data.train, data.test
    tr_cum, te_cum = tr.cumulative, te.cumulative
    hands = tr.counters.shape[1]
    sizes = [n for n in p1["sample_sizes"] if n <= hands]
    X, y, g = [], [], []
    for n in sizes:
        w = window_totals(tr_cum, n, p1["train_windows_max"])
        X.append(profile_features(shrinker, w.totals))
        y.append(tr.labels[w.player])
        g.append(w.player)
    X, y, g = np.concatenate(X), np.concatenate(y), np.concatenate(g)
    clf = ArchetypeClassifier("lr", calibrate=True, random_state=data.seed).fit(X, y, g)
    support = SupportCheck(cfg["support_alpha"], random_state=data.seed).fit(X, y, g)
    bots = {}
    for n in sizes:
        w = window_totals(te_cum, n, p1["test_windows_max"])
        Xb = profile_features(shrinker, w.totals)
        bots[n] = {
            "flagged": support.flags(Xb).astype(float),
            "accuracy": float((clf.predict(Xb) == te.labels[w.player]).mean()),
            "top_prob": float(clf.predict_proba(Xb).max(axis=1).mean()),
        }
    ref = cfg["consistency_reference"]
    consistency = {}
    if hands >= 2 * ref:  # one chunk per held-out bot
        padded = np.concatenate([np.zeros_like(te_cum[:, :1]), te_cum], axis=1)
        ref_label = clf.predict(profile_features(shrinker, padded[:, 2 * ref] - padded[:, ref]))
        for n in (n for n in sizes if n <= ref):
            starts = n * np.arange(min(cfg["windows_max"], ref // n))
            win = padded[:, starts + n] - padded[:, starts]  # bots x windows x counters
            proba = clf.predict_proba(profile_features(shrinker, win.reshape(-1, N_COUNTERS)))
            conf = proba.max(axis=1).reshape(len(te_cum), -1).mean(axis=1)
            agree = (proba.argmax(axis=1).reshape(len(te_cum), -1) == ref_label[:, None]).mean(1)
            consistency[n] = {
                "confidence": conf,
                "agreement": agree,
                "labels": te.labels,
                "reference": ref_label,
            }

    # action models exactly as in Phase 1: decisions in the checkpoint windows, profile from
    # the hands before each window
    cps, window = p1["action_checkpoints"], p1["action_window"]
    rows, which = _windowed(tr.decisions[:, 1], cps, window)
    d = tr.decisions[rows]
    player = d[:, 0].astype(int)
    sit = d[:, 2 : 2 + N_SIT]
    labels = d[:, -1].astype(int)
    profiles = np.stack([profile_features(shrinker, prefix_totals(tr_cum, c)) for c in cps])
    prof = np.hstack([sit, profiles[which, player]])
    sub = rng.choice(len(labels), size=min(cfg["action_train_rows"], len(labels)), replace=False)
    return SyntheticModels(
        shrinker=shrinker,
        sizes=sizes,
        clf=clf,
        support=support,
        bots=bots,
        bots_consistency=consistency,
        sit_model=ActionPredictor(random_state=data.seed).fit(sit[sub], labels[sub]),
        prof_model=ActionPredictor(random_state=data.seed).fit(prof[sub], labels[sub]),
        freq=np.bincount(labels, minlength=3) / len(labels),
    )


def apply_classifier(m: SyntheticModels, pop: Population, cfg) -> dict[str, Any]:
    """The replicate's classifier and support check on a population's players."""
    players, shrinker, clf = pop.players, m.shrinker, m.clf
    out: dict[str, Any] = {"sizes": m.sizes, "bots": m.bots, "players": {}, "consistency": {}}
    for i in [*pop.humans, *([pop.bot] if pop.bot is not None else [])]:
        rows = {}
        for n in m.sizes:
            win = _player_windows(players, i, n, cfg["windows_max"])
            if not len(win):
                continue
            Xh = profile_features(shrinker, win)
            proba = clf.predict_proba(Xh)
            rows[n] = {
                "flagged": float(m.support.flags(Xh).mean()),
                "pvalue_median": float(np.median(m.support.pvalue(Xh))),
                "labels": np.bincount(proba.argmax(axis=1), minlength=len(ARCHETYPES)) / len(proba),
                "top_prob": float(proba.max(axis=1).mean()),
                "windows": len(win),
            }
        out["players"][players.ids[i]] = rows
    ref = cfg["consistency_reference"]
    for n, bots in m.bots_consistency.items():
        humans = {}
        for i in pop.humans:
            cum = players.cum[i]
            chunks = []  # (mean top probability, agreement, reference label) per chunk
            for c in range(int(players.hands[i]) // (2 * ref)):
                base = 2 * ref * c
                block = (cum[base + 2 * ref] - cum[base + ref])[None]
                label = int(clf.predict(profile_features(shrinker, block))[0])
                starts = base + n * np.arange(min(cfg["windows_max"], ref // n))
                proba = clf.predict_proba(profile_features(shrinker, cum[starts + n] - cum[starts]))
                agree = float((proba.argmax(axis=1) == label).mean())
                chunks.append((float(proba.max(axis=1).mean()), agree, label))
            if chunks:
                humans[players.ids[i]] = chunks
        out["consistency"][n] = {"bots": bots, "humans": humans}
    return out


def score_synthetic_actions(m: SyntheticModels, test: HumanRows) -> dict[str, np.ndarray]:
    prof = np.hstack([test.situation, profile_features(m.shrinker, test.profile_totals)])
    return {
        "syn_sit": m.sit_model.predict_proba(test.situation),
        "syn_prof": m.prof_model.predict_proba(prof),
        "syn_marginal": np.tile(m.freq, (len(test.label), 1)),
    }


def summarize_classifier(per_rep: list[dict], humans_ids: list[str], bot_id, rng, n_boot):
    sizes = per_rep[0]["sizes"]
    out: dict[str, Any] = {"sizes": sizes, "support": {}, "labels": {}, "consistency": {}}
    for n in sizes:
        bots_flag = np.concatenate([r["bots"][n]["flagged"] for r in per_rep])
        human_rows = {
            pid: [r["players"][pid][n] for r in per_rep if n in r["players"].get(pid, {})]
            for pid in humans_ids
        }
        human_rows = {p: v for p, v in human_rows.items() if v}
        by_player = {p: float(np.mean([x["flagged"] for x in v])) for p, v in human_rows.items()}
        entry = {
            "bots_flagged": float(bots_flag.mean()),
            "bots_windows": len(bots_flag),
            "bots_accuracy": float(np.mean([r["bots"][n]["accuracy"] for r in per_rep])),
            "bots_top_prob": float(np.mean([r["bots"][n]["top_prob"] for r in per_rep])),
            "humans_flagged": float(np.mean(list(by_player.values()))) if by_player else None,
            "humans_by_player": by_player,
            "human_players": len(by_player),
        }
        labels = {
            p: np.mean([x["labels"] for x in v], axis=0).tolist() for p, v in human_rows.items()
        }
        top = {p: float(np.mean([x["top_prob"] for x in v])) for p, v in human_rows.items()}
        lab = {
            "humans": np.mean(list(labels.values()), axis=0).tolist() if labels else None,
            "humans_top_prob": float(np.mean(list(top.values()))) if top else None,
            "by_player": labels,
        }
        if bot_id is not None:
            b = [r["players"][bot_id][n] for r in per_rep if n in r["players"].get(bot_id, {})]
            if b:
                entry["bot_flagged"] = float(np.mean([x["flagged"] for x in b]))
                lab["bot"] = np.mean([x["labels"] for x in b], axis=0).tolist()
                lab["bot_top_prob"] = float(np.mean([x["top_prob"] for x in b]))
        out["support"][str(n)] = entry
        out["labels"][str(n)] = lab
    for n in per_rep[0]["consistency"]:
        reps = [r["consistency"][n] for r in per_rep]
        if "bots" not in reps[0] or not reps[0].get("humans"):
            continue
        pids = sorted(reps[0]["humans"])
        conf, agree, expected, raw = [], [], [], []
        for r in reps:
            b = r["bots"]
            bot_gap = b["confidence"] - b["agreement"]
            # bots whose own reference block got label k: the gap expected of a player labeled k
            by_ref = {
                k: float(bot_gap[b["reference"] == k].mean())
                if (b["reference"] == k).any()
                else 0.0
                for k in range(len(ARCHETYPES))
            }
            raw.append(float(bot_gap.mean()))
            chunks = [r["humans"][p] for p in pids]
            conf.append([np.mean([c[0] for c in ch]) for ch in chunks])
            agree.append([np.mean([c[1] for c in ch]) for ch in chunks])
            expected.append([np.mean([by_ref[c[2]] for c in ch]) for ch in chunks])
        conf, agree, expected = (np.mean(x, axis=0) for x in (conf, agree, expected))
        gaps = conf - agree
        diff = _player_bootstrap((gaps - expected)[:, None], rng, n_boot)
        by_arch = {
            a.value: float(
                np.mean(
                    [
                        np.mean(
                            (r["bots"]["confidence"] - r["bots"]["agreement"])[
                                r["bots"]["labels"] == k
                            ]
                        )
                        for r in reps
                    ]
                )
            )
            for k, a in enumerate(ARCHETYPES)
        }
        ref_labels = np.bincount(
            [c[2] for r in reps for p in pids for c in r["humans"][p]], minlength=len(ARCHETYPES)
        )
        out["consistency"][str(n)] = {
            "bots": {
                "confidence": float(np.mean([r["bots"]["confidence"].mean() for r in reps])),
                "agreement": float(np.mean([r["bots"]["agreement"].mean() for r in reps])),
                "gap": float(np.mean(raw)),
                "gap_by_archetype": by_arch,
            },
            "humans": {
                "confidence": float(conf.mean()),
                "agreement": float(agree.mean()),
                "gap": float(gaps.mean()),
                "expected_gap": float(expected.mean()),
                "by_player": {p: float(x) for p, x in zip(pids, gaps - expected, strict=True)},
                "reference_labels": (ref_labels / ref_labels.sum()).tolist(),
            },
            "difference": {k: float(np.ravel(v)[0]) for k, v in diff.items()},
            "players": pids,
        }
    return out


# ---------------------------------------------------------------------------- part D
def _windowed(seq: np.ndarray, checkpoints, window) -> tuple[np.ndarray, np.ndarray]:
    rows, which = [], []
    for i, c in enumerate(checkpoints):
        idx = np.nonzero((seq >= c) & (seq < c + window))[0]
        rows.append(idx)
        which.append(np.full(len(idx), i))
    return np.concatenate(rows), np.concatenate(which)


@dataclass
class HumanRows:
    """Test decisions of the humans in the checkpoint windows."""

    player: np.ndarray  # index into Players
    situation: np.ndarray
    label: np.ndarray
    which: np.ndarray  # checkpoint index
    profile_totals: np.ndarray  # counters of the player's first c hands


def human_test_rows(players: Players, humans: list[int], checkpoints, window) -> HumanRows:
    d = players.decisions[np.isin(players.decisions[:, 0], humans)]
    rows, which = _windowed(d[:, 1], checkpoints, window)
    d = d[rows]
    player = d[:, 0].astype(int)
    cps = np.asarray(checkpoints)[which]
    totals = np.stack([players.cum[p][c] for p, c in zip(player, cps, strict=True)])
    return HumanRows(player, d[:, 2 : 2 + N_SIT], d[:, -1].astype(int), which, totals)


@dataclass
class Population:
    """Players analysed as "humans" (the real ones, or a simulated control): their
    leave-one-out priors, the prior fit to all of them, and their action-prediction test rows."""

    name: str
    players: Players
    seating: list[tuple[tuple[str, ...], int]]
    humans: list[int]
    bot: int | None
    lopo_shrinkers: dict[int, EmpiricalBayesShrinker]
    lopo: dict[int, np.ndarray]
    prior: np.ndarray
    test: HumanRows

    @property
    def human_ids(self) -> list[str]:
        return [self.players.ids[i] for i in self.humans]

    @property
    def bot_id(self) -> str | None:
        return None if self.bot is None else self.players.ids[self.bot]


def make_population(name: str, players: Players, seating, cfg) -> Population:
    bot = players.ids.index(cfg["bot_player"]) if cfg["bot_player"] in players.ids else None
    humans = [i for i in range(len(players.ids)) if i != bot]
    totals = players.totals(humans)
    first, second = players.split_totals(humans)
    lopo = {}
    for j, i in enumerate(humans):
        keep = [x for x in range(len(humans)) if x != j]
        lopo[i] = EmpiricalBayesShrinker.fit(totals[keep], (first[keep], second[keep]))
    return Population(
        name=name,
        players=players,
        seating=seating,
        humans=humans,
        bot=bot,
        lopo_shrinkers=lopo,
        lopo={i: _ab(sh) for i, sh in lopo.items()},
        prior=fit_binomial(totals),
        test=human_test_rows(players, humans, cfg["action_checkpoints"], cfg["action_window"]),
    )


def human_action(players: Players, humans, lopo, test: HumanRows, cfg, seed: int, rng) -> dict:
    """Leave-one-player-out action models trained on the other humans' decisions."""
    window = cfg["action_window"]
    d = players.decisions[np.isin(players.decisions[:, 0], humans)]
    out = {k: np.zeros((len(test.label), 3)) for k in ("hum_sit", "hum_prof", "hum_marginal")}
    for i in humans:
        train = d[d[:, 0] != i]
        player = train[:, 0].astype(int)
        block = (train[:, 1].astype(int) // window) * window
        totals = np.stack([players.cum[p][c] for p, c in zip(player, block, strict=True)])
        sit = train[:, 2 : 2 + N_SIT]
        y = train[:, -1].astype(int)
        prof = np.hstack([sit, profile_features(lopo[i], totals)])
        sub = rng.choice(len(y), size=min(cfg["action_train_rows"], len(y)), replace=False)
        mask = test.player == i
        if not mask.any():
            continue
        sit_model = ActionPredictor(random_state=seed).fit(sit[sub], y[sub])
        prof_model = ActionPredictor(random_state=seed).fit(prof[sub], y[sub])
        test_prof = np.hstack(
            [test.situation[mask], profile_features(lopo[i], test.profile_totals[mask])]
        )
        out["hum_sit"][mask] = sit_model.predict_proba(test.situation[mask])
        out["hum_prof"][mask] = prof_model.predict_proba(test_prof)
        out["hum_marginal"][mask] = np.bincount(y, minlength=3) / len(y)
    return out


def summarize_action(test: HumanRows, syn: list[dict], hum: dict, players, cfg, rng) -> dict:
    n_boot = cfg["bootstrap"]
    out: dict[str, Any] = {
        "checkpoints": cfg["action_checkpoints"],
        "window": cfg["action_window"],
        "by_checkpoint": {},
    }
    ll = {k: per_row_log_loss(test.label, hum[k]) for k in ("hum_sit", "hum_prof", "hum_marginal")}
    for k in ("syn_sit", "syn_prof", "syn_marginal"):
        ll[k] = np.mean([per_row_log_loss(test.label, r[k]) for r in syn], axis=0)
    for ci, c in enumerate(cfg["action_checkpoints"]):
        mask = test.which == ci
        if not mask.any():
            continue
        pids = np.unique(test.player[mask])

        def per_player(values, mask=mask, pids=pids):
            return np.array([values[mask & (test.player == p)].mean() for p in pids])

        means = {k: per_player(v) for k, v in ll.items()}
        y = test.label[mask]
        ece = {k: reliability(y, hum[k][mask]).ece for k in ("hum_sit", "hum_prof")}
        for k in ("syn_sit", "syn_prof"):
            ece[k] = float(np.mean([reliability(y, r[k][mask]).ece for r in syn]))
        # bootstrap over players: log-loss differences and the ECE difference
        idx = rng.integers(0, len(pids), size=(n_boot, len(pids)))
        d1 = (means["syn_sit"] - means["hum_sit"])[idx].mean(axis=1)
        rows_of = [np.nonzero(mask & (test.player == p))[0] for p in pids]
        d3 = np.empty(n_boot)
        for b in range(n_boot):
            r = np.concatenate([rows_of[j] for j in idx[b]])
            syn_e = np.mean([reliability(test.label[r], s["syn_sit"][r]).ece for s in syn])
            d3[b] = syn_e - reliability(test.label[r], hum["hum_sit"][r]).ece
        log_loss = {k: float(v.mean()) for k, v in means.items()}
        accuracy = {k: float((hum[k][mask].argmax(axis=1) == y).mean()) for k in hum}
        for k in ("syn_sit", "syn_prof", "syn_marginal"):
            accuracy[k] = float(np.mean([(s[k][mask].argmax(axis=1) == y).mean() for s in syn]))
        out["by_checkpoint"][str(c)] = {
            "decisions": int(mask.sum()),
            "players": len(pids),
            "log_loss": log_loss,
            "ece": ece,
            "accuracy": accuracy,
            "reduction": {
                "synthetic": 1 - log_loss["syn_prof"] / log_loss["syn_sit"],
                "human": 1 - log_loss["hum_prof"] / log_loss["hum_sit"],
            },
            "syn_minus_hum_log_loss": _interval(d1, log_loss["syn_sit"] - log_loss["hum_sit"]),
            "syn_minus_hum_ece": _interval(d3, ece["syn_sit"] - ece["hum_sit"]),
        }
    return out


# ---------------------------------------------------------------------------- predictions
def consistency_verdict(cons: dict[str, Any], margin: float = 0.10) -> dict[str, Any]:
    """C3's pre-set verdict on the intervals of gap(players) - gap(bots): "transfers" if every
    interval lies within +-margin, "fails" if some interval excludes 0, else "inconclusive"."""
    diffs = {n: v["difference"] for n, v in cons.items()}
    if not diffs:
        return {"differences": {}, "verdict": "no data"}
    if any(d["lower"] > 0 or d["upper"] < 0 for d in diffs.values()):
        verdict = "fails"
    elif all(-margin <= d["lower"] and d["upper"] <= margin for d in diffs.values()):
        verdict = "transfers"
    else:
        verdict = "inconclusive"
    return {"differences": diffs, "verdict": verdict}


def predictions(pop: dict[str, Any], reverse: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered predictions B1-B3, C1, C3 and D1-D3 for the humans."""
    est = pop["estimation"]
    small = [str(n) for n in (25, 50, 100, 200) if str(n) in est]

    def above_zero(key):
        fails = [
            f"{STATS[j]} at {n}"
            for n in small
            for j in range(N_STATS)
            if not est[n][key]["lower"][j] > 0
        ]
        return {"holds": bool(small) and not fails, "failures": fails}

    b3 = {
        n: [STATS[j] for j in range(N_STATS) if reverse[n]["human_minus_uniform"]["lower"][j] > 0]
        for n in ("100", "500")
        if n in reverse
    }
    sup = pop["classifier"]["support"]
    c1a = {n: v["bots_flagged"] for n, v in sup.items()}
    c1b = {n: sup[n]["humans_flagged"] for n in ("500", "1000", "2000") if n in sup}
    c3 = consistency_verdict(pop["classifier"]["consistency"])
    act = pop["action"]["by_checkpoint"]
    d1 = {c: v["syn_minus_hum_log_loss"]["lower"] for c, v in act.items()}
    d2 = {c: v["reduction"] for c, v in act.items() if int(c) >= 25}
    d3 = {c: v["syn_minus_hum_ece"]["lower"] for c, v in act.items()}
    return {
        "B1": above_zero("uniform_minus_human"),
        "B2": above_zero("synthetic_minus_human"),
        "B3": {
            "worse_than_uniform": b3,
            "holds": bool(b3) and all(len(v) >= 5 for v in b3.values()),
        },
        "C1a": {"flag_rates": c1a, "holds": all(v <= 0.07 for v in c1a.values())},
        "C1b": {
            "flag_rates": c1b,
            "holds": bool(c1b) and all(v is not None and v >= 0.8 for v in c1b.values()),
        },
        "C3": {**c3, "holds": c3["verdict"] == "fails"},
        "D1": {"lower_bounds": d1, "holds": bool(d1) and all(v > 0 for v in d1.values())},
        "D2": {
            "reductions": d2,
            "holds": bool(d2)
            and all(v["synthetic"] < 0.03 and v["human"] < 0.03 for v in d2.values()),
        },
        "D3": {"lower_bounds": d3, "holds": bool(d3) and all(v > 0 for v in d3.values())},
    }


def control_predictions(pop: dict[str, Any]) -> dict[str, Any]:
    """The control's predictions: the transfer checks pass on a stand-in from the bot family."""
    sup = pop["classifier"]["support"]
    c1 = {n: v["humans_flagged"] for n, v in sup.items() if v["humans_flagged"] is not None}
    c3 = consistency_verdict(pop["classifier"]["consistency"])
    act = pop["action"]["by_checkpoint"]
    d1 = {c: v["syn_minus_hum_log_loss"]["lower"] for c, v in act.items()}
    d2 = {c: v["reduction"]["synthetic"] for c, v in act.items() if int(c) >= 25}
    return {
        "C1": {"flag_rates": c1, "holds": bool(c1) and all(v <= 0.25 for v in c1.values())},
        "C3": {**c3, "holds": c3["verdict"] not in ("fails", "no data")},
        "D1": {"lower_bounds": d1, "holds": bool(d1) and all(v <= 0 for v in d1.values())},
        "D2": {"reductions": d2, "holds": bool(d2) and all(v >= 0.03 for v in d2.values())},
    }


def data_summary(pop: Population) -> dict[str, Any]:
    players = pop.players
    return {
        **players.source,
        "players": {pid: int(h) for pid, h in zip(players.ids, players.hands, strict=True)},
        "humans": pop.human_ids,
        "bot": pop.bot_id,
        "decisions": len(players.decisions),
        "test_decisions": len(pop.test.label),
        "seating_blocks": len(pop.seating),
    }


# ---------------------------------------------------------------------------- run
def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase3" / "priors"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()
    ss = [int(x) for x in np.random.SeedSequence([seed, 37]).generate_state(8)]
    rng = {k: np.random.default_rng(x) for k, x in zip("ABCDKS", ss, strict=False)}
    if cfg["real_source"] == "pluribus":
        players, seating = load_pluribus(cfg, workers)
    else:
        plan = random_seating(cfg, rng["S"])
        players, seating = simulate_standin(plan, cfg["bot_player"], ss[6], "standin", workers)
    pops = {"humans": make_population("humans", players, seating, cfg)}
    if cfg.get("control", True):
        sim, sim_seating = simulate_standin(seating, cfg["bot_player"], ss[7], "control", workers)
        pops["control"] = make_population("control", sim, sim_seating, cfg)
    for name, pop in pops.items():
        hands = int(pop.players.hands[pop.humans].sum())
        print(f"{name}: {len(pop.humans)} players, {hands} player-hands", flush=True)

    p1 = load_config(REPO_ROOT / cfg["phase1_config"])
    p1_seeds = derive_seeds(cfg["phase1_master_seed"], p1["n_seeds"])
    r: dict[str, Any] = {
        "meta": meta,
        "phase1": {"config": cfg["phase1_config"], "seeds": p1_seeds},
        "populations": {
            name: {
                "data": data_summary(pop),
                "priors": part_a(pop.players, pop.humans, pop.bot, rng["A"], cfg),
            }
            for name, pop in pops.items()
        },
    }
    syn_priors, syn_tables, reverse, bot_k = [], [], [], []
    clf_reps: dict[str, list] = {name: [] for name in pops}
    syn_action: dict[str, list] = {name: [] for name in pops}
    for rep, rep_seed in enumerate(p1_seeds):
        t0 = time.perf_counter()
        data = generate(rep_seed, p1, workers=workers)
        shrinker = _syn_shrinker(data)
        syn_tables.append(_prior_table(shrinker))
        syn_priors.append(_ab(shrinker))
        reverse.append(reverse_estimation(data, syn_priors[-1], pops["humans"].prior, cfg))
        bot_k.append(bot_min_players(data, cfg["min_players_grid"], rng["K"], cfg))
        models = train_synthetic(data, shrinker, p1, cfg, rng["D"])
        for name, pop in pops.items():
            clf_reps[name].append(apply_classifier(models, pop, cfg))
            syn_action[name].append(score_synthetic_actions(models, pop.test))
        print(f"replicate {rep + 1}/{len(p1_seeds)}: {time.perf_counter() - t0:.0f}s", flush=True)
    r["synthetic_priors"] = {
        key: {f: [t[key][f] for t in syn_tables] for f in syn_tables[0][key]}
        for key in syn_tables[0]
    }
    r["reverse"] = summarize_reverse(reverse)
    for name, pop in pops.items():
        print(f"{name}: estimation, classifier, actions", flush=True)
        block = r["populations"][name]
        est = human_estimation(pop.players, pop.humans, pop.lopo, syn_priors, cfg)
        block["estimation"] = summarize_estimation(est, rng["B"], cfg["bootstrap"])
        block["classifier"] = summarize_classifier(
            clf_reps[name], pop.human_ids, pop.bot_id, rng["C"], cfg["bootstrap"]
        )
        hum = human_action(
            pop.players, pop.humans, pop.lopo_shrinkers, pop.test, cfg, seed, rng["D"]
        )
        block["action"] = summarize_action(
            pop.test, syn_action[name], hum, pop.players, cfg, rng["D"]
        )

    humans = pops["humans"]
    hk = human_min_players(humans.players, humans.humans, cfg["min_players_grid"], rng["K"], cfg)
    syn_k = {
        k: {n: np.mean([b[k][n] for b in bot_k], axis=0) for n in bot_k[0][k]} for k in bot_k[0]
    }
    k_h, k_s = smallest_k(hk), smallest_k(syn_k)
    r["min_players"] = {
        "grid": cfg["min_players_grid"],
        "human": {str(k): {str(n): v for n, v in d.items()} for k, d in hk.items()},
        "synthetic": {str(k): {str(n): v for n, v in d.items()} for k, d in syn_k.items()},
        "k_human": k_h,
        "k_synthetic": k_s,
        "default": 30 if k_h is None or k_s is None else max(k_h, k_s),
    }
    r["predictions"] = predictions(r["populations"]["humans"], r["reverse"])
    if "control" in pops:
        r["control_predictions"] = control_predictions(r["populations"]["control"])
    r["config"] = {k: cfg[k] for k in ("estimate_sizes", "consistency_reference", "support_alpha")}
    r["seconds"] = time.perf_counter() - t_start
    save_json(r, out_dir / "priors.json")
    (out_dir / "priors_summary.md").write_text(markdown_summary(r), encoding="utf-8", newline="\n")
    make_figures(r, out_dir)
    return r


# ---------------------------------------------------------------------------- report
STAT_LABELS = {
    "vpip": "VPIP",
    "pfr": "PFR",
    "three_bet": "3-bet",
    "fold_to_3bet": "Fold to 3-bet",
    "cbet": "C-bet",
    "fold_to_cbet": "Fold to c-bet",
    "aggr_share": "Aggression share",
    "fold_to_bet": "Fold to a bet",
    "wtsd": "WTSD",
    "wsd": "W$SD",
}
PRIOR_LABELS = {"uniform": "Uniform Beta(1, 1)", "synthetic": "Synthetic", "human": "Human"}
MODEL_LABELS = {
    "syn_sit": "Synthetic, situation",
    "syn_prof": "Synthetic, + profile",
    "hum_sit": "Human, situation",
    "hum_prof": "Human, + profile",
}


def _verdict(holds: bool) -> str:
    return "**holds**" if holds else "**does not hold**"


def mean_rmse(block: dict[str, Any], prior: str) -> float:
    """Mean over the binomial stats of the noise-corrected RMSE (percentage points)."""
    return float(100 * np.nanmean(np.asarray(block["rmse_vs_truth"][prior], dtype=float)))


def _iv(d: dict[str, Any], j: int | None = None, scale: float = 1.0, fmt: str = "+.3f") -> str:
    def pick(x):
        return float(np.ravel(x)[j] if j is not None else np.ravel(x)[0]) * scale

    return f"{pick(d['value']):{fmt}} [{pick(d['lower']):{fmt}}, {pick(d['upper']):{fmt}}]"


def markdown_summary(r: dict[str, Any]) -> str:
    hum = r["populations"]["humans"]
    ctl = r["populations"].get("control")
    pred = r["predictions"]
    data = hum["data"]
    n_hands = sum(v for k, v in data["players"].items() if k in data["humans"])
    lines = [
        "# P3.7: priors fit to real players, and do the synthetic-trained models transfer?",
        "",
        f"{len(data['humans'])} players ({data['kind']}), {n_hands:,} player-hands, "
        f"{data['test_decisions']:,} test decisions; synthetic: {len(r['phase1']['seeds'])} "
        "Phase 1 replicates. Brackets: 95% bootstrap intervals over players unless stated.",
        "",
        "## Pre-registered predictions",
        "",
        f"- B1 human prior beats uniform (n <= 200, every stat): {_verdict(pred['B1']['holds'])}"
        + (f"; fails: {', '.join(pred['B1']['failures'])}" if pred["B1"]["failures"] else ""),
        f"- B2 human prior beats synthetic: {_verdict(pred['B2']['holds'])}"
        + (f"; fails: {', '.join(pred['B2']['failures'])}" if pred["B2"]["failures"] else ""),
        "- B3 human prior worse than uniform on bots for >= 5 of 10 stats at 100 and 500 hands: "
        + "; ".join(f"{n} hands {len(v)}" for n, v in pred["B3"]["worse_than_uniform"].items())
        + f"; {_verdict(pred['B3']['holds'])}",
        "- C1a support check flags at most 7% of held-out bot windows: max "
        f"{max(pred['C1a']['flag_rates'].values()):.1%}; {_verdict(pred['C1a']['holds'])}",
        "- C1b flags at least 80% of human windows at 500 / 1,000 / 2,000 hands: "
        + ", ".join("n/a" if v is None else f"{v:.0%}" for v in pred["C1b"]["flag_rates"].values())
        + f"; {_verdict(pred['C1b']['holds'])}",
        f"- C3 consistency verdict *{pred['C3']['verdict']}* (predicted: fails); "
        f"{_verdict(pred['C3']['holds'])}",
        "- D1 synthetic situation model worse than the human one at every checkpoint: "
        f"{_verdict(pred['D1']['holds'])}",
        "- D2 the profile reduces log-loss by less than 3%: "
        + ", ".join(
            f"{c} hands {v['synthetic']:.1%} / {v['human']:.1%}"
            for c, v in pred["D2"]["reductions"].items()
        )
        + f" (synthetic / human); {_verdict(pred['D2']['holds'])}",
        f"- D3 synthetic situation model worse calibrated: {_verdict(pred['D3']['holds'])}",
    ]
    if "control_predictions" in r:
        cp = r["control_predictions"]
        lines += [
            "",
            "Control (the same seating replayed by Phase 1 bots):",
            "",
            f"- C1 at most 25% flagged: max {max(cp['C1']['flag_rates'].values()):.1%}; "
            f"{_verdict(cp['C1']['holds'])}",
            f"- C3 not *fails*: verdict *{cp['C3']['verdict']}*; {_verdict(cp['C3']['holds'])}",
            f"- D1 synthetic model not worse: {_verdict(cp['D1']['holds'])}",
            "- D2 profile reduction at least 3%: "
            + ", ".join(f"{c} hands {v:.1%}" for c, v in cp["D2"]["reductions"].items())
            + f"; {_verdict(cp['D2']['holds'])}",
        ]
    mp = r["min_players"]
    lines += [
        "",
        f"B4 decision: `refit_priors` default `min_players` = **{mp['default']}** "
        f"(humans: {mp['k_human']}, bots: {mp['k_synthetic']}; was 30).",
        "",
        "## A. Priors",
        "",
        "| Stat | Humans: mean | Humans: concentration | Synthetic: mean | "
        "Synthetic: concentration | Pluribus |",
        "|---|---|---|---|---|---|",
    ]
    pri, syn = hum["priors"], r["synthetic_priors"]
    for s in STATS:
        h = pri["human"][s]
        sm, sk = np.asarray(syn[s]["mean"]), np.asarray(syn[s]["concentration"])
        bot = pri.get("bot", {}).get(s)
        lines.append(
            f"| {STAT_LABELS[s]} | {h['mean']:.1%} [{h['mean_ci'][0]:.1%}, {h['mean_ci'][1]:.1%}] "
            f"| {h['concentration']:,.0f} [{h['concentration_ci'][0]:,.0f}, "
            f"{h['concentration_ci'][1]:,.0f}] | {sm.mean():.1%} | {sk.mean():,.1f} | "
            + ("-" if bot is None else f"{bot:.1%}")
            + " |"
        )
    lines += [
        "",
        "Concentration = alpha + beta (pseudo-opportunities). Synthetic: mean over replicates.",
        "",
        "## B. Estimating a new player's stats (leave one player out)",
        "",
        "RMSE against the true rate (noise-corrected, percentage points, mean over the ten "
        "stats); stats where the human prior is better (95% interval above 0):",
        "",
        "| Hands | Players | Uniform | Synthetic | Human | Beats uniform | Beats synthetic |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, b in hum["estimation"].items():
        better_u = int(np.sum(np.asarray(b["uniform_minus_human"]["lower"], float) > 0))
        better_s = int(np.sum(np.asarray(b["synthetic_minus_human"]["lower"], float) > 0))
        lines.append(
            f"| {int(n):,} | {len(b['players'])} | {mean_rmse(b, 'uniform'):.2f} | "
            f"{mean_rmse(b, 'synthetic'):.2f} | {mean_rmse(b, 'human'):.2f} | {better_u}/10 | "
            f"{better_s}/10 |"
        )
    lines += [
        "",
        "B3: the human prior on the synthetic held-out bots (t-intervals over replicates):",
        "",
        "| Hands | Uniform | Synthetic | Human | Stats where human is worse than uniform |",
        "|---|---|---|---|---|",
    ]
    for n, b in r["reverse"].items():
        worse = int(np.sum(np.asarray(b["human_minus_uniform"]["lower"], float) > 0))
        lines.append(
            f"| {int(n):,} | {mean_rmse(b, 'uniform'):.2f} | {mean_rmse(b, 'synthetic'):.2f} | "
            f"{mean_rmse(b, 'human'):.2f} | {worse}/10 |"
        )
    lines += [
        "",
        "B4: refit on k players vs uniform (MSE difference x 10^4, positive = refit better): "
        "the smallest over n of the mean over the ten stats (the decision rule), and the worst "
        "stat:",
        "",
        "| k | Humans: mean | Humans: worst stat | Bots: mean | Bots: worst stat |",
        "|---|---|---|---|---|",
    ]
    for k in mp["grid"]:
        cells = []
        for who in ("human", "synthetic"):
            d = mp[who].get(str(k))
            if not d:
                cells += ["-", "-"]
                continue
            rows = np.array([np.asarray(v, dtype=float) for v in d.values()])
            worst = np.nanmin(rows, axis=0)
            j = int(np.nanargmin(worst))
            cells += [
                f"{1e4 * np.nanmin(np.nanmean(rows, axis=1)):+.2f}",
                f"{STAT_LABELS[STATS[j]]} {1e4 * worst[j]:+.2f}",
            ]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    clf = hum["classifier"]
    lines += [
        "",
        "## C. The synthetic-trained classifier on these players",
        "",
        "| Hands | Bots flagged | Players flagged | Control flagged | Pluribus flagged | "
        "Bots: accuracy | Players: top probability | Players' labels (nit/TAG/LAG/station/"
        "maniac) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n, s in clf["support"].items():
        lab = clf["labels"][n]
        c = ctl["classifier"]["support"].get(n, {}) if ctl else {}
        lines.append(
            f"| {int(n):,} | {s['bots_flagged']:.1%} | "
            + ("-" if s["humans_flagged"] is None else f"{s['humans_flagged']:.0%}")
            + " | "
            + ("-" if c.get("humans_flagged") is None else f"{c['humans_flagged']:.0%}")
            + " | "
            + (f"{s['bot_flagged']:.0%}" if "bot_flagged" in s else "-")
            + f" | {s['bots_accuracy']:.1%} | "
            + ("-" if lab["humans_top_prob"] is None else f"{lab['humans_top_prob']:.2f}")
            + " | "
            + ("-" if lab["humans"] is None else " / ".join(f"{x:.0%}" for x in lab["humans"]))
            + " |"
        )
    lines += [
        "",
        f"C3 consistency (reference: {r['config']['consistency_reference']:,} hands; "
        "gap = mean top probability - agreement with the reference label):",
        "",
        "| Hands | Bots: gap | Players: gap | Players: expected gap | Difference | "
        "Control: difference |",
        "|---|---|---|---|---|---|",
    ]
    for n, v in clf["consistency"].items():
        c = ctl["classifier"]["consistency"].get(n) if ctl else None
        lines.append(
            f"| {int(n):,} | {v['bots']['gap']:+.3f} | {v['humans']['gap']:+.3f} | "
            f"{v['humans']['expected_gap']:+.3f} | {_iv(v['difference'])} | "
            + (_iv(c["difference"]) if c else "-")
            + " |"
        )
    act = hum["action"]
    lines += [
        "",
        "## D. Next-action log-loss on these players' decisions",
        "",
        "| Hands observed | Decisions | "
        + " | ".join(MODEL_LABELS.values())
        + " | Synthetic - human (situation) | ECE synthetic / human |",
        "|---" * (4 + len(MODEL_LABELS)) + "|",
    ]
    for c, v in act["by_checkpoint"].items():
        ll = v["log_loss"]
        lines.append(
            f"| {int(c):,} | {v['decisions']:,} | "
            + " | ".join(f"{ll[k]:.3f}" for k in MODEL_LABELS)
            + f" | {_iv(v['syn_minus_hum_log_loss'])} | "
            f"{v['ece']['syn_sit']:.3f} / {v['ece']['hum_sit']:.3f} |"
        )
    if ctl:
        lines += ["", "Control:", "", "| Hands observed | " + " | ".join(MODEL_LABELS.values())]
        lines[-1] += " |"
        lines.append("|---" * (1 + len(MODEL_LABELS)) + "|")
        for c, v in ctl["action"]["by_checkpoint"].items():
            lines.append(
                f"| {int(c):,} | "
                + " | ".join(f"{v['log_loss'][k]:.3f}" for k in MODEL_LABELS)
                + " |"
            )
    lines += ["", f"Run time {r['seconds']:.0f} s."]
    return "\n".join(lines) + "\n"


def make_figures(r: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import band, legend, line, new_figure, save_both, style_axes, title

    hum = r["populations"]["humans"]
    ctl = r["populations"].get("control")

    def draw_estimation(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(12, 4.6), sharey=True)
        s = theme["series"]
        colors = {"uniform": s[1], "synthetic": s[3], "human": s[0]}
        panels = (
            (hum["estimation"], "Leave-one-out on the players"),
            (r["reverse"], "On the synthetic held-out bots"),
        )
        for ax, (block, name) in zip(axes, panels, strict=True):
            style_axes(ax, theme)
            ns = [int(n) for n in block]
            for prior, color in colors.items():
                y = [mean_rmse(block[str(n)], prior) for n in ns]
                line(ax, theme, ns, y, color, PRIOR_LABELS[prior] + " prior")
            ax.set_xscale("log")
            ax.set_xticks(ns, [f"{n:,}" for n in ns])
            ax.minorticks_off()
            ax.set_xlabel("Hands observed")
            ax.set_title(name, loc="left", fontsize=10, color=theme["ink"])
        axes[0].set_ylabel("RMSE vs true rate (pp, mean of 10 stats)")
        legend(axes[1], theme, loc="upper right")
        title(
            fig,
            theme,
            "Which prior estimates a player's stats best?",
            "Noise-corrected RMSE of the posterior mean against each player's rate in other "
            "hands; the human prior is fit without the player scored.",
        )
        return fig

    save_both(draw_estimation, out_dir, "priors_estimation")

    def draw_support(theme):
        fig, ax = new_figure(theme, figsize=(8.5, 4.6))
        style_axes(ax, theme)
        s = theme["series"]
        sup = hum["classifier"]["support"]
        ns = [int(n) for n in sup]
        series = [
            ("Synthetic held-out bots", [sup[str(n)]["bots_flagged"] for n in ns], s[3]),
            ("Professionals", [sup[str(n)]["humans_flagged"] for n in ns], s[0]),
        ]
        if ctl:
            cs = ctl["classifier"]["support"]
            series.append(
                ("Control (bots in their seats)", [cs[str(n)]["humans_flagged"] for n in ns], s[2])
            )
        if all("bot_flagged" in sup[str(n)] for n in ns):
            series.append(("Pluribus", [sup[str(n)]["bot_flagged"] for n in ns], s[6]))
        for label, y, color in series:
            y = [np.nan if v is None else 100 * v for v in y]
            line(ax, theme, ns, y, color, label)
        alpha = 100 * r["config"]["support_alpha"]
        ax.axhline(alpha, color=theme["muted"], linewidth=1, linestyle="--")
        ax.text(ns[0], alpha + 2, f"alpha = {alpha:.0f}%", color=theme["ink2"], fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(ns, [f"{n:,}" for n in ns])
        ax.minorticks_off()
        ax.set_ylim(0, 102)
        ax.set_xlabel("Hands in the window")
        ax.set_ylabel("Windows flagged (%)")
        legend(ax, theme, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        title(
            fig,
            theme,
            "Do these players look like any training archetype?",
            "Support check: share of profiles flagged as unlike every archetype the classifier "
            "was trained on.",
        )
        return fig

    save_both(draw_support, out_dir, "priors_support")

    def draw_consistency(theme):
        fig, ax = new_figure(theme, figsize=(8.5, 4.6))
        style_axes(ax, theme)
        s = theme["series"]
        cons = hum["classifier"]["consistency"]
        ns = [int(n) for n in cons]
        blocks = [("Professionals", cons, s[0])]
        if ctl:
            blocks.append(("Control (bots in their seats)", ctl["classifier"]["consistency"], s[2]))
        for label, block, color in blocks:
            d = [block[str(n)]["difference"] for n in ns]
            band(ax, ns, [x["lower"] for x in d], [x["upper"] for x in d], color)
            line(ax, theme, ns, [x["value"] for x in d], color, label)
        for y in (-0.10, 0.10):
            ax.axhline(y, color=theme["muted"], linewidth=1, linestyle="--")
        ax.axhline(0, color=theme["axis"], linewidth=1)
        ax.set_xscale("log")
        ax.set_xticks(ns, [f"{n:,}" for n in ns])
        ax.minorticks_off()
        ax.set_xlabel("Hands in the window")
        ax.set_ylabel("Gap minus the bots' gap")
        legend(ax, theme, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        title(
            fig,
            theme,
            "Does the classifier's confidence hold up on more of the same player's hands?",
            "Gap = top probability - agreement with the label of a later 1,000-hand block, "
            "relative to held-out bots; bands: 95% intervals; dashed: the +-0.10 margin.",
        )
        return fig

    if hum["classifier"]["consistency"]:
        save_both(draw_consistency, out_dir, "priors_consistency")

    def draw_action(theme):
        pops = [("Professionals", hum)] + ([("Control (bots in their seats)", ctl)] if ctl else [])
        fig, axes = new_figure(theme, 1, len(pops), figsize=(6 * len(pops) + 1, 4.6), squeeze=False)
        s = theme["series"]
        colors = {"syn_sit": s[3], "syn_prof": s[1], "hum_sit": s[0], "hum_prof": s[2]}
        for ax, (name, pop) in zip(axes[0], pops, strict=True):
            style_axes(ax, theme)
            by = pop["action"]["by_checkpoint"]
            cs = [int(c) for c in by]
            x = np.arange(len(cs))
            for k, color in colors.items():
                line(ax, theme, x, [by[str(c)]["log_loss"][k] for c in cs], color, MODEL_LABELS[k])
            ax.set_xticks(x, [f"{c:,}" for c in cs])
            ax.set_xlabel("Hands observed before the predicted window")
            ax.set_title(name, loc="left", fontsize=10, color=theme["ink"])
        axes[0][0].set_ylabel("Log-loss per decision (nats)")
        legend(axes[0][-1], theme, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        title(
            fig,
            theme,
            "Predicting the next action: models trained on bots vs on the other players",
            "Fold / check-call / bet-raise; each player's decisions are predicted by models "
            "that never saw that player.",
        )
        return fig

    save_both(draw_action, out_dir, "priors_action")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)["priors"]
    r = run(args.seed, cfg=cfg, workers=args.workers)
    print(markdown_summary(r))


if __name__ == "__main__":
    main()

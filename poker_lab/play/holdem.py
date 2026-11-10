"""No-limit hold'em mode: you against archetype bots or advisors, and the model's read of you.

The table is heads-up or six-handed, blinds 1/2, 100 bb, with every stack topped up each hand
(the simulator's cash game). Opponents are either Phase 1's archetype bots (archetypes drawn at
random and revealed in the report) or advisors: a pot-odds baseline adjusted by the exploit slot
(``exploit_slot``), each profiling the table from completed hands.

The read is Phase 1's profile of you: shrunk stats with 95% intervals (empirical-Bayes priors
from a bot league) and archetype probabilities from the calibrated logistic classifier. Both were
built on six-handed bot tables, so they describe which bot you most resemble, not a diagnosis;
heads-up the stats themselves shift (everyone plays wider), so the classifier is out of its
training range there, and the page says so.

The classifier's training rows come from a small league (``ClassifierConfig``) simulated once and
cached as plain arrays in the data folder; the classifier is refit from them at each session
start (under a second).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from poker_lab.advisor import Advisor, AdvisorAgent
from poker_lab.bots import ARCHETYPES, ParametricBot, make_population, sample_params
from poker_lab.bots.league import LeagueConfig, run_league
from poker_lab.engine.agent import Agent
from poker_lab.play.exploit_slot import ENGINES
from poker_lab.profiling.classifier import ArchetypeClassifier
from poker_lab.profiling.features import profile_features, window_totals
from poker_lab.profiling.profile_store import ProfileStore
from poker_lab.profiling.shrinkage import EmpiricalBayesShrinker
from poker_lab.profiling.stats import BINOMIAL_STATS
from poker_lab.schema import HandHistory, TableConfig, redact_hole_cards
from poker_lab.strategy.baselines import PotOddsStrategy

SMALL_BLIND, BIG_BLIND, STACK = 1, 2, 200
HUMAN_ID = "you"
STAT_LABELS = {
    "vpip": "VPIP (voluntarily put chips in preflop)",
    "pfr": "PFR (raised preflop)",
    "three_bet": "3-bet (re-raised preflop)",
    "fold_to_3bet": "Fold to a 3-bet",
    "cbet": "Continuation bet on the flop",
    "fold_to_cbet": "Fold to a flop c-bet",
    "aggr_share": "Aggression share (bets+raises of postflop actions)",
    "fold_to_bet": "Fold to a postflop bet",
    "wtsd": "Went to showdown after the flop",
    "wsd": "Won at showdown",
    "af": "Aggression factor (bets+raises / calls)",
    "bet_size": "Mean postflop bet (fraction of pot)",
    "open_size": "Mean open raise (bb)",
}


def table(players: int) -> TableConfig:
    return TableConfig("nl", players, SMALL_BLIND, BIG_BLIND)


@dataclass(frozen=True)
class ClassifierConfig:
    per_archetype: int = 24  # 120 players: a multiple of the table size
    hands: int = 600
    sizes: tuple[int, ...] = (25, 50, 100, 200, 300, 600)
    windows_max: int = 10
    equity_samples: int = 150
    workers: int = 4
    seed: int = 11

    def cache_name(self) -> str:
        sizes = "-".join(map(str, self.sizes))
        return (
            f"nl-training-a{self.per_archetype}-h{self.hands}-s{sizes}-w{self.windows_max}-"
            f"e{self.equity_samples}-r{self.seed}"
        )


def training_rows(cfg: ClassifierConfig, cache_dir=None) -> dict[str, Any]:
    """Classifier training rows from a bot league (cached in ``cache_dir`` when given)."""
    if cache_dir is not None:
        npz, prior_json = (
            cache_dir / f"{cfg.cache_name()}.npz",
            cache_dir / f"{cfg.cache_name()}.json",
        )
        if npz.exists() and prior_json.exists():
            import json

            with np.load(npz, allow_pickle=False) as f:
                rows = {k: f[k] for k in ("X", "y", "g")}
            rows["shrinker"] = json.loads(prior_json.read_text(encoding="utf-8"))
            return rows
    rng = np.random.default_rng(cfg.seed)
    pop = make_population(cfg.per_archetype, rng, id_prefix="t")
    league = run_league(
        pop,
        LeagueConfig(hands_per_player=cfg.hands, equity_samples=cfg.equity_samples),
        seed=cfg.seed,
        workers=min(cfg.workers, os.cpu_count() or 1),
    )
    labels = np.array([ARCHETYPES.index(p.archetype) for p in pop])
    cum = league.counters.astype(np.int64).cumsum(axis=1)
    half = cfg.hands // 2
    shrinker = EmpiricalBayesShrinker.fit(
        cum[:, -1], (cum[:, half - 1], cum[:, -1] - cum[:, half - 1])
    )
    X, y, g = [], [], []
    for n in cfg.sizes:
        w = window_totals(cum, n, cfg.windows_max)
        X.append(profile_features(shrinker, w.totals))
        y.append(labels[w.player])
        g.append(w.player)
    rows = {"X": np.concatenate(X), "y": np.concatenate(y), "g": np.concatenate(g)}
    rows["shrinker"] = shrinker.to_dict()
    if cache_dir is not None:
        import json

        np.savez(npz, X=rows["X"], y=rows["y"], g=rows["g"])
        prior_json.write_text(json.dumps(rows["shrinker"]), encoding="utf-8")
    return rows


class HoldemModel:
    """Profiles you from completed hands (redacted to public information) and classifies you."""

    def __init__(self, rows: dict[str, Any], *, seed: int = 0) -> None:
        self.shrinker = EmpiricalBayesShrinker.from_dict(rows["shrinker"])
        self.classifier = ArchetypeClassifier("lr", calibrate=True, random_state=seed).fit(
            rows["X"], rows["y"], rows["g"]
        )
        self.store = ProfileStore(self.shrinker)

    def observe(self, hand: HandHistory) -> None:
        self.store.ingest(redact_hole_cards(hand))

    def read(self, players: int) -> dict[str, Any]:
        p = self.store.profile(HUMAN_ID)
        x = profile_features(self.shrinker, np.array(p.counters))
        proba = self.classifier.predict_proba(x[None, :])[0]
        stats = []
        for name, label in STAT_LABELS.items():
            if name not in p.stats:
                continue
            iv = p.intervals.get(name)
            stats.append(
                {
                    "name": name,
                    "label": label,
                    "value": p.stats[name],
                    "interval": list(iv) if iv else None,
                    "opportunities": p.opportunities.get(name),
                    "rate": name in {s.name for s in BINOMIAL_STATS},
                }
            )
        return {
            "hands": p.hands,
            "stats": stats,
            "archetypes": [
                {"name": a.value, "probability": float(q)}
                for a, q in sorted(zip(ARCHETYPES, proba, strict=True), key=lambda t: -t[1])
            ],
            "caveat": (
                "Trained on six-handed bot tables: this says which bot archetype your stats "
                "resemble, not how a person plays."
                + (
                    " Heads-up everyone plays wider, so the classifier is outside its training "
                    "range here."
                    if players == 2
                    else ""
                )
            ),
        }


def opponents(
    kind: str, n: int, seed: int, engine: str, shrinker: EmpiricalBayesShrinker | None = None
) -> list[tuple[str, Agent, dict[str, str]]]:
    """``n`` opponents as (player id, agent, what the report reveals about them)."""
    rng = np.random.default_rng([seed, 7])
    out = []
    for i in range(1, n + 1):
        if kind == "bots":
            archetype = ARCHETYPES[int(rng.integers(len(ARCHETYPES)))]
            agent = ParametricBot(sample_params(archetype, rng))
            out.append((f"bot{i}", agent, {"archetype": archetype.value}))
        else:
            slot = ENGINES[engine]
            advisor = Advisor(
                PotOddsStrategy(seed=seed + i), slot.factory(), sample=True, seed=seed + 100 + i
            )
            out.append(
                (
                    f"advisor{i}",
                    AdvisorAgent(advisor, ProfileStore(shrinker)),
                    {"engine": slot.name, "kind": slot.kind},
                )
            )
    return out


def exploit_summary(agents: list[Agent], engine: str) -> dict[str, Any] | None:
    """What the exploit slot did in this session (advisor seats only)."""
    logs = [a for a in agents if isinstance(a, AdvisorAgent)]
    if not logs:
        return None
    advice = [x for a in logs for x in a.advice_log]
    adjusted = [x for x in advice if x.strategy.total_variation(x.baseline) > 1e-12]
    entry = ENGINES[engine]
    return {
        "engine": entry.name,
        "kind": entry.kind,
        "description": entry.description,
        "decisions": len(advice),
        "adjusted": len(adjusted),
        "mean_shift": (
            float(np.mean([x.strategy.total_variation(x.baseline) for x in adjusted]))
            if adjusted
            else 0.0
        ),
        "examples": [x.rationale for x in adjusted[-3:]],
    }

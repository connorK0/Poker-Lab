"""P2.12: exploitation in Leduc when the model is wrong.

    python -m experiments.phase2_misspec --seed 2026 [--quick]
    python -m experiments.run_phase2 --seed 2026 --only misspec

The plan and predictions below are pre-registered: written after development on seeds 98 and 99,
reviewed, and committed before the one real run on seed 2026.

The question. Phase 2's exploitation conclusions came from opponents inside the five-parameter
family the model fits. How do they hold when the opponent follows other rules or adapts?

Opponents, and how misspecified each is (measured on seeds 98 and 99 before any result on 2026,
and independently of how any hero scores: the KL divergence from the hero's view of the bot to
its closest family member, in nats per hand; hands = log(20) / KL, the hands at which the expected
evidence against the family reaches 20:1, an optimistic bound because it assumes the alternative
is known (at the median bot's KL in brackets, since the mean KL overstates a typical bot);
reach-weighted total variation per decision):

    kind           KL     hands      TV     why it is outside the family
    family         0      never      0      the control: P2.7's bots, repeated exactly
    slight_street  .025   118 (125)  .057   a quarter of `street` mixed with the family
    board_blind    .075   40 (47)    .097   ignores the board: rank buckets on both streets
    noisy          .112   27 (27)    .124   each infoset drawn around the family's rates
    honest_raiser  .115   26 (35)    .091   raises only when strong, folds more when raised
    near_eq        .160   19 (19)    .163   three quarters equilibrium, a quarter family
    positional     .196   15 (15)    .150   aggression depends on the seat
    equilibrium    .295   10 (10)    .207   the exact equilibrium: nothing can be won from it
    street         .403   7 (7)      .232   aggression and folding depend on the street

No kind is barely distinguishable by this measure. The least distinguishable is slight_street,
which needs at least about 120 hands; the others need 10 to 50.

Adaptive: a static bot starts from a family member and models the hero from its last `window`
hands. `family_*` steps its five rates towards the family rates of its best response, and stays in
the family. `tabular_*` mixes realization plans towards its full best response (fictitious play,
Kuhn's theorem). Against the equilibrium hero the free adapters end at KL about .7 to .76.

Heroes. Static part (P2.7's seeds and bots, exact values): equilibrium, best response and
restricted Nash (p = .5, .9) and data-biased response to the fitted model, a card-blind model, the
population prior, the clairvoyant oracle, and responses to the closest family member (the limit
of infinite data). Sessions (2,000 hands, scored by exact values per 50-hand block): equilibrium,
fit once at 300 hands (BR, RNR .9), refit every 100 hands (BR, RNR .9, RNR .5), refit on the last
300 hands (BR), and the clairvoyant.

Scoring. Gain over the equilibrium hero against the same bot; capture = mean gain / mean oracle
gain; share of bots where the hero does worse than equilibrium (Wilson interval over 150 bots);
exploitability. Intervals are t over 5 seeds. Per-block random streams are shared across heroes,
which is only partial common random numbers once the heroes' histories differ.

Predictions are in PREDICTIONS below, with thresholds set from seeds 98 and 99. A1-A5 are checks
whose outcome is known in advance (P2.7's committed numbers fix them, or a theorem does): they
test the code, not the question, and are counted apart. In development, the capture against an
off-family kind was almost exactly the capture of best response to its closest family member, so
C1 measures the model's bias, not the cost of too little data. The paired comparisons in C7-C10
and D1 were added after the plan review. Their thresholds come from the development seeds' means,
because those runs predate the fields; the run on seed 2026 is the first to compute them.

What this cannot show: Leduc only, these rule bots and two adaptation schemes only, the hero's
family stays fixed (no flexible model or stop-loss is tried), and nothing is about real players.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments import phase2_duplicate as pd
from experiments.common import (
    REPO_ROOT,
    RESULTS_ROOT,
    derive_seeds,
    load_config,
    run_metadata,
    save_json,
)
from poker_lab.evaluation.intervals import t_interval, wilson_interval
from poker_lab.games import LeducHoldem, compile_tree, exploitability, head_to_head
from poker_lab.games.leduc_bots import (
    BetaPrior,
    BotPolicyBuilder,
    LeducBot,
    Observation,
    ObservationIndex,
    card_blind_policy,
    fit_params,
    public_visit_counts,
    sample_population,
    simulate_hands,
)
from poker_lab.games.leduc_misspec import (
    RULES,
    STATIC,
    Adaptation,
    AdaptiveLeducBot,
    FamilyProjection,
    equilibrium_prior,
    full_best_response,
    infoset_features,
    play_session,
    rule_policy,
)
from poker_lab.games.policy import TabularPolicy
from poker_lab.solvers import RobustResponder, solve_lp

CONFIG = REPO_ROOT / "experiments" / "configs" / "phase2.yaml"
COMMITTED = RESULTS_ROOT / "phase2" / "exploit" / "exploit.json"
CONTROL = "family"
STATIC_KINDS = (CONTROL, *RULES)
SESSION_HEROES = (
    "equilibrium",
    "fit_once_br",
    "fit_once_rnr",
    "refit_br",
    "refit_rnr",
    "refit_rnr_low",
    "refit_recent_br",
    "clairvoyant",
)
SIZE_METHODS = ("br", "rnr_0.5", "rnr_0.9", "dbr")
REFERENCE_METHODS = ("br", "rnr_0.5", "rnr_0.9")
TAG_STATIC, TAG_RULE, TAG_SESSION = 1, 2, 3  # streams, apart from P2.7's [seed, bot]
LOSS_TOL = 1e-9  # a gain below -LOSS_TOL is a loss to equilibrium play
_CONTEXT: dict[str, Any] = {}


def _context() -> dict[str, Any]:
    """Per-process cache of the game, its tree, the equilibrium and the projections."""
    if not _CONTEXT:
        game = LeducHoldem()
        tree = compile_tree(game)
        builder = BotPolicyBuilder(tree, game)
        eq = solve_lp(tree).policy
        projection = FamilyProjection(builder, eq)
        _CONTEXT.update(
            game=game,
            tree=tree,
            builder=builder,
            eq=eq,
            responder=RobustResponder(tree),
            features=infoset_features(builder),
            projection=projection,
            eq_theta=projection.closest(eq),
        )
    return _CONTEXT


def opponent_policy(kind: str, bot: LeducBot, seed: int, index: int) -> TabularPolicy:
    ctx = _context()
    if kind == CONTROL:
        return ctx["builder"].policy(bot.params)
    # only ``noisy`` draws from this generator
    rng = np.random.default_rng(np.random.SeedSequence([seed, TAG_RULE, index]))
    return rule_policy(
        kind, ctx["builder"], bot.params, features=ctx["features"], rng=rng, equilibrium=ctx["eq"]
    )


def responses(
    model: TabularPolicy, ps: list[float], confidence: np.ndarray | None
) -> dict[str, TabularPolicy]:
    rr = _context()["responder"]
    out = {"br": full_best_response(model)}
    for p in ps:
        out[f"rnr_{p:g}"] = rr.both_seats("rnr", model, p=p)
    if confidence is not None:
        out["dbr"] = rr.both_seats("dbr", model, confidence=confidence)
    return out


def static_task(
    seed: int, bot: LeducBot, kind: str, prior: BetaPrior, cfg: dict[str, Any], index: int
) -> dict[str, Any]:
    """One opponent, static: the same analysis as P2.7, plus how far outside the family it is."""
    ctx = _context()
    game, tree, builder, eq = ctx["game"], ctx["tree"], ctx["builder"], ctx["eq"]
    truth = opponent_policy(kind, bot, seed, index)
    measured = ctx["projection"].measure(truth)
    eq_value = head_to_head(eq, truth)
    entropy = (
        [seed, index] if kind == CONTROL else [seed, TAG_STATIC, index, STATIC_KINDS.index(kind)]
    )
    rng = np.random.default_rng(np.random.SeedSequence(entropy))  # P2.7's stream for the control
    obs = simulate_hands(tree, game, eq, truth, max(cfg["sizes"]), rng)

    def score(policies: dict[str, TabularPolicy]) -> dict[str, dict[str, float]]:
        return {
            name: {
                "gain": head_to_head(pol, truth) - eq_value,
                "exploitability": exploitability(pol),
            }
            for name, pol in policies.items()
        }

    record: dict[str, Any] = {
        "kind": kind,
        "bot_id": bot.bot_id,
        "archetype": bot.archetype,
        "params": bot.params.as_array(),
        "kl": measured.kl,
        "tv": measured.tv,
        "closest": measured.theta.as_array(),
        "bot_exploitability": exploitability(truth),
        "eq_value": eq_value,
        "oracle": score(responses(truth, cfg["ps"], None)),
        "limit": score(responses(builder.policy(measured.theta), cfg["ps"], None)),
        "population": score(responses(builder.policy(prior.mean), cfg["ps"], None)),
        "sizes": {},
    }
    for n in cfg["sizes"]:
        fit = fit_params(builder, ObservationIndex(tree, game, obs[:n]), prior)
        counts = public_visit_counts(tree, game, obs[:n])
        confidence = counts / (counts + cfg["dbr_pseudocount"])
        blind = full_best_response(card_blind_policy(builder, game, obs[:n]))
        scores = score(responses(builder.policy(fit), cfg["ps"], confidence))
        scores["blind_br"] = score({"blind_br": blind})["blind_br"]
        record["sizes"][str(n)] = scores
    return record


class FitHero:
    """Re-decides from its redacted view every ``every`` hands from ``start`` on (the equilibrium
    before), on all completed hands or the last ``window``; with ``once``, a single time."""

    def __init__(
        self,
        prior: BetaPrior,
        p: float | None,
        *,
        every: int,
        start: int = 0,
        window: int | None = None,
        once: bool = False,
    ) -> None:
        self.prior, self.p, self.every, self.start = prior, p, every, start
        self.window, self.once, self.decided = window, once, False
        self.current = _context()["eq"]

    def __call__(self, seen: list[Observation], hand: int) -> TabularPolicy:
        if hand < self.start or (hand - self.start) % self.every or (self.once and self.decided):
            return self.current
        ctx = _context()
        recent = seen if self.window is None else seen[-self.window :]
        index = ObservationIndex(ctx["tree"], ctx["game"], recent)
        model = ctx["builder"].policy(fit_params(ctx["builder"], index, self.prior))
        self.current = (
            full_best_response(model)
            if self.p is None
            else ctx["responder"].both_seats("rnr", model, p=self.p)
        )
        self.decided = True
        return self.current


def make_hero(name: str, prior: BetaPrior, s: dict[str, Any], bot: AdaptiveLeducBot):
    ctx = _context()
    every, p = s["refit_every"], s["rnr_p"]
    once = {"every": 1, "start": s["fit_once_at"], "once": True}
    heroes = {
        "equilibrium": lambda seen, hand: ctx["eq"],
        "fit_once_br": lambda: FitHero(prior, None, **once),
        "fit_once_rnr": lambda: FitHero(prior, p, **once),
        "refit_br": lambda: FitHero(prior, None, every=every),
        "refit_rnr": lambda: FitHero(prior, p, every=every),
        "refit_rnr_low": lambda: FitHero(prior, s["rnr_p_low"], every=every),
        "refit_recent_br": lambda: FitHero(prior, None, every=every, window=s["recent_window"]),
        "clairvoyant": lambda seen, hand: full_best_response(bot.policy),
    }
    hero = heroes[name]
    return hero if name in ("equilibrium", "clairvoyant") else hero()


def adaptations(cfg: dict[str, Any]) -> dict[str, Adaptation]:
    s = cfg["session"]
    out = {"static": STATIC}
    for kind, a in cfg["adaptive"].items():
        out[kind] = Adaptation(a["speed"], a["window"], s["period"], s["pseudo"], a["mode"])
    return out


def session_task(
    seed: int, bot: LeducBot, kind: str, prior: BetaPrior, cfg: dict[str, Any], index: int
) -> dict[str, Any]:
    """One starting bot and one way of adapting, met by every hero on the same random streams."""
    ctx = _context()
    builder, projection = ctx["builder"], ctx["projection"]
    s = cfg["session"]
    adaptation = adaptations(cfg)[kind]
    hero_prior = equilibrium_prior(ctx["eq_theta"], adaptation.pseudo)
    kind_index = list(adaptations(cfg)).index(kind)
    record: dict[str, Any] = {"kind": kind, "bot_id": bot.bot_id, "heroes": {}}

    def block_rng(block: int) -> np.random.Generator:
        entropy = [seed, TAG_SESSION, index, kind_index, block]
        return np.random.default_rng(np.random.SeedSequence(entropy))

    for name in SESSION_HEROES:
        adaptive = AdaptiveLeducBot(builder, bot.params, adaptation, hero_prior)
        hero = make_hero(name, prior, s, adaptive)
        session = play_session(hero, adaptive, s["hands"], block_rng)
        half = s["hands"] // 2
        record["heroes"][name] = {
            "value": session.value(),
            "value_first_half": session.value(0, half),
            "value_second_half": session.value(half),
            "blocks": session.values,
            "exploitability": float(session.hero_exploitability.mean()),
            "bot_exploitability": float(session.bot_exploitability.mean()),
            "bot_moved": projection.total_variation(session.bot_start, session.bot_final),
            "bot_kl_end": projection.measure(session.bot_final).kl
            if name == "equilibrium"
            else None,
        }
    return record


# ---- aggregation --------------------------------------------------------------------------


def _ci(values) -> dict[str, float]:
    iv = t_interval(np.asarray(values, dtype=float))
    return {"mean": iv.estimate, "lower": iv.lower, "upper": iv.upper, "n": iv.n}


def _share(flags: list[bool]) -> dict[str, float]:
    iv = wilson_interval(int(np.sum(flags)), len(flags))
    return {"share": iv.estimate, "lower": iv.lower, "upper": iv.upper, "n": len(flags)}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _method_summary(
    bots_by_seed: list[list[dict[str, Any]]], get, oracle_gain_by_seed: list[list[float]]
) -> dict[str, Any]:
    """Gain, exploitability, share of the available gain and losses for one method."""
    gains = [[get(b)["gain"] for b in bots] for bots in bots_by_seed]
    expl = [[get(b)["exploitability"] for b in bots] for bots in bots_by_seed]
    capture = [
        float(np.mean(g) / np.mean(o)) if abs(np.mean(o)) > 1e-12 else float("nan")
        for g, o in zip(gains, oracle_gain_by_seed, strict=True)
    ]
    shortfall = [
        float(np.mean(o) - np.mean(g)) for g, o in zip(gains, oracle_gain_by_seed, strict=True)
    ]
    return {
        "gain": _ci([np.mean(g) for g in gains]),
        "exploitability": _ci([np.mean(e) for e in expl]),
        "capture": _ci(capture) if np.isfinite(capture).all() else None,
        "capture_by_seed": capture if np.isfinite(capture).all() else None,
        "shortfall": _ci(shortfall),
        "loses": _share([v < -LOSS_TOL for g in gains for v in g]),
    }


def _versus_br(by_seed: list[list[dict[str, Any]]], cfg) -> dict[str, Any]:
    """Restricted Nash and the data-biased response against best response at the largest size
    (ratios of seed means, and the paired gain difference), and the population model against
    best response after the fewest hands."""
    top, first = str(max(cfg["sizes"])), str(min(cfg["sizes"]))

    def mean(bots, n, m, metric):
        return float(np.mean([b["sizes"][n][m][metric] for b in bots]))

    out: dict[str, Any] = {
        m: {
            "gain_ratio": _ci(
                [mean(b, top, m, "gain") / mean(b, top, "br", "gain") for b in by_seed]
            ),
            "exploitability_ratio": _ci(
                [
                    mean(b, top, m, "exploitability") / mean(b, top, "br", "exploitability")
                    for b in by_seed
                ]
            ),
            "gain_difference": _ci(
                [mean(b, top, m, "gain") - mean(b, top, "br", "gain") for b in by_seed]
            ),
        }
        for m in ("rnr_0.9", "dbr")
    }
    out["population_minus_first"] = _ci(
        [
            float(np.mean([b["population"]["br"]["gain"] for b in bots]))
            - mean(bots, first, "br", "gain")
            for bots in by_seed
        ]
    )
    return out


def aggregate_static(records: list[dict[str, Any]], seeds: list[int], cfg) -> dict[str, Any]:
    out: dict[str, Any] = {}
    per = len(records) // (len(seeds) * len(STATIC_KINDS))
    for k, kind in enumerate(STATIC_KINDS):
        by_seed = [
            records[(j * len(STATIC_KINDS) + k) * per : (j * len(STATIC_KINDS) + k + 1) * per]
            for j in range(len(seeds))
        ]
        oracle = [[b["oracle"]["br"]["gain"] for b in bots] for bots in by_seed]
        entry: dict[str, Any] = {
            "kl": _ci([np.mean([b["kl"] for b in bots]) for bots in by_seed]),
            "tv": _ci([np.mean([b["tv"] for b in bots]) for bots in by_seed]),
            "bot_exploitability": _ci(
                [np.mean([b["bot_exploitability"] for b in bots]) for bots in by_seed]
            ),
            "eq_value": _ci([np.mean([b["eq_value"] for b in bots]) for bots in by_seed]),
            "oracle_gain": _ci([np.mean(o) for o in oracle]),
            "eq_value_min": float(min(b["eq_value"] for bots in by_seed for b in bots)),
            "kl_median": float(np.median([b["kl"] for bots in by_seed for b in bots])),
        }
        mean_kl = entry["kl"]["mean"]
        entry["hands_to_evidence"] = float(np.log(20.0) / mean_kl) if mean_kl > 0 else None
        for group in ("oracle", "limit", "population"):
            entry[group] = {
                m: _method_summary(by_seed, lambda b, g=group, m=m: b[g][m], oracle)
                for m in REFERENCE_METHODS
            }
        entry["sizes"] = {
            str(n): {
                m: _method_summary(by_seed, lambda b, n=str(n), m=m: b["sizes"][n][m], oracle)
                for m in (*SIZE_METHODS, "blind_br")
            }
            for n in cfg["sizes"]
        }
        entry["fit_minus_blind"] = {
            str(n): _ci(
                [
                    np.mean(
                        [
                            b["sizes"][str(n)]["br"]["gain"]
                            - b["sizes"][str(n)]["blind_br"]["gain"]
                            for b in bots
                        ]
                    )
                    for bots in by_seed
                ]
            )
            for n in cfg["sizes"]
        }
        entry["versus_br"] = _versus_br(by_seed, cfg)
        out[kind] = entry
    # What the wrong model costs against the same starting bots: the control's share of the
    # available gain minus this kind's, seed by seed.
    for kind in STATIC_KINDS:
        out[kind]["capture_below_control"] = {
            m: {
                str(n): _ci(
                    np.subtract(
                        out[CONTROL]["sizes"][str(n)][m]["capture_by_seed"],
                        out[kind]["sizes"][str(n)][m]["capture_by_seed"],
                    )
                )
                for n in cfg["sizes"]
            }
            for m in ("br", "rnr_0.9")
            if out[kind]["sizes"][str(cfg["sizes"][0])][m]["capture_by_seed"] is not None
        }
    top = str(max(cfg["sizes"]))
    pooled = [
        (b["kl"], b["oracle"]["br"]["gain"] - b["sizes"][top]["br"]["gain"])
        for b in records
        if b["kind"] != CONTROL
    ]
    x, y = np.array(pooled).T
    out["kl_shortfall_spearman"] = _spearman(x, y)
    return out


def aggregate_sessions(records: list[dict[str, Any]], seeds: list[int], cfg) -> dict[str, Any]:
    out: dict[str, Any] = {}
    kinds = list(adaptations(cfg))
    for k, kind in enumerate(kinds):
        by_seed = _session_seeds(records, seeds, kinds, k)
        eq_rows = [[b["heroes"]["equilibrium"] for b in bots] for bots in by_seed]
        entry: dict[str, Any] = {
            "bot_moved": _ci([np.mean([r["bot_moved"] for r in rows]) for rows in eq_rows]),
            "bot_kl_end": _ci([np.mean([r["bot_kl_end"] for r in rows]) for rows in eq_rows]),
            "bot_exploitability": _ci(
                [np.mean([r["bot_exploitability"] for r in rows]) for rows in eq_rows]
            ),
            "eq_value_min": float(min(r["value"] for rows in eq_rows for r in rows)),
            "heroes": {},
        }
        for hero in SESSION_HEROES:
            rows = [[b["heroes"][hero] for b in bots] for bots in by_seed]
            gains = [
                [r["value"] - e["value"] for r, e in zip(a, c, strict=True)]
                for a, c in zip(rows, eq_rows, strict=True)
            ]
            second = [
                [r["value_second_half"] - e["value_second_half"] for r, e in zip(a, c, strict=True)]
                for a, c in zip(rows, eq_rows, strict=True)
            ]
            curve = np.mean(
                [
                    [
                        np.asarray(r["blocks"]) - np.asarray(e["blocks"])
                        for r, e in zip(a, c, strict=True)
                    ]
                    for a, c in zip(rows, eq_rows, strict=True)
                ],
                axis=(0, 1),
            )
            entry["heroes"][hero] = {
                "value": _ci([np.mean([r["value"] for r in a]) for a in rows]),
                "gain": _ci([np.mean(g) for g in gains]),
                "gain_second_half": _ci([np.mean(g) for g in second]),
                "loses": _share([v < -LOSS_TOL for g in gains for v in g]),
                "exploitability": _ci([np.mean([r["exploitability"] for r in a]) for a in rows]),
                "curve": curve,
            }

        entry["differences"] = {
            name: _ci(
                [
                    np.mean([b["heroes"][a]["value"] - b["heroes"][c]["value"] for b in bots])
                    for bots in by_seed
                ]
            )
            for name, a, c in (
                ("refit_minus_fit_once", "refit_br", "fit_once_br"),
                ("recent_minus_refit", "refit_recent_br", "refit_br"),
                ("rnr_minus_br", "refit_rnr", "refit_br"),
                ("rnr_low_minus_br", "refit_rnr_low", "refit_br"),
            )
        }
        # Each refitting hero's share of the clairvoyant's gain, a ratio of seed means.
        clair = [
            np.mean(
                [
                    b["heroes"]["clairvoyant"]["value"] - b["heroes"]["equilibrium"]["value"]
                    for b in bots
                ]
            )
            for bots in by_seed
        ]
        entry["capture"] = {
            hero: _ci(
                [
                    np.mean(
                        [
                            b["heroes"][hero]["value"] - b["heroes"]["equilibrium"]["value"]
                            for b in bots
                        ]
                    )
                    / c
                    for bots, c in zip(by_seed, clair, strict=True)
                ]
            )
            for hero in ("refit_br", "refit_rnr", "refit_recent_br")
        }
        out[kind] = entry
    # What adapting takes from the equilibrium hero, against the same starting bots.
    control = [
        [b["heroes"]["equilibrium"]["value"] for b in bots]
        for bots in _session_seeds(records, seeds, kinds, 0)
    ]
    for k, kind in enumerate(kinds):
        rows = [
            [b["heroes"]["equilibrium"]["value"] for b in bots]
            for bots in _session_seeds(records, seeds, kinds, k)
        ]
        out[kind]["eq_value_ratio"] = _ci(
            [np.mean(a) / np.mean(c) for a, c in zip(rows, control, strict=True)]
        )
    return out


def _session_seeds(records, seeds, kinds, k: int) -> list[list[dict[str, Any]]]:
    """The records of the ``k``-th kind, grouped by seed."""
    per = len(records) // (len(seeds) * len(kinds))
    return [
        records[(j * len(kinds) + k) * per : (j * len(kinds) + k + 1) * per]
        for j in range(len(seeds))
    ]


def committed_check(seed: int, records: list[dict[str, Any]], cfg) -> dict[str, Any] | None:
    """The control must repeat P2.7's numbers: same bots, same hands (seed 2026 only)."""
    if not COMMITTED.exists():
        return None
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    if committed["meta"]["seed"] != seed:
        return None
    by_id = {
        (s, b["bot_id"]): b
        for s, bots in zip(committed["seeds"], committed["per_seed"], strict=True)
        for b in bots
    }
    worst, n = 0.0, 0
    for r in records:
        if r["kind"] != CONTROL:
            continue
        old = by_id[(r["seed"], r["bot_id"])]
        worst = max(worst, abs(old["eq_value"] - r["eq_value"]))
        n += 1
        for size in cfg["sizes"]:
            for m in ("br", "rnr_0.5", "rnr_0.9", "dbr"):
                for metric in ("gain", "exploitability"):
                    a = old["sizes"][str(size)]["scores"][m][metric]
                    worst = max(worst, abs(a - r["sizes"][str(size)][m][metric]))
    return {"bots": n, "max_abs_difference": worst}


def _slim(r: dict[str, Any], cfg) -> dict[str, Any]:
    top = str(max(cfg["sizes"]))
    return {
        k: r[k]
        for k in (
            "kind",
            "seed",
            "bot_id",
            "archetype",
            "kl",
            "tv",
            "bot_exploitability",
            "eq_value",
        )
    } | {
        "oracle_gain": r["oracle"]["br"]["gain"],
        "limit_gain": r["limit"]["br"]["gain"],
        "br_gain_top": r["sizes"][top]["br"]["gain"],
        "rnr_gain_top": r["sizes"][top]["rnr_0.9"]["gain"],
    }


def run(
    seed: int, *, cfg: dict[str, Any], workers: int | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or RESULTS_ROOT / "phase2" / "misspec"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, cfg)
    t_start = time.perf_counter()
    seeds = derive_seeds(seed, cfg["seeds"])
    static_jobs, session_jobs = [], []
    for s in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([s, 0]))
        train = sample_population(cfg["train_per_archetype"], rng, prefix="t")
        test = sample_population(cfg["test_per_archetype"], rng, prefix="b")
        prior = BetaPrior.fit(np.array([b.params.as_array() for b in train]))
        for kind in STATIC_KINDS:
            static_jobs += [(s, bot, kind, prior, cfg, k) for k, bot in enumerate(test)]
        for kind in adaptations(cfg):
            session_jobs += [(s, bot, kind, prior, cfg, k) for k, bot in enumerate(test)]
    workers = max(1, workers or 8)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        t0 = time.perf_counter()
        futures = [pool.submit(static_task, *job) for job in static_jobs]
        static = [
            f.result() | {"seed": job[0]} for job, f in zip(static_jobs, futures, strict=True)
        ]
        static_seconds = round(time.perf_counter() - t0, 1)
        print(f"static opponents done ({static_seconds:.0f} s)", flush=True)
        t0 = time.perf_counter()
        futures = [pool.submit(session_task, *job) for job in session_jobs]
        sessions = [f.result() for f in futures]
        session_seconds = round(time.perf_counter() - t0, 1)
        print(f"sessions done ({session_seconds:.0f} s)", flush=True)
    out = {
        "meta": meta,
        "seeds": seeds,
        "static": aggregate_static(static, seeds, cfg),
        "sessions": aggregate_sessions(sessions, seeds, cfg),
        "committed_check": committed_check(seed, static, cfg),
        "per_bot": [_slim(r, cfg) for r in static],
        "static_seconds": static_seconds,
        "session_seconds": session_seconds,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    out["predictions"] = pd.predictions(out, PREDICTIONS)
    save_json(out, out_dir / "misspec.json")
    (out_dir / "misspec_summary.md").write_text(
        markdown_summary(out), encoding="utf-8", newline="\n"
    )
    make_figures(out, out_dir)
    return out


_OFF = ("board_blind", "street", "positional", "honest_raiser", "noisy", "near_eq")
_FAMILY_ADAPT = ("family_slow", "family_fast")
_FREE_ADAPT = ("tabular_slow", "tabular_fast")
_TOP = "3000"


def _static(kinds, *path) -> list[tuple]:
    return [("static", k, *path) for k in kinds]


def _session(kinds, *path) -> list[tuple]:
    return [("sessions", k, *path) for k in kinds]


def _p(claim: str, values: list[tuple], test: tuple) -> dict[str, Any]:
    return {"claim": claim, "values": values, "test": test}


PREDICTIONS: dict[str, dict[str, Any]] = {
    # ---- checks: known in advance, they test the code ----
    "A1": _p(
        "Check: the control repeats P2.7's committed eq value, gain and exploitability of BR, "
        "RNR .5/.9 and DBR at every size",
        [("committed_check", "max_abs_difference")],
        ("<=", 1e-9),
    ),
    "A2": _p(
        "Check: the comparison covers all 150 control bots",
        [("committed_check", "bots")],
        ("between", [150, 150]),
    ),
    "A3": _p(
        "Check: the control's KL to the family is at most 1e-6",
        [("static", CONTROL, "kl", "mean")],
        ("<=", 1e-6),
    ),
    "A4": _p(
        "Check: the equilibrium hero never has negative value against any bot",
        _static(STATIC_KINDS, "eq_value_min")
        + _session(("static", *_FAMILY_ADAPT, *_FREE_ADAPT), "eq_value_min"),
        (">=", -1e-6),
    ),
    "A5": _p(
        "Check: with 3,000 hands BR captures at least 97% of the control's oracle gain",
        [("static", CONTROL, "sizes", _TOP, "br", "capture", "mean")],
        (">=", 0.97),
    ),
    # ---- how misspecified ----
    "B1": _p(
        "Every rule kind's KL-implied hands to 20:1 (an optimistic bound) are at most 150",
        _static(RULES, "hands_to_evidence"),
        ("<=", 150),
    ),
    "B2": _p(
        "slight_street's KL interval lies below .04",
        [("static", "slight_street", "kl", "upper")],
        ("<=", 0.04),
    ),
    # ---- opponents that do not adapt ----
    "C1": _p(
        "With 3,000 hands BR captures between 60% and 90% against each off-family kind",
        _static(_OFF, "sizes", _TOP, "br", "capture", "mean"),
        ("between", [0.6, 0.9]),
    ),
    "C2": _p(
        "slight_street's BR capture at 3,000 hands is between 92% and 99%",
        [("static", "slight_street", "sizes", _TOP, "br", "capture", "mean")],
        ("between", [0.92, 0.99]),
    ),
    "C3": _p(
        "BR's gain over equilibrium at 3,000 hands has an interval above .1 for each off-family "
        "kind",
        _static(_OFF, "sizes", _TOP, "br", "gain", "lower"),
        (">=", 0.1),
    ),
    "C4": _p(
        "BR does worse than equilibrium on at most 20% of bots of each off-family kind at 3,000",
        _static(_OFF, "sizes", _TOP, "br", "loses", "share"),
        ("<=", 0.2),
    ),
    "C5a": _p(
        "Against the exact equilibrium BR's gain interval at 3,000 hands lies below -.01",
        [("static", "equilibrium", "sizes", _TOP, "br", "gain", "upper")],
        ("<=", -0.01),
    ),
    "C5b": _p(
        "Against the exact equilibrium BR does worse than equilibrium on at least 95% of bots",
        [("static", "equilibrium", "sizes", _TOP, "br", "loses", "share")],
        (">=", 0.95),
    ),
    "C6": _p(
        "The rank correlation of KL with BR's shortfall at 3,000 hands is between .2 and .55",
        [("static", "kl_shortfall_spearman")],
        ("between", [0.2, 0.55]),
    ),
    "C7": _p(
        "RNR .9 keeps at least 87% of BR's gain at 3,000 hands against each off-family kind",
        _static(_OFF, "versus_br", "rnr_0.9", "gain_ratio", "mean"),
        (">=", 0.87),
    ),
    "C8": _p(
        "RNR .9's exploitability at 3,000 hands is at most 62% of BR's against every kind",
        _static(STATIC_KINDS, "versus_br", "rnr_0.9", "exploitability_ratio", "mean"),
        ("<=", 0.62),
    ),
    "C9": _p(
        "DBR gains at least .03 less than BR at 3,000 hands against each off-family kind",
        _static(_OFF, "versus_br", "dbr", "gain_difference", "upper"),
        ("<=", -0.03),
    ),
    "C10": _p(
        "Against near_eq and board_blind the population model beats BR fitted on 10 hands",
        _static(("near_eq", "board_blind"), "versus_br", "population_minus_first", "lower"),
        (">=", 0.0),
    ),
    "C11": _p(
        "The five-parameter model beats the card-blind one at 3,000 hands for each off-family kind",
        _static(_OFF, "fit_minus_blind", _TOP, "lower"),
        (">=", 0.0),
    ),
    # ---- opponents that adapt ----
    "D1": _p(
        "Refit BR keeps at least 85% of the clairvoyant's gain against adapters inside the family",
        _session(_FAMILY_ADAPT, "capture", "refit_br", "mean"),
        (">=", 0.85),
    ),
    "D2": _p(
        "Refit BR's gain over fit-once BR has an interval above 0 against all four adapters",
        _session((*_FAMILY_ADAPT, *_FREE_ADAPT), "differences", "refit_minus_fit_once", "lower"),
        (">=", 0.0),
    ),
    "D3": _p(
        "Against free adapters refit BR and RNR .9 lose to equilibrium in the second half",
        _session(_FREE_ADAPT, "heroes", "refit_br", "gain_second_half", "upper")
        + _session(_FREE_ADAPT, "heroes", "refit_rnr", "gain_second_half", "upper"),
        ("<=", 0.0),
    ),
    "D4": _p(
        "Against the fast free adapter refit BR, RNR .9 and recent-window BR do worse than "
        "equilibrium on at least 90% of bots",
        [
            ("sessions", "tabular_fast", "heroes", h, "loses", "share")
            for h in ("refit_br", "refit_rnr", "refit_recent_br")
        ],
        (">=", 0.9),
    ),
    "D5": _p(
        "Against free adapters RNR .9 does not protect: its gain over BR has an upper bound of "
        "at most .02",
        _session(_FREE_ADAPT, "differences", "rnr_minus_br", "upper"),
        ("<=", 0.02),
    ),
    "D6": _p(
        "Against the fast free adapter refit RNR .5's gain interval lies within .02 of 0",
        [
            ("sessions", "tabular_fast", "heroes", "refit_rnr_low", "gain", "lower"),
            ("sessions", "tabular_fast", "heroes", "refit_rnr_low", "gain", "upper"),
        ],
        ("between", [-0.02, 0.02]),
    ),
    "D7": _p(
        "The clairvoyant gains at least .7 against free adapters",
        _session(_FREE_ADAPT, "heroes", "clairvoyant", "gain", "mean"),
        (">=", 0.7),
    ),
    "D8": _p(
        "Equilibrium's value against free adapters is at most half its value against fixed bots",
        _session(_FREE_ADAPT, "eq_value_ratio", "upper"),
        ("<=", 0.5),
    ),
    "D9": _p(
        "Against the slow free adapter the 300-hand window beats refit on all hands",
        [("sessions", "tabular_slow", "differences", "recent_minus_refit", "lower")],
        (">=", 0.0),
    ),
}

# ---- reporting ----------------------------------------------------------------------------

KIND_LABELS = {
    "family": "Family bot (control)",
    "board_blind": "Ignores the board",
    "street": "Passive, then aggressive",
    "slight_street": "A quarter of that",
    "positional": "Stabs in position",
    "honest_raiser": "Raises only when strong",
    "noisy": "Noise at every decision",
    "near_eq": "75% equilibrium",
    "equilibrium": "The equilibrium",
}
SESSION_LABELS = {
    "static": "Static family bot (control)",
    "family_slow": "Adapts inside the family, slowly",
    "family_fast": "Adapts inside the family, fast",
    "tabular_slow": "Adapts freely, slowly",
    "tabular_fast": "Adapts freely, fast",
}
HERO_LABELS = {
    "equilibrium": "Equilibrium",
    "fit_once_br": "Fit once, best response",
    "fit_once_rnr": "Fit once, restricted Nash",
    "refit_br": "Refit, best response",
    "refit_rnr": "Refit, restricted Nash, p = 0.9",
    "refit_rnr_low": "Refit, restricted Nash, p = 0.5",
    "refit_recent_br": "Refit on recent hands, best response",
    "clairvoyant": "Clairvoyant (reference)",
}
METHOD_LABELS = {
    "br": "Best response",
    "rnr_0.5": "Restricted Nash, p = 0.5",
    "rnr_0.9": "Restricted Nash, p = 0.9",
    "dbr": "Data-biased",
    "blind_br": "Best response, card-blind model",
}


def _signed(q: dict[str, float], digits: int = 3) -> str:
    return f"{q['mean']:+.{digits}f} [{q['lower']:+.{digits}f}, {q['upper']:+.{digits}f}]"


def _share_text(q: dict[str, float]) -> str:
    return f"{q['share']:.0%}"


def markdown_summary(r: dict[str, Any]) -> str:
    meta, cfg, st, se = r["meta"], r["meta"]["config"], r["static"], r["sessions"]
    sizes = [str(n) for n in cfg["sizes"]]
    top = sizes[-1]
    per_kind = cfg["seeds"] * cfg["test_per_archetype"] * 5
    lines = [
        "# P2.12: exploitation when the model is wrong",
        "",
        f"Seed {meta['seed']}, commit {meta.get('git_commit')} (dirty: {meta.get('git_dirty')}), "
        f"{r['elapsed_seconds']:.0f} s. {per_kind} starting bots per kind ({cfg['seeds']} seeds x "
        f"{cfg['test_per_archetype'] * 5}). Chips per hand, exact. Gain = value against the true "
        "opponent minus the equilibrium's. Means over bots, 95% t-intervals across seeds; "
        '"loses" = share of bots with a negative gain.',
        "",
        "## How far outside the family (no hero's result is used)",
        "",
        "| Opponent | KL (nats/hand) | Hands to 20:1 | Mean TV per decision | Its exploitability | "
        "Equilibrium's value | Available gain (oracle) |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind in STATIC_KINDS:
        e = st[kind]
        hands = "never" if kind == CONTROL else f"{e['hands_to_evidence']:,.0f}"
        lines.append(
            f"| {KIND_LABELS[kind]} | {e['kl']['mean']:.4f} | {hands} | {e['tv']['mean']:.3f} | "
            f"{e['bot_exploitability']['mean']:.3f} | {e['eq_value']['mean']:+.3f} | "
            f"{e['oracle_gain']['mean']:+.3f} |"
        )
    lines += [
        "",
        "Spearman correlation over off-family bots of KL and best response's shortfall at "
        f"{int(top):,} hands: {st['kl_shortfall_spearman']:.2f}.",
        "",
        "## Best response to the fitted model: gain by hands observed",
        "",
        "| Opponent | "
        + " | ".join(f"{int(n):,}" for n in sizes)
        + " | Limit (infinite data) | Oracle |",
        "|---" * (len(sizes) + 3) + "|",
    ]
    for kind in STATIC_KINDS:
        e = st[kind]
        cells = [f"{e['sizes'][n]['br']['gain']['mean']:+.3f}" for n in sizes]
        lines.append(
            f"| {KIND_LABELS[kind]} | "
            + " | ".join(cells)
            + f" | {e['limit']['br']['gain']['mean']:+.3f} | {e['oracle_gain']['mean']:+.3f} |"
        )
    lines += [
        "",
        f"## Every response at {int(top):,} hands",
        "",
        "| Opponent | Response | Gain | Share of the available gain | Loses | Exploitability |",
        "|---|---|---|---|---|---|",
    ]
    for kind in STATIC_KINDS:
        for m in (*SIZE_METHODS, "blind_br"):
            q = st[kind]["sizes"][top][m]
            capture = f"{q['capture']['mean']:.0%}" if q["capture"] else ""
            lines.append(
                f"| {KIND_LABELS[kind]} | {METHOD_LABELS[m]} | {_signed(q['gain'])} | {capture} | "
                f"{_share_text(q['loses'])} | {q['exploitability']['mean']:.3f} |"
            )
    lines += [
        "",
        "## How often each response loses to equilibrium play, by hands observed",
        "",
        "| Opponent | Response | " + " | ".join(f"{int(n):,}" for n in sizes) + " |",
        "|---|---" + "|---" * len(sizes) + "|",
    ]
    for kind in STATIC_KINDS:
        for m in ("br", "rnr_0.9"):
            cells = [_share_text(st[kind]["sizes"][n][m]["loses"]) for n in sizes]
            lines.append(
                f"| {KIND_LABELS[kind]} | {METHOD_LABELS[m]} | " + " | ".join(cells) + " |"
            )
    s = cfg["session"]
    lines += [
        "",
        f"## Sessions against adapters: {s['hands']:,} hands, updates every {s['period']}",
        "",
    ]
    for kind, e in se.items():
        lines += [
            f"### {SESSION_LABELS[kind]}",
            "",
            f"With the equilibrium hero, the bot moves {e['bot_moved']['mean']:.3f} (TV per "
            f"decision), ends {e['bot_kl_end']['mean']:.4f} nats/hand from the family, and its "
            f"mean exploitability is {e['bot_exploitability']['mean']:.3f}.",
            "",
            "| Hero | Value | Gain over equilibrium | Second half | Loses | Exploitability |",
            "|---|---|---|---|---|---|",
        ]
        for hero, x in e["heroes"].items():
            lines.append(
                f"| {HERO_LABELS[hero]} | {_signed(x['value'])} | {_signed(x['gain'])} | "
                f"{_signed(x['gain_second_half'])} | {_share_text(x['loses'])} | "
                f"{x['exploitability']['mean']:.3f} |"
            )
        lines.append("")
    pred = r["predictions"]
    if pred:
        lines += ["## Pre-registered predictions", "", "| | Prediction | Values | Held |"]
        lines.append("|---|---|---|---|")
        for key, p in pred.items():
            values = ", ".join("n/a" if v is None else f"{v:.4g}" for v in p["values"].values())
            lines.append(f"| {key} | {p['claim']} | {values} | {'yes' if p['passed'] else 'NO'} |")
    check = r["committed_check"]
    if check:
        lines += [
            "",
            f"Control against the committed P2.7 results: {check['bots']} bots, largest difference "
            f"{check['max_abs_difference']:.3g}.",
        ]
    return "\n".join(lines) + "\n"


def make_figures(r: dict[str, Any], out_dir: Path) -> None:
    from experiments.plotting import legend, new_figure, save_both, style_axes, title

    st, se, cfg = r["static"], r["sessions"], r["meta"]["config"]
    sizes = [str(n) for n in cfg["sizes"]]

    def by_hands(theme):
        fig, axes = new_figure(theme, 1, 2, figsize=(11, 4.6))
        s = theme["series"]
        for ax in axes:
            style_axes(ax, theme, grid_axis="y")
            ax.set_xscale("log")
            ax.axhline(0, color=theme["axis"], linewidth=1)
            ax.set_xlabel("Hands observed (log)")
        kinds = [k for k in STATIC_KINDS if k != "equilibrium"]
        xs = [int(n) for n in sizes]
        for i, kind in enumerate(kinds):
            color = theme["ink"] if kind == CONTROL else s[i - 1]
            ys = [st[kind]["sizes"][n]["br"]["capture"]["mean"] for n in sizes]
            axes[0].plot(xs, ys, marker="o", color=color, label=KIND_LABELS[kind])
            ys = [st[kind]["sizes"][n]["rnr_0.9"]["capture"]["mean"] for n in sizes]
            axes[1].plot(xs, ys, marker="o", color=color)
        for ax, name in zip(axes, ("Best response", "Restricted Nash, p = 0.9"), strict=True):
            ax.set_ylabel("Share of the available gain")
            ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
            ax.set_title(name, color=theme["ink"], fontsize=10)
        legend(axes[0], theme, loc="upper center", bbox_to_anchor=(1.1, -0.17), ncol=4)
        title(
            fig,
            theme,
            "How much of the available gain a wrong model captures",
            "Exact values; the available gain is the oracle's, which knows the opponent exactly",
            top=0.84,
        )
        return fig

    def sessions(theme):
        kinds = list(se)
        heroes = [h for h in SESSION_HEROES if h != "equilibrium"]
        fig, ax = new_figure(theme, 1, 1, figsize=(10, 5))
        style_axes(ax, theme, grid_axis="x")
        s = theme["series"]
        height = 0.8 / len(heroes)
        ys = np.arange(len(kinds))
        for i, hero in enumerate(heroes):
            q = [se[k]["heroes"][hero]["gain"]["mean"] for k in kinds]
            ax.barh(ys + i * height, q, height=height, color=s[i % len(s)], label=HERO_LABELS[hero])
        ax.axvline(0, color=theme["axis"], linewidth=1)
        ax.set_yticks(ys + 0.4 - height / 2)
        ax.set_yticklabels([SESSION_LABELS[k] for k in kinds], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Gain over the equilibrium hero (chips/hand)")
        legend(ax, theme, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3)
        title(
            fig,
            theme,
            "Exploiting opponents that adapt",
            f"{cfg['session']['hands']:,}-hand sessions, exact value of the strategies in force",
            top=0.88,
        )
        return fig

    save_both(by_hands, out_dir, "misspec_by_hands")
    save_both(sessions, out_dir, "misspec_sessions")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true", help="use configs/phase2_quick.yaml")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    name = "phase2_quick.yaml" if args.quick else "phase2.yaml"
    cfg = load_config(REPO_ROOT / "experiments" / "configs" / name)
    out = args.out or RESULTS_ROOT / cfg["name"] / "misspec"
    workers = args.workers or cfg["misspec"].get("workers")
    results = run(args.seed, cfg=cfg["misspec"], workers=workers, out_dir=out)
    print(f"wrote {out} in {results['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()

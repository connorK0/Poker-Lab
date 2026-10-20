"""Phase 1 analyses for one seed (one independent replicate).

Everything that is *fit* uses training players only: empirical-Bayes priors, classifiers,
calibration temperatures (held-out training players), regressors, action models. Test players
are only scored. Within-seed uncertainty comes from a percentile bootstrap over test players;
the across-seed intervals are computed in ``phase1_report``.

Pre-registered vs. added later: the four classifier variants, global temperature scaling,
regression vs. the population mean, and action prediction were planned before any results.
Three analyses were added after the first full run and are labeled as such in the report:
per-sample-size temperature scaling (motivated by underconfidence at large samples), the
no-bet-sizing ablation (to measure how much identification rests on sizing tells), and
within-archetype R^2.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.phase1_data import SeedData
from poker_lab.bots import ARCHETYPES, PARAM_NAMES
from poker_lab.evaluation import accuracy, brier, confusion_matrix, log_loss, reliability
from poker_lab.evaluation.metrics import per_row_log_loss
from poker_lab.profiling import (
    SITUATION_FEATURES,
    ActionPredictor,
    ArchetypeClassifier,
    EmpiricalBayesShrinker,
    ParamRegressor,
    marginal_baseline,
    oos_r2,
    prefix_totals,
    profile_features,
    window_totals,
)

N_CLASSES = len(ARCHETYPES)
MAIN_MODELS = ("lr", "gbm")
VARIANTS = ("shrunk", "raw")
NO_SIZING = "lr_shrunk_nosizing"  # post-hoc ablation: logistic, shrunk stats, no sizing features
SIZING_COLUMNS = ("bet_size", "open_size")


def _cluster_bootstrap(values: np.ndarray, groups: np.ndarray, rng, n_boot: int = 1000):
    """95% percentile interval of the mean of per-group means (groups = test players)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    means = np.bincount(inv, weights=values) / np.bincount(inv)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    lo, hi = np.quantile(means[idx].mean(axis=1), [0.025, 0.975])
    return [float(lo), float(hi)]


def _class_metrics(y, probs: dict[str, np.ndarray], groups, rng) -> dict[str, Any]:
    """``probs`` maps calibration mode ('none', 'global', 'by_size') to probabilities."""
    proba = probs["global"]
    pred = proba.argmax(axis=1)
    cm = confusion_matrix(y, pred, N_CLASSES)
    correct = (pred == y).astype(float)
    return {
        "accuracy": accuracy(y, proba),
        "log_loss": log_loss(y, proba),
        "log_loss_uncal": log_loss(y, probs["none"]),
        "log_loss_by_size": log_loss(y, probs["by_size"]),
        "ece": reliability(y, proba).ece,
        "ece_uncal": reliability(y, probs["none"]).ece,
        "ece_by_size": reliability(y, probs["by_size"]).ece,
        "brier": brier(y, proba),
        "n_windows": len(y),
        "n_players": len(np.unique(groups)),
        "accuracy_ci_players": _cluster_bootstrap(correct, groups, rng),
        "log_loss_ci_players": _cluster_bootstrap(per_row_log_loss(y, proba), groups, rng),
        "confusion": cm.tolist(),
        "recall": (np.diag(cm) / np.maximum(cm.sum(axis=1), 1)).tolist(),
    }


def _reliability_dict(y, proba, n_bins) -> dict[str, list]:
    r = reliability(y, proba, n_bins)
    return {
        "confidence": r.confidence.tolist(),
        "accuracy": r.accuracy.tolist(),
        "count": r.count.tolist(),
        "ece": r.ece,
    }


def _without_sizing(X: np.ndarray, shrinker: EmpiricalBayesShrinker) -> np.ndarray:
    """Replace the sizing features with their prior means: the model cannot use them."""
    X = X.copy()
    for name in SIZING_COLUMNS:
        X[:, shrinker.names.index(name)] = shrinker.continuous[name].mean
    return X


def _r2_vs(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray) -> list[float]:
    """1 - SSE(prediction) / SSE(baseline prediction), per column."""
    return (1 - ((y_true - y_pred) ** 2).sum(0) / ((y_true - y_base) ** 2).sum(0)).tolist()


def analyze(data: SeedData, cfg: dict) -> dict[str, Any]:
    seed = data.seed
    rng = np.random.default_rng(seed + 1)
    tr, te = data.train, data.test
    tr_cum, te_cum = tr.cumulative, te.cumulative
    hands = tr.counters.shape[1]
    sizes = [n for n in cfg["sample_sizes"] if n <= hands]
    out: dict[str, Any] = {
        "seed": seed,
        "simulation": {
            "train_players": len(tr.player_ids),
            "test_players": len(te.player_ids),
            "hands_per_player": hands,
            "hands_simulated": tr.hands_simulated + te.hands_simulated,
            "seconds": tr.seconds + te.seconds,
            "train_decisions": len(tr.decisions),
            "test_decisions": len(te.decisions),
        },
    }
    assert not set(tr.player_ids) & set(te.player_ids), "player leakage between splits"

    # ---- 1. empirical-Bayes priors: training players' full histories only
    half = hands // 2
    halves = (tr_cum[:, half - 1], tr_cum[:, -1] - tr_cum[:, half - 1])
    shrinker = EmpiricalBayesShrinker.fit(tr_cum[:, -1], halves)
    out["priors"] = shrinker.to_dict()

    # ---- 2. training rows: disjoint windows at every sample size
    feats = {v: [] for v in VARIANTS}
    y_tr, g_tr, p_tr = [], [], []
    for n in sizes:
        w = window_totals(tr_cum, n, cfg["train_windows_max"])
        for v in VARIANTS:
            feats[v].append(profile_features(shrinker, w.totals, shrink=v == "shrunk"))
        y_tr.append(tr.labels[w.player])
        g_tr.append(w.player)
        p_tr.append(tr.params[w.player])
    X_tr = {v: np.concatenate(feats[v]) for v in VARIANTS}
    y_tr, g_tr, p_tr = np.concatenate(y_tr), np.concatenate(g_tr), np.concatenate(p_tr)
    out["training_rows"] = len(y_tr)

    # ---- 3. classifiers (+ no-sizing ablation, oracle, permutation control)
    clfs = {
        f"{m}_{v}": ArchetypeClassifier(m, calibrate=True, random_state=seed).fit(
            X_tr[v], y_tr, g_tr
        )
        for m in MAIN_MODELS
        for v in VARIANTS
    }
    clfs[NO_SIZING] = ArchetypeClassifier("lr", calibrate=True, random_state=seed).fit(
        _without_sizing(X_tr["shrunk"], shrinker), y_tr, g_tr
    )
    out["temperatures"] = {k: c.temperature for k, c in clfs.items()}
    out["temperatures_by_size"] = {
        k: None
        if c.size_temperatures is None
        else {
            str(round(float(np.exp(n)))): float(t)
            for n, t in zip(*c.size_temperatures, strict=True)
        }
        for k, c in clfs.items()
    }
    oracle = {
        m: ArchetypeClassifier(m, calibrate=True, random_state=seed, feature_space="identity").fit(
            tr.params, tr.labels, np.arange(len(tr.labels))
        )
        for m in MAIN_MODELS
    }
    out["oracle"] = {
        m: {
            "accuracy": accuracy(te.labels, c.predict_proba(te.params)),
            "log_loss": log_loss(te.labels, c.predict_proba(te.params)),
        }
        for m, c in oracle.items()
    }
    # Permutation control: shuffle labels across *training players*; any accuracy well above
    # chance on test players would indicate leakage through the features or the split.
    shuffled = tr.labels[rng.permutation(len(tr.labels))][g_tr]
    control = ArchetypeClassifier("gbm", calibrate=False, random_state=seed).fit(
        X_tr["shrunk"], shuffled, g_tr
    )
    n_ctrl = cfg["permutation_n"]
    w = window_totals(te_cum, n_ctrl, cfg["test_windows_max"])
    p_ctrl = control.predict_proba(profile_features(shrinker, w.totals), calibration="none")
    out["permutation_control"] = {
        "n": n_ctrl,
        "accuracy": accuracy(te.labels[w.player], p_ctrl),
        "chance": 1.0 / N_CLASSES,
    }

    # ---- 4. test evaluation per sample size
    out["classification"] = {k: {} for k in clfs}
    out["reliability"] = {mode: {k: {} for k in clfs} for mode in ("none", "global", "by_size")}
    test_probs: dict[int, np.ndarray] = {}
    test_windows = {}
    for n in sizes:
        w = window_totals(te_cum, n, cfg["test_windows_max"])
        test_windows[n] = w
        y = te.labels[w.player]
        for key, clf in clfs.items():
            X = profile_features(shrinker, w.totals, shrink=not key.endswith("_raw"))
            if key == NO_SIZING:
                X = _without_sizing(X, shrinker)
            probs = {
                mode: clf.predict_proba(X, calibration=mode)
                for mode in ("none", "global", "by_size")
            }
            out["classification"][key][str(n)] = _class_metrics(y, probs, w.player, rng)
            if n in cfg["reliability_sizes"]:
                for mode, p in probs.items():
                    out["reliability"][mode][key][str(n)] = _reliability_dict(
                        y, p, cfg["calib_bins"]
                    )
            if key == cfg["primary_model"]:
                test_probs[n] = probs["global"]

    # ---- 5. parameter regression (+ "classify, then use the archetype's mean params")
    regs = {
        k: ParamRegressor(k, random_state=seed).fit(X_tr["shrunk"], p_tr) for k in ("gbm", "ridge")
    }
    type_means = np.stack([tr.params[tr.labels == k].mean(axis=0) for k in range(N_CLASSES)])
    train_mean = tr.params.mean(axis=0)
    out["regression"] = {"param_names": list(PARAM_NAMES)}
    for k in ("gbm", "ridge", "type_mean"):
        out["regression"][k] = {}
        out["regression"][f"{k}_within_type"] = {}
    for n in sizes:
        w = test_windows[n]
        y_true = te.params[w.player]
        X = profile_features(shrinker, w.totals)
        preds = {k: reg.predict(X) for k, reg in regs.items()}
        preds["type_mean"] = test_probs[n] @ type_means
        true_type_mean = type_means[te.labels[w.player]]  # knowing the archetype exactly
        for k, pred in preds.items():
            out["regression"][k][str(n)] = oos_r2(y_true, pred, train_mean).tolist()
            out["regression"][f"{k}_within_type"][str(n)] = _r2_vs(y_true, pred, true_type_mean)

    # ---- 6. action prediction from profiles built on *earlier* hands only
    out["action"] = _action_prediction(data, shrinker, cfg, rng)
    return out


def _windowed(decisions: np.ndarray, checkpoints, window) -> tuple[np.ndarray, np.ndarray]:
    """Row indices and checkpoint index for decisions in each window ``[c, c + window)``.
    Windows may overlap; a decision in two windows is used once per window, each time with
    the profile of that window's checkpoint."""
    seq = decisions[:, 1].astype(int)
    rows, which = [], []
    for i, c in enumerate(checkpoints):
        idx = np.nonzero((seq >= c) & (seq < c + window))[0]
        rows.append(idx)
        which.append(np.full(len(idx), i))
    return np.concatenate(rows), np.concatenate(which)


def _action_features(pool, decisions, which, shrinker, checkpoints):
    cum = pool.cumulative
    player = decisions[:, 0].astype(int)
    situation = decisions[:, 2 : 2 + len(SITUATION_FEATURES)]
    labels = decisions[:, -1].astype(int)
    profiles = np.stack(
        [profile_features(shrinker, prefix_totals(cum, c)) for c in checkpoints]
    )  # (checkpoints, players, features): built from hands before the window starts
    prof = profiles[which, player]
    onehot = np.eye(N_CLASSES)[pool.labels[player]]
    return (
        {
            "situation": situation,
            "profile": np.hstack([situation, prof]),
            "type_oracle": np.hstack([situation, onehot]),
            "param_oracle": np.hstack([situation, pool.params[player]]),
        },
        labels,
        player,
    )


def _action_prediction(data: SeedData, shrinker, cfg: dict, rng) -> dict[str, Any]:
    checkpoints, window = cfg["action_checkpoints"], cfg["action_window"]
    tr, te = data.train, data.test
    rows_tr, which_tr = _windowed(tr.decisions, checkpoints, window)
    rows_te, which_te = _windowed(te.decisions, checkpoints, window)
    X_tr, y_tr, _ = _action_features(tr, tr.decisions[rows_tr], which_tr, shrinker, checkpoints)
    X_te, y_te, player_te = _action_features(
        te, te.decisions[rows_te], which_te, shrinker, checkpoints
    )
    n_rows = min(cfg["action_train_rows"], len(y_tr))
    sub = rng.choice(len(y_tr), size=n_rows, replace=False)
    models = {
        name: ActionPredictor(random_state=data.seed).fit(X[sub], y_tr[sub])
        for name, X in X_tr.items()
    }
    out: dict[str, Any] = {"train_rows_used": int(n_rows), "window": window, "checkpoints": {}}
    for i, c in enumerate(checkpoints):
        mask = which_te == i
        y = y_te[mask]
        probs = {name: models[name].predict_proba(X_te[name][mask]) for name in models}
        probs["marginal"] = marginal_baseline(y_tr, int(mask.sum()))
        row_ll = {name: per_row_log_loss(y, p) for name, p in probs.items()}
        delta = row_ll["situation"] - row_ll["profile"]  # > 0: the profile helps
        out["checkpoints"][str(c)] = {
            "hands_observed": c,
            "predicted_hands": [c, c + window],
            "n_decisions": int(mask.sum()),
            "n_players": len(np.unique(player_te[mask])),
            "log_loss": {name: float(v.mean()) for name, v in row_ll.items()},
            "accuracy": {name: accuracy(y, p) for name, p in probs.items()},
            "delta_vs_situation": float(delta.mean()),
            "delta_ci_players": _cluster_bootstrap(delta, player_te[mask], rng),
        }
    return out

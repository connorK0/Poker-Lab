"""Phase 1 analyses for one seed (one independent replicate).

Everything that is *fit* uses training players only: empirical-Bayes priors, classifiers,
calibration temperatures (held-out training players), regressors, action models. Test players
are only scored. Within-seed uncertainty comes from a percentile bootstrap over test players;
the across-seed intervals are computed in ``phase1_report``.
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


def _cluster_bootstrap(values: np.ndarray, groups: np.ndarray, rng, n_boot: int = 1000):
    """95% percentile interval of the mean of per-group means (groups = test players)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    means = sums / counts
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boots = means[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return [float(lo), float(hi)]


def _class_metrics(y, proba, proba_uncal, groups, rng) -> dict[str, Any]:
    pred = proba.argmax(axis=1)
    cm = confusion_matrix(y, pred, N_CLASSES)
    correct = (pred == y).astype(float)
    row_ll = per_row_log_loss(y, proba)
    return {
        "accuracy": accuracy(y, proba),
        "log_loss": log_loss(y, proba),
        "log_loss_uncal": log_loss(y, proba_uncal),
        "ece": reliability(y, proba).ece,
        "ece_uncal": reliability(y, proba_uncal).ece,
        "brier": brier(y, proba),
        "n_windows": len(y),
        "n_players": len(np.unique(groups)),
        "accuracy_ci_players": _cluster_bootstrap(correct, groups, rng),
        "log_loss_ci_players": _cluster_bootstrap(row_ll, groups, rng),
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

    # ---- 3. classifiers (+ oracle on true parameters, + permutation control)
    clfs = {
        (m, v): ArchetypeClassifier(m, calibrate=True, random_state=seed).fit(X_tr[v], y_tr, g_tr)
        for m in MAIN_MODELS
        for v in VARIANTS
    }
    out["temperatures"] = {f"{m}_{v}": c.temperature for (m, v), c in clfs.items()}
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
    p_ctrl = control.predict_proba(profile_features(shrinker, w.totals))
    out["permutation_control"] = {
        "n": n_ctrl,
        "accuracy": accuracy(te.labels[w.player], p_ctrl),
        "chance": 1.0 / N_CLASSES,
    }

    # ---- 4. test evaluation per sample size
    out["classification"] = {f"{m}_{v}": {} for m, v in clfs}
    out["reliability"] = {f"{m}_{v}": {} for m, v in clfs}
    out["reliability_uncal"] = {f"{m}_{v}": {} for m, v in clfs}
    test_probs: dict[int, np.ndarray] = {}
    test_windows = {}
    for n in sizes:
        w = window_totals(te_cum, n, cfg["test_windows_max"])
        test_windows[n] = w
        y = te.labels[w.player]
        for (m, v), clf in clfs.items():
            X = profile_features(shrinker, w.totals, shrink=v == "shrunk")
            proba = clf.predict_proba(X)
            proba_uncal = clf.predict_proba(X, calibrated=False)
            key = f"{m}_{v}"
            out["classification"][key][str(n)] = _class_metrics(
                y, proba, proba_uncal, w.player, rng
            )
            if n in cfg["reliability_sizes"]:
                out["reliability"][key][str(n)] = _reliability_dict(y, proba, cfg["calib_bins"])
                out["reliability_uncal"][key][str(n)] = _reliability_dict(
                    y, proba_uncal, cfg["calib_bins"]
                )
            if key == cfg["primary_model"]:
                test_probs[n] = proba

    # ---- 5. parameter regression (+ "classify, then use the archetype's mean params")
    regs = {
        k: ParamRegressor(k, random_state=seed).fit(X_tr["shrunk"], p_tr) for k in ("gbm", "ridge")
    }
    type_means = np.stack([tr.params[tr.labels == k].mean(axis=0) for k in range(N_CLASSES)])
    train_mean = tr.params.mean(axis=0)
    out["regression"] = {"param_names": list(PARAM_NAMES), "gbm": {}, "ridge": {}, "type_mean": {}}
    for n in sizes:
        w = test_windows[n]
        y_true = te.params[w.player]
        X = profile_features(shrinker, w.totals)
        for k, reg in regs.items():
            out["regression"][k][str(n)] = oos_r2(y_true, reg.predict(X), train_mean).tolist()
        out["regression"]["type_mean"][str(n)] = oos_r2(
            y_true, test_probs[n] @ type_means, train_mean
        ).tolist()

    # ---- 6. action prediction from profiles built on *earlier* hands only
    out["action"] = _action_prediction(data, shrinker, cfg, rng)
    return out


def _decision_split(decisions: np.ndarray, checkpoints, window):
    seq = decisions[:, 1].astype(int)
    which = np.full(len(seq), -1)
    for i, c in enumerate(checkpoints):
        which[(seq >= c) & (seq < c + window)] = i
    if (which < 0).any():
        raise AssertionError("decision outside every prediction window")
    return which


def _action_features(pool, cum, decisions, which, shrinker, checkpoints):
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
    which_tr = _decision_split(tr.decisions, checkpoints, window)
    which_te = _decision_split(te.decisions, checkpoints, window)
    X_tr, y_tr, _ = _action_features(
        tr, tr.cumulative, tr.decisions, which_tr, shrinker, checkpoints
    )
    X_te, y_te, player_te = _action_features(
        te, te.cumulative, te.decisions, which_te, shrinker, checkpoints
    )
    n_rows = min(cfg["action_train_rows"], len(y_tr))
    sub = rng.choice(len(y_tr), size=n_rows, replace=False)
    models = {
        name: ActionPredictor(random_state=data.seed).fit(X[sub], y_tr[sub])
        for name, X in X_tr.items()
    }
    out: dict[str, Any] = {"train_rows_used": int(n_rows), "checkpoints": {}}
    for i, c in enumerate(checkpoints):
        mask = which_te == i
        y = y_te[mask]
        probs = {name: models[name].predict_proba(X_te[name][mask]) for name in models}
        probs["marginal"] = marginal_baseline(y_tr, int(mask.sum()))
        row_ll = {name: per_row_log_loss(y, p) for name, p in probs.items()}
        delta = row_ll["situation"] - row_ll["profile"]  # > 0: the profile helps
        out["checkpoints"][str(c)] = {
            "hands_observed": c,
            "n_decisions": int(mask.sum()),
            "n_players": len(np.unique(player_te[mask])),
            "log_loss": {name: float(v.mean()) for name, v in row_ll.items()},
            "accuracy": {name: accuracy(y, p) for name, p in probs.items()},
            "delta_vs_situation": float(delta.mean()),
            "delta_ci_players": _cluster_bootstrap(delta, player_te[mask], rng),
        }
    return out

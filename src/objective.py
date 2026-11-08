"""Week 3: evaluate a candidate on fixed training OOF predictions only."""

from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from .baseline import apply_threshold, compute_cost_metrics
from .preprocessing import build_preprocessor


def constraint_objective(metrics: dict, config: dict) -> dict:
    """Keep raw cost f separate from the feasible-first penalty J."""
    cost = metrics["cost_per_sample"]
    g = metrics["rejection_rate"] - config["policy"]["max_rejection_rate"]
    feasible = g <= 0
    c_max = max(config["cost"].values())
    objective = cost if feasible else c_max + cost + config["policy"]["penalty_multiplier"] * g
    return {"constraint_g": g, "constraint_violation": max(0.0, g),
            "feasible": feasible, "objective": objective}


def suggest_params(trial, config: dict) -> dict:
    params = {}
    for name, bounds in config["optimization"]["search_space"].items():
        suggest = trial.suggest_int if bounds["type"] == "int" else trial.suggest_float
        params[name] = suggest(name, bounds["low"], bounds["high"])
    return params


def candidate_key(record: dict) -> tuple:
    """Within one seed: lower raw cost, fewer trees, shallower depth, earlier trial."""
    return (record["metrics"]["cost_per_sample"], record["params"]["n_estimators"],
            record["params"]["max_depth"], record["trial_number"])


def evaluate_candidate(params: dict, X, y, splits: list, config: dict) -> dict:
    started = perf_counter()
    if not X.index.equals(y.index):
        raise ValueError("Feature and label row order must match.")
    oof = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)
    counts = np.zeros(len(y), dtype=int)
    fold_metrics = []
    # Validate the entire CV partition before fitting anything.
    for train_pos, val_pos in splits:
        if (not len(train_pos) or not len(val_pos)
                or np.intersect1d(train_pos, val_pos).size
                or len(np.unique(val_pos)) != len(val_pos)
                or len(np.unique(train_pos)) != len(train_pos)
                or not np.array_equal(np.sort(np.r_[train_pos, val_pos]), np.arange(len(y)))):
            raise ValueError("Each fold must partition train rows without overlap.")
        counts[val_pos] += 1
    if not np.all(counts == 1):
        raise ValueError("Each training row must receive exactly one OOF prediction.")
    for fold, (train_pos, val_pos) in enumerate(splits):
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                class_weight={0: 1.0, 1: params["bad_class_weight"]},
                random_state=config["model_seed"], n_jobs=1,
            )),
        ])
        pipe.fit(X.iloc[train_pos], y.iloc[train_pos])
        bad_col = list(pipe.classes_).index(1)
        oof[val_pos] = pipe.predict_proba(X.iloc[val_pos])[:, bad_col]
        fold_ids[val_pos] = fold
        fold_metrics.append({"fold": fold, **compute_cost_metrics(
            y.iloc[val_pos].to_numpy(), apply_threshold(oof[val_pos], params["threshold"]),
            config["cost"]["false_negative"], config["cost"]["false_positive"])})
    if not np.isfinite(oof).all():
        raise ValueError("OOF probabilities must be finite for every row.")
    pred = apply_threshold(oof, params["threshold"])
    metrics = compute_cost_metrics(y.to_numpy(), pred, config["cost"]["false_negative"],
                                   config["cost"]["false_positive"])
    metrics.update({
        "accuracy": float(accuracy_score(y, pred)),
        "precision_bad": float(precision_score(y, pred, zero_division=0)),
        "recall_bad": float(recall_score(y, pred, zero_division=0)),
        "f1_bad": float(f1_score(y, pred, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
    })
    return {"metrics": metrics, **constraint_objective(metrics, config),
            "fold_details": fold_metrics, "fit_count": len(splits),
            "evaluation_seconds": perf_counter() - started,
            "oof_proba": oof, "fold_ids": fold_ids}

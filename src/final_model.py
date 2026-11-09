"""Fit-ready pipeline and inference using the frozen classification threshold."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def build_final_pipeline(selection):
    chosen = selection["selected"]
    if chosen["method"] == "baseline_threshold":
        params = {}
    else:
        p = chosen["params"]
        params = {"n_estimators": p["n_estimators"], "max_depth": p["max_depth"],
                  "min_samples_split": p["min_samples_split"],
                  "class_weight": {0: 1.0, 1: p["bad_class_weight"]}}
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(**params, random_state=selection["model_seed"], n_jobs=1)),
    ])


def predict_with_threshold(bundle, X):
    """Return P(bad) and 0=approve/1=reject, using the selected threshold (not .5)."""
    if X.columns.tolist() != bundle["feature_columns"]:
        raise ValueError("Feature names/order must match the training schema; exclude row ID and target.")
    pipe = bundle["pipeline"]
    bad_column = list(pipe.classes_).index(1)
    probabilities = pipe.predict_proba(X)[:, bad_column]
    return probabilities, (probabilities >= bundle["threshold"]).astype(np.int64)

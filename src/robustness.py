"""Exploratory nested CV on original training data only; original test is blocked.

python -m src.robustness
python -m src.robustness --resume results/robustness/run-<UTC>
"""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .artifacts import write_csv, write_json
from .baseline import apply_threshold, compute_cost_metrics
from .config import CONFIG_PATH, DATA_DIR, PROJECT_ROOT, load_config
from .data_utils import _environment_versions, file_sha256, load_training_data
from .evaluate_final import read_json, score_predictions
from .final_model import build_final_pipeline, predict_with_threshold
from .preprocessing import build_preprocessor
from .resumable_search import run_search
from .robustness_analysis import summarize, plot_results

PROTOCOL_PATH = PROJECT_ROOT / "configs" / "robustness.json"


def block_test(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        if Path(os.fsdecode(args[0])).name == "credit_test.csv":
            raise PermissionError("Original test is locked: supplemental nested CV uses train only.")


def cap_label(cap):
    return f"cap{round(cap * 100)}"


def default_pipeline(config):
    return Pipeline([("preprocessor", build_preprocessor()),
                     ("classifier", RandomForestClassifier(random_state=config["model_seed"], n_jobs=1))])


def choose_threshold(y, probabilities, cap, config, grid):
    """Choose on inner OOF only; minimum cost then lower threshold, as in Week 2."""
    rows = []
    thresholds = np.round(np.arange(grid["low"], grid["high"] + grid["step"] / 2, grid["step"]), 10)
    for threshold in thresholds:
        metrics = compute_cost_metrics(y, apply_threshold(probabilities, threshold),
                                       config["cost"]["false_negative"], config["cost"]["false_positive"])
        rows.append({"threshold": float(threshold), **metrics,
                     "feasible": metrics["rejection_rate"] <= cap})
    feasible = [r for r in rows if r["feasible"]]
    return (min(feasible, key=lambda r: (r["cost_per_sample"], r["threshold"])) if feasible else None), rows


def make_splits(X, y, protocol):
    outer = StratifiedKFold(protocol["outer_folds"], shuffle=True, random_state=protocol["outer_seed"])
    partitions, manifest = [], []
    for fold, (train, validation) in enumerate(outer.split(X, y)):
        inner = list(StratifiedKFold(protocol["inner_folds"], shuffle=True,
                                    random_state=protocol["inner_seed"]).split(X.iloc[train], y.iloc[train]))
        partitions.append((train, validation, inner))
        ids = X.iloc[train].index
        manifest.append({"outer_fold": fold, "train_row_ids": ids.tolist(),
                         "validation_row_ids": X.iloc[validation].index.tolist(),
                         "inner": [{"train_row_ids": ids[t].tolist(), "validation_row_ids": ids[v].tolist()}
                                   for t, v in inner]})
    return partitions, manifest


def run_fold(fold, X_train, y_train, X_outer, y_outer, splits, config, protocol, folder):
    """All threshold/hyperparameter choices are frozen before outer prediction/scoring."""
    marker = folder / "fold_summary.json"
    if marker.exists():
        saved = read_json(marker)
        for name, digest in saved["output_sha256"].items():
            if file_sha256(folder / name) != digest:
                raise ValueError(f"Completed outer fold artifact changed: {folder / name}")
        return saved["rows"], pd.read_csv(folder / "predictions.csv")
    folder.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    default = default_pipeline(config)
    inner_proba = cross_val_predict(default, X_train, y_train, cv=splits, method="predict_proba", n_jobs=1)[:, 1]
    write_csv(folder / "baseline_inner_oof.csv", pd.DataFrame({"source_row_id": X_train.index,
              "target": y_train.to_numpy(), "probability_bad": inner_proba}))
    choices = {"baseline_default": {"threshold": .5, "inner_cap": None, "params": None}}
    for cap in protocol["inner_rejection_caps"]:
        best, sweep = choose_threshold(y_train.to_numpy(), inner_proba, cap, config, protocol["threshold_grid"])
        write_csv(folder / f"baseline_{cap_label(cap)}_sweep.csv", pd.DataFrame(sweep))
        choices[f"baseline_{cap_label(cap)}"] = {"inner_cap": cap, "params": None,
            "threshold": best["threshold"] if best else None, "inner_selection": best}
        scenario = deepcopy(config)
        scenario["policy"]["max_rejection_rate"] = cap
        scenario["cv"]["n_splits"] = protocol["inner_folds"]
        scenario["cv"]["seed"] = protocol["inner_seed"]
        search_dir = folder / f"tpe_{cap_label(cap)}"
        with (folder / f"tpe_{cap_label(cap)}_console.txt").open("a", encoding="utf-8", newline="\n") as log:
            with redirect_stdout(log):
                summary, _ = run_search("tpe", protocol["sampler_seed"], protocol["trials_per_search"],
                                        X_train, y_train, splits, scenario, search_dir)
        best = summary["best_feasible"]
        choices[f"tpe_{cap_label(cap)}"] = {"inner_cap": cap, "params": best["params"] if best else None,
            "threshold": best["params"]["threshold"] if best else None, "inner_selection": best}
        print(f"Outer fold {fold + 1}: completed TPE cap={cap:.0%}, {summary['completed_trials']} trials", flush=True)
    write_json(folder / "inner_choices.json", choices)
    # Outer features and labels have not been used by any choice above.
    default.fit(X_train, y_train)
    default_prob = default.predict_proba(X_outer)[:, list(default.classes_).index(1)]
    rows, prediction_frames = [], []
    refits = 1
    for method, choice in choices.items():
        row = {"outer_fold": fold, "method": method, "inner_cap": choice["inner_cap"],
               "threshold": choice["threshold"], "n_outer": len(y_outer),
               "selection_status": "no_feasible_inner_candidate" if choice["threshold"] is None else "selected"}
        if choice["threshold"] is not None:
            if method.startswith("baseline"):
                probabilities = default_prob
                predictions = apply_threshold(probabilities, choice["threshold"])
            else:
                pipe = build_final_pipeline({"selected": {"method": "tpe", "params": choice["params"]},
                                             "model_seed": config["model_seed"]})
                pipe.fit(X_train, y_train)
                refits += 1
                probabilities, predictions = predict_with_threshold({"pipeline": pipe, "threshold": choice["threshold"],
                    "feature_columns": X_train.columns.tolist()}, X_outer)
            row.update(score_predictions(y_outer, predictions, config))  # Always measured against policy 35%.
            prediction_frames.append(pd.DataFrame({"outer_fold": fold, "method": method,
                "source_row_id": X_outer.index, "target": y_outer.to_numpy(), "probability_bad": probabilities,
                "prediction": predictions}))
        rows.append(row)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    write_csv(folder / "predictions.csv", predictions)
    outputs = {p.relative_to(folder).as_posix(): file_sha256(p) for p in folder.rglob("*")
               if p.is_file() and p.name not in {"search.checkpoint.json", marker.name}}
    write_json(marker, {"rows": rows, "inner_baseline_fit_count": len(splits), "outer_refit_count": refits,
                       "elapsed_seconds": perf_counter() - started, "output_sha256": outputs})
    return rows, predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    config, protocol = load_config(), read_json(PROTOCOL_PATH)
    if config["policy"]["max_rejection_rate"] != protocol["evaluation_rejection_cap"]:
        raise ValueError("Keep the original evaluation policy.")
    X, y = load_training_data()
    partitions, manifest = make_splits(X, y, protocol)
    base = PROJECT_ROOT / "results" / "robustness"
    output = (PROJECT_ROOT / args.resume).resolve() if args.resume else base / ("run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if not output.resolve().is_relative_to(base.resolve()):
        parser.error("Resume path must be inside results/robustness.")
    identity = {"protocol": protocol, "protocol_sha256": file_sha256(PROTOCOL_PATH),
                "config_sha256": file_sha256(CONFIG_PATH), "environment": _environment_versions(),
                "input_sha256": {n: file_sha256(DATA_DIR / n) for n in ("credit_train.csv", "dataset_metadata.json")},
                "code_sha256": {n: file_sha256(PROJECT_ROOT / "src" / n) for n in
                    ("robustness.py", "robustness_analysis.py", "resumable_search.py", "objective.py", "optimize.py",
                     "baseline.py", "preprocessing.py", "final_model.py", "evaluate_final.py", "artifacts.py", "data_utils.py", "config.py")}}
    if args.resume:
        metadata = read_json(output / "run_metadata.json")
        if metadata["identity"] != identity:
            raise ValueError("Resume identity changed: code/config/environment/data must match.")
        if metadata["status"] == "complete":
            for name, digest in metadata["output_sha256"].items():
                if file_sha256(output / name) != digest:
                    raise ValueError(f"Output checksum mismatch: {name}")
            print(f"Already complete; verified saved results: {output}")
            return
    else:
        output.mkdir(parents=True, exist_ok=False)
        metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
                    "status": "running", "test_read": False,
                    "expected_searches": len(partitions) * len(protocol["inner_rejection_caps"]),
                    "expected_search_fits": len(partitions) * len(protocol["inner_rejection_caps"]) * protocol["trials_per_search"] * protocol["inner_folds"]}
        write_json(output / "splits.json", {"folds": manifest})
    metadata["status"] = "running"
    write_json(output / "run_metadata.json", metadata)
    print(f"Output: {output}", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        rows, predictions = [], []
        for fold, (train, val, inner) in enumerate(partitions):
            r, p = run_fold(fold, X.iloc[train], y.iloc[train], X.iloc[val], y.iloc[val], inner,
                            config, protocol, output / f"outer{fold}")
            rows.extend(r)
            predictions.append(p)
            print(f"Outer fold {fold + 1}/{len(partitions)} saved", flush=True)
        predictions = pd.concat(predictions, ignore_index=True)
        aggregate = summarize(rows, predictions, X.index.tolist(), config, protocol)
        write_csv(output / "per_fold.csv", pd.DataFrame(rows))
        write_csv(output / "outer_predictions.csv", predictions)
        write_csv(output / "aggregate.csv", aggregate)
        plot_results(aggregate, pd.DataFrame(rows), output)
        metadata.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        output_sha256={p.relative_to(output).as_posix(): file_sha256(p) for p in output.rglob("*")
                                       if p.is_file() and p.name not in {"search.checkpoint.json", "run_metadata.json"}})
        print(aggregate[["method", "pooled_cost", "pooled_rejection_rate", "pooled_feasible", "outer_folds_policy_passed"]].to_string(index=False))
    except BaseException as error:
        metadata.update(status="interrupted_or_failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run_metadata.json", metadata)
        raise
    write_json(output / "run_metadata.json", metadata)
    print(f"Finished exploratory nested CV: {output}. Original test was blocked.")


if __name__ == "__main__":
    sys.addaudithook(block_test)
    main()

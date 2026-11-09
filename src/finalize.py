"""Week 5 parts 1–2: freeze CV selection, then fit once on training data only.

python -m src.finalize select
python -m src.finalize fit
Neither command reads the test CSV or computes a test metric.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
from time import perf_counter, sleep

import joblib
import numpy as np
import pandas as pd

from .artifacts import write_csv, write_json
from .baseline import apply_threshold, compute_cost_metrics
from .config import CONFIG_PATH, DATA_DIR, PROJECT_ROOT, load_config
from .data_utils import _environment_versions, file_sha256, load_training_data, load_cv_splits
from .final_model import build_final_pipeline, predict_with_threshold
from .optimize import read_baselines
from .week4 import read_json, validate_reuse, validate_search

DEFAULT_SOURCE = "results/week4/run-20261108T104055472068Z"
METHOD_ORDER = ("baseline_threshold", "random", "tpe")
TOLERANCE = 1e-12


def choose_method_and_candidate(searches, baselines, config):
    """Mean across all planned seeds; select one feasible trial in the chosen method."""
    seeds = config["optimization"]["sampler_seeds"]
    candidates, comparison = {}, []
    for baseline in baselines:
        if baseline["method"] == "baseline_threshold":
            eligible = baseline["feasible"]
            comparison.append({"method": "baseline_threshold", "eligible": eligible,
                               "n_seeds": 1, "n_feasible_seeds": int(eligible),
                               "mean_oof_cost": baseline["cost_per_sample"] if eligible else None,
                               "std_oof_cost": None})
            candidates["baseline_threshold"] = {"method": "baseline_threshold", "sampler_seed": None,
                "trial_number": None, "params": {"threshold": baseline["threshold"]},
                "metrics": {k: v for k, v in baseline.items() if k not in ("method", "threshold")}}
    for method in ("random", "tpe"):
        group = [s for s in searches if s["method"] == method]
        if len(group) != len(seeds) or {s["sampler_seed"] for s in group} != set(seeds):
            raise ValueError(f"Exactly one completed search per planned seed is required: {method}")
        best = [s["best_feasible"] for s in group if s["best_feasible"] is not None]
        eligible = len(best) == len(seeds)
        costs = [b["metrics"]["cost_per_sample"] for b in best]
        comparison.append({"method": method, "eligible": eligible, "n_seeds": len(seeds),
                           "n_feasible_seeds": len(best),
                           "mean_oof_cost": float(np.mean(costs)) if eligible else None,
                           "std_oof_cost": float(np.std(costs, ddof=1)) if eligible and len(costs) > 1 else None})
        if eligible:
            minimum = min(costs)
            tied = [b for b in best if abs(b["metrics"]["cost_per_sample"] - minimum) <= TOLERANCE]
            candidates[method] = min(tied, key=lambda b: (b["params"]["n_estimators"],
                b["params"]["max_depth"], seeds.index(b["sampler_seed"]), b["trial_number"]))
    valid = [r for r in comparison if r["eligible"]]
    if not valid:
        raise ValueError("No method eligible for final selection. Do not relax constraints using test.")
    minimum = min(r["mean_oof_cost"] for r in valid)
    tied = [r for r in valid if abs(r["mean_oof_cost"] - minimum) <= TOLERANCE]
    winner = min(tied, key=lambda r: METHOD_ORDER.index(r["method"]))
    return candidates[winner["method"]], comparison


def verify_oof(path, best, X, y, splits, config):
    frame = pd.read_csv(path, converters={"probability_bad": float})
    folds = np.full(len(y), -1, dtype=int)
    for i, (_, val) in enumerate(splits):
        folds[val] = i
    if (frame.source_row_id.tolist() != X.index.tolist() or frame.target.tolist() != y.tolist()
            or frame.fold.tolist() != folds.tolist()
            or not np.isfinite(frame.probability_bad).all() or not frame.probability_bad.between(0, 1).all()):
        raise ValueError("Selected OOF identities/probabilities do not match training data.")
    pred = apply_threshold(frame.probability_bad.to_numpy(), best["params"]["threshold"])
    actual = compute_cost_metrics(y, pred, config["cost"]["false_negative"], config["cost"]["false_positive"])
    for key in ("TN", "FP", "FN", "TP", "cost_per_sample", "rejection_rate"):
        if not math.isclose(actual[key], best["metrics"][key], abs_tol=TOLERANCE, rel_tol=0):
            raise ValueError(f"Selected OOF and reported {key} differ.")
    if actual["rejection_rate"] > config["policy"]["max_rejection_rate"]:
        raise ValueError("Selected OOF candidate violates the main policy.")


def freeze_selection(source):
    output = PROJECT_ROOT / "results" / "week5"
    if (output / "selection.json").exists():
        raise FileExistsError("selection.json already frozen. Inspect it; no automatic overwrite.")
    config = load_config()
    X, y = load_training_data()
    splits = load_cv_splits(X)
    m = read_json(source / "run_metadata.json")
    if (m["status"] != "complete" or m["config"] != config or m["environment"] != _environment_versions()
            or m["config_sha256"] != file_sha256(CONFIG_PATH)):
        raise ValueError("Week 4 inputs/environment/config are incompatible or incomplete.")
    # Read only explicitly allowed training inputs, never the test CSV.
    for name in ("credit_train.csv", "cv_folds.json", "dataset_metadata.json"):
        if m["input_sha256"][name] != file_sha256(DATA_DIR / name):
            raise ValueError(f"Input checksum mismatch: {name}")
    for name in ("objective.py", "preprocessing.py", "baseline.py", "config.py", "data_utils.py"):
        if m["code_sha256"][name] != file_sha256(PROJECT_ROOT / "src" / name):
            raise ValueError(f"Evaluation code differs from the experiment: {name}")
    for name, expected in m["output_sha256"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or file_sha256(path) != expected:
            raise ValueError(f"Week 4 artifact checksum mismatch: {name}")
    reused = (PROJECT_ROOT / m["reuse_path"]).resolve()
    if not reused.is_relative_to(PROJECT_ROOT.resolve()) or file_sha256(reused / "run_metadata.json") != m["reuse_metadata_sha256"]:
        raise ValueError("Reused seed 42 provenance changed.")
    validate_reuse(reused, config)
    summaries, folders = [], {}
    for method in ("random", "tpe"):
        for seed in config["optimization"]["sampler_seeds"]:
            folder = (reused if seed == 42 else source / "main") / f"{method}_seed{seed}"
            summary, rows = validate_search(folder, method, seed, config)
            best = summary["best_feasible"]
            if best:
                if best != read_json(folder / "best_feasible.json"):
                    raise ValueError("Selected candidate and summary disagree.")
                trial = rows[best["trial_number"]]
                if any(best["params"][k] != trial[k] for k in config["optimization"]["search_space"]):
                    raise ValueError("Candidate params differ from trial history.")
                verify_oof(folder / "best_oof.csv", best, X, y, splits, config)
            summaries.append(summary)
            folders[method, seed] = folder
    baselines, _ = read_baselines()
    selected, comparison = choose_method_and_candidate(summaries, baselines, config)
    if selected["method"] == "baseline_threshold":
        oof_path = PROJECT_ROOT / "results" / "baseline_oof.csv"
        baseline_meta = read_json(PROJECT_ROOT / "results" / "baseline_run_metadata.json")
        if file_sha256(oof_path) != baseline_meta["result_sha256"][oof_path.name]:
            raise ValueError("Baseline OOF checksum mismatch.")
        verify_oof(oof_path, selected, X, y, splits, config)
    else:
        oof_path = folders[selected["method"], selected["sampler_seed"]] / "best_oof.csv"
    decision = {
        "format_version": 1, "status": "frozen_before_test", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": {"method_metric": "Lowest mean of best feasible OOF cost over all 5 planned sampler seeds; baseline threshold uses its single deterministic OOF cost.",
                 "eligibility": "Search methods must have a feasible candidate in every planned seed.",
                 "method_tie_order": METHOD_ORDER, "numeric_tolerance": TOLERANCE,
                 "candidate_tie_order": ["cost", "n_estimators", "max_depth", "sampler_seed_config_order", "trial_number"],
                 "timing": "Method comparison rule explicitly recorded at week 5 after viewing CV, before any test evaluation; not a preregistered blind comparison."},
        "source_run": source.relative_to(PROJECT_ROOT).as_posix(), "source_metadata_sha256": file_sha256(source / "run_metadata.json"),
        "candidate_oof_path": oof_path.relative_to(PROJECT_ROOT).as_posix(), "candidate_oof_sha256": file_sha256(oof_path),
        "comparison": comparison, "selected": selected, "model_seed": config["model_seed"],
        "config": config, "config_sha256": file_sha256(CONFIG_PATH), "environment": _environment_versions(),
        "feature_columns": X.columns.tolist(), "n_train": len(y),
        "train_sha256": file_sha256(DATA_DIR / "credit_train.csv"),
        "cv_sha256": file_sha256(DATA_DIR / "cv_folds.json"),
        "implementation_sha256": {n: file_sha256(PROJECT_ROOT / "src" / n) for n in ("finalize.py", "final_model.py", "preprocessing.py")},
        "test_evaluated": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "method_comparison.csv", pd.DataFrame(comparison))
    write_json(output / "selection.json", decision)
    print(f"Frozen: {selected['method']}, sampler seed={selected['sampler_seed']}, trial={selected['trial_number']}")
    return decision


def fit_frozen_selection():
    output = PROJECT_ROOT / "results" / "week5"
    selection_path = output / "selection.json"
    selection = read_json(selection_path)
    model_path = output / "final_model.joblib"
    if model_path.exists() or (output / "fit_metadata.json").exists():
        raise FileExistsError("Final fitted model/metadata already exists; no automatic refit or overwrite.")
    if (selection["status"] != "frozen_before_test" or selection["test_evaluated"]
            or selection["config_sha256"] != file_sha256(CONFIG_PATH)
            or selection["environment"] != _environment_versions()):
        raise ValueError("Frozen selection is incompatible with current state.")
    for name, expected in selection["implementation_sha256"].items():
        if file_sha256(PROJECT_ROOT / "src" / name) != expected:
            raise ValueError(f"Implementation changed after selection: {name}")
    X, y = load_training_data()
    if (file_sha256(DATA_DIR / "credit_train.csv") != selection["train_sha256"]
            or X.columns.tolist() != selection["feature_columns"] or len(y) != selection["n_train"]):
        raise ValueError("Training data changed after selection.")
    lock_hash = file_sha256(selection_path)
    pipe = build_final_pipeline(selection)
    started = perf_counter()
    pipe.fit(X, y)
    fit_seconds = perf_counter() - started
    bundle = {"format_version": 1, "pipeline": pipe, "threshold": selection["selected"]["params"]["threshold"],
              "feature_columns": X.columns.tolist(), "selection_sha256": lock_hash, "class_labels": {0: "good/approve", 1: "bad/reject"}}
    # Serialization smoke check on training features only, not a performance estimate.
    expected_p, expected_y = predict_with_threshold(bundle, X.iloc[:8])
    temp = output / "final_model.pending.joblib"
    try:
        joblib.dump(bundle, temp, compress=3)
        restored = joblib.load(temp)  # File just created locally by this command.
        actual_p, actual_y = predict_with_threshold(restored, X.iloc[:8])
        np.testing.assert_array_equal(actual_p, expected_p)
        np.testing.assert_array_equal(actual_y, expected_y)
        if lock_hash != file_sha256(selection_path):
            raise ValueError("Selection changed during fit.")
        for attempt in range(6):
            try:
                os.replace(temp, model_path)
                break
            except OSError:
                if attempt == 5:
                    raise
                sleep(min(.1 * 2**attempt, 1.6))
    finally:
        temp.unlink(missing_ok=True)
    write_json(output / "fit_metadata.json", {
        "status": "fitted_train_only", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": len(y), "label_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "fit_count": 1, "fit_seconds": fit_seconds, "selection_sha256": lock_hash,
        "model_file": model_path.name, "model_sha256": file_sha256(model_path),
        "model_size_bytes": model_path.stat().st_size, "environment": _environment_versions(), "joblib_version": version("joblib"),
        "threshold": bundle["threshold"], "rf_params": pipe.named_steps["classifier"].get_params(),
        "serialization_check": "Exact probability and prediction match on first 8 training feature rows; no accuracy scored.",
        "test_evaluated": False, "next_step": "Week 5 part 3 only after explicit request: evaluate the frozen bundle on test.",
    })
    print(f"Fitted once on {len(y)} training rows. Saved {model_path}\nNo test data loaded or evaluated.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "fit"))
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    args = parser.parse_args(argv)
    if args.action == "select":
        source = (PROJECT_ROOT / args.source).resolve()
        if not source.is_relative_to((PROJECT_ROOT / "results" / "week4").resolve()):
            parser.error("Selection source must be an explicit run inside results/week4.")
        freeze_selection(source)
    else:
        fit_frozen_selection()


if __name__ == "__main__":
    main()

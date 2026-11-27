"""Fit an exploratory TPE candidate with a 30% train-OOF rejection ceiling.

python -m src.cap30
python -m src.cap30 --resume results/experiments/cap30/run-<UTC>
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import sys

import joblib
import numpy as np
import optuna
import pandas as pd

from .artifacts import write_csv, write_json
from .config import CONFIG_PATH, DATA_DIR, PROJECT_ROOT, load_config
from .data_utils import _environment_versions, file_sha256, load_cv_splits, load_training_data
from .final_model import build_final_pipeline, predict_with_threshold
from .finalize import verify_oof
from .optimize import read_baselines
from .resumable_search import run_search
from .robustness import block_test
from .week4 import load_baseline_oof, read_json, validate_search
from .week4_analysis import baseline_scenario, per_run_row


BASE = PROJECT_ROOT / "results" / "experiments" / "cap30"
SEARCH_CAP = 0.30


def reference_rows(X, y, splits, config):
    baselines, _ = read_baselines()
    if len(baselines) != 2:
        raise ValueError("Both original OOF baselines are required.")
    rows = [{"method": r["method"], "threshold": r["threshold"], "cost_per_sample": r["cost_per_sample"],
             "rejection_rate": r["rejection_rate"]} for r in baselines]
    baseline_proba = load_baseline_oof(X, y, splits)
    adjusted = baseline_scenario(y, baseline_proba, config, "cap30")[1]
    if adjusted["feasible"]:
        rows.append({"method": "baseline_threshold_cap30", **{k: adjusted[k] for k in
                    ("threshold", "cost_per_sample", "rejection_rate")}})
    original = read_json(PROJECT_ROOT / "results/week5/selection.json")
    selected = original["selected"]
    if (original["train_sha256"] != file_sha256(DATA_DIR / "credit_train.csv")
            or original["cv_sha256"] != file_sha256(DATA_DIR / "cv_folds.json")
            or original["config_sha256"] != file_sha256(CONFIG_PATH)):
        raise ValueError("Week 5 selection used different inputs.")
    rows.append({"method": "week5_" + selected["method"], "threshold": selected["params"]["threshold"],
                 "cost_per_sample": selected["metrics"]["cost_per_sample"],
                 "rejection_rate": selected["metrics"]["rejection_rate"]})
    return rows


def fit_candidate(output, selection, X, y):
    selection_path = output / "selection.json"
    if selection_path.exists() and read_json(selection_path) != selection:
        raise ValueError("Candidate differs from the frozen selection.")
    if not selection_path.exists():
        write_json(selection_path, selection)
    model_path = output / "candidate_model.joblib"
    pipeline = build_final_pipeline(selection).fit(X, y)
    bundle = {"pipeline": pipeline, "format_version": 1, "threshold": selection["selected"]["params"]["threshold"],
              "feature_columns": X.columns.tolist(), "selection_sha256": file_sha256(selection_path)}
    probe = X.iloc[:8]
    expected_p, expected_y = predict_with_threshold(bundle, probe)
    pending = output / "candidate_model.pending.joblib"
    try:
        joblib.dump(bundle, pending, compress=3)
        restored = joblib.load(pending)
        actual_p, actual_y = predict_with_threshold(restored, probe)
        np.testing.assert_array_equal(actual_p, expected_p)
        np.testing.assert_array_equal(actual_y, expected_y)
        pending.replace(model_path)
    finally:
        pending.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, help="Continue or verify a local cap30 run directory.")
    args = parser.parse_args(argv)
    config = load_config()
    policy_cap = config["policy"]["max_rejection_rate"]
    if not SEARCH_CAP < policy_cap:
        raise ValueError("The experimental search cap must be below the evaluation policy cap.")
    config["policy"]["max_rejection_rate"] = SEARCH_CAP
    seeds = config["optimization"]["sampler_seeds"]
    budget = config["optimization"]["trials_per_seed"]
    X, y = load_training_data()
    splits = load_cv_splits(X)
    references = reference_rows(X, y, splits, config)
    protocol = {
        "scope": "Exploratory after disclosure of the original holdout; selection uses fixed train OOF only.",
        "evaluation_policy_cap": policy_cap, "search_config": config,
        "eligibility": "Fit a candidate only if all planned seeds have a feasible trial.",
        "tie_order": ["cost_per_sample", "n_estimators", "max_depth", "seed_config_order", "trial_number"],
        "inputs": {n: file_sha256(DATA_DIR / n) for n in ("credit_train.csv", "cv_folds.json", "dataset_metadata.json")},
        "environment": _environment_versions(), "joblib_version": version("joblib"),
        "code": {n: file_sha256(PROJECT_ROOT / "src" / n) for n in
                 ("cap30.py", "final_model.py", "finalize.py", "robustness.py", "resumable_search.py", "config.py",
                  "optimize.py", "objective.py", "preprocessing.py", "baseline.py", "week4.py", "week4_analysis.py",
                  "artifacts.py", "data_utils.py")},
        "historical_oof_and_cap30_baselines": references,
    }
    if args.resume:
        output = (PROJECT_ROOT / args.resume).resolve()
        if output.parent != BASE.resolve():
            parser.error("Resume an existing run directly inside results/experiments/cap30.")
        metadata = read_json(output / "run_metadata.json")
        if metadata["protocol"] != protocol:
            raise ValueError("Resume protocol, inputs, code or environment changed.")
        if metadata["status"] == "complete":
            for name, checksum in metadata["output_sha256"].items():
                if file_sha256(output / name) != checksum:
                    raise ValueError(f"Output checksum mismatch: {name}")
            print(f"Verified completed experiment: {output}")
            return
    else:
        now = datetime.now(timezone.utc)
        output = BASE / now.strftime("run-%Y%m%dT%H%M%S%fZ")
        output.mkdir(parents=True, exist_ok=False)
        metadata = {"created_at_utc": now.isoformat(), "protocol": protocol, "test_evaluated": False}

    metadata.update(status="running")
    metadata.pop("error", None)
    write_json(output / "run_metadata.json", metadata)
    print(f"Output: {output}", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        rows, candidates = [], []
        for seed in seeds:
            folder = output / f"tpe_seed{seed}"
            run_search("tpe", seed, budget, X, y, splits, config, folder)
            summary, _ = validate_search(folder, "tpe", seed, config)
            rows.append(per_run_row(summary, "cap30"))
            best = summary["best_feasible"]
            if best is not None:
                verify_oof(folder / "best_oof.csv", best, X, y, splits, config)
                candidates.append(best)
        write_csv(output / "seed_summary.csv", pd.DataFrame(rows))
        comparison = list(references)
        if len(candidates) == len(seeds):
            selected = min(candidates, key=lambda r: (r["metrics"]["cost_per_sample"], r["params"]["n_estimators"],
                           r["params"]["max_depth"], seeds.index(r["sampler_seed"]), r["trial_number"]))
            oof_file = f"tpe_seed{selected['sampler_seed']}/best_oof.csv"
            selection = {"selected": selected, "model_seed": config["model_seed"], "oof_file": oof_file,
                         "oof_sha256": file_sha256(output / oof_file), "search_cap": SEARCH_CAP,
                         "evaluation_policy_cap": policy_cap, "status": "exploratory_train_only"}
            fit_candidate(output, selection, X, y)
            comparison.append({"method": "tpe_cap30", "threshold": selected["params"]["threshold"],
                               "cost_per_sample": selected["metrics"]["cost_per_sample"],
                               "rejection_rate": selected["metrics"]["rejection_rate"]})
        write_csv(output / "oof_comparison.csv", pd.DataFrame(comparison))
        metadata.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        n_train=len(y), feasible_seeds=len(candidates), completed_trials=sum(r["completed_trials"] for r in rows),
                        search_fits=sum(r["completed_fit_count"] for r in rows), candidate_fitted=len(candidates) == len(seeds),
                        output_sha256={p.relative_to(output).as_posix(): file_sha256(p) for p in output.rglob("*")
                                       if p.is_file() and p.name not in {"run_metadata.json", "search.checkpoint.json"}})
    except BaseException as exc:
        metadata.update(status="interrupted_or_failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run_metadata.json", metadata)
        raise
    write_json(output / "run_metadata.json", metadata)
    print(pd.DataFrame(comparison).to_string(index=False))
    print(f"Completed train-only experiment. Candidate fitted: {metadata['candidate_fitted']}")


if __name__ == "__main__":
    sys.addaudithook(block_test)
    main()

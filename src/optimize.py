"""Week 3 CLI: paired Random Search/TPE on train OOF; never evaluate test.

Run: python -m src.optimize
Default: 50 trials per method, sampler seed 42. Each run gets a new directory.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from .config import CONFIG_PATH, DATA_DIR, PROJECT_ROOT, load_config
from .data_utils import (_environment_versions, file_sha256, load_cv_splits,
                         load_training_data, write_json)
from .objective import candidate_key, evaluate_candidate, suggest_params


def make_sampler(method: str, seed: int):
    if method == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if method == "tpe":
        return optuna.samplers.TPESampler(seed=seed, n_startup_trials=10,
                                         multivariate=False, constant_liar=False)
    raise ValueError(f"Unknown method: {method}")


def history_rows(study) -> list:
    rows = []
    best_cost = None
    for trial in study.trials:
        attrs = trial.user_attrs
        metrics = attrs.get("metrics", {})
        if trial.state == optuna.trial.TrialState.COMPLETE and attrs.get("feasible"):
            cost = metrics["cost_per_sample"]
            best_cost = cost if best_cost is None else min(best_cost, cost)
        rows.append({"trial_number": trial.number, "evaluation_number": trial.number + 1,
                     "state": trial.state.name, **trial.params, **metrics,
                     "constraint_g": attrs.get("constraint_g"),
                     "constraint_violation": attrs.get("constraint_violation"),
                     "feasible": attrs.get("feasible"), "objective": trial.value,
                     "evaluation_seconds": attrs.get("evaluation_seconds"),
                     "fit_count": attrs.get("fit_count"),
                     "best_feasible_cost_so_far": best_cost})
    return rows


def run_search(method, seed, n_trials, X, y, splits, config, output_dir):
    """Persist logs after every trial; a failed run never pretends to be complete."""
    output_dir.mkdir(parents=True, exist_ok=False)
    study = optuna.create_study(direction="minimize", sampler=make_sampler(method, seed),
                                pruner=optuna.pruners.NopPruner())
    started = perf_counter()
    best = None
    status = "running"

    def objective(trial):
        nonlocal best
        params = suggest_params(trial, config)
        result = evaluate_candidate(params, X, y, splits, config)
        oof = result.pop("oof_proba")
        fold_ids = result.pop("fold_ids")
        for key, value in result.items():
            trial.set_user_attr(key, value)
        record = {"method": method, "sampler_seed": seed,
                  "trial_number": trial.number, "params": params, **result}
        if result["feasible"] and (best is None or candidate_key(record) < candidate_key(best)):
            best = record
            pd.DataFrame({"source_row_id": X.index, "target": y.to_numpy(),
                          "fold": fold_ids, "probability_bad": oof,
                          "prediction": (oof >= params["threshold"]).astype(int)
                          }).to_csv(output_dir / "best_oof.csv", index=False)
            write_json(output_dir / "best_feasible.json", best)
        return result["objective"]

    def save_history(study, trial=None):
        rows = [{"method": method, "sampler_seed": seed, **row} for row in history_rows(study)]
        pd.DataFrame(rows).to_csv(output_dir / "trials.csv", index=False)
        write_json(output_dir / "trials.json", {"trials": [
            {**row, "fold_details": t.user_attrs.get("fold_details")}
            for row, t in zip(rows, study.trials)]})
        if trial is not None:
            value = trial.user_attrs.get("metrics", {}).get("cost_per_sample")
            print(f"{method} seed={seed} trial={trial.number + 1}/{n_trials} "
                  f"cost={value} feasible={trial.user_attrs.get('feasible')}", flush=True)

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=[save_history])
        status = "complete"
    finally:
        save_history(study)
        rows = history_rows(study)
        summary = {"method": method, "sampler_seed": seed, "status": status,
                   "requested_trials": n_trials,
                   "completed_trials": sum(r["state"] == "COMPLETE" for r in rows),
                   "feasible_trials": sum(r.get("feasible") is True for r in rows),
                   "completed_fit_count": sum(r.get("fit_count") or 0 for r in rows),
                   "elapsed_seconds": perf_counter() - started,
                   "best_feasible": best,
                   "selection_status": "feasible_found" if best else "no feasible solution found within budget"}
        if status != "complete":
            summary["status"] = "interrupted_or_failed"
            summary["selection_status"] = "partial_run_do_not_compare_as_full_budget"
        write_json(output_dir / "summary.json", summary)
    return summary, rows


def read_baselines():
    """Verify the local baseline inputs/results before drawing comparison lines."""
    folder = PROJECT_ROOT / "results"
    metadata_path = folder / "baseline_run_metadata.json"
    if not metadata_path.exists():
        return [], "Baseline metadata missing; run python -m src.baseline for comparison."
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_inputs = metadata["input_sha256"]
    if metadata["config_sha256"] != file_sha256(CONFIG_PATH) or any(
            expected_inputs[name] != file_sha256(DATA_DIR / name)
            for name in ("credit_train.csv", "cv_folds.json")):
        raise ValueError("Baseline inputs/config differ; rerun baseline before comparing.")
    for name in ("baseline_default.json", "baseline_threshold_sweep.json"):
        if metadata["result_sha256"][name] != file_sha256(folder / name):
            raise ValueError(f"Baseline result checksum mismatch: {name}")
    if metadata["environment"] != _environment_versions():
        raise ValueError("Baseline environment differs; rerun baseline before comparing.")
    baseline = json.loads((folder / "baseline_default.json").read_text(encoding="utf-8"))
    sweep = json.loads((folder / "baseline_threshold_sweep.json").read_text(encoding="utf-8"))
    rows = [{"method": "baseline_default", "threshold": baseline["threshold"], **baseline["pooled_oof"],
             **baseline["ml_metrics"], "feasible": baseline["feasible"]}]
    if sweep["best_feasible"] is not None:
        rows.append({"method": "baseline_threshold", **sweep["best_feasible"],
                     **sweep["ml_metrics_best"]})
    return rows, "Baseline values from the same fixed train/CV inputs and environment."


def plot_progress(histories, baselines, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for (method, seed), rows in histories.items():
        ax.step([r["evaluation_number"] for r in rows],
                [np.nan if r["best_feasible_cost_so_far"] is None else r["best_feasible_cost_so_far"]
                 for r in rows], where="post", label=f"{method}, seed={seed}")
    for row in baselines:
        if row["feasible"]:
            ax.axhline(row["cost_per_sample"], linestyle="--", label=row["method"], alpha=0.7)
    ax.set(xlabel="Completed trial evaluations", ylabel="Best feasible OOF cost / sample",
           title="Train OOF search progress (lower is better; test unused)")
    ax.text(0.01, 0.02, "No line before the first feasible trial.", transform=ax.transAxes, fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Sampler seeds from config.")
    parser.add_argument("--trials", type=int, help="Default: config budget (50). Other budgets are exploratory.")
    args = parser.parse_args(argv)
    config = load_config()
    budget = args.trials if args.trials is not None else config["optimization"]["trials_per_seed"]
    if budget < 1 or len(set(args.seeds)) != len(args.seeds) or any(
            seed not in config["optimization"]["sampler_seeds"] for seed in args.seeds):
        parser.error("Use a positive budget and distinct seeds present in config.")
    # Always use config seed order for deterministic presentation/tie ordering.
    seeds = [s for s in config["optimization"]["sampler_seeds"] if s in args.seeds]
    X, y = load_training_data()
    splits = load_cv_splits(X)
    baselines, baseline_note = read_baselines()
    output = PROJECT_ROOT / "results" / "week3" / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "config": config, "config_sha256": file_sha256(CONFIG_PATH),
        "sampler_seeds": seeds, "trials_per_method_per_seed": budget,
        "matches_planned_budget": budget == config["optimization"]["trials_per_seed"],
        "environment": _environment_versions(), "model_seed": config["model_seed"],
        "samplers": {"random": "RandomSampler", "tpe": "TPESampler: 10 startup trials, univariate, constant_liar=False"},
        "input_sha256": {n: file_sha256(DATA_DIR / n) for n in ("credit_train.csv", "cv_folds.json", "dataset_metadata.json")},
        "code_sha256": {n: file_sha256(PROJECT_ROOT / "src" / n) for n in
                        ("objective.py", "optimize.py", "baseline.py", "preprocessing.py", "data_utils.py", "config.py")},
        "baseline_note": baseline_note, "baseline_comparison": baselines,
        "scope": "train OOF selection only; no independent test estimate",
        "expected_fit_count": 2 * len(seeds) * budget * len(splits),
        "pruning": False, "parallel_trials": 1, "rf_n_jobs": 1,
        "tie_break": "Within seed: cost, n_estimators, max_depth, trial_number. No final method selection in week 3.",
    }
    write_json(output / "run_metadata.json", metadata)
    print(f"Output: {output}\nExpected RF fits: {metadata['expected_fit_count']}", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    started = perf_counter()
    histories, summaries = {}, []
    try:
        for seed in seeds:
            for method in ("random", "tpe"):
                summary, history = run_search(method, seed, budget, X, y, splits, config,
                                              output / f"{method}_seed{seed}")
                summaries.append(summary)
                histories[method, seed] = history
        comparison = list(baselines)
        for summary in summaries:
            best = summary["best_feasible"]
            comparison.append({"method": summary["method"], "sampler_seed": summary["sampler_seed"],
                               "feasible": best is not None,
                               **(best["metrics"] if best else {}),
                               **(best["params"] if best else {}),
                               "trial_number": best["trial_number"] if best else None,
                               "feasible_trials": summary["feasible_trials"],
                               "completed_fit_count": summary["completed_fit_count"],
                               "search_seconds": summary["elapsed_seconds"]})
        pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
        write_json(output / "summary.json", {"runs": summaries, "comparison": comparison})
        plot_progress(histories, baselines, output / "best_feasible_cost.png")
        print(pd.DataFrame(comparison).reindex(columns=[
            "method", "cost_per_sample", "rejection_rate", "feasible"
        ]).to_string(index=False))
        metadata["status"] = "complete"
    finally:
        if metadata["status"] != "complete":
            metadata["status"] = "interrupted_or_failed"
        metadata["elapsed_seconds"] = perf_counter() - started
        metadata["output_sha256"] = {p.relative_to(output).as_posix(): file_sha256(p)
                                      for p in output.rglob("*") if p.is_file() and p.name != "run_metadata.json"}
        write_json(output / "run_metadata.json", metadata)
    print(f"Finished. Inspect comparison.csv and best_feasible_cost.png in:\n{output}")
    print("OOF selection results only. Test data was not loaded.")


if __name__ == "__main__":
    main()

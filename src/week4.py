"""Week 4: reuse verified seed 42, run remaining seeds and one-factor sensitivity.

python -m src.week4 --check-only  # read-only preflight
python -m src.week4               # full experiment, no test evaluation
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import optuna
import pandas as pd

from .config import CONFIG_PATH, DATA_DIR, PROJECT_ROOT, load_config
from .data_utils import (_environment_versions, file_sha256, load_cv_splits,
                         load_training_data)
from .artifacts import write_csv, write_json
from .optimize import read_baselines
from .resumable_search import run_search
from .week4_analysis import (METHODS, aggregate_runs, baseline_scenario, convergence_table,
                             paired_differences, per_run_row, plot_results)

DEFAULT_REUSE = "results/week3/run-20261108T052245407488Z"

# Exact reviewed compatibility pair: log I/O fix only; evaluator/sampler unchanged.
IO_COMPATIBLE_OPTIMIZE = ("7114bc1839e02fd7bc2fcf96f58f732c4308d2e390e73570f4dd64c7ef74e755",
                         "a0ae590d87898f4cd20013fe3dc0d2d30aba27a581ee3080a061247f7a21e3f7")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_search(folder, method, seed, config):
    summary = read_json(folder / "summary.json")
    rows = read_json(folder / "trials.json")["trials"]
    budget = config["optimization"]["trials_per_seed"]
    folds = config["cv"]["n_splits"]
    if (summary["status"] != "complete" or summary["method"] != method
            or summary["sampler_seed"] != seed or summary["requested_trials"] != budget
            or summary["completed_trials"] != budget or len(rows) != budget
            or summary["completed_fit_count"] != budget * folds):
        raise ValueError(f"Incomplete or incompatible search: {folder}")
    for i, r in enumerate(rows):
        if (r["state"] != "COMPLETE" or r["trial_number"] != i
                or r["method"] != method or r["sampler_seed"] != seed or r["fit_count"] != folds):
            raise ValueError(f"Invalid trial history: {folder}")
    feasible = [r for r in rows if r["feasible"]]
    expected = min(feasible, key=lambda r: (r["cost_per_sample"], r["n_estimators"],
                                           r["max_depth"], r["trial_number"])) if feasible else None
    best = summary["best_feasible"]
    if (summary["feasible_trials"] != len(feasible)
            or (expected is None) != (best is None)
            or (best is not None and (not best["feasible"] or best["trial_number"] != expected["trial_number"]
                                     or best["metrics"]["cost_per_sample"] != expected["cost_per_sample"]))):
        raise ValueError(f"Best selection and trial history disagree: {folder}")
    return summary, rows


def validate_reuse(source, config):
    """Reject changed inputs, environment, search code, incomplete budgets or artifacts."""
    metadata = read_json(source / "run_metadata.json")
    if (metadata["status"] != "complete" or metadata["config"] != config
            or metadata["config_sha256"] != file_sha256(CONFIG_PATH)
            or metadata["environment"] != _environment_versions()
            or metadata["trials_per_method_per_seed"] != config["optimization"]["trials_per_seed"]
            or 42 not in metadata["sampler_seeds"]):
        raise ValueError("Week 3 run is not compatible with the current protocol/environment.")
    for group, root in (("input_sha256", DATA_DIR), ("code_sha256", PROJECT_ROOT / "src"),
                        ("output_sha256", source)):
        for name, checksum in metadata[group].items():
            path = (root / name).resolve()
            current = file_sha256(path) if path.is_relative_to(root.resolve()) else None
            compatible_io = (group == "code_sha256" and name == "optimize.py"
                             and (checksum, current) == IO_COMPATIBLE_OPTIMIZE)
            if not path.is_relative_to(root.resolve()) or (current != checksum and not compatible_io):
                raise ValueError(f"Checksum mismatch in reused run: {group}/{name}")
    return {method: validate_search(source / f"{method}_seed42", method, 42, config) for method in METHODS}


def load_baseline_oof(X, y, splits):
    folder = PROJECT_ROOT / "results"
    metadata = read_json(folder / "baseline_run_metadata.json")
    path = folder / "baseline_oof.csv"
    if file_sha256(path) != metadata["result_sha256"][path.name]:
        raise ValueError("Baseline OOF checksum mismatch.")
    frame = pd.read_csv(path)
    fold_ids = np.full(len(y), -1, dtype=int)
    for i, (_, val) in enumerate(splits):
        fold_ids[val] = i
    if (frame.source_row_id.tolist() != X.index.tolist()
            or frame.target.tolist() != y.tolist() or frame.fold.tolist() != fold_ids.tolist()
            or not np.isfinite(frame.probability_bad).all()
            or not frame.probability_bad.between(0, 1).all()):
        raise ValueError("Baseline OOF rows, targets, folds or probabilities are invalid.")
    return frame.probability_bad.to_numpy()


def scenario_configs(config):
    scenarios = {"main": deepcopy(config)}
    for cost in config["sensitivity"]["planned_false_negative_costs"]:
        if cost != config["cost"]["false_negative"]:
            new = deepcopy(config)
            new["cost"]["false_negative"] = float(cost)
            scenarios[f"fn{cost:g}"] = new
    for rate in config["sensitivity"]["planned_max_rejection_rates"]:
        if rate != config["policy"]["max_rejection_rate"]:
            new = deepcopy(config)
            new["policy"]["max_rejection_rate"] = float(rate)
            scenarios[f"rejection{rate * 100:g}"] = new
    return scenarios


def write_review(output, aggregate, pairs):
    lines = ["# Week 4: generated result review", "",
             "Train OOF selection results; test was not loaded.",
             "Sample SD (ddof=1) is across sampler seeds on the same split/folds.",
             "Cost summaries are conditional on finding a feasible solution; inspect success counts.", ""]
    for r in aggregate.to_dict(orient="records"):
        lines.append(f"- {r['method']}: {r['n_feasible_seeds']}/{r['n_seeds']} feasible seeds; "
                     f"cost mean={r['cost_per_sample_mean']}, SD={r['cost_per_sample_std']}; "
                     f"mean search seconds={r['search_seconds_mean']}.")
    lines += ["", "Paired comparisons (one difference per sampler seed):"]
    for winner, count in pairs.lower_cost_method.value_counts().items():
        lines.append(f"- {winner}: {count} seeds.")
    lines += ["", "Interpretation limits:",
              "- No claim of statistical significance from five seeds alone.",
              "- Runtime includes log writing and can change with system load; reused seed 42 ran earlier.",
              "- Sensitivity reruns searches per scenario with seed 42 only; other seeds are not tested for sensitivity.",
              "- Changing FN cost changes the objective scale. Compare methods within the same scenario.",
              "- Fixed RF baseline OOF is rescored; its threshold is selected again within each scenario.",
              "- No final method/configuration is selected automatically. Keep the 5:1 / 35% main protocol."]
    (output / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", default=DEFAULT_REUSE, help="Week 3 run containing seed 42.")
    parser.add_argument("--check-only", action="store_true", help="Verify inputs/reuse without training or output writes.")
    parser.add_argument("--resume", type=Path, help="Continue a checkpoint-enabled week4 run directory.")
    args = parser.parse_args(argv)
    config = load_config()
    source = Path(args.reuse)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    source = source.resolve()
    if not source.is_relative_to(PROJECT_ROOT.resolve()):
        parser.error("Keep the reused run inside this repository for portable provenance.")
    X, y = load_training_data()
    splits = load_cv_splits(X)
    baselines, _ = read_baselines()
    reused = validate_reuse(source, config)
    proba = load_baseline_oof(X, y, splits)
    scenarios = scenario_configs(config)
    seeds = config["optimization"]["sampler_seeds"]
    if 42 not in seeds:
        parser.error("This protocol requires seed 42 for reuse and sensitivity.")
    budget = config["optimization"]["trials_per_seed"]
    new_fits = 2 * (len(seeds) - 1 + len(scenarios) - 1) * budget * len(splits)
    print(f"Verified seed 42 reuse. Main seeds: {seeds}. New RF fits: {new_fits}.", flush=True)
    if args.check_only:
        print("Preflight passed; no training performed and no results written.")
        return
    output = args.resume if args.resume else PROJECT_ROOT / "results" / "week4" / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    if not output.is_relative_to((PROJECT_ROOT / "results" / "week4").resolve()):
        parser.error("Resume directory must be inside results/week4.")
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "config_sha256": file_sha256(CONFIG_PATH),
        "environment": _environment_versions(), "sampler_seeds": seeds,
        "scenario_configs": scenarios, "sensitivity_seed": 42, "trials_per_method_per_seed": budget,
        "reuse_path": source.relative_to(PROJECT_ROOT).as_posix(),
        "reuse_metadata_sha256": file_sha256(source / "run_metadata.json"),
        "reuse_compatibility_note": "Reviewed optimize.py log-I/O fix accepted by exact old/new SHA-256 pair; all other recorded code must match.",
        "reuse_output_sha256": read_json(source / "run_metadata.json")["output_sha256"],
        "new_expected_fit_count": new_fits, "reused_fit_count": 2 * budget * len(splits),
        "code_sha256": {p.name: file_sha256(p) for p in (PROJECT_ROOT / "src").glob("*.py")},
        "input_sha256": {n: file_sha256(DATA_DIR / n) for n in ("credit_train.csv", "cv_folds.json", "dataset_metadata.json")},
        "baseline_oof_sha256": file_sha256(PROJECT_ROOT / "results" / "baseline_oof.csv"),
        "scope": "train OOF only; 5 sampler seeds for main protocol; sensitivity rerun seed 42 only",
        "completed_searches": [],
        "checkpoint_version": 1,
    }
    if args.resume:
        previous = read_json(output / "run_metadata.json")
        if previous.get("checkpoint_version") != 1:
            parser.error("This older run has no resumable checkpoints. Keep its existing results.")
        for key in ("config", "environment", "code_sha256", "input_sha256", "reuse_metadata_sha256", "baseline_oof_sha256"):
            if previous[key] != metadata[key]:
                parser.error(f"Resume refused: {key} changed.")
        if previous["status"] == "complete":
            print(f"Run already complete: {output}. No searches repeated.")
            return
        metadata["created_at_utc"] = previous["created_at_utc"]
    write_json(output / "run_metadata.json", metadata)
    print(f"Output: {output}", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    started = perf_counter()
    summaries, histories, sensitivity = [], {}, []
    try:
        for seed in seeds:
            for method in METHODS:
                if seed == 42:
                    summary, rows = reused[method]
                else:
                    folder = output / "main" / f"{method}_seed{seed}"
                    run_search(method, seed, budget, X, y, splits, config, folder)
                    summary, rows = validate_search(folder, method, seed, config)
                    metadata["completed_searches"].append(f"main/{method}_seed{seed}")
                    write_json(output / "run_metadata.json", metadata)
                summaries.append(per_run_row(summary))
                histories[method, seed] = rows
        for name, scenario in scenarios.items():
            scenario_rows = baseline_scenario(y.to_numpy(), proba, scenario, name)
            for method in METHODS:
                if name == "main":
                    summary = reused[method][0]
                else:
                    folder = output / "sensitivity" / name / f"{method}_seed42"
                    run_search(method, 42, budget, X, y, splits, scenario, folder)
                    summary, _ = validate_search(folder, method, 42, scenario)
                    metadata["completed_searches"].append(f"sensitivity/{name}/{method}_seed42")
                    write_json(output / "run_metadata.json", metadata)
                scenario_rows.append(per_run_row(summary, name))
            for row in scenario_rows:
                row.update(cost_fn=scenario["cost"]["false_negative"],
                           cost_fp=scenario["cost"]["false_positive"],
                           max_rejection_rate=scenario["policy"]["max_rejection_rate"])
            sensitivity.extend(scenario_rows)
        runs = pd.DataFrame(summaries)
        aggregate = aggregate_runs(runs)
        pairs = paired_differences(runs, seeds)
        convergence = convergence_table(histories, seeds)
        sensitive = pd.DataFrame(sensitivity)
        for name, frame in (("per_seed", runs), ("aggregate", aggregate), ("paired_differences", pairs),
                            ("convergence", convergence), ("sensitivity", sensitive)):
            write_csv(output / f"{name}.csv", frame)
        write_json(output / "summary.json", {
            "per_seed": summaries, "baseline_main": baselines, "sensitivity": sensitivity,
            "aggregate": json.loads(aggregate.to_json(orient="records")),
            "paired_differences": json.loads(pairs.to_json(orient="records")),
            "new_fit_count": new_fits, "total_fit_count_including_reused": new_fits + metadata["reused_fit_count"],
        })
        plot_results(runs, convergence, sensitive, baselines, output / "figures")
        write_review(output, aggregate, pairs)
        metadata["status"] = "complete"
        print(aggregate[["method", "n_feasible_seeds", "cost_per_sample_mean", "cost_per_sample_std", "search_seconds_mean"]].to_string(index=False))
    finally:
        if metadata["status"] != "complete":
            metadata["status"] = "interrupted_or_failed"
        metadata["elapsed_seconds"] = perf_counter() - started
        metadata["output_sha256"] = {p.relative_to(output).as_posix(): file_sha256(p)
                                     for p in output.rglob("*") if p.is_file() and p.name not in {"run_metadata.json", "search.checkpoint.json"}}
        write_json(output / "run_metadata.json", metadata)
    print(f"Finished WEEK 4. Results: {output}\nTest data was not loaded. Read review.md before interpreting results.")


if __name__ == "__main__":
    main()

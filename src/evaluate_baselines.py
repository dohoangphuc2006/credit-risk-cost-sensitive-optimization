"""Supplemental test comparison of the two Week 2 baselines, after TPE test disclosure.

Run: python -m src.evaluate_baselines
Fit one default RF on train; apply both historical thresholds to the same probabilities.
Reuse saved TPE test predictions. Do not tune, select a new winner or overwrite prior results.
"""

from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .artifacts import write_csv, write_json
from .baseline import apply_threshold
from .config import CONFIG_PATH, DATA_DIR, PROJECT_ROOT
from .data_utils import _environment_versions, file_sha256, load_training_data
from .evaluate_final import load_holdout, read_json, score_predictions
from .preprocessing import build_preprocessor


def verify_hashes(root, hashes):
    for name, expected in hashes.items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"Historical artifact checksum mismatch: {name}")


def prepare_comparison():
    """Validate existing records before fitting or reading the test CSV."""
    results = PROJECT_ROOT / "results"
    final = results / "week5"
    baseline = read_json(results / "baseline_run_metadata.json")
    selection = read_json(final / "selection.json")
    tpe = read_json(final / "test_evaluation.json")
    if tpe["status"] != "complete":
        raise ValueError("A completed original TPE evaluation is required.")
    verify_hashes(final, {**tpe["frozen_artifact_sha256"], **tpe["output_sha256"]})
    if (baseline["config_sha256"] != file_sha256(CONFIG_PATH)
            or baseline["config"] != selection["config"]
            or baseline["environment"] != _environment_versions()
            or selection["environment"] != _environment_versions()):
        raise ValueError("Use the original baseline configuration and environment.")
    verify_hashes(DATA_DIR, baseline["input_sha256"])
    verify_hashes(PROJECT_ROOT / "src", baseline["code_sha256"])
    verify_hashes(results, baseline["result_sha256"])
    default = read_json(results / "baseline_default.json")
    swept = read_json(results / "baseline_threshold_sweep.json")
    thresholds = {"baseline_default": default["threshold"],
                  "baseline_threshold": swept["best_feasible"]["threshold"]}
    if thresholds != {"baseline_default": .5, "baseline_threshold": .38}:
        raise ValueError("Expected the two historical thresholds 0.50 and 0.38; no reselection allowed.")
    if (baseline["model_params"]["class_weight"] is not None
            or baseline["model_params"]["random_state"] != selection["model_seed"]):
        raise ValueError("Baseline parameters do not match the original protocol.")
    paths = [results / "baseline_run_metadata.json", final / "test_evaluation.json"]
    paths += [results / n for n in baseline["result_sha256"]]
    paths += [final / n for n in {**tpe["frozen_artifact_sha256"], **tpe["output_sha256"]}]
    protected = {p.relative_to(PROJECT_ROOT).as_posix(): file_sha256(p) for p in paths}
    return selection, baseline, thresholds, protected


def verify_saved_tpe(frame, X, y, metrics, threshold, config):
    """Require the very same test rows and full-precision stored decisions."""
    if (frame.source_row_id.tolist() != X.index.tolist() or frame.target.tolist() != y.tolist()
            or not np.isfinite(frame.probability_bad).all()
            or not frame.probability_bad.between(0, 1).all()
            or not np.array_equal(apply_threshold(frame.probability_bad.to_numpy(), threshold), frame.prediction)):
        raise ValueError("Saved TPE predictions do not match test identities or frozen threshold.")
    computed = score_predictions(y, frame.prediction.to_numpy(), config)
    if computed != metrics:
        raise ValueError("Saved TPE metrics do not match its predictions.")


def plot_comparison(rows, cap, path):
    labels = ["RF default\nthreshold 0.50", "RF threshold-only\nthreshold 0.38", "Frozen TPE"]
    colors = ["#7c8b9c", "#477ba7", "#d2783c"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for ax, key, title in zip(axes, ["cost_per_sample", "rejection_rate"],
                              ["Relative cost / sample (lower is better)", "Rejection rate (policy cap = 35%)"]):
        values = [r[key] for r in rows]
        bars = ax.bar(labels, values, color=colors, width=.6)
        ax.bar_label(bars, labels=[f"{v:.3f}" if key == "cost_per_sample" else f"{v:.1%}" for v in values], padding=4)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(max(values), cap) * 1.3)
        ax.spines[["top", "right"]].set_visible(False)
        if key == "rejection_rate":
            ax.axhline(cap, color="#b52b2b", ls="--", label="35% cap")
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.))
            ax.legend(loc="upper left")
    fig.suptitle(f"Same {rows[0]['n_samples']} test rows | Supplemental comparison after TPE test disclosure", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_baselines():
    final = PROJECT_ROOT / "results" / "week5"
    output = final / "baseline_test"
    ledger = output / "evaluation.json"
    if ledger.exists():
        record = read_json(ledger)
        if record["status"] != "complete":
            raise RuntimeError("Incomplete supplemental evaluation exists; inspect it before any further action.")
        verify_hashes(PROJECT_ROOT, record["protected_artifact_sha256"])
        verify_hashes(output, record["output_sha256"])
        result = read_json(output / "metrics.json")
        print("Verified saved comparison; no training, test read or inference repeated.")
        print(pd.DataFrame(result["comparison"])[["method", "cost_per_sample", "rejection_rate", "feasible"]].to_string(index=False))
        return result

    selection, baseline, thresholds, protected = prepare_comparison()
    output.mkdir(exist_ok=False)  # Refuse overwriting an incomplete directory, including concurrent runs.
    record = {"status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Supplemental descriptive comparison requested after observing TPE test results; not preregistered.",
              "thresholds": thresholds, "threshold_source": "Unchanged Week 2 train-OOF results",
              "model_params": baseline["model_params"], "fit_count": 0, "test_evaluated": False,
              "protected_artifact_sha256": protected, "environment": _environment_versions(),
              "joblib_version": version("joblib"),
              "implementation_sha256": {n: file_sha256(PROJECT_ROOT / "src" / n) for n in
                   ("evaluate_baselines.py", "evaluate_final.py", "baseline.py", "preprocessing.py")},
              "tpe_policy": "Reuse original saved predictions; no TPE refit or inference; keep original selected model."}
    write_json(ledger, record)
    try:
        X_train, y_train = load_training_data(DATA_DIR)
        if X_train.columns.tolist() != selection["feature_columns"] or len(y_train) != selection["n_train"]:
            raise ValueError("Training data schema/count differ from the frozen selection.")
        pipe = Pipeline([("preprocessor", build_preprocessor()),
                         ("classifier", RandomForestClassifier(**baseline["model_params"]))])
        started = perf_counter()
        pipe.fit(X_train, y_train)
        record.update(fit_count=1, n_train=len(y_train), fit_seconds=perf_counter() - started)
        joblib.dump({"pipeline": pipe, "thresholds": thresholds, "feature_columns": X_train.columns.tolist()},
                    output / "baseline_model.joblib", compress=3)
        write_json(ledger, record)  # Fit and historical thresholds recorded before opening test.
        X, y, metadata = load_holdout(selection)
        old = read_json(final / "test_evaluation.json")
        if old["test_sha256"] != metadata["file_sha256"]["credit_test.csv"]:
            raise ValueError("The original and supplemental evaluations must use the same test file.")
        saved = pd.read_csv(final / "test_predictions.csv", converters={"probability_bad": float})
        tpe_metrics = read_json(final / "test_metrics.json")
        verify_saved_tpe(saved, X, y, tpe_metrics["metrics"], tpe_metrics["threshold"], selection["config"])
        started = perf_counter()
        p = pipe.predict_proba(X)[:, list(pipe.classes_).index(1)]
        record.update(test_evaluated=True, n_test=len(y), inference_seconds=perf_counter() - started,
                      test_sha256=old["test_sha256"], probability_inference_count=1)
        predictions = pd.DataFrame({"source_row_id": X.index, "target": y.to_numpy(), "baseline_probability_bad": p})
        rows = []
        for method, threshold in thresholds.items():
            pred = apply_threshold(p, threshold)
            rows.append({"method": method, "threshold": threshold, **score_predictions(y, pred, selection["config"])})
            predictions[f"{method}_prediction"] = pred
        rows.append({"method": "tpe", "threshold": tpe_metrics["threshold"], **tpe_metrics["metrics"]})
        predictions["tpe_probability_bad"] = saved.probability_bad.to_numpy()
        predictions["tpe_prediction"] = saved.prediction.to_numpy()
        result = {"scope": record["scope"], "cost": selection["config"]["cost"],
                  "max_rejection_rate": selection["config"]["policy"]["max_rejection_rate"],
                  "comparison": rows,
                  "interpretation": "Descriptive comparison on the same test samples. Compare both cost and feasibility; lower cost alone does not satisfy the constraint. No test-based threshold tuning or replacement of the selected model."}
        write_csv(output / "predictions.csv", predictions)
        write_csv(output / "comparison.csv", pd.DataFrame(rows))
        write_json(output / "metrics.json", result)
        plot_comparison(rows, result["max_rejection_rate"], output / "comparison.png")
        verify_hashes(PROJECT_ROOT, protected)
        record.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      output_sha256={n: file_sha256(output / n) for n in
                         ("baseline_model.joblib", "predictions.csv", "comparison.csv", "metrics.json", "comparison.png")})
    except BaseException as error:
        record.update(status="interrupted_or_failed", error=f"{type(error).__name__}: {error}")
        write_json(ledger, record)
        raise
    write_json(ledger, record)
    print(pd.DataFrame(rows)[["method", "cost_per_sample", "FN", "FP", "rejection_rate", "feasible"]].to_string(index=False))
    print(f"Supplemental comparison saved: {output}")
    return result


if __name__ == "__main__":
    evaluate_baselines()

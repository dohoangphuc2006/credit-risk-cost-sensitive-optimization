"""Evaluate the frozen bundle once; subsequent calls verify and show saved results."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .artifacts import write_csv, write_json
from .baseline import compute_cost_metrics
from .config import CONFIG_PATH, DATA_DIR, PROJECT_ROOT
from .data_utils import _environment_versions, file_sha256
from .final_model import predict_with_threshold


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_frozen(output):
    """Check the decision and fitted artifact before opening any holdout file."""
    selection = read_json(output / "selection.json")
    fit = read_json(output / "fit_metadata.json")
    if (selection["status"] != "frozen_before_test" or selection["test_evaluated"]
            or fit["status"] != "fitted_train_only" or fit["test_evaluated"]):
        raise ValueError("Expected immutable pre-test selection and fit records.")
    if file_sha256(output / "selection.json") != fit["selection_sha256"]:
        raise ValueError("Frozen selection checksum mismatch.")
    if file_sha256(output / "final_model.joblib") != fit["model_sha256"]:
        raise ValueError("Fitted model checksum mismatch.")
    if file_sha256(CONFIG_PATH) != selection["config_sha256"]:
        raise ValueError("Experiment configuration changed after selection.")
    for name, expected in selection["implementation_sha256"].items():
        if file_sha256(PROJECT_ROOT / "src" / name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    if (selection["environment"] != _environment_versions()
            or fit["environment"] != _environment_versions()
            or fit["joblib_version"] != version("joblib")):
        raise ValueError("Use the recorded Python/package versions for the fitted model.")
    if (file_sha256(DATA_DIR / "credit_train.csv") != selection["train_sha256"]
            or file_sha256(DATA_DIR / "cv_folds.json") != selection["cv_sha256"]):
        raise ValueError("Training data or folds changed after selection.")
    bundle = joblib.load(output / "final_model.joblib")  # Verified, trusted project artifact.
    if (bundle["selection_sha256"] != fit["selection_sha256"]
            or bundle["threshold"] != selection["selected"]["params"]["threshold"]
            or bundle["threshold"] != fit["threshold"]
            or bundle["feature_columns"] != selection["feature_columns"]
            or list(bundle["pipeline"].classes_) != [0, 1]):
        raise ValueError("Bundle does not match the frozen decision.")
    return selection, fit, bundle


def load_holdout(selection):
    metadata = read_json(DATA_DIR / "dataset_metadata.json")
    if (metadata["artifact_format_version"] != 2
            or metadata["config_sha256"] != selection["config_sha256"]
            or metadata["feature_columns"] != selection["feature_columns"]
            or metadata["file_sha256"]["credit_train.csv"] != selection["train_sha256"]):
        raise ValueError("Holdout metadata does not match the frozen training protocol.")
    path = DATA_DIR / "credit_test.csv"
    if file_sha256(path) != metadata["file_sha256"][path.name]:
        raise ValueError("Holdout checksum mismatch.")
    frame = pd.read_csv(path, index_col="source_row_id")
    expected_n = metadata["test_set"]["n_samples"]
    if (len(frame) != expected_n or not frame.index.is_unique
            or frame.index.tolist() != metadata["test_source_row_ids"]
            or set(frame.index).intersection(metadata["train_source_row_ids"])):
        raise ValueError("Holdout row IDs/count differ from the saved disjoint split.")
    y = frame.pop("target")
    if (frame.columns.tolist() != selection["feature_columns"]
            or not y.isin([0, 1]).all()
            or int(y.eq(1).sum()) != metadata["test_set"]["n_bad"]
            or int(y.eq(0).sum()) != metadata["test_set"]["n_good"]):
        raise ValueError("Holdout schema or labels differ from the saved dataset.")
    return frame, y.astype("int64"), metadata


def score_predictions(y, predictions, config):
    metrics = compute_cost_metrics(y, predictions, config["cost"]["false_negative"],
                                   config["cost"]["false_positive"])
    metrics.update(accuracy=float(accuracy_score(y, predictions)),
                   precision_bad=float(precision_score(y, predictions, zero_division=0)),
                   recall_bad=float(recall_score(y, predictions, zero_division=0)),
                   f1_bad=float(f1_score(y, predictions, zero_division=0)),
                   f1_macro=float(f1_score(y, predictions, average="macro", zero_division=0)))
    g = metrics["rejection_rate"] - config["policy"]["max_rejection_rate"]
    metrics.update(constraint_g=g, constraint_violation=max(0., g), feasible=bool(g <= 0))
    return metrics


def plot_test(metrics, output):
    matrix = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    fig, ax = plt.subplots(figsize=(6.5, 5.2), layout="constrained")
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for i in range(2):
        for j in range(2):
            name = [["TN", "FP"], ["FN", "TP"]][i][j]
            ax.text(j, i, f"{name} = {matrix[i, j]}", ha="center", va="center", fontsize=16,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Approve (0)", "Reject (1)"],
           yticklabels=["Good (0)", "Bad (1)"], xlabel="Frozen model decision", ylabel="Actual class",
           title=f"Final holdout: {metrics['n_samples']} rows\nCost/sample = {metrics['cost_per_sample']:.5f}; rejection = {metrics['rejection_rate']:.1%}")
    fig.savefig(output / "test_confusion_matrix.png", dpi=160)
    plt.close(fig)


def evaluate_final():
    output = PROJECT_ROOT / "results" / "week5"
    ledger = output / "test_evaluation.json"
    if ledger.exists():
        saved = read_json(ledger)
        if saved["status"] != "complete":
            raise RuntimeError("An incomplete test attempt exists. Inspect it; do not tune or silently repeat evaluation.")
        for name, digest in {**saved["frozen_artifact_sha256"], **saved["output_sha256"]}.items():
            if file_sha256(output / name) != digest:
                raise ValueError(f"Saved evaluation checksum mismatch: {name}")
        print("Showing verified saved results. Test data was not read; model was not run again.")
        result = read_json(output / "test_metrics.json")
        print(json.dumps(result["metrics"], indent=2))
        return result

    selection, fit, bundle = verify_frozen(output)
    # Exclusive creation protects the one-time evaluation against two concurrent processes.
    record = {"format_version": 1, "status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_count": 1, "fit_count": 0, "test_evaluated": False,
              "frozen_artifact_sha256": {n: file_sha256(output / n) for n in
                    ("selection.json", "fit_metadata.json", "final_model.joblib")},
              "implementation_sha256": {n: file_sha256(PROJECT_ROOT / "src" / n) for n in
                    ("evaluate_final.py", "final_model.py", "baseline.py")},
              "environment": _environment_versions(), "joblib_version": version("joblib"),
              "protocol": "One frozen pipeline/threshold; no fit, threshold sweep, model selection or test-based tuning."}
    with ledger.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        X, y, metadata = load_holdout(selection)
        started = perf_counter()
        probabilities, predictions = predict_with_threshold(bundle, X)
        elapsed = perf_counter() - started
        metrics = score_predictions(y, predictions, selection["config"])
        record.update(test_evaluated=True, n_test=len(y), inference_seconds=elapsed,
                      test_sha256=metadata["file_sha256"]["credit_test.csv"],
                      dataset_metadata_sha256=file_sha256(DATA_DIR / "dataset_metadata.json"))
        result = {"evaluation_scope": "Independent held-out test after frozen train-CV selection and refit",
                  "method": selection["selected"]["method"], "sampler_seed": selection["selected"].get("sampler_seed"),
                  "trial_number": selection["selected"].get("trial_number"), "threshold": bundle["threshold"],
                  "selection_sha256": fit["selection_sha256"], "model_sha256": fit["model_sha256"],
                  "cost": selection["config"]["cost"],
                  "max_rejection_rate": selection["config"]["policy"]["max_rejection_rate"], "metrics": metrics,
                  "interpretation": "Test is descriptive validation of the frozen model. OOF was used for selection; its score is optimistic. No baseline was evaluated on test, so no test improvement over baseline is claimed."}
        write_csv(output / "test_predictions.csv", pd.DataFrame({"source_row_id": X.index,
                  "target": y.to_numpy(), "probability_bad": probabilities, "prediction": predictions}))
        write_json(output / "test_metrics.json", result)
        cv = dict(selection["selected"]["metrics"], constraint_g=selection["selected"]["constraint_g"],
                  constraint_violation=selection["selected"]["constraint_violation"], feasible=selection["selected"]["feasible"])
        write_csv(output / "oof_vs_test.csv", pd.DataFrame([
            {"scope": "train_oof_used_for_selection", **cv}, {"scope": "independent_test", **metrics}]))
        plot_test(metrics, output)
        for name, digest in record["frozen_artifact_sha256"].items():
            if file_sha256(output / name) != digest:
                raise ValueError(f"Frozen artifact changed during evaluation: {name}")
        record.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      output_sha256={n: file_sha256(output / n) for n in
                           ("test_predictions.csv", "test_metrics.json", "oof_vs_test.csv", "test_confusion_matrix.png")})
    except BaseException as error:
        record.update(status="interrupted_or_failed", error=f"{type(error).__name__}: {error}")
        write_json(ledger, record)
        raise
    write_json(ledger, record)
    print(json.dumps(metrics, indent=2))
    print(f"Finished final evaluation: {output}. No fitting or tuning on test.")
    return result


if __name__ == "__main__":
    evaluate_final()

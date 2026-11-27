"""Compare the frozen cap-30 candidate on the previously disclosed Week 5 holdout."""

import joblib
import pandas as pd

from .artifacts import write_csv, write_json
from .baseline import apply_threshold
from .config import DATA_DIR, PROJECT_ROOT
from .data_utils import file_sha256
from .evaluate_baselines import verify_hashes
from .evaluate_final import load_holdout, read_json, score_predictions
from .final_model import predict_with_threshold


def evaluate():
    results = PROJECT_ROOT / "results"
    run = results / "experiments/cap30/run-20261126T074756136292Z"
    output = run / "disclosed_test"
    final = results / "week5"
    baseline = final / "baseline_test"
    metadata = read_json(run / "run_metadata.json")
    previous = read_json(baseline / "evaluation.json")
    if not metadata["candidate_fitted"] or metadata["status"] != "complete" or previous["status"] != "complete":
        raise ValueError("Both the cap-30 fit and the historical comparison must be complete.")
    verify_hashes(run, {k: metadata["output_sha256"][k] for k in ("selection.json", "candidate_model.joblib")})
    verify_hashes(baseline, {k: previous["output_sha256"][k] for k in ("predictions.csv", "metrics.json")})
    verify_hashes(PROJECT_ROOT, previous["protected_artifact_sha256"])
    verify_hashes(DATA_DIR, metadata["protocol"]["inputs"])
    verify_hashes(DATA_DIR, {"credit_test.csv": previous["test_sha256"]})
    ledger = output / "evaluation.json"
    if ledger.exists():
        verify_hashes(output, read_json(ledger)["output_sha256"])
        return pd.read_csv(output / "comparison.csv")

    selection = read_json(final / "selection.json")
    X, y, _ = load_holdout(selection)
    predictions = pd.read_csv(baseline / "predictions.csv",
                              converters={"baseline_probability_bad": float, "tpe_probability_bad": float})
    if predictions.source_row_id.tolist() != X.index.tolist() or predictions.target.tolist() != y.tolist():
        raise ValueError("Historical predictions and holdout rows differ.")
    candidate = joblib.load(run / "candidate_model.joblib")
    chosen = read_json(run / "selection.json")["selected"]
    if (candidate["selection_sha256"] != metadata["output_sha256"]["selection.json"]
            or candidate["threshold"] != chosen["params"]["threshold"]):
        raise ValueError("Candidate does not match the frozen selection.")
    predictions["tpe_cap30_probability_bad"], predictions["tpe_cap30_prediction"] = predict_with_threshold(candidate, X)
    config = selection["config"]
    historical = read_json(baseline / "metrics.json")["comparison"]
    rows = []
    for row in historical:
        score = score_predictions(y, predictions[row["method"] + "_prediction"], config)
        if any(score[k] != row[k] for k in score):
            raise ValueError("Historical scores could not be reproduced.")
        rows.append(row)

    thresholds = {row["method"]: row["threshold"]
                  for row in metadata["protocol"]["historical_oof_and_cap30_baselines"]}
    rf_threshold = thresholds["baseline_threshold_cap30"]
    predictions["baseline_threshold_cap30_prediction"] = apply_threshold(
        predictions.baseline_probability_bad.to_numpy(), rf_threshold)
    for method, threshold in [("baseline_threshold_cap30", rf_threshold), ("tpe_cap30", candidate["threshold"])]:
        rows.append({"method": method, "threshold": threshold,
                     **score_predictions(y, predictions[method + "_prediction"], config)})

    comparison = pd.DataFrame(rows)
    output.mkdir(exist_ok=True)
    write_csv(output / "comparison.csv", comparison)
    write_csv(output / "predictions.csv", predictions)
    write_json(ledger, {
        "scope": "Post hoc comparison on the disclosed Week 5 test records. No tuning or refitting.",
        "test_sha256": previous["test_sha256"], "n_test": len(y),
        "candidate_model_sha256": metadata["output_sha256"]["candidate_model.joblib"],
        "policy_cap": config["policy"]["max_rejection_rate"], "cost": config["cost"],
        "output_sha256": {k: file_sha256(output / k) for k in ("comparison.csv", "predictions.csv")},
    })
    return comparison


if __name__ == "__main__":
    print(evaluate()[["method", "cost_per_sample", "rejection_rate", "FN", "FP", "feasible"]].to_string(index=False))

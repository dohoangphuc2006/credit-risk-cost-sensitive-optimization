"""Supplemental comparison tests use synthetic rows only, never the real holdout."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.artifacts import write_csv, write_json
from src.config import load_config
from src.data_utils import file_sha256
from src.evaluate_baselines import evaluate_baselines, verify_hashes, verify_saved_tpe
from src.evaluate_final import read_json, score_predictions


class EvaluateBaselineTests(unittest.TestCase):
    def test_saved_tpe_requires_same_order_labels_threshold_and_metrics(self):
        X = pd.DataFrame({"x": [0, 1, 2, 3]}, index=[10, 11, 12, 13])
        y = pd.Series([0, 0, 1, 1], index=X.index)
        frame = pd.DataFrame({"source_row_id": X.index, "target": y.to_numpy(),
                              "probability_bad": [.2, .4, .55, .8], "prediction": [0, 0, 1, 1]})
        metrics = score_predictions(y, frame.prediction, load_config())
        verify_saved_tpe(frame, X, y, metrics, .55, load_config())
        with self.assertRaisesRegex(ValueError, "identities"):
            verify_saved_tpe(frame.iloc[::-1], X, y, metrics, .55, load_config())
        with self.assertRaisesRegex(ValueError, "threshold"):
            verify_saved_tpe(frame, X, y, metrics, .6, load_config())
        wrong = deepcopy(metrics)
        wrong["FN"] = 1
        with self.assertRaisesRegex(ValueError, "metrics"):
            verify_saved_tpe(frame, X, y, wrong, .55, load_config())

    def test_changed_historical_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "baseline.json"
            write_json(p, {"threshold": .38})
            checksums = {p.name: file_sha256(p)}
            verify_hashes(root, checksums)
            write_json(p, {"threshold": .4})
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_hashes(root, checksums)

    def test_one_train_fit_two_thresholds_then_cached_result_without_test_read(self):
        with ExitStack() as stack:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            final = root / "results" / "week5"
            final.mkdir(parents=True)
            (root / "src").mkdir()
            for name in ("evaluate_baselines.py", "evaluate_final.py", "baseline.py", "preprocessing.py"):
                (root / "src" / name).write_text("synthetic fixture")
            X_train = pd.DataFrame({"x": range(16), "cat": ["a", "b"] * 8})
            y_train = pd.Series([0, 1] * 8)
            X_test = pd.DataFrame({"x": [2, 3, 4, 5], "cat": ["a", "b", "new", "b"]}, index=[20, 21, 22, 23])
            y_test = pd.Series([0, 1, 0, 1], index=X_test.index)
            proba = np.array([.2, .8, .6, .4])
            saved = pd.DataFrame({"source_row_id": X_test.index, "target": y_test.to_numpy(),
                                  "probability_bad": proba, "prediction": (proba >= .55).astype(int)})
            write_csv(final / "test_predictions.csv", saved)
            config = load_config()
            write_json(final / "test_metrics.json", {"threshold": .55,
                "metrics": score_predictions(y_test, saved.prediction.to_numpy(), config)})
            write_json(final / "test_evaluation.json", {"test_sha256": "synthetic-hash"})
            protected = {p.relative_to(root).as_posix(): file_sha256(p) for p in final.iterdir()}
            selection = {"feature_columns": list(X_train.columns), "n_train": 16, "config": config}
            baseline = {"model_params": RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1).get_params()}
            thresholds = {"baseline_default": .5, "baseline_threshold": .38}
            stack.enter_context(patch("src.evaluate_baselines.PROJECT_ROOT", root))
            stack.enter_context(patch("src.evaluate_baselines.prepare_comparison", return_value=(selection, baseline, thresholds, protected)))
            stack.enter_context(patch("src.evaluate_baselines.load_training_data", return_value=(X_train, y_train)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            fits = []
            original_fit = Pipeline.fit
            def train_only_fit(pipe, X, y):
                self.assertIs(X, X_train)
                self.assertIs(y, y_train)
                fits.append(True)
                return original_fit(pipe, X, y)
            def load_test(_):
                self.assertEqual(len(fits), 1)
                return X_test, y_test, {"file_sha256": {"credit_test.csv": "synthetic-hash"}}
            stack.enter_context(patch.object(Pipeline, "fit", train_only_fit))
            load = stack.enter_context(patch("src.evaluate_baselines.load_holdout", side_effect=load_test))
            first = evaluate_baselines()
            second = evaluate_baselines()
            self.assertEqual(first, second)
            self.assertEqual(len(fits), 1)
            load.assert_called_once()
            out = final / "baseline_test"
            preds = pd.read_csv(out / "predictions.csv", converters={"baseline_probability_bad": float})
            for name, threshold in thresholds.items():
                self.assertEqual(preds[name + "_prediction"].tolist(), (preds.baseline_probability_bad >= threshold).astype(int).tolist())
            self.assertEqual(first["comparison"][2]["cost_per_sample"], read_json(final / "test_metrics.json")["metrics"]["cost_per_sample"])
            self.assertEqual(read_json(out / "evaluation.json")["fit_count"], 1)
            (out / "comparison.csv").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                evaluate_baselines()

    def test_incomplete_evaluation_cannot_silently_repeat(self):
        with tempfile.TemporaryDirectory() as tmp, patch("src.evaluate_baselines.PROJECT_ROOT", Path(tmp)):
            output = Path(tmp) / "results" / "week5" / "baseline_test"
            output.mkdir(parents=True)
            write_json(output / "evaluation.json", {"status": "started"})
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                evaluate_baselines()

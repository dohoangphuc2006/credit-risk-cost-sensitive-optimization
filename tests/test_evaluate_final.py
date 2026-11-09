"""Final evaluation checks with synthetic data only; never open the real holdout."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from importlib.metadata import version
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd

from src.artifacts import write_json
from src.config import CONFIG_PATH, load_config
from src.data_utils import _environment_versions, file_sha256
from src.evaluate_final import evaluate_final, load_holdout, score_predictions
from src.final_model import build_final_pipeline


class EvaluateFinalTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.data = self.root / "data"
        self.output = self.root / "results" / "week5"
        self.data.mkdir()
        self.output.mkdir(parents=True)
        (self.root / "src").mkdir()
        for name in ("evaluate_final.py", "final_model.py", "baseline.py"):
            (self.root / "src" / name).write_text("synthetic fixture")
        for name in ("credit_train.csv", "cv_folds.json"):
            (self.data / name).write_text("synthetic fixture")
        X = pd.DataFrame({"n": range(8), "cat": ["a", "b"] * 4})
        y = pd.Series([0, 1] * 4)
        config = load_config()
        m = score_predictions(y, y, config)
        self.selection = {"status": "frozen_before_test", "test_evaluated": False,
            "config_sha256": file_sha256(CONFIG_PATH), "config": config,
            "environment": _environment_versions(), "implementation_sha256": {},
            "train_sha256": file_sha256(self.data / "credit_train.csv"),
            "cv_sha256": file_sha256(self.data / "cv_folds.json"),
            "feature_columns": list(X.columns), "model_seed": 42,
            "selected": {"method": "tpe", "sampler_seed": 42, "trial_number": 0,
                "params": {"threshold": .55, "n_estimators": 20, "max_depth": 2,
                           "min_samples_split": 2, "bad_class_weight": 2.},
                "metrics": m, "constraint_g": m["constraint_g"],
                "constraint_violation": m["constraint_violation"], "feasible": m["feasible"]}}
        write_json(self.output / "selection.json", self.selection)
        pipe = build_final_pipeline(self.selection).fit(X, y)
        bundle = {"pipeline": pipe, "threshold": .55, "feature_columns": list(X.columns),
                  "selection_sha256": file_sha256(self.output / "selection.json")}
        joblib.dump(bundle, self.output / "final_model.joblib")
        fit = {"status": "fitted_train_only", "test_evaluated": False, "threshold": .55,
               "selection_sha256": bundle["selection_sha256"],
               "model_sha256": file_sha256(self.output / "final_model.joblib"),
               "environment": _environment_versions(), "joblib_version": version("joblib")}
        write_json(self.output / "fit_metadata.json", fit)
        X.assign(target=y).set_axis(range(8, 16)).to_csv(self.data / "credit_test.csv", index_label="source_row_id")
        self.metadata = {"artifact_format_version": 2, "config_sha256": self.selection["config_sha256"],
            "feature_columns": list(X.columns), "train_source_row_ids": list(range(8)),
            "test_source_row_ids": list(range(8, 16)), "test_set": {"n_samples": 8, "n_good": 4, "n_bad": 4},
            "file_sha256": {n: file_sha256(self.data / n) for n in ("credit_train.csv", "credit_test.csv")}}
        write_json(self.data / "dataset_metadata.json", self.metadata)
        self.stack.enter_context(patch("src.evaluate_final.PROJECT_ROOT", self.root))
        self.stack.enter_context(patch("src.evaluate_final.DATA_DIR", self.data))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def test_metrics_keep_cost_and_infeasibility_separate(self):
        m = score_predictions([0, 0, 1, 1], [0, 1, 0, 1], load_config())
        self.assertEqual(m["cost_per_sample"], 1.5)
        self.assertAlmostEqual(m["constraint_violation"], .15)
        self.assertFalse(m["feasible"])
        c = deepcopy(load_config())
        c["policy"]["max_rejection_rate"] = .5
        self.assertTrue(score_predictions([0, 1], [0, 1], c)["feasible"])

    def test_evaluate_once_then_show_saved_without_holdout_or_refit(self):
        with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("No refit allowed")):
            first = evaluate_final()
            (self.data / "credit_test.csv").unlink()
            with patch("src.evaluate_final.predict_with_threshold", side_effect=AssertionError("No second inference")):
                second = evaluate_final()
        self.assertEqual(first, second)
        self.assertTrue((self.output / "test_confusion_matrix.png").exists())
        self.assertEqual(len(pd.read_csv(self.output / "test_predictions.csv")), 8)

    def test_changed_model_stops_before_test_is_opened(self):
        (self.output / "final_model.joblib").write_bytes(b"changed model")
        with patch("src.evaluate_final.load_holdout", side_effect=AssertionError("Holdout must stay closed")):
            with self.assertRaisesRegex(ValueError, "model checksum"):
                evaluate_final()
        self.assertFalse((self.output / "test_evaluation.json").exists())

    def test_overlapping_rows_and_changed_checksum_rejected(self):
        self.metadata["train_source_row_ids"].append(8)
        write_json(self.data / "dataset_metadata.json", self.metadata)
        with self.assertRaisesRegex(ValueError, "disjoint split"):
            load_holdout(self.selection)
        (self.data / "credit_test.csv").write_text("changed data")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_holdout(self.selection)

    def test_incomplete_attempt_and_changed_saved_results_rejected(self):
        write_json(self.output / "test_evaluation.json", {"status": "started"})
        with self.assertRaisesRegex(RuntimeError, "incomplete test attempt"):
            evaluate_final()
        (self.output / "test_evaluation.json").unlink()
        evaluate_final()
        (self.output / "test_metrics.json").write_text("changed result")
        with self.assertRaisesRegex(ValueError, "checksum"):
            evaluate_final()


if __name__ == "__main__":
    unittest.main()

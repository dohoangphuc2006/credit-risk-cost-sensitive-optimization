"""Final selection/refit tests on controlled fixtures; test holdout is never loaded."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd

from src.artifacts import write_json
from src.config import CONFIG_PATH, load_config
from src.data_utils import _environment_versions, file_sha256
from src.final_model import predict_with_threshold
from src.finalize import choose_method_and_candidate, fit_frozen_selection, verify_oof


class FinalizeTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.seeds = self.config["optimization"]["sampler_seeds"]
        self.baselines = [{"method": "baseline_threshold", "threshold": .38,
                           "feasible": True, "cost_per_sample": .72}]

    def searches(self, random_costs, tpe_costs):
        rows = []
        for method, costs in (("random", random_costs), ("tpe", tpe_costs)):
            for seed, cost in zip(self.seeds, costs):
                best = None if cost is None else {"method": method, "sampler_seed": seed,
                    "trial_number": 10, "metrics": {"cost_per_sample": cost},
                    "params": {"threshold": .55, "n_estimators": 20, "max_depth": 2,
                               "min_samples_split": 2, "bad_class_weight": 2.}}
                rows.append({"method": method, "sampler_seed": seed, "best_feasible": best})
        return rows

    def test_method_uses_mean_not_single_lucky_seed(self):
        searches = self.searches([.1, 1., 1., 1., 1.], [.68, .7, .7, .7, .7])
        best, comparison = choose_method_and_candidate(searches, self.baselines, self.config)
        self.assertEqual(best["method"], "tpe")
        self.assertEqual(best["sampler_seed"], 42)
        self.assertAlmostEqual(comparison[-1]["mean_oof_cost"], .696)

    def test_baseline_wins_method_tie(self):
        best, _ = choose_method_and_candidate(self.searches([.72]*5, [.72]*5), self.baselines, self.config)
        self.assertEqual(best["method"], "baseline_threshold")

    def test_failed_seed_disqualifies_method_and_duplicates_rejected(self):
        rows = self.searches([.5, .5, .5, .5, None], [.7]*5)
        best, comparison = choose_method_and_candidate(rows, self.baselines, self.config)
        self.assertEqual(best["method"], "tpe")
        self.assertFalse(comparison[1]["eligible"])
        rows[-1]["sampler_seed"] = 42
        with self.assertRaises(ValueError):
            choose_method_and_candidate(rows, self.baselines, self.config)

    def test_candidate_tie_uses_tree_count_then_depth_then_seed(self):
        rows = self.searches([.8]*5, [.6]*5)
        rows[5]["best_feasible"]["params"]["n_estimators"] = 40
        rows[6]["best_feasible"]["params"]["max_depth"] = 8
        best, _ = choose_method_and_candidate(rows, self.baselines, self.config)
        self.assertEqual(best["sampler_seed"], 2024)

    def test_inference_uses_stored_threshold_and_class_order(self):
        pipe = Mock()
        pipe.classes_ = np.array([1, 0])
        pipe.predict_proba.return_value = np.array([[.53, .47], [.55, .45]])
        bundle = {"pipeline": pipe, "threshold": .55, "feature_columns": ["x"]}
        _, pred = predict_with_threshold(bundle, pd.DataFrame({"x": [1, 2]}))
        self.assertEqual(pred.tolist(), [0, 1])
        with self.assertRaises(ValueError):
            predict_with_threshold(bundle, pd.DataFrame({"wrong": [1]}))

    def test_oof_row_order_and_metric_mismatch_rejected(self):
        X = pd.DataFrame({"x": [1, 2, 3, 4]})
        y = pd.Series([0, 0, 0, 1])
        splits = [(np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3]))]
        best = {"params": {"threshold": .5}, "metrics": {"TN": 3, "FP": 0, "FN": 0, "TP": 1,
                                                                "cost_per_sample": 0., "rejection_rate": .25}}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "oof.csv"
            frame = pd.DataFrame({"source_row_id": [0, 1, 2, 3], "target": y, "fold": [0, 0, 1, 1], "probability_bad": [.1, .2, .3, .8]})
            frame.to_csv(p, index=False)
            verify_oof(p, best, X, y, splits, self.config)
            bad = deepcopy(best)
            bad["metrics"]["FN"] = 1
            with self.assertRaises(ValueError):
                verify_oof(p, bad, X, y, splits, self.config)
            frame.iloc[::-1].to_csv(p, index=False)
            with self.assertRaises(ValueError):
                verify_oof(p, best, X, y, splits, self.config)

    def test_train_refit_serializes_bundle_and_refuses_overwrite(self):
        X = pd.DataFrame({"n": range(8), "cat": ["a", "b"] * 4})
        y = pd.Series([0, 1] * 4)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data = root / "data"
            output = root / "results" / "week5"
            data.mkdir()
            output.mkdir(parents=True)
            # No test CSV exists in this fixture.
            train = data / "credit_train.csv"
            train.write_text("synthetic fixture")
            selection = {"status": "frozen_before_test", "test_evaluated": False,
                "config_sha256": file_sha256(CONFIG_PATH), "environment": _environment_versions(),
                "implementation_sha256": {}, "train_sha256": file_sha256(train),
                "feature_columns": X.columns.tolist(), "n_train": 8, "model_seed": 42,
                "selected": {"method": "tpe", "params": {"threshold": .55, "n_estimators": 20,
                    "max_depth": 2, "min_samples_split": 2, "bad_class_weight": 2.}}}
            write_json(output / "selection.json", selection)
            with patch("src.finalize.PROJECT_ROOT", root), patch("src.finalize.DATA_DIR", data), patch("src.finalize.load_training_data", return_value=(X, y)):
                fit_frozen_selection()
                with self.assertRaises(FileExistsError):
                    fit_frozen_selection()
            bundle = joblib.load(output / "final_model.joblib")
            self.assertEqual(bundle["threshold"], .55)
            self.assertEqual(bundle["pipeline"].named_steps["classifier"].class_weight, {0: 1., 1: 2.})
            meta = json.loads((output / "fit_metadata.json").read_text())
            self.assertFalse(meta["test_evaluated"])
            self.assertEqual(meta["fit_count"], 1)
            self.assertEqual(meta["n_train"], 8)

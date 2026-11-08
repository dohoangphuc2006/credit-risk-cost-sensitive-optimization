"""Protocol checks with synthetic data; no credit model experiment is run."""

import contextlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import optuna

from src.config import load_config
from src.objective import candidate_key, constraint_objective, evaluate_candidate
from src.optimize import history_rows, main, run_search


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.X = pd.DataFrame({"numeric": range(8), "category": ["a"] * 4 + ["b"] * 4})
        self.y = pd.Series([0, 1] * 4)
        self.splits = [(np.arange(4, 8), np.arange(4)), (np.arange(4), np.arange(4, 8))]
        self.params = {"threshold": .5, "n_estimators": 20, "max_depth": 2,
                       "min_samples_split": 2, "bad_class_weight": 2.0}
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def test_penalty_dominates_all_feasible_costs_and_boundary_is_exact(self):
        self.assertEqual(constraint_objective({"cost_per_sample": .72, "rejection_rate": .35}, self.config)["objective"], .72)
        result = constraint_objective({"cost_per_sample": 0, "rejection_rate": .350001}, self.config)
        self.assertFalse(result["feasible"])
        self.assertGreater(result["objective"], 5)
        result = constraint_objective({"cost_per_sample": .2, "rejection_rate": .45}, self.config)
        self.assertAlmostEqual(result["objective"], 6.2)

    def test_real_pipeline_synthetic_oof_with_unseen_categories(self):
        # Validation categories are absent from each training fold.
        result = evaluate_candidate(self.params, self.X, self.y, self.splits, self.config)
        self.assertEqual(result["fit_count"], 2)
        self.assertTrue(np.isfinite(result["oof_proba"]).all())
        self.assertEqual(result["fold_ids"].tolist(), [0] * 4 + [1] * 4)
        m = result["metrics"]
        self.assertEqual(m["cost_per_sample"], (5*m["FN"]+m["FP"])/8)

    def test_overlapping_or_repeated_folds_rejected_before_fit(self):
        bad_splits = [([0, 1], [1, 2]), self.splits, self.splits + self.splits]
        for splits in (bad_splits[0:1], bad_splits[2]):
            with self.assertRaises(ValueError):
                evaluate_candidate(self.params, self.X, self.y, splits, self.config)

    def test_tie_break_prefers_smaller_model_then_earlier_trial(self):
        first = {"metrics": {"cost_per_sample": .7}, "params": self.params, "trial_number": 0}
        other = deepcopy(first)
        other["params"]["n_estimators"] = 30
        self.assertLess(candidate_key(first), candidate_key(other))
        other = deepcopy(first)
        other["trial_number"] = 1
        self.assertLess(candidate_key(first), candidate_key(other))

    def fake_evaluation(self, params, X, y, splits, config):
        # Deliberately controlled optimizer fixture, never a reported experiment.
        metrics = {"cost_per_sample": 1 - params["threshold"],
                   "rejection_rate": 0.2 if params["threshold"] >= .5 else .5,
                   "n_samples": len(y)}
        return {"metrics": metrics, **constraint_objective(metrics, config),
                "fold_details": [], "fit_count": len(splits), "evaluation_seconds": 0.,
                "oof_proba": np.full(len(y), .4), "fold_ids": np.zeros(len(y), dtype=int)}

    def test_both_samplers_logs_and_seed_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp, patch("src.optimize.evaluate_candidate", side_effect=self.fake_evaluation), contextlib.redirect_stdout(io.StringIO()):
            for method in ("random", "tpe"):
                histories = []
                for repeat in range(2):
                    folder = Path(tmp) / f"{method}{repeat}"
                    summary, rows = run_search(method, 42, 12, self.X, self.y, self.splits, self.config, folder)
                    self.assertEqual(summary["completed_trials"], 12)
                    self.assertEqual(summary["completed_fit_count"], 24)
                    self.assertTrue(summary["best_feasible"]["feasible"])
                    self.assertTrue((folder / "best_oof.csv").exists())
                    costs = [r["best_feasible_cost_so_far"] for r in rows if r["best_feasible_cost_so_far"] is not None]
                    self.assertEqual(costs, sorted(costs, reverse=True))
                    histories.append(rows)
                self.assertEqual(histories[0], histories[1])

    def test_no_feasible_trial_produces_null_selection(self):
        def infeasible(*args):
            r = self.fake_evaluation(*args)
            r["metrics"]["rejection_rate"] = .9
            r.update(constraint_objective(r["metrics"], self.config))
            return r
        with tempfile.TemporaryDirectory() as tmp, patch("src.optimize.evaluate_candidate", side_effect=infeasible), contextlib.redirect_stdout(io.StringIO()):
            folder = Path(tmp) / "search"
            summary, rows = run_search("random", 42, 2, self.X, self.y, self.splits, self.config, folder)
            self.assertIsNone(summary["best_feasible"])
            self.assertFalse((folder / "best_oof.csv").exists())
            self.assertTrue(all(r["best_feasible_cost_so_far"] is None for r in rows))

    def test_failed_trial_is_preserved_and_error_propagates(self):
        with tempfile.TemporaryDirectory() as tmp, patch("src.optimize.evaluate_candidate", side_effect=ValueError("fixture failure")), contextlib.redirect_stdout(io.StringIO()):
            folder = Path(tmp) / "failed"
            with self.assertRaisesRegex(ValueError, "fixture failure"):
                run_search("random", 42, 2, self.X, self.y, self.splits, self.config, folder)
            summary = json.loads((folder / "summary.json").read_text())
            self.assertEqual(summary["status"], "interrupted_or_failed")
            self.assertEqual(summary["completed_trials"], 0)
            self.assertEqual(json.loads((folder / "trials.json").read_text())["trials"][0]["state"], "FAIL")

    def test_cli_exports_comparison_plot_metadata_without_test_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            for name in ("objective.py", "optimize.py", "baseline.py", "preprocessing.py", "data_utils.py", "config.py"):
                (root / "src" / name).write_text("fixture")
            (root / "data").mkdir()
            for name in ("credit_train.csv", "cv_folds.json", "dataset_metadata.json"):
                (root / "data" / name).write_text("fixture")
            with patch("src.optimize.PROJECT_ROOT", root), patch("src.optimize.DATA_DIR", root / "data"), patch("src.optimize.load_training_data", return_value=(self.X, self.y)), patch("src.optimize.load_cv_splits", return_value=self.splits), patch("src.optimize.read_baselines", return_value=([], "fixture")), patch("src.optimize.evaluate_candidate", side_effect=self.fake_evaluation), contextlib.redirect_stdout(io.StringIO()):
                main(["--trials", "2"])
            output = next((root / "results" / "week3").iterdir())
            meta = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(meta["status"], "complete")
            self.assertFalse(meta["matches_planned_budget"])
            self.assertTrue((output / "best_feasible_cost.png").exists())
            self.assertEqual(len(pd.read_csv(output / "comparison.csv")), 2)


if __name__ == "__main__":
    unittest.main()

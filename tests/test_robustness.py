"""Nested CV checks on synthetic data; original holdout is never read."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluate_final import read_json, score_predictions
from src.robustness import PROTOCOL_PATH, block_test, choose_threshold, make_splits, run_fold
from src.robustness_analysis import summarize


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.protocol = read_json(PROTOCOL_PATH)
        self.X = pd.DataFrame({"x": range(40), "cat": ["a", "b"] * 20}, index=range(100, 140))
        self.y = pd.Series([0, 1] * 20, index=self.X.index)

    def test_inner_rows_never_overlap_outer_validation(self):
        partitions, manifest = make_splits(self.X, self.y, self.protocol)
        outer_seen = []
        for (train, val, inner), saved in zip(partitions, manifest):
            outer_seen.extend(val)
            self.assertFalse(set(train) & set(val))
            inner_seen = []
            for it, iv in inner:
                self.assertFalse(set(it) & set(iv))
                self.assertEqual(set(it) | set(iv), set(range(len(train))))
                self.assertFalse(set(self.X.iloc[train].iloc[iv].index) & set(self.X.iloc[val].index))
                inner_seen.extend(iv)
            self.assertEqual(sorted(inner_seen), list(range(len(train))))
            self.assertEqual(saved["validation_row_ids"], self.X.iloc[val].index.tolist())
        self.assertEqual(sorted(outer_seen), list(range(len(self.X))))

    def test_inner_threshold_cap_and_no_feasible_case(self):
        best, _ = choose_threshold([0, 0, 1, 1], np.array([.1, .4, .5, .8]), .25,
                                   self.config, self.protocol["threshold_grid"])
        self.assertEqual(best["threshold"], .51)
        self.assertEqual(best["rejection_rate"], .25)
        best, _ = choose_threshold([0, 1], np.ones(2), .3, self.config, self.protocol["threshold_grid"])
        self.assertIsNone(best)

    def test_outer_labels_do_not_affect_inner_choices_and_resume_preserves_fold(self):
        train, val, inner = make_splits(self.X, self.y, self.protocol)[0][0]
        calls = []
        def search(method, seed, budget, X, y, splits, cfg, folder):
            self.assertEqual(X.index.tolist(), self.X.iloc[train].index.tolist())
            self.assertEqual(y.tolist(), self.y.iloc[train].tolist())
            self.assertFalse(set(X.index) & set(self.X.iloc[val].index))
            calls.append(cfg["policy"]["max_rejection_rate"])
            params = {"threshold": .7, "n_estimators": 20, "max_depth": 2,
                      "min_samples_split": 2, "bad_class_weight": 1.}
            best = None if cfg["policy"]["max_rejection_rate"] == .30 else {"params": params}
            return {"best_feasible": best, "completed_trials": budget}, []
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), patch("src.robustness.run_search", side_effect=search):
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            args = (0, self.X.iloc[train], self.y.iloc[train], self.X.iloc[val])
            rows, _ = run_fold(*args, self.y.iloc[val], inner, self.config, self.protocol, a)
            run_fold(*args, 1 - self.y.iloc[val], inner, self.config, self.protocol, b)
            self.assertEqual(read_json(a / "inner_choices.json"), read_json(b / "inner_choices.json"))
            self.assertEqual(len(calls), 6)
            missing = next(r for r in rows if r["method"] == "tpe_cap30")
            self.assertEqual(missing["selection_status"], "no_feasible_inner_candidate")
            self.assertNotIn("cost_per_sample", missing)
            run_fold(*args, self.y.iloc[val], inner, self.config, self.protocol, a)
            self.assertEqual(len(calls), 6)
            (a / "predictions.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                run_fold(*args, self.y.iloc[val], inner, self.config, self.protocol, a)

    def test_missing_fold_not_reported_as_complete_or_zero_cost(self):
        cfg = deepcopy(self.protocol)
        cfg["outer_folds"] = 2
        metrics = score_predictions([0, 1], [0, 1], self.config)
        rows = [{"method": "tpe", "outer_fold": 0, "selection_status": "selected", "n_outer": 2, **metrics},
                {"method": "tpe", "outer_fold": 1, "selection_status": "no_feasible_inner_candidate", "n_outer": 2}]
        p = pd.DataFrame({"method": ["tpe"] * 2, "outer_fold": [0, 0], "source_row_id": [1, 2],
                          "target": [0, 1], "prediction": [0, 1]})
        result = summarize(rows, p, [1, 2, 3, 4], self.config, cfg).iloc[0]
        self.assertFalse(result.complete_coverage)
        self.assertTrue(pd.isna(result.pooled_cost))
        with self.assertRaisesRegex(ValueError, "unique"):
            summarize(rows, pd.concat([p, p]), [1, 2, 3, 4], self.config, cfg)

    def test_original_test_guard_blocks_csv(self):
        with self.assertRaises(PermissionError):
            block_test("open", (str(Path("data") / "credit_test.csv"), "r"))
        block_test("open", (str(Path("data") / "credit_train.csv"), "r"))


if __name__ == "__main__":
    unittest.main()

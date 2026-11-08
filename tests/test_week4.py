"""Week 4 checks: controlled fixtures only, no credit search experiment."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.data_utils import write_json
from src.week4 import DEFAULT_REUSE, main, scenario_configs, validate_reuse, validate_search
from src.week4_analysis import (aggregate_runs, baseline_scenario, convergence_table,
                               paired_differences, per_run_row)


class Week4Tests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def fixture(self, method, seed, config):
        metrics = {"cost_per_sample": .5 if method == "random" else .6,
                   "rejection_rate": .2, "approval_rate": .8, "recall_bad": .7,
                   "FN": 2, "FP": 1, "f1_macro": .7}
        params = {"threshold": .5, "n_estimators": 20, "max_depth": 2,
                  "min_samples_split": 2, "bad_class_weight": 1.}
        n = config["optimization"]["trials_per_seed"]
        folds = config["cv"]["n_splits"]
        best = {"params": params, "metrics": metrics, "trial_number": 0, "feasible": True}
        summary = {"method": method, "sampler_seed": seed, "status": "complete",
                   "requested_trials": n, "completed_trials": n, "completed_fit_count": n * folds,
                   "feasible_trials": n, "elapsed_seconds": 2., "best_feasible": best}
        rows = [{"method": method, "sampler_seed": seed, "state": "COMPLETE", "trial_number": i,
                 "fit_count": folds, **params, **metrics, "feasible": True,
                 "best_feasible_cost_so_far": metrics["cost_per_sample"]} for i in range(n)]
        return summary, rows

    def test_sample_std_across_seeds_and_failed_seed_visible(self):
        frame = pd.DataFrame([
            {"method": "random", "feasible": True, "cost_per_sample": .5, "search_seconds": 1},
            {"method": "random", "feasible": True, "cost_per_sample": .7, "search_seconds": 2},
            {"method": "random", "feasible": False, "search_seconds": 3}])
        r = aggregate_runs(frame).iloc[0]
        self.assertEqual(r.n_seeds, 3)
        self.assertEqual(r.n_feasible_seeds, 2)
        self.assertAlmostEqual(r.cost_per_sample_mean, .6)
        self.assertAlmostEqual(r.cost_per_sample_std, np.sqrt(.02))
        self.assertEqual(r.search_seconds_n, 3)

    def test_all_infeasible_has_no_invented_cost(self):
        r = aggregate_runs(pd.DataFrame([{"method": "random", "feasible": False}])).iloc[0]
        self.assertEqual(r.n_feasible_seeds, 0)
        self.assertIsNone(r.cost_per_sample_mean)
        self.assertEqual(r.cost_per_sample_n, 0)

    def test_convergence_waits_for_all_seeds(self):
        histories = {(m, s): [{"best_feasible_cost_so_far": None if s == 42 else .9},
                              {"best_feasible_cost_so_far": .7 if s == 42 else .9}]
                     for m in ("random", "tpe") for s in (42, 100)}
        rows = convergence_table(histories, [42, 100])
        self.assertTrue(pd.isna(rows.iloc[0].mean_best_cost))
        self.assertEqual(rows.iloc[0].n_feasible_seeds, 1)
        self.assertAlmostEqual(rows.iloc[1].mean_best_cost, .8)

    def test_paired_difference_direction_and_unmatched_feasibility(self):
        frame = pd.DataFrame([{"method": m, "sampler_seed": s, "feasible": not (s == 100 and m == "tpe"),
                               "cost_per_sample": c} for s in (42, 100) for m, c in (("random", .7), ("tpe", .6))])
        pairs = paired_differences(frame, [42, 100])
        self.assertAlmostEqual(pairs.iloc[0].tpe_minus_random_cost, -.1)
        self.assertEqual(pairs.iloc[0].lower_cost_method, "tpe")
        self.assertFalse(pairs.iloc[1].both_feasible)
        self.assertTrue(pd.isna(pairs.iloc[1].tpe_minus_random_cost))

    def test_scenarios_change_one_assumption_without_mutating_config(self):
        original = deepcopy(self.config)
        scenarios = scenario_configs(self.config)
        self.assertEqual(set(scenarios), {"main", "fn2", "fn10", "rejection25", "rejection45"})
        self.assertEqual(scenarios["fn2"]["policy"], original["policy"])
        self.assertEqual(scenarios["rejection25"]["cost"], original["cost"])
        self.assertEqual(self.config, original)

    def test_baseline_threshold_reselected_for_constraint(self):
        cfg = deepcopy(self.config)
        cfg["policy"]["max_rejection_rate"] = .25
        rows = baseline_scenario(np.array([0, 0, 1, 1]), np.array([.1, .2, .5, .8]), cfg, "fixture")
        self.assertFalse(rows[0]["feasible"])
        self.assertTrue(rows[1]["feasible"])
        self.assertAlmostEqual(rows[1]["cost_per_sample"], 1.25)
        self.assertEqual(rows[1]["threshold"], .51)

    def test_reuse_rejects_changed_config_and_tampered_output(self):
        source = PROJECT_ROOT / DEFAULT_REUSE
        config = deepcopy(self.config)
        config["optimization"]["trials_per_seed"] = 49
        with self.assertRaises(ValueError):
            validate_reuse(source, config)
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "reuse"
            shutil.copytree(source, copy)
            (copy / "random_seed42" / "trials.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                validate_reuse(copy, self.config)

    def test_incomplete_search_cannot_enter_statistics(self):
        summary, rows = self.fixture("random", 42, self.config)
        summary["completed_trials"] -= 1
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_json(folder / "summary.json", summary)
            write_json(folder / "trials.json", {"trials": rows})
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                validate_search(folder, "random", 42, self.config)

    def test_full_orchestration_reuses42_and_exports_all_scenarios(self):
        cfg = deepcopy(self.config)
        cfg["optimization"]["trials_per_seed"] = 2
        cfg["cv"]["n_splits"] = 2
        reused = {m: self.fixture(m, 42, cfg) for m in ("random", "tpe")}
        X = pd.DataFrame({"x": range(8)})
        y = pd.Series([0, 1] * 4)
        splits = [(np.arange(4, 8), np.arange(4)), (np.arange(4), np.arange(4, 8))]
        calls = []
        def fake_search(method, seed, n, X, y, splits, config, folder):
            calls.append((method, seed, config["cost"]["false_negative"], config["policy"]["max_rejection_rate"]))
            summary, rows = self.fixture(method, seed, config)
            folder.mkdir(parents=True)
            write_json(folder / "summary.json", summary)
            write_json(folder / "trials.json", {"trials": rows})
            return summary, rows
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "fixture.py").write_text("fixture")
            (root / "data").mkdir()
            for name in ("credit_train.csv", "cv_folds.json", "dataset_metadata.json"):
                (root / "data" / name).write_text("fixture")
            source = root / DEFAULT_REUSE
            source.mkdir(parents=True)
            write_json(source / "run_metadata.json", {"output_sha256": {}})
            (root / "results" / "baseline_oof.csv").write_text("fixture")
            with contextlib.ExitStack() as stack:
                for name, value in {"PROJECT_ROOT": root, "DATA_DIR": root / "data"}.items():
                    stack.enter_context(patch("src.week4." + name, value))
                for name, value in {"load_config": cfg, "load_training_data": (X, y),
                                    "load_cv_splits": splits, "read_baselines": ([], "fixture"),
                                    "validate_reuse": reused, "load_baseline_oof": np.linspace(.1, .8, 8)}.items():
                    stack.enter_context(patch("src.week4." + name, return_value=value))
                stack.enter_context(patch("src.week4.run_search", side_effect=fake_search))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                main(["--check-only"])
                self.assertFalse((root / "results" / "week4").exists())
                main([])
            self.assertEqual(len(calls), 16)  # 8 additional main searches + 8 sensitivity searches.
            self.assertFalse(any(s == 42 and c == 5 and r == .35 for _, s, c, r in calls))
            output = next((root / "results" / "week4").iterdir())
            meta = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(len(pd.read_csv(output / "per_seed.csv")), 10)
            self.assertEqual(len(pd.read_csv(output / "sensitivity.csv")), 20)
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 3)
            self.assertTrue((output / "review.md").exists())


if __name__ == "__main__":
    unittest.main()

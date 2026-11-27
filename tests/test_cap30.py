from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import cap30
from src.config import load_config
from src.data_utils import file_sha256


class Cap30Tests(unittest.TestCase):
    def run_case(self, threshold, expect_model):
        X = pd.DataFrame({"value": [1] * 24})
        y = pd.Series([0, 0, 0, 1] * 6)
        splits = list(StratifiedKFold(2).split(X, y))

        def tiny_config():
            config = load_config()
            config["cv"]["n_splits"] = 2
            search = config["optimization"]
            search.update(sampler_seeds=[42, 100], trials_per_seed=1)
            for name, value in {"threshold": threshold, "n_estimators": 3, "max_depth": 2,
                                "min_samples_split": 2, "bad_class_weight": 1.0}.items():
                search["search_space"][name].update(low=value, high=value)
            return config

        with TemporaryDirectory() as temp:
            base = Path(temp)
            with (patch.object(cap30, "BASE", base), patch.object(cap30, "load_config", tiny_config),
                  patch.object(cap30, "load_training_data", return_value=(X, y)),
                  patch.object(cap30, "load_cv_splits", return_value=splits),
                  patch.object(cap30, "reference_rows", return_value=[]), redirect_stdout(StringIO())):
                cap30.main([])
                output, = base.iterdir()
                meta = cap30.read_json(output / "run_metadata.json")
                self.assertEqual(meta["status"], "complete")
                self.assertEqual(meta["candidate_fitted"], expect_model)
                self.assertFalse(meta["test_evaluated"])
                self.assertEqual((meta["completed_trials"], meta["search_fits"]), (2, 4))
                self.assertEqual((output / "candidate_model.joblib").exists(), expect_model)
                self.assertEqual((output / "selection.json").exists(), expect_model)
                if expect_model:
                    selected = cap30.read_json(output / "selection.json")["selected"]
                    self.assertEqual(selected["sampler_seed"], 42)
                    self.assertLessEqual(selected["metrics"]["rejection_rate"], cap30.SEARCH_CAP)
                before = file_sha256(output / "run_metadata.json")
                with patch.object(cap30, "run_search") as search, patch.object(cap30, "fit_candidate") as fit:
                    cap30.main(["--resume", str(output)])
                    search.assert_not_called()
                    fit.assert_not_called()
                self.assertEqual(file_sha256(output / "run_metadata.json"), before)

    def test_infeasible_budget_leaves_no_candidate(self):
        self.run_case(0.0, False)

    def test_feasible_tie_and_verify_without_refitting(self):
        self.run_case(1.0, True)

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy

import numpy as np
import pandas as pd

from src.config import load_config
from src.objective import constraint_objective
from src.resumable_search import run_search


class ResumeTests(unittest.TestCase):
    def test_interruption_preserves_sampler_pending_trial_and_completed_budget(self):
        cfg = load_config()
        X = pd.DataFrame({"x": [1, 2, 3, 4]})
        y = pd.Series([0, 1, 0, 1])
        splits = [(np.array([0, 1]), np.array([2, 3])), (np.array([2, 3]), np.array([0, 1]))]
        def evaluate(params, *args):
            m = {"cost_per_sample": 1-params["threshold"], "rejection_rate": .2}
            return {"metrics": m, **constraint_objective(m, cfg), "fit_count": 2,
                    "evaluation_seconds": 0., "fold_details": [],
                    "oof_proba": np.full(4, .4), "fold_ids": np.array([1, 1, 0, 0])}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for method in ("random", "tpe"):
                full = Path(tmp) / (method + 'full')
                resumed = Path(tmp) / (method + 'resume')
                with patch("src.resumable_search.evaluate_candidate", side_effect=evaluate):
                    _, expected = run_search(method, 42, 14, X, y, splits, cfg, full)
                count = 0
                def interrupted(*args):
                    nonlocal count
                    count += 1
                    if count == 12:
                        raise KeyboardInterrupt()
                    return evaluate(*args)
                with patch("src.resumable_search.evaluate_candidate", side_effect=interrupted):
                    with self.assertRaises(KeyboardInterrupt):
                        run_search(method, 42, 14, X, y, splits, cfg, resumed)
                with patch("src.resumable_search.evaluate_candidate", side_effect=evaluate) as call:
                    summary, actual = run_search(method, 42, 14, X, y, splits, cfg, resumed)
                    self.assertEqual(call.call_count, 3)
                self.assertEqual(actual, expected)
                self.assertEqual(summary["completed_trials"], 14)
                with patch("src.resumable_search.evaluate_candidate") as call:
                    run_search(method, 42, 14, X, y, splits, cfg, resumed)
                    call.assert_not_called()
                changed = deepcopy(cfg)
                changed['cost']['false_negative'] = 10
                with self.assertRaises(ValueError):
                    run_search(method, 42, 14, X, y, splits, changed, resumed)

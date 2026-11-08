"""Regression checks for transient Windows log-write failures."""

import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.artifacts import write_csv, write_json


class ArtifactTests(unittest.TestCase):
    def test_transient_invalid_argument_is_retried(self):
        import os
        replace = os.replace
        attempts = []

        def flaky(source, target):
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError(errno.EINVAL, "simulated file lock")
            return replace(source, target)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.json"
            with patch("src.artifacts.os.replace", side_effect=flaky), patch("src.artifacts.time.sleep"):
                write_json(path, {"trials": [1, 2]})
            self.assertEqual(json.loads(path.read_text()), {"trials": [1, 2]})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_persistent_error_keeps_previous_complete_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.json"
            path.write_text('{"trials": [1]}')
            with patch("src.artifacts.os.replace", side_effect=PermissionError(errno.EACCES, "locked")) as replace, patch("src.artifacts.time.sleep"):
                with self.assertRaises(PermissionError):
                    write_json(path, {"trials": [1, 2]})
            self.assertEqual(replace.call_count, 6)
            self.assertEqual(json.loads(path.read_text()), {"trials": [1]})

    def test_serialization_failure_does_not_touch_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.json"
            path.write_text("original")
            with self.assertRaises(ValueError):
                write_json(path, {"bad": float("nan")})
            self.assertEqual(path.read_text(), "original")

    def test_csv_is_replaced_and_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.csv"
            write_csv(path, pd.DataFrame({"trial": [0], "cost": [.72]}))
            write_csv(path, pd.DataFrame({"trial": [0, 1], "cost": [.72, .7]}))
            self.assertEqual(pd.read_csv(path).cost.tolist(), [.72, .7])


if __name__ == "__main__":
    unittest.main()

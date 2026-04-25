import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from project.scripts import calibrate_segment_thresholds


class CalibrateSegmentThresholdsScriptTests(unittest.TestCase):
    def _write_input_csv(self, path: Path) -> None:
        df = pd.DataFrame(
            [
                {"fraud_label": 0, "model_score": 0.10, "account_age_days": 5, "TransactionAmt": 100.0, "channel": "APP"},
                {"fraud_label": 0, "model_score": 0.20, "account_age_days": 5, "TransactionAmt": 100.0, "channel": "APP"},
                {"fraud_label": 1, "model_score": 0.80, "account_age_days": 5, "TransactionAmt": 100.0, "channel": "APP"},
                {"fraud_label": 1, "model_score": 0.90, "account_age_days": 5, "TransactionAmt": 100.0, "channel": "APP"},
            ]
        )
        df.to_csv(path, index=False)

    def test_blocks_threshold_write_when_delta_exceeds_limits_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_csv = tmp / "input.csv"
            output_json = tmp / "calibration.json"
            output_pkl = tmp / "thresholds.pkl"
            active_pkl = tmp / "active_thresholds.pkl"
            self._write_input_csv(input_csv)
            with active_pkl.open("wb") as handle:
                pickle.dump({"approve_threshold": 0.40, "block_threshold": 0.60}, handle)

            args = [
                "calibrate_segment_thresholds.py",
                "--input-csv",
                str(input_csv),
                "--output-json",
                str(output_json),
                "--output-thresholds-pkl",
                str(output_pkl),
                "--active-thresholds-pkl",
                str(active_pkl),
                "--max-approve-delta",
                "0.01",
                "--max-block-delta",
                "0.01",
                "--min-block-precision",
                "0.0",
                "--max-approve-to-flag-fpr",
                "1.0",
            ]
            with patch("sys.argv", args):
                with self.assertRaises(ValueError):
                    calibrate_segment_thresholds.main()

            self.assertFalse(output_pkl.exists())
            report_path = output_json.with_name("pr_curve_calibration_report.json")
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "blocked_delta_policy")
            self.assertGreater(report["delta_policy"]["violation_count"], 0)

    def test_force_allows_threshold_write_even_when_delta_exceeds_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_csv = tmp / "input.csv"
            output_json = tmp / "calibration.json"
            output_pkl = tmp / "thresholds.pkl"
            active_pkl = tmp / "active_thresholds.pkl"
            self._write_input_csv(input_csv)
            with active_pkl.open("wb") as handle:
                pickle.dump({"approve_threshold": 0.40, "block_threshold": 0.60}, handle)

            args = [
                "calibrate_segment_thresholds.py",
                "--input-csv",
                str(input_csv),
                "--output-json",
                str(output_json),
                "--output-thresholds-pkl",
                str(output_pkl),
                "--active-thresholds-pkl",
                str(active_pkl),
                "--max-approve-delta",
                "0.01",
                "--max-block-delta",
                "0.01",
                "--min-block-precision",
                "0.0",
                "--max-approve-to-flag-fpr",
                "1.0",
                "--force",
            ]
            with patch("sys.argv", args):
                rc = calibrate_segment_thresholds.main()

            self.assertEqual(rc, 0)
            self.assertTrue(output_pkl.exists())
            report = json.loads(output_json.with_name("pr_curve_calibration_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ready_to_write")
            self.assertTrue(report["delta_policy"]["force"])


if __name__ == "__main__":
    unittest.main()

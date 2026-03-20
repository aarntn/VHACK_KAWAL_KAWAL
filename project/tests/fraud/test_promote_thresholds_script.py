import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.scripts import promote_thresholds


class PromoteThresholdsScriptTests(unittest.TestCase):
    def test_refuses_promotion_when_policy_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            calibration = tmp / "context_calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "policy_checks": {"overall_pass": False},
                        "runtime_recommendation": {
                            "approve_threshold": 0.3,
                            "block_threshold": 0.9,
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = [
                "promote_thresholds.py",
                "--calibration-json",
                str(calibration),
                "--active-thresholds",
                str(tmp / "decision_thresholds.pkl"),
                "--archive-dir",
                str(tmp / "archive"),
                "--promotion-record-json",
                str(tmp / "record.json"),
            ]

            with patch("sys.argv", args):
                with self.assertRaises(ValueError):
                    promote_thresholds.main()

    def test_promotes_and_backs_up_existing_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            active = tmp / "decision_thresholds.pkl"
            with active.open("wb") as f:
                pickle.dump({"approve_threshold": 0.2, "block_threshold": 0.8}, f)

            calibration = tmp / "context_calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "policy_checks": {"overall_pass": True},
                        "runtime_recommendation": {
                            "approve_threshold": 0.25,
                            "block_threshold": 0.85,
                        },
                    }
                ),
                encoding="utf-8",
            )

            archive = tmp / "archive"
            record = tmp / "record.json"

            args = [
                "promote_thresholds.py",
                "--calibration-json",
                str(calibration),
                "--active-thresholds",
                str(active),
                "--archive-dir",
                str(archive),
                "--promotion-record-json",
                str(record),
            ]

            with patch("sys.argv", args):
                rc = promote_thresholds.main()

            self.assertEqual(rc, 0)
            with active.open("rb") as f:
                promoted = pickle.load(f)
            self.assertEqual(promoted["approve_threshold"], 0.25)
            self.assertEqual(promoted["block_threshold"], 0.85)

            payload = json.loads(record.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "promoted")
            self.assertIsNotNone(payload["rollback_thresholds_file"])
            self.assertTrue(Path(payload["rollback_thresholds_file"]).exists())


if __name__ == "__main__":
    unittest.main()

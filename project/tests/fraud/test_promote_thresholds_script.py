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
                        "artifact_metadata": {},
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

            with patch("sys.argv", args), patch.object(
                promote_thresholds,
                "collect_artifact_runtime_metadata",
                return_value={"serialization": {"format": "pickle", "pickle_protocol": 5, "python_minor": "3.11"}},
            ), patch.object(
                promote_thresholds,
                "validate_artifact_compatibility",
                return_value={"ok": True, "failed_checks": [], "checks": {}},
            ):
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
                        "artifact_metadata": {},
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

            with patch("sys.argv", args), patch.object(
                promote_thresholds,
                "collect_artifact_runtime_metadata",
                return_value={"serialization": {"format": "pickle", "pickle_protocol": 5, "python_minor": "3.11"}},
            ), patch.object(
                promote_thresholds,
                "validate_artifact_compatibility",
                return_value={"ok": True, "failed_checks": [], "checks": {}},
            ):
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
            self.assertTrue(Path(payload["rollback_metadata_file"]).exists())

    def test_validates_pr_curve_and_updates_hash_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            active = tmp / "decision_thresholds.pkl"
            with active.open("wb") as f:
                pickle.dump({"approve_threshold": 0.1, "block_threshold": 0.7}, f)

            calibration = tmp / "context_calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "policy_checks": {"overall_pass": True},
                        "runtime_recommendation": {
                            "approve_threshold": 0.25,
                            "block_threshold": 0.85,
                        },
                        "artifact_metadata": {},
                    }
                ),
                encoding="utf-8",
            )
            pr_curve = tmp / "pr_curve.json"
            pr_curve.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {"threshold": 0.1, "precision": 0.2, "recall": 0.9, "pr_auc": 0.22},
                            {"threshold": 0.5, "precision": 0.3, "recall": 0.8, "pr_auc": 0.35},
                            {"threshold": 0.95, "precision": 0.8, "recall": 0.2, "pr_auc": 0.21},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            checksums = tmp / "artifact_checksums.sha256"
            checksums.write_text(f"deadbeef  {active.relative_to(tmp)}\n", encoding="utf-8")
            promoted_manifest = tmp / "promoted_artifact_manifest.json"
            promoted_manifest.write_text(
                json.dumps(
                    {
                        "artifacts": {
                            "threshold_file": active.name,
                            "threshold_file_sha256": "oldhash",
                        }
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
                "--pr-curve-report-json",
                str(pr_curve),
                "--min-pr-auc",
                "0.30",
                "--checksum-manifest",
                str(checksums),
                "--promoted-artifact-manifest",
                str(promoted_manifest),
            ]
            with patch("sys.argv", args), patch.object(
                promote_thresholds,
                "collect_artifact_runtime_metadata",
                return_value={"serialization": {"format": "pickle", "pickle_protocol": 5, "python_minor": "3.11"}},
            ), patch.object(
                promote_thresholds,
                "validate_artifact_compatibility",
                return_value={"ok": True, "failed_checks": [], "checks": {}},
            ):
                rc = promote_thresholds.main()

            self.assertEqual(rc, 0)
            checksum_payload = checksums.read_text(encoding="utf-8")
            self.assertIn(str(active.relative_to(tmp)), checksum_payload)
            self.assertNotIn("deadbeef", checksum_payload)

            manifest_payload = json.loads(promoted_manifest.read_text(encoding="utf-8"))
            self.assertNotEqual(manifest_payload["artifacts"]["threshold_file_sha256"], "oldhash")

            record_payload = json.loads(record.read_text(encoding="utf-8"))
            self.assertTrue(record_payload["checksum_update"]["updated"])
            self.assertTrue(record_payload["promoted_manifest_update"]["updated"])
            self.assertTrue(record_payload["pr_curve_validation"]["ok"])

    def test_rejects_pr_curve_report_without_threshold_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            calibration = tmp / "context_calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "policy_checks": {"overall_pass": True},
                        "runtime_recommendation": {
                            "approve_threshold": 0.25,
                            "block_threshold": 0.85,
                        },
                        "artifact_metadata": {},
                    }
                ),
                encoding="utf-8",
            )
            pr_curve = tmp / "pr_curve.json"
            pr_curve.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {"threshold": 0.01, "precision": 0.2, "recall": 0.9, "pr_auc": 0.22},
                            {"threshold": 0.1, "precision": 0.3, "recall": 0.8, "pr_auc": 0.35},
                            {"threshold": 0.2, "precision": 0.8, "recall": 0.2, "pr_auc": 0.21},
                        ]
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
                "--pr-curve-report-json",
                str(pr_curve),
            ]
            with patch("sys.argv", args), patch.object(
                promote_thresholds,
                "collect_artifact_runtime_metadata",
                return_value={"serialization": {"format": "pickle", "pickle_protocol": 5, "python_minor": "3.11"}},
            ), patch.object(
                promote_thresholds,
                "validate_artifact_compatibility",
                return_value={"ok": True, "failed_checks": [], "checks": {}},
            ):
                with self.assertRaises(ValueError):
                    promote_thresholds.main()


if __name__ == "__main__":
    unittest.main()

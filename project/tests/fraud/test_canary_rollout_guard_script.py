import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.scripts import canary_rollout_guard


class CanaryRolloutGuardScriptTests(unittest.TestCase):
    def test_triggers_rollback_when_thresholds_exceeded_consecutively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            telemetry = tmp_path / "canary_telemetry.json"
            output = tmp_path / "canary_decision.json"
            archive_dir = tmp_path / "archive"
            telemetry.write_text(
                json.dumps(
                    {
                        "windows": [
                            {
                                "window_id": "w1",
                                "endpoint_metrics": [
                                    {
                                        "endpoint_name": "score_transaction",
                                        "error_rate_pct": 0.2,
                                        "p95_latency_ms": 120.0,
                                        "error_categories": {"timeout_upstream": 0.0, "unknown_internal": 0.0},
                                    }
                                ],
                            },
                            {
                                "window_id": "w2",
                                "endpoint_metrics": [
                                    {
                                        "endpoint_name": "score_transaction",
                                        "error_rate_pct": 4.0,
                                        "p95_latency_ms": 390.0,
                                        "error_categories": {"timeout_upstream": 2.5, "unknown_internal": 0.2},
                                    }
                                ],
                            },
                            {
                                "window_id": "w3",
                                "endpoint_metrics": [
                                    {
                                        "endpoint_name": "score_transaction",
                                        "error_rate_pct": 2.2,
                                        "p95_latency_ms": 330.0,
                                        "error_categories": {"timeout_upstream": 1.1, "unknown_internal": 0.3},
                                    }
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = [
                "canary_rollout_guard.py",
                "--telemetry-json",
                str(telemetry),
                "--output-json",
                str(output),
                "--archive-dir",
                str(archive_dir),
                "--artifact-id",
                "fraud-model",
                "--artifact-version",
                "2026.03.18.1",
                "--release-id",
                "rel-20260318-01",
                "--rollback-consecutive-windows",
                "2",
            ]
            with patch("sys.argv", args):
                rc = canary_rollout_guard.main()
            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["decision"], "rollback")
            self.assertTrue(payload["rollback_triggered"])
            self.assertTrue(Path(payload["archive_path"]).exists())

    def test_promotes_when_windows_stay_within_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            telemetry = tmp_path / "canary_telemetry.json"
            output = tmp_path / "canary_decision.json"
            telemetry.write_text(
                json.dumps(
                    {
                        "windows": [
                            {
                                "window_id": "w1",
                                "endpoint_metrics": [
                                    {
                                        "endpoint_name": "wallet_authorize_payment",
                                        "error_rate_pct": 0.2,
                                        "p95_latency_ms": 140.0,
                                        "error_categories": {"timeout_upstream": 0.0, "unknown_internal": 0.0},
                                    }
                                ],
                            },
                            {
                                "window_id": "w2",
                                "endpoint_metrics": [
                                    {
                                        "endpoint_name": "wallet_authorize_payment",
                                        "error_rate_pct": 0.3,
                                        "p95_latency_ms": 160.0,
                                        "error_categories": {"timeout_upstream": 0.1, "unknown_internal": 0.0},
                                    }
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = [
                "canary_rollout_guard.py",
                "--telemetry-json",
                str(telemetry),
                "--output-json",
                str(output),
                "--artifact-id",
                "fraud-model",
                "--artifact-version",
                "2026.03.18.1",
                "--release-id",
                "rel-20260318-01",
            ]
            with patch("sys.argv", args):
                rc = canary_rollout_guard.main()
            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["decision"], "promote")
            self.assertFalse(payload["rollback_triggered"])


if __name__ == "__main__":
    unittest.main()

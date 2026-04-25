import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.scripts import release_gate_check


class ReleaseGateCheckScriptTests(unittest.TestCase):
    def test_fails_when_summary_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = Path(tmp_dir) / "nightly_ops_summary.json"
            args = ["release_gate_check.py", "--ops-summary-json", str(summary)]
            with patch("sys.argv", args):
                rc = release_gate_check.main()
            self.assertEqual(rc, 1)

    def test_fails_when_status_not_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = Path(tmp_dir) / "nightly_ops_summary.json"
            summary.write_text(json.dumps({"status": "failed_calibration"}), encoding="utf-8")
            args = ["release_gate_check.py", "--ops-summary-json", str(summary)]
            with patch("sys.argv", args):
                rc = release_gate_check.main()
            self.assertEqual(rc, 1)

    def test_passes_when_status_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = Path(tmp_dir) / "nightly_ops_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "payload_contract_tests": {"ok": True, "reason": "ok"},
                        "artifact_validation_report": {"ok": True},
                        "benchmark_sla_mode": "enforce",
                        "benchmark": {
                            "ok": True,
                            "stdout": json.dumps(
                                {
                                    "sla_evaluation": {
                                        "score_transaction": {
                                            "real_time_viability": "PASS",
                                            "checks": {"error_rate_pct": {"actual": 0.1}},
                                        },
                                        "wallet_authorize_payment": {
                                            "real_time_viability": "PASS",
                                            "checks": {"error_rate_pct": {"actual": 0.1}},
                                        },
                                    }
                                }
                            ),
                        },
                        "retrain_trigger": {
                            "should_retrain": False,
                            "details": {
                                "warn_drift_streak": 0,
                                "benchmark_sla_fail_streak": 0,
                                "endpoint_error_rate_streaks": {},
                            },
                            "reasons": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = [
                "release_gate_check.py",
                "--ops-summary-json",
                str(summary),
                "--required-startup-streak",
                "1",
            ]
            with patch("sys.argv", args):
                rc = release_gate_check.main()
            self.assertEqual(rc, 0)

    def test_fails_when_payload_contract_tests_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = Path(tmp_dir) / "nightly_ops_summary.json"
            summary.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            args = ["release_gate_check.py", "--ops-summary-json", str(summary)]
            with patch("sys.argv", args):
                rc = release_gate_check.main()
            self.assertEqual(rc, 1)

    def test_fails_when_payload_contract_tests_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = Path(tmp_dir) / "nightly_ops_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "payload_contract_tests": {
                            "ok": False,
                            "reason": "payload_contract_tests_failed",
                            "failing_suites": ["score_transaction_v1_contract"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = ["release_gate_check.py", "--ops-summary-json", str(summary)]
            with patch("sys.argv", args):
                rc = release_gate_check.main()
            self.assertEqual(rc, 1)

    def test_fails_when_status_ok_but_benchmark_sla_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = Path(tmp_dir) / "nightly_ops_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "payload_contract_tests": {"ok": True, "reason": "ok"},
                        "benchmark_sla": {
                            "ok": False,
                            "reason": "benchmark_sla_failed",
                            "failing_endpoints": ["score_transaction"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = ["release_gate_check.py", "--ops-summary-json", str(summary)]
            with patch("sys.argv", args):
                rc = release_gate_check.main()
            self.assertEqual(rc, 1)

    def test_fails_when_uncategorized_endpoint_errors_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = Path(tmp_dir) / "nightly_ops_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "payload_contract_tests": {"ok": True, "reason": "ok"},
                        "benchmark_sla": {
                            "ok": True,
                            "reason": "ok",
                            "failing_endpoints": [],
                            "uncategorized_endpoints": ["wallet_authorize_payment"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = ["release_gate_check.py", "--ops-summary-json", str(summary)]
            with patch("sys.argv", args):
                rc = release_gate_check.main()
            self.assertEqual(rc, 1)


    def test_fails_when_inference_regression_gate_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            summary = tmp_path / "nightly_ops_summary.json"
            summary.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            candidate_report = tmp_path / "inference_candidate_report.json"
            candidate_report.write_text(
                json.dumps(
                    {
                        "status": "hold",
                        "regression_gate": {
                            "blocked": True,
                            "reason": "latency_improved_but_kpi_regressed",
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = [
                "release_gate_check.py",
                "--ops-summary-json",
                str(summary),
                "--inference-candidate-report-json",
                str(candidate_report),
            ]
            with patch("sys.argv", args):
                rc = release_gate_check.main()
            self.assertEqual(rc, 1)

    def test_fails_when_status_ok_and_legacy_benchmark_stdout_has_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = Path(tmp_dir) / "nightly_ops_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "payload_contract_tests": {"ok": True, "reason": "ok"},
                        "benchmark": {
                            "ok": True,
                            "stdout": json.dumps(
                                {
                                    "sla_evaluation": {
                                        "score_transaction": {"real_time_viability": "FAIL"},
                                        "wallet_authorize_payment": {"real_time_viability": "PASS"},
                                    }
                                }
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = ["release_gate_check.py", "--ops-summary-json", str(summary)]
            with patch("sys.argv", args):
                rc = release_gate_check.main()
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()

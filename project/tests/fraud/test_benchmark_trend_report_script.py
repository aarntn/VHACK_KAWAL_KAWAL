import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from project.scripts import benchmark_trend_report


class BenchmarkTrendReportScriptTests(unittest.TestCase):
    def test_load_and_compute_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            payload = {
                "generated_at_utc": "2026-01-01T00:00:00+00:00",
                "endpoints": [
                    {
                        "endpoint_name": "fraud_api",
                        "latency_ms": {"p50": 20, "p95": 40, "p99": 60},
                        "error_rate_pct": 0.0,
                        "throughput_rps": 123,
                    }
                ],
                "sla_evaluation": {
                    "fraud_api": {
                        "real_time_viability": "PASS",
                        "checks": {
                            "p95_latency_ms": {"target_max": 250},
                            "p99_latency_ms": {"target_max": 500},
                            "error_rate_pct": {"target_max": 1.0},
                        },
                    }
                },
            }
            (tmp_path / "latency_benchmark_20260101T000000Z.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            rows = benchmark_trend_report.load_benchmarks(tmp_path, history_limit=5)
            self.assertEqual(len(rows), 1)

            latest = benchmark_trend_report.compute_latest(pd.DataFrame(rows))
            self.assertEqual(len(latest), 1)
            self.assertEqual(latest[0]["endpoint_name"], "fraud_api")

    def test_write_no_data_outputs_emits_json_and_csv_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_json = tmp_path / "latency_trend_report.json"
            output_csv = tmp_path / "latency_trend_report.csv"
            benchmark_dir = tmp_path / "missing-benchmarks"

            payload = benchmark_trend_report.write_no_data_outputs(
                output_json=output_json,
                output_csv=output_csv,
                benchmark_dir=benchmark_dir,
            )

            self.assertEqual(payload["status"], "no_data")
            self.assertEqual(payload["reason"], "no benchmark JSON files found")
            self.assertEqual(payload["benchmark_dir"], str(benchmark_dir))

            json_payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(json_payload["status"], "no_data")
            self.assertEqual(json_payload["reason"], "no benchmark JSON files found")
            self.assertIn("generated_at_utc", json_payload)
            self.assertEqual(json_payload["benchmark_dir"], str(benchmark_dir))

            csv_payload = pd.read_csv(output_csv)
            self.assertEqual(len(csv_payload), 1)
            self.assertEqual(csv_payload.iloc[0]["status"], "no_data")
            self.assertEqual(csv_payload.iloc[0]["reason"], "no benchmark JSON files found")
            self.assertEqual(csv_payload.iloc[0]["benchmark_dir"], str(benchmark_dir))


if __name__ == "__main__":
    unittest.main()

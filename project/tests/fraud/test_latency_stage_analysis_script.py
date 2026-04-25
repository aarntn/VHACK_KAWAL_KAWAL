import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class LatencyStageAnalysisScriptTests(unittest.TestCase):
    def test_latency_stage_analysis_outputs_summary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audit_log = root / "audit.jsonl"
            out_json = root / "latency_stage_analysis.json"
            out_csv = root / "latency_stage_analysis.csv"

            rows = [
                {"stage_timings_ms": {"model_inference_ms": 1.2, "context_scoring_ms": 0.8, "total_pipeline_ms": 3.0}},
                {"stage_timings_ms": {"model_inference_ms": 1.5, "context_scoring_ms": 1.0, "total_pipeline_ms": 3.5}},
            ]
            with audit_log.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            cmd = [
                "python",
                "project/scripts/latency_stage_analysis.py",
                "--audit-log",
                str(audit_log),
                "--output-json",
                str(out_json),
                "--output-csv",
                str(out_csv),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            self.assertTrue(out_json.exists())
            self.assertTrue(out_csv.exists())

            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["records_analyzed"], 2)
            self.assertIn("dominant_stage_by_p95", report)

    def test_latency_stage_analysis_writes_empty_artifacts_when_audit_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audit_log = root / "missing.jsonl"
            out_json = root / "latency_stage_analysis.json"
            out_csv = root / "latency_stage_analysis.csv"

            cmd = [
                "python",
                "project/scripts/latency_stage_analysis.py",
                "--audit-log",
                str(audit_log),
                "--output-json",
                str(out_json),
                "--output-csv",
                str(out_csv),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            self.assertTrue(out_json.exists())
            self.assertTrue(out_csv.exists())

            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["records_analyzed"], 0)
            self.assertFalse(report["has_stage_data"])
            self.assertEqual(report["stages"], [])


if __name__ == "__main__":
    unittest.main()

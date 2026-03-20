import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import random

from project.scripts import explainability_stability_check


class ExplainabilityStabilityCheckScriptTests(unittest.TestCase):
    def test_perturb_payload_updates_ieee_field_names(self) -> None:
        payload = {
            "TransactionAmt": 50.0,
            "TransactionDT": 1000.0,
            "device_risk_score": 0.4,
            "ip_risk_score": 0.3,
            "location_risk_score": 0.2,
            "Time": 123.0,
            "Amount": 321.0,
        }
        perturbed = explainability_stability_check.perturb_payload(payload, random.Random(7), jitter=0.1)

        self.assertNotEqual(perturbed["TransactionAmt"], 50.0)
        self.assertNotEqual(perturbed["TransactionDT"], 1000.0)
        # Legacy aliases are intentionally left untouched.
        self.assertEqual(perturbed["Time"], 123.0)
        self.assertEqual(perturbed["Amount"], 321.0)

    def test_explainability_script_generates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_json = Path(tmp_dir) / "explainability.json"
            out_csv = Path(tmp_dir) / "explainability.csv"

            cmd = [
                "python",
                "project/scripts/explainability_stability_check.py",
                "--samples-per-case",
                "3",
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
            self.assertIn("summary", report)
            self.assertIn("decision_consistency_mean", report["summary"])
            self.assertIn("reason_jaccard_mean", report["summary"])


if __name__ == "__main__":
    unittest.main()

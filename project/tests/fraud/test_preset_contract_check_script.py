import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.scripts import preset_contract_check


class PresetContractCheckScriptTests(unittest.TestCase):
    def test_run_suite_includes_all_presets_and_checks(self) -> None:
        result = preset_contract_check.run_suite(approve_margin=0.01, block_margin=0.005)

        self.assertTrue(result["ok"], result)
        self.assertSetEqual(
            set(result["presets"].keys()),
            {"everyday_purchase", "large_amount", "cross_border", "custom"},
        )
        self.assertTrue(all(result["checks"].values()), result["checks"])

    def test_main_writes_demo_readiness_summary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_json = Path(tmp_dir) / "demo_readiness_summary.json"
            args = [
                "preset_contract_check.py",
                "--output-json",
                str(output_json),
            ]
            with patch("sys.argv", args):
                rc = preset_contract_check.main()

            self.assertEqual(rc, 0)
            self.assertTrue(output_json.exists())
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertTrue(payload.get("ok"), payload)
            self.assertEqual(payload.get("failing_checks"), [])


if __name__ == "__main__":
    unittest.main()

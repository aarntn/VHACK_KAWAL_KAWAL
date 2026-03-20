import subprocess
import sys
import unittest
from pathlib import Path


class ReplayBehaviorProfilesScriptTests(unittest.TestCase):
    def test_script_help_import_bootstrap(self) -> None:
        script = Path("project/scripts/replay_behavior_profiles.py").resolve()
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Replay historical transactions", result.stdout)


if __name__ == "__main__":
    unittest.main()

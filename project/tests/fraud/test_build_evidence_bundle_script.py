import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class BuildEvidenceBundleScriptTests(unittest.TestCase):
    def test_bundle_script_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir) / "bundles"
            cmd = [
                "python",
                "project/scripts/build_evidence_bundle.py",
                "--output-root",
                str(out_root),
                "--bundle-name",
                "test_bundle",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            manifest = out_root / "test_bundle" / "manifest.json"
            self.assertTrue(manifest.exists())

            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertIn("copied_artifacts", data)
            self.assertIn("missing_artifacts", data)
            self.assertIn("archive_path", data)
            self.assertIn("release_tag", data)

    def test_require_complete_fails_when_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir) / "bundles"
            missing_test_log = Path(tmp_dir) / "definitely_missing.log"
            cmd = [
                "python",
                "project/scripts/build_evidence_bundle.py",
                "--output-root",
                str(out_root),
                "--bundle-name",
                "test_bundle_require_complete",
                "--include-test-log",
                str(missing_test_log),
                "--require-complete",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('"status": "incomplete"', result.stdout)

            manifest = out_root / "test_bundle_require_complete" / "manifest.json"
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertIn("full_test_run_log", data["missing_artifacts"])

    def test_release_tag_creates_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir) / "bundles"
            archive_root = Path(tmp_dir) / "archives"
            cmd = [
                "python",
                "project/scripts/build_evidence_bundle.py",
                "--output-root",
                str(out_root),
                "--archive-root",
                str(archive_root),
                "--bundle-name",
                "test_bundle_release_tag",
                "--release-tag",
                "v1.2.3-rc1",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            manifest = out_root / "test_bundle_release_tag" / "manifest.json"
            data = json.loads(manifest.read_text(encoding="utf-8"))
            archive_path = Path(data["archive_path"])
            self.assertTrue(archive_path.exists())
            self.assertEqual(data["release_tag"], "v1.2.3-rc1")


if __name__ == "__main__":
    unittest.main()

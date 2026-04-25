import argparse
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "project" / "outputs" / "evidence_bundles"
DEFAULT_RELEASE_ARCHIVE_ROOT = REPO_ROOT / "project" / "outputs" / "release_artifacts"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_file(pattern: str, root: Path) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect latest evaluation artifacts into one evidence bundle.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bundle-name", default=f"evidence_bundle_{_utc_stamp()}")
    parser.add_argument(
        "--include-test-log",
        type=Path,
        default=REPO_ROOT / "project" / "outputs" / "monitoring" / "full_test_run.log",
        help="Optional path to saved full test output log.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail with non-zero exit when manifest.missing_artifacts is non-empty.",
    )
    parser.add_argument(
        "--release-tag",
        default="",
        help="Optional release tag to archive the bundle as a tar.gz under --archive-root.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_RELEASE_ARCHIVE_ROOT,
        help="Root directory for release-tag archives.",
    )
    return parser.parse_args()


def artifact_map() -> Dict[str, Path | None]:
    return {
        "class_imbalance_metrics_csv": REPO_ROOT / "project" / "outputs" / "figures" / "tables" / "class_imbalance_experiment_metrics.csv",
        "class_imbalance_metrics_md": REPO_ROOT / "project" / "outputs" / "figures" / "tables" / "class_imbalance_experiment_metrics.md",
        "context_calibration_json": REPO_ROOT / "project" / "outputs" / "figures" / "tables" / "context_calibration.json",
        "context_calibration_trials_csv": REPO_ROOT / "project" / "outputs" / "figures" / "tables" / "context_calibration_trials.csv",
        "drift_report_json": REPO_ROOT / "project" / "outputs" / "monitoring" / "drift_report.json",
        "drift_feature_psi_csv": REPO_ROOT / "project" / "outputs" / "monitoring" / "drift_feature_psi.csv",
        "nightly_ops_summary_json": REPO_ROOT / "project" / "outputs" / "monitoring" / "nightly_ops_summary.json",
        "profile_health_json": REPO_ROOT / "project" / "outputs" / "monitoring" / "behavior_profile_health.json",
        "profile_replay_json": REPO_ROOT / "project" / "outputs" / "monitoring" / "profile_replay_summary.json",
        "cohort_kpi_json": REPO_ROOT / "project" / "outputs" / "monitoring" / "cohort_kpi_report.json",
        "cohort_kpi_csv": REPO_ROOT / "project" / "outputs" / "monitoring" / "cohort_kpi_report.csv",
        "latency_trend_json": REPO_ROOT / "project" / "outputs" / "monitoring" / "latency_trend_report.json",
        "latency_trend_csv": REPO_ROOT / "project" / "outputs" / "monitoring" / "latency_trend_report.csv",
        "latest_benchmark_json": _latest_file("latency_benchmark_*.json", REPO_ROOT / "project" / "outputs" / "benchmark"),
        "latest_benchmark_csv": _latest_file("latency_benchmark_*.csv", REPO_ROOT / "project" / "outputs" / "benchmark"),
    }


def copy_if_exists(src: Path | None, dst_dir: Path) -> str | None:
    if src is None or not src.exists():
        return None
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return str(dst)


def sanitize_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in tag).strip("_")


def archive_bundle(bundle_dir: Path, release_tag: str, archive_root: Path) -> Path:
    safe_tag = sanitize_tag(release_tag)
    if not safe_tag:
        raise ValueError("release_tag must contain at least one valid character.")

    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / f"evidence_bundle_{safe_tag}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)
    return archive_path


def main() -> int:
    args = parse_args()
    bundle_dir = args.output_root / args.bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    artifacts = artifact_map()
    copied: Dict[str, str] = {}
    missing: List[str] = []

    if args.include_test_log.exists():
        copied["full_test_run_log"] = copy_if_exists(args.include_test_log, bundle_dir) or ""
    else:
        missing.append("full_test_run_log")

    for key, path in artifacts.items():
        copied_path = copy_if_exists(path, bundle_dir)
        if copied_path is None:
            missing.append(key)
        else:
            copied[key] = copied_path

    archive_path: str | None = None
    if args.release_tag:
        archive_path = str(archive_bundle(bundle_dir, args.release_tag, args.archive_root))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle_dir),
        "release_tag": args.release_tag or None,
        "archive_path": archive_path,
        "copied_artifacts": copied,
        "missing_artifacts": missing,
    }

    manifest_path = bundle_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if args.require_complete and missing:
        print(json.dumps({"bundle_dir": str(bundle_dir), "copied": len(copied), "missing": len(missing), "status": "incomplete"}, indent=2))
        print(f"Saved manifest: {manifest_path}")
        return 1

    print(json.dumps({"bundle_dir": str(bundle_dir), "copied": len(copied), "missing": len(missing), "status": "ok"}, indent=2))
    print(f"Saved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

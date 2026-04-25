import argparse
import hashlib
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.scripts.artifact_compatibility import collect_artifact_runtime_metadata, validate_artifact_compatibility
DEFAULT_CALIBRATION_JSON = REPO_ROOT / "project" / "outputs" / "figures" / "tables" / "context_calibration.json"
DEFAULT_ACTIVE_THRESHOLDS = REPO_ROOT / "project" / "models" / "decision_thresholds.pkl"
DEFAULT_ARCHIVE_DIR = REPO_ROOT / "project" / "outputs" / "threshold_governance"
DEFAULT_PROMOTION_RECORD = DEFAULT_ARCHIVE_DIR / "latest_promotion_record.json"
DEFAULT_MODEL_PATH = REPO_ROOT / "project" / "models" / "final_xgboost_model_promoted_preproc.pkl"
DEFAULT_FEATURE_PATH = REPO_ROOT / "project" / "models" / "feature_columns_promoted_preproc.pkl"
DEFAULT_PREPROCESSING_ARTIFACT_PATH = REPO_ROOT / "project" / "models" / "preprocessing_artifact_promoted.pkl"
DEFAULT_CHECKSUM_MANIFEST = REPO_ROOT / "project" / "models" / "artifact_checksums.sha256"
DEFAULT_PROMOTED_ARTIFACT_MANIFEST = REPO_ROOT / "project" / "models" / "promoted_artifact_manifest.json"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote calibrated thresholds to active runtime file only when policy checks pass, with rollback backup."
    )
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION_JSON)
    parser.add_argument("--active-thresholds", type=Path, default=DEFAULT_ACTIVE_THRESHOLDS)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--promotion-record-json", type=Path, default=DEFAULT_PROMOTION_RECORD)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--preprocessing-artifact-path", type=Path, default=DEFAULT_PREPROCESSING_ARTIFACT_PATH)
    parser.add_argument("--pr-curve-report-json", type=Path, help="Optional PR-curve report JSON used for pre-promotion validation.")
    parser.add_argument("--min-pr-auc", type=float, default=0.0, help="Minimum PR-AUC required from PR-curve report.")
    parser.add_argument("--min-pr-curve-points", type=int, default=3, help="Minimum number of PR-curve points required from report.")
    parser.add_argument("--checksum-manifest", type=Path, default=DEFAULT_CHECKSUM_MANIFEST)
    parser.add_argument("--promoted-artifact-manifest", type=Path, default=DEFAULT_PROMOTED_ARTIFACT_MANIFEST)
    parser.add_argument("--allow-missing-policy-checks", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_thresholds(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Threshold file must contain a dict: {path}")
    return data


def validate_threshold_values(approve: Any, block: Any) -> tuple[float, float]:
    approve_f = float(approve)
    block_f = float(block)
    if not (0.0 <= approve_f < block_f <= 1.0):
        raise ValueError(
            f"Invalid threshold pair: approve_threshold={approve_f}, block_threshold={block_f}."
        )
    return approve_f, block_f


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_pr_curve_points(report: Dict[str, Any]) -> list[Dict[str, float]]:
    if isinstance(report.get("profiles"), list):
        points: list[Dict[str, float]] = []
        for row in report["profiles"]:
            if not isinstance(row, dict):
                continue
            if row.get("threshold") is None:
                continue
            points.append(
                {
                    "threshold": float(row["threshold"]),
                    "precision": float(row.get("precision", 0.0)),
                    "recall": float(row.get("recall", 0.0)),
                    "pr_auc": float(row.get("pr_auc", 0.0)),
                }
            )
        return points

    for key in ("curve", "points", "rows"):
        raw = report.get(key)
        if isinstance(raw, list):
            points = []
            for row in raw:
                if not isinstance(row, dict) or row.get("threshold") is None:
                    continue
                points.append(
                    {
                        "threshold": float(row["threshold"]),
                        "precision": float(row.get("precision", 0.0)),
                        "recall": float(row.get("recall", 0.0)),
                        "pr_auc": float(row.get("pr_auc", 0.0)),
                    }
                )
            if points:
                return points
    return []


def validate_pr_curve_report(
    report: Dict[str, Any],
    *,
    approve_threshold: float,
    block_threshold: float,
    min_pr_auc: float,
    min_points: int,
) -> dict[str, Any]:
    points = _extract_pr_curve_points(report)
    if len(points) < int(min_points):
        raise ValueError(f"PR-curve report has insufficient points: {len(points)} < {min_points}.")

    max_pr_auc = max(float(row.get("pr_auc", 0.0)) for row in points)
    if max_pr_auc < float(min_pr_auc):
        raise ValueError(f"PR-curve report max PR-AUC={max_pr_auc:.6f} below required minimum={min_pr_auc:.6f}.")

    thresholds = sorted(float(row.get("threshold", 0.0)) for row in points)
    if not (thresholds[0] <= approve_threshold <= thresholds[-1]):
        raise ValueError(
            f"Approve threshold {approve_threshold:.6f} is outside PR-curve threshold range "
            f"[{thresholds[0]:.6f}, {thresholds[-1]:.6f}]."
        )
    if not (thresholds[0] <= block_threshold <= thresholds[-1]):
        raise ValueError(
            f"Block threshold {block_threshold:.6f} is outside PR-curve threshold range "
            f"[{thresholds[0]:.6f}, {thresholds[-1]:.6f}]."
        )

    return {
        "ok": True,
        "point_count": len(points),
        "threshold_min": thresholds[0],
        "threshold_max": thresholds[-1],
        "max_pr_auc": max_pr_auc,
    }


def update_checksum_manifest(path: Path, artifact_path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"updated": False, "reason": "missing_checksum_manifest"}
    resolved_artifact = artifact_path.expanduser().resolve()
    target_candidates = {resolved_artifact.name, str(resolved_artifact).replace("\\", "/")}
    try:
        repo_relative = str(resolved_artifact.relative_to(REPO_ROOT)).replace("\\", "/")
        target_candidates.add(repo_relative)
    except ValueError:
        pass
    lines = path.read_text(encoding="utf-8").splitlines()
    new_hash = sha256_file(resolved_artifact)
    updated_lines: list[str] = []
    changed = False
    previous_hash = None
    matched_artifact = None
    for line in lines:
        if not line.strip():
            updated_lines.append(line)
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            updated_lines.append(line)
            continue
        checksum, rel = parts
        normalized_rel = rel.replace("\\", "/")
        if normalized_rel in target_candidates:
            previous_hash = checksum
            matched_artifact = rel
            updated_lines.append(f"{new_hash}  {rel}")
            changed = True
        else:
            updated_lines.append(line)
    if changed:
        path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    return {
        "updated": changed,
        "artifact": matched_artifact,
        "previous_hash": previous_hash,
        "new_hash": new_hash,
    }


def update_promoted_artifact_manifest(path: Path, active_thresholds: Path) -> dict[str, Any]:
    if not path.exists():
        return {"updated": False, "reason": "missing_promoted_artifact_manifest"}
    payload = load_json(path)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return {"updated": False, "reason": "manifest_missing_artifacts"}
    manifest_threshold_file = artifacts.get("threshold_file")
    if not isinstance(manifest_threshold_file, str):
        return {"updated": False, "reason": "manifest_missing_threshold_file"}
    if Path(manifest_threshold_file).name != active_thresholds.name:
        return {"updated": False, "reason": "active_threshold_path_not_referenced"}
    previous_hash = artifacts.get("threshold_file_sha256")
    new_hash = sha256_file(active_thresholds)
    artifacts["threshold_file_sha256"] = new_hash
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"updated": True, "previous_hash": previous_hash, "new_hash": new_hash}


def extract_recommended_thresholds(calibration: Dict[str, Any], allow_missing_policy_checks: bool) -> Dict[str, float]:
    policy_checks = calibration.get("policy_checks")
    if policy_checks is None and not allow_missing_policy_checks:
        raise ValueError("Calibration JSON missing 'policy_checks'. Refusing promotion.")

    if policy_checks is not None:
        if not bool(policy_checks.get("overall_pass", False)):
            raise ValueError("Calibration policy checks failed (overall_pass=false). Refusing promotion.")

    recommendation = calibration.get("runtime_recommendation") or {}
    if not recommendation:
        raise ValueError("Calibration JSON missing 'runtime_recommendation'.")

    approve, block = validate_threshold_values(
        recommendation.get("approve_threshold"),
        recommendation.get("block_threshold"),
    )
    return {
        "approve_threshold": approve,
        "block_threshold": block,
    }


def main() -> int:
    args = parse_args()
    ts = utc_timestamp()

    if not args.calibration_json.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {args.calibration_json}")

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    args.promotion_record_json.parent.mkdir(parents=True, exist_ok=True)

    calibration = load_json(args.calibration_json)
    new_thresholds = extract_recommended_thresholds(calibration, args.allow_missing_policy_checks)
    pr_curve_validation = None
    if args.pr_curve_report_json is not None:
        if not args.pr_curve_report_json.exists():
            raise FileNotFoundError(f"PR-curve report JSON not found: {args.pr_curve_report_json}")
        pr_curve_payload = load_json(args.pr_curve_report_json)
        pr_curve_validation = validate_pr_curve_report(
            pr_curve_payload,
            approve_threshold=float(new_thresholds["approve_threshold"]),
            block_threshold=float(new_thresholds["block_threshold"]),
            min_pr_auc=float(args.min_pr_auc),
            min_points=int(args.min_pr_curve_points),
        )
    expected_metadata = calibration.get("artifact_metadata")
    runtime_metadata = collect_artifact_runtime_metadata(
        model_path=args.model_path,
        feature_path=args.feature_path,
        preprocessing_artifact_path=args.preprocessing_artifact_path,
    )
    compatibility_report = validate_artifact_compatibility(
        expected_metadata=expected_metadata if isinstance(expected_metadata, dict) else None,
        runtime_metadata=runtime_metadata,
    )
    if not compatibility_report.get("ok", False):
        raise ValueError(
            "Artifact compatibility checks failed. Refusing threshold promotion: "
            + ",".join(compatibility_report.get("failed_checks", []))
        )
    previous_thresholds = load_thresholds(args.active_thresholds)

    backup_file = None
    if args.active_thresholds.exists():
        backup_file = args.archive_dir / f"decision_thresholds.backup.{ts}.pkl"
        shutil.copy2(args.active_thresholds, backup_file)

    staged_file = args.active_thresholds.with_suffix(args.active_thresholds.suffix + ".tmp")
    with staged_file.open("wb") as f:
        pickle.dump(new_thresholds, f)
    staged_file.replace(args.active_thresholds)

    calibration_copy = args.archive_dir / f"context_calibration.promoted.{ts}.json"
    shutil.copy2(args.calibration_json, calibration_copy)
    pr_curve_copy = None
    if args.pr_curve_report_json is not None:
        pr_curve_copy = args.archive_dir / f"pr_curve_report.promoted.{ts}.json"
        shutil.copy2(args.pr_curve_report_json, pr_curve_copy)

    checksum_backup = None
    if args.checksum_manifest.exists():
        checksum_backup = args.archive_dir / f"artifact_checksums.backup.{ts}.sha256"
        shutil.copy2(args.checksum_manifest, checksum_backup)
    checksum_update = update_checksum_manifest(args.checksum_manifest, args.active_thresholds)

    promoted_manifest_backup = None
    if args.promoted_artifact_manifest.exists():
        promoted_manifest_backup = args.archive_dir / f"promoted_artifact_manifest.backup.{ts}.json"
        shutil.copy2(args.promoted_artifact_manifest, promoted_manifest_backup)
    promoted_manifest_update = update_promoted_artifact_manifest(args.promoted_artifact_manifest, args.active_thresholds)

    rollback_metadata = {
        "timestamp_utc": ts,
        "rollback_thresholds_file": str(backup_file) if backup_file is not None else None,
        "rollback_calibration_file": str(calibration_copy),
        "rollback_pr_curve_report_file": str(pr_curve_copy) if pr_curve_copy is not None else None,
        "rollback_checksum_manifest_file": str(checksum_backup) if checksum_backup is not None else None,
        "rollback_promoted_manifest_file": str(promoted_manifest_backup) if promoted_manifest_backup is not None else None,
        "previous_thresholds": previous_thresholds,
        "new_thresholds": new_thresholds,
        "checksum_update": checksum_update,
        "promoted_manifest_update": promoted_manifest_update,
    }
    rollback_metadata_path = args.archive_dir / f"rollback_metadata.{ts}.json"
    rollback_metadata_path.write_text(json.dumps(rollback_metadata, indent=2), encoding="utf-8")

    promotion_record = {
        "timestamp_utc": ts,
        "status": "promoted",
        "calibration_json": str(args.calibration_json),
        "calibration_copy": str(calibration_copy),
        "policy_checks": calibration.get("policy_checks"),
        "pr_curve_validation": pr_curve_validation,
        "artifact_compatibility": compatibility_report,
        "previous_thresholds": previous_thresholds,
        "new_thresholds": new_thresholds,
        "rollback_thresholds_file": str(backup_file) if backup_file is not None else None,
        "rollback_metadata_file": str(rollback_metadata_path),
        "checksum_update": checksum_update,
        "promoted_manifest_update": promoted_manifest_update,
    }
    with args.promotion_record_json.open("w", encoding="utf-8") as f:
        json.dump(promotion_record, f, indent=2)

    print(json.dumps({"status": "promoted", "record": str(args.promotion_record_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

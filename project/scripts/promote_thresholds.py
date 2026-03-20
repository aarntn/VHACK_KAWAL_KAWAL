import argparse
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from artifact_compatibility import collect_artifact_runtime_metadata, validate_artifact_compatibility

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_JSON = REPO_ROOT / "project" / "outputs" / "figures" / "tables" / "context_calibration.json"
DEFAULT_ACTIVE_THRESHOLDS = REPO_ROOT / "project" / "models" / "decision_thresholds.pkl"
DEFAULT_ARCHIVE_DIR = REPO_ROOT / "project" / "outputs" / "threshold_governance"
DEFAULT_PROMOTION_RECORD = DEFAULT_ARCHIVE_DIR / "latest_promotion_record.json"
DEFAULT_MODEL_PATH = REPO_ROOT / "project" / "models" / "final_xgboost_model_promoted_preproc.pkl"
DEFAULT_FEATURE_PATH = REPO_ROOT / "project" / "models" / "feature_columns_promoted_preproc.pkl"
DEFAULT_PREPROCESSING_ARTIFACT_PATH = REPO_ROOT / "project" / "models" / "preprocessing_artifact_promoted.pkl"


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

    promotion_record = {
        "timestamp_utc": ts,
        "status": "promoted",
        "calibration_json": str(args.calibration_json),
        "calibration_copy": str(calibration_copy),
        "policy_checks": calibration.get("policy_checks"),
        "artifact_compatibility": compatibility_report,
        "previous_thresholds": previous_thresholds,
        "new_thresholds": new_thresholds,
        "rollback_thresholds_file": str(backup_file) if backup_file is not None else None,
    }
    with args.promotion_record_json.open("w", encoding="utf-8") as f:
        json.dump(promotion_record, f, indent=2)

    print(json.dumps({"status": "promoted", "record": str(args.promotion_record_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from artifact_compatibility import collect_artifact_runtime_metadata, validate_artifact_compatibility


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate model/feature/preprocessing artifacts against metadata expectations.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--feature-path", type=Path, required=True)
    parser.add_argument("--preprocessing-artifact-path", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--promotion-record-json", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("project/outputs/monitoring/artifact_validation_report.json"),
    )
    return parser.parse_args()


def _load_expected_metadata(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    if args.promotion_record_json and args.promotion_record_json.exists():
        payload = json.loads(args.promotion_record_json.read_text(encoding="utf-8"))
        metadata = ((payload.get("artifact_compatibility") or {}).get("expected_metadata"))
        if isinstance(metadata, dict):
            return metadata, f"promotion_record:{args.promotion_record_json}"

    if args.calibration_json and args.calibration_json.exists():
        payload = json.loads(args.calibration_json.read_text(encoding="utf-8"))
        metadata = payload.get("artifact_metadata")
        if isinstance(metadata, dict):
            return metadata, f"calibration_json:{args.calibration_json}"

    return None, "none"


def main() -> int:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    expected_metadata, expected_source = _load_expected_metadata(args)
    runtime_metadata = collect_artifact_runtime_metadata(
        model_path=args.model_path,
        feature_path=args.feature_path,
        preprocessing_artifact_path=args.preprocessing_artifact_path,
    )
    compatibility = validate_artifact_compatibility(expected_metadata=expected_metadata, runtime_metadata=runtime_metadata)

    report = {
        "generated_at_utc": _utc_now_iso(),
        "ok": compatibility.get("ok", False),
        "expected_metadata_source": expected_source,
        "failed_checks": compatibility.get("failed_checks", []),
        "checks": compatibility.get("checks", {}),
        "runtime_metadata": compatibility.get("runtime_metadata"),
        "expected_metadata": compatibility.get("expected_metadata"),
    }
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "output_json": str(args.output_json), "failed_checks": report["failed_checks"]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

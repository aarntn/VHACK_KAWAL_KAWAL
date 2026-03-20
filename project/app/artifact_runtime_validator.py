from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromotedArtifactManifest:
    manifest_path: Path
    model_file: Path
    feature_file: Path
    threshold_file: Path
    preprocessing_bundle_file: Path
    model_metadata_family: str
    preprocessing_bundle_version: str
    expected_feature_schema_hash: str


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Promoted artifact manifest field '{field_name}' must be a non-empty string")
    return value.strip()


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_promoted_artifact_manifest(manifest_path: str | Path) -> PromotedArtifactManifest:
    resolved_manifest = Path(manifest_path).expanduser().resolve()
    if not resolved_manifest.exists():
        raise FileNotFoundError(f"Promoted artifact manifest not found: {resolved_manifest}")

    with resolved_manifest.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    artifacts = payload.get("artifacts")
    metadata = payload.get("metadata")
    if not isinstance(artifacts, dict):
        raise ValueError("Promoted artifact manifest is missing object field: artifacts")
    if not isinstance(metadata, dict):
        raise ValueError("Promoted artifact manifest is missing object field: metadata")

    return PromotedArtifactManifest(
        manifest_path=resolved_manifest,
        model_file=_resolve_path(resolved_manifest.parent, _require_string(artifacts.get("model_file"), "artifacts.model_file")),
        feature_file=_resolve_path(resolved_manifest.parent, _require_string(artifacts.get("feature_file"), "artifacts.feature_file")),
        threshold_file=_resolve_path(resolved_manifest.parent, _require_string(artifacts.get("threshold_file"), "artifacts.threshold_file")),
        preprocessing_bundle_file=_resolve_path(
            resolved_manifest.parent,
            _require_string(artifacts.get("preprocessing_bundle_file"), "artifacts.preprocessing_bundle_file"),
        ),
        model_metadata_family=_require_string(metadata.get("model_metadata_family"), "metadata.model_metadata_family"),
        preprocessing_bundle_version=_require_string(
            metadata.get("preprocessing_bundle_version"),
            "metadata.preprocessing_bundle_version",
        ),
        expected_feature_schema_hash=_require_string(
            metadata.get("expected_feature_schema_hash"),
            "metadata.expected_feature_schema_hash",
        ),
    )


def compute_feature_schema_hash(feature_columns: list[str]) -> str:
    canonical = [str(col) for col in feature_columns]
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_promoted_artifacts(
    manifest: PromotedArtifactManifest,
    *,
    feature_columns: list[str],
    preprocessing_bundle: Any,
) -> dict[str, Any]:
    runtime_bundle_version = str(getattr(preprocessing_bundle, "bundle_version", ""))
    runtime_feature_schema_hash = compute_feature_schema_hash(feature_columns)

    checks = {
        "model_metadata_family_ieee": {
            "expected": "ieee",
            "actual": manifest.model_metadata_family,
            "ok": manifest.model_metadata_family.lower() == "ieee",
        },
        "preprocessing_bundle_version_match": {
            "expected": manifest.preprocessing_bundle_version,
            "actual": runtime_bundle_version,
            "ok": runtime_bundle_version == manifest.preprocessing_bundle_version,
        },
        "feature_schema_hash_match": {
            "expected": manifest.expected_feature_schema_hash,
            "actual": runtime_feature_schema_hash,
            "ok": runtime_feature_schema_hash == manifest.expected_feature_schema_hash,
        },
    }

    failed_checks = [name for name, details in checks.items() if not details["ok"]]
    return {
        "ok": not failed_checks,
        "manifest_path": str(manifest.manifest_path),
        "model_metadata_family": manifest.model_metadata_family,
        "preprocessing_bundle_version": {
            "expected": manifest.preprocessing_bundle_version,
            "runtime": runtime_bundle_version,
        },
        "feature_schema_hash": {
            "expected": manifest.expected_feature_schema_hash,
            "runtime": runtime_feature_schema_hash,
        },
        "failed_checks": failed_checks,
        "checks": checks,
    }


def raise_on_failed_validation(validation_report: dict[str, Any]) -> None:
    if validation_report.get("ok", False):
        return
    failed = validation_report.get("failed_checks", [])
    detail = ", ".join(failed) if failed else "unknown_check_failure"
    raise RuntimeError(
        "Promoted artifact validation failed; refusing startup to avoid degraded traffic. "
        f"failed_checks={detail}"
    )

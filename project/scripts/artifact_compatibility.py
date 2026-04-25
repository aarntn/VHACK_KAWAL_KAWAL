from __future__ import annotations

import hashlib
import json
import pickle
import pickletools
import platform
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.preprocessing import load_preprocessing_bundle


ARTIFACT_METADATA_SCHEMA_VERSION = "1.0"
SUPPORTED_SERIALIZATION_FORMATS = {"pickle"}


def _optional_lib_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def runtime_library_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "xgboost": _optional_lib_version("xgboost"),
        "scikit_learn": _optional_lib_version("sklearn"),
        "numpy": _optional_lib_version("numpy"),
        "pandas": _optional_lib_version("pandas"),
    }


def extract_pickle_protocol(path: str | Path) -> int | None:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as handle:
        raw = handle.read()
    try:
        for opcode, arg, _ in pickletools.genops(raw):
            if opcode.name == "PROTO":
                return int(arg)
            break
    except Exception:
        return None
    if raw.startswith(b"\x80") and len(raw) > 1:
        return int(raw[1])
    return 0


def compute_feature_schema_hash(feature_columns: list[str]) -> str:
    canonical = [str(col) for col in feature_columns]
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_python_minor(version: str | None) -> str | None:
    if not version:
        return None
    pieces = str(version).split(".")
    if len(pieces) < 2:
        return None
    return f"{pieces[0]}.{pieces[1]}"


def collect_artifact_runtime_metadata(
    model_path: str | Path,
    feature_path: str | Path,
    preprocessing_artifact_path: str | Path,
) -> dict[str, Any]:
    resolved_model = Path(model_path).expanduser().resolve()
    resolved_features = Path(feature_path).expanduser().resolve()
    resolved_bundle = Path(preprocessing_artifact_path).expanduser().resolve()

    with resolved_features.open("rb") as handle:
        feature_columns = pickle.load(handle)
    if not isinstance(feature_columns, list):
        raise TypeError(f"Feature artifact must be a list[str], got {type(feature_columns)!r}")

    bundle = load_preprocessing_bundle(resolved_bundle)
    bundle_version = getattr(bundle, "bundle_version", None)

    model_protocol = extract_pickle_protocol(resolved_model)

    model_class_name = None
    try:
        with resolved_model.open("rb") as handle:
            model_obj = pickle.load(handle)
        model_class_name = f"{model_obj.__class__.__module__}.{model_obj.__class__.__name__}"
    except Exception:
        model_class_name = None

    return {
        "schema_version": ARTIFACT_METADATA_SCHEMA_VERSION,
        "model_path": str(resolved_model),
        "feature_path": str(resolved_features),
        "preprocessing_artifact_path": str(resolved_bundle),
        "feature_schema_hash": compute_feature_schema_hash(feature_columns),
        "feature_count": len(feature_columns),
        "preprocessing_bundle_version": bundle_version,
        "serialization": {
            "format": "pickle",
            "pickle_protocol": model_protocol,
            "python_minor": _normalize_python_minor(platform.python_version()),
        },
        "library_versions": runtime_library_versions(),
        "model_class": model_class_name,
    }


def validate_artifact_compatibility(
    expected_metadata: dict[str, Any] | None,
    runtime_metadata: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    runtime_serialization = runtime_metadata.get("serialization") or {}
    expected = expected_metadata or {}
    expected_serialization = expected.get("serialization") or {}

    fmt = runtime_serialization.get("format")
    protocol = runtime_serialization.get("pickle_protocol")
    checks["serialization_format_supported"] = {
        "required_formats": sorted(SUPPORTED_SERIALIZATION_FORMATS),
        "actual": fmt,
        "passed": fmt in SUPPORTED_SERIALIZATION_FORMATS,
    }
    checks["pickle_protocol_supported"] = {
        "required_max": pickle.HIGHEST_PROTOCOL,
        "actual": protocol,
        "passed": isinstance(protocol, int) and protocol <= pickle.HIGHEST_PROTOCOL,
    }

    runtime_py_minor = runtime_serialization.get("python_minor")
    current_runtime_py_minor = _normalize_python_minor((runtime_metadata.get("library_versions") or {}).get("python"))
    checks["python_minor_runtime_compatible"] = {
        "required": current_runtime_py_minor,
        "actual": runtime_py_minor,
        "passed": bool(
            current_runtime_py_minor
            and runtime_py_minor
            and current_runtime_py_minor == runtime_py_minor
        ),
    }

    serialization_mismatches: dict[str, dict[str, Any]] = {}
    for key in ("format", "pickle_protocol", "python_minor"):
        expected_value = expected_serialization.get(key)
        if expected_value is None:
            continue
        actual_value = runtime_serialization.get(key)
        if actual_value != expected_value:
            serialization_mismatches[key] = {"expected": expected_value, "actual": actual_value}
    checks["serialization_match"] = {
        "expected": expected_serialization,
        "actual": runtime_serialization,
        "mismatches": serialization_mismatches,
        "passed": len(serialization_mismatches) == 0,
    }

    expected_libs = expected.get("library_versions") or {}
    runtime_libs = runtime_metadata.get("library_versions") or {}
    lib_mismatches: dict[str, dict[str, Any]] = {}
    for key, expected_value in expected_libs.items():
        if expected_value is None:
            continue
        actual = runtime_libs.get(key)
        if actual != expected_value:
            lib_mismatches[key] = {"expected": expected_value, "actual": actual}
    checks["library_versions_match"] = {
        "expected": expected_libs,
        "actual": runtime_libs,
        "mismatches": lib_mismatches,
        "passed": len(lib_mismatches) == 0,
    }

    expected_feature_hash = expected.get("feature_schema_hash")
    runtime_feature_hash = runtime_metadata.get("feature_schema_hash")
    checks["feature_schema_hash_match"] = {
        "expected": expected_feature_hash,
        "actual": runtime_feature_hash,
        "passed": bool(expected_feature_hash and runtime_feature_hash and expected_feature_hash == runtime_feature_hash),
    }

    expected_bundle_version = expected.get("preprocessing_bundle_version")
    runtime_bundle_version = runtime_metadata.get("preprocessing_bundle_version")
    checks["preprocessing_bundle_version_match"] = {
        "expected": expected_bundle_version,
        "actual": runtime_bundle_version,
        "passed": bool(
            expected_bundle_version
            and runtime_bundle_version
            and str(expected_bundle_version) == str(runtime_bundle_version)
        ),
    }

    expected_model_class = expected.get("model_class")
    runtime_model_class = runtime_metadata.get("model_class")
    checks["model_class_match"] = {
        "expected": expected_model_class,
        "actual": runtime_model_class,
        "passed": bool(
            expected_model_class
            and runtime_model_class
            and str(expected_model_class) == str(runtime_model_class)
        ),
    }

    if expected_metadata is None:
        checks["expected_metadata_present"] = {
            "passed": False,
            "reason": "missing_expected_metadata",
        }

    required_expected_fields = [
        "library_versions",
        "feature_schema_hash",
        "preprocessing_bundle_version",
        "serialization",
        "model_class",
    ]
    missing_expected_fields = [field for field in required_expected_fields if field not in expected]
    checks["expected_metadata_fields_present"] = {
        "required": required_expected_fields,
        "missing": missing_expected_fields,
        "passed": len(missing_expected_fields) == 0,
    }

    failed_checks = [name for name, payload in checks.items() if not payload.get("passed", False)]
    return {
        "ok": len(failed_checks) == 0,
        "failed_checks": failed_checks,
        "checks": checks,
        "runtime_metadata": runtime_metadata,
        "expected_metadata": expected_metadata,
    }

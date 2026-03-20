# Canonical runtime service for demos/repo review.
import os
from contextlib import asynccontextmanager
import threading
import queue
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any
from pathlib import Path
import numpy as np
import pickle
import time
import json
import hashlib
import hmac
import uuid
import contextvars
import logging
from datetime import datetime, timezone
import pandas as pd

from .behavior_profile import BehaviorProfiler
from .aggregate_ingestion import (
    AggregateUpdateEvent,
    QueueAggregateIngestionAdapter,
    MockStreamAggregateIngestionAdapter,
)
from .contracts import PAYLOAD_SCHEMA_VERSION as IEEE_PAYLOAD_SCHEMA_VERSION
from .schema_spec import (
    CONTEXT_FEATURE_FIELDS,
    EXPECTED_RAW_MODEL_FEATURES,
    REQUIRED_SERVING_INPUT_FIELDS,
    serving_schema_summary,
)
from .profile_store import InMemoryProfileStore, SQLiteProfileStore, RedisProfileStore
from .feature_store import (
    CassandraFeatureStore,
    FeatureStoreConfig,
    InMemoryFeatureStore,
    RedisFeatureStore,
)
from project.data.feature_registry import map_to_canonical_features  # noqa: F401
from project.data.entity_aggregation import (
    EntitySmoothingConfig,
    EntitySmoothingState,
    validate_smoothing_method,
)
from project.data.entity_identity import build_entity_id
from project.data.preprocessing import (
    load_preprocessing_bundle,
    prepare_preprocessing_inputs,  # noqa: F401
    transform_runtime_record_with_bundle,
    transform_with_bundle,  # noqa: F401
)
from .artifact_runtime_validator import (
    load_promoted_artifact_manifest,
    validate_promoted_artifacts,
    raise_on_failed_validation,
)
from .rules import (
    RuleStateStore,
    SegmentThresholds,
    apply_segmented_decision,
    compute_user_segment,
    determine_step_up_action,
    evaluate_hard_rules,
    validate_segment_thresholds,
)
from .domain_exceptions import (
    ArtifactSchemaMismatchError,
    ConfigurationError,
    InvalidRiskScoreError,
    ReviewQueueRecordNotFoundError,
    UnknownTransactionTypeError,
    UserProfileMismatchError,
)
from .errors import (
    ApiErrorResponse,
    DomainError,
    build_api_error_response,
    build_validation_error_details,
    classify_known_error,
    extract_http_exception_detail,
)
from .inference_backends import create_inference_backend
from .schemas import DecisionSource, ScoreTransactionRequest, ScoreTransactionResponse
from .alerts import AlertEvent, get_alert_notifier

logger = logging.getLogger(__name__)
alert_notifier = get_alert_notifier()

# ============================================================
# CONFIG
# ============================================================
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_FILE = Path(
    os.getenv("FRAUD_MODEL_FILE", str(PROJECT_ROOT / "models" / "final_xgboost_model.pkl"))
)
FEATURE_FILE = Path(
    os.getenv("FRAUD_FEATURE_FILE", str(PROJECT_ROOT / "models" / "feature_columns.pkl"))
)
THRESHOLD_FILE = Path(
    os.getenv("FRAUD_THRESHOLD_FILE", str(PROJECT_ROOT / "models" / "decision_thresholds.pkl"))
)
USE_PREPROCESSING_INFERENCE = os.getenv("FRAUD_USE_PREPROCESSING_INFERENCE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CONTEXT_ONLY_BASE_SCORE = float(os.getenv("FRAUD_CONTEXT_ONLY_BASE_SCORE", "0.23"))
ANCHOR_PROFILE = os.getenv("FRAUD_ANCHOR_PROFILE", "balanced").strip().lower()
ANCHOR_PROFILE_DEFAULTS: Dict[str, Dict[str, float]] = {
    # Keeps safe/large/risky preset bands stable for demos with dynamic anchor.
    "demo_stable": {
        "ANCHOR_BASELINE_MERCHANT": 0.21,
        "ANCHOR_BASELINE_P2P": 0.24,
        "ANCHOR_BASELINE_CASH_IN": 0.23,
        "ANCHOR_BASELINE_CASH_OUT": 0.30,
        "ANCHOR_CROSS_BORDER_BOOST": 0.08,
        "ANCHOR_SIM_CHANGE_BOOST": 0.04,
        "ANCHOR_NEW_ACCOUNT_LT7_BOOST": 0.03,
        "ANCHOR_NEW_ACCOUNT_LT3_BOOST": 0.05,
        "ANCHOR_DEVICE_RISK_WEIGHT": 0.04,
        "ANCHOR_IP_RISK_WEIGHT": 0.04,
        "ANCHOR_LOCATION_RISK_WEIGHT": 0.02,
        "ANCHOR_ESTABLISHED_MERCHANT_DISCOUNT": 0.03,
        "ANCHOR_VERY_ESTABLISHED_MERCHANT_DISCOUNT": 0.02,
        "ANCHOR_MIN": 0.17,
        "ANCHOR_MAX": 0.60,
    },
    # Backwards-compatible profile close to previous behavior.
    "balanced": {
        "ANCHOR_BASELINE_MERCHANT": CONTEXT_ONLY_BASE_SCORE,
        "ANCHOR_BASELINE_P2P": 0.25,
        "ANCHOR_BASELINE_CASH_IN": 0.24,
        "ANCHOR_BASELINE_CASH_OUT": 0.30,
        "ANCHOR_CROSS_BORDER_BOOST": 0.08,
        "ANCHOR_SIM_CHANGE_BOOST": 0.04,
        "ANCHOR_NEW_ACCOUNT_LT7_BOOST": 0.03,
        "ANCHOR_NEW_ACCOUNT_LT3_BOOST": 0.05,
        "ANCHOR_DEVICE_RISK_WEIGHT": 0.04,
        "ANCHOR_IP_RISK_WEIGHT": 0.04,
        "ANCHOR_LOCATION_RISK_WEIGHT": 0.02,
        "ANCHOR_ESTABLISHED_MERCHANT_DISCOUNT": 0.04,
        "ANCHOR_VERY_ESTABLISHED_MERCHANT_DISCOUNT": 0.02,
        "ANCHOR_MIN": 0.18,
        "ANCHOR_MAX": 0.60,
    },
}


def _anchor_profile_default(key: str, fallback: float) -> float:
    profile_defaults = ANCHOR_PROFILE_DEFAULTS.get(ANCHOR_PROFILE, ANCHOR_PROFILE_DEFAULTS["balanced"])
    return float(profile_defaults.get(key, fallback))


DYNAMIC_IMPUTED_ANCHOR_ENABLED = os.getenv("FRAUD_DYNAMIC_IMPUTED_ANCHOR", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ANCHOR_BASELINE_MERCHANT = float(
    os.getenv("FRAUD_ANCHOR_BASE_MERCHANT", str(_anchor_profile_default("ANCHOR_BASELINE_MERCHANT", CONTEXT_ONLY_BASE_SCORE)))
)
ANCHOR_BASELINE_P2P = float(os.getenv("FRAUD_ANCHOR_BASE_P2P", str(_anchor_profile_default("ANCHOR_BASELINE_P2P", 0.25))))
ANCHOR_BASELINE_CASH_IN = float(
    os.getenv("FRAUD_ANCHOR_BASE_CASH_IN", str(_anchor_profile_default("ANCHOR_BASELINE_CASH_IN", 0.24)))
)
ANCHOR_BASELINE_CASH_OUT = float(
    os.getenv("FRAUD_ANCHOR_BASE_CASH_OUT", str(_anchor_profile_default("ANCHOR_BASELINE_CASH_OUT", 0.30)))
)
ANCHOR_CROSS_BORDER_BOOST = float(
    os.getenv("FRAUD_ANCHOR_CROSS_BORDER_BOOST", str(_anchor_profile_default("ANCHOR_CROSS_BORDER_BOOST", 0.08)))
)
ANCHOR_SIM_CHANGE_BOOST = float(
    os.getenv("FRAUD_ANCHOR_SIM_CHANGE_BOOST", str(_anchor_profile_default("ANCHOR_SIM_CHANGE_BOOST", 0.04)))
)
ANCHOR_NEW_ACCOUNT_LT7_BOOST = float(
    os.getenv("FRAUD_ANCHOR_NEW_ACCOUNT_LT7_BOOST", str(_anchor_profile_default("ANCHOR_NEW_ACCOUNT_LT7_BOOST", 0.03)))
)
ANCHOR_NEW_ACCOUNT_LT3_BOOST = float(
    os.getenv("FRAUD_ANCHOR_NEW_ACCOUNT_LT3_BOOST", str(_anchor_profile_default("ANCHOR_NEW_ACCOUNT_LT3_BOOST", 0.05)))
)
ANCHOR_DEVICE_RISK_WEIGHT = float(
    os.getenv("FRAUD_ANCHOR_DEVICE_RISK_WEIGHT", str(_anchor_profile_default("ANCHOR_DEVICE_RISK_WEIGHT", 0.04)))
)
ANCHOR_IP_RISK_WEIGHT = float(
    os.getenv("FRAUD_ANCHOR_IP_RISK_WEIGHT", str(_anchor_profile_default("ANCHOR_IP_RISK_WEIGHT", 0.04)))
)
ANCHOR_LOCATION_RISK_WEIGHT = float(
    os.getenv("FRAUD_ANCHOR_LOCATION_RISK_WEIGHT", str(_anchor_profile_default("ANCHOR_LOCATION_RISK_WEIGHT", 0.02)))
)
ANCHOR_ESTABLISHED_MERCHANT_DISCOUNT = float(
    os.getenv(
        "FRAUD_ANCHOR_ESTABLISHED_MERCHANT_DISCOUNT",
        str(_anchor_profile_default("ANCHOR_ESTABLISHED_MERCHANT_DISCOUNT", 0.04)),
    )
)
ANCHOR_VERY_ESTABLISHED_MERCHANT_DISCOUNT = float(
    os.getenv(
        "FRAUD_ANCHOR_VERY_ESTABLISHED_MERCHANT_DISCOUNT",
        str(_anchor_profile_default("ANCHOR_VERY_ESTABLISHED_MERCHANT_DISCOUNT", 0.02)),
    )
)
ANCHOR_MIN = float(os.getenv("FRAUD_ANCHOR_MIN", str(_anchor_profile_default("ANCHOR_MIN", 0.18))))
ANCHOR_MAX = float(os.getenv("FRAUD_ANCHOR_MAX", str(_anchor_profile_default("ANCHOR_MAX", 0.60))))
PREPROCESSING_BUNDLE_FILE = Path(
    os.getenv(
        "FRAUD_PREPROCESSING_BUNDLE_FILE",
        str(PROJECT_ROOT / "models" / "preprocessing_artifact_promoted.pkl"),
    )
)
PROMOTED_ARTIFACT_MANIFEST_FILE = Path(
    os.getenv(
        "FRAUD_ARTIFACT_MANIFEST_FILE",
        str(PROJECT_ROOT / "models" / "promoted_artifact_manifest.json"),
    )
)
CONTEXT_CALIBRATION_FILE = os.getenv("CONTEXT_CALIBRATION_FILE")

API_VERSION = "4.0.0"
PAYLOAD_SCHEMA_VERSION = IEEE_PAYLOAD_SCHEMA_VERSION
MODEL_NAME = "Hybrid Fraud Risk Engine with Behavioral Profiling"
MODEL_VERSION = "xgb-v1"
MODEL_INFERENCE_THREADS = max(1, int(os.getenv("MODEL_INFERENCE_THREADS", "1")))
FRAUD_INFERENCE_BACKEND = os.getenv("FRAUD_INFERENCE_BACKEND", "xgboost_inplace_predict").strip().lower()

DEFAULT_APPROVE_THRESHOLD = 0.30
DEFAULT_BLOCK_THRESHOLD = 0.90

CONTEXT_ADJUSTMENT_MAX = float(os.getenv("CONTEXT_ADJUSTMENT_MAX", "0.30"))

ENTITY_SMOOTHING_METHOD = validate_smoothing_method(os.getenv("FRAUD_ENTITY_SMOOTHING_METHOD", "none"))
ENTITY_SMOOTHING_MIN_HISTORY = max(1, int(os.getenv("FRAUD_ENTITY_SMOOTHING_MIN_HISTORY", "2")))
ENTITY_SMOOTHING_EMA_ALPHA = float(os.getenv("FRAUD_ENTITY_SMOOTHING_EMA_ALPHA", "0.3"))
ENTITY_SMOOTHING_BLEND_ALPHA = float(os.getenv("FRAUD_ENTITY_SMOOTHING_BLEND_ALPHA", "0.5"))
ENTITY_SMOOTHING_BLEND_CAP = float(os.getenv("FRAUD_ENTITY_SMOOTHING_BLEND_CAP", "0.25"))
ENTITY_SMOOTHING_FALLBACK = os.getenv("FRAUD_ENTITY_SMOOTHING_FALLBACK", "raw").strip().lower()
if ENTITY_SMOOTHING_FALLBACK not in {"raw", "zero"}:
    raise ConfigurationError("FRAUD_ENTITY_SMOOTHING_FALLBACK must be one of: raw, zero")

CONTEXT_WEIGHT_DEFAULTS = {
    "device_risk_weight": 0.05,
    "ip_risk_weight": 0.05,
    "location_risk_weight": 0.05,
    "amount_over_200": 0.02,
    "amount_over_1000": 0.04,
    "early_time_weight": 0.01,
    "shared_device_ge_3": 0.03,
    "shared_device_ge_5": 0.07,
    "sim_change_weight": 0.05,
    "new_account_lt_7d": 0.03,
    "new_account_lt_3d": 0.06,
    "cashout_over_300": 0.04,
    "agent_high_risk_weight": 0.03,
    "flow_velocity_ge_8": 0.05,
    "p2p_counterparties_ge_12": 0.04,
    "cross_border_weight": 0.03,
}


def parse_context_weights() -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for key, default in CONTEXT_WEIGHT_DEFAULTS.items():
        env_key = f"CONTEXT_WEIGHT_{key.upper()}"
        raw = os.getenv(env_key)
        value = default if raw is None else safe_float(raw, env_key)
        if value < 0:
            raise ConfigurationError(f"{env_key} must be >= 0")
        weights[key] = float(value)
    return weights


CONTEXT_WEIGHTS = parse_context_weights()

IMPUTED_BASELINE_DEFAULTS: Dict[str, float] = {
    "MERCHANT": CONTEXT_ONLY_BASE_SCORE,
    "P2P": 0.25,
    "CASH_IN": 0.19,
    "CASH_OUT": 0.28,
}
IMPUTED_ADJUSTMENT_DEFAULTS: Dict[str, float] = {
    "cross_border": 0.03,
    "account_age_lt_3d": 0.06,
    "account_age_lt_30d": 0.02,
    "channel_agent": 0.02,
    "channel_api": 0.01,
    "severe_device_ip_risk": 0.05,
}
IMPUTED_ANCHOR_CLAMP_DEFAULTS = {
    "min": 0.05,
    "max": 0.80,
}
IMPUTED_SEVERE_RISK_THRESHOLD_DEFAULT = 0.85


def parse_imputed_base_anchor_config() -> Dict[str, Any]:
    baselines: Dict[str, float] = {}
    for tx_type, default in IMPUTED_BASELINE_DEFAULTS.items():
        env_key = f"FRAUD_IMPUTED_BASELINE_{tx_type}"
        baselines[tx_type] = float(os.getenv(env_key, str(default)))

    adjustments: Dict[str, float] = {}
    for adjustment_name, default in IMPUTED_ADJUSTMENT_DEFAULTS.items():
        env_key = f"FRAUD_IMPUTED_WEIGHT_{adjustment_name.upper()}"
        adjustments[adjustment_name] = float(os.getenv(env_key, str(default)))

    clamp_min = float(os.getenv("FRAUD_IMPUTED_ANCHOR_MIN", str(IMPUTED_ANCHOR_CLAMP_DEFAULTS["min"])))
    clamp_max = float(os.getenv("FRAUD_IMPUTED_ANCHOR_MAX", str(IMPUTED_ANCHOR_CLAMP_DEFAULTS["max"])))
    if clamp_min > clamp_max:
        raise ConfigurationError(
            "FRAUD_IMPUTED_ANCHOR_MIN must be <= FRAUD_IMPUTED_ANCHOR_MAX"
        )

    severe_threshold = float(
        os.getenv("FRAUD_IMPUTED_SEVERE_RISK_THRESHOLD", str(IMPUTED_SEVERE_RISK_THRESHOLD_DEFAULT))
    )
    if severe_threshold < 0.0 or severe_threshold > 1.0:
        raise ConfigurationError("FRAUD_IMPUTED_SEVERE_RISK_THRESHOLD must be between 0 and 1")

    return {
        "baselines": baselines,
        "adjustments": adjustments,
        "clamp": {"min": clamp_min, "max": clamp_max},
        "severe_risk_threshold": severe_threshold,
    }


IMPUTED_BASE_ANCHOR_CONFIG = parse_imputed_base_anchor_config()
SEGMENT_THRESHOLDS: Dict[str, SegmentThresholds] = {}
RULE_STATE_STORE = RuleStateStore()

AUDIT_DIR = Path(os.getenv("FRAUD_AUDIT_DIR", str(PROJECT_ROOT / "outputs" / "audit")))
AUDIT_LOG_FILE = AUDIT_DIR / os.getenv("FRAUD_AUDIT_FILE", "fraud_audit_log.jsonl")
REVIEW_QUEUE_FILE = AUDIT_DIR / os.getenv("FRAUD_REVIEW_QUEUE_FILE", "review_queue.jsonl")
ANALYST_OUTCOMES_FILE = AUDIT_DIR / os.getenv("FRAUD_ANALYST_OUTCOMES_FILE", "analyst_outcomes.jsonl")
RETRAINING_CURATION_FILE = AUDIT_DIR / os.getenv("FRAUD_RETRAINING_CURATION_FILE", "retraining_curation.jsonl")
REVIEW_BORDERLINE_MARGIN = float(os.getenv("FRAUD_REVIEW_BORDERLINE_MARGIN", "0.03"))
AUDIT_ASYNC_WRITE_ENABLED = os.getenv("FRAUD_AUDIT_ASYNC_WRITE_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
AUDIT_BATCH_SIZE = max(1, int(os.getenv("FRAUD_AUDIT_BATCH_SIZE", "64")))
AUDIT_BATCH_FLUSH_INTERVAL_SECONDS = max(
    0.01, float(os.getenv("FRAUD_AUDIT_BATCH_FLUSH_INTERVAL_SECONDS", "0.05"))
)
AUDIT_QUEUE_MAXSIZE = max(100, int(os.getenv("FRAUD_AUDIT_QUEUE_MAXSIZE", "20000")))

_AUDIT_STOP_SENTINEL = object()
_audit_write_queue: queue.Queue[Dict[str, Any] | object] = queue.Queue(maxsize=AUDIT_QUEUE_MAXSIZE)
_audit_writer_stop_event = threading.Event()
_audit_writer_thread: threading.Thread | None = None
_audit_writer_lock = threading.Lock()
_audit_last_signature_lock = threading.Lock()
_audit_last_signature: str | None = None


def load_versioned_secrets(
    secrets_env_var: str,
    fallback_env_var: str,
    local_default_secret: str,
) -> tuple[Dict[str, str], str]:
    raw_secrets = os.getenv(secrets_env_var)
    if raw_secrets:
        try:
            parsed = json.loads(raw_secrets)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{secrets_env_var} must be valid JSON") from exc

        if not isinstance(parsed, dict) or not parsed:
            raise ValueError(f"{secrets_env_var} must be a non-empty JSON object")

        normalized: Dict[str, str] = {}
        for version, secret in parsed.items():
            if not isinstance(version, str) or not version.strip():
                raise ValueError(f"{secrets_env_var} contains invalid key version: {version}")
            if not isinstance(secret, str) or not secret.strip():
                raise ValueError(f"{secrets_env_var}[{version}] must be a non-empty string")
            normalized[version.strip()] = secret.strip()

        return normalized, secrets_env_var

    fallback_secret = os.getenv(fallback_env_var, local_default_secret).strip()
    if not fallback_secret:
        raise ValueError(
            f"{secrets_env_var} is unset and {fallback_env_var} is empty; configure one of them"
        )
    return {"v1": fallback_secret}, fallback_env_var


def resolve_active_key_version(
    configured_version: str,
    available_secrets: Dict[str, str],
    version_env_var: str,
) -> tuple[str, bool]:
    if configured_version in available_secrets:
        return configured_version, False

    fallback_version = sorted(available_secrets.keys())[-1]
    logger.warning(
        "%s=%s not found in configured secret versions %s; using fallback version %s",
        version_env_var,
        configured_version,
        sorted(available_secrets.keys()),
        fallback_version,
    )
    return fallback_version, True


HASH_SECRETS, HASH_SECRET_SOURCE = load_versioned_secrets(
    "FRAUD_HASH_SECRETS",
    "HASH_SALT",
    "local_dev_only_change_me",
)
HASH_KEY_VERSION, HASH_KEY_VERSION_FALLBACK_USED = resolve_active_key_version(
    os.getenv("FRAUD_HASH_KEY_VERSION", "v1"),
    HASH_SECRETS,
    "FRAUD_HASH_KEY_VERSION",
)

AUDIT_SIGNING_SECRETS, AUDIT_SIGNING_SECRET_SOURCE = load_versioned_secrets(
    "FRAUD_AUDIT_SIGNING_SECRETS",
    "FRAUD_AUDIT_SIGNING_SECRET",
    "local_dev_only_audit_signing_secret",
)
AUDIT_SIGNING_KEY_VERSION, AUDIT_SIGNING_KEY_VERSION_FALLBACK_USED = resolve_active_key_version(
    os.getenv("FRAUD_AUDIT_SIGNING_KEY_VERSION", "v1"),
    AUDIT_SIGNING_SECRETS,
    "FRAUD_AUDIT_SIGNING_KEY_VERSION",
)

AUDIT_RETENTION_DAYS = int(os.getenv("FRAUD_AUDIT_RETENTION_DAYS", "365"))
AUDIT_DELETION_SLA_DAYS = int(os.getenv("FRAUD_AUDIT_DELETION_SLA_DAYS", "30"))

# ============================================================
# HELPERS
# ============================================================
def clamp_score(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def safe_float(x: Any, field_name: str) -> float:
    try:
        value = float(x)
    except Exception as e:
        raise InvalidRiskScoreError(f"Invalid numeric value for {field_name}: {x}", field=field_name) from e

    if value != value:
        raise InvalidRiskScoreError(f"{field_name} cannot be NaN", field=field_name)

    if value in (float("inf"), float("-inf")):
        raise InvalidRiskScoreError(f"{field_name} cannot be infinite", field=field_name)

    return value


def ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def ensure_dir_exists(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path):
    ensure_file_exists(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def validate_thresholds(approve_threshold: float, block_threshold: float) -> None:
    if not (0.0 <= approve_threshold < block_threshold <= 1.0):
        raise ValueError(
            f"Invalid thresholds: approve_threshold={approve_threshold}, "
            f"block_threshold={block_threshold}. Expected 0 <= approve < block <= 1."
        )


def load_segment_thresholds_config(
    payload: Any,
    *,
    fallback_approve: float,
    fallback_block: float,
    field_name: str,
) -> Dict[str, SegmentThresholds]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be an object")

    segment_thresholds: Dict[str, SegmentThresholds] = {}
    for segment, threshold_payload in payload.items():
        if not isinstance(threshold_payload, dict):
            raise ValueError(f"{field_name}.{segment} must be an object")

        segment_approve = safe_float(
            threshold_payload.get("approve_threshold", fallback_approve),
            f"{field_name}.{segment}.approve_threshold",
        )
        segment_block = safe_float(
            threshold_payload.get("block_threshold", fallback_block),
            f"{field_name}.{segment}.block_threshold",
        )
        validate_thresholds(segment_approve, segment_block)
        min_block_precision = safe_float(
            threshold_payload.get("min_block_precision", 0.0),
            f"{field_name}.{segment}.min_block_precision",
        )
        max_approve_to_flag_fpr = safe_float(
            threshold_payload.get("max_approve_to_flag_fpr", 1.0),
            f"{field_name}.{segment}.max_approve_to_flag_fpr",
        )

        calibration_metrics_payload = threshold_payload.get("calibration_metrics", {})
        calibration_metrics: Dict[str, float] = {}
        if calibration_metrics_payload is not None:
            if not isinstance(calibration_metrics_payload, dict):
                raise ValueError(f"{field_name}.{segment}.calibration_metrics must be an object")
            for metric_name, metric_value in calibration_metrics_payload.items():
                calibration_metrics[str(metric_name)] = safe_float(
                    metric_value,
                    f"{field_name}.{segment}.calibration_metrics.{metric_name}",
                )

        cfg = SegmentThresholds(
            approve_threshold=float(segment_approve),
            block_threshold=float(segment_block),
            min_block_precision=float(min_block_precision),
            max_approve_to_flag_fpr=float(max_approve_to_flag_fpr),
            calibration_metrics=calibration_metrics,
        )
        validate_segment_thresholds(cfg, str(segment))
        segment_thresholds[str(segment)] = cfg

    return segment_thresholds


def segment_thresholds_summary() -> Dict[str, Dict[str, float]]:
    return {
        segment: {
            "approve_threshold": round(cfg.approve_threshold, 4),
            "block_threshold": round(cfg.block_threshold, 4),
            "min_block_precision": round(cfg.min_block_precision, 4),
            "max_approve_to_flag_fpr": round(cfg.max_approve_to_flag_fpr, 4),
        }
        for segment, cfg in sorted(SEGMENT_THRESHOLDS.items())
    }


def validate_context_cap(context_adjustment_max: float) -> None:
    if not (0.0 <= context_adjustment_max <= 1.0):
        raise ValueError(
            f"Invalid CONTEXT_ADJUSTMENT_MAX={context_adjustment_max}. Expected 0 <= value <= 1."
        )


validate_context_cap(CONTEXT_ADJUSTMENT_MAX)


def detect_feature_columns_shape(columns: list[str]) -> str:
    if not columns:
        return "unknown"

    as_text = [str(col) for col in columns]
    raw_markers = {"TransactionDT", "TransactionAmt", "device_risk_score"}
    has_raw_markers = raw_markers.issubset(set(as_text))
    has_preprocessed_markers = any(
        name.startswith("numeric_canonical__")
        or name.startswith("categorical_passthrough__")
        or "__" in name
        for name in as_text
    )

    if has_raw_markers and not has_preprocessed_markers:
        return "raw"
    if has_preprocessed_markers and not has_raw_markers:
        return "preprocessed"
    if has_preprocessed_markers and has_raw_markers:
        return "mixed"
    return "unknown"


def validate_inference_artifact_compatibility(
    use_preprocessing_inference: bool,
    columns: list[str],
    bundle: Any,
) -> dict[str, Any]:
    shape = detect_feature_columns_shape(columns)
    issues: list[str] = []

    if use_preprocessing_inference:
        if bundle is None:
            issues.append(
                "Preprocessing inference mode is enabled but preprocessing bundle is missing."
            )
        if shape != "preprocessed":
            issues.append(
                f"Preprocessing inference mode requires preprocessed feature columns, detected '{shape}'."
            )
        if bundle is not None and getattr(bundle, "feature_names_out", None):
            bundle_features = set(str(name) for name in bundle.feature_names_out)
            missing = [name for name in columns if str(name) not in bundle_features]
            if missing:
                issues.append(
                    "Configured feature_columns are not present in preprocessing bundle output."
                )
    else:
        if shape != "raw":
            issues.append(
                f"Raw inference mode requires IEEE-style raw feature columns (TransactionDT/TransactionAmt/...), detected '{shape}'."
            )

    result = {
        "ok": len(issues) == 0,
        "mode": "preprocessing" if use_preprocessing_inference else "raw",
        "detected_feature_shape": shape,
        "issues": issues,
    }
    return result


def build_artifact_mismatch_runtime_error(validation: dict[str, Any]) -> RuntimeError:
    mode = validation.get("mode", "unknown")
    issues = validation.get("issues", [])
    issue_text = " ".join(issues) if issues else "Artifact compatibility validation failed."
    remediation = (
        "Remediation: "
        "(1) For raw API mode, regenerate runtime artifacts with: "
        "python project/models/final_xgboost_model.py --dataset-source ieee_cis --ieee-transaction-path ieee-fraud-detection/train_transaction.csv --ieee-identity-path ieee-fraud-detection/train_identity.csv "
        "--model-output project/models/final_xgboost_model.pkl --features-output project/models/feature_columns.pkl "
        "--thresholds-output project/models/decision_thresholds.pkl. "
        "(2) For preprocessing mode, set FRAUD_USE_PREPROCESSING_INFERENCE=true and point FRAUD_MODEL_FILE/FRAUD_FEATURE_FILE/"
        "FRAUD_THRESHOLD_FILE/FRAUD_PREPROCESSING_BUNDLE_FILE to promoted preprocessing artifacts."
    )
    return RuntimeError(f"Inference artifact mismatch for mode '{mode}': {issue_text} {remediation}")


def load_context_calibration_overrides(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}

    calibration_path = Path(path)
    ensure_file_exists(calibration_path)
    with calibration_path.open("r", encoding="utf-8") as calibration_file:
        payload = json.load(calibration_file)

    recommendation = payload.get("runtime_recommendation", payload)
    if not isinstance(recommendation, dict):
        raise ValueError("Context calibration payload must be a JSON object")
    return recommendation


def apply_context_calibration_overrides(overrides: Dict[str, Any]) -> None:
    global approve_threshold, block_threshold, CONTEXT_ADJUSTMENT_MAX, CONTEXT_WEIGHTS, SEGMENT_THRESHOLDS

    if not overrides:
        return

    if "approve_threshold" in overrides:
        approve_threshold = safe_float(overrides["approve_threshold"], "approve_threshold")
    if "block_threshold" in overrides:
        block_threshold = safe_float(overrides["block_threshold"], "block_threshold")
    validate_thresholds(approve_threshold, block_threshold)

    if "context_adjustment_max" in overrides:
        CONTEXT_ADJUSTMENT_MAX = safe_float(
            overrides["context_adjustment_max"], "context_adjustment_max"
        )
        validate_context_cap(CONTEXT_ADJUSTMENT_MAX)

    override_weights = overrides.get("context_weights")
    if override_weights is not None:
        if not isinstance(override_weights, dict):
            raise ValueError("context_weights override must be an object")

        merged = dict(CONTEXT_WEIGHTS)
        for key, value in override_weights.items():
            if key not in CONTEXT_WEIGHT_DEFAULTS:
                continue
            numeric_value = safe_float(value, f"context_weights.{key}")
            if numeric_value < 0:
                raise ValueError(f"context_weights.{key} must be >= 0")
            merged[key] = float(numeric_value)
        CONTEXT_WEIGHTS = merged

    override_segment_thresholds = overrides.get("segment_thresholds")
    if override_segment_thresholds is not None:
        SEGMENT_THRESHOLDS = load_segment_thresholds_config(
            override_segment_thresholds,
            fallback_approve=approve_threshold,
            fallback_block=block_threshold,
            field_name="segment_thresholds",
        )


def hash_user_id(user_id: str) -> str:
    secret = HASH_SECRETS[HASH_KEY_VERSION]
    raw = f"{HASH_KEY_VERSION}:{user_id}".encode("utf-8")
    hashed = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return f"{HASH_KEY_VERSION}:{hashed}"


def build_record_signature_payload(record: Dict[str, Any]) -> bytes:
    signing_fields = {k: v for k, v in record.items() if k != "record_signature"}
    canonical = json.dumps(signing_fields, sort_keys=True, separators=(",", ":"))
    return canonical.encode("utf-8")


def sign_audit_record(record: Dict[str, Any]) -> str:
    key_version = record["signature_key_version"]
    signing_secret = AUDIT_SIGNING_SECRETS[key_version]
    payload = build_record_signature_payload(record)
    return hmac.new(signing_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def get_previous_audit_signature() -> str:
    if not AUDIT_LOG_FILE.exists() or AUDIT_LOG_FILE.stat().st_size == 0:
        return "GENESIS"

    with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as audit_file:
        for line in reversed(audit_file.readlines()):
            last_line = line.strip()
            if not last_line:
                continue

            try:
                previous_record = json.loads(last_line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed audit log line while resolving previous signature",
                    extra={"audit_log_path": str(AUDIT_LOG_FILE)},
                )
                continue

            if not isinstance(previous_record, dict):
                continue

            signature = previous_record.get("record_signature")
            if isinstance(signature, str) and signature:
                return signature

    return "GENESIS"


def _initialize_audit_signature_cache() -> None:
    global _audit_last_signature
    with _audit_last_signature_lock:
        _audit_last_signature = get_previous_audit_signature()


def _append_audit_records(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    ensure_dir_exists(AUDIT_DIR)
    with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sign_audit_record_for_write(record: Dict[str, Any]) -> Dict[str, Any]:
    global _audit_last_signature
    out = dict(record)
    with _audit_last_signature_lock:
        previous_signature = _audit_last_signature or "GENESIS"
        out["previous_record_signature"] = previous_signature
        out["record_signature"] = sign_audit_record(out)
        _audit_last_signature = str(out["record_signature"])
    return out


def _flush_audit_batch(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    signed_batch = [_sign_audit_record_for_write(record) for record in records]
    _append_audit_records(signed_batch)


def _audit_writer_loop() -> None:
    pending: List[Dict[str, Any]] = []
    while not _audit_writer_stop_event.is_set():
        try:
            item = _audit_write_queue.get(timeout=AUDIT_BATCH_FLUSH_INTERVAL_SECONDS)
        except queue.Empty:
            _flush_audit_batch(pending)
            pending.clear()
            continue
        if item is _AUDIT_STOP_SENTINEL:
            break
        if isinstance(item, dict):
            pending.append(item)
            if len(pending) >= AUDIT_BATCH_SIZE:
                _flush_audit_batch(pending)
                pending.clear()
    _flush_audit_batch(pending)


def start_audit_writer() -> None:
    global _audit_writer_thread
    with _audit_writer_lock:
        if _audit_writer_thread is not None and _audit_writer_thread.is_alive():
            return
        _initialize_audit_signature_cache()
        _audit_writer_stop_event.clear()
        if not AUDIT_ASYNC_WRITE_ENABLED:
            _audit_writer_thread = None
            return
        _audit_writer_thread = threading.Thread(
            target=_audit_writer_loop,
            name="fraud-audit-writer",
            daemon=True,
        )
        _audit_writer_thread.start()


def stop_audit_writer() -> None:
    global _audit_writer_thread
    with _audit_writer_lock:
        if _audit_writer_thread is None:
            return
        _audit_writer_stop_event.set()
        try:
            _audit_write_queue.put_nowait(_AUDIT_STOP_SENTINEL)
        except queue.Full:
            pass
        _audit_writer_thread.join(timeout=2.0)
        _audit_writer_thread = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_request_id() -> str:
    return str(uuid.uuid4())


REQUEST_ID_CTX: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


def classify_exception(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "artifact_missing"
    if isinstance(exc, TimeoutError):
        return "timeout_upstream"
    if isinstance(exc, ReviewQueueRecordNotFoundError):
        return "review_queue_record_not_found"
    if isinstance(exc, UnknownTransactionTypeError):
        return "unknown_transaction_type"
    if isinstance(exc, UserProfileMismatchError):
        return "user_profile_mismatch"
    if isinstance(exc, ArtifactSchemaMismatchError):
        return "artifact_schema_mismatch"

    message = str(exc).lower()
    if isinstance(exc, ValueError) and (
        "validation" in message
        or "out of allowed numeric range" in message
        or "must contain a non-empty list" in message
    ):
        return "schema_mismatch"
    if "inference artifact mismatch" in message or "artifact mismatch" in message:
        return "artifact_incompatible"
    if "model scoring failed" in message or "predict" in message:
        return "model_runtime_error"
    return "unknown_internal"


def exception_signature(exc: Exception) -> str:
    return f"{exc.__class__.__name__}:{str(exc)[:200]}"


def structured_log(event: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "event": event,
    }
    payload.update(fields)
    logger.info(json.dumps(payload))


def resolve_correlation_id(request: Request) -> str:
    return request.headers.get("x-correlation-id") or request.headers.get("x-request-id") or generate_request_id()



def derive_transaction_id(tx_dict: Dict[str, Any], correlation_id: str) -> str:
    tx_fingerprint = {
        "user_id": tx_dict.get("user_id"),
        "TransactionDT": tx_dict.get("TransactionDT"),
        "TransactionAmt": tx_dict.get("TransactionAmt"),
        "device_id": tx_dict.get("device_id"),
        "tx_type": tx_dict.get("tx_type"),
    }
    digest = hashlib.sha256(
        json.dumps(tx_fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"{digest}:{correlation_id}"


def extract_request_id(request: Request) -> str:
    return REQUEST_ID_CTX.get() or request.headers.get("x-request-id") or generate_request_id()


def extract_user_id(request: Request) -> str | None:
    user_id = request.path_params.get("user_id")
    if user_id is None:
        user_id = request.query_params.get("user_id")
    if user_id is None:
        user_id = request.headers.get("x-user-id") or request.headers.get("user-id")
    return str(user_id) if user_id is not None else None


def extract_transaction_identifiers(request: Request) -> dict[str, str]:
    candidates = {
        "transaction_id": request.path_params.get("transaction_id")
        or request.query_params.get("transaction_id")
        or request.headers.get("x-transaction-id"),
        "tx_id": request.path_params.get("tx_id")
        or request.query_params.get("tx_id")
        or request.headers.get("x-tx-id"),
        "device_id": request.path_params.get("device_id")
        or request.query_params.get("device_id")
        or request.headers.get("x-device-id"),
        "request_id": request.path_params.get("request_id")
        or request.query_params.get("request_id")
        or request.headers.get("x-request-id"),
    }
    return {k: str(v) for k, v in candidates.items() if v is not None}


def extract_trace_id(request: Request, request_id: str) -> str:
    return (
        request.headers.get("x-trace-id")
        or request.headers.get("x-request-id")
        or request.headers.get("traceparent")
        or request_id
    )


def log_exception(request: Request, exc: Exception) -> str:
    request_id = extract_request_id(request)
    trace_id = extract_trace_id(request, request_id)
    category = classify_exception(exc)
    signature = exception_signature(exc)
    route = request.url.path
    user_id = extract_user_id(request)
    transaction_identifiers = extract_transaction_identifiers(request)
    logger.error(
        json.dumps(
            {
                "timestamp": utc_now_iso(),
                "event": "response_boundary",
                "status": "exception",
                "route": route,
                "request_id": request_id,
                "correlation_id": request_id,
                "trace_id": trace_id,
                "user_id": user_id,
                "transaction_identifiers": transaction_identifiers,
                "error_category": category,
                "exception_signature": signature,
            }
        )
    )
    logger.error(
        json.dumps(
            {
                "timestamp": utc_now_iso(),
                "request_id": request_id,
                "correlation_id": request_id,
                "trace_id": trace_id,
                "route": route,
                "user_id": user_id,
                "transaction_identifiers": transaction_identifiers,
                "error_category": category,
                "exception_signature": signature,
                "exception_type": type(exc).__name__,
                "exception_class": exc.__class__.__name__,
                "exception_message": str(exc),
            }
        )
    )
    return request_id

# LOAD ARTIFACTS
artifact_compatibility: dict[str, Any] = {}
promoted_artifact_validation: dict[str, Any] = {}
promoted_manifest = None
preprocessing_bundle = None
model = None
inference_backend = None
inference_backend_runtime = "uninitialized"
feature_columns: list[str] = []
feature_column_indices: np.ndarray | None = None
bundle_feature_name_set: set[str] = set()
_preprocessing_worker_cache = threading.local()
approve_threshold = DEFAULT_APPROVE_THRESHOLD
block_threshold = DEFAULT_BLOCK_THRESHOLD
_artifact_cache_lock = threading.Lock()
_artifact_init_count = 0
runtime_services: dict[str, Any] = {}


def initialize_runtime_artifacts() -> None:
    global artifact_compatibility, promoted_artifact_validation, promoted_manifest
    global preprocessing_bundle, model, inference_backend, inference_backend_runtime
    global feature_columns, feature_column_indices, bundle_feature_name_set
    global approve_threshold, block_threshold, SEGMENT_THRESHOLDS
    global MODEL_FILE, FEATURE_FILE, THRESHOLD_FILE, PREPROCESSING_BUNDLE_FILE, _artifact_init_count
    global runtime_services

    with _artifact_cache_lock:
        if model is not None and feature_columns:
            return

        logger.info("Loading fraud engine artifacts into in-memory singleton cache...")
        promoted_manifest = load_promoted_artifact_manifest(PROMOTED_ARTIFACT_MANIFEST_FILE)
        local_preprocessing_bundle = load_preprocessing_bundle(promoted_manifest.preprocessing_bundle_file)
        local_model = load_pickle(promoted_manifest.model_file)
        local_feature_columns = load_pickle(promoted_manifest.feature_file)
        threshold_config = load_pickle(promoted_manifest.threshold_file)

        if not isinstance(local_feature_columns, list) or not local_feature_columns:
            raise ValueError("feature_columns.pkl must contain a non-empty list")

        local_approve_threshold = float(threshold_config.get("approve_threshold", DEFAULT_APPROVE_THRESHOLD))
        local_block_threshold = float(threshold_config.get("block_threshold", DEFAULT_BLOCK_THRESHOLD))
        validate_thresholds(local_approve_threshold, local_block_threshold)
        local_segment_thresholds = load_segment_thresholds_config(
            threshold_config.get("segment_thresholds"),
            fallback_approve=local_approve_threshold,
            fallback_block=local_block_threshold,
            field_name="segment_thresholds",
        )

        local_promoted_validation = validate_promoted_artifacts(
            promoted_manifest,
            feature_columns=local_feature_columns,
            preprocessing_bundle=local_preprocessing_bundle,
        )
        raise_on_failed_validation(local_promoted_validation)

        local_artifact_compatibility = validate_inference_artifact_compatibility(
            use_preprocessing_inference=USE_PREPROCESSING_INFERENCE,
            columns=local_feature_columns,
            bundle=local_preprocessing_bundle,
        )
        if not local_artifact_compatibility["ok"]:
            raise build_artifact_mismatch_runtime_error(local_artifact_compatibility)

        calibration_overrides = load_context_calibration_overrides(CONTEXT_CALIBRATION_FILE)
        approve_threshold = local_approve_threshold
        block_threshold = local_block_threshold
        SEGMENT_THRESHOLDS = local_segment_thresholds
        apply_context_calibration_overrides(calibration_overrides)

        if hasattr(local_model, "set_params"):
            try:
                local_model.set_params(n_jobs=MODEL_INFERENCE_THREADS)
            except Exception:
                pass
        if hasattr(local_model, "get_booster"):
            try:
                local_model.get_booster().set_param({"nthread": MODEL_INFERENCE_THREADS})
            except Exception:
                pass

        model = local_model
        inference_backend = create_inference_backend(FRAUD_INFERENCE_BACKEND)
        inference_backend_runtime = getattr(inference_backend, "runtime_name", inference_backend.backend_name)
        feature_columns = [str(col) for col in local_feature_columns]
        preprocessing_bundle = local_preprocessing_bundle
        bundle_feature_names = [str(name) for name in local_preprocessing_bundle.feature_names_out]
        bundle_feature_name_set = set(bundle_feature_names)
        index_map = {name: idx for idx, name in enumerate(bundle_feature_names)}
        missing_in_bundle = [name for name in feature_columns if name not in index_map]
        if missing_in_bundle:
            preview = missing_in_bundle[:20]
            remainder = len(missing_in_bundle) - len(preview)
            suffix = f" ... (+{remainder} more)" if remainder > 0 else ""
            raise ValueError(f"Missing required model features in preprocessing bundle: {preview}{suffix}")
        feature_column_indices = np.asarray([index_map[name] for name in feature_columns], dtype=np.int64)
        promoted_artifact_validation = local_promoted_validation
        artifact_compatibility = local_artifact_compatibility
        MODEL_FILE = promoted_manifest.model_file
        FEATURE_FILE = promoted_manifest.feature_file
        THRESHOLD_FILE = promoted_manifest.threshold_file
        PREPROCESSING_BUNDLE_FILE = promoted_manifest.preprocessing_bundle_file
        _artifact_init_count += 1
        runtime_services = {
            "model": model,
            "feature_columns": tuple(feature_columns),
            "feature_column_indices": feature_column_indices,
            "preprocessing_bundle": preprocessing_bundle,
            "approve_threshold": approve_threshold,
            "block_threshold": block_threshold,
            "segment_thresholds": SEGMENT_THRESHOLDS,
            "inference_backend": inference_backend,
            "inference_backend_runtime": inference_backend_runtime,
            "artifact_compatibility": artifact_compatibility,
            "promoted_artifact_validation": promoted_artifact_validation,
            "manifest_path": str(promoted_manifest.manifest_path),
            "init_count": _artifact_init_count,
        }
        logger.info("Artifacts loaded successfully (init_count=%s)", _artifact_init_count)

# BEHAVIOR PROFILER
BEHAVIOR_PROFILE_STORE_BACKEND = os.getenv("BEHAVIOR_PROFILE_STORE_BACKEND", "memory").lower()
BEHAVIOR_PROFILE_TTL_SECONDS = int(os.getenv("BEHAVIOR_PROFILE_TTL_SECONDS", "86400"))
BEHAVIOR_PROFILE_SQLITE_PATH = Path(
    os.getenv(
        "BEHAVIOR_PROFILE_SQLITE_PATH",
        str(PROJECT_ROOT / "outputs" / "behavior_profiles.sqlite3"),
    )
)
AGGREGATE_CACHE_TTL_SECONDS = float(os.getenv("FRAUD_AGGREGATE_CACHE_TTL_SECONDS", "2.0"))
AGGREGATE_INGESTION_ADAPTER = os.getenv("FRAUD_INGESTION_ADAPTER", "mock_stream").strip().lower()
AGGREGATE_INGESTION_IDEMPOTENCY_TTL_SECONDS = float(
    os.getenv("FRAUD_INGESTION_IDEMPOTENCY_TTL_SECONDS", "300")
)
AGGREGATE_INGESTION_QUEUE_SIZE = max(1, int(os.getenv("FRAUD_INGESTION_QUEUE_SIZE", "10000")))
FEATURE_STORE_BACKEND = os.getenv("FRAUD_FEATURE_STORE_BACKEND", "memory").strip().lower()
FEATURE_STORE_TTL_SECONDS = int(os.getenv("FRAUD_FEATURE_STORE_TTL_SECONDS", "7200"))
FEATURE_SCHEMA_VERSION = os.getenv("FRAUD_FEATURE_SCHEMA_VERSION", PAYLOAD_SCHEMA_VERSION)
AGGREGATION_WINDOW_SECONDS = int(os.getenv("FRAUD_AGGREGATION_WINDOW_SECONDS", "3600"))


def create_feature_store():
    config = FeatureStoreConfig(
        schema_version=FEATURE_SCHEMA_VERSION,
        aggregation_window_seconds=AGGREGATION_WINDOW_SECONDS,
    )

    if FEATURE_STORE_BACKEND == "redis":
        try:
            import redis

            redis_url = os.getenv("FRAUD_FEATURE_STORE_REDIS_URL", "redis://localhost:6379/1")
            client = redis.Redis.from_url(redis_url, decode_responses=False)
            client.ping()
            return RedisFeatureStore(client, config=config, ttl_seconds=FEATURE_STORE_TTL_SECONDS)
        except Exception as exc:
            logger.warning(f"Redis feature store unavailable, falling back to memory: {exc}")

    if FEATURE_STORE_BACKEND == "cassandra":
        try:
            from cassandra.cluster import Cluster

            contact_points = [
                host.strip()
                for host in os.getenv("FRAUD_FEATURE_STORE_CASSANDRA_HOSTS", "127.0.0.1").split(",")
                if host.strip()
            ]
            port = int(os.getenv("FRAUD_FEATURE_STORE_CASSANDRA_PORT", "9042"))
            keyspace = os.getenv("FRAUD_FEATURE_STORE_CASSANDRA_KEYSPACE", "fraud")
            cluster = Cluster(contact_points=contact_points, port=port)
            session = cluster.connect()
            return CassandraFeatureStore(session=session, config=config, keyspace=keyspace)
        except Exception as exc:
            logger.warning(f"Cassandra feature store unavailable, falling back to memory: {exc}")

    return InMemoryFeatureStore(config=config, ttl_seconds=FEATURE_STORE_TTL_SECONDS)


def create_profile_store():
    if BEHAVIOR_PROFILE_STORE_BACKEND == "redis":
        try:
            import redis

            redis_url = os.getenv("BEHAVIOR_PROFILE_REDIS_URL", "redis://localhost:6379/0")
            client = redis.Redis.from_url(redis_url, decode_responses=False)
            client.ping()
            return RedisProfileStore(client, ttl_seconds=BEHAVIOR_PROFILE_TTL_SECONDS)
        except Exception as exc:
            logger.warning(f"Redis profile store unavailable, falling back to SQLite: {exc}")

    if BEHAVIOR_PROFILE_STORE_BACKEND in {"sqlite", "redis"}:
        ensure_dir_exists(BEHAVIOR_PROFILE_SQLITE_PATH.parent)
        return SQLiteProfileStore(
            BEHAVIOR_PROFILE_SQLITE_PATH,
            ttl_seconds=BEHAVIOR_PROFILE_TTL_SECONDS,
        )

    return InMemoryProfileStore(ttl_seconds=BEHAVIOR_PROFILE_TTL_SECONDS)


behavior_profiler = BehaviorProfiler(
    profile_store=create_profile_store(),
    aggregate_cache_ttl_seconds=AGGREGATE_CACHE_TTL_SECONDS,
)
feature_store = create_feature_store()
entity_smoothing_state = EntitySmoothingState(
    EntitySmoothingConfig(
        method=ENTITY_SMOOTHING_METHOD,
        min_history=ENTITY_SMOOTHING_MIN_HISTORY,
        ema_alpha=ENTITY_SMOOTHING_EMA_ALPHA,
        blend_alpha=ENTITY_SMOOTHING_BLEND_ALPHA,
        blend_cap=ENTITY_SMOOTHING_BLEND_CAP,
        fallback_for_unseen=ENTITY_SMOOTHING_FALLBACK,
    )
)


def update_aggregate_pipeline(event: AggregateUpdateEvent) -> None:
    behavior_profiler.record_transaction(
        user_id=event.user_id,
        amount=event.amount,
        hour_of_day=event.hour_of_day,
        location_risk_score=event.location_risk_score,
    )


def create_aggregate_ingestion_adapter():
    if AGGREGATE_INGESTION_ADAPTER == "mock_stream":
        return MockStreamAggregateIngestionAdapter(
            handler=update_aggregate_pipeline,
            idempotency_ttl_seconds=AGGREGATE_INGESTION_IDEMPOTENCY_TTL_SECONDS,
        )
    return QueueAggregateIngestionAdapter(
        handler=update_aggregate_pipeline,
        max_queue_size=AGGREGATE_INGESTION_QUEUE_SIZE,
        idempotency_ttl_seconds=AGGREGATE_INGESTION_IDEMPOTENCY_TTL_SECONDS,
    )


aggregate_ingestion_adapter = create_aggregate_ingestion_adapter()


behavior_profiler.seed_profile(
    user_id="user_safe",
    historical_transactions=[
        (25.0, 9, 0.05),
        (18.0, 10, 0.03),
        (32.0, 11, 0.04),
        (21.0, 9, 0.02),
        (27.0, 10, 0.05),
        (19.0, 11, 0.04),
        (23.0, 9, 0.03),
    ]
)

behavior_profiler.seed_profile(
    user_id="user_flag",
    historical_transactions=[
        (75.0, 13, 0.10),
        (110.0, 14, 0.15),
        (95.0, 12, 0.12),
        (82.0, 13, 0.14),
        (105.0, 15, 0.10),
        (98.0, 14, 0.13),
        (120.0, 13, 0.11),
    ]
)

behavior_profiler.seed_profile(
    user_id="user_block",
    historical_transactions=[
        (40.0, 18, 0.08),
        (55.0, 19, 0.10),
        (38.0, 20, 0.06),
        (61.0, 19, 0.09),
        (47.0, 18, 0.07),
        (50.0, 20, 0.08),
        (44.0, 19, 0.09),
    ]
)

# FASTAPI APP
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_runtime_artifacts()
    start_audit_writer()
    app.state.runtime_services = runtime_services
    logger.info(
        "startup_artifact_load_guard worker_pid=%s init_count=%s",
        os.getpid(),
        _artifact_init_count,
    )
    startup_validation = validate_promoted_artifacts(
        promoted_manifest,
        feature_columns=feature_columns,
        preprocessing_bundle=preprocessing_bundle,
    )
    raise_on_failed_validation(startup_validation)
    yield
    close_profile_store = getattr(behavior_profiler.profile_store, "close", None)
    if callable(close_profile_store):
        close_profile_store()
    stop_audit_writer()


app = FastAPI(
    title=MODEL_NAME,
    description="Real-time fraud scoring engine with ML + context + behavioral profiling + privacy-safe audit logging",
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Eager initialization keeps module-level callers/tests aligned even when ASGI lifespan hooks are bypassed.
initialize_runtime_artifacts()
start_audit_writer()
app.state.runtime_services = runtime_services

# RESPONSE MODELS
class HealthResponse(BaseModel):
    status: Literal["ok"]
    api_version: str
    model_name: str
    model_version: str
    feature_count: int
    approve_threshold: float
    block_threshold: float
    segment_thresholds_summary: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class ReadinessResponse(BaseModel):
    status: Literal["ok"]
    artifacts: Dict[str, str]
    audit_log_path: str


class ScoreResponse(BaseModel):
    request_id: str
    correlation_id: str
    api_version: str
    model_name: str
    model_version: str
    base_model_score: float
    context_adjustment: float
    behavior_adjustment: float
    final_risk_score: float
    decision: Literal["APPROVE", "FLAG", "BLOCK"]
    user_segment: str
    segment_thresholds: Dict[str, float]
    hard_rule_hits: List[str]
    verification_action: Literal["NONE", "STEP_UP_OTP", "STEP_UP_KYC", "STEP_UP_MANUAL_REVIEW"]
    verification_reason: str | None
    reasons: List[str]
    context_summary: Dict[str, Any]
    audit_logging_enabled: bool
    latency_ms: float
    stage_timings_ms: Dict[str, float]


class PrivacyResponse(BaseModel):
    api_version: str
    model_name: str
    privacy_principles: List[str]
    stored_fields_in_audit_log: List[str]
    not_stored_in_audit_log: List[str]
    data_integrity_controls: List[str]
    retention_policy: Dict[str, Any]
class ReviewQueueItem(BaseModel):
    request_id: str
    correlation_id: str
    created_at_utc: str
    status: Literal["pending", "resolved"]
    decision: Literal["APPROVE", "FLAG", "BLOCK"]
    final_risk_score: float
    reasons: List[str] = Field(default_factory=list)
    context_summary: Dict[str, Any] = Field(default_factory=dict)


class AnalystOutcomeInput(BaseModel):
    analyst_id: str
    analyst_decision: Literal["FRAUD", "LEGIT", "ESCALATE"]
    analyst_confidence: float = Field(ge=0.0, le=1.0)
    analyst_notes: str | None = None
    transaction_amount: float = Field(default=0.0, ge=0.0)


class ReviewOutcomeResponse(BaseModel):
    status: Literal["ok"]
    request_id: str
    review_status: Literal["resolved"]
    curated_label: str
    retraining_curation_file: str


class RetrainingCurationResponse(BaseModel):
    status: Literal["ok"]
    count: int
    items: List[Dict[str, Any]]


class DashboardLatencyThroughputError(BaseModel):
    requests: int
    throughput_per_min: float
    error_rate: float
    latency_ms_p50: float
    latency_ms_p95: float


class DashboardDriftScoreDistribution(BaseModel):
    baseline_mean_score: float
    observed_mean_score: float
    mean_delta: float
    score_histogram: Dict[str, int]


class DashboardFraudLossFalsePositivesAnalystAgreement(BaseModel):
    estimated_fraud_loss: float
    false_positives: int
    analyst_agreement: float
    confirmed_fraud_cases: int
    analyst_reviews: int


class DashboardDecisionSourceKpiRow(BaseModel):
    decision_source: DecisionSource
    audit_volume: int
    flag_rate: float
    analyst_reviews: int
    confirmed_fraud_conversion: float
    false_positive_rate: float


class DashboardViewsResponse(BaseModel):
    window_hours: int
    generated_at_utc: str
    freshest_record_utc: str | None = None
    data_freshness_seconds: float | None = None
    latency_throughput_error: DashboardLatencyThroughputError
    drift_score_distribution: DashboardDriftScoreDistribution
    fraud_loss_false_positives_analyst_agreement: DashboardFraudLossFalsePositivesAnalystAgreement
    decision_source_kpis: List[DashboardDecisionSourceKpiRow] = Field(default_factory=list)


SHARED_ERROR_RESPONSES = {
    400: {"model": ApiErrorResponse, "description": "Bad request / domain validation error"},
    404: {"model": ApiErrorResponse, "description": "Resource not found"},
    409: {"model": ApiErrorResponse, "description": "Domain conflict"},
    422: {"model": ApiErrorResponse, "description": "Validation error payload"},
    500: {"model": ApiErrorResponse, "description": "Internal server error"},
}




SCORE_TRANSACTION_RESPONSES = {
    200: {"model": ScoreTransactionResponse, "description": "Transaction scored successfully"},
    **SHARED_ERROR_RESPONSES,
}


# ============================================================
# BUSINESS LOGIC
# ============================================================
def get_context_adjustment_breakdown(tx: Dict[str, Any]) -> Dict[str, float]:
    shared_users = int(tx.get("device_shared_users_24h", 0))
    age_days = int(tx.get("account_age_days", 0))
    tx_type = tx.get("tx_type", "MERCHANT")
    channel = tx.get("channel", "APP")

    return {
        "device_risk": CONTEXT_WEIGHTS["device_risk_weight"] * tx["device_risk_score"],
        "ip_risk": CONTEXT_WEIGHTS["ip_risk_weight"] * tx["ip_risk_score"],
        "location_risk": CONTEXT_WEIGHTS["location_risk_weight"] * tx["location_risk_score"],
        "amount_over_200": CONTEXT_WEIGHTS["amount_over_200"] if tx["TransactionAmt"] > 200 else 0.0,
        "amount_over_1000": CONTEXT_WEIGHTS["amount_over_1000"] if tx["TransactionAmt"] > 1000 else 0.0,
        "early_time": CONTEXT_WEIGHTS["early_time_weight"] if tx["TransactionDT"] < 1000 else 0.0,
        "shared_device": (
            CONTEXT_WEIGHTS["shared_device_ge_5"]
            if shared_users >= 5
            else CONTEXT_WEIGHTS["shared_device_ge_3"]
            if shared_users >= 3
            else 0.0
        ),
        "sim_change": CONTEXT_WEIGHTS["sim_change_weight"] if tx.get("sim_change_recent", False) else 0.0,
        "new_account": (
            CONTEXT_WEIGHTS["new_account_lt_3d"]
            if age_days < 3
            else CONTEXT_WEIGHTS["new_account_lt_7d"]
            if age_days < 7
            else 0.0
        ),
        "cashout_over_300": (
            CONTEXT_WEIGHTS["cashout_over_300"]
            if tx_type == "CASH_OUT" and tx["TransactionAmt"] > 300
            else 0.0
        ),
        "agent_high_risk": (
            CONTEXT_WEIGHTS["agent_high_risk_weight"]
            if channel == "AGENT" and (tx["device_risk_score"] >= 0.5 or tx["ip_risk_score"] >= 0.5)
            else 0.0
        ),
        "flow_velocity": (
            CONTEXT_WEIGHTS["flow_velocity_ge_8"]
            if int(tx.get("cash_flow_velocity_1h", 0)) >= 8
            else 0.0
        ),
        "p2p_counterparty_burst": (
            CONTEXT_WEIGHTS["p2p_counterparties_ge_12"]
            if int(tx.get("p2p_counterparties_24h", 0)) >= 12
            else 0.0
        ),
        "cross_border": CONTEXT_WEIGHTS["cross_border_weight"] if tx.get("is_cross_border", False) else 0.0,
    }


def normalize_external_score_transaction_request(tx: ScoreTransactionRequest) -> Dict[str, Any]:
    """
    Normalize the external `/score_transaction` contract into a canonical dict.

    This stage intentionally stays aligned with `ScoreTransactionRequest` so request-shape
    changes are isolated from downstream model-feature construction.
    """
    return tx.model_dump()


def build_model_features_from_normalized_request(tx_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the model-facing feature dictionary from normalized external payload data.

    The model expects IEEE-style feature names, so this mapping is the only place where
    external contract keys are transformed to serving feature names.
    """
    return {
        "schema_version": tx_payload["schema_version"],
        "user_id": tx_payload["user_id"],
        "TransactionDT": float(time.time()),
        "TransactionAmt": tx_payload["transaction_amount"],
        **{f"V{i}": 0.0 for i in range(1, 18)},
        "_v_features_source": "imputed_zero_vector",
        "device_risk_score": tx_payload["device_risk_score"],
        "ip_risk_score": tx_payload["ip_risk_score"],
        "location_risk_score": tx_payload["location_risk_score"],
        "device_id": tx_payload.get("device_id"),
        "device_shared_users_24h": tx_payload.get("device_shared_users_24h", 0),
        "account_age_days": tx_payload.get("account_age_days", 0),
        "sim_change_recent": tx_payload.get("sim_change_recent", False),
        "tx_type": tx_payload.get("tx_type", "MERCHANT"),
        "channel": tx_payload.get("channel", "APP"),
        "cash_flow_velocity_1h": tx_payload.get("cash_flow_velocity_1h", 0),
        "p2p_counterparties_24h": tx_payload.get("p2p_counterparties_24h", 0),
        "is_cross_border": tx_payload.get("is_cross_border", False),
        "override_source": tx_payload.get("override_source", "preset"),
        "support_mode": tx_payload.get("support_mode", False),
        "support_actor_id": tx_payload.get("support_actor_id"),
        "override_fields": tx_payload.get("override_fields", []),
    }


def build_model_features_from_score_transaction_request(
    tx: ScoreTransactionRequest,
    *,
    transaction_dt: float | None = None,
) -> Dict[str, Any]:
    """
    Hot-path builder used by /score_transaction to avoid intermediate payload dict copies.
    """
    resolved_transaction_dt = float(transaction_dt if transaction_dt is not None else time.time())
    return {
        "schema_version": tx.schema_version,
        "user_id": tx.user_id,
        "TransactionDT": resolved_transaction_dt,
        "TransactionAmt": tx.transaction_amount,
        **{f"V{i}": 0.0 for i in range(1, 18)},
        "_v_features_source": "imputed_zero_vector",
        "device_risk_score": tx.device_risk_score,
        "ip_risk_score": tx.ip_risk_score,
        "location_risk_score": tx.location_risk_score,
        "device_id": tx.device_id,
        "device_shared_users_24h": tx.device_shared_users_24h,
        "account_age_days": tx.account_age_days,
        "sim_change_recent": tx.sim_change_recent,
        "tx_type": tx.tx_type,
        "channel": tx.channel,
        "cash_flow_velocity_1h": tx.cash_flow_velocity_1h,
        "p2p_counterparties_24h": tx.p2p_counterparties_24h,
        "is_cross_border": tx.is_cross_border,
        "override_source": tx.override_source,
        "support_mode": tx.support_mode,
        "support_actor_id": tx.support_actor_id,
        "override_fields": tx.override_fields,
    }


def get_context_adjustment(tx: Dict[str, Any]) -> float:
    breakdown = get_context_adjustment_breakdown(tx)
    raw_adjustment = sum(breakdown.values())
    return clamp_score(min(raw_adjustment, CONTEXT_ADJUSTMENT_MAX))


def build_aggregate_input_fingerprint(tx: Dict[str, Any]) -> str:
    fingerprint_payload = {
        "TransactionAmt": round(float(tx.get("TransactionAmt", 0.0)), 6),
        "hour_of_day": int(float(tx.get("TransactionDT", 0.0)) % 24),
        "location_risk_score": round(float(tx.get("location_risk_score", 0.0)), 6),
        "device_risk_score": round(float(tx.get("device_risk_score", 0.0)), 6),
        "ip_risk_score": round(float(tx.get("ip_risk_score", 0.0)), 6),
        "device_shared_users_24h": int(tx.get("device_shared_users_24h", 0)),
        "account_age_days": int(tx.get("account_age_days", 0)),
        "sim_change_recent": bool(tx.get("sim_change_recent", False)),
        "tx_type": str(tx.get("tx_type", "")),
        "channel": str(tx.get("channel", "")),
        "cash_flow_velocity_1h": int(tx.get("cash_flow_velocity_1h", 0)),
        "p2p_counterparties_24h": int(tx.get("p2p_counterparties_24h", 0)),
        "is_cross_border": bool(tx.get("is_cross_border", False)),
    }
    encoded = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compute_imputed_base_anchor(tx: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
    tx_type = str(tx.get("tx_type", "MERCHANT"))
    tx_type_baseline = {
        "MERCHANT": ANCHOR_BASELINE_MERCHANT,
        "P2P": ANCHOR_BASELINE_P2P,
        "CASH_IN": ANCHOR_BASELINE_CASH_IN,
        "CASH_OUT": ANCHOR_BASELINE_CASH_OUT,
    }
    base = float(tx_type_baseline.get(tx_type, ANCHOR_BASELINE_MERCHANT))

    adjustments: Dict[str, float] = {
        "cross_border": ANCHOR_CROSS_BORDER_BOOST if bool(tx.get("is_cross_border", False)) else 0.0,
        "sim_change": ANCHOR_SIM_CHANGE_BOOST if bool(tx.get("sim_change_recent", False)) else 0.0,
        "new_account_lt3": 0.0,
        "new_account_lt7": 0.0,
        "device_risk": ANCHOR_DEVICE_RISK_WEIGHT * float(tx.get("device_risk_score", 0.0) or 0.0),
        "ip_risk": ANCHOR_IP_RISK_WEIGHT * float(tx.get("ip_risk_score", 0.0) or 0.0),
        "location_risk": ANCHOR_LOCATION_RISK_WEIGHT * float(tx.get("location_risk_score", 0.0) or 0.0),
        "established_merchant_discount": 0.0,
        "very_established_merchant_discount": 0.0,
    }
    account_age_days = int(tx.get("account_age_days", 0) or 0)
    if account_age_days < 3:
        adjustments["new_account_lt3"] = ANCHOR_NEW_ACCOUNT_LT3_BOOST
    elif account_age_days < 7:
        adjustments["new_account_lt7"] = ANCHOR_NEW_ACCOUNT_LT7_BOOST

    is_low_risk_merchant = (
        tx_type == "MERCHANT"
        and not bool(tx.get("is_cross_border", False))
        and float(tx.get("device_risk_score", 0.0) or 0.0) <= 0.10
        and float(tx.get("ip_risk_score", 0.0) or 0.0) <= 0.10
        and float(tx.get("location_risk_score", 0.0) or 0.0) <= 0.10
    )
    if is_low_risk_merchant and account_age_days >= 365:
        adjustments["established_merchant_discount"] = -abs(ANCHOR_ESTABLISHED_MERCHANT_DISCOUNT)
        if account_age_days >= 730:
            adjustments["very_established_merchant_discount"] = -abs(ANCHOR_VERY_ESTABLISHED_MERCHANT_DISCOUNT)

    pre_clamp = base + sum(adjustments.values())
    anchored = min(ANCHOR_MAX, max(ANCHOR_MIN, pre_clamp))
    diagnostics = {
        "enabled": True,
        "source": "dynamic",
        "mode": "dynamic",
        "tx_type": tx_type,
        "baseline": round(base, 4),
        "components": {k: round(v, 4) for k, v in adjustments.items()},
        "adjustments": {k: round(v, 4) for k, v in adjustments.items()},
        "pre_clamp": round(pre_clamp, 4),
        "post_clamp": round(anchored, 4),
        "bounds": {"min": ANCHOR_MIN, "max": ANCHOR_MAX},
    }
    return float(anchored), diagnostics


def get_behavior_adjustment(
    tx: Dict[str, float],
    cached_aggregates: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    hour_of_day = int(tx["TransactionDT"] % 24)
    store_aggregates = cached_aggregates
    if store_aggregates is None:
        as_of_ts = float(tx["TransactionDT"])
        store_aggregates = feature_store.get_user_aggregates(user_id=tx["user_id"], as_of_ts=as_of_ts)
    geo_device_mismatch = bool(
        tx.get("is_cross_border", False)
        or tx.get("location_risk_score", 0.0) >= 0.70
        or (
            tx.get("device_shared_users_24h", 0) >= 3
            and tx.get("location_risk_score", 0.0) >= 0.40
        )
    )

    expected_fingerprint = build_aggregate_input_fingerprint(tx)
    has_matching_fingerprint = (
        store_aggregates is not None
        and store_aggregates.get("input_fingerprint") == expected_fingerprint
    )
    if has_matching_fingerprint and isinstance(store_aggregates.get("behavior_features"), dict):
        behavior_features = dict(store_aggregates.get("behavior_features", {}))
        behavior_adjustment = float(store_aggregates.get("behavior_adjustment", 0.0))
        behavior_reasons = list(store_aggregates.get("behavior_reasons", []))
        source = "feature_store"
    else:
        behavior_features = behavior_profiler.compute_behavior_features(
            user_id=tx["user_id"],
            amount=tx["TransactionAmt"],
            hour_of_day=hour_of_day,
            location_risk_score=tx["location_risk_score"],
            event_timestamp=tx["TransactionDT"],
            geo_device_mismatch=geo_device_mismatch,
            counterparties_24h=int(tx.get("p2p_counterparties_24h", 0)),
        )
        # Supportive layer only
        behavior_adjustment = 0.10 * behavior_features["behavior_risk_score"]
        behavior_reasons = behavior_profiler.generate_behavior_reasons(behavior_features)
        source = "on_the_fly"

    return {
        "behavior_features": behavior_features,
        "behavior_adjustment": round(behavior_adjustment, 4),
        "behavior_reasons": behavior_reasons,
        "source": source,
    }


def apply_low_history_policy(
    final_score: float,
    raw_decision: Literal["APPROVE", "FLAG", "BLOCK"],
    behavior_features: Dict[str, Any],
) -> Literal["APPROVE", "FLAG", "BLOCK"]:
    # Low-history policy should only add friction to lower-confidence approvals.
    # Never downgrade an already-blocked transaction.
    if raw_decision == "BLOCK":
        return "BLOCK"

    if not behavior_features.get("is_low_history", False):
        return raw_decision

    if final_score >= 0.97:
        return "BLOCK"

    if final_score >= approve_threshold:
        return "FLAG"

    return raw_decision


def resolve_decision_source(
    raw_decision: Literal["APPROVE", "FLAG", "BLOCK"],
    decision_after_low_history: Literal["APPROVE", "FLAG", "BLOCK"],
    final_decision: Literal["APPROVE", "FLAG", "BLOCK"],
    hard_rule_action: str,
    verification_action: str,
) -> DecisionSource:
    if hard_rule_action in {"FLAG", "BLOCK"} and final_decision != decision_after_low_history:
        return "hard_rule_override"
    if decision_after_low_history != raw_decision:
        return "low_history_policy"
    if verification_action != "NONE":
        return "step_up_policy"
    return "score_band"



def apply_entity_smoothing(base_score: float, tx_dict: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
    if ENTITY_SMOOTHING_METHOD == "none":
        return base_score, {"enabled": False, "method": "none", "fallback_used": False, "prior_history_count": 0}

    features_df = pd.DataFrame([tx_dict])
    entity_series, _ = build_entity_id(features_df, dataset_source="ieee_cis")
    entity_id = str(entity_series.iloc[0])
    smoothed, diagnostics = entity_smoothing_state.smooth(entity_id, base_score)
    return float(smoothed), {"enabled": True, **diagnostics}


def get_decision(final_score: float) -> Literal["APPROVE", "FLAG", "BLOCK"]:
    if final_score < approve_threshold:
        return "APPROVE"
    elif final_score < block_threshold:
        return "FLAG"
    return "BLOCK"


def generate_reasons(
    tx: Dict[str, Any],
    base_score: float,
    context_adjustment: float,
    behavior_reasons: List[str],
    final_score: float,
) -> List[str]:
    reasons: List[str] = []

    if base_score >= 0.90:
        reasons.append("Very high fraud probability from transaction model")
    elif base_score >= 0.60:
        reasons.append("High fraud probability from transaction model")
    elif base_score >= approve_threshold:
        reasons.append("Elevated fraud probability from transaction model")

    if tx["device_risk_score"] >= 0.70:
        reasons.append("High-risk device profile")
    elif tx["device_risk_score"] >= 0.40 and final_score >= approve_threshold:
        reasons.append("Moderate device risk")

    if tx["ip_risk_score"] >= 0.70:
        reasons.append("High-risk IP reputation")
    elif tx["ip_risk_score"] >= 0.40 and final_score >= approve_threshold:
        reasons.append("Moderate IP risk")

    if tx["location_risk_score"] >= 0.70:
        reasons.append("Location mismatch or elevated location risk")
    elif tx["location_risk_score"] >= 0.40 and final_score >= approve_threshold:
        reasons.append("Moderate location risk")

    if tx["TransactionAmt"] > 1000:
        reasons.append("Very high transaction amount")
    elif tx["TransactionAmt"] > 200 and final_score >= approve_threshold:
        reasons.append("Above-normal transaction amount")

    if context_adjustment >= 0.10:
        reasons.append("Contextual risk signals increased final risk score")
    elif context_adjustment >= 0.05 and final_score >= approve_threshold:
        reasons.append("Context contributed to elevated risk")

    if int(tx.get("device_shared_users_24h", 0)) >= 3:
        reasons.append("Device seen across multiple users in short window")

    if tx.get("sim_change_recent", False):
        reasons.append("Recent SIM change detected")

    if int(tx.get("account_age_days", 0)) < 7 and final_score >= approve_threshold:
        reasons.append("Very new account under elevated risk policy")

    if tx.get("tx_type") == "CASH_OUT" and tx["TransactionAmt"] > 300:
        reasons.append("High cash-out risk pattern")

    if int(tx.get("cash_flow_velocity_1h", 0)) >= 8:
        reasons.append("Unusually high wallet flow velocity")

    if int(tx.get("p2p_counterparties_24h", 0)) >= 12:
        reasons.append("High number of P2P counterparties in 24h")

    if tx.get("is_cross_border", False):
        reasons.append("Cross-border transaction risk context")

    reasons.extend(behavior_reasons)
    reasons = list(dict.fromkeys(reasons))

    if not reasons and final_score < approve_threshold:
        reasons.append("Low-risk transaction profile")

    return reasons


def build_context_summary(tx: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "device_risk": {
            "score": round(tx["device_risk_score"], 4),
            "source": "Simulated device fingerprint trust service",
            "meaning": "Measures whether the device appears trusted, consistent, and non-compromised"
        },
        "ip_risk": {
            "score": round(tx["ip_risk_score"], 4),
            "source": "Simulated IP reputation intelligence service",
            "meaning": "Measures network/IP reputation risk such as proxy/VPN abuse or malicious IP history"
        },
        "location_risk": {
            "score": round(tx["location_risk_score"], 4),
            "source": "Simulated geolocation mismatch engine",
            "meaning": "Measures whether the observed transaction location deviates from expected user geography"
        },
        "asean_shared_device_signals": {
            "device_id": tx.get("device_id"),
            "device_shared_users_24h": int(tx.get("device_shared_users_24h", 0)),
            "account_age_days": int(tx.get("account_age_days", 0)),
            "sim_change_recent": bool(tx.get("sim_change_recent", False)),
            "meaning": "Captures shared-device and low-history account-takeover signals common in mobile-first wallet ecosystems"
        },
        "wallet_flow_signals": {
            "tx_type": tx.get("tx_type", "MERCHANT"),
            "channel": tx.get("channel", "APP"),
            "cash_flow_velocity_1h": int(tx.get("cash_flow_velocity_1h", 0)),
            "p2p_counterparties_24h": int(tx.get("p2p_counterparties_24h", 0)),
            "is_cross_border": bool(tx.get("is_cross_border", False)),
            "meaning": "Captures cash-in/out bursts, P2P spread, and cross-border context often seen in super-app transactions"
        }
    }


def build_ml_input(tx_dict: Dict[str, float]) -> np.ndarray:
    missing_features = [col for col in feature_columns if col not in tx_dict]
    if missing_features:
        preview = missing_features[:20]
        remainder = len(missing_features) - len(preview)
        suffix = f" ... (+{remainder} more)" if remainder > 0 else ""
        raise ArtifactSchemaMismatchError(f"Missing required model features: {preview}{suffix}")

    values = [float(tx_dict[col]) for col in feature_columns]
    input_array = np.asarray([values], dtype=np.float32)

    if not np.isfinite(input_array).all():
        raise ArtifactSchemaMismatchError("Input contains null/NaN/inf values after preprocessing")

    return input_array


def build_ml_input_preprocessed(tx_dict: Dict[str, Any]) -> np.ndarray:
    if preprocessing_bundle is None:
        raise RuntimeError(
            "Preprocessing inference mode is enabled but preprocessing bundle is not loaded"
        )

    cache = getattr(_preprocessing_worker_cache, "static_artifacts", None)
    cache_key = (id(preprocessing_bundle), id(feature_column_indices))
    if cache is None or cache.get("key") != cache_key:
        if feature_column_indices is None:
            raise RuntimeError("feature column indices are not initialized")
        cache = {
            "key": cache_key,
            "feature_column_indices": feature_column_indices,
        }
        _preprocessing_worker_cache.static_artifacts = cache

    transformed = transform_runtime_record_with_bundle(preprocessing_bundle, tx_dict)
    transformed_dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)

    ordered = np.take(transformed_dense, cache["feature_column_indices"], axis=1).astype(np.float32, copy=False)
    if not np.isfinite(ordered).all():
        raise ArtifactSchemaMismatchError("Input contains null/NaN/inf values after preprocessing")
    return ordered


# ============================================================
# PRIVACY / AUDIT LOGGING
# ============================================================
def build_audit_record(
    request_id: str,
    tx_dict: Dict[str, Any],
    base_score: float,
    context_adjustment: float,
    behavior_adjustment: float,
    final_score: float,
    decision: str,
    decision_source: DecisionSource,
    user_segment: str,
    applied_thresholds: Dict[str, float],
    hard_rule_hits: List[str],
    verification_action: str,
    verification_reason: str | None,
    reasons: List[str],
    latency_ms: float,
) -> Dict[str, Any]:
    hashed_user_id = hash_user_id(tx_dict["user_id"])

    record = {
        "request_id": request_id,
        "timestamp_utc": utc_now_iso(),
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "hashed_user_id": hashed_user_id,
        "hash_key_version": HASH_KEY_VERSION,
        "decision": decision,
        "decision_source": decision_source,
        "user_segment": user_segment,
        "applied_thresholds": applied_thresholds,
        "hard_rule_hits": hard_rule_hits,
        "verification_action": verification_action,
        "verification_reason": verification_reason,
        "base_model_score": round(base_score, 4),
        "context_adjustment": round(context_adjustment, 4),
        "behavior_adjustment": round(behavior_adjustment, 4),
        "final_risk_score": round(final_score, 4),
        "latency_ms": latency_ms,
        "reason_codes": reasons,
        "context_scores": {
            "device_risk_score": round(tx_dict["device_risk_score"], 4),
            "ip_risk_score": round(tx_dict["ip_risk_score"], 4),
            "location_risk_score": round(tx_dict["location_risk_score"], 4),
        },
        "support_override": {
            "override_source": str(tx_dict.get("override_source", "preset")),
            "support_mode": bool(tx_dict.get("support_mode", False)),
            "support_actor_id": tx_dict.get("support_actor_id"),
            "override_fields": list(tx_dict.get("override_fields", [])),
        },
        "feature_integrity": {
            "feature_count": len(feature_columns),
            "model_inference_threads": MODEL_INFERENCE_THREADS,
            "inference_backend": FRAUD_INFERENCE_BACKEND,
            "inference_backend_runtime": (
                getattr(inference_backend, "runtime_name", inference_backend.backend_name)
                if inference_backend is not None
                else inference_backend_runtime
            ),
            "contains_raw_user_id": False,
            "contains_raw_transaction_vector": False,
        },
        "signature_key_version": AUDIT_SIGNING_KEY_VERSION,
    }
    return record


def write_audit_log(record: Dict[str, Any]) -> None:
    if AUDIT_ASYNC_WRITE_ENABLED:
        try:
            _audit_write_queue.put_nowait(record)
            return
        except queue.Full:
            logger.warning("audit_write_queue_full_falling_back_to_sync_write")
    _flush_audit_batch([record])



def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_dir_exists(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                loaded = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                records.append(loaded)
    return records


def overwrite_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    ensure_dir_exists(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def needs_manual_review(decision: str, final_score: float) -> bool:
    if decision == "BLOCK":
        return True
    if decision == "FLAG":
        return True
    return abs(final_score - approve_threshold) <= REVIEW_BORDERLINE_MARGIN


def queue_review_case(
    request_id: str,
    correlation_id: str,
    decision: str,
    final_score: float,
    reasons: List[str],
    context_summary: Dict[str, Any],
) -> None:
    if not needs_manual_review(decision, final_score):
        return

    queue_items = read_jsonl(REVIEW_QUEUE_FILE)
    for item in queue_items:
        if item.get("request_id") == request_id:
            return

    queue_item = {
        "request_id": request_id,
        "correlation_id": correlation_id,
        "created_at_utc": utc_now_iso(),
        "status": "pending",
        "decision": decision,
        "final_risk_score": round(final_score, 4),
        "reasons": reasons,
        "context_summary": context_summary,
    }
    append_jsonl(REVIEW_QUEUE_FILE, queue_item)


def find_audit_record(request_id: str) -> Dict[str, Any] | None:
    records = read_jsonl(AUDIT_LOG_FILE)
    for record in reversed(records):
        if record.get("request_id") == request_id:
            return record
    return None


def curate_retraining_example(outcome: Dict[str, Any], audit_record: Dict[str, Any] | None) -> Dict[str, Any]:
    analyst_decision = str(outcome.get("analyst_decision", "")).upper()
    label = 1 if analyst_decision == "FRAUD" else 0
    return {
        "request_id": outcome.get("request_id"),
        "curated_at_utc": utc_now_iso(),
        "label": label,
        "analyst_decision": analyst_decision,
        "analyst_confidence": outcome.get("analyst_confidence"),
        "analyst_notes": outcome.get("analyst_notes"),
        "model_decision": (audit_record or {}).get("decision"),
        "model_final_risk_score": (audit_record or {}).get("final_risk_score"),
        "transaction_amount": outcome.get("transaction_amount"),
        "reason_codes": (audit_record or {}).get("reason_codes", []),
        "context_scores": (audit_record or {}).get("context_scores", {}),
        "hashed_user_id": (audit_record or {}).get("hashed_user_id"),
    }


def summarize_dashboard_metrics(window_hours: int = 24) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    audits = read_jsonl(AUDIT_LOG_FILE)
    outcomes = read_jsonl(ANALYST_OUTCOMES_FILE)

    def parse_record_timestamp(record: Dict[str, Any]) -> datetime | None:
        timestamp = record.get("timestamp_utc")
        if not timestamp:
            return None
        try:
            return datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except ValueError:
            return None

    def in_window(record: Dict[str, Any]) -> bool:
        ts = parse_record_timestamp(record)
        if ts is None:
            return False
        return (now - ts).total_seconds() <= window_hours * 3600

    window_audits = [record for record in audits if in_window(record)]
    audit_by_request_id = {str(record.get("request_id")): record for record in window_audits if record.get("request_id")}
    scores = [float(record.get("final_risk_score", 0.0)) for record in window_audits]
    latencies = [float(record.get("latency_ms", 0.0)) for record in window_audits]
    window_timestamps = [ts for record in window_audits if (ts := parse_record_timestamp(record)) is not None]
    freshest_timestamp = max(window_timestamps) if window_timestamps else None
    freshness_seconds = (
        round((now - freshest_timestamp).total_seconds(), 3)
        if freshest_timestamp is not None
        else None
    )

    throughput_per_min = round(len(window_audits) / max(window_hours * 60, 1), 3)
    error_rate = 0.0

    baseline_mean = float(os.getenv("FRAUD_SCORE_BASELINE_MEAN", "0.25"))
    observed_mean = float(np.mean(scores)) if scores else 0.0
    drift_delta = round(observed_mean - baseline_mean, 4)

    matching = 0
    false_positives = 0
    confirmed_fraud = 0
    fraud_loss = 0.0
    for outcome in outcomes:
        analyst_decision = str(outcome.get("analyst_decision", "")).upper()
        model_decision = str(outcome.get("model_decision", "")).upper()
        if analyst_decision == "FRAUD":
            confirmed_fraud += 1
            fraud_loss += float(outcome.get("transaction_amount", 0.0) or 0.0)
        if analyst_decision == "LEGIT" and model_decision in {"FLAG", "BLOCK"}:
            false_positives += 1
        if (analyst_decision == "FRAUD" and model_decision in {"FLAG", "BLOCK"}) or (
            analyst_decision == "LEGIT" and model_decision == "APPROVE"
        ):
            matching += 1

    source_order: List[DecisionSource] = [
        "score_band",
        "hard_rule_override",
        "low_history_policy",
        "step_up_policy",
    ]
    source_rollups: Dict[str, Dict[str, int]] = {
        source: {
            "audit_volume": 0,
            "flag_count": 0,
            "review_count": 0,
            "confirmed_fraud_count": 0,
            "false_positive_count": 0,
        }
        for source in source_order
    }

    for record in window_audits:
        source = str(record.get("decision_source", "score_band"))
        if source not in source_rollups:
            continue
        source_rollups[source]["audit_volume"] += 1
        if str(record.get("decision", "")).upper() == "FLAG":
            source_rollups[source]["flag_count"] += 1

    for outcome in outcomes:
        request_id = str(outcome.get("request_id", ""))
        linked_audit = audit_by_request_id.get(request_id)
        source = str((linked_audit or {}).get("decision_source", "score_band"))
        if source not in source_rollups:
            continue
        source_rollups[source]["review_count"] += 1
        analyst_decision = str(outcome.get("analyst_decision", "")).upper()
        model_decision = str(outcome.get("model_decision", "")).upper()
        if analyst_decision == "FRAUD":
            source_rollups[source]["confirmed_fraud_count"] += 1
        if analyst_decision == "LEGIT" and model_decision in {"FLAG", "BLOCK"}:
            source_rollups[source]["false_positive_count"] += 1

    decision_source_kpis = []
    for source in source_order:
        counts = source_rollups[source]
        audit_volume = counts["audit_volume"]
        review_count = counts["review_count"]
        decision_source_kpis.append(
            {
                "decision_source": source,
                "audit_volume": audit_volume,
                "flag_rate": round(counts["flag_count"] / audit_volume, 4) if audit_volume else 0.0,
                "analyst_reviews": review_count,
                "confirmed_fraud_conversion": (
                    round(counts["confirmed_fraud_count"] / review_count, 4) if review_count else 0.0
                ),
                "false_positive_rate": (
                    round(counts["false_positive_count"] / review_count, 4) if review_count else 0.0
                ),
            }
        )

    agreement = round(matching / len(outcomes), 4) if outcomes else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0

    histogram = {"0-0.3": 0, "0.3-0.6": 0, "0.6-0.9": 0, "0.9-1.0": 0}
    for score in scores:
        if score < 0.3:
            histogram["0-0.3"] += 1
        elif score < 0.6:
            histogram["0.3-0.6"] += 1
        elif score < 0.9:
            histogram["0.6-0.9"] += 1
        else:
            histogram["0.9-1.0"] += 1

    return {
        "window_hours": window_hours,
        "generated_at_utc": now.isoformat(),
        "freshest_record_utc": freshest_timestamp.isoformat() if freshest_timestamp else None,
        "data_freshness_seconds": freshness_seconds,
        "latency_throughput_error": {
            "requests": len(window_audits),
            "throughput_per_min": throughput_per_min,
            "error_rate": error_rate,
            "latency_ms_p50": round(p50, 3),
            "latency_ms_p95": round(p95, 3),
        },
        "drift_score_distribution": {
            "baseline_mean_score": baseline_mean,
            "observed_mean_score": round(observed_mean, 4),
            "mean_delta": drift_delta,
            "score_histogram": histogram,
        },
        "fraud_loss_false_positives_analyst_agreement": {
            "estimated_fraud_loss": round(fraud_loss, 2),
            "false_positives": false_positives,
            "analyst_agreement": agreement,
            "confirmed_fraud_cases": confirmed_fraud,
            "analyst_reviews": len(outcomes),
        },
        "decision_source_kpis": decision_source_kpis,
    }

# ============================================================
# ERROR HANDLING
# ============================================================
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    return JSONResponse(
        status_code=400,
        content=build_api_error_response(
            error="Bad Request",
            detail=str(exc),
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(DomainError)
async def domain_error_handler(request: Request, exc: DomainError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    return JSONResponse(
        status_code=400,
        content=build_api_error_response(
            error="Bad Request",
            detail=str(exc),
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(UnknownTransactionTypeError)
async def unknown_transaction_type_handler(request: Request, exc: UnknownTransactionTypeError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    return JSONResponse(
        status_code=422,
        content=build_api_error_response(
            error="ValidationError",
            detail=str(exc),
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(UserProfileMismatchError)
async def user_profile_mismatch_handler(request: Request, exc: UserProfileMismatchError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    return JSONResponse(
        status_code=409,
        content=build_api_error_response(
            error="Conflict",
            detail=str(exc),
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(ReviewQueueRecordNotFoundError)
async def review_queue_record_not_found_handler(request: Request, exc: ReviewQueueRecordNotFoundError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    return JSONResponse(
        status_code=404,
        content=build_api_error_response(
            error="Not Found",
            detail=str(exc),
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(ArtifactSchemaMismatchError)
async def artifact_schema_mismatch_handler(request: Request, exc: ArtifactSchemaMismatchError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    alert_notifier.notify_domain_failure(
        AlertEvent(
            service=MODEL_NAME,
            severity="high",
            title="Hybrid Fraud API artifact schema mismatch",
            details={
                "route": request.url.path,
                "request_id": request_id,
                "exception_type": type(exc).__name__,
            },
        )
    )
    return JSONResponse(
        status_code=422,
        content=build_api_error_response(
            error="ValidationError",
            detail=str(exc),
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    details = build_validation_error_details(exc)
    return JSONResponse(
        status_code=422,
        content=build_api_error_response(
            error="ValidationError",
            detail="Request payload failed schema validation",
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=details,
            schema_version_expected=PAYLOAD_SCHEMA_VERSION,
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    if exc.status_code >= 500:
        alert_notifier.notify_internal_error(
            AlertEvent(
                service=MODEL_NAME,
                severity="critical",
                title="Hybrid Fraud API HTTP internal error",
                details={
                    "route": request.url.path,
                    "request_id": request_id,
                    "status_code": exc.status_code,
                    "exception_type": type(exc).__name__,
                },
            )
        )
    detail = extract_http_exception_detail(exc, "Request failed")
    return JSONResponse(
        status_code=exc.status_code,
        content=build_api_error_response(
            error="HTTPException",
            detail=detail,
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    alert_notifier.notify_internal_error(
        AlertEvent(
            service=MODEL_NAME,
            severity="critical",
            title="Hybrid Fraud API missing server artifact",
            details={
                "route": request.url.path,
                "request_id": request_id,
                "exception_type": type(exc).__name__,
                "error_category": category,
            },
        )
    )
    return JSONResponse(
        status_code=500,
        content=build_api_error_response(
            error="Server Configuration Error",
            detail="Server configuration error",
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    alert_notifier.notify_internal_error(
        AlertEvent(
            service=MODEL_NAME,
            severity="critical",
            title="Hybrid Fraud API unhandled exception",
            details={
                "route": request.url.path,
                "request_id": request_id,
                "exception_type": type(exc).__name__,
                "error_category": category,
            },
        )
    )
    return JSONResponse(
        status_code=500,
        content=build_api_error_response(
            error="Internal Server Error",
            detail="Internal server error",
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/api/info", response_model=HealthResponse, responses=SHARED_ERROR_RESPONSES)
def api_info():
    return HealthResponse(
        status="ok",
        api_version=API_VERSION,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        feature_count=len(feature_columns),
        approve_threshold=approve_threshold,
        block_threshold=block_threshold,
        segment_thresholds_summary=segment_thresholds_summary(),
    )


@app.get("/health", response_model=HealthResponse, responses=SHARED_ERROR_RESPONSES)
def health():
    return HealthResponse(
        status="ok",
        api_version=API_VERSION,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        feature_count=len(feature_columns),
        approve_threshold=approve_threshold,
        block_threshold=block_threshold,
        segment_thresholds_summary=segment_thresholds_summary(),
    )


@app.get("/health/ready", response_model=ReadinessResponse, responses=SHARED_ERROR_RESPONSES)
def readiness():
    artifact_paths = {
        "model_file": str(MODEL_FILE),
        "feature_file": str(FEATURE_FILE),
        "threshold_file": str(THRESHOLD_FILE),
    }

    for path in artifact_paths.values():
        ensure_file_exists(Path(path))

    ensure_dir_exists(AUDIT_DIR)

    return ReadinessResponse(
        status="ok",
        artifacts=artifact_paths,
        audit_log_path=str(AUDIT_LOG_FILE),
    )


@app.get("/health/artifacts", response_model=Dict[str, Any], responses=SHARED_ERROR_RESPONSES)
def health_artifacts():
    return {
        "status": "ok",
        "manifest_path": str(promoted_manifest.manifest_path),
        "model_metadata_family": promoted_artifact_validation.get("model_metadata_family"),
        "preprocessing_bundle_version": promoted_artifact_validation.get("preprocessing_bundle_version"),
        "feature_schema_hash": promoted_artifact_validation.get("feature_schema_hash"),
        "checks": promoted_artifact_validation.get("checks"),
    }


@app.get("/privacy", response_model=PrivacyResponse, responses=SHARED_ERROR_RESPONSES)
def privacy():
    return PrivacyResponse(
        api_version=API_VERSION,
        model_name=MODEL_NAME,
        privacy_principles=[
            "Raw user identifiers are not returned in scoring responses",
            "User identifiers are HMAC-hashed before audit logging with versioned secrets",
            "Secrets are read from FRAUD_* versioned env vars with local fallback for development",
            "Raw transaction vectors are not stored in audit logs",
            "Only derived risk scores, decision outputs, and reason codes are logged",
            "Scoring runs server-side to reduce exposure of sensitive logic and data"
        ],
        stored_fields_in_audit_log=[
            "request_id",
            "timestamp_utc",
            "api_version",
            "model_version",
            "hashed_user_id",
            "decision",
            "user_segment",
            "applied_thresholds",
            "hard_rule_hits",
            "verification_action",
            "verification_reason",
            "base_model_score",
            "context_adjustment",
            "behavior_adjustment",
            "final_risk_score",
            "latency_ms",
            "reason_codes",
            "context_scores",
            "hash_key_version",
            "signature_key_version",
            "previous_record_signature",
            "record_signature"
        ],
        not_stored_in_audit_log=[
            "raw user_id",
            "full transaction feature vector",
            "raw vector embeddings",
            "raw payment credentials",
            "card number",
            "wallet password or PIN"
        ],
        data_integrity_controls=[
            "strict input schema validation",
            "feature count enforcement",
            "server-side model artifact loading",
            "versioned model metadata",
            "hash-chained HMAC signature per audit record"
        ],
        retention_policy={
            "audit_retention_days": AUDIT_RETENTION_DAYS,
            "deletion_sla_days": AUDIT_DELETION_SLA_DAYS,
            "deletion_scope": "Audit records exceeding retention period are deleted from active storage and backups within SLA.",
            "hash_key_version": HASH_KEY_VERSION,
            "hash_secret_source": HASH_SECRET_SOURCE,
            "audit_signing_key_version": AUDIT_SIGNING_KEY_VERSION,
            "audit_signing_secret_source": AUDIT_SIGNING_SECRET_SOURCE,
        },
    )


@app.post("/score_transaction", response_model=ScoreTransactionResponse, responses=SCORE_TRANSACTION_RESPONSES)
def score_transaction(tx: ScoreTransactionRequest, request: Request):
    start = time.perf_counter()
    request_id = resolve_correlation_id(request)
    REQUEST_ID_CTX.set(request_id)
    stage_timings: Dict[str, float] = {}
    endpoint_name = "score_transaction"

    structured_log(
        "request_ingress",
        request_id=request_id,
        correlation_id=request_id,
        endpoint=endpoint_name,
        payload_schema_version=PAYLOAD_SCHEMA_VERSION,
    )

    stage_start = time.perf_counter()
    structured_log(
        "feature_preparation_start",
        request_id=request_id,
        correlation_id=request_id,
        endpoint=endpoint_name,
    )
    tx_dict = build_model_features_from_score_transaction_request(tx)
    feature_preparation_ms = round((time.perf_counter() - stage_start) * 1000, 3)
    structured_log(
        "feature_preparation_end",
        request_id=request_id,
        correlation_id=request_id,
        endpoint=endpoint_name,
        feature_preparation_ms=feature_preparation_ms,
    )
    stage_timings["feature_preparation_ms"] = feature_preparation_ms

    stage_start = time.perf_counter()
    structured_log(
        "preprocessing_start",
        request_id=request_id,
        correlation_id=request_id,
        endpoint=endpoint_name,
    )
    input_df = (
        build_ml_input_preprocessed(tx_dict)
        if USE_PREPROCESSING_INFERENCE
        else build_ml_input(tx_dict)
    )
    preprocessing_ms = round((time.perf_counter() - stage_start) * 1000, 3)
    stage_timings["preprocessing_ms"] = preprocessing_ms
    structured_log(
        "preprocessing_end",
        request_id=request_id,
        correlation_id=request_id,
        endpoint=endpoint_name,
        preprocessing_ms=preprocessing_ms,
    )

    stage_start = time.perf_counter()
    structured_log(
        "model_inference_start",
        request_id=request_id,
        correlation_id=request_id,
        endpoint=endpoint_name,
    )
    try:
        backend_scores = inference_backend.predict_positive_proba(model, input_df)
        base_score = float(np.asarray(backend_scores, dtype=np.float64)[0])
    except Exception as e:
        raise RuntimeError(f"Model scoring failed using backend {FRAUD_INFERENCE_BACKEND}: {e}") from e
    raw_model_base_score = base_score
    base_anchor_diag: Dict[str, Any] | None = None
    if tx_dict.get("_v_features_source") == "imputed_zero_vector":
        # External contract currently does not provide IEEE PCA components (V1..V17).
        # Using a hard-coded zero-vector can inflate risk unrealistically; anchor to a
        # calibrated prior so context/behavior signals remain meaningful.
        if DYNAMIC_IMPUTED_ANCHOR_ENABLED:
            base_score, base_anchor_diag = compute_imputed_base_anchor(tx_dict)
        else:
            base_score = CONTEXT_ONLY_BASE_SCORE
            base_anchor_diag = {
                "enabled": False,
                "source": "flat",
                "mode": "flat",
                "value": round(base_score, 4),
                "components": {},
            }
    model_predict_ms = round((time.perf_counter() - stage_start) * 1000, 3)
    stage_timings["model_inference_ms"] = model_predict_ms
    stage_timings["model_predict_ms"] = model_predict_ms
    structured_log(
        "model_inference_end",
        request_id=request_id,
        correlation_id=request_id,
        endpoint=endpoint_name,
        model_inference_ms=model_predict_ms,
    )

    if not (0.0 <= base_score <= 1.0):
        raise InvalidRiskScoreError(f"Model returned invalid probability: {base_score}", field="base_score")

    stage_start = time.perf_counter()
    smoothed_base_score, smoothing_diag = apply_entity_smoothing(base_score, tx_dict)
    stage_timings["entity_smoothing_ms"] = round((time.perf_counter() - stage_start) * 1000, 3)

    stage_start = time.perf_counter()
    as_of_ts = float(tx_dict["TransactionDT"])
    input_fingerprint = build_aggregate_input_fingerprint(tx_dict)
    cached_aggregates = feature_store.get_user_aggregates(
        user_id=tx_dict["user_id"],
        as_of_ts=as_of_ts,
    )
    if (
        cached_aggregates
        and cached_aggregates.get("input_fingerprint") == input_fingerprint
        and "context_adjustment" in cached_aggregates
    ):
        context_adjustment = float(cached_aggregates["context_adjustment"])
        context_source = "feature_store"
    else:
        context_adjustment = get_context_adjustment(tx_dict)
        context_source = "on_the_fly"
    context_adjustment_ms = round((time.perf_counter() - stage_start) * 1000, 3)
    stage_timings["context_scoring_ms"] = context_adjustment_ms
    stage_timings["context_adjustment_ms"] = context_adjustment_ms

    stage_start = time.perf_counter()
    behavior_result = get_behavior_adjustment(tx_dict, cached_aggregates=cached_aggregates)
    behavior_adjustment = behavior_result["behavior_adjustment"]
    behavior_reasons = behavior_result["behavior_reasons"]
    behavior_scoring_ms = round((time.perf_counter() - stage_start) * 1000, 3)
    stage_timings["behavior_scoring_ms"] = behavior_scoring_ms

    stage_start = time.perf_counter()
    final_score = clamp_score(smoothed_base_score + context_adjustment + behavior_adjustment)
    user_segment = compute_user_segment(tx_dict)
    segment_thresholds = SEGMENT_THRESHOLDS.get(
        user_segment,
        SegmentThresholds(approve_threshold=approve_threshold, block_threshold=block_threshold),
    )
    raw_decision = apply_segmented_decision(final_score, segment_thresholds)
    decision_after_low_history = apply_low_history_policy(
        final_score,
        raw_decision,
        behavior_result.get("behavior_features", {}),
    )
    decision = decision_after_low_history
    hard_rule_evaluation = evaluate_hard_rules(tx_dict, RULE_STATE_STORE)
    if hard_rule_evaluation.action == "BLOCK":
        decision = "BLOCK"
    elif hard_rule_evaluation.action == "FLAG" and decision == "APPROVE":
        decision = "FLAG"
    step_up_decision = determine_step_up_action(
        final_score=final_score,
        decision=decision,
        segment_thresholds=segment_thresholds,
        rule_hits=hard_rule_evaluation.rule_hits,
        tx=tx_dict,
    )
    decision_source = resolve_decision_source(
        raw_decision=raw_decision,
        decision_after_low_history=decision_after_low_history,
        final_decision=decision,
        hard_rule_action=hard_rule_evaluation.action,
        verification_action=step_up_decision.verification_action,
    )

    reasons = generate_reasons(
        tx_dict,
        smoothed_base_score,
        context_adjustment,
        behavior_reasons,
        final_score
    )

    if behavior_result.get("behavior_features", {}).get("is_low_history", False) and decision == "FLAG":
        reasons.append("Low-history account routed to verification (progressive friction policy)")
    if hard_rule_evaluation.rule_hits:
        reasons.append(f"Hard risk rules triggered: {', '.join(hard_rule_evaluation.rule_hits)}")
    if step_up_decision.verification_action != "NONE":
        reasons.append(f"Verification action required: {step_up_decision.verification_action}")

    context_summary = build_context_summary(tx_dict)
    context_summary["entity_smoothing"] = smoothing_diag
    context_summary["v_features_source"] = tx_dict.get("_v_features_source", "provided")
    context_summary["base_anchor"] = base_anchor_diag
    context_summary["base_model_score_raw_model"] = round(raw_model_base_score, 4)
    context_summary["base_model_score_raw"] = round(base_score, 4)
    if base_anchor_diag is not None:
        context_summary["base_anchor_source"] = base_anchor_diag["source"]
        context_summary["base_anchor_components"] = {
            key: round(float(value), 4)
            for key, value in base_anchor_diag["components"].items()
        }
        context_summary["base_anchor_pre_clamp"] = round(float(base_anchor_diag["pre_clamp"]), 4)
        context_summary["base_anchor_post_clamp"] = round(float(base_anchor_diag["post_clamp"]), 4)
    context_summary["base_model_score_smoothed"] = round(smoothed_base_score, 4)
    context_summary["user_segment"] = user_segment
    context_summary["segment_thresholds"] = {
        "approve_threshold": round(segment_thresholds.approve_threshold, 4),
        "block_threshold": round(segment_thresholds.block_threshold, 4),
        "min_block_precision": round(segment_thresholds.min_block_precision, 4),
        "max_approve_to_flag_fpr": round(segment_thresholds.max_approve_to_flag_fpr, 4),
    }
    context_summary["hard_rule_hits"] = hard_rule_evaluation.rule_hits
    context_summary["decision_source"] = decision_source
    context_summary["decision_attribution"] = {
        "decision_source": decision_source,
        "summary": (
            "Decision came from score bands."
            if decision_source == "score_band"
            else "Decision came from hard risk-rule override."
            if decision_source == "hard_rule_override"
            else "Decision came from low-history policy."
            if decision_source == "low_history_policy"
            else "Decision came from step-up policy."
        ),
    }
    context_summary["aggregate_sources"] = {
        "context": context_source,
        "behavior": behavior_result.get("source", "on_the_fly"),
    }
    stage_timings["decision_and_reasoning_ms"] = round((time.perf_counter() - stage_start) * 1000, 3)

    feature_store.upsert_user_aggregates(
        user_id=str(tx_dict["user_id"]),
        as_of_ts=as_of_ts,
        aggregates={
            "input_fingerprint": input_fingerprint,
            "context_adjustment": round(context_adjustment, 4),
            "behavior_features": behavior_result.get("behavior_features", {}),
            "behavior_adjustment": round(behavior_adjustment, 4),
            "behavior_reasons": behavior_reasons,
        },
    )

    # Asynchronously ingest aggregate update event (keeps scorer stateless/lightweight).
    stage_start = time.perf_counter()
    correlation_id = request_id
    transaction_id = derive_transaction_id(tx_dict, correlation_id)
    event = AggregateUpdateEvent(
        transaction_id=transaction_id,
        correlation_id=correlation_id,
        user_id=str(tx_dict["user_id"]),
        amount=float(tx_dict["TransactionAmt"]),
        hour_of_day=int(tx_dict["TransactionDT"] % 24),
        location_risk_score=float(tx_dict["location_risk_score"]),
        observed_at=time.time(),
    )
    queued = aggregate_ingestion_adapter.ingest(event)
    stage_timings["aggregate_enqueue_ms"] = round((time.perf_counter() - stage_start) * 1000, 3)
    stage_timings["aggregate_event_queued"] = 1.0 if queued else 0.0

    stage_start = time.perf_counter()
    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    # Privacy-safe audit record
    audit_record = build_audit_record(
        request_id=request_id,
        tx_dict=tx_dict,
        base_score=base_score,
        context_adjustment=context_adjustment,
        behavior_adjustment=behavior_adjustment,
        final_score=final_score,
        decision=decision,
        decision_source=decision_source,
        user_segment=user_segment,
        applied_thresholds={
            "approve_threshold": round(segment_thresholds.approve_threshold, 4),
            "block_threshold": round(segment_thresholds.block_threshold, 4),
            "min_block_precision": round(segment_thresholds.min_block_precision, 4),
            "max_approve_to_flag_fpr": round(segment_thresholds.max_approve_to_flag_fpr, 4),
        },
        hard_rule_hits=hard_rule_evaluation.rule_hits,
        verification_action=step_up_decision.verification_action,
        verification_reason=step_up_decision.reason,
        reasons=reasons,
        latency_ms=latency_ms,
    )
    audit_record["stage_timings_ms"] = stage_timings
    write_audit_log(audit_record)
    audit_logging_ms = round((time.perf_counter() - stage_start) * 1000, 3)
    stage_timings["audit_log_write_ms"] = audit_logging_ms
    stage_timings["audit_logging_ms"] = audit_logging_ms
    stage_timings["total_pipeline_ms"] = latency_ms

    logger.info(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "fraud_score_timing",
                "request_id": request_id,
                "schema_version": tx.schema_version,
                "decision": decision,
                "decision_source": decision_source,
                "user_segment": user_segment,
                "hard_rule_hits": hard_rule_evaluation.rule_hits,
                "verification_action": step_up_decision.verification_action,
                "final_risk_score": round(final_score, 4),
                "stage_timings_ms": stage_timings,
            }
        )
    )
    queue_review_case(
        request_id=request_id,
        correlation_id=request_id,
        decision=decision,
        final_score=final_score,
        reasons=reasons,
        context_summary=context_summary,
    )

    structured_log(
        "response_boundary",
        request_id=request_id,
        correlation_id=request_id,
        endpoint=endpoint_name,
        status="success",
        decision=decision,
        error_category=None,
    )

    final_score_rounded = round(final_score, 4)
    return ScoreTransactionResponse(
        request_id=request_id,
        correlation_id=request_id,
        risk_score=final_score_rounded,
        final_risk_score=final_score_rounded,
        decision=decision,
        decision_source=decision_source,
        fraud_reasons=reasons,
        reasons=reasons,
        explainability={
            "base": round(base_score, 4),
            "context": round(context_adjustment, 4),
            "behavior": round(behavior_adjustment, 4),
        },
        context_summary=context_summary,
        stage_timings_ms={
            "total_pipeline_ms": stage_timings.get("total_pipeline_ms"),
            "details": stage_timings,
        },
    )


@app.get("/review_queue", response_model=List[ReviewQueueItem], responses=SHARED_ERROR_RESPONSES)
def review_queue(status: Literal["pending", "resolved", "all"] = "pending"):
    queue_items = read_jsonl(REVIEW_QUEUE_FILE)
    if status == "all":
        filtered = queue_items
    else:
        filtered = [item for item in queue_items if item.get("status") == status]
    return list(reversed(filtered))


@app.post(
    "/review_queue/{request_id}/outcome",
    response_model=ReviewOutcomeResponse,
    responses=SHARED_ERROR_RESPONSES,
)
def submit_review_outcome(request_id: str, outcome: AnalystOutcomeInput):
    queue_items = read_jsonl(REVIEW_QUEUE_FILE)
    queue_record = None
    for item in queue_items:
        if item.get("request_id") == request_id:
            queue_record = item
            break

    if queue_record is None:
        raise ReviewQueueRecordNotFoundError(f"Unknown request_id '{request_id}' in review queue")

    queue_record["status"] = "resolved"
    queue_record["resolved_at_utc"] = utc_now_iso()
    overwrite_jsonl(REVIEW_QUEUE_FILE, queue_items)

    audit_record = find_audit_record(request_id)
    outcome_record = {
        "request_id": request_id,
        "reviewed_at_utc": utc_now_iso(),
        "analyst_id": outcome.analyst_id,
        "analyst_decision": outcome.analyst_decision,
        "analyst_confidence": outcome.analyst_confidence,
        "analyst_notes": outcome.analyst_notes,
        "transaction_amount": outcome.transaction_amount,
        "model_decision": queue_record.get("decision") if audit_record is None else audit_record.get("decision"),
        "model_final_risk_score": queue_record.get("final_risk_score") if audit_record is None else audit_record.get("final_risk_score"),
    }
    append_jsonl(ANALYST_OUTCOMES_FILE, outcome_record)

    curated = curate_retraining_example(outcome_record, audit_record)
    append_jsonl(RETRAINING_CURATION_FILE, curated)

    return {
        "status": "ok",
        "request_id": request_id,
        "review_status": queue_record["status"],
        "curated_label": curated["label"],
        "retraining_curation_file": str(RETRAINING_CURATION_FILE),
    }


@app.get(
    "/retraining/curation",
    response_model=RetrainingCurationResponse,
    responses=SHARED_ERROR_RESPONSES,
)
def retraining_curation(limit: int = 200):
    records = read_jsonl(RETRAINING_CURATION_FILE)
    return {
        "status": "ok",
        "count": len(records),
        "items": list(reversed(records))[: max(1, min(limit, 1000))],
    }


@app.get("/dashboard/views", response_model=DashboardViewsResponse, responses=SHARED_ERROR_RESPONSES)
def dashboard_views(window_hours: int = 24):
    bounded_window = max(1, min(window_hours, 24 * 30))
    return summarize_dashboard_metrics(window_hours=bounded_window)


@app.get("/config", response_model=Dict[str, Any], responses=SHARED_ERROR_RESPONSES)
def get_config():
    return {
        "api_version": API_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "approve_threshold": approve_threshold,
        "block_threshold": block_threshold,
        "segment_thresholds": segment_thresholds_summary(),
        "feature_count": len(feature_columns),
        "model_inference_threads": MODEL_INFERENCE_THREADS,
        "inference_backend": FRAUD_INFERENCE_BACKEND,
        "inference_backend_runtime": (
            getattr(inference_backend, "runtime_name", inference_backend.backend_name)
            if inference_backend is not None
            else inference_backend_runtime
        ),
        "artifact_init_count": _artifact_init_count,
        "use_preprocessing_inference": USE_PREPROCESSING_INFERENCE,
        "artifact_compatibility": artifact_compatibility,
        "required_transaction_fields": REQUIRED_SERVING_INPUT_FIELDS,
        "supported_context_fields": CONTEXT_FEATURE_FIELDS,
        "serving_schema": serving_schema_summary(),
        "expected_raw_model_features": EXPECTED_RAW_MODEL_FEATURES,
        "behavioral_profiling_enabled": True,
        "context_source_modeling_enabled": True,
        "entity_smoothing": {
            "method": ENTITY_SMOOTHING_METHOD,
            "min_history": ENTITY_SMOOTHING_MIN_HISTORY,
            "ema_alpha": ENTITY_SMOOTHING_EMA_ALPHA,
            "blend_alpha": ENTITY_SMOOTHING_BLEND_ALPHA,
            "blend_cap": ENTITY_SMOOTHING_BLEND_CAP,
            "fallback_for_unseen": ENTITY_SMOOTHING_FALLBACK,
        },
        "context_adjustment_max": CONTEXT_ADJUSTMENT_MAX,
        "context_weights": CONTEXT_WEIGHTS,
        "privacy_safe_audit_logging_enabled": True,
        "hash_key_version": HASH_KEY_VERSION,
        "hash_secret_versions_loaded": sorted(HASH_SECRETS.keys()),
        "hash_secret_source": HASH_SECRET_SOURCE,
        "hash_key_version_fallback_used": HASH_KEY_VERSION_FALLBACK_USED,
        "audit_signing_key_version": AUDIT_SIGNING_KEY_VERSION,
        "audit_signing_secret_versions_loaded": sorted(AUDIT_SIGNING_SECRETS.keys()),
        "audit_signing_secret_source": AUDIT_SIGNING_SECRET_SOURCE,
        "audit_signing_key_version_fallback_used": AUDIT_SIGNING_KEY_VERSION_FALLBACK_USED,
        "audit_retention_days": AUDIT_RETENTION_DAYS,
        "audit_deletion_sla_days": AUDIT_DELETION_SLA_DAYS,
        "artifact_paths": {
            "model_file": str(MODEL_FILE),
            "feature_file": str(FEATURE_FILE),
            "threshold_file": str(THRESHOLD_FILE),
            "preprocessing_bundle_file": str(PREPROCESSING_BUNDLE_FILE) if USE_PREPROCESSING_INFERENCE else None,
            "context_calibration_file": CONTEXT_CALIBRATION_FILE,
            "audit_log_file": str(AUDIT_LOG_FILE),
        }
    }

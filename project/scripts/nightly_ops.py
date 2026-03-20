import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests

from project.data.dataset_loader import load_ieee_cis


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXEC = sys.executable or "python"
DEFAULT_MONITORING_DIR = REPO_ROOT / "project" / "outputs" / "monitoring"
DEFAULT_DRIFT_JSON = DEFAULT_MONITORING_DIR / "drift_report.json"
DEFAULT_DRIFT_CSV = DEFAULT_MONITORING_DIR / "drift_feature_psi.csv"
DEFAULT_OPS_SUMMARY_JSON = DEFAULT_MONITORING_DIR / "nightly_ops_summary.json"
DEFAULT_PROFILE_HEALTH_JSON = DEFAULT_MONITORING_DIR / "behavior_profile_health.json"
DEFAULT_PROFILE_REPLAY_JSON = DEFAULT_MONITORING_DIR / "profile_replay_summary.json"
DEFAULT_COHORT_KPI_JSON = DEFAULT_MONITORING_DIR / "cohort_kpi_report.json"
DEFAULT_COHORT_KPI_CSV = DEFAULT_MONITORING_DIR / "cohort_kpi_report.csv"
DEFAULT_LATENCY_TREND_JSON = DEFAULT_MONITORING_DIR / "latency_trend_report.json"
DEFAULT_LATENCY_TREND_CSV = DEFAULT_MONITORING_DIR / "latency_trend_report.csv"
DEFAULT_LATENCY_STAGE_ANALYSIS_JSON = DEFAULT_MONITORING_DIR / "latency_stage_analysis.json"
DEFAULT_LATENCY_STAGE_ANALYSIS_CSV = DEFAULT_MONITORING_DIR / "latency_stage_analysis.csv"
DEFAULT_LATENCY_STAGE_MATRIX_JSON = DEFAULT_MONITORING_DIR / "latency_stage_matrix_report.json"
DEFAULT_LATENCY_STAGE_MATRIX_CSV = DEFAULT_MONITORING_DIR / "latency_stage_matrix_report.csv"
DEFAULT_CALIBRATION_JSON = REPO_ROOT / "project" / "outputs" / "figures" / "tables" / "context_calibration.json"
DEFAULT_CALIBRATION_CSV = REPO_ROOT / "project" / "outputs" / "figures" / "tables" / "context_calibration_trials.csv"
DEFAULT_MODEL_PATH = REPO_ROOT / "project" / "models" / "final_xgboost_model.pkl"
DEFAULT_FEATURE_PATH = REPO_ROOT / "project" / "models" / "feature_columns.pkl"
DEFAULT_PREPROCESSING_ARTIFACT_PATH = REPO_ROOT / "project" / "models" / "preprocessing_artifact_promoted.pkl"
DEFAULT_THRESHOLDS_OUTPUT = REPO_ROOT / "project" / "models" / "decision_thresholds.pkl"
DEFAULT_LEGACY_MODEL_PATH = REPO_ROOT / "project" / "legacy_creditcard" / "models" / "final_xgboost_model.pkl"
DEFAULT_LEGACY_FEATURE_PATH = REPO_ROOT / "project" / "legacy_creditcard" / "models" / "feature_columns.pkl"
DEFAULT_LEGACY_THRESHOLDS_OUTPUT = REPO_ROOT / "project" / "legacy_creditcard" / "models" / "decision_thresholds.pkl"
DEFAULT_IEEE_MODEL_PATH = REPO_ROOT / "project" / "models" / "final_xgboost_model_promoted_preproc.pkl"
DEFAULT_IEEE_FEATURE_PATH = REPO_ROOT / "project" / "models" / "feature_columns_promoted_preproc.pkl"
DEFAULT_IEEE_THRESHOLDS_OUTPUT = REPO_ROOT / "project" / "models" / "decision_thresholds_promoted_preproc.pkl"
DEFAULT_PROMOTION_RECORD_JSON = REPO_ROOT / "project" / "outputs" / "threshold_governance" / "latest_promotion_record.json"
DEFAULT_OPS_RUN_ARCHIVE_DIR = REPO_ROOT / "project" / "outputs" / "ops_runs"
DEFAULT_ARTIFACT_VALIDATION_JSON = DEFAULT_MONITORING_DIR / "artifact_validation_report.json"

# Deterministic nightly-ops summary schema.
# Keep this tuple in the exact output order expected by downstream consumers.
OPS_SUMMARY_SCHEMA_KEYS: tuple[str, ...] = (
    "generated_at_utc",
    "status",
    "dataset_artifact_validation",
    "drift_monitor",
    "drift_report_summary",
    "drift_recalibration_recommendation",
    "calibration",
    "threshold_promotion",
    "artifact_validation",
    "artifact_validation_report",
    "benchmark",
    "benchmark_sla",
    "benchmark_error_summary",
    "benchmark_sla_mode",
    "latency_trend",
    "latency_stage_analysis",
    "latency_stage_matrix",
    "cohort_kpi",
    "decision_source_kpi",
    "profile_replay",
    "profile_health",
    "profile_health_summary",
    "profile_health_status",
    "data_checks",
    "threshold_policy_guardrails",
    "retrain_trigger",
    "archive",
    "alert",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nightly orchestration: drift monitor -> optional recalibration -> optional benchmark."
    )
    parser.add_argument("--dataset-source", choices=["creditcard", "ieee_cis"], default="ieee_cis")
    parser.add_argument("--dataset-path", type=Path, help="Legacy credit-card CSV path (used when --dataset-source creditcard)")
    parser.add_argument("--ieee-transaction-path", type=Path)
    parser.add_argument("--ieee-identity-path", type=Path)
    parser.add_argument(
        "--ieee-merged-cache-path",
        type=Path,
        default=DEFAULT_MONITORING_DIR / "ieee_ops_merged_dataset.csv",
        help="Cache path used to materialize merged IEEE dataset for ops scripts that require a single CSV.",
    )
    parser.add_argument("--audit-log", type=Path, default=REPO_ROOT / "project" / "outputs" / "audit" / "fraud_audit_log.jsonl")
    parser.add_argument("--analyst-outcomes-log", type=Path, default=REPO_ROOT / "project" / "outputs" / "audit" / "analyst_outcomes.jsonl")
    parser.add_argument("--drift-json", type=Path, default=DEFAULT_DRIFT_JSON)
    parser.add_argument("--drift-csv", type=Path, default=DEFAULT_DRIFT_CSV)
    parser.add_argument("--ops-summary-json", type=Path, default=DEFAULT_OPS_SUMMARY_JSON)

    parser.add_argument("--skip-data-checks", action="store_true")
    parser.add_argument("--skip-threshold-policy-guardrails", action="store_true")
    parser.add_argument("--archive-runs-dir", type=Path, default=DEFAULT_OPS_RUN_ARCHIVE_DIR)
    parser.add_argument("--skip-archive", action="store_true")

    parser.add_argument("--retrain-feature-alert-threshold", type=int, default=2)
    parser.add_argument("--retrain-decision-drift-status", default="alert", choices=["warn", "alert"])
    parser.add_argument(
        "--retrain-warn-streak",
        type=int,
        default=3,
        help="Trigger retraining when decision drift status=warn persists for N consecutive archived+current runs.",
    )
    parser.add_argument(
        "--retrain-sla-fail-streak",
        type=int,
        default=3,
        help="Trigger retraining when benchmark SLA fails for N consecutive archived+current runs.",
    )
    parser.add_argument(
        "--retrain-endpoint-error-rate-streak",
        type=int,
        default=3,
        help="Trigger retraining when any endpoint exceeds --retrain-endpoint-error-rate-threshold for N consecutive runs.",
    )
    parser.add_argument(
        "--retrain-endpoint-error-rate-threshold",
        type=float,
        default=5.0,
        help="Endpoint error-rate percentage threshold used to compute consecutive high-error streaks.",
    )
    parser.add_argument("--retrain-on-data-check-fail", action="store_true", default=True)
    parser.add_argument("--retrain-on-threshold-guardrail-fail", action="store_true", default=True)

    parser.add_argument("--psi-warn", type=float, default=0.10)
    parser.add_argument("--psi-alert", type=float, default=0.25)
    parser.add_argument("--decision-drift-warn", type=float, default=0.10)
    parser.add_argument("--decision-drift-alert", type=float, default=0.20)

    parser.add_argument("--run-calibration-on-drift", action="store_true", default=True)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--calibration-trials", type=int, default=500)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--target-fpr", type=float, default=0.005)
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION_JSON)
    parser.add_argument("--calibration-csv", type=Path, default=DEFAULT_CALIBRATION_CSV)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--preprocessing-artifact-path", type=Path, default=DEFAULT_PREPROCESSING_ARTIFACT_PATH)
    parser.add_argument("--thresholds-output", type=Path, default=DEFAULT_THRESHOLDS_OUTPUT)
    parser.add_argument("--promote-thresholds-on-pass", action="store_true", default=True)
    parser.add_argument("--skip-threshold-promotion", action="store_true")
    parser.add_argument("--promotion-record-json", type=Path, default=DEFAULT_PROMOTION_RECORD_JSON)
    parser.add_argument("--artifact-validation-json", type=Path, default=DEFAULT_ARTIFACT_VALIDATION_JSON)
    parser.add_argument("--skip-artifact-validation", action="store_true")

    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument(
        "--benchmark-sla-mode",
        choices=["enforce", "warn", "off"],
        default="enforce",
        help="How benchmark SLA failures affect nightly status: enforce=fail status, warn=record only, off=skip SLA parsing.",
    )
    parser.add_argument("--benchmark-requests", type=int, default=200)
    parser.add_argument("--benchmark-concurrency", type=int, default=12)
    parser.add_argument("--benchmark-timeout", type=float, default=8.0)
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--benchmark-sla-p95-ms", type=float, default=250.0)
    parser.add_argument("--benchmark-sla-p99-ms", type=float, default=500.0)
    parser.add_argument("--benchmark-sla-error-rate-pct", type=float, default=1.0)
    parser.add_argument("--benchmark-trend-history-limit", type=int, default=20)
    parser.add_argument("--latency-trend-json", type=Path, default=DEFAULT_LATENCY_TREND_JSON)
    parser.add_argument("--latency-trend-csv", type=Path, default=DEFAULT_LATENCY_TREND_CSV)
    parser.add_argument("--latency-stage-analysis-json", type=Path, default=DEFAULT_LATENCY_STAGE_ANALYSIS_JSON)
    parser.add_argument("--latency-stage-analysis-csv", type=Path, default=DEFAULT_LATENCY_STAGE_ANALYSIS_CSV)
    parser.add_argument("--run-benchmark-matrix", action="store_true")
    parser.add_argument("--latency-stage-matrix-json", type=Path, default=DEFAULT_LATENCY_STAGE_MATRIX_JSON)
    parser.add_argument("--latency-stage-matrix-csv", type=Path, default=DEFAULT_LATENCY_STAGE_MATRIX_CSV)
    parser.add_argument("--latency-stage-matrix-history-runs", type=int, default=7)

    parser.add_argument("--run-cohort-kpi", action="store_true")
    parser.add_argument("--cohort-kpi-json", type=Path, default=DEFAULT_COHORT_KPI_JSON)
    parser.add_argument("--cohort-kpi-csv", type=Path, default=DEFAULT_COHORT_KPI_CSV)

    parser.add_argument("--run-profile-replay", action="store_true")
    parser.add_argument("--profile-replay-json", type=Path, default=DEFAULT_PROFILE_REPLAY_JSON)
    parser.add_argument("--profile-replay-user-count", type=int, default=200)
    parser.add_argument("--profile-replay-transactions-per-user", type=int, default=12)

    parser.add_argument("--alert-webhook-url", default="")
    parser.add_argument("--alert-timeout", type=float, default=5.0)

    parser.add_argument("--run-profile-health", action="store_true")
    parser.add_argument("--profile-store-backend", default="sqlite")
    parser.add_argument(
        "--profile-sqlite-path",
        type=Path,
        default=REPO_ROOT / "project" / "outputs" / "behavior_profiles.sqlite3",
    )
    parser.add_argument("--profile-health-json", type=Path, default=DEFAULT_PROFILE_HEALTH_JSON)
    parser.add_argument("--profile-min-history", type=int, default=5)
    parser.add_argument("--profile-stale-seconds", type=int, default=7 * 24 * 3600)

    return parser.parse_args()


def resolve_ops_dataset_csv(args: argparse.Namespace) -> Path:
    dataset_source = str(getattr(args, "dataset_source", "creditcard"))
    if dataset_source != "ieee_cis":
        return Path(args.dataset_path)

    tx_path = getattr(args, "ieee_transaction_path", None)
    id_path = getattr(args, "ieee_identity_path", None)
    if tx_path is None or id_path is None:
        raise ValueError("--ieee-transaction-path and --ieee-identity-path are required for ieee_cis source")

    features, labels, _ = load_ieee_cis(tx_path, id_path)
    merged_df = features.copy()
    merged_df["Class"] = labels.astype(int).to_numpy()

    cache_path = Path(getattr(args, "ieee_merged_cache_path", DEFAULT_MONITORING_DIR / "ieee_ops_merged_dataset.csv"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(cache_path, index=False)
    return cache_path


def run_command(cmd: list[str]) -> Dict[str, Any]:
    started = _utc_now_iso()
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "command": cmd,
        "started_at_utc": started,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ok": result.returncode == 0,
    }


def evaluate_benchmark_sla(benchmark_cmd: Dict[str, Any] | None) -> Dict[str, Any] | None:
    def _classify_failure(
        *,
        ok: bool,
        reason: str | None = None,
        benchmark_mode: str | None = None,
        failure_mode: str | None = None,
    ) -> tuple[str, str]:
        if ok:
            return ("none", "healthy")

        normalized_reason = str(reason or "").strip().lower()
        normalized_mode = str(benchmark_mode or "").strip().lower()
        normalized_failure_mode = str(failure_mode or "").strip().lower()

        if normalized_failure_mode == "sla_violation" or normalized_reason == "benchmark_sla_failed":
            return ("runtime_sla_failure", "model_performance_degradation")
        if normalized_failure_mode == "invalid_payload_contract" or normalized_mode == "contract_validation_failed":
            return ("invalid_payload_contract", "pipeline_integration_bug")
        if normalized_mode == "preflight_failed" or normalized_reason == "benchmark_unreachable_services":
            return ("preflight_validation_failure", "pipeline_integration_bug")
        return ("integration_setup_failure", "pipeline_integration_bug")

    if benchmark_cmd is None:
        return None

    stdout = benchmark_cmd.get("stdout", "")
    if not isinstance(stdout, str) or not stdout.strip():
        failure_category, operator_diagnosis = _classify_failure(ok=False, reason="benchmark_stdout_missing")
        return {
            "ok": False,
            "reason": "benchmark_stdout_missing",
            "failing_endpoints": [],
            "failure_category": failure_category,
            "operator_diagnosis": operator_diagnosis,
        }

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        failure_category, operator_diagnosis = _classify_failure(ok=False, reason="benchmark_stdout_not_json")
        return {
            "ok": False,
            "reason": "benchmark_stdout_not_json",
            "failing_endpoints": [],
            "failure_category": failure_category,
            "operator_diagnosis": operator_diagnosis,
        }

    if str(payload.get("benchmark_mode", "")).lower() == "preflight_failed":
        failed_preflight = payload.get("failed_preflight_endpoints") or []
        failure_category, operator_diagnosis = _classify_failure(
            ok=False,
            reason="benchmark_unreachable_services",
            benchmark_mode=payload.get("benchmark_mode"),
            failure_mode=payload.get("failure_mode"),
        )
        return {
            "ok": False,
            "reason": "benchmark_unreachable_services",
            "failing_endpoints": list(failed_preflight) if isinstance(failed_preflight, list) else [],
            "benchmark_mode": payload.get("benchmark_mode"),
            "failure_mode": payload.get("failure_mode"),
            "preflight": payload.get("preflight"),
            "failure_category": failure_category,
            "operator_diagnosis": operator_diagnosis,
        }

    if str(payload.get("benchmark_mode", "")).lower() == "contract_validation_failed":
        contract_validation = payload.get("contract_validation") if isinstance(payload.get("contract_validation"), dict) else {}
        errors = contract_validation.get("errors") if isinstance(contract_validation.get("errors"), list) else []
        failure_category, operator_diagnosis = _classify_failure(
            ok=False,
            reason="benchmark_contract_validation_failed",
            benchmark_mode=payload.get("benchmark_mode"),
            failure_mode=payload.get("failure_mode"),
        )
        return {
            "ok": False,
            "reason": "benchmark_contract_validation_failed",
            "failing_endpoints": [],
            "benchmark_mode": payload.get("benchmark_mode"),
            "failure_mode": payload.get("failure_mode"),
            "failure_category": failure_category,
            "operator_diagnosis": operator_diagnosis,
            "contract_validation": contract_validation,
            "remediation": (
                "Benchmark payload contract validation failed. Regenerate benchmark payload builders from "
                "project/app/contracts.py and project/app/schema_spec.py before rerunning nightly ops."
            ),
            "contract_validation_error_count": len(errors),
        }

    if not benchmark_cmd.get("ok", False):
        failure_category, operator_diagnosis = _classify_failure(
            ok=False,
            reason="benchmark_subprocess_failed",
            benchmark_mode=payload.get("benchmark_mode"),
            failure_mode=payload.get("failure_mode"),
        )
        return {
            "ok": False,
            "reason": "benchmark_subprocess_failed",
            "failing_endpoints": [],
            "benchmark_mode": payload.get("benchmark_mode"),
            "failure_mode": payload.get("failure_mode"),
            "failure_category": failure_category,
            "operator_diagnosis": operator_diagnosis,
        }

    sla_evaluation = payload.get("sla_evaluation")
    if not isinstance(sla_evaluation, dict):
        failure_category, operator_diagnosis = _classify_failure(
            ok=False,
            reason="benchmark_sla_evaluation_missing",
            benchmark_mode=payload.get("benchmark_mode"),
            failure_mode=payload.get("failure_mode"),
        )
        return {
            "ok": False,
            "reason": "benchmark_sla_evaluation_missing",
            "failing_endpoints": [],
            "benchmark_mode": payload.get("benchmark_mode"),
            "failure_mode": payload.get("failure_mode"),
            "failure_category": failure_category,
            "operator_diagnosis": operator_diagnosis,
        }

    failing_endpoints = []
    uncategorized_endpoints = []
    for endpoint, endpoint_eval in sla_evaluation.items():
        endpoint_eval = endpoint_eval or {}
        if str(endpoint_eval.get("real_time_viability", "")).upper() != "PASS":
            failing_endpoints.append(endpoint)
        checks = endpoint_eval.get("checks") or {}
        unknown_guardrail = checks.get("unknown_internal_guardrail") or {}
        if unknown_guardrail.get("pass") is False:
            uncategorized_endpoints.append(endpoint)
    ok = len(failing_endpoints) == 0
    reason = "ok" if ok else "benchmark_sla_failed"
    failure_category, operator_diagnosis = _classify_failure(
        ok=ok,
        reason=reason,
        benchmark_mode=payload.get("benchmark_mode"),
        failure_mode=payload.get("failure_mode"),
    )
    return {
        "ok": ok,
        "reason": reason,
        "failing_endpoints": failing_endpoints,
        "uncategorized_endpoints": uncategorized_endpoints,
        "benchmark_mode": payload.get("benchmark_mode"),
        "failure_mode": payload.get("failure_mode"),
        "failure_category": failure_category,
        "operator_diagnosis": operator_diagnosis,
        "sla_evaluation": sla_evaluation,
    }


def summarize_benchmark_errors(benchmark_cmd: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if benchmark_cmd is None:
        return None
    stdout = benchmark_cmd.get("stdout", "")
    if not isinstance(stdout, str) or not stdout.strip():
        return None
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None

    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, list):
        return None

    endpoint_summary: Dict[str, Dict[str, Any]] = {}
    for endpoint_data in endpoints:
        if not isinstance(endpoint_data, dict):
            continue
        endpoint_name = str(endpoint_data.get("endpoint_name", "unknown"))
        endpoint_summary[endpoint_name] = {
            "error_category_distribution": endpoint_data.get("error_categories") or {},
            "first_failure_timestamp_utc": endpoint_data.get("started_at_utc")
            if int(endpoint_data.get("error_count", 0) or 0) > 0
            else None,
            "top_exception_signatures": endpoint_data.get("top_exception_signatures") or [],
        }
    return endpoint_summary


def normalize_benchmark_sla_mode(mode: str | None) -> str:
    value = str(mode or "enforce").strip().lower()
    if value in {"enforce", "warn", "off"}:
        return value
    return "enforce"


def append_dataset_source_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    out = list(cmd)
    out.extend(["--dataset-source", str(args.dataset_source)])
    if args.dataset_source == "creditcard":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for creditcard source (legacy assets now in project/legacy_creditcard)")
        out.extend(["--dataset-path", str(args.dataset_path)])
        return out

    if not args.ieee_transaction_path or not args.ieee_identity_path:
        raise ValueError("--ieee-transaction-path and --ieee-identity-path are required for ieee_cis source")
    out.extend(["--ieee-transaction-path", str(args.ieee_transaction_path)])
    out.extend(["--ieee-identity-path", str(args.ieee_identity_path)])
    return out


def resolve_dataset_display_path(args: argparse.Namespace) -> str:
    if args.dataset_source == "creditcard":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for creditcard source (legacy assets now in project/legacy_creditcard)")
        return str(args.dataset_path)
    return f"{args.ieee_transaction_path}|{args.ieee_identity_path}"


def resolve_model_artifact_paths(args: argparse.Namespace) -> Dict[str, Path]:
    if str(getattr(args, "dataset_source", "creditcard")) != "ieee_cis":
        return {
            "model_path": Path(getattr(args, "model_path", DEFAULT_MODEL_PATH)),
            "feature_path": Path(getattr(args, "feature_path", DEFAULT_FEATURE_PATH)),
            "thresholds_output": Path(getattr(args, "thresholds_output", DEFAULT_THRESHOLDS_OUTPUT)),
            "preprocessing_artifact_path": Path(getattr(args, "preprocessing_artifact_path", DEFAULT_PREPROCESSING_ARTIFACT_PATH)),
        }

    model_path = Path(getattr(args, "model_path", DEFAULT_MODEL_PATH))
    feature_path = Path(getattr(args, "feature_path", DEFAULT_FEATURE_PATH))
    thresholds_output = Path(getattr(args, "thresholds_output", DEFAULT_THRESHOLDS_OUTPUT))
    preprocessing_artifact_path = Path(getattr(args, "preprocessing_artifact_path", DEFAULT_PREPROCESSING_ARTIFACT_PATH))

    if model_path == DEFAULT_MODEL_PATH:
        model_path = DEFAULT_IEEE_MODEL_PATH
    if feature_path == DEFAULT_FEATURE_PATH:
        feature_path = DEFAULT_IEEE_FEATURE_PATH
    if thresholds_output == DEFAULT_THRESHOLDS_OUTPUT:
        thresholds_output = DEFAULT_IEEE_THRESHOLDS_OUTPUT

    return {
        "model_path": model_path,
        "feature_path": feature_path,
        "thresholds_output": thresholds_output,
        "preprocessing_artifact_path": preprocessing_artifact_path,
    }


def validate_dataset_artifact_compatibility(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_source = str(getattr(args, "dataset_source", "creditcard"))
    artifacts = resolve_model_artifact_paths(args)
    model_path = Path(artifacts["model_path"])
    feature_path = Path(artifacts["feature_path"])
    thresholds_output = Path(artifacts["thresholds_output"])

    def _canonical_path(path_value: Path | str) -> Path:
        path = Path(path_value).expanduser()
        return path.resolve(strict=False)

    report: Dict[str, Any] = {
        "started_at_utc": _utc_now_iso(),
        "dataset_source": dataset_source,
        "model_path": str(model_path),
        "feature_path": str(feature_path),
        "thresholds_output": str(thresholds_output),
    }

    if dataset_source == "ieee_cis":
        missing_ieee_inputs = []
        if not getattr(args, "ieee_transaction_path", None):
            missing_ieee_inputs.append("ieee_transaction_path")
        if not getattr(args, "ieee_identity_path", None):
            missing_ieee_inputs.append("ieee_identity_path")
        if missing_ieee_inputs:
            report["ok"] = False
            report["reason"] = f"missing_ieee_dataset_inputs:{','.join(missing_ieee_inputs)}"
            return report

        expected_ieee = {
            "model_path": DEFAULT_IEEE_MODEL_PATH,
            "feature_path": DEFAULT_IEEE_FEATURE_PATH,
            "thresholds_output": DEFAULT_IEEE_THRESHOLDS_OUTPUT,
        }
        mismatches = {
            key: {"expected": str(expected_path), "actual": report[key]}
            for key, expected_path in expected_ieee.items()
            if _canonical_path(report[key]) != _canonical_path(expected_path)
        }
        if mismatches:
            report["ok"] = False
            report["reason"] = "ieee_source_requires_promoted_ieee_artifacts"
            report["mismatches"] = mismatches
            return report

        report["ok"] = True
        return report

    if not getattr(args, "dataset_path", None):
        report["ok"] = False
        report["reason"] = "missing_creditcard_dataset_path"
        return report

    expected_legacy = {
        "model_path": DEFAULT_LEGACY_MODEL_PATH,
        "feature_path": DEFAULT_LEGACY_FEATURE_PATH,
        "thresholds_output": DEFAULT_LEGACY_THRESHOLDS_OUTPUT,
    }
    mismatches = {
        key: {"expected": str(expected_path), "actual": report[key]}
        for key, expected_path in expected_legacy.items()
        if _canonical_path(report[key]) != _canonical_path(expected_path)
    }
    if mismatches:
        report["ok"] = False
        report["reason"] = "creditcard_source_requires_legacy_artifacts"
        report["mismatches"] = mismatches
        return report

    report["ok"] = True
    return report


def run_drift_monitor(args: argparse.Namespace) -> Dict[str, Any]:
    baseline_dataset = resolve_ops_dataset_csv(args)
    cmd = [
        PYTHON_EXEC,
        "project/scripts/drift_monitor.py",
        "--baseline-dataset",
        str(baseline_dataset),
        "--audit-log",
        str(args.audit_log),
        "--output-json",
        str(args.drift_json),
        "--output-csv",
        str(args.drift_csv),
        "--psi-warn",
        str(args.psi_warn),
        "--psi-alert",
        str(args.psi_alert),
        "--decision-drift-warn",
        str(args.decision_drift_warn),
        "--decision-drift-alert",
        str(args.decision_drift_alert),
    ]
    return run_command(cmd)


def run_data_checks(args: argparse.Namespace) -> Dict[str, Any]:
    started = _utc_now_iso()

    if args.dataset_source == "ieee_cis" and (not args.ieee_transaction_path or not args.ieee_identity_path):
        return {
            "started_at_utc": started,
            "ok": False,
            "reason": "missing_ieee_paths",
            "dataset_source": args.dataset_source,
            "dataset_path": resolve_dataset_display_path(args),
        }

    dataset = resolve_ops_dataset_csv(args)
    if not dataset.exists():
        return {
            "started_at_utc": started,
            "ok": False,
            "reason": f"dataset_missing:{dataset}",
            "dataset_source": args.dataset_source,
            "dataset_path": resolve_dataset_display_path(args),
        }

    df = pd.read_csv(dataset)
    required = ["Class"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {
            "started_at_utc": started,
            "ok": False,
            "reason": f"missing_columns:{missing}",
            "dataset_source": args.dataset_source,
            "dataset_path": resolve_dataset_display_path(args),
            "row_count": int(len(df)),
        }

    labels = pd.to_numeric(df["Class"], errors="coerce")
    invalid_label_count = int((~labels.isin([0, 1])).sum())
    null_ratio = float(df.isna().mean().mean()) if len(df.columns) else 0.0
    return {
        "started_at_utc": started,
        "ok": invalid_label_count == 0,
        "dataset_source": args.dataset_source,
        "dataset_path": resolve_dataset_display_path(args),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "invalid_label_count": invalid_label_count,
        "global_null_ratio": round(null_ratio, 6),
    }


def run_threshold_policy_guardrails(args: argparse.Namespace) -> Dict[str, Any]:
    started = _utc_now_iso()
    thresholds_path = args.thresholds_output
    if not thresholds_path.exists():
        return {
            "started_at_utc": started,
            "ok": False,
            "reason": f"thresholds_missing:{thresholds_path}",
            "thresholds_path": str(thresholds_path),
        }

    try:
        import pickle

        with thresholds_path.open("rb") as f:
            payload = pickle.load(f)
        approve = float(payload.get("approve_threshold", 0.30))
        block = float(payload.get("block_threshold", 0.90))
        ok = 0.0 <= approve < block <= 1.0
        report = {
            "started_at_utc": started,
            "ok": ok,
            "thresholds_path": str(thresholds_path),
            "approve_threshold": approve,
            "block_threshold": block,
        }
        if not ok:
            report["reason"] = "invalid_threshold_order_or_range"

        promotion_record = getattr(args, "promotion_record_json", None)
        if promotion_record is not None and Path(promotion_record).exists():
            record = json.loads(Path(promotion_record).read_text(encoding="utf-8"))
            policy = record.get("policy_checks") or {}
            if policy and policy.get("overall_pass") is False:
                report["ok"] = False
                report["reason"] = "promotion_policy_overall_pass_false"
            report["promotion_record"] = str(promotion_record)
        return report
    except Exception as exc:
        return {
            "started_at_utc": started,
            "ok": False,
            "reason": f"threshold_guardrail_exception:{exc.__class__.__name__}",
            "thresholds_path": str(thresholds_path),
        }


def evaluate_retrain_trigger(
    drift_report: Dict[str, Any] | None,
    data_checks: Dict[str, Any] | None,
    threshold_guardrails: Dict[str, Any] | None,
    args: argparse.Namespace,
    benchmark_sla: Dict[str, Any] | None = None,
    benchmark_cmd: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    def _load_archived_ops_summaries(archive_dir: Path, limit: int) -> list[Dict[str, Any]]:
        if limit <= 0 or not archive_dir.exists():
            return []
        summaries: list[Dict[str, Any]] = []
        for run_dir in sorted([p for p in archive_dir.iterdir() if p.is_dir()], reverse=True):
            summary_path = run_dir / "nightly_ops_summary.json"
            if not summary_path.exists():
                continue
            try:
                entry = {"summary": load_json(summary_path)}
                drift_path = run_dir / "drift_report.json"
                if drift_path.exists():
                    try:
                        entry["drift_report"] = load_json(drift_path)
                    except (OSError, json.JSONDecodeError):
                        entry["drift_report"] = None
                summaries.append(entry)
            except (OSError, json.JSONDecodeError):
                continue
            if len(summaries) >= limit:
                break
        return summaries

    def _extract_sla_from_summary(entry: Dict[str, Any]) -> Dict[str, Any] | None:
        summary = entry.get("summary") if isinstance(entry, dict) else None
        if not isinstance(summary, dict):
            return None
        sla = summary.get("benchmark_sla")
        if isinstance(sla, dict):
            return sla
        return None

    def _extract_sla_from_benchmark_cmd(benchmark_cmd_payload: Dict[str, Any] | None) -> Dict[str, Any] | None:
        return evaluate_benchmark_sla(benchmark_cmd_payload)

    def _is_runtime_sla_failure(sla_payload: Dict[str, Any] | None) -> bool:
        return bool(
            isinstance(sla_payload, dict)
            and not sla_payload.get("ok", False)
            and str(sla_payload.get("failure_category", "")).strip().lower() == "runtime_sla_failure"
        )

    def _is_integration_failure(sla_payload: Dict[str, Any] | None) -> bool:
        if not isinstance(sla_payload, dict) or sla_payload.get("ok", False):
            return False
        return str(sla_payload.get("operator_diagnosis", "")).strip().lower() == "pipeline_integration_bug"

    def _extract_endpoint_error_rates(sla_payload: Dict[str, Any] | None) -> Dict[str, float]:
        if not isinstance(sla_payload, dict):
            return {}
        sla_evaluation = sla_payload.get("sla_evaluation")
        if not isinstance(sla_evaluation, dict):
            return {}
        rates: Dict[str, float] = {}
        for endpoint, endpoint_eval in sla_evaluation.items():
            checks = (endpoint_eval or {}).get("checks") or {}
            error_rate = (checks.get("error_rate_pct") or {}).get("actual")
            try:
                rates[str(endpoint)] = float(error_rate)
            except (TypeError, ValueError):
                continue
        return rates

    def _compute_bool_streak(series: list[bool]) -> int:
        streak = 0
        for item in series:
            if not item:
                break
            streak += 1
        return streak

    reasons: list[str] = []
    details: Dict[str, Any] = {"thresholds": {}}
    if drift_report is not None:
        feature_alerts = int((drift_report.get("summary") or {}).get("feature_alert_count", 0))
        if feature_alerts >= int(getattr(args, "retrain_feature_alert_threshold", 2)):
            reasons.append(f"feature_alert_count={feature_alerts}")

        decision_status = str((drift_report.get("decision_drift") or {}).get("status", "ok"))
        decision_trigger = str(getattr(args, "retrain_decision_drift_status", "alert"))
        if decision_trigger == "warn" and decision_status in {"warn", "alert"}:
            reasons.append(f"decision_drift_status={decision_status}")
        elif decision_trigger == "alert" and decision_status == "alert":
            reasons.append(f"decision_drift_status={decision_status}")
        details["current_decision_drift_status"] = decision_status

    if bool(getattr(args, "retrain_on_data_check_fail", True)) and data_checks is not None and not data_checks.get("ok", False):
        reasons.append("data_checks_failed")

    if bool(getattr(args, "retrain_on_threshold_guardrail_fail", True)) and threshold_guardrails is not None and not threshold_guardrails.get("ok", False):
        reasons.append("threshold_policy_guardrails_failed")

    warn_streak_required = max(0, int(getattr(args, "retrain_warn_streak", 0)))
    sla_fail_streak_required = max(0, int(getattr(args, "retrain_sla_fail_streak", 0)))
    endpoint_error_streak_required = max(0, int(getattr(args, "retrain_endpoint_error_rate_streak", 0)))
    endpoint_error_threshold = float(getattr(args, "retrain_endpoint_error_rate_threshold", 5.0))
    details["thresholds"] = {
        "warn_streak": warn_streak_required,
        "sla_fail_streak": sla_fail_streak_required,
        "endpoint_error_rate_streak": endpoint_error_streak_required,
        "endpoint_error_rate_threshold_pct": endpoint_error_threshold,
    }

    history_window = max(warn_streak_required, sla_fail_streak_required, endpoint_error_streak_required, 1)
    archived_summaries = _load_archived_ops_summaries(Path(getattr(args, "archive_runs_dir", DEFAULT_OPS_RUN_ARCHIVE_DIR)), history_window - 1)

    historical_warns = []
    for entry in archived_summaries:
        drift = entry.get("drift_report") if isinstance(entry, dict) else None
        decision_status = str(((drift or {}).get("decision_drift") or {}).get("status", "unknown")).lower()
        historical_warns.append(decision_status == "warn")
    current_warn = str(((drift_report or {}).get("decision_drift") or {}).get("status", "unknown")).lower() == "warn"
    warn_series = [current_warn, *historical_warns]
    warn_streak = _compute_bool_streak(warn_series)
    details["warn_drift_streak"] = {
        "required": warn_streak_required,
        "current_streak": warn_streak,
        "evaluated_runs": len(warn_series),
    }
    if warn_streak_required > 0 and warn_streak >= warn_streak_required:
        reasons.append(f"warn_drift_streak={warn_streak}")

    current_sla = benchmark_sla if benchmark_sla is not None else _extract_sla_from_benchmark_cmd(benchmark_cmd)
    historical_sla = [_extract_sla_from_summary(entry) for entry in archived_summaries]
    sla_series = [_is_runtime_sla_failure(current_sla)]
    sla_series.extend(_is_runtime_sla_failure(sla) for sla in historical_sla)
    sla_fail_streak = _compute_bool_streak(sla_series)
    details["benchmark_sla_fail_streak"] = {
        "required": sla_fail_streak_required,
        "current_streak": sla_fail_streak,
        "evaluated_runs": len(sla_series),
        "category": "runtime_sla_failure_only",
    }
    if sla_fail_streak_required > 0 and sla_fail_streak >= sla_fail_streak_required:
        reasons.append(f"benchmark_sla_fail_streak={sla_fail_streak}")

    integration_series = [_is_integration_failure(current_sla)]
    integration_series.extend(_is_integration_failure(sla) for sla in historical_sla)
    integration_streak = _compute_bool_streak(integration_series)
    details["integration_failure_streak"] = {
        "current_streak": integration_streak,
        "evaluated_runs": len(integration_series),
        "category": "pipeline_integration_bug",
    }
    if isinstance(current_sla, dict):
        details["current_benchmark_failure_category"] = current_sla.get("failure_category", "none")
        details["current_benchmark_operator_diagnosis"] = current_sla.get("operator_diagnosis", "healthy")

    endpoint_streaks: Dict[str, int] = {}
    endpoint_latest_rates: Dict[str, float] = {}
    endpoint_rate_runs = [_extract_endpoint_error_rates(current_sla)] + [_extract_endpoint_error_rates(sla) for sla in historical_sla]
    all_endpoints: set[str] = set()
    for run_rates in endpoint_rate_runs:
        all_endpoints.update(run_rates.keys())

    for endpoint in sorted(all_endpoints):
        streak = 0
        latest_rate = None
        for run_rates in endpoint_rate_runs:
            endpoint_rate = run_rates.get(endpoint)
            if latest_rate is None and endpoint_rate is not None:
                latest_rate = endpoint_rate
            if endpoint_rate is None or endpoint_rate < endpoint_error_threshold:
                break
            streak += 1
        endpoint_streaks[endpoint] = streak
        if latest_rate is not None:
            endpoint_latest_rates[endpoint] = round(float(latest_rate), 6)

    triggered_endpoints = sorted(
        endpoint
        for endpoint, streak in endpoint_streaks.items()
        if endpoint_error_streak_required > 0 and streak >= endpoint_error_streak_required
    )
    details["endpoint_error_rate_streaks"] = {
        "required": endpoint_error_streak_required,
        "threshold_pct": endpoint_error_threshold,
        "streaks": endpoint_streaks,
        "latest_error_rate_pct": endpoint_latest_rates,
        "triggered_endpoints": triggered_endpoints,
        "evaluated_runs": len(endpoint_rate_runs),
    }
    if triggered_endpoints:
        reasons.append(
            "endpoint_error_rate_streak="
            + ",".join(f"{endpoint}:{endpoint_streaks[endpoint]}" for endpoint in triggered_endpoints)
        )

    return {
        "should_retrain": bool(reasons),
        "reasons": reasons if reasons else ["none"],
        "details": details,
    }


def select_run_artifacts_for_archive(
    args: argparse.Namespace,
    drift_cmd: Dict[str, Any],
    calibration_cmd: Dict[str, Any] | None,
    threshold_promotion_cmd: Dict[str, Any] | None,
    benchmark_cmd: Dict[str, Any] | None,
    latency_trend_cmd: Dict[str, Any] | None,
    latency_stage_analysis_cmd: Dict[str, Any] | None,
    latency_stage_matrix_cmd: Dict[str, Any] | None,
    cohort_kpi_cmd: Dict[str, Any] | None,
    profile_health_cmd: Dict[str, Any] | None,
) -> Dict[str, Path]:
    artifacts = resolve_model_artifact_paths(args)
    selected: Dict[str, Path] = {
        "drift_json": Path(args.drift_json),
        "drift_csv": Path(args.drift_csv),
        "ops_summary_json": Path(args.ops_summary_json),
    }

    if calibration_cmd is not None and calibration_cmd.get("ok", False):
        selected["calibration_json"] = Path(args.calibration_json)
        selected["calibration_csv"] = Path(args.calibration_csv)
        selected["thresholds_output"] = Path(artifacts["thresholds_output"])

    if threshold_promotion_cmd is not None and threshold_promotion_cmd.get("ok", False):
        selected["promotion_record_json"] = Path(args.promotion_record_json)
        selected["thresholds_output"] = Path(artifacts["thresholds_output"])
    if Path(args.artifact_validation_json).exists():
        selected["artifact_validation_json"] = Path(args.artifact_validation_json)

    if cohort_kpi_cmd is not None and cohort_kpi_cmd.get("ok", False):
        selected["cohort_kpi_json"] = Path(args.cohort_kpi_json)

    if benchmark_cmd is not None and benchmark_cmd.get("ok", False) and latency_trend_cmd is not None and latency_trend_cmd.get("ok", False):
        selected["latency_trend_json"] = Path(args.latency_trend_json)
    if latency_stage_analysis_cmd is not None and latency_stage_analysis_cmd.get("ok", False):
        selected["latency_stage_analysis_json"] = Path(args.latency_stage_analysis_json)
        selected["latency_stage_analysis_csv"] = Path(args.latency_stage_analysis_csv)
    if latency_stage_matrix_cmd is not None and latency_stage_matrix_cmd.get("ok", False):
        selected["latency_stage_matrix_json"] = Path(args.latency_stage_matrix_json)
        selected["latency_stage_matrix_csv"] = Path(args.latency_stage_matrix_csv)

    if profile_health_cmd is not None and profile_health_cmd.get("ok", False):
        selected["profile_health_json"] = Path(args.profile_health_json)

    return selected


def archive_run_artifacts(archive_dir: Path, run_id: str, artifacts: Dict[str, Path]) -> Dict[str, Any]:
    run_dir = archive_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    checksums: Dict[str, str] = {}
    missing: Dict[str, str] = {}

    for key, src in artifacts.items():
        src_path = Path(src)
        if not src_path.exists() or not src_path.is_file():
            missing[key] = str(src_path)
            continue
        dst = run_dir / src_path.name
        shutil.copy2(src_path, dst)
        copied[key] = str(dst)
        checksums[key] = hashlib.sha256(dst.read_bytes()).hexdigest()

    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "copied": copied,
        "missing": missing,
        "sha256": checksums,
    }
    (run_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def should_recalibrate(drift_report: Dict[str, Any], args: argparse.Namespace) -> bool:
    if args.skip_calibration:
        return False
    if not args.run_calibration_on_drift:
        return False
    rec = drift_report.get("recalibration_recommendation", {})
    return bool(rec.get("should_recalibrate", False))


def run_calibration(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = resolve_ops_dataset_csv(args)
    artifacts = resolve_model_artifact_paths(args)
    cmd = [
        PYTHON_EXEC,
        "project/scripts/calibrate_context.py",
        "--dataset-source",
        str(getattr(args, "dataset_source", "creditcard")),
        "--dataset-path",
        str(dataset_path),
        "--trials",
        str(args.calibration_trials),
        "--seed",
        str(args.calibration_seed),
        "--target-fpr",
        str(args.target_fpr),
        "--target-precision",
        str(args.target_precision),
        "--output-json",
        str(args.calibration_json),
        "--output-csv",
        str(args.calibration_csv),
        "--model-path",
        str(artifacts["model_path"]),
        "--feature-path",
        str(artifacts["feature_path"]),
        "--preprocessing-artifact-path",
        str(artifacts["preprocessing_artifact_path"]),
        "--thresholds-output",
        str(artifacts["thresholds_output"]),
    ]
    if str(getattr(args, "dataset_source", "creditcard")) == "ieee_cis":
        cmd.extend([
            "--ieee-transaction-path",
            str(args.ieee_transaction_path),
            "--ieee-identity-path",
            str(args.ieee_identity_path),
        ])
    return run_command(cmd)


def run_threshold_promotion(args: argparse.Namespace) -> Dict[str, Any]:
    artifacts = resolve_model_artifact_paths(args)
    cmd = [
        PYTHON_EXEC,
        "project/scripts/promote_thresholds.py",
        "--calibration-json",
        str(args.calibration_json),
        "--active-thresholds",
        str(artifacts["thresholds_output"]),
        "--promotion-record-json",
        str(args.promotion_record_json),
        "--model-path",
        str(artifacts["model_path"]),
        "--feature-path",
        str(artifacts["feature_path"]),
        "--preprocessing-artifact-path",
        str(artifacts["preprocessing_artifact_path"]),
    ]
    return run_command(cmd)


def run_artifact_validation(args: argparse.Namespace) -> Dict[str, Any]:
    artifacts = resolve_model_artifact_paths(args)
    cmd = [
        PYTHON_EXEC,
        "project/scripts/validate_artifact_compatibility.py",
        "--model-path",
        str(artifacts["model_path"]),
        "--feature-path",
        str(artifacts["feature_path"]),
        "--preprocessing-artifact-path",
        str(artifacts["preprocessing_artifact_path"]),
        "--calibration-json",
        str(args.calibration_json),
        "--promotion-record-json",
        str(args.promotion_record_json),
        "--output-json",
        str(args.artifact_validation_json),
    ]
    return run_command(cmd)


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    timeout = float(getattr(args, "benchmark_timeout", 8.0))
    warmup = int(getattr(args, "benchmark_warmup", 3))
    cmd = [
        PYTHON_EXEC,
        "project/scripts/benchmark_latency.py",
        "--requests",
        str(args.benchmark_requests),
        "--concurrency",
        str(args.benchmark_concurrency),
        "--timeout",
        str(timeout),
        "--warmup",
        str(warmup),
        "--sla-p95-ms",
        str(float(getattr(args, "benchmark_sla_p95_ms", 250.0))),
        "--sla-p99-ms",
        str(float(getattr(args, "benchmark_sla_p99_ms", 500.0))),
        "--sla-error-rate-pct",
        str(float(getattr(args, "benchmark_sla_error_rate_pct", 1.0))),
    ]
    result = run_command(cmd)
    stdout = result.get("stdout", "")
    if not isinstance(stdout, str) or not stdout.strip():
        return result
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return result

    if str(payload.get("benchmark_mode", "")).lower() == "contract_validation_failed":
        result["contract_validation_failed"] = True
        result["contract_validation"] = payload.get("contract_validation")
        result["remediation"] = (
            "Benchmark payload contract validation failed. Align benchmark payload builders with "
            "project/app/contracts.py and project/app/schema_spec.py, then rerun nightly ops."
        )
    return result


def run_latency_trend(args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [
        PYTHON_EXEC,
        "project/scripts/benchmark_trend_report.py",
        "--history-limit",
        str(args.benchmark_trend_history_limit),
        "--output-json",
        str(args.latency_trend_json),
        "--output-csv",
        str(args.latency_trend_csv),
    ]
    result = run_command(cmd)
    if not result.get("ok", False):
        return result

    try:
        payload = json.loads(Path(args.latency_trend_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return result

    if payload.get("status") == "no_data":
        result["warning"] = True
        result["warning_reason"] = str(payload.get("reason", "no benchmark JSON files found"))
        result["no_data"] = True
        result["trend_report"] = {
            "status": "no_data",
            "reason": payload.get("reason"),
            "generated_at_utc": payload.get("generated_at_utc"),
            "benchmark_dir": payload.get("benchmark_dir"),
        }
    return result


def run_latency_stage_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [
        PYTHON_EXEC,
        "project/scripts/latency_stage_analysis.py",
        "--audit-log",
        str(args.audit_log),
        "--output-json",
        str(args.latency_stage_analysis_json),
        "--output-csv",
        str(args.latency_stage_analysis_csv),
    ]
    return run_command(cmd)


def run_latency_stage_matrix(args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [
        PYTHON_EXEC,
        "project/scripts/benchmark_stage_matrix.py",
        "--fraud-url",
        "http://127.0.0.1:8000/score_transaction",
        "--wallet-url",
        "http://127.0.0.1:8001/wallet/authorize_payment",
        "--timeout",
        "5.0",
        "--history-runs",
        str(getattr(args, "latency_stage_matrix_history_runs", 7)),
        "--output-json",
        str(args.latency_stage_matrix_json),
        "--output-csv",
        str(args.latency_stage_matrix_csv),
    ]
    return run_command(cmd)


def run_cohort_kpi(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = resolve_ops_dataset_csv(args)
    artifacts = resolve_model_artifact_paths(args)
    cmd = [
        PYTHON_EXEC,
        "project/scripts/cohort_kpi_report.py",
        "--dataset-path",
        str(dataset_path),
        "--model-path",
        str(artifacts["model_path"]),
        "--feature-path",
        str(artifacts["feature_path"]),
        "--thresholds-path",
        str(artifacts["thresholds_output"]),
        "--preprocessing-artifact-path",
        str(artifacts["preprocessing_artifact_path"]),
        "--output-json",
        str(args.cohort_kpi_json),
        "--output-csv",
        str(args.cohort_kpi_csv),
    ]
    return run_command(cmd)


def run_profile_replay(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = resolve_ops_dataset_csv(args)
    cmd = [
        PYTHON_EXEC,
        "project/scripts/replay_behavior_profiles.py",
        "--dataset-source",
        str(getattr(args, "dataset_source", "creditcard")),
        "--dataset-path",
        str(dataset_path),
        "--sqlite-path",
        str(args.profile_sqlite_path),
        "--output-json",
        str(args.profile_replay_json),
        "--user-count",
        str(args.profile_replay_user_count),
        "--transactions-per-user",
        str(args.profile_replay_transactions_per_user),
    ]
    if str(getattr(args, "dataset_source", "creditcard")) == "ieee_cis":
        cmd.extend([
            "--ieee-transaction-path",
            str(args.ieee_transaction_path),
            "--ieee-identity-path",
            str(args.ieee_identity_path),
        ])
    return run_command(cmd)


def run_profile_health(args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [
        PYTHON_EXEC,
        "project/scripts/behavior_profile_health.py",
        "--store-backend",
        str(args.profile_store_backend),
        "--sqlite-path",
        str(args.profile_sqlite_path),
        "--output-json",
        str(args.profile_health_json),
        "--min-history",
        str(args.profile_min_history),
        "--stale-seconds",
        str(args.profile_stale_seconds),
    ]
    return run_command(cmd)


def send_alert(webhook_url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    if not webhook_url:
        return {"sent": False, "reason": "webhook_not_configured"}

    try:
        response = requests.post(webhook_url, json=payload, timeout=timeout_s)
        return {
            "sent": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "response_text": response.text[:240],
        }
    except requests.RequestException as exc:
        return {
            "sent": False,
            "reason": f"request_exception:{exc.__class__.__name__}",
        }


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def compute_decision_source_kpi(audit_log_path: Path, analyst_outcomes_path: Path) -> Dict[str, Any]:
    source_order = ["score_band", "hard_rule_override", "low_history_policy", "step_up_policy"]
    counters = {
        source: {
            "audit_volume": 0,
            "flag_count": 0,
            "analyst_reviews": 0,
            "confirmed_fraud_count": 0,
            "false_positive_count": 0,
        }
        for source in source_order
    }
    audits = load_jsonl_records(audit_log_path)
    outcomes = load_jsonl_records(analyst_outcomes_path)
    audits_by_request_id = {str(row.get("request_id")): row for row in audits if row.get("request_id")}

    for row in audits:
        source = str(row.get("decision_source", "score_band"))
        if source not in counters:
            continue
        counters[source]["audit_volume"] += 1
        if str(row.get("decision", "")).upper() == "FLAG":
            counters[source]["flag_count"] += 1

    for row in outcomes:
        request_id = str(row.get("request_id", ""))
        source = str((audits_by_request_id.get(request_id) or {}).get("decision_source", "score_band"))
        if source not in counters:
            continue
        counters[source]["analyst_reviews"] += 1
        analyst_decision = str(row.get("analyst_decision", "")).upper()
        model_decision = str(row.get("model_decision", "")).upper()
        if analyst_decision == "FRAUD":
            counters[source]["confirmed_fraud_count"] += 1
        if analyst_decision == "LEGIT" and model_decision in {"FLAG", "BLOCK"}:
            counters[source]["false_positive_count"] += 1

    rows: list[Dict[str, Any]] = []
    for source in source_order:
        entry = counters[source]
        volume = int(entry["audit_volume"])
        reviews = int(entry["analyst_reviews"])
        rows.append(
            {
                "decision_source": source,
                "audit_volume": volume,
                "flag_rate": round(float(entry["flag_count"]) / volume, 6) if volume else 0.0,
                "analyst_reviews": reviews,
                "confirmed_fraud_conversion": round(float(entry["confirmed_fraud_count"]) / reviews, 6) if reviews else 0.0,
                "false_positive_rate": round(float(entry["false_positive_count"]) / reviews, 6) if reviews else 0.0,
            }
        )

    return {
        "audit_log": str(audit_log_path),
        "analyst_outcomes_log": str(analyst_outcomes_path),
        "rows": rows,
    }


def build_ops_summary(
    dataset_artifact_validation: Dict[str, Any] | None,
    drift_cmd: Dict[str, Any],
    drift_report: Dict[str, Any] | None,
    calibration_cmd: Dict[str, Any] | None,
    threshold_promotion_cmd: Dict[str, Any] | None,
    artifact_validation_cmd: Dict[str, Any] | None,
    artifact_validation_report: Dict[str, Any] | None,
    benchmark_cmd: Dict[str, Any] | None,
    latency_trend_cmd: Dict[str, Any] | None,
    latency_stage_analysis_cmd: Dict[str, Any] | None,
    latency_stage_matrix_cmd: Dict[str, Any] | None,
    cohort_kpi_cmd: Dict[str, Any] | None,
    decision_source_kpi: Dict[str, Any] | None = None,
    profile_replay_cmd: Dict[str, Any] | None = None,
    profile_health_cmd: Dict[str, Any] | None = None,
    profile_health_report: Dict[str, Any] | None = None,
    benchmark_sla: Dict[str, Any] | None = None,
    benchmark_error_summary: Dict[str, Any] | None = None,
    data_checks: Dict[str, Any] | None = None,
    threshold_policy_guardrails: Dict[str, Any] | None = None,
    retrain_trigger: Dict[str, Any] | None = None,
    archive_result: Dict[str, Any] | None = None,
    alert_result: Dict[str, Any] | None = None,
    benchmark_sla_mode: str = "enforce",
) -> Dict[str, Any]:
    """Build the nightly summary using the documented OPS_SUMMARY_SCHEMA_KEYS order."""
    normalized_sla_mode = normalize_benchmark_sla_mode(benchmark_sla_mode)
    status = "ok"
    if dataset_artifact_validation is not None and not dataset_artifact_validation.get("ok", False):
        status = "failed_dataset_artifact_validation"
    elif not drift_cmd.get("ok", False):
        status = "failed_drift_monitor"
    elif calibration_cmd is not None and not calibration_cmd.get("ok", False):
        status = "failed_calibration"
    elif threshold_promotion_cmd is not None and not threshold_promotion_cmd.get("ok", False):
        status = "failed_threshold_promotion"
    elif artifact_validation_cmd is not None and not artifact_validation_cmd.get("ok", False):
        status = "failed_artifact_validation"
    elif benchmark_cmd is not None and not benchmark_cmd.get("ok", False):
        if bool(benchmark_cmd.get("contract_validation_failed", False)):
            status = "failed_benchmark_contract_validation"
        else:
            status = "failed_benchmark"
    elif normalized_sla_mode == "enforce" and benchmark_sla is not None and not benchmark_sla.get("ok", False):
        status = "failed_benchmark_sla"
    elif latency_trend_cmd is not None and not latency_trend_cmd.get("ok", False):
        status = "failed_latency_trend"
    elif latency_stage_analysis_cmd is not None and not latency_stage_analysis_cmd.get("ok", False):
        status = "failed_latency_stage_analysis"
    elif latency_stage_matrix_cmd is not None and not latency_stage_matrix_cmd.get("ok", False):
        status = "failed_latency_stage_matrix"
    elif cohort_kpi_cmd is not None and not cohort_kpi_cmd.get("ok", False):
        status = "failed_cohort_kpi"
    elif profile_replay_cmd is not None and not profile_replay_cmd.get("ok", False):
        status = "failed_profile_replay"
    elif profile_health_cmd is not None and not profile_health_cmd.get("ok", False):
        status = "failed_profile_health"
    elif data_checks is not None and not data_checks.get("ok", False):
        status = "failed_data_checks"
    elif threshold_policy_guardrails is not None and not threshold_policy_guardrails.get("ok", False):
        status = "failed_threshold_policy_guardrails"

    recalibration = drift_report.get("recalibration_recommendation", {}) if drift_report else {}
    summary_values_by_key: Dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "status": status,
        "dataset_artifact_validation": dataset_artifact_validation,
        "drift_monitor": drift_cmd,
        "drift_report_summary": drift_report.get("summary") if drift_report else None,
        "drift_recalibration_recommendation": recalibration,
        "calibration": calibration_cmd,
        "threshold_promotion": threshold_promotion_cmd,
        "artifact_validation": artifact_validation_cmd,
        "artifact_validation_report": artifact_validation_report,
        "benchmark": benchmark_cmd,
        "benchmark_sla": benchmark_sla,
        "benchmark_error_summary": benchmark_error_summary,
        "benchmark_sla_mode": normalized_sla_mode,
        "latency_trend": latency_trend_cmd,
        "latency_stage_analysis": latency_stage_analysis_cmd,
        "latency_stage_matrix": latency_stage_matrix_cmd,
        "cohort_kpi": cohort_kpi_cmd,
        "decision_source_kpi": decision_source_kpi,
        "profile_replay": profile_replay_cmd,
        "profile_health": profile_health_cmd,
        "profile_health_summary": profile_health_report.get("summary") if profile_health_report else None,
        "profile_health_status": profile_health_report.get("status") if profile_health_report else None,
        "data_checks": data_checks,
        "threshold_policy_guardrails": threshold_policy_guardrails,
        "retrain_trigger": retrain_trigger or {},
        "archive": archive_result,
        "alert": alert_result or {},
    }
    return {key: summary_values_by_key.get(key) for key in OPS_SUMMARY_SCHEMA_KEYS}


def main() -> int:
    args = parse_args()
    benchmark_sla_mode = normalize_benchmark_sla_mode(getattr(args, "benchmark_sla_mode", "enforce"))
    args.ops_summary_json.parent.mkdir(parents=True, exist_ok=True)

    dataset_artifact_validation = validate_dataset_artifact_compatibility(args)
    can_run_followups = bool(dataset_artifact_validation.get("ok", False))
    drift_cmd: Dict[str, Any] = {
        "started_at_utc": _utc_now_iso(),
        "ok": False,
        "skipped": True,
        "reason": "dataset_artifact_validation_failed",
    }
    if can_run_followups:
        drift_cmd = run_drift_monitor(args)
    drift_report: Dict[str, Any] | None = None
    calibration_cmd: Dict[str, Any] | None = None
    threshold_promotion_cmd: Dict[str, Any] | None = None
    artifact_validation_cmd: Dict[str, Any] | None = None
    artifact_validation_report: Dict[str, Any] | None = None
    benchmark_cmd: Dict[str, Any] | None = None
    latency_trend_cmd: Dict[str, Any] | None = None
    latency_stage_analysis_cmd: Dict[str, Any] | None = None
    latency_stage_matrix_cmd: Dict[str, Any] | None = None
    cohort_kpi_cmd: Dict[str, Any] | None = None
    decision_source_kpi: Dict[str, Any] | None = None
    profile_replay_cmd: Dict[str, Any] | None = None
    profile_health_cmd: Dict[str, Any] | None = None
    profile_health_report: Dict[str, Any] | None = None
    benchmark_sla: Dict[str, Any] | None = None
    benchmark_error_summary: Dict[str, Any] | None = None
    data_checks: Dict[str, Any] | None = None
    threshold_policy_guardrails: Dict[str, Any] | None = None
    archive_result: Dict[str, Any] | None = None

    if drift_cmd["ok"] and args.drift_json.exists():
        drift_report = load_json(args.drift_json)

    if can_run_followups and not bool(getattr(args, "skip_data_checks", False)):
        data_checks = run_data_checks(args)

    if can_run_followups and not bool(getattr(args, "skip_threshold_policy_guardrails", False)):
        threshold_policy_guardrails = run_threshold_policy_guardrails(args)

    if drift_cmd["ok"] and drift_report is not None and should_recalibrate(drift_report, args):
        calibration_cmd = run_calibration(args)
        if (
            calibration_cmd.get("ok", False)
            and args.promote_thresholds_on_pass
            and not args.skip_threshold_promotion
        ):
            threshold_promotion_cmd = run_threshold_promotion(args)

    if can_run_followups and not bool(getattr(args, "skip_artifact_validation", False)):
        artifact_validation_cmd = run_artifact_validation(args)
        if artifact_validation_cmd.get("ok", False) and args.artifact_validation_json.exists():
            artifact_validation_report = load_json(args.artifact_validation_json)

    if can_run_followups and args.run_benchmark:
        benchmark_cmd = run_benchmark(args)
        if benchmark_sla_mode != "off":
            benchmark_sla = evaluate_benchmark_sla(benchmark_cmd)
        contract_validation_failed = bool(benchmark_cmd.get("contract_validation_failed", False)) if benchmark_cmd is not None else False
        if benchmark_cmd is not None and not contract_validation_failed:
            benchmark_error_summary = summarize_benchmark_errors(benchmark_cmd)
            latency_trend_cmd = run_latency_trend(args)
            latency_stage_analysis_cmd = run_latency_stage_analysis(args)
        elif benchmark_cmd is not None and contract_validation_failed:
            benchmark_error_summary = {
                "status": "contract_validation_failed",
                "remediation": benchmark_cmd.get("remediation"),
            }
            latency_trend_cmd = {
                "started_at_utc": _utc_now_iso(),
                "ok": False,
                "skipped": True,
                "reason": "benchmark_contract_validation_failed",
            }
            latency_stage_analysis_cmd = {
                "started_at_utc": _utc_now_iso(),
                "ok": False,
                "skipped": True,
                "reason": "benchmark_contract_validation_failed",
            }

    if can_run_followups and bool(getattr(args, "run_benchmark_matrix", False)):
        latency_stage_matrix_cmd = run_latency_stage_matrix(args)

    if can_run_followups and args.run_cohort_kpi:
        cohort_kpi_cmd = run_cohort_kpi(args)
    analyst_outcomes_log = Path(
        getattr(
            args,
            "analyst_outcomes_log",
            REPO_ROOT / "project" / "outputs" / "audit" / "analyst_outcomes.jsonl",
        )
    )
    decision_source_kpi = compute_decision_source_kpi(args.audit_log, analyst_outcomes_log)

    if can_run_followups and args.run_profile_replay:
        profile_replay_cmd = run_profile_replay(args)

    if can_run_followups and args.run_profile_health:
        profile_health_cmd = run_profile_health(args)
        if profile_health_cmd["ok"] and args.profile_health_json.exists():
            profile_health_report = load_json(args.profile_health_json)

    alert_payload = {
        "event": "nightly_ops",
        "drift_status": drift_report.get("decision_drift", {}).get("status") if drift_report else "unknown",
        "recalibration_triggered": bool(calibration_cmd is not None),
        "recalibration_needed": bool(
            drift_report.get("recalibration_recommendation", {}).get("should_recalibrate", False)
        ) if drift_report else False,
        "profile_health_status": (profile_health_report or {}).get("status", "unknown"),
        "retrain_trigger": evaluate_retrain_trigger(
            drift_report,
            data_checks,
            threshold_policy_guardrails,
            args,
            benchmark_sla=benchmark_sla,
            benchmark_cmd=benchmark_cmd,
        ),
        "generated_at_utc": _utc_now_iso(),
    }

    retrain_trigger = alert_payload["retrain_trigger"]

    should_alert = False
    if drift_report is not None:
        decision_status = drift_report.get("decision_drift", {}).get("status", "ok")
        priority = drift_report.get("recalibration_recommendation", {}).get("priority", "low")
        should_alert = decision_status in {"warn", "alert"} or priority in {"medium", "high"}
    if profile_health_report is not None and profile_health_report.get("status") == "warn":
        should_alert = True

    alert_result = {"sent": False, "reason": "not_required"}
    if should_alert or (drift_cmd["ok"] is False):
        alert_result = send_alert(args.alert_webhook_url, alert_payload, args.alert_timeout)

    ops_summary = build_ops_summary(
        dataset_artifact_validation,
        drift_cmd,
        drift_report,
        calibration_cmd,
        threshold_promotion_cmd,
        artifact_validation_cmd,
        artifact_validation_report,
        benchmark_cmd,
        latency_trend_cmd,
        latency_stage_analysis_cmd,
        latency_stage_matrix_cmd,
        cohort_kpi_cmd,
        decision_source_kpi,
        profile_replay_cmd,
        profile_health_cmd,
        profile_health_report,
        benchmark_sla,
        benchmark_error_summary,
        data_checks,
        threshold_policy_guardrails,
        retrain_trigger,
        archive_result,
        alert_result,
        benchmark_sla_mode=benchmark_sla_mode,
    )

    if not bool(getattr(args, "skip_archive", False)):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_result = archive_run_artifacts(
            archive_dir=Path(getattr(args, "archive_runs_dir", DEFAULT_OPS_RUN_ARCHIVE_DIR)),
            run_id=run_id,
            artifacts=select_run_artifacts_for_archive(
                args,
                drift_cmd=drift_cmd,
                calibration_cmd=calibration_cmd,
                threshold_promotion_cmd=threshold_promotion_cmd,
                benchmark_cmd=benchmark_cmd,
                latency_trend_cmd=latency_trend_cmd,
                latency_stage_analysis_cmd=latency_stage_analysis_cmd,
                latency_stage_matrix_cmd=latency_stage_matrix_cmd,
                cohort_kpi_cmd=cohort_kpi_cmd,
                profile_health_cmd=profile_health_cmd,
            ),
        )
        ops_summary["archive"] = archive_result

    with args.ops_summary_json.open("w", encoding="utf-8") as f:
        json.dump(ops_summary, f, indent=2)

    print(json.dumps({"status": ops_summary["status"], "summary_file": str(args.ops_summary_json)}, indent=2))
    return 0 if ops_summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

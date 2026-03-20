#!/usr/bin/env python3
"""Lightweight latency benchmark for fraud and wallet APIs.

Runs controlled-concurrency POST traffic against:
- /score_transaction
- /wallet/authorize_payment

Collects p50/p95/p99 latencies, throughput, and error rate.
Exports results under project/outputs/benchmark/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import requests
from project.app.contracts import (
    FraudTransactionContract,
    PAYLOAD_SCHEMA_VERSION,
    WalletPaymentContract,
)
from project.app.schema_spec import ALL_SERVING_INPUT_FIELDS, IEEE_V_FEATURE_FIELDS


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "benchmark"


@dataclass
class SampleResult:
    latency_ms: float
    ok: bool
    status_code: int | None
    error: str | None
    correlation_id: str
    failure_category: str | None = None
    error_category: str | None = None
    exception_signature: str | None = None
    endpoint_name: str | None = None
    stage_timings_ms: Dict[str, float] | None = None
    upstream_attempts: int | None = None
    fallback_used: bool | None = None


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]

    low_value = sorted_values[low]
    high_value = sorted_values[high]
    fraction = rank - low
    return low_value + (high_value - low_value) * fraction


def _supported_v_feature_names() -> List[str]:
    contract_v_fields = sorted(
        (
            field_name
            for field_name in FraudTransactionContract.model_fields
            if field_name.startswith("V") and field_name[1:].isdigit()
        ),
        key=lambda field_name: int(field_name[1:]),
    )
    schema_v_fields = sorted(IEEE_V_FEATURE_FIELDS, key=lambda field_name: int(field_name[1:]))
    if contract_v_fields != schema_v_fields:
        raise RuntimeError(
            "Benchmark payload contract mismatch: FraudTransactionContract V-features diverge from schema_spec IEEE_V_FEATURE_FIELDS."
        )
    return contract_v_fields


def _base_payload_defaults(amount: float) -> Dict[str, Any]:
    payload_defaults: Dict[str, Any] = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "user_id": "benchmark_user_001",
        "TransactionDT": 34567.0,
        "TransactionAmt": amount,
        "device_risk_score": 0.1,
        "ip_risk_score": 0.05,
        "location_risk_score": 0.05,
        "device_id": "bench-device-1",
        "device_shared_users_24h": 1,
        "account_age_days": 180,
        "sim_change_recent": False,
        "tx_type": "MERCHANT",
        "channel": "APP",
        "cash_flow_velocity_1h": 1,
        "p2p_counterparties_24h": 0,
        "is_cross_border": False,
    }
    payload_defaults.update(_sample_realistic_v_feature_row(amount=amount, user_id=str(payload_defaults["user_id"])))
    return payload_defaults


def _realistic_v_feature_profiles() -> List[Dict[str, float]]:
    """IEEE-style V-feature profiles captured from non-zero fixture-like rows."""
    profile_values: List[List[float]] = [
        [
            -0.04408235138891037,
            7.369614545953526,
            -7.83943086711723,
            6.965419518529064,
            -5.420185217520521,
            -5.714678500677076,
            -5.19026410759605,
            5.052695914119111,
            -5.850861364386235,
            7.593304393413657,
            7.317782322114278,
            -6.071981004101701,
            5.293266247129321,
            7.187205673204595,
            1.9776032224067457,
            -4.539117812432908,
            -1.3149943466980165,
        ],
        [
            -6.1045906684903635,
            -7.220626670744885,
            -4.957298476778099,
            3.528627096445259,
            1.8606477692251477,
            -5.058535427749566,
            -4.595234933097075,
            -5.673993712115449,
            -2.4249455782519327,
            -5.023296046509476,
            6.573711318437564,
            -2.1363538681471805,
            -7.078260653924705,
            -0.06399854377598935,
            -5.939779021605007,
            0.3723280986105646,
            -2.99315065646541,
        ],
        [
            -3.074336509939637,
            0.07449393760432061,
            -6.398364671947665,
            5.247023307487161,
            -1.7797687241476867,
            -5.386606964213321,
            -4.892749520346562,
            -0.31064889899816915,
            -4.137903471319084,
            1.2850041734520904,
            6.945746820275921,
            -4.104167436124441,
            -0.8924972033976921,
            3.5616035647143027,
            -1.9810878995991306,
            -2.0833948569111715,
            -2.154072501581713,
        ],
    ]
    fields = _supported_v_feature_names()
    return [{field_name: float(values[idx]) for idx, field_name in enumerate(fields)} for values in profile_values]


def _sample_realistic_v_feature_row(*, amount: float, user_id: str) -> Dict[str, float]:
    """
    Build deterministic benchmark V-features from realistic non-zero profiles.

    We sample a baseline profile and derive a mild amount-based scaling so benchmark
    traffic better reflects live-like variation than an all-zero vector.
    """
    profiles = _realistic_v_feature_profiles()
    seed_input = f"{user_id}:{round(float(amount), 2)}".encode("utf-8")
    digest = hashlib.sha256(seed_input).hexdigest()
    profile_idx = int(digest[:8], 16) % len(profiles)
    selected = profiles[profile_idx]
    amount_scale = 1.0 + max(-0.08, min(0.08, (float(amount) - 250.0) / 5000.0))
    return {
        field_name: round(float(value) * amount_scale, 6)
        for field_name, value in selected.items()
    }


def _wallet_payload_defaults() -> Dict[str, Any]:
    return {
        "wallet_id": "wallet_bench_001",
        "merchant_name": "Benchmark Merchant",
        "currency": "USD",
    }


def build_base_tx(amount: float = 72.5) -> Dict[str, Any]:
    contract_fields = list(FraudTransactionContract.model_fields.keys())
    if contract_fields != ALL_SERVING_INPUT_FIELDS:
        raise RuntimeError(
            "Benchmark payload contract mismatch: FraudTransactionContract fields diverge from schema_spec ALL_SERVING_INPUT_FIELDS."
        )
    payload_defaults = _base_payload_defaults(amount)
    missing_fields = [field_name for field_name in contract_fields if field_name not in payload_defaults]
    if missing_fields:
        raise RuntimeError(f"Benchmark payload defaults missing FraudTransactionContract fields: {missing_fields}")
    return {field_name: payload_defaults[field_name] for field_name in contract_fields}


def build_wallet_payload() -> Dict[str, Any]:
    payload = build_base_tx()
    wallet_only_fields = [
        field_name
        for field_name in WalletPaymentContract.model_fields
        if field_name not in FraudTransactionContract.model_fields
    ]
    wallet_defaults = _wallet_payload_defaults()
    missing_wallet_fields = [field_name for field_name in wallet_only_fields if field_name not in wallet_defaults]
    if missing_wallet_fields:
        raise RuntimeError(f"Benchmark payload defaults missing WalletPaymentContract fields: {missing_wallet_fields}")
    for field_name in wallet_only_fields:
        payload[field_name] = wallet_defaults[field_name]
    return payload


def _coerce_stage_timings(payload: Any) -> Dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, float] = {}
    details = payload.get("details")
    if isinstance(details, dict):
        for key, value in details.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    for key, value in payload.items():
        if key == "details":
            continue
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def run_request(url: str, payload: Dict[str, Any], timeout_s: float, correlation_id: str) -> SampleResult:
    start = time.perf_counter()
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout_s,
            headers={"x-correlation-id": correlation_id},
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        is_ok = 200 <= response.status_code < 300
        failure_category: str | None = None
        error_category: str | None = None
        signature: str | None = None
        error_text: str | None = None
        endpoint_name: str | None = None
        stage_timings_ms: Dict[str, float] = {}
        upstream_attempts: int | None = None
        fallback_used: bool | None = None
        response_json: Dict[str, Any] = {}
        try:
            parsed_payload = response.json()
            if isinstance(parsed_payload, dict):
                response_json = parsed_payload
        except ValueError:
            response_json = {}

        endpoint_name = response_json.get("endpoint_name") if isinstance(response_json.get("endpoint_name"), str) else None
        stage_timings_ms = _coerce_stage_timings(response_json.get("stage_timings_ms"))
        if isinstance(response_json.get("upstream_attempts"), int):
            upstream_attempts = int(response_json["upstream_attempts"])
        if isinstance(response_json.get("fallback_used"), bool):
            fallback_used = bool(response_json["fallback_used"])

        if not is_ok:
            if 400 <= response.status_code < 500:
                failure_category = "http_4xx"
            elif 500 <= response.status_code < 600:
                failure_category = "http_5xx"
            else:
                failure_category = "http_other"
            error_text = response.text[:240]
            if isinstance(response_json, dict):
                error_category = response_json.get("error_category")
                signature = response_json.get("exception_signature")
            if not error_category:
                error_category = "unknown_internal"
        return SampleResult(
            latency_ms=latency_ms,
            ok=is_ok,
            status_code=response.status_code,
            error=error_text,
            correlation_id=correlation_id,
            failure_category=failure_category,
            error_category=error_category,
            exception_signature=signature,
            endpoint_name=endpoint_name,
            stage_timings_ms=stage_timings_ms,
            upstream_attempts=upstream_attempts,
            fallback_used=fallback_used,
        )
    except requests.exceptions.Timeout as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return SampleResult(
            latency_ms=latency_ms,
            ok=False,
            status_code=None,
            error=str(exc),
            correlation_id=correlation_id,
            failure_category="timeout",
            error_category="timeout_upstream",
            exception_signature=f"{exc.__class__.__name__}:{str(exc)[:200]}",
            stage_timings_ms={},
        )
    except requests.exceptions.ConnectionError as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return SampleResult(
            latency_ms=latency_ms,
            ok=False,
            status_code=None,
            error=str(exc),
            correlation_id=correlation_id,
            failure_category="connection_refused",
            error_category="artifact_missing",
            exception_signature=f"{exc.__class__.__name__}:{str(exc)[:200]}",
            stage_timings_ms={},
        )
    except requests.RequestException as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return SampleResult(
            latency_ms=latency_ms,
            ok=False,
            status_code=None,
            error=str(exc),
            correlation_id=correlation_id,
            failure_category="request_exception",
            error_category="model_runtime_error",
            exception_signature=f"{exc.__class__.__name__}:{str(exc)[:200]}",
            stage_timings_ms={},
        )


def summarize_latency_distribution(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "p50": math.nan, "p95": math.nan, "p99": math.nan, "mean": math.nan}
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "mean": statistics.fmean(sorted_values),
        "p50": percentile(sorted_values, 0.50),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
    }


def summarize_stage_timings(samples: List[SampleResult]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = {}
    for sample in samples:
        stage_dict = sample.stage_timings_ms or {}
        for stage_name, stage_ms in stage_dict.items():
            buckets.setdefault(stage_name, []).append(float(stage_ms))
    return {
        stage: summarize_latency_distribution(stage_values)
        for stage, stage_values in sorted(buckets.items())
    }


def summarize_stage_contribution(samples: List[SampleResult]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    stage_totals: Dict[str, float] = {}
    stage_counts: Dict[str, int] = {}
    share_values: Dict[str, List[float]] = {}

    for sample in samples:
        stage_dict = sample.stage_timings_ms or {}
        total = stage_dict.get("total_pipeline_ms")
        if total is None or total <= 0:
            continue
        for stage_name, value in stage_dict.items():
            if stage_name == "total_pipeline_ms":
                continue
            stage_totals[stage_name] = stage_totals.get(stage_name, 0.0) + float(value)
            stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1
            share_values.setdefault(stage_name, []).append(max(0.0, float(value)) / float(total))

    for stage_name, total in stage_totals.items():
        count = stage_counts.get(stage_name, 0)
        shares = share_values.get(stage_name, [])
        rows.append(
            {
                "stage": stage_name,
                "mean_ms": (total / count) if count > 0 else math.nan,
                "mean_share_of_total_pct": (statistics.fmean(shares) * 100.0) if shares else math.nan,
                "observations": count,
            }
        )

    return sorted(rows, key=lambda row: (row.get("mean_share_of_total_pct") or 0.0), reverse=True)


def run_preflight_check(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    sample = run_request(url, payload, timeout_s, correlation_id=f"preflight-{uuid.uuid4()}")
    return {
        "ok": sample.ok,
        "status_code": sample.status_code,
        "failure_category": sample.failure_category,
        "error_category": sample.error_category,
        "correlation_id": sample.correlation_id,
        "error": sample.error,
        "latency_ms": sample.latency_ms,
    }


def validate_benchmark_payload_contract() -> Dict[str, Any]:
    errors: List[Dict[str, str]] = []
    try:
        FraudTransactionContract.model_validate(build_base_tx())
    except Exception as exc:
        errors.append({"endpoint": "score_transaction", "error": str(exc)})

    try:
        WalletPaymentContract.model_validate(build_wallet_payload())
    except Exception as exc:
        errors.append({"endpoint": "wallet_authorize_payment", "error": str(exc)})

    return {"ok": len(errors) == 0, "errors": errors}


def benchmark_endpoint(
    *,
    name: str,
    url: str,
    payload: Dict[str, Any],
    requests_count: int,
    concurrency: int,
    timeout_s: float,
    warmup_requests: int,
) -> Dict[str, Any]:
    if requests_count <= 0:
        raise ValueError("requests_count must be > 0")
    if concurrency <= 0:
        raise ValueError("concurrency must be > 0")

    for _ in range(max(0, warmup_requests)):
        run_request(url, payload, timeout_s, correlation_id=f"warmup-{name}-{uuid.uuid4()}")

    started_at = iso_now()
    t0 = time.perf_counter()
    samples: List[SampleResult] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_request, url, payload, timeout_s, f"{name}-{uuid.uuid4()}")
            for _ in range(requests_count)
        ]
        for future in as_completed(futures):
            samples.append(future.result())

    elapsed_s = time.perf_counter() - t0

    latencies = sorted(sample.latency_ms for sample in samples)
    success_count = sum(1 for s in samples if s.ok)
    error_count = len(samples) - success_count

    status_histogram: Dict[str, int] = {}
    for sample in samples:
        key = str(sample.status_code) if sample.status_code is not None else "request_exception"
        status_histogram[key] = status_histogram.get(key, 0) + 1

    errors = [s.error for s in samples if s.error]
    failure_categories: Dict[str, int] = {}
    error_categories: Dict[str, int] = {}
    exception_signatures: Dict[str, int] = {}
    for sample in samples:
        if sample.ok:
            continue
        category = sample.failure_category or "unknown"
        failure_categories[category] = failure_categories.get(category, 0) + 1
        error_category = sample.error_category or "unknown_internal"
        error_categories[error_category] = error_categories.get(error_category, 0) + 1
        if sample.exception_signature:
            signature = str(sample.exception_signature)
            exception_signatures[signature] = exception_signatures.get(signature, 0) + 1

    top_exception_signatures = [
        {"signature": signature, "count": count}
        for signature, count in sorted(exception_signatures.items(), key=lambda item: item[1], reverse=True)[:5]
    ]

    success_latencies = [sample.latency_ms for sample in samples if sample.ok]
    error_latencies = [sample.latency_ms for sample in samples if not sample.ok]
    latency_by_status = {
        "success": summarize_latency_distribution(success_latencies),
        "error": summarize_latency_distribution(error_latencies),
    }
    success_p95 = latency_by_status["success"]["p95"]
    error_p95 = latency_by_status["error"]["p95"]
    error_latency_spike = bool(
        not math.isnan(success_p95)
        and not math.isnan(error_p95)
        and success_p95 > 0
        and error_p95 >= (success_p95 * 1.2)
    )

    stage_all = summarize_stage_timings(samples)
    stage_success = summarize_stage_timings([sample for sample in samples if sample.ok])
    stage_error = summarize_stage_timings([sample for sample in samples if not sample.ok])
    stage_contribution_ranked = summarize_stage_contribution(samples)

    retry_samples = [sample for sample in samples if sample.upstream_attempts is not None]
    retry_histogram: Dict[str, int] = {}
    retry_latencies: Dict[str, List[float]] = {}
    fallback_count = 0
    for sample in retry_samples:
        attempt_key = str(int(sample.upstream_attempts or 0))
        retry_histogram[attempt_key] = retry_histogram.get(attempt_key, 0) + 1
        retry_latencies.setdefault(attempt_key, []).append(sample.latency_ms)
        if bool(sample.fallback_used):
            fallback_count += 1
    retry_backoff_impact = {
        "attempt_histogram": retry_histogram,
        "latency_ms_by_attempt": {
            key: summarize_latency_distribution(values) for key, values in sorted(retry_latencies.items(), key=lambda item: int(item[0]))
        },
        "fallback_used_count": fallback_count,
    }

    summary = {
        "endpoint_name": name,
        "url": url,
        "started_at_utc": started_at,
        "duration_s": elapsed_s,
        "concurrency": concurrency,
        "requests_total": len(samples),
        "success_count": success_count,
        "error_count": error_count,
        "error_rate_pct": (error_count / len(samples)) * 100.0,
        "throughput_rps": len(samples) / elapsed_s if elapsed_s > 0 else math.nan,
        "latency_ms": {
            "min": min(latencies),
            "mean": statistics.fmean(latencies),
            "max": max(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "status_histogram": status_histogram,
        "failure_categories": failure_categories,
        "error_categories": error_categories,
        "error_category": "none"
        if error_count == 0
        else max(error_categories.items(), key=lambda item: item[1])[0],
        "latency_by_status_ms": latency_by_status,
        "error_path_latency_spike_detected": error_latency_spike,
        "stage_latency_ms": {
            "all": stage_all,
            "success": stage_success,
            "error": stage_error,
        },
        "ranked_stage_latency_contribution": stage_contribution_ranked,
        "retry_backoff_impact": retry_backoff_impact,
        "top_exception_signatures": top_exception_signatures,
        "sample_errors": errors[:10],
        "sample_correlation_ids": [sample.correlation_id for sample in samples[:10]],
    }
    return summary


def benchmark_endpoint_sustained(
    *,
    name: str,
    url: str,
    payload: Dict[str, Any],
    duration_seconds: int,
    concurrency: int,
    timeout_s: float,
    warmup_requests: int,
) -> Dict[str, Any]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    estimated_requests = max(concurrency * duration_seconds, concurrency)
    return benchmark_endpoint(
        name=name,
        url=url,
        payload=payload,
        requests_count=estimated_requests,
        concurrency=concurrency,
        timeout_s=timeout_s,
        warmup_requests=warmup_requests,
    )


def evaluate_sla(summary: Dict[str, Any], sla: Dict[str, float]) -> Dict[str, Any]:
    p95 = summary["latency_ms"]["p95"]
    p99 = summary["latency_ms"]["p99"]
    error_rate = summary["error_rate_pct"]

    unknown_internal_count = int((summary.get("error_categories") or {}).get("unknown_internal", 0))
    uncategorized_violation = error_rate > sla["error_rate_pct_max"] and unknown_internal_count > 0
    checks = {
        "p95_latency_ms": {"actual": p95, "target_max": sla["p95_latency_ms_max"], "pass": p95 <= sla["p95_latency_ms_max"]},
        "p99_latency_ms": {"actual": p99, "target_max": sla["p99_latency_ms_max"], "pass": p99 <= sla["p99_latency_ms_max"]},
        "error_rate_pct": {"actual": error_rate, "target_max": sla["error_rate_pct_max"], "pass": error_rate <= sla["error_rate_pct_max"]},
        "unknown_internal_guardrail": {
            "actual_unknown_internal_count": unknown_internal_count,
            "error_rate_pct": error_rate,
            "error_rate_threshold_pct": sla["error_rate_pct_max"],
            "pass": not uncategorized_violation,
        },
    }

    overall = all(item["pass"] for item in checks.values())
    return {
        "target_sla": sla,
        "checks": checks,
        "unknown_internal_count": unknown_internal_count,
        "real_time_viability": "PASS" if overall else "FAIL",
    }


def write_outputs(report: Dict[str, Any]) -> Dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = OUTPUT_DIR / f"latency_benchmark_{stamp}.json"
    csv_path = OUTPUT_DIR / f"latency_benchmark_{stamp}.csv"

    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "endpoint_name",
                "url",
                "concurrency",
                "requests_total",
                "success_count",
                "error_count",
                "error_rate_pct",
                "throughput_rps",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "viability",
            ],
        )
        writer.writeheader()
        for endpoint in report["endpoints"]:
            eval_result = report["sla_evaluation"][endpoint["endpoint_name"]]
            writer.writerow(
                {
                    "endpoint_name": endpoint["endpoint_name"],
                    "url": endpoint["url"],
                    "concurrency": endpoint["concurrency"],
                    "requests_total": endpoint["requests_total"],
                    "success_count": endpoint["success_count"],
                    "error_count": endpoint["error_count"],
                    "error_rate_pct": f"{endpoint['error_rate_pct']:.4f}",
                    "throughput_rps": f"{endpoint['throughput_rps']:.2f}",
                    "p50_ms": f"{endpoint['latency_ms']['p50']:.2f}",
                    "p95_ms": f"{endpoint['latency_ms']['p95']:.2f}",
                    "p99_ms": f"{endpoint['latency_ms']['p99']:.2f}",
                    "viability": eval_result["real_time_viability"],
                }
            )

    return {"json": str(json_path), "csv": str(csv_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fraud and wallet latency at controlled concurrency.")
    parser.add_argument("--fraud-url", default="http://127.0.0.1:8000/score_transaction")
    parser.add_argument("--wallet-url", default="http://127.0.0.1:8001/wallet/authorize_payment")
    parser.add_argument("--requests", type=int, default=160, help="Requests per endpoint")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--concurrency-sweep",
        type=str,
        default="",
        help="Comma-separated concurrency values (e.g., 2,6,12) for p95/p99 curve comparison.",
    )
    parser.add_argument("--timeout", type=float, default=2.5, help="Per-request timeout in seconds")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup requests per endpoint")
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=0,
        help="If >0, run sustained load mode for this duration instead of fixed request count.",
    )
    parser.add_argument("--sla-p95-ms", type=float, default=250.0)
    parser.add_argument("--sla-p99-ms", type=float, default=500.0)
    parser.add_argument("--sla-error-rate-pct", type=float, default=1.0)
    parser.add_argument(
        "--preflight-attempts",
        type=int,
        default=20,
        help="Retry count for endpoint preflight checks to reduce transient startup failures.",
    )
    parser.add_argument(
        "--preflight-retry-delay-ms",
        type=float,
        default=500.0,
        help="Delay between preflight retries in milliseconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sla = {
        "p95_latency_ms_max": args.sla_p95_ms,
        "p99_latency_ms_max": args.sla_p99_ms,
        "error_rate_pct_max": args.sla_error_rate_pct,
    }

    contract_validation = validate_benchmark_payload_contract()
    if not contract_validation["ok"]:
        report = {
            "generated_at_utc": iso_now(),
            "benchmark_mode": "contract_validation_failed",
            "failure_mode": "invalid_payload_contract",
            "contract_validation": contract_validation,
            "preflight": {},
            "endpoints": [],
            "sla_evaluation": {},
        }
        output_paths = write_outputs(report)
        print(
            json.dumps(
                {
                    "outputs": output_paths,
                    "benchmark_mode": report["benchmark_mode"],
                    "failure_mode": report["failure_mode"],
                    "contract_validation": contract_validation,
                },
                indent=2,
            )
        )
        return 2

    preflight_attempts = max(1, int(args.preflight_attempts))
    preflight_retry_delay_s = max(0.0, float(args.preflight_retry_delay_ms) / 1000.0)

    preflight: Dict[str, Dict[str, Any]] = {}
    preflight_attempt_log: Dict[str, List[Dict[str, Any]]] = {}
    preflight_targets = {
        "score_transaction": (args.fraud_url, build_base_tx()),
        "wallet_authorize_payment": (args.wallet_url, build_wallet_payload()),
    }
    for endpoint_name, (url, payload) in preflight_targets.items():
        attempts: List[Dict[str, Any]] = []
        final_result: Dict[str, Any] | None = None
        for attempt in range(preflight_attempts):
            check = run_preflight_check(url, payload, args.timeout)
            check["attempt"] = attempt + 1
            attempts.append(check)
            final_result = check
            if check.get("ok"):
                break
            if attempt < (preflight_attempts - 1) and preflight_retry_delay_s > 0:
                time.sleep(preflight_retry_delay_s)
        preflight[endpoint_name] = final_result or {"ok": False, "error": "preflight_attempts_exhausted"}
        preflight_attempt_log[endpoint_name] = attempts
    failed_preflight = [name for name, check in preflight.items() if not check["ok"]]
    if failed_preflight:
        report = {
            "generated_at_utc": iso_now(),
            "benchmark_mode": "preflight_failed",
            "failure_mode": "unreachable_services",
            "preflight": preflight,
            "preflight_attempts": preflight_attempt_log,
            "endpoints": [],
            "sla_evaluation": {},
        }
        output_paths = write_outputs(report)
        print(
            json.dumps(
                {
                    "outputs": output_paths,
                    "benchmark_mode": report["benchmark_mode"],
                    "failure_mode": report["failure_mode"],
                    "preflight": preflight,
                    "preflight_attempts": preflight_attempt_log,
                    "failed_preflight_endpoints": failed_preflight,
                    "sla_evaluation": report["sla_evaluation"],
                },
                indent=2,
            )
        )
        return 2

    sweep_values: List[int] = []
    if args.concurrency_sweep.strip():
        sweep_values = sorted({
            int(token.strip())
            for token in args.concurrency_sweep.split(",")
            if token.strip()
        })
        sweep_values = [value for value in sweep_values if value > 0]

    benchmark_fn = benchmark_endpoint_sustained if args.duration_seconds > 0 else benchmark_endpoint
    benchmark_kwargs = (
        {
            "duration_seconds": args.duration_seconds,
            "concurrency": args.concurrency,
            "timeout_s": args.timeout,
            "warmup_requests": args.warmup,
        }
        if args.duration_seconds > 0
        else {
            "requests_count": args.requests,
            "concurrency": args.concurrency,
            "timeout_s": args.timeout,
            "warmup_requests": args.warmup,
        }
    )

    def run_pair(concurrency_value: int) -> List[Dict[str, Any]]:
        run_kwargs = dict(benchmark_kwargs)
        run_kwargs["concurrency"] = concurrency_value
        fraud_summary_local = benchmark_fn(
            name="score_transaction",
            url=args.fraud_url,
            payload=build_base_tx(),
            **run_kwargs,
        )
        wallet_summary_local = benchmark_fn(
            name="wallet_authorize_payment",
            url=args.wallet_url,
            payload=build_wallet_payload(),
            **run_kwargs,
        )
        return [fraud_summary_local, wallet_summary_local]

    all_endpoint_runs: List[Dict[str, Any]] = []
    curve_comparison: Dict[str, List[Dict[str, Any]]] = {
        "score_transaction": [],
        "wallet_authorize_payment": [],
    }

    sweep_plan = sweep_values if sweep_values else [args.concurrency]
    for concurrency_value in sweep_plan:
        pair = run_pair(concurrency_value)
        all_endpoint_runs.extend(pair)
        for summary in pair:
            curve_comparison[summary["endpoint_name"]].append(
                {
                    "concurrency": concurrency_value,
                    "p95_ms": summary["latency_ms"]["p95"],
                    "p99_ms": summary["latency_ms"]["p99"],
                    "error_rate_pct": summary["error_rate_pct"],
                    "throughput_rps": summary["throughput_rps"],
                }
            )

    report = {
        "benchmark_run_id": str(uuid.uuid4()),
        "generated_at_utc": iso_now(),
        "benchmark_mode": "sustained_load_test" if args.duration_seconds > 0 else "load_test",
        "failure_mode": "none",
        "contract_validation": contract_validation,
        "preflight": preflight,
        "preflight_attempts": preflight_attempt_log,
        "hardware_profile": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor() or "unknown",
            "logical_cpu_count": __import__("os").cpu_count(),
        },
        "benchmark_config": {
            "requests_per_endpoint": args.requests,
            "duration_seconds": args.duration_seconds,
            "concurrency": args.concurrency,
            "concurrency_sweep": sweep_values,
            "request_timeout_s": args.timeout,
            "warmup_requests": args.warmup,
        },
        "endpoints": all_endpoint_runs,
        "latency_curve_comparison": curve_comparison,
        "sla_evaluation": {
            "score_transaction": evaluate_sla(
                next(item for item in all_endpoint_runs if item["endpoint_name"] == "score_transaction" and item["concurrency"] == sweep_plan[-1]),
                sla,
            ),
            "wallet_authorize_payment": evaluate_sla(
                next(item for item in all_endpoint_runs if item["endpoint_name"] == "wallet_authorize_payment" and item["concurrency"] == sweep_plan[-1]),
                sla,
            ),
        },
    }
    if any(result["real_time_viability"] != "PASS" for result in report["sla_evaluation"].values()):
        report["failure_mode"] = "sla_violation"

    output_paths = write_outputs(report)
    print(
        json.dumps(
            {
                "outputs": output_paths,
                "benchmark_mode": report["benchmark_mode"],
                "failure_mode": report["failure_mode"],
                "preflight": preflight,
                "sla_evaluation": report["sla_evaluation"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

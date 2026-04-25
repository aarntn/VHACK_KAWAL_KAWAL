#!/usr/bin/env python3
"""Run benchmark matrix and summarize stage-level latency behavior.

Matrix dimensions:
- concurrency: 5, 10, 20
- request_count: 200, 1000
- start_mode: cold (warmup=0) vs warm (warmup=5)

Uses existing benchmark instrumentation from benchmark_latency.py and produces:
- matrix-level endpoint latency summary
- stage p50/p95/p99 by endpoint + run, including success vs error
- ranked stage contribution table
- retry/backoff impact summary
- trend vs previous 7 runs by endpoint/stage
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXEC = sys.executable or "python"
BENCHMARK_OUTPUT_DIR = REPO_ROOT / "project" / "outputs" / "benchmark"
MONITORING_DIR = REPO_ROOT / "project" / "outputs" / "monitoring"
DEFAULT_MATRIX_JSON = MONITORING_DIR / "latency_stage_matrix_report.json"
DEFAULT_MATRIX_CSV = MONITORING_DIR / "latency_stage_matrix_report.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark matrix and generate stage-level latency report.")
    parser.add_argument("--fraud-url", default="http://127.0.0.1:8000/score_transaction")
    parser.add_argument("--wallet-url", default="http://127.0.0.1:8001/wallet/authorize_payment")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--sla-p95-ms", type=float, default=250.0)
    parser.add_argument("--sla-p99-ms", type=float, default=500.0)
    parser.add_argument("--sla-error-rate-pct", type=float, default=1.0)
    parser.add_argument("--history-runs", type=int, default=7)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_MATRIX_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_MATRIX_CSV)
    parser.add_argument("--concurrency", type=int, nargs="*", default=[5, 10, 20])
    parser.add_argument("--requests", type=int, nargs="*", default=[200, 1000])
    parser.add_argument("--cold-warmup", type=int, default=0)
    parser.add_argument("--warm-warmup", type=int, default=5)
    return parser.parse_args()


def _run_command(cmd: List[str]) -> Dict[str, Any]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _extract_json_from_stdout(stdout: str) -> Dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        raise RuntimeError("benchmark script emitted empty stdout")
    return json.loads(text)


def run_single_benchmark(args: argparse.Namespace, concurrency: int, requests_count: int, start_mode: str) -> Dict[str, Any]:
    warmup = args.warm_warmup if start_mode == "warm" else args.cold_warmup
    cmd = [
        PYTHON_EXEC,
        "project/scripts/benchmark_latency.py",
        "--fraud-url",
        str(args.fraud_url),
        "--wallet-url",
        str(args.wallet_url),
        "--requests",
        str(requests_count),
        "--concurrency",
        str(concurrency),
        "--warmup",
        str(warmup),
        "--timeout",
        str(args.timeout),
        "--sla-p95-ms",
        str(args.sla_p95_ms),
        "--sla-p99-ms",
        str(args.sla_p99_ms),
        "--sla-error-rate-pct",
        str(args.sla_error_rate_pct),
    ]
    execution = _run_command(cmd)
    if not execution["stdout"].strip():
        raise RuntimeError(f"benchmark command produced no stdout: {' '.join(cmd)}\n{execution['stderr']}")

    parsed = _extract_json_from_stdout(execution["stdout"])
    outputs = parsed.get("outputs") or {}
    json_path = outputs.get("json")
    if not json_path:
        raise RuntimeError(f"benchmark command missing outputs.json path: {execution['stdout']}")

    report_payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return {
        "execution": execution,
        "parsed": parsed,
        "report": report_payload,
        "config": {
            "concurrency": concurrency,
            "requests": requests_count,
            "start_mode": start_mode,
            "warmup": warmup,
        },
        "benchmark_output_json": json_path,
    }


def _extract_stage_rows(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    report = run["report"]
    cfg = run["config"]
    run_stamp = report.get("generated_at_utc")
    for endpoint in report.get("endpoints", []):
        endpoint_name = endpoint.get("endpoint_name")
        stage_views = endpoint.get("stage_latency_ms") or {}
        for status_name, status_stages in stage_views.items():
            if not isinstance(status_stages, dict):
                continue
            for stage_name, values in status_stages.items():
                if not isinstance(values, dict):
                    continue
                rows.append(
                    {
                        "run_generated_at_utc": run_stamp,
                        "run_start_mode": cfg["start_mode"],
                        "run_concurrency": cfg["concurrency"],
                        "run_requests": cfg["requests"],
                        "endpoint_name": endpoint_name,
                        "status_class": status_name,
                        "stage": stage_name,
                        "count": values.get("count"),
                        "mean_ms": values.get("mean"),
                        "p50_ms": values.get("p50"),
                        "p95_ms": values.get("p95"),
                        "p99_ms": values.get("p99"),
                    }
                )
    return rows


def _extract_contribution_rows(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    report = run["report"]
    cfg = run["config"]
    run_stamp = report.get("generated_at_utc")
    for endpoint in report.get("endpoints", []):
        endpoint_name = endpoint.get("endpoint_name")
        for rank, entry in enumerate(endpoint.get("ranked_stage_latency_contribution") or [], start=1):
            rows.append(
                {
                    "run_generated_at_utc": run_stamp,
                    "run_start_mode": cfg["start_mode"],
                    "run_concurrency": cfg["concurrency"],
                    "run_requests": cfg["requests"],
                    "endpoint_name": endpoint_name,
                    "rank": rank,
                    "stage": entry.get("stage"),
                    "mean_ms": entry.get("mean_ms"),
                    "mean_share_of_total_pct": entry.get("mean_share_of_total_pct"),
                    "observations": entry.get("observations"),
                }
            )
    return rows


def _extract_endpoint_rows(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    report = run["report"]
    cfg = run["config"]
    run_stamp = report.get("generated_at_utc")
    for endpoint in report.get("endpoints", []):
        retry_impact = endpoint.get("retry_backoff_impact") or {}
        rows.append(
            {
                "run_generated_at_utc": run_stamp,
                "run_start_mode": cfg["start_mode"],
                "run_concurrency": cfg["concurrency"],
                "run_requests": cfg["requests"],
                "endpoint_name": endpoint.get("endpoint_name"),
                "requests_total": endpoint.get("requests_total"),
                "success_count": endpoint.get("success_count"),
                "error_count": endpoint.get("error_count"),
                "error_rate_pct": endpoint.get("error_rate_pct"),
                "latency_p50_ms": (endpoint.get("latency_ms") or {}).get("p50"),
                "latency_p95_ms": (endpoint.get("latency_ms") or {}).get("p95"),
                "latency_p99_ms": (endpoint.get("latency_ms") or {}).get("p99"),
                "success_p95_ms": ((endpoint.get("latency_by_status_ms") or {}).get("success") or {}).get("p95"),
                "error_p95_ms": ((endpoint.get("latency_by_status_ms") or {}).get("error") or {}).get("p95"),
                "error_path_latency_spike_detected": endpoint.get("error_path_latency_spike_detected"),
                "dominant_error_category": endpoint.get("error_category"),
                "retry_attempt_histogram": retry_impact.get("attempt_histogram"),
                "fallback_used_count": retry_impact.get("fallback_used_count"),
            }
        )
    return rows


def load_previous_matrix_reports(history_runs: int, latest_path: Path) -> List[Dict[str, Any]]:
    files = sorted(MONITORING_DIR.glob("latency_stage_matrix_report*.json"))
    previous_files = [path for path in files if path.resolve() != latest_path.resolve()]
    if history_runs > 0:
        previous_files = previous_files[-history_runs:]

    historical_rows: List[Dict[str, Any]] = []
    for path in previous_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("stage_rows", []):
            row_copy = dict(row)
            row_copy["history_report_file"] = path.name
            historical_rows.append(row_copy)
    return historical_rows


def build_stage_trend_vs_history(stage_rows: List[Dict[str, Any]], history_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    current_df = pd.DataFrame(stage_rows)
    if current_df.empty:
        return []

    key_cols = ["endpoint_name", "status_class", "stage"]
    latest = (
        current_df.sort_values("run_generated_at_utc")
        .groupby(key_cols, dropna=False)
        .tail(1)
        .reset_index(drop=True)
    )

    history_df = pd.DataFrame(history_rows)
    if history_df.empty:
        latest["history_p95_mean_ms"] = None
        latest["history_p95_delta_ms"] = None
        return latest[
            [
                "endpoint_name",
                "status_class",
                "stage",
                "p95_ms",
                "history_p95_mean_ms",
                "history_p95_delta_ms",
            ]
        ].to_dict(orient="records")

    history_agg = (
        history_df.groupby(key_cols, dropna=False)["p95_ms"]
        .mean()
        .rename("history_p95_mean_ms")
        .reset_index()
    )
    merged = latest.merge(history_agg, on=key_cols, how="left")
    merged["history_p95_delta_ms"] = merged["p95_ms"] - merged["history_p95_mean_ms"]

    return merged[
        [
            "endpoint_name",
            "status_class",
            "stage",
            "p95_ms",
            "history_p95_mean_ms",
            "history_p95_delta_ms",
        ]
    ].to_dict(orient="records")


def build_report_payload(args: argparse.Namespace, runs: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    endpoint_rows: List[Dict[str, Any]] = []
    stage_rows: List[Dict[str, Any]] = []
    contribution_rows: List[Dict[str, Any]] = []

    for run in runs:
        endpoint_rows.extend(_extract_endpoint_rows(run))
        stage_rows.extend(_extract_stage_rows(run))
        contribution_rows.extend(_extract_contribution_rows(run))

    endpoint_df = pd.DataFrame(endpoint_rows)
    stage_df = pd.DataFrame(stage_rows)
    contribution_df = pd.DataFrame(contribution_rows)

    previous_history_rows = load_previous_matrix_reports(args.history_runs, args.output_json)
    stage_trend = build_stage_trend_vs_history(stage_rows, previous_history_rows)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix": {
            "concurrency": list(args.concurrency),
            "requests": list(args.requests),
            "start_modes": ["cold", "warm"],
            "cold_warmup": args.cold_warmup,
            "warm_warmup": args.warm_warmup,
        },
        "total_runs": len(runs),
        "runs": [
            {
                **run["config"],
                "benchmark_output_json": run["benchmark_output_json"],
                "generated_at_utc": run["report"].get("generated_at_utc"),
                "benchmark_mode": run["report"].get("benchmark_mode"),
                "failure_mode": run["report"].get("failure_mode"),
            }
            for run in runs
        ],
        "endpoint_rows": endpoint_rows,
        "stage_rows": stage_rows,
        "ranked_stage_contribution_rows": contribution_rows,
        "stage_breakdown_trend_vs_previous_7_runs": stage_trend,
        "history_reference": {
            "history_runs": int(args.history_runs),
            "history_points": len(previous_history_rows),
        },
    }

    if endpoint_df.empty:
        csv_df = pd.DataFrame(columns=["endpoint_name", "run_start_mode", "run_concurrency", "run_requests"])
    else:
        csv_df = endpoint_df.sort_values(["endpoint_name", "run_start_mode", "run_concurrency", "run_requests"])  # type: ignore[arg-type]

    payload["summary"] = {
        "error_spike_runs": int(endpoint_df["error_path_latency_spike_detected"].fillna(False).astype(bool).sum()) if not endpoint_df.empty else 0,
        "endpoints_observed": sorted([str(x) for x in endpoint_df["endpoint_name"].dropna().unique()]) if not endpoint_df.empty else [],
        "top_stage_contributors_overall": contribution_df.sort_values("mean_share_of_total_pct", ascending=False).head(10).to_dict(orient="records")
        if not contribution_df.empty
        else [],
    }

    return payload, csv_df


def main() -> int:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    for start_mode in ["cold", "warm"]:
        for requests_count in args.requests:
            for concurrency in args.concurrency:
                runs.append(run_single_benchmark(args, concurrency, requests_count, start_mode))

    payload, csv_df = build_report_payload(args, runs)

    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_df.to_csv(args.output_csv, index=False)

    print(
        json.dumps(
            {
                "total_runs": len(runs),
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

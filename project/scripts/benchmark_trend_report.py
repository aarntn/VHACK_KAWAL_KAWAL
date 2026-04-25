import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "project" / "outputs" / "benchmark"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "project" / "outputs" / "monitoring" / "latency_trend_report.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "project" / "outputs" / "monitoring" / "latency_trend_report.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build latency trend report from benchmark json history.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--history-limit", type=int, default=20)
    return parser.parse_args()


def load_benchmarks(benchmark_dir: Path, history_limit: int) -> List[Dict]:
    files = sorted(benchmark_dir.glob("latency_benchmark_*.json"))
    if history_limit > 0:
        files = files[-history_limit:]

    rows: List[Dict] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        generated_at = payload.get("generated_at_utc", "")

        for endpoint in payload.get("endpoints", []):
            eval_payload = payload.get("sla_evaluation", {}).get(endpoint.get("endpoint_name", ""), {})
            checks = eval_payload.get("checks", {})
            rows.append(
                {
                    "benchmark_file": path.name,
                    "generated_at_utc": generated_at,
                    "endpoint_name": endpoint.get("endpoint_name"),
                    "p50_ms": endpoint.get("latency_ms", {}).get("p50"),
                    "p95_ms": endpoint.get("latency_ms", {}).get("p95"),
                    "p99_ms": endpoint.get("latency_ms", {}).get("p99"),
                    "error_rate_pct": endpoint.get("error_rate_pct"),
                    "throughput_rps": endpoint.get("throughput_rps"),
                    "viability": eval_payload.get("real_time_viability", "unknown"),
                    "p95_target_max": checks.get("p95_latency_ms", {}).get("target_max"),
                    "p99_target_max": checks.get("p99_latency_ms", {}).get("target_max"),
                    "error_target_max": checks.get("error_rate_pct", {}).get("target_max"),
                }
            )
    return rows


def compute_latest(rows_df: pd.DataFrame) -> List[Dict]:
    latest_rows: List[Dict] = []
    for endpoint in rows_df["endpoint_name"].dropna().unique():
        endpoint_df = rows_df[rows_df["endpoint_name"] == endpoint].sort_values("generated_at_utc")
        latest = endpoint_df.iloc[-1]
        latest_rows.append(
            {
                "endpoint_name": endpoint,
                "latest_generated_at_utc": latest["generated_at_utc"],
                "latest_p50_ms": float(latest["p50_ms"]),
                "latest_p95_ms": float(latest["p95_ms"]),
                "latest_p99_ms": float(latest["p99_ms"]),
                "latest_error_rate_pct": float(latest["error_rate_pct"]),
                "latest_viability": str(latest["viability"]),
            }
        )
    return latest_rows


def write_no_data_outputs(output_json: Path, output_csv: Path, benchmark_dir: Path) -> Dict[str, str]:
    timestamp = datetime.now(timezone.utc).isoformat()
    payload: Dict[str, str] = {
        "status": "no_data",
        "reason": "no benchmark JSON files found",
        "generated_at_utc": timestamp,
        "benchmark_dir": str(benchmark_dir),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([payload]).to_csv(output_csv, index=False)
    return payload


def main() -> None:
    args = parse_args()
    rows = load_benchmarks(args.benchmark_dir, args.history_limit)
    if not rows:
        no_data_payload = write_no_data_outputs(args.output_json, args.output_csv, args.benchmark_dir)
        print("Latency trend report complete (no data)")
        print(json.dumps(no_data_payload, indent=2))
        print(f"Saved trend JSON: {args.output_json}")
        print(f"Saved trend CSV: {args.output_csv}")
        return

    rows_df = pd.DataFrame(rows)
    rows_df = rows_df.sort_values(["generated_at_utc", "endpoint_name"]).reset_index(drop=True)

    latest_by_endpoint = compute_latest(rows_df)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_dir": str(args.benchmark_dir),
        "history_limit": args.history_limit,
        "history_points": int(len(rows_df)),
        "latest_by_endpoint": latest_by_endpoint,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows_df.to_csv(args.output_csv, index=False)

    print("Latency trend report complete")
    print(json.dumps({"history_points": len(rows_df), "endpoints": len(latest_by_endpoint)}, indent=2))
    print(f"Saved trend JSON: {args.output_json}")
    print(f"Saved trend CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

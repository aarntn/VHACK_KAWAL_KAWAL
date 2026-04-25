import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate canary telemetry windows against error-rate and p95 thresholds, "
            "emit rollout decision, and archive telemetry with release identifiers."
        )
    )
    parser.add_argument("--telemetry-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, default=Path("project/outputs/rollout_telemetry"))
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--commit-sha", default="")
    parser.add_argument("--canary-traffic-percent", type=float, default=5.0)
    parser.add_argument("--max-error-rate-pct", type=float, default=1.0)
    parser.add_argument("--max-p95-latency-ms", type=float, default=250.0)
    parser.add_argument("--max-unknown-error-pct", type=float, default=0.1)
    parser.add_argument("--rollback-consecutive-windows", type=int, default=3)
    return parser.parse_args()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _window_has_threshold_breach(
    window: dict[str, Any],
    max_error_rate_pct: float,
    max_p95_latency_ms: float,
    max_unknown_error_pct: float,
) -> tuple[bool, list[dict[str, Any]]]:
    endpoint_metrics = window.get("endpoint_metrics")
    if not isinstance(endpoint_metrics, list):
        return True, [{"endpoint": "unknown", "reason": "endpoint_metrics_missing"}]

    breaches: list[dict[str, Any]] = []
    for endpoint in endpoint_metrics:
        if not isinstance(endpoint, dict):
            continue
        name = str(endpoint.get("endpoint_name", "unknown"))
        error_rate_pct = float(endpoint.get("error_rate_pct", 0.0) or 0.0)
        p95_latency_ms = float(endpoint.get("p95_latency_ms", 0.0) or 0.0)
        categories = endpoint.get("error_categories") or {}
        if not isinstance(categories, dict):
            categories = {}
        unknown_error_pct = float(categories.get("unknown_internal", 0.0) or 0.0)

        if error_rate_pct > max_error_rate_pct:
            breaches.append(
                {
                    "endpoint": name,
                    "metric": "error_rate_pct",
                    "actual": error_rate_pct,
                    "threshold": max_error_rate_pct,
                }
            )
        if p95_latency_ms > max_p95_latency_ms:
            breaches.append(
                {
                    "endpoint": name,
                    "metric": "p95_latency_ms",
                    "actual": p95_latency_ms,
                    "threshold": max_p95_latency_ms,
                }
            )
        if unknown_error_pct > max_unknown_error_pct:
            breaches.append(
                {
                    "endpoint": name,
                    "metric": "unknown_internal_pct",
                    "actual": unknown_error_pct,
                    "threshold": max_unknown_error_pct,
                }
            )
    return len(breaches) > 0, breaches


def evaluate_rollout(
    windows: list[dict[str, Any]],
    max_error_rate_pct: float,
    max_p95_latency_ms: float,
    max_unknown_error_pct: float,
    rollback_consecutive_windows: int,
) -> dict[str, Any]:
    window_results: list[dict[str, Any]] = []
    consecutive_breach_streak = 0
    max_streak = 0
    for window in windows:
        breached, breaches = _window_has_threshold_breach(
            window=window,
            max_error_rate_pct=max_error_rate_pct,
            max_p95_latency_ms=max_p95_latency_ms,
            max_unknown_error_pct=max_unknown_error_pct,
        )
        if breached:
            consecutive_breach_streak += 1
            max_streak = max(max_streak, consecutive_breach_streak)
        else:
            consecutive_breach_streak = 0
        window_results.append(
            {
                "window_id": window.get("window_id"),
                "started_at_utc": window.get("started_at_utc"),
                "ended_at_utc": window.get("ended_at_utc"),
                "breached": breached,
                "breaches": breaches,
            }
        )

    should_rollback = max_streak >= rollback_consecutive_windows
    return {
        "window_results": window_results,
        "max_consecutive_breach_streak": max_streak,
        "rollback_triggered": should_rollback,
        "decision": "rollback" if should_rollback else "promote",
        "reason": (
            f"threshold_breach_streak_{max_streak}_gte_{rollback_consecutive_windows}"
            if should_rollback
            else "all_windows_within_thresholds_or_nonconsecutive_breaches"
        ),
    }


def archive_rollout_telemetry(
    telemetry_payload: dict[str, Any],
    archive_dir: Path,
    release_metadata: dict[str, Any],
) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = _utc_timestamp()
    release_id = str(release_metadata.get("release_id", "release"))
    out_path = archive_dir / f"{release_id}_{ts}.json"
    archival_payload = {
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_metadata": release_metadata,
        "telemetry": telemetry_payload,
    }
    out_path.write_text(json.dumps(archival_payload, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    args = parse_args()
    telemetry_payload = load_json(args.telemetry_json)
    windows = telemetry_payload.get("windows")
    if not isinstance(windows, list) or not windows:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "status": "fail",
                    "reason": "windows_missing_or_empty",
                    "telemetry_json": str(args.telemetry_json),
                }
            ),
            encoding="utf-8",
        )
        return 1

    decision = evaluate_rollout(
        windows=windows,
        max_error_rate_pct=args.max_error_rate_pct,
        max_p95_latency_ms=args.max_p95_latency_ms,
        max_unknown_error_pct=args.max_unknown_error_pct,
        rollback_consecutive_windows=args.rollback_consecutive_windows,
    )
    release_metadata = {
        "release_id": args.release_id,
        "artifact_id": args.artifact_id,
        "artifact_version": args.artifact_version,
        "commit_sha": args.commit_sha,
        "canary_traffic_percent": args.canary_traffic_percent,
        "thresholds": {
            "max_error_rate_pct": args.max_error_rate_pct,
            "max_p95_latency_ms": args.max_p95_latency_ms,
            "max_unknown_error_pct": args.max_unknown_error_pct,
            "rollback_consecutive_windows": args.rollback_consecutive_windows,
        },
    }
    archive_path = archive_rollout_telemetry(
        telemetry_payload=telemetry_payload,
        archive_dir=args.archive_dir,
        release_metadata=release_metadata,
    )

    payload = {
        "status": "ok",
        "decision": decision["decision"],
        "rollback_triggered": decision["rollback_triggered"],
        "reason": decision["reason"],
        "max_consecutive_breach_streak": decision["max_consecutive_breach_streak"],
        "window_results": decision["window_results"],
        "release_metadata": release_metadata,
        "archive_path": str(archive_path),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

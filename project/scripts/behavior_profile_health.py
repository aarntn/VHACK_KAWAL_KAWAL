import argparse
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE_PATH = REPO_ROOT / "project" / "outputs" / "behavior_profiles.sqlite3"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "project" / "outputs" / "monitoring" / "behavior_profile_health.json"


@dataclass
class ProfileRow:
    user_id: str
    version: int
    updated_at: float
    expires_at: float
    payload: Dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect behavioral profile-store health (coverage, staleness, low-history ratio)."
    )
    parser.add_argument("--store-backend", choices=["sqlite", "memory", "redis"], default="sqlite")
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--stale-seconds", type=int, default=7 * 24 * 3600)
    parser.add_argument("--coverage-warn-min-active", type=int, default=50)
    parser.add_argument("--low-history-warn-ratio", type=float, default=0.50)
    parser.add_argument("--stale-warn-ratio", type=float, default=0.40)
    return parser.parse_args()


def load_sqlite_profiles(sqlite_path: Path) -> List[ProfileRow]:
    if not sqlite_path.exists():
        return []

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT user_id, payload, version, updated_at, expires_at FROM behavior_profiles"
        ).fetchall()
    finally:
        conn.close()

    parsed: List[ProfileRow] = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            payload = {}
        parsed.append(
            ProfileRow(
                user_id=str(row["user_id"]),
                version=int(row["version"]),
                updated_at=float(row["updated_at"]),
                expires_at=float(row["expires_at"]),
                payload=payload,
            )
        )
    return parsed


def compute_health(
    profiles: List[ProfileRow],
    now_ts: float,
    min_history: int,
    stale_seconds: int,
    coverage_warn_min_active: int,
    low_history_warn_ratio: float,
    stale_warn_ratio: float,
) -> Dict[str, Any]:
    total = len(profiles)
    active = [p for p in profiles if p.expires_at >= now_ts]
    expired = total - len(active)

    if not active:
        return {
            "summary": {
                "total_profiles": total,
                "active_profiles": 0,
                "expired_profiles": expired,
                "low_history_ratio": 1.0 if total > 0 else 0.0,
                "stale_ratio": 0.0,
                "avg_history_count": 0.0,
            },
            "warnings": [
                "No active behavioral profiles found in store"
            ],
            "status": "warn" if total > 0 else "ok",
        }

    history_counts: List[int] = []
    low_history_count = 0
    stale_count = 0

    for profile in active:
        amounts = profile.payload.get("amounts", [])
        history_count = len(amounts) if isinstance(amounts, list) else 0
        history_counts.append(history_count)
        if history_count < min_history:
            low_history_count += 1
        if (now_ts - profile.updated_at) > stale_seconds:
            stale_count += 1

    low_history_ratio = low_history_count / len(active)
    stale_ratio = stale_count / len(active)
    avg_history_count = sum(history_counts) / len(history_counts)

    warnings: List[str] = []
    if len(active) < coverage_warn_min_active:
        warnings.append(
            f"Active behavior-profile coverage below threshold: {len(active)} < {coverage_warn_min_active}"
        )
    if low_history_ratio >= low_history_warn_ratio:
        warnings.append(
            f"High low-history ratio: {low_history_ratio:.3f} >= {low_history_warn_ratio:.3f}"
        )
    if stale_ratio >= stale_warn_ratio:
        warnings.append(
            f"High stale-profile ratio: {stale_ratio:.3f} >= {stale_warn_ratio:.3f}"
        )

    return {
        "summary": {
            "total_profiles": total,
            "active_profiles": len(active),
            "expired_profiles": expired,
            "low_history_ratio": round(low_history_ratio, 6),
            "stale_ratio": round(stale_ratio, 6),
            "avg_history_count": round(avg_history_count, 3),
        },
        "warnings": warnings,
        "status": "warn" if warnings else "ok",
    }


def main() -> None:
    args = parse_args()
    now_ts = time.time()

    if args.store_backend != "sqlite":
        output = {
            "generated_at_utc": utc_now_iso(),
            "store_backend": args.store_backend,
            "status": "warn",
            "summary": {
                "total_profiles": 0,
                "active_profiles": 0,
                "expired_profiles": 0,
                "low_history_ratio": 0.0,
                "stale_ratio": 0.0,
                "avg_history_count": 0.0,
            },
            "warnings": [
                f"Health inspection currently supports sqlite backend only; received {args.store_backend}"
            ],
        }
    else:
        profiles = load_sqlite_profiles(args.sqlite_path)
        health = compute_health(
            profiles=profiles,
            now_ts=now_ts,
            min_history=args.min_history,
            stale_seconds=args.stale_seconds,
            coverage_warn_min_active=args.coverage_warn_min_active,
            low_history_warn_ratio=args.low_history_warn_ratio,
            stale_warn_ratio=args.stale_warn_ratio,
        )

        output = {
            "generated_at_utc": utc_now_iso(),
            "store_backend": args.store_backend,
            "sqlite_path": str(args.sqlite_path),
            "status": health["status"],
            "summary": health["summary"],
            "warnings": health["warnings"],
            "thresholds": {
                "min_history": args.min_history,
                "stale_seconds": args.stale_seconds,
                "coverage_warn_min_active": args.coverage_warn_min_active,
                "low_history_warn_ratio": args.low_history_warn_ratio,
                "stale_warn_ratio": args.stale_warn_ratio,
            },
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Behavior profile health check complete")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved health report JSON: {args.output_json}")


if __name__ == "__main__":
    main()

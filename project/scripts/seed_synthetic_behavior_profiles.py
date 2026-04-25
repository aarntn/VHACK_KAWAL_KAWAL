"""
Seed behavior profiles with synthetic data.

Self-contained — writes directly to SQLite without importing project.app,
so it works even when fastapi / uvicorn are not installed in the current env.

Usage:
    python project/scripts/seed_synthetic_behavior_profiles.py
    python project/scripts/seed_synthetic_behavior_profiles.py --users 500 --txns 20
"""
import argparse
import json
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SQLITE = REPO_ROOT / "project" / "outputs" / "behavior_profiles.sqlite3"
DEFAULT_SUMMARY = REPO_ROOT / "project" / "outputs" / "monitoring" / "profile_replay_summary.json"

# Realistic ASEAN wallet amount ranges (normalised to USD-equivalent)
_AMOUNT_BUCKETS = {
    "low":  (5.0,   50.0),    # small top-ups, transport fares
    "mid":  (50.0,  300.0),   # grocery, utility bills
    "high": (300.0, 2000.0),  # remittances, merchant payouts
}

_HOUR_BUCKETS = {
    "morning":   (6, 11),
    "afternoon": (12, 17),
    "evening":   (18, 22),
    "night":     (0, 5),
}


def _random_amount(rng: random.Random, bucket: str) -> float:
    lo, hi = _AMOUNT_BUCKETS[bucket]
    return round(rng.uniform(lo, hi), 2)


def _random_hour(rng: random.Random, bucket: str) -> int:
    lo, hi = _HOUR_BUCKETS[bucket]
    return rng.randint(lo, hi)


def _build_history(rng: random.Random, n: int, fraud_rate: float):
    """Return list of (amount, hour, location_risk) with realistic ASEAN distribution."""
    amt_bucket = rng.choice(["low", "low", "mid", "mid", "high"])
    hr_bucket = rng.choice(["morning", "afternoon", "afternoon", "evening", "night"])
    rows = []
    for _ in range(n):
        is_fraud = rng.random() < fraud_rate
        amt = round(_random_amount(rng, amt_bucket) * rng.uniform(0.7, 1.4), 2)
        hour = max(0, min(23, _random_hour(rng, hr_bucket) + rng.randint(-1, 1)))
        loc = round(rng.uniform(0.6, 0.95) if is_fraud else rng.uniform(0.0, 0.25), 3)
        rows.append((amt, hour, loc))
    return rows


def _build_payload(user_id: str, history, version: int, now: float) -> dict:
    amounts = [r[0] for r in history]
    hours   = [r[1] for r in history]
    locs    = [r[2] for r in history]
    return {
        "user_id": user_id,
        "amounts": amounts,
        "hours": hours,
        "location_risks": locs,
        "event_timestamps": [0.0] * len(history),
        "geo_device_mismatch_flags": [0] * len(history),
        "counterparties_24h": [0] * len(history),
        "total_transactions": len(history),
        "geo_device_mismatch_count": 0,
        "payload_schema_version": 2,
        "version": version,
        "updated_at": now,
    }


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS behavior_profiles (
            user_id   TEXT PRIMARY KEY,
            payload   TEXT NOT NULL,
            version   INTEGER NOT NULL,
            updated_at REAL NOT NULL,
            expires_at REAL NOT NULL
        )
        """
    )
    conn.commit()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed synthetic behavior profiles (no dataset required)")
    p.add_argument("--users",      type=int,   default=300,  help="Number of synthetic users")
    p.add_argument("--txns",       type=int,   default=18,   help="Transactions per user")
    p.add_argument("--fraud-rate", type=float, default=0.05, help="Fraction labelled fraud")
    p.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE)
    p.add_argument("--output-json", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--ttl-days",   type=int,   default=365,  help="Profile TTL in days")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    ttl_seconds = args.ttl_days * 86400
    now = time.time()
    expires_at = now + ttl_seconds

    args.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(args.sqlite_path))
    conn.row_factory = sqlite3.Row
    _init_db(conn)

    seeded = 0
    total_txns = 0
    for i in range(args.users):
        user_id = f"synthetic_user_{i:04d}"
        history = _build_history(rng, args.txns, args.fraud_rate)
        existing = conn.execute(
            "SELECT version FROM behavior_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        version = 1 if existing is None else int(existing["version"]) + 1
        payload = _build_payload(user_id, history, version, now)
        conn.execute(
            """
            INSERT INTO behavior_profiles (user_id, payload, version, updated_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                payload    = excluded.payload,
                version    = excluded.version,
                updated_at = excluded.updated_at,
                expires_at = excluded.expires_at
            """,
            (user_id, json.dumps(payload), version, now, expires_at),
        )
        seeded += 1
        total_txns += len(history)

    conn.commit()
    conn.close()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "synthetic",
        "sqlite_path": str(args.sqlite_path),
        "users_written": seeded,
        "transactions_replayed": total_txns,
        "transactions_per_user": args.txns,
        "ttl_seconds": ttl_seconds,
        "ttl_days": args.ttl_days,
        "seed": args.seed,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Seeded {seeded} synthetic behavior profiles  ({total_txns} transactions total)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

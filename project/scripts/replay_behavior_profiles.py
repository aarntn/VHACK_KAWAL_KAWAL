import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.app.behavior_profile import BehaviorProfiler
from project.app.profile_store import SQLiteProfileStore
from project.data.dataset_loader import load_creditcard, load_ieee_cis


DEFAULT_SQLITE_PATH = REPO_ROOT / "project" / "outputs" / "behavior_profiles.sqlite3"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "project" / "outputs" / "monitoring" / "profile_replay_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay historical transactions to seed behavior profiles.")
    parser.add_argument("--dataset-source", choices=["creditcard", "ieee_cis"], default="ieee_cis")
    parser.add_argument("--dataset-path", type=Path, help="Legacy credit-card CSV path (used when --dataset-source creditcard)")
    parser.add_argument("--ieee-transaction-path", type=Path)
    parser.add_argument("--ieee-identity-path", type=Path)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--user-count", type=int, default=200)
    parser.add_argument("--transactions-per-user", type=int, default=12)
    parser.add_argument("--ttl-seconds", type=int, default=30 * 24 * 3600)
    parser.add_argument("--history-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_hour(value: float) -> int:
    return int((float(value) // 3600) % 24)


def build_output(args: argparse.Namespace, users_written: int, replayed_rows: int) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.dataset_path),
        "sqlite_path": str(args.sqlite_path),
        "users_written": users_written,
        "transactions_replayed": replayed_rows,
        "transactions_per_user": args.transactions_per_user,
        "ttl_seconds": args.ttl_seconds,
        "history_limit": args.history_limit,
    }


def main() -> None:
    args = parse_args()
    if args.user_count <= 0 or args.transactions_per_user <= 0:
        raise ValueError("user-count and transactions-per-user must be > 0")

    if args.dataset_source == "ieee_cis":
        if not args.ieee_transaction_path or not args.ieee_identity_path:
            raise ValueError("--ieee-transaction-path and --ieee-identity-path are required for ieee_cis source")
        features, labels, _ = load_ieee_cis(args.ieee_transaction_path, args.ieee_identity_path)
        amount_col = "TransactionAmt"
        time_col = "TransactionDT"
    else:
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for creditcard source (legacy assets now in project/legacy_creditcard)")
        if not args.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
        features, labels, _ = load_creditcard(args.dataset_path)
        amount_col = "Amount"
        time_col = "Time"

    df = features.copy()
    df["Class"] = labels.astype(int).to_numpy()
    if amount_col not in df.columns or time_col not in df.columns:
        raise ValueError(f"Dataset must include {amount_col} and {time_col} columns")

    sample_size = args.user_count * args.transactions_per_user
    if len(df) > sample_size:
        sampled = df.sample(n=sample_size, random_state=args.seed)
    else:
        sampled = df.copy()

    store = SQLiteProfileStore(db_path=args.sqlite_path, ttl_seconds=args.ttl_seconds)
    profiler = BehaviorProfiler(profile_store=store, history_limit=args.history_limit, min_history=5)

    replayed = 0
    for idx, (_, row) in enumerate(sampled.iterrows()):
        user_idx = idx % args.user_count
        user_id = f"replay_user_{user_idx:04d}"
        amount = float(row[amount_col])
        hour = to_hour(float(row[time_col]))
        location_risk = 0.85 if int(row.get("Class", 0)) == 1 else 0.10

        profiler.record_transaction(
            user_id=user_id,
            amount=amount,
            hour_of_day=hour,
            location_risk_score=location_risk,
        )
        replayed += 1

    output = build_output(args, users_written=args.user_count, replayed_rows=replayed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("Behavior profile replay complete")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

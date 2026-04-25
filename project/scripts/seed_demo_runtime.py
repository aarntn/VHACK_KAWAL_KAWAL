from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.app.profile_store import SQLiteProfileStore

SCENARIO_PRESETS_PATH = REPO_ROOT / "project" / "frontend" / "src" / "scenarioPresets.json"
PROFILE_DB_PATH = REPO_ROOT / "project" / "outputs" / "behavior_profiles.sqlite3"
AUDIT_DIR = REPO_ROOT / "project" / "outputs" / "audit"
AUDIT_LOG_PATH = AUDIT_DIR / "fraud_audit_log.jsonl"
REVIEW_QUEUE_PATH = AUDIT_DIR / "review_queue.jsonl"
ANALYST_OUTCOMES_PATH = AUDIT_DIR / "analyst_outcomes.jsonl"
RETRAINING_CURATION_PATH = AUDIT_DIR / "retraining_curation.jsonl"
SUMMARY_PATH = REPO_ROOT / "project" / "outputs" / "monitoring" / "demo_runtime_seed_summary.json"

SUPPORTED_COUNTRIES = {"SG", "MY", "ID", "TH", "PH", "VN"}
BACKGROUND_PREFIX = "demo_bg_user_"
DEFAULT_WALLET_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_FRAUD_BASE_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class DemoStoryPlan:
    story_id: str
    title: str
    description: str
    expected_decision: str
    expected_wallet_action: str
    users: list[str]
    count: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed a believable local demo dataset for the wallet + fraud runtime."
    )
    parser.add_argument("--wallet-base-url", default=DEFAULT_WALLET_BASE_URL)
    parser.add_argument("--fraud-base-url", default=DEFAULT_FRAUD_BASE_URL)
    parser.add_argument("--summary-json", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--profile-db", type=Path, default=PROFILE_DB_PATH)
    parser.add_argument("--background-users", type=int, default=48)
    parser.add_argument("--background-history", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inter-request-delay-ms", type=float, default=25.0)
    parser.add_argument("--reset-runtime", action="store_true")
    parser.add_argument("--resolve-reviews", type=int, default=0)
    parser.add_argument("--operator-api-key", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def empty_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def delete_demo_profiles(db_path: Path, named_users: list[str]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS behavior_profiles (
                user_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                version INTEGER NOT NULL,
                updated_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "DELETE FROM behavior_profiles WHERE user_id LIKE ?",
            (f"{BACKGROUND_PREFIX}%",),
        )
        if named_users:
            conn.executemany(
                "DELETE FROM behavior_profiles WHERE user_id = ?",
                [(user_id,) for user_id in named_users],
            )
        conn.commit()
    finally:
        conn.close()


def build_profile_payload(
    user_id: str,
    history: list[tuple[float, int, float]],
    *,
    counterparties: list[int] | None = None,
    mismatch_flags: list[int] | None = None,
) -> dict[str, Any]:
    now = time.time()
    counterparties = counterparties or [0] * len(history)
    mismatch_flags = mismatch_flags or [0] * len(history)
    return {
        "user_id": user_id,
        "amounts": [round(float(amount), 2) for amount, _, _ in history],
        "hours": [int(hour) for _, hour, _ in history],
        "location_risks": [round(float(location_risk), 4) for _, _, location_risk in history],
        "event_timestamps": [now - (len(history) - idx) * 3600 for idx in range(len(history))],
        "geo_device_mismatch_flags": [int(flag) for flag in mismatch_flags],
        "counterparties_24h": [max(0, int(value)) for value in counterparties],
        "total_transactions": len(history),
        "geo_device_mismatch_count": sum(int(flag) for flag in mismatch_flags),
        "payload_schema_version": 2,
    }


def seed_named_profiles(store: SQLiteProfileStore) -> dict[str, int]:
    profiles = {
        "user_id_qr_local": build_profile_payload(
            "user_id_qr_local",
            [(165000, 8, 0.03), (178000, 9, 0.04), (152000, 12, 0.05), (188000, 13, 0.04), (171000, 19, 0.03), (161000, 20, 0.04)],
        ),
        "user_id_qr_commuter": build_profile_payload(
            "user_id_qr_commuter",
            [(92000, 7, 0.04), (106000, 8, 0.05), (98000, 12, 0.04), (112000, 18, 0.05), (101000, 19, 0.04), (97000, 20, 0.05)],
        ),
        "user_id_qr_market": build_profile_payload(
            "user_id_qr_market",
            [(210000, 9, 0.05), (235000, 11, 0.04), (198000, 13, 0.05), (224000, 17, 0.04), (206000, 19, 0.05), (219000, 20, 0.05)],
        ),
        "user_sgph_new": build_profile_payload(
            "user_sgph_new",
            [(120.0, 18, 0.12), (135.0, 19, 0.13), (118.0, 20, 0.14), (142.0, 21, 0.12), (131.0, 19, 0.13)],
            counterparties=[1, 1, 1, 1, 1],
        ),
        "user_agent_my": build_profile_payload(
            "user_agent_my",
            [(220.0, 10, 0.08), (240.0, 11, 0.09), (215.0, 13, 0.08), (235.0, 15, 0.09), (225.0, 16, 0.10), (210.0, 17, 0.09)],
            counterparties=[2, 2, 3, 2, 3, 2],
        ),
        "user_thvn_block": build_profile_payload(
            "user_thvn_block",
            [(480.0, 21, 0.18), (520.0, 22, 0.20), (505.0, 20, 0.19), (495.0, 21, 0.18), (530.0, 22, 0.20)],
            counterparties=[3, 4, 4, 3, 5],
            mismatch_flags=[1, 1, 1, 1, 1],
        ),
        "account_00119": build_profile_payload(
            "account_00119",
            [(320.0, 20, 0.14), (355.0, 21, 0.16), (340.0, 22, 0.15), (365.0, 21, 0.17), (330.0, 20, 0.16)],
            counterparties=[4, 5, 4, 6, 5],
            mismatch_flags=[1, 1, 1, 1, 1],
        ),
    }

    counts: dict[str, int] = {}
    for user_id, payload in profiles.items():
        store.save_profile(user_id, payload)
        counts[user_id] = len(payload["amounts"])
    return counts


def seed_background_profiles(store: SQLiteProfileStore, *, users: int, history_length: int, seed: int) -> int:
    if users <= 0 or history_length <= 0:
        return 0

    rng = random.Random(seed)
    total_events = 0
    country_currency_pairs = [
        ("ID", "IDR"),
        ("MY", "MYR"),
        ("PH", "PHP"),
        ("VN", "VND"),
        ("TH", "THB"),
        ("SG", "SGD"),
    ]
    bucket_ranges = {
        "IDR": (45000, 240000),
        "MYR": (18, 120),
        "PHP": (90, 420),
        "VND": (65000, 380000),
        "THB": (75, 560),
        "SGD": (8, 90),
    }
    for idx in range(users):
        country, currency = country_currency_pairs[idx % len(country_currency_pairs)]
        low, high = bucket_ranges[currency]
        history: list[tuple[float, int, float]] = []
        counterparties: list[int] = []
        mismatch_flags: list[int] = []
        for _ in range(history_length):
            history.append(
                (
                    round(rng.uniform(low, high), 2),
                    rng.choice([7, 8, 9, 12, 13, 18, 19, 20]),
                    round(rng.uniform(0.02, 0.18), 4),
                )
            )
            counterparties.append(rng.randint(0, 4))
            mismatch_flags.append(0)
        payload = build_profile_payload(
            f"{BACKGROUND_PREFIX}{country.lower()}_{idx:03d}",
            history,
            counterparties=counterparties,
            mismatch_flags=mismatch_flags,
        )
        store.save_profile(payload["user_id"], payload)
        total_events += history_length
    return total_events


def _wallet_payload_from_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": scenario["risk"]["schema_version"],
        "user_id": scenario["ui"]["user_id"],
        "wallet_id": scenario["ui"]["wallet_id"],
        "merchant_name": scenario["ui"]["merchant_name"],
        "transaction_amount": float(scenario["ui"]["amount"]),
        "currency": scenario["ui"]["currency"],
        "device_risk_score": float(scenario["risk"]["device_risk_score"]),
        "ip_risk_score": float(scenario["risk"]["ip_risk_score"]),
        "location_risk_score": float(scenario["risk"]["location_risk_score"]),
        "device_id": scenario["risk"]["device_id"],
        "device_shared_users_24h": int(scenario["risk"]["device_shared_users_24h"]),
        "account_age_days": int(scenario["risk"]["account_age_days"]),
        "sim_change_recent": bool(scenario["risk"]["sim_change_recent"]),
        "tx_type": scenario["ui"]["tx_type"],
        "channel": scenario["risk"]["channel"],
        "cash_flow_velocity_1h": int(scenario["risk"]["cash_flow_velocity_1h"]),
        "p2p_counterparties_24h": int(scenario["risk"]["p2p_counterparties_24h"]),
        "is_cross_border": bool(scenario["ui"]["is_cross_border"]),
        "source_country": scenario["ui"].get("source_country"),
        "destination_country": scenario["ui"].get("destination_country"),
        "is_agent_assisted": bool(scenario["ui"].get("is_agent_assisted", False)),
        "connectivity_mode": scenario["ui"].get("connectivity_mode", "online"),
    }


def build_demo_transactions(scenarios: dict[str, Any]) -> tuple[list[dict[str, Any]], list[DemoStoryPlan]]:
    events: list[dict[str, Any]] = []
    story_plans: list[DemoStoryPlan] = []

    qr_base = _wallet_payload_from_scenario(scenarios["everyday_purchase"])
    qr_variants = [
        ("user_id_qr_local", "wallet_id_qr_local", "Warung Nusantara QR", 185000),
        ("user_id_qr_commuter", "wallet_id_qr_commuter", "Jakarta Transit Tap", 118000),
        ("user_id_qr_market", "wallet_id_qr_market", "Pasar Pagi Grocer", 210000),
        ("user_id_qr_local", "wallet_id_qr_local", "Warung Nusantara QR", 156000),
        ("user_id_qr_commuter", "wallet_id_qr_commuter", "Kopi Pagi QR", 98000),
        ("user_id_qr_market", "wallet_id_qr_market", "Warung Makan Local", 224000),
        ("user_id_qr_local", "wallet_id_qr_local", "Biller ID Utility", 132000),
        ("user_id_qr_market", "wallet_id_qr_market", "Pasar Pagi Grocer", 175000),
    ]
    for idx, (user_id, wallet_id, merchant_name, amount) in enumerate(qr_variants, start=1):
        payload = dict(qr_base)
        payload.update(
            {
                "user_id": user_id,
                "wallet_id": wallet_id,
                "merchant_name": merchant_name,
                "transaction_amount": float(amount),
                "device_id": f"device_id_qr_safe_{1 + ((idx - 1) % 3):02d}",
            }
        )
        events.append(
            {
                "story_id": "id_domestic_qr_approve",
                "story_title": "Domestic QR merchant payments in Indonesia",
                "expected_decision": "APPROVE",
                "expected_wallet_action": "APPROVED",
                "payload": payload,
            }
        )
    story_plans.append(
        DemoStoryPlan(
            story_id="id_domestic_qr_approve",
            title="Domestic QR merchant payments in Indonesia",
            description="Established local users make routine QR merchant payments in IDR with low device/IP/location risk.",
            expected_decision="APPROVE",
            expected_wallet_action="APPROVED",
            users=["user_id_qr_local", "user_id_qr_commuter", "user_id_qr_market"],
            count=len(qr_variants),
        )
    )

    remittance_base = _wallet_payload_from_scenario(scenarios["large_amount"])
    remittance_amounts = [420.0, 460.0, 390.0, 510.0]
    for idx, amount in enumerate(remittance_amounts, start=1):
        payload = dict(remittance_base)
        payload.update(
            {
                "wallet_id": f"wallet_sgph_00{idx}",
                "merchant_name": "Remittance Corridor Demo",
                "transaction_amount": amount,
                "device_id": f"device_sgph_new_0{idx}",
            }
        )
        events.append(
            {
                "story_id": "sg_ph_first_remittance_step_up",
                "story_title": "First-time SG-PH remittance",
                "expected_decision": "FLAG",
                "expected_wallet_action": "PENDING_VERIFICATION",
                "payload": payload,
            }
        )
    story_plans.append(
        DemoStoryPlan(
            story_id="sg_ph_first_remittance_step_up",
            title="First-time SG-PH remittance",
            description="A new Singapore wallet user sends a first remittance to the Philippines and gets progressive friction instead of an outright block.",
            expected_decision="FLAG",
            expected_wallet_action="PENDING_VERIFICATION",
            users=["user_sgph_new"],
            count=len(remittance_amounts),
        )
    )

    cashout_base = _wallet_payload_from_scenario(scenarios["agent_cash_out"])
    cashout_amounts = [780.0, 830.0, 760.0, 910.0]
    for idx, amount in enumerate(cashout_amounts, start=1):
        payload = dict(cashout_base)
        payload.update(
            {
                "wallet_id": f"wallet_agent_my_{idx:02d}",
                "merchant_name": "Agent Tunai MY",
                "transaction_amount": amount,
                "device_id": f"device_agent_cashout_0{idx}",
            }
        )
        events.append(
            {
                "story_id": "my_agent_cash_out_flag",
                "story_title": "Agent-assisted cash-out in Malaysia",
                "expected_decision": "FLAG",
                "expected_wallet_action": "PENDING_VERIFICATION",
                "payload": payload,
            }
        )
    story_plans.append(
        DemoStoryPlan(
            story_id="my_agent_cash_out_flag",
            title="Agent-assisted cash-out in Malaysia",
            description="A local MYR cash-out performed through an agent in intermittent connectivity mode stays reviewable and explainable.",
            expected_decision="FLAG",
            expected_wallet_action="PENDING_VERIFICATION",
            users=["user_agent_my"],
            count=len(cashout_amounts),
        )
    )

    block_base = _wallet_payload_from_scenario(scenarios["cross_border"])
    block_amounts = [28000.0, 32000.0, 29500.0, 34000.0]
    for idx, amount in enumerate(block_amounts, start=1):
        payload = dict(block_base)
        payload.update(
            {
                "wallet_id": f"wallet_thvn_{idx:02d}",
                "merchant_name": "Cross-Border Transfer Demo",
                "transaction_amount": amount,
                "device_id": f"new_device_thvn_99{idx}",
            }
        )
        events.append(
            {
                "story_id": "th_vn_repeated_suspicious_block",
                "story_title": "Repeated TH-VN suspicious transfer pattern",
                "expected_decision": "BLOCK",
                "expected_wallet_action": "DECLINED_FRAUD_RISK",
                "payload": payload,
            }
        )
    story_plans.append(
        DemoStoryPlan(
            story_id="th_vn_repeated_suspicious_block",
            title="Repeated TH-VN suspicious transfer pattern",
            description="High-risk cross-border transfers from a new account trigger hard friction and become clear block examples on the dashboard.",
            expected_decision="BLOCK",
            expected_wallet_action="DECLINED_FRAUD_RISK",
            users=["user_thvn_block"],
            count=len(block_amounts),
        )
    )

    ring_payloads = [
        {
            "schema_version": "ieee_fraud_tx_v1",
            "user_id": "account_00119",
            "wallet_id": "wallet_ring_00119",
            "merchant_name": "Cross-Border Transfer Demo",
            "transaction_amount": 2100.0,
            "currency": "SGD",
            "device_risk_score": 0.78,
            "ip_risk_score": 0.84,
            "location_risk_score": 0.72,
            "device_id": "dev-ring16-0",
            "device_shared_users_24h": 7,
            "account_age_days": 12,
            "sim_change_recent": True,
            "tx_type": "P2P",
            "channel": "WEB",
            "cash_flow_velocity_1h": 11,
            "p2p_counterparties_24h": 9,
            "is_cross_border": True,
            "source_country": "SG",
            "destination_country": "VN",
            "is_agent_assisted": False,
            "connectivity_mode": "online",
        },
        {
            "schema_version": "ieee_fraud_tx_v1",
            "user_id": "account_00119",
            "wallet_id": "wallet_ring_00119",
            "merchant_name": "Cross-Border Transfer Demo",
            "transaction_amount": 1950.0,
            "currency": "SGD",
            "device_risk_score": 0.74,
            "ip_risk_score": 0.81,
            "location_risk_score": 0.70,
            "device_id": "dev-ring17-0",
            "device_shared_users_24h": 6,
            "account_age_days": 12,
            "sim_change_recent": True,
            "tx_type": "P2P",
            "channel": "WEB",
            "cash_flow_velocity_1h": 10,
            "p2p_counterparties_24h": 10,
            "is_cross_border": True,
            "source_country": "SG",
            "destination_country": "VN",
            "is_agent_assisted": False,
            "connectivity_mode": "online",
        },
        {
            "schema_version": "ieee_fraud_tx_v1",
            "user_id": "account_00119",
            "wallet_id": "wallet_ring_00119",
            "merchant_name": "Cross-Border Transfer Demo",
            "transaction_amount": 2250.0,
            "currency": "SGD",
            "device_risk_score": 0.80,
            "ip_risk_score": 0.86,
            "location_risk_score": 0.73,
            "device_id": "dev-ring16-2",
            "device_shared_users_24h": 8,
            "account_age_days": 12,
            "sim_change_recent": True,
            "tx_type": "P2P",
            "channel": "WEB",
            "cash_flow_velocity_1h": 12,
            "p2p_counterparties_24h": 11,
            "is_cross_border": True,
            "source_country": "SG",
            "destination_country": "VN",
            "is_agent_assisted": False,
            "connectivity_mode": "online",
        },
        {
            "schema_version": "ieee_fraud_tx_v1",
            "user_id": "account_00119",
            "wallet_id": "wallet_ring_00119",
            "merchant_name": "Cross-Border Transfer Demo",
            "transaction_amount": 2050.0,
            "currency": "SGD",
            "device_risk_score": 0.77,
            "ip_risk_score": 0.83,
            "location_risk_score": 0.71,
            "device_id": "dev-ring17-1",
            "device_shared_users_24h": 7,
            "account_age_days": 12,
            "sim_change_recent": True,
            "tx_type": "P2P",
            "channel": "WEB",
            "cash_flow_velocity_1h": 10,
            "p2p_counterparties_24h": 10,
            "is_cross_border": True,
            "source_country": "SG",
            "destination_country": "VN",
            "is_agent_assisted": False,
            "connectivity_mode": "online",
        },
    ]
    for payload in ring_payloads:
        events.append(
            {
                "story_id": "known_ring_member_escalation",
                "story_title": "Known ring member with shared-attribute evidence",
                "expected_decision": "BLOCK",
                "expected_wallet_action": "DECLINED_FRAUD_RISK",
                "payload": payload,
            }
        )
    story_plans.append(
        DemoStoryPlan(
            story_id="known_ring_member_escalation",
            title="Known ring member with shared-attribute evidence",
            description="A user already present in the fraud ring artifact exercises the ring match path and shows ring-linked reasoning in live responses.",
            expected_decision="BLOCK",
            expected_wallet_action="DECLINED_FRAUD_RISK",
            users=["account_00119"],
            count=len(ring_payloads),
        )
    )

    return events, story_plans


def ensure_services_ready(wallet_base_url: str, fraud_base_url: str) -> None:
    with httpx.Client(timeout=5.0) as client:
        wallet_health = client.get(f"{wallet_base_url.rstrip('/')}/health")
        wallet_health.raise_for_status()
        fraud_health = client.get(f"{fraud_base_url.rstrip('/')}/health")
        fraud_health.raise_for_status()


def validate_countries(events: list[dict[str, Any]]) -> None:
    for event in events:
        payload = event["payload"]
        for field_name in ("source_country", "destination_country"):
            code = payload.get(field_name)
            if code is not None and code not in SUPPORTED_COUNTRIES:
                raise ValueError(f"Unsupported {field_name}: {code}")


def post_wallet_events(
    *,
    wallet_base_url: str,
    events: list[dict[str, Any]],
    inter_request_delay_ms: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=15.0) as client:
        for index, event in enumerate(events, start=1):
            response = client.post(
                f"{wallet_base_url.rstrip('/')}/wallet/authorize_payment",
                json=event["payload"],
            )
            response.raise_for_status()
            body = response.json()
            result = {
                "ordinal": index,
                "story_id": event["story_id"],
                "story_title": event["story_title"],
                "expected_decision": event["expected_decision"],
                "expected_wallet_action": event["expected_wallet_action"],
                "user_id": event["payload"]["user_id"],
                "wallet_id": event["payload"]["wallet_id"],
                "transaction_amount": event["payload"]["transaction_amount"],
                "currency": event["payload"]["currency"],
                "decision": body.get("fraud_engine_decision") or body.get("decision"),
                "wallet_action": body.get("wallet_action"),
                "runtime_mode": body.get("runtime_mode"),
                "corridor": body.get("corridor"),
                "final_risk_score": body.get("final_risk_score"),
                "reason_codes": body.get("reason_codes", []),
                "request_id": body.get("correlation_id") or body.get("wallet_request_id"),
            }
            results.append(result)
            if inter_request_delay_ms > 0:
                time.sleep(inter_request_delay_ms / 1000.0)
    return results


def maybe_resolve_reviews(
    *,
    fraud_base_url: str,
    operator_api_key: str,
    resolve_reviews: int,
) -> list[dict[str, Any]]:
    if resolve_reviews <= 0 or not operator_api_key:
        return []

    headers = {"X-Operator-Api-Key": operator_api_key}
    resolutions: list[dict[str, Any]] = []
    with httpx.Client(timeout=10.0, headers=headers) as client:
        queue_response = client.get(f"{fraud_base_url.rstrip('/')}/review_queue?status=pending")
        queue_response.raise_for_status()
        pending = queue_response.json()
        for item in pending[:resolve_reviews]:
            decision = str(item.get("decision", "")).upper()
            analyst_decision = "FRAUD" if decision == "BLOCK" else "LEGIT"
            payload = {
                "analyst_id": "demo_analyst_01",
                "analyst_decision": analyst_decision,
                "analyst_confidence": 0.88,
                "analyst_notes": "Demo seeding outcome for operator workflow.",
                "transaction_amount": 0.0,
            }
            response = client.post(
                f"{fraud_base_url.rstrip('/')}/review_queue/{item['request_id']}/outcome",
                json=payload,
            )
            response.raise_for_status()
            resolutions.append(response.json())
    return resolutions


def count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def build_summary(
    *,
    background_history_events: int,
    named_profile_counts: dict[str, int],
    story_plans: list[DemoStoryPlan],
    event_results: list[dict[str, Any]],
    resolved_reviews: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    decisions: dict[str, int] = {}
    wallet_actions: dict[str, int] = {}
    mismatches: list[dict[str, Any]] = []
    for result in event_results:
        decisions[result["decision"]] = decisions.get(result["decision"], 0) + 1
        wallet_actions[result["wallet_action"]] = wallet_actions.get(result["wallet_action"], 0) + 1
        if result["decision"] != result["expected_decision"]:
            mismatches.append(
                {
                    "story_id": result["story_id"],
                    "user_id": result["user_id"],
                    "expected_decision": result["expected_decision"],
                    "actual_decision": result["decision"],
                    "wallet_action": result["wallet_action"],
                }
            )

    return {
        "generated_at_utc": utc_now_iso(),
        "mode": "dry_run" if args.dry_run else "live_seed",
        "wallet_base_url": args.wallet_base_url,
        "fraud_base_url": args.fraud_base_url,
        "profile_seed": {
            "background_users": args.background_users,
            "background_history_events": background_history_events,
            "named_users": sorted(named_profile_counts.keys()),
            "named_history_events": sum(named_profile_counts.values()),
        },
        "story_plan": [
            {
                "story_id": plan.story_id,
                "title": plan.title,
                "description": plan.description,
                "expected_decision": plan.expected_decision,
                "expected_wallet_action": plan.expected_wallet_action,
                "users": plan.users,
                "count": plan.count,
            }
            for plan in story_plans
        ],
        "transaction_batch": {
            "requested_events": sum(plan.count for plan in story_plans),
            "posted_events": len(event_results),
            "decision_counts": decisions,
            "wallet_action_counts": wallet_actions,
            "mismatches": mismatches,
        },
        "review_workflow": {
            "resolved_reviews_requested": args.resolve_reviews,
            "resolved_reviews_completed": len(resolved_reviews),
            "resolved_review_request_ids": [item.get("request_id") for item in resolved_reviews],
            "review_queue_records": count_jsonl_records(REVIEW_QUEUE_PATH),
            "analyst_outcome_records": count_jsonl_records(ANALYST_OUTCOMES_PATH),
            "retraining_curation_records": count_jsonl_records(RETRAINING_CURATION_PATH),
        },
        "local_artifacts": {
            "audit_log": str(AUDIT_LOG_PATH),
            "review_queue": str(REVIEW_QUEUE_PATH),
            "analyst_outcomes": str(ANALYST_OUTCOMES_PATH),
            "retraining_curation": str(RETRAINING_CURATION_PATH),
            "profile_db": str(args.profile_db),
        },
        "operator_note": (
            "If FRAUD_OPERATOR_API_KEY is enabled, enter it in the dashboard Access panel to view /dashboard and /rings analytics."
        ),
    }


def main() -> int:
    args = parse_args()
    scenarios = load_json(SCENARIO_PRESETS_PATH)
    events, story_plans = build_demo_transactions(scenarios)
    validate_countries(events)

    named_users = sorted({user for plan in story_plans for user in plan.users})

    if args.reset_runtime:
        empty_jsonl(AUDIT_LOG_PATH)
        empty_jsonl(REVIEW_QUEUE_PATH)
        empty_jsonl(ANALYST_OUTCOMES_PATH)
        empty_jsonl(RETRAINING_CURATION_PATH)
        delete_demo_profiles(args.profile_db, named_users)

    args.profile_db.parent.mkdir(parents=True, exist_ok=True)
    profile_store = SQLiteProfileStore(args.profile_db, ttl_seconds=365 * 86400)
    try:
        named_profile_counts = seed_named_profiles(profile_store)
        background_history_events = seed_background_profiles(
            profile_store,
            users=args.background_users,
            history_length=args.background_history,
            seed=args.seed,
        )
    finally:
        profile_store.close()

    if args.dry_run:
        event_results: list[dict[str, Any]] = []
        resolved_reviews: list[dict[str, Any]] = []
    else:
        ensure_services_ready(args.wallet_base_url, args.fraud_base_url)
        event_results = post_wallet_events(
            wallet_base_url=args.wallet_base_url,
            events=events,
            inter_request_delay_ms=args.inter_request_delay_ms,
        )
        resolved_reviews = maybe_resolve_reviews(
            fraud_base_url=args.fraud_base_url,
            operator_api_key=args.operator_api_key,
            resolve_reviews=args.resolve_reviews,
        )

    summary = build_summary(
        background_history_events=background_history_events,
        named_profile_counts=named_profile_counts,
        story_plans=story_plans,
        event_results=event_results,
        resolved_reviews=resolved_reviews,
        args=args,
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Demo runtime summary written to: {args.summary_json}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

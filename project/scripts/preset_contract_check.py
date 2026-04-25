import argparse
import json
import sys
from datetime import datetime, timezone
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from project.app import wallet_gateway_api
from project.app.hybrid_fraud_api import app as fraud_app
from project.app.hybrid_fraud_api import approve_threshold, block_threshold


class MockFraudResponse:
    def __init__(self, status_code: int, payload: dict[str, Any], text: str | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


PRESET_DEFINITION_PATH = REPO_ROOT / "project/frontend/src/scenarioPresets.json"


def _load_preset_inputs() -> dict[str, dict[str, Any]]:
    raw_presets: dict[str, dict[str, Any]] = json.loads(PRESET_DEFINITION_PATH.read_text(encoding="utf-8"))
    preset_inputs: dict[str, dict[str, Any]] = {}
    for preset_name, preset_def in raw_presets.items():
        ui = preset_def["ui"]
        risk = preset_def["risk"]
        preset_inputs[preset_name] = {
            "schema_version": risk["schema_version"],
            "user_id": ui["user_id"] or "user_custom_demo",
            "transaction_amount": float(ui["amount"] or 0.0),
            "currency": ui.get("currency"),
            "device_risk_score": float(risk["device_risk_score"]),
            "ip_risk_score": float(risk["ip_risk_score"]),
            "location_risk_score": float(risk["location_risk_score"]),
            "device_id": risk["device_id"],
            "device_shared_users_24h": int(risk["device_shared_users_24h"]),
            "account_age_days": int(risk["account_age_days"]),
            "sim_change_recent": bool(risk["sim_change_recent"]),
            "tx_type": ui["tx_type"],
            "channel": risk["channel"],
            "cash_flow_velocity_1h": int(risk["cash_flow_velocity_1h"]),
            "p2p_counterparties_24h": int(risk["p2p_counterparties_24h"]),
            "is_cross_border": bool(ui["is_cross_border"]),
            "source_country": ui.get("source_country"),
            "destination_country": ui.get("destination_country"),
            "is_agent_assisted": bool(ui.get("is_agent_assisted", False)),
            "connectivity_mode": ui.get("connectivity_mode", "online"),
        }
    return preset_inputs


PRESET_INPUTS = _load_preset_inputs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run preset contract checks across fraud + wallet endpoints and emit demo readiness summary JSON."
        )
    )
    parser.add_argument(
        "--approve-margin",
        type=float,
        default=0.01,
        help="Required distance below approve threshold for everyday_purchase.",
    )
    parser.add_argument(
        "--block-margin",
        type=float,
        default=0.005,
        help="Required distance above block threshold for cross_border unless hard rule evidence is present.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("project/outputs/monitoring/demo_readiness_summary.json"),
        help="Where to write readiness JSON artifact.",
    )
    return parser.parse_args()


def _fraud_post(client: TestClient, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/score_transaction", json=payload)
    if response.status_code != 200:
        raise AssertionError(f"Fraud scoring failed: status={response.status_code} body={response.text}")
    return response.json()


def _wallet_post(wallet_client: TestClient, fraud_client: TestClient, payload: dict[str, Any]) -> dict[str, Any]:
    wallet_gateway_api._reset_circuit_breaker_for_tests()

    def mocked_post(url: str, json: dict[str, Any], headers: dict[str, str] | None = None) -> MockFraudResponse:
        del url, headers
        fraud_response = fraud_client.post("/score_transaction", json=json)
        return MockFraudResponse(fraud_response.status_code, fraud_response.json(), fraud_response.text)

    with patch.object(wallet_gateway_api.UPSTREAM_HTTP, "post", new=AsyncMock(side_effect=mocked_post)):
        response = wallet_client.post(
            "/wallet/authorize_payment",
            json={
                **payload,
                "wallet_id": "wallet_demo_001",
                "merchant_name": "Demo Merchant",
                "currency": payload.get("currency", "SGD"),
            },
        )
    if response.status_code != 200:
        raise AssertionError(f"Wallet authorize failed: status={response.status_code} body={response.text}")
    return response.json()


def _contains_hard_rule_signal(fraud_response: dict[str, Any]) -> bool:
    reasons = fraud_response.get("fraud_reasons") or fraud_response.get("reasons") or []
    joined = " | ".join(str(reason) for reason in reasons)
    return "hard risk rules triggered" in joined.lower()


def _has_reason_code(preset_result: dict[str, Any], reason_code: str) -> bool:
    return reason_code in set(preset_result.get("reason_codes", []))


def run_suite(approve_margin: float, block_margin: float) -> dict[str, Any]:
    fraud_client = TestClient(fraud_app)
    wallet_client = TestClient(wallet_gateway_api.app, raise_server_exceptions=False)

    suite: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "approve_threshold": float(approve_threshold),
            "block_threshold": float(block_threshold),
            "approve_margin": float(approve_margin),
            "block_margin": float(block_margin),
        },
        "presets": {},
        "checks": {},
    }

    run_id = uuid.uuid4().hex[:8]

    for preset_name, payload in PRESET_INPUTS.items():
        payload_for_run = {**payload, "user_id": f"{payload['user_id']}_{run_id}_{preset_name}"}
        fraud_response = _fraud_post(fraud_client, payload_for_run)
        wallet_response = _wallet_post(wallet_client, fraud_client, payload_for_run)

        suite["presets"][preset_name] = {
            "fraud_decision": fraud_response.get("decision"),
            "decision_source": fraud_response.get("decision_source"),
            "final_risk_score": float(fraud_response.get("final_risk_score", 0.0)),
            "fraud_reasons": fraud_response.get("fraud_reasons", []),
            "reason_codes": fraud_response.get("reason_codes", []),
            "runtime_mode": fraud_response.get("runtime_mode"),
            "corridor": fraud_response.get("corridor"),
            "normalized_amount_reference": fraud_response.get("normalized_amount_reference"),
            "normalization_basis": fraud_response.get("normalization_basis"),
            "wallet_action": wallet_response.get("wallet_action"),
            "wallet_decision": wallet_response.get("fraud_engine_decision"),
            "wallet_runtime_mode": wallet_response.get("runtime_mode"),
            "fallback_used": bool(wallet_response.get("fallback_used", False)),
        }

    everyday_score = suite["presets"]["everyday_purchase"]["final_risk_score"]
    cross_score = suite["presets"]["cross_border"]["final_risk_score"]

    checks = {
        "all_presets_include_asean_metadata": all(
            preset["corridor"]
            and preset["normalized_amount_reference"] is not None
            and preset["normalization_basis"]
            for preset in suite["presets"].values()
        ),
        "everyday_id_id_local_approve": (
            suite["presets"]["everyday_purchase"]["fraud_decision"] == "APPROVE"
            and suite["presets"]["everyday_purchase"]["corridor"] == "ID-ID"
            and everyday_score < float(approve_threshold) - approve_margin
        ),
        "sg_ph_step_up_ready": (
            suite["presets"]["large_amount"]["corridor"] == "SG-PH"
            and suite["presets"]["large_amount"]["wallet_decision"] == "FLAG"
            and suite["presets"]["large_amount"]["wallet_action"] == "PENDING_VERIFICATION"
            and _has_reason_code(
                suite["presets"]["large_amount"],
                "ASEAN_FIRST_CROSS_BORDER_REMITTANCE",
            )
        ),
        "my_my_agent_cash_out_review_ready": (
            suite["presets"]["agent_cash_out"]["corridor"] == "MY-MY"
            and suite["presets"]["agent_cash_out"]["wallet_decision"] == "FLAG"
            and suite["presets"]["agent_cash_out"]["wallet_action"] == "PENDING_VERIFICATION"
            and _has_reason_code(
                suite["presets"]["agent_cash_out"],
                "ASEAN_AGENT_ASSISTED_CASH_OUT",
            )
        ),
        "th_vn_block_band_or_rule_hit": (
            suite["presets"]["cross_border"]["fraud_decision"] == "BLOCK"
            and suite["presets"]["cross_border"]["corridor"] == "TH-VN"
            and suite["presets"]["cross_border"]["wallet_action"] == "DECLINED_FRAUD_RISK"
            and (
                cross_score > float(block_threshold) + block_margin
                or _contains_hard_rule_signal(suite["presets"]["cross_border"])
            )
        ),
        "custom_accepted": suite["presets"]["custom"]["fraud_decision"] in {"APPROVE", "FLAG", "BLOCK"},
    }

    suite["checks"] = checks
    suite["ok"] = all(checks.values())
    suite["failing_checks"] = [name for name, ok in checks.items() if not ok]
    return suite


def main() -> int:
    args = parse_args()
    suite = run_suite(approve_margin=args.approve_margin, block_margin=args.block_margin)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(suite, indent=2), encoding="utf-8")

    print(json.dumps(suite, indent=2))

    if not suite["ok"]:
        print("preset_contract_check: FAILED", flush=True)
        return 1
    print("preset_contract_check: PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

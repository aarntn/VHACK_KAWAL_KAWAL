from __future__ import annotations

from typing import Any

PAYLOAD_SCHEMA_VERSION = "ieee_fraud_tx_v1"

# Final serving payload contract: IEEE transaction core + V1..V17 + context fields.
CORE_IDENTIFIER_FIELDS = ["schema_version", "user_id"]
CORE_TRANSACTION_FIELDS = ["TransactionDT", "TransactionAmt"]
IEEE_V_FEATURE_FIELDS = [f"V{i}" for i in range(1, 18)]
CONTEXT_FEATURE_FIELDS = [
    "device_risk_score",
    "ip_risk_score",
    "location_risk_score",
    "device_id",
    "device_shared_users_24h",
    "account_age_days",
    "sim_change_recent",
    "tx_type",
    "channel",
    "cash_flow_velocity_1h",
    "p2p_counterparties_24h",
    "is_cross_border",
]

REQUIRED_SERVING_INPUT_FIELDS = CORE_IDENTIFIER_FIELDS + CORE_TRANSACTION_FIELDS + IEEE_V_FEATURE_FIELDS
OPTIONAL_SERVING_INPUT_FIELDS = CONTEXT_FEATURE_FIELDS
ALL_SERVING_INPUT_FIELDS = REQUIRED_SERVING_INPUT_FIELDS + OPTIONAL_SERVING_INPUT_FIELDS

# Expected raw model feature set for direct/raw inference mode.
EXPECTED_RAW_MODEL_FEATURES = CORE_TRANSACTION_FIELDS + IEEE_V_FEATURE_FIELDS + CONTEXT_FEATURE_FIELDS


def serving_schema_summary() -> dict[str, Any]:
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "required_input_fields": REQUIRED_SERVING_INPUT_FIELDS,
        "optional_input_fields": OPTIONAL_SERVING_INPUT_FIELDS,
        "all_input_fields": ALL_SERVING_INPUT_FIELDS,
        "expected_raw_model_features": EXPECTED_RAW_MODEL_FEATURES,
    }

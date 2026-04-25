from __future__ import annotations

"""
Internal IEEE-style serving contracts.

These models reflect the model-serving feature layout (`TransactionDT`, `TransactionAmt`,
`V1`..`V17`) used inside the fraud engine pipeline.

External API callers should use `project.app.schemas.api.ScoreTransactionRequest`
and `WalletAuthorizeRequest` instead. IEEE-style request payloads are maintained
for internal compatibility and test fixtures, not as the public API contract.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schema_spec import PAYLOAD_SCHEMA_VERSION
MAX_ABS_FEATURE_VALUE = 1_000_000
MAX_AMOUNT = 10_000_000
MIN_AMOUNT = 0.0
MAX_TRANSACTION_DT = 10_000_000


def safe_float_contract(x: Any, field_name: str) -> float:
    try:
        value = float(x)
    except Exception as exc:
        raise ValueError(f"Invalid numeric value for {field_name}: {x}") from exc

    if value != value:
        raise ValueError(f"{field_name} cannot be NaN")

    if value in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} cannot be infinite")

    return value


class FraudTransactionContract(BaseModel):
    # Internal-only model serving contract: not the public API payload shape.
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[PAYLOAD_SCHEMA_VERSION]
    user_id: str = Field(..., min_length=1, max_length=100)

    TransactionDT: float = Field(..., ge=0.0, le=MAX_TRANSACTION_DT)
    TransactionAmt: float = Field(..., ge=MIN_AMOUNT, le=MAX_AMOUNT)
    # IEEE-CIS compatibility fields: always supplied as 0.0 in the current runtime pipeline.
    # Preserved for schema compatibility and test fixtures; not used for inference decisions.
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float

    device_risk_score: float = Field(0.0, ge=0.0, le=1.0)
    ip_risk_score: float = Field(0.0, ge=0.0, le=1.0)
    location_risk_score: float = Field(0.0, ge=0.0, le=1.0)

    device_id: str | None = Field(default=None, max_length=128)
    device_shared_users_24h: int = Field(0, ge=0, le=50)
    account_age_days: int = Field(0, ge=0, le=36500)
    sim_change_recent: bool = False

    tx_type: Literal["P2P", "MERCHANT", "CASH_IN", "CASH_OUT"] = "MERCHANT"
    channel: Literal["APP", "AGENT", "QR", "WEB"] = "APP"
    cash_flow_velocity_1h: int = Field(0, ge=0, le=500)
    p2p_counterparties_24h: int = Field(0, ge=0, le=1000)
    is_cross_border: bool = False
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    source_country: Literal["SG", "MY", "ID", "TH", "PH", "VN"] | None = None
    destination_country: Literal["SG", "MY", "ID", "TH", "PH", "VN"] | None = None
    is_agent_assisted: bool = False
    connectivity_mode: Literal["online", "intermittent", "offline_buffered"] = "online"

    @field_validator(
        "TransactionDT",
        "TransactionAmt",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "device_risk_score",
        "ip_risk_score",
        "location_risk_score",
        mode="before",
    )
    @classmethod
    def validate_numeric(cls, v, info):
        value = safe_float_contract(v, info.field_name)
        if abs(value) > MAX_ABS_FEATURE_VALUE and info.field_name not in {
            "device_risk_score",
            "ip_risk_score",
            "location_risk_score",
        }:
            raise ValueError(f"{info.field_name} is out of allowed numeric range")
        return value

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, value):
        if value is None:
            return None
        text = str(value).strip().upper()
        return text or None

    @field_validator("source_country", "destination_country", mode="before")
    @classmethod
    def normalize_country_code(cls, value):
        if value is None:
            return None
        text = str(value).strip().upper()
        return text or None

    @field_validator("connectivity_mode", mode="before")
    @classmethod
    def normalize_connectivity_mode(cls, value):
        if value is None:
            return "online"
        text = str(value).strip().lower()
        return text or "online"


class WalletPaymentContract(FraudTransactionContract):
    wallet_id: str = Field(..., min_length=1, max_length=100)
    merchant_name: str = Field(..., min_length=1, max_length=150)
    currency: str = Field(..., min_length=3, max_length=3)

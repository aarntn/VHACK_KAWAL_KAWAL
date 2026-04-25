from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, AliasChoices, field_validator

from project.app.errors import ApiErrorResponse
from project.app.schema_spec import PAYLOAD_SCHEMA_VERSION

DecisionSource = Literal["score_band", "hard_rule_override", "low_history_policy", "step_up_policy"]
AseanCountryCode = Literal["SG", "MY", "ID", "TH", "PH", "VN"]
ConnectivityMode = Literal["online", "intermittent", "offline_buffered"]
RuntimeMode = Literal["primary", "cached_context", "degraded_local"]


class _RiskValidationMixin(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_version: Literal[PAYLOAD_SCHEMA_VERSION]
    user_id: str = Field(..., min_length=1, max_length=100)
    transaction_amount: float = Field(
        ...,
        ge=0.0,
        le=10_000_000,
        validation_alias=AliasChoices("transaction_amount", "TransactionAmt"),
        serialization_alias="transaction_amount",
        description="Transaction amount. Must be between 0.0 and 10,000,000.",
    )
    device_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Device risk score must satisfy 0.0 <= device_risk_score <= 1.0",
    )
    ip_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="IP risk score must satisfy 0.0 <= ip_risk_score <= 1.0",
    )
    location_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Location risk score must satisfy 0.0 <= location_risk_score <= 1.0",
    )

    # Optional advanced context (UX spec).
    device_id: str | None = Field(default=None, max_length=128)
    device_shared_users_24h: int = Field(default=0, ge=0, le=50)
    account_age_days: int = Field(default=0, ge=0, le=36500)
    sim_change_recent: bool = False
    tx_type: Literal["P2P", "MERCHANT", "CASH_IN", "CASH_OUT"] = "MERCHANT"
    channel: Literal["APP", "AGENT", "QR", "WEB"] = "APP"
    cash_flow_velocity_1h: int = Field(default=0, ge=0, le=500)
    p2p_counterparties_24h: int = Field(default=0, ge=0, le=1000)
    is_cross_border: bool = False
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    source_country: AseanCountryCode | None = None
    destination_country: AseanCountryCode | None = None
    is_agent_assisted: bool = False
    connectivity_mode: ConnectivityMode = "online"
    override_source: Literal["preset", "support_manual"] = "preset"
    support_mode: bool = False
    support_actor_id: str | None = Field(default=None, max_length=100)
    override_fields: List[str] = Field(default_factory=list, max_length=32)

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, value: Any) -> Any:
        if value is None:
            return None
        return str(value).strip().upper()

    @field_validator("source_country", "destination_country", mode="before")
    @classmethod
    def normalize_country_code(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip().upper()
        return text or None

    @field_validator("connectivity_mode", mode="before")
    @classmethod
    def normalize_connectivity_mode(cls, value: Any) -> Any:
        if value is None:
            return "online"
        text = str(value).strip().lower()
        return text or "online"


class ScoreTransactionRequest(_RiskValidationMixin):
    pass


class WalletAuthorizeRequest(_RiskValidationMixin):
    wallet_id: str = Field(..., min_length=1, max_length=100)
    merchant_name: str = Field(..., min_length=1, max_length=150)
    currency: str = Field(..., min_length=3, max_length=3)


class TopFeatureDriver(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_risk" | "reduces_risk"


class ExplainabilityBreakdown(BaseModel):
    base: float
    context: float
    behavior: float
    ring: float = 0.0
    ring_reason_codes: List[str] = Field(default_factory=list)
    external: float = 0.0
    top_feature_drivers: List[TopFeatureDriver] = Field(default_factory=list)


class StageTimingsMs(BaseModel):
    total_pipeline_ms: float | None = None
    details: Dict[str, float] = Field(default_factory=dict)


class ScoreTransactionResponse(BaseModel):
    request_id: str
    correlation_id: str
    risk_score: float
    final_risk_score: float
    decision: Literal["APPROVE", "FLAG", "BLOCK"]
    decision_source: DecisionSource
    runtime_mode: RuntimeMode = "primary"
    fraud_reasons: List[str]
    reasons: List[str]
    reason_codes: List[str] = Field(default_factory=list)
    ring_match_type: Literal["account_member", "attribute_match", "none"] = "none"
    ring_evidence_summary: Dict[str, Any] = Field(default_factory=dict)
    corridor: str | None = None
    normalized_amount_reference: float | None = None
    normalization_basis: str | None = None
    explainability: ExplainabilityBreakdown
    context_summary: Dict[str, Any] = Field(default_factory=dict)
    stage_timings_ms: StageTimingsMs | None = None


class WalletAuthorizeResponse(BaseModel):
    wallet_request_id: str
    correlation_id: str
    timestamp_utc: str
    wallet_action: Literal["APPROVED", "PENDING_VERIFICATION", "DECLINED_FRAUD_RISK"]
    wallet_message: str
    fraud_engine_decision: Literal["APPROVE", "FLAG", "BLOCK"]
    risk_score: float
    final_risk_score: float
    decision: Literal["APPROVE", "FLAG", "BLOCK"]
    decision_source: DecisionSource
    runtime_mode: RuntimeMode = "primary"
    fraud_reasons: List[str]
    reason_codes: List[str] = Field(default_factory=list)
    next_step: str
    corridor: str | None = None
    normalized_amount_reference: float | None = None
    normalization_basis: str | None = None
    explainability: ExplainabilityBreakdown | None = None
    upstream_attempts: int
    circuit_breaker_state: Literal["CLOSED", "OPEN", "HALF_OPEN"]
    fallback_used: bool
    upstream_error: str | None = None
    stage_timings_ms: StageTimingsMs | None = None


__all__ = [
    "ApiErrorResponse",
    "DecisionSource",
    "ScoreTransactionRequest",
    "ScoreTransactionResponse",
    "WalletAuthorizeRequest",
    "WalletAuthorizeResponse",
]

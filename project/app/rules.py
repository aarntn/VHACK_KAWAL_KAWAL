from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal

from .domain_exceptions import DomainValidationError, UnknownChannelError, UnknownTransactionTypeError


HardRuleAction = Literal["ALLOW", "FLAG", "BLOCK"]
StepUpAction = Literal["NONE", "STEP_UP_OTP", "STEP_UP_KYC", "STEP_UP_MANUAL_REVIEW"]


@dataclass
class SegmentThresholds:
    approve_threshold: float
    block_threshold: float
    min_block_precision: float = 0.0
    max_approve_to_flag_fpr: float = 1.0
    calibration_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class RuleEvaluation:
    action: HardRuleAction
    rule_hits: List[str]


@dataclass
class StepUpDecision:
    verification_action: StepUpAction
    reason: str | None


@dataclass(frozen=True)
class HardBlockThresholdPolicy:
    amount_vs_user_median_multiplier: float


@dataclass(frozen=True)
class RiskGatePolicy:
    ip_risk_score_gte: float
    device_risk_score_gte: float
    location_risk_score_gte: float


@dataclass(frozen=True)
class PrecedencePolicy:
    order: List[str]
    conflict_resolution: str


@dataclass(frozen=True)
class RulePolicy:
    version: str
    hard_block_thresholds: HardBlockThresholdPolicy
    step_up_auth_triggers: RiskGatePolicy
    precedence: PrecedencePolicy


def _safe_positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DomainValidationError(f"{label} must be a numeric value", field=label) from exc
    if parsed <= 0:
        raise DomainValidationError(f"{label} must be > 0", field=label)
    return parsed


def _safe_probability(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DomainValidationError(f"{label} must be a numeric value", field=label) from exc
    if parsed < 0:
        raise DomainValidationError(f"{label} must be >= 0", field=label)
    if parsed > 1:
        raise DomainValidationError(f"{label} must be <= 1", field=label)
    return parsed


def load_rule_policy(config_path: Path | None = None) -> RulePolicy:
    default_path = Path(__file__).resolve().parents[1] / "data" / "rule_policy.v1.json"
    path = config_path or default_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DomainValidationError("Rule policy payload must be a JSON object")

    version = str(payload.get("version", "")).strip()
    if not version:
        raise DomainValidationError("Rule policy must include a non-empty version", field="version")

    hard_block_raw = payload.get("hard_block_thresholds")
    if not isinstance(hard_block_raw, dict):
        raise DomainValidationError("hard_block_thresholds must be an object", field="hard_block_thresholds")
    hard_block_policy = HardBlockThresholdPolicy(
        amount_vs_user_median_multiplier=_safe_positive_float(
            hard_block_raw.get("amount_vs_user_median_multiplier"),
            "hard_block_thresholds.amount_vs_user_median_multiplier",
        )
    )

    triggers_raw = payload.get("step_up_auth_triggers")
    if not isinstance(triggers_raw, dict):
        raise DomainValidationError("step_up_auth_triggers must be an object", field="step_up_auth_triggers")
    risk_gate_policy = RiskGatePolicy(
        ip_risk_score_gte=_safe_probability(triggers_raw.get("ip_risk_score_gte"), "step_up_auth_triggers.ip_risk_score_gte"),
        device_risk_score_gte=_safe_probability(
            triggers_raw.get("device_risk_score_gte"), "step_up_auth_triggers.device_risk_score_gte"
        ),
        location_risk_score_gte=_safe_probability(
            triggers_raw.get("location_risk_score_gte"), "step_up_auth_triggers.location_risk_score_gte"
        ),
    )

    precedence_raw = payload.get("precedence")
    if not isinstance(precedence_raw, dict):
        raise DomainValidationError("precedence must be an object", field="precedence")
    order = precedence_raw.get("order")
    if not isinstance(order, list) or not order or not all(isinstance(item, str) and item for item in order):
        raise DomainValidationError("precedence.order must be a non-empty string list", field="precedence.order")
    conflict_resolution = str(precedence_raw.get("conflict_resolution", "")).strip().lower()
    if conflict_resolution not in {"most_restrictive_wins"}:
        raise DomainValidationError("precedence.conflict_resolution must be 'most_restrictive_wins'", field="precedence.conflict_resolution")
    precedence_policy = PrecedencePolicy(order=order, conflict_resolution=conflict_resolution)

    return RulePolicy(
        version=version,
        hard_block_thresholds=hard_block_policy,
        step_up_auth_triggers=risk_gate_policy,
        precedence=precedence_policy,
    )


RULE_POLICY = load_rule_policy()


class RuleStateStore:
    """In-memory state used by high-signal hard rules."""

    def __init__(self) -> None:
        self._last_geo_by_user: Dict[str, str] = {}
        self._last_tx_dt_by_user: Dict[str, float] = {}
        self._last_country_switch_dt_by_user: Dict[str, float] = {}
        self._burst_window_start_by_user: Dict[str, float] = {}
        self._burst_count_by_user: Dict[str, int] = {}

    def get_last_geo(self, user_id: str) -> str | None:
        return self._last_geo_by_user.get(user_id)

    def get_last_tx_dt(self, user_id: str) -> float | None:
        return self._last_tx_dt_by_user.get(user_id)

    def get_last_country_switch_dt(self, user_id: str) -> float | None:
        return self._last_country_switch_dt_by_user.get(user_id)

    def get_burst_window(self, user_id: str) -> tuple[float | None, int]:
        return self._burst_window_start_by_user.get(user_id), self._burst_count_by_user.get(user_id, 0)

    def set_burst_window(self, user_id: str, window_start: float, window_count: int) -> None:
        self._burst_window_start_by_user[user_id] = window_start
        self._burst_count_by_user[user_id] = window_count

    def update_after_evaluation(self, user_id: str, tx_dt: float, geo_bucket: str, switched_country: bool) -> None:
        self._last_geo_by_user[user_id] = geo_bucket
        self._last_tx_dt_by_user[user_id] = tx_dt
        if switched_country:
            self._last_country_switch_dt_by_user[user_id] = tx_dt


def infer_geo_bucket(tx: Dict[str, Any]) -> str:
    if bool(tx.get("is_cross_border", False)):
        return "CROSS_BORDER"
    return "DOMESTIC"


def compute_user_segment(tx: Dict[str, Any]) -> str:
    tx_type = str(tx.get("tx_type", "MERCHANT")).strip().upper() or "MERCHANT"
    allowed_types = {"P2P", "MERCHANT", "CASH_IN", "CASH_OUT"}
    if tx_type not in allowed_types:
        raise UnknownTransactionTypeError(f"Unsupported tx_type '{tx_type}'")

    newness = "new" if int(tx.get("account_age_days", 0)) < 30 else "established"
    amount_band = "high_ticket" if float(tx.get("TransactionAmt", 0.0)) >= 1000 else "low_ticket"
    channel = str(tx.get("channel", "APP")).strip().upper() or "APP"
    allowed_channels = {"APP", "AGENT", "QR", "WEB"}
    if channel not in allowed_channels:
        raise UnknownChannelError(f"Unsupported channel '{channel}'", field="channel")
    return f"{newness}:{amount_band}:{channel}"


def resolve_thresholds_for_segment(
    segment: str,
    base_approve: float,
    base_block: float,
    segment_thresholds: Dict[str, SegmentThresholds],
) -> SegmentThresholds:
    thresholds = segment_thresholds.get(segment)
    if thresholds is None:
        return SegmentThresholds(approve_threshold=base_approve, block_threshold=base_block)
    return thresholds


def validate_segment_thresholds(thresholds: SegmentThresholds, segment_name: str) -> None:
    if not (0.0 <= thresholds.approve_threshold < thresholds.block_threshold <= 1.0):
        raise DomainValidationError(
            (
                f"segment_thresholds.{segment_name} has invalid threshold ordering: "
                f"approve={thresholds.approve_threshold}, block={thresholds.block_threshold}"
            ),
            field=f"segment_thresholds.{segment_name}",
        )

    if not (0.0 <= thresholds.min_block_precision <= 1.0):
        raise DomainValidationError(
            f"segment_thresholds.{segment_name}.min_block_precision must be in [0, 1]",
            field=f"segment_thresholds.{segment_name}.min_block_precision",
        )
    if not (0.0 <= thresholds.max_approve_to_flag_fpr <= 1.0):
        raise DomainValidationError(
            f"segment_thresholds.{segment_name}.max_approve_to_flag_fpr must be in [0, 1]",
            field=f"segment_thresholds.{segment_name}.max_approve_to_flag_fpr",
        )

    block_precision = thresholds.calibration_metrics.get("block_precision")
    if block_precision is not None and block_precision < thresholds.min_block_precision:
        raise DomainValidationError(
            (
                f"segment_thresholds.{segment_name} violates min_block_precision gate: "
                f"metric={block_precision}, minimum={thresholds.min_block_precision}"
            ),
            field=f"segment_thresholds.{segment_name}.calibration_metrics.block_precision",
        )

    approve_to_flag_fpr = thresholds.calibration_metrics.get("approve_to_flag_fpr")
    if approve_to_flag_fpr is not None and approve_to_flag_fpr > thresholds.max_approve_to_flag_fpr:
        raise DomainValidationError(
            (
                f"segment_thresholds.{segment_name} violates max_approve_to_flag_fpr gate: "
                f"metric={approve_to_flag_fpr}, maximum={thresholds.max_approve_to_flag_fpr}"
            ),
            field=f"segment_thresholds.{segment_name}.calibration_metrics.approve_to_flag_fpr",
        )


def evaluate_hard_rules(tx: Dict[str, Any], store: RuleStateStore, policy: RulePolicy = RULE_POLICY) -> RuleEvaluation:
    user_id = str(tx["user_id"])
    tx_dt = float(tx.get("TransactionDT", 0.0))
    geo_bucket = infer_geo_bucket(tx)
    rule_hits: List[str] = []

    # Rule 1: geo impossible travel proxy
    # If geo bucket switches too quickly with elevated location risk, treat as impossible travel.
    last_geo = store.get_last_geo(user_id)
    last_tx_dt = store.get_last_tx_dt(user_id)
    switched_country = bool(last_geo and last_geo != geo_bucket)
    if switched_country and last_tx_dt is not None:
        delta = abs(tx_dt - last_tx_dt)
        if delta <= 6 and float(tx.get("location_risk_score", 0.0)) >= 0.75:
            rule_hits.append("geo_impossible_travel")

    # Rule 2: rapid device-country switches
    # Consecutive country bucket switches in a short interval are high-signal account-takeover behavior.
    if switched_country:
        last_switch_dt = store.get_last_country_switch_dt(user_id)
        if last_switch_dt is not None and abs(tx_dt - last_switch_dt) <= 24:
            rule_hits.append("rapid_device_country_switch")

    # Rule 3: abuse bursts
    # High transaction velocity + shared device pressure in same time window.
    window_start, window_count = store.get_burst_window(user_id)
    if window_start is None or abs(tx_dt - window_start) > 1:
        window_start = tx_dt
        window_count = 1
    else:
        window_count += 1

    store.set_burst_window(user_id=user_id, window_start=window_start, window_count=window_count)

    if (
        window_count >= 5
        and int(tx.get("device_shared_users_24h", 0)) >= 3
        and int(tx.get("cash_flow_velocity_1h", 0)) >= 8
    ):
        rule_hits.append("abuse_burst")

    # Rule 4: abnormal amount spike vs customer baseline.
    user_median_amount_30d = float(tx.get("user_median_amount_30d", 0.0))
    tx_amount = float(tx.get("TransactionAmt", 0.0))
    if user_median_amount_30d > 0:
        amount_multiplier = tx_amount / user_median_amount_30d
        if amount_multiplier >= policy.hard_block_thresholds.amount_vs_user_median_multiplier:
            rule_hits.append("amount_over_user_median_multiplier")

    store.update_after_evaluation(user_id=user_id, tx_dt=tx_dt, geo_bucket=geo_bucket, switched_country=switched_country)

    hard_block_rules = {"geo_impossible_travel", "abuse_burst", "amount_over_user_median_multiplier"}
    if any(rule in hard_block_rules for rule in rule_hits):
        return RuleEvaluation(action="BLOCK", rule_hits=rule_hits)
    if "rapid_device_country_switch" in rule_hits:
        return RuleEvaluation(action="FLAG", rule_hits=rule_hits)
    return RuleEvaluation(action="ALLOW", rule_hits=rule_hits)


def apply_segmented_decision(final_score: float, thresholds: SegmentThresholds) -> Literal["APPROVE", "FLAG", "BLOCK"]:
    if final_score < thresholds.approve_threshold:
        return "APPROVE"
    if final_score < thresholds.block_threshold:
        return "FLAG"
    return "BLOCK"


def determine_step_up_action(
    final_score: float,
    decision: Literal["APPROVE", "FLAG", "BLOCK"],
    segment_thresholds: SegmentThresholds,
    rule_hits: List[str],
    tx: Dict[str, Any] | None = None,
    policy: RulePolicy = RULE_POLICY,
) -> StepUpDecision:
    context = tx or {}
    ip_risk_score = float(context.get("ip_risk_score", 0.0))
    device_risk_score = float(context.get("device_risk_score", 0.0))
    location_risk_score = float(context.get("location_risk_score", 0.0))

    hard_block_rules = {"geo_impossible_travel", "abuse_burst", "amount_over_user_median_multiplier"}
    if decision == "BLOCK":
        if rule_hits:
            hard_rule_hit = next((rule_id for rule_id in rule_hits if rule_id in hard_block_rules), rule_hits[0])
            return StepUpDecision(
                verification_action="STEP_UP_MANUAL_REVIEW",
                reason=hard_rule_hit,
            )
        if final_score >= min(0.99, segment_thresholds.block_threshold + 0.05):
            return StepUpDecision(verification_action="NONE", reason=None)
        return StepUpDecision(
            verification_action="STEP_UP_KYC",
            reason="model_block",
        )

    if device_risk_score >= policy.step_up_auth_triggers.device_risk_score_gte:
        return StepUpDecision(
            verification_action="STEP_UP_KYC",
            reason="device_risk_gate",
        )
    if location_risk_score >= policy.step_up_auth_triggers.location_risk_score_gte:
        return StepUpDecision(
            verification_action="STEP_UP_KYC",
            reason="location_risk_gate",
        )
    if ip_risk_score >= policy.step_up_auth_triggers.ip_risk_score_gte:
        return StepUpDecision(
            verification_action="STEP_UP_OTP",
            reason="ip_risk_gate",
        )

    if decision == "FLAG":
        near_block = final_score >= max(segment_thresholds.approve_threshold, segment_thresholds.block_threshold - 0.05)
        if near_block:
            return StepUpDecision(
                verification_action="STEP_UP_KYC",
                reason="score_near_block_threshold",
            )
        return StepUpDecision(
            verification_action="STEP_UP_OTP",
            reason="score_flag_band",
        )

    if decision == "APPROVE" and final_score >= max(0.0, segment_thresholds.approve_threshold - 0.02):
        return StepUpDecision(
            verification_action="STEP_UP_OTP",
            reason="score_near_approve_threshold",
        )

    return StepUpDecision(verification_action="NONE", reason=None)

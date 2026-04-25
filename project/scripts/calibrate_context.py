import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from project.data.dataset_loader import load_creditcard, load_ieee_cis
from project.data.entity_aggregation import apply_entity_smoothing_batch, validate_smoothing_method
from project.data.entity_identity import build_entity_id
from project.data.preprocessing import load_preprocessing_bundle, prepare_preprocessing_inputs, transform_with_bundle
from artifact_compatibility import collect_artifact_runtime_metadata

DEFAULT_MODEL_PATH = REPO_ROOT / "project" / "models" / "final_xgboost_model.pkl"
DEFAULT_FEATURE_PATH = REPO_ROOT / "project" / "models" / "feature_columns.pkl"
DEFAULT_DATASET_PATH = REPO_ROOT / "project" / "legacy_creditcard" / "creditcard.csv"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "project" / "outputs" / "calibration" / "context_calibration.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "project" / "outputs" / "calibration" / "context_calibration_trials.csv"
DEFAULT_PR_CURVE_CALIBRATION_REPORT = REPO_ROOT / "project" / "outputs" / "calibration" / "pr_curve_calibration_report.json"
DEFAULT_THRESHOLDS_OUTPUT = REPO_ROOT / "project" / "models" / "decision_thresholds.pkl"
DEFAULT_BASELINE_THRESHOLDS_PATH = REPO_ROOT / "project" / "models" / "decision_thresholds.pkl"
DEFAULT_BASELINE_CALIBRATION_JSON = REPO_ROOT / "project" / "outputs" / "calibration" / "context_calibration.json"
DEFAULT_PREPROCESSING_ARTIFACT_PATH = REPO_ROOT / "project" / "models" / "preprocessing_artifact_promoted.pkl"

CONTEXT_WEIGHT_DEFAULTS: Dict[str, float] = {
    "device_risk_weight": 0.05,
    "ip_risk_weight": 0.05,
    "location_risk_weight": 0.05,
    "amount_over_200": 0.02,
    "amount_over_1000": 0.04,
    "early_time_weight": 0.01,
    "shared_device_ge_3": 0.03,
    "shared_device_ge_5": 0.07,
    "sim_change_weight": 0.05,
    "new_account_lt_7d": 0.03,
    "new_account_lt_3d": 0.06,
    "cashout_over_300": 0.04,
    "agent_high_risk_weight": 0.03,
    "flow_velocity_ge_8": 0.05,
    "p2p_counterparties_ge_12": 0.04,
    "cross_border_weight": 0.03,
}

DEFAULT_CONTEXT_VALUES = {
    "device_risk_score": 0.0,
    "ip_risk_score": 0.0,
    "location_risk_score": 0.0,
    "device_shared_users_24h": 0,
    "account_age_days": 30,
    "sim_change_recent": 0,
    "tx_type": "MERCHANT",
    "channel": "APP",
    "cash_flow_velocity_1h": 0,
    "p2p_counterparties_24h": 0,
    "is_cross_border": 0,
}


@dataclass
class CandidateResult:
    objective: float
    precision: float
    recall: float
    f1: float
    fpr: float
    approve_threshold: float
    block_threshold: float
    cap: float
    weights: Dict[str, float]
    ring_weight: float
    ring_cap: float
    fairness_max_fpr_gap: float
    fairness_max_recall_gap: float
    policy_pass: bool
    fairness_pass: bool


@dataclass
class RuntimeConfig:
    approve_threshold: float
    block_threshold: float
    cap: float
    weights: Dict[str, float]


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def detect_feature_columns_shape(columns: list[str]) -> str:
    if not columns:
        return "unknown"
    as_text = [str(col) for col in columns]
    has_preprocessed = any(name.startswith("numeric_canonical__") or name.startswith("categorical_passthrough__") for name in as_text)
    has_raw = any(name in {"Time", "Amount", "TransactionDT", "TransactionAmt"} for name in as_text)
    if has_preprocessed and not has_raw:
        return "preprocessed"
    if has_raw and not has_preprocessed:
        return "raw"
    if has_preprocessed and has_raw:
        return "mixed"
    return "unknown"


def build_model_input(df: pd.DataFrame, feature_columns: list[str], args: argparse.Namespace) -> pd.DataFrame:
    shape = detect_feature_columns_shape(feature_columns)
    if shape in {"raw", "unknown"}:
        return df[feature_columns]

    bundle = load_preprocessing_bundle(args.preprocessing_artifact_path)
    canonical_df, passthrough_df, _ = prepare_preprocessing_inputs(df, args.dataset_source)
    transformed = transform_with_bundle(bundle, canonical_df, passthrough_df)
    transformed_df = pd.DataFrame(transformed, columns=bundle.feature_names_out, index=df.index)
    missing = [col for col in feature_columns if col not in transformed_df.columns]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Preprocessed feature columns missing from transformed calibration input: {preview}")
    return transformed_df.loc[:, feature_columns]


def validate_threshold_order(approve_threshold: float, block_threshold: float) -> None:
    if not (0.0 <= approve_threshold <= 1.0 and 0.0 <= block_threshold <= 1.0):
        raise ValueError("Thresholds must be within [0, 1].")
    if not approve_threshold < block_threshold:
        raise ValueError(
            f"Threshold ordering violation: approve_threshold={approve_threshold} must be < block_threshold={block_threshold}."
        )


def ensure_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, default in DEFAULT_CONTEXT_VALUES.items():
        if col not in out.columns:
            out[col] = default
    return out


def build_context_component_matrix(df: pd.DataFrame, dataset_source: str = "creditcard") -> Dict[str, np.ndarray]:
    device_risk = pd.to_numeric(df["device_risk_score"], errors="coerce").fillna(0.0).to_numpy(float)
    ip_risk = pd.to_numeric(df["ip_risk_score"], errors="coerce").fillna(0.0).to_numpy(float)
    location_risk = pd.to_numeric(df["location_risk_score"], errors="coerce").fillna(0.0).to_numpy(float)

    amount_col = "TransactionAmt" if dataset_source == "ieee_cis" else "Amount"
    time_name = "TransactionDT" if dataset_source == "ieee_cis" else "Time"
    amount = pd.to_numeric(df.get(amount_col, 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    time_col = pd.to_numeric(df.get(time_name, 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    shared_users = pd.to_numeric(df["device_shared_users_24h"], errors="coerce").fillna(0).to_numpy(int)
    age_days = pd.to_numeric(df["account_age_days"], errors="coerce").fillna(30).to_numpy(int)
    flow_velocity = pd.to_numeric(df["cash_flow_velocity_1h"], errors="coerce").fillna(0).to_numpy(int)
    p2p_count = pd.to_numeric(df["p2p_counterparties_24h"], errors="coerce").fillna(0).to_numpy(int)

    sim_change = df["sim_change_recent"].astype(str).str.lower().isin(["1", "true", "yes"]).to_numpy(float)
    cross_border = df["is_cross_border"].astype(str).str.lower().isin(["1", "true", "yes"]).to_numpy(float)

    tx_type = df["tx_type"].astype(str).str.upper()
    channel = df["channel"].astype(str).str.upper()

    return {
        "device_risk_weight": device_risk,
        "ip_risk_weight": ip_risk,
        "location_risk_weight": location_risk,
        "amount_over_200": (amount > 200).astype(float),
        "amount_over_1000": (amount > 1000).astype(float),
        "early_time_weight": (time_col < 1000).astype(float),
        "shared_device_ge_3": (shared_users >= 3).astype(float),
        "shared_device_ge_5": (shared_users >= 5).astype(float),
        "sim_change_weight": sim_change,
        "new_account_lt_7d": (age_days < 7).astype(float),
        "new_account_lt_3d": (age_days < 3).astype(float),
        "cashout_over_300": ((tx_type == "CASH_OUT") & (amount > 300)).astype(float),
        "agent_high_risk_weight": ((channel == "AGENT") & ((device_risk >= 0.5) | (ip_risk >= 0.5))).astype(float),
        "flow_velocity_ge_8": (flow_velocity >= 8).astype(float),
        "p2p_counterparties_ge_12": (p2p_count >= 12).astype(float),
        "cross_border_weight": cross_border,
    }


def compute_context_adjustment(components: Dict[str, np.ndarray], weights: Dict[str, float], cap: float) -> np.ndarray:
    context = np.zeros_like(next(iter(components.values())), dtype=float)
    for key, signal in components.items():
        context += float(weights[key]) * signal
    context = np.minimum(context, float(cap))
    return clamp01(context)


def evaluate_runtime_metrics(scores: np.ndarray, labels: np.ndarray, block_threshold: float) -> Dict[str, float]:
    pred_fraud = scores >= block_threshold
    precision = precision_score(labels, pred_fraud, zero_division=0)
    recall = recall_score(labels, pred_fraud, zero_division=0)
    f1 = f1_score(labels, pred_fraud, zero_division=0)
    tn, fp, _, _ = confusion_matrix(labels, pred_fraud, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def classify_decisions(scores: np.ndarray, approve_threshold: float, block_threshold: float) -> np.ndarray:
    return np.where(scores <= approve_threshold, "approve", np.where(scores >= block_threshold, "block", "flag"))


def compute_decision_drift(
    pre_scores: np.ndarray,
    post_scores: np.ndarray,
    approve_threshold: float,
    block_threshold: float,
) -> Dict[str, float]:
    pre = classify_decisions(pre_scores, approve_threshold, block_threshold)
    post = classify_decisions(post_scores, approve_threshold, block_threshold)

    labels = ["approve", "flag", "block"]
    drift: Dict[str, float] = {}
    total_abs_delta = 0.0
    for label in labels:
        pre_rate = float(np.mean(pre == label))
        post_rate = float(np.mean(post == label))
        delta = post_rate - pre_rate
        drift[f"pre_{label}_rate"] = pre_rate
        drift[f"post_{label}_rate"] = post_rate
        drift[f"delta_{label}_rate"] = delta
        total_abs_delta += abs(delta)
    drift["total_absolute_rate_shift"] = float(total_abs_delta)
    return drift


def _clamp_quantile(q: float) -> float:
    return float(np.clip(float(q), 0.0, 1.0))


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, _clamp_quantile(q)))


def build_raw_to_hybrid_threshold_mapping(
    raw_scores: np.ndarray,
    hybrid_scores: np.ndarray,
    quantiles: tuple[float, ...] = (0.50, 0.75, 0.90),
) -> dict:
    raw = np.asarray(raw_scores, dtype=float)
    hybrid = np.asarray(hybrid_scores, dtype=float)
    if raw.shape[0] != hybrid.shape[0]:
        raise ValueError("raw_scores and hybrid_scores must have matching lengths")

    quantile_rows = []
    for q in quantiles:
        raw_q = _quantile(raw, q)
        hybrid_q = _quantile(hybrid, q)
        quantile_rows.append(
            {
                "quantile": float(q),
                "raw_threshold": raw_q,
                "hybrid_threshold": hybrid_q,
                "shift": float(hybrid_q - raw_q),
            }
        )

    def map_threshold(raw_threshold: float) -> dict:
        quantile_rank = _clamp_quantile(float(np.mean(raw <= float(raw_threshold))))
        mapped_hybrid = _quantile(hybrid, quantile_rank)
        return {
            "raw_threshold": float(raw_threshold),
            "quantile_rank": quantile_rank,
            "hybrid_threshold": mapped_hybrid,
            "shift": float(mapped_hybrid - float(raw_threshold)),
        }

    p50 = next((row for row in quantile_rows if abs(row["quantile"] - 0.50) < 1e-9), quantile_rows[0] if quantile_rows else {})
    p75 = next((row for row in quantile_rows if abs(row["quantile"] - 0.75) < 1e-9), quantile_rows[-1] if quantile_rows else {})

    return {
        "quantile_threshold_mapping": quantile_rows,
        "distribution_shift_summary": {
            "raw_mean": float(np.mean(raw)) if raw.size else 0.0,
            "hybrid_mean": float(np.mean(hybrid)) if hybrid.size else 0.0,
            "raw_std": float(np.std(raw)) if raw.size else 0.0,
            "hybrid_std": float(np.std(hybrid)) if hybrid.size else 0.0,
            "p50_shift": float(p50.get("shift", 0.0)),
            "p75_shift": float(p75.get("shift", 0.0)),
        },
        "map_threshold": map_threshold,
    }


def evaluate_candidate(
    base_scores: np.ndarray,
    labels: np.ndarray,
    components: Dict[str, np.ndarray],
    weights: Dict[str, float],
    cap: float,
    approve_threshold: float,
    block_threshold: float,
    fpr_penalty: float,
    ring_scores: np.ndarray,
    ring_weight: float,
    ring_cap: float,
    max_fpr: float,
    min_recall: float,
    fairness_segments: Dict[str, np.ndarray],
    fairness_max_fpr_gap: float,
    fairness_max_recall_gap: float,
) -> CandidateResult:
    context_adj = compute_context_adjustment(components, weights, cap)
    ring_adj = np.minimum(np.maximum(0.0, float(ring_weight)) * ring_scores, float(ring_cap))
    final_scores = clamp01(base_scores + context_adj + ring_adj)

    pred_fraud = final_scores >= block_threshold
    precision = precision_score(labels, pred_fraud, zero_division=0)
    recall = recall_score(labels, pred_fraud, zero_division=0)
    f1 = f1_score(labels, pred_fraud, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, pred_fraud, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    overall_recall = float(recall)
    overall_fpr = float(fpr)
    fpr_gaps: list[float] = []
    recall_gaps: list[float] = []
    for segment_mask in fairness_segments.values():
        segment_labels = labels[segment_mask]
        segment_pred = pred_fraud[segment_mask]
        if segment_labels.size < 50:
            continue
        seg_tn, seg_fp, _, _ = confusion_matrix(segment_labels, segment_pred, labels=[0, 1]).ravel()
        seg_fpr = seg_fp / (seg_fp + seg_tn) if (seg_fp + seg_tn) else 0.0
        seg_recall = recall_score(segment_labels, segment_pred, zero_division=0)
        fpr_gaps.append(abs(float(seg_fpr) - overall_fpr))
        recall_gaps.append(abs(float(seg_recall) - overall_recall))
    observed_fpr_gap = float(max(fpr_gaps) if fpr_gaps else 0.0)
    observed_recall_gap = float(max(recall_gaps) if recall_gaps else 0.0)
    policy_pass = (overall_fpr <= max_fpr) and (overall_recall >= min_recall)
    fairness_pass = (observed_fpr_gap <= fairness_max_fpr_gap) and (observed_recall_gap <= fairness_max_recall_gap)

    objective = (0.55 * recall) + (0.45 * f1) - (fpr_penalty * fpr)
    if not policy_pass:
        objective -= 1.0
    if not fairness_pass:
        objective -= 1.0

    return CandidateResult(
        objective=float(objective),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        fpr=float(fpr),
        approve_threshold=float(approve_threshold),
        block_threshold=float(block_threshold),
        cap=float(cap),
        weights={k: float(v) for k, v in weights.items()},
        ring_weight=float(ring_weight),
        ring_cap=float(ring_cap),
        fairness_max_fpr_gap=observed_fpr_gap,
        fairness_max_recall_gap=observed_recall_gap,
        policy_pass=bool(policy_pass),
        fairness_pass=bool(fairness_pass),
    )


def sample_candidate_weights(rng: np.random.Generator, defaults: Dict[str, float]) -> Dict[str, float]:
    global_scale = rng.uniform(0.7, 1.4)
    out: Dict[str, float] = {}
    for key, default in defaults.items():
        local_scale = rng.uniform(0.75, 1.25)
        out[key] = max(0.0, default * global_scale * local_scale)
    return out


def candidate_grid(rng: np.random.Generator, trials: int) -> Iterable[Tuple[Dict[str, float], float, float, float]]:
    yield CONTEXT_WEIGHT_DEFAULTS.copy(), 0.30, 0.30, 0.90
    for _ in range(trials - 1):
        weights = sample_candidate_weights(rng, CONTEXT_WEIGHT_DEFAULTS)
        cap = float(rng.uniform(0.12, 0.45))
        approve = float(rng.uniform(0.20, 0.45))
        block = float(rng.uniform(max(approve + 0.20, 0.55), 0.98))
        yield weights, cap, approve, block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate context weights/cap and decision thresholds against labels.")
    parser.add_argument("--dataset-source", choices=["creditcard", "ieee_cis"], default="ieee_cis")
    parser.add_argument("--dataset-path", type=Path, help="Legacy credit-card CSV path (used when --dataset-source creditcard)")
    parser.add_argument("--ieee-transaction-path", type=Path)
    parser.add_argument("--ieee-identity-path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--preprocessing-artifact-path", type=Path, default=DEFAULT_PREPROCESSING_ARTIFACT_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--pr-curve-calibration-report", type=Path, default=DEFAULT_PR_CURVE_CALIBRATION_REPORT)
    parser.add_argument("--thresholds-output", type=Path, default=DEFAULT_THRESHOLDS_OUTPUT)
    parser.add_argument("--baseline-thresholds-path", type=Path, default=DEFAULT_BASELINE_THRESHOLDS_PATH)
    parser.add_argument("--baseline-calibration-json", type=Path, default=DEFAULT_BASELINE_CALIBRATION_JSON)
    parser.add_argument("--trials", type=int, default=350)
    parser.add_argument("--fpr-penalty", type=float, default=0.35)
    parser.add_argument("--target-fpr", type=float, default=0.005)
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument("--min-recall", type=float, default=0.30)
    parser.add_argument("--ring-weight-min", type=float, default=0.02)
    parser.add_argument("--ring-weight-max", type=float, default=0.20)
    parser.add_argument("--ring-weight-step", type=float, default=0.02)
    parser.add_argument("--ring-cap-min", type=float, default=0.02)
    parser.add_argument("--ring-cap-max", type=float, default=0.20)
    parser.add_argument("--ring-cap-step", type=float, default=0.02)
    parser.add_argument("--fairness-max-fpr-gap", type=float, default=0.03)
    parser.add_argument("--fairness-max-recall-gap", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entity-smoothing-method", choices=["none", "mean", "ema", "blend"], default="none")
    parser.add_argument("--entity-smoothing-min-history", type=int, default=2)
    parser.add_argument("--entity-smoothing-ema-alpha", type=float, default=0.3)
    parser.add_argument("--entity-smoothing-blend-alpha", type=float, default=0.5)
    parser.add_argument("--entity-smoothing-blend-cap", type=float, default=0.25)
    return parser.parse_args()




def load_source(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series, dict]:
    if args.dataset_source == "creditcard":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for creditcard source (legacy assets now in project/legacy_creditcard)")
        return load_creditcard(args.dataset_path)

    if not args.ieee_transaction_path or not args.ieee_identity_path:
        raise ValueError("--ieee-transaction-path and --ieee-identity-path are required for ieee_cis source")
    return load_ieee_cis(args.ieee_transaction_path, args.ieee_identity_path)


def resolve_amount_and_time_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    amount_col = "Amount" if "Amount" in df.columns else "TransactionAmt"
    time_col = "Time" if "Time" in df.columns else "TransactionDT"
    amount = pd.to_numeric(df.get(amount_col, 0.0), errors="coerce").fillna(0.0)
    time_values = pd.to_numeric(df.get(time_col, 0.0), errors="coerce").fillna(0.0)
    return amount, time_values


def build_ring_scores(df: pd.DataFrame) -> np.ndarray:
    if "ring_score" in df.columns:
        return pd.to_numeric(df["ring_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(float)
    return np.zeros(len(df), dtype=float)


def build_fairness_segments(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    channel = df.get("channel", pd.Series("UNKNOWN", index=df.index)).astype(str).str.upper()
    tx_type = df.get("tx_type", pd.Series("UNKNOWN", index=df.index)).astype(str).str.upper()
    cross_border = df.get("is_cross_border", pd.Series(0, index=df.index)).astype(str).str.lower().isin(["1", "true", "yes"])
    return {
        "channel_app": (channel == "APP").to_numpy(bool),
        "channel_agent": (channel == "AGENT").to_numpy(bool),
        "tx_type_p2p": (tx_type == "P2P").to_numpy(bool),
        "tx_type_cash_out": (tx_type == "CASH_OUT").to_numpy(bool),
        "cross_border_true": cross_border.to_numpy(bool),
    }

def load_baseline_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    approve = 0.30
    block = 0.90
    cap = 0.30
    weights = CONTEXT_WEIGHT_DEFAULTS.copy()

    calibration_path = args.baseline_calibration_json.expanduser().resolve()
    if calibration_path.exists() and calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        recommendation = calibration.get("runtime_recommendation") or {}
        cap = float(recommendation.get("context_adjustment_max", cap))
        weights = {**weights, **(recommendation.get("context_weights") or {})}
        approve = float(recommendation.get("approve_threshold", approve))
        block = float(recommendation.get("block_threshold", block))

    thresholds_path = args.baseline_thresholds_path.expanduser().resolve()
    if thresholds_path.exists() and thresholds_path.is_file():
        baseline_thresholds = load_pickle(thresholds_path)
        approve = float(baseline_thresholds.get("approve_threshold", approve))
        block = float(baseline_thresholds.get("block_threshold", block))

    validate_threshold_order(approve, block)
    return RuntimeConfig(
        approve_threshold=float(approve),
        block_threshold=float(block),
        cap=float(cap),
        weights={k: float(v) for k, v in weights.items()},
    )




def compute_metrics_with_optional_entity_smoothing(
    scores: np.ndarray,
    labels: np.ndarray,
    eval_frame: pd.DataFrame,
    block_threshold: float,
    *,
    method: str,
    min_history: int,
    ema_alpha: float,
    blend_alpha: float,
    blend_cap: float,
) -> tuple[dict, np.ndarray]:
    resolved = validate_smoothing_method(method)
    if resolved == "none":
        return evaluate_runtime_metrics(scores, labels, block_threshold), scores

    eval_df = eval_frame.copy()
    eval_df["raw_score"] = scores
    eval_df["label"] = labels
    eval_df = eval_df.sort_values(["event_time"], kind="mergesort")
    smoothed = apply_entity_smoothing_batch(
        eval_df,
        method=resolved,
        entity_col="entity_id",
        raw_col="raw_score",
        min_history=int(min_history),
        ema_alpha=float(ema_alpha),
        blend_alpha=float(blend_alpha),
        blend_cap=float(blend_cap),
    )
    y_sorted = eval_df["label"].to_numpy(dtype=int)
    return evaluate_runtime_metrics(smoothed, y_sorted, block_threshold), smoothed

def main() -> None:
    args = parse_args()
    if args.dataset_source == "ieee_cis":
        if not args.ieee_transaction_path or not args.ieee_identity_path:
            raise ValueError("--ieee-transaction-path and --ieee-identity-path are required for ieee_cis source")
        features_df, labels_series, _ = load_ieee_cis(
            args.ieee_transaction_path,
            args.ieee_identity_path,
        )
        amount_col = "TransactionAmt"
        time_col_name = "TransactionDT"
    else:
        if not args.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
        features_df, labels_series, _ = load_creditcard(args.dataset_path)
        amount_col = "Amount"
        time_col_name = "Time"

    df = features_df.copy()
    df["__label__"] = labels_series.astype(int).to_numpy()

    feature_columns = load_pickle(args.feature_path)
    model = load_pickle(args.model_path)
    artifact_metadata = collect_artifact_runtime_metadata(
        model_path=args.model_path,
        feature_path=args.feature_path,
        preprocessing_artifact_path=args.preprocessing_artifact_path,
    )

    df = ensure_context_columns(df)

    X = build_model_input(df, feature_columns, args)
    y = df["__label__"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X,
        y,
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    del X_train, y_train, df_train

    base_scores = model.predict_proba(X_test)[:, 1]
    base_auc = roc_auc_score(y_test, base_scores)

    event_time = pd.to_numeric(
        df_test.get(time_col_name, pd.Series(np.arange(len(df_test)), index=df_test.index)),
        errors="coerce",
    ).fillna(0.0)
    entity_id, entity_diag = build_entity_id(df_test, dataset_source=args.dataset_source)
    eval_frame = pd.DataFrame({"entity_id": entity_id, "event_time": event_time.to_numpy()}, index=df_test.index)

    components = build_context_component_matrix(df_test, dataset_source=args.dataset_source)
    ring_scores = build_ring_scores(df_test)
    fairness_segments = build_fairness_segments(df_test)
    rng = np.random.default_rng(args.seed)
    baseline_runtime = load_baseline_runtime_config(args)

    pre_context_adj = compute_context_adjustment(components, baseline_runtime.weights, baseline_runtime.cap)
    pre_scores = clamp01(base_scores + pre_context_adj)
    pre_metrics, pre_scores_smoothed = compute_metrics_with_optional_entity_smoothing(
        pre_scores,
        y_test,
        eval_frame,
        baseline_runtime.block_threshold,
        method=args.entity_smoothing_method,
        min_history=args.entity_smoothing_min_history,
        ema_alpha=args.entity_smoothing_ema_alpha,
        blend_alpha=args.entity_smoothing_blend_alpha,
        blend_cap=args.entity_smoothing_blend_cap,
    )
    pre_objective = (0.55 * pre_metrics["recall"]) + (0.45 * pre_metrics["f1"]) - (args.fpr_penalty * pre_metrics["fpr"])

    best: CandidateResult | None = None
    rows: List[dict] = []

    ring_weights = np.arange(args.ring_weight_min, args.ring_weight_max + 1e-9, args.ring_weight_step)
    ring_caps = np.arange(args.ring_cap_min, args.ring_cap_max + 1e-9, args.ring_cap_step)

    for i, (weights, cap, approve, block) in enumerate(candidate_grid(rng, args.trials), start=1):
        for ring_weight in ring_weights:
            for ring_cap in ring_caps:
                result = evaluate_candidate(
                    base_scores=base_scores,
                    labels=y_test,
                    components=components,
                    weights=weights,
                    cap=cap,
                    approve_threshold=approve,
                    block_threshold=block,
                    fpr_penalty=args.fpr_penalty,
                    ring_scores=ring_scores,
                    ring_weight=float(ring_weight),
                    ring_cap=float(ring_cap),
                    max_fpr=args.target_fpr,
                    min_recall=args.min_recall,
                    fairness_segments=fairness_segments,
                    fairness_max_fpr_gap=args.fairness_max_fpr_gap,
                    fairness_max_recall_gap=args.fairness_max_recall_gap,
                )

                rows.append(
                    {
                        "trial": i,
                        "objective": result.objective,
                        "precision": result.precision,
                        "recall": result.recall,
                        "f1": result.f1,
                        "fpr": result.fpr,
                        "approve_threshold": result.approve_threshold,
                        "block_threshold": result.block_threshold,
                        "cap": result.cap,
                        "ring_weight": result.ring_weight,
                        "ring_cap": result.ring_cap,
                        "fairness_max_fpr_gap": result.fairness_max_fpr_gap,
                        "fairness_max_recall_gap": result.fairness_max_recall_gap,
                        "fpr_le_target": result.fpr <= args.target_fpr,
                        "precision_ge_target": result.precision >= args.target_precision,
                        "recall_ge_target": result.recall >= args.min_recall,
                        "policy_pass": result.policy_pass,
                        "fairness_pass": result.fairness_pass,
                    }
                )

                if best is None or result.objective > best.objective:
                    best = result

    assert best is not None
    validate_threshold_order(best.approve_threshold, best.block_threshold)

    best_context_adj = compute_context_adjustment(components, best.weights, best.cap)
    best_ring_adj = np.minimum(best.ring_weight * ring_scores, best.ring_cap)
    best_scores = clamp01(base_scores + best_context_adj + best_ring_adj)
    best_metrics, _ = compute_metrics_with_optional_entity_smoothing(
        best_scores,
        y_test,
        eval_frame,
        best.block_threshold,
        method=args.entity_smoothing_method,
        min_history=args.entity_smoothing_min_history,
        ema_alpha=args.entity_smoothing_ema_alpha,
        blend_alpha=args.entity_smoothing_blend_alpha,
        blend_cap=args.entity_smoothing_blend_cap,
    )

    non_regression = {
        "objective_non_regressed": bool(best.objective >= pre_objective),
        "f1_non_regressed": bool(best_metrics["f1"] >= pre_metrics["f1"]),
        "recall_non_regressed": bool(best_metrics["recall"] >= pre_metrics["recall"]),
    }

    selected = best
    regression_fallback_applied = False
    if not all(non_regression.values()):
        regression_fallback_applied = True
        selected = CandidateResult(
            objective=float(pre_objective),
            precision=float(pre_metrics["precision"]),
            recall=float(pre_metrics["recall"]),
            f1=float(pre_metrics["f1"]),
            fpr=float(pre_metrics["fpr"]),
            approve_threshold=baseline_runtime.approve_threshold,
            block_threshold=baseline_runtime.block_threshold,
            cap=baseline_runtime.cap,
            weights=baseline_runtime.weights,
            ring_weight=float(args.ring_weight_min),
            ring_cap=float(args.ring_cap_max),
            fairness_max_fpr_gap=0.0,
            fairness_max_recall_gap=0.0,
            policy_pass=bool((pre_metrics["fpr"] <= args.target_fpr) and (pre_metrics["recall"] >= args.min_recall)),
            fairness_pass=False,
        )

    validate_threshold_order(selected.approve_threshold, selected.block_threshold)
    selected_context_adj = compute_context_adjustment(components, selected.weights, selected.cap)
    selected_ring_adj = np.minimum(selected.ring_weight * ring_scores, selected.ring_cap)
    selected_scores = clamp01(base_scores + selected_context_adj + selected_ring_adj)
    selected_metrics, selected_scores_smoothed = compute_metrics_with_optional_entity_smoothing(
        selected_scores,
        y_test,
        eval_frame,
        selected.block_threshold,
        method=args.entity_smoothing_method,
        min_history=args.entity_smoothing_min_history,
        ema_alpha=args.entity_smoothing_ema_alpha,
        blend_alpha=args.entity_smoothing_blend_alpha,
        blend_cap=args.entity_smoothing_blend_cap,
    )
    decision_drift = compute_decision_drift(
        pre_scores=pre_scores_smoothed,
        post_scores=selected_scores_smoothed,
        approve_threshold=selected.approve_threshold,
        block_threshold=selected.block_threshold,
    )
    threshold_mapping = build_raw_to_hybrid_threshold_mapping(base_scores, selected_scores_smoothed)
    approve_mapped = threshold_mapping["map_threshold"](selected.approve_threshold)
    block_mapped = threshold_mapping["map_threshold"](selected.block_threshold)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.pr_curve_calibration_report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(by="objective", ascending=False).to_csv(args.output_csv, index=False)

    fpr_pass = selected.fpr <= args.target_fpr
    precision_pass = selected.precision >= args.target_precision
    recall_pass = selected.recall >= args.min_recall
    fairness_fpr_pass = selected.fairness_max_fpr_gap <= args.fairness_max_fpr_gap
    fairness_recall_pass = selected.fairness_max_recall_gap <= args.fairness_max_recall_gap
    output = {
        "summary": {
            "base_model_roc_auc": float(base_auc),
            "objective": selected.objective,
            "precision": selected.precision,
            "recall": selected.recall,
            "f1": selected.f1,
            "fpr": selected.fpr,
            "dataset_source": args.dataset_source,
            "dataset_path": str(args.dataset_path),
            "dataset_source": args.dataset_source,
            "ieee_transaction_path": str(args.ieee_transaction_path) if args.ieee_transaction_path else None,
            "ieee_identity_path": str(args.ieee_identity_path) if args.ieee_identity_path else None,
            "amount_column": amount_col,
            "trials": args.trials,
            "fpr_penalty": args.fpr_penalty,
            "target_fpr": args.target_fpr,
            "target_precision": args.target_precision,
            "min_recall": args.min_recall,
            "seed": args.seed,
            "regression_fallback_applied": regression_fallback_applied,
            "ring_sweep": {
                "ring_weight_min": args.ring_weight_min,
                "ring_weight_max": args.ring_weight_max,
                "ring_weight_step": args.ring_weight_step,
                "ring_cap_min": args.ring_cap_min,
                "ring_cap_max": args.ring_cap_max,
                "ring_cap_step": args.ring_cap_step,
            },
        },
        "pre_calibration": {
            "runtime": {
                "approve_threshold": baseline_runtime.approve_threshold,
                "block_threshold": baseline_runtime.block_threshold,
                "context_adjustment_max": baseline_runtime.cap,
            },
            "metrics": pre_metrics,
            "objective": float(pre_objective),
        },
        "post_calibration": {
            "runtime": {
                "approve_threshold": selected.approve_threshold,
                "block_threshold": selected.block_threshold,
                "context_adjustment_max": selected.cap,
                "ring_score_weight": selected.ring_weight,
                "ring_score_cap": selected.ring_cap,
            },
            "metrics": selected_metrics,
            "objective": float(selected.objective),
        },
        "decision_drift_pre_post": decision_drift,
        "entity_smoothing": {
            "method": args.entity_smoothing_method,
            "min_history": args.entity_smoothing_min_history,
            "ema_alpha": args.entity_smoothing_ema_alpha,
            "blend_alpha": args.entity_smoothing_blend_alpha,
            "blend_cap": args.entity_smoothing_blend_cap,
            "entity_diagnostics": entity_diag,
            "pre_calibration_smoothed_metrics": pre_metrics,
            "post_calibration_smoothed_metrics": selected_metrics,
        },
        "non_regression_check": non_regression,
        "policy_checks": {
            "fpr_le_target": {
                "metric": "fpr",
                "operator": "<=",
                "actual": selected.fpr,
                "target": args.target_fpr,
                "pass": fpr_pass,
            },
            "precision_ge_target": {
                "metric": "precision",
                "operator": ">=",
                "actual": selected.precision,
                "target": args.target_precision,
                "pass": precision_pass,
            },
            "recall_ge_target": {
                "metric": "recall",
                "operator": ">=",
                "actual": selected.recall,
                "target": args.min_recall,
                "pass": selected.recall >= args.min_recall,
            },
            "fairness_max_fpr_gap": {
                "metric": "max_abs_segment_fpr_gap",
                "operator": "<=",
                "actual": selected.fairness_max_fpr_gap,
                "target": args.fairness_max_fpr_gap,
                "pass": selected.fairness_max_fpr_gap <= args.fairness_max_fpr_gap,
            },
            "fairness_max_recall_gap": {
                "metric": "max_abs_segment_recall_gap",
                "operator": "<=",
                "actual": selected.fairness_max_recall_gap,
                "target": args.fairness_max_recall_gap,
                "pass": selected.fairness_max_recall_gap <= args.fairness_max_recall_gap,
            },
            "overall_pass": fpr_pass and precision_pass and recall_pass and fairness_fpr_pass and fairness_recall_pass,
            "threshold_order_valid": selected.approve_threshold < selected.block_threshold,
        },
        "runtime_recommendation": {
            "approve_threshold": approve_mapped["hybrid_threshold"],
            "block_threshold": block_mapped["hybrid_threshold"],
            "context_adjustment_max": selected.cap,
            "context_weights": selected.weights,
            "ring_score_weight": selected.ring_weight,
            "ring_score_cap": selected.ring_cap,
            "threshold_calibration_method": "quantile_distribution_mapping",
            "raw_to_hybrid_quantile_rank": {
                "approve": approve_mapped["quantile_rank"],
                "block": block_mapped["quantile_rank"],
            },
        },
        "selection_rationale": {
            "chosen_objective": selected.objective,
            "policy_constraints": {
                "max_fpr": args.target_fpr,
                "min_precision": args.target_precision,
                "min_recall": args.min_recall,
            },
            "fairness_guardrails": {
                "max_abs_segment_fpr_gap": args.fairness_max_fpr_gap,
                "max_abs_segment_recall_gap": args.fairness_max_recall_gap,
                "observed_max_abs_segment_fpr_gap": selected.fairness_max_fpr_gap,
                "observed_max_abs_segment_recall_gap": selected.fairness_max_recall_gap,
            },
            "ring_parameters": {
                "ring_score_weight": selected.ring_weight,
                "ring_score_cap": selected.ring_cap,
            },
            "selection_mode": "constrained_objective_with_ring_grid_sweep",
        },
        "artifact_metadata": artifact_metadata,
    }

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    pr_curve_report = {
        "summary": {
            "rows": int(len(base_scores)),
            "mapping_method": "empirical_quantile_mapping",
        },
        "threshold_mapping": {
            "approve": approve_mapped,
            "block": block_mapped,
        },
        "quantile_threshold_mapping": threshold_mapping["quantile_threshold_mapping"],
        "distribution_shift_summary": threshold_mapping["distribution_shift_summary"],
    }
    with args.pr_curve_calibration_report.open("w", encoding="utf-8") as f:
        json.dump(pr_curve_report, f, indent=2)

    args.thresholds_output.parent.mkdir(parents=True, exist_ok=True)
    with args.thresholds_output.open("wb") as f:
        pickle.dump(
            {
                "approve_threshold": selected.approve_threshold,
                "block_threshold": selected.block_threshold,
            },
            f,
        )

    print("Calibration complete")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved calibration JSON: {args.output_json}")
    print(f"Saved trial table: {args.output_csv}")
    print(f"Saved PR-curve calibration report: {args.pr_curve_calibration_report}")
    print(f"Saved decision thresholds: {args.thresholds_output}")


if __name__ == "__main__":
    main()

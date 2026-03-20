import argparse
import hashlib
import json
import pickle
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.preprocessing import load_preprocessing_bundle, prepare_preprocessing_inputs, transform_with_bundle

DEFAULT_DATASET_PATH = REPO_ROOT / "project" / "legacy_creditcard" / "creditcard.csv"
DEFAULT_MODEL_PATH = REPO_ROOT / "project" / "models" / "final_xgboost_model.pkl"
DEFAULT_FEATURE_PATH = REPO_ROOT / "project" / "models" / "feature_columns.pkl"
DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "project" / "models" / "decision_thresholds.pkl"
DEFAULT_PREPROCESSING_ARTIFACT_PATH = REPO_ROOT / "project" / "models" / "preprocessing_artifact_promoted.pkl"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "project" / "outputs" / "monitoring" / "cohort_kpi_report.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "project" / "outputs" / "monitoring" / "cohort_kpi_report.csv"
DEFAULT_EVALUATION_MODE = "time_aware"

DEFAULT_COHORT_DEFINITIONS: dict[str, Any] = {
    "version": "builtin-v1",
    "cohorts": [
        {"name": "all_users", "all": []},
        {"name": "new_users", "all": [{"column": "account_age_days", "op": "<", "value": 7}]},
        {
            "name": "rural_merchants_proxy",
            "all": [{"column": "channel", "op": "==", "value": "AGENT"}, {"column": "__amount__", "op": "<=", "value": 250}],
        },
        {"name": "gig_workers_proxy", "all": [{"column": "tx_type", "op": "in", "value": ["P2P", "CASH_OUT"]}]},
        {
            "name": "low_history_proxy",
            "any": [
                {"column": "account_age_days", "op": "<", "value": 14},
                {"all": [{"column": "channel", "op": "==", "value": "AGENT"}, {"column": "__amount__", "op": "<", "value": 100}]},
            ],
        },
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cohort-level fraud KPIs (precision/recall/FPR/flag-rate).")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH, help="Legacy credit-card CSV path (used when --dataset-source creditcard)")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--preprocessing-artifact-path", type=Path, default=DEFAULT_PREPROCESSING_ARTIFACT_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--context-adjustment-max", type=float, default=0.30)
    parser.add_argument("--sample-size", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-mode",
        choices=["time_aware", "random_sample"],
        default=DEFAULT_EVALUATION_MODE,
        help="Evaluation selection mode; defaults to time-aware (most recent rows), not random.",
    )
    parser.add_argument(
        "--cohort-config-path",
        type=Path,
        default=None,
        help="Optional path to cohort definitions YAML/JSON. Enables versioned cohort logic.",
    )
    parser.add_argument(
        "--min-positive-support",
        type=int,
        default=5,
        help="Minimum fraud-positive support required for reliable precision/recall interpretation.",
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_path: Path) -> Path:
    if dataset_path.exists():
        return dataset_path

    raw = str(dataset_path)
    candidates: list[Path] = []

    if raw.startswith("/tmp/") or raw.startswith("\\tmp\\") or raw.startswith("/tmp\\"):
        temp_root = Path(tempfile.gettempdir())
        candidates.append(temp_root / dataset_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    hint = ""
    if raw.startswith("/tmp/") or raw.startswith("\\tmp\\") or raw.startswith("/tmp\\"):
        hint = f" (hint: on Windows, '/tmp/...' usually maps to '{Path(tempfile.gettempdir())}')"

    raise FileNotFoundError(f"Dataset not found: {dataset_path}{hint}")


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
    canonical_df, passthrough_df, _ = prepare_preprocessing_inputs(df, dataset_source="ieee_cis")
    transformed = transform_with_bundle(bundle, canonical_df, passthrough_df)
    transformed_df = pd.DataFrame(transformed, columns=bundle.feature_names_out, index=df.index)
    missing = [col for col in feature_columns if col not in transformed_df.columns]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Preprocessed feature columns missing from transformed cohort KPI input: {preview}")
    return transformed_df.loc[:, feature_columns]


def _bool_from_series(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.lower()
    return lowered.isin(["1", "true", "yes"])


def _resolve_time_column(df: pd.DataFrame) -> str:
    return "TransactionDT" if "TransactionDT" in df.columns else "Time"


def _resolve_amount_column(df: pd.DataFrame) -> str:
    return "TransactionAmt" if "TransactionAmt" in df.columns else "Amount"


def ensure_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    time_col = _resolve_time_column(out)
    amount_col = _resolve_amount_column(out)

    if "account_age_days" not in out.columns:
        out["account_age_days"] = (pd.to_numeric(out[time_col], errors="coerce").fillna(0).astype(int) % 120) + 1

    if "channel" not in out.columns:
        out["channel"] = np.where(
            (pd.to_numeric(out[time_col], errors="coerce").fillna(0).astype(int) % 10) < 2,
            "AGENT",
            "APP",
        )

    if "tx_type" not in out.columns:
        amount = pd.to_numeric(out[amount_col], errors="coerce").fillna(0.0)
        time_mod = pd.to_numeric(out[time_col], errors="coerce").fillna(0).astype(int)
        out["tx_type"] = np.where(
            (amount > 300) & (time_mod % 7 == 0),
            "CASH_OUT",
            np.where((amount < 80) & (time_mod % 5 == 0), "P2P", "MERCHANT"),
        )

    defaults = {
        "device_risk_score": 0.0,
        "ip_risk_score": 0.0,
        "location_risk_score": 0.0,
        "device_shared_users_24h": 0,
        "sim_change_recent": False,
        "cash_flow_velocity_1h": 0,
        "p2p_counterparties_24h": 0,
        "is_cross_border": False,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    return out


def compute_context_adjustment(df: pd.DataFrame, cap: float) -> np.ndarray:
    amount_col = _resolve_amount_column(df)
    time_name = _resolve_time_column(df)
    amount = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0).to_numpy(float)
    time_col = pd.to_numeric(df[time_name], errors="coerce").fillna(0.0).to_numpy(float)
    account_age = pd.to_numeric(df["account_age_days"], errors="coerce").fillna(30).to_numpy(int)

    device_risk = pd.to_numeric(df["device_risk_score"], errors="coerce").fillna(0.0).to_numpy(float)
    ip_risk = pd.to_numeric(df["ip_risk_score"], errors="coerce").fillna(0.0).to_numpy(float)
    location_risk = pd.to_numeric(df["location_risk_score"], errors="coerce").fillna(0.0).to_numpy(float)

    shared_users = pd.to_numeric(df["device_shared_users_24h"], errors="coerce").fillna(0).to_numpy(int)
    flow_velocity = pd.to_numeric(df["cash_flow_velocity_1h"], errors="coerce").fillna(0).to_numpy(int)
    p2p_count = pd.to_numeric(df["p2p_counterparties_24h"], errors="coerce").fillna(0).to_numpy(int)

    sim_change = _bool_from_series(df["sim_change_recent"]).to_numpy(float)
    cross_border = _bool_from_series(df["is_cross_border"]).to_numpy(float)

    tx_type = df["tx_type"].astype(str).str.upper()
    channel = df["channel"].astype(str).str.upper()

    context = (
        0.05 * device_risk
        + 0.05 * ip_risk
        + 0.05 * location_risk
        + 0.02 * (amount > 200).astype(float)
        + 0.04 * (amount > 1000).astype(float)
        + 0.01 * (time_col < 1000).astype(float)
        + 0.03 * (shared_users >= 3).astype(float)
        + 0.07 * (shared_users >= 5).astype(float)
        + 0.05 * sim_change
        + 0.03 * (account_age < 7).astype(float)
        + 0.06 * (account_age < 3).astype(float)
        + 0.04 * ((tx_type == "CASH_OUT") & (amount > 300)).astype(float)
        + 0.03 * ((channel == "AGENT") & ((device_risk >= 0.5) | (ip_risk >= 0.5))).astype(float)
        + 0.05 * (flow_velocity >= 8).astype(float)
        + 0.04 * (p2p_count >= 12).astype(float)
        + 0.03 * cross_border
    )

    return np.clip(np.minimum(context, cap), 0.0, 1.0)


def assign_cohorts(df: pd.DataFrame) -> Dict[str, pd.Series]:
    amount = pd.to_numeric(df[_resolve_amount_column(df)], errors="coerce").fillna(0.0)
    channel = df["channel"].astype(str).str.upper()
    tx_type = df["tx_type"].astype(str).str.upper()
    account_age = pd.to_numeric(df["account_age_days"], errors="coerce").fillna(30)

    return {
        "all_users": pd.Series([True] * len(df), index=df.index),
        "new_users": account_age < 7,
        "rural_merchants_proxy": (channel == "AGENT") & (amount <= 250),
        "gig_workers_proxy": tx_type.isin(["P2P", "CASH_OUT"]),
        "low_history_proxy": (account_age < 14) | ((channel == "AGENT") & (amount < 100)),
    }


def _maybe_load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("YAML cohort config requires PyYAML to be installed.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_cohort_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return DEFAULT_COHORT_DEFINITIONS
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        return _maybe_load_yaml(path)
    raise ValueError(f"Unsupported cohort config extension: {suffix}. Use .json/.yaml/.yml.")


def cohort_config_metadata(config: dict[str, Any], source_path: Path | None) -> dict[str, str]:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "version": str(config.get("version", "unspecified")),
        "definition_hash_sha256": hashlib.sha256(canonical).hexdigest(),
        "source": str(source_path) if source_path is not None else "builtin_default",
    }


def _normalize_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return series.astype(str).str.upper()
    return pd.to_numeric(series, errors="coerce")


def _eval_condition(df: pd.DataFrame, condition: dict[str, Any]) -> pd.Series:
    if "all" in condition:
        nested = condition.get("all", [])
        result = pd.Series([True] * len(df), index=df.index)
        for part in nested:
            result &= _eval_condition(df, part)
        return result
    if "any" in condition:
        nested = condition.get("any", [])
        result = pd.Series([False] * len(df), index=df.index)
        for part in nested:
            result |= _eval_condition(df, part)
        return result

    column = str(condition["column"])
    op = str(condition["op"])
    value = condition.get("value")

    if column == "__amount__":
        column = _resolve_amount_column(df)
    elif column == "__time__":
        column = _resolve_time_column(df)

    left = _normalize_series(df[column])
    right: Any = value
    if isinstance(value, str):
        right = value.upper()
    if isinstance(value, list):
        right = [str(item).upper() if isinstance(item, str) else item for item in value]

    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "<":
        left_numeric = pd.to_numeric(df[column], errors="coerce")
        return left_numeric < pd.to_numeric(right, errors="coerce")
    if op == "<=":
        left_numeric = pd.to_numeric(df[column], errors="coerce")
        return left_numeric <= pd.to_numeric(right, errors="coerce")
    if op == ">":
        left_numeric = pd.to_numeric(df[column], errors="coerce")
        return left_numeric > pd.to_numeric(right, errors="coerce")
    if op == ">=":
        left_numeric = pd.to_numeric(df[column], errors="coerce")
        return left_numeric >= pd.to_numeric(right, errors="coerce")
    if op == "in":
        return left.isin(right)
    if op == "not_in":
        return ~left.isin(right)
    raise ValueError(f"Unsupported cohort operator: {op}")


def assign_cohorts_from_config(df: pd.DataFrame, cohort_config: dict[str, Any]) -> Dict[str, pd.Series]:
    items = cohort_config.get("cohorts", [])
    if not items:
        raise ValueError("Cohort config must include non-empty 'cohorts' list.")
    cohorts: Dict[str, pd.Series] = {}
    for item in items:
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("Each cohort definition must include non-empty 'name'.")
        cohorts[name] = _eval_condition(df, item)
    return cohorts


def select_evaluation_frame(df: pd.DataFrame, sample_size: int, seed: int, mode: str) -> pd.DataFrame:
    if sample_size <= 0 or len(df) <= sample_size:
        return df.reset_index(drop=True)
    if mode == "random_sample":
        return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    time_col = _resolve_time_column(df)
    ordered = df.assign(__time_order=pd.to_numeric(df[time_col], errors="coerce").fillna(-np.inf))
    return ordered.sort_values("__time_order").tail(sample_size).drop(columns=["__time_order"]).reset_index(drop=True)


def metric_reliability_label(fraud_positive_count: int, min_positive_support: int) -> str:
    return "reliable" if fraud_positive_count >= min_positive_support else "low_support"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + (z * z / total)
    center = (phat + (z * z) / (2 * total)) / denom
    margin = (z / denom) * np.sqrt((phat * (1 - phat) / total) + ((z * z) / (4 * total * total)))
    return max(0.0, center - margin), min(1.0, center + margin)


def support_weighted_reliability(fraud_positive_count: int, min_positive_support: int) -> float:
    return min(1.0, fraud_positive_count / max(1, min_positive_support))


def metric_row(
    name: str,
    mask: pd.Series,
    labels: np.ndarray,
    pred_block: np.ndarray,
    pred_flag: np.ndarray,
    min_positive_support: int,
) -> Dict[str, float | int | str]:
    idx = np.where(mask.to_numpy())[0]
    y_all = np.asarray(labels)
    pred_block_all = np.asarray(pred_block)
    pred_flag_all = np.asarray(pred_flag)

    if len(idx) == 0:
        return {
            "cohort": name,
            "sample_count": 0,
            "fraud_positive_count": 0,
            "nonfraud_count": 0,
            "metric_reliability": "low_support",
            "support_weighted_reliability": 0.0,
            "fraud_rate": 0.0,
            "precision": 0.0,
            "precision_ci_lower": 0.0,
            "precision_ci_upper": 0.0,
            "recall": 0.0,
            "recall_ci_lower": 0.0,
            "recall_ci_upper": 0.0,
            "false_positive_rate": 0.0,
            "false_positive_rate_ci_lower": 0.0,
            "false_positive_rate_ci_upper": 0.0,
            "flag_rate": 0.0,
            "block_rate": 0.0,
        }

    y = y_all[idx]
    y_pred = pred_block_all[idx]
    y_flag = pred_flag_all[idx]

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    fpr = (fp / (fp + tn)) if (fp + tn) else 0.0

    fraud_positive_count = int((y == 1).sum())
    nonfraud_count = int((y == 0).sum())
    tp = int(tp)
    fp = int(fp)
    fn = int(fn)
    tn = int(tn)
    precision_ci = wilson_interval(tp, tp + fp)
    recall_ci = wilson_interval(tp, tp + fn)
    fpr_ci = wilson_interval(fp, fp + tn)

    return {
        "cohort": name,
        "sample_count": int(len(idx)),
        "fraud_positive_count": fraud_positive_count,
        "nonfraud_count": nonfraud_count,
        "metric_reliability": metric_reliability_label(fraud_positive_count, min_positive_support),
        "support_weighted_reliability": round(float(support_weighted_reliability(fraud_positive_count, min_positive_support)), 6),
        "fraud_rate": round(float(y.mean()), 6),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 6),
        "precision_ci_lower": round(float(precision_ci[0]), 6),
        "precision_ci_upper": round(float(precision_ci[1]), 6),
        "recall": round(float(recall_score(y, y_pred, zero_division=0)), 6),
        "recall_ci_lower": round(float(recall_ci[0]), 6),
        "recall_ci_upper": round(float(recall_ci[1]), 6),
        "false_positive_rate": round(float(fpr), 6),
        "false_positive_rate_ci_lower": round(float(fpr_ci[0]), 6),
        "false_positive_rate_ci_upper": round(float(fpr_ci[1]), 6),
        "flag_rate": round(float(y_flag.mean()), 6),
        "block_rate": round(float(y_pred.mean()), 6),
    }


def iter_rows(
    cohorts: Dict[str, pd.Series],
    labels: np.ndarray,
    pred_block: np.ndarray,
    pred_flag: np.ndarray,
    min_positive_support: int,
) -> Iterable[Dict[str, float | int | str]]:
    for name, mask in cohorts.items():
        yield metric_row(name, mask, labels, pred_block, pred_flag, min_positive_support)


def infer_decision_source(
    final_scores: np.ndarray,
    pred_flag: np.ndarray,
    pred_block: np.ndarray,
    approve_threshold: float,
) -> np.ndarray:
    # Offline approximation used for KPI split: score-only band, conservative low-history proxy,
    # hard-rule proxy at extreme-risk tails, and step-up proxy near approve boundary.
    source = np.full(final_scores.shape, "score_band", dtype=object)
    low_history_proxy = (final_scores >= approve_threshold) & (final_scores < (approve_threshold + 0.03))
    hard_rule_proxy = pred_block & (final_scores >= 0.97)
    step_up_proxy = pred_flag & (final_scores >= approve_threshold) & (final_scores < (approve_threshold + 0.02))

    source[low_history_proxy] = "low_history_policy"
    source[step_up_proxy] = "step_up_policy"
    source[hard_rule_proxy] = "hard_rule_override"
    return source


def build_decision_source_kpis(
    labels: np.ndarray,
    pred_flag: np.ndarray,
    pred_block: np.ndarray,
    decision_sources: np.ndarray,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for source in ["score_band", "hard_rule_override", "low_history_policy", "step_up_policy"]:
        mask = decision_sources == source
        total = int(mask.sum())
        if total == 0:
            rows.append(
                {
                    "decision_source": source,
                    "sample_count": 0,
                    "flag_rate": 0.0,
                    "confirmed_fraud_conversion": 0.0,
                    "false_positive_rate": 0.0,
                }
            )
            continue
        y = labels[mask]
        y_flag = pred_flag[mask]
        y_block = pred_block[mask]
        flagged_or_blocked = y_flag | y_block
        review_denominator = int(flagged_or_blocked.sum())
        confirmed_fraud = int((y == 1).sum())
        false_positive = int(((y == 0) & flagged_or_blocked).sum())
        rows.append(
            {
                "decision_source": source,
                "sample_count": total,
                "flag_rate": round(float(y_flag.mean()), 6),
                "confirmed_fraud_conversion": (
                    round(float(confirmed_fraud) / review_denominator, 6) if review_denominator else 0.0
                ),
                "false_positive_rate": (
                    round(float(false_positive) / review_denominator, 6) if review_denominator else 0.0
                ),
            }
        )
    return rows


def build_low_support_warnings(rows: List[Dict[str, float | int | str]], min_positive_support: int) -> List[Dict[str, int | str]]:
    warnings: List[Dict[str, int | str]] = []
    for row in rows:
        if row["metric_reliability"] == "low_support":
            warnings.append(
                {
                    "cohort": str(row["cohort"]),
                    "fraud_positive_count": int(row["fraud_positive_count"]),
                    "min_positive_support": min_positive_support,
                    "message": "Precision/recall may be unstable due to low fraud-positive support.",
                }
            )
    return warnings


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset_path)

    model = load_pickle(args.model_path)
    feature_columns = load_pickle(args.feature_path)
    thresholds = load_pickle(args.thresholds_path)
    approve_threshold = float(thresholds.get("approve_threshold", 0.30))
    block_threshold = float(thresholds.get("block_threshold", 0.90))

    df = pd.read_csv(dataset_path)
    if "Class" not in df.columns:
        raise ValueError("Dataset must include Class label column")

    df = ensure_context_columns(df)

    df = select_evaluation_frame(df, sample_size=args.sample_size, seed=args.seed, mode=args.evaluation_mode)

    x = build_model_input(df, feature_columns, args)
    labels = df["Class"].astype(int).to_numpy()

    base_scores = model.predict_proba(x)[:, 1]
    context_adjustment = compute_context_adjustment(df, cap=args.context_adjustment_max)
    final_scores = np.clip(base_scores + context_adjustment, 0.0, 1.0)

    pred_block = final_scores >= block_threshold
    pred_flag = (final_scores > approve_threshold) & (final_scores < block_threshold)
    decision_sources = infer_decision_source(final_scores, pred_flag, pred_block, approve_threshold)

    cohort_config = load_cohort_config(args.cohort_config_path)
    cohort_config_meta = cohort_config_metadata(cohort_config, args.cohort_config_path)
    cohorts = assign_cohorts_from_config(df, cohort_config)
    rows = list(iter_rows(cohorts, labels, pred_block, pred_flag, args.min_positive_support))
    results_df = pd.DataFrame(rows)

    warnings = build_low_support_warnings(rows, args.min_positive_support)
    decision_source_kpis = build_decision_source_kpis(labels, pred_flag, pred_block, decision_sources)

    output_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "thresholds": {
            "approve_threshold": approve_threshold,
            "block_threshold": block_threshold,
            "context_adjustment_max": args.context_adjustment_max,
            "min_positive_support": args.min_positive_support,
        },
        "sample_size": int(len(df)),
        "evaluation_mode": args.evaluation_mode,
        "cohort_definition": cohort_config_meta,
        "cohorts": rows,
        "decision_source_kpis": decision_source_kpis,
        "warnings": {
            "low_support_cohorts": warnings,
            "count": len(warnings),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    args.output_json.write_text(json.dumps(output_json, indent=2), encoding="utf-8")
    results_df.to_csv(args.output_csv, index=False)

    print("Cohort KPI report complete")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "sample_size": len(df),
                "low_support_warnings": len(warnings),
            },
            indent=2,
        )
    )
    print(f"Saved report JSON: {args.output_json}")
    print(f"Saved report CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

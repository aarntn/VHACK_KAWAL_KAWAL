import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.dataset_loader import load_creditcard, load_ieee_cis
from project.data.preprocessing import load_preprocessing_bundle, prepare_preprocessing_inputs, transform_with_bundle

DEFAULT_DATASET_PATH = REPO_ROOT / "project" / "legacy_creditcard" / "creditcard.csv"
DEFAULT_MODEL_PATH = REPO_ROOT / "project" / "models" / "final_xgboost_model.pkl"
DEFAULT_FEATURE_PATH = REPO_ROOT / "project" / "models" / "feature_columns.pkl"
DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "project" / "models" / "decision_thresholds.pkl"
DEFAULT_PREPROCESSING_ARTIFACT_PATH = REPO_ROOT / "project" / "models" / "preprocessing_artifact_promoted.pkl"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "project" / "outputs" / "governance" / "fairness_explainability_report.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "project" / "outputs" / "governance" / "fairness_explainability_report.md"
DEFAULT_SEGMENT_CSV = REPO_ROOT / "project" / "outputs" / "governance" / "fairness_segment_metrics.csv"
DEFAULT_SHAP_CSV = REPO_ROOT / "project" / "outputs" / "governance" / "model_top_feature_drivers.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Pass-5 fairness + explainability governance report.")
    parser.add_argument("--dataset-source", choices=["creditcard", "ieee_cis"], default="ieee_cis")
    parser.add_argument("--dataset-path", type=Path, help="Legacy credit-card CSV path (used when --dataset-source creditcard)")
    parser.add_argument("--ieee-transaction-path", type=Path)
    parser.add_argument("--ieee-identity-path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--preprocessing-artifact-path", type=Path, default=DEFAULT_PREPROCESSING_ARTIFACT_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--segment-output-csv", type=Path, default=DEFAULT_SEGMENT_CSV)
    parser.add_argument("--shap-output-csv", type=Path, default=DEFAULT_SHAP_CSV)
    parser.add_argument("--sample-size", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-fpr-gap", type=float, default=0.08)
    parser.add_argument("--max-recall-gap", type=float, default=0.12)
    parser.add_argument("--max-precision-gap", type=float, default=0.12)
    parser.add_argument("--min-segment-size", type=int, default=200)
    parser.add_argument("--top-features", type=int, default=20)
    return parser.parse_args()


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
        raise ValueError(f"Preprocessed feature columns missing from transformed fairness input: {preview}")
    return transformed_df.loc[:, feature_columns]


def ensure_segment_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    time_col = "TransactionDT" if "TransactionDT" in out.columns else "Time"
    amount_col = "TransactionAmt" if "TransactionAmt" in out.columns else "Amount"

    if "account_age_days" not in out.columns:
        out["account_age_days"] = (pd.to_numeric(out.get(time_col, 0), errors="coerce").fillna(0).astype(int) % 120) + 1

    if "channel" not in out.columns:
        out["channel"] = np.where(
            (pd.to_numeric(out.get(time_col, 0), errors="coerce").fillna(0).astype(int) % 10) < 2,
            "AGENT",
            "APP",
        )

    if "region" not in out.columns:
        if "is_cross_border" in out.columns:
            cross_border = out["is_cross_border"].astype(str).str.lower().isin(["1", "true", "yes"])
            out["region"] = np.where(cross_border, "cross_border", "domestic")
        else:
            out["region"] = "unknown"

    if "device_segment" not in out.columns:
        if "channel" in out.columns:
            out["device_segment"] = out["channel"].astype(str).str.upper().replace({"APP": "mobile_app", "AGENT": "agent"})
        else:
            out["device_segment"] = "unknown"

    return out


def assign_governance_segments(df: pd.DataFrame) -> Dict[str, pd.Series]:
    source = infer_dataset_source(df)
    account_age = pd.to_numeric(df["account_age_days"], errors="coerce").fillna(30)
    amount_col = "TransactionAmt" if "TransactionAmt" in df.columns else "Amount"
    amount = pd.to_numeric(df.get(amount_col, 0), errors="coerce").fillna(0.0)
    tx_channel = df["channel"].astype(str).str.upper()
    device_segment = df["device_segment"].astype(str).str.lower()
    region = df["region"].astype(str).str.lower()

    segments = {
        "cohort:new_users": account_age < 7,
        "cohort:established_users": account_age >= 7,
        "cohort:agent_small_ticket": (tx_channel == "AGENT") & (amount <= 250),
        "device:mobile_app": device_segment == "mobile_app",
        "device:agent": device_segment == "agent",
        "region:domestic": region == "domestic",
        "region:cross_border": region == "cross_border",
    }

    if source == "ieee_cis":
        # Prefer concrete cohorts from IEEE-CIS columns when available.
        if "DeviceType" in df.columns:
            device_type = df["DeviceType"].astype(str).str.lower()
            segments["ieee:device_mobile"] = device_type.eq("mobile")
            segments["ieee:device_desktop"] = device_type.eq("desktop")

        if "ProductCD" in df.columns:
            product_cd = df["ProductCD"].astype(str).str.upper()
            for product in sorted(product_cd.dropna().unique()):
                if product and product != "NAN":
                    segments[f"ieee:product_{product}"] = product_cd.eq(product)

        card_cols = [c for c in ["card1", "card2", "card3", "card4", "card5", "card6"] if c in df.columns]
        addr_cols = [c for c in ["addr1", "addr2"] if c in df.columns]
        email_cols = [c for c in ["P_emaildomain", "R_emaildomain"] if c in df.columns]

        card_present = pd.Series(False, index=df.index)
        for c in card_cols:
            card_present = card_present | df[c].notna()

        addr_present = pd.Series(False, index=df.index)
        for c in addr_cols:
            addr_present = addr_present | df[c].notna()

        email_present = pd.Series(False, index=df.index)
        for c in email_cols:
            email_present = email_present | df[c].notna()

        segments["ieee:identity_high_confidence"] = card_present & addr_present & email_present
        segments["ieee:identity_medium_confidence"] = card_present & (addr_present | email_present)
        segments["ieee:identity_low_confidence"] = (~card_present) | ((~addr_present) & (~email_present))

    return segments


def infer_dataset_source(df: pd.DataFrame) -> str:
    if "TransactionDT" in df.columns or "card1" in df.columns:
        return "ieee_cis"
    return "creditcard"


def assign_identity_buckets(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    from project.data.entity_identity import build_entity_id

    source = infer_dataset_source(df)
    entity_id, diagnostics = build_entity_id(df, source)

    tier = entity_id.astype(str).str.split(":", n=1).str[0]
    counts = entity_id.value_counts(dropna=False)
    entity_count = entity_id.map(counts).fillna(1).astype(int)

    unknown = entity_id.astype(str).str.startswith("fallback:")
    weak_tier = tier.isin(["ieee_card_only", "credit_user_proxy"])
    weak_linked = (~unknown) & ((entity_count < 3) | weak_tier)
    known = (~unknown) & (~weak_linked)

    bucket = pd.Series("unknown_no_entity", index=df.index, dtype="object")
    bucket.loc[weak_linked] = "uncertain_weakly_linked_entity_id"
    bucket.loc[known] = "known_high_confidence_entity_id"

    bucket_counts = bucket.value_counts(dropna=False).to_dict()
    diagnostics["identity_bucket_counts"] = {str(k): int(v) for k, v in bucket_counts.items()}
    diagnostics["identity_bucket_rules"] = {
        "known_high_confidence_entity_id": "non-fallback entity with >=3 occurrences and not weak-tier proxy",
        "uncertain_weakly_linked_entity_id": "non-fallback entity with <3 occurrences or weak-tier key",
        "unknown_no_entity": "fallback entity id (no usable identity key)",
    }

    return bucket, diagnostics


def compute_identity_bucket_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    pred_block: np.ndarray,
    identity_bucket: pd.Series,
) -> dict:
    overall = {
        "pr_auc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, pred_block, zero_division=0)),
        **metric_dict(labels, pred_block),
    }

    rows: list[dict[str, Any]] = []
    for bucket_name in [
        "known_high_confidence_entity_id",
        "uncertain_weakly_linked_entity_id",
        "unknown_no_entity",
    ]:
        idx = np.where(identity_bucket.to_numpy() == bucket_name)[0]
        if len(idx) == 0:
            continue
        y = labels[idx]
        s = scores[idx]
        p = pred_block[idx]
        row_metrics = {
            "pr_auc": float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else 0.0,
            "f1": float(f1_score(y, p, zero_division=0)),
            **metric_dict(y, p),
        }
        rows.append(
            {
                "bucket": bucket_name,
                "sample_count": int(len(idx)),
                "fraud_positive_count": int((y == 1).sum()),
                "pr_auc": round(row_metrics["pr_auc"], 6),
                "f1": round(row_metrics["f1"], 6),
                "precision": round(row_metrics["precision"], 6),
                "recall": round(row_metrics["recall"], 6),
                "false_positive_rate": round(row_metrics["false_positive_rate"], 6),
                "pr_auc_gap_vs_overall": round(row_metrics["pr_auc"] - overall["pr_auc"], 6),
                "f1_gap_vs_overall": round(row_metrics["f1"] - overall["f1"], 6),
                "precision_gap_vs_overall": round(row_metrics["precision"] - overall["precision"], 6),
                "recall_gap_vs_overall": round(row_metrics["recall"] - overall["recall"], 6),
                "fpr_gap_vs_overall": round(row_metrics["false_positive_rate"] - overall["false_positive_rate"], 6),
            }
        )

    gap_summary: dict[str, float] = {}
    for metric in ["pr_auc", "f1", "precision", "recall", "false_positive_rate"]:
        vals = [float(r[metric]) for r in rows]
        gap_summary[f"{metric}_max_gap"] = round((max(vals) - min(vals)) if vals else 0.0, 6)

    return {
        "overall": {k: round(float(v), 6) for k, v in overall.items()},
        "buckets": rows,
        "gap_summary": gap_summary,
    }


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = (fp / (fp + tn)) if (fp + tn) else 0.0
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fpr),
    }


def classify_segment_severity(
    row: Dict[str, Any],
    max_fpr_gap: float,
    max_fnr_gap: float,
    max_precision_gap: float,
) -> Dict[str, Any]:
    if row.get("is_low_support", False):
        return {"severity_label": "low_support", "violations": [], "severity_score": 0.0}

    if int(row.get("fraud_positive_count", 0)) == 0:
        return {"severity_label": "no_positive_labels", "violations": [], "severity_score": 0.0}

    fpr_gap = abs(float(row.get("fpr_gap_vs_overall", 0.0)))
    fnr_gap = abs(float(row.get("fnr_gap_vs_overall", 0.0)))
    precision_gap = abs(float(row.get("precision_gap_vs_overall", 0.0)))

    violations: list[str] = []
    if fpr_gap > max_fpr_gap:
        violations.append("fpr_gap")
    if fnr_gap > max_fnr_gap:
        violations.append("fnr_gap")
    if precision_gap > max_precision_gap:
        violations.append("precision_gap")

    warning = (
        fpr_gap > (0.5 * max_fpr_gap)
        or fnr_gap > (0.5 * max_fnr_gap)
        or precision_gap > (0.5 * max_precision_gap)
    )
    severity_label = "severe" if violations else ("warning" if warning else "ok")
    severity_score = max(
        fpr_gap / max(max_fpr_gap, 1e-12),
        fnr_gap / max(max_fnr_gap, 1e-12),
        precision_gap / max(max_precision_gap, 1e-12),
    )
    return {
        "severity_label": severity_label,
        "violations": violations,
        "severity_score": round(float(severity_score), 6),
    }


def compute_segment_metrics(
    labels: np.ndarray,
    pred_block: np.ndarray,
    segments: Dict[str, pd.Series],
    min_segment_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    overall = metric_dict(labels, pred_block)
    overall_fnr = 1.0 - float(overall["recall"])
    rows: List[Dict[str, Any]] = []
    for name, mask in segments.items():
        idx = np.where(mask.to_numpy())[0]
        if len(idx) == 0:
            continue
        y = labels[idx]
        p = pred_block[idx]
        metrics = metric_dict(y, p)
        row = {
            "segment": name,
            "sample_count": int(len(idx)),
            "fraud_positive_count": int((y == 1).sum()),
            "precision": round(metrics["precision"], 6),
            "recall": round(metrics["recall"], 6),
            "false_positive_rate": round(metrics["false_positive_rate"], 6),
            "false_negative_rate": round(1.0 - float(metrics["recall"]), 6),
            "fpr_gap_vs_overall": round(metrics["false_positive_rate"] - overall["false_positive_rate"], 6),
            "fnr_gap_vs_overall": round((1.0 - float(metrics["recall"])) - overall_fnr, 6),
            "recall_gap_vs_overall": round(metrics["recall"] - overall["recall"], 6),
            "precision_gap_vs_overall": round(metrics["precision"] - overall["precision"], 6),
            "is_low_support": len(idx) < min_segment_size,
        }
        rows.append(row)
    return rows, overall


def detect_disparities(
    rows: List[Dict[str, Any]],
    max_fpr_gap: float,
    max_recall_gap: float,
    max_precision_gap: float,
) -> List[Dict[str, Any]]:
    severe: List[Dict[str, Any]] = []
    for row in rows:
        severity = classify_segment_severity(
            row,
            max_fpr_gap=max_fpr_gap,
            max_fnr_gap=max_recall_gap,
            max_precision_gap=max_precision_gap,
        )
        row["severity_label"] = severity["severity_label"]
        row["severity_score"] = severity["severity_score"]
        if severity["severity_label"] != "severe":
            continue
        severe.append(
            {
                "segment": row["segment"],
                "violations": severity["violations"],
                "severity_label": severity["severity_label"],
                "severity_score": severity["severity_score"],
                "row": row,
            }
        )

    severe.sort(key=lambda x: (-float(x.get("severity_score", 0.0)), str(x.get("segment", ""))))
    return severe


def compute_feature_drivers(model: Any, x: pd.DataFrame, top_features: int, seed: int) -> Tuple[pd.DataFrame, str]:
    sampled = x
    if len(sampled) > 4000:
        sampled = sampled.sample(n=4000, random_state=seed)

    # Preferred path: SHAP package when available.
    try:
        import shap  # type: ignore

        explainer = shap.Explainer(model, sampled)
        shap_values = explainer(sampled)
        values = shap_values.values
        if values.ndim == 3:
            values = values[:, :, 1]
        mean_abs = np.abs(values).mean(axis=0)
        importance = pd.DataFrame({"feature": sampled.columns, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False
        )
        return importance.head(top_features).reset_index(drop=True), "shap_explainer"
    except Exception:
        pass

    # Fallback: XGBoost native pred_contribs is SHAP-compatible for tree models.
    try:
        import xgboost as xgb

        booster = model.get_booster() if hasattr(model, "get_booster") else None
        if booster is not None:
            contribs = booster.predict(xgb.DMatrix(sampled), pred_contribs=True)
            mean_abs = np.abs(contribs[:, :-1]).mean(axis=0)
            importance = pd.DataFrame({"feature": sampled.columns, "mean_abs_shap": mean_abs}).sort_values(
                "mean_abs_shap", ascending=False
            )
            return importance.head(top_features).reset_index(drop=True), "xgboost_pred_contribs"
    except Exception:
        pass

    # Last fallback for non-tree models
    if hasattr(model, "feature_importances_"):
        values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
        importance = pd.DataFrame({"feature": sampled.columns, "mean_abs_shap": np.abs(values)}).sort_values(
            "mean_abs_shap", ascending=False
        )
        return importance.head(top_features).reset_index(drop=True), "feature_importances_fallback"

    raise RuntimeError("Unable to compute explainability feature drivers for this model.")


def build_markdown_report(
    generated_at: str,
    overall: Dict[str, float],
    disparities: List[Dict[str, Any]],
    segment_rows: List[Dict[str, Any]],
    top_features_df: pd.DataFrame,
    explainability_method: str,
) -> str:
    lines = [
        "# Fairness + Explainability Governance Report",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "## Overall block-decision metrics",
        "",
        f"- Precision: `{overall['precision']:.4f}`",
        f"- Recall: `{overall['recall']:.4f}`",
        f"- FPR: `{overall['false_positive_rate']:.4f}`",
        "",
        "## Segment disparity assessment",
        "",
    ]

    if disparities:
        lines.extend(
            [
                "Severe disparity detected in the following segments:",
                "",
            ]
        )
        for item in disparities:
            lines.append(f"- `{item['segment']}` violations: `{', '.join(item['violations'])}`")
        lines.extend(
            [
                "",
                "Mitigation notes:",
                "- Review threshold-by-segment policy only where legally/compliance-approved.",
                "- Rebalance training data for impacted segments and rerun Pass 1/2/4/5.",
                "- Add segment-specific monitoring alerts for FPR/recall drift.",
            ]
        )
    else:
        lines.append("No severe segment disparity detected under configured gap thresholds.")

    if disparities:
        lines.extend(
            [
                "",
                "### Severe segment ranking (deterministic)",
                "",
                "| Rank | Segment | Severity score | Violations |",
                "| ---: | --- | ---: | --- |",
            ]
        )
        for idx, d in enumerate(disparities, start=1):
            lines.append(
                f"| {idx} | {d['segment']} | {float(d.get('severity_score', 0.0)):.3f} | {', '.join(d.get('violations', []))} |"
            )

    lines.extend(
        [
            "",
            "## Explainability (top drivers)",
            "",
            f"Method: `{explainability_method}`",
            "",
            "| Feature | Mean |SHAP| |",
            "| --- | ---: |",
        ]
    )
    for _, row in top_features_df.iterrows():
        lines.append(f"| {row['feature']} | {float(row['mean_abs_shap']):.6f} |")

    lines.extend(
        [
            "",
            "## Segment metrics sample",
            "",
            "| Segment | Samples | Precision | Recall | FPR | FNR | Severity |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in segment_rows[:10]:
        lines.append(
            f"| {row['segment']} | {row['sample_count']} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | {float(row['false_positive_rate']):.4f} | {float(row.get('false_negative_rate', 0.0)):.4f} | {row.get('severity_label', 'n/a')} |"
        )

    return "\n".join(lines) + "\n"


def sample_frame_and_labels(
    df: pd.DataFrame,
    labels: pd.Series,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if sample_size <= 0 or len(df) <= sample_size:
        return df.reset_index(drop=True), labels.reset_index(drop=True)

    sampled_index = df.sample(n=sample_size, random_state=seed).index
    sampled_df = df.loc[sampled_index].reset_index(drop=True)
    sampled_labels = labels.loc[sampled_index].reset_index(drop=True)
    return sampled_df, sampled_labels


def load_source(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series, dict]:
    if args.dataset_source == "creditcard":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for creditcard source (legacy assets now in project/legacy_creditcard)")
        return load_creditcard(args.dataset_path)

    if not args.ieee_transaction_path or not args.ieee_identity_path:
        raise ValueError("--ieee-transaction-path and --ieee-identity-path are required for ieee_cis source")
    return load_ieee_cis(args.ieee_transaction_path, args.ieee_identity_path)


def main() -> None:
    args = parse_args()

    model = load_pickle(args.model_path)
    feature_columns = load_pickle(args.feature_path)
    thresholds = load_pickle(args.thresholds_path)

    approve_threshold = float(thresholds.get("approve_threshold", 0.30))
    block_threshold = float(thresholds.get("block_threshold", 0.90))
    if not approve_threshold < block_threshold:
        raise ValueError("Threshold ordering violation: approve_threshold must be < block_threshold")

    if args.dataset_source == "ieee_cis":
        if not args.ieee_transaction_path or not args.ieee_identity_path:
            raise ValueError("--ieee-transaction-path and --ieee-identity-path are required for ieee_cis source")
        features_df, labels_series, _ = load_ieee_cis(args.ieee_transaction_path, args.ieee_identity_path)
    else:
        if not args.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
        features_df, labels_series, _ = load_creditcard(args.dataset_path)

    df = features_df.copy()

    df = ensure_segment_columns(df)
    labels_series = labels_series.copy()
    df, labels_series = sample_frame_and_labels(df, labels_series, args.sample_size, args.seed)

    x = build_model_input(df, feature_columns, args)
    labels = labels_series.astype(int).to_numpy()
    scores = model.predict_proba(x)[:, 1]
    pred_block = scores >= block_threshold

    identity_bucket, identity_bucket_diagnostics = assign_identity_buckets(df)

    segments = assign_governance_segments(df)
    segment_rows, overall = compute_segment_metrics(
        labels=labels,
        pred_block=pred_block,
        segments=segments,
        min_segment_size=args.min_segment_size,
    )
    identity_bucket_metrics = compute_identity_bucket_metrics(
        labels=labels,
        scores=scores,
        pred_block=pred_block,
        identity_bucket=identity_bucket,
    )

    disparities = detect_disparities(
        rows=segment_rows,
        max_fpr_gap=args.max_fpr_gap,
        max_recall_gap=args.max_recall_gap,
        max_precision_gap=args.max_precision_gap,
    )

    top_features_df, explainability_method = compute_feature_drivers(
        model=model,
        x=x,
        top_features=args.top_features,
        seed=args.seed,
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    markdown_report = build_markdown_report(
        generated_at=generated_at,
        overall=overall,
        disparities=disparities,
        segment_rows=segment_rows,
        top_features_df=top_features_df,
        explainability_method=explainability_method,
    )

    output = {
        "generated_at": generated_at,
        "dataset_source": args.dataset_source,
        "dataset_path": str(args.dataset_path),
        "ieee_transaction_path": str(args.ieee_transaction_path) if args.ieee_transaction_path else None,
        "ieee_identity_path": str(args.ieee_identity_path) if args.ieee_identity_path else None,
        "model_path": str(args.model_path),
        "feature_path": str(args.feature_path),
        "thresholds_path": str(args.thresholds_path),
        "thresholds": {
            "approve_threshold": approve_threshold,
            "block_threshold": block_threshold,
        },
        "overall_metrics": overall,
        "segment_policy": {
            "max_fpr_gap": args.max_fpr_gap,
            "max_recall_gap": args.max_recall_gap,
            "max_precision_gap": args.max_precision_gap,
            "min_segment_size": args.min_segment_size,
        },
        "segment_disparity": {
            "severe_count": len(disparities),
            "no_severe_disparity": len(disparities) == 0,
            "severe_segments": disparities,
            "mitigation": [
                "Review threshold-by-segment policy where compliance allows.",
                "Rebalance training data for impacted segments and rerun Pass 1/2/4/5.",
                "Enable drift alerts for segment FPR/recall gaps.",
            ],
        },
        "segments": segment_rows,
        "identity_bucket_metrics": identity_bucket_metrics,
        "identity_bucket_diagnostics": identity_bucket_diagnostics,
        "explainability": {
            "method": explainability_method,
            "top_features": top_features_df.to_dict(orient="records"),
            "candidate_model_explainability_generated": True,
        },
    }

    for p in [args.output_json, args.output_markdown, args.segment_output_csv, args.shap_output_csv]:
        p.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    args.output_json.expanduser().resolve().write_text(json.dumps(output, indent=2), encoding="utf-8")
    args.output_markdown.expanduser().resolve().write_text(markdown_report, encoding="utf-8")
    pd.DataFrame(segment_rows).to_csv(args.segment_output_csv.expanduser().resolve(), index=False)
    top_features_df.to_csv(args.shap_output_csv.expanduser().resolve(), index=False)

    print("Fairness + explainability governance report generated")
    print(f"- JSON: {args.output_json}")
    print(f"- Markdown: {args.output_markdown}")
    print(f"- Segment CSV: {args.segment_output_csv}")
    print(f"- Feature drivers CSV: {args.shap_output_csv}")
    print(f"- Severe segment disparities: {len(disparities)}")


if __name__ == "__main__":
    main()

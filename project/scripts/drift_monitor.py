import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_DATASET = REPO_ROOT / "project" / "legacy_creditcard" / "creditcard.csv"
DEFAULT_AUDIT_LOG = REPO_ROOT / "project" / "outputs" / "audit" / "fraud_audit_log.jsonl"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "project" / "outputs" / "monitoring" / "drift_report.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "project" / "outputs" / "monitoring" / "drift_feature_psi.csv"

# Defaults aligned to fields present in current privacy-safe audit records.
DEFAULT_FEATURE_COLUMNS = [
    "Amount",
    "base_model_score",
    "context_adjustment",
    "behavior_adjustment",
    "final_risk_score",
    "context_scores.device_risk_score",
    "context_scores.ip_risk_score",
    "context_scores.location_risk_score",
]
DEFAULT_DECISIONS = ["APPROVE", "FLAG", "BLOCK"]
EPS = 1e-9


@dataclass
class DriftThresholds:
    psi_warn: float
    psi_alert: float
    decision_drift_warn: float
    decision_drift_alert: float


@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute daily fraud drift metrics and alerting signals (PSI + decision distribution drift)."
    )
    parser.add_argument("--baseline-dataset", type=Path, default=DEFAULT_BASELINE_DATASET)
    parser.add_argument("--audit-log", type=Path, default=DEFAULT_AUDIT_LOG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--psi-warn", type=float, default=0.10)
    parser.add_argument("--psi-alert", type=float, default=0.25)
    parser.add_argument("--decision-drift-warn", type=float, default=0.10)
    parser.add_argument("--decision-drift-alert", type=float, default=0.20)
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=DEFAULT_FEATURE_COLUMNS,
        help="Columns compared with PSI. Missing columns are skipped.",
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--max-audit-records",
        type=int,
        default=25000,
        help="Cap the number of recent audit rows used for window estimates.",
    )
    parser.add_argument(
        "--audit-baseline-ratio",
        type=float,
        default=0.5,
        help="Fraction of audit rows used as baseline window when baseline dataset lacks a feature.",
    )
    return parser.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_series(values: Iterable[float]) -> pd.Series:
    s = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return s.astype(float)


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = _safe_series(expected)
    actual = _safe_series(actual)
    if expected.empty or actual.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(expected, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0

    expected_hist, _ = np.histogram(expected, bins=edges)
    actual_hist, _ = np.histogram(actual, bins=edges)

    expected_pct = expected_hist / max(1, expected_hist.sum())
    actual_pct = actual_hist / max(1, actual_hist.sum())

    expected_pct = np.clip(expected_pct, EPS, None)
    actual_pct = np.clip(actual_pct, EPS, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def classify_drift_level(value: float, warn_threshold: float, alert_threshold: float) -> str:
    if value >= alert_threshold:
        return "alert"
    if value >= warn_threshold:
        return "warn"
    return "ok"


def compute_decision_distribution(records: pd.DataFrame, decisions: List[str]) -> Dict[str, float]:
    if records.empty or "decision" not in records.columns:
        return {decision: 0.0 for decision in decisions}

    counts = records["decision"].value_counts(normalize=True)
    return {decision: float(counts.get(decision, 0.0)) for decision in decisions}


def decision_distribution_drift(
    baseline_distribution: Dict[str, float],
    current_distribution: Dict[str, float],
    decisions: List[str],
) -> float:
    # L1/TV-style drift score in [0, 1]
    l1 = sum(abs(current_distribution.get(d, 0.0) - baseline_distribution.get(d, 0.0)) for d in decisions)
    return float(0.5 * l1)


def load_baseline_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Baseline dataset not found: {path}")
    return pd.read_csv(path)


def load_audit_log(path: Path, max_records: int) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        return pd.DataFrame()

    # Flatten nested JSON keys like context_scores.device_risk_score.
    df = pd.json_normalize(rows, sep=".")
    if max_records > 0 and len(df) > max_records:
        df = df.tail(max_records).reset_index(drop=True)
    return df


def split_audit_windows(audit_df: pd.DataFrame, baseline_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if audit_df.empty:
        return audit_df, audit_df

    ratio = max(0.1, min(0.9, baseline_ratio))
    split_idx = max(1, min(len(audit_df) - 1, int(len(audit_df) * ratio)))
    baseline_window = audit_df.iloc[:split_idx].reset_index(drop=True)
    current_window = audit_df.iloc[split_idx:].reset_index(drop=True)
    return baseline_window, current_window


def evaluate_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: List[str],
    bins: int,
    thresholds: DriftThresholds,
    audit_baseline_df: pd.DataFrame | None = None,
) -> List[FeatureDriftResult]:
    results: List[FeatureDriftResult] = []

    for feature in feature_columns:
        expected_series = None
        actual_series = None

        if feature in baseline_df.columns and feature in current_df.columns:
            expected_series = baseline_df[feature]
            actual_series = current_df[feature]
        elif audit_baseline_df is not None and feature in audit_baseline_df.columns and feature in current_df.columns:
            expected_series = audit_baseline_df[feature]
            actual_series = current_df[feature]

        if expected_series is None or actual_series is None:
            continue

        psi = compute_psi(expected_series, actual_series, bins=bins)
        status = classify_drift_level(psi, thresholds.psi_warn, thresholds.psi_alert)
        results.append(FeatureDriftResult(feature=feature, psi=psi, status=status))

    return results


def build_recalibration_recommendation(feature_results: List[FeatureDriftResult], decision_drift_status: str) -> Dict[str, object]:
    psi_alert_features = [result.feature for result in feature_results if result.status == "alert"]
    psi_warn_features = [result.feature for result in feature_results if result.status == "warn"]

    should_recalibrate = bool(psi_alert_features) or decision_drift_status == "alert"
    priority = "high" if should_recalibrate else "medium" if psi_warn_features or decision_drift_status == "warn" else "low"

    reasons: List[str] = []
    if psi_alert_features:
        reasons.append(f"Feature PSI alert for: {', '.join(psi_alert_features)}")
    if decision_drift_status == "alert":
        reasons.append("Decision distribution drift exceeded alert threshold")
    if not reasons and (psi_warn_features or decision_drift_status == "warn"):
        reasons.append("Warning-level drift detected; monitor closely")
    if not reasons:
        reasons.append("No material drift detected")

    return {
        "should_recalibrate": should_recalibrate,
        "priority": priority,
        "reasons": reasons,
    }


def main() -> None:
    args = parse_args()
    thresholds = DriftThresholds(
        psi_warn=args.psi_warn,
        psi_alert=args.psi_alert,
        decision_drift_warn=args.decision_drift_warn,
        decision_drift_alert=args.decision_drift_alert,
    )

    baseline_df = load_baseline_dataset(args.baseline_dataset)
    audit_df = load_audit_log(args.audit_log, max_records=args.max_audit_records)
    audit_baseline_df, current_df = split_audit_windows(audit_df, baseline_ratio=args.audit_baseline_ratio)

    feature_results = evaluate_feature_drift(
        baseline_df=baseline_df,
        current_df=current_df,
        feature_columns=args.feature_columns,
        bins=args.bins,
        thresholds=thresholds,
        audit_baseline_df=audit_baseline_df,
    )

    if not audit_baseline_df.empty and "decision" in audit_baseline_df.columns:
        baseline_decisions = compute_decision_distribution(audit_baseline_df, decisions=DEFAULT_DECISIONS)
    else:
        baseline_decisions = compute_decision_distribution(
            baseline_df.rename(columns={"Class": "decision"}).assign(
                decision=lambda x: x["decision"].map({0: "APPROVE", 1: "BLOCK"})
            ),
            decisions=DEFAULT_DECISIONS,
        )
    current_decisions = compute_decision_distribution(current_df, decisions=DEFAULT_DECISIONS)
    decision_drift = decision_distribution_drift(
        baseline_distribution=baseline_decisions,
        current_distribution=current_decisions,
        decisions=DEFAULT_DECISIONS,
    )
    decision_drift_status = classify_drift_level(
        decision_drift,
        thresholds.decision_drift_warn,
        thresholds.decision_drift_alert,
    )

    recalibration = build_recalibration_recommendation(feature_results, decision_drift_status)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    feature_rows = [
        {"feature": result.feature, "psi": round(result.psi, 8), "status": result.status}
        for result in sorted(feature_results, key=lambda x: x.psi, reverse=True)
    ]
    pd.DataFrame(feature_rows).to_csv(args.output_csv, index=False)

    alert_count = sum(1 for r in feature_results if r.status == "alert")
    warn_count = sum(1 for r in feature_results if r.status == "warn")

    output = {
        "generated_at_utc": _utc_now_iso(),
        "inputs": {
            "baseline_dataset": str(args.baseline_dataset),
            "audit_log": str(args.audit_log),
            "max_audit_records": args.max_audit_records,
            "audit_baseline_ratio": args.audit_baseline_ratio,
            "feature_columns": args.feature_columns,
            "bins": args.bins,
        },
        "thresholds": {
            "psi_warn": thresholds.psi_warn,
            "psi_alert": thresholds.psi_alert,
            "decision_drift_warn": thresholds.decision_drift_warn,
            "decision_drift_alert": thresholds.decision_drift_alert,
        },
        "summary": {
            "baseline_row_count": int(len(baseline_df)),
            "audit_baseline_row_count": int(len(audit_baseline_df)),
            "current_row_count": int(len(current_df)),
            "feature_count_evaluated": len(feature_results),
            "feature_alert_count": alert_count,
            "feature_warn_count": warn_count,
        },
        "feature_drift": feature_rows,
        "decision_drift": {
            "score": round(decision_drift, 8),
            "status": decision_drift_status,
            "baseline_distribution": baseline_decisions,
            "current_distribution": current_decisions,
        },
        "recalibration_recommendation": recalibration,
    }

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Drift monitoring complete")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved drift report JSON: {args.output_json}")
    print(f"Saved feature PSI table: {args.output_csv}")


if __name__ == "__main__":
    main()

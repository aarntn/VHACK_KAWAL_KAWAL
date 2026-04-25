import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.dataset_loader import build_ieee_eda_gate


DEFAULT_OUTPUT_JSON = REPO_ROOT / "project" / "outputs" / "monitoring" / "ieee_profile_summary.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "project" / "outputs" / "monitoring" / "ieee_column_profile.csv"
TARGET_COL = "isFraud"
JOIN_KEY = "TransactionID"
HIGH_MISSING_THRESHOLD = 0.60
ULTRA_HIGH_CARDINALITY_THRESHOLD = 1000
TOP_N_VALUES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile IEEE-CIS transaction and identity datasets with reproducible summary outputs."
    )
    parser.add_argument("--transaction-path", type=Path, required=True, help="Path to train_transaction.csv")
    parser.add_argument("--identity-path", type=Path, required=True, help="Path to train_identity.csv")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Output summary JSON path")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output column-profile CSV path")
    parser.add_argument(
        "--allow-eda-failures",
        action="store_true",
        help="Continue even if basic EDA sanity checks fail (warning-only mode).",
    )
    return parser.parse_args()


def resolve_existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"{label} file not found: {resolved}")
    return resolved


def validate_columns(transaction_df: pd.DataFrame, identity_df: pd.DataFrame) -> None:
    missing = []
    if JOIN_KEY not in transaction_df.columns:
        missing.append(f"transaction.{JOIN_KEY}")
    if JOIN_KEY not in identity_df.columns:
        missing.append(f"identity.{JOIN_KEY}")
    if TARGET_COL not in transaction_df.columns:
        missing.append(f"transaction.{TARGET_COL}")
    if missing:
        raise ValueError("Required columns missing: " + ", ".join(missing))


def is_categorical(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype)


def needs_top_values(column_name: str, series: pd.Series) -> bool:
    if is_categorical(series):
        return True
    if column_name in {"P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo", "ProductCD"}:
        return True
    if column_name.startswith("card") or column_name.startswith("addr") or column_name.startswith("id_"):
        return True
    return False


def summarize_top_values(series: pd.Series, top_n: int) -> list[dict[str, object]]:
    if series.empty:
        return []
    counts = series.astype("string").fillna("<NA>").value_counts(dropna=False).head(top_n)
    return [{"value": str(k), "count": int(v)} for k, v in counts.items()]


def build_column_profile(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    rows = []
    top_value_summary: dict[str, list[dict[str, object]]] = {}
    high_risk_columns: list[dict[str, object]] = []

    row_count = len(df)

    for column in df.columns:
        series = df[column]
        non_null_count = int(series.notna().sum())
        missing_count = int(row_count - non_null_count)
        missing_ratio = float(missing_count / row_count) if row_count else 0.0
        unique_non_null = int(series.nunique(dropna=True))
        dtype_name = str(series.dtype)
        categorical_like = bool(is_categorical(series))

        risk_reasons = []
        if missing_ratio > HIGH_MISSING_THRESHOLD:
            risk_reasons.append(f"missing_ratio>{HIGH_MISSING_THRESHOLD:.2f}")
        if unique_non_null <= 1:
            risk_reasons.append("single_value_or_constant")
        if categorical_like and unique_non_null > ULTRA_HIGH_CARDINALITY_THRESHOLD:
            risk_reasons.append(f"high_cardinality>{ULTRA_HIGH_CARDINALITY_THRESHOLD}")

        rows.append(
            {
                "column": column,
                "dtype": dtype_name,
                "row_count": int(row_count),
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_ratio": round(missing_ratio, 6),
                "unique_non_null": unique_non_null,
                "is_categorical_like": categorical_like,
                "risk_flags": "|".join(risk_reasons),
            }
        )

        if needs_top_values(column, series):
            top_value_summary[column] = summarize_top_values(series, TOP_N_VALUES)

        if risk_reasons:
            high_risk_columns.append(
                {
                    "column": column,
                    "dtype": dtype_name,
                    "missing_ratio": round(missing_ratio, 6),
                    "unique_non_null": unique_non_null,
                    "risk_reasons": risk_reasons,
                }
            )

    profile_df = pd.DataFrame(rows).sort_values(by=["missing_ratio", "unique_non_null"], ascending=[False, False])
    return profile_df, top_value_summary, high_risk_columns


def target_distribution(merged_df: pd.DataFrame) -> dict[str, object]:
    label_series = pd.to_numeric(merged_df[TARGET_COL], errors="coerce")
    non_null = label_series.dropna()
    count_total = int(len(merged_df))
    count_labeled = int(len(non_null))
    fraud_positive_count = int((non_null == 1).sum())
    fraud_negative_count = int((non_null == 0).sum())
    positive_rate = float(fraud_positive_count / count_labeled) if count_labeled else 0.0

    return {
        "target_column": TARGET_COL,
        "total_rows": count_total,
        "labeled_rows": count_labeled,
        "fraud_positive_count": fraud_positive_count,
        "fraud_negative_count": fraud_negative_count,
        "fraud_positive_rate": round(positive_rate, 6),
    }


def main() -> None:
    args = parse_args()

    transaction_path = resolve_existing_file(args.transaction_path, "Transaction")
    identity_path = resolve_existing_file(args.identity_path, "Identity")
    output_json = args.output_json.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()

    transaction_df = pd.read_csv(transaction_path)
    identity_df = pd.read_csv(identity_path)
    validate_columns(transaction_df, identity_df)

    merged_df = transaction_df.merge(identity_df, how="left", on=JOIN_KEY, suffixes=("", "_identity"))

    profile_df, top_values, high_risk_columns = build_column_profile(merged_df)
    dtype_summary = profile_df.groupby("dtype", dropna=False).size().sort_values(ascending=False)
    eda_gate = build_ieee_eda_gate(transaction_df, identity_df, merged_df, transaction_df[TARGET_COL])

    summary = {
        "dataset": "ieee_cis",
        "paths": {
            "transaction_path": str(transaction_path),
            "identity_path": str(identity_path),
            "output_json": str(output_json),
            "output_csv": str(output_csv),
        },
        "join": {
            "join_key": JOIN_KEY,
            "join_type": "left",
            "transaction_row_count": int(len(transaction_df)),
            "identity_row_count": int(len(identity_df)),
            "merged_row_count": int(len(merged_df)),
        },
        "target_distribution": target_distribution(merged_df),
        "dtype_summary": {str(dtype): int(count) for dtype, count in dtype_summary.items()},
        "high_risk_columns": high_risk_columns,
        "eda_gate": eda_gate,
        "top_n_cardinality_columns": top_values,
        "high_risk_thresholds": {
            "missing_ratio_gt": HIGH_MISSING_THRESHOLD,
            "ultra_high_cardinality_gt": ULTRA_HIGH_CARDINALITY_THRESHOLD,
            "single_value_unique_non_null_lte": 1,
            "top_n_values": TOP_N_VALUES,
        },
    }

    if not eda_gate.get("passed", True):
        failed = ", ".join(eda_gate.get("failed_checks", [])) or "unknown"
        if args.allow_eda_failures:
            print(f"WARNING: EDA gate failed: {failed}")
        else:
            raise RuntimeError(
                f"EDA gate failed: {failed}. Use --allow-eda-failures to continue with warning-only mode."
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    profile_df.to_csv(output_csv, index=False)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("IEEE-CIS dataset profiling complete")
    print(f"- Merged rows: {len(merged_df)}")
    print(f"- Fraud positive rate: {summary['target_distribution']['fraud_positive_rate']:.6f}")
    print(f"- EDA gate passed: {eda_gate.get('passed', True)}")
    print(f"- Column profile CSV: {output_csv}")
    print(f"- Summary JSON: {output_json}")


if __name__ == "__main__":
    main()

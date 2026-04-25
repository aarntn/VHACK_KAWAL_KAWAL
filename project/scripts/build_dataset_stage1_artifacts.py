import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.feature_registry import map_to_canonical_features


DEFAULT_OUTPUT_DIR = REPO_ROOT / "project" / "outputs" / "dataset_stage1"
DEFAULT_CREDITCARD_PATH = REPO_ROOT / "project" / "legacy_creditcard" / "creditcard.csv"
DEFAULT_LABEL_COL = "isFraud"
JOIN_KEY = "TransactionID"
HEAD_ROWS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage-1 dataset onboarding artifacts (ingestion validation only)."
    )
    parser.add_argument(
        "--dataset-source",
        required=True,
        choices=["creditcard", "ieee_cis"],
        help="Dataset source to validate and profile.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_CREDITCARD_PATH,
        help="Path to creditcard CSV (required for --dataset-source creditcard).",
    )
    parser.add_argument(
        "--transaction-path",
        type=Path,
        help="Path to IEEE-CIS train_transaction.csv (required for --dataset-source ieee_cis).",
    )
    parser.add_argument(
        "--identity-path",
        type=Path,
        help="Path to IEEE-CIS train_identity.csv (required for --dataset-source ieee_cis).",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=DEFAULT_LABEL_COL,
        help="Target label column name for IEEE-CIS transaction data (default: isFraud).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for stage-1 artifacts.",
    )
    parser.add_argument(
        "--head-rows",
        type=int,
        default=HEAD_ROWS,
        help="Number of sample rows to save in *_head.csv artifacts.",
    )
    return parser.parse_args()


def resolve_existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"{label} file not found: {resolved}")
    return resolved


def ensure_binary_labels(labels: pd.Series, label_name: str) -> dict[str, Any]:
    non_null = labels.dropna()
    if non_null.empty:
        raise ValueError(f"Target column '{label_name}' is empty after dropping null values.")

    unique_values = sorted(non_null.unique().tolist())
    unique_set = set(unique_values)
    if unique_set - {0, 1}:
        raise ValueError(
            f"Target column '{label_name}' must be binary (0/1). Found values: {unique_values}"
        )

    return {
        "label_column": label_name,
        "unique_values": unique_values,
        "labeled_row_count": int(len(non_null)),
    }


def schema_payload(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns],
    }


def quality_payload(features: pd.DataFrame, labels: pd.Series, label_name: str) -> dict[str, Any]:
    label_counts = labels.value_counts(dropna=False).to_dict()
    total = int(len(labels))
    class_ratio = {
        str(label): {
            "count": int(count),
            "ratio": float(count / total) if total else 0.0,
        }
        for label, count in label_counts.items()
    }
    return {
        "row_count": int(len(features)),
        "missingness": features.isna().mean().to_dict(),
        "class_ratio": class_ratio,
        "target_column": label_name,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_creditcard_artifacts(args: argparse.Namespace, output_dir: Path) -> None:
    dataset_path = resolve_existing_file(args.dataset_path, "Creditcard dataset")
    df = pd.read_csv(dataset_path)

    if "Class" not in df.columns:
        raise ValueError("creditcard dataset must include target column 'Class'.")

    labels = pd.to_numeric(df["Class"], errors="coerce")
    target_check = ensure_binary_labels(labels, "Class")

    features = df.drop(columns=["Class"]).copy()
    canonical_df, _ = map_to_canonical_features(features, "creditcard")

    prefix = "creditcard"
    head_path = output_dir / f"{prefix}_head.csv"
    schema_path = output_dir / f"{prefix}_schema.json"
    quality_path = output_dir / f"{prefix}_quality.json"
    canonical_preview_path = output_dir / f"{prefix}_canonical_preview.csv"

    df.head(args.head_rows).to_csv(head_path, index=False)
    write_json(schema_path, schema_payload(df))
    write_json(
        quality_path,
        {
            **quality_payload(features, labels, "Class"),
            "acceptance_checks": {
                "target_column_exists": True,
                "target_non_empty": target_check["labeled_row_count"] > 0,
                "labels_binary": True,
            },
            "source_paths": {"dataset_path": str(dataset_path)},
        },
    )
    canonical_df.head(args.head_rows).to_csv(canonical_preview_path, index=False)


def build_ieee_artifacts(args: argparse.Namespace, output_dir: Path) -> None:
    if not args.transaction_path or not args.identity_path:
        raise ValueError("--transaction-path and --identity-path are required for --dataset-source ieee_cis")

    tx_path = resolve_existing_file(args.transaction_path, "IEEE transaction")
    id_path = resolve_existing_file(args.identity_path, "IEEE identity")

    tx_df = pd.read_csv(tx_path)
    id_df = pd.read_csv(id_path)

    if JOIN_KEY not in tx_df.columns:
        raise ValueError(f"Transaction file must include join key '{JOIN_KEY}'.")
    if JOIN_KEY not in id_df.columns:
        raise ValueError(f"Identity file must include join key '{JOIN_KEY}'.")
    if args.label_col not in tx_df.columns:
        raise ValueError(f"Transaction file must include target column '{args.label_col}'.")

    duplicate_join_keys = int(tx_df[JOIN_KEY].duplicated().sum())
    if duplicate_join_keys > 0:
        raise ValueError(
            f"Transaction table join key '{JOIN_KEY}' must be unique. Found {duplicate_join_keys} duplicates."
        )

    merged_df = tx_df.merge(id_df, how="left", on=JOIN_KEY, suffixes=("", "_identity"))

    labels = pd.to_numeric(merged_df[args.label_col], errors="coerce")
    target_check = ensure_binary_labels(labels, args.label_col)

    features = merged_df.drop(columns=[args.label_col]).copy()
    canonical_df, _ = map_to_canonical_features(features, "ieee_cis")

    prefix = "ieee_cis"
    head_path = output_dir / f"{prefix}_head.csv"
    schema_path = output_dir / f"{prefix}_schema.json"
    quality_path = output_dir / f"{prefix}_quality.json"
    canonical_preview_path = output_dir / f"{prefix}_canonical_preview.csv"

    merged_df.head(args.head_rows).to_csv(head_path, index=False)
    write_json(schema_path, schema_payload(merged_df))
    write_json(
        quality_path,
        {
            **quality_payload(features, labels, args.label_col),
            "acceptance_checks": {
                "target_column_exists": True,
                "target_non_empty": target_check["labeled_row_count"] > 0,
                "labels_binary": True,
                "join_key_unique_in_transaction": duplicate_join_keys == 0,
            },
            "join": {
                "join_key": JOIN_KEY,
                "join_type": "left",
                "transaction_row_count": int(len(tx_df)),
                "identity_row_count": int(len(id_df)),
                "merged_row_count": int(len(merged_df)),
            },
            "source_paths": {
                "transaction_path": str(tx_path),
                "identity_path": str(id_path),
            },
        },
    )
    canonical_df.head(args.head_rows).to_csv(canonical_preview_path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_source == "creditcard":
        build_creditcard_artifacts(args, output_dir)
        print("Stage-1 artifacts generated for creditcard dataset")
    else:
        build_ieee_artifacts(args, output_dir)
        print("Stage-1 artifacts generated for IEEE-CIS dataset")

    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

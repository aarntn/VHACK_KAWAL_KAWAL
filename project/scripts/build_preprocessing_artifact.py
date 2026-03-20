import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.dataset_loader import load_creditcard, load_ieee_cis
from project.data.preprocessing import (
    fit_preprocessing_bundle,
    prepare_preprocessing_inputs,
    save_preprocessing_bundle,
)

DEFAULT_DATASET_PATH = REPO_ROOT / "project" / "legacy_creditcard" / "creditcard.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "project" / "outputs" / "dataset_stage1"
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "project" / "models" / "preprocessing_artifact.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and persist preprocessing artifact from dataset loader outputs.")
    parser.add_argument(
        "--dataset-source",
        choices=["creditcard", "ieee_cis"],
        default="ieee_cis",
        help="Dataset source mode.",
    )
    parser.add_argument("--dataset-path", type=Path, help="Legacy credit-card CSV path (used when --dataset-source creditcard)")
    parser.add_argument("--ieee-transaction-path", type=Path)
    parser.add_argument("--ieee-identity-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--sample-rows", type=int, default=50)
    parser.add_argument("--scaler", choices=["standard", "robust"], default="standard")
    parser.add_argument("--categorical-encoding", choices=["onehot", "frequency"], default="onehot")
    parser.add_argument(
        "--include-passthrough-categorical",
        action="store_true",
        help="Include categorical passthrough columns in preprocessing pipeline.",
    )
    return parser.parse_args()


def load_source(args: argparse.Namespace):
    if args.dataset_source == "creditcard":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for creditcard source (legacy assets now in project/legacy_creditcard)")
        return load_creditcard(args.dataset_path)

    if not args.ieee_transaction_path or not args.ieee_identity_path:
        raise ValueError("--ieee-transaction-path and --ieee-identity-path are required for ieee_cis source.")
    return load_ieee_cis(args.ieee_transaction_path, args.ieee_identity_path)


def main() -> None:
    args = parse_args()
    features, labels, metadata = load_source(args)

    canonical_df, passthrough_df, behavior_diagnostics = prepare_preprocessing_inputs(features, args.dataset_source)
    bundle, transformed = fit_preprocessing_bundle(
        canonical_df=canonical_df,
        passthrough_df=passthrough_df,
        dataset_source=args.dataset_source,
        include_passthrough=args.include_passthrough_categorical,
        scaler=args.scaler,
        categorical_encoding=args.categorical_encoding,
        behavior_feature_diagnostics=behavior_diagnostics,
    )

    artifact_path = save_preprocessing_bundle(bundle, args.artifact_output)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_rows = min(args.sample_rows, transformed.shape[0])
    preview = pd.DataFrame(transformed[:preview_rows], columns=bundle.feature_names_out)
    preview_path = output_dir / f"{args.dataset_source}_preprocessed_preview.csv"
    preview.to_csv(preview_path, index=False)

    metadata_payload = {
        "dataset_source": args.dataset_source,
        "row_count": int(transformed.shape[0]),
        "transformed_feature_count": int(transformed.shape[1]),
        "canonical_column_count": int(len(bundle.canonical_columns)),
        "passthrough_categorical_column_count": int(len(bundle.passthrough_categorical_columns)),
        "include_passthrough_categorical": bundle.include_passthrough,
        "scaler": bundle.scaler,
        "categorical_encoding": bundle.categorical_encoding,
        "artifact_output": str(artifact_path),
        "preview_output": str(preview_path),
        "input_metadata_summary": {
            "row_count": metadata.get("row_count"),
            "dataset_source": metadata.get("dataset_source"),
        },
        "label_distribution": {
            str(k): int(v) for k, v in pd.Series(labels).value_counts(dropna=False).to_dict().items()
        },
        "behavior_feature_diagnostics": behavior_diagnostics,
        "transformed_preview_stats": {
            "mean_abs": float(np.mean(np.abs(transformed))) if transformed.size else 0.0,
            "std": float(np.std(transformed)) if transformed.size else 0.0,
        },
    }

    metadata_path = output_dir / f"{args.dataset_source}_preprocessing_artifact_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2)

    print("Preprocessing artifact build complete")
    print(f"- Artifact: {artifact_path}")
    print(f"- Preview: {preview_path}")
    print(f"- Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

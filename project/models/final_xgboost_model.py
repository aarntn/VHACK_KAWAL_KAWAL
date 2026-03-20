import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_IEEE_TRANSACTION_PATH = REPO_ROOT / "ieee-fraud-detection" / "train_transaction.csv"
DEFAULT_IEEE_IDENTITY_PATH = REPO_ROOT / "ieee-fraud-detection" / "train_identity.csv"
DEFAULT_MODEL_PATH = REPO_ROOT / "project" / "models" / "final_xgboost_model.pkl"
DEFAULT_FEATURES_PATH = REPO_ROOT / "project" / "models" / "feature_columns.pkl"
DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "project" / "models" / "decision_thresholds.pkl"
DEFAULT_PREPROCESSING_ARTIFACT_PATH = REPO_ROOT / "project" / "models" / "preprocessing_artifact.pkl"
DEFAULT_PREPROCESSED_MODEL_PATH = REPO_ROOT / "project" / "models" / "final_xgboost_model_preprocessed.pkl"
DEFAULT_PREPROCESSED_FEATURES_PATH = REPO_ROOT / "project" / "models" / "feature_columns_preprocessed.pkl"
DEFAULT_PREPROCESSED_THRESHOLDS_PATH = REPO_ROOT / "project" / "models" / "decision_thresholds_preprocessed.pkl"


def _resolve_path(arg_value: str | None, env_var: str, default_path: Path) -> Path:
    chosen = arg_value or os.getenv(env_var)
    return Path(chosen).expanduser().resolve() if chosen else default_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final XGBoost fraud model.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Legacy path to creditcard.csv (used only when --dataset-source creditcard).",
    )
    parser.add_argument(
        "--dataset-source",
        type=str,
        choices=["creditcard", "ieee_cis", "combined"],
        default="ieee_cis",
        help="Dataset source mode to load training data (defaults to ieee_cis).",
    )
    parser.add_argument(
        "--label-policy",
        choices=["transaction", "account_propagated"],
        default="transaction",
        help="Label policy used at dataset load time.",
    )
    parser.add_argument(
        "--ieee-transaction-path",
        type=str,
        help="Path to IEEE-CIS train_transaction.csv.",
    )
    parser.add_argument(
        "--ieee-identity-path",
        type=str,
        help="Path to IEEE-CIS train_identity.csv.",
    )
    parser.add_argument("--model-output", type=str, help="Output path for trained model pickle")
    parser.add_argument("--features-output", type=str, help="Output path for feature column pickle")
    parser.add_argument("--thresholds-output", type=str, help="Output path for threshold pickle")
    parser.add_argument(
        "--use-preprocessing",
        action="store_true",
        help="Use canonical preprocessing pipeline before training (default keeps legacy raw-feature behavior).",
    )
    parser.add_argument(
        "--include-passthrough-categorical",
        action="store_true",
        help="When preprocessing is enabled, include categorical passthrough columns.",
    )
    parser.add_argument(
        "--preprocessing-scaler",
        type=str,
        choices=["standard", "robust"],
        default="standard",
        help="Numeric scaler for preprocessing pipeline.",
    )
    parser.add_argument(
        "--preprocessing-categorical-encoding",
        type=str,
        choices=["onehot", "frequency"],
        default="onehot",
        help="Categorical encoding strategy for preprocessing pipeline.",
    )
    parser.add_argument(
        "--preprocessing-artifact-output",
        type=str,
        help="Output path for preprocessing artifact pickle.",
    )
    parser.add_argument(
        "--overwrite-runtime-artifacts",
        action="store_true",
        help="Allow preprocessing-enabled training to overwrite runtime API artifacts under project/models/*.pkl.",
    )
    parser.add_argument(
        "--feature-whitelist-file",
        type=str,
        help="Optional path to a whitelist file of stable feature names (json list/object or newline text).",
    )
    parser.add_argument(
        "--allow-empty-feature-whitelist",
        action="store_true",
        help="When whitelist file resolves to zero usable names, skip whitelist filtering instead of failing.",
    )
    parser.add_argument(
        "--adversarial-drop-list-file",
        type=str,
        help="Optional adversarial drop-list file (json/csv/text) to remove split-predictive features before training.",
    )
    parser.add_argument(
        "--adversarial-max-importance",
        type=float,
        help="When reading adversarial report JSON with ranked_features, drop features with combined importance above this threshold.",
    )
    return parser.parse_args()


def _print_dataset_summary(metadata: dict) -> None:
    print("\nDataset summary:")
    print(f"- Source: {metadata.get('dataset_source', 'unknown')}")
    print(f"- Row count: {metadata.get('row_count', 'n/a')}")

    class_balance = metadata.get("class_balance", {})
    if class_balance:
        print("- Class balance:")
        for label, stats in class_balance.items():
            count = stats.get("count", 0)
            rate = stats.get("rate", 0.0)
            print(f"  - class={label}: count={count}, rate={rate:.6f}")

    missingness = metadata.get("missingness", {})
    if missingness:
        avg_missingness = sum(float(v) for v in missingness.values()) / len(missingness)
        print(f"- Average feature missingness: {avg_missingness:.6f}")

    if metadata.get("dataset_source") == "ieee_cis":
        print(f"- Transaction rows: {metadata.get('transaction_row_count', 'n/a')}")
        print(f"- Identity rows: {metadata.get('identity_row_count', 'n/a')}")
        print(f"- Merged rows: {metadata.get('merged_row_count', 'n/a')}")


def _resolve_output_targets(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    model_output = _resolve_path(args.model_output, "FINAL_MODEL_OUTPUT", DEFAULT_MODEL_PATH)
    features_output = _resolve_path(args.features_output, "FEATURE_COLUMNS_OUTPUT", DEFAULT_FEATURES_PATH)
    thresholds_output = _resolve_path(args.thresholds_output, "DECISION_THRESHOLDS_OUTPUT", DEFAULT_THRESHOLDS_PATH)
    preprocessing_output = _resolve_path(
        args.preprocessing_artifact_output,
        "PREPROCESSING_ARTIFACT_OUTPUT",
        DEFAULT_PREPROCESSING_ARTIFACT_PATH,
    )

    if args.use_preprocessing and not args.overwrite_runtime_artifacts:
        uses_default_runtime_targets = (
            args.model_output is None
            and args.features_output is None
            and args.thresholds_output is None
            and args.preprocessing_artifact_output is None
        )
        if uses_default_runtime_targets:
            model_output = DEFAULT_PREPROCESSED_MODEL_PATH
            features_output = DEFAULT_PREPROCESSED_FEATURES_PATH
            thresholds_output = DEFAULT_PREPROCESSED_THRESHOLDS_PATH
            print(
                "\n[Safety] --use-preprocessing detected without explicit output paths. "
                "Writing to preprocessed artifact files to avoid breaking runtime API tests."
            )
            print(f"[Safety] model_output={model_output}")
            print(f"[Safety] features_output={features_output}")
            print(f"[Safety] thresholds_output={thresholds_output}")
            print(f"[Safety] preprocessing_artifact_output={preprocessing_output}")

    return model_output, features_output, thresholds_output, preprocessing_output


def _load_feature_whitelist(path: str | None, allow_empty: bool = False) -> list[str] | None:
    if not path:
        return None

    whitelist_path = Path(path).expanduser().resolve()
    content = whitelist_path.read_text(encoding="utf-8")

    if whitelist_path.suffix.lower() == ".json":
        payload = json.loads(content)
        if isinstance(payload, dict):
            if "kept_single_features" in payload and isinstance(payload["kept_single_features"], list):
                names = payload["kept_single_features"]
            elif (
                isinstance(payload.get("summary"), dict)
                and isinstance(payload["summary"].get("kept_single_features"), list)
            ):
                names = payload["summary"]["kept_single_features"]
            elif "feature_whitelist" in payload and isinstance(payload["feature_whitelist"], list):
                names = payload["feature_whitelist"]
            else:
                names = []
        elif isinstance(payload, list):
            names = payload
        else:
            names = []
    else:
        names = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]

    cleaned = [str(name) for name in names if str(name).strip()]
    if not cleaned:
        if allow_empty:
            print(
                "[Warning] Feature whitelist resolved empty; skipping whitelist filter due to "
                f"--allow-empty-feature-whitelist. Path: {whitelist_path}"
            )
            return None
        raise ValueError(
            "Feature whitelist file contains no usable feature names. "
            "Accepted JSON keys: kept_single_features, summary.kept_single_features, feature_whitelist. "
            f"Path: {whitelist_path}"
        )
    return cleaned


def _apply_feature_whitelist(X, whitelist: list[str] | None):
    if whitelist is None:
        return X

    missing = [name for name in whitelist if name not in X.columns]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing)-10} more)"
        raise ValueError(f"Feature whitelist includes columns not present in dataset: {preview}{suffix}")

    if not whitelist:
        raise ValueError("Feature whitelist resolved to an empty feature set.")

    return X.loc[:, whitelist]


def _load_adversarial_drop_features(path: str | None, max_importance: float | None) -> list[str] | None:
    if not path:
        return None

    drop_path = Path(path).expanduser().resolve()
    content = drop_path.read_text(encoding="utf-8")
    suffix = drop_path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(content)
        if isinstance(payload, dict):
            if isinstance(payload.get("drop_features"), list):
                names = payload["drop_features"]
            elif isinstance(payload.get("ranked_features"), list):
                threshold = 0.0 if max_importance is None else float(max_importance)
                names = [
                    row.get("feature")
                    for row in payload["ranked_features"]
                    if isinstance(row, dict) and row.get("importance_combined") is not None and float(row["importance_combined"]) > threshold
                ]
            else:
                names = []
        elif isinstance(payload, list):
            names = payload
        else:
            names = []
    elif suffix == ".csv":
        frame = pd.read_csv(drop_path)
        if "feature" not in frame.columns:
            raise ValueError(f"Adversarial drop CSV missing 'feature' column: {drop_path}")
        if "importance_combined" in frame.columns and max_importance is not None:
            frame = frame[frame["importance_combined"] > float(max_importance)]
        names = frame["feature"].astype(str).tolist()
    else:
        names = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]

    cleaned = sorted(set(str(name) for name in names if str(name).strip()))
    return cleaned or None


def _drop_adversarial_features(X, drop_features: list[str] | None):
    if not drop_features:
        return X
    keep_cols = [col for col in X.columns if col not in set(drop_features)]
    if not keep_cols:
        raise ValueError("Adversarial drop list removed all features; cannot continue training.")
    return X.loc[:, keep_cols]


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve() if args.dataset_path else None
    model_output, features_output, thresholds_output, preprocessing_output = _resolve_output_targets(args)

    # Delayed third-party imports so `--help` works even if ML deps are not installed.
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    from project.data.dataset_loader import load_combined, load_creditcard, load_ieee_cis

    print("=" * 60)
    print("TRAINING FINAL XGBOOST MODEL")
    print("=" * 60)

    if args.dataset_source == "ieee_cis":
        ieee_tx = _resolve_path(args.ieee_transaction_path, "IEEE_TRANSACTION_PATH", DEFAULT_IEEE_TRANSACTION_PATH)
        ieee_id = _resolve_path(args.ieee_identity_path, "IEEE_IDENTITY_PATH", DEFAULT_IEEE_IDENTITY_PATH)
        X, y, metadata = load_ieee_cis(
            ieee_tx,
            ieee_id,
            label_policy=args.label_policy,
        )
    elif args.dataset_source == "creditcard":
        if dataset_path is None:
            raise ValueError("--dataset-path is required when --dataset-source creditcard")
        X, y, metadata = load_creditcard(dataset_path, label_policy=args.label_policy)
    else:
        X, y, metadata = load_combined()

    _print_dataset_summary(metadata)

    adversarial_drop_features = _load_adversarial_drop_features(
        args.adversarial_drop_list_file,
        args.adversarial_max_importance,
    )
    X = _drop_adversarial_features(X, adversarial_drop_features)
    if adversarial_drop_features:
        print(f"- Adversarial drop-list enabled: dropped {len(adversarial_drop_features)} columns")

    feature_whitelist = _load_feature_whitelist(args.feature_whitelist_file, allow_empty=args.allow_empty_feature_whitelist)
    X = _apply_feature_whitelist(X, feature_whitelist)
    if feature_whitelist is not None:
        print(f"- Feature whitelist enabled: {len(feature_whitelist)} columns")

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    feature_names_to_save = X.columns.tolist()

    if args.use_preprocessing:
        from project.data.preprocessing import (
            fit_preprocessing_bundle,
            prepare_preprocessing_inputs,
            save_preprocessing_bundle,
        )

        canonical_train, passthrough_train, behavior_diagnostics = prepare_preprocessing_inputs(X_train, args.dataset_source)
        bundle, X_train_processed = fit_preprocessing_bundle(
            canonical_df=canonical_train,
            passthrough_df=passthrough_train,
            dataset_source=args.dataset_source,
            include_passthrough=args.include_passthrough_categorical,
            scaler=args.preprocessing_scaler,
            categorical_encoding=args.preprocessing_categorical_encoding,
            behavior_feature_diagnostics=behavior_diagnostics,
        )
        save_preprocessing_bundle(bundle, preprocessing_output)
        feature_names_to_save = bundle.feature_names_out
        X_train_for_model = X_train_processed

        print("\nPreprocessing enabled")
        print(f"- Preprocessing artifact: {preprocessing_output}")
        print(f"- Transformed train shape: {X_train_processed.shape}")
        print(f"- Include passthrough categorical: {bundle.include_passthrough}")
        print(f"- Categorical encoding: {bundle.categorical_encoding}")
        print(f"- Scaler: {bundle.scaler}")
        print(f"- Behavior features: {len(behavior_diagnostics.get('feature_columns', []))}")
    else:
        X_train_for_model = X_train
        print("\nPreprocessing disabled (legacy raw-feature training mode)")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"\nscale_pos_weight = {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train_for_model, y_train)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    features_output.parent.mkdir(parents=True, exist_ok=True)
    thresholds_output.parent.mkdir(parents=True, exist_ok=True)

    with model_output.open("wb") as f:
        pickle.dump(model, f)

    with features_output.open("wb") as f:
        pickle.dump(feature_names_to_save, f)

    with thresholds_output.open("wb") as f:
        pickle.dump({"approve_threshold": 0.30, "block_threshold": 0.90}, f)

    print("\nSaved files:")
    print(f"- {model_output}")
    print(f"- {features_output}")
    print(f"- {thresholds_output}")
    if args.use_preprocessing:
        print(f"- {preprocessing_output}")


if __name__ == "__main__":
    main()

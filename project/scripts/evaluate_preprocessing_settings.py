import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.dataset_loader import enforce_eda_gate, load_creditcard, load_ieee_cis
from project.data.behavior_features import get_uid_candidates_for_source
from project.data.entity_aggregation import build_uid
from project.data.preprocessing import (
    fit_preprocessing_bundle,
    prepare_preprocessing_inputs,
    transform_with_bundle,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "project" / "outputs" / "monitoring"


@dataclass
class EvalResult:
    preprocessing_setting: str
    split_mode: str
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    pr_auc: float
    roc_auc: float
    tn: int
    fp: int
    fn: int
    tp: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate preprocessing settings with threshold tuning on a leakage-safe split."
    )
    parser.add_argument("--dataset-source", choices=["creditcard", "ieee_cis"], default="ieee_cis")
    parser.add_argument("--label-policy", choices=["transaction", "account_propagated"], default="transaction")
    parser.add_argument("--dataset-path", type=Path, help="Legacy credit-card CSV path (used when --dataset-source creditcard)")
    parser.add_argument("--ieee-transaction-path", type=Path)
    parser.add_argument("--ieee-identity-path", type=Path)
    parser.add_argument("--split-mode", choices=["time", "random", "groupkfold_uid"], default="time")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--groupkfold-n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument(
        "--feature-decisions-json",
        type=Path,
        help="Optional time_consistency_feature_decisions.json used to whitelist stable single features.",
    )
    parser.add_argument(
        "--feature-whitelist-file",
        type=Path,
        help="Optional newline-delimited feature whitelist. Overrides --feature-decisions-json when both are set.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-eda-failures",
        action="store_true",
        help="Continue even if dataset EDA gate checks fail (warning-only mode).",
    )
    return parser.parse_args()


def load_source(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series, dict]:
    if args.dataset_source == "creditcard":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for creditcard source (legacy assets now in project/legacy_creditcard)")
        x, y, metadata = load_creditcard(args.dataset_path, label_policy=args.label_policy)
        return x, y, metadata

    if not args.ieee_transaction_path or not args.ieee_identity_path:
        raise ValueError("--ieee-transaction-path and --ieee-identity-path are required for ieee_cis source")
    x, y, metadata = load_ieee_cis(
        args.ieee_transaction_path,
        args.ieee_identity_path,
        label_policy=args.label_policy,
    )
    return x, y, metadata


def resolve_event_time_column(features: pd.DataFrame, dataset_source: str) -> pd.Series:
    col = "TransactionDT" if dataset_source == "ieee_cis" else "Time"
    if col in features.columns:
        return pd.to_numeric(features[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    return pd.Series(np.zeros(len(features), dtype=float), index=features.index)


def time_based_split(
    x: pd.DataFrame,
    y: pd.Series,
    event_time: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    order = np.argsort(event_time.to_numpy(), kind="mergesort")
    cutoff = int((1.0 - test_size) * len(order))
    cutoff = max(1, min(cutoff, len(order) - 1))

    train_idx = x.index[order[:cutoff]]
    test_idx = x.index[order[cutoff:]]

    return x.loc[train_idx], x.loc[test_idx], y.loc[train_idx], y.loc[test_idx]


def random_split(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)




def groupkfold_uid_split(
    x: pd.DataFrame,
    y: pd.Series,
    dataset_source: str,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    time_col = "TransactionDT" if dataset_source == "ieee_cis" else "Time"
    uid_candidates = get_uid_candidates_for_source(dataset_source)
    uid = build_uid(x, time_col=time_col, uid_candidates=uid_candidates)
    unique_groups = uid.nunique(dropna=False)
    folds = max(2, min(int(n_splits), int(unique_groups)))
    splitter = GroupKFold(n_splits=folds)

    chosen = None
    for train_idx, test_idx in splitter.split(x, y, groups=uid):
        chosen = (train_idx, test_idx)
    if chosen is None:
        raise ValueError("Unable to build GroupKFold UID split")
    train_idx, test_idx = chosen
    train_labels = y.iloc[train_idx]
    test_labels = y.iloc[test_idx]
    if train_labels.nunique() < 2 or test_labels.nunique() < 2:
        raise ValueError("GroupKFold UID split produced single-class train/test partition")
    return x.iloc[train_idx], x.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def load_feature_whitelist(
    whitelist_file: Path | None,
    feature_decisions_json: Path | None,
    available_columns: list[str],
) -> tuple[list[str], str | None]:
    source: str | None = None
    requested: list[str] | None = None
    available_set = set(available_columns)

    if whitelist_file is not None:
        source = "feature_whitelist_file"
        lines = whitelist_file.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        requested = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    elif feature_decisions_json is not None:
        source = "feature_decisions_json"
        payload = json.loads(feature_decisions_json.expanduser().resolve().read_text(encoding="utf-8"))
        summary = payload.get("summary") if isinstance(payload, dict) else {}
        kept = summary.get("kept_single_features") if isinstance(summary, dict) else None
        if not isinstance(kept, list):
            raise ValueError("feature-decisions-json missing summary.kept_single_features list")
        requested = [str(col) for col in kept]

    if requested is None:
        return available_columns, source

    filtered = [col for col in requested if col in available_set]
    if not filtered:
        raise ValueError("Feature whitelist filtering removed all columns; nothing left to evaluate.")
    return filtered, source


def feature_whitelist_hash(selected_columns: list[str], active: bool) -> str | None:
    if not active:
        return None
    canonical = "\n".join(sorted(selected_columns))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evaluate_thresholds(y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray, setting: str, split_mode: str) -> list[EvalResult]:
    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    rows: list[EvalResult] = []

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            EvalResult(
                preprocessing_setting=setting,
                split_mode=split_mode,
                threshold=float(threshold),
                precision=float(precision_score(y_true, y_pred, zero_division=0)),
                recall=float(recall_score(y_true, y_pred, zero_division=0)),
                f1=float(f1_score(y_true, y_pred, zero_division=0)),
                false_positive_rate=float(false_positive_rate(y_true, y_pred)),
                pr_auc=float(pr_auc),
                roc_auc=float(roc_auc),
                tn=int(tn),
                fp=int(fp),
                fn=int(fn),
                tp=int(tp),
            )
        )
    return rows


def orient_scores_for_positive_class(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    """Orient scores so larger values correspond to higher positive-class likelihood.

    In drift-heavy temporal splits a model can become anti-correlated on the holdout
    (ROC-AUC < 0.5). For threshold sweeps used by promotion policies, use the
    monotonic inverse score (1 - p) in that case so candidate selection reflects
    the best available ranking direction.
    """
    unique = np.unique(y_true)
    if len(unique) < 2:
        return y_score, {
            "score_inversion_applied": False,
            "roc_auc_raw": 0.5,
            "roc_auc_oriented": 0.5,
            "orientation": "identity",
        }

    roc_raw = float(roc_auc_score(y_true, y_score))
    if roc_raw < 0.5:
        flipped = 1.0 - y_score
        roc_flipped = float(roc_auc_score(y_true, flipped))
        return flipped, {
            "score_inversion_applied": True,
            "roc_auc_raw": roc_raw,
            "roc_auc_oriented": roc_flipped,
            "orientation": "inverted",
        }

    return y_score, {
        "score_inversion_applied": False,
        "roc_auc_raw": roc_raw,
        "roc_auc_oriented": roc_raw,
        "orientation": "identity",
    }


def run_setting(
    setting_name: str,
    categorical_encoding: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_source: str,
    args: argparse.Namespace,
    thresholds: np.ndarray,
) -> list[EvalResult]:
    canonical_train, passthrough_train, _ = prepare_preprocessing_inputs(x_train, dataset_source)
    bundle, x_train_transformed = fit_preprocessing_bundle(
        canonical_df=canonical_train,
        passthrough_df=passthrough_train,
        dataset_source=dataset_source,
        include_passthrough=True,
        scaler="robust",
        categorical_encoding=categorical_encoding,
    )

    canonical_test, passthrough_test, _ = prepare_preprocessing_inputs(x_test, dataset_source)
    x_test_transformed = transform_with_bundle(bundle, canonical_test, passthrough_test)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = (neg / pos) if pos else 1.0

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=args.seed,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(x_train_transformed, y_train)
    y_score_raw = model.predict_proba(x_test_transformed)[:, 1]
    y_score, _ = orient_scores_for_positive_class(y_test.to_numpy(), y_score_raw)

    return evaluate_thresholds(
        y_true=y_test.to_numpy(),
        y_score=y_score,
        thresholds=thresholds,
        setting=setting_name,
        split_mode=args.split_mode,
    )


def to_markdown(best_df: pd.DataFrame) -> str:
    headers = ["setting", "threshold", "f1", "recall", "precision", "fpr", "pr_auc", "roc_auc", "tn", "fp", "fn", "tp"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in best_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["preprocessing_setting"]),
                    f"{row['threshold']:.2f}",
                    f"{row['f1']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['false_positive_rate']:.4f}",
                    f"{row['pr_auc']:.4f}",
                    f"{row['roc_auc']:.4f}",
                    str(int(row["tn"])),
                    str(int(row["fp"])),
                    str(int(row["fn"])),
                    str(int(row["tp"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    x, y, source_metadata = load_source(args)
    enforce_eda_gate(
        source_metadata,
        allow_failures=args.allow_eda_failures,
        context=f"{Path(__file__).name}::{args.dataset_source}",
    )

    selected_columns, whitelist_source = load_feature_whitelist(
        whitelist_file=args.feature_whitelist_file,
        feature_decisions_json=args.feature_decisions_json,
        available_columns=x.columns.tolist(),
    )
    whitelist_active = bool(args.feature_whitelist_file or args.feature_decisions_json)
    if whitelist_active:
        x = x[selected_columns].copy()

    if args.split_mode == "time":
        event_time = resolve_event_time_column(x, args.dataset_source)
        x_train, x_test, y_train, y_test = time_based_split(x, y, event_time, args.test_size)
    elif args.split_mode == "groupkfold_uid":
        x_train, x_test, y_train, y_test = groupkfold_uid_split(
            x,
            y,
            dataset_source=args.dataset_source,
            n_splits=args.groupkfold_n_splits,
        )
    else:
        x_train, x_test, y_train, y_test = random_split(x, y, args.test_size, args.seed)

    thresholds = np.arange(args.threshold_start, args.threshold_stop + 1e-9, args.threshold_step)

    all_rows: list[EvalResult] = []
    all_rows.extend(
        run_setting(
            setting_name="onehot_robust",
            categorical_encoding="onehot",
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            dataset_source=args.dataset_source,
            args=args,
            thresholds=thresholds,
        )
    )
    all_rows.extend(
        run_setting(
            setting_name="frequency_robust",
            categorical_encoding="frequency",
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            dataset_source=args.dataset_source,
            args=args,
            thresholds=thresholds,
        )
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_df = pd.DataFrame([asdict(row) for row in all_rows])
    rows_df["feature_whitelist_active"] = whitelist_active
    rows_df["selected_feature_count"] = int(len(selected_columns))
    rows_df["label_policy"] = args.label_policy
    full_csv = output_dir / f"{args.dataset_source}_preprocessing_threshold_comparison.csv"
    rows_df.to_csv(full_csv, index=False)

    best_df = rows_df.sort_values("f1", ascending=False).groupby("preprocessing_setting", as_index=False).head(1)
    best_df = best_df.sort_values("preprocessing_setting")

    best_csv = output_dir / f"{args.dataset_source}_preprocessing_threshold_best.csv"
    best_df.to_csv(best_csv, index=False)

    md = to_markdown(best_df)
    md_path = output_dir / f"{args.dataset_source}_preprocessing_threshold_summary.md"
    md_path.write_text(md, encoding="utf-8")

    meta = {
        "dataset_source": args.dataset_source,
        "split_mode": args.split_mode,
        "test_size": args.test_size,
        "threshold_start": args.threshold_start,
        "threshold_stop": args.threshold_stop,
        "threshold_step": args.threshold_step,
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "label_policy": args.label_policy,
        "label_policy_diagnostics": source_metadata.get("label_policy_diagnostics"),
        "eda_gate": source_metadata.get("eda_gate"),
        "eda_gate_status": "passed" if source_metadata.get("eda_gate", {}).get("passed", True) else "failed",
        "feature_whitelist_active": whitelist_active,
        "feature_whitelist_source": whitelist_source,
        "selected_feature_count": int(len(selected_columns)),
        "feature_whitelist_hash": feature_whitelist_hash(selected_columns, active=whitelist_active),
        "full_csv": str(full_csv),
        "best_csv": str(best_csv),
        "summary_markdown": str(md_path),
    }
    meta_path = output_dir / f"{args.dataset_source}_preprocessing_threshold_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Preprocessing setting comparison complete")
    print(f"- Full thresholds CSV: {full_csv}")
    print(f"- Best-per-setting CSV: {best_csv}")
    print(f"- Markdown summary: {md_path}")
    print(f"- Metadata JSON: {meta_path}")


if __name__ == "__main__":
    main()

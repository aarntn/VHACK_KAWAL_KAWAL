import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.dataset_loader import load_creditcard, load_ieee_cis

DEFAULT_OUTPUT_DIR = REPO_ROOT / "project" / "outputs" / "monitoring"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train adversarial models to identify split-predictive features (train-vs-validation drift)."
    )
    parser.add_argument("--dataset-source", choices=["creditcard", "ieee_cis"], default="ieee_cis")
    parser.add_argument("--label-policy", choices=["transaction", "account_propagated"], default="transaction")
    parser.add_argument("--dataset-path", type=Path, help="Legacy credit-card CSV path (used when --dataset-source creditcard)")
    parser.add_argument("--ieee-transaction-path", type=Path)
    parser.add_argument("--ieee-identity-path", type=Path)
    parser.add_argument("--split-mode", choices=["time", "random"], default="time")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-csv", type=Path, help="Optional explicit early/train feature CSV.")
    parser.add_argument("--test-csv", type=Path, help="Optional explicit late/holdout feature CSV.")
    parser.add_argument(
        "--max-adversarial-importance",
        type=float,
        help="Optional threshold on combined importance to auto-generate drop feature list.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_event_time_column(features: pd.DataFrame, dataset_source: str) -> pd.Series:
    col = "TransactionDT" if dataset_source == "ieee_cis" else "Time"
    if col in features.columns:
        return pd.to_numeric(features[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    return pd.Series(np.arange(len(features), dtype=float), index=features.index)


def time_based_split_indices(event_time: pd.Series, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(event_time.to_numpy(), kind="mergesort")
    cutoff = int((1.0 - test_size) * len(order))
    cutoff = max(1, min(cutoff, len(order) - 1))
    return order[:cutoff], order[cutoff:]


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].fillna("<NA>").astype(str)
            out[col] = pd.factorize(out[col])[0]
        elif str(out[col].dtype).startswith("category"):
            out[col] = out[col].cat.codes
    out = out.apply(pd.to_numeric, errors="coerce")
    return out.fillna(out.median(numeric_only=True)).fillna(0.0)


def _load_source(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    if args.dataset_source == "creditcard":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for creditcard source (legacy assets now in project/legacy_creditcard)")
        x, _, metadata = load_creditcard(args.dataset_path, label_policy=args.label_policy)
        return x, metadata
    if not args.ieee_transaction_path or not args.ieee_identity_path:
        raise ValueError("--ieee-transaction-path and --ieee-identity-path are required for ieee_cis source")
    x, _, metadata = load_ieee_cis(
        args.ieee_transaction_path,
        args.ieee_identity_path,
        label_policy=args.label_policy,
    )
    return x, metadata


def build_adversarial_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, dict]:
    if bool(args.train_csv) ^ bool(args.test_csv):
        raise ValueError("--train-csv and --test-csv must be provided together when using explicit split files")

    if args.train_csv and args.test_csv:
        train_df = pd.read_csv(args.train_csv.expanduser().resolve())
        test_df = pd.read_csv(args.test_csv.expanduser().resolve())
        train_df = train_df.drop(columns=[c for c in ["Class", "is_fraud", "label"] if c in train_df.columns], errors="ignore")
        test_df = test_df.drop(columns=[c for c in ["Class", "is_fraud", "label"] if c in test_df.columns], errors="ignore")
        common = [c for c in train_df.columns if c in set(test_df.columns)]
        if not common:
            raise ValueError("Explicit train/test CSVs do not share common feature columns")
        train_df = train_df[common]
        test_df = test_df[common]
        x_adv = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        y_adv = np.concatenate([np.zeros(len(train_df), dtype=int), np.ones(len(test_df), dtype=int)])
        meta = {
            "split_source": "explicit_files",
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "feature_count": int(len(common)),
        }
        meta["label_policy"] = args.label_policy
        return x_adv, y_adv, meta

    x, source_metadata = _load_source(args)
    if args.split_mode == "time":
        event_time = resolve_event_time_column(x, args.dataset_source)
        train_idx, test_idx = time_based_split_indices(event_time, args.test_size)
    else:
        idx = np.arange(len(x))
        train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed)

    x_train = x.iloc[train_idx]
    x_test = x.iloc[test_idx]
    x_adv = pd.concat([x_train, x_test], axis=0, ignore_index=True)
    y_adv = np.concatenate([np.zeros(len(x_train), dtype=int), np.ones(len(x_test), dtype=int)])
    meta = {
        "split_source": "dataset_split",
        "split_mode": args.split_mode,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": int(x.shape[1]),
    }
    meta["label_policy"] = args.label_policy
    meta["label_policy_diagnostics"] = source_metadata.get("label_policy_diagnostics")
    return x_adv, y_adv, meta


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0:
        return np.zeros_like(values, dtype=float)
    return values / total


def compute_ranked_importance(x_adv: pd.DataFrame, y_adv: np.ndarray, seed: int) -> tuple[pd.DataFrame, dict]:
    x_num = _coerce_numeric(x_adv)

    log_model = LogisticRegression(max_iter=500, random_state=seed)
    log_model.fit(x_num, y_adv)
    log_score = log_model.predict_proba(x_num)[:, 1]

    tree_model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=seed, n_jobs=1)
    tree_model.fit(x_num, y_adv)
    tree_score = tree_model.predict_proba(x_num)[:, 1]

    log_imp = np.abs(log_model.coef_[0]).astype(float)
    tree_imp = tree_model.feature_importances_.astype(float)
    log_norm = _normalize(log_imp)
    tree_norm = _normalize(tree_imp)
    combined = 0.5 * (log_norm + tree_norm)

    ranked = pd.DataFrame(
        {
            "feature": x_num.columns.tolist(),
            "importance_combined": combined,
            "importance_logistic": log_norm,
            "importance_tree": tree_norm,
        }
    ).sort_values(["importance_combined", "importance_tree", "importance_logistic", "feature"], ascending=[False, False, False, True])
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    metrics = {
        "logistic": {
            "roc_auc": float(roc_auc_score(y_adv, log_score)),
            "pr_auc": float(average_precision_score(y_adv, log_score)),
        },
        "tree": {
            "roc_auc": float(roc_auc_score(y_adv, tree_score)),
            "pr_auc": float(average_precision_score(y_adv, tree_score)),
        },
    }
    return ranked, metrics


def derive_drop_features(ranked: pd.DataFrame, max_importance: float | None) -> list[str]:
    if max_importance is None:
        return []
    selected = ranked.loc[ranked["importance_combined"] >= float(max_importance), "feature"]
    return selected.astype(str).tolist()


def main() -> None:
    args = parse_args()
    x_adv, y_adv, split_meta = build_adversarial_dataset(args)
    ranked, metrics = compute_ranked_importance(x_adv, y_adv, seed=args.seed)

    drop_features = derive_drop_features(ranked, args.max_adversarial_importance)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "adversarial_validation_feature_importance.csv"
    report_path = output_dir / "adversarial_validation_report.json"
    drop_list_path = output_dir / "adversarial_validation_drop_features.txt"

    ranked["label_policy"] = args.label_policy
    ranked.to_csv(csv_path, index=False)
    drop_list_path.write_text("\n".join(drop_features) + ("\n" if drop_features else ""), encoding="utf-8")

    report = {
        "dataset_source": args.dataset_source,
        "label_policy": args.label_policy,
        "seed": args.seed,
        "split": split_meta,
        "models": metrics,
        "max_adversarial_importance": args.max_adversarial_importance,
        "drop_feature_count": int(len(drop_features)),
        "drop_features": drop_features,
        "ranked_features": ranked.to_dict(orient="records"),
        "outputs": {
            "feature_importance_csv": str(csv_path),
            "drop_list_file": str(drop_list_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Adversarial validation complete")
    print(f"- Importance CSV: {csv_path}")
    print(f"- Report JSON: {report_path}")
    print(f"- Drop list file: {drop_list_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.dataset_loader import load_creditcard, load_ieee_cis
from project.data.entity_aggregation import apply_entity_smoothing_batch
from project.data.entity_identity import build_entity_id

DEFAULT_DATASET_PATH = REPO_ROOT / "project" / "legacy_creditcard" / "creditcard.csv"
DEFAULT_OUTPUT = REPO_ROOT / "project" / "outputs" / "monitoring" / "entity_aggregation_report.json"


@dataclass
class MetricRow:
    method: str
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
    parser = argparse.ArgumentParser(description="Benchmark entity-level score aggregation for fraud predictions")
    parser.add_argument("--dataset-source", choices=["creditcard", "ieee_cis"], default="ieee_cis")
    parser.add_argument("--label-policy", choices=["transaction", "account_propagated"], default="transaction")
    parser.add_argument("--dataset-path", type=Path, help="Legacy credit-card CSV path (used when --dataset-source creditcard)")
    parser.add_argument("--ieee-transaction-path", type=Path)
    parser.add_argument("--ieee-identity-path", type=Path)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--blend-alpha", type=float, default=0.5)
    parser.add_argument("--blend-cap", type=float, default=0.25)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
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
    return pd.Series(np.arange(len(features), dtype=float), index=features.index)


def time_based_split_idx(event_time: pd.Series, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(event_time.to_numpy(), kind="mergesort")
    cutoff = int((1.0 - test_size) * len(order))
    cutoff = max(1, min(cutoff, len(order) - 1))
    return order[:cutoff], order[cutoff:]


def train_raw_model(x_train: pd.DataFrame, y_train: pd.Series, seed: int):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float, method: str) -> MetricRow:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return MetricRow(
        method=method,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        false_positive_rate=float(false_positive_rate(y_true, y_pred)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )



def champion(rows: list[MetricRow]) -> MetricRow:
    return sorted(rows, key=lambda r: (r.f1, r.pr_auc, r.roc_auc, r.recall, r.precision), reverse=True)[0]


def main() -> None:
    args = parse_args()
    x, y, source_metadata = load_source(args)
    event_time = resolve_event_time_column(x, args.dataset_source)
    train_order, test_order = time_based_split_idx(event_time, args.test_size)

    x_train = x.iloc[train_order]
    y_train = y.iloc[train_order]
    x_test = x.iloc[test_order].copy()

    model = train_raw_model(x_train, y_train, seed=args.seed)
    raw_score = model.predict_proba(x_test)[:, 1]

    entity_id, entity_diag = build_entity_id(x_test, dataset_source=args.dataset_source)
    eval_df = pd.DataFrame(
        {
            "entity_id": entity_id,
            "event_time": event_time.iloc[test_order].to_numpy(),
            "raw_score": raw_score,
        },
        index=x_test.index,
    ).sort_values(["event_time"], kind="mergesort")

    mean_score = apply_entity_smoothing_batch(eval_df, method="mean", min_history=1)
    ema_score = apply_entity_smoothing_batch(eval_df, method="ema", min_history=1, ema_alpha=args.ema_alpha)
    capped_blend = apply_entity_smoothing_batch(
        eval_df,
        method="blend",
        min_history=1,
        blend_alpha=args.blend_alpha,
        blend_cap=args.blend_cap,
    )

    y_sorted = y.loc[eval_df.index].to_numpy()
    rows = [
        evaluate(y_sorted, eval_df["raw_score"].to_numpy(), args.threshold, "raw"),
        evaluate(y_sorted, mean_score, args.threshold, "entity_mean"),
        evaluate(y_sorted, ema_score, args.threshold, "entity_ema"),
        evaluate(y_sorted, capped_blend, args.threshold, "entity_capped_blend"),
    ]

    best = champion(rows)
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset_source": args.dataset_source,
        "rows_total": int(len(x)),
        "rows_test": int(len(x_test)),
        "seed": args.seed,
        "label_policy": args.label_policy,
        "label_policy_diagnostics": source_metadata.get("label_policy_diagnostics"),
        "test_size": args.test_size,
        "threshold": args.threshold,
        "ema_alpha": args.ema_alpha,
        "blend_alpha": args.blend_alpha,
        "blend_cap": args.blend_cap,
        "entity_diagnostics": entity_diag,
        "metrics": [asdict(row) for row in rows],
        "champion": asdict(best),
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Entity aggregation benchmark complete")
    print(f"- Report: {output_path}")
    print(f"- Champion: {best.method} (f1={best.f1:.4f})")


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.dataset_loader import enforce_eda_gate, load_creditcard, load_ieee_cis

DEFAULT_DATASET_PATH = REPO_ROOT / "project" / "legacy_creditcard" / "creditcard.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "project" / "outputs" / "monitoring"


@dataclass
class CandidateResult:
    candidate: str
    dataset_source: str
    split_mode: str
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
    parser = argparse.ArgumentParser(description="Benchmark ensemble candidates with time-aware OOF predictions.")
    parser.add_argument("--dataset-source", choices=["creditcard", "ieee_cis"], default="ieee_cis")
    parser.add_argument("--label-policy", choices=["transaction", "account_propagated"], default="transaction")
    parser.add_argument("--dataset-path", type=Path, help="Legacy credit-card CSV path (used when --dataset-source creditcard)")
    parser.add_argument("--ieee-transaction-path", type=Path)
    parser.add_argument("--ieee-identity-path", type=Path)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
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


def build_time_oof_splits(event_time: pd.Series, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 3:
        raise ValueError("n_splits must be >= 3 for time-aware OOF")

    order = np.argsort(event_time.to_numpy(), kind="mergesort")
    segments = np.array_split(order, n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, len(segments)):
        train_idx = np.concatenate(segments[:i])
        val_idx = segments[i]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        splits.append((train_idx, val_idx))

    if not splits:
        raise ValueError("Unable to build time-based OOF splits; dataset too small for chosen n_splits")
    return splits


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def evaluate_scores(
    candidate: str,
    dataset_source: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> CandidateResult:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return CandidateResult(
        candidate=candidate,
        dataset_source=dataset_source,
        split_mode="time_oof",
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


def maybe_create_lightgbm(seed: int):
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None, "lightgbm not installed; skipping"

    return (
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            n_jobs=1,
            objective="binary",
            class_weight="balanced",
        ),
        None,
    )


def maybe_create_catboost(seed: int):
    try:
        from catboost import CatBoostClassifier
    except Exception:
        return None, "catboost not installed; skipping"

    return (
        CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=seed,
        ),
        None,
    )


def generate_base_oof_predictions(
    models: dict[str, Any],
    x: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    covered_indices = np.concatenate([val_idx for _, val_idx in splits])
    y_covered = y.iloc[covered_indices].to_numpy()
    predictions: dict[str, np.ndarray] = {name: np.zeros(len(covered_indices), dtype=float) for name in models}
    position = {idx: pos for pos, idx in enumerate(covered_indices)}

    for train_idx, val_idx in splits:
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_val = x.iloc[val_idx]

        for name, model in models.items():
            model.fit(x_train, y_train)
            scores = model.predict_proba(x_val)[:, 1]
            for row_idx, score in zip(val_idx, scores):
                predictions[name][position[int(row_idx)]] = float(score)

    return predictions, y_covered


def optimize_weighted_blend(scores: dict[str, np.ndarray], y_true: np.ndarray, threshold: float) -> tuple[dict[str, float], np.ndarray]:
    names = list(scores.keys())
    if len(names) == 1:
        return {names[0]: 1.0}, scores[names[0]]

    grid = np.linspace(0.0, 1.0, 11)
    best_score = -1.0
    best_weights: dict[str, float] = {}
    best_blend = np.zeros_like(y_true, dtype=float)

    if len(names) == 2:
        for w0 in grid:
            w1 = 1.0 - w0
            blend = (w0 * scores[names[0]]) + (w1 * scores[names[1]])
            f1 = f1_score(y_true, (blend >= threshold).astype(int), zero_division=0)
            if f1 > best_score:
                best_score = f1
                best_weights = {names[0]: float(w0), names[1]: float(w1)}
                best_blend = blend
        return best_weights, best_blend

    for w0 in grid:
        for w1 in grid:
            if w0 + w1 > 1.0:
                continue
            w2 = 1.0 - (w0 + w1)
            blend = (w0 * scores[names[0]]) + (w1 * scores[names[1]]) + (w2 * scores[names[2]])
            f1 = f1_score(y_true, (blend >= threshold).astype(int), zero_division=0)
            if f1 > best_score:
                best_score = f1
                best_weights = {names[0]: float(w0), names[1]: float(w1), names[2]: float(w2)}
                best_blend = blend

    return best_weights, best_blend


def build_stacker_oof(base_scores: dict[str, np.ndarray], y_true: np.ndarray, seed: int) -> np.ndarray:
    x_meta = np.column_stack([base_scores[name] for name in base_scores.keys()])
    n = len(y_true)
    split = max(int(n * 0.7), 1)
    split = min(split, n - 1)
    stacker = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    stacker.fit(x_meta[:split], y_true[:split])
    pred = np.zeros(n, dtype=float)
    pred[:split] = stacker.predict_proba(x_meta[:split])[:, 1]
    pred[split:] = stacker.predict_proba(x_meta[split:])[:, 1]
    return pred


def rank_candidates(results_df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    tie_break = [metric, "pr_auc", "roc_auc", "recall", "precision"]
    cols = [c for c in tie_break if c in results_df.columns]
    return results_df.sort_values(cols, ascending=[False] * len(cols))


def feature_hash(columns: list[str]) -> str:
    joined = "\n".join(columns)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    x, y, source_metadata = load_source(args)
    enforce_eda_gate(
        source_metadata,
        allow_failures=args.allow_eda_failures,
        context=f"{Path(__file__).name}::{args.dataset_source}",
    )
    event_time = resolve_event_time_column(x, args.dataset_source)
    splits = build_time_oof_splits(event_time, n_splits=args.n_splits)

    notes: list[str] = []

    from xgboost import XGBClassifier

    models: dict[str, Any] = {
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=args.seed,
        )
    }

    lightgbm_model, lightgbm_note = maybe_create_lightgbm(seed=args.seed)
    if lightgbm_model is not None:
        models["lightgbm"] = lightgbm_model
    elif lightgbm_note:
        notes.append(lightgbm_note)

    catboost_model, catboost_note = maybe_create_catboost(seed=args.seed)
    if catboost_model is not None:
        models["catboost"] = catboost_model
    elif catboost_note:
        notes.append(catboost_note)

    base_scores, y_covered = generate_base_oof_predictions(models=models, x=x, y=y, splits=splits)

    results: list[CandidateResult] = []
    for name, score in base_scores.items():
        results.append(
            evaluate_scores(
                candidate=name,
                dataset_source=args.dataset_source,
                y_true=y_covered,
                y_score=score,
                threshold=args.threshold,
            )
        )

    score_matrix = np.column_stack([base_scores[name] for name in base_scores.keys()])
    equal_blend = np.mean(score_matrix, axis=1)
    results.append(
        evaluate_scores(
            candidate="ensemble_equal_weight",
            dataset_source=args.dataset_source,
            y_true=y_covered,
            y_score=equal_blend,
            threshold=args.threshold,
        )
    )

    weighted_weights, weighted_blend = optimize_weighted_blend(base_scores, y_covered, threshold=args.threshold)
    results.append(
        evaluate_scores(
            candidate="ensemble_weighted",
            dataset_source=args.dataset_source,
            y_true=y_covered,
            y_score=weighted_blend,
            threshold=args.threshold,
        )
    )

    stacked_scores = build_stacker_oof(base_scores, y_covered, seed=args.seed)
    results.append(
        evaluate_scores(
            candidate="ensemble_stack_logistic",
            dataset_source=args.dataset_source,
            y_true=y_covered,
            y_score=stacked_scores,
            threshold=args.threshold,
        )
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([asdict(row) for row in results])
    ranked_df = rank_candidates(results_df, metric="f1")

    csv_path = output_dir / f"{args.dataset_source}_ensemble_benchmark.csv"
    ranked_df.to_csv(csv_path, index=False)

    base_names = list(base_scores.keys())
    split_info = [
        {"fold": i + 1, "train_count": int(len(train_idx)), "val_count": int(len(val_idx))}
        for i, (train_idx, val_idx) in enumerate(splits)
    ]

    best_row = ranked_df.iloc[0].to_dict()
    report = {
        "dataset_source": args.dataset_source,
        "label_policy": args.label_policy,
        "label_policy_diagnostics": source_metadata.get("label_policy_diagnostics"),
        "eda_gate": source_metadata.get("eda_gate"),
        "eda_gate_status": "passed" if source_metadata.get("eda_gate", {}).get("passed", True) else "failed",
        "split_mode": "time_oof",
        "n_splits": args.n_splits,
        "seed": args.seed,
        "threshold": args.threshold,
        "feature_count": int(x.shape[1]),
        "feature_hash": feature_hash(list(x.columns)),
        "rows_total": int(len(x)),
        "rows_oof_evaluated": int(len(y_covered)),
        "base_models_used": base_names,
        "weighted_blend_weights": weighted_weights,
        "split_windows": split_info,
        "notes": notes,
        "candidate_count": int(len(ranked_df)),
        "champion": best_row,
        "results_csv": str(csv_path),
    }
    report_path = output_dir / f"{args.dataset_source}_ensemble_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Ensemble benchmark complete")
    print(f"- Results CSV: {csv_path}")
    print(f"- Report JSON: {report_path}")
    print(f"- Champion: {best_row.get('candidate')} (f1={best_row.get('f1'):.4f})")


if __name__ == "__main__":
    main()

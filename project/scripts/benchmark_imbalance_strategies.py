import argparse
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
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.dataset_loader import load_creditcard, load_ieee_cis

DEFAULT_OUTPUT_DIR = REPO_ROOT / "project" / "outputs" / "monitoring"


@dataclass
class StrategyResult:
    strategy: str
    dataset_source: str
    split_mode: str
    train_rows: int
    test_rows: int
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
        description="Benchmark class-imbalance strategies on a time-aware holdout split."
    )
    parser.add_argument("--dataset-source", choices=["creditcard", "ieee_cis"], default="ieee_cis")
    parser.add_argument("--label-policy", choices=["transaction", "account_propagated"], default="transaction")
    parser.add_argument("--dataset-path", type=Path, help="Legacy credit-card CSV path (used when --dataset-source creditcard)")
    parser.add_argument("--ieee-transaction-path", type=Path)
    parser.add_argument("--ieee-identity-path", type=Path)
    parser.add_argument("--split-mode", choices=["time", "random"], default="time")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--stability-windows",
        type=int,
        default=3,
        help="Number of chronological expanding windows for per-strategy stability stats (time split only).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def encode_categorical_columns(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            as_text = series.astype("string").fillna("<NA>")
            out[col] = pd.Categorical(as_text).codes.astype(np.int32)
    return out


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
    from sklearn.model_selection import train_test_split

    return train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def prepare_resampling_features(x_train: pd.DataFrame) -> pd.DataFrame:
    """Return finite numeric features for samplers that reject NaN/inf values.

    SMOTE/ADASYN in imbalanced-learn require finite numeric input. IEEE-CIS has
    substantial missingness, so we apply deterministic train-only imputation here.
    """
    out = x_train.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    medians = out.median(numeric_only=True)
    out = out.fillna(medians)
    return out.fillna(0.0)


def apply_resampling(
    strategy: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    class_count = int(y_train.nunique(dropna=False))
    if class_count < 2:
        return x_train, y_train

    if strategy == "baseline":
        return x_train, y_train

    x_train_prepared = prepare_resampling_features(x_train)

    try:
        from imblearn.over_sampling import ADASYN, SMOTE
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError as exc:
        raise ImportError(
            "Imbalance benchmarking requires imbalanced-learn. Install with `pip install imbalanced-learn`."
        ) from exc

    if strategy == "random_undersample":
        sampler = RandomUnderSampler(random_state=seed)
        x_out, y_out = sampler.fit_resample(x_train_prepared, y_train)
        return pd.DataFrame(x_out, columns=x_train.columns), pd.Series(y_out)

    if strategy == "smote":
        sampler = SMOTE(random_state=seed)
        x_out, y_out = sampler.fit_resample(x_train_prepared, y_train)
        return pd.DataFrame(x_out, columns=x_train.columns), pd.Series(y_out)

    if strategy == "adasyn":
        sampler = ADASYN(random_state=seed)
        x_out, y_out = sampler.fit_resample(x_train_prepared, y_train)
        return pd.DataFrame(x_out, columns=x_train.columns), pd.Series(y_out)

    if strategy == "smote_then_undersample":
        smote = SMOTE(random_state=seed)
        x_smote, y_smote = smote.fit_resample(x_train_prepared, y_train)
        under = RandomUnderSampler(random_state=seed)
        x_out, y_out = under.fit_resample(x_smote, y_smote)
        return pd.DataFrame(x_out, columns=x_train.columns), pd.Series(y_out)

    raise ValueError(f"Unsupported strategy: {strategy}")


def evaluate_strategy(
    strategy: str,
    dataset_source: str,
    split_mode: str,
    threshold: float,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
) -> StrategyResult:
    x_train_res, y_train_res = apply_resampling(strategy, x_train, y_train, seed=seed)

    neg = int((y_train_res == 0).sum())
    pos = int((y_train_res == 1).sum())
    scale_pos_weight = (neg / pos) if pos else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
    )
    if y_train_res.nunique(dropna=False) < 2:
        y_score = np.zeros(len(x_test), dtype=float)
        y_pred = np.zeros(len(x_test), dtype=int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        pr_auc = average_precision_score(y_test, y_score)
        roc_auc = roc_auc_score(y_test, y_score) if y_test.nunique(dropna=False) > 1 else 0.5
        return StrategyResult(
            strategy=strategy,
            dataset_source=dataset_source,
            split_mode=split_mode,
            train_rows=int(len(x_train_res)),
            test_rows=int(len(x_test)),
            threshold=float(threshold),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1=float(f1_score(y_test, y_pred, zero_division=0)),
            false_positive_rate=float(false_positive_rate(y_test.to_numpy(), y_pred)),
            pr_auc=float(pr_auc),
            roc_auc=float(roc_auc),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            tp=int(tp),
        )

    model.fit(x_train_res, y_train_res)
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return StrategyResult(
        strategy=strategy,
        dataset_source=dataset_source,
        split_mode=split_mode,
        train_rows=int(len(x_train_res)),
        test_rows=int(len(x_test)),
        threshold=float(threshold),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        false_positive_rate=float(false_positive_rate(y_test.to_numpy(), y_pred)),
        pr_auc=float(average_precision_score(y_test, y_score)),
        roc_auc=float(roc_auc_score(y_test, y_score) if y_test.nunique(dropna=False) > 1 else 0.5),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def build_expanding_time_windows(
    event_time: pd.Series,
    test_size: float,
    n_windows: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")

    order = np.argsort(event_time.to_numpy(), kind="mergesort")
    n_rows = len(order)
    if n_rows < 3:
        return []

    test_len = max(1, int(round(n_rows * float(test_size))))
    test_len = min(test_len, n_rows - 1)
    max_train_end = n_rows - test_len
    if max_train_end <= 0:
        return []

    if n_windows == 1:
        train_ends = [max_train_end]
    else:
        train_ends = np.linspace(1, max_train_end, num=n_windows, dtype=int).tolist()

    windows: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, int]] = set()
    for train_end in train_ends:
        train_end = max(1, min(int(train_end), max_train_end))
        test_end = min(train_end + test_len, n_rows)
        key = (train_end, test_end)
        if key in seen:
            continue
        seen.add(key)

        train_order = order[:train_end]
        test_order = order[train_end:test_end]
        if len(train_order) == 0 or len(test_order) == 0:
            continue
        windows.append((train_order, test_order))
    return windows


def summarize_stability(window_results: list[StrategyResult]) -> dict[str, float | int]:
    metrics = {
        "f1": np.array([r.f1 for r in window_results], dtype=float),
        "pr_auc": np.array([r.pr_auc for r in window_results], dtype=float),
        "recall": np.array([r.recall for r in window_results], dtype=float),
    }
    summary: dict[str, float | int] = {"stability_window_count": int(len(window_results))}
    for metric, values in metrics.items():
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))
        summary[f"{metric}_worst_window"] = float(np.min(values))
    return summary


def add_stability_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    enriched = results_df.copy()
    enriched["rank_pr_auc_mean"] = enriched["pr_auc_mean"].rank(method="dense", ascending=False).astype(int)
    enriched["rank_recall_worst"] = enriched["recall_worst_window"].rank(method="dense", ascending=False).astype(int)

    stability_upside = (
        enriched["f1_mean"] + enriched["pr_auc_mean"] + enriched["recall_mean"]
        + enriched["f1_worst_window"] + enriched["pr_auc_worst_window"] + enriched["recall_worst_window"]
    ) / 6.0
    stability_penalty = (enriched["f1_std"] + enriched["pr_auc_std"] + enriched["recall_std"]) / 3.0
    enriched["robustness_score"] = stability_upside - stability_penalty
    enriched["rank_robustness_score"] = enriched["robustness_score"].rank(method="dense", ascending=False).astype(int)
    return enriched


def main() -> None:
    args = parse_args()
    x, y, source_metadata = load_source(args)
    x = encode_categorical_columns(x)

    if args.split_mode == "time":
        event_time = resolve_event_time_column(x, args.dataset_source)
        x_train, x_test, y_train, y_test = time_based_split(x, y, event_time, args.test_size)
    else:
        x_train, x_test, y_train, y_test = random_split(x, y, args.test_size, args.seed)

    strategies = [
        "baseline",
        "random_undersample",
        "smote",
        "adasyn",
        "smote_then_undersample",
    ]
    rows = [
        evaluate_strategy(
            strategy=s,
            dataset_source=args.dataset_source,
            split_mode=args.split_mode,
            threshold=args.threshold,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            seed=args.seed,
        )
        for s in strategies
    ]

    stability_rows: list[dict[str, float | int | str]] = []
    if args.split_mode == "time":
        event_time = resolve_event_time_column(x, args.dataset_source)
        windows = build_expanding_time_windows(event_time, test_size=args.test_size, n_windows=args.stability_windows)
        for strategy in strategies:
            strategy_window_results: list[StrategyResult] = []
            for train_order, test_order in windows:
                x_train_window = x.iloc[train_order]
                y_train_window = y.iloc[train_order]
                x_test_window = x.iloc[test_order]
                y_test_window = y.iloc[test_order]
                strategy_window_results.append(
                    evaluate_strategy(
                        strategy=strategy,
                        dataset_source=args.dataset_source,
                        split_mode="time",
                        threshold=args.threshold,
                        x_train=x_train_window,
                        y_train=y_train_window,
                        x_test=x_test_window,
                        y_test=y_test_window,
                        seed=args.seed,
                    )
                )

            if strategy_window_results:
                row = {"strategy": strategy}
                row.update(summarize_stability(strategy_window_results))
                stability_rows.append(row)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([asdict(row) for row in rows])
    if stability_rows:
        stability_df = pd.DataFrame(stability_rows)
        results_df = results_df.merge(stability_df, on="strategy", how="left")
        results_df = add_stability_columns(results_df)
    results_df = results_df.sort_values("f1", ascending=False)
    csv_path = output_dir / f"{args.dataset_source}_imbalance_strategy_benchmark.csv"
    results_df.to_csv(csv_path, index=False)

    meta = {
        "dataset_source": args.dataset_source,
        "split_mode": args.split_mode,
        "test_size": args.test_size,
        "seed": args.seed,
        "threshold": args.threshold,
        "label_policy": args.label_policy,
        "label_policy_diagnostics": source_metadata.get("label_policy_diagnostics"),
        "stability_windows": args.stability_windows,
        "strategies": strategies,
        "output_csv": str(csv_path),
    }
    if stability_rows:
        meta["stability_metrics"] = [
            "f1_mean",
            "f1_std",
            "f1_worst_window",
            "pr_auc_mean",
            "pr_auc_std",
            "pr_auc_worst_window",
            "recall_mean",
            "recall_std",
            "recall_worst_window",
            "rank_pr_auc_mean",
            "rank_recall_worst",
            "robustness_score",
            "rank_robustness_score",
        ]
    meta_path = output_dir / f"{args.dataset_source}_imbalance_strategy_benchmark_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Imbalance strategy benchmark complete")
    print(f"- Metrics CSV: {csv_path}")
    print(f"- Metadata JSON: {meta_path}")


if __name__ == "__main__":
    main()

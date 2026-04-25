import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.dataset_loader import load_creditcard, load_ieee_cis
from project.data.behavior_features import get_uid_candidates_for_source
from project.data.entity_aggregation import build_uid

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
    parser = argparse.ArgumentParser(
        description="Tune model candidates (Pass 2): XGBoost + lightweight algorithm baselines on reproducible holdout and temporal robustness windows."
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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--include-logistic-baseline", action="store_true", help="Include LogisticRegression baseline (disabled by default for IEEE NaN-heavy data).")
    parser.add_argument("--window-configs", type=str, default="4:1:1,3:1:1", help="Comma-separated train:gap:validate windows in pseudo-month groups")
    parser.add_argument("--month-group-count", type=int, default=8, help="Number of contiguous temporal groups for robustness evaluation")
    parser.add_argument(
        "--ensemble-report-json",
        type=Path,
        help="Optional ensemble benchmark report JSON to merge into promotion-oriented candidate summary.",
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


def encode_categorical_columns(features: pd.DataFrame) -> pd.DataFrame:
    """Convert object/category columns to deterministic integer codes for model compatibility."""
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
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def evaluate_candidate(
    candidate_name: str,
    dataset_source: str,
    split_mode: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> CandidateResult:
    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return CandidateResult(
        candidate=candidate_name,
        dataset_source=dataset_source,
        split_mode=split_mode,
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        false_positive_rate=float(false_positive_rate(y_test.to_numpy(), y_pred)),
        pr_auc=float(average_precision_score(y_test, y_score)),
        roc_auc=float(_safe_roc_auc(y_test.to_numpy(), y_score)),
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


def tune_xgboost_with_optuna(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    n_trials: int,
) -> tuple[Any, dict[str, Any], list[str]]:
    notes: list[str] = []

    from xgboost import XGBClassifier

    try:
        import optuna
    except Exception:
        notes.append("optuna not installed; using default XGBoost config")
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=seed,
        )
        return model, {}, notes

    x_tune_train, x_val, y_tune_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        stratify=y_train,
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": seed,
        }
        model = XGBClassifier(**params)
        model.fit(x_tune_train, y_tune_train)
        y_val_score = model.predict_proba(x_val)[:, 1]
        return float(average_precision_score(y_val, y_val_score))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    best_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": seed,
        }
    )
    tuned_model = XGBClassifier(**best_params)
    return tuned_model, study.best_params, notes


def rank_candidates(results_df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    tie_break = [metric, "pr_auc", "roc_auc", "recall", "precision"]
    cols = [c for c in tie_break if c in results_df.columns]
    return results_df.sort_values(cols, ascending=[False] * len(cols))


def parse_window_configs(raw: str) -> list[tuple[int, int, int]]:
    configs: list[tuple[int, int, int]] = []
    for token in [p.strip() for p in raw.split(",") if p.strip()]:
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid window config '{token}'. Expected train:gap:validate")
        train_n, gap_n, val_n = (int(parts[0]), int(parts[1]), int(parts[2]))
        if train_n <= 0 or gap_n < 0 or val_n <= 0:
            raise ValueError(f"Invalid window config '{token}'. Values must satisfy train>0, gap>=0, validate>0")
        configs.append((train_n, gap_n, val_n))
    if not configs:
        raise ValueError("No valid window config provided")
    return configs


def assign_time_groups(event_time: pd.Series, n_groups: int) -> pd.Series:
    if n_groups < 3:
        raise ValueError("month-group-count must be >= 3")
    order = np.argsort(event_time.to_numpy(), kind="mergesort")
    groups = np.array_split(order, min(n_groups, len(order)))
    group_ids = np.zeros(len(event_time), dtype=int)
    for gid, idxs in enumerate(groups):
        if len(idxs):
            group_ids[idxs] = gid
    return pd.Series(group_ids, index=event_time.index)


def build_temporal_windows(group_ids: pd.Series, configs: list[tuple[int, int, int]]) -> list[dict[str, Any]]:
    max_group = int(group_ids.max())
    windows: list[dict[str, Any]] = []
    for train_n, gap_n, val_n in configs:
        end_cap = max_group - (train_n + gap_n + val_n) + 1
        for start in range(0, max(0, end_cap)):
            train_groups = list(range(start, start + train_n))
            val_start = start + train_n + gap_n
            val_groups = list(range(val_start, val_start + val_n))
            train_idx = group_ids.index[group_ids.isin(train_groups)].to_numpy()
            val_idx = group_ids.index[group_ids.isin(val_groups)].to_numpy()
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            windows.append(
                {
                    "validation_mode": "temporal_holdout",
                    "window_config": f"{train_n}:{gap_n}:{val_n}",
                    "train_groups": train_groups,
                    "val_groups": val_groups,
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                }
            )
    return windows


def build_grouped_folds(group_ids: pd.Series, gap_groups: int = 1) -> list[dict[str, Any]]:
    unique_groups = sorted(group_ids.unique().tolist())
    folds: list[dict[str, Any]] = []
    for val_group in unique_groups[2:]:
        train_max = val_group - gap_groups - 1
        train_groups = [g for g in unique_groups if g <= train_max]
        if not train_groups:
            continue
        train_idx = group_ids.index[group_ids.isin(train_groups)].to_numpy()
        val_idx = group_ids.index[group_ids == val_group].to_numpy()
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append(
            {
                "validation_mode": "month_group_fold",
                "window_config": f"expanding_gap{gap_groups}",
                "train_groups": train_groups,
                "val_groups": [val_group],
                "train_idx": train_idx,
                "val_idx": val_idx,
            }
        )
    return folds


def summarize_robustness(window_df: pd.DataFrame) -> dict[str, Any]:
    metrics = ["precision", "recall", "false_positive_rate", "pr_auc", "f1"]
    out: dict[str, Any] = {}
    for candidate, cdf in window_df.groupby("candidate"):
        summary: dict[str, Any] = {
            "window_count": int(len(cdf)),
            "validation_modes": sorted(cdf["validation_mode"].unique().tolist()),
            "metrics": {},
        }
        for metric in metrics:
            vals = cdf[metric].astype(float)
            summary["metrics"][metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "worst": float(vals.min()),
            }
        out[candidate] = summary
    return out


def load_ensemble_champion(report_path: Path | None) -> dict[str, Any] | None:
    if report_path is None:
        return None
    resolved = report_path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        return None

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    champion = payload.get("champion")
    if not isinstance(champion, dict):
        return None
    out = dict(champion)
    out["_source_report"] = str(resolved)
    out["_source_type"] = "ensemble_benchmark"
    return out


def select_overall_best(single_best: dict[str, Any], ensemble_best: dict[str, Any] | None) -> dict[str, Any]:
    if not ensemble_best:
        out = dict(single_best)
        out["candidate_origin"] = "single_model"
        return out

    ranked = pd.DataFrame([single_best, ensemble_best])
    best = rank_candidates(ranked, metric="f1").iloc[0].to_dict()
    best["candidate_origin"] = "ensemble" if best.get("_source_type") == "ensemble_benchmark" else "single_model"
    return best


def append_ensemble_to_robustness(robustness: dict[str, Any], ensemble_best: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(robustness)
    if not ensemble_best:
        return merged

    candidate_name = str(ensemble_best.get("candidate", "unknown"))
    merged[f"ensemble::{candidate_name}"] = {
        "window_count": 0,
        "validation_modes": ["time_oof"],
        "metrics": {
            "precision": {"mean": float(ensemble_best.get("precision", float("nan"))), "std": 0.0, "worst": float(ensemble_best.get("precision", float("nan")))},
            "recall": {"mean": float(ensemble_best.get("recall", float("nan"))), "std": 0.0, "worst": float(ensemble_best.get("recall", float("nan")))},
            "false_positive_rate": {"mean": float(ensemble_best.get("false_positive_rate", float("nan"))), "std": 0.0, "worst": float(ensemble_best.get("false_positive_rate", float("nan")))},
            "pr_auc": {"mean": float(ensemble_best.get("pr_auc", float("nan"))), "std": 0.0, "worst": float(ensemble_best.get("pr_auc", float("nan")))},
            "f1": {"mean": float(ensemble_best.get("f1", float("nan"))), "std": 0.0, "worst": float(ensemble_best.get("f1", float("nan")))},
        },
        "source": "ensemble_benchmark",
    }
    return merged


def main() -> None:
    args = parse_args()
    x, y, source_metadata = load_source(args)
    x = encode_categorical_columns(x)

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
        event_time = resolve_event_time_column(x, args.dataset_source)
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y,
        )
        event_time = resolve_event_time_column(x, args.dataset_source)

    results: list[CandidateResult] = []
    notes: list[str] = []

    tuned_xgb, tuned_params, tuning_notes = tune_xgboost_with_optuna(
        x_train=x_train,
        y_train=y_train,
        seed=args.seed,
        n_trials=args.optuna_trials,
    )
    notes.extend(tuning_notes)

    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression

    xgb_default = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=args.seed,
    )

    candidates: list[tuple[str, Any]] = [
        ("xgboost_default", xgb_default),
        ("xgboost_tuned", tuned_xgb),
    ]
    if args.include_logistic_baseline:
        candidates.append(("logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)))
    else:
        notes.append("logistic_regression baseline disabled by default (use --include-logistic-baseline to enable).")

    lightgbm_model, lightgbm_note = maybe_create_lightgbm(seed=args.seed)
    if lightgbm_model is not None:
        candidates.append(("lightgbm", lightgbm_model))
    elif lightgbm_note:
        notes.append(lightgbm_note)

    catboost_model, catboost_note = maybe_create_catboost(seed=args.seed)
    if catboost_model is not None:
        candidates.append(("catboost", catboost_model))
    elif catboost_note:
        notes.append(catboost_note)

    for name, model in candidates:
        try:
            results.append(
                evaluate_candidate(
                    candidate_name=name,
                    dataset_source=args.dataset_source,
                    split_mode=args.split_mode,
                    model=model,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    threshold=args.threshold,
                )
            )
        except ValueError as exc:
            if name == "logistic_regression":
                notes.append(f"logistic_regression skipped due to fit error: {exc}")
                continue
            raise

    # Robustness evaluation across repeated windows + month-group folds.
    group_ids = assign_time_groups(event_time, n_groups=args.month_group_count)
    configs = parse_window_configs(args.window_configs)
    windows = build_temporal_windows(group_ids, configs=configs) + build_grouped_folds(group_ids, gap_groups=1)

    window_rows: list[dict[str, Any]] = []
    for i, w in enumerate(windows):
        xw_train = x.loc[w["train_idx"]]
        yw_train = y.loc[w["train_idx"]]
        xw_val = x.loc[w["val_idx"]]
        yw_val = y.loc[w["val_idx"]]

        for name, base_model in candidates:
            try:
                row = evaluate_candidate(
                    candidate_name=name,
                    dataset_source=args.dataset_source,
                    split_mode=w["validation_mode"],
                    model=clone(base_model),
                    x_train=xw_train,
                    y_train=yw_train,
                    x_test=xw_val,
                    y_test=yw_val,
                    threshold=args.threshold,
                )
            except ValueError as exc:
                if name == "logistic_regression":
                    notes.append(f"logistic_regression skipped in robustness window {i}: {exc}")
                    continue
                raise
            payload = asdict(row)
            payload.update(
                {
                    "window_id": i,
                    "validation_mode": w["validation_mode"],
                    "window_config": w["window_config"],
                    "train_groups": ",".join(str(g) for g in w["train_groups"]),
                    "validation_groups": ",".join(str(g) for g in w["val_groups"]),
                }
            )
            window_rows.append(payload)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([asdict(row) for row in results])
    ranked_df = rank_candidates(results_df, metric="f1")

    csv_path = output_dir / f"{args.dataset_source}_model_candidate_benchmark.csv"
    ranked_df.to_csv(csv_path, index=False)

    windows_df = pd.DataFrame(window_rows)
    windows_csv_path = output_dir / f"{args.dataset_source}_validation_windows.csv"
    windows_df.to_csv(windows_csv_path, index=False)

    robustness = summarize_robustness(windows_df) if not windows_df.empty else {}

    ensemble_report_path = args.ensemble_report_json
    if ensemble_report_path is None:
        ensemble_report_path = output_dir / f"{args.dataset_source}_ensemble_benchmark_report.json"
    ensemble_best = load_ensemble_champion(ensemble_report_path)
    combined_robustness = append_ensemble_to_robustness(robustness, ensemble_best)

    best_row = ranked_df.iloc[0].to_dict()
    overall_best = select_overall_best(best_row, ensemble_best)
    best_candidate = str(best_row.get("candidate"))
    best_robust = robustness.get(best_candidate, {})
    robustness_gate = {
        "required": True,
        "passed": bool(best_robust),
        "reason": "ok" if best_robust else "missing_best_candidate_robustness",
    }

    report = {
        "dataset_source": args.dataset_source,
        "label_policy": args.label_policy,
        "label_policy_diagnostics": source_metadata.get("label_policy_diagnostics"),
        "eda_gate": source_metadata.get("eda_gate"),
        "eda_gate_status": "passed" if source_metadata.get("eda_gate", {}).get("passed", True) else "failed",
        "split_mode": args.split_mode,
        "test_size": args.test_size,
        "seed": args.seed,
        "threshold": args.threshold,
        "optuna_trials": args.optuna_trials,
        "window_configs": [f"{t}:{g}:{h}" for t, g, h in configs],
        "month_group_count": args.month_group_count,
        "window_count": int(len(windows_df["window_id"].unique())) if not windows_df.empty else 0,
        "xgboost_tuned_best_params": tuned_params,
        "notes": notes,
        "candidate_count": int(len(ranked_df)),
        "best_candidate": best_row,
        "ensemble_best_candidate": ensemble_best,
        "overall_best_candidate": overall_best,
        "robustness_by_candidate": combined_robustness,
        "best_candidate_robustness": best_robust,
        "robustness_gate": robustness_gate,
        "results_csv": str(csv_path),
        "validation_windows_csv": str(windows_csv_path),
        "ensemble_report_json": str(ensemble_report_path) if ensemble_best else None,
    }
    report_path = output_dir / f"{args.dataset_source}_model_candidate_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    robustness_report_path = output_dir / f"{args.dataset_source}_validation_robustness_report.json"
    robustness_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Model candidate benchmark complete")
    print(f"- Results CSV: {csv_path}")
    print(f"- Report JSON: {report_path}")
    print(f"- Validation windows CSV: {windows_csv_path}")
    print(f"- Robustness report JSON: {robustness_report_path}")
    print(f"- Best candidate: {best_row.get('candidate')} (f1={best_row.get('f1'):.4f})")


if __name__ == "__main__":
    main()

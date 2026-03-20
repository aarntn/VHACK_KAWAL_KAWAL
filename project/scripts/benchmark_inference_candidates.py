import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.dataset_loader import load_creditcard, load_ieee_cis

DEFAULT_OUTPUT_DIR = REPO_ROOT / "project" / "outputs" / "monitoring"


@dataclass
class CandidateMetrics:
    candidate: str
    runtime: str
    fit_time_ms: float
    startup_time_ms: float
    memory_bytes: int
    p95_latency_ms: float
    p99_latency_ms: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference latency/KPI candidates and produce promotion gate report.")
    parser.add_argument("--dataset-source", choices=["creditcard", "ieee_cis"], default="ieee_cis")
    parser.add_argument("--label-policy", choices=["transaction", "account_propagated"], default="transaction")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--ieee-transaction-path", type=Path)
    parser.add_argument("--ieee-identity-path", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--latency-samples", type=int, default=512)
    parser.add_argument("--sla-max-p95-ms", type=float, default=20.0)
    parser.add_argument("--min-pr-auc", type=float, default=0.70)
    parser.add_argument("--min-recall", type=float, default=0.60)
    parser.add_argument("--max-pr-auc-drop", type=float, default=0.01)
    parser.add_argument("--max-recall-drop", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_source(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series]:
    if args.dataset_source == "creditcard":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required when --dataset-source creditcard")
        x, y, _ = load_creditcard(args.dataset_path, label_policy=args.label_policy)
        return x, y

    if not args.ieee_transaction_path or not args.ieee_identity_path:
        raise ValueError("--ieee-transaction-path and --ieee-identity-path are required for ieee_cis source")
    x, y, _ = load_ieee_cis(args.ieee_transaction_path, args.ieee_identity_path, label_policy=args.label_policy)
    return x, y


def encode_categorical_columns(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or isinstance(out[col].dtype, pd.CategoricalDtype):
            encoded = out[col].astype("string").fillna("<NA>")
            out[col] = pd.Categorical(encoded).codes.astype(np.int32)
    return out


def resolve_event_time_column(features: pd.DataFrame, dataset_source: str) -> pd.Series:
    col = "TransactionDT" if dataset_source == "ieee_cis" else "Time"
    if col in features.columns:
        return pd.to_numeric(features[col], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(features)), index=features.index)


def time_split(
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


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def measure_row_level_latency_ms(model: Any, x_test: pd.DataFrame, max_samples: int) -> tuple[float, float, float]:
    sample = x_test.iloc[: max(1, min(max_samples, len(x_test)))]
    latencies: list[float] = []

    warm_start = time.perf_counter()
    model.predict_proba(sample.iloc[[0]])
    startup_ms = (time.perf_counter() - warm_start) * 1000.0

    for idx in range(len(sample)):
        start = time.perf_counter()
        model.predict_proba(sample.iloc[[idx]])
        latencies.append((time.perf_counter() - start) * 1000.0)

    return percentile(latencies, 95), percentile(latencies, 99), startup_ms


def evaluate_trained_candidate(
    candidate: str,
    runtime: str,
    trained_model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    latency_samples: int,
    notes: list[str] | None = None,
) -> CandidateMetrics:
    y_score = trained_model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    p95, p99, startup_ms = measure_row_level_latency_ms(trained_model, x_test, latency_samples)
    return CandidateMetrics(
        candidate=candidate,
        runtime=runtime,
        fit_time_ms=0.0,
        startup_time_ms=float(startup_ms),
        memory_bytes=len(pickle.dumps(trained_model)),
        p95_latency_ms=float(p95),
        p99_latency_ms=float(p99),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        pr_auc=float(average_precision_score(y_test, y_score)),
        roc_auc=float(_safe_roc_auc(y_test.to_numpy(), y_score)),
        notes=list(notes or []),
    )


def maybe_build_onnx_runtime(model: Any, feature_count: int) -> tuple[Any | None, str | None]:
    try:
        import onnxruntime as ort
        from onnxmltools import convert_xgboost
        from skl2onnx.common.data_types import FloatTensorType
    except Exception:
        return None, "onnxruntime conversion dependencies unavailable"

    try:
        onx = convert_xgboost(model, initial_types=[("input", FloatTensorType([None, feature_count]))])
        session = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return None, f"onnx conversion failed: {exc}"

    class OnnxRuntimeWrapper:
        def __init__(self, sess: Any):
            self.sess = sess
            self.input_name = sess.get_inputs()[0].name

        def predict_proba(self, x_frame: pd.DataFrame) -> np.ndarray:
            data = x_frame.to_numpy(dtype=np.float32)
            outputs = self.sess.run(None, {self.input_name: data})
            if len(outputs) >= 2:
                probs = np.asarray(outputs[1])
                if probs.ndim == 2 and probs.shape[1] == 2:
                    return probs
            score = np.asarray(outputs[0]).reshape(-1)
            return np.column_stack([1.0 - score, score])

    return OnnxRuntimeWrapper(session), None


def build_candidate_factories(seed: int) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    factories: dict[str, Any] = {}

    from xgboost import XGBClassifier

    factories["xgb_baseline"] = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=1,
        random_state=seed,
    )
    factories["xgb_shallow_regularized"] = XGBClassifier(
        n_estimators=180,
        max_depth=3,
        min_child_weight=8,
        reg_alpha=1.5,
        reg_lambda=8.0,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=1,
        random_state=seed,
    )

    try:
        from lightgbm import LGBMClassifier

        factories["lightgbm_fast"] = LGBMClassifier(
            n_estimators=160,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=6,
            objective="binary",
            n_jobs=1,
            random_state=seed,
        )
    except Exception:
        notes.append("lightgbm not installed; skipped")

    try:
        from catboost import CatBoostClassifier

        factories["catboost_fast"] = CatBoostClassifier(
            iterations=160,
            learning_rate=0.08,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=seed,
        )
    except Exception:
        notes.append("catboost not installed; skipped")

    return factories, notes


def apply_regression_gate(
    baseline: CandidateMetrics,
    promoted: CandidateMetrics,
    max_pr_auc_drop: float,
    max_recall_drop: float,
) -> dict[str, Any]:
    latency_improved = promoted.p95_latency_ms < baseline.p95_latency_ms
    pr_auc_drop = baseline.pr_auc - promoted.pr_auc
    recall_drop = baseline.recall - promoted.recall
    blocked = bool(latency_improved and (pr_auc_drop > max_pr_auc_drop or recall_drop > max_recall_drop))
    return {
        "blocked": blocked,
        "reason": "latency_improved_but_kpi_regressed" if blocked else "ok",
        "latency_improved": latency_improved,
        "pr_auc_drop": float(pr_auc_drop),
        "recall_drop": float(recall_drop),
        "tolerances": {
            "max_pr_auc_drop": float(max_pr_auc_drop),
            "max_recall_drop": float(max_recall_drop),
        },
    }


def pick_promotion_candidate(results: list[CandidateMetrics], sla_max_p95_ms: float, min_pr_auc: float, min_recall: float) -> tuple[CandidateMetrics | None, list[str]]:
    notes: list[str] = []
    eligible = [
        row
        for row in results
        if row.p95_latency_ms <= sla_max_p95_ms and row.pr_auc >= min_pr_auc and row.recall >= min_recall
    ]
    if not eligible:
        notes.append("No candidate satisfied SLA/KPI eligibility constraints")
        return None, notes

    ranked = sorted(eligible, key=lambda row: (row.p95_latency_ms, row.p99_latency_ms, -row.pr_auc, -row.recall))
    return ranked[0], notes


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x, y = load_source(args)
    x = encode_categorical_columns(x)
    event_time = resolve_event_time_column(x, args.dataset_source)
    x_train, x_test, y_train, y_test = time_split(x, y, event_time, args.test_size)

    factories, skipped_notes = build_candidate_factories(args.seed)
    results: list[CandidateMetrics] = []

    trained_models: dict[str, Any] = {}
    for name, model in factories.items():
        local_model = clone(model)
        fit_start = time.perf_counter()
        local_model.fit(x_train, y_train)
        fit_ms = (time.perf_counter() - fit_start) * 1000.0
        metrics = evaluate_trained_candidate(
            candidate=name,
            runtime="native",
            trained_model=local_model,
            x_test=x_test,
            y_test=y_test,
            threshold=args.threshold,
            latency_samples=args.latency_samples,
        )
        metrics.fit_time_ms = float(fit_ms)
        results.append(metrics)
        trained_models[name] = local_model

    baseline = next((row for row in results if row.candidate == "xgb_baseline" and row.runtime == "native"), None)

    if baseline is not None and "xgb_baseline" in trained_models:
        onnx_runtime, error = maybe_build_onnx_runtime(trained_models["xgb_baseline"], x_test.shape[1])
        if onnx_runtime is None:
            skipped_notes.append(error or "onnx runtime candidate skipped")
        else:
            onnx_metrics = evaluate_trained_candidate(
                candidate="xgb_baseline",
                runtime="onnxruntime",
                trained_model=onnx_runtime,
                x_test=x_test,
                y_test=y_test,
                threshold=args.threshold,
                latency_samples=args.latency_samples,
                notes=["ONNX runtime variant generated from trained xgb_baseline"],
            )
            results.append(onnx_metrics)

    promoted, selection_notes = pick_promotion_candidate(
        results,
        sla_max_p95_ms=args.sla_max_p95_ms,
        min_pr_auc=args.min_pr_auc,
        min_recall=args.min_recall,
    )

    gate = {"blocked": True, "reason": "baseline_missing", "latency_improved": False}
    if baseline is not None and promoted is not None:
        gate = apply_regression_gate(
            baseline,
            promoted,
            max_pr_auc_drop=args.max_pr_auc_drop,
            max_recall_drop=args.max_recall_drop,
        )

    status = "promote"
    if promoted is None or gate.get("blocked", False):
        status = "hold"

    report = {
        "status": status,
        "dataset_source": args.dataset_source,
        "split": "time_holdout",
        "sla": {"max_p95_ms": args.sla_max_p95_ms},
        "kpi_thresholds": {"min_pr_auc": args.min_pr_auc, "min_recall": args.min_recall},
        "candidates": [asdict(item) for item in results],
        "baseline_candidate": asdict(baseline) if baseline else None,
        "promotion_candidate": asdict(promoted) if promoted else None,
        "regression_gate": gate,
        "notes": skipped_notes + selection_notes,
    }

    out_json = args.output_dir / f"{args.dataset_source}_inference_candidate_report.json"
    out_csv = args.output_dir / f"{args.dataset_source}_inference_candidate_report.csv"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(item) for item in results]).to_csv(out_csv, index=False)

    print(json.dumps({"status": status, "report_json": str(out_json), "report_csv": str(out_csv)}))
    return 0 if status == "promote" else 1


if __name__ == "__main__":
    raise SystemExit(main())

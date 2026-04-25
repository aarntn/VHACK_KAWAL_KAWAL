from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLDS = REPO_ROOT / "project" / "models" / "decision_thresholds.pkl"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "project" / "outputs" / "monitoring" / "ring_ablation_report.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "project" / "outputs" / "monitoring" / "ring_ablation_report.md"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "project" / "outputs" / "monitoring" / "ring_ablation_high_risk_cohorts.csv"


@dataclass
class EvalResult:
    precision: float
    recall: float
    f1: float
    fpr: float
    pr_auc: float
    approve_rate: float
    flag_rate: float
    block_rate: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baseline vs baseline+ring under identical split/threshold settings.")
    p.add_argument("--n-samples", type=int, default=120_000)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ring-weight", type=float, default=0.08)
    p.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS)
    p.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    p.add_argument("--output-high-risk-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return p.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def synthetic_dataset(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ring_member = rng.random(n) < 0.18
    ring_score = np.where(ring_member, rng.beta(3.5, 1.6, n), rng.beta(0.9, 8.0, n))

    device_shared_users_24h = np.where(ring_member, rng.poisson(5.0, n), rng.poisson(1.0, n))
    cash_flow_velocity_1h = np.where(ring_member, rng.poisson(9.0, n), rng.poisson(2.0, n))
    p2p_counterparties_24h = np.where(ring_member, rng.poisson(11.0, n), rng.poisson(2.5, n))

    tx_type = np.where(rng.random(n) < (0.45 + 0.20 * ring_member), "CASH_OUT", "MERCHANT")
    channel = np.where(rng.random(n) < (0.42 + 0.18 * ring_member), "AGENT", "APP")
    account_age_days = np.where(ring_member, rng.integers(1, 90, n), rng.integers(30, 720, n))

    base_signal = (
        0.020 * device_shared_users_24h
        + 0.017 * cash_flow_velocity_1h
        + 0.013 * p2p_counterparties_24h
        + 0.060 * (tx_type == "CASH_OUT").astype(float)
        + 0.050 * (channel == "AGENT").astype(float)
        + 0.030 * (account_age_days < 14).astype(float)
    )

    # Fraud probability depends strongly on ring_score, which baseline does not directly observe.
    true_logit = -3.90 + 2.40 * ring_score + 2.00 * base_signal
    y = (rng.random(n) < sigmoid(true_logit)).astype(int)

    # Baseline score: calibrated noisy proxy from non-ring features.
    score_logit = -0.20 + 2.20 * base_signal + rng.normal(0.0, 0.50, n)
    baseline_score = sigmoid(score_logit)

    return pd.DataFrame(
        {
            "is_fraud": y,
            "baseline_score": baseline_score,
            "ring_score": ring_score,
            "device_shared_users_24h": device_shared_users_24h,
            "cash_flow_velocity_1h": cash_flow_velocity_1h,
            "p2p_counterparties_24h": p2p_counterparties_24h,
            "tx_type": tx_type,
            "channel": channel,
            "account_age_days": account_age_days,
        }
    )


def decisions_from_score(score: np.ndarray, approve_threshold: float, block_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    approve = score < approve_threshold
    block = score >= block_threshold
    flag = (~approve) & (~block)
    return approve, flag, block


def compute_metrics(y_true: np.ndarray, score: np.ndarray, block_pred: np.ndarray, approve: np.ndarray, flag: np.ndarray) -> EvalResult:
    tn, fp, fn, tp = confusion_matrix(y_true, block_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    pr_auc = float(average_precision_score(y_true, score)) if int(np.sum(y_true)) > 0 else 0.0
    return EvalResult(
        precision=float(precision_score(y_true, block_pred, zero_division=0)),
        recall=float(recall_score(y_true, block_pred, zero_division=0)),
        f1=float(f1_score(y_true, block_pred, zero_division=0)),
        fpr=fpr,
        pr_auc=pr_auc,
        approve_rate=float(np.mean(approve)),
        flag_rate=float(np.mean(flag)),
        block_rate=float(np.mean(block_pred)),
    )


def metric_row(name: str, baseline: EvalResult, ring: EvalResult) -> dict[str, float | str]:
    return {
        "segment": name,
        "baseline_precision": round(baseline.precision, 6),
        "baseline_recall": round(baseline.recall, 6),
        "baseline_f1": round(baseline.f1, 6),
        "baseline_fpr": round(baseline.fpr, 6),
        "baseline_pr_auc": round(baseline.pr_auc, 6),
        "ring_precision": round(ring.precision, 6),
        "ring_recall": round(ring.recall, 6),
        "ring_f1": round(ring.f1, 6),
        "ring_fpr": round(ring.fpr, 6),
        "ring_pr_auc": round(ring.pr_auc, 6),
        "delta_precision": round(ring.precision - baseline.precision, 6),
        "delta_recall": round(ring.recall - baseline.recall, 6),
        "delta_f1": round(ring.f1 - baseline.f1, 6),
        "delta_fpr": round(ring.fpr - baseline.fpr, 6),
        "delta_pr_auc": round(ring.pr_auc - baseline.pr_auc, 6),
        "baseline_approve_rate": round(baseline.approve_rate, 6),
        "baseline_flag_rate": round(baseline.flag_rate, 6),
        "baseline_block_rate": round(baseline.block_rate, 6),
        "ring_approve_rate": round(ring.approve_rate, 6),
        "ring_flag_rate": round(ring.flag_rate, 6),
        "ring_block_rate": round(ring.block_rate, 6),
        "delta_approve_rate": round(ring.approve_rate - baseline.approve_rate, 6),
        "delta_flag_rate": round(ring.flag_rate - baseline.flag_rate, 6),
        "delta_block_rate": round(ring.block_rate - baseline.block_rate, 6),
    }


def to_markdown(report: dict) -> str:
    agg = report["aggregate"]
    rows = [
        "# Baseline vs Baseline+Ring Evaluation",
        "",
        f"Evidence class: `{report['evidence_class']}`",
        "This report is generated from a synthetic projection dataset and is useful for directional analysis, not production-calibrated proof.",
        "",
        f"Generated at (UTC): `{report['generated_at_utc']}`",
        f"Split: stratified test_size=`{report['split']['test_size']}` with random_state=`{report['split']['seed']}` (identical for both variants).",
        f"Thresholds (shared): approve=`{report['thresholds']['approve_threshold']:.6f}`, block=`{report['thresholds']['block_threshold']:.6f}`.",
        "",
        "## Aggregate metrics",
        "",
        "| Variant | Precision | Recall | F1 | FPR | PR-AUC | Approve rate | Flag rate | Block rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| baseline | {agg['baseline_precision']:.4f} | {agg['baseline_recall']:.4f} | {agg['baseline_f1']:.4f} | "
            f"{agg['baseline_fpr']:.4f} | {agg['baseline_pr_auc']:.4f} | {agg['baseline_approve_rate']:.4f} | "
            f"{agg['baseline_flag_rate']:.4f} | {agg['baseline_block_rate']:.4f} |"
        ),
        (
            f"| baseline+ring | {agg['ring_precision']:.4f} | {agg['ring_recall']:.4f} | {agg['ring_f1']:.4f} | "
            f"{agg['ring_fpr']:.4f} | {agg['ring_pr_auc']:.4f} | {agg['ring_approve_rate']:.4f} | "
            f"{agg['ring_flag_rate']:.4f} | {agg['ring_block_rate']:.4f} |"
        ),
        (
            f"| Δ (ring - baseline) | {agg['delta_precision']:+.4f} | {agg['delta_recall']:+.4f} | {agg['delta_f1']:+.4f} | "
            f"{agg['delta_fpr']:+.4f} | {agg['delta_pr_auc']:+.4f} | {agg['delta_approve_rate']:+.4f} | "
            f"{agg['delta_flag_rate']:+.4f} | {agg['delta_block_rate']:+.4f} |"
        ),
        "",
        "## High-risk mule-prone cohorts",
        "",
        "| Cohort | N | Baseline Recall | Ring Recall | Δ Recall | Baseline F1 | Ring F1 | Δ F1 | Baseline FPR | Ring FPR | Δ FPR | Δ Block rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["high_risk_cohorts"]:
        rows.append(
            f"| {row['segment']} | {row['sample_count']} | {row['baseline_recall']:.4f} | {row['ring_recall']:.4f} | {row['delta_recall']:+.4f} | "
            f"{row['baseline_f1']:.4f} | {row['ring_f1']:.4f} | {row['delta_f1']:+.4f} | {row['baseline_fpr']:.4f} | "
            f"{row['ring_fpr']:.4f} | {row['delta_fpr']:+.4f} | {row['delta_block_rate']:+.4f} |"
        )
    rows.append("")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()

    thresholds = pickle.loads(args.thresholds_path.read_bytes())
    approve_threshold = float(thresholds["approve_threshold"])
    block_threshold = float(thresholds["block_threshold"])

    df = synthetic_dataset(args.n_samples, args.seed)
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["is_fraud"],
    )

    baseline_score = test_df["baseline_score"].to_numpy()
    ring_score = clamp01(baseline_score + (args.ring_weight * test_df["ring_score"].to_numpy()))
    y_true = test_df["is_fraud"].to_numpy()

    base_approve, base_flag, base_block = decisions_from_score(baseline_score, approve_threshold, block_threshold)
    ring_approve, ring_flag, ring_block = decisions_from_score(ring_score, approve_threshold, block_threshold)

    base_metrics = compute_metrics(y_true, baseline_score, base_block, base_approve, base_flag)
    ring_metrics = compute_metrics(y_true, ring_score, ring_block, ring_approve, ring_flag)

    aggregate = metric_row("all", base_metrics, ring_metrics)

    cohorts: dict[str, pd.Series] = {
        "mule:ring_score_ge_0.70": test_df["ring_score"] >= 0.70,
        "mule:shared_device_ge_5": test_df["device_shared_users_24h"] >= 5,
        "mule:cashout_velocity": (test_df["tx_type"] == "CASH_OUT") & (test_df["cash_flow_velocity_1h"] >= 8),
        "mule:agent_new_account": (test_df["channel"] == "AGENT") & (test_df["account_age_days"] < 21),
        "mule:composite_any": (
            (test_df["ring_score"] >= 0.60)
            | (test_df["device_shared_users_24h"] >= 5)
            | ((test_df["tx_type"] == "CASH_OUT") & (test_df["cash_flow_velocity_1h"] >= 8))
        ),
    }

    rows: list[dict[str, float | str | int]] = []
    for name, mask in cohorts.items():
        idx = np.where(mask.to_numpy())[0]
        if len(idx) == 0:
            continue
        c_y = y_true[idx]
        c_base = compute_metrics(c_y, baseline_score[idx], base_block[idx], base_approve[idx], base_flag[idx])
        c_ring = compute_metrics(c_y, ring_score[idx], ring_block[idx], ring_approve[idx], ring_flag[idx])
        row = metric_row(name, c_base, c_ring)
        row["sample_count"] = int(len(idx))
        rows.append(row)

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "synthetic_projection",
        "data": {
            "type": "synthetic",
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "fraud_rate_train": round(float(train_df["is_fraud"].mean()), 6),
            "fraud_rate_test": round(float(test_df["is_fraud"].mean()), 6),
        },
        "split": {
            "strategy": "stratified_holdout",
            "test_size": float(args.test_size),
            "seed": int(args.seed),
        },
        "thresholds": {
            "approve_threshold": approve_threshold,
            "block_threshold": block_threshold,
        },
        "ring_weight": float(args.ring_weight),
        "aggregate": aggregate,
        "high_risk_cohorts": rows,
    }

    for p in [args.output_json, args.output_markdown, args.output_high_risk_csv]:
        p.parent.mkdir(parents=True, exist_ok=True)

    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    args.output_markdown.write_text(to_markdown(output), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output_high_risk_csv, index=False)

    print("Ring ablation evaluation complete")
    print(f"- JSON: {args.output_json}")
    print(f"- Markdown: {args.output_markdown}")
    print(f"- High-risk CSV: {args.output_high_risk_csv}")


if __name__ == "__main__":
    main()

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from project.app.rules import compute_user_segment


@dataclass
class SegmentCalibrationResult:
    segment: str
    sample_count: int
    approve_threshold: float
    block_threshold: float
    min_block_precision: float
    max_approve_to_flag_fpr: float
    block_precision: float
    approve_to_flag_fpr: float
    pr_auc: float
    objective: float


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _objective(tp: float, fp: float, fn: float, pr_auc: float, fp_cost: float, fn_cost: float) -> float:
    return float(pr_auc - (fp_cost * fp) - (fn_cost * fn) + (0.1 * tp))


def calibrate_segment(
    segment: str,
    scores: np.ndarray,
    labels: np.ndarray,
    min_block_precision: float,
    max_approve_to_flag_fpr: float,
    fp_cost: float,
    fn_cost: float,
) -> SegmentCalibrationResult:
    if len(scores) == 0:
        raise ValueError(f"segment '{segment}' has no samples")

    pr_auc = float(average_precision_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0
    candidates = np.linspace(0.02, 0.98, 97)

    best: SegmentCalibrationResult | None = None
    for approve_threshold in candidates:
        block_candidates = candidates[candidates > approve_threshold]
        for block_threshold in block_candidates:
            approve_bucket = scores < approve_threshold
            flag_bucket = (scores >= approve_threshold) & (scores < block_threshold)
            block_bucket = scores >= block_threshold

            tp = float(np.sum(labels[block_bucket] == 1))
            fp = float(np.sum(labels[block_bucket] == 0))
            fn = float(np.sum(labels[~block_bucket] == 1))
            tn = float(np.sum(labels[~block_bucket] == 0))

            block_precision = _safe_ratio(tp, tp + fp)
            approve_to_flag_fpr = _safe_ratio(np.sum((labels == 0) & flag_bucket), np.sum(labels == 0))

            if block_precision < min_block_precision:
                continue
            if approve_to_flag_fpr > max_approve_to_flag_fpr:
                continue

            objective = _objective(tp=tp, fp=fp, fn=fn, pr_auc=pr_auc, fp_cost=fp_cost, fn_cost=fn_cost)
            candidate = SegmentCalibrationResult(
                segment=segment,
                sample_count=int(len(scores)),
                approve_threshold=float(approve_threshold),
                block_threshold=float(block_threshold),
                min_block_precision=float(min_block_precision),
                max_approve_to_flag_fpr=float(max_approve_to_flag_fpr),
                block_precision=float(block_precision),
                approve_to_flag_fpr=float(approve_to_flag_fpr),
                pr_auc=pr_auc,
                objective=float(objective),
            )
            if best is None or candidate.objective > best.objective:
                best = candidate

    if best is None:
        raise ValueError(
            f"No threshold pair passed acceptance gates for segment '{segment}'. "
            f"Relax min_block_precision or max_approve_to_flag_fpr."
        )
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate per-segment approve/block thresholds.")
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV with fraud_label and model_score columns")
    parser.add_argument("--label-col", default="fraud_label")
    parser.add_argument("--score-col", default="model_score")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-thresholds-pkl", type=Path, required=True)
    parser.add_argument("--min-block-precision", type=float, default=0.80)
    parser.add_argument("--max-approve-to-flag-fpr", type=float, default=0.03)
    parser.add_argument("--fp-cost", type=float, default=0.0002)
    parser.add_argument("--fn-cost", type=float, default=0.0010)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    required = {args.label_col, args.score_col, "account_age_days", "TransactionAmt", "channel"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df.copy()
    df["segment"] = df.apply(lambda row: compute_user_segment(row.to_dict()), axis=1)

    segment_thresholds: Dict[str, Dict[str, float | Dict[str, float]]] = {}
    calibration_rows = []

    for segment, sdf in df.groupby("segment"):
        scores = sdf[args.score_col].to_numpy(float)
        labels = sdf[args.label_col].to_numpy(int)
        result = calibrate_segment(
            segment=segment,
            scores=scores,
            labels=labels,
            min_block_precision=float(args.min_block_precision),
            max_approve_to_flag_fpr=float(args.max_approve_to_flag_fpr),
            fp_cost=float(args.fp_cost),
            fn_cost=float(args.fn_cost),
        )
        segment_thresholds[segment] = {
            "approve_threshold": result.approve_threshold,
            "block_threshold": result.block_threshold,
            "min_block_precision": result.min_block_precision,
            "max_approve_to_flag_fpr": result.max_approve_to_flag_fpr,
            "calibration_metrics": {
                "block_precision": result.block_precision,
                "approve_to_flag_fpr": result.approve_to_flag_fpr,
                "pr_auc": result.pr_auc,
                "objective": result.objective,
                "sample_count": float(result.sample_count),
            },
        }
        calibration_rows.append(result.__dict__)

    payload = {
        "runtime_recommendation": {
            "segment_thresholds": segment_thresholds,
        },
        "segment_calibration": calibration_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_thresholds_pkl.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with args.output_thresholds_pkl.open("wb") as handle:
        pickle.dump({"segment_thresholds": segment_thresholds}, handle)

    print(f"Calibrated {len(segment_thresholds)} segments")
    print(f"JSON: {args.output_json}")
    print(f"thresholds.pkl: {args.output_thresholds_pkl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

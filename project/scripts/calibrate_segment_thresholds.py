import argparse
import json
import pickle
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from project.app.rules import compute_user_segment

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTIVE_THRESHOLDS_PKL = REPO_ROOT / "project" / "models" / "decision_thresholds.pkl"


@dataclass
class SegmentCalibrationResult:
    segment: str
    sample_count: int
    approve_threshold: float
    block_threshold: float
    min_block_precision: float
    max_approve_to_flag_fpr: float
    approve_selection_constraint: str
    approve_selection_target: float
    approve_selection_metric_name: str
    approve_selection_metric_value: float
    block_precision: float
    approve_to_flag_fpr: float
    legit_approve_rate: float
    approve_band_fnr: float
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
    approve_selection_constraint: str,
    approve_band_fnr_ceiling: float,
    legit_approve_rate_target: float,
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
            legit_approve_rate = _safe_ratio(np.sum((labels == 0) & approve_bucket), np.sum(labels == 0))
            approve_band_fnr = _safe_ratio(np.sum((labels == 1) & approve_bucket), np.sum(labels == 1))

            if block_precision < min_block_precision:
                continue
            if approve_to_flag_fpr > max_approve_to_flag_fpr:
                continue
            if approve_selection_constraint == "approve_band_fnr_ceiling":
                if approve_band_fnr > approve_band_fnr_ceiling:
                    continue
                selection_target = float(approve_band_fnr_ceiling)
                selection_metric_name = "approve_band_fnr"
                selection_metric_value = float(approve_band_fnr)
            elif approve_selection_constraint == "legit_approve_rate_target":
                if legit_approve_rate < legit_approve_rate_target:
                    continue
                selection_target = float(legit_approve_rate_target)
                selection_metric_name = "legit_approve_rate"
                selection_metric_value = float(legit_approve_rate)
            else:
                raise ValueError(f"Unknown approve selection constraint '{approve_selection_constraint}'")

            objective = _objective(tp=tp, fp=fp, fn=fn, pr_auc=pr_auc, fp_cost=fp_cost, fn_cost=fn_cost)
            candidate = SegmentCalibrationResult(
                segment=segment,
                sample_count=int(len(scores)),
                approve_threshold=float(approve_threshold),
                block_threshold=float(block_threshold),
                min_block_precision=float(min_block_precision),
                max_approve_to_flag_fpr=float(max_approve_to_flag_fpr),
                approve_selection_constraint=approve_selection_constraint,
                approve_selection_target=float(selection_target),
                approve_selection_metric_name=selection_metric_name,
                approve_selection_metric_value=float(selection_metric_value),
                block_precision=float(block_precision),
                approve_to_flag_fpr=float(approve_to_flag_fpr),
                legit_approve_rate=float(legit_approve_rate),
                approve_band_fnr=float(approve_band_fnr),
                pr_auc=pr_auc,
                objective=float(objective),
            )
            if best is None:
                best = candidate
                continue
            if candidate.legit_approve_rate > best.legit_approve_rate:
                best = candidate
                continue
            if candidate.legit_approve_rate == best.legit_approve_rate and candidate.objective > best.objective:
                best = candidate

    if best is None:
        raise ValueError(
            f"No threshold pair passed acceptance gates for segment '{segment}'. "
            f"Relax min_block_precision, max_approve_to_flag_fpr, or the approve selection constraint."
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
    parser.add_argument(
        "--approve-selection-constraint",
        choices=["approve_band_fnr_ceiling", "legit_approve_rate_target"],
        default="approve_band_fnr_ceiling",
        help="Constraint used to select approve_threshold while minimizing false declines on legitimate traffic.",
    )
    parser.add_argument(
        "--approve-band-fnr-ceiling",
        type=float,
        default=0.05,
        help="Maximum fraud false-negative rate allowed in approve band when using approve_band_fnr_ceiling constraint.",
    )
    parser.add_argument(
        "--legit-approve-rate-target",
        type=float,
        default=0.80,
        help="Minimum approve rate for legitimate traffic when using legit_approve_rate_target constraint.",
    )
    parser.add_argument("--fp-cost", type=float, default=0.0002)
    parser.add_argument("--fn-cost", type=float, default=0.0010)
    parser.add_argument("--active-thresholds-pkl", type=Path, default=DEFAULT_ACTIVE_THRESHOLDS_PKL)
    parser.add_argument("--max-approve-delta", type=float, default=0.08)
    parser.add_argument("--max-block-delta", type=float, default=0.08)
    parser.add_argument("--force", action="store_true", help="Allow threshold updates beyond configured deltas.")
    return parser.parse_args()


def load_active_thresholds(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {}
    with resolved.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Active thresholds payload must be a dict: {resolved}")
    return payload


def _segment_active_thresholds(active_payload: Dict[str, Any], segment: str) -> tuple[float | None, float | None]:
    segment_payload = active_payload.get("segment_thresholds")
    if isinstance(segment_payload, dict):
        segment_row = segment_payload.get(segment)
        if isinstance(segment_row, dict):
            approve_value = segment_row.get("approve_threshold")
            block_value = segment_row.get("block_threshold")
            if approve_value is not None and block_value is not None:
                return float(approve_value), float(block_value)

    approve_value = active_payload.get("approve_threshold")
    block_value = active_payload.get("block_threshold")
    if approve_value is None or block_value is None:
        return None, None
    return float(approve_value), float(block_value)


def evaluate_threshold_delta_policy(
    segment_thresholds: Dict[str, Dict[str, float | Dict[str, float]]],
    active_payload: Dict[str, Any],
    max_approve_delta: float,
    max_block_delta: float,
) -> Dict[str, Any]:
    checks: list[Dict[str, Any]] = []
    violations: list[Dict[str, Any]] = []

    for segment in sorted(segment_thresholds.keys()):
        candidate = segment_thresholds[segment]
        candidate_approve = float(candidate["approve_threshold"])
        candidate_block = float(candidate["block_threshold"])
        active_approve, active_block = _segment_active_thresholds(active_payload, segment)

        if active_approve is None or active_block is None:
            checks.append(
                {
                    "segment": segment,
                    "status": "no_active_reference",
                    "reason": "No active threshold baseline found; delta policy skipped for this segment.",
                    "deltas": None,
                }
            )
            continue

        approve_delta = round(candidate_approve - active_approve, 6)
        block_delta = round(candidate_block - active_block, 6)
        approve_delta_abs = round(abs(approve_delta), 6)
        block_delta_abs = round(abs(block_delta), 6)
        within_limits = approve_delta_abs <= max_approve_delta and block_delta_abs <= max_block_delta
        reason = (
            "within configured delta limits"
            if within_limits
            else (
                f"Delta exceeds configured limits: |approve_delta|={approve_delta_abs:.6f} (max={max_approve_delta:.6f}), "
                f"|block_delta|={block_delta_abs:.6f} (max={max_block_delta:.6f})."
            )
        )
        check = {
            "segment": segment,
            "status": "ok" if within_limits else "violation",
            "reason": reason,
            "active_thresholds": {
                "approve_threshold": active_approve,
                "block_threshold": active_block,
            },
            "candidate_thresholds": {
                "approve_threshold": candidate_approve,
                "block_threshold": candidate_block,
            },
            "deltas": {
                "approve_delta": approve_delta,
                "approve_delta_abs": approve_delta_abs,
                "block_delta": block_delta,
                "block_delta_abs": block_delta_abs,
            },
        }
        checks.append(check)
        if not within_limits:
            violations.append(check)

    return {"checks": checks, "violations": violations}


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
            approve_selection_constraint=str(args.approve_selection_constraint),
            approve_band_fnr_ceiling=float(args.approve_band_fnr_ceiling),
            legit_approve_rate_target=float(args.legit_approve_rate_target),
            fp_cost=float(args.fp_cost),
            fn_cost=float(args.fn_cost),
        )
        segment_thresholds[segment] = {
            "approve_threshold": result.approve_threshold,
            "block_threshold": result.block_threshold,
            "min_block_precision": result.min_block_precision,
            "max_approve_to_flag_fpr": result.max_approve_to_flag_fpr,
            "approve_selection_constraint": result.approve_selection_constraint,
            "calibration_metrics": {
                "block_precision": result.block_precision,
                "approve_to_flag_fpr": result.approve_to_flag_fpr,
                "legit_approve_rate": result.legit_approve_rate,
                "approve_band_fnr": result.approve_band_fnr,
                "approve_selection_target": result.approve_selection_target,
                "approve_selection_metric_name": result.approve_selection_metric_name,
                "approve_selection_metric_value": result.approve_selection_metric_value,
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

    active_thresholds = load_active_thresholds(args.active_thresholds_pkl)
    delta_policy = evaluate_threshold_delta_policy(
        segment_thresholds=segment_thresholds,
        active_payload=active_thresholds,
        max_approve_delta=float(args.max_approve_delta),
        max_block_delta=float(args.max_block_delta),
    )
    blocked_by_delta_policy = bool(delta_policy["violations"]) and not args.force
    delta_reason = (
        "threshold movement within configured deltas"
        if not delta_policy["violations"]
        else (
            "threshold movement exceeded configured deltas but override accepted because --force was set"
            if args.force
            else "threshold movement exceeded configured deltas; refusing to write new thresholds without --force"
        )
    )
    audit_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked_delta_policy" if blocked_by_delta_policy else "ready_to_write",
        "delta_policy": {
            "max_approve_delta": float(args.max_approve_delta),
            "max_block_delta": float(args.max_block_delta),
            "force": bool(args.force),
            "reason": delta_reason,
            "checks": delta_policy["checks"],
            "violation_count": len(delta_policy["violations"]),
        },
    }
    report_path = args.output_json.with_name("pr_curve_calibration_report.json")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")

    if blocked_by_delta_policy:
        raise ValueError(
            "Threshold movement exceeds configured deltas. Re-run with --force to override. "
            f"See audit report: {report_path}"
        )

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

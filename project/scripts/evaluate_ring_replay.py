from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.fraud_ring_graph import RingAttributeLookup, RingScoreLookup

DEFAULT_THRESHOLDS = REPO_ROOT / "project" / "models" / "decision_thresholds.pkl"
DEFAULT_RING_SCORES = REPO_ROOT / "project" / "outputs" / "monitoring" / "fraud_ring_scores.json"
DEFAULT_ATTRIBUTE_INDEX = REPO_ROOT / "project" / "outputs" / "monitoring" / "fraud_ring_attribute_index.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "project" / "outputs" / "monitoring" / "ring_replay_report.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "project" / "outputs" / "monitoring" / "ring_replay_report.md"


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
    parser = argparse.ArgumentParser(
        description="Evaluate ring uplift on labeled replay data. Requires baseline_score and is_fraud."
    )
    parser.add_argument("--events-path", type=Path, required=True, help="Replay dataset (.jsonl or .csv)")
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--ring-scores-path", type=Path, default=DEFAULT_RING_SCORES)
    parser.add_argument("--attribute-index-path", type=Path, default=DEFAULT_ATTRIBUTE_INDEX)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--label-field", default="is_fraud")
    parser.add_argument("--baseline-score-field", default="baseline_score")
    parser.add_argument("--ring-weight", type=float, default=0.08)
    parser.add_argument("--device-member-cap", type=int, default=8)
    parser.add_argument("--ip-member-cap", type=int, default=16)
    parser.add_argument("--card-member-cap", type=int, default=8)
    parser.add_argument("--attribute-match-min", type=int, default=1)
    parser.add_argument("--attribute-match-component-floor", type=int, default=4)
    parser.add_argument("--max-report-age-hours", type=float, default=72.0)
    return parser.parse_args()


def _parse_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value for {field_name}: {value}")


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"replay dataset not found: {path}")
    records: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                raw = line.strip()
                if not raw:
                    continue
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError(f"invalid JSON object at {path}:{line_no}")
                records.append(parsed)
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(dict(row))
    else:
        raise ValueError("events-path must end with .jsonl or .csv")
    if not records:
        raise ValueError(f"no replay records found in {path}")
    return records


def _clamp01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def _decisions_from_score(
    score: np.ndarray,
    approve_threshold: float,
    block_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    approve = score < approve_threshold
    block = score >= block_threshold
    flag = (~approve) & (~block)
    return approve, flag, block


def _compute_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    block_pred: np.ndarray,
    approve: np.ndarray,
    flag: np.ndarray,
) -> EvalResult:
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


def _metric_row(name: str, baseline: EvalResult, ring: EvalResult) -> dict[str, float | str]:
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


def _member_cap(attr_type: str, args: argparse.Namespace) -> int:
    if attr_type == "device":
        return int(args.device_member_cap)
    if attr_type == "ip":
        return int(args.ip_member_cap)
    if attr_type == "card":
        return int(args.card_member_cap)
    return int(args.device_member_cap)


def _collect_tokens(record: dict[str, Any]) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    if record.get("device_id"):
        tokens.append(("device", f"device:{record['device_id']}"))
    if record.get("ip_subnet"):
        tokens.append(("ip", f"ip:{record['ip_subnet']}"))
    if record.get("card_prefix"):
        tokens.append(("card", f"card:{record['card_prefix']}"))
    return tokens


def _resolve_ring_match(
    record: dict[str, Any],
    args: argparse.Namespace,
    ring_scores: RingScoreLookup,
    attribute_index: RingAttributeLookup,
) -> tuple[float, str]:
    account_score = float(ring_scores.get(str(record.get("user_id", "")), 0.0) or 0.0)
    if account_score > 0.0:
        return account_score, "account_member"

    matches: list[dict[str, Any]] = []
    max_report_age_seconds = float(args.max_report_age_hours) * 3600.0
    now_ts = time.time()
    for attr_type, token in _collect_tokens(record):
        entry = attribute_index.get(token)
        if not entry:
            continue
        member_count = int(entry.get("member_count", 0) or 0)
        if _member_cap(attr_type, args) > 0 and member_count > _member_cap(attr_type, args):
            continue
        generated_at = float(entry.get("generated_at", 0.0) or 0.0)
        if generated_at > 0.0 and now_ts - generated_at > max_report_age_seconds:
            continue
        matches.append(
            {
                "attr_type": attr_type,
                "max_ring_score": float(entry.get("max_ring_score", 0.0) or 0.0),
                "max_ring_size": int(entry.get("max_ring_size", 0) or 0),
            }
        )

    if not matches:
        return 0.0, "none"

    distinct_types = {match["attr_type"] for match in matches}
    strongest = max(matches, key=lambda match: (match["max_ring_score"], match["max_ring_size"]))
    corroboration_ok = (
        len(matches) >= max(1, int(args.attribute_match_min))
        and (
            len(matches) >= 2
            or len(distinct_types) >= 2
            or int(strongest["max_ring_size"]) >= int(args.attribute_match_component_floor)
        )
    )
    if not corroboration_ok:
        return 0.0, "none"
    return float(strongest["max_ring_score"]), "attribute_match"


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Ring Replay Evaluation",
        "",
        f"Evidence class: `{report['evidence_class']}`",
        f"Generated at (UTC): `{report['generated_at_utc']}`",
        f"Replay rows: `{report['data']['rows_total']}`",
        "",
        "This report is generated from labeled replay data and should be used as the primary technical-evidence artifact when available.",
        "",
        "## Aggregate metrics",
        "",
        "| Variant | Precision | Recall | F1 | FPR | PR-AUC | Approve rate | Flag rate | Block rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    agg = report["aggregate"]
    for variant, prefix in (("baseline", "baseline"), ("baseline+ring", "ring")):
        lines.append(
            f"| {variant} | {agg[f'{prefix}_precision']:.4f} | {agg[f'{prefix}_recall']:.4f} | {agg[f'{prefix}_f1']:.4f} | "
            f"{agg[f'{prefix}_fpr']:.4f} | {agg[f'{prefix}_pr_auc']:.4f} | {agg[f'{prefix}_approve_rate']:.4f} | "
            f"{agg[f'{prefix}_flag_rate']:.4f} | {agg[f'{prefix}_block_rate']:.4f} |"
        )
    lines.append(
        f"| Δ (ring - baseline) | {agg['delta_precision']:+.4f} | {agg['delta_recall']:+.4f} | {agg['delta_f1']:+.4f} | "
        f"{agg['delta_fpr']:+.4f} | {agg['delta_pr_auc']:+.4f} | {agg['delta_approve_rate']:+.4f} | "
        f"{agg['delta_flag_rate']:+.4f} | {agg['delta_block_rate']:+.4f} |"
    )
    lines.extend(
        [
            "",
            "## Match cohorts",
            "",
            "| Cohort | N | Δ Recall | Δ F1 | Δ FPR | Δ Block rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["match_cohorts"]:
        lines.append(
            f"| {row['segment']} | {row['sample_count']} | {row['delta_recall']:+.4f} | {row['delta_f1']:+.4f} | "
            f"{row['delta_fpr']:+.4f} | {row['delta_block_rate']:+.4f} |"
        )
    if report["fairness_segments"]:
        lines.extend(
            [
                "",
                "## Fairness-oriented segments",
                "",
                "| Segment | N | Δ Recall | Δ F1 | Δ FPR | Δ Block rate |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["fairness_segments"]:
            lines.append(
                f"| {row['segment']} | {row['sample_count']} | {row['delta_recall']:+.4f} | {row['delta_f1']:+.4f} | "
                f"{row['delta_fpr']:+.4f} | {row['delta_block_rate']:+.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    records = _load_records(args.events_path)
    if not args.thresholds_path.exists():
        raise FileNotFoundError(f"thresholds file not found: {args.thresholds_path}")
    thresholds = pickle.loads(args.thresholds_path.read_bytes())
    approve_threshold = float(thresholds["approve_threshold"])
    block_threshold = float(thresholds["block_threshold"])

    ring_scores = RingScoreLookup(args.ring_scores_path)
    attribute_index = RingAttributeLookup(args.attribute_index_path)

    baseline_scores: list[float] = []
    ring_scores_applied: list[float] = []
    labels: list[int] = []
    match_types: list[str] = []
    fairness_segments: list[str] = []

    for idx, record in enumerate(records, 1):
        if args.label_field not in record:
            raise ValueError(f"missing {args.label_field} in replay row {idx}")
        if args.baseline_score_field not in record:
            raise ValueError(f"missing {args.baseline_score_field} in replay row {idx}")
        label = int(_parse_bool(record[args.label_field], field_name=args.label_field))
        baseline_score = float(record[args.baseline_score_field])
        ring_signal, match_type = _resolve_ring_match(record, args, ring_scores, attribute_index)
        ring_score = float(_clamp01(np.asarray([baseline_score + (args.ring_weight * ring_signal)]))[0])

        labels.append(label)
        baseline_scores.append(baseline_score)
        ring_scores_applied.append(ring_score)
        match_types.append(match_type)

        account_age = int(float(record.get("account_age_days", 0) or 0))
        agent_assisted = str(record.get("channel", "")).strip().upper() == "AGENT" or _parse_bool(
            record.get("is_agent_assisted", False),
            field_name="is_agent_assisted",
        )
        if agent_assisted:
            fairness_segments.append("agent_assisted")
        elif account_age < 30:
            fairness_segments.append("new_user")
        else:
            fairness_segments.append("established_user")

    y_true = np.asarray(labels)
    baseline = np.asarray(baseline_scores)
    ring_variant = np.asarray(ring_scores_applied)

    base_approve, base_flag, base_block = _decisions_from_score(baseline, approve_threshold, block_threshold)
    ring_approve, ring_flag, ring_block = _decisions_from_score(ring_variant, approve_threshold, block_threshold)

    aggregate = _metric_row(
        "all",
        _compute_metrics(y_true, baseline, base_block, base_approve, base_flag),
        _compute_metrics(y_true, ring_variant, ring_block, ring_approve, ring_flag),
    )

    match_rows: list[dict[str, Any]] = []
    for match_type in ("account_member", "attribute_match", "none"):
        idx = np.where(np.asarray(match_types) == match_type)[0]
        if len(idx) == 0:
            continue
        row = _metric_row(
            f"match_type:{match_type}",
            _compute_metrics(y_true[idx], baseline[idx], base_block[idx], base_approve[idx], base_flag[idx]),
            _compute_metrics(y_true[idx], ring_variant[idx], ring_block[idx], ring_approve[idx], ring_flag[idx]),
        )
        row["sample_count"] = int(len(idx))
        match_rows.append(row)

    fairness_rows: list[dict[str, Any]] = []
    fairness_segment_array = np.asarray(fairness_segments)
    for segment in sorted(set(fairness_segments)):
        idx = np.where(fairness_segment_array == segment)[0]
        if len(idx) == 0:
            continue
        row = _metric_row(
            f"fairness:{segment}",
            _compute_metrics(y_true[idx], baseline[idx], base_block[idx], base_approve[idx], base_flag[idx]),
            _compute_metrics(y_true[idx], ring_variant[idx], ring_block[idx], ring_approve[idx], ring_flag[idx]),
        )
        row["sample_count"] = int(len(idx))
        fairness_rows.append(row)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "measured_replay",
        "data": {
            "rows_total": int(len(records)),
            "fraud_rate": round(float(np.mean(y_true)), 6),
        },
        "thresholds": {
            "approve_threshold": approve_threshold,
            "block_threshold": block_threshold,
            "ring_weight": float(args.ring_weight),
        },
        "artifacts": {
            "events_path": str(args.events_path),
            "ring_scores_path": str(args.ring_scores_path),
            "attribute_index_path": str(args.attribute_index_path),
        },
        "aggregate": aggregate,
        "match_cohorts": match_rows,
        "fairness_segments": fairness_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_markdown.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Measured replay report written: {args.output_json}")
    print(f"Measured replay markdown written: {args.output_markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

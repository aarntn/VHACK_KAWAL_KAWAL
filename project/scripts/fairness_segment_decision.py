import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENT_CSV = REPO_ROOT / "project" / "outputs" / "governance" / "fairness_segment_metrics.csv"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "project" / "outputs" / "governance" / "fairness_action_plan.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "project" / "outputs" / "governance" / "fairness_action_plan.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fairness action plan from segment fairness metrics.")
    parser.add_argument("--segment-metrics-csv", type=Path, default=DEFAULT_SEGMENT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--max-fpr-gap", type=float, default=0.08)
    parser.add_argument("--max-fnr-gap", type=float, default=0.12)
    parser.add_argument("--max-precision-gap", type=float, default=0.12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-segment-size", type=int, default=200)
    return parser.parse_args()


def _score_row(row: pd.Series, max_fpr_gap: float, max_fnr_gap: float, max_precision_gap: float) -> float:
    return max(
        abs(float(row.get("fpr_gap_vs_overall", 0.0))) / max(max_fpr_gap, 1e-12),
        abs(float(row.get("fnr_gap_vs_overall", 0.0))) / max(max_fnr_gap, 1e-12),
        abs(float(row.get("precision_gap_vs_overall", 0.0))) / max(max_precision_gap, 1e-12),
    )


def rank_worst_segments(
    df: pd.DataFrame,
    max_fpr_gap: float,
    max_fnr_gap: float,
    max_precision_gap: float,
    min_segment_size: int,
    top_k: int,
) -> list[dict]:
    work = df.copy()
    if "sample_count" in work.columns:
        work = work[work["sample_count"] >= int(min_segment_size)]
    if work.empty:
        return []

    work["severity_score"] = work.apply(
        lambda r: _score_row(r, max_fpr_gap=max_fpr_gap, max_fnr_gap=max_fnr_gap, max_precision_gap=max_precision_gap),
        axis=1,
    )

    ranked = work.sort_values(["severity_score", "segment"], ascending=[False, True]).head(int(top_k))
    records: list[dict] = []
    for _, row in ranked.iterrows():
        violations: list[str] = []
        if abs(float(row.get("fpr_gap_vs_overall", 0.0))) > max_fpr_gap:
            violations.append("fpr_gap")
        if abs(float(row.get("fnr_gap_vs_overall", 0.0))) > max_fnr_gap:
            violations.append("fnr_gap")
        if abs(float(row.get("precision_gap_vs_overall", 0.0))) > max_precision_gap:
            violations.append("precision_gap")

        threshold_action = "hold"
        if float(row.get("fpr_gap_vs_overall", 0.0)) > max_fpr_gap:
            threshold_action = "raise_threshold_for_segment"
        elif float(row.get("fnr_gap_vs_overall", 0.0)) > max_fnr_gap:
            threshold_action = "lower_threshold_for_segment"

        records.append(
            {
                "segment": str(row.get("segment")),
                "sample_count": int(row.get("sample_count", 0)),
                "severity_score": round(float(row.get("severity_score", 0.0)), 6),
                "violations": violations,
                "threshold_action": threshold_action,
                "fpr_gap_vs_overall": round(float(row.get("fpr_gap_vs_overall", 0.0)), 6),
                "fnr_gap_vs_overall": round(float(row.get("fnr_gap_vs_overall", 0.0)), 6),
                "precision_gap_vs_overall": round(float(row.get("precision_gap_vs_overall", 0.0)), 6),
            }
        )

    return records


def build_action_plan(ranked_segments: list[dict]) -> dict:
    severe_count = sum(1 for r in ranked_segments if r["violations"])
    retraining_required = severe_count > 0 and any(r["threshold_action"] == "hold" for r in ranked_segments)
    status = "block_release" if severe_count > 0 else "permit_with_monitoring"

    next_actions = [
        "Apply per-segment threshold changes only where compliance policy permits.",
        "Recompute fairness_explainability_report.py after any threshold/model change.",
        "Escalate to retraining when severe disparities persist for two consecutive runs.",
    ]

    return {
        "status": status,
        "severe_segment_count": severe_count,
        "retraining_required": retraining_required,
        "worst_segments": ranked_segments,
        "next_actions": next_actions,
    }


def build_markdown(plan: dict) -> str:
    lines = [
        "# Fairness Action Plan",
        "",
        f"- Status: `{plan['status']}`",
        f"- Severe segment count: `{plan['severe_segment_count']}`",
        f"- Retraining required: `{plan['retraining_required']}`",
        "",
        "## Worst segments",
        "",
        "| Segment | Severity score | Violations | Suggested threshold action |",
        "| --- | ---: | --- | --- |",
    ]
    for row in plan["worst_segments"]:
        lines.append(
            f"| {row['segment']} | {row['severity_score']:.3f} | {', '.join(row['violations']) or 'none'} | {row['threshold_action']} |"
        )

    lines.extend(["", "## Next actions", ""])
    for item in plan["next_actions"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    path = args.segment_metrics_csv.expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Segment metrics CSV not found: {path}")

    df = pd.read_csv(path)
    ranked = rank_worst_segments(
        df,
        max_fpr_gap=args.max_fpr_gap,
        max_fnr_gap=args.max_fnr_gap,
        max_precision_gap=args.max_precision_gap,
        min_segment_size=args.min_segment_size,
        top_k=args.top_k,
    )
    plan = build_action_plan(ranked)

    output_json = args.output_json.expanduser().resolve()
    output_md = args.output_markdown.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown(plan), encoding="utf-8")

    print("Fairness action plan generated")
    print(f"- JSON: {output_json}")
    print(f"- Markdown: {output_md}")
    print(f"- Status: {plan['status']}")


if __name__ == "__main__":
    main()

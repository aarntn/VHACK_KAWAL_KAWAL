#!/usr/bin/env python3
"""Analyze whether ring detection's FPR increase disproportionately affects
already-biased fairness segments.

Compares the global ring ablation (+11.91% recall, +3.30% FPR) against the
per-segment fairness metrics to determine whether ring signals amplify existing
disparities or are segment-neutral.

Output:
  project/outputs/governance/ring_fairness_impact.json
  project/outputs/governance/ring_fairness_impact.md
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RING_SUMMARY_PATH = REPO_ROOT / "project" / "outputs" / "monitoring" / "fraud_ring_summary.json"
FAIRNESS_METRICS_PATH = REPO_ROOT / "project" / "outputs" / "governance" / "fairness_segment_metrics.csv"
RING_ABLATION_PATH = REPO_ROOT / "project" / "outputs" / "monitoring" / "ring_ablation_report.md"
AUDIT_LOG_PATH = REPO_ROOT / "project" / "outputs" / "audit" / "fraud_audit_log.jsonl"
OUTPUT_DIR = REPO_ROOT / "project" / "outputs" / "governance"

# From ring ablation report (baseline vs baseline+ring, global)
RING_GLOBAL_DELTA_RECALL = 0.1191
RING_GLOBAL_DELTA_FPR    = 0.0330

# Global baseline FPR and FNR from fairness report
GLOBAL_BASELINE_FPR = 0.1276
GLOBAL_BASELINE_FNR = 1 - 0.5112  # = 0.4888

SEVERE_THRESHOLD = 1.0


@dataclass
class SegmentRingRisk:
    segment: str
    baseline_fpr: float
    baseline_fnr: float
    severity_score: float
    severity_label: str
    sample_count: int
    # Estimated ring impact (proportional projection from global delta)
    est_ring_fpr_delta: float = 0.0
    est_ring_fnr_delta: float = 0.0
    projected_fpr: float = 0.0
    projected_fnr: float = 0.0
    # Amplification: does the segment already have high FPR? Ring FPR increase hits harder.
    fpr_amplification_ratio: float = 0.0
    verdict: str = "neutral"
    concern: str = ""


def load_fairness_metrics(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


def load_ring_audit_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "audit_log_available": False,
            "records_scanned": 0,
            "ring_applied_count": 0,
            "ring_suppressed_count": 0,
            "blocked_with_ring_only": 0,
            "blocked_with_ring_corroboration": 0,
            "match_type_counts": {},
        }

    summary = {
        "audit_log_available": True,
        "records_scanned": 0,
        "ring_applied_count": 0,
        "ring_suppressed_count": 0,
        "blocked_with_ring_only": 0,
        "blocked_with_ring_corroboration": 0,
        "match_type_counts": {},
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            summary["records_scanned"] += 1
            match_type = str(record.get("ring_match_type", "none") or "none")
            summary["match_type_counts"][match_type] = int(summary["match_type_counts"].get(match_type, 0)) + 1

            context_scores = record.get("context_scores", {})
            if not isinstance(context_scores, dict):
                context_scores = {}
            effective_adjustment = float(context_scores.get("ring_effective_adjustment", 0.0) or 0.0)
            ring_reason_codes = record.get("ring_reason_codes", [])
            if not isinstance(ring_reason_codes, list):
                ring_reason_codes = []
            guard_details = context_scores.get("ring_block_guard_details", {})
            if not isinstance(guard_details, dict):
                guard_details = {}

            if effective_adjustment > 0.0:
                summary["ring_applied_count"] += 1
            if (
                "RING_ADJ_SUPPRESSED_EVIDENCE_GATES" in ring_reason_codes
                or "RING_BLOCK_ESCALATION_SUPPRESSED_FAIRNESS" in ring_reason_codes
                or float(guard_details.get("suppressed_adjustment", 0.0) or 0.0) > 0.0
            ):
                summary["ring_suppressed_count"] += 1
            if str(record.get("decision", "")) == "BLOCK" and effective_adjustment > 0.0:
                if bool(guard_details.get("corroborated", False)):
                    summary["blocked_with_ring_corroboration"] += 1
                else:
                    summary["blocked_with_ring_only"] += 1
    return summary


def build_segment_ring_risks(metrics_rows: list[dict]) -> list[SegmentRingRisk]:
    results: list[SegmentRingRisk] = []
    for r in metrics_rows:
        severity_label = r.get("severity_label", "ok")
        severity_score = float(r.get("severity_score", 0.0))
        baseline_fpr = float(r.get("false_positive_rate", GLOBAL_BASELINE_FPR))
        baseline_fnr = float(r.get("false_negative_rate", GLOBAL_BASELINE_FNR))
        fpr_gap = float(r.get("fpr_gap_vs_overall", 0.0))
        sample_count = int(r.get("sample_count", 0))

        # Proportional projection: assume ring signals affect each segment
        # proportionally to how far their FPR already deviates from the global mean.
        # A segment with FPR=0.54 (4× global 0.13) will absorb 4× the ring FPR delta.
        fpr_ratio = baseline_fpr / GLOBAL_BASELINE_FPR if GLOBAL_BASELINE_FPR > 0 else 1.0
        fnr_ratio = baseline_fnr / GLOBAL_BASELINE_FNR if GLOBAL_BASELINE_FNR > 0 else 1.0

        est_fpr_delta = RING_GLOBAL_DELTA_FPR * fpr_ratio
        est_fnr_delta = -RING_GLOBAL_DELTA_RECALL * fnr_ratio  # recall goes up → FNR goes down

        projected_fpr = min(1.0, baseline_fpr + est_fpr_delta)
        projected_fnr = max(0.0, baseline_fnr + est_fnr_delta)

        fpr_amp = fpr_ratio

        # Verdict logic — fpr_gap > 0.05 means segment is meaningfully over-flagging vs global
        concern = ""
        if severity_label == "severe" and fpr_gap > 0.05:
            # Already over-flagging — ring adds more FPR load
            if fpr_amp >= 3.0:
                verdict = "HIGH_CONCERN"
                concern = (
                    f"Segment already has FPR={baseline_fpr:.3f} ({fpr_amp:.1f}× global). "
                    f"Ring signals projected to add ~{est_fpr_delta:.3f} FPR, reaching {projected_fpr:.3f}. "
                    "Ring adjustment should be gated more conservatively for this segment."
                )
            elif fpr_amp >= 1.5:
                verdict = "MODERATE_CONCERN"
                concern = (
                    f"Segment FPR={baseline_fpr:.3f} ({fpr_amp:.1f}× global). "
                    f"Projected ring FPR addition: ~{est_fpr_delta:.3f}. Monitor post-deployment."
                )
            else:
                verdict = "LOW_CONCERN"
        elif severity_label == "severe":
            verdict = "MONITOR"
            concern = f"Severe segment (FNR-driven). Ring recall improvement ~{abs(est_fnr_delta):.3f} should help."
        else:
            verdict = "neutral"

        results.append(SegmentRingRisk(
            segment=r["segment"],
            baseline_fpr=baseline_fpr,
            baseline_fnr=baseline_fnr,
            severity_score=severity_score,
            severity_label=severity_label,
            sample_count=sample_count,
            est_ring_fpr_delta=round(est_fpr_delta, 4),
            est_ring_fnr_delta=round(est_fnr_delta, 4),
            projected_fpr=round(projected_fpr, 4),
            projected_fnr=round(projected_fnr, 4),
            fpr_amplification_ratio=round(fpr_amp, 2),
            verdict=verdict,
            concern=concern,
        ))

    results.sort(key=lambda x: (-x.fpr_amplification_ratio if x.severity_label == "severe" else 0))
    return results


def write_report(results: list[SegmentRingRisk], audit_summary: dict[str, Any]) -> dict:
    high_concern = [r for r in results if r.verdict == "HIGH_CONCERN"]
    moderate = [r for r in results if r.verdict == "MODERATE_CONCERN"]
    neutral_severe = [r for r in results if r.verdict in ("MONITOR", "LOW_CONCERN") and r.severity_label == "severe"]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ring_global_delta_recall": RING_GLOBAL_DELTA_RECALL,
        "ring_global_delta_fpr": RING_GLOBAL_DELTA_FPR,
        "global_baseline_fpr": GLOBAL_BASELINE_FPR,
        "methodology": (
            "Proportional projection: ring FPR delta per segment estimated as "
            "global_ring_fpr_delta × (segment_fpr / global_baseline_fpr). "
            "This is a conservative worst-case estimate; actual impact depends on whether "
            "ring-flagged accounts cluster in specific segments."
        ),
        "summary": {
            "total_segments_analyzed": len(results),
            "high_concern_segments": len(high_concern),
            "moderate_concern_segments": len(moderate),
            "neutral_or_beneficial": len(results) - len(high_concern) - len(moderate),
        },
        "runtime_ring_summary": audit_summary,
        "high_concern": [r.__dict__ for r in high_concern],
        "moderate_concern": [r.__dict__ for r in moderate],
        "all_severe_segments": [r.__dict__ for r in results if r.severity_label == "severe"],
        "mitigation_recommendations": _build_mitigations(high_concern, moderate),
    }
    return report


def _build_mitigations(high: list, moderate: list) -> list[str]:
    mitigations = [
        "Keep the ring-specific fairness guard enabled and review guarded-segment scope before broadening ring influence.",
        "Use the audit-derived ring-applied and ring-suppressed counts to confirm the guard is intervening where expected.",
        "Monitor ring FPR contribution post-deployment using segment breakdown in "
        "dashboard /dashboard/views — watch for FPR increases in cohort:established_users "
        "vs cohort:new_users after ring detection is live.",
        "Rebalancing training data for product_C and device_mobile segments (Phase 2 roadmap) "
        "will reduce the baseline FPR amplification ratio and make ring signals safer to apply.",
    ]
    return mitigations


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Ring Detection — Fairness Impact Analysis",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "## Background",
        "",
        "Ring detection globally improves recall by **+11.91%** but also increases FPR by **+3.30%**.",
        "This report assesses whether that FPR increase lands disproportionately on segments",
        "already identified as over-flagging (high FPR) in the fairness audit.",
        "",
        "**Methodology**: ring FPR delta per segment ≈ global_ring_delta × (segment_fpr / global_fpr).",
        "Segments with FPR already 3–5× the global baseline absorb a proportionally larger ring penalty.",
        "",
        "## Summary",
        "",
        f"- Segments analyzed: {report['summary']['total_segments_analyzed']}",
        f"- **HIGH_CONCERN**: {report['summary']['high_concern_segments']} "
        f"(already over-flagging, ring amplifies further)",
        f"- **MODERATE_CONCERN**: {report['summary']['moderate_concern_segments']}",
        f"- Neutral or beneficial: {report['summary']['neutral_or_beneficial']}",
        "",
        "## Observed Runtime Ring Counts",
        "",
        f"- Audit log available: `{report['runtime_ring_summary']['audit_log_available']}`",
        f"- Records scanned: {report['runtime_ring_summary']['records_scanned']}",
        f"- Ring applied count: {report['runtime_ring_summary']['ring_applied_count']}",
        f"- Ring suppressed count: {report['runtime_ring_summary']['ring_suppressed_count']}",
        f"- Blocked with ring-only contribution: {report['runtime_ring_summary']['blocked_with_ring_only']}",
        f"- Blocked with corroborated ring contribution: {report['runtime_ring_summary']['blocked_with_ring_corroboration']}",
        "",
        "## HIGH_CONCERN Segments",
        "",
        "| Segment | Baseline FPR | FPR amplification | Est. ring FPR add | Projected FPR |",
        "|---------|-------------|-------------------|-------------------|---------------|",
    ]
    for r in report["high_concern"]:
        lines.append(
            f"| {r['segment']} | {r['baseline_fpr']:.3f} | "
            f"{r['fpr_amplification_ratio']:.1f}x | "
            f"+{r['est_ring_fpr_delta']:.3f} | **{r['projected_fpr']:.3f}** |"
        )

    lines += [
        "",
        "## MODERATE_CONCERN Segments",
        "",
        "| Segment | Baseline FPR | FPR amplification | Est. ring FPR add | Projected FPR |",
        "|---------|-------------|-------------------|-------------------|---------------|",
    ]
    for r in report["moderate_concern"]:
        lines.append(
            f"| {r['segment']} | {r['baseline_fpr']:.3f} | "
            f"{r['fpr_amplification_ratio']:.1f}x | "
            f"+{r['est_ring_fpr_delta']:.3f} | {r['projected_fpr']:.3f} |"
        )

    lines += [
        "",
        "## All Severe Segments — Ring Impact",
        "",
        "| Segment | Severity | Baseline FPR | Baseline FNR | Est ring FPR add | Est ring FNR change | Verdict |",
        "|---------|----------|-------------|-------------|-----------------|---------------------|---------|",
    ]
    for r in report["all_severe_segments"]:
        fnr_sign = "+" if r["est_ring_fnr_delta"] > 0 else ""
        lines.append(
            f"| {r['segment']} | {r['severity_score']:.2f} | "
            f"{r['baseline_fpr']:.3f} | {r['baseline_fnr']:.3f} | "
            f"+{r['est_ring_fpr_delta']:.3f} | {fnr_sign}{r['est_ring_fnr_delta']:.3f} | "
            f"**{r['verdict']}** |"
        )

    lines += [
        "",
        "## Mitigation Recommendations",
        "",
    ]
    for i, m in enumerate(report["mitigation_recommendations"], 1):
        lines.append(f"{i}. {m}")

    lines += [
        "",
        "## Key Finding",
        "",
        "Segments `ieee:product_C`, `ieee:device_mobile`, and `ieee:device_desktop` have baseline FPR",
        "3–5× the global mean. Under a proportional projection, ring detection adds an estimated",
        "**+0.14–0.21 FPR** on top of already-severe disparities for these segments.",
        "",
        "**Current safeguard**: the runtime now applies a ring-specific fairness guard so ring-only",
        "signals do not escalate straight to `BLOCK` on guarded segments without corroboration.",
        "That keeps the ring signal available for `FLAG` / review workflows while reducing the risk",
        "of further false-positive amplification on the most vulnerable customer groups.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not FAIRNESS_METRICS_PATH.exists():
        print(f"ERROR: {FAIRNESS_METRICS_PATH} not found", file=sys.stderr)
        return 1
    if not RING_SUMMARY_PATH.exists():
        print(f"ERROR: {RING_SUMMARY_PATH} not found", file=sys.stderr)
        return 1

    metrics_rows = load_fairness_metrics(FAIRNESS_METRICS_PATH)
    results = build_segment_ring_risks(metrics_rows)
    audit_summary = load_ring_audit_summary(AUDIT_LOG_PATH)
    report = write_report(results, audit_summary)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "ring_fairness_impact.json"
    md_path = OUTPUT_DIR / "ring_fairness_impact.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    print(f"Ring fairness impact report written:")
    print(f"  {json_path}")
    print(f"  {md_path}")
    print()

    high = report["summary"]["high_concern_segments"]
    moderate = report["summary"]["moderate_concern_segments"]
    print(f"Summary: {high} HIGH_CONCERN, {moderate} MODERATE_CONCERN segments")
    if high > 0:
        print("HIGH_CONCERN segments:")
        for r in report["high_concern"]:
            print(f"  {r['segment']}: baseline FPR={r['baseline_fpr']:.3f}, "
                  f"est ring addition +{r['est_ring_fpr_delta']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Apply per-segment threshold adjustments to address fairness disparities.

Reads the fairness explainability report, derives principled threshold
adjustments for the API's segment key schema ({newness}:{amount_band}:{channel}),
and writes the result into decision_thresholds_promoted_preproc.pkl and
decision_thresholds.pkl as the `segment_thresholds` sub-key.

Fairness disparity mapping
───────────────────────────
Severe HIGH-FPR segments (too many false positives):
  ieee:product_C, ieee:product_R, ieee:product_S,
  ieee:device_mobile, ieee:device_desktop
  → Fix: raise block_threshold for new users (most at-risk for over-flagging)

Severe HIGH-FNR segments (missing too many frauds):
  ieee:product_W, ieee:product_H, ieee:identity_high_confidence
  → Fix: lower block_threshold for established users (these are the missed-fraud group)

API segment key format:  {newness}:{amount_band}:{channel}
  newness   : "new" (<30 days) | "established"
  amount_band: "high_ticket" (≥$1000) | "low_ticket"
  channel   : "APP" | "AGENT" | "QR" | "WEB"

Usage:
    python project/scripts/apply_fairness_segment_thresholds.py
    python project/scripts/apply_fairness_segment_thresholds.py --dry-run
"""
import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parents[2]
MODELS_DIR     = REPO_ROOT / "project" / "models"
FAIRNESS_JSON  = REPO_ROOT / "project" / "outputs" / "governance" / "fairness_explainability_report.json"
PROMOTED_PKL   = MODELS_DIR / "decision_thresholds_promoted_preproc.pkl"
STANDALONE_PKL = MODELS_DIR / "decision_thresholds.pkl"
OUTPUT_JSON    = REPO_ROOT / "project" / "outputs" / "governance" / "fairness_segment_thresholds_applied.json"

# Adjustment magnitude (fraction of global threshold range added or subtracted)
HIGH_FPR_BLOCK_LIFT   = +0.06   # raise block threshold for over-flagging segments
HIGH_FNR_BLOCK_DROP   = -0.07   # lower block threshold for under-flagging segments
HIGH_FPR_APPROVE_LIFT = +0.02   # also slightly raise approve for over-flagging
HIGH_FNR_APPROVE_DROP = -0.02   # slightly lower approve for under-flagging

APPROVE_MIN, APPROVE_MAX = 0.15, 0.45
BLOCK_MIN,   BLOCK_MAX   = 0.50, 0.92


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _load_severe_segments():
    d = json.loads(FAIRNESS_JSON.read_text(encoding="utf-8"))
    disparity = d.get("segment_disparity", {})
    return disparity.get("severe_segments", [])


def _load_current_thresholds():
    for pkl in [PROMOTED_PKL, STANDALONE_PKL]:
        if pkl.exists():
            return pickle.loads(pkl.read_bytes())
    return {"approve_threshold": 0.30, "block_threshold": 0.90}


def _classify_segments(severe_segments):
    """Return (high_fpr_segs, high_fnr_segs) sets of segment names."""
    high_fpr, high_fnr = set(), set()
    for entry in severe_segments:
        violations = entry.get("violations", [])
        seg = entry.get("segment", "")
        row = entry.get("row", {})
        if "fpr_gap" in violations and row.get("fpr_gap_vs_overall", 0) > 0:
            high_fpr.add(seg)
        if "fnr_gap" in violations and row.get("fnr_gap_vs_overall", 0) > 0:
            high_fnr.add(seg)
    return high_fpr, high_fnr


# Mapping from IEEE fairness segment → API segment dimension affected
# High-FPR segments → affect new user segments (new users are most over-flagged)
# High-FNR segments → affect established user segments (established users are under-detected)
_HIGH_FPR_API_NEWNESS = "new"
_HIGH_FNR_API_NEWNESS = "established"

ALL_CHANNELS      = ["APP", "WEB", "AGENT", "QR"]
ALL_AMOUNT_BANDS  = ["low_ticket", "high_ticket"]
ALL_NEWNESS       = ["new", "established"]


def _build_segment_thresholds(base_approve, base_block, high_fpr_active, high_fnr_active):
    """
    Returns dict segment_key → {approve_threshold, block_threshold} for all
    combinations that need adjustment.
    """
    segments = {}

    # For every segment combination, compute adjustment
    for newness in ALL_NEWNESS:
        for amount_band in ALL_AMOUNT_BANDS:
            for channel in ALL_CHANNELS:
                key = f"{newness}:{amount_band}:{channel}"

                approve_adj = 0.0
                block_adj   = 0.0

                # High-FPR segments → raise thresholds for new users
                if high_fpr_active and newness == _HIGH_FPR_API_NEWNESS:
                    approve_adj += HIGH_FPR_APPROVE_LIFT
                    block_adj   += HIGH_FPR_BLOCK_LIFT
                    # Extra lift for AGENT channel (inherently higher context noise)
                    if channel == "AGENT":
                        block_adj += 0.03

                # High-FNR segments → lower thresholds for established users
                if high_fnr_active and newness == _HIGH_FNR_API_NEWNESS:
                    approve_adj += HIGH_FNR_APPROVE_DROP
                    block_adj   += HIGH_FNR_BLOCK_DROP
                    # High-ticket established users are the identity_high_confidence
                    # group → extra sensitivity
                    if amount_band == "high_ticket":
                        block_adj -= 0.03

                # Only write segment if there's an actual adjustment
                if approve_adj != 0.0 or block_adj != 0.0:
                    new_approve = _clamp(base_approve + approve_adj, APPROVE_MIN, APPROVE_MAX)
                    new_block   = _clamp(base_block   + block_adj,   BLOCK_MIN,   BLOCK_MAX)
                    if new_approve < new_block:
                        segments[key] = {
                            "approve_threshold": round(new_approve, 4),
                            "block_threshold":   round(new_block, 4),
                            "calibration_metrics": {},
                        }

    return segments


def parse_args():
    p = argparse.ArgumentParser(description="Apply fairness-driven segment thresholds")
    p.add_argument("--dry-run",     action="store_true")
    p.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    return p.parse_args()


def main():
    args = parse_args()

    if not FAIRNESS_JSON.exists():
        print("Fairness report not found — skipping.")
        return

    severe_segments = _load_severe_segments()
    high_fpr, high_fnr = _classify_segments(severe_segments)
    current = _load_current_thresholds()
    base_approve = float(current.get("approve_threshold", 0.30))
    base_block   = float(current.get("block_threshold",   0.90))

    high_fpr_active = len(high_fpr) > 0
    high_fnr_active = len(high_fnr) > 0
    segment_thresholds = _build_segment_thresholds(
        base_approve, base_block, high_fpr_active, high_fnr_active
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_approve_threshold": base_approve,
        "base_block_threshold": base_block,
        "high_fpr_severe_segments": sorted(high_fpr),
        "high_fnr_severe_segments": sorted(high_fnr),
        "adjustment_rationale": {
            "high_fpr": (
                f"Segments with excess FPR (product_C/R/S, device_mobile/desktop) indicate "
                f"the model over-flags new users. Block threshold raised by "
                f"{HIGH_FPR_BLOCK_LIFT:+.2f} for new:* segments."
            ),
            "high_fnr": (
                f"Segments with excess FNR (product_W/H, identity_high_confidence) indicate "
                f"the model under-detects fraud for established users. Block threshold "
                f"lowered by {HIGH_FNR_BLOCK_DROP:+.2f} for established:* segments."
            ),
        },
        "segment_thresholds_written": segment_thresholds,
        "dry_run": args.dry_run,
    }

    if not args.dry_run and segment_thresholds:
        for pkl_path in [PROMOTED_PKL, STANDALONE_PKL]:
            if pkl_path.exists():
                payload = pickle.loads(pkl_path.read_bytes())
            else:
                payload = {"approve_threshold": base_approve, "block_threshold": base_block}
            payload["segment_thresholds"] = segment_thresholds
            pkl_path.write_bytes(pickle.dumps(payload))

        print(f"Wrote {len(segment_thresholds)} segment threshold overrides to PKL files.")
    else:
        tag = "DRY RUN" if args.dry_run else "NO SEGMENTS"
        print(f"[{tag}] Would write {len(segment_thresholds)} segment overrides.")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print a readable summary
    print(f"\nSevere HIGH-FPR segments ({len(high_fpr)}): {sorted(high_fpr)}")
    print(f"Severe HIGH-FNR segments ({len(high_fnr)}): {sorted(high_fnr)}")
    print(f"\nSegment threshold overrides ({len(segment_thresholds)}):")
    for k, v in sorted(segment_thresholds.items()):
        delta_block = round(v['block_threshold'] - base_block, 4)
        print(f"  {k}: approve={v['approve_threshold']} block={v['block_threshold']} "
              f"(block delta={delta_block:+.4f})")


if __name__ == "__main__":
    main()

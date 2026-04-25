"""
Drift-triggered threshold recalibration (no dataset required).

Reads the latest drift report from the ops run archive, detects whether
the decision distribution has shifted away from baseline, and adjusts
approve_threshold / block_threshold to counteract the drift.

The approach is a linear correction:
    excess_block_rate = current_block_rate - baseline_block_rate
    new_block_threshold = current_block_threshold
                          + BLOCK_SCALE * excess_block_rate

    excess_flag_rate  = current_flag_rate - baseline_flag_rate
    new_approve_threshold = current_approve_threshold
                            - APPROVE_SCALE * excess_flag_rate

Guardrails:
  - new_approve_threshold clamped to [0.15, 0.45]
  - new_block_threshold   clamped to [0.55, 0.92]
  - approve < block enforced

Usage:
    python project/scripts/recalibrate_thresholds_from_drift.py
    python project/scripts/recalibrate_thresholds_from_drift.py --dry-run
"""
import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODELS_DIR     = REPO_ROOT / "project" / "models"
OPS_RUNS_DIR   = REPO_ROOT / "project" / "outputs" / "ops_runs"
OUTPUT_JSON    = REPO_ROOT / "project" / "outputs" / "calibration" / "drift_recalibration_report.json"
PROMOTED_PKL   = MODELS_DIR / "decision_thresholds_promoted_preproc.pkl"
STANDALONE_PKL = MODELS_DIR / "decision_thresholds.pkl"

BLOCK_SCALE   = 2.0   # how aggressively to raise block_threshold per unit excess block rate
APPROVE_SCALE = 0.5   # how aggressively to lower approve_threshold per unit excess flag rate
APPROVE_MIN, APPROVE_MAX = 0.15, 0.45
BLOCK_MIN,   BLOCK_MAX   = 0.55, 0.92

DRIFT_ALERT_THRESHOLD  = 0.15  # minimum decision drift score to trigger recalibration
BLOCK_EXCESS_THRESHOLD = 0.05  # minimum excess block rate to trigger block adjustment


def _load_latest_drift_report() -> dict | None:
    if not OPS_RUNS_DIR.exists():
        return None
    runs = sorted(OPS_RUNS_DIR.iterdir())
    for run in reversed(runs):
        p = run / "drift_report.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    # fallback to monitoring dir
    p = REPO_ROOT / "project" / "outputs" / "monitoring" / "drift_report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _load_current_thresholds() -> dict:
    for pkl in [PROMOTED_PKL, STANDALONE_PKL]:
        if pkl.exists():
            return pickle.loads(pkl.read_bytes())
    return {"approve_threshold": 0.30, "block_threshold": 0.90}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drift-corrected threshold recalibration")
    p.add_argument("--dry-run", action="store_true", help="Compute but do not write PKL files")
    p.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    p.add_argument("--block-scale",   type=float, default=BLOCK_SCALE)
    p.add_argument("--approve-scale", type=float, default=APPROVE_SCALE)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    drift_report = _load_latest_drift_report()
    if drift_report is None:
        print("No drift report found — nothing to recalibrate.")
        sys.exit(0)

    decision_drift = drift_report.get("decision_drift", {})
    drift_score    = float(decision_drift.get("score", 0.0))
    drift_status   = decision_drift.get("status", "ok")
    baseline_dist  = decision_drift.get("baseline_distribution", {})
    current_dist   = decision_drift.get("current_distribution", {})

    current_thresholds = _load_current_thresholds()
    current_approve = float(current_thresholds.get("approve_threshold", 0.30))
    current_block   = float(current_thresholds.get("block_threshold", 0.90))

    baseline_block = float(baseline_dist.get("BLOCK", 0.066))
    current_block_rate = float(current_dist.get("BLOCK", current_block_rate := 0.0))
    baseline_flag  = float(baseline_dist.get("FLAG", 0.069))
    current_flag_rate  = float(current_dist.get("FLAG", 0.0))

    excess_block = current_block_rate - baseline_block
    excess_flag  = current_flag_rate  - baseline_flag

    # Decision: recalibrate only when drift is material
    should_recalibrate = (
        drift_status in ("alert", "warn")
        and drift_score >= DRIFT_ALERT_THRESHOLD
        and excess_block >= BLOCK_EXCESS_THRESHOLD
    )

    if not should_recalibrate:
        new_approve = current_approve
        new_block   = current_block
        action      = "no_change"
        reason      = (
            f"drift_score={drift_score:.4f} or excess_block={excess_block:.4f} "
            f"below recalibration thresholds"
        )
    else:
        raw_new_block   = current_block   + args.block_scale   * excess_block
        raw_new_approve = current_approve - args.approve_scale * max(0.0, excess_flag)
        new_block   = _clamp(raw_new_block,   BLOCK_MIN,   BLOCK_MAX)
        new_approve = _clamp(raw_new_approve, APPROVE_MIN, APPROVE_MAX)
        # Ensure approve < block
        if new_approve >= new_block:
            new_approve = max(APPROVE_MIN, new_block - 0.10)
        action = "recalibrated"
        reason = (
            f"drift_status={drift_status}, drift_score={drift_score:.4f}, "
            f"excess_block_rate={excess_block:.4f}, excess_flag_rate={excess_flag:.4f}"
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "reason": reason,
        "drift_score": drift_score,
        "drift_status": drift_status,
        "baseline_distribution": baseline_dist,
        "current_distribution": current_dist,
        "excess_block_rate": round(excess_block, 4),
        "excess_flag_rate":  round(excess_flag, 4),
        "thresholds_before": {"approve_threshold": current_approve, "block_threshold": current_block},
        "thresholds_after":  {"approve_threshold": round(new_approve, 4), "block_threshold": round(new_block, 4)},
        "dry_run": args.dry_run,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.dry_run and action == "recalibrated":
        payload = {"approve_threshold": new_approve, "block_threshold": new_block}
        PROMOTED_PKL.write_bytes(pickle.dumps(payload))
        STANDALONE_PKL.write_bytes(pickle.dumps(payload))
        print(f"Thresholds updated: approve={new_approve:.4f}  block={new_block:.4f}")
    else:
        print(f"[{'DRY RUN' if args.dry_run else 'NO CHANGE'}] "
              f"approve={new_approve:.4f}  block={new_block:.4f}")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

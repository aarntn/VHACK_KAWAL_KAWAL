# Fairness Demo Note

Use this note when talking about fairness so the claim stays grounded in artifacts instead of UI estimates.

## What we can defend

- Fairness analysis exists in:
  - `project/outputs/governance/fairness_explainability_report.md`
  - `project/outputs/governance/fairness_segment_metrics.csv`
  - `project/outputs/governance/fairness_segment_thresholds_applied.json`
- If you generate a local mitigation memo with `project/scripts/fairness_segment_decision.py`, it may also emit `project/outputs/governance/fairness_action_plan.md`, but that file is not assumed to be bundled in the repo by default.
- Segment-aware thresholds were applied on `2026-04-19`.
- Residual severe segments still exist, so the fairness story is “mitigation in place, not fully solved.”
- ASEAN-specific handoff framing lives in `docs/asean_fairness_summary.md`.
- Ring-specific safeguards are now in the runtime too: ring-only evidence is prevented from escalating straight to `BLOCK` on guarded segments unless another signal corroborates it.

## How to explain it

Use this structure:

1. We audited disparity by segment.
2. We applied threshold adjustments for severe cohorts as a mitigation.
3. We preserved an artifact trail showing what changed.
4. We still treat some segments as requiring follow-up retraining and policy review.

## Good phrasing

> We already have a fairness audit and mitigation loop in the delivery pipeline. For the demo we present fairness from governance artifacts, not from live dashboard approximations.

For ring fairness phrasing:

> We found that ring signals can amplify false positives on some cohorts, so we added a ring-specific fairness guard and we report ring-applied versus ring-suppressed outcomes from the audit log.

For ASEAN phrasing:

> We now carry country, corridor, channel, and agent-assisted metadata through the runtime and audit path. The next fairness step is regenerating metrics on those ASEAN-aware cohorts from production-like traffic.

## Avoid

- Avoid quoting fairness numbers from frontend-derived cohort tables.
- Avoid saying fairness is “fully resolved.”
- Avoid implying the live dashboard is the canonical fairness source.
- Avoid claiming ring detection is unbiased by default; say that it is now bounded by evidence gates plus a fairness guard.

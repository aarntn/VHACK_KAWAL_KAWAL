# ASEAN Fairness Summary

This summary keeps the ASEAN fairness story conservative and artifact-backed.

## Current measured governance baseline

Source artifacts:

- `project/outputs/governance/fairness_explainability_report.md`
- `project/outputs/governance/fairness_segment_thresholds_applied.json`

Measured overall block-decision metrics:

- Precision: `0.1294`
- Recall: `0.5112`
- FPR: `0.1276`

## Severe segments already identified

The current fairness report still shows severe disparity in IEEE-derived cohorts, especially:

- `ieee:product_C`
- `ieee:device_mobile`
- `ieee:device_desktop`
- `ieee:product_W`
- `ieee:product_R`

These are the strongest evidence-backed gaps today. We should present them honestly as known monitored cohorts rather than claiming the disparity problem is already solved.

## Live mitigation that is already active

Threshold mitigation has already been written and applied through `project/outputs/governance/fairness_segment_thresholds_applied.json`:

- `new:*` segments: block threshold raised to roughly `0.805`, with `AGENT` variants up to `0.835`
- `established:*` segments: block threshold lowered to roughly `0.675`, with high-ticket segments down to `0.645`

This is the current live governance control we can defend in Q&A.

## ASEAN-specific fairness framing for the demo

For the ASEAN demo, the right framing is:

- We already support country/corridor/channel metadata in runtime scoring and audit provenance.
- We do **not** yet claim a fully retrained ASEAN fairness pack.
- The next fairness evidence layer is region-aware segmentation across:
  - country / corridor
  - channel
  - new user vs established user
  - agent-assisted vs self-serve

## Safe line to use

> We already have live threshold governance and audit-safe provenance. ASEAN-specific fairness segmentation is now wired into the request and audit path, while the next step is to regenerate fairness metrics on those region-aware cohorts from production-like data.

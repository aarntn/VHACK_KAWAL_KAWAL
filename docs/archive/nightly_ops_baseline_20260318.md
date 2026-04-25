# Nightly Ops Baseline Capture — 2026-03-18

## Context calibration inspection (`project/outputs/figures/tables/context_calibration.json`)

- `policy_checks.fpr_le_target.pass=true` (`actual=0.00014068852966417648`, `target=0.005`)
- `policy_checks.precision_ge_target.pass=true` (`actual=0.9137931034482759`, `target=0.85`)
- `policy_checks.overall_pass=true`

Runtime recommendation snapshot:

- `approve_threshold=0.21663971423879455`
- `block_threshold=0.8056064754152743`
- `context_adjustment_max=0.15474176635186124`
- `context_weights` present for the full 17-feature promoted context set.

Interpretation: calibration policy is already in strict-pass state (`overall_pass=true`), so no strict-policy relaxation is required for this artifact.

## Diagnostic rerun attempt with policy-promotion bypass

Per diagnostic-run guidance, nightly ops was re-run with threshold promotion bypassed:

```bash
python -m project.scripts.nightly_ops \
  --dataset-source ieee_cis \
  --ieee-transaction-path ieee-fraud-detection/train_transaction.csv \
  --ieee-identity-path ieee-fraud-detection/train_identity.csv \
  --model-path project/models/final_xgboost_model_promoted_preproc.pkl \
  --feature-path project/models/feature_columns_promoted_preproc.pkl \
  --preprocessing-artifact-path project/models/preprocessing_artifact_promoted.pkl \
  --thresholds-output project/models/decision_thresholds_promoted_preproc.pkl \
  --audit-log project/outputs/audit/fraud_audit_log.jsonl \
  --run-benchmark \
  --run-cohort-kpi \
  --run-profile-health \
  --skip-threshold-promotion \
  --ops-summary-json project/outputs/monitoring/nightly_ops_summary.json
```

Result: run failed early because IEEE source files are not present in this environment:

- Missing: `ieee-fraud-detection/train_transaction.csv`
- Missing: `ieee-fraud-detection/train_identity.csv`

## Release readiness status

- Current summary remains non-green (`status=failed_calibration` in `project/outputs/monitoring/nightly_ops_summary.json`).
- Threshold promotion was intentionally bypassed for diagnostic continuation (`--skip-threshold-promotion`) but full pipeline completion is still blocked by missing dataset inputs.
- **Production-ready is NOT marked** while promotion/gating pathways are blocked or incomplete.

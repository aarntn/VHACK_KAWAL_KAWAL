#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python project/scripts/preset_contract_check.py \
  --output-json project/outputs/monitoring/demo_readiness_summary.json

python project/scripts/nightly_ops.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path ieee-fraud-detection/train_transaction.csv \
  --ieee-identity-path ieee-fraud-detection/train_identity.csv \
  --model-path project/models/final_xgboost_model_promoted_preproc.pkl \
  --feature-path project/models/feature_columns_promoted_preproc.pkl \
  --preprocessing-artifact-path project/models/preprocessing_artifact_promoted.pkl \
  --thresholds-output project/models/decision_thresholds_promoted_preproc.pkl \
  --audit-log project/outputs/audit/fraud_audit_log.jsonl \
  --run-benchmark \
  --benchmark-sla-mode warn \
  --run-cohort-kpi \
  --run-profile-replay \
  --run-profile-health \
  --ops-summary-json project/outputs/monitoring/nightly_ops_summary.json

python project/scripts/release_gate_check.py \
  --ops-summary-json project/outputs/monitoring/nightly_ops_summary.json

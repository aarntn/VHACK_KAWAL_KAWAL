$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "../..")
Set-Location $RepoRoot

python project/scripts/preset_contract_check.py `
  --output-json project/outputs/monitoring/demo_readiness_summary.json

python project/scripts/nightly_ops.py `
  --dataset-source ieee_cis `
  --ieee-transaction-path ieee-fraud-detection/train_transaction.csv `
  --ieee-identity-path ieee-fraud-detection/train_identity.csv `
  --audit-log project/outputs/audit/fraud_audit_log.jsonl `
  --run-benchmark `
  --benchmark-sla-mode warn `
  --run-cohort-kpi `
  --run-profile-replay `
  --run-profile-health `
  --ops-summary-json project/outputs/monitoring/nightly_ops_summary.json

python project/scripts/release_gate_check.py `
  --ops-summary-json project/outputs/monitoring/nightly_ops_summary.json

# Automation Defaults (Nightly Ops + Threshold Governance)

This document defines the default automation path for operations checks and threshold governance.

## 1) Nightly scheduler options

Use one scheduler (or more) to run nightly automation once per day:

- Linux/macOS cron: `project/scripts/run_nightly_ops.sh`
- Windows Task Scheduler: `project/scripts/run_nightly_ops.ps1`
- GitHub Actions schedule: `.github/workflows/nightly-ops.yml`

The scheduled job runs:

1. Preset contract check across fraud + wallet paths (`preset_contract_check.py`)
2. Drift monitor
3. Calibration (when drift recommends recalibration)
4. Threshold promotion only when policy checks pass
5. Release gate check (`release_gate_check.py`)

Interim automation default: run nightly with `--benchmark-sla-mode warn` so benchmark SLA misses are recorded while still producing full benchmark evidence artifacts on every run.

## 2) Threshold promotion rules

`project/scripts/promote_thresholds.py` enforces:

- `policy_checks.overall_pass == true` in calibration JSON
- Threshold sanity (`0 <= approve < block <= 1`)
- Backup of previous active threshold file before replacement
- Promotion record output for auditability

Artifacts:

- Active thresholds: `project/models/decision_thresholds.pkl`
- Rollback backup(s): `project/outputs/threshold_governance/decision_thresholds.backup.<timestamp>.pkl`
- Promotion log: `project/outputs/threshold_governance/latest_promotion_record.json`

## 3) Release gating rule

`project/scripts/release_gate_check.py` returns non-zero exit code unless the backend
execution gate is fully satisfied.

Required checks:

1. **Consecutive startup proof**: `N` consecutive runs with successful nightly status,
   artifact validation pass, and no benchmark preflight startup failure.
2. **Artifact compatibility**: `artifact_validation_report.ok == true`.
3. **Payload/API contract checks**: `payload_contract_tests.ok == true` (or benchmark
   contract validation fallback in legacy payloads).
4. **Benchmark SLA + error-rate policy**:
   - `benchmark_sla_mode` must be `enforce` for production-ready gate decisions.
   - `benchmark_sla.ok == true`, or an explicit warn-mode exception approval is provided for non-production interim reviews.
   - Endpoint error rates remain under policy (`--max-endpoint-error-rate-pct`).
   - Any `unknown_internal_guardrail` breach fails the gate.
5. **Nightly retraining trigger policy review**:
   - Drift streak
   - SLA-fail streak
   - Endpoint-error streak
   - If retraining is recommended, an explicit review ack is required.
6. **Inference regression gate**:
   - Inference candidate report must not be blocked/hold when present.

On every run, the script writes a one-page backend readiness report:

- `project/outputs/monitoring/backend_readiness_report.md`
- `project/outputs/monitoring/demo_readiness_summary.json`

When the gate passes, backend API contracts are considered frozen and frontend
integration is allowed only against stable endpoints listed by the gate output.

Example:

```bash
python project/scripts/release_gate_check.py \
  --ops-summary-json project/outputs/monitoring/nightly_ops_summary.json \
  --required-startup-streak 3 \
  --max-endpoint-error-rate-pct 5.0
```

This is intended to be used as a deployment precondition.

## 4) Canary rollout guardrails + automated rollback

Use a small canary traffic slice first (recommended: `5%`) and evaluate endpoint
error categories + p95 latency in rolling windows.

Automation script: `project/scripts/canary_rollout_guard.py`

Example:

```bash
python -m project.scripts.canary_rollout_guard \
  --telemetry-json project/outputs/monitoring/canary_windows.json \
  --output-json project/outputs/monitoring/canary_rollout_decision.json \
  --archive-dir project/outputs/rollout_telemetry \
  --artifact-id fraud-model \
  --artifact-version 2026.03.18.1 \
  --release-id rel-20260318-01 \
  --commit-sha <git_sha> \
  --canary-traffic-percent 5 \
  --max-error-rate-pct 1.0 \
  --max-p95-latency-ms 250 \
  --max-unknown-error-pct 0.1 \
  --rollback-consecutive-windows 3
```

Behavior:

- Monitors endpoint-level `error_rate_pct`, `p95_latency_ms`, and `unknown_internal` error share.
- Triggers `decision=rollback` when thresholds are exceeded for consecutive windows.
- Writes release decision payload + archives raw telemetry with artifact/version identifiers.

## 5) Incident runbook for taxonomy/stage triage

Use the rollout triage flow in `docs/rollout-incident-runbook.md` to investigate:

- Error-category regressions (timeout/upstream/validation/unknown internal)
- Stage-level latency regressions (feature extraction/model inference/context/behavior/profile-store)
- Canary rollback and escalation paths

## 6) What to run after model/ops changes

You do **not** need to run the full nightly flow for every small code change.
Use a tiered workflow:

1. Fast local unit tests (for the files you touched).
2. Full nightly ops run before a demo/release candidate.
3. Release gate check after nightly ops summary is generated.

### Full nightly command (IEEE-CIS)

```bash
python -m project.scripts.nightly_ops \
  --dataset-source ieee_cis \
  --ieee-transaction-path "$TX" \
  --ieee-identity-path "$ID" \
  --model-path project/models/final_xgboost_model_promoted_preproc.pkl \
  --feature-path project/models/feature_columns_promoted_preproc.pkl \
  --preprocessing-artifact-path project/models/preprocessing_artifact_promoted.pkl \
  --thresholds-output project/models/decision_thresholds_promoted_preproc.pkl \
  --run-benchmark \
  --benchmark-sla-mode warn \
  --run-cohort-kpi \
  --run-profile-replay \
  --run-profile-health
```

### Then run the gate

```bash
python -m project.scripts.release_gate_check \
  --ops-summary-json project/outputs/monitoring/nightly_ops_summary.json
```

### Final submission / demo hardening mode

Before final submission or demo hardening, rerun nightly with strict SLA enforcement and require the release gate to pass:

```bash
python -m project.scripts.nightly_ops \
  --dataset-source ieee_cis \
  --ieee-transaction-path "$TX" \
  --ieee-identity-path "$ID" \
  --run-benchmark \
  --benchmark-sla-mode enforce \
  --run-cohort-kpi \
  --run-profile-replay \
  --run-profile-health

python -m project.scripts.release_gate_check \
  --ops-summary-json project/outputs/monitoring/nightly_ops_summary.json
```

### One-liner wrapper (recommended for consistency)

```bash
bash project/scripts/run_nightly_ops.sh
```

## 7) Release-candidate evidence policy

Every release candidate must produce a complete evidence pack.

Required command:

```bash
python project/scripts/build_evidence_bundle.py \
  --bundle-name evidence_bundle_<release_tag> \
  --release-tag <release_tag> \
  --require-complete
```

Pass condition: `manifest.json` has `missing_artifacts: []`.

A tagged archive is written to `project/outputs/release_artifacts/evidence_bundle_<release_tag>.tar.gz`.

Workflow support: `.github/workflows/release-candidate-evidence.yml`

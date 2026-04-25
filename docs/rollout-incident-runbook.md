# Rollout Incident Runbook (Canary + Stage Metrics)

This runbook defines incident triage for canary rollouts using the current error
taxonomy and stage-level latency metrics.

## 1) Preconditions before canary

Do not start canary unless all pre-deploy gates are green:

1. Consecutive startup gate passes (`required_startup_streak` satisfied with artifact validation passing).
2. Artifact parity check passes (`artifact_validation_report.ok=true`).
3. Payload contract tests pass (`payload_contract_tests.ok=true`).
4. Benchmark thresholds pass (`benchmark_sla.ok=true`) **or** warn-mode exception is explicitly approved.
5. Endpoint error rates are below policy thresholds (`max_endpoint_error_rate_pct`).
6. Retraining trigger policy review is complete (drift streak, SLA-fail streak, endpoint-error streak).
7. Artifact integrity checks pass (model checksum manifest + audit signing configuration sanity).

Gate command:

```bash
python -m project.scripts.release_gate_check \
  --ops-summary-json project/outputs/monitoring/nightly_ops_summary.json \
  --required-startup-streak 3 \
  --max-endpoint-error-rate-pct 5.0
```

Integrity command (must pass before canary):

```bash
bash project/scripts/verify_model_artifact_checksums.sh
```

Readiness evidence report output:

- `project/outputs/monitoring/backend_readiness_report.md`

## 2) Canary setup

- Start with a small traffic slice (default `5%`).
- Window duration recommendation: `5 minutes`.
- Minimum observation horizon: `6 windows` before promotion.

Monitor endpoint metrics per window:

- `error_rate_pct`
- `p95_latency_ms`
- Error category distribution:
  - `timeout_upstream`
  - `validation_error`
  - `dependency_failure`
  - `unknown_internal`

## 3) Rollout policy / rollback rule

Run evaluator:

```bash
python -m project.scripts.canary_rollout_guard \
  --telemetry-json project/outputs/monitoring/canary_windows.json \
  --output-json project/outputs/monitoring/canary_rollout_decision.json \
  --archive-dir project/outputs/rollout_telemetry \
  --artifact-id fraud-model \
  --artifact-version <version> \
  --release-id <release_id> \
  --commit-sha <git_sha> \
  --canary-traffic-percent 5 \
  --max-error-rate-pct 1.0 \
  --max-p95-latency-ms 250 \
  --max-unknown-error-pct 0.1 \
  --rollback-consecutive-windows 3
```

Decision outcomes:

- `promote`: thresholds stayed inside limits (or breaches were not consecutive).
- `rollback`: thresholds exceeded for the configured consecutive-window streak.

## 4) Incident triage flow

1. **Check rollback decision payload**
   - Open `project/outputs/monitoring/canary_rollout_decision.json`.
   - Identify breached endpoints, metrics, and streak length.
2. **Classify error category spike**
   - If `timeout_upstream` increases: inspect wallet->fraud connectivity and downstream saturation.
   - If `validation_error` increases: compare payload contract release diff and client deploy timeline.
   - If `dependency_failure` increases: inspect external service health and retries/circuit breaker.
   - If `unknown_internal` increases: treat as SEV-1 candidate, freeze promotion, escalate to on-call.
3. **Check stage latency diagnostics**
   - Review `project/outputs/monitoring/latency_stage_analysis.json`.
   - Locate dominant stage by `p95_ms`.
   - Compare stage deltas against prior stable run.
4. **Run integrity validation**
   - Execute checksum verification (`bash project/scripts/verify_model_artifact_checksums.sh`).
   - Verify `/health/artifacts` for manifest/check consistency (`feature_schema_hash`, `checks`).
   - If audit log tampering is suspected, verify hash-chain continuity (`previous_record_signature` → `record_signature`).
5. **Mitigate**
   - Roll back canary when rollback condition is met.
   - If non-consecutive breaches, hold at canary slice and observe additional windows.
   - If taxonomy indicates schema mismatch, roll back client contract change first.
6. **Escalate**
   - SEV-1: `unknown_internal` breach or hard availability failure.
   - SEV-1: confirmed artifact/signature tampering.
   - SEV-2: p95 regression above threshold with low error rate.
   - SEV-2: checksum/signature-chain failure without confirmed active tampering.
   - SEV-3: transient non-consecutive threshold breach.

## 5) Telemetry retention / auditability

Archive location:

- `project/outputs/rollout_telemetry/<release_id>_<timestamp>.json`

Each archived bundle must contain:

- Release identifiers: `release_id`, `artifact_id`, `artifact_version`, `commit_sha`
- Canary settings: traffic percent and thresholds
- Raw rollout telemetry windows used in decisioning
- Final decision payload (promote/rollback + reason)

Retention recommendation:

- Keep telemetry bundles for at least 90 days for audit traceability.

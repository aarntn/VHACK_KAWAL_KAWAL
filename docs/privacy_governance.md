# Privacy Governance for Audit, Analyst Review, and Retraining Curation

This document defines what the fraud service stores under `project/outputs/audit/*.jsonl`, how privacy is preserved, retention/deletion expectations, and access controls for analyst workflows.

Primary implementation references:
- Runtime audit/review endpoints and record builders: `project/app/hybrid_fraud_api.py`
- Incident playbook: `docs/rollout-incident-runbook.md`

## 1) Audit and analyst JSONL data contract

### 1.1 `fraud_audit_log.jsonl`

Written by `build_audit_record(...)` + `write_audit_log(...)`.

**Stored fields (per line):**
- `request_id`, `timestamp_utc`, `api_version`, `model_version`
- `hashed_user_id`, `hash_key_version`
- `decision`, `decision_source`, `user_segment`, `applied_thresholds`, `hard_rule_hits`
- `verification_action`, `verification_reason`
- `base_model_score`, `context_adjustment`, `behavior_adjustment`, `final_risk_score`
- `latency_ms`, `reason_codes`, `context_scores`
- `feature_integrity` metadata (`feature_count`, backend/runtime, and privacy assertions)
- `signature_key_version`, `previous_record_signature`, `record_signature`

**Hashed/tokenized treatment:**
- `user_id` is never persisted; it is HMAC-SHA256 hashed into `hashed_user_id` using `FRAUD_HASH_SECRETS` and `FRAUD_HASH_KEY_VERSION`.
- `request_id` is a generated UUID correlation handle (pseudonymous operational token), not a bank/account identifier.
- No additional tokenization layer is currently implemented for audit fields.

### 1.2 `review_queue.jsonl`

Written by `queue_review_case(...)`; updated by `submit_review_outcome(...)`.

**Stored fields (per line):**
- `request_id`, `correlation_id`, `created_at_utc`, `status`
- `decision`, `final_risk_score`, `reasons`, `context_summary`
- On resolution: `resolved_at_utc`

**Hashed/tokenized treatment:**
- No raw `user_id` stored.
- Uses `request_id`/`correlation_id` indirection only.

### 1.3 `analyst_outcomes.jsonl`

Written by `submit_review_outcome(...)`.

**Stored fields (per line):**
- `request_id`, `reviewed_at_utc`
- `analyst_id`, `analyst_decision`, `analyst_confidence`, `analyst_notes`
- `transaction_amount`, `model_decision`, `model_final_risk_score`

**Hashed/tokenized treatment:**
- Analyst identity is stored as `analyst_id` for accountability/auditability.
- No customer direct identifier is written.

### 1.4 `retraining_curation.jsonl`

Written by `curate_retraining_example(...)` + `append_jsonl(...)`.

**Stored fields (per line):**
- `request_id`, `curated_at_utc`
- `label`, `analyst_decision`, `analyst_confidence`, `analyst_notes`
- `model_decision`, `model_final_risk_score`, `transaction_amount`
- `reason_codes`, `context_scores`, `hashed_user_id`

**Hashed/tokenized treatment:**
- User linkage remains via `hashed_user_id` only.
- Raw account/payment identifiers are excluded.

## 2) Retention windows and deletion workflow

The service exposes baseline retention settings via config:
- `FRAUD_AUDIT_RETENTION_DAYS` (default `365`)
- `FRAUD_AUDIT_DELETION_SLA_DAYS` (default `30`)

### 2.1 Recommended operating windows

- **Hot storage (`0-90 days`)**
  - Keep JSONL audit/review files in active storage for investigation, on-call, and model QA loops.
  - Includes `fraud_audit_log.jsonl`, `review_queue.jsonl`, `analyst_outcomes.jsonl`, `retraining_curation.jsonl`.
- **Archive storage (`91-365 days`)**
  - Move immutable snapshots/bundles to restricted archive for governance/compliance evidence.
  - Archive should preserve integrity metadata (checksums/signatures) and access logs.
- **Beyond retention (`>365 days` by default)**
  - Delete from hot and archive systems; complete backup purge within deletion SLA (`<=30 days`).

### 2.2 Deletion workflow

1. **Identify expired records** by timestamp (`timestamp_utc`, `created_at_utc`, `reviewed_at_utc`, `curated_at_utc`).
2. **Export compliance report** (counts, date boundaries, operator/service principal, run time).
3. **Delete from hot JSONL stores** and any query indices/materialized views.
4. **Delete from archives/backups** before `FRAUD_AUDIT_DELETION_SLA_DAYS` deadline.
5. **Record deletion evidence** in release/ops artifacts (run id, commit SHA, deletion counts).
6. **Escalate SLA breaches** through the incident runbook as an operational compliance incident.

## 3) Role-based access expectations for analyst endpoints

Endpoints in `project/app/hybrid_fraud_api.py`:
- `GET /review_queue`
- `POST /review_queue/{request_id}/outcome`
- `GET /retraining/curation`
- `GET /dashboard/views` (aggregate metrics)

The current repo implementation does not enforce authz at route level. Deployment **must** enforce role controls at API gateway/service mesh/identity proxy.

### 3.1 Expected RBAC matrix

- **Fraud Analyst (read/write review)**
  - Allow: `GET /review_queue`, `POST /review_queue/{request_id}/outcome`
  - Deny: platform config, raw model artifacts, broad archive deletion operations
- **ML Engineer / Model Risk (read curation)**
  - Allow: `GET /retraining/curation`, `GET /dashboard/views`
  - Conditional access to outcome notes only when required for label QA
- **SRE / Security Operations (operational oversight)**
  - Allow: read-only access to queues/curation for incident triage
  - Allow: integrity verification tooling and retention job status
- **Auditor / Compliance (read-only)**
  - Allow: immutable evidence bundles, retention/deletion reports, integrity attestations
  - Deny: write access to analyst outcomes and retraining labels

### 3.2 Additional control expectations

- Enforce least-privilege scopes and short-lived credentials.
- Emit access logs for every analyst endpoint call (`who`, `what`, `when`, `request_id`).
- Apply dual-control approvals for bulk export/delete operations.

## 4) Integrity controls and incident runbook mapping

Integrity controls already present in runtime and scripts:

1. **Hash-chained audit signatures**
   - Every audit record includes `previous_record_signature` + `record_signature`.
   - Signature uses HMAC-SHA256 with versioned keys (`FRAUD_AUDIT_SIGNING_SECRETS`, `FRAUD_AUDIT_SIGNING_KEY_VERSION`).
2. **Versioned hashing secret management**
   - `hashed_user_id` includes key version (`hash_key_version`) for key rotation traceability.
3. **Model artifact checksum verification**
   - `project/scripts/verify_model_artifact_checksums.sh` verifies `project/models/artifact_checksums.sha256`.
4. **Artifact validation surfaces at runtime**
   - `/health/artifacts` reports manifest/check status (`feature_schema_hash`, checks).

### 4.1 Incident runbook linkage

Use `docs/rollout-incident-runbook.md` together with these integrity checks:

- **Pre-canary gate:** run checksum verification and artifact health checks before rollout.
- **During incident triage:** if errors suggest schema/artifact drift, verify checksums and artifact manifest first.
- **SEV escalation trigger:** any failed checksum/signature-chain verification or unexplained signature discontinuity is treated as at least SEV-2, and SEV-1 if active tampering is suspected.
- **Post-incident evidence:** attach checksum outputs and signature verification findings to archived telemetry bundle metadata.

## 5) “What we never store” (unbanked-user protection)

To protect unbanked users and reduce sensitive-data blast radius, the audit/review pipeline never stores:

- Raw `user_id` values (only HMAC hashes)
- Full raw transaction feature vectors from requests
- Card PAN / card number / CVV / expiry
- Wallet password, PIN, OTP values, or secret reset material
- Government ID/passport/national-ID numbers
- Full bank or wallet account numbers
- Raw device fingerprint payloads beyond bounded risk summaries
- Private keys, auth bearer tokens, or session secrets
- Biometric templates (face/voice/fingerprint)

This list should be used in demos and judge reviews as a strict storage baseline; any exception requires explicit governance approval and README/runbook updates.

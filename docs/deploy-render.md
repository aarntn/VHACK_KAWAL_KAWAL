# Deploying Fraud API + Wallet API on Render

This repo now supports Docker deployment of both services:

- `fraud-api` (`Dockerfile.fraud-api`)
- `wallet-api` (`Dockerfile.wallet-api`)

## 1) Prerequisites

1. Push this repository to GitHub.
2. Create a Render account and connect the GitHub repo.

## 2) Create the Fraud API service

1. In Render dashboard, click **New +** → **Web Service**.
2. Select your repository.
3. Configure:
   - **Name**: `fraud-api`
   - **Environment**: `Docker`
   - **Dockerfile Path**: `Dockerfile.fraud-api`
   - **Branch**: your deployment branch
4. Set env vars:
   - `PORT=8000`
   - `GUNICORN_WORKERS=2`
   - `GUNICORN_TIMEOUT=60`
   - `FRAUD_MODEL_FILE=/app/project/models/final_xgboost_model.pkl`
   - `FRAUD_FEATURE_FILE=/app/project/models/feature_columns.pkl`
   - `FRAUD_THRESHOLD_FILE=/app/project/models/decision_thresholds.pkl`
   - `FRAUD_AUDIT_DIR=/app/project/outputs/audit`
   - `FRAUD_AUDIT_FILE=fraud_audit_log.jsonl`
5. Add a health check path: `/health/ready`.

### Persistent disk for audit logs

1. Open `fraud-api` service settings → **Disks** → **Add Disk**.
2. Mount path: `/app/project/outputs/audit`.
3. Size: at least `1 GB`.

This ensures `project/outputs/audit/fraud_audit_log.jsonl` is persisted across deploys/restarts.

## 3) Create the Wallet API service

1. In Render dashboard, click **New +** → **Web Service**.
2. Select the same repository.
3. Configure:
   - **Name**: `wallet-api`
   - **Environment**: `Docker`
   - **Dockerfile Path**: `Dockerfile.wallet-api`
   - **Branch**: your deployment branch
4. Set env vars:
   - `PORT=8001`
   - `GUNICORN_WORKERS=4`
   - `GUNICORN_TIMEOUT=60`
   - `UPSTREAM_TIMEOUT_SECONDS=2.0`
   - `UPSTREAM_MAX_RETRIES=2`
   - `UPSTREAM_BACKOFF_MS=40`
   - `UPSTREAM_BACKOFF_MAX_MS=400`
   - `MAX_INFLIGHT_REQUESTS=400`
   - `FRAUD_ENGINE_URL=https://<fraud-api-render-url>/score_transaction`
   - `FRAUD_ENGINE_HEALTH_URL=https://<fraud-api-render-url>/health`
5. Add a health check path: `/health/ready`.

## 4) Verify deployment

After both services are live:

1. Open `https://<fraud-api-render-url>/health/ready`
2. Open `https://<wallet-api-render-url>/health/ready`
3. Submit a payment payload to:
   - `POST https://<wallet-api-render-url>/wallet/authorize_payment`

## Model artifact strategy

Model artifacts are baked into the fraud image at build time from:

- `project/models/final_xgboost_model.pkl`
- `project/models/feature_columns.pkl`
- `project/models/decision_thresholds.pkl`

If you need to rotate models, commit updated artifacts and trigger a new Render deploy.

## Audit log persistence strategy

- Runtime path in container: `/app/project/outputs/audit/fraud_audit_log.jsonl`
- Persistence mechanism: Render persistent disk mounted at `/app/project/outputs/audit`
- Result: audit data survives container restarts and image redeploys.

## Deployment architecture (production baseline)

Use this section as the minimum deployment standard before demo/go-live.

### 1) CI pipeline: hard gates + deployment gating

This repo includes concrete GitHub Actions workflows:

1. **`.github/workflows/ci-gates.yml`**
   - Runs test suite: `python -m unittest discover -s project/tests -t . -p 'test_*.py'`
   - Verifies model artifact checksums via `project/scripts/verify_model_artifact_checksums.sh`
   - Runs vulnerability scanning via `pip-audit -r requirements.txt`
2. **`.github/workflows/deploy-gated.yml`**
   - Triggers only when `ci-gates` completes successfully on `main`
   - Calls Render deploy hooks (if secrets are configured)

Model artifact checksum source of truth:

- `project/models/artifact_checksums.sha256`

Gate behavior:

- Any failed gate blocks deploy.
- CI status is the single source of truth for release promotion.

Threshold governance add-on:

- Nightly ops can auto-run calibration and threshold promotion when policy checks pass.
- Promotion is recorded in `project/outputs/threshold_governance/latest_promotion_record.json` and previous thresholds are backed up for rollback.
- Use `project/scripts/release_gate_check.py` to block release when nightly ops status is not `ok`.

### 2) Secret management policy

**Do not hardcode salts, URLs, or API keys in application code.** Configure all sensitive values through environment variables sourced from Render encrypted environment groups or an external secrets manager.

Required controls:

- Store secrets only in Render secret env vars (or Vault/1Password/AWS Secrets Manager integration).
- Rotate secrets on a fixed cadence (for judges demo: at least once before final scoring window).
- Maintain separate secret sets per environment (`dev`, `staging`, `prod`).
- Restrict secret visibility to deployment maintainers.
- Add secret scanning in CI (for example: `gitleaks`).

### 3) Logging, metrics, and alerts

Capture JSON structured logs from both services and forward to a log backend (Render logs + external sink recommended).

Minimum telemetry:

- **Latency**
  - P50/P95/P99 for `/score_transaction` and `/wallet/authorize_payment`.
- **Error rate**
  - 4xx and 5xx rates per endpoint, plus upstream timeout rate from wallet -> fraud API.
- **Fraud model quality drift**
  - Monitor false-positive-rate (FPR) drift vs baseline window.
  - Alert when FPR exceeds agreed threshold band for consecutive windows.
- **Infrastructure signals**
  - Container restarts, memory saturation, disk usage for audit/profile stores.

Recommended alert thresholds (starter values):

- P95 latency > 750 ms for 10 minutes.
- 5xx error rate > 2% for 5 minutes.
- FPR drift > +20% relative to validated baseline for 3 evaluation windows.

### 4) Profile store production choice + backup/retention

The behavior profile store supports in-memory, SQLite, and Redis backends.

- **Preferred production choice:** `redis` (primary), with persistence enabled (AOF/RDB) and controlled TTL.
- **Fallback/low-scale option:** `sqlite` on persistent disk.
- **Do not use `memory` in production** (state loss on restart).

Backup and retention policy:

- **Redis**
  - Enable snapshot + append-only persistence.
  - Daily backup, 14-day retention minimum.
  - Quarterly restore drill.
- **SQLite**
  - Store DB on persistent disk.
  - Daily compressed snapshot copy to object storage.
  - 30-day retention minimum.
  - Weekly integrity check (`PRAGMA integrity_check;`).

## Environment variable templates

Use templates like below for environment groups (never commit real secret values).

### Fraud API (`fraud-api`)

```bash
# Runtime
PORT=8000
GUNICORN_WORKERS=2
GUNICORN_TIMEOUT=60

# Artifact paths
FRAUD_MODEL_FILE=/app/project/models/final_xgboost_model.pkl
FRAUD_FEATURE_FILE=/app/project/models/feature_columns.pkl
FRAUD_THRESHOLD_FILE=/app/project/models/decision_thresholds.pkl

# Audit logs
FRAUD_AUDIT_DIR=/app/project/outputs/audit
FRAUD_AUDIT_FILE=fraud_audit_log.jsonl

# Security / privacy (REQUIRED: set via secret manager)
FRAUD_HASH_KEY_VERSION=v2
FRAUD_HASH_SECRETS={"v1":"<old-secret>","v2":"<active-secret>"}
FRAUD_AUDIT_SIGNING_KEY_VERSION=v2
FRAUD_AUDIT_SIGNING_SECRETS={"v1":"<old-signing-secret>","v2":"<active-signing-secret>"}
# Optional backward-compatible fallback (prefer FRAUD_HASH_SECRETS instead):
# HASH_SALT=<legacy-single-secret>

# Behavior profiling
BEHAVIOR_PROFILE_STORE_BACKEND=redis
BEHAVIOR_PROFILE_TTL_SECONDS=86400
BEHAVIOR_PROFILE_REDIS_URL=<set-in-render-secret-env>
# If using sqlite fallback:
# BEHAVIOR_PROFILE_SQLITE_PATH=/app/project/outputs/behavior_profiles.sqlite3
```

### Wallet API (`wallet-api`)

```bash
# Runtime
PORT=8001
GUNICORN_WORKERS=4
GUNICORN_TIMEOUT=60
UPSTREAM_TIMEOUT_SECONDS=2.0
UPSTREAM_MAX_RETRIES=2
UPSTREAM_BACKOFF_MS=40
UPSTREAM_BACKOFF_MAX_MS=400
MAX_INFLIGHT_REQUESTS=400

# Fraud API routing
FRAUD_ENGINE_URL=https://<fraud-api-render-url>/score_transaction
FRAUD_ENGINE_HEALTH_URL=https://<fraud-api-render-url>/health

# Optional auth key for internal service calls (if enabled)
FRAUD_ENGINE_API_KEY=<set-in-render-secret-env>
```

## Go-live checklist (judges)

Before judges validation, confirm all checks below:

- [ ] CI is green on default branch (lint, tests, build, security scan, artifact checksum verification).
- [ ] Fraud and wallet images are tagged with release version + commit SHA.
- [ ] Model artifact checksums match approved release manifest.
- [ ] No hardcoded secrets/salts remain in app code; all sensitive values come from secret env vars.
- [ ] Render services expose healthy readiness endpoints:
  - `GET /health/ready` on fraud API
  - `GET /health/ready` on wallet API
- [ ] Persistent storage configured:
  - Audit log disk mounted for fraud API
  - Profile store backend set to Redis (or SQLite with backup job)
- [ ] Dashboards available for latency/error/FPR drift and alerts are enabled.
- [ ] Backup job has succeeded at least once; restore procedure is documented.
- [ ] End-to-end payment authorization smoke test passes against live URLs.

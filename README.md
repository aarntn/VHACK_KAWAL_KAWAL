# Fraud Shield — Real-Time Fraud Detection Stack

A production-grade fraud detection engine for digital wallet authorization. Combines a 5-component hybrid scorer, graph-based fraud ring detection, external MCP signals, async inference, segment-level fairness calibration, and a React dashboard.

**Case Study:** Digital Trust – Real-Time Fraud Shield for the Unbanked  
**Track:** Machine Learning (Fraud & Anomaly Detection)  
**SDG:** 8.10 — Expand access to financial services for unbanked populations  
**Dataset:** IEEE-CIS Fraud Detection (590,540 transactions, 3.5% fraud rate)

> Legacy `creditcard` training helpers remain for compatibility only. For judging and GitHub handoff, treat the IEEE-CIS pipeline and current runtime stack as the primary path.

> ASEAN demo context is now first-class in runtime behavior, not just in docs. The live request/response path supports corridor metadata, local-currency normalization, agent-assisted flows, and degraded-connectivity provenance for `SG`, `MY`, `ID`, `TH`, `PH`, and `VN`.

---

## Architecture

```
[React Dashboard / External Client]
           |
           v
+-------------------------------+
|  wallet_gateway_api  (:8001)  |  ← /wallet/authorize_payment
|  Circuit breaker + retries    |
+-------------------------------+
           |
           v
+-------------------------------+        +---------------------------+
|  hybrid_fraud_api    (:8000)  |  <-->  |  mock_mcp_server  (:8002) |
|                               |        |  Watchlist + device intel |
|  1. XGBoost base score        |        +---------------------------+
|  2. Context adjustment        |
|  3. Behavior EMA profile      |
|  4. Fraud ring graph score    |
|  5. MCP external signal       |
|                               |
|  SHAP explainability          |
|  HMAC audit chain             |
|  Segment thresholds           |
|  Async inference pipeline     |
+-------------------------------+
      |           |          |
      v           v          v
 models/*.pkl  profile    outputs/audit/
               store      *.jsonl
```

**Decision outputs:** `APPROVE` · `FLAG` (step-up auth) · `BLOCK`

---

## Key Documents

| Document | Location |
|----------|----------|
| Architecture + calibration pipeline | `docs/architecture_calibration_release_gates.md` |
| All metrics explained | `docs/metrics_glossary.md` |
| Results readout (model + latency + fairness) | `docs/preliminary_readout.md` |
| ASEAN runtime demo note | `docs/asean_demo_note.md` |
| ASEAN fairness summary | `docs/asean_fairness_summary.md` |
| Demo-safe backend checks | `docs/demo_safe_backend_checklist.md` |
| Live demo runtime seed plan | `docs/live_demo_runtime_seed_plan.md` |
| Backend proof-point narrative | `docs/backend_health_narrative.md` |
| Fairness demo note | `docs/fairness_demo_note.md` |
| Repo handoff notes | `docs/repo_handoff_notes.md` |
| Roadmap (Phases 1–4) | `project/ROADMAP.md` |
| Fairness thresholds applied | `project/outputs/governance/fairness_segment_thresholds_applied.json` |
| Ring fairness impact | `project/outputs/governance/ring_fairness_impact.md` |

---

## Prerequisites

- Python 3.11 or 3.12
- Node.js 18+
- IEEE-CIS dataset (Kaggle) — only needed to re-run training scripts
- Ports 8000, 8001, 8002 free

---

## 1. Install Python Dependencies

```bash
# From repo root
python -m pip install -r requirements.txt
```

On Windows, activate your virtual environment first:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 2. Start Backend Services

The recommended way is the unified launcher, which starts all three services together:

```bash
python project/scripts/launch_services.py
```

This starts:
- Fraud API on `:8000`
- Wallet Gateway on `:8001`
- Mock MCP server on `:8002`

### Launcher options

| Flag | Default | Description |
|------|---------|-------------|
| `--fraud-port` | 8000 | Fraud API port |
| `--wallet-port` | 8001 | Wallet gateway port |
| `--mcp-port` | 8002 | Mock MCP server port |
| `--workload-profile` | `balanced` | `balanced` / `cpu` / `io` — drives default worker count |
| `--fraud-workers N` | auto (2–4) | Explicit uvicorn worker count for fraud API |
| `--wallet-workers N` | auto (2–4) | Explicit uvicorn worker count for wallet gateway |
| `--upstream-timeout` | 0.8 | Wallet → fraud HTTP timeout in seconds |
| `--upstream-max-retries` | 1 | Retry attempts before circuit trips |
| `--upstream-backoff-ms` | 25 | Backoff base in milliseconds |
| `--no-mcp` | off | Skip the mock MCP server entirely |
| `--reload` | off | Enable uvicorn auto-reload (dev only) |
| `--allow-windows-multi-worker` | off | Allow >1 worker on Windows (unstable) |

> **Windows note:** Multi-worker mode is unstable on Windows. The launcher automatically downgrades to 1 worker per service unless `--allow-windows-multi-worker` is passed. Use WSL2 or Linux for load testing.

**Example — dev mode with MCP disabled:**
```bash
python project/scripts/launch_services.py --no-mcp --reload
```

**Example — load test profile:**
```bash
python project/scripts/launch_services.py --workload-profile cpu --fraud-workers 4 --wallet-workers 4
```

### Manual start (without launcher)

```bash
# Terminal 1 — fraud API
python -m uvicorn project.app.hybrid_fraud_api:app --port 8000

# Terminal 2 — wallet gateway
FRAUD_ENGINE_URL=http://127.0.0.1:8000/score_transaction \
python -m uvicorn project.app.wallet_gateway_api:app --port 8001

# Terminal 3 — mock MCP (optional)
python -m uvicorn project.scripts.mock_mcp_server:app --port 8002
```

### Health checks

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health
```

The fraud API health response includes MCP status:
```json
{
  "status": "ok",
  "mcp": {
    "mode": "enabled",
    "breaker_open": false,
    "breaker_failure_count": 0,
    "mcp_url": "http://127.0.0.1:8002"
  }
}
```

`mode` values: `"enabled"` · `"disabled"` (started with `--no-mcp`) · `"degraded"` (circuit open after failures)

---

## 3. Start the Frontend

```bash
cd project/frontend
npm install
cp .env.example .env          # sets VITE_FRAUD_API_BASE_URL and VITE_WALLET_API_BASE_URL
npm run dev
```

Open the Vite URL shown in the terminal (default `http://localhost:5173`).

**Frontend .env variables:**
```
VITE_FRAUD_API_BASE_URL=http://127.0.0.1:8000
VITE_WALLET_API_BASE_URL=http://127.0.0.1:8001
```

Operator analytics access is no longer bundled into the frontend build. Enter the operator access code at runtime in the dashboard Access panel instead of shipping `VITE_*` secrets to the browser.

**Build for production:**
```bash
npm run build
npm run preview
```

**Frontend pages:**
- `/` — Transaction scorer (submit a transaction, see APPROVE/FLAG/BLOCK + explainability)
- `/dashboard` — Admin dashboard (latency benchmarks, drift monitor, ring KPIs)
- `/rings` — Fraud ring graph (interactive D3 force visualization)

### Optional: seed a live demo dataset

If localhost looks too empty for a convincing B2B demo, seed realistic behavior history plus a
repeatable batch of wallet transactions:

```powershell
.\.venv\Scripts\Activate.ps1
python project\scripts\seed_demo_runtime.py --reset-runtime
```

This populates:
- behavior profiles
- audit log traffic
- review queue entries
- dashboard metrics

Reference plan:
- `docs/live_demo_runtime_seed_plan.md`

---

## 4. Environment Variables

### Fraud API (`hybrid_fraud_api.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAUD_INFERENCE_BACKEND` | `xgboost_inplace_predict` | Inference backend: `xgboost_inplace_predict` / `onnx` |
| `FRAUD_SHAP_TOP_N` | `5` | Number of top SHAP feature drivers returned per request. Set to `0` to disable |
| `FRAUD_RING_SCORES_PATH` | `project/outputs/monitoring/fraud_ring_scores.json` | Path to prebuilt ring score lookup |
| `FRAUD_RING_SCORE_WEIGHT` | `0.08` | Weight applied to ring score when adjusting final risk score |
| `FRAUD_RING_SCORE_CAP` | `0.20` | Maximum absolute ring score adjustment |
| `FRAUD_MCP_URL` | `http://127.0.0.1:8002` | MCP external signal server URL |
| `FRAUD_MCP_ENABLED` | `true` | Set `false` to disable MCP lookups entirely |
| `FRAUD_MCP_API_KEY` | _(empty)_ | API key sent as `X-MCP-Api-Key` header |
| `FRAUD_MCP_TIMEOUT_MS` | `50` | Per-request MCP timeout in milliseconds |
| `FRAUD_MCP_CACHE_TTL_S` | `60.0` | TTL for cached MCP responses (seconds) |
| `FRAUD_MCP_CACHE_MAXSIZE` | `2000` | Maximum entries in the MCP result cache |
| `FRAUD_MCP_CB_FAILURES` | `5` | MCP failures before circuit opens |
| `FRAUD_MCP_CB_RESET_S` | `30.0` | Seconds before circuit tries to close again |
| `FRAUD_MCP_BOOST_HIGH` | `0.08` | Score adjustment for high-tier watchlist hit |
| `FRAUD_MCP_BOOST_MED` | `0.05` | Score adjustment for medium-tier watchlist hit |
| `FRAUD_MCP_BOOST_LOW` | `0.02` | Score adjustment for low-tier watchlist hit |
| `FRAUD_MCP_CLEAN_DISCOUNT` | `-0.02` | Score adjustment for verified-clean device |
| `FRAUD_MCP_SHARED_BOOST` | `0.04` | Boost for device shared by 5+ accounts |
| `FRAUD_MCP_ADJUSTMENT_CAP` | `0.10` | Maximum absolute MCP adjustment (prevents signal dominance) |
| `FRAUD_OPERATOR_API_KEY` | _(empty)_ | Operator access code required for `/dashboard/views`, `/config`, `/metrics`, and review/curation endpoints |
| `FRAUD_OPERATOR_AUTH_MODE` | `required` | `required` / `enabled` / `disabled` / `auto`; keep `required` for demos and deployment integrity |
| `FRAUD_CORS_ALLOW_ORIGINS` | localhost dev origins | Comma-separated allowed browser origins for the fraud API |

### Wallet Gateway (`wallet_gateway_api.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAUD_ENGINE_URL` | `http://127.0.0.1:8000/score_transaction` | Fraud API endpoint |
| `FRAUD_ENGINE_HEALTH_URL` | `http://fraud-api:8000/health` | Health check URL for circuit breaker probe |
| `UPSTREAM_TIMEOUT_SECONDS` | `0.8` | HTTP timeout per attempt |
| `UPSTREAM_MAX_RETRIES` | `1` | Retry count before giving up |
| `UPSTREAM_BACKOFF_MS` | `20` | Base backoff between retries |
| `UPSTREAM_BACKOFF_MAX_MS` | `150` | Maximum backoff cap |
| `UPSTREAM_BACKOFF_JITTER_RATIO` | `0.3` | Jitter fraction added to backoff |
| `UPSTREAM_RETRY_STATUS_CODES` | `429,500,502,503,504` | HTTP status codes that trigger retry |
| `MAX_INFLIGHT_REQUESTS` | `400` | Concurrency cap on the wallet gateway |
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Failures before wallet → fraud circuit opens |
| `CIRCUIT_BREAKER_RESET_SECONDS` | `30` | Circuit reset window |
| `FALLBACK_ENGINE_DECISION` | `FLAG` | Decision returned when fraud API is unreachable |

---

## 5. API Reference

### `POST /score_transaction`  (port 8000)

**Request schema version:** `ieee_fraud_tx_v1`

**ASEAN runtime additions:**
- optional `currency`
- optional `source_country` / `destination_country` using ISO 3166-1 alpha-2 for the supported demo set
- optional `is_agent_assisted`
- optional `connectivity_mode`

**Minimum payload (public / frontend):**
```json
{
  "schema_version": "ieee_fraud_tx_v1",
  "user_id": "user_001",
  "transaction_amount": 72.50,
  "device_risk_score": 0.10,
  "ip_risk_score": 0.05,
  "location_risk_score": 0.05
}
```

**Full payload with optional context fields:**
```json
{
  "schema_version": "ieee_fraud_tx_v1",
  "user_id": "user_001",
  "transaction_amount": 72.50,
  "device_risk_score": 0.10,
  "ip_risk_score": 0.05,
  "location_risk_score": 0.05,
  "device_id": "device-abc-123",
  "device_shared_users_24h": 1,
  "account_age_days": 180,
  "sim_change_recent": false,
  "tx_type": "MERCHANT",
  "channel": "APP",
  "cash_flow_velocity_1h": 1,
  "p2p_counterparties_24h": 0,
  "is_cross_border": false,
  "currency": "SGD",
  "source_country": "SG",
  "destination_country": "SG",
  "is_agent_assisted": false,
  "connectivity_mode": "online"
}
```

`tx_type` options: `P2P` · `MERCHANT` · `CASH_IN` · `CASH_OUT`  
`channel` options: `APP` · `AGENT` · `QR` · `WEB`
`connectivity_mode` options: `online` · `intermittent` · `offline_buffered`

**Internal benchmark payload (IEEE-style V-features):**
```json
{
  "schema_version": "ieee_fraud_tx_v1",
  "user_id": "user_safe",
  "TransactionDT": 67070.12,
  "TransactionAmt": 454.84,
  "V1": -0.04, "V2": 7.36, "V3": -7.83, "V4": 6.96,
  "V5": -5.42, "V6": -5.71, "V7": -5.19, "V8": 5.05,
  "V9": -5.85, "V10": 7.59, "V11": 7.31, "V12": -6.07,
  "V13": 5.29, "V14": 7.18, "V15": 1.97, "V16": -4.53,
  "V17": -1.31,
  "device_risk_score": 0.36,
  "ip_risk_score": 0.83,
  "location_risk_score": 0.68
}
```

**Response:**
```json
{
  "request_id": "...",
  "correlation_id": "...",
  "risk_score": 0.42,
  "final_risk_score": 0.47,
  "decision": "FLAG",
  "decision_source": "score_band",
  "runtime_mode": "primary",
  "corridor": "SG-PH",
  "normalized_amount_reference": 310.8,
  "normalization_basis": "static_demo_reference_snapshot_2026q2:SGD->USD_EQUIVALENT",
  "fraud_reasons": ["high_ip_risk", "cash_out_tx"],
  "reasons": ["high_ip_risk", "cash_out_tx"],
  "reason_codes": ["MODEL_ELEVATED_FRAUD_PROBABILITY", "ASEAN_FIRST_CROSS_BORDER_REMITTANCE"],
  "explainability": {
    "base": 0.31,
    "context": 0.08,
    "behavior": 0.03,
    "ring": 0.00,
    "ring_reason_codes": [],
    "external": 0.05,
    "top_feature_drivers": [
      {"feature": "device signal present", "shap_value": 0.608, "direction": "increases_risk"},
      {"feature": "amount raw", "shap_value": 0.358, "direction": "increases_risk"}
    ]
  },
  "stage_timings_ms": {
    "total_pipeline_ms": 42.1,
    "details": {}
  }
}
```

**Validation errors** return HTTP `422`:
```json
{
  "error": "ValidationError",
  "schema_version_expected": "ieee_fraud_tx_v1",
  "detail": "...",
  "details": [{"field": "transaction_amount", "code": "...", "message": "...", "input": "..."}]
}
```

### `POST /wallet/authorize_payment`  (port 8001)

```json
{
  "schema_version": "ieee_fraud_tx_v1",
  "user_id": "user_001",
  "wallet_id": "wallet_abc",
  "merchant_name": "GrabFood",
  "currency": "MYR",
  "source_country": "MY",
  "destination_country": "MY",
  "is_agent_assisted": true,
  "connectivity_mode": "intermittent",
  "transaction_amount": 25.00,
  "device_risk_score": 0.05,
  "ip_risk_score": 0.05,
  "location_risk_score": 0.05
}
```

Response includes `wallet_action` (`APPROVED` / `PENDING_VERIFICATION` / `DECLINED_FRAUD_RISK`), `runtime_mode`, `corridor`, normalized-amount provenance, circuit breaker state, `attempt_durations_ms`, `backoff_total_ms`, and `upstream_call_ms` in `stage_timings_ms`.

### Other Endpoints

| Endpoint | Service | Description |
|----------|---------|-------------|
| `GET /health` | 8000 / 8001 | Service health + MCP circuit breaker state |
| `GET /api/info` | 8000 | Model artifact info, thresholds, MCP config, ring graph stats |
| `GET /ring/graph` | 8000 | Ring nodes + exact observed edges for D3 visualization when evidence links are available |
| `GET /privacy` | 8000 | Audit log field contracts |
| `GET /review_queue` | 8000 | Pending FLAG/BLOCK cases for analyst review |
| `GET /docs` | 8000 / 8001 | Interactive Swagger UI |

---

## 6. Run Tests

```bash
# All tests
python -m unittest discover -s project/tests -t . -p "test_*.py"

# Specific suite
python -m unittest discover -s project/tests/fraud -t . -p "test_*.py"

# Single file
python -m unittest project/tests/fraud/test_health_mcp_fields.py
python -m unittest project/tests/fraud/test_benchmark_compare_script.py
```

Tests are in `project/tests/fraud/` and `project/tests/wallet/`. Notable test files:
- `test_health_mcp_fields.py` — MCP circuit breaker snapshot, health response fields, wallet timing fields (26 tests)
- `test_benchmark_compare_script.py` — CSV delta table, SLA transitions, cp1252-safe output (20 tests)
- `test_behavior_profile_health_script.py` — behavior EMA profile health
- `test_drift_monitor_script.py` — PSI drift detection

---

## 7. Latency Benchmark

```bash
# Start services first
python project/scripts/launch_services.py --fraud-workers 2 --wallet-workers 2

# Run benchmark (new terminal)
python project/scripts/benchmark_latency.py --requests 200 --concurrency 6

# Judge-mode sweep (strict SLA gate)
python project/scripts/benchmark_latency.py \
  --requests 200 \
  --concurrency-sweep 2,6,12 \
  --sla-p95-ms 250 \
  --sla-p99-ms 500 \
  --sla-error-rate-pct 1.0
```

Outputs: `project/outputs/benchmark/latency_benchmark_*.json` and `.csv`

**Compare two benchmark runs:**
```bash
python project/scripts/benchmark_compare.py \
  --before project/outputs/benchmark/latency_benchmark_BEFORE.csv \
  --after  project/outputs/benchmark/latency_benchmark_AFTER.csv
```

Prints a p50/p95/p99 delta table with SLA pass/fail transitions and saves JSON + CSV to `project/outputs/benchmark/`.

---

## 8. Fraud Ring Graph

The ring layer now uses three complementary artifacts:

- `fraud_ring_scores.json` for known account-member lookups
- `fraud_ring_evidence_links.json` for exact account-to-attribute graph edges
- `fraud_ring_attribute_index.json` for unseen-account attribute matching

To rebuild the ring artifacts from backend events:

```bash
python project/scripts/build_fraud_ring_graph.py --events-path path/to/events.jsonl
```

Outputs (generated on first run):
- `project/outputs/monitoring/fraud_ring_summary.json` — build summary + canonical artifact pointers
- `project/outputs/monitoring/fraud_ring_reports.json` — ring reports for the primary inference window
- `project/outputs/monitoring/fraud_ring_scores.json` — per-account ring scores loaded by the API
- `project/outputs/monitoring/fraud_ring_evidence_links.json` — exact graph evidence links
- `project/outputs/monitoring/fraud_ring_attribute_index.json` — per-attribute risk index (generated when events data is provided)

Artifact modes:
- `label_mode=labeled` — may emit `fraud_count` and `fraud_rate`
- `label_mode=topology_only` — structural evidence only; no label-derived claims

The `/ring/graph` endpoint only returns exact evidence links when that artifact exists. If evidence links are missing, it returns summaries and nodes with `links=[]` instead of fabricating implied edges.

To measure directional ring uplift on synthetic data:
```bash
python project/scripts/evaluate_ring_ablation.py
```

Outputs `project/outputs/monitoring/ring_ablation_report.md` with `evidence_class=synthetic_projection`.

To measure ring uplift on labeled replay data:
```bash
python project/scripts/evaluate_ring_replay.py --events-path path/to/labeled_replay.jsonl
```

Outputs (generated when labeled replay data is available):
- `project/outputs/monitoring/ring_replay_report.json`
- `project/outputs/monitoring/ring_replay_report.md`

Use the replay report as the primary proof artifact when labeled replay data is available. Use the synthetic ablation report (`ring_ablation_report.md`) as directional support when labeled data is not yet available.

To assess fairness impact of ring signals per segment:
```bash
python project/scripts/ring_fairness_impact.py
```

Outputs:
- `project/outputs/governance/ring_fairness_impact.json`
- `project/outputs/governance/ring_fairness_impact.md`

The fairness report now also includes audit-derived runtime counts for ring-applied, ring-suppressed, and blocked-with-ring outcomes.

For demo/handoff talking points, see `docs/ring_remediation_note.md`.

---

## 9. Nightly Ops (Automated)

Single command that runs all monitoring, drift detection, KPI reporting, benchmark, and archiving:

```bash
# Linux/macOS
bash project/scripts/run_nightly_ops.sh

# Windows PowerShell
powershell project/scripts/run_nightly_ops.ps1

# Or directly
python project/scripts/nightly_ops.py \
  --audit-log project/outputs/audit/fraud_audit_log.jsonl \
  --run-benchmark \
  --benchmark-sla-mode warn \
  --run-cohort-kpi \
  --run-profile-replay \
  --run-profile-health \
  --archive-runs-dir project/outputs/ops_runs \
  --ops-summary-json project/outputs/monitoring/nightly_ops_summary.json
```

Nightly ops pipeline steps:
1. Dataset + label validity checks
2. Scoring drift monitor (PSI per feature + decision mix drift)
3. Threshold-policy guardrails (approve < block ordering, promotion policy)
4. Optional recalibration + threshold promotion
5. Latency benchmark + trend report
6. Latency stage analysis
7. Cohort KPI generation
8. Behavior profile replay + health check
9. Retrain trigger evaluation + alert metadata
10. Per-run artifact archive with checksum manifest → `project/outputs/ops_runs/<run_id>/`

---

## 10. ML Training Pipeline

> The IEEE-CIS dataset is not bundled. Download from Kaggle:
> `train_transaction.csv` + `train_identity.csv`

### Pass 1 — Imbalance strategy benchmark

```bash
python project/scripts/benchmark_imbalance_strategies.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /data/ieee/train_transaction.csv \
  --ieee-identity-path /data/ieee/train_identity.csv \
  --split-mode time \
  --seed 42
```

Benchmarks 7 strategies (baseline, random undersample, SMOTE, ADASYN, SMOTE+undersample, SMOTE+ENN, ADASYN+ENN). **Winner: `smote_enn`** — best robustness score across 11 temporal windows.

### Pass 2 — Model candidate tuning + comparison

```bash
python project/scripts/tune_model_candidates.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /data/ieee/train_transaction.csv \
  --ieee-identity-path /data/ieee/train_identity.csv \
  --split-mode groupkfold_uid \
  --groupkfold-n-splits 5 \
  --optuna-trials 20 \
  --seed 42
```

Runs XGBoost (default + Optuna-tuned), optional LightGBM/CatBoost. Produces robustness report over 11 validation windows. **Winner: `xgboost_tuned`** (ROC-AUC 0.951, PR-AUC 0.746).

### Pass 3 — Preprocessing evaluation + promotion

```bash
python project/scripts/evaluate_preprocessing_settings.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /data/ieee/train_transaction.csv \
  --ieee-identity-path /data/ieee/train_identity.csv \
  --split-mode time

python project/scripts/promote_best_preprocessing_setting.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /data/ieee/train_transaction.csv \
  --ieee-identity-path /data/ieee/train_identity.csv \
  --validation-robustness-report project/outputs/monitoring/ieee_cis_validation_robustness_report.json
```

### Pass 4 — Context weight calibration

Grid search over 14 context signal weights, maximising F1 at target FPR ≤ 0.15.

```bash
python project/scripts/calibrate_context.py \
  --model-path project/models/final_xgboost_model_promoted_preproc.pkl \
  --feature-path project/models/feature_columns_promoted_preproc.pkl \
  --baseline-thresholds-path project/models/decision_thresholds_promoted_preproc.pkl \
  --trials 350 \
  --seed 42
```

### Pass 5 — Fairness + explainability governance

```bash
python project/scripts/fairness_explainability_report.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /data/ieee/train_transaction.csv \
  --ieee-identity-path /data/ieee/train_identity.csv \
  --model-path project/models/final_xgboost_model_promoted_preproc.pkl \
  --feature-path project/models/feature_columns_promoted_preproc.pkl \
  --thresholds-path project/models/decision_thresholds_promoted_preproc.pkl \
  --max-fpr-gap 0.08 \
  --max-recall-gap 0.12
```

On Windows if you hit `ModuleNotFoundError: No module named project`:
```powershell
python -m project.scripts.fairness_explainability_report --dataset-source ieee_cis ...
```

Outputs:
- `project/outputs/governance/fairness_explainability_report.json`
- `project/outputs/governance/fairness_segment_metrics.csv`
- `project/outputs/governance/model_top_feature_drivers.csv`

Governance decision step (applies per-segment block thresholds):
```bash
python project/scripts/fairness_segment_decision.py \
  --segment-metrics-csv project/outputs/governance/fairness_segment_metrics.csv \
  --max-fpr-gap 0.08 \
  --max-fnr-gap 0.12
```

### Adversarial validation (leakage check)

```bash
python project/scripts/adversarial_validation.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /data/ieee/train_transaction.csv \
  --ieee-identity-path /data/ieee/train_identity.csv
```

Drops features with adversarial importance > 0.02 (8 time-contaminated features removed).

---

## 11. Governance & Evidence

### Release gate check

```bash
python project/scripts/release_gate_check.py \
  --ops-summary-json project/outputs/monitoring/nightly_ops_summary.json
```

Exits non-zero if any gate is violated (drift alert, latency SLA breach, fairness severe segment without waiver).

### Build evidence bundle

```bash
python project/scripts/build_evidence_bundle.py
```

Creates a timestamped archive under `project/outputs/evidence_bundles/` with all monitoring artifacts and a checksum manifest.

### Drift monitor (standalone)

```bash
python project/scripts/drift_monitor.py \
  --audit-log project/outputs/audit/fraud_audit_log.jsonl \
  --output-json project/outputs/monitoring/drift_report.json
```

PSI thresholds: warn > 0.10, alert > 0.25. Decision drift thresholds: warn > 0.10, alert > 0.20.

### Cohort KPI report

```bash
python project/scripts/cohort_kpi_report.py \
  --output-json project/outputs/monitoring/cohort_kpi_report.json
```

Covers: `all_users` · `new_users` · `rural_merchants_proxy` · `gig_workers_proxy` · `low_history_proxy`

### Behavior profile health

```bash
python project/scripts/behavior_profile_health.py \
  --store-backend sqlite \
  --sqlite-path project/outputs/behavior_profiles.sqlite3 \
  --output-json project/outputs/monitoring/behavior_profile_health.json
```

### Explainability stability check

```bash
python project/scripts/explainability_stability_check.py
```

Re-scores 3 representative cases (approve/flag/block) 25 times each with ±5% jitter. Requires `decision_consistency = 1.0` to pass.

---

## 12. Decision Thresholds

| Score range | Decision |
|-------------|----------|
| `< 0.3323` | `APPROVE` |
| `0.3323 – 0.5504` | `FLAG` (step-up verification) |
| `>= 0.5504` | `BLOCK` |

Thresholds are stored in `project/models/decision_thresholds_promoted_preproc.pkl`. Per-segment overrides are applied at runtime from `project/outputs/governance/fairness_segment_thresholds_applied.json`.

---

## 13. Project Structure

```
project/
├── app/
│   ├── hybrid_fraud_api.py       ← Fraud API (FastAPI, async)
│   ├── wallet_gateway_api.py     ← Wallet gateway (FastAPI, async)
│   ├── behavior_profile.py       ← EMA behavior profiler
│   ├── feature_store.py          ← InMemory / Redis feature store
│   ├── profile_store.py          ← InMemory / Redis profile store
│   ├── inference_backends.py     ← XGBoost / ONNX backends
│   ├── rules.py                  ← Hard-rule overrides
│   ├── schemas/api.py            ← Request/response Pydantic models
│   └── mcp/
│       ├── watchlist_client.py   ← MCP client (async, TTL cache, circuit breaker)
│       └── __init__.py
├── scripts/
│   ├── launch_services.py        ← Unified service launcher
│   ├── mock_mcp_server.py        ← Mock external signal server
│   ├── nightly_ops.py            ← Full ops pipeline
│   ├── benchmark_latency.py      ← Latency load tester
│   ├── benchmark_compare.py      ← Before/after delta table
│   ├── build_fraud_ring_graph.py ← Build ring graph from scratch
│   ├── evaluate_ring_ablation.py ← Measure ring recall impact
│   ├── evaluate_ring_replay.py   ← Measure ring uplift on labeled replay data
│   ├── ring_fairness_impact.py   ← Ring FPR per fairness segment
│   ├── fairness_explainability_report.py
│   ├── fairness_segment_decision.py
│   ├── drift_monitor.py
│   ├── cohort_kpi_report.py
│   ├── release_gate_check.py
│   └── build_evidence_bundle.py
├── models/
│   ├── final_xgboost_model_promoted_preproc.pkl
│   ├── feature_columns_promoted_preproc.pkl
│   └── decision_thresholds_promoted_preproc.pkl
├── tests/
│   ├── fraud/                    ← 80+ unit + integration tests
│   └── wallet/
├── outputs/
│   ├── audit/                    ← HMAC-chained audit log (jsonl)
│   ├── benchmark/                ← Latency benchmark CSVs + JSON
│   ├── governance/               ← Fairness reports, ring impact
│   ├── monitoring/               ← Drift, KPI, ring summary, model benchmarks
│   └── ops_runs/                 ← Nightly ops run archives
├── docs/
│   ├── architecture_calibration_release_gates.md
│   ├── metrics_glossary.md
│   └── preliminary_readout.md
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── Dashboard.tsx
│   │   ├── UserForm.tsx
│   │   ├── UserResult.tsx
│   │   ├── DetailUser.tsx
│   │   └── components/
│   │       ├── RingGraph.tsx     ← D3 ring force graph
│   │       ├── DashboardDriftCard.tsx
│   │       ├── DashboardBenchmarkCard.tsx
│   │       └── DashboardKpiCard.tsx
│   └── package.json
└── deploy/
    ├── gunicorn_conf.py
    └── download_artifacts.py
project/ROADMAP.md
requirements.txt
```

---

## 14. Extension Points Already Implemented

These are live in code but not activated in the default single-node deployment:

| Hook | File | Status |
|------|------|--------|
| `RedisFeatureStore` | `app/feature_store.py` | Implemented — activate by setting `FEATURE_STORE_BACKEND=redis` |
| `RedisProfileStore` | `app/profile_store.py` | Implemented — activate by setting `PROFILE_STORE_BACKEND=redis` |
| `CassandraFeatureStore` | `app/feature_store.py` | Stub (protocol defined, not configured) |
| `OnnxOrHummingbirdBackend` | `app/inference_backends.py` | Selectable via `FRAUD_INFERENCE_BACKEND=onnx` |
| `segment_thresholds` | `app/hybrid_fraud_api.py` | Live — applied per cohort key from JSON |
| `FRAUD_SHAP_TOP_N` | `app/hybrid_fraud_api.py` | Live — per-request SHAP, default 5 |
| MCP async client | `app/mcp/watchlist_client.py` | Live — parallel watchlist + device lookup |
| Circuit breaker (fraud→MCP) | `app/mcp/watchlist_client.py` | Live — 5 failures → 30s open |
| Circuit breaker (wallet→fraud) | `app/wallet_gateway_api.py` | Live — 5 failures → 30s open |
| Nightly ops | `scripts/nightly_ops.py` | Live — fully automated |
| Evidence bundles | `scripts/build_evidence_bundle.py` | Live — timestamped archives |

---

## 15. Privacy & Audit

- **HMAC-SHA256 chained audit log** — each record signs the previous record's hash. Any tampering invalidates the chain.
- **User IDs are hashed** before storage (`hash_key_version` tracks key rotation). Raw user IDs are never written to disk.
- **Raw transaction feature vectors are never stored** — audit log records decisions, reasons, and hashes only.
- **MCP adjustment cap** — external signal adjustments are capped at ±0.10 to prevent a single upstream signal from dominating the score.
- **`/privacy` endpoint** — publishes the complete list of stored audit fields.

---

## 16. Runtime Artifact Policy

- Runtime artifacts (`.pyc`, `__pycache__/`) must not be committed.
- Virtual environments (`.venv/`) must not be committed.
- Model artifacts (`*.pkl`) are pre-generated and committed for deployment.
- No exceptions to the above are currently approved.

---

## 17. Scoring Pipeline — How the 5 Components Combine

Every request to `/score_transaction` passes through these stages in order:

```
1. Feature preparation
   Raw request fields → canonical feature vector (preprocessing + imputation)
   ~90–97% of total latency on single-worker Windows due to pandas overhead

2. XGBoost base score  (base)
   Booster.inplace_predict(feature_vector) → float in [0, 1]
   Represents the model's posterior fraud probability at threshold=0.5

3. Context adjustment  (context)
   context_score = Σ weight_i × signal_i
   Signals: device_risk, ip_risk, location_risk, is_cross_border,
            sim_change_recent, is_new_payee, account_age, device_shared_users,
            cash_flow_velocity, p2p_counterparties, tx_type, channel
   Capped at ±CONTEXT_ADJUSTMENT_MAX (default 0.30)

4. Behavior adjustment  (behavior)
   behavior_score = f(amount_deviation, frequency_deviation, time_deviation)
   Derived from per-user EMA profile stored in SQLite / Redis
   New accounts start at 0.0 (no historical signal)

5. Ring score adjustment  (ring)
   ring_adjustment = FRAUD_RING_SCORE_WEIGHT × ring_score
   ring_score comes from account-member lookup first, then attribute-index matching for unseen accounts
   Exact account-to-attribute evidence is persisted separately for `/ring/graph`
   Ring adjustment is gated by ring size, corroboration count, artifact recency, and a ring-only block fairness guard
   Capped at FRAUD_RING_SCORE_CAP (default 0.20)

6. MCP external signal  (external)
   Async parallel lookup: watchlist + device intel
   Adjustment: +0.08 (high-tier hit) / +0.05 (med) / +0.02 (low) / -0.02 (clean)
   Capped at ±FRAUD_MCP_ADJUSTMENT_CAP (default 0.10)
   Falls back to 0.0 on circuit-open or timeout

final_risk_score = base + context + behavior + ring + external
                   clipped to [0.0, 1.0]

7. Segment threshold lookup
   The decision gate (APPROVE / FLAG / BLOCK) uses per-segment thresholds
   for accounts in fairness-adjusted cohorts, falling back to global thresholds

8. SHAP explainability (async, parallel with steps 3–6)
   XGBoost predict_contribs → top-N feature SHAP values
   Returned in response as explainability.top_feature_drivers
   Controlled by FRAUD_SHAP_TOP_N (default 5, 0 = disabled)
```

**Async execution:** steps 3–8 run on the FastAPI event loop. The CPU-bound steps (1–2) are offloaded to a thread-pool executor via `run_in_executor` so the event loop stays unblocked during I/O waits (MCP, audit log write).

---

## 18. Pre-flight Contract Check

Before starting services for a demo or test run, validate that all frontend scenario presets produce the expected decisions end-to-end:

```bash
python project/scripts/preset_contract_check.py
```

This starts the fraud API in-process (no network), scores every preset defined in `project/frontend/src/scenarioPresets.json`, and asserts that each one returns the expected `decision` and is within the expected score band. Exits non-zero if any preset fails.

The default preset suite is now the ASEAN demo pack:
- `ID-ID` domestic QR payment in `IDR`
- `SG-PH` first-time remittance in `SGD`
- `MY-MY` agent-assisted cash-out in `MYR`
- `TH-VN` suspicious cross-border pattern in `THB`

Run this after any change to thresholds, context weights, or model artifacts.

---

## 19. Additional Monitoring Scripts

### Latency stage analysis

Parses the audit log and breaks down per-request latency into named pipeline stages:

```bash
python project/scripts/latency_stage_analysis.py \
  --audit-log project/outputs/audit/fraud_audit_log.jsonl \
  --output-json project/outputs/monitoring/latency_stage_analysis.json \
  --output-csv project/outputs/monitoring/latency_stage_analysis.csv
```

Stage breakdown in output: `feature_preparation_ms`, `model_inference_ms`, `context_scoring_ms`, `audit_log_write_ms`. Useful for identifying which pipeline step dominates latency on a given deployment.

### Latency trend report

Aggregates all benchmark JSON files in the benchmark directory into a time-series trend:

```bash
python project/scripts/benchmark_trend_report.py \
  --benchmark-dir project/outputs/benchmark \
  --output-json project/outputs/monitoring/latency_trend_report.json \
  --output-csv project/outputs/monitoring/latency_trend_report.csv
```

### Replay behavior profiles

Seeds the behavior profile store by replaying historical dataset transactions through the EMA profiler. Run once before starting services in a fresh environment so user profiles are pre-warmed:

```bash
python project/scripts/replay_behavior_profiles.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /data/ieee/train_transaction.csv \
  --ieee-identity-path /data/ieee/train_identity.csv \
  --sqlite-path project/outputs/behavior_profiles.sqlite3 \
  --user-count 200 \
  --output-json project/outputs/monitoring/profile_replay_summary.json
```

### Threshold operating points

Generates a precision/recall/FPR sweep across all thresholds to find Pareto-optimal operating points:

```bash
python project/scripts/threshold_operating_points.py \
  --sweep-csv project/outputs/monitoring/ieee_cis_preprocessing_threshold_comparison.csv \
  --output-csv project/outputs/monitoring/ieee_cis_operating_points.csv \
  --output-json project/outputs/monitoring/ieee_cis_operating_points.json
```

Useful for choosing a block threshold that satisfies a specific FPR budget or minimum recall requirement.

### Time-consistency feature filter

Identifies features whose predictive power degrades significantly between early and late time periods (a form of time leakage / concept drift):

```bash
python project/scripts/time_consistency_feature_filter.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /data/ieee/train_transaction.csv \
  --ieee-identity-path /data/ieee/train_identity.csv
```

Outputs `project/outputs/monitoring/time_consistency_feature_decisions.json` with `keep`/`drop` decisions per feature. The 8 features dropped in the final model were identified here.

### Canary rollout guard

Evaluates staged rollout telemetry windows and emits a `PROCEED` / `ROLLBACK` decision:

```bash
python project/scripts/canary_rollout_guard.py \
  --telemetry-json path/to/canary_telemetry.json \
  --output-json project/outputs/rollout_telemetry/canary_decision.json \
  --artifact-id fraud-engine \
  --artifact-version 1.0.0 \
  --release-id release-2026-04-20 \
  --max-error-rate-pct 1.0 \
  --max-p95-latency-ms 250.0 \
  --rollback-consecutive-windows 3
```

Triggers `ROLLBACK` if 3 consecutive telemetry windows breach error rate or p95 SLA. Archives telemetry with release metadata under `project/outputs/rollout_telemetry/`.

---

## 20. Dataset — Stage 1 Onboarding

Validate dataset ingestion before training (produces reproducible onboarding artifacts, no model trained):

```bash
python project/scripts/build_dataset_stage1_artifacts.py \
  --dataset-source ieee_cis \
  --transaction-path /data/ieee/train_transaction.csv \
  --identity-path /data/ieee/train_identity.csv
```

Outputs under `project/outputs/dataset_stage1/`:
- `ieee_cis_head.csv` — first rows of merged dataset
- `ieee_cis_schema.json` — column types and missingness
- `ieee_cis_quality.json` — target prevalence, cardinality, timestamp sanity
- `ieee_cis_canonical_preview.csv` — preprocessed feature preview

Acceptance checks run automatically:
- Target column exists, is binary (0/1)
- `TransactionID` is unique in the transaction table
- Positive rate within expected range
- No key missingness

---

## 21. MCP Mock Server

The mock MCP server (`project/scripts/mock_mcp_server.py`) simulates an external risk intelligence service. It is started automatically by `launch_services.py` on port 8002.

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /watchlist/{account_id}` | Returns watchlist hit tier: `high`, `medium`, `low`, or `clean` |
| `GET /device/{device_id}` | Returns device risk tier and shared-account count |
| `GET /health` | Mock server health |

### API key auth

If `FRAUD_MCP_API_KEY` is set on the fraud API, the mock server validates `X-MCP-Api-Key` on every request and returns `401` on mismatch.

### Simulating MCP outage (circuit breaker test)

```bash
# 1. Stop the mock MCP server only (leave fraud + wallet running)
#    Ctrl+C the MCP process, or start services with --no-mcp

# 2. Send 5+ requests — each will fail MCP lookup, tripping the circuit
curl -X POST http://127.0.0.1:8000/score_transaction -H "Content-Type: application/json" \
  -d '{"schema_version":"ieee_fraud_tx_v1","user_id":"u1","transaction_amount":50,"device_risk_score":0.1,"ip_risk_score":0.1,"location_risk_score":0.1}'

# 3. Check circuit state in health endpoint
curl http://127.0.0.1:8000/health
# → mcp.mode: "degraded", mcp.breaker_open: true

# 4. Scoring continues normally — MCP external field returns 0.0 (neutral fallback)
```

---

## 22. Windows PowerShell Quickstart

Full startup sequence for Windows, from a fresh terminal:

```powershell
# 0. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 1. Verify dependencies
pip install -r requirements.txt

# 2. Pre-flight check (validates preset contracts end-to-end)
python project/scripts/preset_contract_check.py

# 3. Start all services (single-worker on Windows — multi-worker is unstable)
python project/scripts/launch_services.py

# In a new terminal tab:
# 4. Start frontend
cd project\frontend
npm install
Copy-Item .env.example .env
npm run dev

# In another terminal tab:
# 5. Health check
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health

# 6. Score a test transaction
$body = '{"schema_version":"ieee_fraud_tx_v1","user_id":"demo_user","transaction_amount":150.0,"device_risk_score":0.6,"ip_risk_score":0.7,"location_risk_score":0.3}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/score_transaction `
  -ContentType "application/json" -Body $body
```

Open `http://localhost:5173` in a browser to use the dashboard UI.

---

## 23. Troubleshooting

### `ModuleNotFoundError: No module named 'project'`

Run scripts as modules from the repo root, not as files:

```bash
# Wrong
python project/scripts/fairness_explainability_report.py

# Correct
python -m project.scripts.fairness_explainability_report --dataset-source ieee_cis ...
```

Or add the repo root to `PYTHONPATH`:
```bash
export PYTHONPATH=$(pwd)          # Linux/macOS
$env:PYTHONPATH = (Get-Location)  # PowerShell
```

### Services fail to start — port already in use

```bash
# Find what's on port 8000
netstat -ano | findstr :8000      # Windows
lsof -i :8000                     # Linux/macOS

# Kill it, then restart launcher
```

### Multi-worker mode crashes on Windows

Expected — Windows does not support uvicorn's multi-process forking reliably. The launcher automatically falls back to 1 worker. Add `--allow-windows-multi-worker` only for deliberate testing, not for demos.

### MCP circuit breaker stays open

The circuit resets after `FRAUD_MCP_CB_RESET_S` seconds (default 30). If the mock MCP server is back up, the next request after the reset window will probe it and close the circuit. Check `/health` for `breaker_open` and `opens_again_in_s`.

### Behavior profiles are cold (all users score as new)

Run the replay script to pre-warm profiles before starting the API:

```bash
python project/scripts/replay_behavior_profiles.py \
  --sqlite-path project/outputs/behavior_profiles.sqlite3
```

### Frontend shows no data / CORS errors

Confirm `.env` in `project/frontend/` points to the correct ports:
```
VITE_FRAUD_API_BASE_URL=http://127.0.0.1:8000
VITE_WALLET_API_BASE_URL=http://127.0.0.1:8001
```
Then restart `npm run dev`.

### `UnicodeEncodeError` in benchmark output (Windows terminal)

The benchmark compare script uses ASCII-only output (`->` not `→`). If you still see encoding errors, set your terminal to UTF-8:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

---

## 24. Demo Runbook (Judges)

Fastest path from zero to a live scored transaction:

```bash
# 1. Install
pip install -r requirements.txt

# 2. Start everything
python project/scripts/launch_services.py

# 3. Open UI
cd project/frontend && npm install && cp .env.example .env && npm run dev
# → http://localhost:5173

# 4. Score a high-risk transaction via API (should BLOCK)
curl -s -X POST http://127.0.0.1:8000/score_transaction \
  -H "Content-Type: application/json" \
  -d '{
    "schema_version": "ieee_fraud_tx_v1",
    "user_id": "account_00119",
    "transaction_amount": 4500,
    "device_risk_score": 0.9,
    "ip_risk_score": 0.85,
    "location_risk_score": 0.8,
    "tx_type": "CASH_OUT",
    "is_cross_border": true,
    "sim_change_recent": true
  }' | python -m json.tool

# 5. View the ring graph
curl http://127.0.0.1:8000/ring/graph | python -m json.tool
# Or visit http://localhost:5173/rings in the browser

# 6. Check health + MCP state
curl http://127.0.0.1:8000/health | python -m json.tool

# 7. Run the full governance evidence bundle
python project/scripts/build_evidence_bundle.py
```

Key things to show:
- `explainability.top_feature_drivers` in the score response — live SHAP per request
- `explainability.ring` — non-zero for account_00119 (known ring member)
- `/rings` page — D3 force graph with 19 rings, coloured by risk score
- `/dashboard` — drift PSI, benchmark latency, cohort KPIs
- `project/outputs/governance/ring_fairness_impact.md` — fairness impact of ring signals
- `project/outputs/governance/fairness_segment_thresholds_applied.json` — segment threshold mitigations in the checked-in evidence bundle

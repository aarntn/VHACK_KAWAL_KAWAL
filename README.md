# Vhack Fraud Shield (Backend/ML)

A real-time fraud detection stack for digital wallet authorization.

It combines:
- **Model score** (XGBoost)
- **Context score** (device/IP/location + transaction context)
- **Behavior score** (user profile history)

Outputs are operational decisions:
- `APPROVE`
- `FLAG` (step-up verification)
- `BLOCK`

---

## Key Docs (Submission)

- Architecture: `docs/architecture_calibration_release_gates.md`
- Results (consolidated): `docs/preliminary_readout.md`
- Demo instructions: see **Demo Runbook (Windows/PowerShell)** in this README

---


## Hackathon Case Study Context (Merged from `GOAL.txt`)

**Case Study:** Digital Trust – Real-Time Fraud Shield for the Unbanked  
**Track:** Machine Learning (Fraud & Anomaly Detection)  
**Primary Goal:** SDG 8 (Decent Work and Economic Growth, Target 8.10)

### Real-world context
Digital wallet adoption is growing rapidly across ASEAN, including among unbanked and low-digital-literacy users. A single successful fraud event can wipe out wallet balances for vulnerable users (e.g., gig workers and rural merchants), directly reducing trust in digital finance.

### Problem statement
Rule-based fraud controls alone are often insufficient for adaptive fraud behavior. This project targets real-time, low-latency anomaly scoring that can reduce fraud loss while minimizing friction for legitimate transactions.

### Technical focus
- Behavioral profiling (amount/frequency/location/time baselines)
- Real-time scoring and decisions (`APPROVE`, `FLAG`, `BLOCK`)
- Imbalanced-learning strategy for rare fraud labels
- Contextual signals (IP/device/location risk)
- Privacy-first operations and auditability

### Expected deliverables
- Fraud detection engine (high precision/recall objective)
- Risk API prototype integrated with a wallet authorization flow

Reference dataset challenge: IEEE-CIS Fraud Detection (Kaggle).

---

## Demo Runbook (Windows/PowerShell) (Merged from `HOW-TO-RUN.txt`)

### Prerequisites
- Python 3.11 virtual environment at `D:\Vhack\.venv`
- Node.js installed
- IEEE-CIS dataset at `D:\Vhack\ieee-fraud-detection\`
- Ports `8000` and `8001` are free

### 0) Install dependencies
```powershell
cd D:\Vhack
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1) Set environment variables
```powershell
$env:FRAUD_ENGINE_HEALTH_URL = "http://127.0.0.1:8000/health"
$env:FRAUD_ENGINE_URL = "http://127.0.0.1:8000/score_transaction"
```

### 2) Run preset contract check
```powershell
python project/scripts/preset_contract_check.py
```

### 3) Start backend services
```powershell
python project/scripts/launch_services.py --workload-profile cpu --fraud-workers 4 --wallet-workers 4 --upstream-timeout 2.0 --upstream-max-retries 2
```

### 4) Start frontend (new terminal)
```powershell
cd project/frontend
npm install
Copy-Item .env.example .env
npm run dev
```
Open the Vite URL shown in terminal (typically `http://localhost:5173`).

### 5) Quick health checks
```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health
```
If both return OK JSON, the frontend should be connected.

---

## 1) Architecture

```text
[Wallet Client / Demo UI]
            |
            v
+-----------------------------+
| wallet_gateway_api (8001)   |
| - /wallet/authorize_payment |
+-----------------------------+
            |
            v
+-----------------------------+
| hybrid_fraud_api (8000)     |
| - XGBoost inference         |
| - Context adjustments       |
| - Behavior profiler         |
| - Privacy-safe audit log    |
+-----------------------------+
     |            |            \
     v            v             v
model/*.pkl   profile store   outputs/audit/*.jsonl
```

---

## 2) Quick Start

### Install

```bash
python -m pip install -r requirements.txt
```

### Run APIs (repo root)

```bash
python -m uvicorn project.app.hybrid_fraud_api:app --port 8000 --workers 2 --loop uvloop --http httptools
FRAUD_ENGINE_URL=http://127.0.0.1:8000/score_transaction python -m uvicorn project.app.wallet_gateway_api:app --port 8001 --workers 2 --loop uvloop --http httptools
```

### `/score_transaction` payload contracts (public vs internal)

**Public API schema (frontend + external callers):**
- Send `transaction_amount` plus context fields (`device_risk_score`, `ip_risk_score`, `location_risk_score`, `device_id`, `device_shared_users_24h`, `account_age_days`, `sim_change_recent`, `tx_type`, `channel`, `cash_flow_velocity_1h`, `p2p_counterparties_24h`, `is_cross_border`).
- Request model: `ScoreTransactionRequest` in `project/app/schemas/api.py`.

Example public payload:

```json
{
  "schema_version": "ieee_fraud_tx_v1",
  "user_id": "frontend_demo_user_001",
  "transaction_amount": 72.5,
  "device_risk_score": 0.1,
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
  "is_cross_border": false
}
```

**Internal benchmark/model-serving contract (IEEE-style V-features):**
- The latency benchmark and internal model-serving validation use `FraudTransactionContract` with `TransactionDT`, `TransactionAmt`, and `V1..V17`.
- This internal contract is not the frontend-facing schema.

Internal payload example:

```json
{
  "schema_version": "ieee_fraud_tx_v1",
  "user_id": "user_safe",
  "TransactionDT": 67070.12,
  "TransactionAmt": 454.84,
  "V1": -0.04,
  "V2": 7.36,
  "V3": -7.83,
  "V4": 6.96,
  "V5": -5.42,
  "V6": -5.71,
  "V7": -5.19,
  "V8": 5.05,
  "V9": -5.85,
  "V10": 7.59,
  "V11": 7.31,
  "V12": -6.07,
  "V13": 5.29,
  "V14": 7.18,
  "V15": 1.97,
  "V16": -4.53,
  "V17": -1.31,
  "device_risk_score": 0.36,
  "ip_risk_score": 0.83,
  "location_risk_score": 0.68
}
```

Validation failures return HTTP `422` with machine-readable details:
- `error`: `ValidationError`
- `schema_version_expected`
- `detail`
- `details[]` with `{field, code, message, input}`

Optional wallet override:

```bash
FRAUD_ENGINE_URL=http://127.0.0.1:8000/score_transaction python -m uvicorn project.app.wallet_gateway_api:app --port 8001 --workers 2 --loop uvloop --http httptools
```

### Run tests

```bash
python -m unittest discover -s project/tests -t . -p "test_*.py"
```

---

## Alerting adapter (internal error hooks)

- Alert adapter interface: `project/app/alerts/notifier.py`
- Default runtime wiring: `get_alert_notifier()` returns `NoOpAlertNotifier` (safe in production; never throws)
- Global FastAPI exception handlers in:
  - `project/app/hybrid_fraud_api.py`
  - `project/app/wallet_gateway_api.py`
  call notifier hooks for unhandled/internal failures and high-severity artifact schema mismatches.
- Future Slack/PagerDuty integration should be implemented as a concrete `AlertNotifier` in
  `project/app/alerts/notifier.py` and wired via `get_alert_notifier()` (or startup dependency injection).

---


## Dataset defaults (IEEE-CIS first)

- The project now defaults to **IEEE-CIS** (`dataset_source=ieee_cis`) for training/evaluation scripts.
- Expected default local dataset location: `D:\Vhack\ieee-fraud-detection` (for `train_transaction.csv` and `train_identity.csv`).
- Legacy credit-card assets are archived under `project/legacy_creditcard/` (models + monitoring outputs + optional legacy CSV path).
- Legacy runs are still supported via `--dataset-source creditcard --dataset-path project/legacy_creditcard/creditcard.csv`, but they are no longer the default for training or evaluation.

## 3) Core Model + Decisioning

- Model: **XGBoost classifier**
- Features: `TransactionDT`, `TransactionAmt`, identity/device/address signals + contextual risk features
- Hybrid scoring: base model + context + behavior

Decision thresholds (default policy shape):

| Score Range   | Decision |
| --- | --- |
| `<= approve_threshold` | `APPROVE` |
| `approve_threshold .. block_threshold` | `FLAG` |
| `>= block_threshold` | `BLOCK` |

Active thresholds are loaded from `project/models/decision_thresholds.pkl`.

---

## 4) Important Scripts

### Class imbalance experiments

```bash
python project/models/class_imbalance_experiments.py --dataset-path project/legacy_creditcard/creditcard.csv --seed 42
```

Outputs:
- `project/outputs/figures/tables/class_imbalance_experiment_metrics.csv`
- `project/outputs/figures/tables/class_imbalance_experiment_metrics.md`

### Time-aware imbalance strategy benchmark (Pass 1)

```bash
python project/scripts/benchmark_imbalance_strategies.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /path/to/train_transaction.csv \
  --ieee-identity-path /path/to/train_identity.csv \
  --split-mode time \
  --seed 42
```

Outputs:
- `project/outputs/monitoring/<dataset>_imbalance_strategy_benchmark.csv`
- `project/outputs/monitoring/<dataset>_imbalance_strategy_benchmark_metadata.json`

#### IEEE-CIS quickstart (explicit Kaggle paths)

The IEEE-CIS dataset is not bundled in this repository. Download from Kaggle and pass paths explicitly.

```bash
python project/scripts/benchmark_imbalance_strategies.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /data/ieee/train_transaction.csv \
  --ieee-identity-path /data/ieee/train_identity.csv \
  --split-mode time \
  --seed 42

python project/scripts/tune_model_candidates.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /data/ieee/train_transaction.csv \
  --ieee-identity-path /data/ieee/train_identity.csv \
  --split-mode groupkfold_uid \
  --groupkfold-n-splits 5 \
  --optuna-trials 20 \
  --seed 42

python project/scripts/evaluate_preprocessing_settings.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /data/ieee/train_transaction.csv \
  --ieee-identity-path /data/ieee/train_identity.csv \
  --split-mode time \
  --threshold-start 0.005 \
  --threshold-stop 0.15 \
  --threshold-step 0.005
```

### Model candidate tuning + comparison (Pass 2)

```bash
python project/scripts/tune_model_candidates.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /path/to/train_transaction.csv \
  --ieee-identity-path /path/to/train_identity.csv \
  --split-mode time \
  --optuna-trials 20 \
  --seed 42
```

Outputs:
- `project/outputs/monitoring/<dataset>_model_candidate_benchmark.csv`
- `project/outputs/monitoring/<dataset>_model_candidate_benchmark_report.json`
- `project/outputs/monitoring/<dataset>_validation_windows.csv`
- `project/outputs/monitoring/<dataset>_validation_robustness_report.json`

Notes:
- Includes `xgboost_default`, `xgboost_tuned`, and `logistic_regression` by default.
- Adds `lightgbm`/`catboost` only if those libraries are installed.
- Falls back to default XGBoost config if `optuna` is not installed and records this in report notes.
- Robustness report aggregates repeated temporal holdouts (`train N, gap G, validate H`) + month-group folds and records mean/std/worst-window metrics.

### Ensemble candidate benchmark with time-aware OOF (Pass 7)

```bash
python project/scripts/benchmark_ensemble_candidates.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /path/to/train_transaction.csv \
  --ieee-identity-path /path/to/train_identity.csv \
  --n-splits 5 \
  --seed 42
```

Outputs:
- `project/outputs/monitoring/<dataset>_ensemble_benchmark.csv`
- `project/outputs/monitoring/<dataset>_ensemble_benchmark_report.json`

Notes:
- Trains independent base models (`xgboost`, optional `lightgbm`, optional `catboost`) on time-aware OOF folds.
- Compares single models, equal-weight blend, weighted blend, and logistic stacker.
- Report includes champion, reproducible split windows, seed/folds, and feature-set hash.

### Entity identity + score aggregation benchmark (Pass 8)

```bash
python project/scripts/benchmark_entity_aggregation.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /path/to/train_transaction.csv \
  --ieee-identity-path /path/to/train_identity.csv \
  --threshold 0.5
```

Outputs:
- `project/outputs/monitoring/entity_aggregation_report.json`

Notes:
- Builds stable `entity_id` using reusable tiered identity logic in `project/data/entity_identity.py`.
- Applies transaction-score smoothing by entity using mean, EMA, and capped blend.
- Logs pre/post metrics (`precision`, `recall`, `f1`, `false_positive_rate`, `pr_auc`, `roc_auc`) and champion method.

### Time-consistency feature filter (Pass 9)

```bash
python project/scripts/time_consistency_feature_filter.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /path/to/train_transaction.csv \
  --ieee-identity-path /path/to/train_identity.csv \
  --feature-groups-file /path/to/feature_groups.json
```

Outputs:
- `project/outputs/monitoring/time_consistency_feature_scores.csv`
- `project/outputs/monitoring/time_consistency_feature_decisions.json`

Notes:
- Evaluates each single feature and optional feature group using early-period train vs later-period validation windows.
- Records `train_auc`, `validation_auc`, `auc_delta`, inversion flags, and keep/drop decisions with configured thresholds.
- `project/models/final_xgboost_model.py` supports `--feature-whitelist-file` to train only on stable features from this pass.

### Context recalibration (Pass 4)

```bash
python project/scripts/calibrate_context.py \
  --dataset-path project/legacy_creditcard/creditcard.csv \
  --model-path project/models/final_xgboost_model_promoted_preproc.pkl \
  --feature-path project/models/feature_columns_promoted_preproc.pkl \
  --baseline-thresholds-path project/models/decision_thresholds_promoted_preproc.pkl \
  --trials 350 \
  --seed 42
```

Outputs:
- `project/outputs/calibration/context_calibration.json`
- `project/outputs/calibration/context_calibration_trials.csv`
- `project/models/decision_thresholds.pkl` (or your custom `--thresholds-output`)

Pass-4 report highlights:
- `pre_calibration` and `post_calibration` metrics/objectives.
- `decision_drift_pre_post` to compare decision mix shift after recalibration.
- `non_regression_check` and automatic fallback to baseline runtime config if objective/F1/recall regress.
- Threshold ordering validation (`approve_threshold < block_threshold`) enforced before write.

### Fairness + explainability governance (Pass 5)

```bash
python project/scripts/fairness_explainability_report.py \
  --dataset-path project/legacy_creditcard/creditcard.csv \
  --model-path project/models/final_xgboost_model_promoted_preproc.pkl \
  --feature-path project/models/feature_columns_promoted_preproc.pkl \
  --thresholds-path project/models/decision_thresholds_promoted_preproc.pkl \
  --max-fpr-gap 0.08 \
  --max-recall-gap 0.12 \
  --max-precision-gap 0.12
```

If you hit `ModuleNotFoundError: No module named project` on Windows, run as a module:

```powershell
python -m project.scripts.fairness_explainability_report --dataset-source creditcard --dataset-path project/legacy_creditcard/creditcard.csv
```

Outputs:
- `project/outputs/governance/fairness_explainability_report.json`
- `project/outputs/governance/fairness_explainability_report.md`
- `project/outputs/governance/fairness_segment_metrics.csv`
- `project/outputs/governance/model_top_feature_drivers.csv`

Pass-5 checks:
- Segment metrics (`precision`, `recall`, `false_positive_rate`, `false_negative_rate`) across cohort/device/region slices.
- Per-segment deltas against overall baseline include `fpr_gap_vs_overall`, `fnr_gap_vs_overall`, `precision_gap_vs_overall`.
- Stable severity labels (`ok`, `warning`, `severe`, `low_support`) are attached to segment rows, and severe segments are ranked deterministically by severity score.
- Identity-bucket diagnostics under `identity_bucket_metrics` for `known_high_confidence_entity_id`, `uncertain_weakly_linked_entity_id`, and `unknown_no_entity` with PR-AUC/F1/precision/recall/FPR and gap summaries.
- Severe disparity detection based on configured FPR/FNR/precision gap policy thresholds.
- SHAP-style top feature drivers for the candidate model (SHAP explainer preferred; XGBoost pred-contrib fallback).

Companion governance decision step:

```bash
python project/scripts/fairness_segment_decision.py \
  --segment-metrics-csv project/outputs/governance/fairness_segment_metrics.csv \
  --max-fpr-gap 0.08 \
  --max-fnr-gap 0.12 \
  --max-precision-gap 0.12
```

Outputs:
- `project/outputs/governance/fairness_action_plan.json`
- `project/outputs/governance/fairness_action_plan.md`

Governance action criteria (release decision):
- **Block release** when any non-low-support segment is labeled `severe` (gap beyond policy threshold) and no approved compensating control exists.
- **Permit with waiver** only when severe segments have documented mitigation owner + deadline, and extra monitoring alerts are enabled.
- **Permit** when there are no severe segments (warnings can proceed with monitoring and follow-up work).

### Latency benchmark

Start non-reload service instances before benchmarking (recommended):

```bash
python project/scripts/launch_services.py --fraud-workers 2 --wallet-workers 2
```

Then run benchmark traffic against those running instances:

```bash
python project/scripts/benchmark_latency.py --requests 200 --concurrency 20
```

Judge mode quick validation (single command, sweep 2/6/12 with strict gate):

```bash
python project/scripts/benchmark_latency.py --requests 200 --concurrency-sweep 2,6,12 --sla-p95-ms 250 --sla-p99-ms 500 --sla-error-rate-pct 1.0
```

Expected pass thresholds:
- `score_transaction` and `wallet_authorize_payment` both require:
  - `p95 <= 250 ms`
  - `p99 <= 500 ms`
  - `error_rate <= 1.0%`

Outputs:
- `project/outputs/benchmark/latency_benchmark_*.json`
- `project/outputs/benchmark/latency_benchmark_*.csv`

### Latency trend report

```bash
python project/scripts/benchmark_trend_report.py \
  --benchmark-dir project/outputs/benchmark \
  --output-json project/outputs/monitoring/latency_trend_report.json \
  --output-csv project/outputs/monitoring/latency_trend_report.csv
```

### Cohort KPI report

```bash
python project/scripts/cohort_kpi_report.py \
  --dataset-path project/legacy_creditcard/creditcard.csv \
  --output-json project/outputs/monitoring/cohort_kpi_report.json \
  --output-csv project/outputs/monitoring/cohort_kpi_report.csv
```

Cohorts include:
- `new_users`
- `rural_merchants_proxy`
- `gig_workers_proxy`
- `low_history_proxy`

### Replay behavior profiles (seed profile store)

```bash
python project/scripts/replay_behavior_profiles.py \
  --dataset-path project/legacy_creditcard/creditcard.csv \
  --sqlite-path project/outputs/behavior_profiles.sqlite3 \
  --output-json project/outputs/monitoring/profile_replay_summary.json
```

### Behavior profile health check

```bash
python project/scripts/behavior_profile_health.py \
  --store-backend sqlite \
  --sqlite-path project/outputs/behavior_profiles.sqlite3 \
  --output-json project/outputs/monitoring/behavior_profile_health.json
```

### Drift monitoring

```bash
python project/scripts/drift_monitor.py \
  --baseline-dataset project/legacy_creditcard/creditcard.csv \
  --audit-log project/outputs/audit/fraud_audit_log.jsonl \
  --output-json project/outputs/monitoring/drift_report.json \
  --output-csv project/outputs/monitoring/drift_feature_psi.csv
```

---

## 5) Nightly Ops (single command)

```bash
python project/scripts/nightly_ops.py \
  --dataset-path project/legacy_creditcard/creditcard.csv \
  --audit-log project/outputs/audit/fraud_audit_log.jsonl \
  --run-benchmark \
  --benchmark-sla-mode warn \
  --run-cohort-kpi \
  --run-profile-replay \
  --run-profile-health \
  --archive-runs-dir project/outputs/ops_runs \
  --ops-summary-json project/outputs/monitoring/nightly_ops_summary.json
```

Flow:
1. Data checks (dataset/label validity)
2. Scoring drift monitor (feature PSI + decision drift)
3. Threshold-policy guardrails (threshold ordering + promotion policy pass)
4. Optional recalibration + threshold promotion
5. Benchmark + latency trend generation
6. Latency stage analysis artifact generation
7. Cohort KPI generation
8. Behavior-profile replay + health check
9. Retrain trigger evaluation + summary + alert metadata
10. Per-run artifact archive with checksum manifest under `project/outputs/ops_runs/<run_id>/`

Helper wrappers:
- Linux/macOS: `project/scripts/run_nightly_ops.sh`
- Windows PowerShell: `project/scripts/run_nightly_ops.ps1`

---

## 6) CI/CD and Evidence

### Release gate

```bash
python project/scripts/release_gate_check.py --ops-summary-json project/outputs/monitoring/nightly_ops_summary.json
```

### Build evidence bundle

```bash
python project/scripts/build_evidence_bundle.py
```

Bundle includes latest monitoring artifacts (drift, calibration, benchmark, trend, cohort KPI, profile replay/health).

### Privacy governance and analyst review controls

See `docs/privacy_governance.md` for:
- exact `project/outputs/audit/*.jsonl` field contracts and hash/token handling,
- retention windows (hot vs archive) plus deletion workflow/SLA,
- RBAC expectations for `/review_queue`, `/review_queue/{request_id}/outcome`, and `/retraining/curation`,
- integrity controls and incident-runbook mapping for audit + retraining artifacts.

---

## 7) Dataset

Primary dataset expected by scripts:

```text
project/legacy_creditcard/creditcard.csv
```

Reference:
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 8) Stage 1: IEEE onboarding

Use the Stage-1 artifact builder to validate ingestion only (no training) and produce reproducible onboarding outputs under `project/outputs/dataset_stage1/`.

### IEEE-CIS source

```bash
python project/scripts/build_dataset_stage1_artifacts.py   --dataset-source ieee_cis   --transaction-path /path/to/train_transaction.csv   --identity-path /path/to/train_identity.csv
```

Expected outputs:
- `creditcard_head.csv` / `ieee_cis_head.csv`
- `creditcard_schema.json` / `ieee_cis_schema.json`
- `creditcard_quality.json` / `ieee_cis_quality.json`
- `creditcard_canonical_preview.csv` / `ieee_cis_canonical_preview.csv`

Acceptance checks performed by the script:
- Target column exists and is non-empty
- Target labels are binary (0/1)
- IEEE-CIS `TransactionID` is unique in the transaction table

### Preprocessing setting comparison + threshold tuning

Run a leakage-aware comparison of `onehot + robust` vs `frequency + robust` with threshold sweeps and metric outputs (F1, recall, precision, FPR, PR-AUC, ROC-AUC, confusion matrix components).

```bash
python project/scripts/evaluate_preprocessing_settings.py   --dataset-source ieee_cis   --ieee-transaction-path /path/to/train_transaction.csv   --ieee-identity-path /path/to/train_identity.csv   --split-mode time   --test-size 0.2
```

Outputs under `project/outputs/monitoring/`:
- `ieee_cis_preprocessing_threshold_comparison.csv` (all thresholds)
- `ieee_cis_preprocessing_threshold_best.csv` (best F1 per setting)
- `ieee_cis_preprocessing_threshold_summary.md` (markdown table)
- `ieee_cis_preprocessing_threshold_metadata.json`

#### Promote best preprocessing setting to a trained artifact bundle

```bash
python project/scripts/promote_best_preprocessing_setting.py \
  --dataset-source ieee_cis \
  --ieee-transaction-path /path/to/train_transaction.csv \
  --ieee-identity-path /path/to/train_identity.csv \
  --validation-robustness-report project/outputs/monitoring/ieee_cis_validation_robustness_report.json
```

This selects a winner using `--selection-scope best|full` and `--selection-metric` (`f1`, `recall`, `precision`), supports policy filters (`--max-fpr`, `--min-precision`), adds quality guardrails (`--min-f1`, `--min-pr-auc`, `--min-recall`), requires a validation robustness report with `robustness_gate.passed=true`, writes a promotion report JSON including `quality_gate` pass/fail details, and trains `final_xgboost_model.py` with explicit preprocessing outputs. Use `--dry-run` to inspect without training.

If policy constraints are too strict and remove all rows, promotion now writes `status=blocked_selection_policy` plus remediation steps to the promotion report. Use `--allow-policy-fallback` only with strict quality gates (`--min-f1`, `--min-pr-auc`, `--min-recall`).

---

PowerShell example:

```powershell
python project/scripts/cohort_kpi_report.py `
  --dataset-path project/legacy_creditcard/creditcard.csv `
  --sample-size 2000 `
  --output-json project\outputs\monitoring\cohort_kpi_report.json `
  --output-csv project\outputs\monitoring\cohort_kpi_report.csv
```

---

## 10) Project Paths

- Fraud API: `project/app/hybrid_fraud_api.py`
- Wallet API: `project/app/wallet_gateway_api.py`
- Scripts: `project/scripts/`
- Models: `project/models/`
- Tests: `project/tests/`
- Monitoring outputs: `project/outputs/monitoring/`

## Runtime artifact policy

- Runtime-generated artifacts (for example `.pyc` and `__pycache__/`) must not be committed.
- Virtual environments (for example `.venv/`) must not be committed.
- Exception policy: if an artifact must be versioned for a specific reproducibility or compliance reason, document the rationale in this README and store it outside source trees under a dedicated top-level artifact directory.
- Current status: no runtime artifact exceptions are approved.

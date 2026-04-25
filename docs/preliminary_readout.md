# Preliminary Readout (Consolidated)

_Last updated: 2026-03-19 (UTC)_

This document is the single consolidated readout for submission review, combining architecture context, latest benchmark metrics, and current known limitations.

## 1) Architecture diagram

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
| - Audit logging             |
+-----------------------------+
     |            |            \
     v            v             v
model/*.pkl   profile store   outputs/audit/*.jsonl
```

## 2) Final metrics table

Latency results below reflect the latest tuple benchmark rerun comparison snapshot (baseline vs optimized run) for target tuples `(80,2)`, `(120,6)`, `(200,12)`.

| Endpoint | Requests | Concurrency | p95 before (ms) | p95 after (ms) | Δ p95 (ms) | p99 before (ms) | p99 after (ms) | Δ p99 (ms) | Error-rate status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `score_transaction` | 80 | 2 | 333.13 | 103.84 | -229.29 | 572.37 | 125.62 | -446.75 | 0.0% (pass) |
| `score_transaction` | 120 | 6 | 2209.38 | 363.70 | -1845.68 | 2508.72 | 416.89 | -2091.83 | 2.5% baseline breach |
| `score_transaction` | 200 | 12 | 2514.96 | 692.08 | -1822.88 | 2554.77 | 805.83 | -1748.94 | 22.5% baseline breach |
| `wallet_authorize_payment` | 80 | 2 | 390.76 | 94.38 | -296.38 | 449.58 | 136.99 | -312.59 | 0.0% (pass) |
| `wallet_authorize_payment` | 120 | 6 | 1732.64 | 368.90 | -1363.74 | 1761.62 | 397.49 | -1364.13 | 0.0% |
| `wallet_authorize_payment` | 200 | 12 | 1696.09 | 768.14 | -927.95 | 1745.42 | 846.21 | -899.21 | 0.0% |

Gate target: `p95 <= 250 ms`, `p99 <= 500 ms`, `error_rate <= 1.0%`.

Current gate outcome:
- `(80,2)`: PASS (both endpoints)
- `(120,6)`: FAIL on p95 for both endpoints
- `(200,12)`: FAIL on p95/p99 for both endpoints

## 3) Known limits

1. **Tail latency under higher concurrency remains above SLA** (`120/6` and `200/12` tuples fail gate).
2. **Preprocessing stage dominates tail latency** in stage analysis and remains the main optimization target.
3. **Nightly full-pipeline reproducibility depends on local IEEE dataset availability** (`train_transaction.csv`, `train_identity.csv`).
4. **Production-ready status remains blocked** until benchmark gates and nightly gating paths pass end-to-end.

## 4) Archived detailed experiment logs

Older dated documentation has been moved to `docs/archive/` for traceability while keeping this readout as the single submission-facing summary.

## Threshold recalibration validation (2026-04-19)

- Validation artifact: `project/outputs/monitoring/threshold_recalibration_validation_report.md`

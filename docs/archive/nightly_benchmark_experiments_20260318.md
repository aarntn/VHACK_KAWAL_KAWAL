# Nightly benchmark concurrency experiments (2026-03-18)

## Command baseline used

- Kept the nightly benchmark command shape and varied only `--benchmark-concurrency` with fixed `--benchmark-requests 200`.
- Concurrency values tested: `4`, `8`, `12`.

## Endpoint p95/p99 results

| concurrency | score_transaction p95 (ms) | score_transaction p99 (ms) | wallet_authorize_payment p95 (ms) | wallet_authorize_payment p99 (ms) | score SLA | wallet SLA |
|---:|---:|---:|---:|---:|:---:|:---:|
| 4 | 1832.67 | 1975.45 | 1738.09 | 2016.29 | FAIL | FAIL |
| 8 | 5873.77 | 7222.43 | 1916.15 | 2200.78 | FAIL | FAIL |
| 12 | 6502.49 | 7038.49 | 1789.74 | 1880.24 | FAIL | FAIL |

## Stage correlation (latency_stage_analysis.json)

- Stage analysis records analyzed: **3290**.
- Dominant stage by tail latency: **preprocessing_ms** with p95=1885.37 ms and p99=5555.88 ms.
- This aligns with benchmark endpoint-level stage breakdowns where preprocessing-related stages dominate tail latency under load.

## Optimization priority (preprocessing_ms first)

1. **Reduce preprocessing fan-out and per-request transforms** (cache stable encodings/feature maps per process).
2. **Avoid duplicate preprocessing in wallet path** (wallet adds gateway preprocessing + upstream fraud preprocessing).
3. **Move non-critical audit-path work off request path** where possible; current `audit_log_write_ms` competes for tail budget.
4. **Re-benchmark after each optimization at concurrency 4/8/12 with fixed 200 requests** to isolate impact.

## SLA enforce rerun status

- A strict nightly enforce rerun with the IEEE command failed before benchmark execution because IEEE dataset files are not present in this environment (`ieee-fraud-detection/train_transaction.csv`, `ieee-fraud-detection/train_identity.csv`).
- Direct benchmark enforce runs on the active services still show SLA failures at tested concurrencies, so p95/p99 caps are **not yet passing** for either endpoint.


## Target-load tuple rerun (identical SLA args)

Command pattern used for every tuple (same SLA args each run):

```bash
python project/scripts/benchmark_latency.py \
  --requests <REQ> \
  --concurrency <CONC> \
  --sla-p95-ms 250 \
  --sla-p99-ms 500 \
  --sla-error-rate-pct 1.0
```

Artifacts captured under `project/outputs/benchmarks/20260318_tuples/`.

### Endpoint metrics by tuple

| requests | concurrency | score_transaction p95 (ms) | score_transaction p99 (ms) | score_transaction error rate (%) | wallet_authorize_payment p95 (ms) | wallet_authorize_payment p99 (ms) | wallet_authorize_payment error rate (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 2 | 333.13 | 572.37 | 0.0 | 390.76 | 449.58 | 0.0 |
| 120 | 6 | 2209.38 | 2508.72 | 2.5 | 1732.64 | 1761.62 | 0.0 |
| 200 | 12 | 2514.96 | 2554.77 | 22.5 | 1696.09 | 1745.42 | 0.0 |

## Pass gate definition and status

Gate definition at target load:

- **PASS only if both endpoints meet all of:**
  - p95 <= 250 ms
  - p99 <= 500 ms

Result from the tuple rerun:

- **Gate status: FAIL**.
- Both endpoints miss p95/p99 at all three tuples; `score_transaction` also breaches error-rate SLA at `(120,6)` and `(200,12)`.

## Step tracking + escalation rule

Optimization steps 1–4 remain the active sequence:

1. Reduce preprocessing fan-out/per-request transforms.
2. Remove duplicate preprocessing in wallet path.
3. Shift non-critical audit-path work off request path.
4. Re-benchmark after each optimization using fixed SLA args.

Escalation rule:

- If still failing after completion of steps 1–4, open separate effort for **ONNX / feature-store / offline feature materialization**.
- Given current rerun still fails the gate, escalation planning should be prepared in parallel and executed immediately once step-4 iteration is confirmed complete.

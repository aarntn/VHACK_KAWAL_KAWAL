# Backend Go/No-Go Report (Pass 5)

Decision: **GO**

## Pass/Fail Criteria
- Consecutive nightly ops runs with status `ok`: required 3.
- Endpoint error rate <= 5.0%.
- Endpoint p95 <= 7000 ms.
- Endpoint p99 <= 9000 ms.

## Run Results
### Run 1 (ok)
- Summary: `project/outputs/monitoring/nightly_ops_summary_pass4_run1.json`
- Benchmark JSON: `/workspace/Vhack/project/outputs/benchmark/latency_benchmark_20260318T165036Z.json`
  - `score_transaction`: error_rate=0.00%, p95=750.42 ms, p99=761.15 ms
    - Top stage contributors: feature_preparation_ms (315.22 ms, 93.2%); preprocessing_ms (315.22 ms, 93.2%); audit_log_write_ms (72.61 ms, 24.8%)
  - `wallet_authorize_payment`: error_rate=0.00%, p95=548.46 ms, p99=557.19 ms
    - Top stage contributors: upstream_call_ms (383.64 ms, 99.1%); gateway_preprocessing_ms (0.70 ms, 0.2%); response_mapping_ms (0.00 ms, 0.0%)
- Retrain trigger: should_retrain=True, reasons=['feature_alert_count=2', 'decision_drift_status=alert', 'benchmark_sla_fail_streak=3']
### Run 2 (ok)
- Summary: `project/outputs/monitoring/nightly_ops_summary_pass4_run2.json`
- Benchmark JSON: `/workspace/Vhack/project/outputs/benchmark/latency_benchmark_20260318T165106Z.json`
  - `score_transaction`: error_rate=0.00%, p95=1682.79 ms, p99=1743.15 ms
    - Top stage contributors: feature_preparation_ms (783.44 ms, 97.6%); preprocessing_ms (783.44 ms, 97.6%); audit_log_write_ms (175.63 ms, 23.5%)
  - `wallet_authorize_payment`: error_rate=0.00%, p95=4327.68 ms, p99=4885.20 ms
    - Top stage contributors: upstream_call_ms (1742.10 ms, 100.0%); gateway_preprocessing_ms (0.03 ms, 0.0%); response_mapping_ms (0.00 ms, 0.0%)
- Retrain trigger: should_retrain=True, reasons=['feature_alert_count=2', 'decision_drift_status=alert', 'benchmark_sla_fail_streak=3']
### Run 3 (ok)
- Summary: `project/outputs/monitoring/nightly_ops_summary_pass4_run3.json`
- Benchmark JSON: `/workspace/Vhack/project/outputs/benchmark/latency_benchmark_20260318T165125Z.json`
  - `score_transaction`: error_rate=0.00%, p95=378.79 ms, p99=396.62 ms
    - Top stage contributors: feature_preparation_ms (189.23 ms, 90.2%); preprocessing_ms (189.23 ms, 90.2%); audit_log_write_ms (44.25 ms, 22.3%)
  - `wallet_authorize_payment`: error_rate=0.00%, p95=811.82 ms, p99=813.34 ms
    - Top stage contributors: upstream_call_ms (545.86 ms, 99.9%); gateway_preprocessing_ms (0.07 ms, 0.0%); response_mapping_ms (0.00 ms, 0.0%)
- Retrain trigger: should_retrain=True, reasons=['feature_alert_count=2', 'decision_drift_status=alert', 'benchmark_sla_fail_streak=3']

## Singleton Artifact Checks
- Fraud artifact_init_count: 1 (expected 1)
- Wallet runtime_init_count: 1 (expected 1)

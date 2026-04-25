# Tuple benchmark rerun on chosen promoted model/threshold (2026-03-19)

- Model artifact: `project/models/final_xgboost_model_promoted_preproc.pkl`.
- Threshold artifact: `project/models/decision_thresholds_promoted_preproc.pkl`.
- Thresholds: approve=0.332260, block=0.550426.
- SLA target: p95 <= 250 ms, p99 <= 500 ms, error rate <= 1.0%.

## Endpoint latency/error summary

| endpoint | requests | concurrency | p95 (ms) | p99 (ms) | error rate (%) | SLA verdict |
|---|---:|---:|---:|---:|---:|---|
| score_transaction | 80 | 2 | 287.15 | 442.64 | 0.00 | FAIL |
| wallet_authorize_payment | 80 | 2 | 631.06 | 1014.71 | 0.00 | FAIL |
| score_transaction | 120 | 6 | 927.20 | 1085.31 | 0.00 | FAIL |
| wallet_authorize_payment | 120 | 6 | 1326.17 | 1679.96 | 0.00 | FAIL |
| score_transaction | 200 | 12 | 2360.63 | 2517.78 | 3.00 | FAIL |
| wallet_authorize_payment | 200 | 12 | 1679.26 | 1732.55 | 0.00 | FAIL |

## Stage contribution breakdown (top 3 by mean share of total)

| endpoint | requests | concurrency | stage | mean share of total (%) | mean stage latency (ms) |
|---|---:|---:|---|---:|---:|
| score_transaction | 80 | 2 | feature_preparation_ms | 93.99 | 116.75 |
| score_transaction | 80 | 2 | preprocessing_ms | 93.99 | 116.75 |
| score_transaction | 80 | 2 | model_inference_ms | 4.18 | 5.43 |
| wallet_authorize_payment | 80 | 2 | internal_fraud_scoring_call_ms | 99.56 | 217.99 |
| wallet_authorize_payment | 80 | 2 | wallet_minus_fraud_core_ms | 36.25 | 79.78 |
| wallet_authorize_payment | 80 | 2 | wallet_preprocessing_ms | 0.09 | 0.16 |
| score_transaction | 120 | 6 | feature_preparation_ms | 95.54 | 360.28 |
| score_transaction | 120 | 6 | preprocessing_ms | 95.54 | 360.28 |
| score_transaction | 120 | 6 | model_inference_ms | 3.58 | 12.47 |
| wallet_authorize_payment | 120 | 6 | internal_fraud_scoring_call_ms | 99.84 | 492.66 |
| wallet_authorize_payment | 120 | 6 | wallet_minus_fraud_core_ms | 31.20 | 169.36 |
| wallet_authorize_payment | 120 | 6 | wallet_preprocessing_ms | 0.05 | 0.14 |
| score_transaction | 200 | 12 | feature_preparation_ms | 95.90 | 660.74 |
| score_transaction | 200 | 12 | preprocessing_ms | 95.90 | 660.74 |
| score_transaction | 200 | 12 | model_inference_ms | 3.51 | 20.57 |
| wallet_authorize_payment | 200 | 12 | internal_fraud_scoring_call_ms | 48.42 | 434.09 |
| wallet_authorize_payment | 200 | 12 | wallet_minus_fraud_core_ms | 33.95 | 294.99 |
| wallet_authorize_payment | 200 | 12 | wallet_preprocessing_ms | 12.62 | 0.11 |

## Honest pitch impact at high load

- `(80,2)`: score path slightly above p95 target and wallet path above both p95/p99 targets.
- `(120,6)`: both endpoints exceed p95/p99 materially; error remains low.
- `(200,12)`: score endpoint breaches p95/p99 heavily and now also breaches error-rate SLA (3.0%).
- Net: this stack is **not** production-ready for high concurrency without additional latency/error hardening.

## Optimization roadmap

1. **Preprocessing hot path reduction** (highest impact): feature preparation + preprocessing contributes ~94-96% of score latency; prioritize vectorization/caching and removing duplicate transforms.
2. **Wallet dependency isolation**: internal fraud call dominates wallet time; add bounded queueing + adaptive concurrency limits to cap tail amplification.
3. **Model inference/runtime tuning**: increase inference threads with controlled worker counts and profile GIL/serialization overhead.
4. **Backpressure + shed strategy** at `(>=12)` concurrency: early reject/degrade instead of timeout-driven retries.
5. **Repeat tuple gate after each optimization** using these same tuples plus soak duration mode (`--duration-seconds`) to verify stability.

# Error Taxonomy + Next-Run Actions (2026-03-19)

Source artifacts:
- `project/outputs/governance/fairness_segment_metrics.csv`
- `project/outputs/governance/model_top_feature_drivers.csv`
- `project/outputs/governance/fairness_explainability_report.md`

## 1) FP/FN slice by segment (with estimated counts)

Using:
- `FP_est = false_positive_rate * (sample_count - fraud_positive_count)`
- `FN_est = false_negative_rate * fraud_positive_count`

Rounded to nearest integer for triage.

### Highest false-positive load

| Segment | Samples | Fraud+ | FPR | FP_est | FPR gap vs overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| ieee:identity_medium_confidence | 24,905 | 877 | 0.126 | 3,027 | -0.0016 |
| cohort:established_users | 23,748 | 854 | 0.128 | 2,934 | +0.0006 |
| device:mobile_app | 20,029 | 719 | 0.126 | 2,437 | -0.0014 |
| ieee:product_C | 2,974 | 347 | 0.662 | 1,740 | +0.5348 |
| ieee:device_desktop | 3,707 | 264 | 0.446 | 1,536 | +0.3186 |
| ieee:device_mobile | 2,344 | 240 | 0.538 | 1,131 | +0.4100 |

### Highest false-negative load

| Segment | Samples | Fraud+ | FNR | FN_est | FNR gap vs overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| ieee:identity_medium_confidence | 24,905 | 877 | 0.491 | 431 | +0.0026 |
| cohort:established_users | 23,748 | 854 | 0.489 | 418 | +0.0006 |
| device:mobile_app | 20,029 | 719 | 0.497 | 357 | +0.0077 |
| ieee:product_W | 18,510 | 373 | 0.879 | 328 | +0.3905 |
| ieee:identity_high_confidence | 18,699 | 480 | 0.635 | 305 | +0.1466 |

### Top global feature drivers

| Rank | Feature | Mean |SHAP| |
| ---: | --- | ---: |
| 1 | numeric_canonical__device_signal_present | 0.608 |
| 2 | numeric_canonical__amount_raw | 0.358 |
| 3 | numeric_canonical__time_since_last_tx | 0.237 |
| 4 | numeric_canonical__event_time_raw | 0.192 |
| 5 | numeric_canonical__amount_over_user_avg | 0.135 |
| 6 | numeric_canonical__hour_of_day | 0.125 |
| 7 | numeric_canonical__avg_amount_24h | 0.096 |
| 8 | numeric_canonical__amount_log | 0.084 |
| 9 | numeric_canonical__tx_count_24h | 0.071 |

Interpretation: the model is dominated by device presence + amount/velocity/time features, so segment-level errors are likely mediated by how these features are distributed across product/device cohorts.

## 2) Short error taxonomy

### A) High-risk false positives

Definition: segments with very high FPR and material volume.

- `ieee:product_C` (FPR 0.662, FP_est 1,740)
- `ieee:device_mobile` (FPR 0.538, FP_est 1,131)
- `ieee:device_desktop` (FPR 0.446, FP_est 1,536)

Risk: excessive customer friction and manual-review load despite only moderate precision gains.

### B) Missed fraud (false negatives)

Definition: segments with very high FNR and meaningful fraud-positive counts.

- `ieee:product_W` (FNR 0.879, FN_est 328)
- `ieee:identity_high_confidence` (FNR 0.635, FN_est 305)
- plus high-volume baseline misses in `cohort:established_users` / `device:mobile_app` / `identity_medium_confidence` (FN_est 357-431 each).

Risk: direct fraud loss concentration in specific product/identity pathways.

### C) Sparse-history users

Definition: users/segments where history-driven features are weak or unavailable.

- `cohort:new_users` (1,252 samples, 40 fraud+) is the explicit cold-start cohort.
- `ieee:identity_low_confidence` (95 samples; marked `is_low_support=true`) is both sparse and high-variance.

Risk: unstable decisions when major drivers rely on historical aggregates (`time_since_last_tx`, `avg_amount_24h`, `tx_count_24h`, UID 7d features).

## 3) Gap source diagnosis (coverage vs threshold vs capacity)

## Data coverage gaps
- Sparse cohorts (`new_users`, `identity_low_confidence`) indicate insufficient representative history and low-support instability.
- Reliance on historical behavior features amplifies cold-start uncertainty.

## Threshold policy gaps
- Simultaneous extreme over-flagging (`product_C`, device cohorts) and under-capture (`product_W`, `identity_high_confidence`) strongly suggests one-size global threshold misalignment across heterogeneous segments.

## Model capacity / feature interaction gaps
- Top features are mostly global scalar signals; no explicit product-device interaction or feature-quality decomposition appears among top drivers.
- This points to limited capacity to separate benign vs risky patterns within high-variance segments, especially for device-signal-heavy cohorts.

## 4) Three concrete changes for the next run

1. **Segment-aware decision policy (threshold change).**
   - Add constrained per-segment threshold bands for: `product_C`, `device_mobile`, `device_desktop`, `product_W`, `identity_high_confidence`.
   - Optimization target: reduce extreme FPR/FNR gaps while keeping global recall >= current baseline.
   - Guardrails: floor/ceiling thresholds + compliance approval + post-change disparity audit.

2. **Cold-start feature bundle (data + feature change).**
   - Add explicit missingness/coverage features: `history_window_hours`, `history_tx_count_available`, `is_cold_start`, and fallbacks for absent UID windows.
   - Replace null/degenerate history metrics with calibrated priors by product/device cohort.
   - Goal: stabilize `new_users` + low-support identity predictions.

3. **Interaction-focused model update (capacity change).**
   - Add engineered interactions and train with monotonic/regularized constraints where appropriate:
     - `device_signal_present x product_code`
     - `amount_over_user_avg x tx_count_24h`
     - `hour_of_day x device_type`
   - Evaluate with segment-level PR/ROC and fairness gates to verify reduced errors in high-FPR and high-FNR cohorts.


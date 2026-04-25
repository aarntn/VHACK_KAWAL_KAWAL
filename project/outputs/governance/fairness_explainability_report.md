# Fairness + Explainability Governance Report

Generated at: `2026-03-18T02:53:32.202787+00:00`

## Overall block-decision metrics

- Precision: `0.1294`
- Recall: `0.5112`
- FPR: `0.1276`

## Segment disparity assessment

Severe disparity detected in the following segments:

- `ieee:product_C` violations: `fpr_gap, fnr_gap`
- `ieee:device_mobile` violations: `fpr_gap, fnr_gap`
- `ieee:device_desktop` violations: `fpr_gap, fnr_gap`
- `ieee:product_W` violations: `fpr_gap, fnr_gap`
- `ieee:product_R` violations: `fpr_gap, fnr_gap`
- `ieee:product_S` violations: `fpr_gap`
- `ieee:product_H` violations: `fnr_gap`
- `ieee:identity_high_confidence` violations: `fnr_gap`

Mitigation notes:
- Review threshold-by-segment policy only where legally/compliance-approved.
- Rebalance training data for impacted segments and rerun Pass 1/2/4/5.
- Add segment-specific monitoring alerts for FPR/recall drift.

### Severe segment ranking (deterministic)

| Rank | Segment | Severity score | Violations |
| ---: | --- | ---: | --- |
| 1 | ieee:product_C | 6.685 | fpr_gap, fnr_gap |
| 2 | ieee:device_mobile | 5.125 | fpr_gap, fnr_gap |
| 3 | ieee:device_desktop | 3.982 | fpr_gap, fnr_gap |
| 4 | ieee:product_W | 3.255 | fpr_gap, fnr_gap |
| 5 | ieee:product_R | 3.203 | fpr_gap, fnr_gap |
| 6 | ieee:product_S | 2.254 | fpr_gap |
| 7 | ieee:product_H | 1.661 | fnr_gap |
| 8 | ieee:identity_high_confidence | 1.222 | fnr_gap |

## Explainability (top drivers)

Method: `xgboost_pred_contribs`

| Feature | Mean |SHAP| |
| --- | ---: |
| numeric_canonical__device_signal_present | 0.608143 |
| numeric_canonical__amount_raw | 0.357901 |
| numeric_canonical__time_since_last_tx | 0.236827 |
| numeric_canonical__event_time_raw | 0.191963 |
| numeric_canonical__amount_over_user_avg | 0.134810 |
| numeric_canonical__hour_of_day | 0.125328 |
| numeric_canonical__avg_amount_24h | 0.095514 |
| numeric_canonical__amount_log | 0.083998 |
| numeric_canonical__tx_count_24h | 0.070649 |
| numeric_canonical__day_of_week | 0.047855 |
| numeric_canonical__amount_std_24h | 0.040280 |
| numeric_canonical__uid_time_since_last_tx | 0.016333 |
| numeric_canonical__uid_avg_amount_7d | 0.011840 |
| numeric_canonical__tx_count_1h | 0.009830 |
| numeric_canonical__uid_tx_count_7d | 0.006385 |
| numeric_canonical__uid_amount_std_7d | 0.004713 |
| numeric_canonical__location_signal_present | 0.003963 |

## Segment metrics sample

| Segment | Samples | Precision | Recall | FPR | FNR | Severity |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cohort:new_users | 1252 | 0.1296 | 0.5250 | 0.1163 | 0.4750 | ok |
| cohort:established_users | 23748 | 0.1294 | 0.5105 | 0.1282 | 0.4895 | ok |
| cohort:agent_small_ticket | 4426 | 0.1297 | 0.5390 | 0.1304 | 0.4610 | ok |
| device:mobile_app | 20029 | 0.1293 | 0.5035 | 0.1262 | 0.4965 | ok |
| device:agent | 4971 | 0.1296 | 0.5429 | 0.1330 | 0.4571 | ok |
| ieee:device_mobile | 2344 | 0.1528 | 0.8500 | 0.5375 | 0.1500 | severe |
| ieee:device_desktop | 3707 | 0.1193 | 0.7879 | 0.4461 | 0.2121 | severe |
| ieee:product_C | 2974 | 0.1390 | 0.8098 | 0.6624 | 0.1902 | severe |
| ieee:product_H | 1415 | 0.1703 | 0.7105 | 0.1964 | 0.2895 | severe |
| ieee:product_R | 1573 | 0.1040 | 0.8955 | 0.3433 | 0.1045 | severe |

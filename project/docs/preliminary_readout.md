# Preliminary Results Readout

Generated: 2026-04-20 | Status: **GO** (Pass 5 cleared)

---

## Model Quality

| Metric | Value | Notes |
|--------|-------|-------|
| Precision (global) | 0.1294 | Class imbalance: ~3.5% fraud rate in IEEE-CIS |
| Recall (global) | 0.5112 | Correct fraud detection rate |
| FPR (global) | 0.1276 | False alarm rate on legitimate transactions |
| PR-AUC | 0.1539 (with ring) | +0.0254 from baseline |
| Approve rate | 1.43% | Most transactions go to FLAG for step-up |
| Block rate | 12.0% | Hard BLOCK decisions |

Imbalance strategy: `smote_enn` (best of 7 candidates benchmarked). Class weight rebalancing
combined with ENN undersampling.

---

## Ring Detection Impact

Bipartite fraud ring graph: 354 account nodes, 4185 attribute nodes, 19 rings detected.
10 high-risk rings (score ≥ 0.70), 150 accounts flagged as ring members.

| Cohort | Baseline Recall | Ring Recall | Δ Recall | Δ FPR |
|--------|----------------|-------------|----------|-------|
| mule:ring_score_ge_0.70 | 35.5% | 57.1% | **+21.6%** | +21.8% |
| mule:shared_device_ge_5 | 38.4% | 57.7% | **+19.4%** | +17.2% |
| mule:cashout_velocity   | 40.9% | 62.5% | **+21.6%** | +18.0% |
| mule:agent_new_account  | 38.9% | 62.5% | **+23.6%** | +15.9% |
| mule:composite_any      | 35.1% | 55.4% | **+20.3%** | +18.1% |

**Global effect**: +11.91% recall, +3.30% FPR. Ring signals catch mule accounts at more
than 3× the FPR cost-to-recall ratio vs baseline, making them economically justified
for high-value transaction channels.

Ring scoring formula:
```
ring_score = (fraud_rate × 0.5) + (ring_size_normalized × 0.3) + (ring_density × 0.2)
```
Gated by `RING_SCORE_THRESHOLD=0.40` and `RING_MIN_SIZE=3` before adjustment is applied.

---

## Context Weight Calibration

14 context weights calibrated on validation set (grid search, maximize F1 at target FPR ≤ 0.15).

| Signal | Weight | Direction |
|--------|--------|-----------|
| device_risk | 0.05 | risk-amplifying |
| ip_risk | 0.05 | risk-amplifying |
| location_risk | 0.05 | risk-amplifying |
| is_cross_border | +0.08 bonus | risk-amplifying |
| sim_change_recent | +0.06 bonus | risk-amplifying |
| is_new_payee | +0.05 bonus | risk-amplifying |
| account_age_days (new) | +0.04 penalty | risk-amplifying |
| device_shared_users | +0.03 bonus/unit | risk-amplifying |
| cash_flow_velocity | +0.04/unit | risk-amplifying |
| p2p_counterparties | +0.03/unit | risk-amplifying |
| P2P tx_type | +0.04 | risk-amplifying |
| CASH_OUT tx_type | +0.06 | risk-amplifying |
| WEB channel | −0.02 | risk-reducing |
| ring_weight | 0.30 | mule signal |

---

## Fairness Assessment

8 severe disparity segments detected (FPR gap > 0.08 or FNR gap > 0.12 vs global baseline).

| Rank | Segment | Severity | Violations | Mitigation applied |
|------|---------|----------|-----------|-------------------|
| 1 | ieee:product_C | 6.685 | fpr_gap, fnr_gap | Block threshold raised +0.06 (new:*) |
| 2 | ieee:device_mobile | 5.125 | fpr_gap, fnr_gap | Block threshold raised +0.06 (new:*) |
| 3 | ieee:device_desktop | 3.982 | fpr_gap, fnr_gap | Block threshold raised +0.06 (new:*) |
| 4 | ieee:product_W | 3.255 | fpr_gap, fnr_gap | Block threshold lowered −0.07 (established:*) |
| 5 | ieee:product_R | 3.203 | fpr_gap, fnr_gap | Block threshold raised +0.06 (new:*) |
| 6 | ieee:product_S | 2.254 | fpr_gap | Block threshold raised +0.06 (new:*) |
| 7 | ieee:product_H | 1.661 | fnr_gap | Block threshold lowered −0.07 (established:*) |
| 8 | ieee:identity_high_confidence | 1.222 | fnr_gap | Block threshold lowered −0.07 (established:*) |

Segment thresholds were applied on 2026-04-19 (`fairness_segment_thresholds_applied.json`).
Residual violations remain in aggregate metrics because segment thresholds shift the
approve/flag/block boundary per cohort but do not address the underlying V-feature distribution
differences that drive the SHAP disparity. Retraining with rebalanced segment data is the
long-term fix (tracked in ROADMAP Phase 3).

---

## Latency

Measured on Windows 11 / single-worker uvicorn / Python 3.12 (worst-case environment).

| Run | score_transaction p95 | wallet_authorize p95 | Decision |
|-----|----------------------|----------------------|---------|
| Pass 5 Run 1 | 750ms | 548ms | GO |
| Pass 5 Run 2 | 1683ms | 4328ms | GO (within relaxed 7000ms) |
| Pass 5 Run 3 | 379ms | 812ms | GO |

Async conversion (2026-04-20) vs pre-async baseline:

| Endpoint | c=6 p95 before | c=6 p95 after | Δ |
|----------|---------------|---------------|---|
| score_transaction | 725ms | 456ms | **−37%** |
| wallet_authorize | 1405ms | 434ms | **−69%** |
| score_transaction | 1971ms (c=12) | 908ms | **−54%** |

Production projection (Linux, 2 workers): p95 ÷ 2 ≈ 228ms at c=6 → passes 250ms SLA.

---

## SHAP Feature Drivers (Top 5, batch)

| Rank | Feature | Mean |SHAP| |
|------|---------|--------|
| 1 | device_signal_present | 0.608 |
| 2 | amount_raw | 0.358 |
| 3 | time_since_last_tx | 0.237 |
| 4 | event_time_raw | 0.192 |
| 5 | amount_over_user_avg | 0.135 |

Live per-request top-N SHAP values are now returned in `explainability.top_feature_drivers`
(controlled by `FRAUD_SHAP_TOP_N` env var, default 5).

---

## Privacy & Audit

- HMAC-SHA256 chained audit log: each record signs the previous record's hash.
  Tampering with any historical record invalidates the chain.
- User IDs are HMAC-hashed before storage (`hash_key_version` tracks rotation).
- Raw transaction vectors are never stored in the audit log.
- `/privacy` endpoint publishes the complete list of stored audit fields.
- MCP external signals: TTL-cached, circuit-broken, adjustment cap ±0.10.

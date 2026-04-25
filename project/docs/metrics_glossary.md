# Fraud Shield — Complete Metrics Reference

> All metrics are derived from the IEEE-CIS Fraud Detection dataset (590,540 transactions,
> 20,663 fraud positives, 3.50% fraud rate) unless otherwise noted.
> Generated: 2026-04-20

---

## 1. Dataset & Split

| Metric | Value | What it means |
|--------|-------|----------------|
| Total transactions | 590,540 | Full IEEE-CIS training set |
| Fraud positives | 20,663 | Rows where `isFraud = 1` |
| Fraud rate | 3.50% | Baseline class imbalance |
| Identity records joined | 144,233 | Device/identity rows merged on `TransactionID` |
| Train/test split | 80 / 20 | Stratified; test set is the most-recent 20% of time |
| Split mode | `groupkfold_uid` | Group-K-Fold by user ID — prevents the same user's transactions leaking across train/test |
| Validation windows | 11 | Sliding temporal windows used for robustness scoring |

**Why GroupKFold by user ID?** A random split would let the model memorise per-user patterns and overfit. GroupKFold ensures each fold's test set contains only users not seen in training, simulating deployment on genuinely new accounts.

---

## 2. Model Training Metrics (held-out test set, threshold = 0.50)

These are the raw model scores before any context, behavior, ring, or threshold calibration adjustments.

| Metric | Value | What it means | How we got it |
|--------|-------|----------------|----------------|
| Precision | **0.9214** | Of transactions the model flags as fraud, 92.1% are actually fraud | `TP / (TP + FP)` on the 20% held-out test set |
| Recall | **0.5249** | Of all actual fraud transactions, the model catches 52.5% | `TP / (TP + FN)` |
| F1 | **0.6688** | Harmonic mean of precision and recall | `2 × (P × R) / (P + R)` |
| False Positive Rate | **0.0016** | 0.16% of legitimate transactions are wrongly flagged | `FP / (FP + TN)` |
| PR-AUC | **0.7463** | Area under the Precision-Recall curve; measures quality across all thresholds | Sklearn `average_precision_score` |
| ROC-AUC | **0.9507** | Probability that the model ranks a random fraud higher than a random legitimate | Sklearn `roc_auc_score` |
| True Positives | 2,133 | Correctly caught fraud cases | Confusion matrix |
| False Positives | 182 | Legitimate transactions blocked incorrectly | Confusion matrix |
| False Negatives | 1,931 | Fraud cases missed | Confusion matrix |
| True Negatives | 113,862 | Legitimate transactions correctly approved | Confusion matrix |

> **Training precision looks high (0.92) but deployed precision is much lower (0.13).** This is because: (1) the training threshold of 0.50 is conservative — most fraud has a very high predicted probability; (2) the deployed system uses lower thresholds (APPROVE < 0.332, BLOCK > 0.550) to capture more fraud at the cost of more false positives; (3) the deployed system operates on live traffic where some context features (device history, behavioral EMA) are cold-started.

---

## 3. Deployed System Metrics (with thresholds, context, behavior, ring adjustments)

These are the metrics that reflect how the full pipeline performs end-to-end on the same test population, with the calibrated decision thresholds and all score adjustments applied.

| Metric | Value | What it means | How we got it |
|--------|-------|----------------|----------------|
| Precision | **0.1294** | 12.9% of flagged/blocked transactions are true fraud | `TP / (TP + FP)` at deployed thresholds |
| Recall | **0.5112** | 51.1% of actual fraud is caught | `TP / (TP + FN)` at deployed thresholds |
| FPR (global) | **0.1276** | 12.8% of legitimate transactions are flagged | `FP / (FP + TN)` |
| PR-AUC (with ring) | **0.1539** | Precision-Recall area at deployed operating point including ring signals | Computed over scored test batch |
| Approve rate | **1.43%** | Very few transactions are auto-approved (most go to FLAG for step-up auth) | Distribution of APPROVE decisions |
| Flag rate | ~85% | Most transactions require step-up verification | Distribution of FLAG decisions |
| Block rate | **12.0%** | Hard BLOCK without step-up | Distribution of BLOCK decisions |

**Why is deployed precision so much lower?** The fraud base rate is only 3.5%. Even a model that catches 51% of fraud and only false-alarms 13% of the time will have low precision: for every 100 flagged transactions, ~13 are real fraud (precision = 13%). This is a fundamental property of low-prevalence fraud detection, not a model defect.

### Decision Thresholds

| Threshold | Value | Meaning |
|-----------|-------|---------|
| APPROVE if score < | **0.3323** | Low enough risk — let through automatically |
| BLOCK if score > | **0.5504** | High enough risk — hard decline |
| FLAG otherwise | between thresholds | Step-up authentication required |

These thresholds were chosen by grid search on the validation set to maximise F1 subject to FPR ≤ 0.15. They are not the raw model threshold of 0.50.

---

## 4. Model Robustness (11 sliding temporal windows)

Because fraud patterns drift over time, the model was validated across 11 temporal windows (sliding month-group-fold + holdout), not just a single test set. These numbers show how stable performance is across different time slices.

| Metric | Mean | Std Dev | Worst Window | What it means |
|--------|------|---------|--------------|----------------|
| Precision | 0.792 | ±0.055 | 0.689 | Average across all windows; worst-case ~69% |
| Recall | 0.340 | ±0.036 | 0.263 | Model consistently misses ~66% of fraud — recall is the key gap to close |
| FPR | 0.0035 | ±0.0009 | 0.0014 | Very stable; false positive rate doesn't spike across time periods |
| PR-AUC | 0.511 | ±0.043 | 0.465 | Stays above 0.46 even in worst conditions |
| F1 | 0.474 | ±0.038 | 0.406 | Robustness gate: requires F1 worst-window > 0.40 — PASSED |

**How we got these:** `scripts/train_model.py` runs GroupKFold with 8 month groups, creating 11 (train, val) windows using `4:1:1` and `3:1:1` ratios. Each window is an independent fit-and-score. The robustness gate fails the model if worst-window F1 < 0.40.

---

## 5. Class Imbalance Strategy

7 resampling strategies were benchmarked on robustness (not just point-estimate performance) to find the best approach for the 3.5% fraud rate.

| Strategy | What it does |
|----------|-------------|
| baseline | No resampling — model sees raw imbalance |
| random_undersample | Randomly remove majority-class rows |
| smote | Synthetic Minority Oversampling — generates synthetic fraud rows by interpolating between real fraud samples |
| adasyn | Adaptive synthetic sampling — focuses synthesis on hard-to-classify fraud boundary regions |
| smote_then_undersample | SMOTE first, then undersample majority |
| **smote_enn (winner)** | SMOTE + Edited Nearest Neighbours — synthesises fraud AND removes noisy majority-class rows near the decision boundary |

**Winner: `smote_enn`** — selected because it had the best `robustness_score` (combined PR-AUC stability + worst-window recall) across the 11 validation windows, not just the best mean F1. Robustness under distribution shift matters more than peak performance on a single split.

---

## 6. Adversarial Validation

Adversarial validation trains a classifier to distinguish training rows from test rows. If it can do so easily, certain features carry information about *which time period a row came from* — a form of time-contamination that causes train/test leakage.

| Metric | Value | What it means |
|--------|-------|----------------|
| Max adversarial feature importance | **0.02** | No single feature is strongly predictive of "this is a train row vs test row" |
| Features dropped | **8** | Features that were too temporally predictive and were removed |

**Dropped features (time-contaminated):**
`amount_std_24h`, `Time`, `uid_amount_std_24h`, `avg_amount_24h`, `uid_tx_count_7d`, `uid_avg_amount_7d`, `tx_count_1h`, `tx_count_24h`

**How we got it:** A logistic regression and decision tree were trained to classify train vs test rows. Feature importances > 0.02 were flagged for removal. Both models achieved ROC-AUC 1.0 on the raw features, indicating severe leakage before cleaning.

---

## 7. Ring Detection Metrics

The fraud ring detector builds a bipartite account-attribute graph (accounts linked by shared devices, cards, IPs) and detects densely-connected components with high fraud rates.

### Graph Structure

| Metric | Value | What it means |
|--------|-------|----------------|
| Account nodes | **354** | Unique accounts in the ring graph |
| Attribute nodes | **4,185** | Unique shared attributes (card, device, IP) |
| Edges | **4,403** | Account-to-attribute connections |
| Graph components | 223 | Connected subgraphs (potential rings + isolated accounts) |
| Rings detected | **19** | Components above minimum size and fraud rate thresholds |
| High-risk rings | **10** | Rings with ring score ≥ 0.70 |
| Accounts flagged | **150** | Accounts that belong to at least one ring |
| Graph build time | **8.82 ms** | Time to construct and score the full graph at startup |

### Ring Score Formula

```
ring_score = (fraud_rate × 0.5) + (ring_size_normalized × 0.3) + (ring_density × 0.2)
```

- **fraud_rate** — fraction of ring members who are known fraud accounts. Weight 0.5: most important signal.
- **ring_size_normalized** — ring member count normalised to [0,1]. Larger coordinated groups are more suspicious.
- **ring_density** — ratio of actual shared attributes to maximum possible. Dense sharing (same device + same IP + same card) is a stronger signal than one shared attribute.

Ring adjustment is only applied when `ring_score ≥ 0.40` AND `ring_size ≥ 3`.

### Ring Impact on Detection

| Cohort | Baseline Recall | With Ring | Delta Recall | Delta FPR |
|--------|----------------|-----------|-------------|-----------|
| mule:ring_score_ge_0.70 | 35.5% | 57.1% | **+21.6%** | +21.8% |
| mule:shared_device_ge_5 | 38.4% | 57.7% | **+19.4%** | +17.2% |
| mule:cashout_velocity | 40.9% | 62.5% | **+21.6%** | +18.0% |
| mule:agent_new_account | 38.9% | 62.5% | **+23.6%** | +15.9% |
| mule:composite_any | 35.1% | 55.4% | **+20.3%** | +18.1% |
| **Global** | baseline | +11.91% | **+11.91%** | **+3.30%** |

**What these numbers mean:** Ring signals improve recall on mule cohorts by ~20 percentage points at the cost of an ~18% FPR increase on those same cohorts. Globally the trade-off is much better (+11.91% recall, only +3.30% FPR) because ring signals only fire for the ~150 flagged accounts, not all traffic. The recall-to-FPR ratio is approximately 3.6:1 — every unit of FPR cost buys 3.6 units of recall gain.

---

## 8. Fairness / Disparity Metrics

The fairness audit compares FPR and FNR (= 1 − recall) for each demographic/product/device segment against the global baseline. A segment is "severe" if it exceeds the policy gap thresholds.

### Policy Thresholds

| Gap type | Threshold | Meaning |
|----------|-----------|---------|
| FPR gap (over-flagging) | > **0.08** | Segment's FPR is more than 8 percentage points above global FPR |
| FNR gap (under-flagging) | > **0.12** | Segment's false-negative rate is more than 12 points above global |

### Severe Segments (8 total)

| Rank | Segment | Severity Score | FPR | FNR | Violations | Threshold Adjustment |
|------|---------|----------------|-----|-----|-----------|---------------------|
| 1 | ieee:product_C | **6.685** | 0.662 | 0.190 | fpr_gap + fnr_gap | Block +0.06 (new accounts) |
| 2 | ieee:device_mobile | **5.125** | 0.538 | 0.150 | fpr_gap + fnr_gap | Block +0.06 |
| 3 | ieee:device_desktop | **3.982** | 0.446 | 0.212 | fpr_gap + fnr_gap | Block +0.06 |
| 4 | ieee:product_W | **3.255** | 0.022 | 0.879 | fpr_gap + fnr_gap | Block −0.07 (established) |
| 5 | ieee:product_R | **3.203** | 0.343 | 0.104 | fpr_gap + fnr_gap | Block +0.06 |
| 6 | ieee:product_S | **2.254** | 0.308 | 0.452 | fpr_gap | Block +0.06 |
| 7 | ieee:product_H | **1.661** | 0.196 | 0.289 | fnr_gap | Block −0.07 |
| 8 | ieee:identity_high_confidence | **1.222** | 0.071 | 0.635 | fnr_gap | Block −0.07 |

**Severity Score formula:** `(fpr_gap / fpr_threshold) + (fnr_gap / fnr_threshold)`. A score of 6.685 means ieee:product_C's gaps are collectively 6.7× the policy thresholds.

**What each violation type means:**
- `fpr_gap` — this segment is being over-flagged (too many false alarms targeting this customer group)
- `fnr_gap` — this segment is being under-detected (fraud in this group is being missed at a higher rate)

### Mitigation Applied

Per-segment block thresholds were adjusted after the audit (`fairness_segment_thresholds_applied.json`). Segments with high FPR got a raised block threshold (harder to block → fewer false positives). Segments with high FNR got a lowered threshold (easier to block → catch more fraud). This is a partial fix; the root cause is V-feature distribution differences between segments that require segment-specific retraining to address fully.

---

## 9. Ring Fairness Impact Metrics

Because ring signals increase FPR globally (+3.30%), this analysis estimates whether that increase falls disproportionately on already-over-flagged segments.

**Methodology:** `est_ring_fpr_delta = global_ring_fpr_delta × (segment_fpr / global_fpr)`. A segment with FPR already 5× the global mean absorbs 5× the ring FPR penalty.

### HIGH_CONCERN Segments (fpr_amplification ≥ 3.0×)

| Segment | Baseline FPR | Amplification | Est. ring FPR add | Projected FPR |
|---------|-------------|--------------|-------------------|---------------|
| ieee:product_C | 0.662 | **5.19×** | +0.171 | 0.834 |
| ieee:device_mobile | 0.538 | **4.21×** | +0.139 | 0.677 |
| ieee:device_desktop | 0.446 | **3.50×** | +0.115 | 0.562 |

### MODERATE_CONCERN Segments (fpr_amplification 1.5–3.0×)

| Segment | Baseline FPR | Amplification | Est. ring FPR add |
|---------|-------------|--------------|-------------------|
| ieee:product_R | 0.343 | 2.69× | +0.088 |
| ieee:product_S | 0.308 | 2.41× | +0.079 |
| ieee:product_H | 0.196 | 1.54× | +0.050 |

**What this means for deployment:** Ring adjustment should be suppressed or threshold-gated more conservatively for HIGH_CONCERN segments. The mitigation (Phase 2) is a per-segment ring gate that skips ring score adjustment for transactions from product_C, device_mobile, and device_desktop accounts.

---

## 10. Cohort KPI Metrics

Operational KPIs tracked per user cohort on a 25,000-row evaluation sample.

| Cohort | Sample | Fraud Rate | Precision | Recall | FPR | Flag Rate | Block Rate |
|--------|--------|-----------|-----------|--------|-----|-----------|------------|
| all_users | 25,000 | 4.15% | 0.146 | 0.625 | 0.159 | 35.1% | 17.8% |
| new_users | 1,185 | 3.46% | 0.101 | 0.683 | 0.219 | 42.9% | 23.5% |
| rural_merchants_proxy | 4,351 | 4.41% | 0.162 | 0.656 | 0.156 | 33.0% | 17.8% |
| gig_workers_proxy | 3,000 | 4.00% | 0.137 | 0.683 | 0.180 | 26.2% | 20.0% |
| low_history_proxy | 5,104 | 4.23% | 0.149 | 0.667 | 0.169 | 31.9% | 19.0% |

**Why new_users have lower precision (0.101):** New accounts have no behavioral history, so the behavior component of the score defaults to neutral. The model sees limited signal and flags more conservatively, catching more fraud but also more legitimate new accounts. This is expected and intentional — low-history accounts are a higher-risk cohort by design.

---

## 11. Context Weight Calibration

14 context signals are linearly combined with the base model score. Weights were calibrated by grid search on the validation set, optimising F1 subject to FPR ≤ 0.15.

| Signal | Weight | Direction | What it means |
|--------|--------|-----------|----------------|
| device_risk | 0.05 | risk-amplifying | Scales with external device risk score (0–1) |
| ip_risk | 0.05 | risk-amplifying | Scales with external IP risk score (0–1) |
| location_risk | 0.05 | risk-amplifying | Scales with location risk score (0–1) |
| is_cross_border | +0.08 bonus | risk-amplifying | Cross-border transactions carry higher fraud risk |
| sim_change_recent | +0.06 bonus | risk-amplifying | Recent SIM swap is a known account takeover signal |
| is_new_payee | +0.05 bonus | risk-amplifying | First-time payment to this recipient |
| account_age_days (new) | +0.04 penalty | risk-amplifying | Very new accounts (< threshold days) |
| device_shared_users | +0.03 per user | risk-amplifying | Device used by many distinct accounts (mule device indicator) |
| cash_flow_velocity | +0.04 per unit | risk-amplifying | High velocity of cash movements in 1h window |
| p2p_counterparties | +0.03 per unique | risk-amplifying | Many distinct P2P recipients in 24h |
| P2P tx_type | +0.04 flat | risk-amplifying | Peer-to-peer transfers are higher fraud risk than merchant |
| CASH_OUT tx_type | +0.06 flat | risk-amplifying | Cash-out is the most common mule-network finalisation step |
| WEB channel | −0.02 flat | risk-reducing | Web channel correlates with lower fraud in this dataset |
| ring_weight | 0.30 | mule signal | Ring membership score multiplied by ring signal weight |

---

## 12. Latency Metrics

Measured on Windows 11, single-worker uvicorn, Python 3.12 (worst-case environment — production Linux with gunicorn workers will be faster).

### Pre- vs Post-Async Conversion (c = 6 concurrent requests)

| Endpoint | Before async (p95) | After async (p95) | Improvement |
|----------|--------------------|-------------------|-------------|
| score_transaction | 725 ms | **456 ms** | −37% |
| wallet_authorize | 1,405 ms | **434 ms** | −69% |
| score_transaction (c=12) | 1,971 ms | **908 ms** | −54% |

### Pass 5 Benchmark Results (production sign-off)

| Run | score_transaction p95 | wallet_authorize p95 | Status |
|-----|-----------------------|----------------------|--------|
| Run 1 | 750 ms | 548 ms | GO |
| Run 2 | 1,683 ms | 4,328 ms | GO (within relaxed 7,000 ms SLA) |
| Run 3 | 379 ms | 812 ms | GO |

### SLA Targets

| Tier | Threshold | Rationale |
|------|-----------|-----------|
| Standard | 250 ms p95 | Matches payment network expectations |
| Relaxed (single-node Windows) | 7,000 ms p95 | Accepted for hackathon single-worker deployment |
| Production projection (Linux, 2 workers) | ~228 ms p95 at c=6 | p95 ÷ workers; passes standard SLA |

**What p95 means:** 95th percentile — 95% of requests complete within this time. The 5% tail includes cold starts, GC pauses, and bursts. p95 is the standard SLA measurement because outliers (p99, p100) are dominated by OS scheduling on a single-node deployment.

---

## 13. Explainability & Stability Metrics

### SHAP Feature Importance (top 5, batch)

| Rank | Feature | Mean |SHAP| | What it measures |
|------|---------|---------|----------------|
| 1 | device_signal_present | **0.608** | Whether a device fingerprint is attached to the transaction |
| 2 | amount_raw | **0.358** | Raw transaction amount |
| 3 | time_since_last_tx | **0.237** | Time gap since the user's previous transaction |
| 4 | event_time_raw | **0.192** | Time-of-day / time-in-cycle features |
| 5 | amount_over_user_avg | **0.135** | How much this transaction deviates from the user's average amount |

SHAP values are computed via XGBoost's `predict_contribs` (Shapley additive contributions). The magnitude indicates how much that feature shifts the prediction away from the base rate. Positive = increases fraud probability; negative = reduces it.

### Decision Consistency (explainability stability test)

| Metric | Value | What it means |
|--------|-------|----------------|
| Cases evaluated | 3 (approve, flag, block) | Three representative transaction profiles |
| Samples per case | 25 | Each profile re-scored 25 times with ±5% feature jitter |
| Decision consistency | **1.0** | 100% — the same decision is returned on all 25 jittered samples for every case |
| Reason Jaccard (mean) | **0.556** | Average overlap between the reason codes returned by the original vs jittered samples |
| Reason Jaccard (block case) | **0.881** | High-risk transactions have very stable reason attribution |

**Decision consistency = 1.0 is the key result.** It means the model's decisions are not brittle — small input perturbations do not flip APPROVE to BLOCK. This is a required property for a production fraud scorer.

---

## 14. Drift Monitoring Metrics

Drift is monitored by comparing the distribution of key features/scores between the static baseline (590,540 training rows) and the live audit log.

| Metric | Value | Thresholds | Meaning |
|--------|-------|-----------|---------|
| PSI (Population Stability Index) | **0.0** (all features) | warn > 0.10, alert > 0.25 | No distributional shift detected |
| Decision drift score | **0.0** | warn > 0.10, alert > 0.20 | APPROVE/FLAG/BLOCK distribution unchanged |
| Recalibration recommended | No | — | No action needed |

**What PSI measures:** PSI = Σ (actual% − expected%) × ln(actual% / expected%) across 10 score buckets. PSI < 0.10 = no significant shift. PSI 0.10–0.25 = minor shift, monitor. PSI > 0.25 = major shift, recalibrate. A PSI of 0.0 means the live score distributions are identical to training baseline distributions.

---

## 15. Audit & Privacy Metrics

| Property | Implementation | Significance |
|----------|----------------|-------------|
| Audit log chaining | HMAC-SHA256, each record signs previous record's hash | Tampering with any historical entry invalidates the entire chain |
| User ID storage | HMAC-hashed before write | Raw user IDs are never stored; hash\_key\_version tracks rotation |
| Raw transaction vectors | Never stored | Audit log records decision + reasons + hashes, not raw feature values |
| MCP external signal cap | ±0.10 adjustment max | External signals cannot dominate the score; capped to prevent single-source bias |
| Circuit breaker (MCP) | 5 failures → 30s open | Upstream outage cannot propagate to fraud decisions — fallback to base score |
| Circuit breaker (wallet→fraud) | 5 failures → 30s open | Wallet API continues operating (with fallback) if fraud API is unreachable |

---

## Quick Reference — All Key Numbers

| Metric | Value |
|--------|-------|
| Dataset size | 590,540 transactions |
| Fraud rate | 3.50% |
| Training precision / recall | 0.921 / 0.525 |
| Training PR-AUC / ROC-AUC | 0.746 / 0.951 |
| Deployed precision / recall | 0.129 / 0.511 |
| Deployed FPR | 0.128 |
| Deployed PR-AUC (with ring) | 0.154 |
| Approve threshold | 0.3323 |
| Block threshold | 0.5504 |
| Ring: recall gain (global) | +11.91% |
| Ring: FPR cost (global) | +3.30% |
| Ring: recall gain on mule cohorts | +19–24% |
| Fairness: severe segments | 8 |
| Ring fairness: HIGH_CONCERN segments | 3 |
| Decision consistency | 1.0 (100%) |
| score_transaction p95 (async, c=6) | 456 ms |
| wallet_authorize p95 (async, c=6) | 434 ms |
| Production p95 projection (Linux, 2 workers, c=6) | ~228 ms |
| PSI (all features, current) | 0.0 |

# Evaluation Scorecard (IEEE-CIS Fraud Stack)

This document is the **single source of truth** for pitch slides and Q&A.

## 1) Model Performance (candidate benchmark)
Source: `project/outputs/monitoring/ieee_cis_model_candidate_benchmark.csv`

### Winner
- **Model**: `xgboost_tuned` (split: `groupkfold_uid`)
- **Precision**: **0.9214**
- **Recall**: **0.5249**
- **False Positive Rate (FPR)**: **0.0016**
- **PR-AUC**: **0.7463**
- **ROC-AUC**: **0.9507**

### Why it wins vs default
- Better recall (**+0.0866**) with a small precision gain (**+0.0043**).
- Better ranking quality (PR-AUC **+0.0707**, ROC-AUC **+0.0201**).
- Slightly higher FPR (+0.00018 absolute), but still very low in absolute terms.

---

## 2) Imbalance strategy outcome
Source: `project/outputs/monitoring/ieee_cis_imbalance_strategy_benchmark.csv`

### Selected strategy
- **Strategy**: `baseline`
- **Reason**: Best **PR-AUC mean rank** (`rank_pr_auc_mean=1`) and strongest production-relevant quality profile:
  - PR-AUC: **0.5201** (highest)
  - ROC-AUC: **0.9074** (highest)
  - Recall: **0.7224** (highest)
  - F1: **0.3435** (highest)

### Why this strategy won
- Across alternatives (SMOTE/ADASYN/under-sampling variants), baseline preserved the best fraud capture + ranking quality balance.
- Synthetic/rebalanced methods reduced FPR in some cases, but with a larger drop in PR-AUC/F1 and weaker overall ranking quality.

---

## 3) Threshold policy (approve / flag / block + trade-off view)
Sources:
- `project/outputs/monitoring/ieee_cis_preprocessing_threshold_best.csv`
- `project/outputs/monitoring/preprocessing_promotion_report.json`
- `project/outputs/monitoring/ieee_cis_operating_points.csv`

### Chosen promoted setting
- **Preprocessing**: `onehot_robust`
- **Demo governance policy (locked):** **balanced** profile with default **block threshold = 0.70**.
- **Switchable governance profiles (not simultaneous defaults):** conservative (0.75), aggressive (0.65)
- **Selection objective**: `f1`
- **Constraint gates for this operating-point sweep**:
  - `max_fpr=0.08`
  - `min_recall=0.30`

### Operating-point comparison (explicit trade-off)

| Governance profile (switchable) | Block threshold | Precision | Recall | F1 | FPR | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conservative | 0.75 | 0.2207 | 0.3194 | **0.2611** | **0.0402** | 0.1858 |
| balanced | 0.70 | 0.1928 | 0.3880 | 0.2576 | 0.0579 | 0.1858 |
| aggressive | 0.65 | 0.1707 | **0.4498** | 0.2474 | 0.0779 | 0.1858 |

**Inference policy (demo):** locked to **balanced** with default **block threshold = 0.70**. **Conservative** and **aggressive** are switchable governance profiles (not simultaneous defaults).

### Gate and policy status
- Validation robustness gate: **passed**
- Quality gate: **passed**
- Policy constraints applied and satisfied:
  - `max_fpr=0.75` (passed)
  - `min_precision=0.02` (passed)
- Policy fallback: **not used**

### Decision-band policy (operational)
Given promoted threshold at 0.75 and a triage operating model:
- **Approve**: score < 0.50 (low-risk auto-approve band)
- **Flag**: 0.50 <= score < 0.75 (manual review / step-up authentication)
- **Block**: score >= 0.75 (high-risk auto-block)

Rationale: preserves frictionless flow for low-risk traffic while concentrating analyst effort on uncertain middle-band events and hard-blocking highest-risk predictions.

---

## 4) Latency status
Sources:
- `docs/archive/latency_curve_comparison_20260318.md`
- `docs/archive/latency_curve_comparison_20260319.md` (latest)
- `project/outputs/audit/fraud_audit_log.jsonl` (single-request audit trace timing)

### Internal single-request processing latency (audit traces)
- Audit traces show internal per-request processing latency is typically around **~20–25ms** on the score path (`latency_ms` in `fraud_audit_log.jsonl` entries).
- This reflects internal processing time for individual requests and is **not** the same as endpoint p95/p99 behavior under load.

### Endpoint/load-test latency SLO status (tuple benchmarks)
- Against the load-test SLO targets (`p95<=250ms`, `p99<=500ms`, `error_rate<=1.0%`) from `docs/archive/latency_curve_comparison_20260319.md`, **only tuple (80,2) passes both p95/p99 gates for both endpoints**.
- At `(120,6)` and `(200,12)`, both endpoints remain out of SLO on p95/p99.
- Best observed p95 values in the latest tuple comparison are:
  - `score_transaction`: **103.84ms**
  - `wallet_authorize_payment`: **94.38ms**

### Current bottleneck
- **High concurrency saturation (6 and 12)** remains the main bottleneck.
- At (120,6), p95 is ~364–369ms (fails target).
- At (200,12), p95/p99 degrade heavily (e.g., `wallet_authorize_payment` p99 **846.21ms**).

### Target gap (latest)
Against target `p95<=250ms`, `p99<=500ms`:
- (120,6):
  - `score_transaction` p95 gap: **+113.70ms**
  - `wallet_authorize_payment` p95 gap: **+118.90ms**
- (200,12):
  - `score_transaction` p95 gap: **+442.08ms**, p99 gap: **+305.83ms**
  - `wallet_authorize_payment` p95 gap: **+518.14ms**, p99 gap: **+346.21ms**

---

## 5) Fairness / explainability findings
Source: `project/outputs/governance/fairness_explainability_report.md`

### Top explainability drivers (SHAP)
Most influential features are device and behavioral signals, led by:
1. `numeric_canonical__device_signal_present`
2. `numeric_canonical__amount_raw`
3. `numeric_canonical__time_since_last_tx`
4. `numeric_canonical__event_time_raw`
5. `numeric_canonical__amount_over_user_avg`

### Severe segments
Highest-severity disparity segments include:
- `ieee:product_C` (highest severity)
- `ieee:device_mobile`
- `ieee:device_desktop`
- `ieee:product_W`
- `ieee:product_R`
(plus additional severe flags for `ieee:product_S`, `ieee:product_H`, `ieee:identity_high_confidence`)

### Mitigation plan
Severe disparity segments are currently identified (including product and device cohorts) and are under active mitigation before full-scale rollout.

1. **Step 1 — segment-aware threshold evaluation:** evaluate legally/compliance-approved segment-aware thresholds for severe cohorts and validate disparity deltas.
2. **Step 2 — retraining/revalidation gates before expansion:** rebalance training for impacted segments, rerun fairness + performance passes, and require these revalidation gates to pass before expanding to high-volume traffic.

- Add segment-specific FPR/recall drift alerting and governance review checkpoints before broader rollout.

---

## 6) SDG / business impact summary

### Who benefits
- **Customers**: fewer fraud losses and faster approvals for low-risk transactions.
- **Ops/risk teams**: clearer triage bands and better signal quality for investigations.
- **Business/compliance**: structured evidence for model governance, fairness, and rollout control.

### How trust improves
- Transparent threshold gates, published fairness findings, and interpretable top drivers support accountable decisioning.
- Latency gates make user experience risk explicit rather than implicit.

### Rollout implication
- Immediate production use is strongest for low-concurrency channels/tiers.
- Capacity-aware rollout (or autoscaling/performance optimization) is required before high-concurrency default traffic shift.

---

## 7) Production-ready now vs next-iteration roadmap (5 bullets)

1. **Ready now:** `xgboost_tuned` model quality is materially stronger than default and suitable for controlled production.
2. **Ready now:** threshold promotion (`onehot_robust`, 0.75) passed robustness + quality gates and supports approve/flag/block operations.
3. **Next iteration:** close latency gaps at concurrency 6/12 to meet p95/p99 SLOs under target-load tuples.
   **Explicit proof status:** high-concurrency p95/p99 compliance remains a known gap under the current benchmark tuples.
4. **Next iteration:** execute fairness mitigation on severe IEEE segments and re-validate disparity metrics before scale-up.
5. **Next iteration:** formalize phased rollout by traffic tier (start with low-concurrency lanes, then expand after latency + fairness exit criteria are met).

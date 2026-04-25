# Business Case: ASEAN Fraud Decisioning Platform

## One-Page Summary (Pitch Deck Transfer)

**Problem:** Digital wallet growth in ASEAN is outpacing fraud controls for underserved user segments (gig workers, rural merchants, and agent banking channels), creating measurable losses, customer churn, and compliance pressure.

**Solution:** A real-time fraud decisioning stack that combines ML scoring, contextual risk, and behavior profiling to reduce fraud losses while minimizing false positives and preserving transaction approval rates.

**Who pays:** Wallet operators, super-apps, acquirers, and agent banking networks that need improved risk control without degrading user experience.

**Primary user segments enabled by the product:**
- **Gig worker:** high transaction frequency, variable income cycles, sensitive to payment friction.
- **Rural merchant:** low digital trust baseline, intermittent connectivity, high dependence on wallet reliability.
- **Agent banking partner:** cash-in/cash-out and assisted onboarding endpoints with elevated mule and synthetic identity exposure.

**Why now:** ASEAN’s digital payments volume, financial inclusion mandates, and regulator attention to operational resilience make explainable, configurable anti-fraud infrastructure an immediate priority.

**Business model:**
1. Usage-based transaction scoring fee.
2. Tiered SaaS platform subscription for tooling and governance.
3. Fraud-loss-share option for mature deployments.

**12-month traction target:** Land 2 lighthouse customers, expand to 6 paid deployments, and reach positive gross margin contribution from combined usage + SaaS fees.

## Judge-Facing Evidence: What Already Exists

### Model candidate benchmark (IEEE-CIS, grouped holdout)

Quoted directly from `project/outputs/monitoring/ieee_cis_model_candidate_benchmark.csv`:

| Candidate | Precision | Recall | False Positive Rate (FPR) | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| xgboost_tuned | 0.921382 | 0.524852 | 0.001596 | 0.746345 |
| xgboost_default | 0.917096 | 0.438238 | 0.001412 | 0.675639 |

### Imbalance strategy experiments (IEEE-CIS, time split)

Quoted directly from `project/outputs/monitoring/ieee_cis_imbalance_strategy_benchmark.csv`:

| Strategy | Precision | Recall | FPR | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.225326 | 0.722441 | 0.088510 | 0.520113 |
| smote_then_undersample | 0.222167 | 0.441437 | 0.055075 | 0.331732 |
| random_undersample | 0.173763 | 0.581447 | 0.098523 | 0.387407 |
| smote | 0.149035 | 0.456201 | 0.092824 | 0.293845 |
| adasyn | 0.121912 | 0.599902 | 0.153976 | 0.298485 |

### Added pitch slide (explicit threshold trade-off)

**Slide title:** _Three operating points under policy constraints (instead of one static threshold)._

Constrained objective used for the threshold sweep on the tuned promoted setting (`onehot_robust`):
- **Optimize:** highest F1 (PR-AUC shown as ranking context).
- **Subject to:** `FPR <= 0.08` and `Recall >= 0.30`.

| Governance profile (switchable) | Block threshold | Precision | Recall | F1 | FPR | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conservative | 0.75 | 0.2207 | 0.3194 | **0.2611** | **0.0402** | 0.1858 |
| balanced | 0.70 | 0.1928 | 0.3880 | 0.2576 | 0.0579 | 0.1858 |
| aggressive | 0.65 | 0.1707 | **0.4498** | 0.2474 | 0.0779 | 0.1858 |

**Inference policy (demo):** locked to **balanced** with default **block threshold = 0.70**. **Conservative** and **aggressive** are switchable governance profiles (not simultaneous defaults).

---

## 1) Target Users + TAM/SAM/SOM Assumptions (ASEAN)

### Target users

### A) Gig worker
- Typical profile: ride-hailing/logistics, creator economy, freelance services.
- Risk pain points: account takeover, payout redirection, friendly fraud disputes.
- Product value: low-latency authorization support and behavior-based anomaly detection to keep legitimate payouts flowing.

### B) Rural merchant
- Typical profile: neighborhood stores, micro-retailers, QR-acceptance first adopters.
- Risk pain points: social engineering scams, account sharing, device compromise.
- Product value: contextual scoring that adapts to lower-data environments while reducing unnecessary blocks.

### C) Agent banking partner
- Typical profile: field agents/merchant agents supporting onboarding and cash operations.
- Risk pain points: identity fraud, collusive behavior, cash-out abuse patterns.
- Product value: segment-aware rules + explainable risk outputs to support agent oversight and audits.

### Market sizing framework

Assumptions are directional planning estimates for go-to-market design and fundraising narrative. Values below are annualized in USD.

| Metric | Definition | Assumption |
| --- | --- | --- |
| **TAM** | Total ASEAN serviceable fraud decisioning opportunity across wallet + embedded finance transaction volume needing risk scoring | **$1.6B/year** |
| **SAM** | Near-term reachable market (digital wallets/super-apps + agent-bank channels in priority ASEAN corridors) | **$420M/year** |
| **SOM (36 months)** | Realistic share with focused execution and partner-led distribution | **$18M/year** |

### Assumption logic (top-down + bottom-up blend)

1. **Top-down TAM logic**
   - ASEAN digital payment ecosystems process very large annual transaction counts; anti-fraud spend is modeled as a small basis-point share of protected volume plus platform subscription spend.
   - TAM proxy applies blended pricing and penetration to risk-relevant transactions in key ASEAN markets.

2. **SAM logic**
   - Filters TAM to product-adjacent buyers with immediate integration readiness:
     - wallet operators,
     - super-app payment rails,
     - agent banking program operators.
   - Excludes slower-moving institutional segments in first wave.

3. **SOM logic**
   - Built from sales capacity assumptions:
     - Year 1: 2 lighthouse accounts,
     - Year 2: 4 additional accounts,
     - mixed ACV from usage + subscription tiers,
     - progressive expansion of scored transaction volume per account.

---

## 2) Competitor Matrix + Differentiators

| Category | Typical strengths | Typical weaknesses | Our differentiators |
| --- | --- | --- | --- |
| **Rule-based systems** (internal/manual heuristics) | Fast to start, low initial tooling complexity, human-readable policies | Rule explosion, brittle adaptation, high analyst burden, weaker unknown-pattern detection | Hybrid ML + context + behavior scoring with policy controls that preserve explainability and operator override pathways |
| **Incumbent anti-fraud providers** (global enterprise suites) | Mature libraries, broad integrations, proven enterprise credibility | High cost, slower customization for local payment behaviors, integration overhead for smaller teams | ASEAN-focused segment tuning (gig/rural/agent), faster deployment paths, and pricing models aligned to emerging-market unit economics |
| **Wallet-native solutions** (in-house risk modules) | Tight platform integration, ownership of first-party data | Opportunity cost for product teams, uneven model governance, limited external benchmarking | Dedicated fraud stack with explicit governance artifacts, release gates, and explainability-focused operations for regulator-facing workflows |

### Positioning statement

We are not replacing policy teams; we are amplifying them with a configurable decision layer that improves fraud capture while controlling false positives in inclusion-sensitive segments.

---

## 3) Monetization Model + 12-Month Adoption Path

### Monetization model

### A) Per-transaction scoring
- Fee charged per scored authorization event.
- Example planning range: **$0.0015 - $0.006** per transaction (volume and SLA dependent).
- Best for high-scale wallet operators seeking performance-linked cost structure.

### B) Tiered SaaS subscription
- Platform access for policy management, analytics, explainability artifacts, and governance workflows.
- Example tiers:
  - **Growth:** baseline dashboards + alerts + support.
  - **Scale:** advanced segmentation, drift monitoring, integration support.
  - **Enterprise/Regulated:** dedicated controls, audit readiness workflows, stricter SLA terms.

### C) Fraud-loss-share (selective)
- Optional model where part of pricing is tied to measurable fraud loss reduction vs baseline.
- Requires jointly agreed measurement protocol and guardrails to avoid incentive misalignment.

### 12-month adoption path

| Phase | Timeline | Commercial objective | Product/ops objective |
| --- | --- | --- | --- |
| **Phase 1: Design partners** | Months 1-3 | Sign 2 lighthouse customers | Integrate core APIs, establish baseline KPIs, define success metrics |
| **Phase 2: Pilot to paid conversion** | Months 4-6 | Convert both pilots to paid contracts | Calibrate thresholds by segment, publish monthly model/risk reviews |
| **Phase 3: Replication** | Months 7-9 | Add 2-3 new customers via partner references | Package repeatable onboarding + controls templates |
| **Phase 4: Scale foundations** | Months 10-12 | Reach 6 paid customers total and expansion commitments | Formalize governance cadence, launch loss-share pilots where baseline quality is sufficient |

---

## 4) Unit Economics Assumptions: False-Positive Cost vs Fraud-Loss Avoided

### Core framing

Economic value is maximized when incremental fraud-loss reduction exceeds incremental customer friction costs from false positives and review operations.

### Planning assumptions (illustrative)

| Variable | Assumption |
| --- | --- |
| Monthly scored transactions per mid-size customer | 25M |
| Baseline fraud loss rate | 20 bps of GMV |
| Baseline false-positive rate (hard block/step-up causing abandonment) | 1.8% |
| Improved fraud loss rate after deployment | 14 bps |
| Improved false-positive rate after tuning | 1.3% |
| Average economic cost per false positive | $0.20 (lost margin, support, churn proxy) |

### Example impact calculation

1. **Fraud-loss avoided**
   - Improvement: 6 bps.
   - On a hypothetical $1.2B monthly GMV, avoided loss ≈ **$720K/month**.

2. **False-positive cost reduction**
   - Improvement: 0.5 percentage points.
   - On 25M monthly transactions, 125K fewer false positives.
   - At $0.20 each, recovered value ≈ **$25K/month**.

3. **Total gross economic benefit**
   - Approx. **$745K/month** before platform fees and internal ops costs.

### Decision guidance
- Prioritize segment-level threshold tuning (gig/rural/agent) to improve both fraud capture and approval lift.
- Measure unit economics monthly with transparent counterfactual baselines.

---

## 5) Risk & Compliance Considerations

### Data residency
- Support deployment patterns that keep sensitive data in-country where required (or within approved regional jurisdictions).
- Separate model telemetry from personally identifiable data where possible.
- Maintain clear data flow mapping for regulator and partner due diligence.

### Explainability expectations
- Provide transaction-level reason codes suitable for operations teams.
- Maintain model documentation, threshold rationale, and change logs for auditability.
- Ensure adverse-action style explanations can be adapted to local disclosure expectations.

### Operational governance
- Define model lifecycle controls:
  - versioning,
  - validation gates,
  - rollback procedures,
  - exception handling ownership.
- Establish periodic governance forums (risk, compliance, engineering, product).
- Track fairness/inclusion indicators to avoid disproportionate impact on underbanked populations.

### Regulatory posture (operating principle)
- Treat compliance as a product feature: configurable controls, evidentiary logs, and operator oversight by default.

---

## Appendix: KPI Set for Investors and Operators

- Fraud loss rate (bps of GMV)
- False-positive rate
- Approval rate / conversion impact
- Analyst review workload per 10K transactions
- Mean time to policy adjustment
- Explainability coverage (% of decisions with operator-usable reason codes)

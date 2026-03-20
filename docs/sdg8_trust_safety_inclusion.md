# SDG-8 Narrative: Trust, Safety, and Inclusion Outcomes for Unbanked Users

## Context

SDG-8 emphasizes inclusive economic participation. For unbanked users, digital wallets are often the first formal financial tool. Fraud events can permanently reduce trust and adoption, especially among first-time users.

## Trust outcomes

- **Transparent explanations**: Decision reasons are surfaced to support understandable outcomes.
- **Consistent policying**: Similar high-risk patterns follow consistent review/escalation paths.
- **Human override path**: Borderline and high-risk decisions are routed to analyst review rather than hard automation only.

## Safety outcomes

- **Early intervention**: High-risk takeover patterns are blocked or flagged before losses escalate.
- **Defense-in-depth**: Model risk + context risk + behavior risk + analyst review reduce single-point failures.
- **Operational monitoring**: Dashboard KPIs track latency, drift, false positives, and agreement so safety regressions are detected before broad rollout.

## Inclusion outcomes

- **Progressive friction**: Low-risk users proceed quickly; uncertain cases get verification instead of blanket denial.
- **Lower exclusion risk**: Analyst review and retraining curation specifically target reduction of avoidable false positives.
- **Policy feedback loop**: Analyst outcomes become curated retraining data, improving treatment for sparse-history users over time.

## Accountability signals tracked

- False positives and analyst agreement.
- Estimated fraud loss avoided.
- Review queue volume and resolution throughput.
- Drift and score-distribution shifts that may disproportionately impact new-to-digital populations.

## Narrative summary

This release balances access and protection by combining machine decisioning with a formal review queue and retraining feedback loop. For unbanked users, this supports safer onboarding, fewer avoidable denials, and more reliable trust in digital financial services.


## Locked decisioning profiles and SDG trust fit (2026-03-19)

The decisioning profiles are now frozen in `artifacts/config/decisioning_profiles.locked.json` with three operating modes:

- **Conservative** (`block_threshold=0.75`): lowest FPR, highest customer protection against wrongful blocks.
- **Balanced** (`block_threshold=0.70`): recommended judging mode and default for demo.
- **Aggressive** (`block_threshold=0.65`): highest recall for elevated threat windows.

The demo is explicitly frozen to **Balanced** in `artifacts/config/demo_decisioning_profile.locked.json`.

Why this matches the SDG trust objective:

- It protects users from fraud with materially stronger recall than conservative mode.
- It avoids the higher friction burden of aggressive mode, reducing avoidable false-positive harm for first-time and financially vulnerable users.
- It preserves transparent governance by hard-locking profile intent and thresholds as auditable artifacts.

# Fraud Ring Remediation Note

Use this note for judge Q&A and GitHub handoff. It summarizes the ring-system changes made for pre-demo credibility.

## What changed

- `/ring/graph` now prefers exact observed account-to-attribute evidence links from `fraud_ring_evidence_links.json`.
- If exact evidence links are unavailable, the API returns ring summaries and nodes with `links=[]` instead of fabricating implied relationships.
- Ring artifacts are now label-safe:
  - `label_mode=labeled` may emit `fraud_count` and `fraud_rate`
  - `label_mode=topology_only` emits structural evidence only
- The runtime ring scorer now supports:
  - `account_member` matches from `fraud_ring_scores.json`
  - `attribute_match` for unseen accounts via `fraud_ring_attribute_index.json`
- Noisy high-degree attributes are suppressed before they can influence scoring.
- Ring-only evidence is bounded by:
  - component-size gate
  - corroboration gate
  - artifact-recency gate
  - ring-only block fairness guard

## What we can honestly claim

- The graph visualization is now evidence-faithful when the evidence-link artifact is present.
- The ring builder is reproducible across `labeled` and `topology_only` modes.
- Unseen accounts can now inherit ring risk through shared risky attributes instead of relying only on account-level membership.
- Ring influence is constrained before it can escalate a transaction to `BLOCK` on guarded segments.
- The audit trail now records ring provenance, ring match type, evidence-gate outcomes, and fairness-guard outcomes.

## Which artifacts matter

- `project/outputs/monitoring/fraud_ring_reports.json`
- `project/outputs/monitoring/fraud_ring_scores.json`
- `project/outputs/monitoring/fraud_ring_evidence_links.json`
- `project/outputs/monitoring/fraud_ring_attribute_index.json`
- `project/outputs/monitoring/ring_ablation_report.*`
- `project/outputs/monitoring/ring_replay_report.*`
- `project/outputs/governance/ring_fairness_impact.*`

## How to present evidence

- Prefer `ring_replay_report.*` when labeled replay data exists.
- If only synthetic directional evidence is available, explicitly label `ring_ablation_report.*` as `synthetic_projection`.
- Use `ring_fairness_impact.*` to explain why the ring-specific fairness guard exists.

## Still future work

- Truly online graph updates instead of nightly artifact rebuilds
- Additional edge types such as account-to-account counterparties
- Supervised learned ring scoring on stable labeled graph artifacts
- Stronger replay evaluation on larger production-like labeled histories

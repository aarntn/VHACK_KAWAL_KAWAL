# Backend Health Narrative

Use these points for the technical presentation and Q&A. They are intentionally conservative.

## What the backend can prove live

- The fraud API exposes runtime thresholds, model metadata, MCP breaker status, privacy contract, ring graph data, and audit-derived analytics through `/health`, `/api/info`, `/privacy`, `/ring/graph`, and `/dashboard/views`.
- Operator surfaces can be protected with `FRAUD_OPERATOR_API_KEY` using the `X-Operator-Api-Key` header.
- A `/metrics` endpoint now exports plain-text operational metrics derived from live audit/review artifacts.
- The score and wallet APIs now return ASEAN-specific provenance: `corridor`, `normalized_amount_reference`, `normalization_basis`, `runtime_mode`, and region-aware `reason_codes`.
- The runtime supports the demo country set `SG`, `MY`, `ID`, `TH`, `PH`, and `VN` without adding live third-party dependencies during scoring.
- The ring graph endpoint now prefers exact observed account-to-attribute evidence links and falls back to summary-only nodes rather than fabricating links.
- The ring runtime now supports both known ring members and unseen accounts matched through the attribute index, with noisy-attribute suppression and a ring-only block fairness guard.

## Latency story

Use the scorecard and architecture docs for exact language:

- Internal single-request audit traces are typically around `~20–25ms` on the score path.
- Load-test evidence shows only tuple `(80,2)` passes the stated p95/p99 gates for both endpoints.
- The dominant bottleneck is preprocessing plus inference, which the architecture document attributes as roughly `90–97%` of score latency.

Say it this way:

> We are low-latency for the demo configuration, we measured the bottleneck explicitly, and our next scaling step is worker/container deployment plus preprocessing hot-path reduction.

For ASEAN framing:

> We added corridor and local-currency context without introducing blocking external calls. The demo uses a checked-in normalization artifact so latency stays stable and explanations stay reproducible.

## Fraud and review analytics story

From the live dashboard backend:

- requests in window
- throughput per minute
- p50 / p95 latency
- false positives
- confirmed fraud cases
- analyst agreement
- decision-source KPIs

Say it this way:

> The dashboard only uses backend-generated analytics for the operational views we present live. We removed silent frontend fallback for those KPI surfaces.

For ring credibility:

> The graph only shows exact observed evidence links when those artifacts exist. If they are missing, we degrade to summaries instead of inventing edges.

## ASEAN runtime story

Use these four examples because they are now backed by shared preset data and contract checks:

- `ID-ID` domestic QR payment in `IDR` -> `APPROVE`
- `SG-PH` first-time remittance in `SGD` -> `FLAG` / step-up
- `MY-MY` agent-assisted cash-out in `MYR` -> `FLAG`
- `TH-VN` repeated suspicious pattern in `THB` -> `BLOCK`

Say it this way:

> The ASEAN story is no longer just slideware. The runtime accepts corridor metadata, normalizes local currencies into a shared model reference, and records that provenance in both the response and the audit chain.

## Explainability and governance story

- Explainability is returned per request with base/context/behavior/ring/external contributions.
- Privacy-safe audit logs omit raw user IDs and raw feature vectors.
- Audit entries are hash-chained and signed with versioned secrets.
- Runtime artifact validation uses the promoted manifest plus compatibility checks.
- ASEAN provenance is audit-safe: raw identifiers remain excluded, while corridor, normalization artifact version, and explanation codes are recorded for reviewability.
- Ring provenance is audit-safe too: we now record `ring_match_type`, ring evidence summaries, evidence-gate decisions, and fairness-guard outcomes without storing raw identifiers.

## Safe claims to make

- “We have a governed runtime, not just a classifier.”
- “We can show exact live thresholds and audit-safe decision traces.”
- “We know preprocessing is the main latency bottleneck and can point to the measurement.”
- “We can show ASEAN-local scenarios with reproducible corridor and normalization provenance.”

## Claims to avoid

- Do not claim peak-scale readiness beyond the measured benchmark tuples.
- Do not claim live cohort fairness monitoring from the dashboard UI.
- Do not present sandbox/example values as production telemetry.

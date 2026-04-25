# Fraud Shield — Development Roadmap

## Current State (Hackathon Submission — April 2026)

A production-grade real-time fraud detection stack running on a single-node uvicorn deployment.
Core pipeline is complete: 5-component hybrid scorer, ring detection, MCP external signals,
async inference, HMAC audit chain, segment-level fairness calibration, nightly ops automation,
and an evidence-faithful ring layer with exact graph links plus attribute matching for unseen accounts.

---

## Phase 1 — Production Hardening (Next 30 days)

**Goal**: Make the single-node deployment production-safe under real traffic.

| Item | Description | Rationale |
|------|-------------|-----------|
| Multi-worker Linux deployment | Switch from Windows single-worker uvicorn to 4-worker gunicorn on Linux/container | p95 at c=6 drops from ~456ms to ~114ms (linear with workers) |
| Redis feature store | Swap `InMemoryFeatureStore` → `RedisFeatureStore` (protocol already defined) | Behavior profiles survive restarts; share state across workers |
| Redis profile store | Swap `InMemoryProfileStore` → `RedisProfileStore` | Same motivation; per-user EMA survives redeploy |
| Prometheus metrics export | Add `/metrics` endpoint with request count, p50/p95 per endpoint, circuit breaker state | Required for ops alerting without log scraping |
| Structured log → ELK/Loki | Forward JSON structured logs to a log aggregation service | Current JSON logs go to stdout only |
| Docker Compose production profile | Add resource limits, health-check restart policies, secrets via env-file | Current compose file is dev-only |

---

## Phase 2 — Model & Signal Expansion (30–90 days)

**Goal**: Improve fraud detection quality and reduce the fairness gaps identified in the audit.

| Item | Description | Rationale |
|------|-------------|-----------|
| Segment-specific retraining | Retrain with stratified oversampling per `ieee:product_C`, `ieee:device_mobile` (top 2 severity segments) | Addresses root cause of FPR/FNR disparity (threshold adjustment is a band-aid) |
| Online ring detection | Replace the current nightly account-score + attribute-index artifacts with incremental graph updates on each scored transaction | Current runtime can match unseen accounts by shared attributes, but truly new rings still wait for the next artifact rebuild |
| Graph edge types | Add account→account edges (shared P2P counterparties) on top of the current exact account↔attribute evidence graph | Current graph now preserves real account↔attribute provenance, but direct mule-to-mule edges could improve recall further |
| Velocity-based hard rules | Add rate-limit rules: N BLOCK decisions in M minutes from same device/IP triggers auto-escalation | Covers burst attack patterns not captured by model score |
| Live MCP signal quality check | Validate MCP response schema and score distribution on every call; circuit-break on unexpected distribution | Current circuit breaker only fires on network failure, not on bad data |
| Learned ring weights | Replace the current structural ring score with a supervised model trained on reproducible labeled ring artifacts | Today the runtime is label-safe and topology-only capable, but the scoring formula is still hand-tuned |

---

## Phase 3 — Platform Scale (90–180 days)

**Goal**: Multi-tenant, multi-region, real-time stream processing.

| Item | Description | Rationale |
|------|-------------|-----------|
| Kafka transaction stream | Ingest transactions from Kafka topic instead of HTTP POST | Decouples wallet from fraud engine; enables replay and exactly-once guarantees |
| Flink/Spark Streaming aggregations | Move 24h/7d behavioral aggregates from in-memory to Flink stateful operators | Enables sub-second aggregate freshness at scale; removes per-request aggregate recomputation |
| Cassandra feature store | Activate `CassandraFeatureStore` stub (protocol defined, not configured) | Required for multi-region replication of behavioral state |
| A/B model serving | Route X% of traffic to candidate model, compare FPR/FNR in real time | Needed to validate model updates without full rollout risk |
| Federated learning pilot | Train per-bank local model, aggregate gradients centrally (no raw data sharing) | Enables collaboration across ASEAN institutions without PII transfer |
| GDPR/PDPA compliance module | Per-user data deletion API, retention policy enforcement, audit log rotation | Required for regulatory compliance across Malaysia, Thailand, Indonesia |

---

## Phase 4 — ASEAN Expansion (6–12 months)

**Goal**: Localization and cross-border coverage.

| Item | Description | Rationale |
|------|-------------|-----------|
| Live FX normalization upgrade | Replace the current checked-in ASEAN normalization snapshot with a governed real-time FX/merchant corridor layer | Static normalization for SGD/MYR/IDR/THB/PHP/VND is now live for demo/runtime consistency; real-time FX remains future work |
| GovTech / eKYC integration | Connect to national digital identity APIs (MyDigital, SingPass) for identity confidence signals | Reduces `ieee:identity_high_confidence` FNR gap |
| Mobile network signal | Integrate SIM card tenure and carrier-change frequency as context features | `sim_change_recent` is currently a binary flag; carrier API gives richer signal |
| Embedded offline scoring SDK | Replace the current runtime degraded/offline-buffered safeguards with a lightweight ONNX model in the wallet SDK | Low-connectivity rural ASEAN contexts where API call latency is prohibitive |
| Regulatory reporting API | Automated SAR (Suspicious Activity Report) generation for Bank Negara / OJK / BOT | Fraud analytics team needs structured output, not just decision codes |

---

## Extension Points Already in Code

These are implemented but not yet activated in the default deployment:

| Hook | File | State |
|------|------|-------|
| `RedisFeatureStore` | `app/feature_store.py` | Implemented, not configured |
| `RedisProfileStore` | `app/profile_store.py` | Implemented, not configured |
| `CassandraFeatureStore` | `app/feature_store.py` | Stub, protocol defined |
| `OnnxOrHummingbirdBackend` | `app/inference_backends.py` | Implemented, selectable via `FRAUD_INFERENCE_BACKEND=onnx` |
| `segment_thresholds` | `app/hybrid_fraud_api.py` | Live, applied per cohort key |
| `FRAUD_SHAP_TOP_N` | `app/hybrid_fraud_api.py` | Live, per-request SHAP (default 5) |
| MCP async client | `app/mcp/watchlist_client.py` | Live, parallel watchlist+device lookup |
| Circuit breaker (fraud→MCP) | `app/mcp/watchlist_client.py` | Live, 5 failures → 30s open |
| Circuit breaker (wallet→fraud) | `app/wallet_gateway_api.py` | Live, 5 failures → 30s open |
| Nightly ops | `scripts/nightly_ops.py` | Live, fully automated |
| Evidence bundle | `scripts/build_evidence_bundle.py` | Live, timestamped archives |

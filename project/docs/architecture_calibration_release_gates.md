# Architecture, Calibration & Release Gates

## System Architecture

### 5-Component Hybrid Scorer

Every transaction flows through five additive components. The final risk score is the clipped
sum; each component is independently observable in the API response under `explainability`.

```
Request
  │
  ├─► [1] Base model score         XGBoost (inplace_predict, float32)
  │       feature_columns (17)     V1..V17, TransactionDT, TransactionAmt, + behavioral aggregates
  │
  ├─► [2] Context adjustment       device_risk_weight × device_risk_score
  │                                + ip_risk_weight × ip_risk_score
  │                                + location_risk_weight × location_risk_score
  │                                + channel/type/payee bonus/penalty weights (14 total)
  │
  ├─► [3] Behavior adjustment      entity-level EMA profile (tx_count_24h, avg_amount_24h,
  │                                p2p_counterparties_24h) vs observed values
  │                                TTL-cached per user_id, threading.Lock guarded
  │
  ├─► [4] Ring adjustment          bipartite graph lookup (account ↔ attribute edges)
  │                                ring_score × ring_weight, gated by size + density thresholds
  │                                RING_SCORE_THRESHOLD=0.40, RING_MIN_SIZE=3
  │
  └─► [5] External (MCP) signal    async watchlist + device lookup via httpx.AsyncClient
          FRAUD_MCP_BOOST_HIGH=0.08 / FRAUD_MCP_CLEAN_DISCOUNT=-0.02
          TTL cache (60s, 2000 entries), circuit breaker (5 failures → 30s open)

Final score = clip(sum of components, 0.0, 1.0)
Decision    = BLOCK if score ≥ block_threshold
              FLAG  if score ≥ approve_threshold
              APPROVE otherwise
              (hard rules and policy overrides checked first)
```

### Inference Pipeline (async)

```
POST /score_transaction
  │
  ├─ [event loop]   request parsing + validation         ~0.5ms
  ├─ [event loop]   tx_dict build + ring lookup          ~1ms
  ├─ [executor]     preprocessing transform + XGBoost    90–97% of latency
  │                 (run_in_executor → thread pool, releases GIL)
  ├─ [event loop]   context/behavior/ring adjustments    ~0.5ms
  ├─ [event loop]   await MCP async lookup               11–50ms (cached: <1ms)
  ├─ [event loop]   decision logic + response build      ~0.5ms
  └─ [async queue]  audit log write (non-blocking)       0ms on hot path
```

The preprocessing + inference step uses `asyncio.get_running_loop().run_in_executor(None, fn)`
to offload CPU work. This lets the event loop serve other requests during the ~100–400ms
model scoring window.

### Feature Preparation Detail

`feature_preparation_ms` (stage 1) = time to call `build_model_features_from_score_transaction_request()`,
which constructs a flat `tx_dict` from the Pydantic request model. This is pure Python attribute
access with no I/O.

`preprocessing_ms` (stage 2) = time to call `build_ml_input_preprocessed()`, which:
1. Calls `transform_runtime_record_with_bundle()` — applies the promoted preprocessing artifact
   (StandardScaler + categorical encoding via a sparse matrix transform)
2. Calls `np.take(..., feature_column_indices)` to reorder into model column order
3. Validates `np.isfinite()` on the result

Both stages run inside the executor. The 90–97% share of total latency attributed to these stages
in nightly ops benchmarks reflects XGBoost inference on Windows single-worker uvicorn, where the
GIL serializes multiple concurrent inferences. On Linux with 2+ gunicorn workers these stages
distribute across cores and p95 drops linearly.

---

## Calibration Pipeline

### Pass 1 — Data preparation
- `build_dataset_stage1_artifacts.py` — IEEE-CIS dataset loading, class balance audit
- `build_preprocessing_artifact.py` — fits scaler/encoder, saves versioned bundle
- `evaluate_preprocessing_settings.py` — grid search over preprocessing configs

### Pass 2 — Model training & selection
- `benchmark_ensemble_candidates.py` — compares XGBoost, LightGBM, RandomForest, ONNX
- `benchmark_inference_candidates.py` — evaluates runtime backends (inplace, ONNX, HB)
- `tune_model_candidates.py` — hyperparameter sweep (Bayesian or grid)
- `promote_best_preprocessing_setting.py` — promotes winning config by ROC-AUC + inference speed

### Pass 3 — Threshold tuning
- `threshold_tuning.py` — sweeps approve/block at global level
- `calibrate_context.py` — tunes 14 context weights against validation set
- `calibrate_segment_thresholds.py` — computes per-segment FPR/FNR at candidate thresholds
- `apply_fairness_segment_thresholds.py` — writes segment overrides to deployed threshold config
- `promote_thresholds.py` — hashes and promotes the threshold bundle

### Pass 4 — Validation & artifact integrity
- `artifact_compatibility.py` — schema contract check between preprocessing bundle and model
- `validate_artifact_compatibility.py` — checksums against promoted manifest
- `validate_inference_backend_parity.py` — verifies all backends give same predictions
- `adversarial_validation.py` — ensures train/test indistinguishable (no leakage)

### Pass 5 — Release gate
- `release_gate_check.py` — asserts 3 consecutive nightly GO passes, 0% error rate
- `build_evidence_bundle.py` — archives all artifacts, reports, checksums into a dated bundle
- `backend_go_no_go_report.md` — human-readable decision with p95/p99 per endpoint

---

## Release Gates Summary

| Gate | Criterion | Pass threshold |
|------|-----------|---------------|
| Error rate | `error_rate_pct` | ≤ 0.0% (nightly ops) |
| p95 latency | `score_transaction` | ≤ 7000ms (relaxed demo SLA) |
| p99 latency | `score_transaction` | ≤ 9000ms |
| Artifact integrity | SHA256 checksums | Exact match to promoted manifest |
| Fairness gate | FPR gap per segment | ≤ 0.08 (segments with segment thresholds applied) |
| Fairness gate | FNR gap per segment | ≤ 0.12 |
| Consecutive runs | Nightly GO passes | ≥ 3 consecutive before release |
| Audit chain | HMAC signature chain | Unbroken (each record signs previous hash) |

---

## Component-Level Artifact Versioning

All artifacts carry SHA256 checksums in `promoted_artifact_manifest.json`:

```json
{
  "model_file": "models/final_xgboost_model_promoted_preproc.pkl",
  "model_file_sha256": "<hash>",
  "feature_file": "models/feature_columns_promoted_preproc.pkl",
  "preprocessing_bundle_file": "models/preprocessing_bundle.pkl",
  "threshold_file": "models/segment_thresholds_promoted.pkl"
}
```

Any mismatch between disk and manifest is caught at startup and raises `ArtifactSchemaMismatchError`,
preventing a bad model from serving traffic.

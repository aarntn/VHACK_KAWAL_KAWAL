# Tuple latency comparison (requests/concurrency = 80/2, 120/6, 200/12)
- Generated at: 2026-03-19T23:48:04+00:00
- Before baseline source: `docs/archive/tuple_benchmark_rerun_20260319.md`.
- After benchmark JSON sources:
  - `project/outputs/benchmark/latency_benchmark_20260319T234716Z.json` (80/2)
  - `project/outputs/benchmark/latency_benchmark_20260319T234737Z.json` (120/6)
  - `project/outputs/benchmark/latency_benchmark_20260319T234804Z.json` (200/12)

## score_transaction

| Requests | Concurrency | p95 before (ms) | p95 after (ms) | Δ p95 (ms) | p99 before (ms) | p99 after (ms) | Δ p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 2 | 287.15 | 166.60 | -120.55 | 442.64 | 184.18 | -258.46 |
| 120 | 6 | 927.20 | 600.58 | -326.62 | 1085.31 | 710.37 | -374.94 |
| 200 | 12 | 2360.63 | 1029.44 | -1331.19 | 2517.78 | 1220.81 | -1296.97 |

## wallet_authorize_payment

| Requests | Concurrency | p95 before (ms) | p95 after (ms) | Δ p95 (ms) | p99 before (ms) | p99 after (ms) | Δ p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 2 | 631.06 | 169.20 | -461.86 | 1014.71 | 181.63 | -833.08 |
| 120 | 6 | 1326.17 | 590.00 | -736.17 | 1679.96 | 679.95 | -1000.01 |
| 200 | 12 | 1679.26 | 1665.42 | -13.84 | 1732.55 | 1689.93 | -42.62 |

## Stage bottleneck focus check (score path)

| Tuple | feature_preparation_ms mean (ms) | preprocessing_ms mean (ms) | preprocessing share of total (%) |
|---|---:|---:|---:|
| (80,2) | 0.05 | 83.99 | 89.88 |
| (120,6) | 0.06 | 183.25 | 91.65 |
| (200,12) | 0.06 | 304.01 | 93.14 |

Notes:
- `feature_preparation_ms` is now measured independently from `preprocessing_ms` to remove duplicate stage accounting in the fraud path.
- `preprocessing_ms` remains the dominant score-stage bottleneck and is the optimization target for next cycle.

## Tuple gate status

- Target: `p95 <= 250 ms`, `p99 <= 500 ms`, `error_rate <= 1.0%`.
- Result:
  - `(80,2)` passes both endpoints.
  - `(120,6)` fails both endpoints on p95/p99.
  - `(200,12)` fails both endpoints on p95/p99.

## Locked benchmark evidence (submission)

| Tuple | JSON artifact | benchmark_run_id | generated_at_utc | sha256 |
|---|---|---|---|---|
| (80,2) | `latency_benchmark_20260319T234716Z.json` | `3f60d2b6-ae57-447e-8507-3529ffef7164` | 2026-03-19T23:47:16.740558+00:00 | `18497f17dcd29cbf445973e56a1dfb581f1e62acd34d033af45b5d9b67a1fdc5` |
| (120,6) | `latency_benchmark_20260319T234737Z.json` | `d95e45a6-0c1e-4df2-ba20-88c2d007a45d` | 2026-03-19T23:47:37.501112+00:00 | `9b445e92c9f293d8de9ee855190e7ca2b6b29ffe7377b9d0360fb58c0ad8cc80` |
| (200,12) | `latency_benchmark_20260319T234804Z.json` | `222dd166-4774-4e23-9113-53b425666d59` | 2026-03-19T23:48:04.537746+00:00 | `1b54ba3a8ba1692511d67a80d5ff28e33f93afadb4104308b0e1de677bffa6af` |

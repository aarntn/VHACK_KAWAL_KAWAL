# Latency curve comparison (concurrency sweep 2/6/12)
- Generated at: 2026-03-18T22:07:57.180534+00:00
- Source benchmark JSON: `project/outputs/benchmark/latency_benchmark_20260318T220757Z.json`

## score_transaction
| Concurrency | p95 (ms) | p99 (ms) | Throughput (rps) | Error rate (%) |
|---:|---:|---:|---:|---:|
| 2 | 173.06 | 219.62 | 21.97 | 0.00 |
| 6 | 617.25 | 668.45 | 20.10 | 0.00 |
| 12 | 1760.55 | 2023.79 | 15.86 | 0.00 |

Trend: p95 changed from 173.06ms at c=2 to 1760.55ms at c=12; p99 changed from 219.62ms to 2023.79ms.

## wallet_authorize_payment
| Concurrency | p95 (ms) | p99 (ms) | Throughput (rps) | Error rate (%) |
|---:|---:|---:|---:|---:|
| 2 | 219.19 | 250.79 | 17.06 | 0.00 |
| 6 | 525.96 | 639.74 | 19.98 | 0.00 |
| 12 | 1660.69 | 1667.24 | 10.40 | 0.00 |

Trend: p95 changed from 219.19ms at c=2 to 1660.69ms at c=12; p99 changed from 250.79ms to 1667.24ms.

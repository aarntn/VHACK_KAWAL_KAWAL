# Threshold Operating-Point Comparison (2026-03-19)

Constrained sweep objective: maximize F1 under `FPR <= 0.08` and `Recall >= 0.30` using `onehot_robust` threshold sweep output.

| Profile | Threshold | Precision | Recall | F1 | FPR | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conservative | 0.75 | 0.2207 | 0.3194 | 0.2611 | 0.0402 | 0.1858 |
| balanced | 0.70 | 0.1928 | 0.3880 | 0.2576 | 0.0579 | 0.1858 |
| aggressive | 0.65 | 0.1707 | 0.4498 | 0.2474 | 0.0779 | 0.1858 |

Inference policy: default to **balanced** for demo; switch to **conservative** if business requires stricter FPR.

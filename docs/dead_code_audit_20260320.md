# Dead Code / Redundancy Audit — 2026-03-20

Scope: quick submission-hardening audit focused on `creditcard.csv` legacy usage and redundant paths.

## Confirmed still in use (do **not** delete in final rush)

- `project.data.dataset_loader.load_creditcard` is still imported by multiple active scripts (`project/scripts/*`) and tests (`project/tests/fraud/*`).
- `project/models/final_xgboost_model.py` still supports `--dataset-source creditcard` and is referenced in docs and runtime guidance.
- Nightly ops still has legacy creditcard compatibility mode in script logic.

## Cleanups already done

- Removed obsolete one-off scripts with hardcoded local Windows paths:
  - `project/models/xgboost_model.py`
  - `project/models/save_final_model.py`
  - `project/scripts/find_flag_case.py`
  - `project/legacy_creditcard/eda.py`

## Additional low-risk quality cleanup done in this pass

1. Updated CI nightly workflow dataset path from `creditcardfraud/creditcard.csv` to `project/legacy_creditcard/creditcard.csv` for consistency with documented legacy location.
2. Replaced machine-specific Windows path example in README (`D:\Vhack\...`) with a generic placeholder path.

## Recommended for separate follow-up PR (not this final rush)

1. Introduce a `project/legacy_creditcard/tools/` namespace and move creditcard-only helper scripts there.
2. Keep deprecation wrappers for one release cycle so existing commands do not break.
3. Then remove wrappers and prune creditcard branches from scripts/tests if IEEE-CIS-only strategy is finalized.

## Submission guidance

- For judges: present IEEE-CIS path as default and call creditcard support “legacy compatibility mode”.
- Keep archive docs as appendix evidence, not as primary source unless they are the latest dated artifact.

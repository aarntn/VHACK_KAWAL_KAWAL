# Dead Code Cleanup — 2026-03-20

This cleanup removes obsolete one-off scripts that were not referenced by the current README runbook, CI workflows, or active training/inference paths.

## Removed files

1. `project/models/xgboost_model.py`
   - Legacy local-path training script with hardcoded Windows path assumptions.
2. `project/models/save_final_model.py`
   - Legacy helper script with hardcoded Windows path assumptions.
3. `project/scripts/find_flag_case.py`
   - One-off local investigation script not used by CI or deployment flows.
4. `project/legacy_creditcard/eda.py`
   - Legacy exploratory notebook-style script with hardcoded local path assumptions.

## Safety checks performed

- Searched repository references to removed filenames to confirm no active references in README, docs, scripts, tests, app code, or workflows.
- Ran a focused pytest module to ensure cleanup did not introduce import-level breakage tied to deleted files.

## Follow-up (separate push)

- Run full test suite in CI-like environment (`PYTHONPATH` + dataset fixtures) and capture before/after artifact diff for final release notes.

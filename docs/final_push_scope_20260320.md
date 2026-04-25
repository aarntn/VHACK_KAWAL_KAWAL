# Final Push Scope (Clean Submission) — 2026-03-20

This guide is for a **clean, judge-friendly final push** and separates what to ship now vs what to postpone.

## Pass now (keep in final push)

1. **Core runtime + contracts**
   - `project/app/` APIs, contracts, rules, stores, inference backends.
2. **Front-end demo app**
   - `project/frontend/` and deployment docs.
3. **Single source-of-truth scorecard + business docs**
   - `docs/evaluation_scorecard.md`, `docs/business_case.md`, `docs/privacy_governance.md`, `docs/sdg8_trust_safety_inclusion.md`.
4. **Locked decisioning config used by demo**
   - `artifacts/config/demo_decisioning_profile.locked.json`
   - `artifacts/config/decisioning_profiles.locked.json`

## Ignore for judging narrative (do not delete in this push)

1. **Legacy creditcard compatibility paths** still present for backward compatibility:
   - `project/models/baseline_model.py`
   - `project/models/class_imbalance_experiments.py`
   - Creditcard branches in scripts under `project/scripts/`.
2. **Historical archive docs** used as evidence trail:
   - `docs/archive/*`
3. **Nightly workflow creditcard branch** retained for compatibility:
   - `.github/workflows/nightly-ops.yml`

## Move to legacy in a separate push (safe follow-up)

If you want a stricter repo later, do this in a dedicated follow-up PR with full CI:

1. Move creditcard-only training helpers to `project/legacy_creditcard/tools/`.
2. Keep thin wrappers (or deprecation notices) for old entrypoints for one release cycle.
3. After one release cycle, remove wrappers and update README commands.

## What is likely irrelevant now

Already removed in prior cleanup:
- `project/models/xgboost_model.py`
- `project/models/save_final_model.py`
- `project/scripts/find_flag_case.py`
- `project/legacy_creditcard/eda.py`

## Tables/figures freshness rule for final deck

Use this rule:

1. **Use latest dated artifact** if multiple versions exist.
   - Example: use `docs/archive/latency_curve_comparison_20260319.md` over `20260318`.
2. **If table is not regenerated after major config/model change, mark as historical**.
3. **Only pitch numbers that are present in `docs/evaluation_scorecard.md`**.

## Final push do/don't checklist

### Do
- Keep README commands aligned with currently supported flows.
- Keep one canonical metric source (scorecard) and point slides to it.
- Keep archived evidence but treat it as appendix.

### Don't
- Don’t delete compatibility scripts and tests in the same PR as presentation cleanup.
- Don’t mix refactors + metric regeneration in one rushed push.
- Don’t quote stale table values without a date label.

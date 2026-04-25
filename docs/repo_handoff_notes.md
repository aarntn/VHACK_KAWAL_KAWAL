# Repo Handoff Notes

This repo is being kept conservative for the demo and GitHub handoff.

## Primary narrative

- IEEE-CIS + current fraud runtime is the primary path.
- The React dashboard is part of the demo surface.
- Governance, privacy, and latency evidence live under `docs/` and `project/outputs/`.
- ASEAN-local runtime behavior now lives in source and docs, not just in presentation material.

## Legacy paths kept on purpose

These remain for backward compatibility and should not be deleted in the final demo rush:

- `project/models/baseline_model.py`
- `project/models/class_imbalance_experiments.py`
- creditcard branches in active scripts/tests
- archive docs in `docs/archive/`

## Cleanup done in this pass

- root `.DS_Store` removed
- `.gitignore` expanded for macOS/editor noise and root `outputs/`
- dashboard analytics contract tightened to prefer live backend data over silent mock fallback
- docs added for demo-safe backend checks and evidence-backed claims
- static ASEAN normalization artifact added under `project/data/`
- preset contract checks updated to use the ASEAN demo scenario pack

## Runtime files

These paths are runtime-local artifacts and should be ignored in normal local development:

- `project/outputs/audit/fraud_audit_log.jsonl`
- `project/outputs/behavior_profiles.sqlite3`

If an older clone still has them tracked, remove them from the Git index once with `git rm --cached -- ...` and keep the local copies on disk.

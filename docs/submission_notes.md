# Submission Notes

## Frontend setup

1. `cd project/frontend`
2. `npm install`

## Packaging checklist

- Remove `project/frontend/node_modules` before creating zip/upload package.
- Include only dependency manifests for reproducibility:
  - `project/frontend/package.json`
  - `project/frontend/package-lock.json`
- Do **not** include `project/frontend/node_modules/`.

## Clean submission repository recipe

1. Create and switch to a dedicated branch (example: `submission-clean`).
2. Build a clean folder/repo and copy only:
   - `project/app`, `project/data`, `project/models`
   - `project/scripts`
   - `project/tests`
   - `project/frontend/src`, `project/frontend/public`, and frontend config files (`package.json`, lockfile, Vite/TypeScript configs)
   - `docs/` key docs (exclude `docs/archive`)
   - root files: `README(.md)`, `requirements.txt`, `docker-compose.yml`, `Dockerfile*`
3. Do **not** copy `.venv`, `project/outputs`, `project/frontend/node_modules`, or `.git`.
4. Initialize git in the clean folder, commit, and push to the submission remote.
5. Add this README note:

> Artifacts omitted intentionally; reproducible via scripts.

## Locked tuple benchmark evidence (2026-03-19)

- Final before/after tuple comparison and locked benchmark hashes are recorded in:
  - `docs/archive/latency_curve_comparison_20260319.md`
- Evidence references benchmark runs for `(80,2)`, `(120,6)`, and `(200,12)` with:
  - benchmark run IDs
  - UTC generation timestamps
  - SHA-256 checksums

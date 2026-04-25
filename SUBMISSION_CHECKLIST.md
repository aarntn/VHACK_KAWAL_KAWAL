# Submission Checklist

Use this checklist before final hackathon submission.

## 1) Required Links
- [ ] Public repository link is accessible.
- [ ] Demo video link is accessible without requesting permission.
- [ ] Any deck/slides link is accessible.
- [ ] If forms require specific URLs, pasted links are final and tested.

## 2) Demo Video Naming & Content
- [ ] Video filename follows organizer-required naming format.
- [ ] Team name and project name are clearly shown.
- [ ] Problem statement, solution flow, and live demo are included.
- [ ] Runtime decision outputs (`APPROVE` / `FLAG` / `BLOCK`) are shown.
- [ ] Video duration is within allowed limit.

## 3) Deadline & Timezone Checks
- [ ] Submission deadline date verified on the official portal.
- [ ] Deadline timezone confirmed (convert to local timezone explicitly).
- [ ] Final upload target time set at least 2 hours before deadline.
- [ ] All assets uploaded and re-opened for final verification.

## 4) Final QA
- [ ] README is up to date and includes setup/run instructions.
- [ ] Broken links check completed.
- [ ] Team contact details in submission form are correct.
- [ ] Final submission confirmation screenshot saved.
- [ ] `.\.venv\Scripts\python.exe project/scripts/preset_contract_check.py` passes with the ASEAN demo presets.
- [ ] Demo presets shown in the recording use ASEAN-local currencies/corridors (`ID-ID`, `SG-PH`, `MY-MY`, `TH-VN`).
- [ ] `.\.venv\Scripts\python.exe project/scripts/ring_fairness_impact.py` runs successfully and the report is referenced consistently.
- [ ] If the live dashboard would otherwise look empty, run `.\.venv\Scripts\python.exe project/scripts/seed_demo_runtime.py --reset-runtime` before recording and confirm `project/outputs/monitoring/demo_runtime_seed_summary.json` looks sane.
- [ ] If presenting ring metrics, use `ring_replay_report.*` when labeled replay exists; otherwise explicitly label `ring_ablation_report.*` as synthetic projection.
- [ ] `/ring/graph` demo is using exact evidence links (`evidence_links_available=true`) or is clearly described as summary-only fallback.


## 5) Final Push Scope Hygiene
- [ ] Keep only judge-facing claims that exist in `docs/evaluation_scorecard.md`.
- [ ] Mark archive tables with explicit dates when used in slides.
- [ ] If keeping creditcard compatibility code, label it as legacy/non-default.
- [ ] Avoid deleting compatibility code and refactoring in the same last-minute PR.
- [ ] Follow `docs/final_push_scope_20260320.md` for pass/ignore/move decisions.
- [ ] Reference `docs/asean_demo_note.md` and `docs/asean_fairness_summary.md` for ASEAN-specific technical claims.
- [ ] Reference the ring remediation updates consistently: exact evidence links, label-safe artifacts, attribute matching for unseen accounts, and ring-only block fairness guard.

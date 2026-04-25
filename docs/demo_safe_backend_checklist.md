# Demo-Safe Backend Checklist

Use this checklist before the demo so the dashboard and talking points stay aligned with real backend outputs.

## 1. Start services

```powershell
python project\scripts\launch_services.py
```

Expected:
- Fraud API on `http://127.0.0.1:8000`
- Wallet API on `http://127.0.0.1:8001`
- Mock MCP on `http://127.0.0.1:8002` unless `--no-mcp` is used

## 1.5 Seed believable live demo activity

If you want the dashboard, queue, and wallet flow to look active instead of empty:

```powershell
.\.venv\Scripts\Activate.ps1
python project\scripts\seed_demo_runtime.py --reset-runtime
```

Reference:
- `docs/live_demo_runtime_seed_plan.md`
- summary output: `project/outputs/monitoring/demo_runtime_seed_summary.json`

## 2. Verify health endpoints

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/ready
curl http://127.0.0.1:8000/api/info
curl http://127.0.0.1:8000/privacy
curl http://127.0.0.1:8001/health
```

Expected:
- `/health` returns `status=ok` plus thresholds and MCP status
- `/health/ready` confirms promoted artifacts and audit directory are present
- `/api/info` exposes runtime model info and thresholds
- `/privacy` shows audit/privacy contract
- `/ring/graph` is verified in the operator-only step below

## 3. Verify operator-only analytics surfaces

Operator analytics are locked by default unless `FRAUD_OPERATOR_AUTH_MODE=disabled`, so include:

```powershell
$headers = @{ "X-Operator-Api-Key" = $env:FRAUD_OPERATOR_API_KEY }
Invoke-RestMethod http://127.0.0.1:8000/dashboard/views -Headers $headers
Invoke-RestMethod http://127.0.0.1:8000/config -Headers $headers
Invoke-WebRequest http://127.0.0.1:8000/metrics -Headers $headers
Invoke-RestMethod http://127.0.0.1:8000/ring/graph -Headers $headers
```

Expected:
- `/dashboard/views` returns live audit-derived analytics
- `/config` returns runtime contract details and artifact metadata
- `/metrics` returns plain-text operational metrics
- `/ring/graph` returns ring nodes, links, and summaries for the protected analytics surface
- the dashboard Access panel accepts the operator access code at runtime without bundling it into the frontend build

## 4. Run one end-to-end decision for each class

Use one example each for:
- `APPROVE`
- `FLAG` / step-up
- `BLOCK`

Check:
- decision
- runtime mode (`primary` / `cached_context` / `degraded_local`)
- corridor
- normalized amount reference + basis
- final risk score
- reasons
- stage timings
- audit trail side effects

Recommended ASEAN demo set:
- `ID-ID` QR merchant payment in `IDR` should land on `APPROVE`
- `SG-PH` first remittance in `SGD` should land on `FLAG` and wallet `PENDING_VERIFICATION`
- `MY-MY` agent-assisted cash-out in `MYR` should land on `FLAG`
- `TH-VN` suspicious transfer in `THB` should land on `BLOCK`

## 5. Confirm dashboard is using live backend data

Before presenting:
- refresh dashboard after backend starts
- confirm request volume and p95 latency appear
- confirm ring graph loads from `/ring/graph`
- confirm the scorer/result view shows the expected runtime mode label instead of a stale previous run
- if analytics do not load, do not present dashboard KPI cards as live

## 6. Git hygiene before push

Before committing:

```powershell
git status
```

Expected after the cleanup pass:
- `project/outputs/audit/fraud_audit_log.jsonl` should stay ignored as a runtime-local artifact
- `project/outputs/behavior_profiles.sqlite3` should stay ignored as a runtime-local artifact

If an older clone still tracks those files, untrack them once and keep the local working copies:

```powershell
git rm --cached -- project/outputs/audit/fraud_audit_log.jsonl project/outputs/behavior_profiles.sqlite3
```

If `behavior_profiles.sqlite3` is locked during inspection or cleanup, stop the running service first.

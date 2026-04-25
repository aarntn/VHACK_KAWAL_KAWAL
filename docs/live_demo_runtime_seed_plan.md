# Live Demo Runtime Seed Plan

Use this plan to make localhost feel like a real B2B wallet-fraud product instead of an empty form demo.

The seed flow does **not** fake dashboard numbers. It:
- seeds realistic behavior history into the local profile store
- sends a repeatable batch of wallet authorization requests through the real wallet API
- lets the fraud API generate the real audit log, review queue, and dashboard metrics

## What gets seeded

### 1. Named demo users

These are the users you can talk about in the presentation.

| User ID | Role in demo | Notes |
|---|---|---|
| `user_id_qr_local` | Safe domestic QR payer | Indonesia local wallet, everyday merchant payments |
| `user_id_qr_commuter` | Safe domestic QR payer | Smaller routine QR transactions |
| `user_id_qr_market` | Safe domestic QR payer | Slightly higher-ticket domestic merchant pattern |
| `user_sgph_new` | Borderline remittance user | First-time `SG-PH` remittance, meant to step up |
| `user_agent_my` | Agent-assisted cash-out user | `MY-MY` agent flow with intermittent connectivity |
| `user_thvn_block` | Clear high-risk user | Repeated suspicious `TH-VN` transfer pattern |
| `account_00119` | Known ring member | Present in the checked-in ring artifact for live ring-linked scoring |

### 2. Background behavior history

Default seed:
- `48` synthetic background users
- `12` historical transactions each
- `576` behavior-history events total

Purpose:
- makes the behavior-profile store feel populated
- avoids a “fresh empty system” feeling
- keeps the demo deterministic and lightweight

### 3. Live transaction batch

Default live batch:
- `24` wallet authorization events total

Breakdown:

| Story | Count | Expected result | Why it matters |
|---|---:|---|---|
| Domestic `ID-ID` QR merchant payments | 8 | `APPROVE` | Shows low-friction local wallet flow |
| First-time `SG-PH` remittance | 4 | `FLAG` / `PENDING_VERIFICATION` | Shows progressive friction instead of blunt denial |
| `MY-MY` agent-assisted cash-out | 4 | `FLAG` / `PENDING_VERIFICATION` | Shows agent + intermittent-connectivity context |
| Repeated `TH-VN` suspicious transfer pattern | 4 | `BLOCK` | Shows strong high-risk intervention |
| Known ring member escalation (`account_00119`) | 4 | `BLOCK` | Shows live ring-linked reasoning |

This gives you:
- multiple `APPROVE`
- multiple `FLAG`
- multiple `BLOCK`
- a populated review queue
- enough dashboard traffic to look alive

## Exact commands

### Terminal 1: start services

```powershell
cd C:\Users\kikee_07xsqul\Downloads\VHACK\Vhack
.\.venv\Scripts\Activate.ps1
python project\scripts\launch_services.py
```

### Terminal 2: seed the live demo state

```powershell
cd C:\Users\kikee_07xsqul\Downloads\VHACK\Vhack
.\.venv\Scripts\Activate.ps1
python project\scripts\seed_demo_runtime.py --reset-runtime
```

This writes a summary to:

```text
project/outputs/monitoring/demo_runtime_seed_summary.json
```

### Optional: resolve a couple of review cases for analyst-flow storytelling

If operator auth is enabled:

```powershell
cd C:\Users\kikee_07xsqul\Downloads\VHACK\Vhack
.\.venv\Scripts\Activate.ps1
python project\scripts\seed_demo_runtime.py --resolve-reviews 2 --operator-api-key $env:FRAUD_OPERATOR_API_KEY
```

## What screens should now look populated

- `/`
  - manual scoring still works for live input
- `/dashboard`
  - request volume
  - latency metrics
  - decision-source metrics
  - fraud / false-positive / analyst-agreement summaries from real audit-derived data
- `/rings`
  - existing ring artifact remains available, while seeded transactions add ring-linked score examples in the live API responses
- review workflow
  - `FLAG` and `BLOCK` cases automatically create queue entries

## Best presentation order

1. Show one `APPROVE` example from the main user flow.
2. Show one `FLAG` example and explain step-up verification.
3. Show one `BLOCK` example and explain why it is high confidence.
4. Open the dashboard and show that the system has live traffic, not just one manual request.
5. Open the ring view and explain that graph intelligence is one layer inside the hybrid scorer.

## Notes

- The seeding script uses the real wallet API path, so the live dashboard state comes from your actual backend behavior.
- `FLAG` and `BLOCK` cases automatically populate the review queue because the fraud API queues all such cases for analyst review.
- If operator auth is enabled, enter the access code in the dashboard Access panel before showing `/dashboard` or `/rings`.

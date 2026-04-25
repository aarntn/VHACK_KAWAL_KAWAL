# ASEAN Context Layer — Integration Sandbox (/demo)

**Date:** 2026-04-24  
**Status:** Approved  
**Scope:** `project/frontend/src/IntegrationSandbox.tsx` only

---

## Goal

The ASEAN runtime context layer (corridor, source/destination country, connectivity mode, agent-assisted) already exists in the backend and in the main user flow (`UserForm.tsx`). It is completely absent from the Integration Sandbox at `/demo`. This spec adds it so that judges see the ASEAN differentiator on the primary demo page.

---

## Architecture

All changes are confined to `IntegrationSandbox.tsx`. No new files, no API changes.

The sandbox uses local offline scoring (`deriveRiskScoreFromForm`) — the ASEAN fields feed into that scoring function so their effect is immediately visible in the live result panel.

---

## Changes

### 1. Types

Add two type aliases at the top of the file (matching `UserForm.tsx`):

```ts
type CountryCode = "SG" | "MY" | "ID" | "TH" | "PH" | "VN";
type ConnectivityMode = "online" | "intermittent" | "offline_buffered";
```

Extend `SandboxFormState` with four new fields:

```ts
sourceCountry: CountryCode;
destinationCountry: CountryCode;
isAgentAssisted: boolean;
connectivityMode: ConnectivityMode;
```

---

### 2. State initialisation

`createInitialFormState()` reads new fields from `SCENARIO_PRESETS.everyday_purchase.ui`:

```ts
sourceCountry: basePreset.ui.source_country ?? "ID",
destinationCountry: basePreset.ui.destination_country ?? "ID",
isAgentAssisted: basePreset.ui.is_agent_assisted ?? false,
connectivityMode: basePreset.ui.connectivity_mode ?? "online",
```

---

### 3. Scenario preset application

`applyScenarioPreset()` maps preset UI fields into the new state fields for all non-custom scenarios. For `custom`, default to `"SG"` / `"SG"` / `false` / `"online"` (same pattern as UserForm).

---

### 4. Form UI — Payment Details card (PROMINENT placement)

**This is the emphasis.** The ASEAN fields are placed directly in the Payment Details card, immediately after the cross-border checkbox. They are always visible — not collapsed, not behind a toggle.

Layout (matching `UserForm.tsx` exactly):

**Row 1 — Countries (2-col grid, `mt-4 gap-4`):**
- Left: `FieldLabel` "Source country" + `FieldSelect` for `sourceCountry` (all 6 ASEAN codes)
- Right: `FieldLabel` "Destination country" + `FieldSelect` for `destinationCountry`, **disabled + opacity 0.7 when `isCrossBorder` is false** (mirrors `sourceCountry` when domestic)

**Row 2 — Mode + Agent (2-col grid, `mt-4 gap-4`):**
- Left: `FieldLabel` "Connectivity" + `FieldSelect` for `connectivityMode`:
  - `online` → "Live"
  - `intermittent` → "Intermittent"
  - `offline_buffered` → "Degraded"
- Right: agent-assisted toggle button (checkbox + label "Agent-assisted"), aligned to bottom of cell via `flex items-end`

All four fields update via the existing `updateField` helper and reset `mobileStage` to `'review'`.

---

### 5. Risk scoring

`deriveRiskScoreFromForm` gains two additive bumps after the existing factors:

```ts
composite += form.isAgentAssisted ? 0.06 : 0;
composite += form.connectivityMode === "offline_buffered" ? 0.04
           : form.connectivityMode === "intermittent"     ? 0.02
           : 0;
```

Effect: an agent-assisted + offline + cross-border transaction visibly pushes the score toward FLAG/BLOCK, making the ASEAN context legible in the live result panel.

---

### 6. ReviewScreen — corridor row

Add one row to the "Transaction summary" block, between Channel and Currency:

- **Label:** `Corridor`
- **Value:** `"ID → SG"` when `isCrossBorder` is true, otherwise just `sourceCountry` (e.g. `"ID"`)

This makes the ASEAN-awareness explicit in the mock mobile UI that judges see on the right side of the screen.

---

### 7. `handleSubmitReview` — history record

No change needed. `addTransactionToHistory` already receives `crossBorder` — the new fields don't need to be stored in history for the demo to work.

---

## What is NOT changing

- No API calls added (sandbox stays offline)
- No changes to `UserForm.tsx`, `UserApp.tsx`, or backend
- No new components or files
- No changes to the left-panel header or mobile phone frame

---

## Success criteria

1. Selecting the "Send to abroad" preset auto-fills source = `ID`, destination = `SG`, and the corridor row in the ReviewScreen shows `ID → SG`
2. Toggling agent-assisted or changing connectivity mode updates the live risk score visibly
3. When cross-border is unchecked, destination country select is greyed out
4. All four new fields reset correctly when switching presets

# ASEAN Context Layer — Integration Sandbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ASEAN runtime context fields (source/destination country, connectivity mode, agent-assisted) to the Integration Sandbox at `/demo` so judges see the ASEAN differentiator on the primary demo page.

**Architecture:** All changes are in `project/frontend/src/IntegrationSandbox.tsx` only. The sandbox uses local offline scoring — ASEAN fields are wired into `SandboxFormState`, the form UI, `applyScenarioPreset`, `deriveRiskScoreFromForm`, and the `ReviewScreen` corridor row.

**Tech Stack:** React, TypeScript, Tailwind CSS, Vite

---

## Files

- Modify: `project/frontend/src/IntegrationSandbox.tsx`

---

### Task 1: Add types and extend SandboxFormState

**Files:**
- Modify: `project/frontend/src/IntegrationSandbox.tsx`

- [ ] **Step 1: Add type aliases after the existing `type TxType` and `type Channel` lines (lines 15–16)**

```ts
type CountryCode = 'SG' | 'MY' | 'ID' | 'TH' | 'PH' | 'VN';
type ConnectivityMode = 'online' | 'intermittent' | 'offline_buffered';
```

- [ ] **Step 2: Add four fields to `SandboxFormState` (after `p2pCounterparties24h`)**

```ts
type SandboxFormState = {
  formScenario: ScenarioId;
  userId: string;
  amount: string;
  txType: TxType;
  isCrossBorder: boolean;
  walletId: string;
  currency: string;
  merchantName: string;
  deviceRiskScore: string;
  ipRiskScore: string;
  locationRiskScore: string;
  deviceId: string;
  deviceSharedUsers24h: string;
  accountAgeDays: string;
  simChangeRecent: boolean;
  channel: Channel;
  cashFlowVelocity1h: string;
  p2pCounterparties24h: string;
  sourceCountry: CountryCode;
  destinationCountry: CountryCode;
  isAgentAssisted: boolean;
  connectivityMode: ConnectivityMode;
};
```

- [ ] **Step 3: Verify TypeScript compiles**

Run from `project/frontend/`:
```bash
npx tsc --noEmit
```
Expected: errors about `createInitialFormState` and `applyScenarioPreset` missing the new fields (that's correct — fix in next task).

- [ ] **Step 4: Commit**

```bash
git add project/frontend/src/IntegrationSandbox.tsx
git commit -m "feat(sandbox): add ASEAN type aliases and SandboxFormState fields"
```

---

### Task 2: Wire new fields into createInitialFormState and applyScenarioPreset

**Files:**
- Modify: `project/frontend/src/IntegrationSandbox.tsx`

- [ ] **Step 1: Update `createInitialFormState` to read ASEAN fields from the everyday_purchase preset**

Replace the existing `createInitialFormState` function body with:

```ts
function createInitialFormState(): SandboxFormState {
  const basePreset = SCENARIO_PRESETS.everyday_purchase;

  return {
    formScenario: 'everyday_purchase',
    userId: basePreset.ui.user_id,
    amount: basePreset.ui.amount,
    txType: basePreset.ui.tx_type,
    isCrossBorder: basePreset.ui.is_cross_border,
    walletId: basePreset.ui.wallet_id,
    currency: basePreset.ui.currency,
    merchantName: basePreset.ui.merchant_name,
    deviceRiskScore: String(basePreset.risk.device_risk_score),
    ipRiskScore: String(basePreset.risk.ip_risk_score),
    locationRiskScore: String(basePreset.risk.location_risk_score),
    deviceId: basePreset.risk.device_id,
    deviceSharedUsers24h: String(basePreset.risk.device_shared_users_24h),
    accountAgeDays: String(basePreset.risk.account_age_days),
    simChangeRecent: basePreset.risk.sim_change_recent,
    channel: basePreset.risk.channel,
    cashFlowVelocity1h: String(basePreset.risk.cash_flow_velocity_1h),
    p2pCounterparties24h: String(basePreset.risk.p2p_counterparties_24h),
    sourceCountry: (basePreset.ui.source_country ?? 'ID') as CountryCode,
    destinationCountry: (basePreset.ui.destination_country ?? 'ID') as CountryCode,
    isAgentAssisted: basePreset.ui.is_agent_assisted ?? false,
    connectivityMode: (basePreset.ui.connectivity_mode ?? 'online') as ConnectivityMode,
  };
}
```

- [ ] **Step 2: Update the non-custom branch of `applyScenarioPreset` to include ASEAN fields**

Replace the `return { ...current, formScenario: scenarioId, ... }` block in the non-custom branch with:

```ts
return {
  ...current,
  formScenario: scenarioId,
  userId: preset.ui.user_id,
  amount: preset.ui.amount,
  txType: preset.ui.tx_type,
  isCrossBorder: preset.ui.is_cross_border,
  walletId: preset.ui.wallet_id,
  currency: preset.ui.currency,
  merchantName: preset.ui.merchant_name,
  deviceRiskScore: String(preset.risk.device_risk_score),
  ipRiskScore: String(preset.risk.ip_risk_score),
  locationRiskScore: String(preset.risk.location_risk_score),
  deviceId: preset.risk.device_id,
  deviceSharedUsers24h: String(preset.risk.device_shared_users_24h),
  accountAgeDays: String(preset.risk.account_age_days),
  simChangeRecent: preset.risk.sim_change_recent,
  channel: preset.risk.channel,
  cashFlowVelocity1h: String(preset.risk.cash_flow_velocity_1h),
  p2pCounterparties24h: String(preset.risk.p2p_counterparties_24h),
  sourceCountry: (preset.ui.source_country ?? 'ID') as CountryCode,
  destinationCountry: (preset.ui.destination_country ?? 'ID') as CountryCode,
  isAgentAssisted: preset.ui.is_agent_assisted ?? false,
  connectivityMode: (preset.ui.connectivity_mode ?? 'online') as ConnectivityMode,
};
```

- [ ] **Step 3: Update the `custom` branch of `applyScenarioPreset` to include ASEAN defaults**

In the `if (scenarioId === 'custom')` return block, add after `p2pCounterparties24h`:

```ts
sourceCountry: 'SG' as CountryCode,
destinationCountry: 'SG' as CountryCode,
isAgentAssisted: false,
connectivityMode: 'online' as ConnectivityMode,
```

- [ ] **Step 4: Verify TypeScript compiles clean**

Run from `project/frontend/`:
```bash
npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add project/frontend/src/IntegrationSandbox.tsx
git commit -m "feat(sandbox): wire ASEAN fields into initial state and scenario presets"
```

---

### Task 3: Add ASEAN fields to the Payment Details form UI

**Files:**
- Modify: `project/frontend/src/IntegrationSandbox.tsx`

- [ ] **Step 1: Add `CheckCircle2` to the lucide-react import if not already present**

The top import line should read:
```ts
import {
  ArrowDown,
  CheckCircle2,
  Globe,
  PencilLine,
  ShoppingCart,
  Wallet,
  WalletCards,
} from 'lucide-react';
```
(`CheckCircle2` is already imported — confirm it's there, no change needed if so.)

- [ ] **Step 2: Add the ASEAN fields block immediately after the cross-border checkbox button in the Payment Details card**

Locate the closing `</button>` of the cross-border checkbox (the one with "Sending to someone in another country"). Add the following JSX directly after it, still inside the same `<div className="mt-4 rounded-[20px] ...">` card:

```tsx
{/* ASEAN context — source/destination countries */}
<div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
  <div>
    <FieldLabel>Source country</FieldLabel>
    <FieldSelect<CountryCode>
      value={form.sourceCountry}
      onChange={(value) => updateField('sourceCountry', value)}
      options={[
        { value: 'SG', label: 'SG — Singapore' },
        { value: 'MY', label: 'MY — Malaysia' },
        { value: 'ID', label: 'ID — Indonesia' },
        { value: 'TH', label: 'TH — Thailand' },
        { value: 'PH', label: 'PH — Philippines' },
        { value: 'VN', label: 'VN — Vietnam' },
      ]}
    />
  </div>
  <div style={{ opacity: form.isCrossBorder ? 1 : 0.5 }}>
    <FieldLabel>Destination country</FieldLabel>
    <FieldSelect<CountryCode>
      value={form.isCrossBorder ? form.destinationCountry : form.sourceCountry}
      onChange={(value) => updateField('destinationCountry', value)}
      options={[
        { value: 'SG', label: 'SG — Singapore' },
        { value: 'MY', label: 'MY — Malaysia' },
        { value: 'ID', label: 'ID — Indonesia' },
        { value: 'TH', label: 'TH — Thailand' },
        { value: 'PH', label: 'PH — Philippines' },
        { value: 'VN', label: 'VN — Vietnam' },
      ]}
    />
  </div>
</div>

{/* ASEAN context — connectivity + agent */}
<div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
  <div>
    <FieldLabel>Connectivity</FieldLabel>
    <FieldSelect<ConnectivityMode>
      value={form.connectivityMode}
      onChange={(value) => updateField('connectivityMode', value)}
      options={[
        { value: 'online', label: 'Live' },
        { value: 'intermittent', label: 'Intermittent' },
        { value: 'offline_buffered', label: 'Degraded' },
      ]}
    />
  </div>
  <div className="flex items-end">
    <button
      type="button"
      onClick={() => updateField('isAgentAssisted', !form.isAgentAssisted)}
      className="flex items-center gap-3 border-0 bg-transparent p-0 text-left shadow-none"
      style={{ minHeight: 0 }}
    >
      <div className={`flex h-[18px] w-[18px] items-center justify-center rounded-[4px] border ${form.isAgentAssisted ? 'border-[#1273E7] bg-[#1273E7]' : 'border-[#6B7280]'}`}>
        {form.isAgentAssisted ? <CheckCircle2 size={12} color="white" /> : null}
      </div>
      <span className="text-[13px] text-white">Agent-assisted</span>
    </button>
  </div>
</div>
```

- [ ] **Step 3: Verify TypeScript compiles clean**

Run from `project/frontend/`:
```bash
npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add project/frontend/src/IntegrationSandbox.tsx
git commit -m "feat(sandbox): add ASEAN source/dest country, connectivity, agent-assisted to Payment Details"
```

---

### Task 4: Update offline risk scoring to factor in ASEAN fields

**Files:**
- Modify: `project/frontend/src/IntegrationSandbox.tsx`

- [ ] **Step 1: Add ASEAN bumps to `deriveRiskScoreFromForm`**

Locate `deriveRiskScoreFromForm`. After the existing `composite += counterparties >= 3 ? ...` line and before the final `return`, add:

```ts
composite += form.isAgentAssisted ? 0.06 : 0;
composite += form.connectivityMode === 'offline_buffered' ? 0.04
           : form.connectivityMode === 'intermittent'     ? 0.02
           : 0;
```

The full updated function should look like:

```ts
function deriveRiskScoreFromForm(form: SandboxFormState): number {
  const avgRisk =
    (parseNumericValue(form.deviceRiskScore) +
      parseNumericValue(form.ipRiskScore) +
      parseNumericValue(form.locationRiskScore)) /
    3;
  const sharedUsers = parseNumericValue(form.deviceSharedUsers24h);
  const accountAge = parseNumericValue(form.accountAgeDays);
  const cashVelocity = parseNumericValue(form.cashFlowVelocity1h);
  const counterparties = parseNumericValue(form.p2pCounterparties24h);
  const amount = parseNumericValue(form.amount);

  let composite = avgRisk;
  composite += form.simChangeRecent ? 0.14 : 0;
  composite += form.isCrossBorder ? 0.12 : 0;
  composite += sharedUsers >= 3 ? Math.min((sharedUsers - 2) * 0.035, 0.16) : 0;
  composite += accountAge < 30 ? 0.12 : accountAge < 90 ? 0.05 : 0;
  composite += cashVelocity >= 4 ? Math.min((cashVelocity - 3) * 0.028, 0.16) : 0;
  composite += counterparties >= 3 ? Math.min((counterparties - 2) * 0.018, 0.12) : 0;
  composite += amount >= 750 ? 0.08 : amount >= 250 ? 0.04 : 0;
  composite += form.isAgentAssisted ? 0.06 : 0;
  composite += form.connectivityMode === 'offline_buffered' ? 0.04
             : form.connectivityMode === 'intermittent'     ? 0.02
             : 0;

  return Math.min(0.99, Math.max(0.01, Number(composite.toFixed(3))));
}
```

- [ ] **Step 2: Verify TypeScript compiles clean**

Run from `project/frontend/`:
```bash
npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add project/frontend/src/IntegrationSandbox.tsx
git commit -m "feat(sandbox): factor ASEAN agent-assisted and connectivity into offline risk score"
```

---

### Task 5: Show corridor in ReviewScreen

**Files:**
- Modify: `project/frontend/src/IntegrationSandbox.tsx`

- [ ] **Step 1: Pass ASEAN fields into ReviewScreen**

`ReviewScreen` currently receives `data: SandboxFormState` — it already has access to `sourceCountry`, `destinationCountry`, and `isCrossBorder` via `data`. No prop changes needed.

- [ ] **Step 2: Add corridor row to the transaction summary block in ReviewScreen**

Locate the transaction summary rows (the block with "Payment type", "Channel", "Currency"). Add a "Corridor" row between "Channel" and "Currency":

```tsx
<div className="flex items-center justify-between gap-4 text-[14px] tracking-[-0.14px]">
  <span className="text-[#8C909F]">Corridor</span>
  <span className="font-medium text-[#DFE2EB]">
    {data.isCrossBorder
      ? `${data.sourceCountry} → ${data.destinationCountry}`
      : data.sourceCountry}
  </span>
</div>
```

The full updated summary rows block:

```tsx
<div className="mt-5 space-y-3">
  <div className="flex items-center justify-between gap-4 text-[14px] tracking-[-0.14px]">
    <span className="text-[#8C909F]">Payment type</span>
    <span className="font-medium text-[#DFE2EB]">{txTypeLabel}</span>
  </div>
  <div className="flex items-center justify-between gap-4 text-[14px] tracking-[-0.14px]">
    <span className="text-[#8C909F]">Channel</span>
    <span className="font-medium text-[#DFE2EB]">{data.channel}</span>
  </div>
  <div className="flex items-center justify-between gap-4 text-[14px] tracking-[-0.14px]">
    <span className="text-[#8C909F]">Corridor</span>
    <span className="font-medium text-[#DFE2EB]">
      {data.isCrossBorder
        ? `${data.sourceCountry} → ${data.destinationCountry}`
        : data.sourceCountry}
    </span>
  </div>
  <div className="flex items-center justify-between gap-4 text-[14px] tracking-[-0.14px]">
    <span className="text-[#8C909F]">Currency</span>
    <span className="font-medium text-[#DFE2EB]">{data.currency || 'USD'}</span>
  </div>
</div>
```

- [ ] **Step 3: Verify TypeScript compiles clean**

Run from `project/frontend/`:
```bash
npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add project/frontend/src/IntegrationSandbox.tsx
git commit -m "feat(sandbox): show ASEAN corridor in ReviewScreen transaction summary"
```

---

### Task 6: Manual verification

- [ ] **Step 1: Start the dev server**

Run from `project/frontend/`:
```bash
npm run dev
```
Open `http://localhost:5173/demo`.

- [ ] **Step 2: Verify everyday_purchase preset**

On load, confirm:
- Source country = ID
- Destination country = ID (greyed out)
- Connectivity = Live
- Agent-assisted = unchecked
- ReviewScreen corridor row = `ID`

- [ ] **Step 3: Verify cross_border preset**

Click "Send to abroad" scenario card. Confirm:
- Source country auto-fills (check scenarioPresets.json for the value)
- Destination country auto-fills and is no longer greyed out
- ReviewScreen corridor row = `{source} → {dest}`

- [ ] **Step 4: Verify risk score responds to ASEAN fields**

With any preset, toggle Agent-assisted on. Confirm the risk score in the result panel increases. Change Connectivity to "Degraded". Confirm score increases further.

- [ ] **Step 5: Verify destination country greys out when cross-border is unchecked**

With cross_border preset, uncheck "Sending to someone in another country". Confirm destination country select becomes `opacity: 0.5` and corridor row shows just the source country.

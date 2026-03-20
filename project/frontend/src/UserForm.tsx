import { useState } from "react";
import { Globe, Edit3, ShoppingCart, WalletCards } from "lucide-react";
import { t, type Locale } from "./i18n";
import { SCENARIO_ORDER, SCENARIO_PRESETS, type ScenarioId } from "./scenarioPresets";


const tokens = {
  bgBase: "#10141a",
  bgSurface: "#181c22",
  bgCard: "#1c2026",
  bgInput: "#22252a",
  border: "rgba(66,71,84,0.3)",
  borderFocus: "rgba(173,198,255,0.3)",
  blueHover: "#acc6e6ff",
  blue: "#4d8eff",
  blueDim: "rgba(77,142,255,0.2)",
  blueBright: "#bfdbfe",
  textPrimary: "#ffffff",
  textSecondary: "#c2c6d6",
  textMuted: "#8c909f",
  textBlue: "#adc6ff",
};

interface Scenario {
  id: ScenarioId;
  icon: React.ElementType;
  label: string;
  desc: string;
  presetRisk?: {
    schema_version: string;
    device_risk_score: number;
    ip_risk_score: number;
    location_risk_score: number;
    device_id: string;
    device_shared_users_24h: number;
    account_age_days: number;
    sim_change_recent: boolean;
    channel: "APP" | "WEB" | "AGENT" | "QR";
    cash_flow_velocity_1h: number;
    p2p_counterparties_24h: number;
  };
  prefillUserId?: string;
  prefillAmount: string;
  prefillType: string;
  prefillWalletId?: string;
  prefillCurrency?: string;
  prefillMerchant?: string;
  crossBorder?: boolean;
}

const SCENARIO_ICON_BY_ID: Record<ScenarioId, React.ElementType> = {
  everyday_purchase: ShoppingCart,
  large_amount: WalletCards,
  cross_border: Globe,
  custom: Edit3,
};

const SCENARIOS: Scenario[] = SCENARIO_ORDER.map((scenarioId) => {
  const preset = SCENARIO_PRESETS[scenarioId];
  return {
    id: scenarioId,
    icon: SCENARIO_ICON_BY_ID[scenarioId],
    label: "",
    desc: "",
    presetRisk: preset.risk,
    prefillAmount: preset.ui.amount,
    prefillType: preset.ui.tx_type,
    prefillUserId: preset.ui.user_id,
    prefillWalletId: preset.ui.wallet_id,
    prefillCurrency: preset.ui.currency,
    prefillMerchant: preset.ui.merchant_name,
    crossBorder: preset.ui.is_cross_border,
  };
});

interface PaymentType {
  value: string;
}

const PAYMENT_TYPES: PaymentType[] = [
  { value: "MERCHANT" },
  { value: "P2P" },
  { value: "CASH_IN" },
  { value: "CASH_OUT" },
];

const SCENARIO_I18N_KEY_BY_ID: Record<string, string> = {
  everyday_purchase: "scenario.everyday_purchase",
  large_amount: "scenario.large_amount",
  cross_border: "scenario.cross_border",
  custom: "scenario.custom",
};

interface ScenarioCardProps {
  scenario: Scenario;
  selected: string;
  onSelect: (scenario: Scenario) => void;
}

function ScenarioCard({ scenario, selected, onSelect }: ScenarioCardProps) {
  const active = selected === scenario.id;
  const Icon = scenario.icon;
  return (
    <button
      onClick={() => onSelect(scenario)}
      style={{
        background: active ? tokens.blueDim : tokens.bgCard,
        border: `1px solid ${active ? "rgba(173,198,255,0.3)" : "rgba(66,71,84,0.3)"}`,
        borderRadius: 12,
        padding: "13px",
        textAlign: "left",
        cursor: "pointer",
        transition: "all 0.2s ease",
        display: "flex",
        flexDirection: "column",
        gap: 8,
        height: "100%",
      }}
    >
      <div
        style={{
          width: 16,
          height: 16,
          color: active ? tokens.blueBright : tokens.textSecondary,
          transition: "color 0.2s ease",
          flexShrink: 0,
        }}
      >
        <Icon size={16} strokeWidth={2} />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span
          style={{
            fontSize: 14,
            fontWeight: 500,
            color: active ? tokens.blueBright : tokens.textPrimary,
            fontFamily: "'Inter Tight', sans-serif",
            lineHeight: 1.38,
          }}
        >
          {scenario.label}
        </span>
        <span
          style={{
            fontSize: 12,
            fontWeight: 400,
            color: tokens.textSecondary,
            lineHeight: 1.25,
            fontFamily: "'Inter Tight', sans-serif",
          }}
        >
          {scenario.desc}
        </span>
      </div>
    </button>
  );
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <label
      style={{
        display: "block",
        fontSize: 11,
        color: tokens.textSecondary,
        marginBottom: 4,
        fontFamily: "'Inter Tight', sans-serif",
        fontWeight: 500,
        paddingLeft: 4,
      }}
    >
      {children}
    </label>
  );
}

interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  style?: React.CSSProperties;
}

function TextInput({ value, onChange, placeholder, style = {} }: TextInputProps) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      style={{
        width: "100%",
        background: tokens.bgInput,
        border: `0.9px solid ${tokens.border}`,
        borderRadius: 8,
        padding: "11.5px 12px",
        fontSize: 12,
        color: tokens.textPrimary,
        fontFamily: "'Inter Tight', sans-serif",
        outline: "none",
        boxSizing: "border-box",
        transition: "border-color 0.15s ease",
        ...style,
      }}
    />
  );
}

interface SelectInputProps {
  value: string;
  onChange: (value: string) => void;
  options: PaymentType[];
  placeholder: string;
  translate: (key: string) => string;
}

function SelectInput({ value, onChange, options, placeholder, translate }: SelectInputProps) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        width: "100%",
        background: tokens.bgInput,
        border: `0.9px solid ${tokens.border}`,
        borderRadius: 8,
        padding: "11.5px 12px",
        fontSize: 12,
        color: value ? tokens.textPrimary : tokens.textSecondary,
        fontFamily: "'Inter Tight', sans-serif",
        outline: "none",
        appearance: "none",
        cursor: "pointer",
        transition: "border-color 0.15s ease",
        backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%238c909f' stroke-width='2'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E")`,
        backgroundRepeat: "no-repeat",
        backgroundPosition: "right 12px center",
        paddingRight: 36,
      }}
    >
      <option value="" disabled>{placeholder}</option>
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {translate(`option.txType.${o.value}`)}
        </option>
      ))}
    </select>
  );
}

interface SectionCardProps {
  title: string;
  children: React.ReactNode;
}

function SectionCard({ title, children }: SectionCardProps) {
  return (
    <div
      style={{
        background: tokens.bgSurface,
        border: `1px solid rgba(66,71,84,0.2)`,
        borderRadius: 16,
        padding: "17px",
        marginBottom: 12,
      }}
    >
      <p
        style={{
          margin: "0 0 16px",
          fontSize: 14,
          fontWeight: 600,
          color: tokens.textPrimary,
          fontFamily: "'Inter Tight', sans-serif",
        }}
      >
        {title}
      </p>
      {children}
    </div>
  );
}

export type UserFormPayload = {
  scenario_id: ScenarioId;
  override_source?: "preset" | "support_manual";
  support_mode?: boolean;
  support_actor_id?: string;
  override_fields?: string[];
  user_id: string;
  transaction_amount: string;
  tx_type: string;
  is_cross_border: boolean;
  wallet_id: string;
  currency: string;
  merchant_name: string;
  preset_risk?: Scenario["presetRisk"];
};

interface CheckYourPaymentProps {
  locale?: Locale;
  onSubmit: (payload: UserFormPayload) => void | Promise<void>;
  isSubmitting?: boolean;
  submitError?: string;
  onBack?: () => void;
  supportMode?: boolean;
  supportActorId?: string;
  onSupportModeChange?: (enabled: boolean) => void;
  onSupportActorIdChange?: (actorId: string) => void;
}

export default function CheckYourPayment({
  locale = "en",
  onSubmit,
  isSubmitting = false,
  submitError,
  onBack,
  supportMode = false,
  supportActorId,
  onSupportModeChange,
  onSupportActorIdChange,
}: CheckYourPaymentProps) {
  const [selectedScenario, setSelectedScenario] = useState<ScenarioId>("everyday_purchase");
  const [userId, setUserId] = useState("user_1001");
  const [amount, setAmount] = useState("12.40");
  const [paymentType, setPaymentType] = useState("MERCHANT");
  const [crossBorder, setCrossBorder] = useState(false);
  const [walletId, setWalletId] = useState("user_1001");
  const [currency, setCurrency] = useState("USD");
  const [merchant, setMerchant] = useState("Demo Merchant");
  const [deviceRiskScore, setDeviceRiskScore] = useState("0.05");
  const [ipRiskScore, setIpRiskScore] = useState("0.06");
  const [locationRiskScore, setLocationRiskScore] = useState("0.05");
  const [deviceId, setDeviceId] = useState("device_safe_01");
  const [deviceSharedUsers24h, setDeviceSharedUsers24h] = useState("1");
  const [accountAgeDays, setAccountAgeDays] = useState("365");
  const [simChangeRecent, setSimChangeRecent] = useState(false);
  const [channel, setChannel] = useState<"APP" | "WEB" | "AGENT" | "QR">("APP");
  const [cashFlowVelocity1h, setCashFlowVelocity1h] = useState("1");
  const [p2pCounterparties24h, setP2pCounterparties24h] = useState("0");
  const [btnHover, setBtnHover] = useState(false);
  const translate = (key: string, vars?: Record<string, string | number>): string => t(locale, key, vars);

  const getScenarioRisk = (id: ScenarioId) => SCENARIO_PRESETS[id].risk;

  const applyRiskPreset = (id: ScenarioId) => {
    const risk = getScenarioRisk(id);
    setDeviceRiskScore(String(risk.device_risk_score));
    setIpRiskScore(String(risk.ip_risk_score));
    setLocationRiskScore(String(risk.location_risk_score));
    setDeviceId(risk.device_id);
    setDeviceSharedUsers24h(String(risk.device_shared_users_24h));
    setAccountAgeDays(String(risk.account_age_days));
    setSimChangeRecent(risk.sim_change_recent);
    setChannel(risk.channel);
    setCashFlowVelocity1h(String(risk.cash_flow_velocity_1h));
    setP2pCounterparties24h(String(risk.p2p_counterparties_24h));
  };

  const resetAdvancedToPreset = () => applyRiskPreset(selectedScenario);

  const clampFloat = (value: string, min: number, max: number, fallback: string): string => {
    const parsed = Number.parseFloat(value);
    if (!Number.isFinite(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed)).toFixed(2);
  };

  const clampInt = (value: string, min: number, max: number, fallback: string): string => {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed)) return fallback;
    return String(Math.min(max, Math.max(min, parsed)));
  };

  const getOverrideFields = (): string[] => {
    const preset = getScenarioRisk(selectedScenario);
    const changed: string[] = [];
    if (Number.parseFloat(deviceRiskScore) !== preset.device_risk_score) changed.push("device_risk_score");
    if (Number.parseFloat(ipRiskScore) !== preset.ip_risk_score) changed.push("ip_risk_score");
    if (Number.parseFloat(locationRiskScore) !== preset.location_risk_score) changed.push("location_risk_score");
    if (deviceId !== preset.device_id) changed.push("device_id");
    if (Number.parseInt(deviceSharedUsers24h, 10) !== preset.device_shared_users_24h) changed.push("device_shared_users_24h");
    if (Number.parseInt(accountAgeDays, 10) !== preset.account_age_days) changed.push("account_age_days");
    if (simChangeRecent !== preset.sim_change_recent) changed.push("sim_change_recent");
    if (channel !== preset.channel) changed.push("channel");
    if (Number.parseInt(cashFlowVelocity1h, 10) !== preset.cash_flow_velocity_1h) changed.push("cash_flow_velocity_1h");
    if (Number.parseInt(p2pCounterparties24h, 10) !== preset.p2p_counterparties_24h) changed.push("p2p_counterparties_24h");
    return changed;
  };

  function handleScenarioSelect(scenario: Scenario) {
    setSelectedScenario(scenario.id);
    
    if (scenario.id === "custom") {
      setUserId("");
      setAmount("");
      setPaymentType("");
      setCrossBorder(false);
      setWalletId("");
      setCurrency("");
      setMerchant("");
      return;
    }

    if (scenario.prefillAmount) setAmount(scenario.prefillAmount);
    if (scenario.prefillType) setPaymentType(scenario.prefillType);
    if (scenario.prefillUserId) setUserId(scenario.prefillUserId);
    if (scenario.prefillWalletId) setWalletId(scenario.prefillWalletId);
    if (scenario.prefillCurrency) setCurrency(scenario.prefillCurrency);
    if (scenario.prefillMerchant) setMerchant(scenario.prefillMerchant);

    // Explicitly set crossBorder based on the scenario's setting, 
    // defaulting to false if it's not specified (unticking it for other scenarios).
    setCrossBorder(scenario.crossBorder === true);
    applyRiskPreset(scenario.id);
  }

  function handleSubmit() {
    // Validate all required fields before submitting
    if (!userId?.trim() || !amount?.trim() || !paymentType || !walletId?.trim() || !merchant?.trim() || !currency?.trim()) {
      return; // Button should be disabled, but just in case
    }

    const overrideFields = supportMode && selectedScenario !== "custom" ? getOverrideFields() : [];
    const hasManualOverride = overrideFields.length > 0;

    const payload: UserFormPayload = {
      scenario_id: selectedScenario,
      user_id: userId.trim(),
      transaction_amount: amount.trim(),
      tx_type: paymentType,
      is_cross_border: crossBorder,
      wallet_id: walletId.trim(),
      currency: currency.trim(),
      merchant_name: merchant.trim(),
      preset_risk:
        selectedScenario === "custom"
          ? undefined
          : {
              schema_version: SCENARIO_PRESETS[selectedScenario].risk.schema_version,
              device_risk_score: Number.parseFloat(clampFloat(deviceRiskScore, 0, 1, String(SCENARIO_PRESETS[selectedScenario].risk.device_risk_score))),
              ip_risk_score: Number.parseFloat(clampFloat(ipRiskScore, 0, 1, String(SCENARIO_PRESETS[selectedScenario].risk.ip_risk_score))),
              location_risk_score: Number.parseFloat(clampFloat(locationRiskScore, 0, 1, String(SCENARIO_PRESETS[selectedScenario].risk.location_risk_score))),
              device_id: deviceId.trim() || SCENARIO_PRESETS[selectedScenario].risk.device_id,
              device_shared_users_24h: Number.parseInt(clampInt(deviceSharedUsers24h, 0, 50, String(SCENARIO_PRESETS[selectedScenario].risk.device_shared_users_24h)), 10),
              account_age_days: Number.parseInt(clampInt(accountAgeDays, 0, 36500, String(SCENARIO_PRESETS[selectedScenario].risk.account_age_days)), 10),
              sim_change_recent: simChangeRecent,
              channel,
              cash_flow_velocity_1h: Number.parseInt(clampInt(cashFlowVelocity1h, 0, 500, String(SCENARIO_PRESETS[selectedScenario].risk.cash_flow_velocity_1h)), 10),
              p2p_counterparties_24h: Number.parseInt(clampInt(p2pCounterparties24h, 0, 1000, String(SCENARIO_PRESETS[selectedScenario].risk.p2p_counterparties_24h)), 10),
            },
    };

    if (supportMode) {
      payload.override_source = hasManualOverride ? "support_manual" : "preset";
      payload.support_mode = true;
      payload.support_actor_id = supportActorId;
      payload.override_fields = overrideFields;
    }

    onSubmit(payload);
  }

  function handleBack() {
    if (onBack) {
      onBack();
      return;
    }
    if (window.history.length > 1) {
      window.history.back();
    }
  }

  const canSubmit = Boolean(userId?.trim() && amount?.trim() && paymentType && walletId?.trim() && merchant?.trim() && currency?.trim());
  const editedFields = supportMode && selectedScenario !== "custom" ? getOverrideFields() : [];

  return (
    <div
      style={{
        minHeight: "100vh",
        background: tokens.bgBase,
        fontFamily: "'Inter Tight', sans-serif",
        maxWidth: 520,
        margin: "0 auto",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* scrollable body */}
      <div style={{ flex: 1, padding: "8px 16px 140px 16px", overflowY: "auto" }}>

        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 0 }}>
          <button
            onClick={handleBack}
            aria-label={translate("form.back")}
            style={{
              background: "transparent",
              border: "none",
              cursor: "pointer",
              padding: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: tokens.textSecondary,
              transition: "color 0.2s ease",
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLElement).style.color = tokens.textPrimary;
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLElement).style.color = tokens.textSecondary;
            }}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M15 18l-6-6 6-6" />
            </svg>
          </button>

          <button
            onClick={() => onSupportModeChange?.(!supportMode)}
            style={{
              border: "none",
              background: "transparent",
              color: supportMode ? tokens.blueBright : tokens.textMuted,
              fontSize: 11,
              cursor: "pointer",
              padding: "4px 0",
            }}
          >
            Support
          </button>
        </div>

        {supportMode && (
          <div
            style={{
              background: tokens.bgCard,
              border: `1px solid ${tokens.border}`,
              borderRadius: 10,
              padding: "10px 12px",
              marginBottom: 10,
            }}
          >
            <FieldLabel>Support staff ID (optional)</FieldLabel>
            <TextInput
              value={supportActorId ?? ""}
              onChange={(value) => onSupportActorIdChange?.(value)}
              placeholder="agent_123"
            />
          </div>
        )}
        <h1
            style={{
              margin: 0,
              fontSize: 16,
              fontWeight: 600,
              color: tokens.textPrimary,
              fontFamily: "'Inter Tight', sans-serif",
            }}
          >
            {translate("form.title")}
          </h1>
        <p style={{ margin: "4px 0px 12px", fontSize: 11, color: tokens.textSecondary, lineHeight: 1.5, fontWeight: 400 }}>
          {translate("form.subtitle")}
        </p>
        

        {/* Scenario grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 12,
            marginBottom: 12,
          }}
        >
          {SCENARIOS.map((s) => (
            <ScenarioCard
              key={s.id}
              scenario={{
                ...s,
                label: translate(`${SCENARIO_I18N_KEY_BY_ID[s.id]}.label`),
                desc: translate(`${SCENARIO_I18N_KEY_BY_ID[s.id]}.narrative`),
              }}
              selected={selectedScenario}
              onSelect={handleScenarioSelect}
            />
          ))}
        </div>

        {/* Payment Details */}
        <SectionCard title={translate("form.section.paymentDetails")}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 12 }}>
            <div>
              <FieldLabel>{translate("form.field.userId")}</FieldLabel>
              <TextInput value={userId} onChange={setUserId} placeholder={translate("form.placeholder.userId")} />
            </div>
            <div>
              <FieldLabel>{translate("form.field.amount")}</FieldLabel>
              <TextInput
                value={amount}
                onChange={setAmount}
                placeholder={translate("form.placeholder.amount")}
              />
            </div>
          </div>

          <div style={{ marginBottom: 12 }}>
            <FieldLabel>{translate("form.field.paymentType")}</FieldLabel>
            <SelectInput
              value={paymentType}
              onChange={setPaymentType}
              options={PAYMENT_TYPES}
              placeholder={translate("form.placeholder.paymentType")}
              translate={translate}
            />
          </div>

          {/* Cross-border checkbox */}
          <button
            onClick={() => setCrossBorder(!crossBorder)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              background: "transparent",
              border: "none",
              cursor: "pointer",
              padding: 0,
              fontFamily: "'Inter Tight', sans-serif",
            }}
          >
            <div
              style={{
                width: 18,
                height: 18,
                borderRadius: 3.6,
                border: `0.9px solid ${tokens.textMuted}`,
                background: crossBorder ? tokens.blue : "transparent",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
                transition: "all 0.15s ease",
              }}
            >
              {crossBorder && (
                <svg width="10" height="8" viewBox="0 0 11 9" fill="none">
                  <path d="M1 4L4 7L10 1" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </div>
            <span style={{ fontSize: 12, color: tokens.textPrimary, textAlign: "left", lineHeight: 1.4, fontWeight: 400 }}>
              {translate("form.field.crossBorder")}
            </span>
          </button>
        </SectionCard>

        {supportMode && (
          <SectionCard title="Advanced context (support staff only)">
            <p style={{ margin: "0 0 10px", fontSize: 11, color: tokens.textSecondary }}>
              Auto-filled from selected preset. Adjust only when investigating customer-reported edge cases.
            </p>
            <p style={{ margin: "0 0 10px", fontSize: 11, color: tokens.textSecondary }}>
              Risk scores (0.00-1.00)
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 12 }}>
              <div>
                <FieldLabel>Device risk score</FieldLabel>
                <TextInput value={deviceRiskScore} onChange={setDeviceRiskScore} />
              </div>
              <div>
                <FieldLabel>IP risk score</FieldLabel>
                <TextInput value={ipRiskScore} onChange={setIpRiskScore} />
              </div>
              <div>
                <FieldLabel>Location risk score</FieldLabel>
                <TextInput value={locationRiskScore} onChange={setLocationRiskScore} />
              </div>
            </div>

                <p style={{ margin: "0 0 10px", fontSize: 11, color: tokens.textSecondary }}>
                  Device signals
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                  <div>
                    <FieldLabel>Device ID</FieldLabel>
                    <TextInput value={deviceId} onChange={setDeviceId} />
                  </div>
                  <div>
                    <FieldLabel>Device shared users (24h)</FieldLabel>
                    <TextInput value={deviceSharedUsers24h} onChange={setDeviceSharedUsers24h} />
                  </div>
                  <div>
                    <FieldLabel>Account age (days)</FieldLabel>
                    <TextInput value={accountAgeDays} onChange={setAccountAgeDays} />
                  </div>
                  <button
                    onClick={() => setSimChangeRecent((prev) => !prev)}
                    style={{ display: "flex", alignItems: "center", gap: 8, background: "transparent", border: "none", color: tokens.textPrimary, cursor: "pointer", paddingTop: 20 }}
                  >
                    <input type="checkbox" checked={simChangeRecent} readOnly />
                    SIM recently changed
                  </button>
                </div>

                <p style={{ margin: "0 0 10px", fontSize: 11, color: tokens.textSecondary }}>
                  Behavior velocity and channel
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 12 }}>
                  <div>
                    <FieldLabel>Channel</FieldLabel>
                    <select
                      value={channel}
                      onChange={(e) => setChannel(e.target.value as "APP" | "WEB" | "AGENT" | "QR")}
                      style={{ width: "100%", background: tokens.bgInput, border: `0.9px solid ${tokens.border}`, borderRadius: 8, padding: "11.5px 12px", fontSize: 12, color: tokens.textPrimary }}
                    >
                      <option value="APP">APP</option>
                      <option value="WEB">WEB</option>
                      <option value="AGENT">AGENT</option>
                      <option value="QR">QR</option>
                    </select>
                  </div>
                  <div>
                    <FieldLabel>Cash flow velocity (1h)</FieldLabel>
                    <TextInput value={cashFlowVelocity1h} onChange={setCashFlowVelocity1h} />
                  </div>
                  <div>
                    <FieldLabel>P2P counterparties (24h)</FieldLabel>
                    <TextInput value={p2pCounterparties24h} onChange={setP2pCounterparties24h} />
                  </div>
                </div>

            <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 8 }}>
              <button
                onClick={resetAdvancedToPreset}
                style={{ border: `1px solid ${tokens.border}`, background: "transparent", color: tokens.textSecondary, borderRadius: 8, padding: "7px 10px", fontSize: 12, cursor: "pointer" }}
              >
                Reset to preset defaults
              </button>
              {editedFields.length > 0 && (
                <span style={{ fontSize: 11, color: tokens.blueBright }}>
                  Edited from preset ({editedFields.length})
                </span>
              )}
            </div>
          </SectionCard>
        )}

        {/* Your Wallet */}
        <SectionCard title={translate("form.section.wallet")}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
            <div>
              <FieldLabel>{translate("form.field.walletId")}</FieldLabel>
              <TextInput value={walletId} onChange={setWalletId} placeholder={translate("form.placeholder.walletId")} />
            </div>
            <div>
              <FieldLabel>{translate("form.field.currency")}</FieldLabel>
              <TextInput value={currency} onChange={setCurrency} placeholder={translate("form.placeholder.currency")} />
            </div>
          </div>
          <div>
            <FieldLabel>{translate("form.field.merchant")}</FieldLabel>
            <TextInput value={merchant} onChange={setMerchant} placeholder={translate("form.placeholder.merchant")} />
          </div>
        </SectionCard>

        {!supportMode && (
          <SectionCard title="How decisions are explained">
            <ul style={{ margin: 0, paddingLeft: 18, color: tokens.textSecondary, fontSize: 12, lineHeight: 1.5 }}>
              <li>Everyday purchase: Approved quickly when behavior looks normal.</li>
              <li>Big payment: May need extra verification for your safety.</li>
              <li>Send abroad: Blocked when policy risk is too high.</li>
            </ul>
          </SectionCard>
        )}
      </div>

      {/* Sticky CTA */}
      <div
        style={{
          position: "fixed",
          bottom: 0,
          left: "50%",
          transform: "translateX(-50%)",
          width: "100%",
          maxWidth: 520,
          padding: "21px 16px 20px",
          background: "backdrop-blur(12px) rgba(28,32,38,0.6)",
          backdropFilter: "blur(12px)",
          borderTop: "1px solid rgba(66,71,84,0.15)",
          boxShadow: "0px -4px 32px 0px rgba(173,198,255,0.08)",
          borderTopLeftRadius: 24,
          borderTopRightRadius: 24,
        }}
      >
        {submitError && (
          <p style={{ margin: "0 0 10px", color: "#fca5a5", fontSize: 12 }}>
            {submitError}
          </p>
        )}
        <button
          onMouseEnter={() => setBtnHover(true)}
          onMouseLeave={() => setBtnHover(false)}
          onClick={handleSubmit}
          disabled={!canSubmit || isSubmitting}
          style={{
            width: "100%",
            background: !canSubmit || isSubmitting
              ? tokens.bgSurface
              : btnHover
              ? tokens.blueHover
              : tokens.blueBright,
            color: !canSubmit || isSubmitting
              ? tokens.textMuted
              : "#002e6a",
            border: "none",
            borderRadius: 12,
            padding: "12px 16px",
            fontSize: 14,
            fontWeight: 700,
            fontFamily: "'Inter Tight', sans-serif",
            cursor: canSubmit && !isSubmitting ? "pointer" : "not-allowed",
            transition: "all 0.2s ease",
            boxShadow: !canSubmit || isSubmitting 
              ? "none" 
              : "0px 10px 15px -3px rgba(173,198,255,0.2), 0px 4px 6px -4px rgba(173,198,255,0.2)",
          }}
        >
          {isSubmitting ? translate("form.button.checking") : translate("form.button.check")}
        </button>
      </div>
    </div>
  );
}

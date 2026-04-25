import { useState, useEffect, useRef, type ElementType } from "react";
import { TECHNICAL_DEFAULTS, resolveTechnicalFields, toNum } from "./utils/transaction";
import { WalletCards, KeyRound, Shield, Hand } from "lucide-react";
import CheckYourPayment, { type UserFormPayload } from "./UserForm";
import FraudResultScreen, { type Decision, type Transaction } from "./UserResult";
import {
  authorizePayment as authorizePaymentRequest,
  scoreTransaction as scoreTransactionRequest,
} from "./api";
import { localeOptions, t, type Locale } from "./i18n";
import { addTransactionToHistory, updateTransactionInHistory } from "./transactionHistoryUtils";

const tokens = {
  bgBase: "#0d1117",
  bgSurface: "#161b22",
  bgCard: "#1c2230",
  border: "#2d3748",
  borderLight: "#374151",
  blueHover: "#acc6e6ff",
  blueDim: "rgba(59,130,246,0.12)",
  blueBright: "#bfdbfe",
  green: "#10b981",
  greenDim: "rgba(16,185,129,0.1)",
  textPrimary: "#f0f6fc",
  textSecondary: "#8b949e",
  textMuted: "#4b5563",
};

type TxType = "MERCHANT" | "P2P" | "CASH_IN" | "CASH_OUT";
type Channel = "APP" | "WEB" | "AGENT" | "QR";

type TxPayload = {
  schema_version: string;
  user_id: string;
  transaction_amount: string;
  currency?: string;
  device_risk_score: string;
  ip_risk_score: string;
  location_risk_score: string;
  device_id: string;
  device_shared_users_24h: string;
  account_age_days: string;
  sim_change_recent: boolean;
  tx_type: TxType;
  channel: Channel;
  cash_flow_velocity_1h: string;
  p2p_counterparties_24h: string;
  is_cross_border: boolean;
  source_country?: "SG" | "MY" | "ID" | "TH" | "PH" | "VN";
  destination_country?: "SG" | "MY" | "ID" | "TH" | "PH" | "VN";
  is_agent_assisted?: boolean;
  connectivity_mode?: "online" | "intermittent" | "offline_buffered";
};

type FraudResponse = {
  decision: Decision;
  risk_score?: number;
  final_risk_score?: number;
  fraud_reasons?: string[];
  reasons?: string[];
  reason_codes?: string[];
  runtime_mode?: "primary" | "cached_context" | "degraded_local";
  corridor?: string | null;
  normalized_amount_reference?: number | null;
  normalization_basis?: string | null;
  explainability?: {
    base: number;
    context: number;
    behavior: number;
    ring?: number;
    external?: number;
  };
  context_summary?: {
    mcp_watchlist_hit?: boolean;
    mcp_risk_tier?: string;
  };
  stage_timings_ms?: {
    total_pipeline_ms?: number;
  };
};

type WalletResponse = {
  wallet_action: 'APPROVED' | 'PENDING_VERIFICATION' | 'DECLINED' | 'DECLINED_FRAUD_RISK';
  wallet_message?: string;
  fraud_engine_decision: Decision;
  final_risk_score?: number;
  next_step?: string;
  runtime_mode?: "primary" | "cached_context" | "degraded_local";
  reason_codes?: string[];
  corridor?: string | null;
  normalized_amount_reference?: number | null;
  normalization_basis?: string | null;
};

const SCHEMA_VERSION = "ieee_fraud_tx_v1";

const getSupportStorage = (): Storage | null => {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    return window.sessionStorage;
  } catch {
    try {
      return window.localStorage;
    } catch {
      return null;
    }
  }
};

const readSupportSettings = (): { enabled: boolean; actorId?: string } => {
  if (typeof window === "undefined") {
    return { enabled: import.meta.env.VITE_SUPPORT_MODE === "true" };
  }

  const storage = getSupportStorage();
  const enabled =
    import.meta.env.VITE_SUPPORT_MODE === "true" ||
    storage?.getItem("vhack_support_mode") === "true";
  const actorId = storage?.getItem("vhack_support_actor_id") ?? undefined;

  return { enabled, actorId };
};

// ── Animated dot-wave hero canvas ──────────────────────
function HeroCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let t = 0;

    const cols = 28, rows = 18;
    const W = canvas.width, H = canvas.height;
    const cw = W / cols, ch = H / rows;

    function draw() {
      if (!ctx) return;
      ctx.clearRect(0, 0, W, H);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const x = c * cw + cw / 2;
          const y = r * ch + ch / 2;
          const dx = (c - cols / 2) / cols;
          const dy = (r - rows / 2) / rows;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const wave = Math.sin(dist * 14 - t * 1.4) * 0.5 + 0.5;
          const size = 0.8 + wave * 1.4;
          const alpha = 0.08 + wave * 0.28;
          // teal-to-blue gradient by column
          const hue = 210 + c * 1.5;
          ctx.beginPath();
          ctx.arc(x, y, size, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${hue}, 80%, 70%, ${alpha})`;
          ctx.fill();
        }
      }
      t += 0.025;
      rafRef.current = requestAnimationFrame(draw);
    }

    draw();
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={390}
      height={280}
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", display: "block" }}
    />
  );
}

// ── Privacy card ────────────────────────────────────────
interface PrivacyCardProps {
  icon: ElementType;
  title: string;
  body: string;
  delay: number;
}

function PrivacyCard({ icon: Icon, title, body, delay }: PrivacyCardProps) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  return (
    <div
      style={{
        background: "linear-gradient(102deg, #1a202c 0%, #131a24 100%)",
        border: "1px solid rgba(66,71,84,0.18)",
        borderRadius: 18,
        padding: "20px",
        display: "flex",
        alignItems: "flex-start",
        flexDirection: "column",
        gap: 14,
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(12px)",
        transition: "opacity 0.5s ease, transform 0.5s ease",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12, width: "100%" }}>
        <div
          style={{
            width: 40,
            height: 40,
            borderRadius: 8,
            background: "#293142",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#adc6ff",
            flexShrink: 0,
          }}
        >
          <Icon size={20} strokeWidth={2} />
        </div>
        <p
          style={{
            margin: 0,
            fontSize: 16,
            fontWeight: 500,
            color: "#dfe2eb",
            fontFamily: "'Inter Tight', sans-serif",
            lineHeight: "28px",
          }}
        >
          {title}
        </p>
      </div>

      <div style={{ width: "100%" }}>
        <p
          style={{
            margin: 0,
            fontSize: 13,
            color: "#b6bccb",
            lineHeight: "20px",
            fontFamily: "'Inter Tight', sans-serif",
          }}
        >
          {body}
        </p>
      </div>
    </div>
  );
}

const PRIVACY_ITEMS: Array<{ icon: ElementType; titleKey: string; bodyKey: string }> = [
  { icon: WalletCards, titleKey: "landing.privacy.card1.title", bodyKey: "landing.privacy.card1.body" },
  { icon: KeyRound, titleKey: "landing.privacy.card2.title", bodyKey: "landing.privacy.card2.body" },
  { icon: Shield, titleKey: "landing.privacy.card3.title", bodyKey: "landing.privacy.card3.body" },
  { icon: Hand, titleKey: "landing.privacy.card4.title", bodyKey: "landing.privacy.card4.body" },
];

// ── Main screen ─────────────────────────────────────────
export default function FraudShieldPrivacyScreen() {
  const [view, setView] = useState<"landing" | "form" | "result">("landing");
  const [locale, setLocale] = useState<Locale>("en");
  const [btnHover, setBtnHover] = useState(false);
  const [ctaHover, setCtaHover] = useState(false);
  const [heroLoaded, setHeroLoaded] = useState(false);
  const [contentLoaded, setContentLoaded] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isAuthorizing, setIsAuthorizing] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [resultDecision, setResultDecision] = useState<Decision>("APPROVE");
  const [resultReasons, setResultReasons] = useState<string[]>([]);
  const [resultRuntimeMode, setResultRuntimeMode] = useState<"primary" | "cached_context" | "degraded_local">("primary");
  const [resultTransaction, setResultTransaction] = useState<Transaction>({
    merchant: "Demo Merchant",
    amount: "SGD 0.00",
    type: "MERCHANT",
  });
  const [lastTransactionId, setLastTransactionId] = useState<string | null>(null);
  const [lastFormData, setLastFormData] = useState<UserFormPayload | null>(null);
  const initialSupportSettings = readSupportSettings();
  const [supportModeEnabled, setSupportModeEnabled] = useState(
    initialSupportSettings.enabled,
  );
  const [supportActorIdValue, setSupportActorIdValue] = useState<string | undefined>(
    initialSupportSettings.actorId,
  );
  const translate = (key: string, vars?: Record<string, string | number>): string => t(locale, key, vars);

  const handleSupportModeToggle = (enabled: boolean) => {
    setSupportModeEnabled(enabled);
    const storage = getSupportStorage();
    if (storage) {
      if (enabled) {
        storage.setItem("vhack_support_mode", "true");
      } else {
        storage.removeItem("vhack_support_mode");
      }
    }
  };

  const handleSupportActorIdUpdate = (actorId: string) => {
    const trimmed = actorId.trim();
    setSupportActorIdValue(trimmed || undefined);
    const storage = getSupportStorage();
    if (storage) {
      if (trimmed) {
        storage.setItem("vhack_support_actor_id", trimmed);
      } else {
        storage.removeItem("vhack_support_actor_id");
      }
    }
  };

  useEffect(() => {
    if (view === 'landing') {
      const t1 = setTimeout(() => setHeroLoaded(true), 100);
      const t2 = setTimeout(() => setContentLoaded(true), 400);
      return () => { clearTimeout(t1); clearTimeout(t2); };
    }
  }, [view]);

  const handleSubmitPayment = async (form: UserFormPayload) => {
    setIsSubmitting(true);
    setSubmitError("");

    const fraudPayload: TxPayload = {
      schema_version: form.preset_risk?.schema_version ?? SCHEMA_VERSION,
      user_id: form.user_id ?? "",
      transaction_amount: form.transaction_amount,
      device_risk_score: form.preset_risk ? String(form.preset_risk.device_risk_score) : TECHNICAL_DEFAULTS.device_risk_score,
      ip_risk_score: form.preset_risk ? String(form.preset_risk.ip_risk_score) : TECHNICAL_DEFAULTS.ip_risk_score,
      location_risk_score: form.preset_risk ? String(form.preset_risk.location_risk_score) : TECHNICAL_DEFAULTS.location_risk_score,
      device_id: form.preset_risk?.device_id ?? TECHNICAL_DEFAULTS.device_id,
      device_shared_users_24h: form.preset_risk ? String(form.preset_risk.device_shared_users_24h) : TECHNICAL_DEFAULTS.device_shared_users_24h,
      account_age_days: form.preset_risk ? String(form.preset_risk.account_age_days) : TECHNICAL_DEFAULTS.account_age_days,
      sim_change_recent: form.preset_risk?.sim_change_recent ?? false,
      tx_type: (form.tx_type as TxType) || "MERCHANT",
      channel: form.preset_risk?.channel ?? "APP",
      cash_flow_velocity_1h: form.preset_risk ? String(form.preset_risk.cash_flow_velocity_1h) : TECHNICAL_DEFAULTS.cash_flow_velocity_1h,
      p2p_counterparties_24h: form.preset_risk ? String(form.preset_risk.p2p_counterparties_24h) : TECHNICAL_DEFAULTS.p2p_counterparties_24h,
      is_cross_border: Boolean(form.is_cross_border),
      currency: form.currency,
      source_country: form.source_country,
      destination_country: form.destination_country,
      is_agent_assisted: Boolean(form.is_agent_assisted),
      connectivity_mode: form.connectivity_mode,
    };

    const payload = resolveTechnicalFields(fraudPayload);
    const requestBody = {
      ...payload,
      override_source: form.override_source,
      support_mode: form.support_mode,
      support_actor_id: form.support_actor_id,
      override_fields: form.override_fields ?? [],
      transaction_amount: toNum(payload.transaction_amount),
      device_risk_score: toNum(payload.device_risk_score),
      ip_risk_score: toNum(payload.ip_risk_score),
      location_risk_score: toNum(payload.location_risk_score),
      device_shared_users_24h: toNum(payload.device_shared_users_24h),
      account_age_days: toNum(payload.account_age_days),
      cash_flow_velocity_1h: toNum(payload.cash_flow_velocity_1h),
      p2p_counterparties_24h: toNum(payload.p2p_counterparties_24h),
      sim_change_recent: Boolean(payload.sim_change_recent),
      is_cross_border: Boolean(payload.is_cross_border),
      currency: form.currency?.trim().toUpperCase() || undefined,
      source_country: form.source_country,
      destination_country: form.destination_country,
      is_agent_assisted: Boolean(form.is_agent_assisted),
      connectivity_mode: form.connectivity_mode,
    };

    if (form.support_mode) {
      Object.assign(requestBody, {
        override_source: form.override_source,
        support_mode: form.support_mode,
        support_actor_id: form.support_actor_id,
        override_fields: form.override_fields ?? [],
      });
    }

    try {
      const scored = (await scoreTransactionRequest(requestBody)) as FraudResponse;
      setResultDecision(scored.decision);
      setResultReasons(scored.fraud_reasons ?? scored.reasons ?? []);
      setResultRuntimeMode(scored.runtime_mode ?? "primary");
      setResultTransaction({
        merchant: form.merchant_name || "Unknown merchant",
        amount: `${(form.currency || "SGD").toUpperCase()} ${Number.parseFloat(form.transaction_amount || "0").toFixed(2)}`,
        type: form.tx_type,
      });
      
      // Save transaction to history
      const transactionId = `tx_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      addTransactionToHistory({
        id: transactionId,
        userId: form.user_id || 'Anonymous',
        status: scored.decision,
        transactionType: form.tx_type || 'MERCHANT',
        amount: form.transaction_amount || '0',
        merchantName: form.merchant_name || 'Unknown merchant',
        walletId: form.wallet_id || 'Unknown',
        currency: (form.currency || 'SGD').toUpperCase(),
        crossBorder: Boolean(form.is_cross_border),
        timestamp: new Date().toISOString(),
        riskScore: scored.final_risk_score ?? scored.risk_score,
        latencyMs: scored.stage_timings_ms?.total_pipeline_ms,
        explainabilityBase: scored.explainability?.base,
        explainabilityContext: scored.explainability?.context,
        explainabilityBehavior: scored.explainability?.behavior,
        explainabilityRing: scored.explainability?.ring,
        explainabilityExternal: scored.explainability?.external,
        mcpWatchlistHit: scored.context_summary?.mcp_watchlist_hit,
        mcpRiskTier: scored.context_summary?.mcp_risk_tier,
        runtimeMode: scored.runtime_mode,
        corridor: scored.corridor ?? undefined,
      });
      
      // Store transaction ID and form data for wallet authorization
      setLastTransactionId(transactionId);
      setLastFormData(form);
      
      setView("result");
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Failed to score transaction.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleAuthorizePayment = async () => {
    if (!lastTransactionId || !lastFormData) {
      setSubmitError("No transaction to authorize");
      return;
    }

    setIsAuthorizing(true);
    setSubmitError("");

    try {
      const walletResult = (await authorizePaymentRequest({
          user_id: lastFormData.user_id ?? "",
          transaction_amount: Number.parseFloat(lastFormData.transaction_amount),
          wallet_id: lastFormData.wallet_id,
          merchant_name: lastFormData.merchant_name,
          currency: lastFormData.currency,
          source_country: lastFormData.source_country,
          destination_country: lastFormData.destination_country,
          is_agent_assisted: Boolean(lastFormData.is_agent_assisted),
          connectivity_mode: lastFormData.connectivity_mode,
        })) as WalletResponse;

      // Update the transaction with wallet result
      updateTransactionInHistory(lastTransactionId, {
        walletAction: walletResult.wallet_action,
        walletMessage: walletResult.wallet_message,
        walletDecision: walletResult.fraud_engine_decision,
        nextStep: walletResult.next_step,
        runtimeMode: walletResult.runtime_mode,
        corridor: walletResult.corridor ?? undefined,
      });
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Failed to authorize payment");
    } finally {
      setIsAuthorizing(false);
    }
  };

  if (view === "form") {
    return (
      <CheckYourPayment
        locale={locale}
        onSubmit={handleSubmitPayment}
        isSubmitting={isSubmitting}
        submitError={submitError}
        onBack={() => setView("landing")}
        supportMode={supportModeEnabled}
        supportActorId={supportActorIdValue}
        onSupportModeChange={handleSupportModeToggle}
        onSupportActorIdChange={handleSupportActorIdUpdate}
      />
    );
  }

  if (view === "result") {
    return (
      <FraudResultScreen
        locale={locale}
        decision={resultDecision}
        reasons={resultReasons}
        runtimeMode={resultRuntimeMode}
        transaction={resultTransaction}
        onGoBack={() => setView("landing")}
        onCheckAnother={() => {
          setSubmitError("");
          setView("form");
        }}
        onAuthorize={handleAuthorizePayment}
        isAuthorizing={isAuthorizing}
      />
    );
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        background: tokens.bgBase,
        fontFamily: "'Inter Tight', sans-serif",
        maxWidth: 520,
        margin: "0 auto",
        overflowX: "hidden",
      }}
    >
      {/* ── HERO ── */}
      <div
        style={{
          position: "relative",
          height: 280,
          overflow: "hidden",
          background: "linear-gradient(160deg, #0a1628 0%, #0d1117 100%)",
        }}
      >
        <HeroCanvas />

        {/* Logo */}
        <div
          style={{
            position: "absolute",
            top: 24,
            left: 18,
            zIndex: 2,
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        >
        <img src="/Logo.svg" alt="Kawal" style={{ width: 100, height: 32 }} />
        </div>

        {/* language switcher */}
        <div
          style={{
            position: "absolute",
            top: 18,
            right: 18,
            zIndex: 2,
          }}
        >
          <select
            aria-label={translate("lang.label")}
            value={locale}
            onChange={(e) => setLocale(e.target.value as Locale)}
            style={{
              appearance: "none",
              WebkitAppearance: "none",
              MozAppearance: "none",
              background: "rgba(67, 76, 108, 0.55)",
              border: "1px solid rgba(207, 216, 239, 0.28)",
              color: "#eef3ff",
              borderRadius: 12,
              padding: "10px 32px 10px 12px",
              fontSize: 15,
              lineHeight: 1.2,
              fontFamily: "'Inter Tight', sans-serif",
              cursor: "pointer",
              backdropFilter: "blur(8px)",
              boxShadow: "0 6px 18px rgba(0, 0, 0, 0.2)",
            }}
          >
            {localeOptions.map((option) => (
              <option key={option.code} value={option.code}>
                {translate(option.labelKey)}
              </option>
            ))}
          </select>
          <span
            style={{
              position: "absolute",
              right: 14,
              top: "50%",
              transform: "translateY(-50%)",
              pointerEvents: "none",
              color: "#eef3ff",
              fontSize: 12,
            }}
          >
              &#9662;
          </span>
        </div>

        {/* gradient fade bottom */}
        <div
          style={{
            position: "absolute",
            bottom: 0, left: 0, right: 0,
            height: 120,
            background: `linear-gradient(to bottom, transparent, ${tokens.bgBase})`,
          }}
        />

        {/* hero text */}
        <div
          style={{
            position: "absolute",
            bottom: 0, left: 24, right: 24,
            opacity: heroLoaded ? 1 : 0,
            transform: heroLoaded ? "translateY(0)" : "translateY(16px)",
            transition: "opacity 0.7s ease, transform 0.7s ease",
          }}
        >
          <h1
            style={{
              margin: "0 0 8px",
              fontSize: 36,
              fontWeight: 600,
              color: tokens.textPrimary,
              lineHeight: 1.15,
            }}
          >
            {translate("landing.hero.title")}
          </h1>
          <p style={{ margin: 0, fontSize: 14, color: tokens.textSecondary, lineHeight: 1.6 }}>
            {translate("landing.hero.desc")}
          </p>
        </div>
      </div>

      {/* ── BODY ── */}
      <div style={{ padding: "24px 20px 48px" }}>

        {/* Start CTA */}
        <div
          style={{
            opacity: contentLoaded ? 1 : 0,
            transform: contentLoaded ? "translateY(0)" : "translateY(10px)",
            transition: "opacity 0.5s ease, transform 0.5s ease",
            marginBottom: 32,
          }}
        >
          <button
            onClick={() => setView('form')}
            onMouseEnter={() => setBtnHover(true)}
            onMouseLeave={() => setBtnHover(false)}
            style={{
              width: "100%",
              background: btnHover ? tokens.blueHover : tokens.blueBright,
              color: btnHover ? "#1e3a8a" : "#1e3a8a",
              border: "none",
              borderRadius: 14,
              padding: "16px",
              fontSize: 15,
              fontWeight: 700,
              fontFamily: "inherit",
              cursor: "pointer",
              transition: "all 0.2s ease",
            }}
          >
            {translate("landing.start")}
          </button>
        </div>

        {/* Privacy section header */}
        <div
          style={{
            marginBottom: 16,
            opacity: contentLoaded ? 1 : 0,
            transition: "opacity 0.5s ease 0.1s",
          }}
        >
          <h2
            style={{
              margin: "0 0 6px",
              fontSize: 18,
              fontWeight: 500,
              color: tokens.textPrimary,
            }}
          >
            {translate("landing.privacy.title")}
          </h2>
          <p style={{ margin: 0, fontSize: 13, color: tokens.textSecondary, lineHeight: 1.6 }}>
            {translate("landing.privacy.desc")}
          </p>
        </div>

        {/* Privacy cards */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 28 }}>
          {PRIVACY_ITEMS.map((item, i) => (
            <PrivacyCard
              key={i}
              icon={item.icon}
              title={translate(item.titleKey)}
              body={translate(item.bodyKey)}
              delay={500 + i * 100}
            />
          ))}
        </div>

        {/* Bottom CTA card */}
        <div
          style={{
            background: tokens.bgSurface,
            border: `0.5px solid ${tokens.border}`,
            borderRadius: 20,
            padding: "28px 20px",
            textAlign: "center",
            opacity: contentLoaded ? 1 : 0,
            transition: "opacity 0.5s ease 0.9s",
          }}
        >
          <h3
            style={{
              margin: "0 0 8px",
              fontSize: 24,
              fontWeight: 600,
              color: tokens.textPrimary,
              lineHeight: 1.25,
            }}
          >
            {translate("landing.cta.title")}
          </h3>
          <p style={{ margin: "0 0 20px", fontSize: 13, color: tokens.textSecondary, lineHeight: 1.6 }}>
            {translate("landing.cta.desc")}
          </p>

          <button
            onClick={() => setView('form')}
            onMouseEnter={() => setCtaHover(true)}
            onMouseLeave={() => setCtaHover(false)}
            style={{
              width: "100%",
              background: ctaHover ? "#c6c6c6" : "#ffffffff",
              color: "#000000ff",
              border: "none",
              borderRadius: 14,
              padding: "16px",
              fontSize: 15,
              fontWeight: 700,
              fontFamily: "inherit",
              cursor: "pointer",
              transition: "background 0.2s ease",
            }}
          >
            {translate("landing.cta.button")}
          </button>
        </div>
      </div>
    </div>
  );
}

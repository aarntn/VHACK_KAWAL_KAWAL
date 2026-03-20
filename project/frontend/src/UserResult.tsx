import { useState, useEffect } from "react";
import { t, type Locale } from "./i18n";

// ── Design tokens ────────────────────────────────────────
const tokens = {
  bgBase: "#10141a",
  bgSurface: "#181c22",
  bgCard: "#1c2026",
  border: "rgba(66,71,84,0.05)",
  green: "#4edea3",
  greenDim: "rgba(78,222,163,0.1)",
  greenDeep: "rgba(0,165,114,0.2)",
  greenBorder: "rgba(78,222,163,0.2)",
  yellow: "#fbbf24",
  yellowDim: "rgba(251,191,36,0.1)",
  yellowDeep: "rgba(251,191,36,0.15)",
  yellowBorder: "rgba(251,191,36,0.3)",
  red: "#df5858",
  redDim: "rgba(223,88,88,0.1)",
  redDeep: "rgba(223,88,88,0.12)",
  redBorder: "rgba(223,88,88,0.2)",
  blue: "#4d8eff",
  blueBright: "#adc6ff",
  textPrimary: "#dfe2eb",
  textSecondary: "#c2c6d6",
  textMuted: "#8c909f",
} as const;

// ── Types ────────────────────────────────────────────────
export type PaymentType =
  | "merchant_purchase"
  | "p2p_transfer"
  | "bill_payment"
  | "topup"
  | "MERCHANT"
  | "P2P"
  | "CASH_IN"
  | "CASH_OUT"
  | (string & {});

export interface Transaction {
  merchant: string;
  amount: string;
  type: PaymentType;
}

export type Decision = "APPROVE" | "FLAG" | "BLOCK";

export interface FraudResultScreenProps {
  locale?: Locale;
  decision?: Decision;
  transaction?: Transaction;
  /** Raw reason strings from the backend — auto-translated to plain language */
  reasons?: string[];
  onGoBack?: () => void;
  onCheckAnother?: () => void;
  onAuthorize?: () => Promise<void>;
  isAuthorizing?: boolean;
}

function paymentTypeLabel(type: string, translate: (key: string) => string): string {
  const map: Record<string, string> = {
    merchant_purchase: "result.paymentType.merchant_purchase",
    p2p_transfer: "result.paymentType.p2p_transfer",
    bill_payment: "result.paymentType.bill_payment",
    topup: "result.paymentType.topup",
    MERCHANT: "option.txType.MERCHANT",
    P2P: "option.txType.P2P",
    CASH_IN: "option.txType.CASH_IN",
    CASH_OUT: "option.txType.CASH_OUT",
  };
  const key = map[type];
  return key ? translate(key) : type;
}

function translateReason(raw: string, translate: (key: string) => string): string {
  const reasonMap: Record<string, string> = {
    "High fraud probability from transaction model": "result.reason.highFraudProb",
    "Above-normal transaction amount": "result.reason.aboveNormalAmount",
    "Context contributed to elevated risk": "result.reason.contextElevated",
    "Behavior is consistent with the user's normal baseline": "result.reason.normalBaseline",
  };
  const key = reasonMap[raw];
  return key ? translate(key) : raw;
}

// ── Sub-components ───────────────────────────────────────

interface ResultIconProps {
  decision: Decision;
}

function ResultIcon({ decision }: ResultIconProps) {
  const [popped, setPopped] = useState<boolean>(false);

  useEffect(() => {
    const t = setTimeout(() => setPopped(true), 120);
    return () => clearTimeout(t);
  }, []);

  const getColors = (decision: Decision) => {
    if (decision === "APPROVE") {
      return { color: tokens.green, dimColor: tokens.greenDim, deepColor: tokens.greenDeep, borderColor: tokens.greenBorder };
    }
    if (decision === "FLAG") {
      return { color: tokens.yellow, dimColor: tokens.yellowDim, deepColor: tokens.yellowDeep, borderColor: tokens.yellowBorder };
    }
    return { color: tokens.red, dimColor: tokens.redDim, deepColor: tokens.redDeep, borderColor: tokens.redBorder };
  };

  const { color, dimColor, deepColor, borderColor } = getColors(decision);

  return (
    <div style={{ position: "relative", display: "inline-block", margin: "0 auto 24px" }}>
      {/* Background blur glow */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: dimColor,
          filter: "blur(32px)",
          borderRadius: "9999px",
          opacity: popped ? 1 : 0,
          transform: popped ? "scale(1)" : "scale(0.6)",
          transition: "opacity 0.4s ease, transform 0.4s cubic-bezier(0.34,1.56,0.64,1)",
        }}
      />
      {/* Badge container with border */}
      <div
        style={{
          position: "relative",
          width: 96,
          height: 96,
          borderRadius: "9999px",
          background: deepColor,
          border: `1px solid ${borderColor}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: 1,
          opacity: popped ? 1 : 0,
          transform: popped ? "scale(1)" : "scale(0.6)",
          transition: "opacity 0.4s ease, transform 0.4s cubic-bezier(0.34,1.56,0.64,1)",
        }}
      >
        {/* Icon container */}
        <div
          style={{
            width: 40,
            height: 40,
            borderRadius: "50%",
            background: color,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          {decision === "APPROVE" ? (
            <svg width="22" height="18" viewBox="0 0 22 18" fill="none">
              <path
                d="M2 9L8 15L20 2"
                stroke="white"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          ) : (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path
                d="M6 6L18 18M18 6L6 18"
                stroke="white"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          )}
        </div>
      </div>
    </div>
  );
}

interface SummaryRowProps {
  label: string;
  value: string;
  last?: boolean;
}

function SummaryRow({ label, value, last = false }: SummaryRowProps) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "12px 0",
        borderBottom: last ? "none" : `1px solid hsla(0,0%,100%,0.04)`,
      }}
    >
      <span style={{ fontSize: 12, color: tokens.textSecondary, fontFamily: "'Inter Tight', sans-serif", letterSpacing: 0.3 }}>{label}</span>
      <span style={{ fontSize: 14, fontWeight: 400, color: tokens.textPrimary, letterSpacing: 0.3, fontFamily: "'Inter Tight', sans-serif" }}>{value}</span>
    </div>
  );
}

interface ReasonsBoxProps {
  decision: Decision;
  reasons: string[];
  translate: (key: string) => string;
}

function ReasonsBox({ decision, reasons, translate }: ReasonsBoxProps) {
  const getReasonColors = (decision: Decision) => {
    if (decision === "APPROVE") {
      return {
        color: tokens.green,
        borderColor: tokens.greenBorder,
        tagBg: "rgba(78,222,163,0.1)",
        tagBorder: tokens.greenBorder,
        tagLabel: translate("result.tag.safe"),
      };
    }
    if (decision === "FLAG") {
      return {
        color: tokens.yellow,
        borderColor: tokens.yellowBorder,
        tagBg: "rgba(251,191,36,0.1)",
        tagBorder: tokens.yellowBorder,
        tagLabel: "CAUTION",
      };
    }
    return {
      color: tokens.red,
      borderColor: tokens.redBorder,
      tagBg: "rgba(223,88,88,0.1)",
      tagBorder: "rgba(223,88,88,0.2)",
      tagLabel: translate("result.tag.worried"),
    };
  };

  const { color, borderColor, tagBg, tagBorder, tagLabel } = getReasonColors(decision);

  // Show first 5 reasons, same as AdminDashboard
  const filteredReasons = reasons.slice(0, 5);

  return (
    <div style={{ marginBottom: 24 }}>
      {/* Tag */}
      <div
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 4,
          background: tagBg,
          border: `1px solid ${tagBorder}`,
          borderRadius: 9999,
          padding: "5px 11px",
          marginBottom: 14,
        }}
      >
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: color,
            flexShrink: 0,
          }}
        />
        <span
          style={{
            fontSize: 10,
            fontWeight: 700,
            color: color,
            fontFamily: "'Inter Tight', sans-serif",
            letterSpacing: 0.3,
            textTransform: "uppercase",
          }}
        >
          {tagLabel}
        </span>
      </div>

      {/* Reason list */}
      <div
        style={{
          background: tokens.bgCard,
          border: `1px solid ${borderColor}`,
          borderRadius: 12,
          overflow: "hidden",
        }}
      >
        {filteredReasons.map((r, i) => (
          <div
            key={i}
            style={{
              padding: "21px",
              borderBottom:
                i < filteredReasons.length - 1 ? `1px solid ${borderColor}` : "none",
              display: "flex",
              alignItems: "flex-start",
              gap: 0,
            }}
          >
            <span style={{ fontSize: 12, color: tokens.textPrimary, lineHeight: 1.4, fontFamily: "'Inter Tight', sans-serif", letterSpacing: 0.3 }}>
              {translateReason(r, translate)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main component ───────────────────────────────────────
const DEFAULT_APPROVE_REASONS = ["Behavior is consistent with the user's normal baseline"];
const DEFAULT_FLAG_REASONS = [
  "High fraud probability from transaction model",
  "Above-normal transaction amount",
  "Context contributed to elevated risk",
];

const DEFAULT_TRANSACTION: Transaction = {
  merchant: "GOJEK",
  amount: "$12.40",
  type: "merchant_purchase",
};

export default function FraudResultScreen({
  locale = "en",
  decision = "APPROVE",
  transaction = DEFAULT_TRANSACTION,
  reasons = decision === "APPROVE" ? DEFAULT_APPROVE_REASONS : DEFAULT_FLAG_REASONS,
  onGoBack,
  onCheckAnother,
  onAuthorize,
  isAuthorizing = false,
}: FraudResultScreenProps) {
  const [headerVisible, setHeaderVisible] = useState<boolean>(false);
  const [bodyVisible, setBodyVisible] = useState<boolean>(false);
  const [finishBtnHover, setFinishBtnHover] = useState<boolean>(false);
  const [checkAgainHover, setCheckAgainHover] = useState<boolean>(false);
  const translate = (key: string, vars?: Record<string, string | number>): string => t(locale, key, vars);

  const handleFinish = async () => {
    if (onAuthorize && !isAuthorizing) {
      await onAuthorize();
    }
    if (onGoBack) {
      onGoBack();
    }
  };

  useEffect(() => {
    const t1 = setTimeout(() => setHeaderVisible(true), 80);
    const t2 = setTimeout(() => setBodyVisible(true), 350);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, []);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: tokens.bgBase,
        fontFamily: "'Inter Tight', sans-serif",
        letterSpacing: 0.3,
        maxWidth: 520,
        margin: "0 auto",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Scrollable body */}
      <div style={{ flex: 1, padding: "32px 16px 140px 16px", overflowY: "auto" }}>

        {/* Hero */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            marginBottom: 32,
            opacity: headerVisible ? 1 : 0,
            transform: headerVisible ? "translateY(0)" : "translateY(12px)",
            transition: "opacity 0.5s ease, transform 0.5s ease",
          }}
        >
          <ResultIcon decision={decision} />
          <h1
            style={{
              margin: "0 0 4px",
              fontSize: 24,
              fontWeight: 700,
              color: tokens.textPrimary,
              lineHeight: 1.33,
              textAlign: "center",
            }}
          >
            {decision === "APPROVE" ? translate("result.hero.safeTitle") : decision === "FLAG" ? "Action Required" : translate("result.hero.warnTitle")}
          </h1>
          <p
            style={{
              margin: 0,
              fontSize: 14,
              color: "#d2d2d2",
              lineHeight: 1.71,
              maxWidth: 280,
              textAlign: "center",
              fontWeight: 400,
            }}
          >
            {decision === "APPROVE"
              ? translate("result.hero.safeDesc")
              : decision === "FLAG"
              ? "We need to verify your identity to proceed with this transaction."
              : translate("result.hero.warnDesc")}
          </p>
        </div>

        {/* Body */}
        <div
          style={{
            opacity: bodyVisible ? 1 : 0,
            transform: bodyVisible ? "translateY(0)" : "translateY(10px)",
            transition: "opacity 0.5s ease, transform 0.5s ease",
          }}
        >
          <p
            style={{
              margin: "0 0 16px",
              fontSize: 16,
              fontWeight: 600,
              color: tokens.textPrimary,
            }}
          >
            {translate("result.summary.title")}
          </p>

          <div
            style={{
              background: tokens.bgCard,
              border: `1px solid ${tokens.border}`,
              borderRadius: 16,
              padding: "17px",
              marginBottom: 24,
            }}
          >
            <SummaryRow label={translate("result.summary.payingTo")} value={transaction.merchant} />
            <SummaryRow label={translate("result.summary.amount")} value={transaction.amount} />
            <SummaryRow
              label={translate("result.summary.paymentType")}
              value={paymentTypeLabel(transaction.type, translate)}
              last
            />
          </div>

          <ReasonsBox decision={decision} reasons={reasons} translate={translate} />
        </div>
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
          gap: 14,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <button
          onMouseEnter={() => setFinishBtnHover(true)}
          onMouseLeave={() => setFinishBtnHover(false)}
          onClick={handleFinish}
          disabled={isAuthorizing}
          style={{
            width: "100%",
            background: finishBtnHover ? "#9db4ff" : tokens.blueBright,
            color: "#002e6a",
            border: "none",
            borderRadius: 12,
            padding: "12.5px 16px",
            fontSize: 14,
            fontWeight: 700,
            fontFamily: "'Inter Tight', sans-serif",
            letterSpacing: 0.3,
            cursor: isAuthorizing ? "not-allowed" : "pointer",
            opacity: isAuthorizing ? 0.6 : 1,
            transition: "background 0.2s ease, opacity 0.2s ease",
            boxShadow: finishBtnHover && !decision || !isAuthorizing
              ? "0px 10px 15px -3px rgba(173,198,255,0.2), 0px 4px 6px -4px rgba(173,198,255,0.2)"
              : "0px 10px 15px -3px rgba(173,198,255,0.2), 0px 4px 6px -4px rgba(173,198,255,0.2)",
          }}
        >
          {translate("result.button.finish")}
        </button>
        <button
          onMouseEnter={() => setCheckAgainHover(true)}
          onMouseLeave={() => setCheckAgainHover(false)}
          onClick={onCheckAnother}
          style={{
            width: "100%",
            background: checkAgainHover ? "#2f2f2f" : "#242424",
            color: "#fff",
            border: "none",
            borderRadius: 12,
            padding: "12.5px 16px",
            fontSize: 14,
            fontWeight: 600,
            fontFamily: "'Inter Tight', sans-serif",
            letterSpacing: 0.3,
            cursor: "pointer",
            transition: "background 0.2s ease",
          }}
        >
          {translate("result.button.checkAgain")}
        </button>
      </div>
    </div>
  );
}
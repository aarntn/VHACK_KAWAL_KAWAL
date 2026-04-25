export const toNum = (value: string): number => {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

export const clampRisk = (value: number): number => Math.min(0.99, Math.max(0.01, value));

export const TECHNICAL_DEFAULTS = {
  device_risk_score: "0.20",
  ip_risk_score: "0.22",
  location_risk_score: "0.18",
  device_id: "device_default_app",
  device_shared_users_24h: "1",
  account_age_days: "90",
  cash_flow_velocity_1h: "2",
  p2p_counterparties_24h: "1",
} as const;

type TxType = "MERCHANT" | "P2P" | "CASH_IN" | "CASH_OUT";

type TechnicalFieldPayload = {
  tx_type: TxType;
  is_cross_border: boolean;
  channel: string;
  device_risk_score: string;
  ip_risk_score: string;
  location_risk_score: string;
  device_id: string;
  device_shared_users_24h: string;
  account_age_days: string;
  cash_flow_velocity_1h: string;
  p2p_counterparties_24h: string;
};

export const resolveTechnicalFields = <T extends TechnicalFieldPayload>(payload: T): T => {
  const txRiskBaseline: Record<TxType, number> = {
    MERCHANT: 0.18,
    P2P: 0.28,
    CASH_IN: 0.24,
    CASH_OUT: 0.34,
  };
  const baseline = txRiskBaseline[payload.tx_type];
  const crossBorderBoost = payload.is_cross_border ? 0.12 : 0;
  const defaultDeviceRisk = clampRisk(baseline + crossBorderBoost);
  const defaultIpRisk = clampRisk(defaultDeviceRisk + 0.04);
  const defaultLocationRisk = clampRisk(defaultDeviceRisk + (payload.is_cross_border ? 0.06 : 0.01));

  return {
    ...payload,
    device_risk_score: payload.device_risk_score || defaultDeviceRisk.toFixed(2),
    ip_risk_score: payload.ip_risk_score || defaultIpRisk.toFixed(2),
    location_risk_score: payload.location_risk_score || defaultLocationRisk.toFixed(2),
    device_id: payload.device_id || `device_${payload.channel.toLowerCase()}_default`,
    device_shared_users_24h: payload.device_shared_users_24h || TECHNICAL_DEFAULTS.device_shared_users_24h,
    account_age_days: payload.account_age_days || TECHNICAL_DEFAULTS.account_age_days,
    cash_flow_velocity_1h: payload.cash_flow_velocity_1h || TECHNICAL_DEFAULTS.cash_flow_velocity_1h,
    p2p_counterparties_24h: payload.p2p_counterparties_24h || TECHNICAL_DEFAULTS.p2p_counterparties_24h,
  };
};

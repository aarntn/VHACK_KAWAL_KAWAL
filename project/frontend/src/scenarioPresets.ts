import scenarioPresetJson from "./scenarioPresets.json";

export type ScenarioId = "everyday_purchase" | "large_amount" | "agent_cash_out" | "cross_border" | "custom";

type TxType = "MERCHANT" | "P2P" | "CASH_IN" | "CASH_OUT";
type Channel = "APP" | "WEB" | "AGENT" | "QR";

type ScenarioUiPreset = {
  icon: "ShoppingCart" | "WalletCards" | "Globe" | "Edit3";
  amount: string;
  tx_type: TxType;
  user_id: string;
  wallet_id: string;
  currency: string;
  merchant_name: string;
  is_cross_border: boolean;
  source_country?: "SG" | "MY" | "ID" | "TH" | "PH" | "VN";
  destination_country?: "SG" | "MY" | "ID" | "TH" | "PH" | "VN";
  is_agent_assisted?: boolean;
  connectivity_mode?: "online" | "intermittent" | "offline_buffered";
};

type ScenarioRiskPreset = {
  schema_version: string;
  device_risk_score: number;
  ip_risk_score: number;
  location_risk_score: number;
  device_id: string;
  device_shared_users_24h: number;
  account_age_days: number;
  sim_change_recent: boolean;
  channel: Channel;
  cash_flow_velocity_1h: number;
  p2p_counterparties_24h: number;
};

export type ScenarioPreset = {
  id: ScenarioId;
  ui: ScenarioUiPreset;
  risk: ScenarioRiskPreset;
};

export const SCENARIO_PRESETS = scenarioPresetJson as Record<ScenarioId, ScenarioPreset>;

export const SCENARIO_ORDER: ScenarioId[] = [
  "everyday_purchase",
  "large_amount",
  "agent_cash_out",
  "cross_border",
  "custom",
];

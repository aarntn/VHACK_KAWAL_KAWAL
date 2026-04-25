export type ApiErrorDetail = {
  field?: string;
  message?: string;
  code?: string;
  input?: unknown;
};

export type ApiErrorResponse = {
  type?: string;
  error?: string;
  message?: string;
  detail?: string;
  field?: string;
  details?: ApiErrorDetail[];
};

export type DashboardViewsResponse = {
  window_hours: number;
  generated_at_utc: string;
  freshest_record_utc?: string | null;
  data_freshness_seconds?: number | null;
  latency_throughput_error: {
    requests: number;
    throughput_per_min: number;
    error_rate: number;
    latency_ms_p50: number;
    latency_ms_p95: number;
  };
  drift_score_distribution: {
    baseline_mean_score: number;
    observed_mean_score: number;
    mean_delta: number;
    score_histogram: Record<string, number>;
  };
  fraud_loss_false_positives_analyst_agreement: {
    estimated_fraud_loss: number;
    false_positives: number;
    analyst_agreement: number;
    confirmed_fraud_cases: number;
    analyst_reviews: number;
  };
  decision_source_kpis?: Array<{
    decision_source: 'score_band' | 'hard_rule_override' | 'low_history_policy' | 'step_up_policy';
    audit_volume: number;
    flag_rate: number;
    analyst_reviews: number;
    confirmed_fraud_conversion: number;
    false_positive_rate: number;
  }>;
  recent_transactions?: Array<{
    request_id: string;
    timestamp_utc: string;
    user_id: string;
    transaction_amount: number;
    currency?: string | null;
    tx_type: string;
    decision: 'APPROVE' | 'FLAG' | 'BLOCK';
    final_risk_score: number;
    latency_ms: number;
    is_cross_border: boolean;
    reason_codes?: string[];
  }>;
  decision_counts?: {
    approve: number;
    flag: number;
    block: number;
  };
};

export type FraudApiInfoResponse = {
  status: string;
  api_version: string;
  model_name: string;
  model_version: string;
  feature_count: number;
  approve_threshold: number;
  block_threshold: number;
};

export type RuntimeThresholds = {
  approveThreshold: number;
  blockThreshold: number;
};

export type WalletApiInfoResponse = {
  status: string;
  service: string;
  api_version: string;
  fraud_engine_url: string;
  fraud_engine_health_url: string;
  resilience: {
    timeout_seconds: number;
    max_retries: number;
    retry_status_codes: number[];
    backoff_ms: number;
    backoff_max_ms: number;
    backoff_jitter_ratio: number;
    circuit_breaker_failure_threshold: number;
    circuit_breaker_reset_seconds: number;
    fallback_engine_decision: string;
    max_inflight_requests: number;
    uvicorn_workers: number;
    runtime_init_count: number;
  };
  message: string;
};

export type ScoreTransactionPayload = {
  schema_version: string;
  user_id: string;
  transaction_amount: number;
  currency?: string;
  device_risk_score: number;
  ip_risk_score: number;
  location_risk_score: number;
  device_id: string;
  device_shared_users_24h: number;
  account_age_days: number;
  sim_change_recent: boolean;
  tx_type: string;
  channel: string;
  cash_flow_velocity_1h: number;
  p2p_counterparties_24h: number;
  is_cross_border: boolean;
  source_country?: 'SG' | 'MY' | 'ID' | 'TH' | 'PH' | 'VN';
  destination_country?: 'SG' | 'MY' | 'ID' | 'TH' | 'PH' | 'VN';
  is_agent_assisted?: boolean;
  connectivity_mode?: 'online' | 'intermittent' | 'offline_buffered';
  [key: string]: unknown;
};

export type ScoreTransactionResponse = {
  decision: 'APPROVE' | 'FLAG' | 'BLOCK';
  decision_source?: 'score_band' | 'hard_rule_override' | 'low_history_policy' | 'step_up_policy';
  risk_score?: number;
  final_risk_score?: number;
  fraud_reasons?: string[];
  reasons?: string[];
  reason_codes?: string[];
  ring_match_type?: 'account_member' | 'attribute_match' | 'none';
  ring_evidence_summary?: Record<string, unknown>;
  runtime_mode?: 'primary' | 'cached_context' | 'degraded_local';
  corridor?: string | null;
  normalized_amount_reference?: number | null;
  normalization_basis?: string | null;
  explainability?: {
    base: number;
    context: number;
    behavior: number;
    ring?: number;
    external?: number;
    top_feature_drivers?: Array<{ feature: string; shap_value: number; direction: string }>;
  };
  stage_timings_ms?: {
    total_pipeline_ms?: number;
  };
};

export type AuthorizePaymentPayload = {
  user_id: string;
  transaction_amount: number;
  wallet_id?: string;
  merchant_name?: string;
  currency?: string;
  source_country?: 'SG' | 'MY' | 'ID' | 'TH' | 'PH' | 'VN';
  destination_country?: 'SG' | 'MY' | 'ID' | 'TH' | 'PH' | 'VN';
  is_agent_assisted?: boolean;
  connectivity_mode?: 'online' | 'intermittent' | 'offline_buffered';
  [key: string]: unknown;
};

export type AuthorizePaymentResponse = {
  wallet_action: 'APPROVED' | 'PENDING_VERIFICATION' | 'DECLINED' | 'DECLINED_FRAUD_RISK';
  wallet_message?: string;
  fraud_engine_decision: 'APPROVE' | 'FLAG' | 'BLOCK';
  final_risk_score?: number;
  next_step?: string;
  runtime_mode?: 'primary' | 'cached_context' | 'degraded_local';
  reason_codes?: string[];
  corridor?: string | null;
  normalized_amount_reference?: number | null;
  normalization_basis?: string | null;
};

export type ApiReferenceCard = {
  path: '/score' | '/explain' | '/audit';
  method: 'POST' | 'GET';
  summary: string;
  latencyMs: number;
  sampleRequest: Record<string, unknown>;
  sampleResponse: Record<string, unknown>;
};

const DEFAULT_HOST = typeof window !== 'undefined' ? window.location.hostname : '127.0.0.1';
const DEFAULT_PROTOCOL = typeof window !== 'undefined' ? window.location.protocol : 'http:';

const FRAUD_BASE_URL = import.meta.env.VITE_FRAUD_API_BASE_URL ?? `${DEFAULT_PROTOCOL}//${DEFAULT_HOST}:8000`;
const WALLET_BASE_URL = import.meta.env.VITE_WALLET_API_BASE_URL ?? `${DEFAULT_PROTOCOL}//${DEFAULT_HOST}:8001`;
const WAIT_MS = 320;
let operatorAccessCode: string = (import.meta.env.VITE_DEFAULT_OPERATOR_KEY as string | undefined ?? '').trim();

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));
const round = (value: number, digits = 3): number => Number(value.toFixed(digits));

export const SANDBOX_TOKEN_LABEL = 'sandbox tenant token issued via operator onboarding';

export function setOperatorAccessCode(value: string): void {
  operatorAccessCode = value.trim();
}

export function clearOperatorAccessCode(): void {
  operatorAccessCode = '';
}

export function hasOperatorAccessCode(): boolean {
  return operatorAccessCode.length > 0;
}

export const DEFAULT_SCORE_TEST_PAYLOAD: ScoreTransactionPayload = {
  schema_version: 'ieee_fraud_tx_v1',
  user_id: 'user_sgph_demo',
  transaction_amount: 420,
  currency: 'SGD',
  device_risk_score: 0.46,
  ip_risk_score: 0.54,
  location_risk_score: 0.39,
  device_id: 'device_sgph_demo_01',
  device_shared_users_24h: 1,
  account_age_days: 14,
  sim_change_recent: false,
  tx_type: 'P2P',
  channel: 'APP',
  cash_flow_velocity_1h: 3,
  p2p_counterparties_24h: 1,
  is_cross_border: true,
  source_country: 'SG',
  destination_country: 'PH',
  is_agent_assisted: false,
  connectivity_mode: 'online',
};

export const API_REFERENCE_CARDS: ApiReferenceCard[] = [
  {
    path: '/score',
    method: 'POST',
    summary: 'Primary fraud scoring endpoint used before wallet authorization.',
    latencyMs: 128,
    sampleRequest: DEFAULT_SCORE_TEST_PAYLOAD,
    sampleResponse: {
      decision: 'APPROVE',
      final_risk_score: 0.24,
      reasons: ['Behavior is consistent with the user baseline'],
      stage_timings_ms: { total_pipeline_ms: 128 },
    },
  },
  {
    path: '/explain',
    method: 'POST',
    summary: 'Returns feature-level contribution details for support and analyst workflows.',
    latencyMs: 164,
    sampleRequest: {
      request_id: 'req_01JSM0Q8J0E4T48M0A',
      include_top_factors: true,
      top_k: 5,
    },
    sampleResponse: {
      request_id: 'req_01JSM0Q8J0E4T48M0A',
      top_factors: [
        { feature: 'device_risk_score', contribution: 0.11 },
        { feature: 'cash_flow_velocity_1h', contribution: 0.05 },
      ],
      narrative: 'Stable device and account tenure kept the risk below escalation threshold.',
    },
  },
  {
    path: '/audit',
    method: 'GET',
    summary: 'Audit trail retrieval for compliance review and internal investigations.',
    latencyMs: 187,
    sampleRequest: {
      request_id: 'req_01JSM0Q8J0E4T48M0A',
      include_manual_review: true,
    },
    sampleResponse: {
      request_id: 'req_01JSM0Q8J0E4T48M0A',
      events: [
        { at: '2026-04-21T09:12:04Z', action: 'scored', actor: 'fraud-engine-v3.4.1' },
        { at: '2026-04-21T09:12:04Z', action: 'decision.approved', actor: 'wallet-orchestrator' },
      ],
    },
  },
];

const MOCK_APPROVE_THRESHOLD = 0.42;
const MOCK_BLOCK_THRESHOLD = 0.75;

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }
    throw new Error('Response JSON parsing failed');
  }

  if (!response.ok) {
    const err = payload as ApiErrorResponse;
    const message = err.message ?? err.detail ?? err.error ?? err.type ?? `Request failed with status ${response.status}`;
    const error = new Error(message);
    (error as Error & { apiError?: ApiErrorResponse; status?: number }).apiError = err;
    (error as Error & { apiError?: ApiErrorResponse; status?: number }).status = response.status;
    throw error;
  }

  return payload as T;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function buildMockReasons(payload: ScoreTransactionPayload, finalRiskScore: number): string[] {
  const reasons: string[] = [];

  if (finalRiskScore >= 0.75) {
    reasons.push('High fraud probability from transaction model');
  }
  if (payload.transaction_amount >= 750) {
    reasons.push('Above-normal transaction amount');
  }
  if (payload.is_cross_border || payload.sim_change_recent || payload.cash_flow_velocity_1h >= 8) {
    reasons.push('Context contributed to elevated risk');
  }
  if (!payload.is_cross_border && payload.channel === 'QR' && (payload.source_country ?? payload.destination_country) === 'ID') {
    reasons.push('Domestic QR merchant payment matches a normal local wallet pattern');
  }
  if (payload.is_cross_border && payload.tx_type === 'P2P' && payload.account_age_days <= 30) {
    reasons.push('First cross-border remittance on a new payment corridor');
  }
  if (payload.is_agent_assisted && payload.tx_type === 'CASH_OUT') {
    reasons.push('Agent-assisted cash-out requires extra review');
  }
  if (payload.cash_flow_velocity_1h >= 8 && payload.tx_type === 'CASH_OUT') {
    reasons.push('Rapid cash-in and cash-out movement increased risk');
  }
  if (
    payload.is_cross_border
    && payload.source_country
    && payload.destination_country
    && payload.source_country !== payload.destination_country
    && payload.location_risk_score >= 0.4
  ) {
    reasons.push('Device geography does not match the payment corridor');
  }
  if (payload.account_age_days <= 7 && finalRiskScore >= MOCK_APPROVE_THRESHOLD) {
    reasons.push('New wallet initiated a high-risk first transfer');
  }
  if (payload.connectivity_mode === 'offline_buffered') {
    reasons.push('Decision used offline-buffered scoring safeguards');
  } else if (payload.connectivity_mode === 'intermittent') {
    reasons.push('Intermittent connectivity triggered conservative verification safeguards');
  }
  if (reasons.length === 0) {
    reasons.push("Behavior is consistent with the user's normal baseline");
  }

  return reasons;
}

export function createMockScoreResponse(payload: ScoreTransactionPayload): ScoreTransactionResponse {
  const amountRisk = clamp(payload.transaction_amount / 1800, 0, 0.24);
  const deviceComponent = payload.device_risk_score * 0.26;
  const ipComponent = payload.ip_risk_score * 0.18;
  const locationComponent = payload.location_risk_score * 0.14;
  const velocityComponent = clamp(payload.cash_flow_velocity_1h / 15, 0, 1) * 0.12;
  const counterpartiesComponent = clamp(payload.p2p_counterparties_24h / 20, 0, 1) * 0.08;
  const accountAgePenalty = clamp((90 - payload.account_age_days) / 90, 0, 1) * 0.08;
  const contextBoost = (payload.is_cross_border ? 0.08 : 0) + (payload.sim_change_recent ? 0.11 : 0);

  const base = clamp(deviceComponent + ipComponent, 0.02, 0.55);
  const context = clamp(locationComponent + velocityComponent + counterpartiesComponent + contextBoost, 0.02, 0.42);
  const behavior = clamp(amountRisk + accountAgePenalty, 0.02, 0.28);
  const finalRiskScore = round(clamp(base + context + behavior, 0.04, 0.98), 4);

  const decision: ScoreTransactionResponse['decision'] = finalRiskScore >= MOCK_BLOCK_THRESHOLD
    ? 'BLOCK'
    : finalRiskScore >= MOCK_APPROVE_THRESHOLD
      ? 'FLAG'
      : 'APPROVE';

  const latency = Math.round(
    88
    + payload.transaction_amount * 0.04
    + payload.cash_flow_velocity_1h * 5
    + (payload.is_cross_border ? 24 : 0)
    + (payload.sim_change_recent ? 18 : 0),
  );

  const reasons = buildMockReasons(payload, finalRiskScore);
  const corridor = payload.source_country && payload.destination_country
    ? `${payload.source_country}-${payload.destination_country}`
    : payload.source_country
      ? `${payload.source_country}-${payload.source_country}`
      : null;
  const runtimeMode: ScoreTransactionResponse['runtime_mode'] = payload.connectivity_mode === 'offline_buffered'
    ? 'degraded_local'
    : payload.connectivity_mode === 'intermittent'
      ? 'cached_context'
      : 'primary';
  const normalizationBasis = payload.currency
    ? `frontend_mock_reference_snapshot:${payload.currency}`
    : 'frontend_mock_reference_snapshot';

  return {
    decision,
    risk_score: finalRiskScore,
    final_risk_score: finalRiskScore,
    fraud_reasons: reasons,
    reasons,
    reason_codes: reasons.map((reason) => reason.toUpperCase().replace(/[^A-Z0-9]+/g, '_')),
    runtime_mode: runtimeMode,
    corridor,
    normalized_amount_reference: round(payload.transaction_amount, 4),
    normalization_basis: normalizationBasis,
    explainability: {
      base: round(base, 3),
      context: round(context, 3),
      behavior: round(behavior, 3),
    },
    stage_timings_ms: {
      total_pipeline_ms: latency,
    },
  };
}

function buildOperatorHeaders(base: Record<string, string>): Record<string, string> {
  if (!operatorAccessCode) {
    return base;
  }
  return {
    ...base,
    'X-Operator-Api-Key': operatorAccessCode,
  };
}

export async function fetchDashboardViews(windowHours = 24): Promise<DashboardViewsResponse> {
  const boundedWindow = Math.max(1, Math.min(Math.trunc(windowHours), 24 * 30));
  const url = new URL(`${FRAUD_BASE_URL}/dashboard/views`);
  url.searchParams.set('window_hours', String(boundedWindow));
  const response = await fetch(url.toString(), {
    method: 'GET',
    headers: buildOperatorHeaders({ accept: 'application/json' }),
    cache: 'no-store',
  });
  return parseJsonOrThrow<DashboardViewsResponse>(response);
}

export async function fetchFraudInfo(): Promise<FraudApiInfoResponse> {
  const response = await fetch(`${FRAUD_BASE_URL}/api/info`, {
    method: 'GET',
    headers: { accept: 'application/json' },
    cache: 'no-store',
  });
  return parseJsonOrThrow<FraudApiInfoResponse>(response);
}

type FraudHealthResponse = Partial<FraudApiInfoResponse> & {
  approve_threshold?: number;
  block_threshold?: number;
};

const isValidThresholdPair = (approveThreshold: unknown, blockThreshold: unknown): boolean => {
  if (typeof approveThreshold !== 'number' || typeof blockThreshold !== 'number') {
    return false;
  }
  return Number.isFinite(approveThreshold)
    && Number.isFinite(blockThreshold)
    && approveThreshold >= 0
    && blockThreshold <= 1
    && approveThreshold < blockThreshold;
};

const toRuntimeThresholds = (source: FraudHealthResponse): RuntimeThresholds | null => {
  if (!isValidThresholdPair(source.approve_threshold, source.block_threshold)) {
    return null;
  }
  return {
    approveThreshold: source.approve_threshold as number,
    blockThreshold: source.block_threshold as number,
  };
};

export async function fetchRuntimeThresholds(): Promise<RuntimeThresholds> {
  const info = await fetchFraudInfo();
  const infoThresholds = toRuntimeThresholds(info);
  if (infoThresholds) {
    return infoThresholds;
  }

  const healthResponse = await fetch(`${FRAUD_BASE_URL}/health`, {
    method: 'GET',
    headers: { accept: 'application/json' },
    cache: 'no-store',
  });
  const health = await parseJsonOrThrow<FraudHealthResponse>(healthResponse);
  const healthThresholds = toRuntimeThresholds(health);
  if (!healthThresholds) {
    throw new Error('Runtime thresholds are unavailable from /api/info and /health');
  }
  return healthThresholds;
}

export async function fetchWalletInfo(): Promise<WalletApiInfoResponse> {
  const response = await fetch(`${WALLET_BASE_URL}/api/info`, {
    method: 'GET',
    headers: { accept: 'application/json' },
    cache: 'no-store',
  });
  return parseJsonOrThrow<WalletApiInfoResponse>(response);
}

export async function scoreTransaction(payload: ScoreTransactionPayload): Promise<ScoreTransactionResponse> {
  const response = await fetch(`${FRAUD_BASE_URL}/score_transaction`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
    cache: 'no-store',
  });
  return parseJsonOrThrow<ScoreTransactionResponse>(response);
}

export async function authorizePayment(payload: AuthorizePaymentPayload): Promise<AuthorizePaymentResponse> {
  const response = await fetch(`${WALLET_BASE_URL}/wallet/authorize_payment`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
    cache: 'no-store',
  });
  return parseJsonOrThrow<AuthorizePaymentResponse>(response);
}

export async function runMockScoreEndpoint(payload: ScoreTransactionPayload): Promise<ScoreTransactionResponse> {
  await wait(WAIT_MS);
  return createMockScoreResponse(payload);
}

export const apiConfig = {
  fraudBaseUrl: FRAUD_BASE_URL,
  walletBaseUrl: WALLET_BASE_URL,
};

export type RingGraphNode = {
  id: string;
  type: 'account' | 'attribute';
  tier?: 'high' | 'medium' | 'low';
  ring_id?: string;
  ring_score?: number;
  // attribute nodes
  attr_type?: 'device' | 'ip' | 'card' | 'unknown';
  display?: string;
  member_count?: number;
  first_seen_utc?: string | null;
  last_seen_utc?: string | null;
  window?: string | null;
};

export type RingMeta = {
  ring_id: string;
  score: number;
  tier: 'high' | 'medium' | 'low';
  fraud_rate?: number | null;
  size: number;
  attribute_types: string[];
  label_mode?: string;
};

export type RingGraphLink = {
  source: string;
  target: string;
  weight: number;
  edge_type?: string;
  attr_type?: 'device' | 'ip' | 'card' | 'unknown';
  support_count?: number;
  member_count?: number;
  first_seen_utc?: string | null;
  last_seen_utc?: string | null;
  window?: string | null;
};

export type RingGraphResponse = {
  nodes: RingGraphNode[];
  links: RingGraphLink[];
  rings: RingMeta[];
  summary: {
    total_rings: number;
    high_risk_rings: number;
    medium_risk_rings: number;
    total_accounts_in_rings: number;
    evidence_links_available?: boolean;
  };
};

export async function fetchRingGraph(): Promise<RingGraphResponse> {
  const response = await fetch(`${FRAUD_BASE_URL}/ring/graph`, {
    method: 'GET',
    headers: buildOperatorHeaders({ accept: 'application/json' }),
    cache: 'no-store',
  });
  return parseJsonOrThrow<RingGraphResponse>(response);
}

export type RingTopEntry = {
  ring_id: string;
  ring_score: number;
  tier: 'high' | 'medium' | 'low';
  ring_size: number;
  fraud_count: number | null;
  fraud_rate: number | null;
  attribute_types: string[];
  shared_attr_count: number;
  label_mode: string;
};

export type RingSummaryResponse = {
  total_rings: number;
  high_risk_rings: number;
  medium_risk_rings: number;
  low_risk_rings: number;
  total_accounts_in_rings: number;
  top_rings: RingTopEntry[];
  evidence_links_available: boolean;
  weight_model: { topology_r2: number; fraud_blend: number; trained_at_utc: string } | null;
  reports_generated_at: number | null;
};

export async function fetchRingSummary(topN = 10): Promise<RingSummaryResponse> {
  const response = await fetch(`${FRAUD_BASE_URL}/ring/summary?top_n=${topN}`, {
    method: 'GET',
    headers: buildOperatorHeaders({ accept: 'application/json' }),
    cache: 'no-store',
  });
  return parseJsonOrThrow<RingSummaryResponse>(response);
}

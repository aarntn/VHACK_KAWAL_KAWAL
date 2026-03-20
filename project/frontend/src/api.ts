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

const DEFAULT_HOST = typeof window !== 'undefined' ? window.location.hostname : '127.0.0.1';
const DEFAULT_PROTOCOL = typeof window !== 'undefined' ? window.location.protocol : 'http:';

const FRAUD_BASE_URL = import.meta.env.VITE_FRAUD_API_BASE_URL ?? `${DEFAULT_PROTOCOL}//${DEFAULT_HOST}:8000`;
const WALLET_BASE_URL = import.meta.env.VITE_WALLET_API_BASE_URL ?? `${DEFAULT_PROTOCOL}//${DEFAULT_HOST}:8001`;

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
    (error as Error & { apiError?: ApiErrorResponse }).apiError = err;
    throw error;
  }

  return payload as T;
}

export async function fetchDashboardViews(windowHours = 24): Promise<DashboardViewsResponse> {
  const boundedWindow = Math.max(1, Math.min(Math.trunc(windowHours), 24 * 30));
  const url = new URL(`${FRAUD_BASE_URL}/dashboard/views`);
  url.searchParams.set('window_hours', String(boundedWindow));
  const response = await fetch(url.toString(), {
    method: 'GET',
    headers: { accept: 'application/json' },
  });
  return parseJsonOrThrow<DashboardViewsResponse>(response);
}

export async function fetchFraudInfo(): Promise<FraudApiInfoResponse> {
  const response = await fetch(`${FRAUD_BASE_URL}/api/info`, {
    method: 'GET',
    headers: { accept: 'application/json' },
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
  });
  return parseJsonOrThrow<WalletApiInfoResponse>(response);
}

export const apiConfig = {
  fraudBaseUrl: FRAUD_BASE_URL,
  walletBaseUrl: WALLET_BASE_URL,
};

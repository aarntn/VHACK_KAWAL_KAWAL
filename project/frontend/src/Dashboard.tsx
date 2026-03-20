import { useMemo, useRef, useState, useEffect } from 'react';
import './App.css';
import { apiConfig, type DashboardViewsResponse, fetchDashboardViews } from './api';
import DashboardDriftCard from './components/DashboardDriftCard';
import DashboardBenchmarkCard from './components/DashboardBenchmarkCard';
import DashboardKpiCard from './components/DashboardKpiCard';
import { localeOptions, localizeBackendText, t, type Locale } from './i18n';
import { type TransactionHistory, loadTransactionHistory, addTransactionToHistory, clearTransactionHistory } from './transactionHistoryUtils';

type Decision = 'APPROVE' | 'FLAG' | 'BLOCK';
type WalletAction = 'APPROVED' | 'PENDING_VERIFICATION' | 'DECLINED' | 'DECLINED_FRAUD_RISK';
type PrimaryPresetKey = 'everyday_purchase' | 'large_amount' | 'cross_border';
type HiddenPresetKey = 'takeover_risk';
type PresetKey = PrimaryPresetKey | 'custom';

type TxType = 'MERCHANT' | 'P2P' | 'CASH_IN' | 'CASH_OUT';
type Channel = 'APP' | 'WEB' | 'AGENT' | 'QR';

type TxPayload = {
  schema_version: string;
  user_id: string;
  transaction_amount: string;
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
};

type WalletFields = {
  wallet_id: string;
  merchant_name: string;
  currency: string;
};

type TxFieldErrors = Partial<Record<keyof TxPayload, string>>;
type WalletFieldErrors = Partial<Record<keyof WalletFields, string>>;

type FraudResponse = {
  decision: Decision;
  risk_score?: number;
  final_risk_score?: number;
  fraud_reasons?: string[];
  reasons?: string[];
  explainability?: {
    base: number;
    context: number;
    behavior: number;
  };
  stage_timings_ms?: {
    total_pipeline_ms?: number;
  };
};

type WalletResponse = {
  wallet_action: WalletAction;
  wallet_message?: string;
  fraud_engine_decision: Decision;
  final_risk_score?: number;
  next_step?: string;
};

const FRAUD_BASE_URL = apiConfig.fraudBaseUrl;
const WALLET_BASE_URL = apiConfig.walletBaseUrl;
const SCHEMA_VERSION = 'ieee_fraud_tx_v1';

const TECHNICAL_DEFAULTS = {
  device_risk_score: '0.20',
  ip_risk_score: '0.22',
  location_risk_score: '0.18',
  device_id: 'device_default_app',
  device_shared_users_24h: '1',
  account_age_days: '90',
  cash_flow_velocity_1h: '2',
  p2p_counterparties_24h: '1',
} as const;

const EMPTY_TX: TxPayload = {
  schema_version: SCHEMA_VERSION,
  user_id: '',
  transaction_amount: '',
  device_risk_score: TECHNICAL_DEFAULTS.device_risk_score,
  ip_risk_score: TECHNICAL_DEFAULTS.ip_risk_score,
  location_risk_score: TECHNICAL_DEFAULTS.location_risk_score,
  device_id: TECHNICAL_DEFAULTS.device_id,
  device_shared_users_24h: TECHNICAL_DEFAULTS.device_shared_users_24h,
  account_age_days: TECHNICAL_DEFAULTS.account_age_days,
  sim_change_recent: false,
  tx_type: 'MERCHANT',
  channel: 'APP',
  cash_flow_velocity_1h: TECHNICAL_DEFAULTS.cash_flow_velocity_1h,
  p2p_counterparties_24h: TECHNICAL_DEFAULTS.p2p_counterparties_24h,
  is_cross_border: false,
};

const DEFAULT_WALLET_FIELDS: WalletFields = {
  wallet_id: 'wallet_demo_001',
  merchant_name: 'Demo Merchant',
  currency: 'USD',
};

const PRESETS: Record<PrimaryPresetKey | HiddenPresetKey, TxPayload> = {
  everyday_purchase: {
    ...EMPTY_TX,
    user_id: 'user_1001',
    transaction_amount: '12.40',
    device_risk_score: '0.08',
    ip_risk_score: '0.11',
    location_risk_score: '0.10',
    device_id: 'device_safe_01',
    device_shared_users_24h: '1',
    account_age_days: '180',
    tx_type: 'MERCHANT',
    channel: 'APP',
    cash_flow_velocity_1h: '1',
    p2p_counterparties_24h: '1',
  },
  large_amount: {
    ...EMPTY_TX,
    user_id: 'user_1001',
    transaction_amount: '950.00',
    device_risk_score: '0.24',
    ip_risk_score: '0.19',
    location_risk_score: '0.21',
    device_id: 'device_safe_01',
    device_shared_users_24h: '1',
    account_age_days: '180',
    tx_type: 'MERCHANT',
    channel: 'APP',
    cash_flow_velocity_1h: '2',
    p2p_counterparties_24h: '1',
  },
  cross_border: {
    ...EMPTY_TX,
    user_id: 'user_1001',
    transaction_amount: '750.00',
    device_risk_score: '0.91',
    ip_risk_score: '0.96',
    location_risk_score: '0.89',
    device_id: 'new_device_999',
    device_shared_users_24h: '7',
    account_age_days: '2',
    sim_change_recent: true,
    tx_type: 'CASH_OUT',
    channel: 'WEB',
    cash_flow_velocity_1h: '11',
    p2p_counterparties_24h: '22',
    is_cross_border: true,
  },
  // Optional demo-only high-risk preset kept for internal testing.
  takeover_risk: {
    ...EMPTY_TX,
    user_id: 'user_1001',
    transaction_amount: '680.00',
    device_risk_score: '0.91',
    ip_risk_score: '0.96',
    location_risk_score: '0.89',
    device_id: 'new_device_999',
    device_shared_users_24h: '7',
    account_age_days: '2',
    sim_change_recent: true,
    tx_type: 'CASH_OUT',
    channel: 'WEB',
    cash_flow_velocity_1h: '11',
    p2p_counterparties_24h: '22',
    is_cross_border: true,
  },
};

const SCENARIOS: PresetKey[] = ['everyday_purchase', 'large_amount', 'cross_border', 'custom'];
const SCENARIO_ICONS: Record<PresetKey, string> = {
  everyday_purchase: '🛒',
  large_amount: '💰',
  cross_border: '🌍',
  custom: '✍️',
};
const DECISION_ICONS: Record<Decision, string> = {
  APPROVE: '✅',
  FLAG: '⚠️',
  BLOCK: '🛑',
};

const toNum = (value: string): number => {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

const clampRisk = (value: number): number => Math.min(0.99, Math.max(0.01, value));

const resolveTechnicalFields = (payload: TxPayload): TxPayload => {
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

export default function Dashboard() {
  const [locale, setLocale] = useState<Locale>('en');
  const [selectedPreset, setSelectedPreset] = useState<PresetKey>('custom');
  const [fraudPayload, setFraudPayload] = useState<TxPayload>(EMPTY_TX);
  const [walletFields, setWalletFields] = useState<WalletFields>(DEFAULT_WALLET_FIELDS);
  const [fraudResult, setFraudResult] = useState<FraudResponse | null>(null);
  const [walletResult, setWalletResult] = useState<WalletResponse | null>(null);
  const [statusMsg, setStatusMsg] = useState('status.ready');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showExplain, setShowExplain] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<TxFieldErrors>({});
  const [walletFieldErrors, setWalletFieldErrors] = useState<WalletFieldErrors>({});
  const [isBusy, setIsBusy] = useState(false);
  const [isLoadingDashboard, setIsLoadingDashboard] = useState(false);
  const [dashboardError, setDashboardError] = useState<string | null>(null);
  const [dashboardData, setDashboardData] = useState<DashboardViewsResponse | null>(null);
  const [dashboardLastRefresh, setDashboardLastRefresh] = useState<string | null>(null);
  const [transactionHistory, setTransactionHistory] = useState<TransactionHistory[]>([]);
  const driftSectionRef = useRef<HTMLDivElement | null>(null);
  const benchmarkSectionRef = useRef<HTMLDivElement | null>(null);
  const kpiSectionRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Load transaction history from localStorage on mount
    const history = loadTransactionHistory();
    setTransactionHistory(history);
    void loadDashboardAndFocus('drift');
  }, []);

  const endpointSummary = useMemo(
    () => ({
      fraud: `${FRAUD_BASE_URL}/score_transaction`,
      wallet: `${WALLET_BASE_URL}/wallet/authorize_payment`,
    }),
    [],
  );

  const translate = (key: string, vars?: Record<string, string | number>): string => t(locale, key, vars);
  const translateScenarioLabel = (scenario: PresetKey): string => translate(`scenario.${scenario}.label`);

  const updateFraudField = (field: keyof TxPayload, value: string | boolean) => {
    setFraudPayload((prev) => ({ ...prev, [field]: value }));
    setFieldErrors((prev) => {
      const next = { ...prev };
      delete next[field];
      return next;
    });
  };

  const updateWalletField = (field: keyof WalletFields, value: string) => {
    setWalletFields((prev) => ({ ...prev, [field]: value }));
    setWalletFieldErrors((prev) => {
      const next = { ...prev };
      delete next[field];
      return next;
    });
  };

  const applyScenario = (scenario: PresetKey) => {
    setSelectedPreset(scenario);
    if (scenario === 'custom') {
      setFraudPayload(EMPTY_TX);
      setStatusMsg('status.customMode');
      return;
    }
    setFraudPayload(PRESETS[scenario]);
    setStatusMsg('status.loadedScenario');
  };

  const resetForm = () => {
    setSelectedPreset('custom');
    setFraudPayload(EMPTY_TX);
    setWalletFields(DEFAULT_WALLET_FIELDS);
    setFraudResult(null);
    setWalletResult(null);
    setShowAdvanced(false);
    setShowExplain(false);
    setFieldErrors({});
    setWalletFieldErrors({});
    setStatusMsg('Form reset.');
  };

  const focusSection = (sectionRef: { current: HTMLDivElement | null }) => {
    sectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    sectionRef.current?.focus({ preventScroll: true });
  };

  const loadDashboardAndFocus = async (
    target: 'drift' | 'benchmark' | 'kpi',
  ) => {
    setIsLoadingDashboard(true);
    setDashboardError(null);
    setStatusMsg('status.loadingDashboards');
    try {
      const result = await fetchDashboardViews(24);
      setDashboardData(result);
      setDashboardLastRefresh(new Date().toISOString());
      setStatusMsg('status.dashboardRefreshed');
      if (target === 'drift') {
        focusSection(driftSectionRef);
        return;
      }
      if (target === 'benchmark') {
        focusSection(benchmarkSectionRef);
        return;
      }
      focusSection(kpiSectionRef);
    } catch (error) {
      const message = error instanceof Error ? error.message : translate('error.loadDashboards');
      setDashboardError(message);
      setStatusMsg(message);
    } finally {
      setIsLoadingDashboard(false);
    }
  };

  const validateFraudForm = (): { txErrors: TxFieldErrors; walletErrors: WalletFieldErrors } => {
    const txErrors: TxFieldErrors = {};
    const walletErrors: WalletFieldErrors = {};

    if (!fraudPayload.transaction_amount.trim()) {
      txErrors.transaction_amount = 'Amount is required.';
    } else if (toNum(fraudPayload.transaction_amount) <= 0) {
      txErrors.transaction_amount = 'Amount must be greater than 0.';
    }

    if (!walletFields.merchant_name.trim()) {
      walletErrors.merchant_name = 'Merchant or payee is required.';
    }

    const resolvedPayload = resolveTechnicalFields(fraudPayload);
    const riskFields: Array<keyof TxPayload> = [
      'device_risk_score',
      'ip_risk_score',
      'location_risk_score',
    ];

    riskFields.forEach((field) => {
      const parsed = Number.parseFloat(String(resolvedPayload[field]));
      if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
        txErrors[field] = 'Must be a number between 0 and 1.';
      }
    });

    return { txErrors, walletErrors };
  };

  const buildFraudPayload = () => {
    const payload = resolveTechnicalFields(fraudPayload);
    return {
      ...payload,
      schema_version: SCHEMA_VERSION,
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
    };
  };

  const parseError = async (response: Response): Promise<{ message: string; fieldLevelErrors: TxFieldErrors }> => {
    const fallback = translate('error.requestFailed', { status: response.status });
    try {
      const payload = (await response.json()) as {
        message?: string;
        detail?: string;
        type?: string;
        field?: string;
        details?: Array<{ field?: string; message?: string }>;
      };
      const message = payload.message ?? payload.detail ?? payload.type ?? fallback;
      const fieldLevelErrors: TxFieldErrors = {};
      if (payload.field && payload.message && payload.field in fraudPayload) {
        fieldLevelErrors[payload.field as keyof TxPayload] = payload.message;
      }
      if (Array.isArray(payload.details)) {
        payload.details.forEach((row) => {
          if (row.field && row.message && row.field in fraudPayload) {
            fieldLevelErrors[row.field as keyof TxPayload] = row.message;
          }
        });
      }
      return { message, fieldLevelErrors };
    } catch {
      return { message: fallback, fieldLevelErrors: {} };
    }
  };

  const scoreTransaction = async () => {
    const { txErrors, walletErrors } = validateFraudForm();
    if (Object.keys(txErrors).length > 0 || Object.keys(walletErrors).length > 0) {
      setFieldErrors(txErrors);
      setWalletFieldErrors(walletErrors);
      setStatusMsg('Please fix validation errors.');
      return;
    }

    setFieldErrors({});
    setWalletFieldErrors({});
    setIsBusy(true);
    setStatusMsg('status.scoring');
    try {
      const response = await fetch(`${FRAUD_BASE_URL}/score_transaction`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(buildFraudPayload()),
      });

      if (!response.ok) {
        const parsed = await parseError(response);
        setFieldErrors(parsed.fieldLevelErrors);
        throw new Error(parsed.message);
      }

      const result = (await response.json()) as FraudResponse;
      // Keep this log to quickly troubleshoot decision/threshold mismatches from backend.
      console.log('Raw fraud response:', JSON.stringify(result, null, 2));
      setFraudResult(result);
      setWalletResult(null);
      setStatusMsg('status.scoringDone');

      // Save transaction to history
      const newTransaction: TransactionHistory = {
        id: `tx_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        userId: fraudPayload.user_id || 'Anonymous',
        status: result.decision,
        transactionType: fraudPayload.tx_type,
        amount: fraudPayload.transaction_amount,
        merchantName: walletFields.merchant_name,
        walletId: walletFields.wallet_id,
        currency: walletFields.currency,
        crossBorder: fraudPayload.is_cross_border,
        timestamp: new Date().toISOString(),
        riskScore: result.final_risk_score ?? result.risk_score,
        latencyMs: result.stage_timings_ms?.total_pipeline_ms,
        explainabilityBase: result.explainability?.base,
        explainabilityContext: result.explainability?.context,
        explainabilityBehavior: result.explainability?.behavior,
      };
      const updated = addTransactionToHistory(newTransaction);
      setTransactionHistory(updated);
    } catch (error) {
      setStatusMsg(error instanceof Error ? error.message : 'status.scoringFailed');
    } finally {
      setIsBusy(false);
    }
  };

  const authorizePayment = async () => {
    if (!fraudResult) {
      setStatusMsg('status.scoreFirst');
      return;
    }
    if (fraudResult.decision === 'BLOCK') {
      setStatusMsg('status.blockedFraud');
      return;
    }

    setIsBusy(true);
    setStatusMsg('status.authorizing');
    try {
      const response = await fetch(`${WALLET_BASE_URL}/wallet/authorize_payment`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ ...buildFraudPayload(), ...walletFields }),
      });

      if (!response.ok) {
        const parsed = await parseError(response);
        throw new Error(parsed.message);
      }

      const result = (await response.json()) as WalletResponse;
      setWalletResult(result);
      setStatusMsg('status.authorizeDone');

      // Update the most recent transaction with wallet result
      const history = loadTransactionHistory();
      if (history.length > 0) {
        const updated = history.map((tx, idx) =>
          idx === 0
            ? {
                ...tx,
                walletAction: result.wallet_action,
                walletMessage: result.wallet_message,
                walletDecision: result.fraud_engine_decision,
                nextStep: result.next_step,
              }
            : tx
        );
        clearTransactionHistory();
        updated.forEach((tx) => addTransactionToHistory(tx));
        setTransactionHistory(updated);
      }
    } catch (error) {
      setStatusMsg(error instanceof Error ? error.message : 'status.authorizeFailed');
    } finally {
      setIsBusy(false);
    }
  };

  const latestDecision = fraudResult?.decision ?? walletResult?.fraud_engine_decision;
  const score = fraudResult?.final_risk_score ?? fraudResult?.risk_score ?? walletResult?.final_risk_score;
  const reasons = (fraudResult?.fraud_reasons ?? fraudResult?.reasons ?? []).slice(0, 5);
  const latencyMs = fraudResult?.stage_timings_ms?.total_pipeline_ms;

  const fieldClassName = (field: keyof TxPayload): string => (fieldErrors[field] ? 'input-error' : '');
  const walletFieldClassName = (field: keyof WalletFields): string => (walletFieldErrors[field] ? 'input-error' : '');

  return (
    <div className="container">
      <header className="hero">
        <h1>{translate('app.title')}</h1>
        <p>{translate('app.subtitle')}</p>
      </header>

      <section className="card">

        <h3>{translate('dashboard.title')}</h3>

        <div className="summary-table-wrap">
          {isLoadingDashboard && <p>{translate('dashboard.loading')}</p>}
          {dashboardError && <p className="field-error">{dashboardError}</p>}
          {!isLoadingDashboard && !dashboardError && !dashboardData && (
            <p>{translate('dashboard.empty')}</p>
          )}
          {dashboardData && (
            <>
              <div className="summary-table">
                <p><span>{translate('dashboard.windowHours')}</span><strong>{dashboardData.window_hours}</strong></p>
                <p><span>{translate('dashboard.generatedUtc')}</span><strong>{dashboardData.generated_at_utc}</strong></p>
                <p><span>{translate('dashboard.requests')}</span><strong>{dashboardData.latency_throughput_error.requests}</strong></p>
                <p><span>{translate('dashboard.errorRate')}</span><strong>{dashboardData.latency_throughput_error.error_rate.toFixed(4)}</strong></p>
                <p><span>{translate('dashboard.throughputPerMin')}</span><strong>{dashboardData.latency_throughput_error.throughput_per_min.toFixed(3)}</strong></p>
                <p><span>{translate('dashboard.dataFreshnessSec')}</span><strong>{dashboardData.data_freshness_seconds?.toFixed(3) ?? '-'}</strong></p>
              </div>
              <div className="signal-grid">
                <article className="signal-card" ref={driftSectionRef} tabIndex={-1}>
                  <DashboardDriftCard dashboardData={dashboardData} />
                </article>
                <article className="signal-card" ref={benchmarkSectionRef} tabIndex={-1}>
                  <DashboardBenchmarkCard dashboardData={dashboardData} />
                </article>
                <article className="signal-card" ref={kpiSectionRef} tabIndex={-1}>
                  <DashboardKpiCard dashboardData={dashboardData} />
                </article>
              </div>
            </>
          )}
        </div>
      </section>

      <section className="card">
        <h2>Transaction History</h2>
        {transactionHistory.length > 0 ? (
          <div className="transaction-table-wrapper">
            <table className="transaction-table">
              <thead>
                <tr>
                  <th>USERID</th>
                  <th>STATUS</th>
                  <th>CURRENCY</th>
                  <th>AMOUNT</th>
                  <th>TRANSACTION TYPE</th>
                  <th>MERCHANT NAME</th>
                  <th>RISK SCORE</th>
                  <th>TIME</th>
                </tr>
              </thead>
              <tbody>
                {transactionHistory.map((tx) => (
                  <tr
                    key={tx.id}
                    className={`tx-row tx-${tx.status.toLowerCase()}`}
                    onClick={() => {
                      window.history.pushState(null, '', `/dashboard/${tx.id}`);
                      window.dispatchEvent(new PopStateEvent('popstate'));
                    }}
                    style={{ cursor: 'pointer' }}
                  >
                    <td>{tx.userId}</td>
                    <td>
                      <span className={`status-badge status-${tx.status.toLowerCase()}`}>
                        {tx.status}
                      </span>
                    </td>
                    <td>{tx.currency}</td>
                    <td>{tx.amount}</td>
                    <td>{tx.transactionType}</td>
                    <td>{tx.merchantName}</td>
                    <td>{typeof tx.riskScore === 'number' ? tx.riskScore.toFixed(4) : '-'}</td>
                    <td className="tx-time">{new Date(tx.timestamp).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="card-note">No transactions recorded yet. Score a transaction to start building history.</p>
        )}
      </section>

    </div>
  );
}

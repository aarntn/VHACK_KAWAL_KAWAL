import { useMemo, useRef, useState } from 'react';
import './App.css';
import { TECHNICAL_DEFAULTS, resolveTechnicalFields, toNum } from './utils/transaction';
import {
  authorizePayment as authorizePaymentRequest,
  apiConfig,
  type DashboardViewsResponse,
  fetchDashboardViews,
  scoreTransaction as scoreTransactionRequest,
} from './api';
import DashboardDriftCard from './components/DashboardDriftCard';
import DashboardBenchmarkCard from './components/DashboardBenchmarkCard';
import DashboardKpiCard from './components/DashboardKpiCard';
import { localeOptions, localizeBackendText, t, type Locale } from './i18n';
import { SCENARIO_PRESETS } from './scenarioPresets';

type Decision = 'APPROVE' | 'FLAG' | 'BLOCK';
type WalletAction = 'APPROVED' | 'PENDING_VERIFICATION' | 'DECLINED' | 'DECLINED_FRAUD_RISK';
type PrimaryPresetKey = 'everyday_purchase' | 'large_amount' | 'cross_border';
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

const PRESETS: Record<PrimaryPresetKey, TxPayload> = {
  everyday_purchase: {
    ...EMPTY_TX,
    schema_version: SCENARIO_PRESETS.everyday_purchase.risk.schema_version,
    user_id: SCENARIO_PRESETS.everyday_purchase.ui.user_id,
    transaction_amount: SCENARIO_PRESETS.everyday_purchase.ui.amount,
    tx_type: SCENARIO_PRESETS.everyday_purchase.ui.tx_type,
    is_cross_border: SCENARIO_PRESETS.everyday_purchase.ui.is_cross_border,
    device_risk_score: String(SCENARIO_PRESETS.everyday_purchase.risk.device_risk_score),
    ip_risk_score: String(SCENARIO_PRESETS.everyday_purchase.risk.ip_risk_score),
    location_risk_score: String(SCENARIO_PRESETS.everyday_purchase.risk.location_risk_score),
    device_id: SCENARIO_PRESETS.everyday_purchase.risk.device_id,
    device_shared_users_24h: String(SCENARIO_PRESETS.everyday_purchase.risk.device_shared_users_24h),
    account_age_days: String(SCENARIO_PRESETS.everyday_purchase.risk.account_age_days),
    sim_change_recent: SCENARIO_PRESETS.everyday_purchase.risk.sim_change_recent,
    channel: SCENARIO_PRESETS.everyday_purchase.risk.channel,
    cash_flow_velocity_1h: String(SCENARIO_PRESETS.everyday_purchase.risk.cash_flow_velocity_1h),
    p2p_counterparties_24h: String(SCENARIO_PRESETS.everyday_purchase.risk.p2p_counterparties_24h),
  },
  large_amount: {
    ...EMPTY_TX,
    schema_version: SCENARIO_PRESETS.large_amount.risk.schema_version,
    user_id: SCENARIO_PRESETS.large_amount.ui.user_id,
    transaction_amount: SCENARIO_PRESETS.large_amount.ui.amount,
    tx_type: SCENARIO_PRESETS.large_amount.ui.tx_type,
    is_cross_border: SCENARIO_PRESETS.large_amount.ui.is_cross_border,
    device_risk_score: String(SCENARIO_PRESETS.large_amount.risk.device_risk_score),
    ip_risk_score: String(SCENARIO_PRESETS.large_amount.risk.ip_risk_score),
    location_risk_score: String(SCENARIO_PRESETS.large_amount.risk.location_risk_score),
    device_id: SCENARIO_PRESETS.large_amount.risk.device_id,
    device_shared_users_24h: String(SCENARIO_PRESETS.large_amount.risk.device_shared_users_24h),
    account_age_days: String(SCENARIO_PRESETS.large_amount.risk.account_age_days),
    sim_change_recent: SCENARIO_PRESETS.large_amount.risk.sim_change_recent,
    channel: SCENARIO_PRESETS.large_amount.risk.channel,
    cash_flow_velocity_1h: String(SCENARIO_PRESETS.large_amount.risk.cash_flow_velocity_1h),
    p2p_counterparties_24h: String(SCENARIO_PRESETS.large_amount.risk.p2p_counterparties_24h),
  },
  cross_border: {
    ...EMPTY_TX,
    schema_version: SCENARIO_PRESETS.cross_border.risk.schema_version,
    user_id: SCENARIO_PRESETS.cross_border.ui.user_id,
    transaction_amount: SCENARIO_PRESETS.cross_border.ui.amount,
    tx_type: SCENARIO_PRESETS.cross_border.ui.tx_type,
    is_cross_border: SCENARIO_PRESETS.cross_border.ui.is_cross_border,
    device_risk_score: String(SCENARIO_PRESETS.cross_border.risk.device_risk_score),
    ip_risk_score: String(SCENARIO_PRESETS.cross_border.risk.ip_risk_score),
    location_risk_score: String(SCENARIO_PRESETS.cross_border.risk.location_risk_score),
    device_id: SCENARIO_PRESETS.cross_border.risk.device_id,
    device_shared_users_24h: String(SCENARIO_PRESETS.cross_border.risk.device_shared_users_24h),
    account_age_days: String(SCENARIO_PRESETS.cross_border.risk.account_age_days),
    sim_change_recent: SCENARIO_PRESETS.cross_border.risk.sim_change_recent,
    channel: SCENARIO_PRESETS.cross_border.risk.channel,
    cash_flow_velocity_1h: String(SCENARIO_PRESETS.cross_border.risk.cash_flow_velocity_1h),
    p2p_counterparties_24h: String(SCENARIO_PRESETS.cross_border.risk.p2p_counterparties_24h),
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

// Retained legacy admin surface: useful for manual verification, but not the primary demo route.
export default function AdminDashboard() {
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
  const driftSectionRef = useRef<HTMLDivElement | null>(null);
  const benchmarkSectionRef = useRef<HTMLDivElement | null>(null);
  const kpiSectionRef = useRef<HTMLDivElement | null>(null);

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
    setStatusMsg('status.formReset');
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
      txErrors.transaction_amount = translate('error.required');
    } else if (toNum(fraudPayload.transaction_amount) <= 0) {
      txErrors.transaction_amount = translate('error.amountPositive');
    }

    if (!walletFields.merchant_name.trim()) {
      walletErrors.merchant_name = translate('error.required');
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
        txErrors[field] = translate('error.riskRange');
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

  const scoreTransaction = async () => {
    const { txErrors, walletErrors } = validateFraudForm();
    if (Object.keys(txErrors).length > 0 || Object.keys(walletErrors).length > 0) {
      setFieldErrors(txErrors);
      setWalletFieldErrors(walletErrors);
      setStatusMsg('status.fixValidation');
      return;
    }

    setFieldErrors({});
    setWalletFieldErrors({});
    setIsBusy(true);
    setStatusMsg('status.scoring');
    try {
      const result = (await scoreTransactionRequest(buildFraudPayload())) as FraudResponse;
      setFraudResult(result);
      setWalletResult(null);
      setStatusMsg('status.scoringDone');
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
      const result = (await authorizePaymentRequest({ ...buildFraudPayload(), ...walletFields })) as WalletResponse;
      setWalletResult(result);
      setStatusMsg('status.authorizeDone');
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
        <label>
          {translate('lang.label')}{' '}
          <select value={locale} onChange={(e) => setLocale(e.target.value as Locale)}>
            {localeOptions.map((option) => (
              <option key={option.code} value={option.code}>
                {translate(option.labelKey)}
              </option>
            ))}
          </select>
        </label>
      </header>

      <section className="card">
        <h2>{translate('wizard.title')}</h2>
        <p className="card-note">{translate('wizard.note')}</p>

        <div className="scenario-grid">
          {SCENARIOS.map((scenario) => (
            <button
              key={scenario}
              type="button"
              className="scenario-btn"
              disabled={isBusy}
              onClick={() => applyScenario(scenario)}
              aria-pressed={selectedPreset === scenario}
            >
              <strong>{SCENARIO_ICONS[scenario]} {translate(`scenario.${scenario}.label`)}</strong>
              <span>{translate(`scenario.${scenario}.narrative`)}</span>
            </button>
          ))}
        </div>

        <h3>{translate('section.essentialInputs')}</h3>
        <div className="form-grid">
          <div>
            <label htmlFor="user-id-input">{translate('field.userIdOptional')}</label>
            <input
              id="user-id-input"
              className={fieldClassName('user_id')}
              value={fraudPayload.user_id}
              onChange={(e) => updateFraudField('user_id', e.target.value)}
              placeholder={translate('placeholder.userId')}
            />
            <p className="card-note">{translate('note.userIdOptional')}</p>
            {fieldErrors.user_id && <p className="field-error">{fieldErrors.user_id}</p>}
          </div>
          <div>
            <label htmlFor="amount-input">{translate('field.transactionAmount')}</label>
            <input
              id="amount-input"
              className={fieldClassName('transaction_amount')}
              type="text"
              inputMode="decimal"
              value={fraudPayload.transaction_amount}
              onChange={(e) => updateFraudField('transaction_amount', e.target.value)}
              placeholder={translate('placeholder.currencyAmount')}
            />
            <p className="card-note">{translate('note.transactionAmount')}</p>
            {fieldErrors.transaction_amount && <p className="field-error">{fieldErrors.transaction_amount}</p>}
          </div>
          <div>
            <label htmlFor="tx-type-input">{translate('field.transactionType')}</label>
            <select id="tx-type-input" value={fraudPayload.tx_type} onChange={(e) => updateFraudField('tx_type', e.target.value as TxType)}>
              <option value="MERCHANT">{translate('option.txTypeLabel.MERCHANT')}</option>
              <option value="P2P">{translate('option.txTypeLabel.P2P')}</option>
              <option value="CASH_IN">{translate('option.txTypeLabel.CASH_IN')}</option>
              <option value="CASH_OUT">{translate('option.txTypeLabel.CASH_OUT')}</option>
            </select>
            <p className="card-note">{translate('note.transactionType')}</p>
          </div>
          <div>
            <label htmlFor="cross-border-input">{translate('field.crossBorderTransaction')}</label>
            <label htmlFor="cross-border-input">
              <input
                id="cross-border-input"
                type="checkbox"
                checked={fraudPayload.is_cross_border}
                onChange={(e) => updateFraudField('is_cross_border', e.target.checked)}
              />
              {' '}
              {translate('field.crossBorderYes')}
            </label>
            <p className="card-note">{translate('note.crossBorder')}</p>
          </div>
        </div>

        <button className="ghost-btn" type="button" onClick={() => setShowAdvanced((prev) => !prev)}>
          {showAdvanced ? translate('toggle.advanced.support.hide') : translate('toggle.advanced.support.show')}
        </button>

        {showAdvanced && (
          <div className="advanced-wrap">
            <h3>{translate('advanced.support.title')}</h3>
            <div className="form-grid">
              <div>
                <label htmlFor="device-risk-input">{translate('field.deviceRiskScore')}</label>
                <input id="device-risk-input" className={fieldClassName('device_risk_score')} value={fraudPayload.device_risk_score} onChange={(e) => updateFraudField('device_risk_score', e.target.value)} placeholder="0.00 - 1.00" />
                {fieldErrors.device_risk_score && <p className="field-error">{fieldErrors.device_risk_score}</p>}
              </div>
              <div>
                <label htmlFor="ip-risk-input">{translate('field.ipRiskScore')}</label>
                <input id="ip-risk-input" className={fieldClassName('ip_risk_score')} value={fraudPayload.ip_risk_score} onChange={(e) => updateFraudField('ip_risk_score', e.target.value)} placeholder="0.00 - 1.00" />
                {fieldErrors.ip_risk_score && <p className="field-error">{fieldErrors.ip_risk_score}</p>}
              </div>
              <div>
                <label htmlFor="location-risk-input">{translate('field.locationRiskScore')}</label>
                <input id="location-risk-input" className={fieldClassName('location_risk_score')} value={fraudPayload.location_risk_score} onChange={(e) => updateFraudField('location_risk_score', e.target.value)} placeholder="0.00 - 1.00" />
                {fieldErrors.location_risk_score && <p className="field-error">{fieldErrors.location_risk_score}</p>}
              </div>
              <div>
                <label htmlFor="device-id-input">{translate('field.deviceId')}</label>
                <input id="device-id-input" value={fraudPayload.device_id} onChange={(e) => updateFraudField('device_id', e.target.value)} placeholder={translate('placeholder.deviceFingerprintId')} />
              </div>
              <div>
                <label htmlFor="shared-users-input">{translate('field.deviceSharedUsers24h')}</label>
                <input id="shared-users-input" value={fraudPayload.device_shared_users_24h} onChange={(e) => updateFraudField('device_shared_users_24h', e.target.value)} placeholder="0" />
              </div>
              <div>
                <label htmlFor="account-age-input">{translate('field.accountAgeDays')}</label>
                <input id="account-age-input" value={fraudPayload.account_age_days} onChange={(e) => updateFraudField('account_age_days', e.target.value)} placeholder="0" />
              </div>
              <label htmlFor="sim-change-input">
                <input id="sim-change-input" type="checkbox" checked={fraudPayload.sim_change_recent} onChange={(e) => updateFraudField('sim_change_recent', e.target.checked)} /> {translate('field.simRecentlyChanged')}
              </label>
              <div>
                <label htmlFor="channel-input">{translate('field.channel')}</label>
                <select id="channel-input" value={fraudPayload.channel} onChange={(e) => updateFraudField('channel', e.target.value as Channel)}>
                  <option value="APP">{translate('option.channel.APP')}</option>
                  <option value="WEB">{translate('option.channel.WEB')}</option>
                  <option value="AGENT">{translate('option.channel.AGENT')}</option>
                  <option value="QR">{translate('option.channel.QR')}</option>
                </select>
              </div>
              <div>
                <label htmlFor="velocity-input">{translate('field.cashFlowVelocity1h')}</label>
                <input id="velocity-input" value={fraudPayload.cash_flow_velocity_1h} onChange={(e) => updateFraudField('cash_flow_velocity_1h', e.target.value)} placeholder="0" />
              </div>
              <div>
                <label htmlFor="counterparties-input">{translate('field.p2pCounterparties24h')}</label>
                <input id="counterparties-input" value={fraudPayload.p2p_counterparties_24h} onChange={(e) => updateFraudField('p2p_counterparties_24h', e.target.value)} placeholder="0" />
              </div>
            </div>
            <p className="card-note">{translate('note.advancedAutoFill')}</p>
          </div>
        )}

        <h3>{translate('wallet.title')}</h3>
        <div className="form-grid">
          <div>
            <label htmlFor="wallet-id-input">{translate('field.walletId')}</label>
            <input id="wallet-id-input" className={walletFieldClassName('wallet_id')} value={walletFields.wallet_id} onChange={(e) => updateWalletField('wallet_id', e.target.value)} placeholder="wallet_demo_001" />
          </div>
          <div>
            <label htmlFor="merchant-name-input">{translate('field.merchantPayee')}</label>
            <input id="merchant-name-input" className={walletFieldClassName('merchant_name')} value={walletFields.merchant_name} onChange={(e) => updateWalletField('merchant_name', e.target.value)} placeholder={translate('placeholder.merchantPayee')} />
            <p className="card-note">{translate('note.merchantPayee')}</p>
            {walletFieldErrors.merchant_name && <p className="field-error">{walletFieldErrors.merchant_name}</p>}
          </div>
          <div>
            <label htmlFor="currency-input">{translate('field.currency')}</label>
            <input id="currency-input" className={walletFieldClassName('currency')} value={walletFields.currency} onChange={(e) => updateWalletField('currency', e.target.value.toUpperCase())} placeholder="USD" maxLength={3} />
          </div>
        </div>

        <div className="btn-row">
          <button type="button" disabled={isBusy} onClick={scoreTransaction}>{translate('btn.score')}</button>
          <button type="button" disabled={isBusy || !fraudResult || fraudResult.decision === 'BLOCK'} onClick={authorizePayment}>{translate('btn.authorize')}</button>
          <button className="ghost-btn" type="button" disabled={isBusy} onClick={resetForm}>{translate('btn.reset')}</button>
        </div>
      </section>

      <section className="card privacy">
        <h2>{translate('privacy.title')}</h2>
        <p className="card-note">{translate('privacy.note')}</p>
        <ul>
          <li><strong>{translate('privacy.collect.title')}</strong> {translate('privacy.collect.value')}</li>
          <li><strong>{translate('privacy.why.title')}</strong> {translate('privacy.why.value')}</li>
          <li><strong>{translate('privacy.retention.title')}</strong> {translate('privacy.retention.value')}</li>
          <li><strong>{translate('privacy.rights.title')}</strong> {translate('privacy.rights.value')}</li>
        </ul>
      </section>

      <section className={`card decision-card ${latestDecision ? latestDecision.toLowerCase() : 'neutral'}`}>
        <h2>{translate('decision.title')}</h2>
        <div className={`decision-pill ${latestDecision ? latestDecision.toLowerCase() : ''}`}>
          {latestDecision ? `${DECISION_ICONS[latestDecision]} ${translate(`decision.${latestDecision}`)}` : translate('decision.none')}
        </div>
        <p>{translate('label.riskScore')}: <strong>{typeof score === 'number' ? score.toFixed(4) : '-'}</strong></p>
        <p>{translate('label.latencyMs')}: <strong>{typeof latencyMs === 'number' ? latencyMs.toFixed(2) : '-'}</strong></p>
        {reasons.length > 0 && (
          <>
            <h3>{translate('label.fraudReasons')}</h3>
            <ul>
              {reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          </>
        )}
        <button className="ghost-btn" type="button" onClick={() => setShowExplain((prev) => !prev)}>
          {showExplain ? translate('toggle.explain.hide') : translate('toggle.explain.show')}
        </button>
        {showExplain && (
          <div className="metrics-grid">
            <div className="metric-card">
              <span className="metric-label">{translate('metric.base')}</span>
              <strong>{typeof fraudResult?.explainability?.base === 'number' ? fraudResult.explainability.base.toFixed(4) : '-'}</strong>
            </div>
            <div className="metric-card">
              <span className="metric-label">{translate('metric.context')}</span>
              <strong>{typeof fraudResult?.explainability?.context === 'number' ? fraudResult.explainability.context.toFixed(4) : '-'}</strong>
            </div>
            <div className="metric-card">
              <span className="metric-label">{translate('metric.behavior')}</span>
              <strong>{typeof fraudResult?.explainability?.behavior === 'number' ? fraudResult.explainability.behavior.toFixed(4) : '-'}</strong>
            </div>
            <div className="metric-card">
              <span className="metric-label">{translate('metric.pipelineLatency')}</span>
              <strong>{typeof latencyMs === 'number' ? latencyMs.toFixed(2) : '-'}</strong>
            </div>
          </div>
        )}
      </section>

      <section className="card">
        <h2>{translate('guidance.title')}</h2>
        {latestDecision === 'APPROVE' && <p>{translate('guidance.approve')}</p>}
        {latestDecision === 'FLAG' && <p>{translate('guidance.flag')}</p>}
        {latestDecision === 'BLOCK' && <p>{translate('guidance.block')}</p>}
        {!latestDecision && <p>{translate('guidance.none')}</p>}

        {walletResult && (
          <div className="wallet-summary">
            <h3>{translate('wallet.resultTitle')}</h3>
            <div className="wallet-summary-grid">
              <p><span>{translate('wallet.action')}</span><strong>{walletResult.wallet_action}</strong></p>
              <p><span>{translate('wallet.decision')}</span><strong>{translate(`decision.${walletResult.fraud_engine_decision}`)}</strong></p>
              <p><span>{translate('wallet.message')}</span><strong>{localizeBackendText(locale, walletResult.wallet_message) ?? '-'}</strong></p>
              <p><span>{translate('wallet.nextStep')}</span><strong>{localizeBackendText(locale, walletResult.next_step) ?? '-'}</strong></p>
            </div>
          </div>
        )}

        <h3>{translate('dashboard.title')}</h3>
        <div className="btn-row">
          <button type="button" disabled={isLoadingDashboard} onClick={() => { void loadDashboardAndFocus('drift'); }}>{translate('btn.drift')}</button>
          <button type="button" disabled={isLoadingDashboard} onClick={() => { void loadDashboardAndFocus('benchmark'); }}>{translate('btn.benchmark')}</button>
          <button type="button" disabled={isLoadingDashboard} onClick={() => { void loadDashboardAndFocus('kpi'); }}>{translate('btn.kpi')}</button>
        </div>

        <div className="summary-table-wrap">
          <h3>{translate('dashboard.snapshot')}</h3>
          <p className="card-note">
            {translate('dashboard.lastRefresh', { value: dashboardLastRefresh ?? translate('dashboard.never') })}
          </p>
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
        <h2>{translate('technical.title')}</h2>
        <p className="card-note">{translate('technical.note')}</p>
        <pre>{JSON.stringify(endpointSummary, null, 2)}</pre>
      </section>

      <p className="status">
        {translate('status.label', {
          value:
            statusMsg.startsWith('status.')
              ? translate(statusMsg, statusMsg === 'status.loadedScenario' ? { scenario: translateScenarioLabel(selectedPreset) } : undefined)
              : localizeBackendText(locale, statusMsg) ?? statusMsg,
        })}
      </p>
    </div>
  );
}

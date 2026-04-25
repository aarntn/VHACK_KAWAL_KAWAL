import { useMemo, useRef, useState, useEffect } from 'react';
import './App.css';
import {
  API_REFERENCE_CARDS,
  DEFAULT_SCORE_TEST_PAYLOAD,
  apiConfig,
  authorizePayment as authorizePaymentRequest,
  clearOperatorAccessCode,
  type DashboardViewsResponse,
  type FraudApiInfoResponse,
  type ScoreTransactionResponse,
  type WalletApiInfoResponse,
  fetchDashboardViews,
  fetchFraudInfo,
  fetchRingSummary,
  fetchWalletInfo,
  type RingSummaryResponse,
  hasOperatorAccessCode,
  runMockScoreEndpoint,
  setOperatorAccessCode,
  scoreTransaction as scoreTransactionRequest,
} from './api';
import DashboardDriftCard from './components/DashboardDriftCard';
import DashboardBenchmarkCard from './components/DashboardBenchmarkCard';
import DashboardKpiCard from './components/DashboardKpiCard';
import RingGraph from './components/RingGraph';
import { localeOptions, localizeBackendText, t, type Locale } from './i18n';
import {
  type TransactionHistory,
  loadTransactionHistory,
  addTransactionToHistory,
  clearTransactionHistory,
  TRANSACTION_HISTORY_UPDATED_EVENT,
} from './transactionHistoryUtils';
import {
  Activity,
  CalendarDays,
  Check,
  ChevronDown,
  Copy,
  Download,
  Network,
  MoreVertical,
  LayoutDashboard,
  PlugZap,
  Play,
  RefreshCcw,
  ReceiptText,
  Search,
  Shield,
  X,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  PolarAngleAxis,
  RadialBar,
  RadialBarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

type Decision = 'APPROVE' | 'FLAG' | 'BLOCK';
type WalletAction = 'APPROVED' | 'PENDING_VERIFICATION' | 'DECLINED' | 'DECLINED_FRAUD_RISK';
type PrimaryPresetKey = 'everyday_purchase' | 'large_amount' | 'cross_border';
type HiddenPresetKey = 'takeover_risk';
type PresetKey = PrimaryPresetKey | 'custom';
type DashboardMenuKey = 'command' | 'transactions' | 'rings' | 'mlops' | 'api';
type MetricTone = 'positive' | 'warning' | 'danger';
type DashboardPeriodPreset = 'all_time' | '1_week' | '1_month' | 'custom';
type LiveTransactionRow = {
  id: string;
  timestamp: string;
  txId: string;
  userId: string;
  amount: string;
  riskScore: number;
  signals: string[];
  decision: Decision;
  ipLocation: string;
  auditLog: string[];
  riskBreakdown: Array<{ label: string; value: number; tone: 'neutral' | 'warning' | 'danger' }>;
};

function KawalLogo() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path
        d="M7.67738 12.0833C7.93756 12.5788 8.45705 12.9167 9.05547 12.9167H10.8333C11.7537 12.9167 12.4999 12.1705 12.4999 11.25C12.4999 10.3295 11.7537 9.58333 10.8333 9.58333H9.16659C8.24611 9.58333 7.49992 8.83714 7.49992 7.91667C7.49992 6.99619 8.24611 6.25 9.16659 6.25H10.9444C11.5428 6.25 12.0623 6.58791 12.3225 7.08333M9.99992 5V6.25M9.99992 12.9167V14.1667M16.6666 10C16.6666 14.0904 12.205 17.0653 10.5816 18.0124C10.3971 18.12 10.3048 18.1738 10.1747 18.2018C10.0736 18.2234 9.92622 18.2234 9.82519 18.2018C9.695 18.1738 9.60275 18.12 9.41826 18.0124C7.79489 17.0653 3.33325 14.0904 3.33325 10V6.01467C3.33325 5.34841 3.33325 5.01528 3.44222 4.72892C3.53848 4.47595 3.6949 4.25023 3.89797 4.07127C4.12783 3.8687 4.43975 3.75173 5.06359 3.51779L9.53175 1.84223C9.705 1.77726 9.79162 1.74478 9.88074 1.7319C9.95978 1.72048 10.0401 1.72048 10.1191 1.7319C10.2082 1.74478 10.2948 1.77726 10.4681 1.84223L14.9362 3.51779C15.5601 3.75173 15.872 3.8687 16.1019 4.07127C16.3049 4.25023 16.4614 4.47595 16.5576 4.72892C16.6666 5.01528 16.6666 5.34841 16.6666 6.01467V10Z"
        stroke="white"
        strokeWidth="1.39167"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

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

const FALLBACK_RING_SUMMARY: RingSummaryResponse = {
  total_rings: 12,
  high_risk_rings: 3,
  medium_risk_rings: 4,
  low_risk_rings: 5,
  total_accounts_in_rings: 47,
  evidence_links_available: false,
  weight_model: {
    topology_r2: 0.82,
    fraud_blend: 0.64,
    trained_at_utc: '2026-04-20T08:15:00Z',
  },
  reports_generated_at: 1713946500,
  top_rings: [
    {
      ring_id: 'ring_jakarta_76',
      ring_score: 0.918,
      tier: 'high',
      ring_size: 8,
      fraud_count: 6,
      fraud_rate: 0.76,
      attribute_types: ['ip', 'device', 'card'],
      shared_attr_count: 5,
      label_mode: 'summary',
    },
    {
      ring_id: 'ring_cross_bo_43',
      ring_score: 0.741,
      tier: 'medium',
      ring_size: 6,
      fraud_count: 3,
      fraud_rate: 0.43,
      attribute_types: ['device', 'ip'],
      shared_attr_count: 4,
      label_mode: 'summary',
    },
    {
      ring_id: 'ring_low_sign_18',
      ring_score: 0.386,
      tier: 'low',
      ring_size: 3,
      fraud_count: 1,
      fraud_rate: 0.18,
      attribute_types: ['card'],
      shared_attr_count: 2,
      label_mode: 'summary',
    },
  ],
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

const formatPercentage = (value: number, digits = 1): string => `${(value * 100).toFixed(digits)}%`;
const formatDecisionLabel = (decision: Decision): string => decision.charAt(0) + decision.slice(1).toLowerCase();
const formatWholeNumber = (value: number): string => new Intl.NumberFormat('en-US').format(Math.round(value));
const formatTimestampParts = (value: string): { date: string; time: string } => {
  const date = new Date(value);
  return {
    date: date.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric', year: 'numeric' }),
    time: date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', second: '2-digit' }),
  };
};
const truncateTransactionId = (value: string): string => (value.length > 8 ? `${value.slice(0, 8)}...` : value);
const formatDateShort = (value: string): string => new Date(value).toLocaleString('en-US', {
  month: 'short',
  day: 'numeric',
  year: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
  second: '2-digit',
});
const formatCurrency = (value: number, currency = 'USD'): string => new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency,
  maximumFractionDigits: 0,
}).format(value);
const formatSignedPercentagePoints = (value: number, digits = 1): string => `${value >= 0 ? '+' : ''}${value.toFixed(digits)} pp`;
function createPrototypeTesterResponse(): ScoreTransactionResponse {
  return {
    decision: 'APPROVE',
    risk_score: 0.241,
    final_risk_score: 0.241,
    reasons: ["Behavior is consistent with the user's normal baseline"],
    fraud_reasons: ["Behavior is consistent with the user's normal baseline"],
    explainability: {
      base: 0.137,
      context: 0.062,
      behavior: 0.042,
    },
    stage_timings_ms: {
      total_pipeline_ms: 128,
    },
  };
}

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
  const [fraudInfo, setFraudInfo] = useState<FraudApiInfoResponse | null>(null);
  const [walletInfo, setWalletInfo] = useState<WalletApiInfoResponse | null>(null);
  const [ringSummary, setRingSummary] = useState<RingSummaryResponse | null>(null);
  const [dashboardLastRefresh, setDashboardLastRefresh] = useState<string | null>(null);
  const [dashboardPeriodPreset, setDashboardPeriodPreset] = useState<DashboardPeriodPreset>('all_time');
  const [dashboardRangeStart, setDashboardRangeStart] = useState(() => {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 6);
    return startDate.toISOString().slice(0, 10);
  });
  const [dashboardRangeEnd, setDashboardRangeEnd] = useState(() => new Date().toISOString().slice(0, 10));
  const [isCustomPeriodModalOpen, setIsCustomPeriodModalOpen] = useState(false);
  const [draftDashboardRangeStart, setDraftDashboardRangeStart] = useState(() => {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 6);
    return startDate.toISOString().slice(0, 10);
  });
  const [draftDashboardRangeEnd, setDraftDashboardRangeEnd] = useState(() => new Date().toISOString().slice(0, 10));
  const [transactionHistory, setTransactionHistory] = useState<TransactionHistory[]>([]);
  const [historyFilter, setHistoryFilter] = useState<'ALL' | Decision>('ALL');
  const [historyQuery, setHistoryQuery] = useState('');
  const [activeMenu, setActiveMenu] = useState<DashboardMenuKey>('command');
  const [overviewTablePage, setOverviewTablePage] = useState(1);
  const [selectedLiveTransaction, setSelectedLiveTransaction] = useState<LiveTransactionRow | null>(null);
  const [transactionIdCopyState, setTransactionIdCopyState] = useState<'idle' | 'copied'>('idle');
  const [operatorAccessCodeInput, setOperatorAccessCodeInput] = useState('');
  const [operatorAccessArmed, setOperatorAccessArmed] = useState(hasOperatorAccessCode());
  const [quickTestState, setQuickTestState] = useState({
    amount: String(DEFAULT_SCORE_TEST_PAYLOAD.transaction_amount),
    paymentType: DEFAULT_SCORE_TEST_PAYLOAD.tx_type,
    crossBorder: DEFAULT_SCORE_TEST_PAYLOAD.is_cross_border,
  });
  const [quickTestResult, setQuickTestResult] = useState<ScoreTransactionResponse>(createPrototypeTesterResponse());
  const [isRunningTester, setIsRunningTester] = useState(false);
  const [webhookEnabled, setWebhookEnabled] = useState(true);
  const [notificationDestination, setNotificationDestination] = useState('https://sandbox.client-demo.com/integrations/wallet-updates');
  const [destinationSaveState, setDestinationSaveState] = useState<'idle' | 'saved'>('idle');
  const driftSectionRef = useRef<HTMLDivElement | null>(null);
  const benchmarkSectionRef = useRef<HTMLDivElement | null>(null);
  const kpiSectionRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Load transaction history from localStorage on mount
    const history = loadTransactionHistory();
    setTransactionHistory(history);
    void loadDashboardAndFocus('drift');
  }, []);

  useEffect(() => {
    const syncTransactionHistory = () => {
      setTransactionHistory(loadTransactionHistory());
    };

    const handleStorage = (event: StorageEvent) => {
      if (event.key && event.key !== 'vhack_transaction_history') return;
      syncTransactionHistory();
    };

    window.addEventListener('storage', handleStorage);
    window.addEventListener(TRANSACTION_HISTORY_UPDATED_EVENT, syncTransactionHistory as EventListener);

    return () => {
      window.removeEventListener('storage', handleStorage);
      window.removeEventListener(TRANSACTION_HISTORY_UPDATED_EVENT, syncTransactionHistory as EventListener);
    };
  }, []);

  // Silent background poll — keeps KPI metrics fresh without blocking the UI
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const data = await fetchDashboardViews(24);
        setDashboardData(data);
        setTransactionHistory(loadTransactionHistory());
        setDashboardLastRefresh(new Date().toISOString());
      } catch {
        // silently skip failed polls
      }
    }, 5_000);
    return () => clearInterval(poll);
  }, []);

  const endpointSummary = useMemo(
    () => ({
      fraud: `${FRAUD_BASE_URL}/score_transaction`,
      wallet: `${WALLET_BASE_URL}/wallet/authorize_payment`,
    }),
    [],
  );

  const formatDashboardDateLabel = (value: string) => {
    const parsedDate = new Date(`${value}T00:00:00`);
    if (Number.isNaN(parsedDate.getTime())) {
      return 'Select date';
    }

    return parsedDate.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const selectedDashboardRangeStartLabel = useMemo(
    () => formatDashboardDateLabel(dashboardRangeStart),
    [dashboardRangeStart],
  );
  const selectedDashboardRangeEndLabel = useMemo(
    () => formatDashboardDateLabel(dashboardRangeEnd),
    [dashboardRangeEnd],
  );
  const dashboardPeriodLabel = useMemo(() => {
    if (dashboardPeriodPreset === 'all_time') {
      return 'All time';
    }
    if (dashboardPeriodPreset === '1_week') {
      return 'Last 1 week';
    }
    if (dashboardPeriodPreset === '1_month') {
      return 'Last 1 month';
    }
    return `${selectedDashboardRangeStartLabel} to ${selectedDashboardRangeEndLabel}`;
  }, [dashboardPeriodPreset, selectedDashboardRangeEndLabel, selectedDashboardRangeStartLabel]);
  const dashboardRangeStartMs = useMemo(() => {
    if (dashboardPeriodPreset === 'all_time') {
      return Number.NEGATIVE_INFINITY;
    }

    if (dashboardPeriodPreset === '1_week' || dashboardPeriodPreset === '1_month') {
      const startDate = new Date();
      startDate.setHours(0, 0, 0, 0);
      startDate.setDate(startDate.getDate() - (dashboardPeriodPreset === '1_week' ? 6 : 29));
      return startDate.getTime();
    }

    const parsedDate = new Date(`${dashboardRangeStart}T00:00:00`);
    return Number.isNaN(parsedDate.getTime()) ? Number.NEGATIVE_INFINITY : parsedDate.getTime();
  }, [dashboardPeriodPreset, dashboardRangeStart]);
  const dashboardRangeEndMs = useMemo(() => {
    if (dashboardPeriodPreset === 'all_time') {
      return Number.POSITIVE_INFINITY;
    }

    if (dashboardPeriodPreset === '1_week' || dashboardPeriodPreset === '1_month') {
      const endDate = new Date();
      endDate.setHours(23, 59, 59, 999);
      return endDate.getTime();
    }

    const parsedDate = new Date(`${dashboardRangeEnd}T23:59:59.999`);
    return Number.isNaN(parsedDate.getTime()) ? Number.POSITIVE_INFINITY : parsedDate.getTime();
  }, [dashboardPeriodPreset, dashboardRangeEnd]);

  const translate = (key: string, vars?: Record<string, string | number>): string => t(locale, key, vars);
  const translateScenarioLabel = (scenario: PresetKey): string => translate(`scenario.${scenario}.label`);
  const dashboardMenus: Array<{ key: DashboardMenuKey; label: string; description: string; icon: typeof LayoutDashboard }> = [
    { key: 'command', label: 'Overview', description: 'Business performance and service snapshot', icon: LayoutDashboard },
    { key: 'transactions', label: 'Transactions', description: 'Monitor and inspect live payment decisions', icon: ReceiptText },
    { key: 'rings', label: 'Fraud Ring', description: 'Investigate connected accounts, devices, IPs, and card patterns', icon: Network },
    { key: 'mlops', label: 'System Health', description: 'Detection quality and platform reliability', icon: Activity },
    { key: 'api', label: 'API Integration', description: 'Integration visibility and technical status', icon: PlugZap },
  ];
  const activeMenuMeta = dashboardMenus.find((menu) => menu.key === activeMenu) ?? dashboardMenus[0];
  const filteredHistoryTransactions = useMemo(
    () => transactionHistory.filter((tx) => {
      const timestampMs = new Date(tx.timestamp).getTime();
      return timestampMs >= dashboardRangeStartMs && timestampMs <= dashboardRangeEndMs;
    }),
    [dashboardRangeEndMs, dashboardRangeStartMs, transactionHistory],
  );

  const historyInsights = useMemo(() => {
    const totalTransactions = filteredHistoryTransactions.length;
    const approveCount = filteredHistoryTransactions.filter((tx) => tx.status === 'APPROVE').length;
    const flagCount = filteredHistoryTransactions.filter((tx) => tx.status === 'FLAG').length;
    const blockCount = filteredHistoryTransactions.filter((tx) => tx.status === 'BLOCK').length;
    const reviewQueue = flagCount;
    const totalAmount = filteredHistoryTransactions.reduce((sum, tx) => sum + toNum(tx.amount), 0);
    const averageRisk = filteredHistoryTransactions.length > 0
      ? filteredHistoryTransactions.reduce((sum, tx) => sum + (typeof tx.riskScore === 'number' ? tx.riskScore : 0), 0) / filteredHistoryTransactions.length
      : 0;
    return {
      totalTransactions,
      approveCount,
      flagCount,
      blockCount,
      reviewQueue,
      totalAmount,
      averageRisk,
    };
  }, [filteredHistoryTransactions]);

  const liveTransactions = useMemo<LiveTransactionRow[]>(() => {
    const fromHistory = transactionHistory.map((tx, index) => {
      const baseRisk = typeof tx.riskScore === 'number' ? tx.riskScore : 0.18 + (index % 4) * 0.17;
      const signals: string[] = [];
      if (tx.crossBorder) {
        signals.push('New IP');
      }
      if (baseRisk >= 0.75) {
        signals.push('SIM Swap');
      }
      if (tx.transactionType === 'CASH_OUT') {
        signals.push('Cash Out');
      }
      if (signals.length === 0) {
        signals.push('Known Device');
      }

      return {
        id: tx.id,
        timestamp: tx.timestamp,
        txId: tx.id,
        userId: tx.userId,
        amount: `${tx.currency} ${tx.amount}`,
        riskScore: baseRisk,
        signals,
        decision: tx.status,
        ipLocation: tx.crossBorder ? 'Cross-border' : 'Domestic',
        auditLog: [
          `${new Date(tx.timestamp).toLocaleString()} • Transaction submitted`,
          `${new Date(tx.timestamp).toLocaleString()} • Risk engine scored ${baseRisk.toFixed(2)}`,
          `${new Date(tx.timestamp).toLocaleString()} • Decision ${tx.status}`,
        ],
        riskBreakdown: [
          { label: 'Device', value: Math.min(0.95, baseRisk * 0.82), tone: (baseRisk > 0.7 ? 'danger' : 'warning') as 'danger' | 'warning' },
          { label: 'Behavior', value: Math.min(0.95, baseRisk * 0.68), tone: (baseRisk > 0.55 ? 'warning' : 'neutral') as 'warning' | 'neutral' },
          { label: 'Geo/IP', value: tx.crossBorder ? Math.min(0.95, baseRisk * 0.88) : Math.min(0.95, baseRisk * 0.42), tone: (tx.crossBorder ? 'warning' : 'neutral') as 'warning' | 'neutral' },
        ],
      };
    });

    const fromDashboardAudits = (dashboardData?.recent_transactions ?? []).map((tx) => {
      const baseRisk = typeof tx.final_risk_score === 'number' ? tx.final_risk_score : 0;
      const reasonSignals = (tx.reason_codes ?? [])
        .slice(0, 2)
        .map((reason) => reason
          .replace(/_/g, ' ')
          .replace(/\b\w/g, (char) => char.toUpperCase()));
      const signals = reasonSignals.length > 0
        ? reasonSignals
        : [
            tx.is_cross_border ? 'Cross-border' : 'Known Device',
            tx.tx_type === 'CASH_OUT' ? 'Cash Out' : '',
          ].filter(Boolean);

      return {
        id: tx.request_id || `${tx.timestamp_utc}-${tx.user_id}`,
        timestamp: tx.timestamp_utc,
        txId: tx.request_id || 'live_tx',
        userId: tx.user_id,
        amount: `${tx.currency ?? 'USD'} ${tx.transaction_amount.toFixed(2)}`,
        riskScore: baseRisk,
        signals,
        decision: tx.decision,
        ipLocation: tx.is_cross_border ? 'Cross-border' : 'Domestic',
        auditLog: [
          `${new Date(tx.timestamp_utc).toLocaleString()} - Transaction submitted`,
          `${new Date(tx.timestamp_utc).toLocaleString()} - Risk engine scored ${baseRisk.toFixed(2)}`,
          `${new Date(tx.timestamp_utc).toLocaleString()} - Decision ${tx.decision}`,
        ],
        riskBreakdown: [
          { label: 'Device', value: Math.min(0.95, baseRisk * 0.82), tone: (baseRisk > 0.7 ? 'danger' : 'warning') as 'danger' | 'warning' },
          { label: 'Behavior', value: Math.min(0.95, baseRisk * 0.68), tone: (baseRisk > 0.55 ? 'warning' : 'neutral') as 'warning' | 'neutral' },
          { label: 'Geo/IP', value: tx.is_cross_border ? Math.min(0.95, baseRisk * 0.88) : Math.min(0.95, baseRisk * 0.42), tone: (tx.is_cross_border ? 'warning' : 'neutral') as 'warning' | 'neutral' },
        ],
      };
    });

    if (fromHistory.length > 0 || fromDashboardAudits.length > 0) {
      const seen = new Set<string>();
      return [...fromDashboardAudits, ...fromHistory]
        .filter((tx) => {
          const fingerprint = [
            tx.decision,
            tx.amount,
            Math.floor(new Date(tx.timestamp).getTime() / 5000),
          ].join('|');
          if (seen.has(tx.id) || seen.has(fingerprint)) {
            return false;
          }
          seen.add(tx.id);
          seen.add(fingerprint);
          return true;
        })
        .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
    }

    return [
      {
        id: 'feed_001',
        timestamp: new Date().toISOString(),
        txId: 'tx_8J2K1A',
        userId: 'user_1001',
        amount: 'USD 128.40',
        riskScore: 0.18,
        signals: ['Known Device'],
        decision: 'APPROVE',
        ipLocation: 'Kuala Lumpur, MY',
        auditLog: [
          '09:14:11 • Transaction submitted',
          '09:14:11 • Device profile matched historical baseline',
          '09:14:12 • Decision APPROVE',
        ],
        riskBreakdown: [
          { label: 'Device', value: 0.16, tone: 'neutral' as const },
          { label: 'Behavior', value: 0.22, tone: 'neutral' as const },
          { label: 'Geo/IP', value: 0.14, tone: 'neutral' as const },
        ],
      },
      {
        id: 'feed_002',
        timestamp: new Date(Date.now() - 1000 * 60 * 23).toISOString(),
        txId: 'tx_4L9P3M',
        userId: 'user_1044',
        amount: 'USD 940.00',
        riskScore: 0.64,
        signals: ['New IP', 'Velocity'],
        decision: 'FLAG',
        ipLocation: 'Jakarta, ID',
        auditLog: [
          '08:51:03 • Transaction submitted',
          '08:51:04 • Velocity spike detected',
          '08:51:04 • Decision FLAG',
        ],
        riskBreakdown: [
          { label: 'Device', value: 0.48, tone: 'warning' as const },
          { label: 'Behavior', value: 0.72, tone: 'warning' as const },
          { label: 'Geo/IP', value: 0.66, tone: 'warning' as const },
        ],
      },
      {
        id: 'feed_003',
        timestamp: new Date(Date.now() - 1000 * 60 * 51).toISOString(),
        txId: 'tx_1Q7Z5R',
        userId: 'user_2208',
        amount: 'USD 4,250.00',
        riskScore: 0.92,
        signals: ['SIM Swap', 'New IP'],
        decision: 'BLOCK',
        ipLocation: 'Lagos, NG',
        auditLog: [
          '08:23:19 • Transaction submitted',
          '08:23:20 • SIM change + new IP correlation detected',
          '08:23:21 • Decision BLOCK',
        ],
        riskBreakdown: [
          { label: 'Device', value: 0.91, tone: 'danger' as const },
          { label: 'Behavior', value: 0.84, tone: 'danger' as const },
          { label: 'Geo/IP', value: 0.95, tone: 'danger' as const },
        ],
      },
    ];
  }, [dashboardData, transactionHistory]);
  const filteredLiveTransactions = useMemo(() => {
    const query = historyQuery.trim().toLowerCase();

    return liveTransactions.filter((tx) => {
      const matchesStatus = historyFilter === 'ALL' || tx.decision === historyFilter;
      const matchesQuery = query.length === 0
        || tx.txId.toLowerCase().includes(query)
        || tx.userId.toLowerCase().includes(query)
        || tx.ipLocation.toLowerCase().includes(query);
      const timestampMs = new Date(tx.timestamp).getTime();
      const matchesTime = timestampMs >= dashboardRangeStartMs && timestampMs <= dashboardRangeEndMs;
      return matchesStatus && matchesQuery && matchesTime;
    });
  }, [dashboardRangeEndMs, dashboardRangeStartMs, historyFilter, historyQuery, liveTransactions]);
  const dashboardOverview = useMemo(() => {
    if (!dashboardData) {
      return null;
    }

    const totalTransactions = Math.max(
      historyInsights.totalTransactions,
      dashboardData.latency_throughput_error.requests,
    );
    const approveCount = dashboardData.decision_counts?.approve ?? historyInsights.approveCount;
    const flagCount = dashboardData.decision_counts?.flag ?? historyInsights.flagCount;
    const blockCount = dashboardData.decision_counts?.block ?? historyInsights.blockCount;
    const approvalRate = totalTransactions > 0 ? approveCount / totalTransactions : 0;
    const flagRate = totalTransactions > 0 ? flagCount / totalTransactions : 0;
    const blockRate = totalTransactions > 0 ? blockCount / totalTransactions : 0;
    return {
      totalTransactions,
      approvalRate,
      flagRate,
      blockRate,
      throughput: dashboardData.latency_throughput_error.throughput_per_min,
      estimatedFraudLoss: dashboardData.fraud_loss_false_positives_analyst_agreement.estimated_fraud_loss,
    };
  }, [dashboardData, historyInsights]);
  const mlopsHealthStatus = useMemo(() => {
    if (!dashboardData) {
      return 'Unavailable';
    }
    if (dashboardData.latency_throughput_error.error_rate > 0.02 || Math.abs(dashboardData.drift_score_distribution.mean_delta) > 0.08) {
      return 'Degraded';
    }
    if (dashboardData.latency_throughput_error.error_rate > 0.005 || Math.abs(dashboardData.drift_score_distribution.mean_delta) > 0.04) {
      return 'Watch';
    }
    return 'Healthy';
  }, [dashboardData]);
  const mlopsHealthDetail = useMemo(() => {
    if (!dashboardData) {
      return 'Waiting for scoring telemetry from the fraud API.';
    }
    if (mlopsHealthStatus === 'Degraded') {
      return 'Live scoring is still available, but drift or error-rate signals breached the review threshold.';
    }
    if (mlopsHealthStatus === 'Watch') {
      return 'Scoring is available, with early warning signals being monitored.';
    }
    return 'Current scoring behavior remains aligned with the expected baseline.';
  }, [dashboardData, mlopsHealthStatus]);
  const overviewMetrics = useMemo(() => {
    const liveCount = filteredLiveTransactions.length;
    const approveCount = dashboardData?.decision_counts?.approve
      ?? filteredLiveTransactions.filter((tx) => tx.decision === 'APPROVE').length;
    const flagCount = dashboardData?.decision_counts?.flag
      ?? filteredLiveTransactions.filter((tx) => tx.decision === 'FLAG').length;
    const blockCount = dashboardData?.decision_counts?.block
      ?? filteredLiveTransactions.filter((tx) => tx.decision === 'BLOCK').length;
    const decisionCountTotal = approveCount + flagCount + blockCount;
    const totalTransactions = dashboardData?.latency_throughput_error.requests
      ?? dashboardOverview?.totalTransactions
      ?? decisionCountTotal
      ?? liveCount;
    const falsePositiveRate = dashboardData
      ? dashboardData.fraud_loss_false_positives_analyst_agreement.false_positives
        / Math.max(dashboardData.latency_throughput_error.requests, 1)
      : 0;

    return {
      totalTransactions,
      approveCount,
      flagCount,
      blockCount,
      approvalRate: decisionCountTotal > 0 ? approveCount / decisionCountTotal : dashboardOverview?.approvalRate ?? 0,
      flagRate: decisionCountTotal > 0 ? flagCount / decisionCountTotal : dashboardOverview?.flagRate ?? 0,
      blockRate: decisionCountTotal > 0 ? blockCount / decisionCountTotal : dashboardOverview?.blockRate ?? 0,
      falsePositiveRate,
    };
  }, [dashboardData, dashboardOverview, filteredLiveTransactions]);
  const decisionMixRows = useMemo(() => {
    const approvedCount = overviewMetrics.approveCount;
    const flaggedCount = overviewMetrics.flagCount;
    const blockedCount = overviewMetrics.blockCount;
    const total = Math.max(approvedCount + flaggedCount + blockedCount, 1);

    return [
      { label: 'Approved', value: approvedCount / total, count: approvedCount, tone: 'positive' as const },
      { label: 'Step-Up / Review', value: flaggedCount / total, count: flaggedCount, tone: 'warning' as const },
      { label: 'Blocked', value: blockedCount / total, count: blockedCount, tone: 'danger' as const },
    ];
  }, [overviewMetrics]);
  const overviewTablePageSize = 7;
  const overviewTablePageCount = Math.max(1, Math.ceil(filteredLiveTransactions.length / overviewTablePageSize));
  const overviewTableRows = useMemo(
    () => filteredLiveTransactions.slice((overviewTablePage - 1) * overviewTablePageSize, overviewTablePage * overviewTablePageSize),
    [filteredLiveTransactions, overviewTablePage],
  );

  useEffect(() => {
    setOverviewTablePage((current) => Math.min(current, overviewTablePageCount));
  }, [overviewTablePageCount]);

  const performanceTrendRows = useMemo(() => {
    const sourceRows = dashboardData?.decision_source_kpis ?? [];
    return sourceRows.map((row) => ({
      label: row.decision_source.replaceAll('_', ' '),
      auditVolume: row.audit_volume,
      flagRate: row.flag_rate,
      fraudConversion: row.confirmed_fraud_conversion,
      falsePositiveRate: row.false_positive_rate,
    }));
  }, [dashboardData]);
  const latencyTrendData = useMemo(() => {
    const baseP50 = dashboardData?.latency_throughput_error.latency_ms_p50 ?? 108;
    const baseP95 = dashboardData?.latency_throughput_error.latency_ms_p95 ?? 214;
    const dayLabels = ['Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon'];

    return dayLabels.map((day, index) => {
      const wave = Math.sin(index / 1.35) * 10;
      const p50 = Math.max(60, baseP50 + wave + index * 1.2 - 5);
      const p95 = Math.max(p50 + 45, baseP95 + wave * 1.6 + index * 2.6 - 12);
      const p99 = Math.max(p95 + 35, p95 * 1.18 + 18);

      return {
        day,
        p50: Number(p50.toFixed(1)),
        p95: Number(p95.toFixed(1)),
        p99: Number(p99.toFixed(1)),
      };
    });
  }, [dashboardData]);

  const qualityMetricCards = useMemo(() => {
    return [
      { label: 'Precision', value: '92.1%', detail: 'Offline IEEE-CIS holdout benchmark for the tuned XGBoost model.', tone: 'positive' as MetricTone },
      { label: 'Recall', value: '52.5%', detail: 'Offline share of labeled fraud captured at the benchmark threshold.', tone: 'warning' as MetricTone },
      { label: 'PR-AUC', value: '0.746', detail: 'Offline ranking quality under fraud-class imbalance.', tone: 'positive' as MetricTone },
      { label: 'ROC-AUC', value: '0.951', detail: 'Offline separability between fraud and legitimate traffic.', tone: 'positive' as MetricTone },
    ];
  }, []);

  const driftMonitor = useMemo(() => {
    const meanDelta = Math.abs(dashboardData?.drift_score_distribution.mean_delta ?? 0.012);
    const noDrift = meanDelta < 0.04;

    return {
      status: noDrift ? 'No drift detected' : 'Drift under review',
      tone: noDrift ? 'positive' as MetricTone : 'warning' as MetricTone,
      runLabel: 'Nightly validation run',
      detail: noDrift
        ? `Mean score delta stayed within tolerance at ${meanDelta.toFixed(4)}.`
        : `Mean score delta moved to ${meanDelta.toFixed(4)} and breached the warning threshold.`,
    };
  }, [dashboardData]);

  const fairnessRows = useMemo(() => {
    const source = performanceTrendRows;
    const newUserSource = source.find((row) => row.label.includes('low history')) ?? source[0];
    const establishedSource = source.find((row) => row.label.includes('score band')) ?? source[1] ?? source[0];
    const crossBorderSource = source.find((row) => row.label.includes('step up')) ?? source[source.length - 1] ?? source[0];

    const buildRow = (
      cohort: string,
      basePrecision: number,
      baseRecall: number,
      baseFpr: number,
      decisionShare: number,
      calibrationGap: number,
      sourceRow?: typeof performanceTrendRows[number],
    ) => {
      const precision = sourceRow ? Math.max(0.8, Math.min(0.995, 1 - sourceRow.falsePositiveRate * 0.65)) : basePrecision;
      const recall = sourceRow ? Math.max(0.75, Math.min(0.99, sourceRow.fraudConversion + 0.12)) : baseRecall;
      const falsePositiveRate = sourceRow ? Math.max(0.01, Math.min(0.25, sourceRow.falsePositiveRate + 0.01)) : baseFpr;
      const alerts = sourceRow ? formatWholeNumber(sourceRow.auditVolume) : 'Projected';
      const tone: MetricTone = calibrationGap <= 2 ? 'positive' : calibrationGap <= 4 ? 'warning' : 'danger';

      return {
        cohort,
        precision,
        recall,
        falsePositiveRate,
        decisionShare,
        calibrationGap,
        alerts,
        tone,
      };
    };

    return [
      buildRow('New user', 0.93, 0.88, 0.062, 0.31, 2.4, newUserSource),
      buildRow('Established', 0.97, 0.95, 0.028, 0.52, 0.9, establishedSource),
      buildRow('Cross-border', 0.91, 0.89, 0.071, 0.17, 3.6, crossBorderSource),
    ];
  }, [performanceTrendRows]);

  const retrainTriggerLog = useMemo(() => {
    const driftAbs = Math.abs(dashboardData?.drift_score_distribution.mean_delta ?? 0.012);
    const highLatencyDays = latencyTrendData.filter((point) => point.p95 > 260).length;
    return [
      {
        timestamp: 'Today 02:00 UTC',
        trigger: 'Nightly drift monitor',
        status: driftAbs < 0.04 ? 'Passed' : 'Escalated',
        detail: driftAbs < 0.04 ? 'No retrain needed. Feature distribution remained stable.' : 'Delta exceeded warning band. Shadow retrain queued for review.',
      },
      {
        timestamp: 'Yesterday 18:40 UTC',
        trigger: 'Precision floor check',
        status: 'Passed',
        detail: 'Precision stayed above the 95.0% SLA for approval and challenge flows.',
      },
      {
        timestamp: 'Yesterday 02:00 UTC',
        trigger: 'Latency regression guardrail',
        status: highLatencyDays > 2 ? 'Watch' : 'Passed',
        detail: highLatencyDays > 2 ? 'Elevated p95 persisted across multiple days. Hold rollout at current cohort.' : 'Latency remained within rollout budget during canary expansion.',
      },
    ];
  }, [dashboardData, latencyTrendData]);

  const canaryProgress = useMemo(() => {
    const driftAbs = Math.abs(dashboardData?.drift_score_distribution.mean_delta ?? 0.012);
    const errorRate = dashboardData?.latency_throughput_error.error_rate ?? 0.0024;
    const progress = driftAbs < 0.02 && errorRate < 0.005 ? 84 : driftAbs < 0.04 && errorRate < 0.01 ? 62 : 38;

    return {
      progress,
      label: `${progress}% of traffic on canary`,
      detail: progress >= 80 ? 'Expansion is on track for full rollout after one more healthy nightly run.' : 'Hold current exposure until the next stability checkpoint clears.',
    };
  }, [dashboardData]);

  const apiSummary = useMemo(() => {
    const successRate = dashboardData ? Math.max(0, 1 - dashboardData.latency_throughput_error.error_rate) : null;
    const fraudHealthy = fraudInfo?.status?.toLowerCase() === 'ok';
    const walletHealthy = walletInfo?.status?.toLowerCase() === 'ok';

    return {
      requestVolume: dashboardData?.latency_throughput_error.requests ?? null,
      successRate,
      errorRate: dashboardData?.latency_throughput_error.error_rate ?? null,
      p95Latency: dashboardData?.latency_throughput_error.latency_ms_p95 ?? null,
      webhookSuccessRate: webhookEnabled ? 0.998 : 0,
      integrationStatus: fraudHealthy && walletHealthy ? 'Operational' : (fraudInfo || walletInfo ? 'Attention needed' : 'Checking'),
    };
  }, [dashboardData, fraudInfo, walletInfo, webhookEnabled]);

  const uptimeGaugeValue = useMemo(() => {
    const baseSuccessRate = apiSummary.successRate ?? 0.9986;
    return Math.min(100, Math.max(0, baseSuccessRate * 100));
  }, [apiSummary.successRate]);

  const uptimeGaugeData = useMemo(
    () => ([
      { name: 'uptime', value: uptimeGaugeValue, fill: uptimeGaugeValue >= 99.9 ? '#22c55e' : '#f59e0b' },
    ]),
    [uptimeGaugeValue],
  );

  const ringSummaryView = ringSummary ?? FALLBACK_RING_SUMMARY;

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

  const copyTransactionId = async () => {
    if (!selectedLiveTransaction) return;

    try {
      await navigator.clipboard.writeText(selectedLiveTransaction.txId);
      setTransactionIdCopyState('copied');
      window.setTimeout(() => setTransactionIdCopyState('idle'), 1600);
    } catch {
      setTransactionIdCopyState('idle');
    }
  };

  const applyOperatorAccess = async () => {
    const trimmed = operatorAccessCodeInput.trim();
    if (!trimmed) {
      setDashboardError('Enter the operator access code to unlock live analytics.');
      return;
    }

    setOperatorAccessCode(trimmed);
    setOperatorAccessArmed(true);
    setDashboardError(null);
    setOperatorAccessCodeInput('');
    await loadDashboardAndFocus('drift');
  };

  const clearOperatorAccess = () => {
    clearOperatorAccessCode();
    setOperatorAccessArmed(false);
    setDashboardData(null);
    setDashboardLastRefresh(null);
    setDashboardError('Operator analytics access was cleared from browser memory.');
  };

  const runRequestTester = async () => {
    setIsRunningTester(true);

    try {
      const payload = {
        ...DEFAULT_SCORE_TEST_PAYLOAD,
        transaction_amount: Number.parseFloat(quickTestState.amount) || DEFAULT_SCORE_TEST_PAYLOAD.transaction_amount,
        tx_type: quickTestState.paymentType,
        is_cross_border: quickTestState.crossBorder,
      };
      const response = await runMockScoreEndpoint(payload);
      setQuickTestResult(response);
    } finally {
      setIsRunningTester(false);
    }
  };

  const saveNotificationDestination = () => {
    setDestinationSaveState('saved');
    window.setTimeout(() => setDestinationSaveState('idle'), 1600);
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
    setTransactionHistory(loadTransactionHistory());
    setStatusMsg('status.loadingDashboards');
    try {
      const [dashboardResult, fraudInfoResult, walletInfoResult, ringSummaryResult] = await Promise.allSettled([
        fetchDashboardViews(24),
        fetchFraudInfo(),
        fetchWalletInfo(),
        fetchRingSummary(),
      ]);

      if (dashboardResult.status !== 'fulfilled') {
        throw dashboardResult.reason;
      }

      setDashboardData(dashboardResult.value);
      if (fraudInfoResult.status === 'fulfilled') {
        setFraudInfo(fraudInfoResult.value);
      }
      if (walletInfoResult.status === 'fulfilled') {
        setWalletInfo(walletInfoResult.value);
      }
      if (ringSummaryResult.status === 'fulfilled') {
        setRingSummary(ringSummaryResult.value);
      }
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
      const result = (await scoreTransactionRequest(buildFraudPayload())) as FraudResponse;

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
      const result = (await authorizePaymentRequest({ ...buildFraudPayload(), ...walletFields })) as WalletResponse;
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
  const exportLiveTransactionsCsv = () => {
    const rows = [
      ['Timestamp', 'TxID', 'UserID', 'Amount', 'Risk Score', 'Signals', 'Decision'],
      ...filteredLiveTransactions.map((tx) => [
        new Date(tx.timestamp).toISOString(),
        tx.txId,
        tx.userId,
        tx.amount,
        tx.riskScore.toFixed(2),
        tx.signals.join('|'),
        tx.decision,
      ]),
    ];
    const csv = rows
      .map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(','))
      .join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'live-transaction-feed.csv';
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleDashboardRangeStartChange = (value: string) => {
    setDashboardRangeStart(value);
    if (value > dashboardRangeEnd) {
      setDashboardRangeEnd(value);
    }
  };

  const handleDashboardRangeEndChange = (value: string) => {
    setDashboardRangeEnd(value);
    if (value < dashboardRangeStart) {
      setDashboardRangeStart(value);
    }
  };

  const handleDashboardPeriodPresetChange = (value: DashboardPeriodPreset) => {
    if (value === 'custom') {
      setDraftDashboardRangeStart(dashboardRangeStart);
      setDraftDashboardRangeEnd(dashboardRangeEnd);
      setIsCustomPeriodModalOpen(true);
      return;
    }

    setDashboardPeriodPreset(value);
    setIsCustomPeriodModalOpen(false);
  };

  const handleDraftDashboardRangeStartChange = (value: string) => {
    setDraftDashboardRangeStart(value);
    if (value > draftDashboardRangeEnd) {
      setDraftDashboardRangeEnd(value);
    }
  };

  const handleDraftDashboardRangeEndChange = (value: string) => {
    setDraftDashboardRangeEnd(value);
    if (value < draftDashboardRangeStart) {
      setDraftDashboardRangeStart(value);
    }
  };

  const applyCustomDashboardPeriod = () => {
    setDashboardRangeStart(draftDashboardRangeStart);
    setDashboardRangeEnd(draftDashboardRangeEnd);
    setDashboardPeriodPreset('custom');
    setIsCustomPeriodModalOpen(false);
  };

  // Retained transaction-scoring helpers support the fuller demo flow even though this route now renders
  // the analytics-focused dashboard only.
  void [
    localeOptions,
    localizeBackendText,
    SCENARIOS,
    SCENARIO_ICONS,
    DECISION_ICONS,
    dashboardOverview,
    setLocale,
    setHistoryFilter,
    setHistoryQuery,
    selectedPreset,
    statusMsg,
    showAdvanced,
    showExplain,
    isBusy,
    endpointSummary,
    translateScenarioLabel,
    updateFraudField,
    updateWalletField,
    applyScenario,
    resetForm,
    scoreTransaction,
    authorizePayment,
    latestDecision,
    score,
    reasons,
    latencyMs,
    fieldClassName,
    walletFieldClassName,
    selectedLiveTransaction,
    setSelectedLiveTransaction,
    exportLiveTransactionsCsv,
  ];

  return (
    <div className="ops-shell">
      <aside className="ops-sidebar">
        <div className="ops-sidebar-brand">
          <div className="ops-sidebar-logo">
            <KawalLogo />
          </div>
          <div className="ops-sidebar-brand-copy">
            <strong>Kawal</strong>
          </div>
        </div>

        <nav className="ops-sidebar-nav" aria-label="Primary navigation">
          {dashboardMenus.map((menu) => {
            const Icon = menu.icon;
            return (
              <button
                key={menu.key}
                type="button"
                className={`ops-sidebar-link ${activeMenu === menu.key ? 'active' : ''}`}
                onClick={() => setActiveMenu(menu.key)}
              >
                <Icon size={18} />
                <span>{menu.label}</span>
              </button>
            );
          })}
        </nav>
      </aside>

      <main className="ops-main">
        <div className="ops-main-scroll">
          <div className="ops-main-header">
            <div>
              <h1>{activeMenuMeta.label}</h1>
              <p>{activeMenuMeta.description}</p>
              {dashboardError && <p className="field-error">{dashboardError}</p>}
            </div>
            <div className="ops-header-actions">
              <div className="ops-period-filter-group">
                <label className="ops-period-filter-select">
                  <CalendarDays size={16} />
                  <select
                    value={dashboardPeriodPreset}
                    onChange={(event) => handleDashboardPeriodPresetChange(event.target.value as DashboardPeriodPreset)}
                    aria-label="Select dashboard period"
                  >
                    <option value="all_time">All time</option>
                    <option value="1_week">Last 1 week</option>
                    <option value="1_month">Last 1 month</option>
                    <option value="custom">Custom period</option>
                  </select>
                  <ChevronDown size={16} aria-hidden="true" />
                </label>
              </div>
              <button type="button" className="ghost-btn ops-refresh-btn" onClick={() => void loadDashboardAndFocus('drift')} disabled={isLoadingDashboard}>
                <RefreshCcw size={16} />
                {isLoadingDashboard ? 'Refreshing' : 'Refresh'}
              </button>
            </div>
          </div>

          {activeMenu === 'transactions' && (
            <section className="ops-live-layout">
              <div className="ops-live-main">
                <section>
                  <div className="ops-live-filters">
                    <div className="ops-search-shell">
                      <Search size={18} />
                      <input
                        type="text"
                        value={historyQuery}
                        onChange={(event) => setHistoryQuery(event.target.value)}
                        placeholder="Search transactions"
                        aria-label="Search transactions"
                      />
                    </div>
                    <label className="ops-status-filter-shell">
                      <select
                        className="ops-status-filter"
                        value={historyFilter}
                        onChange={(event) => setHistoryFilter(event.target.value as 'ALL' | Decision)}
                      >
                        <option value="ALL">All Status</option>
                        <option value="APPROVE">Approve</option>
                        <option value="FLAG">Flag</option>
                        <option value="BLOCK">Block</option>
                      </select>
                      <ChevronDown size={16} aria-hidden="true" />
                    </label>
                    <button type="button" className="ops-export-btn" onClick={exportLiveTransactionsCsv}>
                      <Download size={16} />
                      Export CSV
                    </button>
                  </div>
                </section>

                <section className="card">
                  <div className="transaction-table-wrapper ops-live-table-wrap">
                    <table className="transaction-table ops-live-table">
                      <thead>
                        <tr>
                          <th>TIMESTAMP</th>
                          <th>USER ID</th>
                          <th>AMOUNT</th>
                          <th>SIGNALS</th>
                          <th>RISK SCORE</th>
                          <th>DECISION</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredLiveTransactions.map((tx) => {
                          const timestampParts = formatTimestampParts(tx.timestamp);

                          return (
                            <tr
                              key={tx.id}
                              className={`ops-live-row row-${tx.decision.toLowerCase()}`}
                              onClick={() => setSelectedLiveTransaction(tx)}
                              style={{ cursor: 'pointer' }}
                            >
                              <td className="tx-time">
                                <span className="tx-date">{timestampParts.date}</span>
                                <span className="tx-time-detail">{timestampParts.time}</span>
                              </td>
                              <td>{tx.userId}</td>
                              <td>{tx.amount}</td>
                              <td>
                                <div className="ops-signal-list">
                                  {tx.signals.map((signal) => (
                                    <span key={signal} className="ops-signal-pill ops-signal-pill-neutral">{signal}</span>
                                  ))}
                                </div>
                              </td>
                              <td>{tx.riskScore.toFixed(2)}</td>
                              <td>
                                <span className={`status-badge status-${tx.decision.toLowerCase()}`}>
                                  <span className="status-badge-dot" aria-hidden="true" />
                                  {formatDecisionLabel(tx.decision)}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  {filteredLiveTransactions.length === 0 && (
                    <p className="card-note">No live transactions match the current filters.</p>
                  )}
                </section>
              </div>

              {selectedLiveTransaction && (
                <div className="ops-drawer-overlay" role="presentation" onClick={() => setSelectedLiveTransaction(null)}>
                  <aside
                    className="ops-drawer open"
                    aria-hidden="false"
                    onClick={(event) => event.stopPropagation()}
                  >
                    <div className="ops-drawer-header">
                      <div>
                        <h3>Transaction Details</h3>
                      </div>
                      <button type="button" className="ops-drawer-close" onClick={() => setSelectedLiveTransaction(null)}>
                        <X size={18} />
                      </button>
                    </div>

                    <div className="ops-drawer-section">
                      <div className="ops-detail-grid">
                        <div>
                          <span>User</span>
                          <strong>{selectedLiveTransaction.userId}</strong>
                        </div>
                        <div>
                          <span>Amount</span>
                          <strong>{selectedLiveTransaction.amount}</strong>
                        </div>
                        <div>
                          <span>Decision</span>
                          <strong>{formatDecisionLabel(selectedLiveTransaction.decision)}</strong>
                        </div>
                        <div>
                          <span>IP Location</span>
                          <strong>{selectedLiveTransaction.ipLocation}</strong>
                        </div>
                      </div>
                      <div className="ops-drawer-copy-field-wrap">
                        <span>Transaction ID</span>
                        <div className="ops-drawer-copy-field">
                          <code>{selectedLiveTransaction.txId}</code>
                          <button type="button" className="ops-drawer-copy-btn" onClick={() => void copyTransactionId()}>
                            {transactionIdCopyState === 'copied' ? <Check size={18} /> : <Copy size={18} />}
                            {transactionIdCopyState === 'copied' ? 'Copied' : 'Copy'}
                          </button>
                        </div>
                      </div>
                    </div>

                    <div className="ops-drawer-section">
                      <h4>Risk Breakdown</h4>
                      <div className="ops-breakdown-list">
                        {selectedLiveTransaction.riskBreakdown.map((item) => (
                          <div key={item.label} className="ops-breakdown-row">
                            <div className="ops-breakdown-head">
                              <span>{item.label}</span>
                              <strong className="ops-breakdown-value">{item.value.toFixed(2)}</strong>
                            </div>
                            <div className="ops-breakdown-track">
                              <div className={`ops-breakdown-fill ${item.tone}`} style={{ width: `${item.value * 100}%` }} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="ops-drawer-section">
                      <h4>Audit Log</h4>
                      <div className="ops-audit-log">
                        {selectedLiveTransaction.auditLog.map((entry, index) => {
                          const [meta, ...detailParts] = entry.split(' • ');
                          const detail = detailParts.join(' • ');

                          return (
                            <div key={entry} className="ops-audit-log-item">
                              <div className="ops-audit-log-head">
                                <span>{detail || `Event ${index + 1}`}</span>
                                <strong>{meta}</strong>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </aside>
                </div>
              )}
            </section>
          )}

          {activeMenu === 'rings' && (
            <>
              {ringSummaryView && (
                <>
                <section style={{ marginBottom: '24px' }}>
                  <div className="ops-command-grid ops-command-grid-two">
                    <article className="ops-command-card">
                      <span>Total Rings</span>
                      <strong>{ringSummaryView.total_rings}</strong>
                    </article>
                    <article className="ops-command-card">
                      <span>Accounts in Rings</span>
                      <strong>{ringSummaryView.total_accounts_in_rings}</strong>
                    </article>
                  </div>
                  <div className="ops-command-grid ops-command-grid-three" style={{ marginTop: '24px' }}>
                    <article className="ops-command-card">
                      <span>Low Risk</span>
                      <strong>{ringSummaryView.low_risk_rings}</strong>
                    </article>
                    <article className="ops-command-card">
                      <span>Medium Risk</span>
                      <strong>{ringSummaryView.medium_risk_rings}</strong>
                    </article>
                    <article className="ops-command-card">
                      <span>High Risk</span>
                      <strong>{ringSummaryView.high_risk_rings}</strong>
                    </article>
                  </div>
                </section>
                  {ringSummaryView.top_rings.length > 0 && (
                    <section className="card">
                    <div className="transaction-table-wrapper ops-live-table-wrap" style={{ maxHeight: '480px', overflowY: 'auto' }}>
                      <table className="transaction-table ops-live-table">
                        <thead>
                          <tr>
                            <th>RING ID</th>
                            <th>TIER</th>
                            <th>SCORE</th>
                            <th>FRAUD RATE</th>
                            <th>MEMBERS</th>
                            <th>ATTRIBUTES</th>
                          </tr>
                        </thead>
                        <tbody>
                          {ringSummaryView.top_rings.map((ring) => (
                            <tr key={ring.ring_id}>
                              <td style={{ fontFamily: 'monospace', fontSize: '0.82rem' }}>{ring.ring_id}</td>
                              <td>
                                <span
                                  className={`ops-ring-tier-pill ${
                                    ring.tier === 'high'
                                      ? 'badge-danger'
                                      : ring.tier === 'medium'
                                        ? 'badge-warn'
                                        : 'badge-safe'
                                  }`}
                                >
                                  {ring.tier}
                                </span>
                              </td>
                              <td>{ring.ring_score.toFixed(3)}</td>
                              <td>{ring.fraud_rate !== null ? `${(ring.fraud_rate * 100).toFixed(1)}%` : '—'}</td>
                              <td>{ring.ring_size}</td>
                              <td style={{ fontSize: '0.80rem', color: '#94a3b8' }}>{ring.attribute_types.join(', ')}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    </section>
                  )}
                </>
              )}
              <section className="card">
                <RingGraph variant="embedded" />
              </section>
            </>
          )}

          {activeMenu === 'mlops' && (
            <section className="ops-mlops-layout ops-system-health-layout">
              <div className="ops-mlops-grid ops-mlops-grid-four">
                  {qualityMetricCards.map((metric) => (
                    <article key={metric.label} className="ops-command-card">
                      <span>{metric.label}</span>
                      <strong>{metric.value}</strong>
                      <p className="ops-card-meta">{metric.detail}</p>
                    </article>
                  ))}
              </div>

              <div className="ops-mlops-grid ops-mlops-grid-two">
                  <article className="ops-mlops-panel ops-latency-panel">
                    <div className="ops-panel-head">
                      <div>
                        <strong>Latency Trend</strong>
                        <p>Recharts line view of decisioning latency percentiles.</p>
                      </div>
                    </div>
                    <div className="ops-chart-shell">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={latencyTrendData} margin={{ top: 12, right: 8, left: -18, bottom: 0 }}>
                          <CartesianGrid stroke="rgba(148, 163, 184, 0.14)" vertical={false} />
                          <XAxis dataKey="day" stroke="#94a3b8" tickLine={false} axisLine={false} />
                          <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} width={44} unit="ms" />
                          <Tooltip
                            contentStyle={{
                              background: '#0f172a',
                              border: '1px solid rgba(59, 130, 246, 0.25)',
                              borderRadius: '12px',
                            }}
                          />
                          <Legend />
                          <Line type="monotone" dataKey="p50" stroke="#38bdf8" strokeWidth={2.5} dot={false} name="p50" />
                          <Line type="monotone" dataKey="p95" stroke="#f59e0b" strokeWidth={2.5} dot={false} name="p95" />
                          <Line type="monotone" dataKey="p99" stroke="#f43f5e" strokeWidth={2.5} dot={false} name="p99" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </article>

                  <article className="ops-mlops-panel ops-fairness-panel">
                    <div className="ops-panel-head">
                      <div>
                        <strong>Fairness Monitor</strong>
                        <p>Nightly cohort checks compare calibration and false positive movement.</p>
                      </div>
                    </div>
                    <div className="transaction-table-wrapper ops-compact-table">
                      <table className="transaction-table">
                        <thead>
                          <tr>
                            <th>COHORT</th>
                            <th>PRECISION</th>
                            <th>RECALL</th>
                            <th>FALSE POSITIVE</th>
                            <th>DECISION SHARE</th>
                            <th>CALIBRATION GAP</th>
                            <th>STATUS</th>
                          </tr>
                        </thead>
                        <tbody>
                          {fairnessRows.map((row) => (
                            <tr key={row.cohort}>
                              <td>{row.cohort}</td>
                              <td>{formatPercentage(row.precision, 1)}</td>
                              <td>{formatPercentage(row.recall, 1)}</td>
                              <td>{formatPercentage(row.falsePositiveRate, 1)}</td>
                              <td>{formatPercentage(row.decisionShare, 1)}</td>
                              <td>{formatSignedPercentagePoints(row.calibrationGap, 1)}</td>
                              <td>
                                <span className={`ops-inline-pill ${row.tone}`}>
                                  {row.tone === 'positive' ? 'Within guardrail' : row.tone === 'warning' ? 'Monitor' : 'Needs review'}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </article>
              </div>

              <div className="ops-mlops-grid ops-mlops-grid-three">
                  <article className="ops-mlops-panel">
                    <div className="ops-panel-head">
                      <div>
                        <span>Drift detection</span>
                        <strong>{driftMonitor.runLabel}</strong>
                      </div>
                    </div>
                    <div className={`ops-status-banner ${driftMonitor.tone}`}>
                      <span className="ops-status-dot" aria-hidden="true" />
                      <div>
                        <strong>{driftMonitor.status}</strong>
                        <p>{driftMonitor.detail}</p>
                      </div>
                    </div>
                  </article>

                  <article className="ops-mlops-panel">
                    <div className="ops-panel-head">
                      <div>
                        <span>Retrain trigger log</span>
                        <strong>Latest automation decisions</strong>
                      </div>
                    </div>
                    <div className="ops-timeline">
                      {retrainTriggerLog.map((item) => (
                        <div key={`${item.timestamp}-${item.trigger}`} className="ops-timeline-item">
                          <strong>{item.trigger}</strong>
                          <p>{item.detail}</p>
                          <div className="ops-timeline-meta">{item.timestamp} • {item.status}</div>
                        </div>
                      ))}
                    </div>
                  </article>

                  <article className="ops-mlops-panel">
                    <div className="ops-panel-head">
                      <div>
                        <span>Canary rollout</span>
                        <strong>{canaryProgress.label}</strong>
                      </div>
                    </div>
                    <div className="ops-progress-track" aria-label="Canary rollout progress">
                      <div className="ops-progress-fill" style={{ width: `${canaryProgress.progress}%` }} />
                    </div>
                    <p className="ops-progress-copy">{canaryProgress.detail}</p>
                    <div className="ops-chart-shell ops-chart-shell-compact">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={fairnessRows} margin={{ top: 8, right: 4, left: -20, bottom: 0 }}>
                          <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" vertical={false} />
                          <XAxis dataKey="cohort" stroke="#94a3b8" tickLine={false} axisLine={false} />
                          <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} tickFormatter={(value) => `${value}%`} />
                          <Tooltip
                            formatter={(value) => `${Number(value ?? 0).toFixed(1)}%`}
                            contentStyle={{
                              background: '#0f172a',
                              border: '1px solid rgba(59, 130, 246, 0.25)',
                              borderRadius: '12px',
                            }}
                          />
                          <Bar dataKey="calibrationGap" radius={[8, 8, 0, 0]}>
                            {fairnessRows.map((row) => (
                              <Cell
                                key={row.cohort}
                                fill={row.tone === 'positive' ? '#22c55e' : row.tone === 'warning' ? '#f59e0b' : '#ef4444'}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </article>
              </div>
            </section>
          )}

          {activeMenu === 'api' && (
            <section className="ops-mlops-layout">
              <section className="ops-overview-grid">
                <section className="card">
                  <div className="ops-section-heading">
                    <h3>Access</h3>
                    <p>Everything the client team needs to start connecting their stack to the fraud engine.</p>
                  </div>
                  <div className="ops-card-body ops-api-key-card">
                    <div className="ops-source-list">
                      <div className="ops-source-item">
                        <strong>Environment</strong>
                        <p className="card-note">Sandbox mode active for prototype traffic and safe integration testing.</p>
                      </div>
                      <div className="ops-source-item">
                        <strong>Base endpoints</strong>
                        <p className="card-note">Scoring: {apiConfig.fraudBaseUrl}</p>
                        <p className="card-note">Wallet: {apiConfig.walletBaseUrl}</p>
                      </div>
                    </div>
                    <div className="ops-destination-config">
                      <label htmlFor="operator-access-code">Operator analytics access code</label>
                      <div className="ops-inline-actions ops-inline-actions-spaced">
                        <input
                          id="operator-access-code"
                          type="password"
                          value={operatorAccessCodeInput}
                          onChange={(event) => setOperatorAccessCodeInput(event.target.value)}
                          placeholder="Local dev default: vhack-demo-operator-2026"
                          autoComplete="off"
                          spellCheck={false}
                        />
                        <button type="button" className="ops-export-btn" onClick={() => void applyOperatorAccess()}>
                          Unlock analytics
                        </button>
                      </div>
                    </div>
                    <p className="card-note">
                      Analytics access is held in memory only and is cleared when the page refreshes. Status: {operatorAccessArmed ? 'unlocked in this tab' : 'locked'}.
                    </p>
                  </div>
                </section>

                <section className="card">
                  <div className="ops-section-heading">
                    <h3>SLA Uptime</h3>
                    <p>Target: 99.9% monthly uptime for the public scoring API.</p>
                  </div>
                  <div className="ops-card-body ops-api-gauge-layout">
                    <div className="ops-api-gauge-shell">
                      <ResponsiveContainer width="100%" height="100%">
                        <RadialBarChart
                          data={uptimeGaugeData}
                          startAngle={210}
                          endAngle={-30}
                          innerRadius="72%"
                          outerRadius="100%"
                          barSize={18}
                        >
                          <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
                          <RadialBar background dataKey="value" cornerRadius={12} />
                        </RadialBarChart>
                      </ResponsiveContainer>
                      <div className="ops-api-gauge-center">
                        <strong>{uptimeGaugeValue.toFixed(2)}%</strong>
                        <span>{uptimeGaugeValue >= 99.9 ? 'On target' : 'Monitor'}</span>
                      </div>
                    </div>
                    <div className="ops-source-list">
                      <div className="ops-source-item">
                        <strong>Scoring API</strong>
                        <p className="card-note">{fraudInfo ? `${fraudInfo.model_name} v${fraudInfo.model_version} • ${fraudInfo.status}` : 'Mock scoring service available'}</p>
                      </div>
                      <div className="ops-source-item">
                        <strong>Wallet orchestration</strong>
                        <p className="card-note">{walletInfo ? `${walletInfo.service} • retries ${walletInfo.resilience.max_retries}` : 'Mock wallet orchestration available'}</p>
                      </div>
                      <div className="ops-source-item">
                        <strong>Traffic snapshot</strong>
                        <p className="card-note">{apiSummary.requestVolume != null ? `${formatWholeNumber(apiSummary.requestVolume)} requests in the current window • p95 ${apiSummary.p95Latency?.toFixed(1) ?? 'n/a'} ms` : 'Using realistic prototype traffic defaults.'}</p>
                      </div>
                    </div>
                  </div>
                </section>
              </section>

              <section className="card">
                <div className="ops-section-heading">
                  <h3>Test</h3>
                  <p>Quickly validate how the scoring service behaves with a sample transaction before wiring your wallet flow end to end.</p>
                </div>
                <div className="ops-card-body ops-api-tester-grid">
                  <div>
                    <div className="ops-quick-form">
                      <label>
                        Amount
                        <input
                          type="text"
                          value={quickTestState.amount}
                          onChange={(event) => setQuickTestState((current) => ({ ...current, amount: event.target.value }))}
                        />
                      </label>
                      <label>
                        Payment type
                        <select
                          value={quickTestState.paymentType}
                          onChange={(event) => setQuickTestState((current) => ({ ...current, paymentType: event.target.value }))}
                        >
                          <option value="MERCHANT">Merchant purchase</option>
                          <option value="P2P">P2P transfer</option>
                          <option value="CASH_IN">Cash in</option>
                          <option value="CASH_OUT">Cash out</option>
                        </select>
                      </label>
                      <label className="ops-check-row">
                        <input
                          type="checkbox"
                          checked={quickTestState.crossBorder}
                          onChange={(event) => setQuickTestState((current) => ({ ...current, crossBorder: event.target.checked }))}
                        />
                        <span>Cross-border payment</span>
                      </label>
                    </div>
                    <div className="ops-inline-actions ops-inline-actions-spaced">
                      <button type="button" className="ops-export-btn" onClick={() => void runRequestTester()} disabled={isRunningTester}>
                        <Play size={16} />
                        {isRunningTester ? 'Testing...' : 'Run test'}
                      </button>
                      <button
                        type="button"
                        className="ghost-btn"
                        onClick={() => {
                          setQuickTestState({
                            amount: String(DEFAULT_SCORE_TEST_PAYLOAD.transaction_amount),
                            paymentType: DEFAULT_SCORE_TEST_PAYLOAD.tx_type,
                            crossBorder: DEFAULT_SCORE_TEST_PAYLOAD.is_cross_border,
                          });
                          setQuickTestResult(createPrototypeTesterResponse());
                        }}
                      >
                        Reset
                      </button>
                    </div>
                  </div>
                  <div className="ops-test-result-card">
                    <span className="ops-code-label">Sample result</span>
                    <div className="ops-test-result-header">
                      <strong>{quickTestResult.decision}</strong>
                      <span>{quickTestResult.final_risk_score?.toFixed(2) ?? '0.00'} risk score</span>
                    </div>
                    <div className="ops-source-list">
                      <div className="ops-source-item">
                        <strong>Primary reason</strong>
                        <p className="card-note">{quickTestResult.reasons?.[0] ?? 'Normal baseline behavior detected.'}</p>
                      </div>
                      <div className="ops-source-item">
                        <strong>Response time</strong>
                        <p className="card-note">{quickTestResult.stage_timings_ms?.total_pipeline_ms ?? 128} ms in prototype mode</p>
                      </div>
                      <div className="ops-source-item">
                        <strong>Decision breakdown</strong>
                        <p className="card-note">
                          Base {quickTestResult.explainability?.base?.toFixed(2) ?? '0.00'} •
                          Context {quickTestResult.explainability?.context?.toFixed(2) ?? '0.00'} •
                          Behavior {quickTestResult.explainability?.behavior?.toFixed(2) ?? '0.00'}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </section>

              <section className="card">
                <div className="ops-section-heading" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '16px' }}>
                  <div>
                    <h3>Wallet Integration</h3>
                    <p>Configure how fraud decisions are synchronized back into the client wallet or payment operations stack.</p>
                  </div>
                  <button
                    type="button"
                    className={`ops-toggle ${webhookEnabled ? 'active' : ''}`}
                    aria-pressed={webhookEnabled}
                    onClick={() => setWebhookEnabled((current) => !current)}
                  >
                    <span />
                  </button>
                </div>
                <div className="ops-card-body ops-webhook-card">
                  <div className="ops-source-list">
                      <div className="ops-source-item">
                        <strong>Destination URL</strong>
                      <div className="ops-destination-config">
                        <input
                          type="url"
                          value={notificationDestination}
                          onChange={(event) => setNotificationDestination(event.target.value)}
                          placeholder="https://client.example.com/wallet-updates"
                        />
                        <button type="button" className="ghost-btn" onClick={saveNotificationDestination}>
                          {destinationSaveState === 'saved' ? 'Saved' : 'Save'}
                        </button>
                      </div>
                      <p className="card-note">{webhookEnabled ? 'Used for wallet and ops updates in prototype mode.' : 'Saved for later use when decision sync is turned back on.'}</p>
                    </div>
                    <div className="ops-source-item">
                      <strong>What gets sent</strong>
                      <p className="card-note">Approved, flagged, blocked, and review-needed decisions</p>
                    </div>
                    <div className="ops-source-item">
                      <strong>Delivery health</strong>
                      <p className="card-note">{webhookEnabled ? `${formatPercentage(apiSummary.webhookSuccessRate ?? 0.998, 2)} successful delivery rate in prototype mode` : 'Paused by operator'}</p>
                    </div>
                  </div>
                </div>
              </section>

              <section className="card">
                <div className="ops-section-heading">
                  <h3>Endpoints</h3>
                  <p>A concise reference for the core API surfaces the client team will use during implementation.</p>
                </div>
                <div className="ops-card-body ops-mlops-grid ops-mlops-grid-three">
                  {API_REFERENCE_CARDS.map((endpoint) => (
                    <article key={endpoint.path} className="ops-mlops-panel">
                      <div className="ops-panel-head">
                        <div>
                          <span>{endpoint.method}</span>
                          <strong>{endpoint.path}</strong>
                        </div>
                        <p>{endpoint.latencyMs} ms mock latency</p>
                      </div>
                      <p className="card-note">{endpoint.summary}</p>
                      <div className="ops-source-list">
                        <div className="ops-source-item">
                          <strong>Best for</strong>
                          <p className="card-note">
                            {endpoint.path === '/score'
                              ? 'Scoring a payment before authorization'
                              : endpoint.path === '/explain'
                                ? 'Showing why a decision was made'
                                : 'Reviewing an event trail for support or compliance'}
                          </p>
                        </div>
                        <div className="ops-source-item">
                          <strong>Typical output</strong>
                          <p className="card-note">
                            {endpoint.path === '/score'
                              ? 'Risk score, decision, and timing'
                              : endpoint.path === '/explain'
                                ? 'Top contributing factors and narrative'
                                : 'Timeline of audit events and actors'}
                          </p>
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            </section>
          )}

          {activeMenu === 'command' && (
            <>
              <section className="ops-overview-stack">
                <div className="ops-command-grid ops-command-grid-two">
                  <article className="ops-command-card ops-command-card-wide">
                    <div className="ops-card-kebab" aria-hidden="true">
                      <MoreVertical size={16} />
                    </div>
                    <span>Estimated Fraud Prevented</span>
                    <strong>{dashboardData ? formatCurrency(dashboardOverview?.estimatedFraudLoss ?? 0) : 'Unavailable'}</strong>
                    <p className="ops-card-meta">Estimated value of fraudulent payment exposure stopped before settlement.</p>
                  </article>
                  <article className="ops-command-card ops-command-card-wide">
                    <div className="ops-card-kebab" aria-hidden="true">
                      <MoreVertical size={16} />
                    </div>
                    <span>Risk Engine Health</span>
                    <strong className="ops-health-inline">
                      <span className="ops-health-dot" aria-hidden="true" />
                      {mlopsHealthStatus}
                    </strong>
                    <p className="ops-card-meta">{mlopsHealthDetail}</p>
                  </article>
                </div>

                <div className="ops-command-grid ops-command-grid-three">
                  <article className="ops-command-card">
                    <div className="ops-card-kebab" aria-hidden="true">
                      <MoreVertical size={16} />
                    </div>
                    <span>Transactions Reviewed</span>
                    <strong>{formatWholeNumber(overviewMetrics.totalTransactions)}</strong>
                    <p className="ops-card-meta">Scored transactions across the current reporting window.</p>
                  </article>
                  <article className="ops-command-card">
                    <div className="ops-card-kebab" aria-hidden="true">
                      <MoreVertical size={16} />
                    </div>
                    <span>Processing Latency (p95)</span>
                    <strong>{dashboardData ? `${dashboardData.latency_throughput_error.latency_ms_p95.toFixed(1)} ms` : 'Unavailable'}</strong>
                    <p className="ops-card-meta">Time taken for nearly all scoring requests to complete.</p>
                  </article>
                  <article className="ops-command-card">
                    <div className="ops-card-kebab" aria-hidden="true">
                      <MoreVertical size={16} />
                    </div>
                    <span>False Positive Rate</span>
                    <strong>{formatPercentage(overviewMetrics.falsePositiveRate, 2)}</strong>
                    <p className="ops-card-meta">Legitimate payments incorrectly challenged or declined by the current controls.</p>
                  </article>
                </div>
              </section>

              <section className="card ops-mix-card">
                <div className="ops-section-heading ops-section-heading-tight">
                  <h3>Decision Mix</h3>
                </div>
                <div className="ops-mix-panel">
                  <div className="ops-mix-stack" aria-label="Decision distribution">
                    {decisionMixRows.map((item) => (
                      <div
                        key={item.label}
                        className={`ops-mix-segment ${item.tone === 'positive' ? 'positive' : item.tone}`}
                        style={{ width: `${Math.max(item.value * 100, item.value > 0 ? 10 : 0)}%` }}
                        title={`${item.label}: ${item.count} transactions (${formatPercentage(item.value, 1)})`}
                      />
                    ))}
                  </div>
                  <div className="ops-mix-legend">
                    {decisionMixRows.map((item) => (
                      <article key={item.label} className="ops-mix-item">
                        <span className="ops-mix-dot-wrap" aria-hidden="true">
                          <span className={`ops-mix-dot ${item.tone === 'positive' ? 'positive' : item.tone}`} />
                        </span>
                        <div className="ops-mix-copy">
                          <strong>{item.label === 'Step-Up / Review' ? 'Flag' : item.label}</strong>
                          <p>{item.count} Transactions</p>
                        </div>
                        <span className="ops-mix-value">{formatPercentage(item.value, 0)}</span>
                      </article>
                    ))}
                  </div>
                </div>
              </section>

              <section className="card">
                <div className="ops-table-header ops-table-header-tight">
                  <div>
                    <h3>Transactions</h3>
                    {dashboardLastRefresh && (
                      <p className="card-note">Updated {new Date(dashboardLastRefresh).toLocaleTimeString()} with the latest dashboard snapshot.</p>
                    )}
                  </div>
                  <button type="button" className="ghost-btn" onClick={() => setActiveMenu('transactions')}>
                    View All
                  </button>
                </div>

                <div className="transaction-table-wrapper ops-overview-table-wrap">
                  <table className="transaction-table ops-live-table">
                    <thead>
                      <tr>
                        <th>TIMESTAMP</th>
                        <th>USER ID</th>
                        <th>AMOUNT</th>
                        <th>SIGNALS</th>
                        <th>RISK SCORE</th>
                        <th>DECISION</th>
                      </tr>
                    </thead>
                    <tbody>
                      {overviewTableRows.map((row) => {
                        const timestampParts = formatTimestampParts(row.timestamp);

                        return (
                          <tr key={row.id} className={`ops-live-row row-${row.decision.toLowerCase()}`}>
                            <td className="tx-time">
                              <span className="tx-date">{timestampParts.date}</span>
                              <span className="tx-time-detail">{timestampParts.time}</span>
                            </td>
                            <td className="ops-cell-user">{row.userId}</td>
                            <td className="ops-cell-secondary">{row.amount}</td>
                            <td>
                              <div className="ops-signal-list">
                                {row.signals.map((signal) => (
                                  <span key={signal} className="ops-signal-pill ops-signal-pill-neutral">{signal}</span>
                                ))}
                              </div>
                            </td>
                            <td className="ops-cell-secondary">{row.riskScore.toFixed(2)}</td>
                            <td>
                              <span className={`status-badge status-${row.decision.toLowerCase()}`}>
                                <span className="status-badge-dot" aria-hidden="true" />
                                {formatDecisionLabel(row.decision)}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                {overviewTableRows.length === 0 && (
                  <p className="card-note ops-table-empty">No transactions are available for the current reporting window.</p>
                )}
                <div className="ops-table-footer">
                  <p>Page {overviewTablePage} of {overviewTablePageCount}</p>
                  <div className="ops-inline-actions">
                    <button
                      type="button"
                      className="ghost-btn ops-table-page-btn"
                      onClick={() => setOverviewTablePage((current) => Math.max(1, current - 1))}
                      disabled={overviewTablePage <= 1}
                    >
                      Previous
                    </button>
                    <button
                      type="button"
                      className="ghost-btn ops-table-page-btn"
                      onClick={() => setOverviewTablePage((current) => Math.min(overviewTablePageCount, current + 1))}
                      disabled={overviewTablePage >= overviewTablePageCount}
                    >
                      Next
                    </button>
                  </div>
                </div>
              </section>

              <div className="ops-hidden-diagnostics" aria-hidden="true">
                {dashboardData && (
                  <>
                    <div ref={driftSectionRef}><DashboardDriftCard dashboardData={dashboardData} /></div>
                    <div ref={benchmarkSectionRef}><DashboardBenchmarkCard dashboardData={dashboardData} /></div>
                    <div ref={kpiSectionRef}><DashboardKpiCard dashboardData={dashboardData} /></div>
                  </>
                )}
              </div>
            </>
          )}
        </div>
      </main>

      {isCustomPeriodModalOpen && (
        <div className="ops-modal-backdrop" role="presentation" onClick={() => setIsCustomPeriodModalOpen(false)}>
          <section
            className="ops-period-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="custom-period-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="ops-period-modal-header">
              <div>
                <h2 id="custom-period-title">Custom Period</h2>
                <p>Choose the start and end dates you want this dashboard to show.</p>
              </div>
              <button type="button" className="ops-drawer-close" onClick={() => setIsCustomPeriodModalOpen(false)}>
                <X size={16} />
              </button>
            </div>
            <div className="ops-period-modal-body">
              <label className="ops-period-modal-field">
                <span>From</span>
                <input
                  type="date"
                  value={draftDashboardRangeStart}
                  max={draftDashboardRangeEnd}
                  onChange={(event) => handleDraftDashboardRangeStartChange(event.target.value)}
                />
              </label>
              <label className="ops-period-modal-field">
                <span>To</span>
                <input
                  type="date"
                  value={draftDashboardRangeEnd}
                  min={draftDashboardRangeStart}
                  onChange={(event) => handleDraftDashboardRangeEndChange(event.target.value)}
                />
              </label>
            </div>
            <div className="ops-period-modal-actions">
              <button type="button" className="ghost-btn" onClick={() => setIsCustomPeriodModalOpen(false)}>
                Cancel
              </button>
              <button type="button" onClick={applyCustomDashboardPeriod}>
                Apply
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

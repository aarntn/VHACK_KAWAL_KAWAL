export type TransactionHistory = {
  id: string;
  userId: string;
  status: 'APPROVE' | 'FLAG' | 'BLOCK';
  transactionType: string;
  amount: string;
  merchantName: string;
  walletId: string;
  currency: string;
  crossBorder: boolean;
  timestamp: string;
  riskScore?: number;
  latencyMs?: number;
  explainabilityBase?: number;
  explainabilityContext?: number;
  explainabilityBehavior?: number;
  explainabilityRing?: number;
  explainabilityExternal?: number;
  mcpWatchlistHit?: boolean;
  mcpRiskTier?: string;
  runtimeMode?: string;
  corridor?: string;
  walletAction?: string;
  walletMessage?: string;
  walletDecision?: string;
  nextStep?: string;
};

const STORAGE_KEY = 'vhack_transaction_history';
const HISTORY_UPDATED_EVENT = 'vhack-transaction-history-updated';
const getLocalHistoryStorage = (): Storage | null => {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    return window.localStorage;
  } catch {
    return null;
  }
};

const getSessionHistoryStorage = (): Storage | null => {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
};

const getWritableHistoryStorage = (): Storage | null => (
  getLocalHistoryStorage() ?? getSessionHistoryStorage()
);

const readStoredHistory = (storage: Storage | null): TransactionHistory[] | null => {
  if (!storage) {
    return null;
  }

  try {
    const stored = storage.getItem(STORAGE_KEY);
    if (!stored) {
      return null;
    }
    const parsed = JSON.parse(stored);
    return Array.isArray(parsed) ? parsed.map((record) => sanitizeTransaction(record as TransactionHistory)) : [];
  } catch {
    return [];
  }
};

const clearLegacySessionHistory = (): void => {
  const sessionStorage = getSessionHistoryStorage();
  if (!sessionStorage) {
    return;
  }

  try {
    sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // Best effort cleanup only.
  }
};

const migrateSessionHistoryToLocalStorage = (transactions: TransactionHistory[]): void => {
  const localStorage = getLocalHistoryStorage();
  if (!localStorage || transactions.length === 0) {
    return;
  }

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(transactions.map((transaction) => sanitizeTransaction(transaction))));
    clearLegacySessionHistory();
  } catch {
    try {
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(transactions.map((transaction) => sanitizeTransaction(transaction))));
    } catch {
      // Keep the read path resilient even when browser storage is unavailable.
    }
  }
};

const maskStoredIdentifier = (value: string, prefix: string): string => {
  const trimmed = value.trim();
  if (!trimmed || trimmed.toLowerCase() === 'anonymous' || trimmed.toLowerCase() === 'unknown') {
    return trimmed || `${prefix}_unknown`;
  }
  if (trimmed.length <= 4) {
    return `${prefix}_****`;
  }
  return `${trimmed.slice(0, 2)}***${trimmed.slice(-2)}`;
};

const sanitizeTransaction = (transaction: TransactionHistory): TransactionHistory => ({
  ...transaction,
  userId: maskStoredIdentifier(transaction.userId, 'usr'),
  walletId: maskStoredIdentifier(transaction.walletId, 'wlt'),
});

export const loadTransactionHistory = (): TransactionHistory[] => {
  const localHistory = readStoredHistory(getLocalHistoryStorage());
  if (localHistory !== null) {
    return localHistory;
  }

  const sessionHistory = readStoredHistory(getSessionHistoryStorage());
  if (sessionHistory !== null) {
    migrateSessionHistoryToLocalStorage(sessionHistory);
    return sessionHistory;
  }

  return [];
};

export const saveTransactionHistory = (transactions: TransactionHistory[]): void => {
  const storage = getWritableHistoryStorage();
  if (!storage) {
    return;
  }
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(transactions.map((transaction) => sanitizeTransaction(transaction))));
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent(HISTORY_UPDATED_EVENT, { detail: transactions }));
    }
  } catch {
    console.error('Failed to save transaction history');
  }
};

export const addTransactionToHistory = (transaction: TransactionHistory): TransactionHistory[] => {
  const history = loadTransactionHistory();
  const updated = [transaction, ...history].slice(0, 50); // Keep last 50
  saveTransactionHistory(updated);
  return updated;
};

export const clearTransactionHistory = (): void => {
  try {
    getLocalHistoryStorage()?.removeItem(STORAGE_KEY);
    getSessionHistoryStorage()?.removeItem(STORAGE_KEY);
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent(HISTORY_UPDATED_EVENT));
    }
  } catch {
    console.error('Failed to clear transaction history');
  }
};

export const updateTransactionInHistory = (
  transactionId: string,
  updates: Partial<TransactionHistory>,
): TransactionHistory[] => {
  const history = loadTransactionHistory();
  const updated = history.map((tx) => (tx.id === transactionId ? { ...tx, ...updates } : tx));
  saveTransactionHistory(updated);
  return updated;
};

export const TRANSACTION_HISTORY_UPDATED_EVENT = HISTORY_UPDATED_EVENT;

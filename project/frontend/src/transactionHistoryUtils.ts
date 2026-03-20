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
  walletAction?: string;
  walletMessage?: string;
  walletDecision?: string;
  nextStep?: string;
};

const STORAGE_KEY = 'vhack_transaction_history';

export const loadTransactionHistory = (): TransactionHistory[] => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
};

export const saveTransactionHistory = (transactions: TransactionHistory[]): void => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(transactions));
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
    localStorage.removeItem(STORAGE_KEY);
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

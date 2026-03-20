import { useEffect, useState } from 'react';
import { type TransactionHistory, loadTransactionHistory } from './transactionHistoryUtils';
import { localizeBackendText, t, localeOptions, type Locale } from './i18n';
import './App.css';

type DetailUserProps = {
  transactionId: string;
};

export default function DetailUser({ transactionId }: DetailUserProps) {
  const [transaction, setTransaction] = useState<TransactionHistory | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [locale, setLocale] = useState<Locale>('en');

  useEffect(() => {
    // Load transaction history from localStorage and find the transaction by ID
    const history = loadTransactionHistory();
    const found = history.find((tx) => tx.id === transactionId);
    setTransaction(found || null);
    setIsLoading(false);
  }, [transactionId]);

  const handleBackToDashboard = () => {
    window.history.pushState(null, '', '/dashboard');
    window.dispatchEvent(new PopStateEvent('popstate'));
  };

  const translate = (key: string): string => t(locale, key);

  if (isLoading) {
    return (
      <div className="container">
        <header className="hero">
          <h1>Loading...</h1>
        </header>
      </div>
    );
  }

  if (!transaction) {
    return (
      <div className="container">
        <header className="hero">
          <h1>Transaction Not Found</h1>
          <p>The transaction you're looking for does not exist.</p>
        </header>
        <section className="card">
          <button className="ghost-btn" type="button" onClick={handleBackToDashboard}>
            ← Back to Dashboard
          </button>
        </section>
      </div>
    );
  }

  const statusBadgeClass = `status-badge status-${transaction.status.toLowerCase()}`;
  const txRowClass = `tx-row tx-${transaction.status.toLowerCase()}`;

  return (
    <div className="container">
      <header className="hero">
        <h1>Transaction Details</h1>
        <p>View the complete transaction information below</p>
      </header>

      <section className="card">
        <div className="detail-header">
          <button className="ghost-btn" type="button" onClick={handleBackToDashboard}>
            ← Back to Dashboard
          </button>
        </div>

         <div className={"card"} style={{ padding: '20px', marginBottom: '20px', borderRadius: '8px' }}>    
        <h2>Transaction Overview</h2>
          <div className="detail-grid">
            <div className="detail-item">
              <span className="label">Transaction ID:</span>
              <span className="value">{transaction.id}</span>
            </div>
            <div className="detail-item">
              <span className="label">User ID:</span>
              <span className="value">{transaction.userId}</span>
            </div>
            <div className="detail-item">
              <span className="label">Status:</span>
              <span className={statusBadgeClass}>{transaction.status}</span>
            </div>
            <div className="detail-item">
              <span className="label">Transaction Type:</span>
              <span className="value">{transaction.transactionType}</span>
            </div>
            <div className="detail-item">
              <span className="label">Amount:</span>
              <span className="value">{transaction.amount}</span>
            </div>
            <div className="detail-item">
              <span className="label">Merchant Name:</span>
              <span className="value">{transaction.merchantName}</span>
            </div>
            <div className="detail-item">
              <span className="label">Wallet ID:</span>
              <span className="value">{transaction.walletId}</span>
            </div>
            <div className="detail-item">
              <span className="label">Currency:</span>
              <span className="value">{transaction.currency}</span>
            </div>
            <div className="detail-item">
              <span className="label">Cross Border:</span>
              <span className="value">{transaction.crossBorder ? 'Yes' : 'No'}</span>
            </div>
        </div>
        </div>    

        <div className={"card"} style={{ padding: '20px', marginBottom: '20px', borderRadius: '8px' }}>
          <h2>Detail Reasons</h2>
          <div className="detail-grid">
            <div className="detail-item">
              <span className="label">Risk Score:</span>
              <span className="value">
                {typeof transaction.riskScore === 'number' ? transaction.riskScore.toFixed(4) : '-'}
              </span>
            </div>
            <div className="detail-item">
              <span className="label">Latency (ms):</span>
              <span className="value">
                {typeof transaction.latencyMs === 'number' ? transaction.latencyMs.toFixed(2) : '-'}
              </span>
            </div>
            <div className="detail-item">
              <span className="label">Base Model Score:</span>
              <span className="value">
                {typeof transaction.explainabilityBase === 'number' ? transaction.explainabilityBase.toFixed(4) : '-'}
              </span>
            </div>
            <div className="detail-item">
              <span className="label">Context Adjustment:</span>
              <span className="value">
                {typeof transaction.explainabilityContext === 'number' ? transaction.explainabilityContext.toFixed(4) : '-'}
              </span>
            </div>
            <div className="detail-item">
              <span className="label">Behavior Adjustment:</span>
              <span className="value">
                {typeof transaction.explainabilityBehavior === 'number' ? transaction.explainabilityBehavior.toFixed(4) : '-'}
              </span>
            </div>
            <div className="detail-item">
              <span className="label">Timestamp:</span>
              <span className="value">{new Date(transaction.timestamp).toLocaleString()}</span>
            </div>
          </div>
        </div>

        <section className="card">
          <h2>{translate('guidance.title')}</h2>
          {transaction.status === 'APPROVE' && <p>{translate('guidance.approve')}</p>}
          {transaction.status === 'FLAG' && <p>{translate('guidance.flag')}</p>}
          {transaction.status === 'BLOCK' && <p>{translate('guidance.block')}</p>}
        </section>
    
      </section>
    </div>
  );
}

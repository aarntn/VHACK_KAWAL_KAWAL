import { useMemo, useState } from 'react';
import { loadTransactionHistory } from './transactionHistoryUtils';
import { t, type Locale } from './i18n';
import './App.css';

type DetailUserProps = {
  transactionId: string;
};

export default function DetailUser({ transactionId }: DetailUserProps) {
  const [locale] = useState<Locale>('en');
  const transaction = useMemo(() => {
    const history = loadTransactionHistory();
    return history.find((tx) => tx.id === transactionId) ?? null;
  }, [transactionId]);

  const handleBackToDashboard = () => {
    window.history.pushState(null, '', '/dashboard');
    window.dispatchEvent(new PopStateEvent('popstate'));
  };

  const translate = (key: string): string => t(locale, key);

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
          <h2>Risk Score Breakdown</h2>

          {/* MCP Watchlist Badge */}
          {transaction.mcpWatchlistHit && (
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: 8,
              background: 'rgba(251,191,36,0.12)', border: '1px solid rgba(251,191,36,0.35)',
              borderRadius: 8, padding: '7px 14px', marginBottom: 16, fontSize: 12, fontWeight: 600,
              color: '#fbbf24', letterSpacing: 0.3,
            }}>
              <span style={{ fontSize: 14 }}>⚠</span>
              External Watchlist Hit — tier: {transaction.mcpRiskTier ?? 'unknown'}
            </div>
          )}

          {/* Stacked explainability bar */}
          {(() => {
            const base     = transaction.explainabilityBase     ?? 0;
            const context  = transaction.explainabilityContext  ?? 0;
            const behavior = transaction.explainabilityBehavior ?? 0;
            const ring     = transaction.explainabilityRing     ?? 0;
            const external = transaction.explainabilityExternal ?? 0;
            const total    = Math.max(base + Math.abs(context) + Math.abs(behavior) + Math.abs(ring) + Math.abs(external), 0.001);
            const pct = (v: number) => `${Math.max(0, (Math.abs(v) / total) * 100).toFixed(1)}%`;
            const segments = [
              { label: 'Base',     value: base,     color: '#4d8eff' },
              { label: 'Context',  value: context,  color: '#f59e0b' },
              { label: 'Behavior', value: behavior, color: '#2dd4bf' },
              { label: 'Ring',     value: ring,     color: '#ef4444' },
              { label: 'MCP Ext',  value: external, color: '#a855f7' },
            ].filter(s => Math.abs(s.value) > 0.0001);
            return (
              <div style={{ marginBottom: 20 }}>
                <div style={{ display: 'flex', borderRadius: 6, overflow: 'hidden', height: 20, marginBottom: 10, background: '#1c2026' }}>
                  {segments.map(s => (
                    <div key={s.label} title={`${s.label}: ${s.value.toFixed(4)}`}
                      style={{ width: pct(s.value), background: s.color, transition: 'width 0.4s ease' }} />
                  ))}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 14px' }}>
                  {segments.map(s => (
                    <span key={s.label} style={{ fontSize: 11, color: '#c2c6d6', display: 'flex', alignItems: 'center', gap: 5 }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: s.color, display: 'inline-block' }} />
                      {s.label} <strong style={{ color: '#dfe2eb' }}>{s.value >= 0 ? '+' : ''}{s.value.toFixed(3)}</strong>
                    </span>
                  ))}
                </div>
              </div>
            );
          })()}

          <div className="detail-grid">
            <div className="detail-item">
              <span className="label">Final Risk Score:</span>
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

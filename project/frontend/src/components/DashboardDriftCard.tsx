import type { DashboardViewsResponse } from '../api';

type Props = {
  dashboardData: DashboardViewsResponse;
};

function DashboardDriftCard({ dashboardData }: Props) {
  const drift = dashboardData.drift_score_distribution;
  const histogramEntries = Object.entries(drift.score_histogram);
  const maxCount = Math.max(...histogramEntries.map(([, count]) => count), 1);
  const absDelta = Math.abs(drift.mean_delta);
  const status = absDelta >= 0.1 ? 'needs attention' : absDelta >= 0.04 ? 'watching' : 'stable';

  return (
    <div className="ops-panel-content">
      <div className="ops-panel-header">
        <div>
          <h3>Risk drift</h3>
          <p className="signal-score">Score distribution movement in the latest window</p>
        </div>
        <span className={`ops-badge ops-badge-${status.replace(' ', '-')}`}>{status}</span>
      </div>

      <div className="mini-kpi-grid">
        <div className="mini-kpi-card">
          <span>Delta</span>
          <strong>{drift.mean_delta >= 0 ? '+' : ''}{drift.mean_delta.toFixed(4)}</strong>
        </div>
        <div className="mini-kpi-card">
          <span>Baseline mean</span>
          <strong>{drift.baseline_mean_score.toFixed(4)}</strong>
        </div>
        <div className="mini-kpi-card">
          <span>Observed mean</span>
          <strong>{drift.observed_mean_score.toFixed(4)}</strong>
        </div>
      </div>

      <div className="distribution-list">
        {histogramEntries.map(([range, count]) => (
          <div key={range} className="distribution-row">
            <div className="distribution-meta">
              <span>{range}</span>
              <strong>{count}</strong>
            </div>
            <div className="distribution-track">
              <div className="distribution-fill" style={{ width: `${(count / maxCount) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default DashboardDriftCard;

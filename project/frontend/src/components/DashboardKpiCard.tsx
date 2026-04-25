import type { DashboardViewsResponse } from '../api';

type Props = {
  dashboardData: DashboardViewsResponse;
};

function DashboardKpiCard({ dashboardData }: Props) {
  const kpi = dashboardData.fraud_loss_false_positives_analyst_agreement;
  const sourceRows = dashboardData.decision_source_kpis ?? [];
  const maxVolume = Math.max(...sourceRows.map((row) => row.audit_volume), 1);

  return (
    <div className="ops-panel-content">
      <div className="ops-panel-header">
        <div>
          <h3>Business outcomes</h3>
          <p className="signal-score">Loss, review quality, and decision-source performance</p>
        </div>
      </div>

      <div className="mini-kpi-grid">
        <div className="mini-kpi-card">
          <span>Estimated fraud loss</span>
          <strong>${kpi.estimated_fraud_loss.toFixed(2)}</strong>
        </div>
        <div className="mini-kpi-card">
          <span>False positives</span>
          <strong>{kpi.false_positives}</strong>
        </div>
        <div className="mini-kpi-card">
          <span>Analyst agreement</span>
          <strong>{(kpi.analyst_agreement * 100).toFixed(1)}%</strong>
        </div>
        <div className="mini-kpi-card">
          <span>Confirmed fraud</span>
          <strong>{kpi.confirmed_fraud_cases}</strong>
        </div>
      </div>

      {sourceRows.length > 0 && (
        <div className="source-breakdown">
          <h4>Decision source split</h4>
          {sourceRows.map((row) => (
            <div key={row.decision_source} className="source-row">
              <div className="source-row-head">
                <strong>{row.decision_source.replaceAll('_', ' ')}</strong>
                <span>{row.audit_volume} reviews</span>
              </div>
              <div className="distribution-track">
                <div className="distribution-fill" style={{ width: `${(row.audit_volume / maxVolume) * 100}%` }} />
              </div>
              <p>
                Flag {(row.flag_rate * 100).toFixed(1)}% • Fraud conversion {(row.confirmed_fraud_conversion * 100).toFixed(1)}% • False positives {(row.false_positive_rate * 100).toFixed(1)}%
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default DashboardKpiCard;

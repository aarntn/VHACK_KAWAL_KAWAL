import type { DashboardViewsResponse } from '../api';

type Props = {
  dashboardData: DashboardViewsResponse;
};

function DashboardKpiCard({ dashboardData }: Props) {
  const kpi = dashboardData.fraud_loss_false_positives_analyst_agreement;
  const sourceRows = dashboardData.decision_source_kpis ?? [];

  return (
    <>
      <h3>KPI</h3>
      <p>Estimated fraud loss: <strong>${kpi.estimated_fraud_loss.toFixed(2)}</strong></p>
      <p>False positives: <strong>{kpi.false_positives}</strong></p>
      <p>Analyst agreement: <strong>{kpi.analyst_agreement.toFixed(4)}</strong></p>
      {sourceRows.length > 0 && (
        <>
          <h4>Decision source split</h4>
          {sourceRows.map((row) => (
            <p key={row.decision_source}>
              <strong>{row.decision_source}</strong>: flag {row.flag_rate.toFixed(4)} • confirmed fraud {row.confirmed_fraud_conversion.toFixed(4)} • false positives {row.false_positive_rate.toFixed(4)}
            </p>
          ))}
        </>
      )}
    </>
  );
}

export default DashboardKpiCard;

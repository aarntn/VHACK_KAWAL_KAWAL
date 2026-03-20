import type { DashboardViewsResponse } from '../api';

type Props = {
  dashboardData: DashboardViewsResponse;
};

function DashboardDriftCard({ dashboardData }: Props) {
  return (
    <>
      <h3>Drift</h3>
      <p className="signal-score">
        Δ mean score: <strong>{dashboardData.drift_score_distribution.mean_delta.toFixed(4)}</strong>
      </p>
      <pre>{JSON.stringify(dashboardData.drift_score_distribution.score_histogram, null, 2)}</pre>
    </>
  );
}

export default DashboardDriftCard;

import type { DashboardViewsResponse } from '../api';

type Props = {
  dashboardData: DashboardViewsResponse;
};

function DashboardBenchmarkCard({ dashboardData }: Props) {
  const benchmark = dashboardData.latency_throughput_error;
  const health = benchmark.error_rate > 0.02 || benchmark.latency_ms_p95 > 800
    ? 'needs attention'
    : benchmark.error_rate > 0.005 || benchmark.latency_ms_p95 > 250
      ? 'watching'
      : 'healthy';

  return (
    <div className="ops-panel-content">
      <div className="ops-panel-header">
        <div>
          <h3>System health</h3>
          <p className="signal-score">Operational reliability for scoring traffic</p>
        </div>
        <span className={`ops-badge ops-badge-${health.replace(' ', '-')}`}>{health}</span>
      </div>

      <div className="benchmark-hero">
        <span>P95 latency</span>
        <strong>{benchmark.latency_ms_p95.toFixed(2)} ms</strong>
      </div>

      <div className="mini-kpi-grid">
        <div className="mini-kpi-card">
          <span>P50 latency</span>
          <strong>{benchmark.latency_ms_p50.toFixed(2)} ms</strong>
        </div>
        <div className="mini-kpi-card">
          <span>Requests</span>
          <strong>{benchmark.requests}</strong>
        </div>
        <div className="mini-kpi-card">
          <span>Throughput/min</span>
          <strong>{benchmark.throughput_per_min.toFixed(3)}</strong>
        </div>
        <div className="mini-kpi-card">
          <span>Error rate</span>
          <strong>{(benchmark.error_rate * 100).toFixed(2)}%</strong>
        </div>
      </div>
    </div>
  );
}

export default DashboardBenchmarkCard;

import type { DashboardViewsResponse } from '../api';

type Props = {
  dashboardData: DashboardViewsResponse;
};

function DashboardBenchmarkCard({ dashboardData }: Props) {
  return (
    <>
      <h3>Benchmark</h3>
      <p className="signal-score">
        Latency p50/p95: <strong>{dashboardData.latency_throughput_error.latency_ms_p50.toFixed(2)} / {dashboardData.latency_throughput_error.latency_ms_p95.toFixed(2)} ms</strong>
      </p>
      <p>Requests: <strong>{dashboardData.latency_throughput_error.requests}</strong></p>
    </>
  );
}

export default DashboardBenchmarkCard;

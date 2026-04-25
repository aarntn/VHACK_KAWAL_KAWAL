import unittest

from project.scripts import benchmark_inference_candidates as bic


class BenchmarkInferenceCandidatesScriptTests(unittest.TestCase):
    def _row(self, name: str, p95: float, pr_auc: float, recall: float) -> bic.CandidateMetrics:
        return bic.CandidateMetrics(
            candidate=name,
            runtime="native",
            fit_time_ms=1.0,
            startup_time_ms=1.0,
            memory_bytes=1024,
            p95_latency_ms=p95,
            p99_latency_ms=p95 + 1.0,
            precision=0.8,
            recall=recall,
            f1=0.7,
            pr_auc=pr_auc,
            roc_auc=0.9,
            notes=[],
        )

    def test_pick_candidate_by_latency_with_thresholds(self) -> None:
        rows = [
            self._row("baseline", 9.0, 0.82, 0.78),
            self._row("faster", 6.0, 0.80, 0.76),
            self._row("too_low_recall", 5.0, 0.81, 0.50),
        ]
        promoted, notes = bic.pick_promotion_candidate(rows, sla_max_p95_ms=10.0, min_pr_auc=0.79, min_recall=0.7)
        self.assertIsNotNone(promoted)
        assert promoted is not None
        self.assertEqual(promoted.candidate, "faster")
        self.assertEqual(notes, [])

    def test_regression_gate_blocks_latency_only_win(self) -> None:
        baseline = self._row("baseline", 9.0, 0.85, 0.80)
        promoted = self._row("fast_bad", 5.0, 0.80, 0.70)
        gate = bic.apply_regression_gate(baseline, promoted, max_pr_auc_drop=0.02, max_recall_drop=0.05)
        self.assertTrue(gate["blocked"])
        self.assertEqual(gate["reason"], "latency_improved_but_kpi_regressed")


if __name__ == "__main__":
    unittest.main()

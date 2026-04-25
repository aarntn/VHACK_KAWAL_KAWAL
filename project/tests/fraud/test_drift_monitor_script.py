import unittest

import pandas as pd

from project.scripts.drift_monitor import (
    DriftThresholds,
    build_recalibration_recommendation,
    classify_drift_level,
    compute_decision_distribution,
    compute_psi,
    decision_distribution_drift,
    evaluate_feature_drift,
    split_audit_windows,
)


class DriftMonitorScriptTests(unittest.TestCase):
    def test_compute_psi_near_zero_when_distributions_match(self) -> None:
        expected = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        actual = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

        psi = compute_psi(expected, actual, bins=5)
        self.assertAlmostEqual(psi, 0.0, places=6)

    def test_decision_distribution_drift_detects_shift(self) -> None:
        baseline = {"APPROVE": 0.9, "FLAG": 0.05, "BLOCK": 0.05}
        current = {"APPROVE": 0.6, "FLAG": 0.3, "BLOCK": 0.1}

        drift = decision_distribution_drift(
            baseline_distribution=baseline,
            current_distribution=current,
            decisions=["APPROVE", "FLAG", "BLOCK"],
        )

        self.assertGreater(drift, 0.0)
        self.assertLessEqual(drift, 1.0)

    def test_evaluate_feature_drift_marks_alert_feature(self) -> None:
        thresholds = DriftThresholds(
            psi_warn=0.1,
            psi_alert=0.25,
            decision_drift_warn=0.1,
            decision_drift_alert=0.2,
        )

        baseline = pd.DataFrame(
            {
                "Amount": [10, 12, 13, 11, 14, 12, 10, 11],
                "final_risk_score": [0.01, 0.02, 0.03, 0.01, 0.02, 0.02, 0.01, 0.03],
            }
        )
        current = pd.DataFrame(
            {
                "Amount": [200, 250, 300, 400, 450, 500, 550, 600],
                "final_risk_score": [0.8, 0.85, 0.9, 0.95, 0.88, 0.93, 0.91, 0.89],
            }
        )

        results = evaluate_feature_drift(
            baseline_df=baseline,
            current_df=current,
            feature_columns=["Amount", "final_risk_score"],
            bins=4,
            thresholds=thresholds,
        )

        by_feature = {r.feature: r for r in results}
        self.assertEqual(by_feature["Amount"].status, "alert")
        self.assertEqual(by_feature["final_risk_score"].status, "alert")

    def test_evaluate_feature_drift_uses_audit_baseline_for_audit_only_feature(self) -> None:
        thresholds = DriftThresholds(0.1, 0.25, 0.1, 0.2)
        baseline = pd.DataFrame({"Amount": [10, 11, 12, 13]})
        current = pd.DataFrame(
            {
                "base_model_score": [0.90, 0.91, 0.92, 0.93],
                "context_scores.device_risk_score": [0.95, 0.97, 0.96, 0.98],
            }
        )
        audit_baseline = pd.DataFrame(
            {
                "base_model_score": [0.10, 0.11, 0.12, 0.13],
                "context_scores.device_risk_score": [0.05, 0.07, 0.06, 0.08],
            }
        )

        results = evaluate_feature_drift(
            baseline_df=baseline,
            current_df=current,
            feature_columns=["base_model_score", "context_scores.device_risk_score"],
            bins=4,
            thresholds=thresholds,
            audit_baseline_df=audit_baseline,
        )

        by_feature = {r.feature: r for r in results}
        self.assertIn("base_model_score", by_feature)
        self.assertIn("context_scores.device_risk_score", by_feature)
        self.assertEqual(by_feature["base_model_score"].status, "alert")

    def test_split_audit_windows_returns_non_empty_slices(self) -> None:
        audit_df = pd.DataFrame({"final_risk_score": [0.1, 0.2, 0.3, 0.4, 0.5]})
        baseline_window, current_window = split_audit_windows(audit_df, baseline_ratio=0.5)

        self.assertGreater(len(baseline_window), 0)
        self.assertGreater(len(current_window), 0)
        self.assertEqual(len(baseline_window) + len(current_window), len(audit_df))

    def test_recalibration_recommendation_triggers_on_alert(self) -> None:
        thresholds = DriftThresholds(0.1, 0.25, 0.1, 0.2)
        baseline = pd.DataFrame({"Amount": [10, 11, 12, 13], "final_risk_score": [0.01, 0.02, 0.02, 0.03]})
        current = pd.DataFrame({"Amount": [200, 210, 220, 230], "final_risk_score": [0.7, 0.8, 0.9, 0.95]})

        feature_results = evaluate_feature_drift(
            baseline_df=baseline,
            current_df=current,
            feature_columns=["Amount", "final_risk_score"],
            bins=4,
            thresholds=thresholds,
        )

        recommendation = build_recalibration_recommendation(feature_results, decision_drift_status="ok")
        self.assertTrue(recommendation["should_recalibrate"])
        self.assertEqual(recommendation["priority"], "high")

    def test_compute_decision_distribution_handles_missing_column(self) -> None:
        dist = compute_decision_distribution(pd.DataFrame({"x": [1, 2]}), ["APPROVE", "FLAG", "BLOCK"])
        self.assertEqual(dist["APPROVE"], 0.0)

    def test_classify_drift_level_thresholds(self) -> None:
        self.assertEqual(classify_drift_level(0.05, 0.1, 0.2), "ok")
        self.assertEqual(classify_drift_level(0.15, 0.1, 0.2), "warn")
        self.assertEqual(classify_drift_level(0.25, 0.1, 0.2), "alert")


if __name__ == "__main__":
    unittest.main()

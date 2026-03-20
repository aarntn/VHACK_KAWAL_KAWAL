import unittest

import numpy as np
import pandas as pd

from project.scripts.calibrate_context import (
    build_context_component_matrix,
    classify_decisions,
    compute_decision_drift,
    compute_context_adjustment,
    evaluate_candidate,
    validate_threshold_order,
)


class ContextCalibrationScriptTests(unittest.TestCase):
    def test_component_matrix_contains_expected_keys(self) -> None:
        df = pd.DataFrame(
            {
                "Amount": [10, 500],
                "Time": [5, 5000],
                "device_risk_score": [0.1, 0.9],
                "ip_risk_score": [0.2, 0.8],
                "location_risk_score": [0.3, 0.7],
                "device_shared_users_24h": [0, 6],
                "account_age_days": [20, 1],
                "sim_change_recent": [False, True],
                "tx_type": ["MERCHANT", "CASH_OUT"],
                "channel": ["APP", "AGENT"],
                "cash_flow_velocity_1h": [1, 10],
                "p2p_counterparties_24h": [1, 20],
                "is_cross_border": [False, True],
            }
        )

        components = build_context_component_matrix(df)
        self.assertIn("device_risk_weight", components)
        self.assertIn("cashout_over_300", components)
        self.assertEqual(len(components["device_risk_weight"]), 2)

    def test_component_matrix_supports_ieee_columns(self) -> None:
        df = pd.DataFrame(
            {
                "TransactionAmt": [10, 500],
                "TransactionDT": [5, 5000],
                "device_risk_score": [0.1, 0.9],
                "ip_risk_score": [0.2, 0.8],
                "location_risk_score": [0.3, 0.7],
                "device_shared_users_24h": [0, 6],
                "account_age_days": [20, 1],
                "sim_change_recent": [False, True],
                "tx_type": ["MERCHANT", "CASH_OUT"],
                "channel": ["APP", "AGENT"],
                "cash_flow_velocity_1h": [1, 10],
                "p2p_counterparties_24h": [1, 20],
                "is_cross_border": [False, True],
            }
        )

        components = build_context_component_matrix(df, dataset_source="ieee_cis")
        self.assertIn("device_risk_weight", components)
        self.assertIn("amount_over_200", components)

    def test_context_adjustment_is_capped(self) -> None:
        components = {
            "device_risk_weight": np.array([1.0, 1.0]),
            "ip_risk_weight": np.array([1.0, 1.0]),
        }
        weights = {"device_risk_weight": 0.4, "ip_risk_weight": 0.5}
        context = compute_context_adjustment(components, weights, cap=0.3)
        self.assertTrue(np.all(context <= 0.3))

    def test_evaluate_candidate_returns_valid_metrics(self) -> None:
        labels = np.array([0, 0, 1, 1])
        base_scores = np.array([0.1, 0.2, 0.7, 0.95])
        components = {
            "device_risk_weight": np.array([0.0, 0.1, 0.8, 0.9]),
        }
        weights = {"device_risk_weight": 0.1}

        result = evaluate_candidate(
            base_scores=base_scores,
            labels=labels,
            components=components,
            weights=weights,
            cap=0.3,
            approve_threshold=0.3,
            block_threshold=0.8,
            fpr_penalty=0.3,
        )

        self.assertGreaterEqual(result.precision, 0.0)
        self.assertLessEqual(result.precision, 1.0)
        self.assertGreaterEqual(result.recall, 0.0)
        self.assertLessEqual(result.recall, 1.0)

    def test_validate_threshold_order_rejects_invalid_order(self) -> None:
        with self.assertRaises(ValueError):
            validate_threshold_order(approve_threshold=0.7, block_threshold=0.6)

    def test_classify_decisions_three_way(self) -> None:
        decisions = classify_decisions(np.array([0.1, 0.4, 0.95]), approve_threshold=0.3, block_threshold=0.8)
        self.assertListEqual(decisions.tolist(), ["approve", "flag", "block"])

    def test_compute_decision_drift_has_expected_keys(self) -> None:
        pre_scores = np.array([0.1, 0.4, 0.92, 0.2])
        post_scores = np.array([0.2, 0.81, 0.93, 0.5])
        drift = compute_decision_drift(pre_scores, post_scores, approve_threshold=0.3, block_threshold=0.8)
        self.assertIn("pre_approve_rate", drift)
        self.assertIn("post_flag_rate", drift)
        self.assertIn("delta_block_rate", drift)
        self.assertIn("total_absolute_rate_shift", drift)


if __name__ == "__main__":
    unittest.main()

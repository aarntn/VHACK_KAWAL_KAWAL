import unittest

from project.app.domain_exceptions import UnknownTransactionTypeError
from project.app.domain_exceptions import DomainValidationError
from project.app.rules import (
    RULE_POLICY,
    RuleStateStore,
    SegmentThresholds,
    apply_segmented_decision,
    compute_user_segment,
    determine_step_up_action,
    evaluate_hard_rules,
    resolve_thresholds_for_segment,
    validate_segment_thresholds,
)


class RulesModuleTests(unittest.TestCase):
    def test_compute_user_segment(self) -> None:
        segment = compute_user_segment(
            {
                "account_age_days": 5,
                "TransactionAmt": 1500,
                "channel": "agent",
            }
        )
        self.assertEqual(segment, "new:high_ticket:AGENT")

    def test_apply_segmented_decision(self) -> None:
        thresholds = SegmentThresholds(approve_threshold=0.25, block_threshold=0.8)
        self.assertEqual(apply_segmented_decision(0.2, thresholds), "APPROVE")
        self.assertEqual(apply_segmented_decision(0.4, thresholds), "FLAG")
        self.assertEqual(apply_segmented_decision(0.95, thresholds), "BLOCK")

    def test_segment_resolution_and_boundary_behavior(self) -> None:
        segment_cfg = {
            "new:high_ticket:WEB": SegmentThresholds(approve_threshold=0.2, block_threshold=0.7),
            "established:low_ticket:APP": SegmentThresholds(approve_threshold=0.35, block_threshold=0.85),
        }
        web = resolve_thresholds_for_segment("new:high_ticket:WEB", 0.3, 0.9, segment_cfg)
        app = resolve_thresholds_for_segment("established:low_ticket:APP", 0.3, 0.9, segment_cfg)
        defaulted = resolve_thresholds_for_segment("new:low_ticket:QR", 0.3, 0.9, segment_cfg)

        self.assertEqual(apply_segmented_decision(0.19, web), "APPROVE")
        self.assertEqual(apply_segmented_decision(0.2, web), "FLAG")
        self.assertEqual(apply_segmented_decision(0.7, web), "BLOCK")
        self.assertEqual(apply_segmented_decision(0.34, app), "APPROVE")
        self.assertEqual(apply_segmented_decision(0.35, app), "FLAG")
        self.assertEqual(apply_segmented_decision(0.85, app), "BLOCK")
        self.assertEqual(defaulted.approve_threshold, 0.3)
        self.assertEqual(defaulted.block_threshold, 0.9)

    def test_segment_acceptance_gates_reject_failed_calibration_metrics(self) -> None:
        with self.assertRaises(DomainValidationError):
            validate_segment_thresholds(
                SegmentThresholds(
                    approve_threshold=0.2,
                    block_threshold=0.7,
                    min_block_precision=0.9,
                    max_approve_to_flag_fpr=0.03,
                    calibration_metrics={"block_precision": 0.7},
                ),
                "new:high_ticket:WEB",
            )

    def test_evaluate_hard_rules_for_rapid_geo_switch(self) -> None:
        store = RuleStateStore()
        base_tx = {
            "user_id": "u1",
            "TransactionDT": 1000,
            "is_cross_border": False,
            "location_risk_score": 0.1,
            "device_shared_users_24h": 0,
            "cash_flow_velocity_1h": 0,
        }
        second_tx = {
            "user_id": "u1",
            "TransactionDT": 1005,
            "is_cross_border": True,
            "location_risk_score": 0.9,
            "device_shared_users_24h": 0,
            "cash_flow_velocity_1h": 0,
        }
        third_tx = {
            "user_id": "u1",
            "TransactionDT": 1010,
            "is_cross_border": False,
            "location_risk_score": 0.9,
            "device_shared_users_24h": 0,
            "cash_flow_velocity_1h": 0,
        }

        first = evaluate_hard_rules(base_tx, store)
        self.assertEqual(first.action, "ALLOW")

        second = evaluate_hard_rules(second_tx, store)
        self.assertEqual(second.action, "BLOCK")
        self.assertIn("geo_impossible_travel", second.rule_hits)

        third = evaluate_hard_rules(third_tx, store)
        self.assertIn("rapid_device_country_switch", third.rule_hits)

    def test_determine_step_up_action(self) -> None:
        thresholds = SegmentThresholds(approve_threshold=0.3, block_threshold=0.9)
        result = determine_step_up_action(
            final_score=0.88,
            decision="FLAG",
            segment_thresholds=thresholds,
            rule_hits=[],
            tx={},
        )
        self.assertEqual(result.verification_action, "STEP_UP_KYC")
        self.assertEqual(result.reason, "score_near_block_threshold")

    def test_hard_block_amount_multiplier_rule(self) -> None:
        store = RuleStateStore()
        result = evaluate_hard_rules(
            {
                "user_id": "u1",
                "TransactionDT": 1000,
                "TransactionAmt": 401,
                "user_median_amount_30d": 100,
                "is_cross_border": False,
                "location_risk_score": 0.1,
                "device_shared_users_24h": 0,
                "cash_flow_velocity_1h": 0,
            },
            store,
        )
        self.assertEqual(result.action, "BLOCK")
        self.assertIn("amount_over_user_median_multiplier", result.rule_hits)

    def test_step_up_reason_returns_rule_id(self) -> None:
        thresholds = SegmentThresholds(approve_threshold=0.3, block_threshold=0.9)
        result = determine_step_up_action(
            final_score=0.95,
            decision="BLOCK",
            segment_thresholds=thresholds,
            rule_hits=["abuse_burst"],
            tx={},
        )
        self.assertEqual(result.verification_action, "STEP_UP_MANUAL_REVIEW")
        self.assertEqual(result.reason, "abuse_burst")

    def test_rule_policy_loaded_at_startup(self) -> None:
        self.assertEqual(RULE_POLICY.version, "1.0.0")
        self.assertEqual(RULE_POLICY.precedence.conflict_resolution, "most_restrictive_wins")

    def test_compute_user_segment_rejects_unknown_transaction_type(self) -> None:
        with self.assertRaises(UnknownTransactionTypeError):
            compute_user_segment(
                {
                    "account_age_days": 5,
                    "TransactionAmt": 1500,
                    "channel": "agent",
                    "tx_type": "UNKNOWN_KIND",
                }
            )


if __name__ == "__main__":
    unittest.main()

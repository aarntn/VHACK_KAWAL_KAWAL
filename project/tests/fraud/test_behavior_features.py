import unittest

import pandas as pd

from project.data.behavior_features import build_behavior_features, get_uid_candidates_for_source


class BehaviorFeaturesTests(unittest.TestCase):
    def test_ieee_uid_candidates_match_kaggle_identity_mix(self) -> None:
        candidates = get_uid_candidates_for_source("ieee_cis")
        self.assertListEqual(
            candidates,
            ["card1", "card2", "DeviceType", "DeviceInfo", "addr1", "P_emaildomain"],
        )

    def test_ieee_behavior_features_expected_values_for_ordered_data(self) -> None:
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 1000, 2000],
                "TransactionAmt": [10.0, 20.0, 30.0],
                "card1": [1111, 1111, 1111],
            }
        )

        behavior_df, diagnostics = build_behavior_features(df, "ieee_cis")

        self.assertListEqual(behavior_df["tx_count_1h"].astype(int).tolist(), [0, 1, 2])
        self.assertListEqual(behavior_df["tx_count_24h"].astype(int).tolist(), [0, 1, 2])
        self.assertEqual(behavior_df.loc[0, "avg_amount_24h"], 0.0)
        self.assertEqual(behavior_df.loc[1, "avg_amount_24h"], 10.0)
        self.assertEqual(behavior_df.loc[2, "avg_amount_24h"], 15.0)
        self.assertEqual(behavior_df.loc[0, "amount_over_user_avg"], 1.0)
        self.assertEqual(behavior_df.loc[1, "amount_over_user_avg"], 2.0)
        self.assertEqual(behavior_df.loc[2, "amount_over_user_avg"], 2.0)
        self.assertEqual(behavior_df.loc[0, "time_since_last_tx"], 86400.0)
        self.assertEqual(behavior_df.loc[1, "time_since_last_tx"], 1000.0)
        self.assertEqual(behavior_df.loc[2, "time_since_last_tx"], 1000.0)

        self.assertIn("null_rates", diagnostics)
        self.assertIn("source_coverage", diagnostics)

    def test_no_future_leakage_prior_rows_do_not_change(self) -> None:
        base = pd.DataFrame(
            {
                "TransactionDT": [0, 1000, 2000],
                "TransactionAmt": [10.0, 20.0, 30.0],
                "card1": [1111, 1111, 1111],
            }
        )
        mutated = base.copy()
        mutated.loc[2, "TransactionAmt"] = 3000.0

        base_features, _ = build_behavior_features(base, "ieee_cis")
        mutated_features, _ = build_behavior_features(mutated, "ieee_cis")

        for col in ["tx_count_1h", "tx_count_24h", "avg_amount_24h", "amount_over_user_avg", "time_since_last_tx"]:
            self.assertAlmostEqual(base_features.loc[0, col], mutated_features.loc[0, col], places=9)
            self.assertAlmostEqual(base_features.loc[1, col], mutated_features.loc[1, col], places=9)

    def test_uid_aggregate_features_are_present_and_deterministic(self) -> None:
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 1000, 2000, 90000],
                "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
                "card1": [1111, 1111, 1111, 1111],
                "addr1": [100, 100, 100, 100],
            }
        )
        a, _ = build_behavior_features(df, "ieee_cis")
        b, _ = build_behavior_features(df, "ieee_cis")

        for col in ["uid_tx_count_7d", "uid_avg_amount_7d", "uid_amount_std_7d", "uid_time_since_last_tx"]:
            self.assertIn(col, a.columns)
            self.assertListEqual(a[col].tolist(), b[col].tolist())

    def test_uid_features_no_future_leakage(self) -> None:
        base = pd.DataFrame(
            {
                "TransactionDT": [0, 1000, 2000],
                "TransactionAmt": [10.0, 20.0, 30.0],
                "card1": [1111, 1111, 1111],
                "addr1": [100, 100, 100],
            }
        )
        mutated = base.copy()
        mutated.loc[2, "TransactionAmt"] = 9999.0

        base_features, _ = build_behavior_features(base, "ieee_cis")
        mutated_features, _ = build_behavior_features(mutated, "ieee_cis")

        for col in ["uid_tx_count_7d", "uid_avg_amount_7d", "uid_amount_std_7d", "uid_time_since_last_tx"]:
            self.assertAlmostEqual(base_features.loc[0, col], mutated_features.loc[0, col], places=9)
            self.assertAlmostEqual(base_features.loc[1, col], mutated_features.loc[1, col], places=9)


if __name__ == "__main__":
    unittest.main()

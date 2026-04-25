import unittest

import pandas as pd

from project.data.preprocessing import fit_preprocessing_bundle, prepare_preprocessing_inputs
from project.data.feature_registry import load_registry_config, map_to_canonical_features


class PreprocessingTests(unittest.TestCase):
    def test_feature_registry_includes_ieee_source_and_preprocessing_uses_explicit_routing(self) -> None:
        config = load_registry_config()
        self.assertIn("ieee_cis", config["sources"])
        self.assertEqual(config["sources"]["ieee_cis"]["amount_column"], "TransactionAmt")
        self.assertEqual(config["sources"]["ieee_cis"]["event_time_column"], "TransactionDT")
        self.assertEqual(config["sources"]["ieee_cis"]["required_columns"], ["TransactionAmt", "TransactionDT"])

        with self.assertRaises(ValueError):
            map_to_canonical_features(pd.DataFrame({"TransactionAmt": [1.0]}), "unknown_source")

    def test_prepare_preprocessing_inputs_deduplicates_overlap_columns(self) -> None:
        features = pd.DataFrame(
            {
                "Time": [0, 1000, 2000],
                "Amount": [10.0, 20.0, 30.0],
                "V1": [0.1, 0.2, 0.3],
                "uid_tx_count_7d": [100.0, 200.0, 300.0],
            }
        )

        canonical_df, passthrough_df, diagnostics = prepare_preprocessing_inputs(features, "creditcard")

        self.assertEqual(canonical_df.columns.tolist().count("uid_tx_count_7d"), 1)
        self.assertIn("uid_tx_count_7d", diagnostics["duplicate_canonical_behavior_columns_dropped"])

        bundle, transformed = fit_preprocessing_bundle(
            canonical_df=canonical_df,
            passthrough_df=passthrough_df,
            dataset_source="creditcard",
            include_passthrough=False,
            scaler="standard",
            categorical_encoding="onehot",
            behavior_feature_diagnostics=diagnostics,
        )

        self.assertEqual(transformed.shape[0], 3)
        self.assertEqual(len(bundle.feature_names_out), transformed.shape[1])

    def test_ieee_routing_does_not_silently_fallback_to_creditcard_columns(self) -> None:
        # Intentional credit-card style columns only.
        features = pd.DataFrame(
            {
                "Time": [100.0, 200.0],
                "Amount": [10.0, 15.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "requires columns"):
            prepare_preprocessing_inputs(features, "ieee_cis")

    def test_ieee_mapping_prefers_transaction_fields_over_legacy_aliases(self) -> None:
        features = pd.DataFrame(
            {
                "TransactionDT": [900.0, 1800.0],
                "TransactionAmt": [70.0, 80.0],
                "Time": [100.0, 200.0],
                "Amount": [10.0, 15.0],
            }
        )

        canonical_df, _, _ = prepare_preprocessing_inputs(features, "ieee_cis")
        self.assertListEqual(canonical_df["amount_raw"].tolist(), [70.0, 80.0])
        self.assertListEqual(canonical_df["event_time_raw"].tolist(), [900.0, 1800.0])


if __name__ == "__main__":
    unittest.main()

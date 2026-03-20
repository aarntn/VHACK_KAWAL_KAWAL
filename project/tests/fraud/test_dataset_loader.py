import tempfile
import unittest
from pathlib import Path

import pandas as pd

from project.data.dataset_loader import apply_label_policy, enforce_eda_gate, load_creditcard, load_ieee_cis


class DatasetLoaderTests(unittest.TestCase):
    def test_load_creditcard_missing_file_includes_debug_context(self) -> None:
        missing_path = Path("this/path/does/not/exist/creditcard.csv")

        with self.assertRaises(FileNotFoundError) as ctx:
            load_creditcard(missing_path)

        message = str(ctx.exception)
        self.assertIn("Dataset file not found", message)
        self.assertIn("Current working directory", message)

    def test_load_ieee_cis_missing_transaction_file_includes_debug_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            identity_path = Path(temp_dir) / "train_identity.csv"
            identity_path.write_text("TransactionID\nid_1\n", encoding="utf-8")

            with self.assertRaises(FileNotFoundError) as ctx:
                load_ieee_cis("missing_transaction.csv", identity_path)

        message = str(ctx.exception)
        self.assertIn("Transaction file not found", message)
        self.assertIn("Current working directory", message)

    def test_apply_label_policy_account_propagated_forward_only(self) -> None:
        features = pd.DataFrame(
            {
                "Time": [1, 2, 3, 4],
                "card1": [1111, 1111, 1111, 1111],
                "card2": [222, 222, 222, 222],
            },
            index=[10, 11, 12, 13],
        )
        labels = pd.Series([0, 1, 0, 0], index=features.index)

        out, diag = apply_label_policy(features, labels, dataset_source="ieee_cis", label_policy="account_propagated")

        self.assertListEqual(out.tolist(), [0, 1, 1, 1])
        self.assertEqual(diag["rows_changed"], 2)
        self.assertEqual(diag["positive_before"], 1)
        self.assertEqual(diag["positive_after"], 3)

    def test_apply_label_policy_transaction_no_change(self) -> None:
        features = pd.DataFrame({"Time": [1, 2, 3]})
        labels = pd.Series([0, 1, 0])
        out, diag = apply_label_policy(features, labels, dataset_source="creditcard", label_policy="transaction")
        self.assertListEqual(out.tolist(), [0, 1, 0])
        self.assertEqual(diag["rows_changed"], 0)

    def test_load_creditcard_exposes_behavior_features_for_both_label_policies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "creditcard.csv"
            pd.DataFrame(
                {
                    "Time": [0, 1000, 2000],
                    "Amount": [10.0, 20.0, 30.0],
                    "V1": [0.1, 0.2, 0.3],
                    "Class": [0, 1, 0],
                }
            ).to_csv(dataset_path, index=False)

            x_tx, y_tx, meta_tx = load_creditcard(dataset_path, label_policy="transaction")
            x_ap, y_ap, meta_ap = load_creditcard(dataset_path, label_policy="account_propagated")

            for col in ["uid_tx_count_7d", "uid_avg_amount_7d", "uid_amount_std_7d", "uid_time_since_last_tx"]:
                self.assertIn(col, x_tx.columns)
                self.assertIn(col, x_ap.columns)
            self.assertEqual(meta_tx["label_policy"], "transaction")
            self.assertEqual(meta_ap["label_policy"], "account_propagated")
            self.assertEqual(len(y_tx), len(y_ap))

    def test_ieee_loader_includes_eda_gate_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tx_path = Path(temp_dir) / "train_transaction.csv"
            id_path = Path(temp_dir) / "train_identity.csv"

            pd.DataFrame(
                {
                    "TransactionID": [1, 2, 3],
                    "TransactionDT": [10, 20, 30],
                    "card1": [1111, 1111, 2222],
                    "isFraud": [0, 1, 0],
                }
            ).to_csv(tx_path, index=False)
            pd.DataFrame({"TransactionID": [1, 2]}).to_csv(id_path, index=False)

            _, _, meta = load_ieee_cis(tx_path, id_path)

            self.assertIn("eda_gate", meta)
            self.assertTrue(meta["eda_gate"]["passed"])
            self.assertIn("join_row_counts", meta["eda_gate"]["checks"])

    def test_enforce_eda_gate_raises_when_blocking_enabled(self) -> None:
        metadata = {"eda_gate": {"passed": False, "failed_checks": ["target_prevalence"]}}
        with self.assertRaises(RuntimeError):
            enforce_eda_gate(metadata, allow_failures=False, context="unit-test")

        enforce_eda_gate(metadata, allow_failures=True, context="unit-test")


if __name__ == "__main__":
    unittest.main()

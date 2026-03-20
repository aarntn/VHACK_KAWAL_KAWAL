import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from project.scripts import calibrate_context


class CalibrateContextScriptTests(unittest.TestCase):
    def test_load_source_creditcard_branch(self) -> None:
        args = argparse.Namespace(
            dataset_source="creditcard",
            dataset_path=Path("project/legacy_creditcard/creditcard.csv"),
            ieee_transaction_path=None,
            ieee_identity_path=None,
        )
        with patch.object(calibrate_context, "load_creditcard", return_value=(pd.DataFrame(), pd.Series(dtype=int), {})) as mocked:
            calibrate_context.load_source(args)
        mocked.assert_called_once_with(args.dataset_path)

    def test_load_source_ieee_requires_both_paths(self) -> None:
        args = argparse.Namespace(
            dataset_source="ieee_cis",
            dataset_path=Path("project/legacy_creditcard/creditcard.csv"),
            ieee_transaction_path=None,
            ieee_identity_path=None,
        )
        with self.assertRaisesRegex(ValueError, "--ieee-transaction-path and --ieee-identity-path are required"):
            calibrate_context.load_source(args)

    def test_load_source_ieee_branch(self) -> None:
        args = argparse.Namespace(
            dataset_source="ieee_cis",
            dataset_path=Path("project/legacy_creditcard/creditcard.csv"),
            ieee_transaction_path=Path("tx.csv"),
            ieee_identity_path=Path("id.csv"),
        )
        with patch.object(calibrate_context, "load_ieee_cis", return_value=(pd.DataFrame(), pd.Series(dtype=int), {})) as mocked:
            calibrate_context.load_source(args)
        mocked.assert_called_once_with(args.ieee_transaction_path, args.ieee_identity_path)

    def test_resolve_amount_and_time_columns_ieee(self) -> None:
        df = pd.DataFrame({"TransactionAmt": [1.0], "TransactionDT": [10]})
        amount, event_time = calibrate_context.resolve_amount_and_time_columns(df)
        self.assertEqual(float(amount.iloc[0]), 1.0)
        self.assertEqual(float(event_time.iloc[0]), 10.0)

    def test_build_model_input_raw_feature_path(self) -> None:
        df = pd.DataFrame({"TransactionDT": [1.0], "TransactionAmt": [5.0]})
        args = argparse.Namespace(dataset_source="ieee_cis", preprocessing_artifact_path=Path("bundle.pkl"))
        out = calibrate_context.build_model_input(df, ["TransactionDT", "TransactionAmt"], args)
        self.assertListEqual(out.columns.tolist(), ["TransactionDT", "TransactionAmt"])

    def test_build_model_input_preprocessed_feature_path_uses_bundle(self) -> None:
        df = pd.DataFrame({"TransactionDT": [1.0], "TransactionAmt": [5.0]})
        args = argparse.Namespace(dataset_source="ieee_cis", preprocessing_artifact_path=Path("bundle.pkl"))

        bundle = type("Bundle", (), {"feature_names_out": ["numeric_canonical__amount_raw", "numeric_canonical__event_time_raw"]})()
        transformed = np.array([[0.1, 0.2]], dtype=float)

        with patch.object(calibrate_context, "load_preprocessing_bundle", return_value=bundle), \
            patch.object(calibrate_context, "prepare_preprocessing_inputs", return_value=(pd.DataFrame(), pd.DataFrame(), {})), \
            patch.object(calibrate_context, "transform_with_bundle", return_value=transformed):
            out = calibrate_context.build_model_input(
                df,
                ["numeric_canonical__amount_raw", "numeric_canonical__event_time_raw"],
                args,
            )

        self.assertListEqual(out.columns.tolist(), ["numeric_canonical__amount_raw", "numeric_canonical__event_time_raw"])


if __name__ == "__main__":
    unittest.main()

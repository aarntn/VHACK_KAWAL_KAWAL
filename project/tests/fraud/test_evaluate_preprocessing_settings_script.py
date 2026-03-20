import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from project.scripts import evaluate_preprocessing_settings
from project.scripts.evaluate_preprocessing_settings import (
    feature_whitelist_hash,
    groupkfold_uid_split,
    load_feature_whitelist,
    orient_scores_for_positive_class,
    time_based_split,
    to_markdown,
)


class EvaluatePreprocessingSettingsScriptTests(unittest.TestCase):
    def test_time_based_split_orders_by_event_time(self) -> None:
        x = pd.DataFrame({"a": [1, 2, 3, 4]}, index=[10, 11, 12, 13])
        y = pd.Series([0, 1, 0, 1], index=x.index)
        event_time = pd.Series([300, 100, 400, 200], index=x.index)

        x_train, x_test, y_train, y_test = time_based_split(x, y, event_time, test_size=0.5)

        self.assertEqual(list(x_train.index), [11, 13])
        self.assertEqual(list(x_test.index), [10, 12])
        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(y_test), 2)

    def test_to_markdown_contains_expected_headers(self) -> None:
        df = pd.DataFrame(
            {
                "preprocessing_setting": ["onehot_robust"],
                "threshold": [0.4],
                "f1": [0.8],
                "recall": [0.75],
                "precision": [0.85],
                "false_positive_rate": [0.02],
                "pr_auc": [0.81],
                "roc_auc": [0.9],
                "tn": [100],
                "fp": [2],
                "fn": [5],
                "tp": [15],
            }
        )
        md = to_markdown(df)
        self.assertIn("| setting | threshold | f1 | recall | precision |", md)
        self.assertIn("onehot_robust", md)

    def test_load_feature_whitelist_defaults_to_all_columns_when_off(self) -> None:
        cols = ["f1", "f2", "f3"]
        selected, source = load_feature_whitelist(None, None, cols)
        self.assertEqual(selected, cols)
        self.assertIsNone(source)

    def test_load_feature_whitelist_from_decisions_json(self) -> None:
        cols = ["Time", "Amount", "V1"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "time_consistency_feature_decisions.json"
            path.write_text(
                """
{
  "summary": {
    "kept_single_features": ["Amount", "missing_col", "V1"]
  }
}
""".strip(),
                encoding="utf-8",
            )
            selected, source = load_feature_whitelist(None, path, cols)

        self.assertEqual(selected, ["Amount", "V1"])
        self.assertEqual(source, "feature_decisions_json")

    def test_load_feature_whitelist_from_file_overrides_decisions(self) -> None:
        cols = ["Time", "Amount", "V1"]
        with tempfile.TemporaryDirectory() as tmp:
            decisions = Path(tmp) / "time_consistency_feature_decisions.json"
            decisions.write_text('{"summary": {"kept_single_features": ["Time"]}}', encoding="utf-8")
            whitelist = Path(tmp) / "feature_whitelist.txt"
            whitelist.write_text("# comment\nV1\nAmount\nunknown\n", encoding="utf-8")

            selected, source = load_feature_whitelist(whitelist, decisions, cols)

        self.assertEqual(selected, ["V1", "Amount"])
        self.assertEqual(source, "feature_whitelist_file")

    def test_load_feature_whitelist_raises_when_no_selected_columns(self) -> None:
        cols = ["Time", "Amount"]
        with tempfile.TemporaryDirectory() as tmp:
            whitelist = Path(tmp) / "feature_whitelist.txt"
            whitelist.write_text("missing1\nmissing2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "removed all columns"):
                load_feature_whitelist(whitelist, None, cols)

    def test_feature_whitelist_hash(self) -> None:
        self.assertIsNone(feature_whitelist_hash(["a", "b"], active=False))
        h1 = feature_whitelist_hash(["b", "a"], active=True)
        h2 = feature_whitelist_hash(["a", "b"], active=True)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1 or ""), 64)

    def test_groupkfold_uid_split_returns_non_empty_partitions(self) -> None:
        x = pd.DataFrame(
            {
                "TransactionDT": [0, 1, 2, 3, 4, 5],
                "card1": [1, 1, 2, 2, 3, 3],
                "addr1": [10, 10, 20, 20, 30, 30],
                "f": [0, 1, 0, 1, 0, 1],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])
        x_train, x_test, y_train, y_test = groupkfold_uid_split(x, y, dataset_source="ieee_cis", n_splits=3)
        self.assertGreater(len(x_train), 0)
        self.assertGreater(len(x_test), 0)
        self.assertGreaterEqual(y_train.nunique(), 2)
        self.assertGreaterEqual(y_test.nunique(), 2)

    def test_main_invokes_enforce_eda_gate_before_training(self) -> None:
        fake_x = pd.DataFrame({"TransactionDT": [0, 1, 2, 3], "f": [0.1, 0.2, 0.3, 0.4]})
        fake_y = pd.Series([0, 1, 0, 1])
        fake_meta = {"eda_gate": {"passed": True}}

        args = evaluate_preprocessing_settings.argparse.Namespace(
            dataset_source="ieee_cis",
            label_policy="transaction",
            dataset_path=None,
            ieee_transaction_path=Path("tx.csv"),
            ieee_identity_path=Path("id.csv"),
            split_mode="time",
            test_size=0.5,
            groupkfold_n_splits=2,
            seed=42,
            threshold_start=0.4,
            threshold_stop=0.5,
            threshold_step=0.1,
            n_estimators=10,
            max_depth=2,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            feature_decisions_json=None,
            feature_whitelist_file=None,
            output_dir=Path(tempfile.mkdtemp()),
            allow_eda_failures=False,
        )

        with patch.object(evaluate_preprocessing_settings, "parse_args", return_value=args), \
            patch.object(evaluate_preprocessing_settings, "load_source", return_value=(fake_x, fake_y, fake_meta)), \
            patch.object(evaluate_preprocessing_settings, "enforce_eda_gate") as mocked_gate:
            evaluate_preprocessing_settings.main()

        mocked_gate.assert_called_once()

    def test_orient_scores_for_positive_class_inverts_when_auc_below_half(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])

        oriented, meta = orient_scores_for_positive_class(y_true, y_score)

        self.assertTrue(meta["score_inversion_applied"])
        self.assertEqual(meta["orientation"], "inverted")
        self.assertGreater(float(meta["roc_auc_oriented"]), float(meta["roc_auc_raw"]))
        np.testing.assert_allclose(oriented, 1.0 - y_score)

    def test_orient_scores_for_positive_class_keeps_scores_when_auc_good(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])

        oriented, meta = orient_scores_for_positive_class(y_true, y_score)

        self.assertFalse(meta["score_inversion_applied"])
        self.assertEqual(meta["orientation"], "identity")
        np.testing.assert_allclose(oriented, y_score)


class FrequencyEncoderTests(unittest.TestCase):
    def test_frequency_encoder_handles_integer_column_keys(self) -> None:
        from project.data.preprocessing import FrequencyEncoder

        # Mimic upstream SimpleImputer output where columns are integer positions.
        x = pd.DataFrame({0: ["a", "a", "b"], 1: ["x", "y", "x"]})

        enc = FrequencyEncoder()
        enc.fit(x)
        transformed = enc.transform(x)

        self.assertEqual(transformed.shape, (3, 2))
        feature_names = enc.get_feature_names_out().tolist()
        self.assertEqual(feature_names, ["0__freq", "1__freq"])
        # First row frequencies: a=2/3, x=2/3
        self.assertAlmostEqual(float(transformed[0, 0]), 2 / 3, places=6)
        self.assertAlmostEqual(float(transformed[0, 1]), 2 / 3, places=6)


if __name__ == "__main__":
    unittest.main()

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from project.scripts import cohort_kpi_report


class CohortKPIReportScriptTests(unittest.TestCase):
    def test_ensure_context_columns_adds_missing(self) -> None:
        df = pd.DataFrame({"Time": [0, 10], "Amount": [1.0, 2.0]})
        out = cohort_kpi_report.ensure_context_columns(df)
        self.assertIn("account_age_days", out.columns)
        self.assertIn("channel", out.columns)
        self.assertIn("tx_type", out.columns)

    def test_ensure_context_columns_supports_ieee_columns(self) -> None:
        df = pd.DataFrame({"TransactionDT": [1, 2, 3], "TransactionAmt": [10.0, 20.0, 30.0]})
        out = cohort_kpi_report.ensure_context_columns(df)
        self.assertIn("account_age_days", out.columns)
        self.assertIn("channel", out.columns)
        self.assertIn("tx_type", out.columns)

    def test_assign_cohorts_supports_ieee_amount_column(self) -> None:
        df = cohort_kpi_report.ensure_context_columns(pd.DataFrame({"TransactionDT": [1.0], "TransactionAmt": [10.0]}))
        cohorts = cohort_kpi_report.assign_cohorts(df)
        self.assertIn("all_users", cohorts)
        self.assertIn("new_users", cohorts)
        self.assertIn("rural_merchants_proxy", cohorts)

    def test_assign_cohorts_has_expected_keys(self) -> None:
        df = cohort_kpi_report.ensure_context_columns(pd.DataFrame({"Time": [1.0], "Amount": [10.0]}))
        cohorts = cohort_kpi_report.assign_cohorts(df)
        self.assertIn("all_users", cohorts)
        self.assertIn("new_users", cohorts)
        self.assertIn("rural_merchants_proxy", cohorts)
        self.assertIn("gig_workers_proxy", cohorts)
        self.assertIn("low_history_proxy", cohorts)

    def test_assign_cohorts_from_config_uses_versioned_definitions(self) -> None:
        df = cohort_kpi_report.ensure_context_columns(pd.DataFrame({"Time": [1.0, 5.0], "Amount": [10.0, 300.0]}))
        config = {
            "version": "test-v1",
            "cohorts": [
                {"name": "all_users", "all": []},
                {"name": "small_amount", "all": [{"column": "__amount__", "op": "<", "value": 100}]},
            ],
        }
        cohorts = cohort_kpi_report.assign_cohorts_from_config(df, config)
        self.assertEqual(int(cohorts["all_users"].sum()), 2)
        self.assertEqual(int(cohorts["small_amount"].sum()), 1)

    def test_assign_cohorts_from_config_numeric_comparison_handles_object_amount(self) -> None:
        df = cohort_kpi_report.ensure_context_columns(pd.DataFrame({"Time": [1.0, 5.0], "Amount": ["10.0", "bad"]}))
        config = {
            "version": "test-v1",
            "cohorts": [
                {"name": "amount_under_250", "all": [{"column": "__amount__", "op": "<=", "value": 250}]},
            ],
        }
        cohorts = cohort_kpi_report.assign_cohorts_from_config(df, config)
        self.assertEqual(cohorts["amount_under_250"].tolist(), [True, False])

    def test_load_cohort_config_from_json_and_hash_metadata(self) -> None:
        config = {"version": "v-test", "cohorts": [{"name": "all_users", "all": []}]}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "cohorts.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            loaded = cohort_kpi_report.load_cohort_config(path)
            meta = cohort_kpi_report.cohort_config_metadata(loaded, path)
        self.assertEqual(loaded["version"], "v-test")
        self.assertEqual(meta["version"], "v-test")
        self.assertTrue(meta["definition_hash_sha256"])

    def test_metric_row_empty_mask(self) -> None:
        mask = pd.Series([False, False])
        row = cohort_kpi_report.metric_row(
            "test",
            mask,
            labels=[0, 1],
            pred_block=[False, True],
            pred_flag=[False, False],
            min_positive_support=5,
        )
        self.assertEqual(row["sample_count"], 0)
        self.assertEqual(row["precision"], 0.0)
        self.assertEqual(row["fraud_positive_count"], 0)
        self.assertEqual(row["nonfraud_count"], 0)
        self.assertEqual(row["metric_reliability"], "low_support")
        self.assertEqual(row["support_weighted_reliability"], 0.0)

    def test_detect_feature_columns_shape_identifies_preprocessed_columns(self) -> None:
        shape = cohort_kpi_report.detect_feature_columns_shape(["numeric_canonical__amount_raw", "numeric_canonical__amount_log"])
        self.assertEqual(shape, "preprocessed")

    def test_build_model_input_supports_preprocessed_feature_columns(self) -> None:
        df = pd.DataFrame({"TransactionAmt": [10.0, 20.0], "TransactionDT": [1, 2]})
        feature_columns = ["numeric_canonical__amount_raw", "numeric_canonical__amount_log"]
        args = argparse.Namespace(preprocessing_artifact_path=Path("project/models/preprocessing_artifact_promoted.pkl"))

        bundle = type("Bundle", (), {"feature_names_out": feature_columns})()
        transformed = np.array([[1.0, 2.0], [3.0, 4.0]])

        with patch("project.scripts.cohort_kpi_report.load_preprocessing_bundle", return_value=bundle), \
             patch("project.scripts.cohort_kpi_report.prepare_preprocessing_inputs", return_value=(pd.DataFrame(), pd.DataFrame(), {})), \
             patch("project.scripts.cohort_kpi_report.transform_with_bundle", return_value=transformed):
            result = cohort_kpi_report.build_model_input(df, feature_columns, args)

        self.assertEqual(list(result.columns), feature_columns)
        self.assertEqual(result.to_numpy().tolist(), transformed.tolist())

    def test_reliability_and_warnings_for_low_support(self) -> None:
        row = {
            "cohort": "new_users",
            "sample_count": 20,
            "fraud_positive_count": 2,
            "nonfraud_count": 18,
            "metric_reliability": "low_support",
            "fraud_rate": 0.1,
            "precision": 0.0,
            "recall": 0.0,
            "false_positive_rate": 0.0,
            "flag_rate": 0.0,
            "block_rate": 0.0,
        }
        warnings = cohort_kpi_report.build_low_support_warnings([row], min_positive_support=5)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["cohort"], "new_users")
        self.assertEqual(warnings[0]["fraud_positive_count"], 2)

    def test_select_evaluation_frame_defaults_to_time_aware_recent_rows(self) -> None:
        df = pd.DataFrame({"Time": [1, 2, 3, 4], "Amount": [1.0, 2.0, 3.0, 4.0]})
        out = cohort_kpi_report.select_evaluation_frame(df, sample_size=2, seed=42, mode="time_aware")
        self.assertEqual(out["Time"].tolist(), [3, 4])

    def test_script_runs_help_when_invoked_directly(self) -> None:
        result = subprocess.run(
            ["python", "project/scripts/cohort_kpi_report.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--preprocessing-artifact-path", result.stdout)

    def test_resolve_dataset_path_maps_tmp_style_to_windows_tempdir_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset = tmp_path / "mock_creditcard.csv"
            dataset.write_text("Time,Amount,Class\n0,1.0,0\n", encoding="utf-8")

            with patch("project.scripts.cohort_kpi_report.tempfile.gettempdir", return_value=str(tmp_path)):
                resolved = cohort_kpi_report.resolve_dataset_path(Path("/tmp/mock_creditcard.csv"))

            self.assertEqual(resolved, dataset)

    def test_resolve_dataset_path_raises_with_windows_hint_for_tmp_style(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            cohort_kpi_report.resolve_dataset_path(Path("/tmp/does_not_exist.csv"))

        self.assertIn("hint: on Windows", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from project.scripts.fairness_explainability_report import (
    assign_governance_segments,
    assign_identity_buckets,
    build_markdown_report,
    build_model_input,
    compute_identity_bucket_metrics,
    sample_frame_and_labels,
    compute_segment_metrics,
    detect_feature_columns_shape,
    detect_disparities,
    ensure_segment_columns,
    load_source,
)
from project.scripts.fairness_segment_decision import rank_worst_segments


class FairnessExplainabilityReportScriptTests(unittest.TestCase):

    def test_load_source_creditcard_branch(self) -> None:
        args = argparse.Namespace(
            dataset_source="creditcard",
            dataset_path=Path("project/legacy_creditcard/creditcard.csv"),
            ieee_transaction_path=None,
            ieee_identity_path=None,
        )
        with patch("project.scripts.fairness_explainability_report.load_creditcard", return_value=(pd.DataFrame(), pd.Series(dtype=int), {})) as mocked:
            load_source(args)
        mocked.assert_called_once_with(args.dataset_path)

    def test_load_source_ieee_requires_paths(self) -> None:
        args = argparse.Namespace(
            dataset_source="ieee_cis",
            dataset_path=Path("project/legacy_creditcard/creditcard.csv"),
            ieee_transaction_path=None,
            ieee_identity_path=None,
        )
        with self.assertRaisesRegex(ValueError, "--ieee-transaction-path and --ieee-identity-path are required"):
            load_source(args)

    def test_ensure_segment_columns_adds_defaults(self) -> None:
        df = pd.DataFrame({"Time": [1, 2, 3], "Amount": [10.0, 200.0, 400.0]})
        out = ensure_segment_columns(df)
        self.assertIn("account_age_days", out.columns)
        self.assertIn("channel", out.columns)
        self.assertIn("region", out.columns)
        self.assertIn("device_segment", out.columns)

    def test_ensure_segment_columns_supports_ieee_time_amount(self) -> None:
        df = pd.DataFrame({"TransactionDT": [1, 2, 3], "TransactionAmt": [10.0, 20.0, 30.0]})
        out = ensure_segment_columns(df)
        self.assertIn("account_age_days", out.columns)
        self.assertIn("channel", out.columns)

    def test_detect_feature_columns_shape_identifies_preprocessed_columns(self) -> None:
        shape = detect_feature_columns_shape(["numeric_canonical__amount_raw", "numeric_canonical__amount_log"])
        self.assertEqual(shape, "preprocessed")

    def test_sample_frame_and_labels_keeps_lengths_and_alignment(self) -> None:
        df = pd.DataFrame({"TransactionAmt": [10.0, 20.0, 30.0, 40.0]}, index=[10, 11, 12, 13])
        labels = pd.Series([0, 1, 0, 1], index=[10, 11, 12, 13])

        sampled_df, sampled_labels = sample_frame_and_labels(df, labels, sample_size=2, seed=7)

        self.assertEqual(len(sampled_df), 2)
        self.assertEqual(len(sampled_labels), 2)
        self.assertEqual(list(sampled_df.index), [0, 1])
        self.assertEqual(list(sampled_labels.index), [0, 1])
        self.assertEqual(sampled_labels.tolist(), labels.loc[df.sample(n=2, random_state=7).index].tolist())

    def test_build_model_input_supports_preprocessed_feature_columns(self) -> None:
        df = pd.DataFrame({"TransactionAmt": [10.0, 20.0], "TransactionDT": [1, 2]})
        feature_columns = ["numeric_canonical__amount_raw", "numeric_canonical__amount_log"]
        args = argparse.Namespace(dataset_source="ieee_cis", preprocessing_artifact_path=Path("project/models/preprocessing_artifact_promoted.pkl"))

        bundle = type("Bundle", (), {"feature_names_out": feature_columns})()
        transformed = np.array([[1.0, 2.0], [3.0, 4.0]])

        with patch("project.scripts.fairness_explainability_report.load_preprocessing_bundle", return_value=bundle), \
             patch("project.scripts.fairness_explainability_report.prepare_preprocessing_inputs", return_value=(pd.DataFrame(), pd.DataFrame(), {})), \
             patch("project.scripts.fairness_explainability_report.transform_with_bundle", return_value=transformed):
            result = build_model_input(df, feature_columns, args)

        self.assertEqual(list(result.columns), feature_columns)
        self.assertEqual(result.to_numpy().tolist(), transformed.tolist())

    def test_compute_segment_metrics_contains_gap_fields(self) -> None:
        labels = np.array([0, 0, 1, 1])
        pred_block = np.array([False, True, True, False])
        segments = {
            "s1": pd.Series([True, True, False, False]),
            "s2": pd.Series([False, False, True, True]),
        }

        rows, overall = compute_segment_metrics(labels, pred_block, segments, min_segment_size=1)

        self.assertEqual(len(rows), 2)
        self.assertIn("precision", overall)
        self.assertIn("fpr_gap_vs_overall", rows[0])
        self.assertIn("fnr_gap_vs_overall", rows[0])
        self.assertIn("false_negative_rate", rows[0])
        self.assertIn("recall_gap_vs_overall", rows[0])
        self.assertIn("precision_gap_vs_overall", rows[0])


    def test_assign_governance_segments_adds_ieee_specific_cohorts(self) -> None:
        df = pd.DataFrame({
            "TransactionDT": [1, 2, 3, 4],
            "Time": [1, 2, 3, 4],
            "Amount": [10.0, 20.0, 30.0, 40.0],
            "DeviceType": ["mobile", "desktop", "mobile", np.nan],
            "ProductCD": ["W", "H", "W", "C"],
            "card1": [1000, 1001, np.nan, 1003],
            "addr1": [200, np.nan, 202, np.nan],
            "P_emaildomain": ["gmail.com", "yahoo.com", np.nan, np.nan],
        })
        with_segments = ensure_segment_columns(df)
        segments = assign_governance_segments(with_segments)

        self.assertIn("ieee:device_mobile", segments)
        self.assertIn("ieee:device_desktop", segments)
        self.assertIn("ieee:product_W", segments)
        self.assertIn("ieee:product_H", segments)
        self.assertIn("ieee:identity_high_confidence", segments)
        self.assertIn("ieee:identity_medium_confidence", segments)
        self.assertIn("ieee:identity_low_confidence", segments)

        self.assertEqual(int(segments["ieee:device_mobile"].sum()), 2)
        self.assertEqual(int(segments["ieee:identity_high_confidence"].sum()), 1)
        self.assertEqual(int(segments["ieee:identity_medium_confidence"].sum()), 2)
        self.assertEqual(int(segments["ieee:identity_low_confidence"].sum()), 2)

    def test_detect_disparities_marks_zero_positive_segment_as_no_positive_labels(self) -> None:
        rows = [
            {
                "segment": "ieee:device_mobile",
                "sample_count": 300,
                "fraud_positive_count": 0,
                "is_low_support": False,
                "fpr_gap_vs_overall": 0.20,
                "fnr_gap_vs_overall": 1.00,
                "precision_gap_vs_overall": 1.00,
            }
        ]

        severe = detect_disparities(rows, max_fpr_gap=0.08, max_recall_gap=0.12, max_precision_gap=0.12)

        self.assertEqual(severe, [])
        self.assertEqual(rows[0]["severity_label"], "no_positive_labels")

    def test_detect_disparities_flags_severe_segment(self) -> None:
        rows = [
            {
                "segment": "region:cross_border",
                "sample_count": 1000,
                "fraud_positive_count": 20,
                "is_low_support": False,
                "fpr_gap_vs_overall": 0.15,
                "fnr_gap_vs_overall": 0.01,
                "precision_gap_vs_overall": 0.01,
            },
            {
                "segment": "device:agent",
                "sample_count": 1000,
                "fraud_positive_count": 20,
                "is_low_support": False,
                "fpr_gap_vs_overall": 0.20,
                "fnr_gap_vs_overall": 0.01,
                "precision_gap_vs_overall": 0.01,
            },
            {
                "segment": "device:mobile_app",
                "sample_count": 1000,
                "fraud_positive_count": 20,
                "is_low_support": False,
                "fpr_gap_vs_overall": 0.01,
                "fnr_gap_vs_overall": 0.01,
                "precision_gap_vs_overall": 0.01,
            },
        ]

        severe = detect_disparities(rows, max_fpr_gap=0.08, max_recall_gap=0.12, max_precision_gap=0.12)
        self.assertEqual(len(severe), 2)
        self.assertEqual(severe[0]["segment"], "device:agent")
        self.assertEqual(severe[1]["segment"], "region:cross_border")
        self.assertEqual(rows[2]["severity_label"], "ok")


    def test_rank_worst_segments_is_deterministic(self) -> None:
        frame = pd.DataFrame(
            [
                {"segment": "b", "sample_count": 500, "fpr_gap_vs_overall": 0.09, "fnr_gap_vs_overall": 0.01, "precision_gap_vs_overall": 0.01},
                {"segment": "a", "sample_count": 500, "fpr_gap_vs_overall": 0.09, "fnr_gap_vs_overall": 0.01, "precision_gap_vs_overall": 0.01},
                {"segment": "c", "sample_count": 500, "fpr_gap_vs_overall": 0.20, "fnr_gap_vs_overall": 0.01, "precision_gap_vs_overall": 0.01},
            ]
        )
        ranked = rank_worst_segments(
            frame,
            max_fpr_gap=0.08,
            max_fnr_gap=0.12,
            max_precision_gap=0.12,
            min_segment_size=200,
            top_k=3,
        )
        self.assertEqual([r["segment"] for r in ranked], ["c", "a", "b"])

    def test_build_markdown_report_includes_mitigation_when_disparity_found(self) -> None:
        report = build_markdown_report(
            generated_at="2026-01-01T00:00:00Z",
            overall={"precision": 0.5, "recall": 0.6, "false_positive_rate": 0.02},
            disparities=[{"segment": "region:cross_border", "violations": ["fpr_gap"]}],
            segment_rows=[{"segment": "region:cross_border", "sample_count": 10, "precision": 0.4, "recall": 0.2, "false_positive_rate": 0.1}],
            top_features_df=pd.DataFrame([{"feature": "V14", "mean_abs_shap": 0.1234}]),
            explainability_method="xgboost_pred_contribs",
        )

        self.assertIn("Mitigation notes", report)
        self.assertIn("Explainability (top drivers)", report)


    def test_identity_bucket_metrics_contains_expected_buckets(self) -> None:
        df = pd.DataFrame({
            "card1": [1, 1, 1, 2, np.nan],
            "card2": [10, 10, 10, 20, np.nan],
            "addr1": [100, 100, 100, 200, np.nan],
            "P_emaildomain": ["a.com", "a.com", "a.com", "b.com", np.nan],
        })
        buckets, _ = assign_identity_buckets(df)
        labels = np.array([0, 1, 0, 1, 0])
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        pred_block = scores >= 0.5

        result = compute_identity_bucket_metrics(labels, scores, pred_block, buckets)

        names = [row["bucket"] for row in result["buckets"]]
        self.assertIn("known_high_confidence_entity_id", names)
        self.assertIn("uncertain_weakly_linked_entity_id", names)
        self.assertIn("unknown_no_entity", names)
        self.assertIn("pr_auc_max_gap", result["gap_summary"])



if __name__ == "__main__":
    unittest.main()

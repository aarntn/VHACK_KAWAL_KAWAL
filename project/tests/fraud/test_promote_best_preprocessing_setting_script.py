import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from project.scripts import promote_best_preprocessing_setting as promotion_script
from project.scripts.promote_best_preprocessing_setting import (
    build_training_command,
    evaluate_quality_gate,
    main,
    parse_setting_name,
    resolve_selection_csv,
    select_best_row,
)


class PromoteBestPreprocessingSettingScriptTests(unittest.TestCase):

    def test_evaluate_quality_gate_reports_failed_metrics(self) -> None:
        row = pd.Series({"f1": 0.12, "pr_auc": 0.10, "recall": 0.08, "false_positive_rate": 0.03})
        gate = evaluate_quality_gate(row, min_f1=0.2, min_pr_auc=0.2, min_recall=0.2, max_fpr=0.02)
        self.assertFalse(gate["passed"])
        self.assertIn("f1", gate["failed_metrics"])
        self.assertIn("pr_auc", gate["failed_metrics"])
        self.assertIn("recall", gate["failed_metrics"])
        self.assertIn("false_positive_rate", gate["failed_metrics"])

    def test_parse_setting_name(self) -> None:
        encoding, scaler = parse_setting_name("frequency_robust")
        self.assertEqual(encoding, "frequency")
        self.assertEqual(scaler, "robust")

    def test_select_best_row_by_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            best_csv = Path(temp_dir) / "best.csv"
            pd.DataFrame(
                [
                    {"preprocessing_setting": "onehot_robust", "f1": 0.30, "threshold": 0.2},
                    {"preprocessing_setting": "frequency_robust", "f1": 0.42, "threshold": 0.25},
                ]
            ).to_csv(best_csv, index=False)

            row, diagnostics = select_best_row(best_csv, "f1")

        self.assertEqual(row["preprocessing_setting"], "frequency_robust")
        self.assertAlmostEqual(float(row["f1"]), 0.42, places=6)
        self.assertEqual(diagnostics["candidate_count_before_policy"], 2)

    def test_select_best_row_full_scope_with_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_csv = Path(temp_dir) / "comparison.csv"
            pd.DataFrame(
                [
                    {
                        "preprocessing_setting": "onehot_robust",
                        "threshold": 0.65,
                        "f1": 0.255,
                        "recall": 0.41,
                        "precision": 0.18,
                        "false_positive_rate": 0.064,
                    },
                    {
                        "preprocessing_setting": "onehot_robust",
                        "threshold": 0.70,
                        "f1": 0.262,
                        "recall": 0.35,
                        "precision": 0.21,
                        "false_positive_rate": 0.048,
                    },
                    {
                        "preprocessing_setting": "frequency_robust",
                        "threshold": 0.70,
                        "f1": 0.228,
                        "recall": 0.24,
                        "precision": 0.21,
                        "false_positive_rate": 0.031,
                    },
                ]
            ).to_csv(comparison_csv, index=False)

            row, diagnostics = select_best_row(
                comparison_csv,
                selection_metric="recall",
                selection_scope="full",
                max_fpr=0.05,
                min_precision=0.20,
            )

        self.assertEqual(row["preprocessing_setting"], "onehot_robust")
        self.assertAlmostEqual(float(row["threshold"]), 0.70, places=6)
        self.assertEqual(diagnostics["candidate_count_after_policy"], 2)

    def test_select_best_row_full_scope_constraints_no_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_csv = Path(temp_dir) / "comparison.csv"
            pd.DataFrame(
                [
                    {
                        "preprocessing_setting": "onehot_robust",
                        "threshold": 0.65,
                        "f1": 0.255,
                        "recall": 0.41,
                        "precision": 0.18,
                        "false_positive_rate": 0.064,
                    }
                ]
            ).to_csv(comparison_csv, index=False)

            with self.assertRaises(ValueError):
                select_best_row(
                    comparison_csv,
                    selection_metric="recall",
                    selection_scope="full",
                    max_fpr=0.05,
                    min_precision=0.20,
                )

    def test_resolve_selection_csv_uses_comparison_for_full_scope(self) -> None:
        class Args:
            selection_scope = "full"
            comparison_csv = None
            best_csv = None
            dataset_source = "ieee_cis"

        path = resolve_selection_csv(Args())
        self.assertTrue(str(path).endswith("ieee_cis_preprocessing_threshold_comparison.csv"))

    def test_build_training_command_for_ieee_requires_paths(self) -> None:
        class Args:
            dataset_source = "ieee_cis"
            dataset_path = Path("project/legacy_creditcard/creditcard.csv")
            ieee_transaction_path = Path("/data/train_transaction.csv")
            ieee_identity_path = Path("/data/train_identity.csv")
            model_output = Path("project/models/m.pkl")
            features_output = Path("project/models/f.pkl")
            thresholds_output = Path("project/models/t.pkl")
            preprocessing_artifact_output = Path("project/models/p.pkl")

        cmd = build_training_command(Args(), encoding="onehot", scaler="robust")

        self.assertIn("--dataset-source", cmd)
        self.assertIn("ieee_cis", cmd)
        self.assertIn("--ieee-transaction-path", cmd)
        self.assertIn("--ieee-identity-path", cmd)
        self.assertIn("--use-preprocessing", cmd)

    def test_select_best_row_raises_when_policy_column_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_csv = Path(temp_dir) / "comparison.csv"
            pd.DataFrame(
                [
                    {
                        "preprocessing_setting": "onehot_robust",
                        "threshold": 0.65,
                        "f1": 0.255,
                        "recall": 0.41,
                    }
                ]
            ).to_csv(comparison_csv, index=False)

            with self.assertRaises(ValueError):
                select_best_row(
                    comparison_csv,
                    selection_metric="f1",
                    selection_scope="full",
                    max_fpr=0.05,
                )

    def test_select_best_row_with_policy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_csv = Path(temp_dir) / "comparison.csv"
            pd.DataFrame(
                [
                    {
                        "preprocessing_setting": "onehot_robust",
                        "threshold": 0.65,
                        "f1": 0.30,
                        "recall": 0.41,
                        "precision": 0.18,
                        "false_positive_rate": 0.064,
                    },
                    {
                        "preprocessing_setting": "frequency_standard",
                        "threshold": 0.70,
                        "f1": 0.29,
                        "recall": 0.44,
                        "precision": 0.19,
                        "false_positive_rate": 0.070,
                    },
                ]
            ).to_csv(comparison_csv, index=False)

            row, diagnostics = select_best_row(
                comparison_csv,
                selection_metric="f1",
                selection_scope="full",
                max_fpr=0.05,
                min_precision=0.20,
                allow_policy_fallback=True,
            )

        self.assertEqual(row["preprocessing_setting"], "onehot_robust")
        self.assertTrue(diagnostics["policy_fallback_used"])
        self.assertGreaterEqual(len(diagnostics["fallback_policy_violations"]), 1)

    def test_main_dry_run_writes_policy_and_threshold_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison_csv = root / "comparison.csv"
            promotion_report = root / "promotion_report.json"
            dataset_path = root / "creditcard.csv"
            dataset_path.write_text("stub", encoding="utf-8")
            robustness_report = root / "robustness.json"
            robustness_report.write_text("{\"robustness_gate\": {\"passed\": true}}", encoding="utf-8")

            pd.DataFrame(
                [
                    {
                        "preprocessing_setting": "onehot_robust",
                        "threshold": 0.65,
                        "f1": 0.26,
                        "recall": 0.41,
                        "precision": 0.18,
                        "false_positive_rate": 0.064,
                    },
                    {
                        "preprocessing_setting": "frequency_robust",
                        "threshold": 0.70,
                        "f1": 0.24,
                        "recall": 0.35,
                        "precision": 0.24,
                        "false_positive_rate": 0.03,
                        "pr_auc": 0.25,
                    },
                ]
            ).to_csv(comparison_csv, index=False)

            argv = [
                "promote_best_preprocessing_setting.py",
                "--dataset-source",
                "creditcard",
                "--selection-scope",
                "full",
                "--comparison-csv",
                str(comparison_csv),
                "--selection-metric",
                "f1",
                "--max-fpr",
                "0.05",
                "--min-precision",
                "0.20",
                "--dataset-path",
                str(dataset_path),
                "--promotion-report",
                str(promotion_report),
                "--validation-robustness-report",
                str(robustness_report),
                "--min-f1",
                "0.1",
                "--min-pr-auc",
                "0.1",
                "--min-recall",
                "0.1",
                "--dry-run",
            ]

            with mock.patch.object(promotion_script.sys, "argv", argv), mock.patch.object(
                promotion_script.subprocess, "run"
            ) as run_mock:
                main()

            run_mock.assert_not_called()
            report = pd.read_json(promotion_report, typ="series")
            self.assertAlmostEqual(float(report["selected_threshold"]), 0.70, places=6)
            self.assertEqual(report["selected_setting"], "frequency_robust")
            self.assertEqual(report["selected_objective"], "f1")
            self.assertEqual(report["selection_policy"]["max_fpr"], 0.05)
            self.assertEqual(report["selection_policy"]["min_precision"], 0.2)

    def test_main_blocks_on_mixed_label_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison_csv = root / "comparison.csv"
            promotion_report = root / "promotion_report.json"
            dataset_path = root / "creditcard.csv"
            dataset_path.write_text("stub", encoding="utf-8")
            robustness_report = root / "robustness.json"
            robustness_report.write_text(
                '{"robustness_gate": {"passed": true}, "label_policy": "account_propagated"}',
                encoding="utf-8",
            )

            pd.DataFrame(
                [
                    {
                        "preprocessing_setting": "frequency_robust",
                        "threshold": 0.70,
                        "f1": 0.24,
                        "recall": 0.35,
                        "precision": 0.24,
                        "false_positive_rate": 0.03,
                        "pr_auc": 0.25,
                    }
                ]
            ).to_csv(comparison_csv, index=False)

            argv = [
                "promote_best_preprocessing_setting.py",
                "--dataset-source",
                "creditcard",
                "--label-policy",
                "transaction",
                "--selection-scope",
                "full",
                "--comparison-csv",
                str(comparison_csv),
                "--dataset-path",
                str(dataset_path),
                "--promotion-report",
                str(promotion_report),
                "--validation-robustness-report",
                str(robustness_report),
                "--dry-run",
            ]

            with mock.patch.object(promotion_script.sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "robustness_report=.*preprocessing_metadata=.*observed="):
                    main()

    def test_main_real_run_executes_training_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison_csv = root / "comparison.csv"
            promotion_report = root / "promotion_report.json"
            dataset_path = root / "creditcard.csv"
            dataset_path.write_text("stub", encoding="utf-8")
            robustness_report = root / "robustness.json"
            robustness_report.write_text("{\"robustness_gate\": {\"passed\": true}}", encoding="utf-8")

            pd.DataFrame(
                [
                    {
                        "preprocessing_setting": "onehot_standard",
                        "threshold": 0.55,
                        "f1": 0.30,
                        "recall": 0.39,
                        "precision": 0.25,
                        "false_positive_rate": 0.04,
                        "pr_auc": 0.35,
                    }
                ]
            ).to_csv(comparison_csv, index=False)

            argv = [
                "promote_best_preprocessing_setting.py",
                "--dataset-source",
                "creditcard",
                "--selection-scope",
                "full",
                "--comparison-csv",
                str(comparison_csv),
                "--selection-metric",
                "f1",
                "--dataset-path",
                str(dataset_path),
                "--promotion-report",
                str(promotion_report),
                "--validation-robustness-report",
                str(robustness_report),
                "--min-f1",
                "0.1",
                "--min-pr-auc",
                "0.1",
                "--min-recall",
                "0.1",
            ]

            with mock.patch.object(promotion_script.sys, "argv", argv), mock.patch.object(
                promotion_script.subprocess, "run"
            ) as run_mock:
                main()

            run_mock.assert_called_once()
            called_command = run_mock.call_args[0][0]
            self.assertIn("--use-preprocessing", called_command)
            self.assertIn("onehot", called_command)
            self.assertIn("standard", called_command)
            self.assertTrue(promotion_report.exists())

    def test_main_fallback_hard_floor_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison_csv = root / "comparison.csv"
            promotion_report = root / "promotion_report.json"
            robustness_report = root / "robustness.json"
            robustness_report.write_text('{"robustness_gate": {"passed": true}}', encoding="utf-8")
            dataset_path = root / "creditcard.csv"
            dataset_path.write_text("stub", encoding="utf-8")

            pd.DataFrame([
                {
                    "preprocessing_setting": "onehot_standard",
                    "threshold": 0.55,
                    "f1": 0.05,
                    "pr_auc": 0.2,
                    "recall": 0.3,
                    "precision": 0.01,
                    "false_positive_rate": 0.2,
                }
            ]).to_csv(comparison_csv, index=False)

            argv = [
                "promote_best_preprocessing_setting.py",
                "--dataset-source", "creditcard",
                "--selection-scope", "full",
                "--comparison-csv", str(comparison_csv),
                "--allow-policy-fallback",
                "--max-fpr", "0.01",
                "--min-precision", "0.5",
                "--min-f1", "0.1",
                "--dataset-path", str(dataset_path),
                "--promotion-report", str(promotion_report),
                "--validation-robustness-report", str(robustness_report),
                "--dry-run",
            ]
            with mock.patch.object(promotion_script.sys, "argv", argv), self.assertRaises(ValueError):
                main()

            self.assertTrue(promotion_report.exists())


    def test_main_quality_gate_block_writes_diagnostics_and_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison_csv = root / "comparison.csv"
            promotion_report = root / "promotion_report.json"
            robustness_report = root / "robustness.json"
            robustness_report.write_text('{"robustness_gate": {"passed": true}}', encoding="utf-8")
            dataset_path = root / "creditcard.csv"
            dataset_path.write_text("stub", encoding="utf-8")

            pd.DataFrame(
                [
                    {
                        "preprocessing_setting": "onehot_standard",
                        "threshold": 0.55,
                        "f1": 0.05,
                        "pr_auc": 0.04,
                        "recall": 0.03,
                        "precision": 0.20,
                        "false_positive_rate": 0.01,
                    }
                ]
            ).to_csv(comparison_csv, index=False)

            argv = [
                "promote_best_preprocessing_setting.py",
                "--dataset-source", "creditcard",
                "--selection-scope", "full",
                "--comparison-csv", str(comparison_csv),
                "--dataset-path", str(dataset_path),
                "--promotion-report", str(promotion_report),
                "--validation-robustness-report", str(robustness_report),
                "--min-f1", "0.2",
                "--min-pr-auc", "0.2",
                "--min-recall", "0.2",
                "--dry-run",
            ]

            with mock.patch.object(promotion_script.sys, "argv", argv), mock.patch.object(
                promotion_script.subprocess, "run"
            ) as run_mock:
                with self.assertRaisesRegex(ValueError, "See promotion report") as ctx:
                    main()

            self.assertIn(str(promotion_report), str(ctx.exception))
            run_mock.assert_not_called()
            report = pd.read_json(promotion_report, typ="series")
            self.assertEqual(report["selected_setting"], "onehot_standard")
            self.assertIn("selected_row_metadata", report.index)
            self.assertIn("next_actions", report.index)
            self.assertIn("quality_gate", report.index)
            self.assertFalse(report["quality_gate"]["passed"])
            self.assertIn("f1", report["quality_gate"]["failed_metrics"])
            self.assertIn("pr_auc", report["quality_gate"]["failed_metrics"])
            self.assertIn("recall", report["quality_gate"]["failed_metrics"])
            self.assertEqual(report["quality_gate"]["checks"]["f1"]["required"], 0.2)
            self.assertEqual(report["quality_gate"]["checks"]["f1"]["actual"], 0.05)

    def test_main_policy_block_writes_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison_csv = root / "comparison.csv"
            promotion_report = root / "promotion_report.json"
            robustness_report = root / "robustness.json"
            robustness_report.write_text('{"robustness_gate": {"passed": true}}', encoding="utf-8")
            dataset_path = root / "creditcard.csv"
            dataset_path.write_text("stub", encoding="utf-8")

            pd.DataFrame([
                {
                    "preprocessing_setting": "onehot_standard",
                    "threshold": 0.55,
                    "f1": 0.05,
                    "pr_auc": 0.2,
                    "recall": 0.3,
                    "precision": 0.01,
                    "false_positive_rate": 0.01,
                }
            ]).to_csv(comparison_csv, index=False)

            argv = [
                "promote_best_preprocessing_setting.py",
                "--dataset-source", "creditcard",
                "--selection-scope", "full",
                "--comparison-csv", str(comparison_csv),
                "--min-precision", "0.2",
                "--dataset-path", str(dataset_path),
                "--promotion-report", str(promotion_report),
                "--validation-robustness-report", str(robustness_report),
                "--dry-run",
            ]
            with mock.patch.object(promotion_script.sys, "argv", argv), self.assertRaises(ValueError):
                main()

            report = pd.read_json(promotion_report, typ="series")
            self.assertEqual(report["status"], "blocked_selection_policy")
            self.assertIn("remediation", report.index)



if __name__ == "__main__":
    unittest.main()

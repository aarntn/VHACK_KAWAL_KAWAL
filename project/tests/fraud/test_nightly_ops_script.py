import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from project.scripts import nightly_ops


class NightlyOpsScriptTests(unittest.TestCase):

    def test_append_dataset_source_args_creditcard(self) -> None:
        args = argparse.Namespace(
            dataset_source="creditcard",
            dataset_path=Path("project/legacy_creditcard/creditcard.csv"),
            ieee_transaction_path=None,
            ieee_identity_path=None,
        )
        cmd = nightly_ops.append_dataset_source_args(["python", "x.py"], args)
        self.assertIn("--dataset-source", cmd)
        self.assertIn("creditcard", cmd)
        self.assertIn("--dataset-path", cmd)

    def test_append_dataset_source_args_ieee_requires_paths(self) -> None:
        args = argparse.Namespace(
            dataset_source="ieee_cis",
            dataset_path=Path("project/legacy_creditcard/creditcard.csv"),
            ieee_transaction_path=None,
            ieee_identity_path=None,
        )
        with self.assertRaisesRegex(ValueError, "--ieee-transaction-path and --ieee-identity-path are required"):
            nightly_ops.append_dataset_source_args(["python", "x.py"], args)

    def test_resolve_model_artifact_paths_defaults_to_promoted_ieee_artifacts(self) -> None:
        args = argparse.Namespace(
            dataset_source="ieee_cis",
            model_path=nightly_ops.DEFAULT_MODEL_PATH,
            feature_path=nightly_ops.DEFAULT_FEATURE_PATH,
            thresholds_output=nightly_ops.DEFAULT_THRESHOLDS_OUTPUT,
            preprocessing_artifact_path=nightly_ops.DEFAULT_PREPROCESSING_ARTIFACT_PATH,
        )

        resolved = nightly_ops.resolve_model_artifact_paths(args)

        self.assertEqual(resolved["model_path"], nightly_ops.DEFAULT_IEEE_MODEL_PATH)
        self.assertEqual(resolved["feature_path"], nightly_ops.DEFAULT_IEEE_FEATURE_PATH)
        self.assertEqual(resolved["thresholds_output"], nightly_ops.DEFAULT_IEEE_THRESHOLDS_OUTPUT)
        self.assertEqual(resolved["preprocessing_artifact_path"], nightly_ops.DEFAULT_PREPROCESSING_ARTIFACT_PATH)

    def test_run_cohort_kpi_passes_promoted_ieee_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = argparse.Namespace(
                dataset_source="ieee_cis",
                dataset_path=tmp_path / "unused.csv",
                ieee_transaction_path=tmp_path / "train_transaction.csv",
                ieee_identity_path=tmp_path / "train_identity.csv",
                ieee_merged_cache_path=tmp_path / "cache.csv",
                model_path=nightly_ops.DEFAULT_MODEL_PATH,
                feature_path=nightly_ops.DEFAULT_FEATURE_PATH,
                thresholds_output=nightly_ops.DEFAULT_THRESHOLDS_OUTPUT,
                preprocessing_artifact_path=nightly_ops.DEFAULT_PREPROCESSING_ARTIFACT_PATH,
                cohort_kpi_json=tmp_path / "cohort.json",
                cohort_kpi_csv=tmp_path / "cohort.csv",
            )

            with patch.object(nightly_ops, "resolve_ops_dataset_csv", return_value=tmp_path / "merged.csv"), \
                 patch.object(nightly_ops, "run_command", return_value={"ok": True, "command": []}) as mocked:
                nightly_ops.run_cohort_kpi(args)

            cmd = mocked.call_args.args[0]
            self.assertIn(str(nightly_ops.DEFAULT_IEEE_MODEL_PATH), cmd)
            self.assertIn(str(nightly_ops.DEFAULT_IEEE_FEATURE_PATH), cmd)
            self.assertIn(str(nightly_ops.DEFAULT_IEEE_THRESHOLDS_OUTPUT), cmd)
            self.assertIn(str(nightly_ops.DEFAULT_PREPROCESSING_ARTIFACT_PATH), cmd)

    def test_run_data_checks_ieee_missing_paths(self) -> None:
        args = argparse.Namespace(
            dataset_source="ieee_cis",
            dataset_path=Path("project/legacy_creditcard/creditcard.csv"),
            ieee_transaction_path=None,
            ieee_identity_path=None,
        )
        report = nightly_ops.run_data_checks(args)
        self.assertFalse(report["ok"])
        self.assertEqual(report["reason"], "missing_ieee_paths")

    def test_should_recalibrate_respects_drift_payload_and_flags(self) -> None:
        args = argparse.Namespace(skip_calibration=False, run_calibration_on_drift=True)
        self.assertTrue(
            nightly_ops.should_recalibrate(
                {"recalibration_recommendation": {"should_recalibrate": True}},
                args,
            )
        )

        args.skip_calibration = True
        self.assertFalse(
            nightly_ops.should_recalibrate(
                {"recalibration_recommendation": {"should_recalibrate": True}},
                args,
            )
        )

    def test_build_ops_summary_failed_states(self) -> None:
        summary = nightly_ops.build_ops_summary(
            dataset_artifact_validation=None,
            drift_cmd={"ok": False},
            drift_report=None,
            calibration_cmd=None,
            threshold_promotion_cmd=None,
            artifact_validation_cmd=None,
            artifact_validation_report=None,
            benchmark_cmd=None,
            latency_trend_cmd=None,
            latency_stage_analysis_cmd=None,
            latency_stage_matrix_cmd=None,
            cohort_kpi_cmd=None,
            profile_replay_cmd=None,
            profile_health_cmd=None,
            profile_health_report=None,
            benchmark_sla=None,
            benchmark_error_summary=None,
            data_checks=None,
            threshold_policy_guardrails=None,
            retrain_trigger={"should_retrain": False, "reasons": ["none"]},
            archive_result=None,
            alert_result={"sent": False},
        )
        self.assertEqual(summary["status"], "failed_drift_monitor")

        summary = nightly_ops.build_ops_summary(
            dataset_artifact_validation=None,
            drift_cmd={"ok": True},
            drift_report={"summary": {}},
            calibration_cmd={"ok": False},
            threshold_promotion_cmd=None,
            artifact_validation_cmd=None,
            artifact_validation_report=None,
            benchmark_cmd=None,
            latency_trend_cmd=None,
            latency_stage_analysis_cmd=None,
            latency_stage_matrix_cmd=None,
            cohort_kpi_cmd=None,
            profile_replay_cmd=None,
            profile_health_cmd=None,
            profile_health_report=None,
            benchmark_sla=None,
            benchmark_error_summary=None,
            data_checks=None,
            threshold_policy_guardrails=None,
            retrain_trigger={"should_retrain": False, "reasons": ["none"]},
            archive_result=None,
            alert_result={"sent": False},
        )
        self.assertEqual(summary["status"], "failed_calibration")

        summary = nightly_ops.build_ops_summary(
            dataset_artifact_validation=None,
            drift_cmd={"ok": True},
            drift_report={"summary": {}},
            calibration_cmd=None,
            threshold_promotion_cmd=None,
            artifact_validation_cmd=None,
            artifact_validation_report=None,
            benchmark_cmd={"ok": True},
            latency_trend_cmd=None,
            latency_stage_analysis_cmd=None,
            latency_stage_matrix_cmd=None,
            cohort_kpi_cmd=None,
            profile_replay_cmd=None,
            profile_health_cmd=None,
            profile_health_report=None,
            benchmark_sla={"ok": False, "reason": "benchmark_sla_failed", "failing_endpoints": ["score_transaction"]},
            benchmark_error_summary=None,
            data_checks=None,
            threshold_policy_guardrails=None,
            retrain_trigger={"should_retrain": False, "reasons": ["none"]},
            archive_result=None,
            alert_result={"sent": False},
        )
        self.assertEqual(summary["status"], "failed_benchmark_sla")

        summary = nightly_ops.build_ops_summary(
            dataset_artifact_validation=None,
            drift_cmd={"ok": True},
            drift_report={"summary": {}},
            calibration_cmd=None,
            threshold_promotion_cmd=None,
            artifact_validation_cmd=None,
            artifact_validation_report=None,
            benchmark_cmd={"ok": True},
            latency_trend_cmd=None,
            latency_stage_analysis_cmd=None,
            latency_stage_matrix_cmd=None,
            cohort_kpi_cmd=None,
            profile_replay_cmd=None,
            profile_health_cmd=None,
            profile_health_report=None,
            benchmark_sla={"ok": False, "reason": "benchmark_sla_failed", "failing_endpoints": ["score_transaction"]},
            benchmark_error_summary=None,
            data_checks=None,
            threshold_policy_guardrails=None,
            retrain_trigger={"should_retrain": False, "reasons": ["none"]},
            archive_result=None,
            alert_result={"sent": False},
            benchmark_sla_mode="warn",
        )
        self.assertEqual(summary["status"], "ok")

    def test_build_ops_summary_sla_mode_status_transitions(self) -> None:
        failing_sla = {"ok": False, "reason": "benchmark_sla_failed", "failing_endpoints": ["score_transaction"]}
        expected_by_mode = {
            "enforce": "failed_benchmark_sla",
            "warn": "ok",
            "off": "ok",
        }
        for mode, expected_status in expected_by_mode.items():
            with self.subTest(mode=mode):
                summary = nightly_ops.build_ops_summary(
                    dataset_artifact_validation=None,
                    drift_cmd={"ok": True},
                    drift_report={"summary": {}},
                    calibration_cmd=None,
                    threshold_promotion_cmd=None,
                    artifact_validation_cmd=None,
                    artifact_validation_report=None,
                    benchmark_cmd={"ok": True},
                    latency_trend_cmd=None,
                    latency_stage_analysis_cmd=None,
                    latency_stage_matrix_cmd=None,
                    cohort_kpi_cmd=None,
                    profile_replay_cmd=None,
                    profile_health_cmd=None,
                    profile_health_report=None,
                    benchmark_sla=failing_sla,
                    benchmark_error_summary={"score_transaction": {"error_category_distribution": {"timeout": 1}}},
                    data_checks=None,
                    threshold_policy_guardrails=None,
                    retrain_trigger={"should_retrain": False, "reasons": ["none"]},
                    archive_result=None,
                    alert_result={"sent": False},
                    benchmark_sla_mode=mode,
                )
                self.assertEqual(summary["status"], expected_status)
                self.assertEqual(summary["benchmark_sla_mode"], mode)

    def test_build_ops_summary_sla_mode_invalid_value_defaults_to_enforce(self) -> None:
        summary = nightly_ops.build_ops_summary(
            dataset_artifact_validation=None,
            drift_cmd={"ok": True},
            drift_report={"summary": {}},
            calibration_cmd=None,
            threshold_promotion_cmd=None,
            artifact_validation_cmd=None,
            artifact_validation_report=None,
            benchmark_cmd={"ok": True},
            latency_trend_cmd=None,
            latency_stage_analysis_cmd=None,
            latency_stage_matrix_cmd=None,
            cohort_kpi_cmd=None,
            profile_replay_cmd=None,
            profile_health_cmd=None,
            profile_health_report=None,
            benchmark_sla={"ok": False, "reason": "benchmark_sla_failed", "failing_endpoints": ["score_transaction"]},
            benchmark_error_summary=None,
            data_checks=None,
            threshold_policy_guardrails=None,
            retrain_trigger={"should_retrain": False, "reasons": ["none"]},
            archive_result=None,
            alert_result={"sent": False},
            benchmark_sla_mode="unexpected",
        )
        self.assertEqual(summary["benchmark_sla_mode"], "enforce")
        self.assertEqual(summary["status"], "failed_benchmark_sla")

    def test_build_ops_summary_schema_is_stable_and_unique(self) -> None:
        first_summary = nightly_ops.build_ops_summary(
            dataset_artifact_validation={"ok": True},
            drift_cmd={"ok": True},
            drift_report={"summary": {"feature_alert_count": 0}, "recalibration_recommendation": {}},
            calibration_cmd=None,
            threshold_promotion_cmd=None,
            artifact_validation_cmd={"ok": True},
            artifact_validation_report={"ok": True},
            benchmark_cmd={"ok": True},
            latency_trend_cmd={"ok": True},
            latency_stage_analysis_cmd={"ok": True},
            latency_stage_matrix_cmd={"ok": True},
            cohort_kpi_cmd=None,
            profile_replay_cmd=None,
            profile_health_cmd=None,
            profile_health_report=None,
            benchmark_sla={"ok": True, "reason": "ok", "failing_endpoints": []},
            benchmark_error_summary={"score_transaction": {"error_category_distribution": {}}},
            data_checks={"ok": True},
            threshold_policy_guardrails={"ok": True},
            retrain_trigger={"should_retrain": False, "reasons": ["none"]},
            archive_result=None,
            alert_result={"sent": False},
            benchmark_sla_mode="enforce",
        )
        second_summary = nightly_ops.build_ops_summary(
            dataset_artifact_validation={"ok": True},
            drift_cmd={"ok": True},
            drift_report={"summary": {"feature_alert_count": 0}, "recalibration_recommendation": {}},
            calibration_cmd=None,
            threshold_promotion_cmd=None,
            artifact_validation_cmd={"ok": True},
            artifact_validation_report={"ok": True},
            benchmark_cmd={"ok": True},
            latency_trend_cmd={"ok": True},
            latency_stage_analysis_cmd={"ok": True},
            latency_stage_matrix_cmd={"ok": True},
            cohort_kpi_cmd=None,
            profile_replay_cmd=None,
            profile_health_cmd=None,
            profile_health_report=None,
            benchmark_sla={"ok": True, "reason": "ok", "failing_endpoints": []},
            benchmark_error_summary={"score_transaction": {"error_category_distribution": {}}},
            data_checks={"ok": True},
            threshold_policy_guardrails={"ok": True},
            retrain_trigger={"should_retrain": False, "reasons": ["none"]},
            archive_result=None,
            alert_result={"sent": False},
            benchmark_sla_mode="enforce",
        )
        expected_keys = list(nightly_ops.OPS_SUMMARY_SCHEMA_KEYS)
        self.assertEqual(list(first_summary.keys()), expected_keys)
        self.assertEqual(list(second_summary.keys()), expected_keys)
        self.assertEqual(len(first_summary.keys()), len(set(first_summary.keys())))

    def test_run_latency_trend_marks_no_data_as_warning_not_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            trend_json = tmp_path / "latency_trend_report.json"
            trend_json.write_text(
                json.dumps(
                    {
                        "status": "no_data",
                        "reason": "no benchmark JSON files found",
                        "generated_at_utc": "2026-03-18T00:00:00+00:00",
                        "benchmark_dir": str(tmp_path / "benchmark"),
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                benchmark_trend_history_limit=20,
                latency_trend_json=trend_json,
                latency_trend_csv=tmp_path / "latency_trend_report.csv",
            )

            with patch.object(nightly_ops, "run_command", return_value={"ok": True, "returncode": 0, "stdout": "", "stderr": ""}):
                report = nightly_ops.run_latency_trend(args)

            self.assertTrue(report["ok"])
            self.assertTrue(report["warning"])
            self.assertTrue(report["no_data"])
            self.assertEqual(report["warning_reason"], "no benchmark JSON files found")

    def test_evaluate_benchmark_sla_flags_failed_endpoints(self) -> None:
        benchmark_cmd = {
            "ok": True,
            "stdout": json.dumps(
                {
                    "benchmark_mode": "load_test",
                    "failure_mode": "sla_violation",
                    "sla_evaluation": {
                        "score_transaction": {"real_time_viability": "FAIL"},
                        "wallet_authorize_payment": {"real_time_viability": "PASS"},
                    }
                }
            ),
        }
        result = nightly_ops.evaluate_benchmark_sla(benchmark_cmd)
        self.assertFalse(result["ok"])
        self.assertEqual(result["failing_endpoints"], ["score_transaction"])
        self.assertEqual(result["failure_category"], "runtime_sla_failure")
        self.assertEqual(result["operator_diagnosis"], "model_performance_degradation")

    def test_evaluate_benchmark_sla_classifies_contract_validation_failures(self) -> None:
        benchmark_cmd = {
            "ok": False,
            "stdout": json.dumps(
                {
                    "benchmark_mode": "contract_validation_failed",
                    "failure_mode": "invalid_payload_contract",
                    "contract_validation": {"ok": False, "errors": ["wallet payload mismatch"]},
                }
            ),
        }
        result = nightly_ops.evaluate_benchmark_sla(benchmark_cmd)
        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_category"], "invalid_payload_contract")
        self.assertEqual(result["operator_diagnosis"], "pipeline_integration_bug")

    def test_summarize_benchmark_errors_by_endpoint(self) -> None:
        benchmark_cmd = {
            "ok": True,
            "stdout": json.dumps(
                {
                    "endpoints": [
                        {
                            "endpoint_name": "score_transaction",
                            "started_at_utc": "2026-03-18T00:00:00+00:00",
                            "error_count": 2,
                            "error_categories": {"model_runtime_error": 2},
                            "top_exception_signatures": [{"signature": "RuntimeError:x", "count": 2}],
                        },
                        {
                            "endpoint_name": "wallet_authorize_payment",
                            "started_at_utc": "2026-03-18T00:00:00+00:00",
                            "error_count": 0,
                            "error_categories": {},
                            "top_exception_signatures": [],
                        },
                    ]
                }
            ),
        }
        summary = nightly_ops.summarize_benchmark_errors(benchmark_cmd)
        assert summary is not None
        self.assertEqual(
            summary["score_transaction"]["error_category_distribution"],
            {"model_runtime_error": 2},
        )
        self.assertEqual(
            summary["score_transaction"]["first_failure_timestamp_utc"],
            "2026-03-18T00:00:00+00:00",
        )
        self.assertIsNone(summary["wallet_authorize_payment"]["first_failure_timestamp_utc"])

    def test_evaluate_retrain_trigger_flags_expected_reasons(self) -> None:
        args = argparse.Namespace(
            retrain_feature_alert_threshold=2,
            retrain_decision_drift_status="alert",
            retrain_on_data_check_fail=True,
            retrain_on_threshold_guardrail_fail=True,
        )
        drift_report = {"summary": {"feature_alert_count": 3}, "decision_drift": {"status": "alert"}}
        data_checks = {"ok": False}
        threshold_guardrails = {"ok": False}

        trigger = nightly_ops.evaluate_retrain_trigger(drift_report, data_checks, threshold_guardrails, args)

        self.assertTrue(trigger["should_retrain"])
        self.assertIn("feature_alert_count=3", trigger["reasons"])
        self.assertIn("decision_drift_status=alert", trigger["reasons"])
        self.assertIn("data_checks_failed", trigger["reasons"])
        self.assertIn("threshold_policy_guardrails_failed", trigger["reasons"])

    def test_evaluate_retrain_trigger_excludes_integration_failures_from_sla_streak(self) -> None:
        args = argparse.Namespace(
            retrain_feature_alert_threshold=5,
            retrain_decision_drift_status="alert",
            retrain_on_data_check_fail=False,
            retrain_on_threshold_guardrail_fail=False,
            retrain_warn_streak=0,
            retrain_sla_fail_streak=1,
            retrain_endpoint_error_rate_streak=0,
            retrain_endpoint_error_rate_threshold=5.0,
            archive_runs_dir=Path("/tmp/does-not-exist"),
        )
        benchmark_sla = {
            "ok": False,
            "reason": "benchmark_subprocess_failed",
            "benchmark_mode": "contract_validation_failed",
            "failure_mode": "invalid_payload_contract",
            "failure_category": "invalid_payload_contract",
            "operator_diagnosis": "pipeline_integration_bug",
            "failing_endpoints": [],
        }

        trigger = nightly_ops.evaluate_retrain_trigger(
            drift_report=None,
            data_checks=None,
            threshold_guardrails=None,
            args=args,
            benchmark_sla=benchmark_sla,
            benchmark_cmd=None,
        )

        self.assertFalse(trigger["should_retrain"])
        self.assertNotIn("benchmark_sla_fail_streak=1", trigger["reasons"])
        self.assertEqual(trigger["details"]["benchmark_sla_fail_streak"]["current_streak"], 0)
        self.assertEqual(trigger["details"]["integration_failure_streak"]["current_streak"], 1)

    def test_select_run_artifacts_for_archive_excludes_stale_optional_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = argparse.Namespace(
                dataset_source="ieee_cis",
                drift_json=tmp_path / "drift.json",
                drift_csv=tmp_path / "drift.csv",
                ops_summary_json=tmp_path / "ops.json",
                calibration_json=tmp_path / "calibration.json",
                calibration_csv=tmp_path / "calibration.csv",
                promotion_record_json=tmp_path / "promotion_record.json",
                artifact_validation_json=tmp_path / "artifact_validation.json",
                thresholds_output=nightly_ops.DEFAULT_THRESHOLDS_OUTPUT,
                cohort_kpi_json=tmp_path / "cohort.json",
                latency_trend_json=tmp_path / "latency.json",
                latency_stage_analysis_json=tmp_path / "latency_stage_analysis.json",
                latency_stage_analysis_csv=tmp_path / "latency_stage_analysis.csv",
                latency_stage_matrix_json=tmp_path / "latency_stage_matrix_report.json",
                latency_stage_matrix_csv=tmp_path / "latency_stage_matrix_report.csv",
                profile_health_json=tmp_path / "profile_health.json",
                model_path=nightly_ops.DEFAULT_MODEL_PATH,
                feature_path=nightly_ops.DEFAULT_FEATURE_PATH,
                preprocessing_artifact_path=nightly_ops.DEFAULT_PREPROCESSING_ARTIFACT_PATH,
            )

            selected = nightly_ops.select_run_artifacts_for_archive(
                args,
                drift_cmd={"ok": True},
                calibration_cmd=None,
                threshold_promotion_cmd=None,
                benchmark_cmd={"ok": True},
                latency_trend_cmd={"ok": True},
                cohort_kpi_cmd={"ok": True},
                latency_stage_analysis_cmd={"ok": True},
                latency_stage_matrix_cmd={"ok": True},
                profile_health_cmd={"ok": True},
            )

            self.assertIn("drift_json", selected)
            self.assertIn("cohort_kpi_json", selected)
            self.assertIn("latency_trend_json", selected)
            self.assertIn("latency_stage_analysis_json", selected)
            self.assertIn("latency_stage_analysis_csv", selected)
            self.assertIn("latency_stage_matrix_json", selected)
            self.assertIn("latency_stage_matrix_csv", selected)
            self.assertIn("profile_health_json", selected)
            self.assertNotIn("calibration_json", selected)
            self.assertNotIn("promotion_record_json", selected)
            self.assertNotIn("thresholds_output", selected)

    def test_archive_run_artifacts_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src = tmp_path / "drift.json"
            src.write_text('{"ok": true}', encoding="utf-8")

            result = nightly_ops.archive_run_artifacts(
                archive_dir=tmp_path / "archive",
                run_id="20260101T000000Z",
                artifacts={"drift_json": src, "missing": tmp_path / "nope.json"},
            )

            self.assertIn("drift_json", result["copied"])
            self.assertIn("missing", result["missing"])
            manifest = Path(result["run_dir"]) / "artifact_manifest.json"
            self.assertTrue(manifest.exists())

    def test_main_runs_calibration_when_recommended(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            drift_json = tmp_path / "drift_report.json"
            ops_summary = tmp_path / "nightly_ops_summary.json"

            drift_payload = {
                "summary": {"feature_alert_count": 1},
                "decision_drift": {"status": "alert"},
                "recalibration_recommendation": {
                    "should_recalibrate": True,
                    "priority": "high",
                    "reasons": ["test"],
                },
            }

            call_index = {"count": 0}

            def fake_run_command(cmd):
                call_index["count"] += 1
                if "drift_monitor.py" in cmd[1]:
                    drift_json.write_text(json.dumps(drift_payload), encoding="utf-8")
                return {
                    "command": cmd,
                    "started_at_utc": "2026-01-01T00:00:00+00:00",
                    "returncode": 0,
                    "stdout": "ok",
                    "stderr": "",
                    "ok": True,
                }

            args = argparse.Namespace(
                dataset_source="creditcard",
                dataset_path=tmp_path / "dataset.csv",
                ieee_transaction_path=None,
                ieee_identity_path=None,
                audit_log=tmp_path / "audit.jsonl",
                drift_json=drift_json,
                drift_csv=tmp_path / "drift.csv",
                ops_summary_json=ops_summary,
                psi_warn=0.1,
                psi_alert=0.25,
                decision_drift_warn=0.1,
                decision_drift_alert=0.2,
                run_calibration_on_drift=True,
                skip_calibration=False,
                calibration_trials=10,
                calibration_seed=42,
                target_fpr=0.005,
                target_precision=0.85,
                calibration_json=tmp_path / "calibration.json",
                calibration_csv=tmp_path / "calibration.csv",
                model_path=nightly_ops.DEFAULT_LEGACY_MODEL_PATH,
                feature_path=nightly_ops.DEFAULT_LEGACY_FEATURE_PATH,
                preprocessing_artifact_path=nightly_ops.DEFAULT_PREPROCESSING_ARTIFACT_PATH,
                thresholds_output=nightly_ops.DEFAULT_LEGACY_THRESHOLDS_OUTPUT,
                promote_thresholds_on_pass=True,
                skip_threshold_promotion=False,
                promotion_record_json=tmp_path / "promotion_record.json",
                artifact_validation_json=tmp_path / "artifact_validation.json",
                run_benchmark=False,
                benchmark_sla_mode="enforce",
                benchmark_requests=10,
                benchmark_concurrency=2,
                benchmark_trend_history_limit=10,
                latency_trend_json=tmp_path / "latency_trend.json",
                latency_trend_csv=tmp_path / "latency_trend.csv",
                latency_stage_analysis_json=tmp_path / "latency_stage_analysis.json",
                latency_stage_analysis_csv=tmp_path / "latency_stage_analysis.csv",
                run_cohort_kpi=False,
                cohort_kpi_json=tmp_path / "cohort.json",
                cohort_kpi_csv=tmp_path / "cohort.csv",
                run_profile_replay=False,
                profile_replay_json=tmp_path / "profile_replay.json",
                profile_replay_user_count=20,
                profile_replay_transactions_per_user=5,
                run_profile_health=False,
                profile_store_backend="sqlite",
                profile_sqlite_path=tmp_path / "profiles.sqlite3",
                profile_health_json=tmp_path / "profile_health.json",
                profile_min_history=5,
                profile_stale_seconds=86400,
                skip_data_checks=True,
                skip_threshold_policy_guardrails=True,
                archive_runs_dir=tmp_path / "archive",
                skip_archive=True,
                retrain_feature_alert_threshold=2,
                retrain_decision_drift_status="alert",
                retrain_on_data_check_fail=True,
                retrain_on_threshold_guardrail_fail=True,
                alert_webhook_url="",
                alert_timeout=5.0,
            )

            with patch.object(nightly_ops, "parse_args", return_value=args), patch.object(
                nightly_ops, "run_command", side_effect=fake_run_command
            ):
                exit_code = nightly_ops.main()

            self.assertEqual(exit_code, 0)
            saved_summary = json.loads(ops_summary.read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["status"], "ok")
            self.assertIsNotNone(saved_summary["calibration"])
            self.assertEqual(call_index["count"], 4)
            self.assertIsNotNone(saved_summary["threshold_promotion"])
            self.assertIsNone(saved_summary["latency_stage_analysis"])

    def test_main_runs_latency_stage_analysis_even_when_audit_log_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            drift_json = tmp_path / "drift_report.json"
            ops_summary = tmp_path / "nightly_ops_summary.json"

            drift_payload = {
                "summary": {"feature_alert_count": 0},
                "decision_drift": {"status": "ok"},
                "recalibration_recommendation": {"should_recalibrate": False},
            }

            call_index = {"count": 0}

            def fake_run_command(cmd):
                call_index["count"] += 1
                if "drift_monitor.py" in cmd[1]:
                    drift_json.write_text(json.dumps(drift_payload), encoding="utf-8")
                return {
                    "command": cmd,
                    "started_at_utc": "2026-01-01T00:00:00+00:00",
                    "returncode": 0,
                    "stdout": "ok",
                    "stderr": "",
                    "ok": True,
                }

            args = argparse.Namespace(
                dataset_source="creditcard",
                dataset_path=tmp_path / "dataset.csv",
                ieee_transaction_path=None,
                ieee_identity_path=None,
                audit_log=tmp_path / "audit.jsonl",
                drift_json=drift_json,
                drift_csv=tmp_path / "drift.csv",
                ops_summary_json=ops_summary,
                psi_warn=0.1,
                psi_alert=0.25,
                decision_drift_warn=0.1,
                decision_drift_alert=0.2,
                run_calibration_on_drift=True,
                skip_calibration=False,
                calibration_trials=10,
                calibration_seed=42,
                target_fpr=0.005,
                target_precision=0.85,
                calibration_json=tmp_path / "calibration.json",
                calibration_csv=tmp_path / "calibration.csv",
                model_path=nightly_ops.DEFAULT_LEGACY_MODEL_PATH,
                feature_path=nightly_ops.DEFAULT_LEGACY_FEATURE_PATH,
                preprocessing_artifact_path=nightly_ops.DEFAULT_PREPROCESSING_ARTIFACT_PATH,
                thresholds_output=nightly_ops.DEFAULT_LEGACY_THRESHOLDS_OUTPUT,
                promote_thresholds_on_pass=False,
                skip_threshold_promotion=False,
                promotion_record_json=tmp_path / "promotion_record.json",
                artifact_validation_json=tmp_path / "artifact_validation.json",
                run_benchmark=True,
                benchmark_sla_mode="off",
                benchmark_requests=10,
                benchmark_concurrency=2,
                benchmark_trend_history_limit=10,
                latency_trend_json=tmp_path / "latency_trend.json",
                latency_trend_csv=tmp_path / "latency_trend.csv",
                latency_stage_analysis_json=tmp_path / "latency_stage_analysis.json",
                latency_stage_analysis_csv=tmp_path / "latency_stage_analysis.csv",
                run_cohort_kpi=False,
                cohort_kpi_json=tmp_path / "cohort.json",
                cohort_kpi_csv=tmp_path / "cohort.csv",
                run_profile_replay=False,
                profile_replay_json=tmp_path / "profile_replay.json",
                profile_replay_user_count=20,
                profile_replay_transactions_per_user=5,
                run_profile_health=False,
                profile_store_backend="sqlite",
                profile_sqlite_path=tmp_path / "profiles.sqlite3",
                profile_health_json=tmp_path / "profile_health.json",
                profile_min_history=5,
                profile_stale_seconds=86400,
                skip_data_checks=True,
                skip_threshold_policy_guardrails=True,
                archive_runs_dir=tmp_path / "archive",
                skip_archive=True,
                retrain_feature_alert_threshold=2,
                retrain_decision_drift_status="alert",
                retrain_on_data_check_fail=True,
                retrain_on_threshold_guardrail_fail=True,
                alert_webhook_url="",
                alert_timeout=5.0,
            )

            with patch.object(nightly_ops, "parse_args", return_value=args), patch.object(
                nightly_ops, "run_command", side_effect=fake_run_command
            ):
                exit_code = nightly_ops.main()

            self.assertEqual(exit_code, 0)
            saved_summary = json.loads(ops_summary.read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["status"], "ok")
            self.assertIsNotNone(saved_summary["benchmark"])
            self.assertIsNotNone(saved_summary["latency_trend"])
            self.assertIsNotNone(saved_summary["latency_stage_analysis"])
            self.assertEqual(call_index["count"], 5)

    def test_send_alert_handles_no_webhook(self) -> None:
        result = nightly_ops.send_alert("", {"x": 1}, timeout_s=1.0)
        self.assertFalse(result["sent"])
        self.assertEqual(result["reason"], "webhook_not_configured")

    def test_resolve_ops_dataset_csv_materializes_ieee_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = argparse.Namespace(
                dataset_source="ieee_cis",
                dataset_path=tmp_path / "unused.csv",
                ieee_transaction_path=tmp_path / "train_transaction.csv",
                ieee_identity_path=tmp_path / "train_identity.csv",
                ieee_merged_cache_path=tmp_path / "cache" / "ieee_ops.csv",
            )
            features = pd.DataFrame({"TransactionDT": [1, 2], "TransactionAmt": [10.0, 20.0]})
            labels = pd.Series([0, 1])

            with patch.object(nightly_ops, "load_ieee_cis", return_value=(features, labels, {})):
                out = nightly_ops.resolve_ops_dataset_csv(args)

            self.assertTrue(out.exists())
            loaded = pd.read_csv(out)
            self.assertIn("Class", loaded.columns)


if __name__ == "__main__":
    unittest.main()

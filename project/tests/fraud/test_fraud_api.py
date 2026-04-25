import json
import importlib
import tempfile
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pickle
import pandas as pd

from fastapi.testclient import TestClient

from project.app.inference_backends import create_inference_backend

try:
    from project.app import hybrid_fraud_api
    from project.app.domain_exceptions import ArtifactSchemaMismatchError, DomainValidationError
except ModuleNotFoundError:
    import app.hybrid_fraud_api as hybrid_fraud_api
    from app.domain_exceptions import ArtifactSchemaMismatchError, DomainValidationError


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "fraud_payloads.json"


def load_payload_fixtures() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


class ConstantProbaModel:
    def predict_proba(self, x):
        n = len(x)
        return [[0.7, 0.3] for _ in range(n)]




class DummyBooster:
    def __init__(self) -> None:
        self.last_shape = None
        self.last_dtype = None

    def inplace_predict(self, x, validate_features=False):
        self.last_shape = x.shape
        self.last_dtype = x.dtype
        return [0.25 for _ in range(len(x))]


class DummyXGBModel:
    def __init__(self) -> None:
        self._booster = DummyBooster()

    def get_booster(self):
        return self._booster


class DummyPredictProbaModel:
    def __init__(self) -> None:
        self.last_shape = None
        self.last_dtype = None

    def predict_proba(self, x):
        self.last_shape = x.shape
        self.last_dtype = x.dtype
        return [[0.75, 0.25] for _ in range(len(x))]


class FraudApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixtures = load_payload_fixtures()
        cls.client = TestClient(hybrid_fraud_api.app)

    @staticmethod
    def _serving_payload(payload: dict) -> dict:
        """Trim fixture payloads to the active serving contract (V1..V17)."""
        trimmed = dict(payload)
        for feature_idx in range(18, 29):
            trimmed.pop(f"V{feature_idx}", None)
        return trimmed

    def test_scoring_endpoint_with_fixed_payloads(self) -> None:
        for case_name in ("approve_case", "flag_case", "block_case"):
            with self.subTest(case=case_name):
                response = self.client.post(
                    "/score_transaction", json=self._serving_payload(self.fixtures[case_name])
                )
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertIn(body["decision"], {"APPROVE", "FLAG", "BLOCK"})
                self.assertIn("final_risk_score", body)
                self.assertIn("user_segment", body)
                self.assertIn("segment_thresholds", body)
                self.assertIn("hard_rule_hits", body)
                self.assertIn("verification_action", body)

    def test_validation_failure_for_nan_inf_and_out_of_range_values(self) -> None:
        malformed_payloads = {
            "nan": {**self._serving_payload(self.fixtures["approve_case"]), "V1": "NaN"},
            "infinite": {**self._serving_payload(self.fixtures["approve_case"]), "V2": "inf"},
            "out_of_range": {**self._serving_payload(self.fixtures["approve_case"]), "V3": 2_000_000},
        }

        for name, payload in malformed_payloads.items():
            with self.subTest(name=name):
                response = self.client.post("/score_transaction", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
                body = response.json()
                self.assertEqual(body["error"], "ValidationError")
                self.assertEqual(body["schema_version_expected"], hybrid_fraud_api.PAYLOAD_SCHEMA_VERSION)
                self.assertTrue(body["details"])

    def test_segment_threshold_overrides_loaded_from_context_calibration(self) -> None:
        overrides = {
            "approve_threshold": 0.30,
            "block_threshold": 0.90,
            "ring_score_weight": 0.12,
            "ring_score_cap": 0.18,
            "segment_thresholds": {
                "new:high_ticket:APP": {
                    "approve_threshold": 0.10,
                    "block_threshold": 0.60,
                }
            },
        }

        original_approve = hybrid_fraud_api.approve_threshold
        original_block = hybrid_fraud_api.block_threshold
        original_ring_weight = hybrid_fraud_api.RING_SCORE_WEIGHT
        original_ring_cap = hybrid_fraud_api.RING_SCORE_CAP
        original_segment_thresholds = dict(hybrid_fraud_api.SEGMENT_THRESHOLDS)
        try:
            hybrid_fraud_api.apply_context_calibration_overrides(overrides)
            self.assertIn("new:high_ticket:APP", hybrid_fraud_api.SEGMENT_THRESHOLDS)
            segment_cfg = hybrid_fraud_api.SEGMENT_THRESHOLDS["new:high_ticket:APP"]
            self.assertEqual(segment_cfg.approve_threshold, 0.1)
            self.assertEqual(segment_cfg.block_threshold, 0.6)
            self.assertEqual(hybrid_fraud_api.RING_SCORE_WEIGHT, 0.12)
            self.assertEqual(hybrid_fraud_api.RING_SCORE_CAP, 0.18)
        finally:
            hybrid_fraud_api.approve_threshold = original_approve
            hybrid_fraud_api.block_threshold = original_block
            hybrid_fraud_api.RING_SCORE_WEIGHT = original_ring_weight
            hybrid_fraud_api.RING_SCORE_CAP = original_ring_cap
            hybrid_fraud_api.SEGMENT_THRESHOLDS = original_segment_thresholds

    def test_segment_thresholds_summary_exposed_in_health(self) -> None:
        original_segment_thresholds = dict(hybrid_fraud_api.SEGMENT_THRESHOLDS)
        try:
            hybrid_fraud_api.SEGMENT_THRESHOLDS = {
                "new:high_ticket:WEB": hybrid_fraud_api.SegmentThresholds(
                    approve_threshold=0.22,
                    block_threshold=0.72,
                    min_block_precision=0.85,
                    max_approve_to_flag_fpr=0.02,
                )
            }
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertIn("segment_thresholds_summary", body)
            self.assertEqual(body["segment_thresholds_summary"]["new:high_ticket:WEB"]["approve_threshold"], 0.22)
            self.assertEqual(body["segment_thresholds_summary"]["new:high_ticket:WEB"]["block_threshold"], 0.72)
            self.assertEqual(body["segment_thresholds_summary"]["new:high_ticket:WEB"]["min_block_precision"], 0.85)
            self.assertEqual(body["segment_thresholds_summary"]["new:high_ticket:WEB"]["max_approve_to_flag_fpr"], 0.02)
        finally:
            hybrid_fraud_api.SEGMENT_THRESHOLDS = original_segment_thresholds

    def test_load_segment_thresholds_config_applies_acceptance_gates(self) -> None:
        with self.assertRaises(DomainValidationError):
            hybrid_fraud_api.load_segment_thresholds_config(
                {
                    "new:high_ticket:WEB": {
                        "approve_threshold": 0.2,
                        "block_threshold": 0.8,
                        "min_block_precision": 0.9,
                        "max_approve_to_flag_fpr": 0.03,
                        "calibration_metrics": {
                            "block_precision": 0.88,
                            "approve_to_flag_fpr": 0.01,
                        },
                    }
                },
                fallback_approve=0.3,
                fallback_block=0.9,
                field_name="segment_thresholds",
            )

    def test_validation_failure_for_renamed_or_missing_external_fields(self) -> None:
        renamed_payload = {
            **self._serving_payload(self.fixtures["approve_case"]),
            "Amount": self.fixtures["approve_case"]["TransactionAmt"],
        }
        renamed_payload.pop("TransactionAmt")

        missing_risk_payload = {
            **self._serving_payload(self.fixtures["approve_case"]),
        }
        missing_risk_payload.pop("device_risk_score")

        for name, payload in {
            "renamed_amount_field": renamed_payload,
            "missing_required_risk_field": missing_risk_payload,
        }.items():
            with self.subTest(name=name):
                response = self.client.post("/score_transaction", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
                body = response.json()
                self.assertEqual(body["error"], "ValidationError")
                self.assertEqual(body["schema_version_expected"], hybrid_fraud_api.PAYLOAD_SCHEMA_VERSION)
                detail_fields = {detail["field"] for detail in body["details"]}
                if name == "renamed_amount_field":
                    self.assertIn("transaction_amount", detail_fields)
                else:
                    self.assertIn("device_risk_score", detail_fields)

    def test_build_model_features_marks_imputed_v_features_source(self) -> None:
        payload = {
            "schema_version": hybrid_fraud_api.PAYLOAD_SCHEMA_VERSION,
            "user_id": "user_1001",
            "transaction_amount": 50.0,
            "device_risk_score": 0.1,
            "ip_risk_score": 0.1,
            "location_risk_score": 0.1,
        }
        tx_dict = hybrid_fraud_api.build_model_features_from_normalized_request(payload)
        self.assertEqual(tx_dict["_v_features_source"], "imputed_zero_vector")
        self.assertTrue(all(tx_dict[f"V{i}"] == 0.0 for i in range(1, 18)))

    def test_build_model_features_from_request_avoids_intermediate_payload_copy(self) -> None:
        request_model = hybrid_fraud_api.ScoreTransactionRequest(
            schema_version=hybrid_fraud_api.PAYLOAD_SCHEMA_VERSION,
            user_id="hot_path_user",
            transaction_amount=88.8,
            device_risk_score=0.2,
            ip_risk_score=0.1,
            location_risk_score=0.15,
            device_id="device_hot",
            device_shared_users_24h=2,
            account_age_days=90,
            sim_change_recent=False,
            tx_type="MERCHANT",
            channel="APP",
            cash_flow_velocity_1h=1,
            p2p_counterparties_24h=0,
            is_cross_border=False,
        )

        tx_dict = hybrid_fraud_api.build_model_features_from_score_transaction_request(
            request_model,
            transaction_dt=12345.0,
        )

        self.assertEqual(tx_dict["TransactionDT"], 12345.0)
        self.assertEqual(tx_dict["TransactionAmt"], 88.8)
        self.assertEqual(tx_dict["_v_features_source"], "imputed_zero_vector")
        self.assertTrue(all(tx_dict[f"V{i}"] == 0.0 for i in range(1, 18)))

    def test_score_transaction_rejects_unsupported_asean_country_code(self) -> None:
        payload = {
            **self._serving_payload(self.fixtures["approve_case"]),
            "source_country": "US",
            "destination_country": "SG",
        }

        response = self.client.post("/score_transaction", json=payload)
        self.assertEqual(response.status_code, 422, response.text)

    def test_score_transaction_rejects_invalid_connectivity_mode(self) -> None:
        payload = {
            **self._serving_payload(self.fixtures["approve_case"]),
            "connectivity_mode": "satellite",
        }

        response = self.client.post("/score_transaction", json=payload)
        self.assertEqual(response.status_code, 422, response.text)

    def test_asean_context_infers_domestic_corridor_from_currency(self) -> None:
        request_model = hybrid_fraud_api.ScoreTransactionRequest(
            schema_version=hybrid_fraud_api.PAYLOAD_SCHEMA_VERSION,
            user_id="idr_domestic_user",
            transaction_amount=185000.0,
            currency="IDR",
            device_risk_score=0.08,
            ip_risk_score=0.05,
            location_risk_score=0.04,
            device_id="device_id_idr_01",
            device_shared_users_24h=1,
            account_age_days=320,
            sim_change_recent=False,
            tx_type="MERCHANT",
            channel="QR",
            cash_flow_velocity_1h=1,
            p2p_counterparties_24h=0,
            is_cross_border=False,
        )

        tx_dict = hybrid_fraud_api.build_model_features_from_score_transaction_request(
            request_model,
            transaction_dt=12345.0,
        )

        self.assertEqual(tx_dict["source_country"], "ID")
        self.assertEqual(tx_dict["destination_country"], "ID")
        self.assertEqual(tx_dict["corridor"], "ID-ID")
        self.assertNotEqual(tx_dict["TransactionAmt"], 185000.0)
        self.assertEqual(tx_dict["TransactionAmt"], tx_dict["normalized_amount_reference"])
        self.assertTrue(tx_dict["normalization_basis"].startswith(hybrid_fraud_api.ASEAN_NORMALIZATION_BASIS))

    def test_score_transaction_offline_buffered_returns_asean_runtime_metadata(self) -> None:
        payload = {
            **self._serving_payload(self.fixtures["flag_case"]),
            "currency": "MYR",
            "source_country": "MY",
            "destination_country": "MY",
            "is_agent_assisted": True,
            "connectivity_mode": "offline_buffered",
            "tx_type": "CASH_OUT",
            "channel": "AGENT",
        }

        response = self.client.post("/score_transaction", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["runtime_mode"], "degraded_local")
        self.assertEqual(body["corridor"], "MY-MY")
        self.assertIsInstance(body["normalized_amount_reference"], float)
        self.assertTrue(body["normalization_basis"].startswith(hybrid_fraud_api.ASEAN_NORMALIZATION_BASIS))
        self.assertIn("ASEAN_OFFLINE_BUFFERED_SCORING", body["reason_codes"])
        self.assertIn("ASEAN_AGENT_ASSISTED_CASH_OUT", body["reason_codes"])

    def test_score_transaction_anchors_base_score_when_v_features_are_imputed(self) -> None:
        payload = self._serving_payload(self.fixtures["approve_case"])
        with patch.object(
            hybrid_fraud_api.inference_backend,
            "predict_positive_proba",
            return_value=[0.99],
        ), patch.object(
            hybrid_fraud_api,
            "DYNAMIC_IMPUTED_ANCHOR_ENABLED",
            True,
        ), patch.object(
            hybrid_fraud_api,
            "compute_imputed_base_anchor",
            return_value=(0.31, {"mode": "dynamic"}),
        ), patch.object(
            hybrid_fraud_api,
            "get_context_adjustment",
            return_value=0.0,
        ), patch.object(
            hybrid_fraud_api,
            "get_behavior_adjustment",
            return_value={
                "behavior_features": {"behavior_risk_score": 0.0, "is_low_history": False},
                "behavior_adjustment": 0.0,
                "behavior_reasons": [],
                "source": "test",
            },
        ), patch.object(
            hybrid_fraud_api,
            "apply_entity_smoothing",
            side_effect=lambda base_score, _: (base_score, {"enabled": False}),
        ):
            response = self.client.post("/score_transaction", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["explainability"]["base"], 0.31)

    def test_compute_imputed_base_anchor_increases_for_cross_border_cashout(self) -> None:
        low_risk = {
            "tx_type": "MERCHANT",
            "is_cross_border": False,
            "sim_change_recent": False,
            "account_age_days": 365,
            "device_risk_score": 0.05,
            "ip_risk_score": 0.05,
            "location_risk_score": 0.05,
        }
        risky = {
            "tx_type": "CASH_OUT",
            "is_cross_border": True,
            "sim_change_recent": True,
            "account_age_days": 2,
            "device_risk_score": 0.90,
            "ip_risk_score": 0.95,
            "location_risk_score": 0.90,
        }
        low_anchor, _ = hybrid_fraud_api.compute_imputed_base_anchor(low_risk)
        high_anchor, high_diag = hybrid_fraud_api.compute_imputed_base_anchor(risky)
        self.assertEqual(high_diag["source"], "dynamic")
        self.assertIsInstance(high_diag["components"], dict)
        self.assertLess(low_anchor, high_anchor)

    def test_anchor_profile_defaults_demo_stable(self) -> None:
        with patch.object(hybrid_fraud_api, "ANCHOR_PROFILE", "demo_stable"):
            baseline = hybrid_fraud_api._anchor_profile_default("ANCHOR_BASELINE_MERCHANT", 0.23)
            self.assertEqual(baseline, 0.21)


    def test_missing_features_error_is_truncated_for_large_feature_sets(self) -> None:
        payload = self._serving_payload(self.fixtures["approve_case"])
        synthetic_features = [f"feature_{i}" for i in range(100)]
        with patch.object(hybrid_fraud_api, "feature_columns", synthetic_features):
            with self.assertRaises(ArtifactSchemaMismatchError) as ctx:
                hybrid_fraud_api.build_ml_input(payload)
        msg = str(ctx.exception)
        self.assertIn("Missing required model features", msg)
        self.assertIn("(+80 more)", msg)

    def test_build_ml_input_preprocessed_aligns_to_feature_columns(self) -> None:
        payload = self._serving_payload(self.fixtures["approve_case"])

        fake_bundle = SimpleNamespace(
            dataset_source="ieee_cis",
            feature_names_out=["f2", "f1", "other"],
        )

        with patch.object(hybrid_fraud_api, "preprocessing_bundle", fake_bundle), patch.object(
            hybrid_fraud_api,
            "feature_columns",
            ["f1", "f2"],
        ), patch.object(
            hybrid_fraud_api,
            "map_to_canonical_features",
            return_value=(None, None),
        ), patch.object(
            hybrid_fraud_api,
            "prepare_preprocessing_inputs",
            return_value=(None, None, {}),
        ), patch.object(
            hybrid_fraud_api,
            "transform_with_bundle",
            return_value=[[2.0, 1.0, 9.0]],
        ):
            arr = hybrid_fraud_api.build_ml_input_preprocessed(payload)

        self.assertEqual(arr.shape, (1, 2))
        self.assertEqual(arr.tolist(), [[1.0, 2.0]])

    def test_missing_features_error_is_truncated_in_preprocessing_mode(self) -> None:
        payload = self._serving_payload(self.fixtures["approve_case"])
        synthetic_features = [f"numeric_canonical__feature_{i}" for i in range(100)]
        fake_bundle = SimpleNamespace(dataset_source="ieee_cis", feature_names_out=["f0"])

        with patch.object(hybrid_fraud_api, "preprocessing_bundle", fake_bundle), patch.object(
            hybrid_fraud_api,
            "feature_columns",
            synthetic_features,
        ), patch.object(
            hybrid_fraud_api,
            "map_to_canonical_features",
            return_value=(None, None),
        ), patch.object(
            hybrid_fraud_api,
            "prepare_preprocessing_inputs",
            return_value=(None, None, {}),
        ), patch.object(
            hybrid_fraud_api,
            "transform_with_bundle",
            return_value=[[1.0]],
        ):
            with self.assertRaises(ArtifactSchemaMismatchError) as ctx:
                hybrid_fraud_api.build_ml_input_preprocessed(payload)

        msg = str(ctx.exception)
        self.assertIn("Missing required model features", msg)
        self.assertIn("(+80 more)", msg)

    def test_detect_feature_columns_shape(self) -> None:
        self.assertEqual(
            hybrid_fraud_api.detect_feature_columns_shape(
                ["TransactionDT", "TransactionAmt", "device_risk_score", "V1", "V2"]
            ),
            "raw",
        )
        self.assertEqual(
            hybrid_fraud_api.detect_feature_columns_shape(
                ["numeric_canonical__amount_raw", "categorical_passthrough__ProductCD_W"]
            ),
            "preprocessed",
        )

    def test_validate_inference_artifact_compatibility_reports_mismatch(self) -> None:
        validation = hybrid_fraud_api.validate_inference_artifact_compatibility(
            use_preprocessing_inference=False,
            columns=["numeric_canonical__amount_raw", "numeric_canonical__amount_log"],
            bundle=None,
        )
        self.assertFalse(validation["ok"])
        self.assertEqual(validation["mode"], "raw")
        self.assertEqual(validation["detected_feature_shape"], "preprocessed")
        self.assertTrue(validation["issues"])

    def test_build_artifact_mismatch_runtime_error_contains_remediation(self) -> None:
        error = hybrid_fraud_api.build_artifact_mismatch_runtime_error(
            {
                "mode": "raw",
                "issues": ["Raw inference mode requires raw feature columns."],
            }
        )
        self.assertIn("Inference artifact mismatch", str(error))
        self.assertIn("final_xgboost_model.py", str(error))

    def test_preprocessing_mode_scoring_returns_200_with_artifacts(self) -> None:
        from project.data.preprocessing import fit_preprocessing_bundle, prepare_preprocessing_inputs

        payload = self._serving_payload(self.fixtures["approve_case"])
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            model_path = temp_dir_path / "model.pkl"
            features_path = temp_dir_path / "feature_columns.pkl"
            thresholds_path = temp_dir_path / "thresholds.pkl"
            bundle_path = temp_dir_path / "preprocessing_bundle.pkl"

            with model_path.open("wb") as handle:
                pickle.dump(ConstantProbaModel(), handle)

            df = pd.DataFrame([payload])
            canonical_df, passthrough_df, diagnostics = prepare_preprocessing_inputs(df, "ieee_cis")
            bundle, _ = fit_preprocessing_bundle(
                canonical_df=canonical_df,
                passthrough_df=passthrough_df,
                dataset_source="ieee_cis",
                include_passthrough=False,
                scaler="robust",
                categorical_encoding="onehot",
                behavior_feature_diagnostics=diagnostics,
            )

            with bundle_path.open("wb") as handle:
                pickle.dump(bundle, handle)
            with features_path.open("wb") as handle:
                pickle.dump(bundle.feature_names_out, handle)
            with thresholds_path.open("wb") as handle:
                pickle.dump({"approve_threshold": 0.3, "block_threshold": 0.9}, handle)

            with patch.dict(
                os.environ,
                {
                    "FRAUD_MODEL_FILE": str(model_path),
                    "FRAUD_FEATURE_FILE": str(features_path),
                    "FRAUD_THRESHOLD_FILE": str(thresholds_path),
                    "FRAUD_USE_PREPROCESSING_INFERENCE": "true",
                    "FRAUD_PREPROCESSING_BUNDLE_FILE": str(bundle_path),
                },
                clear=False,
            ):
                reloaded_module = importlib.reload(hybrid_fraud_api)
                client = TestClient(reloaded_module.app)
                response = client.post("/score_transaction", json=payload)
                self.assertEqual(response.status_code, 200, response.text)

        importlib.reload(hybrid_fraud_api)

    def test_startup_fails_fast_on_mode_artifact_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            model_path = temp_dir_path / "model.pkl"
            features_path = temp_dir_path / "feature_columns.pkl"
            thresholds_path = temp_dir_path / "thresholds.pkl"

            with model_path.open("wb") as handle:
                pickle.dump(ConstantProbaModel(), handle)
            with features_path.open("wb") as handle:
                pickle.dump(["numeric_canonical__amount_raw", "numeric_canonical__amount_log"], handle)
            with thresholds_path.open("wb") as handle:
                pickle.dump({"approve_threshold": 0.3, "block_threshold": 0.9}, handle)

            with patch.dict(
                os.environ,
                {
                    "FRAUD_MODEL_FILE": str(model_path),
                    "FRAUD_FEATURE_FILE": str(features_path),
                    "FRAUD_THRESHOLD_FILE": str(thresholds_path),
                    "FRAUD_USE_PREPROCESSING_INFERENCE": "false",
                },
                clear=False,
            ):
                reloaded = importlib.reload(hybrid_fraud_api)
                with self.assertRaises(RuntimeError) as ctx:
                    reloaded.initialize_runtime_artifacts()

        self.assertIn("Inference artifact mismatch", str(ctx.exception))
        importlib.reload(hybrid_fraud_api)

    def test_missing_model_files_raise_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.pkl"
            with self.assertRaises(FileNotFoundError):
                hybrid_fraud_api.ensure_file_exists(missing_path)

            with self.assertRaises(FileNotFoundError):
                hybrid_fraud_api.load_pickle(missing_path)

    def test_malformed_thresholds_are_rejected(self) -> None:
        malformed_thresholds = [
            (0.5, 0.5),
            (0.9, 0.3),
            (-0.1, 0.8),
            (0.2, 1.1),
        ]

        for approve_threshold, block_threshold in malformed_thresholds:
            with self.subTest(
                approve_threshold=approve_threshold,
                block_threshold=block_threshold,
            ):
                with self.assertRaises(ValueError):
                    hybrid_fraud_api.validate_thresholds(
                        approve_threshold=approve_threshold,
                        block_threshold=block_threshold,
                    )

    def test_context_adjustment_is_deterministic_for_known_payloads(self) -> None:
        cases = {
            "approve_case": 0.07,
            "flag_case": 0.1747,
            "block_case": 0.1684,
        }

        for case_name, expected_adjustment in cases.items():
            with self.subTest(case=case_name):
                adjustment = hybrid_fraud_api.get_context_adjustment(self.fixtures[case_name])
                self.assertAlmostEqual(adjustment, expected_adjustment, places=4)

    def test_context_adjustment_is_capped_by_maximum(self) -> None:
        payload = {
            **self.fixtures["approve_case"],
            "TransactionAmt": 2_500.0,
            "TransactionDT": 500.0,
            "device_risk_score": 1.0,
            "ip_risk_score": 1.0,
            "location_risk_score": 1.0,
            "device_shared_users_24h": 5,
            "account_age_days": 0,
            "sim_change_recent": True,
            "tx_type": "CASH_OUT",
            "channel": "AGENT",
            "cash_flow_velocity_1h": 8,
            "p2p_counterparties_24h": 12,
            "is_cross_border": True,
        }

        breakdown = hybrid_fraud_api.get_context_adjustment_breakdown(payload)
        self.assertAlmostEqual(sum(breakdown.values()), 0.59, places=6)
        self.assertEqual(
            hybrid_fraud_api.get_context_adjustment(payload),
            hybrid_fraud_api.CONTEXT_ADJUSTMENT_MAX,
        )

    def test_decision_boundaries_around_thresholds_are_deterministic(self) -> None:
        payload = self._serving_payload(self.fixtures["approve_case"])
        approve_threshold = hybrid_fraud_api.approve_threshold
        block_threshold = hybrid_fraud_api.block_threshold

        boundary_cases = [
            (approve_threshold - 0.0001, "APPROVE"),
            (approve_threshold, "FLAG"),
            (block_threshold - 0.0001, "FLAG"),
            (block_threshold, "BLOCK"),
        ]

        for base_score, expected_decision in boundary_cases:
            with self.subTest(base_score=base_score, expected_decision=expected_decision):
                with patch.object(
                    hybrid_fraud_api,
                    "model",
                    SimpleNamespace(predict_proba=lambda _: [[1 - base_score, base_score]]),
                ), patch.object(
                    hybrid_fraud_api,
                    "get_context_adjustment",
                    return_value=0.0,
                ), patch.object(
                    hybrid_fraud_api,
                    "get_behavior_adjustment",
                    return_value={"behavior_adjustment": 0.0, "behavior_reasons": []},
                ):
                    response = self.client.post("/score_transaction", json=payload)

                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["decision"], expected_decision)

    def test_audit_logs_do_not_include_raw_user_id_or_full_transaction_vector(self) -> None:
        payload = self._serving_payload(self.fixtures["flag_case"])

        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.jsonl"
            with patch.object(hybrid_fraud_api, "AUDIT_LOG_FILE", audit_path), patch.object(
                hybrid_fraud_api,
                "AUDIT_DIR",
                Path(temp_dir),
            ):
                response = self.client.post("/score_transaction", json=payload)

            self.assertEqual(response.status_code, 200, response.text)
            written = audit_path.read_text(encoding="utf-8")
            record = json.loads(written.strip())

            self.assertNotIn("user_id", record)
            self.assertNotIn(payload["user_id"], written)
            self.assertNotIn("raw_vector_embeddings", record)

            for vector_key in [f"V{i}" for i in range(1, 18)] + ["TransactionDT", "TransactionAmt"]:
                self.assertNotIn(vector_key, record)

            self.assertIn("hash_key_version", record)
            self.assertIn("signature_key_version", record)
            self.assertIn("previous_record_signature", record)
            self.assertIn("record_signature", record)
            self.assertTrue(record["record_signature"])

    def test_audit_logs_include_asean_provenance_without_raw_identifier_leakage(self) -> None:
        payload = {
            **self._serving_payload(self.fixtures["flag_case"]),
            "currency": "SGD",
            "source_country": "SG",
            "destination_country": "PH",
            "is_cross_border": True,
            "connectivity_mode": "online",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.jsonl"
            with patch.object(hybrid_fraud_api, "AUDIT_LOG_FILE", audit_path), patch.object(
                hybrid_fraud_api,
                "AUDIT_DIR",
                Path(temp_dir),
            ), patch.object(
                hybrid_fraud_api,
                "AUDIT_ASYNC_WRITE_ENABLED",
                False,
            ):
                hybrid_fraud_api._audit_last_signature = None
                hybrid_fraud_api._initialize_audit_signature_cache()
                response = self.client.post("/score_transaction", json=payload)

            self.assertEqual(response.status_code, 200, response.text)
            records = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            record = records[-1]

            self.assertNotIn(payload["user_id"], json.dumps(record))
            self.assertIn("asean_provenance", record)
            self.assertEqual(record["asean_provenance"]["source_country"], "SG")
            self.assertEqual(record["asean_provenance"]["destination_country"], "PH")
            self.assertEqual(record["asean_provenance"]["currency"], "SGD")
            self.assertIn("normalized_amount_reference", record["asean_provenance"])
            self.assertIn("reason_codes", record)

    def test_audit_log_signatures_are_hash_chained(self) -> None:
        payload = self._serving_payload(self.fixtures["approve_case"])

        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.jsonl"
            with patch.object(hybrid_fraud_api, "AUDIT_LOG_FILE", audit_path), patch.object(
                hybrid_fraud_api,
                "AUDIT_DIR",
                Path(temp_dir),
            ), patch.object(
                hybrid_fraud_api,
                "AUDIT_ASYNC_WRITE_ENABLED",
                False,
            ):
                hybrid_fraud_api._audit_last_signature = None
                hybrid_fraud_api._initialize_audit_signature_cache()
                first_response = self.client.post("/score_transaction", json=payload)
                second_response = self.client.post("/score_transaction", json=payload)

            self.assertEqual(first_response.status_code, 200, first_response.text)
            self.assertEqual(second_response.status_code, 200, second_response.text)

            records = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(records[0]["previous_record_signature"], "GENESIS")
            self.assertEqual(
                records[1]["previous_record_signature"],
                records[0]["record_signature"],
            )

    def test_first_signed_record_uses_genesis_after_empty_flush(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.jsonl"
            record = {
                "request_id": "first-signing-check",
                "signature_key_version": hybrid_fraud_api.AUDIT_SIGNING_KEY_VERSION,
            }
            with patch.object(hybrid_fraud_api, "AUDIT_LOG_FILE", audit_path), patch.object(
                hybrid_fraud_api,
                "AUDIT_DIR",
                Path(temp_dir),
            ):
                hybrid_fraud_api._audit_last_signature = None
                hybrid_fraud_api._initialize_audit_signature_cache()
                hybrid_fraud_api._flush_audit_batch([])
                hybrid_fraud_api._flush_audit_batch([record])

            records = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["previous_record_signature"], "GENESIS")
            self.assertEqual(
                hybrid_fraud_api._audit_last_signature,
                records[0]["record_signature"],
            )

    def test_score_transaction_tolerates_malformed_trailing_audit_log_line(self) -> None:
        payload = self._serving_payload(self.fixtures["approve_case"])

        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.jsonl"
            prior_signature = "previous-valid-signature"
            audit_path.write_text(
                "\n".join(
                    [
                        json.dumps({"record_signature": prior_signature}),
                        "{malformed trailing json",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(hybrid_fraud_api, "AUDIT_LOG_FILE", audit_path), patch.object(
                hybrid_fraud_api,
                "AUDIT_DIR",
                Path(temp_dir),
            ):
                response = self.client.post("/score_transaction", json=payload)

            self.assertEqual(response.status_code, 200, response.text)

            records = []
            for line in audit_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            self.assertGreaterEqual(len(records), 2)
            self.assertEqual(records[-1]["previous_record_signature"], prior_signature)

    def test_score_transaction_uses_genesis_when_latest_audit_line_is_malformed(self) -> None:
        payload = self._serving_payload(self.fixtures["approve_case"])

        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.jsonl"
            audit_path.write_text("{malformed trailing json\n", encoding="utf-8")

            with patch.object(hybrid_fraud_api, "AUDIT_LOG_FILE", audit_path), patch.object(
                hybrid_fraud_api,
                "AUDIT_DIR",
                Path(temp_dir),
            ):
                response = self.client.post("/score_transaction", json=payload)

            self.assertEqual(response.status_code, 200, response.text)

            records = []
            for line in audit_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            self.assertGreaterEqual(len(records), 1)
            self.assertEqual(records[-1]["previous_record_signature"], "GENESIS")

    def test_low_history_policy_does_not_downgrade_block_decision(self) -> None:
        decision = hybrid_fraud_api.apply_low_history_policy(
            final_score=0.78,
            raw_decision="BLOCK",
            behavior_features={"is_low_history": True},
        )
        self.assertEqual(decision, "BLOCK")

    def test_low_history_policy_allows_block_for_extreme_scores(self) -> None:
        decision = hybrid_fraud_api.apply_low_history_policy(
            final_score=0.99,
            raw_decision="FLAG",
            behavior_features={"is_low_history": True},
        )
        self.assertEqual(decision, "BLOCK")

    def test_legacy_fallback_env_vars_are_supported(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HASH_SALT": "legacy-hash-secret",
                "FRAUD_HASH_KEY_VERSION": "v9",
                "FRAUD_AUDIT_SIGNING_SECRET": "legacy-signing-secret",
                "FRAUD_AUDIT_SIGNING_KEY_VERSION": "v8",
            },
            clear=True,
        ):
            reloaded_module = importlib.reload(hybrid_fraud_api)
            self.assertEqual(reloaded_module.HASH_SECRET_SOURCE, "HASH_SALT")
            self.assertEqual(reloaded_module.HASH_KEY_VERSION, "v1")
            self.assertTrue(reloaded_module.HASH_KEY_VERSION_FALLBACK_USED)
            self.assertEqual(reloaded_module.AUDIT_SIGNING_SECRET_SOURCE, "FRAUD_AUDIT_SIGNING_SECRET")
            self.assertEqual(reloaded_module.AUDIT_SIGNING_KEY_VERSION, "v1")
            self.assertTrue(reloaded_module.AUDIT_SIGNING_KEY_VERSION_FALLBACK_USED)

        importlib.reload(hybrid_fraud_api)


    def test_inference_backend_factory_aliases(self) -> None:
        self.assertEqual(create_inference_backend("predict_proba").backend_name, "predict_proba")
        self.assertEqual(create_inference_backend("inplace_predict").backend_name, "xgboost_inplace_predict")
        self.assertEqual(create_inference_backend("xgboost").backend_name, "xgboost_inplace_predict")
        self.assertEqual(create_inference_backend("").backend_name, "xgboost_inplace_predict")
        self.assertEqual(create_inference_backend("onnx").backend_name, "onnx_hummingbird")

    def test_inplace_backend_coerces_1d_input_to_2d_float32_matrix(self) -> None:
        backend = create_inference_backend("xgboost_inplace_predict")
        model = DummyXGBModel()

        scores = backend.predict_positive_proba(model, [1.0, 2.0, 3.0])

        self.assertEqual(model.get_booster().last_shape, (1, 3))
        self.assertEqual(str(model.get_booster().last_dtype), "float32")
        self.assertEqual(scores.tolist(), [0.25])

    def test_predict_proba_backend_coerces_1d_input_to_2d_float32_matrix(self) -> None:
        backend = create_inference_backend("predict_proba")
        model = DummyPredictProbaModel()

        scores = backend.predict_positive_proba(model, [1.0, 2.0])

        self.assertEqual(model.last_shape, (1, 2))
        self.assertEqual(str(model.last_dtype), "float32")
        self.assertEqual(scores.tolist(), [0.25])

    def test_config_and_privacy_reflect_versioned_secret_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FRAUD_HASH_KEY_VERSION": "v2",
                "FRAUD_HASH_SECRETS": '{"v1":"old-hash","v2":"new-hash"}',
                "FRAUD_AUDIT_SIGNING_KEY_VERSION": "v3",
                "FRAUD_AUDIT_SIGNING_SECRETS": '{"v2":"old-sign","v3":"new-sign"}',
                "FRAUD_AUDIT_RETENTION_DAYS": "730",
                "FRAUD_AUDIT_DELETION_SLA_DAYS": "45",
                "FRAUD_OPERATOR_AUTH_MODE": "required",
                "FRAUD_OPERATOR_API_KEY": "config-test-key",
            },
            clear=True,
        ):
            reloaded_module = importlib.reload(hybrid_fraud_api)
            client = TestClient(reloaded_module.app)

            config_response = client.get("/config", headers={"X-Operator-Api-Key": "config-test-key"})
            self.assertEqual(config_response.status_code, 200, config_response.text)
            config_body = config_response.json()
            self.assertEqual(config_body["hash_key_version"], "v2")
            self.assertEqual(config_body["hash_secret_source"], "FRAUD_HASH_SECRETS")
            self.assertFalse(config_body["hash_key_version_fallback_used"])
            self.assertEqual(config_body["audit_signing_key_version"], "v3")
            self.assertEqual(config_body["audit_signing_secret_source"], "FRAUD_AUDIT_SIGNING_SECRETS")
            self.assertFalse(config_body["audit_signing_key_version_fallback_used"])
            self.assertIn("inference_backend", config_body)
            self.assertIn("inference_backend_runtime", config_body)

            privacy_response = client.get("/privacy")
            self.assertEqual(privacy_response.status_code, 200, privacy_response.text)
            retention = privacy_response.json()["retention_policy"]
            self.assertEqual(retention["audit_retention_days"], 730)
            self.assertEqual(retention["deletion_sla_days"], 45)
            self.assertEqual(retention["hash_key_version"], "v2")
            self.assertEqual(retention["audit_signing_key_version"], "v3")

        importlib.reload(hybrid_fraud_api)

    def test_apply_entity_smoothing_uses_history_and_is_deterministic(self) -> None:
        from project.data.entity_aggregation import EntitySmoothingConfig, EntitySmoothingState

        payload = self._serving_payload(self.fixtures["approve_case"])
        state = EntitySmoothingState(EntitySmoothingConfig(method="mean", min_history=2))

        with patch.object(hybrid_fraud_api, "ENTITY_SMOOTHING_METHOD", "mean"), patch.object(
            hybrid_fraud_api,
            "entity_smoothing_state",
            state,
        ):
            first, d1 = hybrid_fraud_api.apply_entity_smoothing(0.9, payload)
            second, d2 = hybrid_fraud_api.apply_entity_smoothing(0.3, payload)
            third, d3 = hybrid_fraud_api.apply_entity_smoothing(0.4, payload)

        self.assertAlmostEqual(first, 0.9, places=6)
        self.assertAlmostEqual(second, 0.3, places=6)
        self.assertAlmostEqual(third, 0.6, places=6)
        self.assertTrue(d1["fallback_used"])
        self.assertTrue(d2["fallback_used"])
        self.assertFalse(d3["fallback_used"])

    def test_get_behavior_adjustment_reads_feature_store_when_available(self) -> None:
        payload = self.fixtures["approve_case"]
        fingerprint = hybrid_fraud_api.build_aggregate_input_fingerprint(payload)
        cached = {
            "input_fingerprint": fingerprint,
            "behavior_features": {"behavior_risk_score": 0.4, "is_low_history": False},
            "behavior_adjustment": 0.04,
            "behavior_reasons": ["cached behavior reasons"],
        }
        with patch.object(
            hybrid_fraud_api.feature_store,
            "get_user_aggregates",
            return_value=cached,
        ), patch.object(
            hybrid_fraud_api.behavior_profiler,
            "compute_behavior_features",
        ) as compute_mock:
            result = hybrid_fraud_api.get_behavior_adjustment(payload, cached_aggregates=cached)

        compute_mock.assert_not_called()
        self.assertEqual(result["source"], "feature_store")
        self.assertEqual(result["behavior_reasons"], ["cached behavior reasons"])
        self.assertAlmostEqual(result["behavior_adjustment"], 0.04, places=6)

    def test_get_behavior_adjustment_falls_back_to_on_the_fly_compute(self) -> None:
        payload = self.fixtures["approve_case"]
        features = {"behavior_risk_score": 0.5, "is_low_history": False}
        with patch.object(
            hybrid_fraud_api.behavior_profiler,
            "compute_behavior_features",
            return_value=features,
        ) as compute_mock, patch.object(
            hybrid_fraud_api.behavior_profiler,
            "generate_behavior_reasons",
            return_value=["computed behavior reasons"],
        ) as reasons_mock:
            result = hybrid_fraud_api.get_behavior_adjustment(payload, cached_aggregates=None)

        compute_mock.assert_called_once()
        reasons_mock.assert_called_once_with(features)
        self.assertEqual(result["source"], "on_the_fly")
        self.assertEqual(result["behavior_reasons"], ["computed behavior reasons"])
        self.assertAlmostEqual(result["behavior_adjustment"], 0.05, places=6)


    def test_review_queue_and_outcome_curation_flow(self) -> None:
        payload = self._serving_payload(self.fixtures["flag_case"])
        response = self.client.post("/score_transaction", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        request_id = response.json()["request_id"]

        queue_response = self.client.get("/review_queue?status=pending")
        self.assertEqual(queue_response.status_code, 200, queue_response.text)
        queue_items = queue_response.json()
        self.assertTrue(any(item["request_id"] == request_id for item in queue_items))

        outcome_response = self.client.post(
            f"/review_queue/{request_id}/outcome",
            json={
                "analyst_id": "analyst_1",
                "analyst_decision": "FRAUD",
                "analyst_confidence": 0.92,
                "analyst_notes": "confirmed mule account pattern",
                "transaction_amount": 250.0,
            },
        )
        self.assertEqual(outcome_response.status_code, 200, outcome_response.text)
        outcome_body = outcome_response.json()
        self.assertEqual(outcome_body["status"], "ok")
        self.assertEqual(outcome_body["curated_label"], 1)

        curation_response = self.client.get("/retraining/curation")
        self.assertEqual(curation_response.status_code, 200, curation_response.text)
        self.assertGreaterEqual(curation_response.json()["count"], 1)

    def test_submit_review_outcome_unknown_request_id_returns_stable_error(self) -> None:
        with patch.object(hybrid_fraud_api, "initialize_runtime_artifacts"), patch.object(
            hybrid_fraud_api, "read_jsonl", return_value=[]
        ):
            response = self.client.post(
                "/review_queue/nonexistent-request/outcome",
                json={
                    "analyst_id": "analyst_1",
                    "analyst_decision": "FRAUD",
                    "analyst_confidence": 0.92,
                    "analyst_notes": "missing request id",
                    "transaction_amount": 250.0,
                },
            )
        self.assertEqual(response.status_code, 404, response.text)
        body = response.json()
        self.assertEqual(body["error_category"], "review_queue_record_not_found")
        self.assertIn("Unknown request_id", body["detail"])

    def test_dashboard_views_endpoint_contract(self) -> None:
        original_mode = hybrid_fraud_api.FRAUD_OPERATOR_AUTH_MODE
        try:
            hybrid_fraud_api.FRAUD_OPERATOR_AUTH_MODE = "disabled"
            response = self.client.get("/dashboard/views?window_hours=24")
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertIn("latency_throughput_error", body)
            self.assertIn("drift_score_distribution", body)
            self.assertIn("fraud_loss_false_positives_analyst_agreement", body)
        finally:
            hybrid_fraud_api.FRAUD_OPERATOR_AUTH_MODE = original_mode

    def test_dashboard_views_counts_only_windowed_outcomes(self) -> None:
        now = "2026-04-23T12:00:00+00:00"
        audit_records = [
            {
                "request_id": "req_recent",
                "timestamp_utc": "2026-04-23T11:30:00+00:00",
                "decision": "FLAG",
                "decision_source": "score_band",
                "final_risk_score": 0.62,
                "latency_ms": 101.0,
            }
        ]
        outcome_records = [
            {
                "request_id": "req_recent",
                "reviewed_at_utc": "2026-04-23T11:45:00+00:00",
                "analyst_decision": "FRAUD",
                "model_decision": "FLAG",
                "transaction_amount": 250.0,
            },
            {
                "request_id": "req_old",
                "reviewed_at_utc": "2026-04-20T11:45:00+00:00",
                "analyst_decision": "LEGIT",
                "model_decision": "BLOCK",
                "transaction_amount": 999.0,
            },
        ]

        def fake_read_jsonl(path):
            if path == hybrid_fraud_api.AUDIT_LOG_FILE:
                return audit_records
            if path == hybrid_fraud_api.ANALYST_OUTCOMES_FILE:
                return outcome_records
            return []

        real_datetime = hybrid_fraud_api.datetime

        class FrozenDatetime:
            @staticmethod
            def now(tz=None):
                return real_datetime.fromisoformat(now)

            @staticmethod
            def fromisoformat(value):
                return real_datetime.fromisoformat(value)

        with patch.object(hybrid_fraud_api, "read_jsonl", side_effect=fake_read_jsonl), patch.object(
            hybrid_fraud_api, "datetime", FrozenDatetime
        ):
            summary = hybrid_fraud_api.summarize_dashboard_metrics(window_hours=24)

        fraud_ops = summary["fraud_loss_false_positives_analyst_agreement"]
        self.assertEqual(fraud_ops["analyst_reviews"], 1)
        self.assertEqual(fraud_ops["confirmed_fraud_cases"], 1)
        self.assertEqual(fraud_ops["false_positives"], 0)
        self.assertEqual(fraud_ops["estimated_fraud_loss"], 250.0)

    def test_health_response_includes_security_headers(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers.get("cache-control"), "no-store")
        self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(response.headers.get("x-frame-options"), "DENY")

    def test_sensitive_route_requires_operator_api_key_when_enabled(self) -> None:
        original_mode = hybrid_fraud_api.FRAUD_OPERATOR_AUTH_MODE
        original_key = hybrid_fraud_api.FRAUD_OPERATOR_API_KEY
        try:
            hybrid_fraud_api.FRAUD_OPERATOR_AUTH_MODE = "required"
            hybrid_fraud_api.FRAUD_OPERATOR_API_KEY = "operator-demo-key"

            missing = self.client.get("/dashboard/views?window_hours=24")
            self.assertEqual(missing.status_code, 401, missing.text)

            wrong = self.client.get(
                "/dashboard/views?window_hours=24",
                headers={"X-Operator-Api-Key": "wrong-key"},
            )
            self.assertEqual(wrong.status_code, 403, wrong.text)

            allowed = self.client.get(
                "/dashboard/views?window_hours=24",
                headers={"X-Operator-Api-Key": "operator-demo-key"},
            )
            self.assertEqual(allowed.status_code, 200, allowed.text)
        finally:
            hybrid_fraud_api.FRAUD_OPERATOR_AUTH_MODE = original_mode
            hybrid_fraud_api.FRAUD_OPERATOR_API_KEY = original_key

    def test_metrics_endpoint_returns_prometheus_text(self) -> None:
        original_mode = hybrid_fraud_api.FRAUD_OPERATOR_AUTH_MODE
        original_key = hybrid_fraud_api.FRAUD_OPERATOR_API_KEY
        try:
            hybrid_fraud_api.FRAUD_OPERATOR_AUTH_MODE = "disabled"
            hybrid_fraud_api.FRAUD_OPERATOR_API_KEY = ""

            response = self.client.get("/metrics?window_hours=24")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("fraud_dashboard_requests_total", response.text)
            self.assertIn("fraud_review_queue_pending_total", response.text)
            self.assertIn("fraud_mcp_breaker_open", response.text)
        finally:
            hybrid_fraud_api.FRAUD_OPERATOR_AUTH_MODE = original_mode
            hybrid_fraud_api.FRAUD_OPERATOR_API_KEY = original_key

    def test_ring_graph_requires_operator_api_key_when_enabled(self) -> None:
        original_mode = hybrid_fraud_api.FRAUD_OPERATOR_AUTH_MODE
        original_key = hybrid_fraud_api.FRAUD_OPERATOR_API_KEY
        try:
            hybrid_fraud_api.FRAUD_OPERATOR_AUTH_MODE = "required"
            hybrid_fraud_api.FRAUD_OPERATOR_API_KEY = "operator-demo-key"

            missing = self.client.get("/ring/graph")
            self.assertEqual(missing.status_code, 401, missing.text)

            wrong = self.client.get(
                "/ring/graph",
                headers={"X-Operator-Api-Key": "wrong-key"},
            )
            self.assertEqual(wrong.status_code, 403, wrong.text)

            allowed = self.client.get(
                "/ring/graph",
                headers={"X-Operator-Api-Key": "operator-demo-key"},
            )
            self.assertEqual(allowed.status_code, 200, allowed.text)
        finally:
            hybrid_fraud_api.FRAUD_OPERATOR_AUTH_MODE = original_mode
            hybrid_fraud_api.FRAUD_OPERATOR_API_KEY = original_key


if __name__ == "__main__":
    unittest.main()

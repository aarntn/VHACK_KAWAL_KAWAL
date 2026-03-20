import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import httpx

try:
    from project.app.hybrid_fraud_api import app as fraud_app
    from project.app.domain_exceptions import ReviewQueueRecordNotFoundError
    from project.app import wallet_gateway_api
    from project.app.schemas import ScoreTransactionRequest
except ModuleNotFoundError:
    from app.hybrid_fraud_api import app as fraud_app
    from app.domain_exceptions import ReviewQueueRecordNotFoundError
    import app.wallet_gateway_api as wallet_gateway_api
    from app.schemas import ScoreTransactionRequest


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "fraud_payloads.json"
EXPECTED_WALLET_ACTIONS = {
    "APPROVE": "APPROVED",
    "FLAG": "PENDING_VERIFICATION",
    "BLOCK": "DECLINED_FRAUD_RISK",
}


def load_payload_fixtures() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


class MockFraudResponse:
    def __init__(self, status_code: int, payload: dict, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload


class WalletGatewayApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixtures = load_payload_fixtures()
        cls.fraud_client = TestClient(fraud_app)
        cls.wallet_client = TestClient(wallet_gateway_api.app, raise_server_exceptions=False)

    def setUp(self) -> None:
        wallet_gateway_api._reset_circuit_breaker_for_tests()

    def _wallet_payload(self, case_name: str) -> dict:
        return {
            **self.fixtures[case_name],
            "wallet_id": "wallet_test_001",
            "merchant_name": "Test Merchant",
            "currency": "USD",
        }

    def test_wallet_authorize_payment_and_action_mapping(self) -> None:
        fraud_client = self.fraud_client

        def mocked_post(url, json, headers=None):
            del url
            fraud_response = fraud_client.post("/score_transaction", json=json)
            return MockFraudResponse(fraud_response.status_code, fraud_response.json())

        with patch.object(wallet_gateway_api.UPSTREAM_HTTP, "post", new=AsyncMock(side_effect=mocked_post)):
            for case_name in ("approve_case", "flag_case", "block_case"):
                with self.subTest(case=case_name):
                    response = self.wallet_client.post(
                        "/wallet/authorize_payment",
                        json=self._wallet_payload(case_name),
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    body = response.json()
                    engine_decision = body["fraud_engine_decision"]
                    self.assertEqual(body["wallet_action"], EXPECTED_WALLET_ACTIONS[engine_decision])
                    self.assertFalse(body["fallback_used"])
                    self.assertEqual(body["circuit_breaker_state"], "CLOSED")

    def test_upstream_failure_uses_fallback_policy(self) -> None:
        with patch.object(
            wallet_gateway_api.UPSTREAM_HTTP,
            "post",
            side_effect=httpx.RequestError("network down"),
        ), patch.object(wallet_gateway_api, "UPSTREAM_MAX_RETRIES", 0):
            response = self.wallet_client.post(
                "/wallet/authorize_payment",
                json=self._wallet_payload("approve_case"),
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["fallback_used"])
        self.assertEqual(body["fraud_engine_decision"], wallet_gateway_api.FALLBACK_ENGINE_DECISION)
        self.assertIn("UPSTREAM_ENGINE_UNAVAILABLE_FALLBACK", body["fraud_reasons"])

    def test_non_200_fraud_engine_response_uses_fallback(self) -> None:
        with patch.object(
            wallet_gateway_api.UPSTREAM_HTTP,
            "post",
            new=AsyncMock(return_value=MockFraudResponse(
                status_code=503,
                payload={"error": "engine down"},
                text="service unavailable",
            )),
        ), patch.object(wallet_gateway_api, "UPSTREAM_MAX_RETRIES", 0):
            response = self.wallet_client.post(
                "/wallet/authorize_payment",
                json=self._wallet_payload("flag_case"),
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["fallback_used"])
        self.assertEqual(body["upstream_error"], "status_503")

    def test_wallet_forwards_asean_context_fields_to_fraud_engine(self) -> None:
        captured_payload = {}

        def mocked_post(url, json, headers=None):
            del url
            captured_payload.update(json)
            captured_payload["forwarded_correlation_id"] = (headers or {}).get("x-correlation-id")
            return MockFraudResponse(
                status_code=200,
                payload={
                    "decision": "FLAG",
                    "final_risk_score": 0.75,
                    "reasons": ["test reason"],
                },
            )

        payload = {
            **self._wallet_payload("flag_case"),
            "device_id": "device_shared_123",
            "device_shared_users_24h": 6,
            "account_age_days": 2,
            "sim_change_recent": True,
            "tx_type": "CASH_OUT",
            "channel": "AGENT",
            "cash_flow_velocity_1h": 10,
            "p2p_counterparties_24h": 15,
            "is_cross_border": True,
        }

        with patch.object(wallet_gateway_api.UPSTREAM_HTTP, "post", new=AsyncMock(side_effect=mocked_post)):
            response = self.wallet_client.post("/wallet/authorize_payment", json=payload)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured_payload.get("device_id"), "device_shared_123")
        self.assertEqual(captured_payload.get("device_shared_users_24h"), 6)
        self.assertEqual(captured_payload.get("account_age_days"), 2)
        self.assertTrue(captured_payload.get("sim_change_recent"))
        self.assertEqual(captured_payload.get("tx_type"), "CASH_OUT")
        self.assertEqual(captured_payload.get("channel"), "AGENT")
        self.assertEqual(captured_payload.get("cash_flow_velocity_1h"), 10)
        self.assertEqual(captured_payload.get("p2p_counterparties_24h"), 15)
        self.assertTrue(captured_payload.get("is_cross_border"))
        self.assertIsNotNone(captured_payload.get("forwarded_correlation_id"))
        self.assertEqual(response.json()["wallet_action"], "PENDING_VERIFICATION")

    def test_wallet_forwards_required_ieee_fields_to_fraud_engine(self) -> None:
        captured_payload = {}

        def mocked_post(url, json, headers=None):
            del url, headers
            captured_payload.update(json)
            return MockFraudResponse(
                status_code=200,
                payload={
                    "decision": "APPROVE",
                    "final_risk_score": 0.12,
                    "reasons": ["contract test"],
                },
            )

        with patch.object(wallet_gateway_api.UPSTREAM_HTTP, "post", new=AsyncMock(side_effect=mocked_post)):
            response = self.wallet_client.post(
                "/wallet/authorize_payment",
                json=self._wallet_payload("approve_case"),
            )

        self.assertEqual(response.status_code, 200, response.text)
        for field_name in [
            "schema_version",
            "user_id",
            "transaction_amount",
            "device_risk_score",
            "ip_risk_score",
            "location_risk_score",
        ]:
            with self.subTest(field_name=field_name):
                self.assertIn(field_name, captured_payload)

    def test_wallet_payload_excludes_legacy_v_features(self) -> None:
        captured_payload = {}

        def mocked_post(url, json, headers=None):
            del url, headers
            captured_payload.update(json)
            return MockFraudResponse(
                status_code=200,
                payload={
                    "decision": "APPROVE",
                    "final_risk_score": 0.12,
                    "reasons": ["contract test"],
                },
            )

        with patch.object(wallet_gateway_api.UPSTREAM_HTTP, "post", new=AsyncMock(side_effect=mocked_post)):
            response = self.wallet_client.post(
                "/wallet/authorize_payment",
                json=self._wallet_payload("approve_case"),
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(any(key.startswith("V") for key in captured_payload))

    def test_wallet_forwarding_contract_matches_score_transaction_request_fields(self) -> None:
        payment = wallet_gateway_api.WalletPaymentRequest(
            **{
                **self._wallet_payload("approve_case"),
                "transaction_amount": self.fixtures["approve_case"]["TransactionAmt"],
            }
        )

        forwarded_keys = set(wallet_gateway_api.build_fraud_payload(payment).keys())
        score_request_keys = set(ScoreTransactionRequest.model_fields.keys())

        self.assertSetEqual(
            forwarded_keys,
            score_request_keys,
            "Wallet forwarding payload keys diverged from ScoreTransactionRequest fields.",
        )

    def test_wallet_stage_timings_include_wrapper_overhead_vs_fraud_core(self) -> None:
        with patch.object(
            wallet_gateway_api.UPSTREAM_HTTP,
            "post",
            new=AsyncMock(
                return_value=MockFraudResponse(
                    status_code=200,
                    payload={
                        "decision": "APPROVE",
                        "final_risk_score": 0.11,
                        "reasons": ["contract test"],
                        "stage_timings_ms": {"total_pipeline_ms": 1.25},
                    },
                )
            ),
        ):
            response = self.wallet_client.post(
                "/wallet/authorize_payment",
                json=self._wallet_payload("approve_case"),
            )

        self.assertEqual(response.status_code, 200, response.text)
        stage_details = response.json()["stage_timings_ms"]["details"]
        self.assertIn("input_validation_ms", stage_details)
        self.assertIn("wallet_preprocessing_ms", stage_details)
        self.assertIn("internal_fraud_scoring_call_ms", stage_details)
        self.assertIn("response_mapping_ms", stage_details)
        self.assertIn("wallet_minus_fraud_core_ms", stage_details)
        self.assertGreaterEqual(stage_details["wallet_minus_fraud_core_ms"], 0.0)

    def test_wallet_validation_fails_for_renamed_ieee_fields(self) -> None:
        payload = self._wallet_payload("approve_case")
        payload["Time"] = payload["TransactionDT"]
        payload["Amount"] = payload["TransactionAmt"]
        payload.pop("TransactionDT")
        payload.pop("TransactionAmt")

        response = self.wallet_client.post("/wallet/authorize_payment", json=payload)
        self.assertEqual(response.status_code, 422, response.text)
        detail_text = response.text
        self.assertIn("transaction_amount", detail_text)

    def test_retry_succeeds_after_transient_error(self) -> None:
        calls = {"count": 0}

        def flaky_post(url, json, headers=None):
            del url, json, headers
            calls["count"] += 1
            if calls["count"] == 1:
                return MockFraudResponse(status_code=503, payload={"error": "temporary"})
            return MockFraudResponse(
                status_code=200,
                payload={"decision": "APPROVE", "final_risk_score": 0.1, "reasons": ["ok"]},
            )

        with patch.object(wallet_gateway_api.UPSTREAM_HTTP, "post", new=AsyncMock(side_effect=flaky_post)), patch.object(
            wallet_gateway_api, "UPSTREAM_MAX_RETRIES", 2
        ), patch.object(wallet_gateway_api.asyncio, "sleep", new=AsyncMock(return_value=None)):
            response = self.wallet_client.post(
                "/wallet/authorize_payment",
                json=self._wallet_payload("approve_case"),
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["fraud_engine_decision"], "APPROVE")
        self.assertEqual(body["upstream_attempts"], 2)
        self.assertFalse(body["fallback_used"])

    def test_circuit_breaker_opens_after_repeated_failures(self) -> None:
        original_threshold = wallet_gateway_api.circuit_breaker.failure_threshold
        wallet_gateway_api.circuit_breaker.failure_threshold = 2

        try:
            with patch.object(
                wallet_gateway_api.UPSTREAM_HTTP,
                "post",
                side_effect=httpx.RequestError("down"),
            ), patch.object(wallet_gateway_api, "UPSTREAM_MAX_RETRIES", 0), patch.object(
                wallet_gateway_api.asyncio, "sleep", new=AsyncMock(return_value=None)
            ):
                first = self.wallet_client.post("/wallet/authorize_payment", json=self._wallet_payload("approve_case"))
                second = self.wallet_client.post("/wallet/authorize_payment", json=self._wallet_payload("approve_case"))
                third = self.wallet_client.post("/wallet/authorize_payment", json=self._wallet_payload("approve_case"))

            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(second.json()["circuit_breaker_state"], "OPEN")
            self.assertEqual(third.status_code, 200)
            self.assertEqual(third.json()["upstream_attempts"], 0)
            self.assertEqual(third.json()["upstream_error"], "circuit_open")
        finally:
            wallet_gateway_api.circuit_breaker.failure_threshold = original_threshold

    def test_load_shedding_returns_503_when_inflight_limit_hit(self) -> None:
        with patch.object(wallet_gateway_api._inflight_guard, "acquire", return_value=False):
            response = self.wallet_client.post(
                "/wallet/authorize_payment",
                json=self._wallet_payload("approve_case"),
            )
        self.assertEqual(response.status_code, 503, response.text)

    def test_domain_exceptions_have_stable_categories(self) -> None:
        category = wallet_gateway_api.classify_exception(
            ReviewQueueRecordNotFoundError("missing")
        )
        self.assertEqual(category, "review_queue_record_not_found")


if __name__ == "__main__":
    unittest.main()

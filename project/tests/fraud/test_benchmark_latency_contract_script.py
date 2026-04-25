import unittest

from pydantic import ValidationError

from project.app.contracts import FraudTransactionContract
from project.app.schema_spec import IEEE_V_FEATURE_FIELDS
from project.scripts import benchmark_latency


class BenchmarkLatencyContractTests(unittest.TestCase):
    def test_benchmark_payloads_match_api_contract(self) -> None:
        validation = benchmark_latency.validate_benchmark_payload_contract()
        self.assertTrue(validation["ok"], validation)

    def test_build_base_tx_is_parseable_by_fraud_contract(self) -> None:
        payload = benchmark_latency.build_base_tx()
        parsed = FraudTransactionContract.model_validate(payload)
        self.assertEqual(parsed.TransactionAmt, payload["TransactionAmt"])

    def test_benchmark_payload_unknown_fields_fail_contract_validation(self) -> None:
        payload = benchmark_latency.build_base_tx()
        payload["unexpected_field"] = "boom"

        with self.assertRaises(ValidationError):
            FraudTransactionContract.model_validate(payload)

    def test_benchmark_payload_v_features_track_contract_exactly(self) -> None:
        payload = benchmark_latency.build_base_tx()

        payload_v_fields = {key for key in payload if key.startswith("V")}
        contract_v_fields = {key for key in FraudTransactionContract.model_fields if key.startswith("V")}
        schema_v_fields = set(IEEE_V_FEATURE_FIELDS)

        extra_v_features = payload_v_fields - schema_v_fields

        self.assertEqual(payload_v_fields, contract_v_fields)
        self.assertEqual(payload_v_fields, schema_v_fields)
        self.assertEqual(len(extra_v_features), 0, f"Unexpected extra V-features in benchmark payload: {sorted(extra_v_features)}")

    def test_benchmark_payload_uses_non_zero_realistic_v_features(self) -> None:
        payload = benchmark_latency.build_base_tx(amount=950.0)
        v_values = [float(payload[field]) for field in IEEE_V_FEATURE_FIELDS]
        self.assertTrue(any(abs(value) > 0.0 for value in v_values))


if __name__ == "__main__":
    unittest.main()

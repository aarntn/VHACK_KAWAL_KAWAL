import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from project.app.contracts import FraudTransactionContract


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "fraud_payloads.json"


def load_payload_fixtures() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


class ContractValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixtures = load_payload_fixtures()

    def test_contract_requires_ieee_fields(self) -> None:
        payload = {**self.fixtures["approve_case"]}
        payload.pop("TransactionDT")

        contract_v_fields = [
            field_name for field_name in FraudTransactionContract.model_fields if field_name.startswith("V")
        ]
        highest_v_field = max(contract_v_fields, key=lambda name: int(name[1:]))
        payload.pop(highest_v_field, None)

        with self.assertRaises(ValidationError) as ctx:
            FraudTransactionContract(**payload)

        message = str(ctx.exception)
        self.assertIn("TransactionDT", message)
        self.assertIn(highest_v_field, message)

    def test_contract_rejects_transactiondt_out_of_range(self) -> None:
        payload = {**self.fixtures["approve_case"], "TransactionDT": 20_000_000}

        with self.assertRaises(ValidationError) as ctx:
            FraudTransactionContract(**payload)

        self.assertIn("TransactionDT", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

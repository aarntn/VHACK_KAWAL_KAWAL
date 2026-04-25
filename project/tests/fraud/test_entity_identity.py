import unittest

import pandas as pd

from project.data.entity_identity import build_entity_id


class EntityIdentityTests(unittest.TestCase):
    def test_build_entity_id_ieee_uses_card_tier(self) -> None:
        df = pd.DataFrame(
            {
                "card1": [1111, 1111, 2222],
                "card2": [100, 100, 200],
                "addr1": [10, 10, 20],
                "P_emaildomain": ["a.com", "a.com", "b.com"],
            }
        )

        entity_id, diagnostics = build_entity_id(df, dataset_source="ieee_cis")

        self.assertEqual(entity_id.iloc[0], entity_id.iloc[1])
        self.assertNotEqual(entity_id.iloc[0], entity_id.iloc[2])
        self.assertEqual(diagnostics["rows_total"], 3)

    def test_build_entity_id_creditcard_fallback_when_minimal_columns(self) -> None:
        df = pd.DataFrame({"Amount": [1.0, 2.0], "Time": [10, 20]})
        entity_id, diagnostics = build_entity_id(df, dataset_source="creditcard")
        self.assertEqual(len(entity_id), 2)
        self.assertEqual(diagnostics["rows_total"], 2)
        self.assertGreaterEqual(diagnostics["unique_entities"], 1)


if __name__ == "__main__":
    unittest.main()

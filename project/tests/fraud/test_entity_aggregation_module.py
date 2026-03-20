import unittest

import pandas as pd

from project.data.entity_aggregation import apply_entity_smoothing_batch, build_uid, compute_entity_rolling_aggregates


class EntityAggregationModuleTests(unittest.TestCase):
    def test_blend_smoothing_deterministic(self) -> None:
        df = pd.DataFrame(
            {
                "entity_id": ["u1", "u1", "u2", "u2"],
                "raw_score": [0.9, 0.1, 0.2, 0.8],
                "event_time": [1, 2, 1, 2],
            }
        ).sort_values(["event_time"], kind="mergesort")

        s1 = apply_entity_smoothing_batch(df, method="blend", min_history=1, blend_alpha=0.5, blend_cap=0.2)
        s2 = apply_entity_smoothing_batch(df, method="blend", min_history=1, blend_alpha=0.5, blend_cap=0.2)
        self.assertEqual(s1.tolist(), s2.tolist())

    def test_uid_build_and_rolling_aggregates_are_deterministic(self) -> None:
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 1000, 2000],
                "TransactionAmt": [10.0, 20.0, 30.0],
                "card1": [1111, 1111, 1111],
                "addr1": [100, 100, 100],
            }
        )
        uid = build_uid(df, time_col="TransactionDT", uid_candidates=["card1", "addr1"], time_bucket_seconds=86400)
        agg1 = compute_entity_rolling_aggregates(
            event_time=df["TransactionDT"],
            amount=df["TransactionAmt"],
            entity_id=uid,
            window_seconds=7 * 86400,
            default_recency=86400.0,
        )
        agg2 = compute_entity_rolling_aggregates(
            event_time=df["TransactionDT"],
            amount=df["TransactionAmt"],
            entity_id=uid,
            window_seconds=7 * 86400,
            default_recency=86400.0,
        )
        self.assertListEqual(agg1["count"].tolist(), agg2["count"].tolist())
        self.assertEqual(agg1.loc[0, "count"], 0.0)
        self.assertEqual(agg1.loc[1, "count"], 1.0)
        self.assertEqual(agg1.loc[2, "count"], 2.0)

    def test_rolling_aggregates_handle_duplicate_index_without_loc_slowdown_path(self) -> None:
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 1000, 2000, 3000],
                "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
                "card1": [1111, 1111, 1111, 1111],
                "addr1": [100, 100, 100, 100],
            },
            index=[0, 0, 1, 1],
        )
        uid = build_uid(df, time_col="TransactionDT", uid_candidates=["card1", "addr1"], time_bucket_seconds=86400)
        agg = compute_entity_rolling_aggregates(
            event_time=df["TransactionDT"],
            amount=df["TransactionAmt"],
            entity_id=uid,
            window_seconds=7 * 86400,
            default_recency=86400.0,
        )
        self.assertEqual(len(agg), len(df))
        self.assertListEqual(agg["count"].tolist(), [0.0, 1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()

import unittest

from project.app.feature_store import FeatureStoreConfig, InMemoryFeatureStore


class InMemoryFeatureStoreTests(unittest.TestCase):
    def test_upsert_and_get_with_versioned_window_key(self) -> None:
        clock = {"t": 100.0}
        store = InMemoryFeatureStore(
            config=FeatureStoreConfig(schema_version="s1", aggregation_window_seconds=60),
            ttl_seconds=120,
            now_fn=lambda: clock["t"],
        )

        store.upsert_user_aggregates(
            user_id="user-1",
            as_of_ts=95.0,
            aggregates={"behavior_adjustment": 0.12},
        )
        observed = store.get_user_aggregates("user-1", as_of_ts=100.0)
        self.assertEqual(observed, {"behavior_adjustment": 0.12})

    def test_window_and_ttl_invalidation(self) -> None:
        clock = {"t": 100.0}
        store = InMemoryFeatureStore(
            config=FeatureStoreConfig(schema_version="s1", aggregation_window_seconds=5),
            ttl_seconds=10,
            now_fn=lambda: clock["t"],
        )

        store.upsert_user_aggregates(
            user_id="user-1",
            as_of_ts=100.0,
            aggregates={"context_adjustment": 0.2},
        )
        self.assertIsNone(store.get_user_aggregates("user-1", as_of_ts=120.0))

        store.upsert_user_aggregates(
            user_id="user-1",
            as_of_ts=100.0,
            aggregates={"context_adjustment": 0.3},
        )
        clock["t"] = 111.0
        self.assertIsNone(store.get_user_aggregates("user-1", as_of_ts=111.0))


if __name__ == "__main__":
    unittest.main()

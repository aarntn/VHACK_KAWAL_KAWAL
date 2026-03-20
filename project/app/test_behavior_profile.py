import tempfile
import unittest

from app.behavior_profile import BehaviorProfiler
from app.domain_exceptions import UserProfileMismatchError
from app.profile_store import InMemoryProfileStore, SQLiteProfileStore


class BehaviorProfilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profiler = BehaviorProfiler()

    def test_cold_start_returns_default_safe_scores(self) -> None:
        features = self.profiler.compute_behavior_features(
            user_id="new_user",
            amount=30.0,
            hour_of_day=14,
            location_risk_score=0.2,
        )

        self.assertEqual(features["amount_deviation_score"], 0.10)
        self.assertEqual(features["time_deviation_score"], 0.10)
        self.assertEqual(features["velocity_risk_score"], 0.10)
        self.assertEqual(features["location_deviation_score"], 0.2)
        self.assertAlmostEqual(features["behavior_risk_score"], 0.14, places=4)

    def test_high_amount_and_time_deviation_scores_increase(self) -> None:
        self.profiler.seed_profile(
            user_id="user_001",
            historical_transactions=[
                (25.0, 9, 0.05),
                (18.0, 10, 0.03),
                (32.0, 11, 0.04),
                (21.0, 9, 0.02),
                (27.0, 10, 0.05),
                (19.0, 11, 0.04),
                (23.0, 9, 0.03),
            ],
        )

        features = self.profiler.compute_behavior_features(
            user_id="user_001",
            amount=250.0,
            hour_of_day=2,
            location_risk_score=0.75,
        )

        self.assertGreaterEqual(features["amount_deviation_score"], 0.70)
        self.assertGreaterEqual(features["time_deviation_score"], 0.45)
        self.assertGreaterEqual(features["velocity_risk_score"], 0.20)
        self.assertGreater(features["behavior_risk_score"], 0.5)

    def test_reason_generation_includes_behavioral_signals(self) -> None:
        reasons = self.profiler.generate_behavior_reasons(
            {
                "amount_deviation_score": 0.7,
                "time_deviation_score": 0.45,
                "velocity_risk_score": 0.35,
                "location_deviation_score": 0.75,
                "behavior_risk_score": 0.6,
            }
        )

        self.assertTrue(any("amount" in reason.lower() for reason in reasons))
        self.assertTrue(any("time" in reason.lower() for reason in reasons))
        self.assertTrue(any("burst" in reason.lower() for reason in reasons))
        self.assertTrue(any("location" in reason.lower() for reason in reasons))

    def test_profile_update_and_versioning(self) -> None:
        store = InMemoryProfileStore(ttl_seconds=300)
        profiler = BehaviorProfiler(profile_store=store, aggregate_cache_ttl_seconds=0)

        first = profiler.record_transaction("user_version", 10.0, 9, 0.1)
        second = profiler.record_transaction("user_version", 12.0, 10, 0.2)

        self.assertEqual(first.version, 1)
        self.assertEqual(second.version, 2)
        self.assertEqual(len(second.amounts), 2)

    def test_profile_ttl_expiration_returns_cold_start(self) -> None:
        current_time = 1000

        def now_fn():
            return current_time

        store = InMemoryProfileStore(ttl_seconds=5, now_fn=now_fn)
        profiler = BehaviorProfiler(profile_store=store, aggregate_cache_ttl_seconds=0)

        profiler.seed_profile("ttl_user", [(20.0, 9, 0.1)] * 5)

        current_time = 1010
        features = profiler.compute_behavior_features("ttl_user", 30.0, 10, 0.25)

        self.assertEqual(features["amount_deviation_score"], 0.10)
        self.assertEqual(features["time_deviation_score"], 0.10)

    def test_aggregate_signals_update_with_new_transactions(self) -> None:
        base_time = 10_000.0
        for offset, counterparties in [(0, 2), (120, 3), (260, 4), (3500, 6), (7000, 5)]:
            self.profiler.record_transaction(
                user_id="agg_user",
                amount=50.0 + offset / 100.0,
                hour_of_day=10,
                location_risk_score=0.2,
                event_timestamp=base_time + offset,
                geo_device_mismatch=(offset % 2 == 0),
                counterparties_24h=counterparties,
            )

        features = self.profiler.compute_behavior_features(
            user_id="agg_user",
            amount=100.0,
            hour_of_day=11,
            location_risk_score=0.7,
            event_timestamp=base_time + 7100,
            geo_device_mismatch=True,
            counterparties_24h=10,
        )
        self.assertEqual(features["velocity_count_5m"], 1.0)
        self.assertEqual(features["velocity_count_1h"], 2.0)
        self.assertEqual(features["velocity_count_24h"], 5.0)
        self.assertGreater(features["geo_device_mismatch_rate"], 0.0)
        self.assertGreater(features["counterpart_diversity_delta"], 0.0)

    def test_sqlite_ttl_expiry_removes_profile_state(self) -> None:
        current_time = 5000.0

        def now_fn():
            return current_time

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/profiles.sqlite3"
            store = SQLiteProfileStore(db_path=db_path, ttl_seconds=10, now_fn=now_fn)
            profiler = BehaviorProfiler(profile_store=store)
            profiler.record_transaction(
                user_id="sqlite_ttl_user",
                amount=42.0,
                hour_of_day=9,
                location_risk_score=0.1,
                event_timestamp=current_time,
            )

            current_time = 5012.0
            profile = profiler.get_or_create_profile("sqlite_ttl_user")
            self.assertEqual(profile.amounts, [])
            self.assertEqual(profile.total_transactions, 0)

    def test_deterministic_recomputation_for_same_history(self) -> None:
        tx_sequence = [
            (20.0, 9, 0.05, 1000.0, False, 1),
            (22.0, 10, 0.08, 1300.0, True, 2),
            (18.0, 11, 0.07, 1600.0, False, 1),
            (24.0, 12, 0.09, 2000.0, True, 3),
            (21.0, 9, 0.06, 2400.0, False, 2),
        ]
        profiler_a = BehaviorProfiler()
        profiler_b = BehaviorProfiler()

        for amount, hour, loc, ts, mismatch, counterparties in tx_sequence:
            profiler_a.record_transaction("det_user", amount, hour, loc, ts, mismatch, counterparties)
            profiler_b.record_transaction("det_user", amount, hour, loc, ts, mismatch, counterparties)

        features_a = profiler_a.compute_behavior_features(
            user_id="det_user",
            amount=26.0,
            hour_of_day=13,
            location_risk_score=0.3,
            event_timestamp=2600.0,
            geo_device_mismatch=True,
            counterparties_24h=5,
        )
        features_b = profiler_b.compute_behavior_features(
            user_id="det_user",
            amount=26.0,
            hour_of_day=13,
            location_risk_score=0.3,
            event_timestamp=2600.0,
            geo_device_mismatch=True,
            counterparties_24h=5,
        )
        self.assertDictEqual(features_a, features_b)

    def test_cross_request_consistency_with_sqlite_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/profiles.sqlite3"
            store = SQLiteProfileStore(db_path=db_path, ttl_seconds=600)

            profiler_request_1 = BehaviorProfiler(profile_store=store)
            profiler_request_1.record_transaction("shared_user", 45.0, 8, 0.1)

            profiler_request_2 = BehaviorProfiler(profile_store=store)
            profile = profiler_request_2.get_or_create_profile("shared_user")

            self.assertEqual(profile.amounts, [45.0])
            self.assertEqual(profile.version, 1)

    def test_aggregate_cache_reduces_repeated_store_fetches(self) -> None:
        class CountingStore(InMemoryProfileStore):
            def __init__(self):
                super().__init__(ttl_seconds=300)
                self.get_calls = 0

            def get_profile(self, user_id: str):
                self.get_calls += 1
                return super().get_profile(user_id)

        store = CountingStore()
        profiler = BehaviorProfiler(profile_store=store, aggregate_cache_ttl_seconds=10)
        profiler.seed_profile("cache_user", [(20.0, 9, 0.1)] * 5)

        profiler.compute_behavior_features("cache_user", 25.0, 10, 0.1)
        profiler.compute_behavior_features("cache_user", 26.0, 11, 0.1)

        self.assertEqual(store.get_calls, 1)

    def test_profile_user_id_mismatch_raises_domain_exception(self) -> None:
        class MismatchStore(InMemoryProfileStore):
            def get_profile(self, user_id: str):
                payload = super().get_profile(user_id)
                if payload is None:
                    return None
                corrupted = dict(payload)
                corrupted["user_id"] = "unexpected_user"
                return corrupted

        store = MismatchStore()
        store.save_profile(
            "expected_user",
            {
                "user_id": "expected_user",
                "amounts": [25.0] * 5,
                "hours": [9] * 5,
                "location_risks": [0.2] * 5,
                "event_timestamps": [0.0] * 5,
                "geo_device_mismatch_flags": [0] * 5,
                "counterparties_24h": [1] * 5,
                "total_transactions": 5,
                "geo_device_mismatch_count": 0,
                "payload_schema_version": 2,
                "version": 1,
                "updated_at": 0.0,
            },
        )
        profiler = BehaviorProfiler(profile_store=store)

        with self.assertRaises(UserProfileMismatchError):
            profiler.get_or_create_profile("expected_user")


if __name__ == "__main__":
    unittest.main()

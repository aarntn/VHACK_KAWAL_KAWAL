import unittest

from project.app.aggregate_ingestion import (
    AggregateUpdateEvent,
    MockStreamAggregateIngestionAdapter,
)


class AggregateIngestionTests(unittest.TestCase):
    def test_mock_stream_idempotency_deduplicates_transaction_and_correlation(self) -> None:
        processed = []

        adapter = MockStreamAggregateIngestionAdapter(handler=processed.append, idempotency_ttl_seconds=120.0)
        base = AggregateUpdateEvent(
            transaction_id="tx-1",
            correlation_id="corr-1",
            user_id="user-1",
            amount=10.0,
            hour_of_day=10,
            location_risk_score=0.2,
            observed_at=1000.0,
        )
        adapter.ingest(base)
        adapter.ingest(base)
        adapter.ingest(
            AggregateUpdateEvent(
                transaction_id="tx-2",
                correlation_id="corr-1",
                user_id="user-1",
                amount=20.0,
                hour_of_day=11,
                location_risk_score=0.3,
                observed_at=1001.0,
            )
        )

        processed_count = adapter.flush()
        self.assertEqual(processed_count, 1)
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0].transaction_id, "tx-1")


if __name__ == "__main__":
    unittest.main()

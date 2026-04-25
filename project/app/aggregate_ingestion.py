from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass(frozen=True)
class AggregateUpdateEvent:
    transaction_id: str
    correlation_id: str
    user_id: str
    amount: float
    hour_of_day: int
    location_risk_score: float
    observed_at: float


class IdempotencyWindow:
    def __init__(self, ttl_seconds: float = 300.0):
        self.ttl_seconds = float(ttl_seconds)
        self._lock = threading.Lock()
        self._seen: Dict[str, float] = {}

    def _evict_expired(self, now: float) -> None:
        expired = [key for key, expires_at in self._seen.items() if expires_at <= now]
        for key in expired:
            self._seen.pop(key, None)

    def check_and_mark(self, transaction_id: str, correlation_id: str) -> bool:
        now = time.time()
        dedupe_keys = [f"tx:{transaction_id}", f"corr:{correlation_id}"]
        with self._lock:
            self._evict_expired(now)
            if any(key in self._seen for key in dedupe_keys):
                return False
            expires_at = now + self.ttl_seconds
            for key in dedupe_keys:
                self._seen[key] = expires_at
            return True


class QueueAggregateIngestionAdapter:
    def __init__(
        self,
        *,
        handler: Callable[[AggregateUpdateEvent], None],
        max_queue_size: int = 10_000,
        idempotency_ttl_seconds: float = 300.0,
    ):
        self._handler = handler
        self._queue: queue.Queue[AggregateUpdateEvent] = queue.Queue(maxsize=max_queue_size)
        self._idempotency = IdempotencyWindow(ttl_seconds=idempotency_ttl_seconds)
        self._dropped_events = 0
        self._worker = threading.Thread(target=self._run, name="aggregate-updater", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            try:
                if self._idempotency.check_and_mark(event.transaction_id, event.correlation_id):
                    self._handler(event)
            finally:
                self._queue.task_done()

    def ingest(self, event: AggregateUpdateEvent) -> bool:
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            self._dropped_events += 1
            return False

    def snapshot(self) -> Dict[str, int]:
        return {
            "queued_events": int(self._queue.qsize()),
            "dropped_events": int(self._dropped_events),
        }


class MockStreamAggregateIngestionAdapter:
    def __init__(
        self,
        *,
        handler: Callable[[AggregateUpdateEvent], None],
        idempotency_ttl_seconds: float = 300.0,
    ):
        self._handler = handler
        self._idempotency = IdempotencyWindow(ttl_seconds=idempotency_ttl_seconds)
        self._stream: List[AggregateUpdateEvent] = []

    def ingest(self, event: AggregateUpdateEvent) -> bool:
        self._stream.append(event)
        return True

    def flush(self) -> int:
        processed = 0
        for event in self._stream:
            if self._idempotency.check_and_mark(event.transaction_id, event.correlation_id):
                self._handler(event)
                processed += 1
        self._stream.clear()
        return processed

    def snapshot(self) -> Dict[str, int]:
        return {"queued_events": len(self._stream), "dropped_events": 0}

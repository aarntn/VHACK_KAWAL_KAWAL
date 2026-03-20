from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


DEFAULT_AGGREGATION_WINDOW_SECONDS = int(
    os.getenv("FRAUD_AGGREGATION_WINDOW_SECONDS", "3600")
)
DEFAULT_FEATURE_SCHEMA_VERSION = os.getenv("FRAUD_FEATURE_SCHEMA_VERSION", "v1")


@dataclass(frozen=True)
class FeatureStoreConfig:
    schema_version: str = DEFAULT_FEATURE_SCHEMA_VERSION
    aggregation_window_seconds: int = DEFAULT_AGGREGATION_WINDOW_SECONDS

    @property
    def version_key(self) -> str:
        return f"schema:{self.schema_version}:window:{self.aggregation_window_seconds}"


class FeatureStore(Protocol):
    def get_user_aggregates(self, user_id: str, as_of_ts: float) -> Optional[Dict[str, Any]]:
        ...

    def upsert_user_aggregates(
        self,
        user_id: str,
        as_of_ts: float,
        aggregates: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...


class InMemoryFeatureStore:
    def __init__(
        self,
        config: FeatureStoreConfig,
        ttl_seconds: int = 7200,
        now_fn=None,
    ):
        self.config = config
        self.ttl_seconds = int(ttl_seconds)
        self._now_fn = now_fn or time.time
        self._data: Dict[str, Dict[str, Any]] = {}

    def _record_key(self, user_id: str) -> str:
        return str(user_id).strip()

    def _is_valid(self, record: Dict[str, Any], as_of_ts: float) -> bool:
        if record.get("version_key") != self.config.version_key:
            return False
        expires_at = float(record.get("expires_at", 0.0))
        if expires_at <= self._now_fn():
            return False
        observed_at = float(record.get("observed_at", 0.0))
        return abs(float(as_of_ts) - observed_at) <= self.config.aggregation_window_seconds

    def get_user_aggregates(self, user_id: str, as_of_ts: float) -> Optional[Dict[str, Any]]:
        key = self._record_key(user_id)
        record = self._data.get(key)
        if record is None:
            return None
        if not self._is_valid(record, as_of_ts):
            self._data.pop(key, None)
            return None
        return dict(record["aggregates"])

    def upsert_user_aggregates(
        self,
        user_id: str,
        as_of_ts: float,
        aggregates: Dict[str, Any],
    ) -> Dict[str, Any]:
        now = self._now_fn()
        key = self._record_key(user_id)
        record = {
            "observed_at": float(as_of_ts),
            "expires_at": now + self.ttl_seconds,
            "version_key": self.config.version_key,
            "aggregates": dict(aggregates),
        }
        self._data[key] = record
        return dict(record["aggregates"])


class RedisFeatureStore:
    def __init__(self, redis_client, config: FeatureStoreConfig, ttl_seconds: int = 7200, prefix: str = "feature:agg:"):
        self.redis_client = redis_client
        self.config = config
        self.ttl_seconds = int(ttl_seconds)
        self.prefix = prefix

    def _key(self, user_id: str) -> str:
        return f"{self.prefix}{self.config.version_key}:{user_id}"

    def get_user_aggregates(self, user_id: str, as_of_ts: float) -> Optional[Dict[str, Any]]:
        raw = self.redis_client.get(self._key(user_id))
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        observed_at = float(payload.get("observed_at", 0.0))
        if abs(float(as_of_ts) - observed_at) > self.config.aggregation_window_seconds:
            return None
        return dict(payload.get("aggregates", {}))

    def upsert_user_aggregates(
        self,
        user_id: str,
        as_of_ts: float,
        aggregates: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "observed_at": float(as_of_ts),
            "version_key": self.config.version_key,
            "aggregates": dict(aggregates),
        }
        self.redis_client.setex(self._key(user_id), self.ttl_seconds, json.dumps(payload))
        return dict(aggregates)


class CassandraFeatureStore:
    def __init__(self, session, config: FeatureStoreConfig, keyspace: str = "fraud"):
        self.session = session
        self.config = config
        self.keyspace = keyspace
        self.table = f"{keyspace}.user_feature_aggregates"
        self._init_schema()

    def _init_schema(self) -> None:
        self.session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                version_key text,
                user_id text,
                observed_at double,
                aggregates text,
                PRIMARY KEY ((version_key), user_id)
            )
            """
        )

    def get_user_aggregates(self, user_id: str, as_of_ts: float) -> Optional[Dict[str, Any]]:
        row = self.session.execute(
            f"SELECT observed_at, aggregates FROM {self.table} WHERE version_key=%s AND user_id=%s",
            (self.config.version_key, str(user_id)),
        ).one()
        if row is None:
            return None
        observed_at = float(getattr(row, "observed_at", 0.0))
        if abs(float(as_of_ts) - observed_at) > self.config.aggregation_window_seconds:
            return None
        aggregates_raw = getattr(row, "aggregates", "{}")
        return json.loads(aggregates_raw)

    def upsert_user_aggregates(
        self,
        user_id: str,
        as_of_ts: float,
        aggregates: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.session.execute(
            f"INSERT INTO {self.table} (version_key, user_id, observed_at, aggregates) VALUES (%s, %s, %s, %s)",
            (self.config.version_key, str(user_id), float(as_of_ts), json.dumps(aggregates)),
        )
        return dict(aggregates)

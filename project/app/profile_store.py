from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def normalize_profile_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    normalized.setdefault("payload_schema_version", 2)
    normalized.setdefault("event_timestamps", [])
    normalized.setdefault("geo_device_mismatch_flags", [])
    normalized.setdefault("counterparties_24h", [])
    normalized.setdefault("total_transactions", len(normalized.get("amounts", [])))
    normalized.setdefault("geo_device_mismatch_count", 0)
    return normalized


class InMemoryProfileStore:
    def __init__(self, ttl_seconds: int = 3600, now_fn: Callable[[], float] | None = None):
        self.ttl_seconds = ttl_seconds
        self._now_fn = now_fn or time.time
        self._profiles: Dict[str, dict] = {}

    def _is_expired(self, payload: dict) -> bool:
        return payload.get("expires_at", 0) < self._now_fn()

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        payload = self._profiles.get(user_id)
        if payload is None:
            return None
        if self._is_expired(payload):
            del self._profiles[user_id]
            return None
        return {k: v for k, v in payload.items() if k != "expires_at"}

    def save_profile(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = self._now_fn()
        current = self._profiles.get(user_id)
        next_version = 1 if current is None else int(current.get("version", 0)) + 1

        to_store = normalize_profile_payload(payload)
        to_store["user_id"] = user_id
        to_store["version"] = next_version
        to_store["updated_at"] = now
        to_store["expires_at"] = now + self.ttl_seconds

        self._profiles[user_id] = to_store
        return {k: v for k, v in to_store.items() if k != "expires_at"}

    def close(self) -> None:
        return None


class SQLiteProfileStore:
    def __init__(self, db_path: str | Path, ttl_seconds: int = 3600, now_fn: Callable[[], float] | None = None):
        self.db_path = str(db_path)
        self.ttl_seconds = ttl_seconds
        self._now_fn = now_fn or time.time
        self._init_db()

    def _build_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self._conn = self._build_conn()
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS behavior_profiles (
                    user_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            self._conn.commit()

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        now = self._now_fn()
        with self._lock:
            row = self._conn.execute(
                "SELECT payload, expires_at FROM behavior_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] < now:
                self._conn.execute("DELETE FROM behavior_profiles WHERE user_id = ?", (user_id,))
                self._conn.commit()
                return None
            payload = json.loads(row["payload"])
            return normalize_profile_payload(payload)

    def save_profile(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = self._now_fn()
        expires_at = now + self.ttl_seconds

        with self._lock:
            existing = self._conn.execute(
                "SELECT version FROM behavior_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            next_version = 1 if existing is None else int(existing["version"]) + 1

            to_store = normalize_profile_payload(payload)
            to_store["user_id"] = user_id
            to_store["version"] = next_version
            to_store["updated_at"] = now

            self._conn.execute(
                """
                INSERT INTO behavior_profiles (user_id, payload, version, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    payload = excluded.payload,
                    version = excluded.version,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                """,
                (user_id, json.dumps(to_store), next_version, now, expires_at),
            )
            self._conn.commit()
            return to_store

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class RedisProfileStore:
    def __init__(self, redis_client, ttl_seconds: int = 3600, prefix: str = "behavior:profile:"):
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix

    def _key(self, user_id: str) -> str:
        return f"{self.prefix}{user_id}"

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        raw = self.redis_client.get(self._key(user_id))
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return normalize_profile_payload(json.loads(raw))

    def save_profile(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_profile(user_id)
        next_version = 1 if current is None else int(current.get("version", 0)) + 1

        to_store = normalize_profile_payload(payload)
        to_store["user_id"] = user_id
        to_store["version"] = next_version
        to_store["updated_at"] = time.time()

        self.redis_client.setex(self._key(user_id), self.ttl_seconds, json.dumps(to_store))
        return to_store

    def close(self) -> None:
        close_fn = getattr(self.redis_client, "close", None)
        if callable(close_fn):
            close_fn()

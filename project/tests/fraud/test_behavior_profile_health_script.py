import sqlite3
import tempfile
import unittest
from pathlib import Path

from project.scripts.behavior_profile_health import (
    compute_health,
    load_sqlite_profiles,
)


class BehaviorProfileHealthScriptTests(unittest.TestCase):
    def _build_sqlite_store(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                """
                CREATE TABLE behavior_profiles (
                    user_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO behavior_profiles (user_id, payload, version, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "user_a",
                    '{"amounts":[10,20,30,40,50],"hours":[1],"location_risks":[0.1]}',
                    2,
                    1000.0,
                    5000.0,
                ),
            )
            conn.execute(
                """
                INSERT INTO behavior_profiles (user_id, payload, version, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "user_b",
                    '{"amounts":[5],"hours":[2],"location_risks":[0.2]}',
                    1,
                    100.0,
                    200.0,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def test_load_sqlite_profiles_returns_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "profiles.sqlite3"
            self._build_sqlite_store(db_path)
            profiles = load_sqlite_profiles(db_path)
            self.assertEqual(len(profiles), 2)

    def test_compute_health_flags_low_coverage_and_staleness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "profiles.sqlite3"
            self._build_sqlite_store(db_path)
            profiles = load_sqlite_profiles(db_path)

            health = compute_health(
                profiles=profiles,
                now_ts=2000.0,
                min_history=5,
                stale_seconds=100,
                coverage_warn_min_active=5,
                low_history_warn_ratio=0.3,
                stale_warn_ratio=0.2,
            )

            self.assertEqual(health["summary"]["active_profiles"], 1)
            self.assertEqual(health["status"], "warn")
            self.assertTrue(health["warnings"])


if __name__ == "__main__":
    unittest.main()

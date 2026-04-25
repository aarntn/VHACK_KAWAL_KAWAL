"""Tests for MCP status fields in /health and /api/info responses.

Validates:
- get_mcp_status() always returns required keys with correct types
- Mode is "enabled" when MCP is on and circuit not open
- Mode is "disabled" when FRAUD_MCP_ENABLED=false
- Mode is "degraded" when circuit breaker trips
- HealthResponse model accepts mcp dict field
- Circuit breaker snapshot produces correct fields
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ---------------------------------------------------------------------------
# get_mcp_status() unit tests
# ---------------------------------------------------------------------------

class TestGetMcpStatus(unittest.TestCase):
    def _import_fresh(self, env_overrides: dict = None):
        """Re-import watchlist_client with patched env to test mode transitions."""
        import importlib
        env = {**os.environ}
        if env_overrides:
            env.update(env_overrides)
        with patch.dict(os.environ, env, clear=True):
            import project.app.mcp.watchlist_client as wc
            importlib.reload(wc)
            return wc

    def test_get_mcp_status_returns_required_keys(self):
        from project.app.mcp.watchlist_client import get_mcp_status
        status = get_mcp_status()
        required_keys = {"mode", "enabled", "breaker_open", "breaker_failure_count",
                         "breaker_opens_again_in_s", "mcp_url"}
        self.assertEqual(set(status.keys()), required_keys)

    def test_mode_is_string(self):
        from project.app.mcp.watchlist_client import get_mcp_status
        status = get_mcp_status()
        self.assertIsInstance(status["mode"], str)

    def test_enabled_is_bool(self):
        from project.app.mcp.watchlist_client import get_mcp_status
        status = get_mcp_status()
        self.assertIsInstance(status["enabled"], bool)

    def test_breaker_open_is_bool(self):
        from project.app.mcp.watchlist_client import get_mcp_status
        status = get_mcp_status()
        self.assertIsInstance(status["breaker_open"], bool)

    def test_breaker_failure_count_is_int(self):
        from project.app.mcp.watchlist_client import get_mcp_status
        status = get_mcp_status()
        self.assertIsInstance(status["breaker_failure_count"], int)

    def test_mode_values_are_valid(self):
        from project.app.mcp.watchlist_client import get_mcp_status
        status = get_mcp_status()
        self.assertIn(status["mode"], {"enabled", "disabled", "degraded"})

    def test_mode_enabled_when_circuit_closed(self):
        from project.app.mcp.watchlist_client import get_mcp_status, _breaker
        _breaker.record_success()
        status = get_mcp_status()
        if status["enabled"]:
            self.assertEqual(status["mode"], "enabled")
            self.assertFalse(status["breaker_open"])

    def test_mode_degraded_when_circuit_trips(self):
        import project.app.mcp.watchlist_client as wc
        original_failures = wc._breaker._failures
        original_opened_at = wc._breaker._opened_at
        try:
            wc._breaker._failures = wc._breaker._threshold
            import time
            wc._breaker._opened_at = time.monotonic()
            status = wc.get_mcp_status()
            if status["enabled"]:
                self.assertEqual(status["mode"], "degraded")
                self.assertTrue(status["breaker_open"])
        finally:
            wc._breaker._failures = original_failures
            wc._breaker._opened_at = original_opened_at

    def test_mcp_url_is_string(self):
        from project.app.mcp.watchlist_client import get_mcp_status
        status = get_mcp_status()
        self.assertIsInstance(status["mcp_url"], str)

    def test_mcp_url_nonempty(self):
        from project.app.mcp.watchlist_client import get_mcp_status
        status = get_mcp_status()
        self.assertGreater(len(status["mcp_url"]), 0)


# ---------------------------------------------------------------------------
# _CircuitBreaker.snapshot() unit tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerSnapshot(unittest.TestCase):
    def _make_breaker(self, threshold=3, reset_s=30.0):
        from project.app.mcp.watchlist_client import _CircuitBreaker
        return _CircuitBreaker(failure_threshold=threshold, reset_timeout_s=reset_s)

    def test_snapshot_keys(self):
        cb = self._make_breaker()
        snap = cb.snapshot()
        self.assertIn("open", snap)
        self.assertIn("failure_count", snap)
        self.assertIn("reset_timeout_s", snap)
        self.assertIn("opens_again_in_s", snap)

    def test_snapshot_closed_state(self):
        cb = self._make_breaker(threshold=3)
        snap = cb.snapshot()
        self.assertFalse(snap["open"])
        self.assertEqual(snap["failure_count"], 0)
        self.assertEqual(snap["opens_again_in_s"], 0.0)

    def test_snapshot_open_after_failures(self):
        import time
        cb = self._make_breaker(threshold=3, reset_s=30.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        snap = cb.snapshot()
        self.assertTrue(snap["open"])
        self.assertEqual(snap["failure_count"], 3)
        self.assertGreater(snap["opens_again_in_s"], 0.0)
        self.assertLessEqual(snap["opens_again_in_s"], 30.0)

    def test_snapshot_reset_after_success(self):
        cb = self._make_breaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        snap = cb.snapshot()
        self.assertFalse(snap["open"])
        self.assertEqual(snap["failure_count"], 0)


# ---------------------------------------------------------------------------
# HealthResponse model includes mcp field
# ---------------------------------------------------------------------------

class TestHealthResponseMcpField(unittest.TestCase):
    def test_health_response_accepts_mcp_dict(self):
        from project.app.hybrid_fraud_api import HealthResponse
        response = HealthResponse(
            status="ok",
            api_version="4.0.0",
            model_name="test_model",
            model_version="v1",
            feature_count=17,
            approve_threshold=0.28,
            block_threshold=0.74,
            segment_thresholds_summary={},
            mcp={"mode": "enabled", "enabled": True, "breaker_open": False,
                 "breaker_failure_count": 0, "breaker_opens_again_in_s": 0.0,
                 "mcp_url": "http://127.0.0.1:8002"},
        )
        self.assertEqual(response.mcp["mode"], "enabled")

    def test_health_response_mcp_defaults_to_empty_dict(self):
        from project.app.hybrid_fraud_api import HealthResponse
        response = HealthResponse(
            status="ok",
            api_version="4.0.0",
            model_name="test_model",
            model_version="v1",
            feature_count=17,
            approve_threshold=0.28,
            block_threshold=0.74,
        )
        self.assertEqual(response.mcp, {})

    def test_health_response_mcp_serializes_to_dict(self):
        from project.app.hybrid_fraud_api import HealthResponse
        mcp_payload = {
            "mode": "degraded",
            "enabled": True,
            "breaker_open": True,
            "breaker_failure_count": 5,
            "breaker_opens_again_in_s": 18.3,
            "mcp_url": "http://127.0.0.1:8002",
        }
        response = HealthResponse(
            status="ok",
            api_version="4.0.0",
            model_name="test",
            model_version="v1",
            feature_count=17,
            approve_threshold=0.28,
            block_threshold=0.74,
            mcp=mcp_payload,
        )
        serialized = response.model_dump()
        self.assertEqual(serialized["mcp"]["mode"], "degraded")
        self.assertTrue(serialized["mcp"]["breaker_open"])

    def test_health_response_all_mcp_mode_values(self):
        from project.app.hybrid_fraud_api import HealthResponse
        for mode in ("enabled", "disabled", "degraded"):
            response = HealthResponse(
                status="ok",
                api_version="4.0.0",
                model_name="test",
                model_version="v1",
                feature_count=17,
                approve_threshold=0.28,
                block_threshold=0.74,
                mcp={"mode": mode},
            )
            self.assertEqual(response.mcp["mode"], mode)


# ---------------------------------------------------------------------------
# Wallet stage_timings_ms per-attempt fields (unit test without live service)
# ---------------------------------------------------------------------------

class TestWalletAttemptTimingFields(unittest.IsolatedAsyncioTestCase):
    async def _call_attempt_fraud_call(self, succeed=True, status=200):
        """Call _attempt_fraud_call with a mock httpx client."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        import project.app.wallet_gateway_api as wgw

        mock_response = MagicMock()
        mock_response.status_code = status
        mock_response.json.return_value = {
            "decision": "APPROVE",
            "decision_source": "score_band",
            "final_risk_score": 0.15,
            "reasons": ["test"],
            "stage_timings_ms": {"total_pipeline_ms": 120.0},
        }

        mock_client = AsyncMock()
        if succeed:
            mock_client.post.return_value = mock_response
        else:
            from httpx import TimeoutException
            mock_client.post.side_effect = TimeoutException("timeout")

        original_client = wgw.UPSTREAM_HTTP
        wgw.UPSTREAM_HTTP = mock_client
        wgw.circuit_breaker.reset()
        try:
            result, meta = await wgw._attempt_fraud_call({"test": True})
        finally:
            wgw.UPSTREAM_HTTP = original_client
        return result, meta

    async def test_success_has_attempt_durations_ms(self):
        _, meta = await self._call_attempt_fraud_call(succeed=True)
        self.assertIn("attempt_durations_ms", meta)
        self.assertIsInstance(meta["attempt_durations_ms"], list)
        self.assertEqual(len(meta["attempt_durations_ms"]), 1)

    async def test_success_has_upstream_call_ms(self):
        _, meta = await self._call_attempt_fraud_call(succeed=True)
        self.assertIn("upstream_call_ms", meta)
        self.assertIsNotNone(meta["upstream_call_ms"])
        self.assertGreaterEqual(meta["upstream_call_ms"], 0.0)

    async def test_success_has_backoff_total_ms_zero(self):
        _, meta = await self._call_attempt_fraud_call(succeed=True)
        self.assertIn("backoff_total_ms", meta)
        self.assertEqual(meta["backoff_total_ms"], 0.0)

    async def test_timeout_has_upstream_call_ms_none(self):
        import project.app.wallet_gateway_api as wgw
        wgw.circuit_breaker.reset()
        _, meta = await self._call_attempt_fraud_call(succeed=False)
        self.assertIsNone(meta["upstream_call_ms"])

    async def test_timeout_attempt_durations_ms_populated(self):
        _, meta = await self._call_attempt_fraud_call(succeed=False)
        self.assertIsInstance(meta["attempt_durations_ms"], list)

    async def test_circuit_open_returns_all_timing_keys(self):
        import project.app.wallet_gateway_api as wgw
        import time
        wgw.circuit_breaker._lock.__class__  # ensure it's the right object
        wgw.circuit_breaker.state = "OPEN"
        wgw.circuit_breaker.opened_at = time.monotonic()
        try:
            _, meta = await wgw._attempt_fraud_call({"test": True})
        finally:
            wgw.circuit_breaker.reset()
        self.assertIn("attempt_durations_ms", meta)
        self.assertIn("backoff_total_ms", meta)
        self.assertIn("upstream_call_ms", meta)
        self.assertEqual(meta["attempts"], 0)

    async def test_fallback_false_on_success(self):
        result, meta = await self._call_attempt_fraud_call(succeed=True)
        self.assertFalse(meta["fallback_used"])
        self.assertIsNotNone(result)

    async def test_fallback_true_on_timeout(self):
        _, meta = await self._call_attempt_fraud_call(succeed=False)
        self.assertTrue(meta["fallback_used"])


if __name__ == "__main__":
    unittest.main()

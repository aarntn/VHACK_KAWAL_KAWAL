"""
MCP external signal client — production-grade edition.

Architecture notes
──────────────────
Fix 1 — Non-blocking hot path
  score_transaction is a sync FastAPI handler running in uvicorn's thread pool.
  Blocking requests.get() there does NOT block the event loop, but it does
  consume a thread for the full round-trip. The TTL cache (Fix 2) eliminates
  most live calls so thread-hold time under load becomes negligible.

Fix 2 — TTL cache + circuit breaker
  TTLCache:        account results cached for FRAUD_MCP_CACHE_TTL_S (default 60s).
                   Eliminates redundant lookups for the same account within a scoring window.
  CircuitBreaker:  after FRAUD_MCP_CB_FAILURES consecutive failures, the circuit opens
                   and all calls return neutral for FRAUD_MCP_CB_RESET_S (default 30s)
                   before a probe is allowed. Prevents per-request overhead when MCP is down.

Fix 3 — Governance: all adjustment weights are env-overridable and documented in
  project/outputs/governance/mcp_weight_rationale.json (written by this module at startup).

Fix 4 — API key auth
  If FRAUD_MCP_API_KEY is set, every request carries X-MCP-Api-Key: <key>.
  The mock server and any real MCP endpoint validate this header.

Environment variables
──────────────────────
  FRAUD_MCP_URL               MCP server base URL  (default: http://127.0.0.1:8002)
  FRAUD_MCP_TIMEOUT_MS        Per-request timeout ms (default: 50)
  FRAUD_MCP_ENABLED           Set "false" to disable entirely (default: true)
  FRAUD_MCP_API_KEY           Bearer key for X-MCP-Api-Key header (default: empty = no auth)
  FRAUD_MCP_CACHE_TTL_S       Cache TTL in seconds (default: 60.0)
  FRAUD_MCP_CACHE_MAXSIZE     Max cached entries (default: 2000)
  FRAUD_MCP_CB_FAILURES       Circuit-open threshold (default: 5)
  FRAUD_MCP_CB_RESET_S        Circuit-reset window in seconds (default: 30.0)
  FRAUD_MCP_BOOST_HIGH        Adjustment for high-tier watchlist hit (default: 0.08)
  FRAUD_MCP_BOOST_MED         Adjustment for medium-tier watchlist hit (default: 0.05)
  FRAUD_MCP_BOOST_LOW         Adjustment for low-tier watchlist hit (default: 0.02)
  FRAUD_MCP_CLEAN_DISCOUNT    Adjustment for verified-clean device (default: -0.02)
  FRAUD_MCP_SHARED_BOOST      Adjustment for device shared by 5+ accounts (default: 0.04)
  FRAUD_MCP_ADJUSTMENT_CAP    Max absolute external adjustment (default: 0.10)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import requests

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_MCP_BASE_URL  = os.getenv("FRAUD_MCP_URL", "http://127.0.0.1:8002")
_MCP_TIMEOUT_S = float(os.getenv("FRAUD_MCP_TIMEOUT_MS", "50")) / 1000.0
_MCP_ENABLED   = os.getenv("FRAUD_MCP_ENABLED", "true").lower() not in ("false", "0", "no")
_MCP_API_KEY   = os.getenv("FRAUD_MCP_API_KEY", "")

_CACHE_TTL_S   = float(os.getenv("FRAUD_MCP_CACHE_TTL_S",   "60.0"))
_CACHE_MAXSIZE = int(os.getenv("FRAUD_MCP_CACHE_MAXSIZE",   "2000"))
_CB_FAILURES   = int(os.getenv("FRAUD_MCP_CB_FAILURES",     "5"))
_CB_RESET_S    = float(os.getenv("FRAUD_MCP_CB_RESET_S",    "30.0"))

# Adjustment weights — all env-overridable for calibration governance
_BOOST_HIGH       = float(os.getenv("FRAUD_MCP_BOOST_HIGH",      "0.08"))
_BOOST_MED        = float(os.getenv("FRAUD_MCP_BOOST_MED",       "0.05"))
_BOOST_LOW        = float(os.getenv("FRAUD_MCP_BOOST_LOW",       "0.02"))
_CLEAN_DISCOUNT   = float(os.getenv("FRAUD_MCP_CLEAN_DISCOUNT",  "-0.02"))
_SHARED_BOOST     = float(os.getenv("FRAUD_MCP_SHARED_BOOST",    "0.04"))
_ADJUSTMENT_CAP   = float(os.getenv("FRAUD_MCP_ADJUSTMENT_CAP",  "0.10"))

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class MCPWatchlistResult:
    watchlist_hit:       bool  = False
    risk_tier:           str   = "clean"
    device_risk_score:   float = 0.0
    device_shared_count: int   = 0
    external_adjustment: float = 0.0
    source:              str   = "unavailable"
    latency_ms:          float = 0.0
    cache_hit:           bool  = False

_NEUTRAL_DISABLED  = MCPWatchlistResult(source="disabled")
_NEUTRAL_OPEN      = MCPWatchlistResult(source="circuit_open")

# ---------------------------------------------------------------------------
# TTL cache (no external deps)
# ---------------------------------------------------------------------------

class _TTLCache:
    """Thread-safe TTL cache keyed by (account_id, device_id)."""

    def __init__(self, maxsize: int, ttl: float) -> None:
        self._store: dict[str, Tuple[MCPWatchlistResult, float]] = {}
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[MCPWatchlistResult]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            result, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            cached = MCPWatchlistResult(**result.__dict__)
            cached.cache_hit = True
            return cached

    def set(self, key: str, value: MCPWatchlistResult) -> None:
        with self._lock:
            if len(self._store) >= self._maxsize:
                oldest_key = min(self._store, key=lambda k: self._store[k][1])
                del self._store[oldest_key]
            self._store[key] = (value, time.monotonic() + self._ttl)

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class _CircuitBreaker:
    """
    Simple three-state circuit breaker: CLOSED → OPEN → HALF-OPEN → CLOSED.

    Thread-safe. Does NOT require external libraries.
    """

    def __init__(self, failure_threshold: int, reset_timeout_s: float) -> None:
        self._failures   = 0
        self._opened_at  = 0.0
        self._threshold  = failure_threshold
        self._reset      = reset_timeout_s
        self._lock       = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._failures >= self._threshold:
                elapsed = time.monotonic() - self._opened_at
                if elapsed < self._reset:
                    return True
                # Half-open: let one probe through
                self._failures = self._threshold - 1
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._opened_at = time.monotonic()
                logger.warning(
                    "MCP circuit opened after %d consecutive failures — "
                    "suppressing calls for %.0fs",
                    self._threshold, self._reset,
                )

    @property
    def failure_count(self) -> int:
        return self._failures

    def snapshot(self) -> dict:
        with self._lock:
            open_flag = self._failures >= self._threshold
            elapsed = time.monotonic() - self._opened_at if open_flag else 0.0
            remaining = max(0.0, self._reset - elapsed) if open_flag else 0.0
            return {
                "open": open_flag and remaining > 0,
                "failure_count": self._failures,
                "reset_timeout_s": self._reset,
                "opens_again_in_s": round(remaining, 1),
            }


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_cache    = _TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_CACHE_TTL_S)
_breaker  = _CircuitBreaker(failure_threshold=_CB_FAILURES, reset_timeout_s=_CB_RESET_S)

_session  = requests.Session()
_session.headers.update({"Accept": "application/json"})
if _MCP_API_KEY:
    _session.headers["X-MCP-Api-Key"] = _MCP_API_KEY

# Async HTTP client — created lazily so it lives inside the running event loop.
_async_headers: dict[str, str] = {"Accept": "application/json"}
if _MCP_API_KEY:
    _async_headers["X-MCP-Api-Key"] = _MCP_API_KEY
_async_client: Optional["httpx.AsyncClient"] = None
_async_client_lock = threading.Lock()


def _get_async_client() -> "httpx.AsyncClient":
    global _async_client
    if _async_client is None:
        with _async_client_lock:
            if _async_client is None:
                if not _HAS_HTTPX:
                    raise ImportError("httpx is required for async MCP calls: pip install httpx")
                _async_client = httpx.AsyncClient(
                    headers=_async_headers,
                    timeout=_MCP_TIMEOUT_S,
                )
    return _async_client

# ---------------------------------------------------------------------------
# Governance: write weight rationale at module import (once)
# ---------------------------------------------------------------------------

def _write_weight_rationale() -> None:
    try:
        out = Path(__file__).resolve().parents[3] / "project" / "outputs" / "governance" / "mcp_weight_rationale.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        rationale = {
            "description": (
                "MCP external signal adjustment weights. All values are env-overridable "
                "for calibration. Weights were set to be proportional to the expected "
                "incremental fraud-lift observed per risk tier in the ring ablation study "
                "(see ring_ablation_report.json: high-risk ring cohorts showed +21.6% recall "
                "lift). External weights are intentionally conservative (max +0.08) so MCP "
                "signals augment rather than override the base model."
            ),
            "weights": {
                "BOOST_HIGH":     {"value": _BOOST_HIGH,     "env": "FRAUD_MCP_BOOST_HIGH",     "rationale": "High-tier hit: ring_score>=0.70; aligned with top-ring fraud rate 68.8%"},
                "BOOST_MED":      {"value": _BOOST_MED,      "env": "FRAUD_MCP_BOOST_MED",       "rationale": "Medium-tier hit: ring_score 0.40–0.70; moderate fraud signal"},
                "BOOST_LOW":      {"value": _BOOST_LOW,      "env": "FRAUD_MCP_BOOST_LOW",       "rationale": "Low-tier: borderline ring; minimal uplift"},
                "CLEAN_DISCOUNT": {"value": _CLEAN_DISCOUNT, "env": "FRAUD_MCP_CLEAN_DISCOUNT",  "rationale": "Verified-clean device reduces FPR for established low-risk users"},
                "SHARED_BOOST":   {"value": _SHARED_BOOST,   "env": "FRAUD_MCP_SHARED_BOOST",    "rationale": "5+ shared device accounts: shared_device cohort showed +19.4% recall lift"},
                "ADJUSTMENT_CAP": {"value": _ADJUSTMENT_CAP, "env": "FRAUD_MCP_ADJUSTMENT_CAP",  "rationale": "Hard cap: MCP signals must not dominate base model output"},
            },
            "calibration_status": "heuristic_with_ablation_evidence",
            "next_step": "Re-run calibrate_mcp_weights.py with labeled holdout to derive data-driven weights",
        }
        out.write_text(json.dumps(rationale, indent=2), encoding="utf-8")
    except Exception:
        pass  # governance write is never allowed to break the import

_write_weight_rationale()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup_external_signals(
    account_id: str,
    device_id: Optional[str] = None,
) -> MCPWatchlistResult:
    """
    Look up account + device external signals. Never raises.

    Returns:
        MCPWatchlistResult with external_adjustment in [−cap, +cap].
        source is one of: fraud_ring_registry | timeout | unavailable |
                          circuit_open | disabled | error | cache_hit(implied by cache_hit=True)
    """
    if not _MCP_ENABLED:
        return _NEUTRAL_DISABLED

    if _breaker.is_open:
        return _NEUTRAL_OPEN

    cache_key = f"{account_id}|{device_id or ''}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    result = MCPWatchlistResult()
    t0 = time.perf_counter()

    try:
        # Watchlist lookup
        resp = _session.get(
            f"{_MCP_BASE_URL}/watchlist/{account_id}",
            timeout=_MCP_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()

        result.watchlist_hit = bool(data.get("hit", False))
        result.risk_tier     = str(data.get("risk_tier", "clean"))
        result.source        = str(data.get("source", "mcp_watchlist"))

        if result.watchlist_hit:
            boost = {
                "high":   _BOOST_HIGH,
                "medium": _BOOST_MED,
                "low":    _BOOST_LOW,
            }.get(result.risk_tier, _BOOST_LOW)
            result.external_adjustment += boost

        # Device reputation (best-effort, won't fail the whole call)
        if device_id:
            try:
                dresp = _session.get(
                    f"{_MCP_BASE_URL}/device/{device_id}",
                    timeout=_MCP_TIMEOUT_S,
                )
                dresp.raise_for_status()
                ddata = dresp.json()
                result.device_risk_score   = float(ddata.get("risk_score", 0.0))
                result.device_shared_count = int(ddata.get("shared_count", 0))

                if result.device_shared_count >= 5:
                    result.external_adjustment += _SHARED_BOOST
                elif result.device_risk_score < 0.1 and result.device_shared_count == 0:
                    result.external_adjustment += _CLEAN_DISCOUNT
            except Exception:
                pass

        _breaker.record_success()

    except requests.exceptions.Timeout:
        result.source = "timeout"
        _breaker.record_failure()
    except requests.exceptions.ConnectionError:
        result.source = "unavailable"
        _breaker.record_failure()
    except Exception as exc:
        logger.debug("MCP lookup error account=%s: %s", account_id, exc)
        result.source = "error"
        _breaker.record_failure()

    result.external_adjustment = max(-_ADJUSTMENT_CAP, min(_ADJUSTMENT_CAP, result.external_adjustment))
    result.latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    _cache.set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Async public API — uses httpx + asyncio.gather for parallel watchlist+device
# ---------------------------------------------------------------------------

async def lookup_external_signals_async(
    account_id: str,
    device_id: Optional[str] = None,
) -> MCPWatchlistResult:
    """
    Async version of lookup_external_signals.

    Runs watchlist and device lookups in parallel via asyncio.gather.
    Falls back gracefully on any error. Shares the same TTL cache and
    circuit breaker as the sync version.
    """
    if not _MCP_ENABLED:
        return _NEUTRAL_DISABLED

    if _breaker.is_open:
        return _NEUTRAL_OPEN

    cache_key = f"{account_id}|{device_id or ''}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    result = MCPWatchlistResult()
    t0 = time.perf_counter()
    client = _get_async_client()

    async def _fetch_watchlist() -> dict:
        r = await client.get(f"{_MCP_BASE_URL}/watchlist/{account_id}")
        r.raise_for_status()
        return r.json()

    async def _fetch_device() -> dict:
        r = await client.get(f"{_MCP_BASE_URL}/device/{device_id}")
        r.raise_for_status()
        return r.json()

    try:
        tasks = [asyncio.create_task(_fetch_watchlist())]
        if device_id:
            tasks.append(asyncio.create_task(_fetch_device()))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Watchlist result (always index 0)
        wl = responses[0]
        if isinstance(wl, Exception):
            raise wl
        result.watchlist_hit = bool(wl.get("hit", False))
        result.risk_tier     = str(wl.get("risk_tier", "clean"))
        result.source        = str(wl.get("source", "mcp_watchlist"))
        if result.watchlist_hit:
            boost = {"high": _BOOST_HIGH, "medium": _BOOST_MED, "low": _BOOST_LOW}.get(
                result.risk_tier, _BOOST_LOW
            )
            result.external_adjustment += boost

        # Device result (index 1, optional)
        if device_id and len(responses) > 1:
            ddata = responses[1]
            if not isinstance(ddata, Exception):
                result.device_risk_score   = float(ddata.get("risk_score", 0.0))
                result.device_shared_count = int(ddata.get("shared_count", 0))
                if result.device_shared_count >= 5:
                    result.external_adjustment += _SHARED_BOOST
                elif result.device_risk_score < 0.1 and result.device_shared_count == 0:
                    result.external_adjustment += _CLEAN_DISCOUNT

        _breaker.record_success()

    except Exception as exc:
        import httpx as _httpx
        if isinstance(exc, _httpx.TimeoutException):
            result.source = "timeout"
        elif isinstance(exc, _httpx.ConnectError):
            result.source = "unavailable"
        else:
            logger.debug("MCP async lookup error account=%s: %s", account_id, exc)
            result.source = "error"
        _breaker.record_failure()

    result.external_adjustment = max(-_ADJUSTMENT_CAP, min(_ADJUSTMENT_CAP, result.external_adjustment))
    result.latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    _cache.set(cache_key, result)
    return result


def get_mcp_status() -> dict:
    """Return a health snapshot of the MCP subsystem for /health endpoints."""
    breaker = _breaker.snapshot()
    if not _MCP_ENABLED:
        mode = "disabled"
    elif breaker["open"]:
        mode = "degraded"
    else:
        mode = "enabled"
    return {
        "mode": mode,
        "enabled": _MCP_ENABLED,
        "breaker_open": breaker["open"],
        "breaker_failure_count": breaker["failure_count"],
        "breaker_opens_again_in_s": breaker["opens_again_in_s"],
        "mcp_url": _MCP_BASE_URL,
    }

"""
Mock MCP External Signal Server — port 8002

Simulates the external data sources a production fraud engine would integrate:
  - Watchlist registry  GET /watchlist/{account_id}
  - Device reputation   GET /device/{device_id}
  - Health              GET /health

Data strategy:
  - Accounts in fraud_ring_scores.json with ring_score >= 0.40 are flagged on the
    watchlist, tiered by score magnitude (high/medium/low).
  - Device reputation is deterministic from device_id hash — provides varied but
    stable risk scores across demo runs.
  - Unknown accounts always return a 'clean' watchlist result so legitimate users
    are never falsely flagged.

Usage:
    python project/scripts/mock_mcp_server.py
    python project/scripts/mock_mcp_server.py --port 8002 --host 127.0.0.1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader

REPO_ROOT        = Path(__file__).resolve().parents[2]
RING_SCORES_PATH = REPO_ROOT / "project" / "outputs" / "monitoring" / "fraud_ring_scores.json"

# API key auth — set FRAUD_MCP_API_KEY env var on both client and server to enable.
# Empty string = auth disabled (dev/demo mode).
_SERVER_API_KEY = os.getenv("FRAUD_MCP_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-MCP-Api-Key", auto_error=False)

app = FastAPI(
    title="Mock MCP External Signal Server",
    description="Simulates watchlist and device reputation for the fraud scoring demo",
    version="1.0.0",
)


def _require_api_key(key: str = Security(_api_key_header)) -> None:
    """Dependency: validate API key when the server is configured with one."""
    if _SERVER_API_KEY and key != _SERVER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-MCP-Api-Key")

# ---------------------------------------------------------------------------
# Load ring scores at module level so lookups are O(1)
# ---------------------------------------------------------------------------
_ring_scores: Dict[str, float] = {}

def _load_ring_scores() -> None:
    global _ring_scores
    if RING_SCORES_PATH.exists():
        try:
            _ring_scores = {
                str(k): float(v)
                for k, v in json.loads(RING_SCORES_PATH.read_text(encoding="utf-8")).items()
            }
        except Exception:
            _ring_scores = {}

_load_ring_scores()


def _risk_tier(ring_score: float) -> str:
    if ring_score >= 0.70:
        return "high"
    if ring_score >= 0.40:
        return "medium"
    return "clean"


def _device_hash_int(device_id: str) -> int:
    return int(hashlib.sha1(device_id.encode()).hexdigest(), 16)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/watchlist/{account_id}", response_class=JSONResponse)
def check_watchlist(account_id: str, _auth: None = Security(_require_api_key)) -> Dict[str, Any]:
    ring_score = _ring_scores.get(account_id, 0.0)
    tier       = _risk_tier(ring_score)
    hit        = tier != "clean"
    return {
        "account_id": account_id,
        "hit":        hit,
        "risk_tier":  tier,
        "ring_score": round(ring_score, 4),
        "source":     "fraud_ring_registry",
        "registry_size": len(_ring_scores),
    }


@app.get("/device/{device_id}", response_class=JSONResponse)
def check_device(device_id: str, _auth: None = Security(_require_api_key)) -> Dict[str, Any]:
    h             = _device_hash_int(device_id)
    shared_count  = h % 8                          # 0–7 shared accounts
    risk_score    = round(min(1.0, shared_count * 0.12), 3)
    return {
        "device_id":     device_id,
        "risk_score":    risk_score,
        "shared_count":  shared_count,
        "source":        "device_reputation_service",
    }


@app.get("/health", response_class=JSONResponse)
def health() -> Dict[str, Any]:
    return {
        "status":              "ok",
        "service":             "mock_mcp_server",
        "accounts_in_registry": len(_ring_scores),
        "ring_scores_path":    str(RING_SCORES_PATH),
        "ring_scores_loaded":  RING_SCORES_PATH.exists(),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mock MCP External Signal Server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    uvicorn.run(
        "project.scripts.mock_mcp_server:app",
        host=args.host,
        port=args.port,
        workers=1,
    )

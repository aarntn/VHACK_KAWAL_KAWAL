# Canonical runtime service for demos/repo review.
import os
import random
import time
import threading
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
import json
import logging
import uuid
import contextvars

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .contracts import PAYLOAD_SCHEMA_VERSION as IEEE_PAYLOAD_SCHEMA_VERSION
from .schema_spec import REQUIRED_SERVING_INPUT_FIELDS, CONTEXT_FEATURE_FIELDS
from .domain_exceptions import (
    ArtifactSchemaMismatchError,
    ConfigurationError,
    ReviewQueueRecordNotFoundError,
    UnknownTransactionTypeError,
    UserProfileMismatchError,
)
from .errors import (
    ApiErrorResponse,
    DomainError,
    build_api_error_response,
    build_validation_error_details,
    classify_known_error,
    extract_http_exception_detail,
)
from .schemas import WalletAuthorizeRequest, WalletAuthorizeResponse
from .alerts import AlertEvent, get_alert_notifier

API_VERSION = "1.1.0"
PAYLOAD_SCHEMA_VERSION = IEEE_PAYLOAD_SCHEMA_VERSION
SERVICE_NAME = "Wallet Gateway Mock"
FRAUD_ENGINE_URL = os.getenv("FRAUD_ENGINE_URL", "http://127.0.0.1:8000/score_transaction")
FRAUD_ENGINE_HEALTH_URL = os.getenv("FRAUD_ENGINE_HEALTH_URL", "http://fraud-api:8000/health")
UPSTREAM_TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "1.5"))

UPSTREAM_MAX_RETRIES = int(os.getenv("UPSTREAM_MAX_RETRIES", "2"))
UPSTREAM_BACKOFF_MS = float(os.getenv("UPSTREAM_BACKOFF_MS", "40"))
UPSTREAM_BACKOFF_MAX_MS = float(os.getenv("UPSTREAM_BACKOFF_MAX_MS", "400"))
UPSTREAM_BACKOFF_JITTER_RATIO = float(os.getenv("UPSTREAM_BACKOFF_JITTER_RATIO", "0.3"))
UPSTREAM_RETRY_STATUS_CODES_RAW = os.getenv("UPSTREAM_RETRY_STATUS_CODES", "429,500,502,503,504")
UPSTREAM_POOL_MAX_CONNECTIONS = int(os.getenv("UPSTREAM_POOL_MAX_CONNECTIONS", "256"))
UPSTREAM_POOL_MAX_KEEPALIVE = int(os.getenv("UPSTREAM_POOL_MAX_KEEPALIVE", "128"))
UPSTREAM_POOL_KEEPALIVE_EXPIRY_SECONDS = float(os.getenv("UPSTREAM_POOL_KEEPALIVE_EXPIRY_SECONDS", "20"))
MAX_INFLIGHT_REQUESTS = int(os.getenv("MAX_INFLIGHT_REQUESTS", "400"))
UVICORN_WORKERS = int(os.getenv("UVICORN_WORKERS", "4"))

CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
CIRCUIT_BREAKER_RESET_SECONDS = float(os.getenv("CIRCUIT_BREAKER_RESET_SECONDS", "30"))
CIRCUIT_BREAKER_HALF_OPEN_SUCCESS_THRESHOLD = int(
    os.getenv("CIRCUIT_BREAKER_HALF_OPEN_SUCCESS_THRESHOLD", "2")
)

FALLBACK_ENGINE_DECISION = os.getenv("FALLBACK_ENGINE_DECISION", "FLAG").upper()
if FALLBACK_ENGINE_DECISION not in {"APPROVE", "FLAG", "BLOCK"}:
    raise ConfigurationError("FALLBACK_ENGINE_DECISION must be one of APPROVE, FLAG, BLOCK")

FALLBACK_RISK_SCORES = {
    "APPROVE": 0.20,
    "FLAG": 0.50,
    "BLOCK": 0.95,
}

SLA_GUARDRAILS = {
    "tier_1_concurrency_25": {"p95_latency_ms_max": 180, "error_rate_max": 0.005},
    "tier_2_concurrency_100": {"p95_latency_ms_max": 350, "error_rate_max": 0.01},
    "tier_3_concurrency_250": {"p95_latency_ms_max": 700, "error_rate_max": 0.02},
    "tier_4_concurrency_400": {"p95_latency_ms_max": 1000, "error_rate_max": 0.03},
}

logger = logging.getLogger(__name__)
alert_notifier = get_alert_notifier()
IEEE_FORWARD_FIELDS = REQUIRED_SERVING_INPUT_FIELDS + CONTEXT_FEATURE_FIELDS

UPSTREAM_HTTP: httpx.AsyncClient | None = None
_runtime_init_lock = threading.Lock()
_runtime_init_count = 0
_inflight_guard = threading.BoundedSemaphore(value=MAX_INFLIGHT_REQUESTS)


def _build_upstream_client() -> httpx.AsyncClient:
    timeout = httpx.Timeout(UPSTREAM_TIMEOUT_SECONDS)
    limits = httpx.Limits(
        max_connections=UPSTREAM_POOL_MAX_CONNECTIONS,
        max_keepalive_connections=UPSTREAM_POOL_MAX_KEEPALIVE,
        keepalive_expiry=UPSTREAM_POOL_KEEPALIVE_EXPIRY_SECONDS,
    )
    return httpx.AsyncClient(timeout=timeout, limits=limits)


# Initialize once for module-level callers/tests; lifespan preserves singleton semantics.
UPSTREAM_HTTP = _build_upstream_client()
_runtime_init_count = 1


@asynccontextmanager
async def lifespan(_: FastAPI):
    global UPSTREAM_HTTP, _runtime_init_count
    with _runtime_init_lock:
        if UPSTREAM_HTTP is None:
            UPSTREAM_HTTP = _build_upstream_client()
            _runtime_init_count += 1
        logger.info(
            "startup_upstream_client_guard worker_pid=%s init_count=%s",
            os.getpid(),
            _runtime_init_count,
        )
    yield
    if UPSTREAM_HTTP is not None:
        await UPSTREAM_HTTP.aclose()
        UPSTREAM_HTTP = None


app = FastAPI(
    title=SERVICE_NAME,
    description="Mock digital wallet integration flow for fraud-aware payment authorization",
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RESILIENCE / RETRY / CIRCUIT BREAKER
# ============================================================

def _parse_retry_status_codes(raw: str) -> set[int]:
    statuses: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            statuses.add(int(token))
        except ValueError:
            continue
    return statuses


UPSTREAM_RETRY_STATUS_CODES = _parse_retry_status_codes(UPSTREAM_RETRY_STATUS_CODES_RAW)


@dataclass
class CircuitBreaker:
    failure_threshold: int
    reset_seconds: float
    half_open_success_threshold: int

    state: str = "CLOSED"
    failure_count: int = 0
    half_open_success_count: int = 0
    opened_at: float | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self.state != "OPEN":
                return True

            now = time.monotonic()
            if self.opened_at is not None and (now - self.opened_at) >= self.reset_seconds:
                self.state = "HALF_OPEN"
                self.half_open_success_count = 0
                return True

            return False

    def record_success(self) -> None:
        with self._lock:
            if self.state == "HALF_OPEN":
                self.half_open_success_count += 1
                if self.half_open_success_count >= self.half_open_success_threshold:
                    self.state = "CLOSED"
                    self.failure_count = 0
                    self.half_open_success_count = 0
                    self.opened_at = None
            else:
                self.state = "CLOSED"
                self.failure_count = 0
                self.half_open_success_count = 0
                self.opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            if self.state == "HALF_OPEN":
                self.state = "OPEN"
                self.opened_at = time.monotonic()
                self.failure_count = self.failure_threshold
                self.half_open_success_count = 0
                return

            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.opened_at = time.monotonic()
                self.half_open_success_count = 0

    def snapshot(self) -> dict:
        with self._lock:
            opened_for_seconds = 0.0
            if self.opened_at is not None:
                opened_for_seconds = max(0.0, time.monotonic() - self.opened_at)
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "half_open_success_count": self.half_open_success_count,
                "opened_for_seconds": round(opened_for_seconds, 3),
            }

    def reset(self) -> None:
        with self._lock:
            self.state = "CLOSED"
            self.failure_count = 0
            self.half_open_success_count = 0
            self.opened_at = None


circuit_breaker = CircuitBreaker(
    failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    reset_seconds=CIRCUIT_BREAKER_RESET_SECONDS,
    half_open_success_threshold=CIRCUIT_BREAKER_HALF_OPEN_SUCCESS_THRESHOLD,
)


def _should_retry(status_code: int) -> bool:
    return status_code in UPSTREAM_RETRY_STATUS_CODES


def _backoff_seconds(attempt_index: int) -> float:
    base = min((UPSTREAM_BACKOFF_MS / 1000.0) * (2 ** attempt_index), UPSTREAM_BACKOFF_MAX_MS / 1000.0)
    jitter = random.uniform(0.0, base * UPSTREAM_BACKOFF_JITTER_RATIO)
    return base + jitter


async def _attempt_fraud_call(fraud_payload: dict) -> tuple[dict | None, dict]:
    if not circuit_breaker.allow_request():
        return None, {
            "attempts": 0,
            "error": "circuit_open",
            "circuit_breaker_state": circuit_breaker.snapshot()["state"],
            "fallback_used": True,
        }

    max_attempts = max(1, UPSTREAM_MAX_RETRIES + 1)
    last_error: str | None = None

    for attempt in range(max_attempts):
        try:
            if UPSTREAM_HTTP is None:
                raise RuntimeError("Upstream HTTP client is not initialized")
            request_id = REQUEST_ID_CTX.get() or generate_request_id()
            response = await UPSTREAM_HTTP.post(
                FRAUD_ENGINE_URL,
                json=fraud_payload,
                headers={"x-correlation-id": request_id},
            )
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            last_error = f"request_exception:{exc.__class__.__name__}"
            if attempt < (max_attempts - 1):
                await asyncio.sleep(_backoff_seconds(attempt))
                continue
            circuit_breaker.record_failure()
            return None, {
                "attempts": attempt + 1,
                "error": last_error,
                "circuit_breaker_state": circuit_breaker.snapshot()["state"],
                "fallback_used": True,
            }

        if response.status_code == 200:
            circuit_breaker.record_success()
            payload = response.json()
            return payload, {
                "attempts": attempt + 1,
                "error": None,
                "circuit_breaker_state": circuit_breaker.snapshot()["state"],
                "fallback_used": False,
            }

        last_error = f"status_{response.status_code}"
        if _should_retry(response.status_code) and attempt < (max_attempts - 1):
            await asyncio.sleep(_backoff_seconds(attempt))
            continue

        circuit_breaker.record_failure()
        return None, {
            "attempts": attempt + 1,
            "error": last_error,
            "circuit_breaker_state": circuit_breaker.snapshot()["state"],
            "fallback_used": True,
        }

    circuit_breaker.record_failure()
    return None, {
        "attempts": max_attempts,
        "error": last_error or "unknown_upstream_error",
        "circuit_breaker_state": circuit_breaker.snapshot()["state"],
        "fallback_used": True,
    }


# ============================================================
# HELPERS
# ============================================================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_request_id() -> str:
    return str(uuid.uuid4())


REQUEST_ID_CTX: contextvars.ContextVar[str | None] = contextvars.ContextVar("wallet_request_id", default=None)


def classify_exception(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "artifact_missing"
    if isinstance(exc, httpx.TimeoutException) or isinstance(exc, TimeoutError):
        return "timeout_upstream"
    if isinstance(exc, ReviewQueueRecordNotFoundError):
        return "review_queue_record_not_found"
    if isinstance(exc, UnknownTransactionTypeError):
        return "unknown_transaction_type"
    if isinstance(exc, UserProfileMismatchError):
        return "user_profile_mismatch"
    if isinstance(exc, ArtifactSchemaMismatchError):
        return "artifact_schema_mismatch"

    message = str(exc).lower()
    if isinstance(exc, ValueError) or "validation" in message:
        return "schema_mismatch"
    if "artifact mismatch" in message or "incompatible" in message:
        return "artifact_incompatible"
    if "model" in message and "runtime" in message:
        return "model_runtime_error"
    return "unknown_internal"


def exception_signature(exc: Exception) -> str:
    return f"{exc.__class__.__name__}:{str(exc)[:200]}"


def structured_log(event: str, **fields: object) -> None:
    payload = {"timestamp": utc_now_iso(), "event": event}
    payload.update(fields)
    logger.info(json.dumps(payload))


def resolve_correlation_id(request: Request) -> str:
    return request.headers.get("x-correlation-id") or request.headers.get("x-request-id") or generate_request_id()


def extract_request_id(request: Request) -> str:
    return REQUEST_ID_CTX.get() or request.headers.get("x-request-id") or generate_request_id()


def extract_user_id(request: Request) -> str | None:
    user_id = request.path_params.get("user_id")
    if user_id is None:
        user_id = request.query_params.get("user_id")
    if user_id is None:
        user_id = request.headers.get("x-user-id") or request.headers.get("user-id")
    return str(user_id) if user_id is not None else None


def extract_transaction_identifiers(request: Request) -> dict[str, str]:
    candidates = {
        "transaction_id": request.path_params.get("transaction_id")
        or request.query_params.get("transaction_id")
        or request.headers.get("x-transaction-id"),
        "tx_id": request.path_params.get("tx_id")
        or request.query_params.get("tx_id")
        or request.headers.get("x-tx-id"),
        "payment_id": request.path_params.get("payment_id")
        or request.query_params.get("payment_id")
        or request.headers.get("x-payment-id"),
        "wallet_request_id": request.path_params.get("wallet_request_id")
        or request.query_params.get("wallet_request_id")
        or request.headers.get("x-wallet-request-id"),
    }
    return {k: str(v) for k, v in candidates.items() if v is not None}


def extract_trace_id(request: Request, request_id: str) -> str:
    return (
        request.headers.get("x-trace-id")
        or request.headers.get("x-request-id")
        or request.headers.get("traceparent")
        or request_id
    )


def log_exception(request: Request, exc: Exception) -> str:
    request_id = extract_request_id(request)
    trace_id = extract_trace_id(request, request_id)
    category = classify_exception(exc)
    signature = exception_signature(exc)
    route = request.url.path
    user_id = extract_user_id(request)
    transaction_identifiers = extract_transaction_identifiers(request)
    logger.error(
        json.dumps(
            {
                "timestamp": utc_now_iso(),
                "event": "response_boundary",
                "status": "exception",
                "route": route,
                "request_id": request_id,
                "correlation_id": request_id,
                "trace_id": trace_id,
                "user_id": user_id,
                "transaction_identifiers": transaction_identifiers,
                "error_category": category,
                "exception_signature": signature,
            }
        )
    )
    logger.error(
        json.dumps(
            {
                "timestamp": utc_now_iso(),
                "request_id": request_id,
                "correlation_id": request_id,
                "trace_id": trace_id,
                "route": route,
                "user_id": user_id,
                "transaction_identifiers": transaction_identifiers,
                "error_category": category,
                "exception_signature": signature,
                "exception_type": type(exc).__name__,
                "exception_class": exc.__class__.__name__,
                "exception_message": str(exc),
            }
        )
    )
    return request_id


def map_engine_decision_to_wallet_action(engine_decision: str) -> str:
    if engine_decision == "APPROVE":
        return "APPROVED"
    elif engine_decision == "FLAG":
        return "PENDING_VERIFICATION"
    return "DECLINED_FRAUD_RISK"


def build_wallet_message(engine_decision: str) -> str:
    if engine_decision == "APPROVE":
        return "Payment approved and processed."
    elif engine_decision == "FLAG":
        return "Payment requires step-up verification before completion."
    return "Payment declined due to elevated fraud risk."


def resolve_next_step(wallet_action: str) -> str:
    if wallet_action == "APPROVED":
        return "Complete payment"
    if wallet_action == "PENDING_VERIFICATION":
        return "Trigger OTP / biometric verification"
    return "Decline payment and notify user"


def build_fallback_fraud_result(error_code: str) -> dict:
    return {
        "decision": FALLBACK_ENGINE_DECISION,
        "decision_source": "score_band",
        "final_risk_score": FALLBACK_RISK_SCORES[FALLBACK_ENGINE_DECISION],
        "reasons": [
            "UPSTREAM_ENGINE_UNAVAILABLE_FALLBACK",
            f"upstream_error={error_code}",
        ],
    }


def validate_wallet_request(payment: "WalletPaymentRequest") -> None:
    if payment.currency != payment.currency.upper():
        raise ValueError("currency must be uppercase ISO-4217 code")


def build_fraud_payload(payment: "WalletPaymentRequest") -> dict:
    # Forward the shared risk contract as-is to avoid duplicate field-level preprocessing
    # in gateway and fraud services.
    return payment.model_dump(mode="python")


# ============================================================
# RESPONSE MODELS
# ============================================================
class WalletApiInfoResponse(BaseModel):
    service: str
    api_version: str
    fraud_engine_url: str
    fraud_engine_health_url: str
    resilience: dict[str, object]
    message: str


class WalletHealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    api_version: str
    fraud_engine_url: str
    circuit_breaker: dict[str, object]


class WalletReadinessResponse(BaseModel):
    status: Literal["ok"]
    service: str
    fraud_engine_url: str
    fraud_engine_health_url: str


WALLET_ERROR_RESPONSES = {
    400: {"model": ApiErrorResponse, "description": "Bad request / domain validation error"},
    404: {"model": ApiErrorResponse, "description": "Resource not found"},
    409: {"model": ApiErrorResponse, "description": "Domain conflict"},
    422: {"model": ApiErrorResponse, "description": "Validation error payload"},
    500: {"model": ApiErrorResponse, "description": "Internal server error"},
    503: {"model": ApiErrorResponse, "description": "Upstream unavailable or overloaded"},
}


# ============================================================
# INPUT SCHEMA
# ============================================================
class WalletPaymentRequest(WalletAuthorizeRequest):
    pass


WALLET_AUTHORIZE_RESPONSES = {
    200: {"model": WalletAuthorizeResponse, "description": "Wallet authorization decision payload"},
    **WALLET_ERROR_RESPONSES,
}


# ============================================================
# ERROR HANDLING
# ============================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    if exc.status_code >= 500:
        alert_notifier.notify_internal_error(
            AlertEvent(
                service=SERVICE_NAME,
                severity="critical",
                title="Wallet Gateway HTTP internal error",
                details={
                    "route": request.url.path,
                    "request_id": request_id,
                    "status_code": exc.status_code,
                    "exception_type": type(exc).__name__,
                },
            )
        )
    detail = extract_http_exception_detail(exc, "Upstream risk engine unavailable")
    return JSONResponse(
        status_code=exc.status_code,
        content=build_api_error_response(
            error="Wallet Gateway Error",
            detail=detail,
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    details = build_validation_error_details(exc)
    return JSONResponse(
        status_code=422,
        content=build_api_error_response(
            error="ValidationError",
            detail="Request payload failed schema validation",
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=details,
            schema_version_expected=PAYLOAD_SCHEMA_VERSION,
        ),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    return JSONResponse(
        status_code=400,
        content=build_api_error_response(
            error="Bad Request",
            detail=str(exc),
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(DomainError)
async def domain_error_handler(request: Request, exc: DomainError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    return JSONResponse(
        status_code=400,
        content=build_api_error_response(
            error="Bad Request",
            detail=str(exc),
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    alert_notifier.notify_internal_error(
        AlertEvent(
            service=SERVICE_NAME,
            severity="critical",
            title="Wallet Gateway unhandled exception",
            details={
                "route": request.url.path,
                "request_id": request_id,
                "exception_type": type(exc).__name__,
                "error_category": category,
            },
        )
    )
    return JSONResponse(
        status_code=500,
        content=build_api_error_response(
            error="Wallet Gateway Error",
            detail="Upstream risk engine unavailable",
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(UnknownTransactionTypeError)
async def unknown_transaction_type_handler(request: Request, exc: UnknownTransactionTypeError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    return JSONResponse(
        status_code=422,
        content=build_api_error_response(
            error="Wallet Gateway Error",
            detail=str(exc),
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(UserProfileMismatchError)
async def user_profile_mismatch_handler(request: Request, exc: UserProfileMismatchError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    return JSONResponse(
        status_code=409,
        content=build_api_error_response(
            error="Wallet Gateway Error",
            detail=str(exc),
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(ReviewQueueRecordNotFoundError)
async def review_queue_record_not_found_handler(request: Request, exc: ReviewQueueRecordNotFoundError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    return JSONResponse(
        status_code=404,
        content=build_api_error_response(
            error="Wallet Gateway Error",
            detail=str(exc),
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


@app.exception_handler(ArtifactSchemaMismatchError)
async def artifact_schema_mismatch_handler(request: Request, exc: ArtifactSchemaMismatchError):
    request_id = log_exception(request, exc)
    category = classify_known_error(exc, classify_exception)
    alert_notifier.notify_domain_failure(
        AlertEvent(
            service=SERVICE_NAME,
            severity="high",
            title="Wallet Gateway artifact schema mismatch",
            details={
                "route": request.url.path,
                "request_id": request_id,
                "exception_type": type(exc).__name__,
            },
        )
    )
    return JSONResponse(
        status_code=422,
        content=build_api_error_response(
            error="Wallet Gateway Error",
            detail=str(exc),
            error_category=category,
            request_id=request_id,
            correlation_id=request_id,
            details=None,
        ),
    )


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/api/info", response_model=WalletApiInfoResponse, responses=WALLET_ERROR_RESPONSES)
def api_info():
    return {
        "service": SERVICE_NAME,
        "api_version": API_VERSION,
        "fraud_engine_url": FRAUD_ENGINE_URL,
        "fraud_engine_health_url": FRAUD_ENGINE_HEALTH_URL,
        "resilience": {
            "max_retries": UPSTREAM_MAX_RETRIES,
            "retry_status_codes": sorted(UPSTREAM_RETRY_STATUS_CODES),
            "upstream_pool_max_connections": UPSTREAM_POOL_MAX_CONNECTIONS,
            "upstream_pool_max_keepalive": UPSTREAM_POOL_MAX_KEEPALIVE,
            "upstream_keepalive_expiry_seconds": UPSTREAM_POOL_KEEPALIVE_EXPIRY_SECONDS,
            "backoff_ms": UPSTREAM_BACKOFF_MS,
            "backoff_max_ms": UPSTREAM_BACKOFF_MAX_MS,
            "backoff_jitter_ratio": UPSTREAM_BACKOFF_JITTER_RATIO,
            "circuit_breaker_failure_threshold": CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            "circuit_breaker_reset_seconds": CIRCUIT_BREAKER_RESET_SECONDS,
            "fallback_engine_decision": FALLBACK_ENGINE_DECISION,
            "max_inflight_requests": MAX_INFLIGHT_REQUESTS,
            "uvicorn_workers": UVICORN_WORKERS,
            "runtime_init_count": _runtime_init_count,
        },
        "message": "Wallet gateway mock is running",
    }


@app.get("/health", response_model=WalletHealthResponse, responses=WALLET_ERROR_RESPONSES)
def health():
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "api_version": API_VERSION,
        "fraud_engine_url": FRAUD_ENGINE_URL,
        "circuit_breaker": circuit_breaker.snapshot(),
    }


@app.get("/health/ready", response_model=WalletReadinessResponse, responses=WALLET_ERROR_RESPONSES)
async def readiness():
    try:
        if UPSTREAM_HTTP is None:
            raise HTTPException(status_code=503, detail="Upstream HTTP client not initialized")
        response = await UPSTREAM_HTTP.get(FRAUD_ENGINE_HEALTH_URL)
        if response.status_code != 200:
            raise HTTPException(
                status_code=503,
                detail=f"Fraud engine health returned {response.status_code}",
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Fraud engine health endpoint is unreachable",
        ) from exc

    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "fraud_engine_url": FRAUD_ENGINE_URL,
        "fraud_engine_health_url": FRAUD_ENGINE_HEALTH_URL,
    }


@app.get("/wallet/decision_policy")
def decision_policy():
    return {
        "APPROVE": {
            "wallet_action": "APPROVED",
            "meaning": "Transaction is processed immediately",
        },
        "FLAG": {
            "wallet_action": "PENDING_VERIFICATION",
            "meaning": "Transaction requires OTP / biometric / secondary verification",
        },
        "BLOCK": {
            "wallet_action": "DECLINED_FRAUD_RISK",
            "meaning": "Transaction is stopped and user may be alerted",
        },
        "fallback_policy": {
            "when": "risk engine unavailable or circuit breaker open",
            "fraud_engine_decision": FALLBACK_ENGINE_DECISION,
            "wallet_action": map_engine_decision_to_wallet_action(FALLBACK_ENGINE_DECISION),
        },
    }


@app.post(
    "/wallet/authorize_payment",
    response_model=WalletAuthorizeResponse,
    responses=WALLET_AUTHORIZE_RESPONSES,
)
async def authorize_payment(payment: WalletPaymentRequest, request: Request):
    request_start = time.perf_counter()
    wallet_request_id = resolve_correlation_id(request)
    REQUEST_ID_CTX.set(wallet_request_id)
    stage_timings_ms: dict[str, float] = {}
    endpoint_name = "wallet_authorize_payment"

    structured_log(
        "request_ingress",
        request_id=wallet_request_id,
        correlation_id=wallet_request_id,
        endpoint=endpoint_name,
        payload_schema_version=PAYLOAD_SCHEMA_VERSION,
    )
    if not _inflight_guard.acquire(blocking=False):
        structured_log(
            "load_shed_503",
            request_id=wallet_request_id,
            correlation_id=wallet_request_id,
            endpoint=endpoint_name,
            reason="max_inflight_exceeded",
            max_inflight_requests=MAX_INFLIGHT_REQUESTS,
        )
        raise HTTPException(status_code=503, detail="Gateway overloaded; retry later")

    try:
        stage_start = time.perf_counter()
        validate_wallet_request(payment)
        stage_timings_ms["input_validation_ms"] = round((time.perf_counter() - stage_start) * 1000, 3)

        stage_start = time.perf_counter()
        structured_log(
            "preprocessing_start",
            request_id=wallet_request_id,
            correlation_id=wallet_request_id,
            endpoint=endpoint_name,
        )
        fraud_payload = build_fraud_payload(payment)
        stage_timings_ms["wallet_preprocessing_ms"] = round((time.perf_counter() - stage_start) * 1000, 3)
        structured_log(
            "preprocessing_end",
            request_id=wallet_request_id,
            correlation_id=wallet_request_id,
            endpoint=endpoint_name,
            preprocessing_ms=stage_timings_ms["wallet_preprocessing_ms"],
        )

        stage_start = time.perf_counter()
        structured_log(
            "model_inference_start",
            request_id=wallet_request_id,
            correlation_id=wallet_request_id,
            endpoint=endpoint_name,
        )
        fraud_result, upstream_meta = await _attempt_fraud_call(fraud_payload)
        stage_timings_ms["internal_fraud_scoring_call_ms"] = round((time.perf_counter() - stage_start) * 1000, 3)
        structured_log(
            "model_inference_end",
            request_id=wallet_request_id,
            correlation_id=wallet_request_id,
            endpoint=endpoint_name,
            model_inference_ms=stage_timings_ms["internal_fraud_scoring_call_ms"],
            upstream_attempts=int(upstream_meta["attempts"]),
        )

        stage_start = time.perf_counter()
        if fraud_result is None:
            fallback_stage_start = time.perf_counter()
            fraud_result = build_fallback_fraud_result(upstream_meta["error"])
            stage_timings_ms["fallback_path_ms"] = round((time.perf_counter() - fallback_stage_start) * 1000, 3)
        else:
            stage_timings_ms["fallback_path_ms"] = 0.0

        engine_decision = fraud_result["decision"]
        decision_source = str(fraud_result.get("decision_source", "score_band"))
        final_risk_score = fraud_result["final_risk_score"]
        fraud_reasons = fraud_result["reasons"]

        wallet_action = map_engine_decision_to_wallet_action(engine_decision)
        wallet_message = build_wallet_message(engine_decision)
        stage_timings_ms["response_mapping_ms"] = round((time.perf_counter() - stage_start) * 1000, 3)
        stage_timings_ms["total_pipeline_ms"] = round((time.perf_counter() - request_start) * 1000, 3)
        fraud_core_total_pipeline_ms = (
            (fraud_result.get("stage_timings_ms") or {}).get("total_pipeline_ms")
            if not bool(upstream_meta["fallback_used"])
            else None
        )
        if fraud_core_total_pipeline_ms is not None:
            fraud_core_total_pipeline_ms = round(float(fraud_core_total_pipeline_ms), 3)
            wallet_minus_fraud_core_ms = round(
                max(0.0, stage_timings_ms["total_pipeline_ms"] - fraud_core_total_pipeline_ms), 3
            )
        else:
            wallet_minus_fraud_core_ms = None
        if wallet_minus_fraud_core_ms is not None:
            stage_timings_ms["wallet_minus_fraud_core_ms"] = wallet_minus_fraud_core_ms

        logger.info(
            json.dumps(
                {
                    "timestamp": utc_now_iso(),
                    "event": "wallet_authorize_payment_timing",
                    "wallet_request_id": wallet_request_id,
                    "correlation_id": wallet_request_id,
                    "fallback_used": bool(upstream_meta["fallback_used"]),
                    "upstream_attempts": int(upstream_meta["attempts"]),
                    "upstream_error": upstream_meta["error"],
                    "input_validation_ms": stage_timings_ms["input_validation_ms"],
                    "wallet_preprocessing_ms": stage_timings_ms["wallet_preprocessing_ms"],
                    "internal_fraud_scoring_call_ms": stage_timings_ms["internal_fraud_scoring_call_ms"],
                    "response_mapping_ms": stage_timings_ms["response_mapping_ms"],
                    "total_pipeline_ms": stage_timings_ms["total_pipeline_ms"],
                    "fraud_core_total_pipeline_ms": fraud_core_total_pipeline_ms,
                    "wallet_minus_fraud_core_ms": wallet_minus_fraud_core_ms,
                    "decision_source": decision_source,
                    "stage_timings_ms": stage_timings_ms,
                }
            )
        )
        structured_log(
            "wallet_timing_summary",
            request_id=wallet_request_id,
            correlation_id=wallet_request_id,
            endpoint=endpoint_name,
            input_validation_ms=stage_timings_ms["input_validation_ms"],
            wallet_preprocessing_ms=stage_timings_ms["wallet_preprocessing_ms"],
            internal_fraud_scoring_call_ms=stage_timings_ms["internal_fraud_scoring_call_ms"],
            response_mapping_ms=stage_timings_ms["response_mapping_ms"],
            total_pipeline_ms=stage_timings_ms["total_pipeline_ms"],
            fraud_core_total_pipeline_ms=fraud_core_total_pipeline_ms,
            wallet_minus_fraud_core_ms=wallet_minus_fraud_core_ms,
            decision_source=decision_source,
        )
        structured_log(
            "response_boundary",
            request_id=wallet_request_id,
            correlation_id=wallet_request_id,
            endpoint=endpoint_name,
            status="success",
            error_category=None,
        )

        return WalletAuthorizeResponse(
            wallet_request_id=wallet_request_id,
            correlation_id=wallet_request_id,
            timestamp_utc=utc_now_iso(),
            wallet_action=wallet_action,
            wallet_message=wallet_message,
            fraud_engine_decision=engine_decision,
            decision=engine_decision,
            decision_source=decision_source,
            risk_score=final_risk_score,
            final_risk_score=final_risk_score,
            fraud_reasons=fraud_reasons,
            next_step=resolve_next_step(wallet_action),
            explainability=None,
            upstream_attempts=int(upstream_meta["attempts"]),
            circuit_breaker_state=upstream_meta["circuit_breaker_state"],
            fallback_used=bool(upstream_meta["fallback_used"]),
            upstream_error=upstream_meta["error"],
            stage_timings_ms={
                "total_pipeline_ms": stage_timings_ms.get("total_pipeline_ms"),
                "details": stage_timings_ms,
            },
        )
    finally:
        _inflight_guard.release()


@app.get("/wallet/demo_cases")
def demo_cases():
    return {
        "safe_case_user_id": "user_safe",
        "flag_case_user_id": "user_flag",
        "block_case_user_id": "user_block",
        "demo_note": "Use the same transaction payloads as the fraud engine, but add wallet_id, merchant_name, and currency.",
    }


@app.get("/wallet/sla_guardrails")
def sla_guardrails():
    return {
        "service": SERVICE_NAME,
        "tiers": SLA_GUARDRAILS,
        "load_shedding_policy": {
            "mode": "immediate_503_when_inflight_full",
            "max_inflight_requests": MAX_INFLIGHT_REQUESTS,
        },
    }


# Helper for deterministic tests only.
def _reset_circuit_breaker_for_tests() -> None:
    circuit_breaker.reset()

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from .domain import (
    ArtifactSchemaMismatchError,
    ConfigurationError,
    DomainError,
    DomainValidationError,
    InvalidRiskScoreError,
    ProfileNotFoundError,
    ReviewQueueRecordNotFoundError,
    UnknownChannelError,
    UnknownTransactionTypeError,
    UserProfileMismatchError,
)
from .handlers import register_exception_handlers


class ApiErrorResponse(BaseModel):
    # Canonical envelope fields used by fraud + wallet APIs.
    error: str | None = None
    detail: str | None = None
    error_category: str | None = None
    request_id: str | None = None
    correlation_id: str | None = None
    details: Any | None = None
    schema_version_expected: str | None = None
    timestamp_utc: str | None = None

    # Legacy-compatible envelope fields used by older handlers.
    type: str | None = None
    message: str | None = None
    field: str | None = None
    code: str | None = None
    trace_id: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_api_error_response(
    *,
    error: str | None,
    detail: str | None,
    error_category: str | None,
    request_id: str | None,
    correlation_id: str | None,
    details: Any | None = None,
    schema_version_expected: str | None = None,
) -> dict[str, Any]:
    payload = ApiErrorResponse(
        error=error,
        detail=detail,
        error_category=error_category,
        request_id=request_id,
        correlation_id=correlation_id,
        details=details,
        schema_version_expected=schema_version_expected,
        timestamp_utc=utc_now_iso(),
    )
    return payload.model_dump(exclude_none=True)


def build_validation_error_details(exc: RequestValidationError) -> list[dict[str, Any]]:
    details = []
    for err in exc.errors():
        loc = [str(item) for item in err.get("loc", []) if item != "body"]
        details.append(
            {
                "field": ".".join(loc) if loc else "body",
                "code": err.get("type", "validation_error"),
                "message": err.get("msg", "Invalid request payload"),
                "input": err.get("input"),
            }
        )
    return details


def extract_http_exception_detail(exc: HTTPException, fallback: str) -> str:
    return exc.detail if isinstance(exc.detail, str) else fallback


def classify_known_error(exc: Exception, classify_exception: Any) -> str:
    if isinstance(exc, HTTPException):
        if exc.status_code >= 500:
            return "http_server_error"
        if exc.status_code == 404:
            return "not_found"
        if exc.status_code in (400, 401, 403, 409, 422):
            return "http_client_error"
        return "http_error"
    if isinstance(exc, RequestValidationError):
        return "schema_mismatch"
    if isinstance(exc, (ValueError, DomainError)):
        return "domain_validation_error"
    return classify_exception(exc)


__all__ = [
    "ApiErrorResponse",
    "ArtifactSchemaMismatchError",
    "build_api_error_response",
    "build_validation_error_details",
    "classify_known_error",
    "ConfigurationError",
    "DomainError",
    "DomainValidationError",
    "extract_http_exception_detail",
    "InvalidRiskScoreError",
    "ProfileNotFoundError",
    "ReviewQueueRecordNotFoundError",
    "UnknownChannelError",
    "UnknownTransactionTypeError",
    "UserProfileMismatchError",
    "register_exception_handlers",
]

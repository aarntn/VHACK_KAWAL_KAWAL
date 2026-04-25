from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .domain import DomainError


def _trace_id(request: Request) -> str | None:
    return (
        getattr(request.state, "request_id", None)
        or request.headers.get("x-correlation-id")
        or request.headers.get("x-request-id")
    )


def _envelope(
    error_type: str,
    message: str,
    *,
    field: str | None = None,
    code: str | None = None,
    details: Any | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": error_type, "message": message}
    if field is not None:
        payload["field"] = field
    if code is not None:
        payload["code"] = code
    if details is not None:
        payload["details"] = details
    if trace_id is not None:
        payload["trace_id"] = trace_id
    return payload


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            status_code=getattr(exc, "status_code", 400),
            content=_envelope(
                getattr(exc, "error_type", "domain_error"),
                exc.message,
                field=getattr(exc, "field", None),
                code=getattr(exc, "code", None),
                details=getattr(exc, "details", None),
                trace_id=_trace_id(request),
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = []
        for err in exc.errors():
            loc = [str(item) for item in err.get("loc", []) if item != "body"]
            details.append(
                {
                    "field": ".".join(loc) if loc else "body",
                    "code": err.get("type", "validation_error"),
                    "message": err.get("msg", "Invalid request payload"),
                }
            )
        first = details[0] if details else None
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "validation_error",
                "Request payload failed schema validation",
                field=first.get("field") if first else None,
                code=first.get("code") if first else "validation_error",
                details=details,
                trace_id=_trace_id(request),
            ),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "HTTP error"
        error_type = "validation_error" if 400 <= exc.status_code < 500 else "internal_error"
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(
                error_type,
                message,
                code=f"http_{exc.status_code}",
                trace_id=_trace_id(request),
            ),
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, _: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=_envelope(
                "internal_error",
                "Internal server error",
                code="internal_error",
                trace_id=_trace_id(request),
            ),
        )

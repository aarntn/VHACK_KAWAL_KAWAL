from __future__ import annotations

from typing import Any


class DomainError(Exception):
    """Base type for domain/business exceptions that map to 4xx responses."""

    error_type = "domain_error"
    error_code = "domain_error"
    status_code = 400

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        code: str | None = None,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        self.code = code or self.error_code
        self.details = details


class DomainValidationError(DomainError):
    error_type = "validation_error"
    error_code = "domain_validation_error"
    status_code = 400


class UnknownTransactionTypeError(DomainValidationError):
    error_code = "unknown_transaction_type"
    status_code = 422


class UnknownChannelError(DomainValidationError):
    error_code = "unknown_channel"
    status_code = 422


class ProfileNotFoundError(DomainError):
    error_type = "not_found"
    error_code = "profile_not_found"
    status_code = 404


class InvalidRiskScoreError(DomainValidationError):
    error_code = "invalid_risk_score"


class UserProfileMismatchError(DomainError):
    error_type = "conflict"
    error_code = "user_profile_mismatch"
    status_code = 409


class ReviewQueueRecordNotFoundError(DomainError):
    error_type = "not_found"
    error_code = "review_queue_record_not_found"
    status_code = 404


class ArtifactSchemaMismatchError(DomainValidationError):
    error_code = "artifact_schema_mismatch"
    status_code = 422


class ConfigurationError(DomainValidationError):
    error_code = "invalid_configuration"

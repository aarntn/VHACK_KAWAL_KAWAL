from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AlertEvent:
    service: str
    severity: str
    title: str
    details: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AlertNotifier:
    """Adapter interface for operational alerts (Slack/PagerDuty/etc)."""

    def notify_internal_error(self, event: AlertEvent) -> None:
        raise NotImplementedError

    def notify_domain_failure(self, event: AlertEvent) -> None:
        raise NotImplementedError


class NoOpAlertNotifier(AlertNotifier):
    """
    Production-safe default notifier.

    Future integration point:
    - Replace this with a concrete Slack/PagerDuty implementation and wire it through
      `get_alert_notifier()` (or dependency injection at app startup).
    """

    def notify_internal_error(self, event: AlertEvent) -> None:
        self._safe_log("internal_error", event)

    def notify_domain_failure(self, event: AlertEvent) -> None:
        self._safe_log("domain_failure", event)

    def _safe_log(self, event_type: str, event: AlertEvent) -> None:
        try:
            logger.debug(
                "Alert notifier noop event_type=%s service=%s severity=%s title=%s",
                event_type,
                event.service,
                event.severity,
                event.title,
            )
        except Exception:
            # Safety guarantee: notifier path must never raise.
            return


def get_alert_notifier() -> AlertNotifier:
    return NoOpAlertNotifier()

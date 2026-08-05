"""Shared types and the planner-facing Dashboard interaction port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

PlannerActivity = Literal["starting", "idle", "busy", "ended"]
DashboardMessageStatus = Literal[
    "pending",
    "sending",
    "sent",
    "withdrawn",
    "failed",
    "unsent",
]


@dataclass(slots=True)
class DashboardMessage:
    """One user message submitted through a Dashboard Session."""

    message_id: str
    text: str
    status: DashboardMessageStatus
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-safe representation exposed to the frontend."""
        return {
            "message_id": self.message_id,
            "text": self.text,
            "status": self.status,
            "error": self.error,
        }


class DashboardInteractionPort(Protocol):
    """Planner-facing access to one Dashboard interaction Session."""

    @property
    def planner_activity(self) -> PlannerActivity:
        """Return the current planner input activity."""
        ...

    @property
    def interaction_version(self) -> int:
        """Return the monotonic interaction-state version."""
        ...

    def wait_for_interaction_change(
        self,
        since: int,
        timeout: float | None = None,
    ) -> int:
        """Wait for an interaction change and return the latest version."""
        ...

    def claim_next_pending_message(self) -> DashboardMessage | None:
        """Claim the next pending message, if task replacement is not pending."""
        ...

    def mark_message_sent(self, message_id: str) -> DashboardMessage:
        """Commit one successfully submitted message."""
        ...

    def mark_message_failed(
        self,
        message_id: str,
        error: str,
    ) -> DashboardMessage:
        """Record one failed message submission."""
        ...

    def claim_interrupt_request(self) -> bool:
        """Claim a queued interrupt request."""
        ...

    def complete_interrupt(self, error: str | None = None) -> None:
        """Complete the claimed interrupt request."""
        ...

    def set_planner_activity(
        self,
        activity: PlannerActivity,
        *,
        accepting_input: bool | None = None,
    ) -> None:
        """Update planner activity and optionally input acceptance."""
        ...

    def seal_interaction(self) -> None:
        """End the interaction Session."""
        ...

    @property
    def task_replacement_requested(self) -> bool:
        """Whether this TaskRun must yield to a newer task command."""
        ...

    def complete_task_replacement(self, error: str | None = None) -> None:
        """Seal the old conversation after reaching a safe boundary."""
        ...


class DashboardInteractionError(RuntimeError):
    """Base class for invalid Dashboard interaction operations."""


class InteractionUnavailableError(DashboardInteractionError):
    """The Session is not currently accepting Dashboard input."""


class UnknownDashboardMessageError(DashboardInteractionError):
    """The requested message does not belong to this Session."""


class DashboardMessageConflictError(DashboardInteractionError):
    """A message can no longer make the requested state transition."""

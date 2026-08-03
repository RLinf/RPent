"""Sequential TaskRun coordination for one long-lived Dashboard Session."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rpent.dashboard.state import ClaimedTask, DashboardState
from rpent.utils.logging import get_logger

logger = get_logger("dashboard_session")


class DashboardSessionController:
    """Own shared services and run one claimed TaskRun at a time."""

    def __init__(
        self,
        *,
        state: DashboardState,
        start_shared: Callable[[], Any],
        run_task: Callable[[ClaimedTask, Any], BaseException | str | None],
    ) -> None:
        self._state = state
        self._start_shared = start_shared
        self._run_task = run_task

    def run(self) -> None:
        """Start shared services, then consume last-write-wins task commands."""
        shared: Any | None = None
        try:
            try:
                shared = self._start_shared()
            except Exception as exc:
                self._state.fail_session(exc)
                return

            self._state.shared_services_ready()
            while True:
                claimed = self._state.wait_for_task()
                if claimed is None:
                    break
                try:
                    error = self._run_task(claimed, shared)
                except Exception as exc:
                    error = exc

                if self._state.task_replacement_requested:
                    state = "cancelled"
                elif error:
                    state = "failed"
                else:
                    state = "succeeded"
                self._state.complete_task(
                    state=state,
                    error=error,
                )
        finally:
            if shared is not None:
                try:
                    shared.close()
                except Exception as exc:
                    logger.warning("shared runtime cleanup failed: %s", exc)

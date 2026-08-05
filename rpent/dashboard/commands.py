"""Pure parsing for Dashboard-local control commands."""

from __future__ import annotations

import re
from dataclasses import dataclass

LIBERO_SUITE_NAMES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_object_task",
    "libero_object_swap",
    "libero_object_lan",
    "libero_goal_task",
    "libero_goal_swap",
    "libero_goal_lan",
    "libero_spatial_task",
    "libero_spatial_swap",
    "libero_spatial_lan",
    "libero_10",
    "libero_10_task",
    "libero_10_swap",
    "libero_10_lan",
)
LIBERO_SUITES = frozenset(LIBERO_SUITE_NAMES)

TASK_COMMAND = "/rpent-task"
_NON_NEGATIVE_INTEGER = re.compile(r"[0-9]+")
_TASK_COMMAND_USAGE = f"{TASK_COMMAND} <suite> <task> <seed>"


@dataclass(frozen=True, slots=True)
class TaskCommand:
    """One validated request to create a fresh Dashboard TaskRun."""

    suite: str
    task: int
    seed: int


class DashboardCommandError(ValueError):
    """Raised when Dashboard command input is invalid or unsupported."""


def parse_dashboard_command(text: str) -> TaskCommand | None:
    """Parse a local task command or return ``None`` for ordinary text.

    Every input whose first token starts with ``/rpent-`` is reserved for the
    Dashboard, so unsupported command names are rejected locally.
    """

    if not isinstance(text, str):
        raise TypeError("Dashboard input must be a string")

    tokens = text.split()
    if not tokens:
        return None

    command_name = tokens[0]
    if command_name.startswith("/rpent-"):
        if command_name != TASK_COMMAND:
            raise DashboardCommandError(
                f"unknown Dashboard command: {command_name}"
            )
    elif command_name.lower() != TASK_COMMAND:
        return None

    if len(tokens) != 4 or command_name != TASK_COMMAND:
        raise DashboardCommandError(f"expected {_TASK_COMMAND_USAGE}")

    _, suite, task_text, seed_text = tokens
    if suite not in LIBERO_SUITES:
        raise DashboardCommandError(f"unsupported LIBERO suite: {suite}")

    task = _parse_non_negative_integer("task", task_text)
    seed = _parse_non_negative_integer("seed", seed_text)
    return TaskCommand(suite=suite, task=task, seed=seed)


def _parse_non_negative_integer(name: str, value: str) -> int:
    if _NON_NEGATIVE_INTEGER.fullmatch(value) is None:
        raise DashboardCommandError(
            f"{name} must be a non-negative integer, got {value!r}"
        )
    return int(value)

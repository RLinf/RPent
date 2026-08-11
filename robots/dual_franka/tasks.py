"""Task definitions for the dual-Franka RPent extension."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DualFrankaTask:
    """One real-robot dual-arm task presented to the planner."""

    name: str
    instruction: str
    success_criteria: str
    constraints: tuple[str, ...]


DUAL_FRANKA_TASKS = {
    0: DualFrankaTask(
        name="primitive_smoke_test",
        instruction=(
            "Inspect the current state, then for each arm in turn exercise one "
            "conservative translation, one conservative rotation, and the gripper "
            "controls."
        ),
        success_criteria=(
            "Every requested per-arm primitive returns a usable result and the "
            "synchronized camera/state snapshots agree with the commanded change."
        ),
        constraints=(
            "Move exactly one arm per motion call; never assume a 'both' mode.",
            "Keep translation commands at or below 0.02 m per call.",
            "Keep rotation commands at or below 0.15 rad per call.",
            "Stop immediately if images or state indicate unsafe motion.",
        ),
    ),
    1: DualFrankaTask(
        name="vla_grasp",
        instruction=(
            "Use bounded analytic motion to stage the chosen arm near the named "
            "object, ensure that arm's gripper is open, then use vla_grasp for the "
            "contact-rich grasp attempt."
        ),
        success_criteria=(
            "The target object visibly follows the chosen gripper upward and remains "
            "stable between the fingers across the camera views."
        ),
        constraints=(
            "Inspect the left-wrist, base, and right-wrist views after every action.",
            "Use VLA only after a safe near-object staging pose is reached.",
            "Do not infer success from gripper position alone.",
        ),
    ),
}


def get_dual_franka_task(task_id: int) -> DualFrankaTask:
    """Return a task by numeric ID with a clear error for unknown IDs."""
    try:
        return DUAL_FRANKA_TASKS[int(task_id)]
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"unknown dual-Franka task {task_id!r}; choices are "
            f"{sorted(DUAL_FRANKA_TASKS)}"
        ) from exc

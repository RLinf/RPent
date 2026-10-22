# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""YAM task registration and deterministic exploration rules."""

TASK_INSTRUCTIONS = {
    0: "Inspect the current state, then exercise one small translation, one small wrist rotation, and the gripper controls for each arm in turn. Move only one arm per call. Keep translations at or below 0.02 metres and wrist rotations at or below 5 degrees per call. Use basic primitives only, without VLA actions. Stop if observations or feedback indicate unsafe motion. Finish only after the onsite operator confirms the result.",
    1: "Tidy up the table. Left and right refer to the top camera view. Sort all three bottles by brand. Place all Pepsi bottles in the bag on the left and all Coca-Cola bottles in the bag on the right. Move the bowls to uncover the spoons, place the white spoon in the white bowl and the pink spoon in the pink bowl, and return both bowls to their original marked positions: white bowl on the left and pink bowl on the right.",
    2: "Tidy up the table. Left and right refer to the top camera view. Sort all three bottles by brand. Place all Coca-Cola bottles in the bag on the left and all Pepsi bottles in the bag on the right. Move the bowls to uncover the spoons, place the white spoon in the white bowl and the pink spoon in the pink bowl, and return both bowls to their original marked positions: white bowl on the left and pink bowl on the right.",
}


def resolve_task_language(task_id: int, task_language: str | None = None) -> str:
    """Resolve the same task instruction for the Agent and ENV server."""
    if type(task_id) is not int or task_id not in TASK_INSTRUCTIONS:
        raise ValueError(f"unknown YAM task {task_id!r}; choices are 0, 1, 2")
    language = str(task_language).strip() if task_language else ""
    return language or TASK_INSTRUCTIONS[task_id]


def classify_episode(status: dict) -> dict:
    """Use fresh server facts; never infer success or clear a stop."""
    if status.get("eval_success") is True:
        reason = "success"
    elif status.get("terminal_event") in {"failure", "abort"}:
        reason = status["terminal_event"]
    elif status.get("stop_requested") is not False:
        reason = "stopped"
    elif int(status.get("take_action_cnt", 0)) >= int(status.get("step_lim", 0)):
        reason = "budget_exhausted"
    elif status.get("ready_for_motion") is not True or not status.get("episode_id"):
        reason = "awaiting_ready"
    else:
        reason = "ready"
    return {
        "reason": reason,
        "can_continue": reason == "ready",
        **status,
    }

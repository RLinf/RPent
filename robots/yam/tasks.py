# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""YAM diagnostic task registration and deterministic exploration rules."""

TASK_INSTRUCTIONS = {
    "tabletop_cleanup_a": "Tidy up the table. Left and right refer to the top camera view. Sort all three bottles by brand. Place all Pepsi bottles in the bag on the left and all Coca-Cola bottles in the bag on the right. Move the bowls to uncover the spoons, place the white spoon in the white bowl and the pink spoon in the pink bowl, and return both bowls to their original marked positions: white bowl on the left and pink bowl on the right.",
    "tabletop_cleanup_b": "Tidy up the table. Left and right refer to the top camera view. Sort all three bottles by brand. Place all Coca-Cola bottles in the bag on the left and all Pepsi bottles in the bag on the right. Move the bowls to uncover the spoons, place the white spoon in the white bowl and the pink spoon in the pink bowl, and return both bowls to their original marked positions: white bowl on the left and pink bowl on the right.",
}

DIAGNOSTIC_TASKS = {103: "manual_primitive_test", 104: "vla_deployment_test"}


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

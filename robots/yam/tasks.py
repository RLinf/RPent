# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""YAM diagnostic task registration and deterministic exploration rules."""

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

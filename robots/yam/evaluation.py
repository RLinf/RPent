# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Persist real YAM outcomes through the public runner finalization hook."""

from __future__ import annotations

from pathlib import Path

from rpent.evaluation import RunFinalizationContext, write_json_atomic


def finalize_run(context: RunFinalizationContext) -> Path:
    """Keep planner claims separate from operator-backed environment success."""
    context.output_dir.mkdir(parents=True, exist_ok=True)
    status = (
        "unknown"
        if context.environment_success is None
        else "success"
        if context.environment_success
        else "not_successful"
    )
    return write_json_atomic(
        context.output_dir / "result.json",
        {
            "robot": context.robot_name,
            "task": dict(context.task_desc),
            "environment_success": context.environment_success,
            "status": status,
            "success_source": "environment_eval_success",
            "planner_finish_request": dict(context.finish_result or {}),
            "agent_error": context.agent_error,
            "elapsed_s": context.elapsed_s,
            "planner": context.planner,
            "model": context.model,
            "stats": dict(context.stats),
        },
    )

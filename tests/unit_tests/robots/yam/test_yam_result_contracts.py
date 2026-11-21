# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.

import json
from pathlib import Path

from robots.yam.evaluation import finalize_run
from rpent.evaluation import RunFinalizationContext
from rpent.robots.base import get_robot_spec as resolve_robot_spec


def test_result_records_the_run_verdict(tmp_path):
    path = finalize_run(
        RunFinalizationContext(
            output_dir=tmp_path,
            robot_name="yam",
            task_desc={"task": "place the cube"},
            environment_success=True,
            agent_error=None,
            elapsed_s=1.0,
            planner="api",
            model="gpt-5.5",
            reasoning_effort="low",
            max_turns=2,
            planner_timeout_s=None,
            finish_result={"status": "success"},
            stats={},
            truth_source="operator",
        )
    )

    record = json.loads(Path(path).read_text(encoding="utf-8"))
    # The result artifact and run.json record the same verdict; YAM's
    # ``eval_success`` is the onsite operator's receipt.
    assert record["success_source"] == "environment_eval_success"
    assert (
        record["accepted"],
        record["retryable"],
        record["failure_category"],
        record["stop_reason"],
        record["truth_source"],
    ) == (True, False, None, "solved", "operator")


def test_the_spec_declares_the_operator_as_its_truth_source():
    assert resolve_robot_spec("yam").truth_source == "operator"

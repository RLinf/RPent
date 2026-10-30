# Copyright 2026 The RPent Authors.
"""Public CLI chain with a scripted planner and real RoboDojo runtime."""

import json
import os
from pathlib import Path

import pytest

from tests.e2e_tests.common import (
    ScriptedToolCall,
    publish_check,
    run_scripted_policy_chain,
)

pytestmark = [
    pytest.mark.timeout(2400),
    pytest.mark.skipif(
        os.environ.get("RPENT_RUN_ROBODOJO_INTEGRATION") != "1",
        reason="requires a reserved GPU, RoboDojo assets and model checkpoints",
    ),
]


@pytest.mark.parametrize("control", ["primitive", "policy"])
def test_policy_chain(control, vla_backend):
    if control == "primitive":
        action = ScriptedToolCall(
            "set_gripper", {"arm": "left", "val": 0.5, "steps": 2}
        )
    else:
        arguments = {"chunks": 1}
        if vla_backend == "vpp2":
            arguments["instruction"] = "Pick up the pink bottle on the table"
        action = ScriptedToolCall(f"{vla_backend}_act", arguments)
    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"]) / (
        f"policy-chain-{action.name}"
        if vla_backend == "vpp2"
        else f"policy-chain-awomo05-{action.name}"
    )
    endpoint = os.environ.get("RPENT_ROBODOJO_VLA_ENDPOINT")
    result = run_scripted_policy_chain(
        robot="robodojo",
        robot_argv=[
            "--task",
            "put_bottles_into_dustbin",
            "--seed",
            "0",
            "--layout",
            "0",
            *(["--vla-backend", vla_backend] if vla_backend != "vpp2" else []),
            *(["--vla-endpoint", endpoint] if endpoint else []),
        ],
        output_dir=output,
        action=action,
        action_count_field="executed_actions",
        before_action_calls=(
            ScriptedToolCall("view_env_state", {"step": 0}),
            ScriptedToolCall(
                "segment", {"text_prompt": "the pink bottle on the table", "step": 0}
            ),
            ScriptedToolCall(
                "sample_world_xyz", {"view": "head", "pixels": [[240, 320]], "step": 0}
            ),
        ),
        include_images=True,
    )
    assert result["post_action_new_image_count"] > 0, result
    transcript = json.loads((output / result["transcript"]).read_text(encoding="utf-8"))
    assert transcript["error"] is None, transcript["error"]
    reads = {
        message["name"]: json.loads(message["content"])
        for message in transcript["messages"]
        if message.get("role") == "tool"
        and message.get("name") in {"view_env_state", "segment", "sample_world_xyz"}
    }
    assert set(reads) == {"view_env_state", "segment", "sample_world_xyz"}
    assert all(not payload.get("error") for payload in reads.values()), reads
    assert reads["sample_world_xyz"]["samples"][0]["valid_points"] > 0, reads
    segments = list(output.glob("segment_*.json/00.json"))
    assert len(segments) == 1, segments
    segment = json.loads(segments[0].read_text(encoding="utf-8"))
    assert segment["found"] and segment["score"] >= 0.2, segment
    assert segment["step_idx"] == 0 and segment["camera"] == "cam_head"
    result["definition"] = (
        f"scripted planner -> public CLI -> toolkit -> {action.name} -> finish"
    )
    publish_check(f"robodojo_{action.name}_policy_chain", result)

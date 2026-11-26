# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline contracts for the EFE toolkit."""

from __future__ import annotations

from robots.efe.toolkit import EfeToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.utils.logging import init_output_dir

READ_ONLY_EFE_TOOLS = [
    "efe_spec",
    "efe_observe",
    "efe_status",
    "efe_recover",
]

RAW_EFE_TOOLS = [*READ_ONLY_EFE_TOOLS, "efe_submit"]

SKILL_TOOLS = ["move_to_pose", "set_gripper"]


def _toolkit(tmp_path, client, *, skills=True):
    init_output_dir(tmp_path)
    memory = MemoryManager(tmp_path / "memory")
    return EfeToolkit(
        client=client,
        dashboard_events=NullDashboardEventSink(),
        memory=memory,
        state_output_dir=tmp_path,
        enable_skills=skills,
    )


def test_toolkit_registers_skills_and_shared_file_tools(tmp_path, efe_client):
    toolkit = _toolkit(tmp_path, efe_client)

    result = toolkit.execute_tool(
        "finish", {"status": "completed", "summary": "Session ended."}
    )

    assert toolkit.memory.root == (tmp_path / "memory").resolve()
    assert [declaration.name for declaration in toolkit.list_tools()] == [
        "read_text_file",
        "write_text_file",
        "list_dir",
        "finish",
        *READ_ONLY_EFE_TOOLS,
        "view_env_state",
        *SKILL_TOOLS,
    ]
    assert result.data == {
        "_finish": True,
        "status": "completed",
        "summary": "Session ended.",
    }


def test_raw_only_mode_hides_skills(tmp_path, efe_client):
    toolkit = _toolkit(tmp_path, efe_client, skills=False)

    assert toolkit.skills_enabled is False
    assert [declaration.name for declaration in toolkit.list_tools()] == [
        "read_text_file",
        "write_text_file",
        "list_dir",
        "finish",
        *RAW_EFE_TOOLS,
        "view_env_state",
    ]


def test_incompatible_layout_skips_skills_and_records_the_reason(tmp_path, efe_client):
    efe_client.spec["kinematics"] = {}
    toolkit = _toolkit(tmp_path, efe_client)

    names = [declaration.name for declaration in toolkit.list_tools()]

    assert toolkit.skills_enabled is False
    assert "efe_submit" in names
    assert "move_to_pose" not in names
    assert "set_gripper" not in names
    assert "move_to_pose" in toolkit.metrics()["skills_error"]


def test_tool_descriptions_do_not_describe_coding_artifacts(tmp_path, efe_client):
    toolkit = _toolkit(tmp_path, efe_client)

    descriptions = {
        declaration.name: declaration.description or ""
        for declaration in toolkit.list_tools()
    }

    for name in ("read_text_file", "write_text_file", "list_dir"):
        lowered = descriptions[name].lower()
        assert "recipe" not in lowered
        assert "audit" not in lowered


def test_state_records_keep_the_frames_of_their_episode(tmp_path, efe_client):
    efe_client.spec["budget"] = {"reset": 2}
    toolkit = _toolkit(tmp_path, efe_client, skills=False)

    toolkit.execute_tool("efe_observe", {"cameras": ["wrist_cam"]})
    toolkit.execute_tool(
        "efe_submit", {"action_type": "joint_position", "actions": [[0.0] * 7]}
    )
    before = toolkit.execute_tool("view_env_state", {"step": -1})

    toolkit.execute_tool("efe_reset", {})
    after = toolkit.execute_tool("view_env_state", {"step": -1})

    assert len(before.images) == 1
    assert before.data["camera_images"]["wrist_cam"].endswith(
        "camera_wrist_cam_ep000_s000000.png"
    )
    assert after.images == []
    assert after.data["state"]["episode"] == 1


def test_solved_reads_the_service_verdict_not_finish(tmp_path, efe_client):
    toolkit = _toolkit(tmp_path, efe_client)

    assert toolkit.solved() is False
    efe_client.done = True
    efe_client.done_reason = "success"
    assert toolkit.solved() is True


def test_metrics_count_session_calls_and_service_progress(tmp_path, efe_client):
    toolkit = _toolkit(tmp_path, efe_client)

    toolkit.execute_tool("set_gripper", {"opening": 0.02})
    toolkit.execute_tool("move_to_pose", {"x": 0.0, "y": 0.0, "z": 0.3})

    metrics = toolkit.metrics()

    assert metrics["tool_calls"] == 2
    assert metrics["skill_calls"] == 2
    assert metrics["efe_submit_calls"] == 1
    assert metrics["efe_control_steps"] == 1
    assert metrics["skills_enabled"] is True
    assert metrics["skills_error"] is None


def test_finish_contract_rejects_unsupported_status(tmp_path, efe_client):
    toolkit = _toolkit(tmp_path, efe_client)
    finish = next(
        declaration
        for declaration in toolkit.list_tools()
        if declaration.name == "finish"
    )

    assert "verifier" in finish.description.lower()
    assert toolkit.execute_tool(
        "finish", {"status": "success", "summary": "x"}
    ).is_error

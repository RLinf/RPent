"""LIBERO environment extension."""
from __future__ import annotations

from physical_agent.envs.base import EnvSpec
from physical_agent.envs.libero.prompt_bundle import PROMPTS
from physical_agent.envs.libero.tools import (
    TOOL_HANDLERS,
    TOOLS_SPEC,
    set_driver_client,
    stop_recording_and_save,
)

_ALLOWED_MCP_TOOL_NAMES = tuple(
    f"mcp__physical_agent__{name}"
    for name in [
        "move_to",
        "pi0_pick",
        "release",
        "set_gripper",
        "rotate_wrist",
        "rotate_pitch",
        "move_pose",
        "view_driver_state",
        "view_camera_meta",
        "back_project",
        "read_text_file",
        "write_text_file",
        "mcp_list_dir",
        "finish",
    ]
)


def get_env_spec() -> EnvSpec:
    return EnvSpec(
        name="libero",
        prompts=PROMPTS,
        tools_spec=TOOLS_SPEC,
        tool_handlers=TOOL_HANDLERS,
        set_driver_client=set_driver_client,
        stop_recording_and_save=stop_recording_and_save,
        allowed_mcp_tool_names=_ALLOWED_MCP_TOOL_NAMES,
    )

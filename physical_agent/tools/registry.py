"""Tool registry composed from common tools and one environment spec."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from physical_agent.envs.base import EnvSpec
from physical_agent.envs.registry import get_env_spec
from physical_agent.tools import common


@dataclass(frozen=True)
class ToolRegistry:
    """Agent-facing tools for one configured environment."""

    env_spec: EnvSpec

    def get_tools_spec(self) -> list[dict[str, Any]]:
        return common.bind_output_dir_descriptions([
            *common.TOOLS_SPEC,
            *self.env_spec.tools_spec,
        ])

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> dict[str, Any]:
        handler = common.TOOL_HANDLERS.get(name)
        if handler is None:
            handler = self.env_spec.tool_handlers.get(name)
        if handler is None:
            return {"error": f"unknown tool: {name}"}
        try:
            return handler(**input_dict)
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}", "got": input_dict}
        except Exception as e:
            import traceback

            return {"error": str(e), "traceback": traceback.format_exc()}


def create_tool_registry(env_name: str | None = None) -> ToolRegistry:
    return ToolRegistry(env_spec=get_env_spec(env_name))

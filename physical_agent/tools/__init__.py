"""Agent tool declarations, handlers, and result serialization."""

from physical_agent.tools.common import tool_result_to_content_blocks
from physical_agent.tools.registry import ToolRegistry, create_tool_registry

__all__ = [
    "ToolRegistry",
    "create_tool_registry",
    "tool_result_to_content_blocks",
]

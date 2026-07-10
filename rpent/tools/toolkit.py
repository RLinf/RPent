"""Base class for agent tools.

``Toolkit`` is the agent-facing tool container. Subclasses can register tools
during ``__init__`` via :meth:`Toolkit.add_tool`; the cerebrum calls the tools through :meth:`Toolkit.get_tools_spec` and
:meth:`Toolkit.execute_tool`.
"""
from __future__ import annotations

import base64
import json
from collections.abc import Callable
from dataclasses import dataclass, field
import traceback
from typing import Any, ClassVar

from rpent.utils.templates import substitute


@dataclass
class ToolResult:
    """Result of executing one tool call.

    Carries the raw result dict (for logging and finish-signal detection)
    alongside the Anthropic-shaped content blocks the LLM consumes.
    """

    name: str
    result: dict[str, Any]
    call_id: str | None = None

    content_blocks: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    is_finish: bool = field(default=False, init=False)

    #: Max bytes of the text block emitted in :attr:`content_blocks`.
    MAX_TEXT_BYTES_IN_RESULT: ClassVar[int] = 60000

    def __post_init__(self) -> None:
        self.content_blocks = self._build_content_blocks()
        self.is_finish = bool(
            isinstance(self.result, dict) and self.result.get("_finish")
        )

    def _build_content_blocks(self) -> list[dict[str, Any]]:
        """Build Anthropic-shaped content blocks (text + optional images).

        Strips image byte payloads from the text block and emits them as
        separate base64 image blocks so the LLM receives the state images as
        multimodal content.
        """
        result = self.result
        if not isinstance(result, dict):
            return [{"type": "text", "text": str(result)[:self.MAX_TEXT_BYTES_IN_RESULT]}]

        result_for_text = dict(result)
        image = result_for_text.pop("_image_bytes", None)
        image_cam = result_for_text.pop("_image_cam_bytes", None)
        image_wrist = result_for_text.pop("_image_wrist_bytes", None)
        text = json.dumps(result_for_text, indent=2, default=str)
        if len(text) > self.MAX_TEXT_BYTES_IN_RESULT:
            text = text[:self.MAX_TEXT_BYTES_IN_RESULT] + "\n[truncated]"

        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]

        def _add_image_bytes(data_bytes: bytes) -> None:
            data = base64.b64encode(data_bytes).decode("utf-8")
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": data,
                },
            })

        if image:
            _add_image_bytes(image)
        if image_cam:
            _add_image_bytes(image_cam)
        if image_wrist:
            _add_image_bytes(image_wrist)
        return blocks


class Toolkit:
    """Base toolkit: registers common tools and dispatches tool calls.

    Subclasses extend ``__init__`` (calling ``super().__init__()`` first)
    and register additional tools with :meth:`add_tool`. Env-specific
    subclasses receive their env/model/etc. as constructor arguments and
    build the underlying primitive driver in ``__init__``; the toolkit
    base class only contributes the common file/IO tools. Override
    :meth:`close` to release env-side drivers at the end of the run.
    """

    def __init__(self, *, dashboard: Any = None) -> None:
        # name -> (spec, handler)
        self._tools: dict[str, tuple[dict[str, Any], Callable[..., dict[str, Any]]]] = {}
        self._dashboard = dashboard
        self._register_common_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_tool(
        self,
        name: str,
        spec: dict[str, Any],
        handler: Callable[..., dict[str, Any]],
    ) -> None:
        """Register one tool under ``name`` with its schema and handler.

        Args:
            name: Tool name as the LLM sees it (e.g. ``"read_text_file"``).
            spec: Anthropic-shaped tool schema dict (``name``,
                ``description``, ``input_schema``).
            handler: Callable invoked with the tool's input kwargs; returns
                a result dict.
        """
        self._tools[name] = (spec, handler)

    def _register_common_tools(self) -> None:
        """Register the file/IO tools shared by every run."""
        from rpent.tools import common

        for spec in common.TOOLS_SPEC:
            name = spec["name"]
            self.add_tool(name, spec, common.TOOL_HANDLERS[name])

    # ------------------------------------------------------------------
    # Cerebrum-facing API
    # ------------------------------------------------------------------

    def get_tools_spec(self) -> list[dict[str, Any]]:
        """Return the tool schemas the LLM sees."""
        return substitute(
            [spec for spec, _ in self._tools.values()]
        )

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call to its registered handler."""
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(name=name, result={"error": f"unknown tool: {name}"})
        handler = entry[1]
        try:
            result = handler(**input_dict)
        except TypeError as e:
            result = {"error": f"bad arguments for {name}: {e}", "got": input_dict}
        except Exception as e:
            result = {"error": str(e), "traceback": traceback.format_exc()}
        if self._dashboard is not None:
            self._dashboard.on_tool_result(name, result)
        return ToolResult(name=name, result=result)

    # ------------------------------------------------------------------
    # Driver lifecycle hooks (overridden by env toolkits)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the env driver at end of run. Default: no-op."""

    def write_recipe(self, recipe_tag: str) -> str | None:
        """Write a replay recipe for this env, if supported."""
        return None

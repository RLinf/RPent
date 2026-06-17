"""Agent-side LLM tool registry.

Defines the schemas the LLM sees, the dispatch glue for tool calls, and
the conversion of tool results into multimodal content blocks.

Generic file/IO tools live in this module. Environment-specific tools are
provided by :mod:`physical_agent.envs` and merged by :mod:`physical_agent.tools`.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from physical_agent.utils.config import get_repo_root
from physical_agent.utils.logging import get_output_dir



def _require_output_dir() -> Path:
    d = get_output_dir()
    if d is None:
        raise RuntimeError("init_output_dir() must be called before using tools")
    return d


def _output_dir_desc() -> str:
    d = get_output_dir()
    return str(d) if d is not None else "<unset>"


# ---------------------------------------------------------------------------
# Tool schema declarations (Anthropic-shaped canonical schema)
# ---------------------------------------------------------------------------

TOOLS_SPEC: list[dict] = [
    {
        "name": "read_text_file",
        "description": (
            "Read a UTF-8 text file. Use for guides (STRICT_HYBRID_GUIDE.md, "
            "PRO_HYBRID_GUIDE.md, env_calibration.md), past recipe JSONLs, "
            "audit JSONs, and memory files. Large files are truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or repo-relative path"},
                "max_chars": {"type": "integer", "description": "Max chars (default 40000)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_text_file",
        "description": (
            "Write a UTF-8 text file (creates parent dirs). Use this to save "
            "the working recipe JSONL and the final audit JSON at the end of "
            "a successful run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "mcp_list_dir",
        "description": (
            "List files in a directory (non-recursive). Default = current output dir. "
            "Use to inspect the driver working directory or to discover existing "
            "recipes in workspace_pro/results_*_pert/."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Default: current output dir"},
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = get_repo_root() / p
    return p


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + f"\n\n[TRUNCATED — file is {len(text)} chars, showed first {max_chars}]"
    )


def read_text_file(path: str, max_chars: int = 40000) -> dict:
    p = _resolve(path)
    if not p.exists():
        return {"error": f"file not found: {p}"}
    if p.is_dir():
        return {"error": f"is a directory: {p}"}
    try:
        text = p.read_text(errors="replace")
    except Exception as e:
        return {"error": str(e)}
    return {"path": str(p), "size": len(text), "content": _truncate(text, max_chars)}


def write_text_file(path: str, content: str) -> dict:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"path": str(p), "bytes_written": len(content.encode("utf-8"))}


def mcp_list_dir(path: str = "") -> dict:
    # Default to the current output dir (so parallel agents see their own).
    p = _resolve(path) if path else _require_output_dir()
    if not p.exists():
        return {"error": f"directory not found: {p}"}
    files = sorted(os.listdir(p))
    return {"path": str(p), "count": len(files), "files": files}


TOOL_HANDLERS: dict = {
    "read_text_file": read_text_file,
    "write_text_file": write_text_file,
    "mcp_list_dir": mcp_list_dir,
}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def bind_output_dir_descriptions(tools_spec: list[dict]) -> list[dict]:
    """Return tool schemas with descriptions bound to the current output dir."""
    tools = json.loads(json.dumps(tools_spec))
    replacements = {
        "current output dir": _output_dir_desc(),
        "Default: current output dir": f"Default: {_output_dir_desc()}",
    }
    for tool in tools:
        desc = tool.get("description", "")
        for old, new in replacements.items():
            desc = desc.replace(old, new)
        tool["description"] = desc
        props = tool.get("input_schema", {}).get("properties", {})
        for prop in props.values():
            prop_desc = prop.get("description", "")
            for old, new in replacements.items():
                prop_desc = prop_desc.replace(old, new)
            prop["description"] = prop_desc
    return tools


def get_tools_spec() -> list[dict]:
    return bind_output_dir_descriptions(TOOLS_SPEC)


def execute_tool(name: str, input_dict: dict) -> dict:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return handler(**input_dict)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}", "got": input_dict}
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Convert tool result -> Anthropic content blocks (text + optional image)
# ---------------------------------------------------------------------------

MAX_TEXT_BYTES_IN_RESULT = 60000


def tool_result_to_content_blocks(result):
    """Build a list of Anthropic content blocks from a tool result dict.

    If the result has private image bytes, those PNGs are included as base64
    image blocks (alongside a text block with the JSON state).
    """
    if not isinstance(result, dict):
        return [{"type": "text", "text": str(result)[:MAX_TEXT_BYTES_IN_RESULT]}]

    result_for_text = dict(result)
    image = result_for_text.pop("_image_bytes", None)
    image_cam = result_for_text.pop("_image_cam_bytes", None)
    text = json.dumps(result_for_text, indent=2, default=str)
    if len(text) > MAX_TEXT_BYTES_IN_RESULT:
        text = text[:MAX_TEXT_BYTES_IN_RESULT] + "\n[truncated]"

    blocks = [{"type": "text", "text": text}]

    def _add_image_bytes(data_bytes: bytes):
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
    return blocks

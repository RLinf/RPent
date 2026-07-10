"""litellm proxy — inline namespace tools for API, restore namespace in
streaming response so Codex routes via its own MCP registry.

Problem
-------

Codex sends ``type: "namespace"`` tool definitions.  The upstream API
(Infini-AI) rejects namespace format, so the proxy inlines them as
``type: "function"`` (``mcp__rpent__echo``).  The API then returns
``function_call`` *without* a ``namespace`` field — Codex cannot route the
flattened name through its MCP namespace registry.

Solution
--------

1.  **Pre-call** — inline namespace tools as function tools (required by API).
2.  **Streaming iterator** — on each ``response.output_item.{added,done}``
    chunk whose ``item.type == "function_call"``:
    a.  If ``"mcp__"`` in the name → restore namespace and tool name (existing).
    b.  If name is a known MCP short name → add namespace field only.
3.  Non-function tools (web_search, …) are silently dropped.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from litellm.integrations.custom_logger import CustomLogger


def _inline_tools(data: dict) -> bool:
    """Inline namespace tools as function tools in-place."""
    tools = data.get("tools")
    if not isinstance(tools, list):
        return False

    inlined: list[dict[str, Any]] = []
    modified = False
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "namespace":
            ns_name: str = t.get("name", "")
            ns_tools = t.get("tools") or []
            _debug(f"  namespace {ns_name}: {len(ns_tools)} nested tools")
            modified = True
            for i, mcp_tool in enumerate(ns_tools):
                if not isinstance(mcp_tool, dict):
                    _debug(f"    tool[{i}]: not a dict, skipped")
                    continue
                fn_name: str = f"{ns_name}__{mcp_tool.get('name', '')}"
                # Codex SDK serialises MCP tool schemas as ``parameters``
                # (OpenAI / Responses-API format), *not* ``input_schema``.
                raw_params = mcp_tool.get("parameters") or mcp_tool.get("input_schema") or {"type": "object"}
                _debug(f"    inlined -> {fn_name}: params_keys={list(dict(raw_params).keys())}")
                params = dict(raw_params)
                if not isinstance(params.get("properties"), dict):
                    params["properties"] = {}
                if not isinstance(params.get("required"), list):
                    params.pop("required", None)
                inlined.append({
                    "type": "function",
                    "name": fn_name,
                    "description": mcp_tool.get("description", ""),
                    "parameters": params,
                })
        elif isinstance(t, dict) and t.get("type") == "function":
            inlined.append(t)
        else:
            # web_search, image_generation, computer, etc. — drop silently
            pass

    if modified:
        data["tools"] = inlined
    return modified


# SDK-internal function tools the Codex binary registers for its own use.
# These should NOT get namespace restoration when called by short name;
# the SDK already knows how to route them directly.
_SDK_TOOLS: frozenset[str] = frozenset({
    "exec_command", "write_stdin",
    "read_text_file", "write_text_file",
    "list_mcp_resources", "list_mcp_resource_templates", "call_mcp_tool",
})


def _extract_mcp_namespace_map(data: dict) -> dict[str, str]:
    """Scan tool list for function tools whose name contains ``__`` (MCP
    tools inlined by either the proxy or the Codex SDK) and return a
    mapping of ``{short_name: namespace}``.

    Example::
        {"back_project": "mcp__rpent",
         "view_driver_state": "mcp__rpent"}
    """
    tools = data.get("tools")
    if not isinstance(tools, list):
        return {}

    result: dict[str, str] = {}
    for t in tools:
        if not isinstance(t, dict):
            continue
        name: str | None = t.get("name")
        if not isinstance(name, str):
            continue
        if "__" not in name:
            continue
        parts = name.split("__")
        if len(parts) < 2:
            continue
        namespace = "__".join(parts[:-1])
        short_name = parts[-1]
        result[short_name] = namespace

    return result


def _maybe_strip_exec_command(data: dict) -> None:
    """Remove ``exec_command`` and ``write_stdin`` from the tool list when
    ``CODEX_PROXY_STRIP_EXEC=1`` is set.

    This forces the model to use inlined MCP function tools for robot control
    instead of falling back to the familiar ``exec_command`` + curl pattern.
    """
    if os.environ.get("CODEX_PROXY_STRIP_EXEC") != "1":
        return
    tools = data.get("tools")
    if not isinstance(tools, list):
        return
    stripped = [t for t in tools if t.get("name") not in ("exec_command", "write_stdin")]
    if len(stripped) != len(tools):
        _debug(f"_maybe_strip_exec_command: {len(tools)} -> {len(stripped)} tools (removed exec_command/write_stdin)")
        data["tools"] = stripped


_DUMP_PATH = os.environ.get("CODEX_PROXY_DUMP_REQUEST")
_DUMP_RESPONSE_PATH = os.environ.get("CODEX_PROXY_DUMP_RESPONSE")
_DEBUG_LOG = os.environ.get("CODEX_PROXY_DEBUG_LOG")

def _debug(msg: str) -> None:
    if not _DEBUG_LOG:
        return
    try:
        Path(_DEBUG_LOG).parent.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_LOG, "a") as f:
            f.write(f"{msg}\n")
    except Exception:
        pass


def _maybe_dump(data: dict) -> None:
    """Append the inbound request body to a JSONL file when
    ``CODEX_PROXY_DUMP_REQUEST`` points to a writable path."""
    if not _DUMP_PATH:
        return
    try:
        Path(_DUMP_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(_DUMP_PATH, "a") as f:
            f.write(json.dumps(data, default=str))
            f.write("\n")
    except Exception:
        pass



class _CodexToolFilter(CustomLogger):
    """Pre-call hook: inline namespace tools as function tools so the
    upstream Chat Completions endpoint accepts the request while keeping
    Codex's ``mcp__``-based MCP routing intact."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mcp_tool_namespaces: dict[str, str] = {}

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[dict]:
        _debug(f"=== pre_call_hook called: call_type={call_type} ===")
        input_val = data.get("input")
        if isinstance(input_val, list):
            _debug(f"input items: {len(input_val)}")
            for ii, item in enumerate(input_val):
                if isinstance(item, dict):
                    itype = item.get('type','?')
                    _debug(f"  input[{ii}]: type={itype} role={item.get('role','?')}")
                    if itype == 'function_call':
                        _debug(f"    name={item.get('name','')} args={str(item.get('arguments',''))[:200]}")
                        _debug(f"    call_id={item.get('call_id','')} status={item.get('status','')}")
                    elif itype == 'function_call_output':
                        _debug(f"    call_id={item.get('call_id','')} output={str(item.get('output',''))[:300]}")
                    else:
                        content = item.get('content', '') or item.get('text', '')
                        if isinstance(content, str) and content.strip():
                            _debug(f"    content preview: {content[:100]}...")
                        elif isinstance(content, list):
                            _debug(f"    content blocks: {len(content)}")
        _debug(f"data keys: {list(data.keys())}")
        _debug(f"tools key exists: {'tools' in data}")
        _debug(f"stream={data.get('stream', 'NOT_SET')} tool_choice={data.get('tool_choice', 'NOT_SET')}")

        tools = data.get("tools")
        if isinstance(tools, list):
            _debug(f"tools count: {len(tools)}")
            for i, t in enumerate(tools):
                _debug(f"  tool[{i}]: type={t.get('type', '?')} name={t.get('name', t.get('function', {}).get('name', '?'))}")
        else:
            _debug(f"tools type: {type(tools).__name__} = {str(tools)[:200]}")
        _inline_tools(data)
        _maybe_strip_exec_command(data)
        # Build a mapping of MCP tool short names → namespace from the
        # (possibly inlined) function tool list.  This is used in response
        # processing to restore the namespace on function_calls that come
        # back with either the full ``ns__tool`` name or just ``tool``.
        self._mcp_tool_namespaces = _extract_mcp_namespace_map(data)
        _debug(f"MCP tool namespace map: {len(self._mcp_tool_namespaces)} entries")
        _maybe_dump(data)
        return data

    async def _convert_to_namespace_response(self, response: Any) -> None:
        """Restore namespace field on function_call items the API flattened."""
        if not self._mcp_tool_namespaces:
            return
        try:
            if hasattr(response, "model_dump"):
                raw = response.model_dump(mode="json", exclude_none=True)
            elif hasattr(response, "dict"):
                raw = response.dict(exclude_none=True)
            elif isinstance(response, dict):
                raw = response
            else:
                _debug(f"  _convert: unsupported response type {type(response).__name__}")
                return
        except Exception as e:
            _debug(f"  _convert: model_dump error: {e}")
            return

        output_items = raw.get("output") if isinstance(raw, dict) else None
        if not isinstance(output_items, list):
            _debug(f"  _convert: no output list found, keys={list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__}")
            return

        modified = False
        for idx, item in enumerate(output_items):
            if not isinstance(item, dict):
                continue
            if item.get("type") != "function_call":
                continue
            name: str = item.get("name", "")
            if "__" in name:
                # Full name: mcp__rpent__back_project → namespace + short name
                parts = name.split("__")
                if len(parts) < 2:
                    continue
                ns = "__".join(parts[:-1])
                tool = parts[-1]
                item["namespace"] = ns
                item["name"] = tool
                modified = True
                _debug(f"  _convert: function_call[{idx}]: {name} -> namespace={ns} name={tool}")
            elif name in self._mcp_tool_namespaces and name not in _SDK_TOOLS:
                # Short name, known MCP tool: add namespace so SDK can route
                ns = self._mcp_tool_namespaces[name]
                item["namespace"] = ns
                modified = True
                _debug(f"  _convert: function_call[{idx}]: {name} -> namespace={ns} (short name)")

        if modified and hasattr(response, "model_dump"):
            try:
                if hasattr(response, "output") and isinstance(response.output, list):
                    for response_item, raw_item in zip(response.output, output_items):
                        if hasattr(response_item, "namespace") and "namespace" in raw_item:
                            response_item.namespace = raw_item["namespace"]
                        if hasattr(response_item, "name") and "name" in raw_item:
                            response_item.name = raw_item["name"]
            except Exception as e:
                _debug(f"  _convert: response update error: {e}")

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        count = 0
        async for chunk in response:
            count += 1
            self._fix_function_call_namespace(chunk)
            yield chunk

    def _fix_function_call_namespace(self, chunk: Any) -> None:
        """Restore namespace on function_call stream items."""
        if not self._mcp_tool_namespaces:
            return
        try:
            if hasattr(chunk, "model_dump"):
                raw = chunk.model_dump(mode="json", exclude_none=True)
            elif hasattr(chunk, "dict"):
                raw = chunk.dict(exclude_none=True)
            elif isinstance(chunk, dict):
                raw = chunk
            else:
                return
        except Exception:
            return

        chunk_type = raw.get("type", "") if isinstance(raw, dict) else ""
        if chunk_type not in ("response.output_item.added", "response.output_item.done"):
            return

        item = raw.get("item")
        if not isinstance(item, dict):
            return
        if item.get("type") != "function_call":
            return
        name: str = item.get("name", "")

        try:
            ci = getattr(chunk, "item", None)
            if ci is None:
                return
            ci_type = type(ci).__name__

            if "mcp__" in name:
                sep = "__"
                parts = name.split(sep)
                if len(parts) < 2:
                    return
                namespace = sep.join(parts[:-1])
                tool = parts[-1]
                if isinstance(ci, dict):
                    ci["namespace"] = namespace
                    ci["name"] = tool
                else:
                    ci.name = tool
                    ci.namespace = namespace
                _debug(f"  _fix_fn_ns: {name} -> namespace={namespace} name={tool} (item_type={ci_type})")
            elif name in self._mcp_tool_namespaces and name not in _SDK_TOOLS:
                namespace = self._mcp_tool_namespaces[name]
                if isinstance(ci, dict):
                    ci["namespace"] = namespace
                else:
                    ci.namespace = namespace
                _debug(f"  _fix_fn_ns (short): {name} -> namespace={namespace} (item_type={ci_type})")
        except Exception as e:
            _debug(f"  _fix_fn_ns error: {e}")

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: Any,
    ) -> None:
        _debug(f"=== post_call_success_hook: type={type(response).__name__} ===")
        _debug(f"  stream={data.get('stream', False)}")
        try:
            if hasattr(response, "model_dump"):
                raw = response.model_dump(mode="json", exclude_none=True)
            elif hasattr(response, "dict"):
                raw = response.dict(exclude_none=True)
            elif isinstance(response, dict):
                raw = response
            else:
                raw = None
            if isinstance(raw, dict):
                outputs = raw.get("output")
                if isinstance(outputs, list):
                    _debug(f"  output items: {len(outputs)}")
                    for idx, item in enumerate(outputs):
                        if isinstance(item, dict):
                            _debug(f"    output[{idx}]: type={item.get('type','?')} name={item.get('name','?')} ns={item.get('namespace','?')}")
        except Exception as e:
            _debug(f"  log error: {e}")

        await self._convert_to_namespace_response(response)
        if not _DUMP_RESPONSE_PATH:
            return
        try:
            Path(_DUMP_RESPONSE_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(_DUMP_RESPONSE_PATH, "a") as f:
                if hasattr(response, "model_dump"):
                    resp_data = response.model_dump(mode="json", exclude_none=True)
                elif hasattr(response, "dict"):
                    resp_data = response.dict(exclude_none=True)
                elif isinstance(response, dict):
                    resp_data = response
                else:
                    resp_data = {"type": type(response).__name__}
                f.write(json.dumps(resp_data, default=str) + "\n")
        except Exception as e:
            _debug(f"response dump error: {e}")


# litellm imports this attribute when the YAML lists
# ``litellm_callbacks.codex_tool_filter``.
codex_tool_filter = _CodexToolFilter()

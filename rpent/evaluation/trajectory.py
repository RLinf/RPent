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


"""Read lossless RPent traces without importing model or simulator packages."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any


def restore(value: Any, root: Path, *, paths: bool = False) -> Any:
    """Restore saved assets after verifying their original byte hashes."""
    if isinstance(value, dict) and "$asset" in value:
        path = root / value["$asset"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != value["sha256"]:
            raise ValueError(f"asset checksum mismatch: {path}")
        if paths:
            return str(path.resolve())
        if value["$encoding"] == "bytes":
            return data
        return value["data_uri_prefix"] + base64.b64encode(data).decode()
    if isinstance(value, list):
        return [restore(x, root, paths=paths) for x in value]
    if isinstance(value, dict):
        return {k: restore(v, root, paths=paths) for k, v in value.items()}
    return value


def read_response(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a complete ordinary or streamed assistant response."""
    metadata = json.loads((directory / "metadata.json").read_text())
    if not metadata.get("complete") or metadata.get("status_code") != 200:
        raise ValueError(f"incomplete HTTP response: {directory}")
    raw = (directory / "response.body").read_text()
    if "text/event-stream" not in metadata.get("content_type", ""):
        result = json.loads(raw)
        choice = result["choices"][0]
        message, reason = choice["message"], choice["finish_reason"]
        usage = result.get("usage")
    else:
        message = {"role": "assistant", "content": "", "reasoning_content": ""}
        calls = {}
        reason = usage = None
        done = False
        for line in raw.splitlines():
            if not line.startswith("data:"):
                continue
            line = line[5:].strip()
            if line == "[DONE]":
                done = True
                continue
            chunk = json.loads(line)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                reason = choice.get("finish_reason") or reason
                delta = choice.get("delta", {})
                for key in ("content", "reasoning_content"):
                    message[key] += delta.get(key) or ""
                for call in delta.get("tool_calls") or []:
                    index = call.get("index", 0)
                    dst = calls.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    dst["id"] += call.get("id") or ""
                    for key in ("name", "arguments"):
                        dst["function"][key] += call.get("function", {}).get(key) or ""
        if not done:
            raise ValueError(f"SSE lacks DONE: {directory}")
        if calls:
            message["tool_calls"] = [calls[k] for k in sorted(calls)]
    if reason not in ("stop", "tool_calls"):
        raise ValueError(f"unusable finish reason {reason}: {directory}")
    for call in message.get("tool_calls") or []:
        args = call["function"]["arguments"]
        if isinstance(args, str):
            json.loads(args)
    return message, {"finish_reason": reason, "usage": usage, **metadata}


def complete_turns(episode: Path) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    """Match complete wire replies to the native response selected for each turn."""
    trace = episode / "api_trace"
    completed = {p.stem for p in (trace / "responses").glob("*.json")}
    turns = {}
    for directory in sorted((trace / "requests").glob("*")):
        if not directory.is_dir():
            continue
        try:
            message, meta = read_response(directory)
            if meta["native_id"] not in completed:
                continue
            native = json.loads(
                (trace / "responses" / f"{meta['native_id']}.json").read_text()
            )[0]
            parts = native["parts"]
            for kind, field in [("thinking", "reasoning_content"), ("text", "content")]:
                content = "".join(p["content"] for p in parts if p["part_kind"] == kind)
                if content != (message.get(field) or ""):
                    raise ValueError(f"wire/native {field} mismatch: {directory}")
            expected = [p for p in parts if p["part_kind"] == "tool-call"]
            actual = message.get("tool_calls") or []
            if len(expected) != len(actual):
                raise ValueError(f"wire/native tool count mismatch: {directory}")
            for a, b in zip(expected, actual, strict=True):
                arguments = a.get("args") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                wire_arguments = b["function"]["arguments"]
                if isinstance(wire_arguments, str):
                    wire_arguments = json.loads(wire_arguments)
                if (
                    a["tool_name"] != b["function"]["name"]
                    or a["tool_call_id"] != b["id"]
                    or arguments != wire_arguments
                ):
                    raise ValueError(f"wire/native tool mismatch: {directory}")
            turns[meta["native_id"]] = (directory, message, meta)
        except (OSError, ValueError, KeyError):
            continue
    return [turns[k] for k in sorted(turns)]


def audit_episode(
    episode: Path, *, verify_assets: bool = False
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any], dict[str, Any]]]]:
    """Require native success agreement and complete, usable responses for every turn."""
    result = json.loads((episode / "result.json").read_text())
    transcript = json.loads(next(episode.glob("transcript_*.json")).read_text())
    states = json.loads((episode / "states.json").read_text())["steps"]
    json.loads((episode / "rollout.json").read_text())
    if result.get("agent_error") or transcript.get("error"):
        raise ValueError(result.get("agent_error") or transcript["error"])
    native_success = bool(states and states[-1]["terminated"])
    if bool(result["success"]) != native_success:
        raise ValueError(f"result/native success mismatch: {episode}")
    turns = complete_turns(episode)
    if not turns or len(turns) != result["stats"]["turns_used"]:
        raise ValueError(
            f"completed turn coverage {len(turns)} != {result['stats']['turns_used']}"
        )
    if verify_assets:
        for d, _, _ in turns:
            restore(json.loads((d / "request.json").read_text()), episode / "api_trace")
    return result, turns

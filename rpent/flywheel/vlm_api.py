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

"""Convert recorded API planner turns to VLM training samples."""

from pathlib import PurePosixPath
from typing import Any

from pydantic_ai.messages import RetryPromptPart, ToolReturnPart

from rpent.flywheel.vlm import (
    VLMImage,
    VLMMessage,
    VLMSample,
    VLMText,
    VLMToolDefinition,
)
from rpent.planner.utils.sdk import json_object
from rpent.session.records import ArtifactRef


class TurnConversionError(ValueError):
    """Identify a rejected turn by a stable report reason."""

    def __init__(self, reason: str, detail: str) -> None:
        """Attach the report reason to the conversion error."""
        super().__init__(detail)
        self.reason = reason


def _assistant_message(parts: list[dict[str, Any]]) -> VLMMessage:
    """Build an assistant training message from recorded text and tool calls."""
    message = VLMMessage("assistant")
    for part in parts:
        kind = part["part_kind"]
        if kind == "text":
            if part["content"]:
                message.content.append({"type": "text", "text": part["content"]})
        elif kind == "thinking":
            continue
        elif kind == "tool-call":
            args = part["args"]
            arguments = {} if args is None or args == "" else json_object(args)
            if arguments is None:
                raise TurnConversionError(
                    "invalid_tool_arguments",
                    f"Invalid tool arguments for {part['tool_call_id']}",
                )
            message.tool_calls.append(
                {
                    "id": part["tool_call_id"],
                    "name": part["tool_name"],
                    "arguments": arguments,
                }
            )
        else:
            raise ValueError(f"Unsupported API response part: {kind}")
    return message


def _image(item: dict[str, Any]) -> VLMImage:
    """Convert a recorded image artifact reference into a VLM image."""
    data = item.get("data")
    media_type = item.get("media_type")
    if not (
        item.get("kind") == "binary"
        and isinstance(data, dict)
        and data.get("$type") == "artifact_ref"
        and data.get("kind") == "bytes"
        and isinstance(media_type, str)
        and media_type.startswith("image/")
    ):
        raise ValueError("Expected an image byte artifact reference")
    path = PurePosixPath(data["path"])
    if len(path.parts) < 2 or path.parts[0] != "artifacts" or ".." in path.parts:
        raise ValueError(f"Invalid image artifact path: {path}")
    return VLMImage(
        type="image",
        artifact=ArtifactRef(path=data["path"], kind="bytes", media_type=media_type),
        observation_id=None,
    )


def _tool_content(part: dict[str, Any]) -> list[VLMText | VLMImage]:
    """Convert a tool result into text and image blocks."""
    result = ToolReturnPart(
        part["tool_name"],
        part["content"],
        part["tool_call_id"],
        outcome=part.get("outcome", "success"),
    )
    items = result.content if isinstance(result.content, list) else [result.content]
    images = {
        index: _image(item)
        for index, item in enumerate(items)
        if isinstance(item, dict)
        and item.get("kind")
        in (
            "binary",
            "image-url",
            "audio-url",
            "video-url",
            "document-url",
            "uploaded-file",
        )
    }
    if not images or result.outcome == "failed":
        if images:
            values = [item for index, item in enumerate(items) if index not in images]
            result.content = values[0] if len(values) == 1 else values
        return [{"type": "text", "text": result.model_response_str()}, *images.values()]
    return [
        images[index] if index in images else VLMText(type="text", text=text)
        for index, text in enumerate(result.content_items(mode="str"))
    ]


def _request_message(part: dict[str, Any]) -> VLMMessage:
    """Convert one SDK request part into a VLM message."""
    kind = part["part_kind"]
    content = part["content"]
    if kind == "system-prompt":
        message = VLMMessage("system")
    elif kind == "user-prompt":
        message = VLMMessage("user")
    elif kind == "tool-return":
        return VLMMessage(
            "tool", content=_tool_content(part), tool_call_id=part["tool_call_id"]
        )
    elif kind == "retry-prompt":
        retry = RetryPromptPart(content=content, tool_name=part.get("tool_name"))
        message = (
            VLMMessage("tool", tool_call_id=part["tool_call_id"])
            if retry.tool_name is not None
            else VLMMessage("user")
        )
        content = retry.model_response()
    else:
        raise ValueError(f"Unsupported API request part: {kind}")

    items = content if isinstance(content, list) else [content]
    for item in items:
        item_kind = item.get("kind") if isinstance(item, dict) else None
        if item_kind == "binary":
            message.content.append(_image(item))
        elif isinstance(item, str):
            message.content.append({"type": "text", "text": item})
        elif item_kind == "text-content" and kind == "user-prompt":
            message.content.append({"type": "text", "text": item["content"]})
        else:
            raise ValueError(f"Unsupported non-text content in {kind}")
    return message


def api_turn_to_sample(task: dict[str, Any], planner: dict[str, Any]) -> VLMSample:
    """Convert one completed API planner record and its history to a turn sample."""
    if planner["backend"] != "api":
        raise ValueError("Expected a completed API planner turn")
    request, response = planner.get("request"), planner.get("response")
    if planner["status"] != "completed" or not request or not response:
        raise TurnConversionError(
            "incomplete_response", "Expected a completed API planner turn"
        )
    if request["format"] != "pydantic_ai.model_request":
        raise ValueError("Expected a pydantic_ai.model_request")
    if response["format"] != "pydantic_ai.model_response":
        raise ValueError("Expected a pydantic_ai.model_response")
    if (
        request.get("completeness") != "complete"
        or response.get("completeness") != "complete"
    ):
        raise TurnConversionError(
            "incomplete_response", "Expected complete API request and response payloads"
        )
    reply = response.get("data") or {}
    if reply.get("state", "complete") != "complete" or reply.get("finish_reason") in {
        "length",
        "content_filter",
        "error",
    }:
        raise TurnConversionError(
            "incomplete_response", "API response did not finish normally"
        )

    data = request["data"]
    parameters = data["model_request_parameters"]
    if parameters.get("instruction_parts") or parameters.get("native_tools"):
        raise ValueError("Separate API instructions and native tools are unsupported")
    messages = []
    for entry in data["messages"]:
        if entry["kind"] == "response":
            message = _assistant_message(entry["parts"])
            if message.content or message.tool_calls:
                messages.append(message)
        elif entry["kind"] == "request" and not entry.get("instructions"):
            messages.extend(_request_message(part) for part in entry["parts"])
        else:
            raise ValueError("Unsupported API request history")

    target = _assistant_message(reply.get("parts") or [])
    if not (target.content or target.tool_calls):
        raise TurnConversionError(
            "incomplete_response", "API response has no text or tool calls to train on"
        )
    definitions = parameters["function_tools"] + parameters.get("output_tools", [])
    return VLMSample(
        episode_id=task["task_id"],
        turn_id=planner["planner_id"],
        messages=messages,
        tools=[
            VLMToolDefinition(
                name=tool["name"],
                description=tool["description"] or "",
                input_schema=tool["parameters_json_schema"],
            )
            for tool in definitions
        ],
        target=target,
        metadata={"backend": "api", "model": data.get("model")},
    )

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

"""Pydantic-AI-faithful Qwen conversation formatting, validation, and compaction.

Reproduces the exact wire format of ``QwenChatModel`` + ``WorkflowCapability`` +
``RequestAwareSlidingWindow`` without requiring pydantic-ai at runtime:

- Opening request counts as 1 logical message (``[system, opening_user]``).
- Text-only tool results serialize as a compact 1-element JSON array ``["{...}"]``.
- Image-bearing tool results emit ``role="tool"`` with ``See file <sha1[:6]>.``
  references plus a trailing ``role="user"`` message wrapped in
  ``<tool_response>\n...\n</tool_response>`` with ``<tool_result ...>`` tags
  so Qwen3.5's chat template keeps multi-step reasoning blocks intact.
- Raw PNG bytes live on ``LogicalMessage.images`` without eager Base64 encoding;
  lightweight ``rpent-image://<file_id>`` placeholders stay on ``wire`` for
  templating and prefix comparisons, and OpenAI JSON data URIs are materialized
  lazily once per message only when requested.
- Validation retries format as ``Validation feedback:\n<msg>\n\nFix the errors and try again.``.
- Sliding-window compaction trims by logical message count or token budget while
  preserving tool-call/return pairs and the opening ``[system, opening_user]``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from PIL import Image

from robots.rl.units import argument_hint, validate_relations

TOOL_RESPONSE_OPEN = "<tool_response>\n"
TOOL_RESPONSE_CLOSE = "\n</tool_response>"
IMAGE_PLACEHOLDER_PREFIX = "rpent-image://"
PATCH_SIZE = 16
MERGE_SIZE = 2
DEFAULT_IMAGE_TOKENS = 1024


def file_id_for(png_bytes: bytes) -> str:
    """Match pydantic-ai's 6-character SHA-1 identifier for binary tool files."""
    return hashlib.sha1(png_bytes).hexdigest()[:6]


def count_image_tokens(png_bytes: bytes) -> int:
    """Return Qwen3.5 vision tokens for one image (32x32 merged patch grid)."""
    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            width, height = img.size
    except Exception:
        return DEFAULT_IMAGE_TOKENS
    block = PATCH_SIZE * MERGE_SIZE
    return math.ceil(width / block) * math.ceil(height / block)


@dataclass(frozen=True)
class LogicalMessage:
    """One Pydantic-AI-equivalent logical message unit for compaction.

    ``wire`` stores lightweight template-ready dicts without embedded Base64
    image strings; raw PNG payloads are kept directly in ``images``.
    """

    kind: str  # "opening", "assistant", "tool", "retry"
    wire: tuple[dict[str, Any], ...]
    tool_call_ids: tuple[str, ...] = ()
    images: tuple[bytes, ...] = ()
    image_tokens: int = 0


def opening_message(
    system_prompt: str, user_message: str, initial_observation_text: str
) -> LogicalMessage:
    """Build the opening [system, user] logical message."""
    return LogicalMessage(
        kind="opening",
        wire=(
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{user_message}\n\n{initial_observation_text}",
            },
        ),
    )


def assistant_message(
    *,
    content: str | None,
    reasoning_content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> LogicalMessage:
    """Build an assistant logical message matching QwenChatModel wire layout."""
    msg: dict[str, Any] = {"role": "assistant", "content": content or None}
    if reasoning_content is not None:
        msg["reasoning_content"] = reasoning_content
    call_ids: list[str] = []
    if tool_calls:
        msg["tool_calls"] = tool_calls
        call_ids = [call["id"] for call in tool_calls]
    return LogicalMessage(kind="assistant", wire=(msg,), tool_call_ids=tuple(call_ids))


def tool_return_message(
    *,
    tool_name: str,
    tool_call_id: str,
    text: str,
    images: Sequence[bytes] = (),
) -> LogicalMessage:
    """Build a tool return logical message (with optional framed Qwen image turn)."""
    if not images:
        payload = json.dumps([text], separators=(",", ":"), ensure_ascii=False)
        return LogicalMessage(
            kind="tool",
            wire=({"role": "tool", "tool_call_id": tool_call_id, "content": payload},),
            tool_call_ids=(tool_call_id,),
        )

    raw_images = tuple(bytes(img) for img in images)
    fids = [file_id_for(img) for img in raw_images]
    tool_payload = json.dumps(
        [text, *(f"See file {fid}." for fid in fids)],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    user_parts: list[dict[str, Any]] = [{"type": "text", "text": TOOL_RESPONSE_OPEN}]
    total_img_tokens = 0
    for img, fid in zip(raw_images, fids, strict=True):
        total_img_tokens += count_image_tokens(img)
        user_parts.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f'<tool_result tool_name="{tool_name}" '
                        f'tool_call_id="{tool_call_id}" file_id="{fid}">'
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"{IMAGE_PLACEHOLDER_PREFIX}{fid}"},
                },
                {"type": "text", "text": "</tool_result>"},
            ]
        )
    user_parts.append({"type": "text", "text": TOOL_RESPONSE_CLOSE})
    return LogicalMessage(
        kind="tool",
        wire=(
            {"role": "tool", "tool_call_id": tool_call_id, "content": tool_payload},
            {"role": "user", "content": user_parts},
        ),
        tool_call_ids=(tool_call_id,),
        images=raw_images,
        image_tokens=total_img_tokens,
    )


def retry_message(feedback: str) -> LogicalMessage:
    """Build Pydantic AI's RetryPromptPart user feedback message."""
    return LogicalMessage(
        kind="retry",
        wire=(
            {
                "role": "user",
                "content": f"Validation feedback:\n{feedback}\n\nFix the errors and try again.",
            },
        ),
    )


def _materialize_openai_wire(msg: LogicalMessage) -> list[dict[str, Any]]:
    """Expand ``rpent-image://`` placeholders to Base64 data URIs once for JSON APIs."""
    if not msg.images:
        return list(msg.wire)
    img_iter = iter(msg.images)
    out: list[dict[str, Any]] = []
    for item in msg.wire:
        content = item.get("content")
        if not isinstance(content, list):
            out.append(item)
            continue
        new_parts: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image_url":
                raw = next(img_iter)
                b64 = base64.b64encode(raw).decode("ascii")
                new_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
            else:
                new_parts.append(part)
        out.append({**item, "content": new_parts})
    return out


def validate_tool_calls(
    calls: Sequence[dict[str, Any]],
    tool_schemas: Mapping[str, dict[str, Any]],
    environment: str,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """Validate one turn's parsed tool calls exactly like WorkflowCapability.

    Returns ``(error_kind, error_message, parsed_arguments)``.
    When ``error_kind is None`` and ``calls`` is non-empty, ``parsed_arguments``
    holds the validated argument mapping for ``calls[0]``.
    """
    if not calls:
        return None, None, None
    if len(calls) > 1:
        return (
            "tool_arguments",
            "Use one tool call per response. No tool executed.",
            None,
        )
    call = calls[0]
    name = call["name"]
    if name not in tool_schemas:
        return "tool_name", f"Unknown tool name: {name}. No tool executed.", None
    raw_args = call["arguments"]
    try:
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        errors = list(Draft202012Validator(tool_schemas[name]).iter_errors(arguments))
        if errors:
            error = errors[0]
            location = ".".join(str(item) for item in error.absolute_path)
            raise ValueError(f"{location or name}: {error.message}")
        validate_relations(name, arguments)
    except (AssertionError, ValueError, TypeError) as exc:
        hint = argument_hint(name, environment)
        return (
            "tool_arguments",
            f"Invalid tool arguments: {exc}. {hint} No tool executed.",
            None,
        )
    return None, None, arguments


@dataclass(frozen=True)
class CompactionConfig:
    """Sliding-window compaction settings matching RequestAwareSlidingWindow."""

    max_messages: int | None = 80
    keep_messages: int = 10
    context_window: int | None = 262144
    max_fraction: float | None = 0.7
    max_tokens: int | None = None
    preserve_first_user_message: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> CompactionConfig | None:
        if data is None:
            return None
        return cls(
            max_messages=data.get("max_messages", 80),
            keep_messages=int(data.get("keep_messages", 10)),
            context_window=data.get("context_window", 262144),
            max_fraction=data.get("max_fraction", 0.7),
            max_tokens=data.get("max_tokens"),
            preserve_first_user_message=bool(
                data.get("preserve_first_user_message", True)
            ),
        )

    @property
    def token_trigger(self) -> int | None:
        if self.max_tokens is not None:
            return int(self.max_tokens)
        if self.max_fraction is not None and self.context_window is not None:
            return int(self.context_window * self.max_fraction)
        return None


def find_safe_cutoff(messages: Sequence[LogicalMessage], keep_messages: int) -> int:
    """Return earliest index >= 0 keeping at least ``keep_messages`` without splitting tool pairs."""
    if keep_messages <= 0 or len(messages) <= keep_messages:
        return 0
    cutoff = len(messages) - keep_messages
    call_index: dict[str, int] = {}
    for idx, msg in enumerate(messages):
        if msg.kind == "assistant":
            for cid in msg.tool_call_ids:
                call_index[cid] = idx
    changed = True
    while changed and cutoff > 0:
        changed = False
        for msg in messages[cutoff:]:
            if msg.kind == "tool":
                for cid in msg.tool_call_ids:
                    owner = call_index.get(cid)
                    if owner is not None and owner < cutoff:
                        cutoff = owner
                        changed = True
    return cutoff


def estimate_logical_tokens(
    messages: Sequence[LogicalMessage],
    tools: Sequence[dict[str, Any]] | None = None,
    count_text: Callable[[str], int] | None = None,
) -> int:
    """Fast token estimate for standalone rollout when exact processor counting is absent."""
    text_chunks: list[str] = []
    img_tokens = 0
    for msg in messages:
        img_tokens += msg.image_tokens
        for wire in msg.wire:
            content = wire.get("content")
            if isinstance(content, str):
                text_chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        text_chunks.append(part["text"])
            if wire.get("reasoning_content"):
                text_chunks.append(wire["reasoning_content"])
            for call in wire.get("tool_calls", ()):
                text_chunks.append(json.dumps(call, ensure_ascii=False))
    if tools:
        text_chunks.append(json.dumps(tools, ensure_ascii=False))
    joined = "\n".join(text_chunks)
    text_tokens = count_text(joined) if count_text is not None else len(joined) // 4
    return text_tokens + img_tokens


@dataclass
class EpisodeConversation:
    """Stateful conversation manager shared by RPent standalone rollout and RLinf."""

    system_prompt: str
    user_message: str
    initial_observation_text: str
    tools: list[dict[str, Any]]
    environment: str
    compaction: CompactionConfig | None = None
    max_output_retries: int = 1
    messages: list[LogicalMessage] = field(default_factory=list)
    output_retries: int = 0

    def __post_init__(self) -> None:
        self._tool_schemas = {
            t["function"]["name"]: t["function"]["parameters"]
            for t in self.tools
            if t.get("type") == "function"
        }
        self._openai_wire_cache: dict[int, list[dict[str, Any]]] = {}
        if not self.messages:
            self.messages = [
                opening_message(
                    self.system_prompt,
                    self.user_message,
                    self.initial_observation_text,
                )
            ]

    def images(self) -> list[bytes]:
        """Return all raw binary PNG images across active messages in order."""
        return [img for msg in self.messages for img in msg.images]

    def wire_messages(self, *, inline_base64: bool = False) -> list[dict[str, Any]]:
        """Flatten logical messages into template dicts (or Base64-inlined OpenAI dicts)."""
        if not inline_base64:
            return [item for msg in self.messages for item in msg.wire]
        out: list[dict[str, Any]] = []
        for msg in self.messages:
            if not msg.images:
                out.extend(msg.wire)
                continue
            key = id(msg)
            cached = self._openai_wire_cache.get(key)
            if cached is None:
                cached = _materialize_openai_wire(msg)
                self._openai_wire_cache[key] = cached
            out.extend(cached)
        return out

    def maybe_compact(
        self,
        *,
        current_tokens: int | None = None,
        count_text: Callable[[str], int] | None = None,
    ) -> bool:
        """Apply sliding-window compaction if message count or token budget is exceeded."""
        cfg = self.compaction
        if cfg is None:
            return False
        over_messages = (
            cfg.max_messages is not None and len(self.messages) > cfg.max_messages
        )
        limit = cfg.token_trigger
        if not over_messages and limit is not None:
            tokens = (
                current_tokens
                if current_tokens is not None
                else estimate_logical_tokens(self.messages, self.tools, count_text)
            )
            over_tokens = tokens > limit
        else:
            over_tokens = False
        if not (over_messages or over_tokens):
            return False

        cutoff = find_safe_cutoff(self.messages, cfg.keep_messages)
        if cutoff <= 0:
            return False
        trimmed = list(self.messages[cutoff:])
        if cfg.preserve_first_user_message:
            opening = self.messages[0]
            if not trimmed or trimmed[0] is not opening:
                trimmed = [opening, *trimmed]
        self.messages = trimmed
        active_ids = {id(m) for m in trimmed}
        self._openai_wire_cache = {
            k: v for k, v in self._openai_wire_cache.items() if k in active_ids
        }
        return True

    def append_assistant(
        self,
        *,
        content: str | None,
        reasoning_content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.messages.append(
            assistant_message(
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
            )
        )

    def append_tool_result(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        text: str,
        images: Sequence[bytes] = (),
    ) -> None:
        self.output_retries = 0
        self.messages.append(
            tool_return_message(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                text=text,
                images=images,
            )
        )

    def validate_or_retry(
        self, calls: Sequence[dict[str, Any]]
    ) -> tuple[str | None, str | None, dict[str, Any] | None, bool]:
        """Validate calls; append retry feedback if within retry budget.

        Returns ``(error_kind, error_msg, arguments, should_retry)``.
        """
        err_kind, err_msg, args = validate_tool_calls(
            calls, self._tool_schemas, self.environment
        )
        if err_kind is None:
            self.output_retries = 0
            return None, None, args, False
        self.output_retries += 1
        if self.output_retries <= self.max_output_retries:
            self.messages.append(retry_message(err_msg or ""))
            return err_kind, err_msg, None, True
        return err_kind, err_msg, None, False

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

"""Native file and image tools shared by robot toolkits."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from rpent.tools.base import ToolContext, ToolResult, readonly, tool


@tool
@readonly
def read_text_file(
    path: str, max_chars: int = 40000, *, ctx: ToolContext
) -> ToolResult:
    """Read a UTF-8 text file. Use for past recipe JSONLs, audit JSONs, and memory files. Large files are truncated. Published memory is read-only. During exploration, you may write only to your current memory inbox. Memory for other robots is unavailable.

    Args:
        path: Absolute or repo-relative path
        max_chars: Max chars (default 40000)
    """
    resolved = ctx.memory.authorize_read(path)
    text = resolved.read_text(encoding="utf-8", errors="replace")
    content = text
    if len(text) > max_chars:
        content = (
            text[:max_chars]
            + f"\n\n[TRUNCATED — file is {len(text)} chars, showed first {max_chars}]"
        )
    return ToolResult(
        data={"path": str(resolved), "size": len(text), "content": content}
    )


@tool
def write_text_file(path: str, content: str, *, ctx: ToolContext) -> ToolResult:
    """Write a UTF-8 text file (creates parent dirs). Use this to save the working recipe JSONL and the final audit JSON at the end of a successful run. Published memory is read-only. During exploration, you may write only to your current memory inbox. Memory for other robots is unavailable."""
    resolved = ctx.memory.authorize_write(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return ToolResult(
        data={"path": str(resolved), "bytes_written": len(content.encode("utf-8"))}
    )


@tool
@readonly
def read_image(
    name: str,
    step: Annotated[int, Field(ge=-1)] = -1,
    *,
    ctx: ToolContext,
) -> ToolResult:
    """Read a step-scoped image artifact as visual input.

    Artifact failures are returned as structured tool errors so a bad
    model-supplied name or step does not abort the agent run.
    """
    try:
        record = ctx.state.get(step)
        path = ctx.state.artifact_path(name, step=record.step_idx)
    except Exception as e:
        return ToolResult(error=str(e))
    path = ctx.memory.authorize_read(path)
    data = {"artifact": name, "step": record.step_idx}
    if name not in record.artifacts or not path.is_file():
        return ToolResult(
            data={**data},
            error=f"Image artifact {name!r} is not available at step {step}.",
        )
    if path.suffix.lower() != ".png":
        return ToolResult(data={**data}, error=f"Artifact {name!r} is not a PNG image.")
    return ToolResult(data=data, images=[path.read_bytes()])


@tool
@readonly
def list_dir(path: str = "", *, ctx: ToolContext) -> ToolResult:
    """List files in a directory (non-recursive). Use to inspect the working directory or discover existing resource files. Published memory is read-only. During exploration, you may write only to your current memory inbox. Memory for other robots is unavailable.

    Args:
        path: Directory path. Defaults to the current task's output directory.
    """
    resolved = ctx.memory.authorize_read(path or ctx.output_dir)
    files = sorted(item.name for item in resolved.iterdir())
    return ToolResult(data={"path": str(resolved), "count": len(files), "files": files})


COMMON_TOOLS = (read_text_file, write_text_file, list_dir, read_image)

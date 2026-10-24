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

"""Shared artifacts and child environment for external agents."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, TextIO

from rpent.planner.base import PlannerResult
from rpent.utils.logging import get_logger

logger = get_logger("agent_runtime")


def agent_environment() -> dict[str, str]:
    """Inherit the environment while bypassing proxies for local MCP traffic."""
    env = dict(os.environ)
    entries = [
        item.strip()
        for key in ("NO_PROXY", "no_proxy")
        for item in env.get(key, "").split(",")
        if item.strip()
    ]
    bypass = ",".join(dict.fromkeys([*entries, "127.0.0.1", "localhost"]))
    env["NO_PROXY"] = env["no_proxy"] = bypass
    return env


class AgentArtifacts:
    """Own transcript paths, elapsed time, and common planner result fields."""

    def __init__(
        self,
        backend: str,
        output_path: Path | None,
        *,
        directory: Path | None = None,
    ) -> None:
        """Allocate an output path, preserving each backend's temporary directory.

        Args:
            backend: Backend identifier used in results and temporary filenames.
            output_path: Explicit transcript path, or None for a temporary file.
            directory: Temporary-file directory; None uses the system default.
        """
        self.backend = backend
        if output_path is None:
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                suffix=".out", prefix=f"{backend}_task_", dir=directory, delete=False
            ) as file_obj:
                output_path = Path(file_obj.name)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.raw_stream_path = self.sidecar(".stream.jsonl")
        self.started = time.monotonic()

    def sidecar(self, suffix: str) -> Path:
        """Return a companion artifact path without replacing the output suffix."""
        return self.output_path.with_suffix(self.output_path.suffix + suffix)

    def result(
        self,
        *,
        text: str,
        finish_result: dict[str, Any] | None,
        final_response: str | None,
        stats: dict[str, Any],
        error: str | None,
        **extra_paths: Path,
    ) -> PlannerResult:
        """Build a result with backend statistics and companion artifact paths.

        Args:
            text: Rendered transcript; falls back to the file when empty.
            finish_result: Successful finish tool result, when present.
            final_response: Last assistant response used for size reporting.
            stats: Backend-specific usage and turn counters.
            error: Execution or backend error, when present.
            **extra_paths: Backend-specific artifact names and paths.

        Returns:
            The common planner result without changing backend event semantics.
        """
        if not text and self.output_path.exists():
            text = self.output_path.read_text(errors="replace")
        logger.info("%s output: %s", self.backend, self.output_path)
        logger.info("%s raw stream: %s", self.backend, self.raw_stream_path)
        return PlannerResult(
            finish_result=finish_result,
            messages=[{"role": self.backend, "content": text}],
            stats={
                "backend": self.backend,
                "elapsed_s": round(time.monotonic() - self.started, 1),
                "output_chars": len(text),
                "output_path": str(self.output_path),
                "raw_stream_path": str(self.raw_stream_path),
                "last_message_chars": len(final_response or ""),
                **{name: str(path) for name, path in extra_paths.items()},
                **stats,
            },
            error=error,
        )


def write_jsonl(stream: TextIO, value: Any) -> None:
    """Write and flush one JSON event."""
    stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    stream.flush()


def short_json(value: Any, *, limit: int = 1000) -> str:
    """Render a bounded JSON preview for a transcript."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + f"...(+{len(text) - limit})"

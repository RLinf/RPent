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

"""Native tool execution boundary shared by dsh runtime integration layers."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from rpent.planner.utils.http_mcp_server import mcp_result
from rpent.tools import ToolResult
from rpent.tools.toolkit import Toolkit


class _ToolBoundary:
    """Serialize MCP calls and latch native completion before returning over HTTP."""

    def __init__(self, toolkit: Toolkit, ledger_path: Path | None = None):
        self.toolkit = toolkit
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.finish_result: dict | None = None
        self.completed_calls = 0
        self.execution_uncertain = False
        self.ledger_path = ledger_path

    def list_tools(self):
        return self.toolkit.list_tools()

    def execute_tool(self, name: str, arguments: dict) -> ToolResult:
        with self.lock:
            if self.stopped.is_set() or self.finish_result is not None:
                return ToolResult(
                    data={"error": "RPent run has stopped; tool refused."}
                )
            try:
                result = self.toolkit.execute_tool(name, arguments)
            except BaseException:
                self.execution_uncertain = True
                self.stopped.set()
                raise
            self.execution_uncertain |= result.execution_uncertain
            if self.execution_uncertain:
                self.stopped.set()
            self.completed_calls += 1
            if (
                not result.is_error
                and not result.execution_uncertain
                and result.data.get("_finish") is True
            ):
                self.finish_result = dict(result.data)
            if self.ledger_path is not None:
                business_error_hash = None
                if (
                    result.is_error
                    and not result.execution_uncertain
                    and not result.images
                ):
                    try:
                        text = mcp_result(result, include_structured=True)["content"][
                            0
                        ]["text"]
                        # MCP errors become text-only runtime exceptions; the
                        # structured copy cannot rescue a truncated text block.
                        json.loads(text)
                    except Exception:
                        # A native refusal whose payload cannot be serialized
                        # is not a correctly delivered business error. The MCP
                        # failure must close admission just like lost output.
                        pass
                    else:
                        business_error_hash = hashlib.sha256(text.encode()).hexdigest()
                temporary = self.ledger_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "seq": self.completed_calls,
                            "name": name,
                            "isError": result.is_error,
                            "executionUncertain": result.execution_uncertain,
                            "canonicalBusinessErrorSha256": business_error_hash,
                            "finishResult": self.finish_result,
                        }
                    )
                )
                temporary.replace(self.ledger_path)
            return result

    def cancel_active_and_wait(self) -> None:
        self.stopped.set()
        self.toolkit.cancel_active_and_wait()
        # Toolkit cancellation may return before the MCP wrapper records its
        # result. Hold this lock once to establish the final action certainty.
        with self.lock:
            pass

    def resume_calls(self) -> None:
        """Reopen admission only after the driver establishes runtime quiescence."""
        with self.lock:
            if self.execution_uncertain or self.finish_result is not None:
                raise RuntimeError("Cannot resume a terminated or uncertain dsh run")
            self.toolkit.resume_calls()
            self.stopped.clear()

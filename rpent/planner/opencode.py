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

"""OpenCode configuration for the shared ACP driver."""

from __future__ import annotations

import json
import os
import shutil

from rpent.planner.utils.acp import AcpPlanner, AcpSession
from rpent.planner.utils.agent_runtime import agent_environment


class OpenCodePlanner(AcpPlanner):
    """Run OpenCode with RPent tools and its existing provider configuration."""

    backend = "opencode"
    label = "OpenCode"

    def _create_session(
        self, root, control, emit, stderr_path, *, mcp_url, max_turns, tool_names
    ) -> AcpSession:
        binary = os.environ.get("OPENCODE_BIN") or shutil.which("opencode")
        if not binary:
            raise FileNotFoundError(
                "OpenCode is not installed; install it or set OPENCODE_BIN"
            )
        env = agent_environment()
        config = json.loads(env.get("OPENCODE_CONFIG_CONTENT") or "{}")
        config.update(
            default_agent=root.name,
            agent={
                root.name: {
                    "mode": "primary",
                    "steps": max(1, max_turns),
                    "permission": {
                        "*": "deny",
                        **{f"rpent_{name}": "allow" for name in tool_names},
                    },
                }
            },
            autoupdate=False,
            share="disabled",
        )
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
        return AcpSession(
            command=[binary, "acp", "--pure"],
            cwd=root,
            env=env,
            model=self._model,
            mcp_url=mcp_url,
            control=control,
            emit_event=emit,
            stderr_path=stderr_path,
            redact=self._redact,
        )

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
from collections.abc import Callable

from rpent.planner.utils.acp import AcpPlanner, AcpSession
from rpent.planner.utils.agent_runtime import agent_environment


class OpenCodePlanner(AcpPlanner):
    """Run OpenCode with RPent tools and its existing provider configuration."""

    backend = "opencode"
    label = "OpenCode"

    @staticmethod
    def _load_secret_redactor() -> Callable[[str], str]:
        try:
            config = json.loads(os.environ.get("OPENCODE_CONFIG_CONTENT") or "{}")
        except json.JSONDecodeError:
            config = {}
        secrets = {}
        providers = config.get("provider") if isinstance(config, dict) else None
        if isinstance(providers, dict):
            for name, provider in providers.items():
                options = (
                    provider.get("options") if isinstance(provider, dict) else None
                )
                key = options.get("apiKey") if isinstance(options, dict) else None
                if isinstance(key, str):
                    secrets[f"provider.{name}.options.apiKey"] = key
        return AcpPlanner._load_secret_redactor(secrets)

    def _create_session(
        self,
        root,
        control,
        emit,
        stderr_path,
        *,
        mcp_url,
        tool_names,
        redact,
    ) -> AcpSession:
        binary = os.environ.get("OPENCODE_BIN") or shutil.which("opencode")
        if not binary:
            raise FileNotFoundError(
                "OpenCode is not installed; install it or set OPENCODE_BIN"
            )
        env = agent_environment()
        env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
        env["OPENCODE_DISABLE_CLAUDE_CODE"] = "1"
        env["OPENCODE_DISABLE_EXTERNAL_SKILLS"] = "1"
        config = json.loads(env.get("OPENCODE_CONFIG_CONTENT") or "{}")
        config.update(
            default_agent=root.name,
            agent={
                root.name: {
                    "mode": "primary",
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
            reasoning_effort=self._reasoning_effort,
            mcp_url=mcp_url,
            control=control,
            emit_event=emit,
            stderr_path=stderr_path,
            redact=redact,
        )

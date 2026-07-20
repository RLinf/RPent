"""LeRobot SO101 environment extension.

Entry point for the env registry: :func:`get_env_spec` and
:func:`get_toolkit` are discovered by
:func:`rpent.envs.base._resolve_env` via
``importlib.import_module("robots.lerobot")`` — dropping this
package on disk is the entire registration step.
"""
from __future__ import annotations

import sys
from typing import Any

from rpent.deployment.launcher import EnvDriverContext
from rpent.envs.env_spec import EnvSpec
from rpent.envs.prompt_bundle import PromptBundle
from robots.lerobot.prompt import (
    system_prompt,
    user_prompt,
)
from rpent.utils.rpc import RpcClient
from rpent.utils.config import get_repo_root


def get_env_spec() -> EnvSpec:
    """Return the SO101 env identity + prompt bundle.

    Tool schemas, handlers, and the MCP allowlist live on the toolkit (see
    :func:`get_toolkit`); driver launch config is attached here.
    """
    return EnvSpec(
        name="lerobot",
        prompts=PromptBundle(
            system=system_prompt,
            user=user_prompt,
        ),
        build_command=_build_driver_command,
        requires_suite_task=False,
    )


def _build_driver_command(context: EnvDriverContext) -> list[str]:
    return [
        sys.executable,
        str(get_repo_root() / "robots" / "lerobot" / "env_server.py"),
        "--output-dir",
        str(context.output_dir),
        "--max-episode-steps",
        str(context.max_episode_steps),
    ]


def get_toolkit(
    *,
    rpc_client: RpcClient,
    driver_context: EnvDriverContext,
    video_path: str | None = None,
    dashboard: Any = None,
):
    """Return the SO101 toolkit (common tools + SO101 primitives).

    ``driver_context`` is accepted for a uniform env extension API; SO101 does
    not currently need any extra context beyond the RPC client.
    """
    del driver_context
    from robots.lerobot.env_client import LerobotEnvClient
    from robots.lerobot.toolkit import LerobotToolkit

    return LerobotToolkit(
        env=LerobotEnvClient(rpc_client),
        video_path=video_path,
        dashboard=dashboard,
    )

"""Franka environment extension."""
from __future__ import annotations

from typing import Any

from rpent.envs.env_spec import EnvSpec
from rpent.envs.prompt_bundle import PromptBundle
from robots.franka.prompt import system_prompt, user_prompt
from rpent.utils.rpc import RpcClient


def get_env_spec() -> EnvSpec:
    """Return the Franka env identity + prompt bundle."""
    return EnvSpec(
        name="franka",
        prompts=PromptBundle(
            system=system_prompt,
            user=user_prompt,
        ),
    )


def get_toolkit(
    *,
    rpc_client: RpcClient,
    video_path: str | None = None,
    dashboard: Any = None,
):
    """Return the Franka toolkit (common tools + Cartesian primitives)."""
    from robots.franka.env_client import FrankaEnvClient
    from robots.franka.toolkit import FrankaToolkit

    return FrankaToolkit(
        env=FrankaEnvClient(rpc_client),
        video_path=video_path,
        dashboard=dashboard,
    )

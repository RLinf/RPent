"""Franka environment extension."""
from __future__ import annotations

from typing import Any

from rpent.deployment.launcher import EnvDriverContext
from rpent.envs.env_spec import EnvSpec
from rpent.envs.prompt_bundle import PromptBundle
from robots.franka.prompt import system_prompt, user_prompt
from rpent.rpc_driver.base import RpcClient
from rpent.utils.config import get_repo_root


def get_env_spec() -> EnvSpec:
    """Return the Franka env identity + prompt bundle."""
    return EnvSpec(
        name="franka",
        prompts=PromptBundle(
            system=system_prompt,
            user=user_prompt,
        ),
        build_command=_build_driver_command,
        requires_suite_task=False,
    )


def _build_driver_command(context: EnvDriverContext) -> list[str]:
    return [
        "bash",
        str(get_repo_root() / "deployment" / "franka" / "run_env_server.sh"),
        "--output-dir",
        str(context.output_dir),
    ]


def get_toolkit(
    *,
    rpc_client: RpcClient,
    driver_context: EnvDriverContext,
    video_path: str | None = None,
    dashboard: Any = None,
):
    """Return the Franka toolkit (common tools + Cartesian primitives)."""
    del driver_context
    from robots.franka.env_client import FrankaEnvClient
    from robots.franka.toolkit import FrankaToolkit

    return FrankaToolkit(
        env=FrankaEnvClient(rpc_client),
        video_path=video_path,
        dashboard=dashboard,
    )

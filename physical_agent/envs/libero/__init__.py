"""LIBERO environment extension."""
from __future__ import annotations

from typing import Any

from physical_agent.envs.prompt_bundle import PromptBundle
from physical_agent.envs.env_spec import EnvSpec
from physical_agent.envs.libero.prompt_bundle import (
    api_system,
    api_user,
    cli_system,
    cli_user,
)


def get_env_spec() -> EnvSpec:
    """Return the LIBERO env identity + prompt bundle.

    Tool schemas, handlers, driver lifecycle, and the MCP allowlist live on
    the LIBERO toolkit (see :func:`get_toolkit`).
    """
    return EnvSpec(
        name="libero",
        prompts=PromptBundle(
            api_system=api_system,
            api_user=api_user,
            cli_system=cli_system,
            cli_user=cli_user,
        ),
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    video_path: str | None = None,
):
    """Return the LIBERO toolkit (common tools + LIBERO primitives)."""
    from physical_agent.envs.libero.toolkit import LiberoToolkit

    return LiberoToolkit(
        primitives_kwargs=primitives_kwargs,
        video_path=video_path,
    )

"""LIBERO environment extension."""
from __future__ import annotations

from typing import Any

from robots.libero.prompt_bundle import (
    system_prompt,
    user_prompt,
)
from rpent.envs.env_spec import EnvSpec
from rpent.envs.prompt_bundle import PromptBundle


def get_env_spec() -> EnvSpec:
    """Return the LIBERO env identity + prompt bundle.

    Tool schemas, handlers, server lifecycle, and the MCP allowlist live on
    the LIBERO toolkit (see :func:`get_toolkit`).
    """
    return EnvSpec(
        name="libero",
        prompts=PromptBundle(
            system=system_prompt,
            user=user_prompt,
        ),
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    sam3_client: Any,
    video_path: str | None = None,
    dashboard: Any = None,
):
    """Return the LIBERO toolkit (common tools + LIBERO primitives)."""
    from robots.libero.toolkit import LiberoToolkit

    return LiberoToolkit(
        primitives_kwargs=primitives_kwargs,
        sam3_client=sam3_client,
        video_path=video_path,
        dashboard=dashboard,
    )

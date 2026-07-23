"""LeRobot SO101 environment extension.

Entry point for the env registry: :func:`get_env_spec` and
:func:`get_toolkit` are discovered by
:func:`rpent.envs.base._resolve_env` via
``importlib.import_module("robots.lerobot")`` — dropping this
package on disk is the entire registration step.
"""
from __future__ import annotations

from typing import Any

from robots.lerobot.prompt import (
    system_prompt,
    user_prompt,
)
from rpent.envs.env_spec import EnvSpec
from rpent.envs.prompt_bundle import PromptBundle


def get_env_spec() -> EnvSpec:
    """Return the SO101 env identity + prompt bundle.

    Tool schemas, handlers, and the MCP allowlist live on the toolkit (see
    :func:`get_toolkit`).
    """
    return EnvSpec(
        name="lerobot",
        prompts=PromptBundle(
            system=system_prompt,
            user=user_prompt,
        ),
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    video_path: str | None = None,
    dashboard: Any = None,
):
    """Return the SO101 toolkit (common tools + SO101 primitives).

    ``primitives_kwargs`` is assembled by ``_init_lerobot`` in
    ``rpent/cli/main.py`` and carries the env RPC stub
    (``{"env": LerobotEnvClient(...)}``, optionally plus ``"model"``).
    """
    from robots.lerobot.toolkit import LerobotToolkit

    return LerobotToolkit(
        video_path=video_path,
        dashboard=dashboard,
        **primitives_kwargs,
    )

"""Franka environment extension."""
from __future__ import annotations

from typing import Any

from robots.franka.prompt import system_prompt, user_prompt
from rpent.envs.env_spec import EnvSpec
from rpent.envs.prompt_bundle import PromptBundle


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
    primitives_kwargs: dict[str, Any],
    video_path: str | None = None,
    dashboard: Any = None,
):
    """Return the Franka toolkit (common tools + Cartesian primitives).

    ``primitives_kwargs`` is assembled by ``_init_franka`` in
    ``rpent/cli/main.py`` and carries the env RPC stub
    (``{"env": FrankaEnvClient(...)}``).
    """
    from robots.franka.toolkit import FrankaToolkit

    return FrankaToolkit(
        video_path=video_path,
        dashboard=dashboard,
        **primitives_kwargs,
    )

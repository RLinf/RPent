"""reBot DevArm RobStride environment extension."""

from __future__ import annotations

from typing import Any

from robots.rebot_robstride.prompt_bundle import system_prompt, user_prompt
from rpent.envs.env_spec import EnvSpec
from rpent.envs.prompt_bundle import PromptBundle


def get_env_spec() -> EnvSpec:
    """Return the physical reBot environment identity and prompts."""
    return EnvSpec(
        name="rebot_robstride",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
    )


def get_runtime(*, args: Any, output_dir: str, dashboard: Any = None):
    """Return the reBot RobStride process lifecycle adapter."""
    from robots.rebot_robstride.runtime import RebotRobstrideRuntime

    return RebotRobstrideRuntime(
        args=args,
        output_dir=output_dir,
        dashboard=dashboard,
    )


def get_toolkit(*, primitives_kwargs: dict[str, Any], dashboard: Any = None, **_):
    """Build the reBot toolkit for callers that manage transport themselves."""
    from robots.rebot_robstride.toolkit import RebotRobstrideToolkit

    return RebotRobstrideToolkit(
        env=primitives_kwargs["env"],
        dashboard=dashboard,
    )

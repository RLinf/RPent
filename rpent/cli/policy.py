# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Policy setup shared by the CLI and Dashboard."""

from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from rpent.code_policy.sandbox import bubblewrap
from rpent.code_policy.toolkit import CodePolicyTools
from rpent.prompt.code_policy import code_policy_api_reference
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.tools import Toolkit


def validate_policy_options(args: Namespace, robot: RobotSpec) -> None:
    """Check policy capabilities before starting robot services."""
    if args.policy_mode == "code":
        bubblewrap()
    if not args.no_am:
        return
    if args.policy_mode != "code":
        raise ValueError("--no-am requires --policy-mode code")
    if not robot.supports_no_am:
        raise ValueError(f"--no-am is not supported for robot {robot.name!r}")


def configure_policy(config: RunConfig, args: Namespace) -> RunConfig:
    """Select the policy prompts and memory configuration."""
    if getattr(args, "policy_mode", "tools") != "code":
        return config
    prompt_vars = {**config.prompt_vars, "policy_mode": "code"}
    for name in ("memory_dir", "memory_inbox"):
        prompt_vars.pop(name, None)
    return replace(config, prompt_vars=prompt_vars, use_memory=False)


@contextmanager
def planner_policy(
    toolkit: Toolkit,
    args: Namespace,
    output_dir: Path,
    prompt_vars: dict[str, Any],
) -> Iterator[tuple[Toolkit, dict[str, Any]]]:
    """Lend a robot toolkit to a planner, wrapping it in Python mode if selected."""
    if getattr(args, "policy_mode", "tools") != "code":
        yield toolkit, prompt_vars
        return

    policy = CodePolicyTools(
        toolkit,
        output_dir=output_dir,
        timeout_s=args.code_timeout_s,
        max_calls=args.code_max_calls,
        include_images=not args.no_images,
    )
    try:
        yield (
            policy,
            {
                **prompt_vars,
                "code_policy_api": code_policy_api_reference(policy.robot_tools()),
            },
        )
    finally:
        policy.close()

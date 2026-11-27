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

"""VLABench robot spec: registry entry and dashboard descriptor."""

from __future__ import annotations

import argparse

from robots.vlabench.adapter import init_runtime
from robots.vlabench.config import (
    add_cli_args,
    finalize_run,
    parse_config,
    prepare_memory,
    validate_args,
)
from robots.vlabench.prompts.domain import system_prompt, user_prompt
from rpent.dashboard.spec import DashboardSpec
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec

PRIMITIVES = ("move_pose", "move_delta", "set_gripper")


def resolve_dashboard(args: argparse.Namespace) -> DashboardSpec:
    """Describe the Dashboard task form and runtime components."""
    del args
    return {
        "task": {
            "command": "/rpent-task",
            "usage": "/rpent-task <task_name> <episode_index> <seed>",
            "fields": (
                {"name": "task_name"},
                {"name": "episode_index", "kind": "integer", "minimum": 0},
                {"name": "seed", "kind": "integer", "minimum": 0},
            ),
            "display": "{task_name} / episode {episode_index} / seed {seed}",
            "output_slug": "{task_name}_e{episode_index}_s{seed}",
        },
        "runtime_components": ({"name": "env", "label": "ENV", "scope": "unique"},),
        "primitives": PRIMITIVES,
    }


def get_robot_spec() -> RobotSpec:
    """Return the VLABench robot spec."""
    return RobotSpec(
        name="vlabench",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=add_cli_args,
        parse_config=parse_config,
        init_runtime=init_runtime,
        resolve_dashboard=resolve_dashboard,
        validate_args=validate_args,
        prepare_memory=prepare_memory,
        finalize_run=finalize_run,
    )

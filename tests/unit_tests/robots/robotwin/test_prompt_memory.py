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

"""Tests for the curated RoboTwin prompt."""

from __future__ import annotations

import pytest

from robots.robotwin.prompt_bundle import system_prompt, user_prompt
from rpent.prompt.utils import format_prompt


def test_prompts_render_with_memory_relative_paths():
    variables = {
        "task_name": "beat_block_hammer",
        "task_config": "demo_randomized",
        "seed": 100000,
        "memory_dir": "memory/robotwin",
        "reference_tag": "robotwin_beat_block_hammer_demo_randomized_s0",
    }

    system = format_prompt(system_prompt(), variables=variables)
    user = format_prompt(user_prompt(), variables=variables)

    assert "robots/robotwin/guides/GUIDE_RPENT.md" in system
    assert (
        "memory/robotwin/task-specific/robotwin_beat_block_hammer_demo_randomized_s0.json"
        in system
    )
    assert (
        "memory/robotwin/task-specific/robotwin_beat_block_hammer_demo_randomized_s0_recipe.jsonl"
        in system
    )
    assert "memory/robotwin/MEMORY.md" in system
    assert "task: beat_block_hammer" in user
    assert "seed: 100000" in user
    assert "task_config: demo_randomized" in user
    assert "{{" not in system + user


@pytest.mark.parametrize("mode", ["eval", "explore"])
def test_uniwam_prompt_selects_wam_tool_and_memory_for_mode(mode):
    variables = {
        "policy_backend": "uniwam",
        "policy_kind": "wam",
        "mode": mode,
        "memory_dir": "shared-memory",
        "task_name": "beat_block_hammer",
        "task_config": "demo_randomized",
        "seed": 7,
        "reference_tag": "beat_block_hammer_s0",
        "recipe_tag": "test",
        "session_number": 1,
        "session_max": 3,
        "memory_inbox": "shared-memory/inbox",
        "explore_attempts_per_session": 5,
        "output_dir": "output",
    }
    system = format_prompt(system_prompt(variables), variables=variables)
    user = format_prompt(user_prompt(variables), variables=variables)
    assert "GUIDE_RPENT.md" in system
    assert "{{" not in system + user
    assert "wam_act" in system + user
    assert "lingbot_act" not in system + user
    if mode == "eval":
        assert "shared-memory/MEMORY.md" in system
    else:
        assert "shared-memory/inbox/wip/" in system


@pytest.mark.parametrize(
    ("backend", "exploring"),
    [
        ("lingbot", False),
        ("lingbot", True),
        ("uniwam", False),
        ("uniwam", True),
        ("fast-wam", False),
    ],
)
def test_configured_prompt_uses_common_workflow(backend, exploring, tmp_path):
    from argparse import ArgumentParser

    from robots.robotwin.robot_spec import get_robot_spec
    from rpent.prompt import evaluate, explore

    spec = get_robot_spec()
    parser = ArgumentParser()
    spec.add_cli_args(parser, False)
    args = parser.parse_args(
        [
            "--task-name",
            "beat_block_hammer",
            "--seed",
            "7",
            "--am-backend",
            backend,
            "--am-endpoint",
            "http://localhost:8117",
        ]
    )
    args.explore = exploring
    args.memory_dir = tmp_path / "memory"
    args.output_dir = tmp_path
    config = spec.parse_config(args)
    variables = {**config.prompt_vars, "output_dir": tmp_path}
    sections = spec.prompts.system(variables)
    common = (explore if exploring else evaluate).system_prompt()
    assert list(sections) == list(common)
    assert all(sections[key].startswith(body) for key, body in common.items())
    assert list(spec.prompts.user(variables)) == ["TASK", "BEGIN"]
    system = spec.prompts.render("system", variables=variables)
    user = spec.prompts.render("user", variables=variables)
    assert "{{" not in system + user
    assert ("GUIDE_RPENT.md" in system) == (backend != "fast-wam")
    assert ("attempt_<N>_failed.json" in system) == exploring
    action = sections["ACTION AND VERIFICATION"]
    assert ("lingbot_act" in action) == (backend == "lingbot")
    assert ("wam_act" in action) == (backend != "lingbot")
    assert ("16 qpos actions" in action) == (backend == "uniwam")
    assert "TASK_ENV.eval_success=true" in sections["SUCCESS AND REPORTING"]

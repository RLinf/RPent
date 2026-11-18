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

from robots.robotwin.robot_spec import get_robot_spec
from rpent.cli.main import _build_argparser
from rpent.robots.prompt_bundle import PromptBundle


@pytest.mark.parametrize("backend", [None, "lingbot"])
def test_prompts_render_with_memory_relative_paths(backend):
    prompts, variables = _configured_prompts(backend, "eval")
    variables.update(
        {
            "task_name": "beat_block_hammer",
            "task_config": "demo_randomized",
            "seed": 100000,
            "memory_dir": "memory/robotwin",
            "reference_tag": "robotwin_beat_block_hammer_demo_randomized_s0",
        }
    )

    system = prompts.system(variables)
    user = prompts.user(variables)

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
    assert "lingbot_act" in system
    assert "checkpoint" not in system + user
    assert "task: beat_block_hammer" in user
    assert "seed: 100000" in user
    assert "task_config: demo_randomized" in user
    assert "{{" not in system + user


@pytest.mark.parametrize("mode", ["eval", "explore"])
def test_uniwam_prompt_selects_wam_tool_and_memory_for_mode(mode):
    prompts, variables = _configured_prompts("uniwam", mode)
    system = prompts.system(variables)
    user = prompts.user(variables)
    assert "GUIDE_RPENT.md" in system
    assert "{{" not in system + user
    assert "wam_act" in system + user
    assert "lingbot_act" not in system + user
    if mode == "eval":
        assert f"{variables['memory_dir']}/MEMORY.md" in system
    else:
        assert f"{variables['memory_inbox']}/wip/" in system


def _configured_prompts(backend, mode):
    spec = get_robot_spec()
    parser = _build_argparser()
    spec.add_cli_args(parser, use_dashboard=False)
    options = [
        "--robot",
        "robotwin",
        "--task-name",
        "beat_block_hammer",
        "--seed",
        "7",
        "--am-endpoint",
        "http://localhost:8117",
        "--memory-profile",
        "local",
    ]
    if backend is not None:
        options.extend(["--am-backend", backend])
    if mode == "explore":
        options.append("--explore")
    run = spec.parse_config(parser.parse_args(options))
    return spec.prompts, {**run.prompt_vars, "output_dir": str(run.output_dir)}


def test_short_wam_protocol():
    prompts, values = _configured_prompts("fast-wam", "eval")
    system = prompts.system(values)
    assert "wam_act" in system
    assert 'read_text_file(path="robots/robotwin/guides/GUIDE_RPENT.md")' in system
    assert "## Read memory before acting" not in system
    assert f"Local memory directory: {values['memory_dir']}." in system
    assert "checkpoint" not in system + prompts.user(values)


@pytest.mark.parametrize("standalone", [True, False])
def test_physical_protocol_requires_explicit_registration(tmp_path, standalone):
    def render(physical):
        return PromptBundle.for_robot(
            "robotwin",
            policy="policies/lingbot.md",
            physical=physical,
        ).system({"standalone_policy": standalone})

    onsite = tmp_path / "onsite.md"
    with pytest.raises(FileNotFoundError):
        render(onsite)
    onsite.write_text("Wait for the onsite signal.")
    assert "Wait for the onsite signal." not in render(None)
    system = render(onsite)
    assert "Wait for the onsite signal." in system
    assert "## Physical runtime" in system
    assert "lingbot_act" in system


def test_prompt_registration_rejects_ambiguous_options():
    with pytest.raises(TypeError, match="memory"):
        PromptBundle.for_robot("robotwin", memory="false")
    with pytest.raises(TypeError, match="Markdown"):
        PromptBundle.for_robot("robotwin", physical=True)

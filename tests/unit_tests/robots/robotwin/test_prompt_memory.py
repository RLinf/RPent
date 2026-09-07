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

from pathlib import Path

import pytest

from robots.robotwin.prompt_bundle import system_prompt, user_prompt
from rpent.prompt.utils import format_prompt


def test_prompts_render_with_memory_relative_paths():
    variables = {
        "task_name": "beat_block_hammer",
        "task_config": "demo_randomized",
        "seed": 100000,
        "memory_dir": "memory/robotwin",
        "reference_tag": "beat_block_hammer_s0",
    }

    system = format_prompt(system_prompt(), variables=variables)
    user = format_prompt(user_prompt(), variables=variables)

    assert "robots/robotwin/guides/GUIDE_RPENT.md" in system
    assert "memory/robotwin/task_only/beat_block_hammer_s0.json" in system
    assert "memory/robotwin/task_only/beat_block_hammer_s0_recipe.jsonl" in system
    assert "memory/robotwin/MEMORY.md" in system
    assert "task: beat_block_hammer" in user
    assert "seed: 100000" in user
    assert "task_config: demo_randomized" in user
    assert "{{" not in system + user


@pytest.mark.parametrize("profile", ["hf", "local"])
def test_memory_profile_keeps_hf_paths_and_reads_local_recipes(profile, tmp_path):
    import argparse
    from robots.robotwin.robot_spec import _add_cli_args, _parse_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(tmp_path))
    parser.add_argument("--memory-dir", default=str(tmp_path / "memory"))
    parser.add_argument("--memory-profile", default=profile)
    _add_cli_args(parser, False)
    args = parser.parse_args(["--task-name", "beat_block_hammer", "--seed", "7"])
    config = _parse_config(args)
    rendered = format_prompt(system_prompt(config.prompt_vars), variables=config.prompt_vars)
    if profile == "hf":
        assert "beat_block_hammer_s0_recipe.jsonl" in rendered
        assert "robotwin_beat_block_hammer_s<seed>" not in rendered
    else:
        assert "robotwin_beat_block_hammer_s<seed>_recipe.jsonl" in rendered
    assert "{{" not in rendered


def test_exploration_prompt_and_inbox_permissions(monkeypatch, tmp_path):
    import argparse
    from types import SimpleNamespace
    from robots.robotwin import robot_spec, toolkit
    from rpent.dashboard.events import NullDashboardEventSink

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(tmp_path))
    parser.add_argument("--memory-dir", default=str(tmp_path / "memory"))
    parser.add_argument("--explore", action="store_true")
    robot_spec._add_cli_args(parser, False)
    args = parser.parse_args(["--task-name", "beat_block_hammer", "--seed", "7", "--explore"])
    config = robot_spec._parse_config(args)
    variables = {**config.prompt_vars, "output_dir": str(tmp_path)}
    rendered = format_prompt(system_prompt(variables), variables=variables)
    assert "{{" not in rendered
    assert "episode_status.eval_success" in rendered
    assert "TASK_ENV.eval_success" in rendered
    assert "configured exact seed" in rendered
    assert "layout determinism has not been verified" in rendered
    assert "remaining_steps = step_lim - take_action_cnt" in rendered
    assert "evaluator implementation" in rendered
    assert "hidden state or rewards" in rendered
    assert "raw expert trajectories" in rendered
    assert "Do not use shell, Python, network clients" in rendered
    assert "registered RoboTwin Toolkit is the only control surface" in rendered
    from robots.robotwin.prompts import system as evaluation_parts

    assert system_prompt(variables)["CONDITIONAL TASK-FAMILY PLAYBOOKS"] == (
        evaluation_parts.TASK_FAMILIES
    )
    assert "must not be restarted" not in rendered
    assert "restores the same seed" not in rendered
    assert "5" in rendered
    assert "no-restart episode" not in rendered
    assert str(tmp_path / (config.recipe_tag + ".json")) in rendered
    monkeypatch.setattr(toolkit, "RoboTwinToolkit", lambda **kw: SimpleNamespace(**kw))
    instance = robot_spec.get_toolkit(primitives_kwargs={},
        dashboard_events=NullDashboardEventSink(), config=config, mode="exploration")
    write = instance.memory.get_common_tool_bindings()["write_text_file"][1]
    own = Path(config.prompt_vars["memory_inbox"]) / "wip" / "notes.md"
    write(str(own), "working notes")
    assert own.read_text() == "working notes"
    with pytest.raises(PermissionError):
        write(str(own.parents[2] / "other_cell" / "notes.md"), "forbidden")

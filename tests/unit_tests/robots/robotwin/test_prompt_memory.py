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

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.robotwin import robot_spec, toolkit, tools
from robots.robotwin.prompt_bundle import system_prompt, user_prompt
from robots.robotwin.robot_spec import _add_cli_args, _parse_config
from rpent.dashboard.events import NullDashboardEventSink
from rpent.prompt.utils import format_prompt


def _exploration_variables(tmp_path: Path) -> dict[str, object]:
    memory_dir = tmp_path / "memory"
    recipe_tag = "robotwin_beat_block_hammer_s7"
    return {
        "task_name": "beat_block_hammer",
        "task_config": "demo_randomized",
        "seed": 7,
        "memory_dir": str(memory_dir),
        "reference_tag": "robotwin_beat_block_hammer_s0",
        "recipe_tag": recipe_tag,
        "mode": "explore",
        "memory_profile": "local",
        "memory_inbox": str(memory_dir / "_internal" / "inbox" / recipe_tag),
        "session_number": 1,
        "session_max": 3,
        "attempts_per_session": 5,
        "output_dir": str(tmp_path),
    }


def _render_exploration_prompt(tmp_path: Path) -> str:
    variables = _exploration_variables(tmp_path)
    return format_prompt(system_prompt(variables), variables=variables)


def _normalized(text: str) -> str:
    return " ".join(text.split())


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(tmp_path))
    parser.add_argument("--memory-dir", default=str(tmp_path / "memory"))
    parser.add_argument("--memory-profile", default=profile)
    _add_cli_args(parser, False)
    args = parser.parse_args(["--task-name", "beat_block_hammer", "--seed", "7"])
    config = _parse_config(args)
    rendered = format_prompt(
        system_prompt(config.prompt_vars), variables=config.prompt_vars
    )
    if profile == "hf":
        assert "beat_block_hammer_s0_recipe.jsonl" in rendered
        assert "robotwin_beat_block_hammer_s<seed>" not in rendered
    else:
        assert "robotwin_beat_block_hammer_s<seed>_recipe.jsonl" in rendered
    assert "{{" not in rendered


def test_exploration_prompt_and_inbox_permissions(monkeypatch, tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(tmp_path))
    parser.add_argument("--memory-dir", default=str(tmp_path / "memory"))
    parser.add_argument("--explore", action="store_true")
    robot_spec._add_cli_args(parser, False)
    args = parser.parse_args(
        ["--task-name", "beat_block_hammer", "--seed", "7", "--explore"]
    )
    config = robot_spec._parse_config(args)
    variables = {**config.prompt_vars, "output_dir": str(tmp_path)}
    rendered = format_prompt(system_prompt(variables), variables=variables)
    normalized = _normalized(rendered)
    assert "{{" not in rendered
    assert "episode_status.eval_success" in rendered
    assert "TASK_ENV.eval_success" in rendered
    assert "configured exact seed" in rendered
    assert "layout determinism has not been verified" in normalized
    assert "remaining_steps = step_lim - take_action_cnt" in rendered
    assert "evaluator implementation" in rendered
    assert "hidden simulator state or rewards" in normalized
    assert "raw expert trajectories" in rendered
    assert "Do not use shell, Python, network clients" in normalized
    assert "registered RoboTwin Toolkit is the only control surface" in normalized
    from robots.robotwin.prompts import system as evaluation_parts

    assert evaluation_parts.TASK_FAMILIES in rendered
    assert "must not be restarted" not in rendered
    assert "restores the same seed" not in rendered
    assert "5 attempts including the initial episode" in normalized
    assert "no-restart episode" not in rendered
    assert str(tmp_path / (config.recipe_tag + ".json")) in rendered
    monkeypatch.setattr(toolkit, "RoboTwinToolkit", lambda **kw: SimpleNamespace(**kw))
    instance = robot_spec.get_toolkit(
        primitives_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        mode="exploration",
    )
    write = instance.memory.get_common_tool_bindings()["write_text_file"][1]
    own = Path(config.prompt_vars["memory_inbox"]) / "wip" / "notes.md"
    write(str(own), "working notes")
    assert own.read_text() == "working notes"
    with pytest.raises(PermissionError):
        write(str(own.parents[2] / "other_cell" / "notes.md"), "forbidden")


def test_exploration_prompt_keeps_live_task_language_authoritative(tmp_path):
    rendered = _render_exploration_prompt(tmp_path)
    normalized = _normalized(rendered)

    assert "complete `task_language`" in normalized
    assert "authoritative objective" in normalized
    assert (
        "Every `lingbot_act` automatically receives that exact, complete" in normalized
    )
    assert (
        "Never replace, shorten, translate, reinterpret, or stage-condition"
        in normalized
    )
    assert "cannot supply a different VLA instruction" in normalized
    assert "Stale subtask text from any prior attempt is equally non-authoritative" in (
        normalized
    )
    assert "Memory contains advisory strategy evidence only" in normalized
    assert "never authoritative scene geometry or object state" in normalized
    assert "historical pixels, coordinates, poses" in normalized


def test_exploration_prompt_requires_accuracy_loop_and_targeted_recovery(tmp_path):
    rendered = _render_exploration_prompt(tmp_path)
    normalized = _normalized(rendered)

    for requirement in (
        "follow observe, act, verify, and adapt",
        "completed and protected subgoals",
        "first unmet postcondition",
        "smallest useful action or recovery",
        "Predict the observable effect",
        "Verify the predicted effect before advancing",
        "do not repeat an unchanged failed strategy without new evidence",
        "Protect completed and nearly completed task state",
        "concise internal ledger",
    ):
        assert requirement in normalized

    assert "continued recovery in that attempt is no longer useful" in normalized
    assert "Use only the registered `reset` lifecycle tool" in normalized
    assert "re-perceive the new live scene before acting" in normalized


def test_exploration_prompt_control_and_verification_match_public_tools(tmp_path):
    rendered = _render_exploration_prompt(tmp_path)
    normalized = _normalized(rendered)
    registered = {spec["name"] for spec in tools.TOOLS_SPEC}

    referenced_controls = {
        "lingbot_act",
        "move_to",
        "rotate_wrist",
        "set_gripper",
        "release",
        "reset",
        "finish",
    }
    assert referenced_controls <= registered
    for name in referenced_controls:
        assert f"`{name}`" in rendered

    assert "pure VLA, pure analytic control, or a hybrid" in normalized
    assert "fine contact" in normalized
    assert "verified free-space transport" in normalized
    assert "Select `chunks` from current progress" in normalized
    for gate in (
        "Grasp and continued hold",
        "Transport",
        "Placement, release, stability, and withdrawal",
        "Contact or mechanism",
        "Handover",
    ):
        assert gate in normalized
    assert (
        "A completed primitive is mechanical feedback, not semantic task success"
        in normalized
    )


def test_exploration_prompt_uses_only_native_success_and_current_scene(tmp_path):
    rendered = _render_exploration_prompt(tmp_path)
    normalized = _normalized(rendered)

    assert "Only current `episode_status.eval_success=true`" in normalized
    assert "sourced from native `TASK_ENV.eval_success`" in normalized
    assert "Stop issuing robot actions immediately" in normalized
    assert "After every reset and every scene-changing event, re-localize" in normalized
    assert "Use only current driver/toolkit-visible" in normalized


def test_exploration_prompt_excludes_standalone_and_unsupported_contracts(tmp_path):
    variables = _exploration_variables(tmp_path)
    rendered = format_prompt(system_prompt(variables), variables=variables)
    rendered += format_prompt(user_prompt(variables), variables=variables)

    for forbidden in (
        "Never RESET",
        "one no-restart episode",
        "No Prior Recipe Or Memory",
        "command.json",
        "done_",
        "WORKDIR",
        "final.json",
        "recipe_{TAG}.jsonl",
        '"agent_commands"',
        '"confirmed_attempts"',
        '"final_state"',
        "`descend`",
        "`reset_session`",
        "`prompt`",
        "`use_length`",
        "chunks=",
        "0.005",
        "0.010",
    ):
        assert forbidden not in rendered

    lingbot = next(spec for spec in tools.TOOLS_SPEC if spec["name"] == "lingbot_act")
    assert "never sent to the policy" in lingbot["description"]
    assert "agent_prompt_ignored" not in rendered
    assert "grasp_success" not in rendered
    assert "collision_free" not in rendered
    assert "mechanism_success" not in rendered
    assert "═" not in rendered
    assert re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", rendered) is None

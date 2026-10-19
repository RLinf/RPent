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

"""Offline tests for dual-Franka environment discovery and config loading."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.dual_franka import get_robot_spec
from robots.dual_franka.runtime_config import load_runtime_config
from rpent.robots.base import enumerate_robots
from rpent.robots.base import get_robot_spec as resolve_robot_spec


@pytest.mark.parametrize("task_id", [0, 1, 2])
def test_named_task_memory_preserves_flash_task_id(
    monkeypatch, dual_franka_robot_config, task_id
):
    from rpent.cli.main import _build_argparser

    spec = get_robot_spec()
    parser = _build_argparser()
    spec.add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(["--robot", "dual_franka", "--task-id", str(task_id)])
    args.robot_config = str(dual_franka_robot_config)
    config = spec.parse_config(args)
    assert (config.recipe_tag == f"dual_franka_t{task_id}") is (task_id == 0)
    assert config.prompt_vars["recipe_tag"] == config.recipe_tag
    assert Path(config.prompt_vars["memory_inbox"]).name == config.recipe_tag
    card_tag = f"dual_franka_t{task_id}"
    toolkit = SimpleNamespace(flash_options={"card": {"task": card_tag}})
    calls = []
    monkeypatch.setattr(
        "robots.franka.flash.replay.replay_card",
        lambda toolkit, tag, note: calls.append(tag) or {"done": True},
    )
    assert spec.run_flash(toolkit, config.recipe_tag, lambda text: None)["done"]
    assert calls == [card_tag]
    with pytest.raises(ValueError, match="does not match"):
        spec.run_flash(toolkit, "different_task", lambda text: None)


def test_dual_franka_extension_is_discoverable():
    assert "dual_franka" in enumerate_robots()
    spec = resolve_robot_spec("dual_franka")
    assert spec.name == "dual_franka"
    assert spec.supports_exploration is True
    assert get_robot_spec().name == spec.name
    runtime_components = {item["name"] for item in spec.dashboard["runtime_components"]}
    assert "sam3" in runtime_components


def test_franka_extensions_declare_real_robot():
    single = resolve_robot_spec("franka")
    dual = resolve_robot_spec("dual_franka")
    assert single.is_real_robot and dual.is_real_robot


def test_dual_franka_uses_rpent_owned_robot_config(
    fake_rlinf_realworld_modules, monkeypatch
):
    from robots.dual_franka.runtime_config import load_mapping

    config_path = Path(__file__).parents[4] / "robots/dual_franka/config/example.yaml"
    runtime = load_runtime_config(None, task_description="test task")
    cfg = runtime.rlinf

    assert config_path.is_file()
    assert cfg.env.eval.init_params.id == "RPentDualFrankaTCPEnv-v1"
    assert cfg.env.eval.override_cfg.task_description == "test task"
    assert runtime.controller["move_tolerance_m"] == 0.006
    assert runtime.controller["rotate_tolerance_rad"] == 0.04
    assert runtime.controller["iteration_multiplier"] == 50
    assert runtime.controller["min_iterations"] == 200
    assert runtime.controller["gripper_settle_s"] == 1.5
    thresholds = runtime.controller["joint_health_thresholds"]
    assert thresholds["left"] == thresholds["right"]
    assert thresholds == load_mapping(config_path)["joint_health"]["thresholds"]
    assert runtime.controller["rotate_max_drift_m"] == 0.10
    raw = load_mapping(config_path)
    monkeypatch.setattr(
        "robots.dual_franka.runtime_config.load_mapping", lambda path: raw
    )
    for rotation_hold, expected in [({}, 0.10), ({"max_drift_m": 0.03}, 0.03)]:
        raw["rotation_hold"] = rotation_hold
        runtime = load_runtime_config(None, task_description="test task")
        assert runtime.controller["rotate_max_drift_m"] == expected

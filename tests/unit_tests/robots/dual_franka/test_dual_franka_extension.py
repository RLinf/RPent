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

"""Offline tests for dual-Franka environment discovery and prompt configuration."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from robots.dual_franka import get_robot_spec
from robots.dual_franka.robot_spec import (
    _add_cli_args,
    _env_server_command,
    _parse_config,
    _spawn_env_server,
    _vla_server_command,
)
from robots.dual_franka.runtime_config import load_runtime_config
from rpent.robots.base import enumerate_robots
from rpent.robots.base import get_robot_spec as resolve_robot_spec
from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from rpent.utils.daemon import ProcessDaemon


def test_dual_franka_extension_is_discoverable_and_renders_task_prompt(tmp_path: Path):
    assert "dual_franka" in enumerate_robots()
    spec = resolve_robot_spec("dual_franka")
    assert spec.name == "dual_franka"

    run_config = _parse_config(
        Namespace(
            task_id=1,
            output_dir=tmp_path,
        )
    )
    user_prompt = spec.prompts.render("user", variables=run_config.prompt_vars)

    assert run_config.recipe_tag == "dual_franka_t1"
    assert run_config.task_desc == {"task_id": 1, "task_name": "vla_grasp"}
    assert "Use bounded analytic motion" in user_prompt
    assert str(tmp_path) in user_prompt


def test_direct_dual_franka_spec_matches_registry_resolution():
    assert get_robot_spec().name == resolve_robot_spec("dual_franka").name


def test_dual_franka_cli_uses_current_interpreter_without_override_option():
    parser = argparse.ArgumentParser()
    _add_cli_args(parser, use_dashboard=False)

    args = parser.parse_args([])

    assert not hasattr(args, "rlinf_python")
    command = _env_server_command(
        args,
        host="127.0.0.1",
        port=5556,
    )
    assert command[:3] == [
        sys.executable,
        "-m",
        "robots.dual_franka.env_server",
    ]
    assert "--config-name" not in command
    assert "--override" not in command
    assert "--task-description" in command


def test_dual_franka_uses_rpent_owned_robot_config():
    pytest.importorskip("rlinf.envs.realworld.franka.franka_env")
    config_path = Path(__file__).parents[4] / "robots/dual_franka/config/example.yaml"
    runtime = load_runtime_config(None, task_description="test task")
    cfg = runtime.rlinf

    assert config_path.is_file()
    assert cfg.env.eval.init_params.id == "DualFrankaTCPEnv-v1"
    hardware = cfg.cluster.node_groups[0].hardware.configs[0]
    assert hardware.left_robot_ip == "172.16.0.2"
    assert hardware.right_robot_ip == "172.16.0.2"
    assert hardware.right_controller_node_rank == 1
    assert cfg.env.eval.override_cfg.task_description == "test task"
    assert runtime.controller["move_max_step_m"] == 0.02


def test_dual_franka_env_server_reuses_preprovisioned_ray_environments(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr("robots.dual_franka.robot_spec.pick_free_port", lambda: 5556)
    monkeypatch.setattr(ProcessDaemon, "start", lambda self: None)

    daemon, _rpc = _spawn_env_server(
        Namespace(env_endpoint=None, task_id=1, robot_config=None),
        tmp_path,
    )

    assert daemon is not None
    assert daemon.subprocess_env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"


def test_dual_franka_vla_server_command_uses_checkpoint_and_repo_id():
    args = Namespace(
        vla_model_path="/models/global_step_5000",
        vla_repo_id="org/dual-franka-tcp-rot6d",
        cuda_device=2,
    )

    command = _vla_server_command(args, host="127.0.0.1", port=6000)

    assert command[:3] == [
        sys.executable,
        "-m",
        "robots.dual_franka.vla_server",
    ]
    assert command[command.index("--model-path") + 1] == args.vla_model_path
    assert command[command.index("--repo-id") + 1] == args.vla_repo_id
    assert command[command.index("--cuda-device") + 1] == "2"


def test_dual_franka_vla_config_and_obs_encoding():
    pytest.importorskip("torch")
    from robots.dual_franka.vla_server import build_model_cfg

    cfg = build_model_cfg("/models/global_step_5000", "org/dataset")

    client = Pi05VLAClient(None, embodiment="dual_franka")
    observation = client.encode_obs(
        {
            "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
            "extra_view_images": np.zeros((2, 8, 8, 3), dtype=np.uint8),
            "states": np.zeros(20, dtype=np.float32),
            "task_descriptions": "grasp the cube",
        }
    )

    assert cfg.openpi.config_name == "pi05_dualfranka_tcp_rot6d"
    assert cfg.openpi_data.repo_id == "org/dataset"
    assert cfg.action_dim == 20
    assert cfg.num_action_chunks == 20
    assert observation["main_images"].shape == (1, 8, 8, 3)
    assert observation["extra_view_images"].shape == (1, 2, 8, 8, 3)
    assert observation["states"].shape == (1, 20)
    assert observation["task_descriptions"] == ["grasp the cube"]

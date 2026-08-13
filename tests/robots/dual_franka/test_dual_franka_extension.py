"""Offline tests for dual-Franka environment discovery and prompt configuration."""

from __future__ import annotations

import argparse
import base64
import io
import sys
from argparse import Namespace
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from robots.dual_franka import (
    _add_cli_args,
    _env_server_command,
    _parse_config,
    _vla_server_command,
    get_env_spec,
)
from robots.dual_franka.env_server import _compose_config
from robots.dual_franka.vla_server import _build_env_obs, build_model_cfg
from rpent.envs.base import enumerate_envs
from rpent.envs.base import get_env_spec as resolve_env_spec


def test_dual_franka_extension_is_discoverable_and_renders_task_prompt(tmp_path: Path):
    assert "dual_franka" in enumerate_envs()
    spec = resolve_env_spec("dual_franka")
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
    assert get_env_spec().name == resolve_env_spec("dual_franka").name


def test_dual_franka_cli_uses_current_interpreter_without_override_option():
    parser = argparse.ArgumentParser()
    _add_cli_args(parser, use_dashboard=False)

    args = parser.parse_args([])

    assert not hasattr(args, "rlinf_python")
    command = _env_server_command(
        args,
        host="127.0.0.1",
        port=5556,
        output_dir=Path("logs/test"),
    )
    assert command[0] == sys.executable


def test_dual_franka_uses_rpent_owned_runtime_config():
    config_path = (
        Path(__file__).parents[3]
        / "robots/dual_franka/config/realworld_physical_agent_eval_dual_franka.yaml"
    )

    cfg = _compose_config("realworld_physical_agent_eval_dual_franka", [])

    assert config_path.is_file()
    assert cfg.env.eval.init_params.id == "DualFrankaTcpEnv-v1"
    hardware = cfg.cluster.node_groups[0].hardware.configs[0]
    assert hardware.left_robot_ip == "LEFT_ROBOT_IP"
    assert hardware.right_robot_ip == "RIGHT_ROBOT_IP"


def test_dual_franka_vla_server_command_uses_checkpoint_and_repo_id():
    args = Namespace(
        vla_model_path="/models/global_step_5000",
        vla_repo_id="org/dual-franka-tcp-rot6d",
        cuda_device=2,
    )

    command = _vla_server_command(args, host="127.0.0.1", port=6000)

    assert command[0] == sys.executable
    assert command[command.index("--model-path") + 1] == args.vla_model_path
    assert command[command.index("--repo-id") + 1] == args.vla_repo_id
    assert command[command.index("--cuda-device") + 1] == "2"


def test_dual_franka_vla_config_and_wire_observation_mapping():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    imageio.imwrite(buffer, image, format="png")
    block = {
        "format": "png",
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }

    cfg = build_model_cfg("/models/global_step_5000", "org/dataset")
    observation = _build_env_obs(
        "grasp the cube",
        {"main": block, "extra": [block, block]},
        [[0.0] * 20],
    )

    assert cfg.openpi.config_name == "pi05_dualfranka_tcp_rot6d"
    assert cfg.openpi_data.repo_id == "org/dataset"
    assert cfg.action_dim == 20
    assert cfg.num_action_chunks == 20
    assert observation["main_images"].shape == (1, 8, 8, 3)
    assert observation["extra_view_images"].shape == (1, 2, 8, 8, 3)
    assert observation["states"].shape == (1, 20)

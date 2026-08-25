"""Offline tests for Franka environment discovery and prompt configuration."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import gymnasium as gym
import numpy as np

from robots.franka import (
    _add_cli_args,
    _env_server_command,
    _parse_config,
    get_env_spec,
)
from robots.franka.physical_agent_env import (
    PhysicalAgentFrankaConfig,
    register_physical_agent_franka_env,
)
from robots.franka.runtime_config import load_runtime_config
from rpent.envs.base import enumerate_envs
from rpent.envs.base import get_env_spec as resolve_env_spec


def test_franka_extension_is_discoverable_and_renders_task_prompt(tmp_path: Path):
    assert "franka" in enumerate_envs()
    spec = resolve_env_spec("franka")
    assert spec.name == "franka"

    run_config = _parse_config(
        Namespace(
            task_id=1,
            output_dir=tmp_path,
        )
    )
    user_prompt = spec.prompts.render("user", variables=run_config.prompt_vars)

    assert run_config.recipe_tag == "franka_t1"
    assert run_config.task_desc == {"task_id": 1, "task_name": "vla_grasp"}
    assert "Use bounded analytic motion" in user_prompt
    assert str(tmp_path) in user_prompt


def test_direct_franka_spec_matches_registry_resolution():
    assert get_env_spec().name == resolve_env_spec("franka").name


def test_franka_cli_uses_current_interpreter_without_override_option():
    parser = argparse.ArgumentParser()
    _add_cli_args(parser, use_dashboard=False)

    args = parser.parse_args([])

    assert not hasattr(args, "rlinf_python")
    command = _env_server_command(
        args,
        host="127.0.0.1",
        port=5555,
    )
    assert command[0] == sys.executable
    assert "--config-name" not in command
    assert "--override" not in command
    assert "--task-description" in command


def test_franka_cli_accepts_local_rlinf_checkout(monkeypatch, tmp_path: Path):
    (tmp_path / "rlinf").mkdir()
    (tmp_path / "rlinf/__init__.py").touch()
    monkeypatch.setenv("RLINF_REPO_PATH", str(tmp_path))
    parser = argparse.ArgumentParser()
    _add_cli_args(parser, use_dashboard=False)

    args = parser.parse_args([])

    assert args.rlinf_root == str(tmp_path)


def test_franka_uses_rpent_owned_robot_config():
    config_path = Path(__file__).parents[3] / "robots/franka/robot_config.yaml"
    runtime = load_runtime_config(config_path, task_description="test task")
    cfg = runtime.rlinf

    assert config_path.is_file()
    assert cfg.env.eval.init_params.id == "PhysicalAgentFrankaEnv-v1"
    assert cfg.cluster.node_groups[0].hardware.configs[0].robot_ip == "ROBOT_IP"
    assert cfg.env.eval.override_cfg.task_description == "test task"
    assert runtime.controller["move"]["tolerance_m"] == 0.005


def test_physical_agent_config_derives_reset_and_safety_bounds():
    target = np.array([0.5, 0.1, 0.2, 3.0, 0.0, 0.25])
    config = PhysicalAgentFrankaConfig(
        target_ee_pose=target,
        clip_x_range=0.2,
        clip_y_range=0.3,
        clip_z_range_low=0.04,
        clip_z_range_high=0.1,
        clip_roll_pitch_range=0.05,
        clip_rz_range=0.4,
        compliance_param={"translational_clip_x": 0.005},
    )

    np.testing.assert_allclose(config.reset_ee_pose, target + [0, 0, 0.1, 0, 0, 0])
    np.testing.assert_allclose(
        config.ee_pose_limit_min,
        [0.3, -0.2, 0.16, 2.95, -0.05, -0.15],
    )
    np.testing.assert_allclose(
        config.ee_pose_limit_max,
        [0.7, 0.4, 0.3, 3.05, 0.05, 0.65],
    )
    assert config.compliance_param["translational_clip_x"] == 0.005


def test_rpent_franka_registration_exists():
    register_physical_agent_franka_env()
    assert gym.spec("PhysicalAgentFrankaEnv-v1") is not None

"""Offline tests for Franka environment discovery and prompt configuration."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import gymnasium as gym
import pytest

from robots.franka import (
    _add_cli_args,
    _env_server_command,
    _parse_config,
    get_robot_spec,
)
from robots.franka.physical_agent_env import register_physical_agent_franka_env
from robots.franka.runtime_config import load_runtime_config
from rpent.robots.base import enumerate_robots
from rpent.robots.base import get_robot_spec as resolve_robot_spec


def test_franka_extension_is_discoverable_and_renders_task_prompt(tmp_path: Path):
    assert "franka" in enumerate_robots()
    spec = resolve_robot_spec("franka")
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
    assert get_robot_spec().name == resolve_robot_spec("franka").name


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


def test_franka_uses_rpent_owned_robot_config(monkeypatch):
    monkeypatch.setenv("ROBOT_IP", "10.0.0.5")
    monkeypatch.setenv("CAMERA_SERIAL_WRIST", "111111111111")
    monkeypatch.setenv("CAMERA_SERIAL_EXTERNAL", "222222222222")
    config_path = Path(__file__).parents[3] / "robots/franka/config/robot_config.yaml"
    runtime = load_runtime_config(config_path, task_description="test task")
    cfg = runtime.rlinf

    assert config_path.is_file()
    assert cfg.env.eval.init_params.id == "PhysicalAgentFrankaEnv-v1"
    assert cfg.cluster.node_groups[0].hardware.configs[0].robot_ip == "10.0.0.5"
    assert cfg.env.eval.override_cfg.task_description == "test task"
    assert runtime.controller["move_tolerance_m"] == 0.005


def test_rpent_franka_registration_exists():
    register_physical_agent_franka_env()
    assert gym.spec("PhysicalAgentFrankaEnv-v1") is not None


def test_franka_identity_resolves_from_flags_over_env(monkeypatch):
    from robots.franka.runtime_config import load_mapping, resolve_identity

    monkeypatch.setenv("ROBOT_IP", "10.0.0.5")
    monkeypatch.setenv("CAMERA_SERIAL_WRIST", "111111111111")
    monkeypatch.setenv("CAMERA_SERIAL_EXTERNAL", "222222222222")
    config_path = Path(__file__).parents[3] / "robots/franka/config/robot_config.yaml"
    config = resolve_identity(load_mapping(config_path), robot_ip="10.0.0.99")

    assert config["robot"]["ip"] == "10.0.0.99"  # flag beats environment
    assert config["cameras"]["devices"]["wrist_1"]["serial"] == "111111111111"
    assert config["cameras"]["devices"]["third_person"]["serial"] == "222222222222"


def test_franka_identity_requires_robot_ip(monkeypatch):
    from robots.franka.runtime_config import load_mapping, resolve_identity

    monkeypatch.delenv("ROBOT_IP", raising=False)
    config_path = Path(__file__).parents[3] / "robots/franka/config/robot_config.yaml"
    with pytest.raises(ValueError, match="ROBOT_IP"):
        resolve_identity(load_mapping(config_path))

"""Offline tests for Franka environment discovery and prompt configuration."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

from robots.franka import (
    _add_cli_args,
    _env_server_command,
    _parse_config,
    get_env_spec,
)
from robots.franka.env_server import _compose_config
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
        output_dir=Path("logs/test"),
    )
    assert command[0] == sys.executable


def test_franka_uses_rpent_owned_runtime_config():
    config_path = (
        Path(__file__).parents[3]
        / "robots/franka/config/realworld_physical_agent_eval.yaml"
    )

    cfg = _compose_config("realworld_physical_agent_eval", [])

    assert config_path.is_file()
    assert cfg.env.eval.init_params.id == "PhysicalAgentFrankaEnv-v1"
    assert cfg.cluster.node_groups[0].hardware.configs[0].robot_ip == "ROBOT_IP"

"""Offline tests for dual-Franka environment discovery and prompt configuration."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

from robots.dual_franka import (
    _add_cli_args,
    _env_server_command,
    _parse_config,
    get_env_spec,
)
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

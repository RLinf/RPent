"""Offline tests for dual-Franka environment discovery and prompt configuration."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from robots.dual_franka import _parse_config, get_env_spec
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

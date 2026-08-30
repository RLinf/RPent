from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from rpent.prompt.utils import format_prompt

REPO_ROOT = Path(__file__).resolve().parents[2]
BEHAVIOR_ROOT = REPO_ROOT / "robots" / "behavior"
pytestmark = pytest.mark.skipif(
    not BEHAVIOR_ROOT.is_dir(),
    reason="BEHAVIOR robot plugin has not landed in this worktree yet",
)


def _prompt_module():
    return importlib.import_module("robots.behavior.prompt_bundle")


def _render(factory: Any, variables: dict[str, object]) -> str:
    return format_prompt(factory(variables), variables=variables)


def _context(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "behavior_mode": "explore",
        "task_name": "picking_up_trash",
        "task_language": (
            "Put the three can of soda from the living room inside the tash can "
            "in the kitchen."
        ),
        "public_seed": 0,
        "recipe_tag": "picking_up_trash_s0",
        "output_dir": "/tmp/behavior-out",
        "max_episode_steps": 43200,
        "global_tool_budget": 350,
        "wall_clock_seconds": 7200,
        "task_instruction": "TASK_INSTRUCTION_SENTINEL",
        "public_capabilities": "CAPABILITY_SENTINEL",
        "episode_memory": "MEMORY_SENTINEL",
        "attempt_index": 1,
        "job_id": "job-abc",
    }
    values.update(overrides)
    return values


def test_rendered_prompts_resolve_placeholders_without_interpolating_input_braces():
    module = _prompt_module()
    variables = _context(
        task_instruction='Reviewed data: {"literal": "{{ not_a_placeholder }}"}',
        episode_memory="Literal {{ prior text }}",
    )

    system = _render(module.system_prompt, variables)
    user = _render(module.user_prompt, variables)

    assert "picking_up_trash" in system + user
    assert "picking_up_trash_s0" in system + user
    assert "{{ not_a_placeholder }}" in system
    assert "{{ prior text }}" in system
    assert "{{ task_name }}" not in system + user
    assert "{{ recipe_tag }}" not in system + user


def test_prompt_uses_runtime_injected_task_capabilities_and_memory() -> None:
    module = _prompt_module()
    system = _render(module.system_prompt, _context())

    assert "TASK_INSTRUCTION_SENTINEL" in system
    assert "CAPABILITY_SENTINEL" in system
    assert "MEMORY_SENTINEL" in system
    assert "task-profile files" in system
    assert "hidden environment metadata" in system
    assert "replace them" in system


def test_prompts_keep_dynamic_chunks_and_runner_owned_termination_semantics() -> None:
    module = _prompt_module()
    system = " ".join(_render(module.system_prompt, _context()).split()).lower()

    for marker in (
        "public capabilities",
        "peer planner tools",
        "no list order implies a required sequence",
        "requires `chunks=n`",
        "choose n as a positive integer",
        "does not impose a fixed chunks value",
        "runtime contract",
        "explicit terminal receipts",
    ):
        assert marker in system
    assert "chunks=20" not in system
    assert "max_chunks" not in system
    assert "finish establishes task_success" not in system


def test_prompt_models_one_invocation_as_one_episode_attempt() -> None:
    module = _prompt_module()
    system = " ".join(_render(module.system_prompt, _context()).split()).lower()

    assert "one planner invocation is one behavior episode attempt" in system
    assert "cannot reset or restart the environment inside the invocation" in system
    assert "fresh `rpent --robot behavior --behavior-mode explore` process" in system
    assert "multi-attempt policy" in system


def test_prompt_does_not_leak_private_instances_or_old_role_cameras() -> None:
    module = _prompt_module()
    system = _render(module.system_prompt, _context())

    for private_instance in (242, 109, 181, 187, 197, 203, 211, 212, 295, 298):
        assert str(private_instance) not in system
    assert "held_wrist" not in system
    assert "press_wrist" not in system

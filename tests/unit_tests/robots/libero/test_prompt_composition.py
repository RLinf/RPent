# Copyright 2026 The RPent Authors.
"""Prompt selection and per-task binding regressions."""

import pytest

from robots.libero.prompt_bundle import system_prompt, user_prompt


@pytest.fixture
def variables():
    return {
        "suite": "libero_10_task",
        "task": 4,
        "seed": 7,
        "recipe_tag": "libero10_task_t4_s7",
        "reference_tag": "libero10_task_t4_s0",
        "memory_dir": "/memory/first",
        "output_dir": "/run/first",
        "memory_inbox": "/memory/first/inbox",
        "session_number": 1,
        "session_max": 3,
        "explore_attempts_per_session": 5,
    }


@pytest.mark.parametrize("profile", ["hf", "local"])
@pytest.mark.parametrize("direct", [False, True])
def test_pi05_evaluation_contract(variables, profile, direct):
    text = system_prompt(
        variables | {"memory_profile": profile, "enable_direct_action": direct}
    )
    assert "SINGLE-ATTEMPT MODE" in text
    assert "MULTI-ATTEMPT EXPLORE MODE" not in text
    assert "`execute_action`" in text if direct else "`execute_action`" not in text
    assert "pi0_pick" in text and "pi0_doubled" in text
    assert "/memory/first/task-specific/libero10_task_t4_s0.json" in text
    assert (
        "/memory/first/task-family/task-family_libero10_<regime>_t4.md" in text
    ) is (profile == "local")
    assert "task_language" in text and "NEVER read the BDDL" in text


def test_explore_and_task_switch_bind_current_context(variables):
    first = system_prompt(variables | {"mode": "explore"})
    changed = variables | {
        "mode": "explore",
        "task": 9,
        "seed": 8,
        "memory_dir": "/memory/second",
        "memory_inbox": "/memory/second/inbox",
        "output_dir": "/run/second",
        "recipe_tag": "second_s8",
        "reference_tag": "second_s0",
        "session_number": 2,
    }
    second = system_prompt(changed)
    assert "MULTI-ATTEMPT EXPLORE MODE" in first
    assert "This is session 2 of 3" in second
    assert "/memory/first" not in second and "/run/first" not in second
    assert "/memory/second/inbox/wip/" in second
    assert "/run/second/attempts/attempt_<N>_failed.json" in second
    assert "task: 9" in user_prompt(changed) and "seed: 8" in user_prompt(changed)


@pytest.mark.parametrize("backend", ["cosmos_policy", "cosmos3", "fastwam"])
def test_wam_stays_minimal_and_moves_startup_to_system(variables, backend):
    values = variables | {"policy_kind": "wam", "policy_backend": backend}
    system, user = system_prompt(values), user_prompt(values)
    assert "at most four per call" in system
    assert "wam_act" in system and "pi0_pick" not in system
    assert "Do not reset" in system and "terminated=true" in system
    assert "MEMORY.md" not in system and "PROVEN LEVERS" not in system
    assert "wam_act" not in user
    assert "Local memory directory: /memory/first" in user


def test_user_values_are_not_reinterpreted_as_templates(variables):
    text = user_prompt(variables | {"suite": "literal {{missing}}"})
    assert "literal {{missing}}" in text


def test_exploration_requires_explicit_attempt_budget(variables):
    values = variables | {"mode": "explore", "explore_attempts_per_session": 9}
    assert "Attempt limit for this session: 9." in system_prompt(values)
    del values["explore_attempts_per_session"]
    with pytest.raises(KeyError, match="explore_attempts_per_session"):
        system_prompt(values)

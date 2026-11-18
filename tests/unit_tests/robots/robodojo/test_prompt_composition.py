# Copyright 2026 The RPent Authors.
"""Final prompt role and task switching contracts."""

import pytest

from robots.robodojo.prompt_bundle import system_prompt, user_prompt


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
@pytest.mark.parametrize("mode", ["dev", "explore"])
def test_task_switch_keeps_literal_task_data_and_policy_boundary(backend, mode):
    variables = {
        "mode": mode,
        "policy_backend": backend,
        "seed": 2,
        "layout": 3,
        "env_cfg_type": "arx_x5",
        "task_summary": "Keep {{literal}} intact",
        "session_number": 1,
        "session_max": 3,
        "explore_attempts_per_session": 3,
    }
    for task in ("stack_blocks", "stack_blocks_random"):
        values = {
            **variables,
            "task": task,
            "reference_tag": f"{task}_s0_l0",
            "recipe_tag": f"{task}_s2_l3",
            "output_dir": f"/output/{task}",
            "memory_dir": f"/memory/{task}",
            "memory_inbox": f"/inbox/{task}",
        }
        system, user = system_prompt(values), user_prompt(values)
        assert "{{literal}}" in user
        assert f"/output/{task}" in user
        assert "Call `view_env_state` first" in system
        assert "Call `view_env_state` first" not in user
        if backend == "awomo05":
            assert "stop all motion, including Primitive actions" in system
            assert "no instruction parameter" in system
            assert "clarify the\ninstruction" not in system
        else:
            assert "inspect the scene and use fresh\nanalytic actions" in system
            assert "vpp2_act(instruction, chunks=1)" in system


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
def test_eval_fair_keeps_memory_and_normal_startup_out_of_prompt(backend):
    values = {"mode": "eval-fair", "policy_backend": backend, "task": "1", "layout": 2}
    system, user = system_prompt(values), user_prompt(values)
    assert "Task success feedback is unavailable" in system
    assert "do not claim a verified verdict" in system
    assert "Use experience in context" not in system
    assert "MEMORY.md" not in system
    assert "Call `view_env_state` first" not in system
    assert "1 / layout 2" in user
    assert "remaining_steps" in system

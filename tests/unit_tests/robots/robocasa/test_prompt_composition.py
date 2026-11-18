# Copyright 2026 The RPent Authors.
"""Final RoboCasa prompt roles and task binding."""

import pytest

from robots.robocasa.prompt_bundle import system_prompt, user_prompt


@pytest.mark.parametrize("mode", ["eval", "explore"])
def test_task_switch_binds_current_paths_once(tmp_path, make_corpus, mode):
    for seed, task in enumerate(("OpenDrawer", "StirVegetables")):
        root = make_corpus(tmp_path / task, tasks=(task,))
        variables = {
            "mode": mode,
            "task_name": task,
            "split": "target",
            "seed": seed,
            "recipe_tag": f"{task}_target_s{seed}",
            "memory_dir": str(root),
            "output_dir": f"/output/{task}/{{{{literal}}}}",
            "memory_inbox": str(root / "_internal/inbox/current"),
            "session_number": 1,
            "session_max": 3,
            "explore_attempts_per_session": 3,
        }
        system, user = system_prompt(variables), user_prompt(variables)
        assert f"Task: {task} / target" in user
        assert "{{literal}}" in user
        assert "complete live task_language" in system
        assert "RLDX_MAX_CHUNKS=40" in system
        assert "default 999" in system
        if mode == "eval":
            assert "Writing an audit alone does not finish the run" in system
            assert "After completion is accepted, stop" in system
            assert "coordinate_decimals" not in system
            assert "after-audit" not in system
            assert "reset or restart" in system
            assert "reset or restart" not in user
            assert str(root / f"task-specific/{task}_s0.json") in system
        else:
            assert "Each new attempt must change a named factor" in system
            assert "Each new attempt must change a named factor" not in user
            assert f"/output/{task}/{{{{literal}}}}" in system

# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Opt-in Fast-WAM checkpoint inference and bounded RoboTwin qpos execution."""

import os
import time

import numpy as np
import pytest

from robots.robotwin.robot_spec import ROBOTWIN_CAMERA_NAMES, get_robot_spec
from tests.e2e_tests.common import (
    ScriptedToolCall,
    parse_runtime_args,
    require_array,
    run_scripted_policy_chain,
    runtime_phase,
)

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("RPENT_FAST_WAM_ROBOTWIN_ENDPOINT")
        or not os.environ.get("ROBOTWIN_ASSETS_PATH"),
        reason="requires a RoboTwin Fast-WAM endpoint and ROBOTWIN_ASSETS_PATH",
    ),
    pytest.mark.timeout(1200),
]


def test_fast_wam_robotwin_closed_loop(tmp_path, record_property) -> None:
    spec = get_robot_spec()
    args = parse_runtime_args(
        spec,
        [
            "--task-name",
            "beat_block_hammer",
            "--seed",
            "100000",
            "--task-config",
            "demo_randomized",
            "--max-episode-steps",
            "64",
            "--wam-backend",
            "fast-wam",
            "--wam-endpoint",
            os.environ["RPENT_FAST_WAM_ROBOTWIN_ENDPOINT"],
        ],
    )
    with runtime_phase(spec, args, tmp_path / "env", {"env", "wam"}) as runtime:
        client = runtime["model"]
        caps = client.wam.get_capabilities()
        record_property("checkpoint", caps.checkpoint)
        try:
            env = runtime["env"]
            env.reset()
            client.reset()
            for index in range(2):
                observation = {
                    "views": {
                        name: {"rgb": env.render_camera(name)}
                        for name in ROBOTWIN_CAMERA_NAMES
                    },
                    "robot_state": env.last_info["robot_state"],
                    "task_language": env.get_task_language(),
                }
                started = time.perf_counter()
                actions = client.predict(observation)
                record_property(
                    f"inference_{index}_seconds", time.perf_counter() - started
                )
                assert require_array(
                    actions, "Fast-WAM qpos actions", ndim=2, last_dim=14
                ).shape == (caps.chunk_size, 14)
                np.save(tmp_path / f"actions-{index}.npy", actions)
                _, _, _, _, info = env.chunk_step(
                    actions, action_type=client.action_type
                )
                record_property(f"executed_actions_{index}", info["executed_actions"])
                assert info["executed_actions"] > 0
                require_array(
                    info["robot_state"]["qpos_target14"],
                    "next joint state",
                    ndim=1,
                    last_dim=14,
                )
                if env.terminated or env.truncated:
                    break
            record_property("terminated", env.terminated)
            record_property("truncated", env.truncated)
            client.reset()
        finally:
            client.close()
    result = run_scripted_policy_chain(
        robot="robotwin",
        robot_argv=[
            "--task-name",
            "beat_block_hammer",
            "--seed",
            "100000",
            "--task-config",
            "demo_randomized",
            "--max-episode-steps",
            "64",
            "--wam-backend",
            "fast-wam",
            "--wam-endpoint",
            os.environ["RPENT_FAST_WAM_ROBOTWIN_ENDPOINT"],
        ],
        output_dir=tmp_path / "chain",
        action=ScriptedToolCall("wam_act", {"max_chunks": 2}),
        action_count_field="executed_steps",
        use_memory=False,
    )
    assert result["status"] == "passed"

# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Opt-in Fast-WAM checkpoint inference and bounded LIBERO action execution."""

import os
import time

import numpy as np
import pytest

from robots.libero.robot_spec import get_robot_spec
from tests.e2e_tests.common import (
    ScriptedToolCall,
    parse_runtime_args,
    require_array,
    run_scripted_policy_chain,
    runtime_phase,
)

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("RPENT_FAST_WAM_ENDPOINT"),
        reason="requires a real RPENT_FAST_WAM_ENDPOINT and LIBERO assets",
    ),
    pytest.mark.timeout(1200),
]


def test_fast_wam_libero_closed_loop(tmp_path, record_property) -> None:
    spec = get_robot_spec()
    args = parse_runtime_args(
        spec,
        [
            "--suite",
            "libero_spatial",
            "--task",
            "0",
            "--seed",
            "0",
            "--max-episode-steps",
            "64",
            "--wam-backend",
            "fast-wam",
            "--wam-endpoint",
            os.environ["RPENT_FAST_WAM_ENDPOINT"],
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
                raw = {**env.raw_obs(), "task_descriptions": env.get_task_language()}
                started = time.perf_counter()
                actions = client.predict(raw)
                record_property(
                    f"inference_{index}_seconds", time.perf_counter() - started
                )
                assert require_array(
                    actions, "Fast-WAM actions", ndim=2, last_dim=7
                ).shape == (caps.chunk_size, 7)
                np.save(tmp_path / f"actions-{index}.npy", actions)
                for action in actions:
                    obs, *_ = env.step(action)
                    require_array(obs["main_images"], "next image", ndim=3, last_dim=3)
                    if env.terminated or env.truncated:
                        break
                if env.terminated or env.truncated:
                    break
            record_property("terminated", env.terminated)
            record_property("truncated", env.truncated)
            client.reset()
        finally:
            client.close()
    result = run_scripted_policy_chain(
        robot="libero",
        robot_argv=[
            "--suite",
            "libero_spatial",
            "--task",
            "0",
            "--seed",
            "0",
            "--max-episode-steps",
            "64",
            "--wam-backend",
            "fast-wam",
            "--wam-endpoint",
            os.environ["RPENT_FAST_WAM_ENDPOINT"],
        ],
        output_dir=tmp_path / "chain",
        action=ScriptedToolCall("wam_act", {"max_chunks": 2}),
        action_count_field="chunks",
        use_memory=False,
    )
    assert result["status"] == "passed"

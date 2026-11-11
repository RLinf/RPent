# Copyright 2026 The RPent Authors.
"""Opt-in SAM3 and native VLA components; SAM3 needs no VLA weights."""

import os
from pathlib import Path

import numpy as np
import pytest

from robots.robodojo.robot_spec import get_robot_spec
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from tests.e2e_tests.common import parse_runtime_args, require_rgb, runtime_phase

pytestmark = [
    pytest.mark.timeout(900),
    pytest.mark.skipif(
        os.environ.get("RPENT_RUN_ROBODOJO_INTEGRATION") != "1",
        reason="requires a reserved GPU, RoboDojo assets and SAM3 checkpoint",
    ),
]


def test_sam3_component(monkeypatch):
    from rpent.utils import logging

    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"]) / "sam3-component"
    monkeypatch.setattr(logging, "_output_dir", output)
    spec = get_robot_spec()
    args = parse_runtime_args(
        spec, ["--task", "put_bottles_into_dustbin", "--seed", "0", "--layout", "0"]
    )
    with runtime_phase(spec, args, output / "runtime", {"env", "sam3"}) as runtime:
        toolkit = RoboDojoToolkit(
            primitives_kwargs=runtime,
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(output / "memory"),
            state_output_dir=output / "states",
        )
        try:
            image = require_rgb(toolkit._state.load("cam_head.png"), "head RGB")
            prompt = "the pink bottle on the table"
            native = runtime["sam3_client"].segment(image, text_prompt=prompt)
            assert native.found and native.score >= 0.2, native.reason
            assert native.mask.shape == image.shape[:2] and native.mask.any()
            before = runtime["env"].get_status()["step"]
            response = toolkit.execute_tool(
                "segment", {"text_prompt": prompt, "step": 0}
            )
            assert not response.is_error and response.data["found"], response.data
            data = response.data
            assert len(response.images) == 1
            row, col = data["point_rc"]
            assert native.mask[row, col]
            projected = toolkit.execute_tool(
                "back_project",
                {
                    "row": row,
                    "col": col,
                    "camera": data["camera"],
                    "step": data["step_idx"],
                },
            )
            assert not projected.is_error, projected.data
            world = toolkit._state.load("head_world_xyz.npy", step=0)[row, col]
            assert np.isfinite(world).all()
            np.testing.assert_allclose(projected.data["world_xyz"], world, atol=5.1e-5)
            assert toolkit._state.latest_step == 0
            assert runtime["env"].get_status()["step"] == before
            assert (
                toolkit._state.load(data["overlay_artifact"], step=0).shape
                == image.shape
            )
        finally:
            toolkit.close()


@pytest.mark.timeout(2400)
def test_vpp2_component(vla_backend):
    if vla_backend != "vpp2":
        pytest.skip("VPP2 is not the selected backend")
    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"]) / "vpp2-component"
    spec = get_robot_spec()
    endpoint = os.environ.get("RPENT_ROBODOJO_VLA_ENDPOINT")
    args = parse_runtime_args(
        spec,
        [
            "--task",
            "put_bottles_into_dustbin",
            *(["--vla-endpoint", endpoint] if endpoint else []),
        ],
    )
    with runtime_phase(spec, args, output / "runtime", {"env", "vla"}) as runtime:
        env = runtime["env"]
        env.step_joint_update("left", gripper=0.5)
        context = env.policy_observation()
        actions = runtime["vla_client"].infer(
            context, "Pick up the pink bottle on the table"
        )
        assert len(actions) == 24
        assert all(
            np.isfinite(value).all() for action in actions for value in action.values()
        )
        obs, _, _, info = env.step(actions[0])
        assert info["executed_actions"] == 1
        after = env.policy_observation()
        assert after["inference_index"] == 1
        np.testing.assert_array_equal(
            after["history"][-1][0], obs["vision"]["cam_head"]["color"]
        )


@pytest.mark.timeout(2400)
def test_awomo05_component(vla_backend):
    if vla_backend != "awomo05":
        pytest.skip("Awomo05 is not the selected backend")
    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"]) / "awomo05-component"
    endpoint = os.environ.get("RPENT_ROBODOJO_VLA_ENDPOINT")
    spec = get_robot_spec()
    args = parse_runtime_args(
        spec,
        [
            "--task",
            "put_bottles_into_dustbin",
            "--vla-backend",
            "awomo05",
            *(["--vla-endpoint", endpoint] if endpoint else []),
        ],
    )
    with runtime_phase(spec, args, output / "runtime", {"env", "vla"}) as runtime:
        env, policy = runtime["env"], runtime["vla_client"]
        try:
            policy.reset_session()
            policy.update_obs(env.last_obs, 0)
            env.step_joint_update("left", gripper=0.5)
            policy.update_obs(env.last_obs, 1)
            actions = policy.get_action()
            assert len(actions) == 8
            assert all(
                np.isfinite(value).all()
                for action in actions
                for value in action.values()
            )
            obs, _, _, info = env.step(actions[0])
            assert info["executed_actions"] == 1
            policy.update_obs(obs, 2)
            assert info["step_after"] == 2
        finally:
            policy.close()

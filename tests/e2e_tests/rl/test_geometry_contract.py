# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Opt-in measured RGB-D, unit and motion checks through the episode boundary."""

import argparse
import contextlib
import json
import os
import shlex
from pathlib import Path

import numpy as np
import pytest

from robots.rl.rl_episode import start_environment
from robots.rl.robot_spec import get_robot_spec
from rpent.evaluation import write_json_atomic
from rpent.utils.logging import init_output_dir

pytestmark = pytest.mark.skipif(
    not os.environ.get("RPENT_GEOMETRY_ARGS"),
    reason="requires assigned simulator/SAM3 GPUs and RPENT_GEOMETRY_ARGS",
)


def _arm_pose(state, arm):
    measured = state if arm is None else state[arm]
    return np.asarray(measured["pose"]["xyz"], dtype=np.float64)


@pytest.mark.timeout(1200)
def test_episode_geometry_and_control():
    parser = argparse.ArgumentParser()
    get_robot_spec().add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(shlex.split(os.environ["RPENT_GEOMETRY_ARGS"]))
    args.output_dir = Path(os.environ["RPENT_E2E_OUTPUT_DIR"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    init_output_dir(args.output_dir)
    task = {
        "environment": args.env,
        "suite": args.suite,
        "task": args.task,
        "seed": args.seed,
    }
    evidence = {"benchmark": False, "arguments": vars(args).copy(), "cameras": []}
    query_points = json.loads(os.environ.get("RPENT_GEOMETRY_POINTS", "{}"))
    evidence["arguments"]["output_dir"] = str(args.output_dir)
    with contextlib.ExitStack() as release:
        tk, _ = start_environment(args, task, release)
        p = tk._primitives
        tk.initial_observation()
        initial = tk._robot_state()
        coordinates = p.coordinates
        factor = 100
        cameras = getattr(p, "cameras", ("agentview", "wrist"))
        step = tk.state.latest_step
        view = tk.execute_tool("view_env", {})
        assert not view.is_error and len(view.images) == len(cameras), view.data
        for camera in cameras:
            rgb = tk.state.load(f"{camera}.png", step=step)
            raw = tk.state.load(f"{camera}_world.npz", step=step)
            assert rgb.shape[:2] == raw.shape[:2] and rgb.shape[-1] == 3
            assert np.std(rgb) > 1
            valid = np.argwhere(np.isfinite(raw).all(-1) & (np.abs(raw).sum(-1) > 0))
            assert len(valid) / raw.shape[0] / raw.shape[1] > 0.2
            query = np.asarray(query_points.get(camera, np.asarray(raw.shape[:2]) // 2))
            pixel = valid[np.argmin(np.linalg.norm(valid - query, axis=1))]
            native = raw[tuple(pixel)].astype(np.float64)
            # Independently invert the saved rigid transform; do not use the
            # production point conversion to construct this expected result.
            expected = np.linalg.solve(
                coordinates.rotation, native - coordinates.origin
            )
            result = tk.execute_tool(
                "back_project",
                {"pixel": pixel.tolist(), "camera": camera, "step": step},
            )
            assert not result.is_error, result.data
            np.testing.assert_allclose(
                result.data["world_xyz"], expected * factor, atol=0.00006 * factor
            )
            evidence["cameras"].append(
                {
                    "name": camera,
                    "rgb_std": float(np.std(rgb)),
                    "valid_fraction": len(valid) / raw.shape[0] / raw.shape[1],
                    "pixel": pixel.tolist(),
                    "native_world_m": native.tolist(),
                    "back_project": result.data,
                }
            )
            segmentation = tk.execute_tool(
                "segment",
                {
                    "point": pixel.tolist(),
                    "step": step,
                    "camera": camera,
                },
            )
            assert not segmentation.is_error, segmentation.data
            evidence["cameras"][-1]["segment"] = segmentation.data
            if segmentation.data.get("code") == "not_found":
                continue
            assert "mask_id" in segmentation.data, segmentation.data
            mask_projection = tk.execute_tool(
                "back_project",
                {
                    "mask_id": segmentation.data["mask_id"],
                },
            )
            assert not mask_projection.is_error, mask_projection.data
            np.testing.assert_allclose(
                mask_projection.data["world_xyz"],
                segmentation.data["world_xyz"],
                atol=0.00006 * factor,
            )
            evidence["cameras"][-1].update(
                segment=segmentation.data, mask_back_project=mask_projection.data
            )
        assert any("mask_back_project" in camera for camera in evidence["cameras"]), (
            evidence
        )
        perception = tk.execute_tool(
            "segment", {"prompt": "table", "camera": cameras[0]}
        )
        assert not perception.is_error, perception.data
        evidence["sam3"] = perception.data
        arms = ("left", "right") if "left" in initial else (None,)
        for arm in arms:
            before = _arm_pose(tk._robot_state(), arm)
            target = before + [0, 0, 0.01]
            action = {"action": "move_to", "xyz": (target * factor).tolist()}
            if arm is not None:
                action["arm"] = arm
            arguments = {"actions": [action]}
            result = tk.execute_tool("execute_actions", arguments)
            assert not result.is_error, result.data
            after = _arm_pose(tk._robot_state(), arm)
            assert np.linalg.norm(after - before) > 0.001, result.data
            assert np.linalg.norm(after - target) < 0.006, result.data
            evidence.setdefault("motions", []).append(
                {
                    "arm": arm,
                    "target_m": target.tolist(),
                    "measured_m": after.tolist(),
                    "result": result.data,
                }
            )
        assert p.env_steps > 0
        evidence.update(
            initial=initial,
            final=tk._robot_state(),
            env_steps=p.env_steps,
            coordinates=p.coordinates.metadata(),
        )
        # Paths in args may include a simulator or asset root.
        write_json_atomic(
            args.output_dir / "geometry-result.json",
            json.loads(json.dumps(evidence, default=str)),
        )


@pytest.mark.timeout(1200)
def test_mobile_frame_and_historical_projection():
    """Measure frame changes against the native base and an unchanged old pixel."""
    parser = argparse.ArgumentParser()
    get_robot_spec().add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(shlex.split(os.environ["RPENT_GEOMETRY_ARGS"]))
    if args.env != "robocasa":
        pytest.skip("requires the native RoboCasa mobile mounting frame")
    args.output_dir = Path(os.environ["RPENT_E2E_OUTPUT_DIR"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    init_output_dir(args.output_dir)
    factor = 100
    task = {"environment": args.env, "task_name": args.task_name, "seed": args.seed}
    with contextlib.ExitStack() as release:
        tk, _ = start_environment(args, task, release)
        tk.initial_observation()
        p = tk._primitives
        step = tk.state.latest_step
        camera = p.cameras[0]
        raw = tk.state.load(f"{camera}_world.npz", step=step)
        valid = np.argwhere(np.isfinite(raw).all(-1) & (np.abs(raw).sum(-1) > 0))
        pixel = valid[len(valid) // 2]
        world_point = raw[tuple(pixel)].astype(np.float64)
        query = {"pixel": pixel.tolist(), "camera": camera, "step": step}
        evidence = {"benchmark": False, "initial_coordinates": p.coordinates.metadata()}
        initial_pose = tk._robot_state()["pose"]
        target = np.asarray(initial_pose["xyz"]) + [0, 0, 0.01]
        actions = [
            {
                "action": "move_base",
                "vx": 0.03 * factor,
                "vy": 0,
                "yaw_rate": 0.08,
                "duration_s": 1,
            },
            {
                "action": "move_lift",
                "delta_cm": 0.01 * factor,
                "duration_s": 3,
            },
            {"action": "move_to", "xyz": (target * factor).tolist()},
        ]
        rejected = tk.execute_tool(
            "execute_actions", {"actions": actions, "frame_version": 0}
        )
        assert rejected.is_error and not p.env_steps
        result = tk.execute_tool("execute_actions", {"actions": actions})
        assert result.data["executed_count"] == 3 and not result.images, result.data
        base, lift, arm = result.data["results"]
        assert base["frame_version"] > base["start_frame_version"]
        assert lift["start_frame_version"] == base["frame_version"]
        assert (
            arm["start_frame_version"]
            == lift["frame_version"]
            > lift["start_frame_version"]
        )
        current = p.coordinates
        measured = p._measure_native()
        expected = current.origin + current.rotation @ target
        np.testing.assert_allclose(measured.xyz, expected, atol=0.005)
        old = tk.execute_tool("back_project", query)
        assert old.data["capture_frame_version"] < old.data["frame_version"]
        assert (
            old.data["step"] == step
            and old.data["coordinate_step"] == tk.state.latest_step
        )
        np.testing.assert_allclose(
            old.data["world_xyz"],
            np.linalg.solve(current.rotation, world_point - current.origin) * factor,
            atol=0.00006 * factor,
        )
        evidence.update(
            actions=result.data,
            historical_point=old.data,
            final_coordinates=current.metadata(),
            env_steps=p.env_steps,
        )
        write_json_atomic(args.output_dir / "mobility-result.json", evidence)

# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native shared-workflow control probe and bounded capacity holder.

These runs are integration evidence, never benchmark or training episodes.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robots.rl.rl_episode import start_environment
from robots.rl.robot_spec import get_robot_spec
from rpent.evaluation import write_json_atomic
from rpent.utils.logging import init_output_dir


def run(args: argparse.Namespace) -> None:
    """Initialize the real task, check its views, and optionally exercise controls."""
    init_output_dir(args.output_dir)
    started = time.time()
    task = {
        "environment": args.env,
        "workflow": args.workflow,
        "seed": args.seed,
        "suite": args.suite,
        "task": args.task,
        "task_name": args.task_name,
        "task_config": args.task_config,
        "split": args.split,
    }
    evidence = {"task": task, "benchmark": False, "started_at": started}
    with contextlib.ExitStack() as release:
        toolkit, _ = start_environment(args, task, release)
        primitive = toolkit._primitives
        before = toolkit._robot_state()
        observation = toolkit.execute_tool("view_env", {})
        assert not observation.is_error and len(observation.images) >= 2
        evidence["initial_state"] = before
        evidence["cameras"] = []
        cameras = ("agentview", "wrist") if args.env == "libero" else primitive.cameras
        for name in cameras:
            rgb = toolkit.state.load(f"{name}.png")
            world = toolkit.state.load(f"{name}_world.npz")
            assert rgb.shape[:2] == world.shape[:2] and rgb.shape[-1] == 3
            assert float(np.std(rgb)) > 1
            assert float(np.isfinite(world).all(axis=-1).mean()) > 0.2
            evidence["cameras"].append(
                {
                    "camera": name,
                    "shape": list(rgb.shape),
                    "finite_depth_fraction": float(
                        np.isfinite(world).all(axis=-1).mean()
                    ),
                }
            )
        if args.mode == "controls":
            if args.env == "robocasa":
                xyz = primitive.measure().xyz.copy()
                desired = xyz + [0.005, 0, 0.01]
                result = toolkit.execute_tool("move_to", {"xyz": desired.tolist()})
                assert not result.is_error, result.data
                assert np.linalg.norm(primitive.measure().xyz - xyz) > 0.001
                assert np.linalg.norm(primitive.measure().xyz - desired) < 0.006
                base_before = np.asarray(toolkit._robot_state()["base_xyz"])
                result = toolkit.execute_tool(
                    "move_base", {"vx": 0.4, "duration_s": 0.5}
                )
                assert not result.is_error, result.data
                displacement = np.linalg.norm(
                    np.asarray(toolkit._robot_state()["base_xyz"]) - base_before
                )
                assert 0.002 < displacement < 0.1, displacement
                evidence["base_displacement_m"] = float(displacement)
                first_delta = (
                    np.asarray(toolkit._robot_state()["base_xyz"]) - base_before
                )
                rotation_before = Rotation.from_quat(
                    toolkit._robot_state()["base_quaternion_xyzw"]
                )
                result = toolkit.execute_tool(
                    "move_base", {"yaw_rate": 0.3, "duration_s": 0.5}
                )
                assert not result.is_error, result.data
                rotation_after = Rotation.from_quat(
                    toolkit._robot_state()["base_quaternion_xyzw"]
                )
                yaw_change = float(
                    (rotation_after * rotation_before.inv()).as_rotvec()[2]
                )
                assert 0.05 < yaw_change < 0.3, yaw_change
                before_second = np.asarray(toolkit._robot_state()["base_xyz"])
                result = toolkit.execute_tool(
                    "move_base", {"vx": 0.4, "duration_s": 0.5}
                )
                assert not result.is_error, result.data
                second_delta = (
                    np.asarray(toolkit._robot_state()["base_xyz"]) - before_second
                )
                expected = (rotation_after * rotation_before.inv()).apply(first_delta)
                cosine = np.dot(expected[:2], second_delta[:2]) / (
                    np.linalg.norm(expected[:2]) * np.linalg.norm(second_delta[:2])
                )
                assert cosine > 0.99, {
                    "expected_direction": expected.tolist(),
                    "measured_delta": second_delta.tolist(),
                }
                evidence["base_turn_equivariance_cosine"] = float(cosine)
                evidence["base_yaw_change_rad"] = yaw_change
            elif args.env == "robotwin":
                left, right = (
                    np.asarray(before[arm]["tcp_xyz"]) for arm in ("left", "right")
                )
                result = toolkit.execute_tool(
                    "move_to",
                    {
                        "arm": "left",
                        "xyz": (left + [0, 0, 0.01]).tolist(),
                        "other_xyz": (right + [0, 0, 0.01]).tolist(),
                    },
                )
                assert not result.is_error and result.data["status"] == "reached", (
                    result.data
                )
                after = toolkit._robot_state()
                assert (
                    np.linalg.norm(np.asarray(after["left"]["tcp_xyz"]) - left) > 0.001
                )
                assert (
                    np.linalg.norm(np.asarray(after["right"]["tcp_xyz"]) - right)
                    > 0.001
                )
                result = toolkit.execute_tool(
                    "set_gripper", {"arm": "left", "opening": 1.0, "other_opening": 1.0}
                )
                assert not result.is_error, result.data
            else:
                xyz = primitive.measure().xyz.copy()
                result = toolkit.execute_tool(
                    "move_to", {"xyz": (xyz + [0, 0, 0.01]).tolist()}
                )
                assert not result.is_error, result.data
            evidence["control_result"] = result.data
            assert primitive.env_steps > 0
            perception = toolkit.execute_tool(
                "segment", {"prompt": "object", "camera": cameras[0]}
            )
            assert not perception.is_error, perception.data
            evidence["perception"] = perception.data
        evidence.update(
            ready_at=time.time(),
            elapsed_seconds=time.time() - started,
            env_steps=primitive.env_steps,
            native_success=toolkit.solved(),
        )
        write_json_atomic(args.output_dir / "READY.json", evidence)
        if args.action_file is not None:
            while not args.action_file.exists():
                time.sleep(0.5)
            if not primitive.ended:
                if args.env == "robotwin":
                    state = toolkit._robot_state()
                    arguments = {
                        "arm": "left",
                        "xyz": (
                            np.asarray(state["left"]["tcp_xyz"]) + [0, 0, 0.01]
                        ).tolist(),
                        "other_xyz": (
                            np.asarray(state["right"]["tcp_xyz"]) + [0, 0, 0.01]
                        ).tolist(),
                    }
                else:
                    arguments = {
                        "xyz": (primitive.measure().xyz + [0, 0, 0.01]).tolist()
                    }
                motion = toolkit.execute_tool("move_to", arguments)
                assert not motion.is_error, motion.data
                evidence["burst_action"] = motion.data
            else:
                evidence["burst_action"] = {
                    "status": "already_terminated",
                    "native_success": toolkit.solved(),
                }
            write_json_atomic(args.output_dir / "ACTION_DONE.json", evidence)
        if args.release_file is not None:
            while not args.release_file.exists():
                time.sleep(0.5)
        evidence["final_state"] = toolkit._robot_state()
    evidence.update(passed=True, finished_at=time.time())
    write_json_atomic(args.output_dir / "verified.json", evidence)


def main() -> None:
    """Run explicitly; no simulator is imported by offline unit tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    get_robot_spec().add_cli_args(parser, use_dashboard=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("controls", "hold"), default="controls")
    parser.add_argument("--release-file", type=Path)
    parser.add_argument("--action-file", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        run(args)
    except BaseException as error:
        write_json_atomic(
            args.output_dir / "error.json",
            {
                "error": str(error),
                "type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "time": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()

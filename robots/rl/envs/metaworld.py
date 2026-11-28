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

"""Native MetaWorld scene, calibrated cameras, and fixed-horizon TCP control."""

import hashlib
import pickle
from typing import Annotated

import numpy as np
from pydantic import Field

from robots.rl import geometry
from robots.rl.geometry import Coordinates, grasp_axes
from robots.rl.perception import RecordedPerception
from robots.rl.toolkit import Vector3
from rpent.tools import ToolResult, tool

NATIVE_HORIZON = 500
TASK_DESCRIPTIONS = {
    "assembly-v3": "Pick up a nut and place it onto a peg",
    "basketball-v3": "Dunk the basketball into the basket",
    "bin-picking-v3": "Grasp the puck from one bin and place it into another bin",
    "box-close-v3": "Grasp the cover and close the box with it",
    "button-press-topdown-v3": "Press a button from the top",
    "button-press-topdown-wall-v3": "Bypass a wall and press a button from the top",
    "button-press-v3": "Press a button",
    "button-press-wall-v3": "Bypass a wall and press a button",
    "coffee-button-v3": "Push a button on the coffee machine",
    "coffee-pull-v3": "Pull a mug from a coffee machine",
    "coffee-push-v3": "Push a mug under a coffee machine",
    "dial-turn-v3": "Rotate a dial 180 degrees",
    "disassemble-v3": "Pick a nut out of a peg",
    "door-close-v3": "Close a door with a revolving joint",
    "door-lock-v3": "Lock the door by rotating the lock clockwise",
    "door-open-v3": "Open a door with a revolving joint",
    "door-unlock-v3": "Unlock the door by rotating the lock counter-clockwise",
    "hand-insert-v3": "Insert the gripper into a hole",
    "drawer-close-v3": "Push and close a drawer",
    "drawer-open-v3": "Open a drawer",
    "faucet-open-v3": "Rotate the faucet counter-clockwise",
    "faucet-close-v3": "Rotate the faucet clockwise",
    "hammer-v3": "Hammer a screw on the wall",
    "handle-press-side-v3": "Press a handle down sideways",
    "handle-press-v3": "Press a handle down",
    "handle-pull-side-v3": "Pull a handle up sideways",
    "handle-pull-v3": "Pull a handle up",
    "lever-pull-v3": "Pull a lever down 90 degrees",
    "peg-insert-side-v3": "Insert a peg sideways",
    "pick-place-wall-v3": "Pick a puck, bypass a wall and place the puck",
    "pick-out-of-hole-v3": "Pick up a puck from a hole",
    "reach-v3": "Reach a goal position",
    "push-back-v3": "Push the puck to a goal",
    "push-v3": "Push the puck to a goal",
    "pick-place-v3": "Pick and place a puck to a goal",
    "plate-slide-v3": "Slide a plate into a cabinet",
    "plate-slide-side-v3": "Slide a plate into a cabinet sideways",
    "plate-slide-back-v3": "Get a plate from the cabinet",
    "plate-slide-back-side-v3": "Get a plate from the cabinet sideways",
    "peg-unplug-side-v3": "Unplug a peg sideways",
    "soccer-v3": "Kick a soccer into the goal",
    "stick-push-v3": "Grasp a stick and push a box using the stick",
    "stick-pull-v3": "Grasp a stick and pull a box with the stick",
    "push-wall-v3": "Bypass a wall and push a puck to a goal",
    "reach-wall-v3": "Bypass a wall and reach a goal",
    "shelf-place-v3": "Pick and place a puck onto a shelf",
    "sweep-into-v3": "Sweep a puck into a hole",
    "sweep-v3": "Sweep a puck off the table",
    "window-open-v3": "Push and open a window",
    "window-close-v3": "Push and close a window",
}

CAMERAS = ("corner2", "gripperPOV")
SIZE = 1024
SERVO_GAIN = 0.15
TCP_TOLERANCE_M = 0.004
MAX_MOTION_STEPS = 100
STALL_STEPS = 15
STALL_PROGRESS_M = 0.002
MAX_GRIPPER_STEPS = 50
OPENING_TOLERANCE = 0.05
GRIPPER_SETTLE_STEPS = 3
GRIPPER_SETTLE_OPENING = 0.01
# Measured with an unloaded Sawyer gripper after 50 native steps at each end.
CLOSED_GAP_M = 0.029667591142120822
OPEN_GAP_M = 0.10034364850608656


def _digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def initial_scene(task_name: str, index: int, benchmark=None):
    """Reset the selected MT1 task and return its native environment and replay hashes."""
    if type(index) is not int or not 0 <= index < 50:
        raise ValueError("MetaWorld task index must be in [0, 49]")
    if benchmark is None:
        import metaworld

        benchmark = metaworld.MT1(task_name, seed=0)
    task = benchmark.train_tasks[index]
    env = benchmark.train_classes[task_name]()
    try:
        env.set_task(task)
        env.seed(index)
        env.reset(seed=index)
        if env.curr_path_length != 0 or env.max_path_length != NATIVE_HORIZON:
            raise RuntimeError("Unexpected MetaWorld native step budget")
        identity = {
            "task_index": index,
            "rand_vec_sha256": _digest(pickle.loads(task.data)["rand_vec"]),
            "init_qpos_sha256": _digest(env.data.qpos),
        }
        return env, identity
    except BaseException:
        env.close()
        raise


class Scene:
    """Own the native Sawyer physics and allocate its RGB-D renderer on first view."""

    def __init__(self, task_name, index, max_episode_steps=NATIVE_HORIZON):
        import mujoco

        self.max_episode_steps = min(int(max_episode_steps), NATIVE_HORIZON)
        self.env, self.identity = initial_scene(task_name, index)
        self.model, self.data = self.env.model, self.env.data
        self.model.cam("corner2").pos[:] = [0.75, 0.075, 0.7]
        self.model.vis.map.znear = 0.01
        self.model.vis.global_.offwidth = self.model.vis.global_.offheight = SIZE
        mujoco.mj_forward(self.model, self.data)
        self.renderer = None
        self.grip_command = -1.0
        self.success = False

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.env.close()

    def calibration(self):
        import mujoco

        m, d = self.model, self.data
        base = d.body("base").xpos
        table = d.body("tablelink")
        forward = table.xpos - base
        forward[2] = 0.0
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        boxes = np.flatnonzero(
            (m.geom_bodyid == table.id)
            & (m.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
            & ((m.geom_contype != 0) | (m.geom_conaffinity != 0))
        )
        top = max(
            d.geom_xpos[i, 2] + abs(d.geom_xmat[i].reshape(3, 3)[2]) @ m.geom_size[i]
            for i in boxes
        )
        hand = d.body("hand")
        hand_rotation = hand.xmat.reshape(3, 3)
        pads = self.pads()
        jaw_in_hand = hand_rotation.T @ (pads[0] - pads[1])
        approach_in_hand = hand_rotation.T @ (pads.mean(axis=0) - hand.xpos)
        return {
            "coordinate_calibration": {
                "base_xyz": base.tolist(),
                "base_rotation": np.column_stack(
                    (forward, np.cross(up, forward), up)
                ).tolist(),
                "table_top_xyz": [*table.xpos[:2], top],
                "tcp": "finger_pad_midpoint",
            },
            "jaw_slide_axes_site": [jaw_in_hand.tolist()],
            "wrist_to_site_in_site": approach_in_hand.tolist(),
        }

    def pads(self):
        return np.stack(
            [self.data.site(n).xpos for n in ("leftEndEffector", "rightEndEffector")]
        )

    def _opening_raw(self):
        pads = self.pads()
        gap = float(np.linalg.norm(pads[0] - pads[1]))
        return float(np.clip((gap - CLOSED_GAP_M) / (OPEN_GAP_M - CLOSED_GAP_M), 0, 1))

    def state(self):
        pads = self.pads()
        gap = float(np.linalg.norm(pads[0] - pads[1]))
        opening = float(
            np.clip((gap - CLOSED_GAP_M) / (OPEN_GAP_M - CLOSED_GAP_M), 0, 1)
        )
        steps = int(self.env.curr_path_length)
        return {
            "tcp_world": pads.mean(axis=0).tolist(),
            "hand_rotation": self.data.body("hand").xmat.reshape(3, 3).copy().tolist(),
            "opening": round(opening, 3),
            "finger_gap_m": round(gap, 5),
            "env_steps": steps,
            "terminated": self.success,
            "truncated": steps >= self.max_episode_steps and not self.success,
        }

    def advance(self, delta, check_cancelled=None):
        if check_cancelled is not None:
            check_cancelled()
        if self.success or self.env.curr_path_length >= self.max_episode_steps:
            return False
        _, _, _, _, info = self.env.step(
            np.r_[np.clip(delta, -1, 1), self.grip_command]
        )
        self.success |= bool(info["success"])
        return True

    def move_to(self, target, check_cancelled=None):
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Expected a finite TCP position")
        errors: list[float] = []
        reason: str | None = None
        for steps in range(MAX_MOTION_STEPS + 1):
            delta = target - self.pads().mean(axis=0)
            error = float(np.linalg.norm(delta))
            errors.append(error)
            if error <= TCP_TOLERANCE_M:
                break
            if self.success or self.env.curr_path_length >= self.max_episode_steps:
                reason = "terminated" if self.success else "truncated"
                break
            if (
                len(errors) > STALL_STEPS
                and errors[-1 - STALL_STEPS] - error < STALL_PROGRESS_M
            ):
                reason = "stalled"
                break
            if steps >= MAX_MOTION_STEPS:
                reason = "max_steps"
                break
            if not self.advance(
                SERVO_GAIN * delta / self.env.action_scale, check_cancelled
            ):
                reason = "terminated" if self.success else "truncated"
                break
        error = float(np.linalg.norm(target - self.pads().mean(axis=0)))
        if error <= TCP_TOLERANCE_M:
            reason = None
        state = self.state()
        return {
            "status": "reached" if reason is None else "partial",
            **({} if reason is None else {"reason": reason}),
            "position_error_m": round(error, 4),
            **state,
        }

    def set_gripper(self, opening, check_cancelled=None):
        if not 0 <= opening <= 1:
            raise ValueError("Expected opening in [0,1]")
        hold_tcp = self.pads().mean(axis=0)
        previous = self._opening_raw()
        still = 0
        reason: str | None = None
        for steps in range(MAX_GRIPPER_STEPS + 1):
            current = self._opening_raw()
            diff = float(opening - current)
            if steps > 0:
                still = (
                    still + 1 if abs(current - previous) < GRIPPER_SETTLE_OPENING else 0
                )
            previous = current
            if opening in (0, 1):
                self.grip_command = 1.0 - 2.0 * opening
            else:
                self.grip_command = float(-np.clip(diff * 5.0, -1, 1))
            if abs(diff) <= OPENING_TOLERANCE:
                break
            if self.success or self.env.curr_path_length >= self.max_episode_steps:
                reason = "terminated" if self.success else "truncated"
                break
            if still >= GRIPPER_SETTLE_STEPS and diff < 0:
                reason = "blocked"
                break
            if steps >= MAX_GRIPPER_STEPS:
                reason = "max_steps"
                break
            if not self.advance(
                SERVO_GAIN
                * (hold_tcp - self.pads().mean(axis=0))
                / self.env.action_scale,
                check_cancelled,
            ):
                reason = "terminated" if self.success else "truncated"
                break
        current = self._opening_raw()
        diff = float(opening - current)
        if abs(diff) <= OPENING_TOLERANCE:
            reason = None
        state = self.state()
        return {
            "status": "reached" if reason is None else "partial",
            **({} if reason is None else {"reason": reason}),
            "opening_error": round(diff, 3),
            **state,
        }

    def frame(self, camera):
        import mujoco

        if camera not in CAMERAS:
            raise ValueError("Unknown camera")
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=SIZE, width=SIZE)
        renderer = self.renderer
        renderer.update_scene(self.data, camera=camera)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        try:
            depth = renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
        # Optical convention: X right, Y down, Z into the image.
        cam = self.data.cam(camera)
        rotation = cam.xmat.reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        if camera == "corner2":
            rgb, depth = rgb[::-1, ::-1], depth[::-1, ::-1]
            rotation = rotation @ np.diag([-1.0, -1.0, 1.0])
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, cam.xpos
        f = SIZE / (2 * np.tan(np.deg2rad(self.model.cam(camera).fovy.item()) / 2))
        c = (SIZE - 1) / 2
        far = self.model.stat.extent * self.model.vis.map.zfar
        depth[depth >= far * (1 - 1e-5)] = np.nan
        return {
            "rgb": np.ascontiguousarray(rgb),
            "depth_m": np.ascontiguousarray(depth),
            "intrinsic_K": [[f, 0, c], [0, f, c], [0, 0, 1]],
            "extrinsic_cam2world": transform.tolist(),
        }


class LocalMetaWorldEnv:
    """One local Sawyer scene with the worker's task metadata."""

    def __init__(
        self, task_name, index, max_episode_steps=NATIVE_HORIZON, task_language=None
    ):
        self.task_language = task_language or TASK_DESCRIPTIONS[task_name]
        self.scene = Scene(task_name, index, max_episode_steps)
        self.identity = self.scene.identity
        self.last_info = {**self.identity, "init_state_id": index}

    def calibration(self):
        return self.scene.calibration()

    def state(self):
        return self.scene.state()

    def move_to(self, target, check_cancelled=None):
        return self.scene.move_to(target, check_cancelled)

    def set_gripper(self, opening, check_cancelled=None):
        return self.scene.set_gripper(opening, check_cancelled)

    def frame(self, camera):
        return self.scene.frame(camera)

    def close(self):
        self.scene.close()

    def get_task_language(self):
        return self.task_language


class MetaWorldRLPrimitives(RecordedPerception):
    """Position-only Sawyer control in episode coordinates."""

    cameras = CAMERAS

    def __init__(self, *, max_episode_steps=NATIVE_HORIZON, **kwargs):
        self._init_perception(
            max_episode_steps=min(int(max_episode_steps), NATIVE_HORIZON), **kwargs
        )
        control = self.env.calibration()
        self.coordinates = Coordinates.from_control(control)
        self.tool_rotation = grasp_axes(
            control["jaw_slide_axes_site"][0], control["wrist_to_site_in_site"]
        )
        self.identity = dict(self.env.identity)
        self._save_artifact("coordinates.json", self.coordinates.metadata(), None)
        self._save_artifact("initial_state.json", self.identity, None)

    def native_success(self):
        return self._success

    def _update_state(self, state):
        self.env_steps = int(state["env_steps"])
        self._success |= bool(state["terminated"])

    def robot_state(self):
        state = self.env.state()
        self._update_state(state)
        tcp = self.coordinates.points(state["tcp_world"])
        hand_rotation = np.asarray(state["hand_rotation"], dtype=np.float64)
        orientation = self.coordinates.orientation(hand_rotation @ self.tool_rotation)
        return {
            "pose": {
                "xyz": [round(float(v), 4) for v in tcp],
                "orientation_rpy": geometry.rpy_of(orientation),
            },
            "opening": round(float(state["opening"]), 3),
            "finger_gap_m": round(float(state["finger_gap_m"]), 5),
            "env_steps": self.env_steps,
        }

    def capture(self, camera):
        frame = self.env.frame(camera)
        rgb, depth = frame.pop("rgb"), frame.pop("depth_m")
        return (
            rgb,
            geometry.world_from_depth(depth, frame),
            {
                **frame,
                "raw_point_map_frame": "sim_world",
                "raw_point_map_unit": "m",
                "tool_rotation_to_native": self.tool_rotation.tolist(),
            },
        )

    def _action_result(self, state, partial_reason):
        self._update_state(state)
        data = {"status": state["status"], "env_steps": self.env_steps}
        for key in ("position_error_m", "opening_error"):
            if key in state:
                data[key] = state[key]
        if state["status"] != "reached":
            data["reason"] = state.get("reason") or (
                self._ended_reason() if self.ended else partial_reason
            )
        return ToolResult(data=data)

    @tool
    def move_to(self, xyz: Vector3) -> ToolResult:
        """Move the finger-pad midpoint to an absolute xyz target."""
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        state = self.env.move_to(
            self.coordinates.world_position(xyz), self._check_cancelled
        )
        return self._action_result(state, "max_steps")

    @tool
    def set_gripper(
        self, opening: Annotated[float, Field(ge=0.0, le=1.0)]
    ) -> ToolResult:
        """Servo the gripper to a normalized opening."""
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        state = self.env.set_gripper(opening, self._check_cancelled)
        return self._action_result(state, "blocked")

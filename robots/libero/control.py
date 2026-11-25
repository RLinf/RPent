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

"""Read native LIBERO grip-site poses without changing RLinf policy observations."""

from __future__ import annotations

import numpy as np

from rpent.robots.components.am_contracts import AMControlSpec


class ControlObservation:
    """Transparent native-env wrapper exposing a picklable robot-state property."""

    def __init__(self, env):
        self.env = env
        self._coordinate_calibration = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        """Invalidate calibration when the native scene resets."""
        self._coordinate_calibration = None
        return self.env.reset(*args, **kwargs)

    @staticmethod
    def _support_surface(native) -> tuple[str, np.ndarray]:
        """Measure the arena's table or floor from compiled collision geometry."""
        model, data = native.sim.model, native.sim.data
        try:
            arena = native.model.mujoco_arena
        except AttributeError as exc:
            raise ValueError(
                "general requires the compiled LIBERO arena model"
            ) from exc
        support = next(
            (
                (field, body)
                for field in (
                    "table_body",
                    "living_room_table_body",
                    "coffee_table_body",
                    "floor_body",
                )
                if (body := getattr(arena, field, None)) is not None
            ),
            None,
        )
        if support is None:
            raise ValueError("general requires a supported LIBERO support body")
        field, body = support
        kind = "floor" if field == "floor_body" else "table"
        root = model.body_name2id(body.get("name"))
        descendants = {root}
        for body in range(root + 1, model.nbody):
            if int(model.body_parentid[body]) in descendants:
                descendants.add(body)
        surfaces = []
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) not in descendants:
                continue
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            rotation = data.geom_xmat[geom].reshape(3, 3)
            point = data.geom_xpos[geom].copy()
            # MuJoCo box=6 and plane=0. A plane's size is not its thickness.
            geom_type = int(model.geom_type[geom])
            if geom_type == 6:
                if np.max(np.abs(rotation[2])) < 1 - 1e-6:
                    continue
                point[2] += np.abs(rotation[2]) @ model.geom_size[geom]
            elif kind != "floor" or geom_type != 0 or rotation[2, 2] < 1 - 1e-6:
                continue
            surfaces.append(point)
        if not surfaces:
            raise ValueError(
                "general requires measured LIBERO support collision boxes or an upward floor plane"
            )
        return kind, max(surfaces, key=lambda point: point[2])

    def _general_state(self, native, robot, arm) -> dict:
        """Read the installed base and support surface from compiled geometry."""
        sim = native.sim
        if self._coordinate_calibration is None:
            base = sim.model.body_name2id(robot.robot_model.root_body)
            kind, surface = self._support_surface(native)
            self._coordinate_calibration = {
                "base_xyz": sim.data.body_xpos[base].copy(),
                "base_rotation": sim.data.body_xmat[base].reshape(3, 3).copy(),
                "tcp": "grip_site",
                f"{kind}_top_xyz": np.asarray(surface, dtype=float).copy(),
            }
        return {"coordinate_calibration": self._coordinate_calibration}

    @property
    def rpent_control_state(self) -> dict:
        """Read MuJoCo's site frame and the active OSC controller configuration."""
        native = self.env
        while hasattr(native, "env"):
            native = native.env
        robot = native.robots[0]
        arm = robot.arms[0]
        controller = robot.part_controllers[arm]
        if controller.name != "OSC_POSE" or controller.input_type != "delta":
            raise ValueError("LIBERO RL requires the native delta OSC_POSE controller")
        site = robot.eef_site_id[arm]
        body = native.sim.model.body_name2id(robot.robot_model.eef_name[arm])
        site_rotation = native.sim.data.site_xmat[site].reshape(3, 3).copy()
        body_rotation = native.sim.data.body_xmat[body].reshape(3, 3).copy()
        origin = (
            np.eye(3)
            if controller.input_ref_frame == "world"
            else np.asarray(controller.origin_ori)
        )
        if origin.shape != (3, 3):
            raise ValueError("native OSC origin rotation is unavailable")
        jaw_axes = []
        for name in robot.gripper[arm].joints:
            joint = native.sim.model.joint_name2id(name)
            joint_body = native.sim.model.jnt_bodyid[joint]
            axis_world = (
                native.sim.data.body_xmat[joint_body].reshape(3, 3)
                @ native.sim.model.jnt_axis[joint]
            )
            jaw_axes.append(site_rotation.T @ axis_world)
        return {
            **self._general_state(native, robot, arm),
            "jaw_slide_axes_site": np.asarray(jaw_axes),
            "xyz": native.sim.data.site_xpos[site].copy(),
            "rotation": site_rotation,
            "body_to_site_rotation": body_rotation.T @ site_rotation,
            "wrist_to_site_in_site": site_rotation.T
            @ (native.sim.data.site_xpos[site] - native.sim.data.body_xpos[body]),
            "control_to_world_rotation": origin.copy(),
            "input_min": np.asarray(controller.input_min).copy(),
            "input_max": np.asarray(controller.input_max).copy(),
            "output_min": np.asarray(controller.output_min).copy(),
            "output_max": np.asarray(controller.output_max).copy(),
            "reference": "grip_site",
            "control_reference": controller.input_ref_frame,
        }


def wrap_factory(factory):
    """Wrap one existing RLinf factory; no simulator import or environment clone."""

    def create():
        return ControlObservation(factory())

    return create


def controlled_env_type():
    """Return a thin LiberoEnv subclass using its supported factory extension."""
    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    class ControlledLiberoEnv(LiberoEnv):
        def get_env_fns(self):
            return [wrap_factory(factory) for factory in super().get_env_fns()]

    return ControlledLiberoEnv


# Values use LIBERO's native OSC scaling and gripper convention. Model
# adapters restore this representation before handing actions to the env.
LIBERO_OSC = AMControlSpec(
    embodiment="libero_7d",
    action_space="libero.osc_pose.delta.v1",
    action_schema=[
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_axis_angle_x",
        "delta_axis_angle_y",
        "delta_axis_angle_z",
        "gripper",
    ],
    action_type=None,
)

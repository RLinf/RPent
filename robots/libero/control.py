# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Read native LIBERO grip-site poses without changing RLinf policy observations."""

from __future__ import annotations

import numpy as np


class ControlObservation:
    """Transparent native-env wrapper exposing a picklable robot-state property."""

    def __init__(self, env, *, world_cord="base", coordinate_calibration=False):
        self.env = env
        self.world_cord = world_cord
        self.coordinate_calibration = coordinate_calibration
        self._coordinate_calibration = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        """Invalidate calibration when the native scene resets."""
        self._coordinate_calibration = None
        return self.env.reset(*args, **kwargs)

    @staticmethod
    def _table_top(native) -> np.ndarray:
        """Measure the arena's support surface from compiled collision boxes."""
        model, data = native.sim.model, native.sim.data
        arena = native.mujoco_arena
        table = next(
            (
                body
                for field in (
                    "table_body",
                    "living_room_table_body",
                    "coffee_table_body",
                )
                if (body := getattr(arena, field, None)) is not None
            ),
            None,
        )
        if table is None:
            raise ValueError("general requires a supported LIBERO support-table body")
        root = model.body_name2id(table.get("name"))
        descendants = {root}
        for body in range(root + 1, model.nbody):
            if int(model.body_parentid[body]) in descendants:
                descendants.add(body)
        surfaces = []
        for geom in range(model.ngeom):
            # MuJoCo mjGEOM_BOX=6; LIBERO's support collision surfaces are boxes.
            if (
                int(model.geom_bodyid[geom]) not in descendants
                or int(model.geom_type[geom]) != 6
            ):
                continue
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            rotation = data.geom_xmat[geom].reshape(3, 3)
            if np.max(np.abs(rotation[2])) < 1 - 1e-6:
                continue
            point = data.geom_xpos[geom].copy()
            point[2] += np.abs(rotation[2]) @ model.geom_size[geom]
            surfaces.append(point)
        if not surfaces:
            raise ValueError("general requires measured LIBERO table collision boxes")
        return max(surfaces, key=lambda point: point[2])

    def _general_state(self, native, robot, arm) -> dict:
        """Read the mounted base, tabletop and actual pad midpoint from MuJoCo."""
        sim = native.sim
        if self._coordinate_calibration is None:
            base = sim.model.body_name2id(robot.robot_model.root_body)
            table = self._table_top(native)
            self._coordinate_calibration = {
                "base_xyz": sim.data.body_xpos[base].copy(),
                "base_rotation": sim.data.body_xmat[base].reshape(3, 3).copy(),
                "table_top_xyz": np.asarray(table, dtype=float).copy(),
            }
        gripper = robot.gripper[arm]
        pads = []
        for side in ("left_fingerpad", "right_fingerpad"):
            names = gripper.important_geoms[side]
            if len(names) != 1:
                raise ValueError("general requires one collision pad per Panda finger")
            pads.append(sim.data.geom_xpos[sim.model.geom_name2id(names[0])])
        return {
            "coordinate_calibration": self._coordinate_calibration,
            "pad_tcp_xyz": np.mean(pads, axis=0),
        }

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
            **(
                self._general_state(native, robot, arm)
                if self.world_cord == "general" or self.coordinate_calibration
                else {}
            ),
            "jaw_slide_axes_site": np.asarray(jaw_axes),
            "xyz": native.sim.data.site_xpos[site].copy(),
            "rotation": site_rotation,
            "body_to_site_rotation": body_rotation.T @ site_rotation,
            "control_to_world_rotation": origin.copy(),
            "input_min": np.asarray(controller.input_min).copy(),
            "input_max": np.asarray(controller.input_max).copy(),
            "output_min": np.asarray(controller.output_min).copy(),
            "output_max": np.asarray(controller.output_max).copy(),
            "reference": "grip_site",
            "control_reference": controller.input_ref_frame,
        }


def wrap_factory(factory, *, world_cord="base", coordinate_calibration=False):
    """Wrap one existing RLinf factory; no simulator import or environment clone."""

    def create():
        return ControlObservation(
            factory(),
            world_cord=world_cord,
            coordinate_calibration=coordinate_calibration,
        )

    return create


def controlled_env_type(*, world_cord="base", coordinate_calibration=False):
    """Return a thin LiberoEnv subclass using its supported factory extension."""
    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    class ControlledLiberoEnv(LiberoEnv):
        def get_env_fns(self):
            return [
                wrap_factory(
                    factory,
                    world_cord=world_cord,
                    coordinate_calibration=coordinate_calibration,
                )
                for factory in super().get_env_fns()
            ]

    return ControlledLiberoEnv

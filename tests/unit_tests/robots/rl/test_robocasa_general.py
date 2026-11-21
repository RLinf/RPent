# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""RoboCasa general coordinates, dynamic mount frame, lift, batch, 1024 and remote contracts."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from robots.rl import remote, robot_spec
from robots.rl.backends import RoboCasaGeneralPrimitives
from robots.rl.contracts import (
    argument_hint,
    validate_coordinates,
    validate_image_size,
    validate_options,
)
from robots.rl.prompt_bundle import system_prompt
from robots.rl.toolkit import RLToolkit
from robots.robocasa.env_server import RoboCasaEnvFacade
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.api_loop import ApiAgentLoop
from rpent.utils.rpc.http_rpc import HttpRpcServer
from tests.unit_tests.robots.rl.test_remote_contracts import Bridge

GRIPPER_DOWN = Rotation.from_euler("xyz", [np.pi, 0.0, 0.0]).as_matrix()


def _control_meta() -> dict:
    return {
        "gripper_max_gap": 0.08,
        "parts": {
            "right": {
                "input_min": [-1.0] * 6,
                "input_max": [1.0] * 6,
                "output_min": [-0.05] * 3 + [-0.5] * 3,
                "output_max": [0.05] * 3 + [0.5] * 3,
                "input_ref_frame": "base",
                "actuator_min": [-1.0] * 6,
                "actuator_max": [1.0] * 6,
            },
            "base": {
                "input_min": [-1.0] * 3,
                "input_max": [1.0] * 3,
                "output_min": [-1.0] * 3,
                "output_max": [1.0] * 3,
                "input_ref_frame": "base",
                "actuator_min": [-2.0, -4.0, -5.0],
                "actuator_max": [2.0, 4.0, 5.0],
            },
            "torso": {
                "input_min": [-1.0],
                "input_max": [1.0],
                "output_min": [-0.025],
                "output_max": [0.025],
                "input_ref_frame": "base",
                "actuator_min": [-0.025],
                "actuator_max": [0.025],
            },
        },
        "slices": {"right": [0, 6], "base": [7, 10], "torso": [10, 11]},
        "control_freq": 20,
    }


class FakeRoboCasaGeneralEnv:
    """Kinematic PandaOmron stub with real mount, site/pad offset, base, lift, and RGB-D."""

    def __init__(
        self,
        *,
        base_xyz=(2.0, -1.0, 0.0),
        base_yaw=np.pi / 3,
        lift_m=0.05,
        support_kind="table",
        support_z=0.90,
        pad_offset_site=(0.001, -0.0005, -0.0036),
        camera_size=1024,
        block_gap: float | None = None,
        stall_arm: bool = False,
        lift_max_m: float = 0.25,
        base_scale: float = 1.0,
        success_when=None,
    ):
        self.meta = _control_meta()
        self.camera_h = camera_size
        self.camera_w = camera_size
        self.base_xyz = np.asarray(base_xyz, dtype=np.float64)
        self.base_yaw = float(base_yaw)
        self.init_yaw = float(base_yaw)
        self.lift_m = float(lift_m)
        self.mount_z_offset = 0.70
        self.support_kind = support_kind
        self.support_z = float(support_z)
        self.pad_offset_site = np.asarray(pad_offset_site, dtype=np.float64)
        self.block_gap = block_gap
        self.stall_arm = stall_arm
        self.lift_max_m = float(lift_max_m)
        self.base_scale = float(base_scale)
        self.success_when = success_when

        # Initial arm pose in current mount frame: [0.25, -0.02, 0.55], pointing down.
        self._mount_arm_xyz = np.array([0.25, -0.02, 0.55], dtype=np.float64)
        self._mount_arm_rot = Rotation.from_matrix(GRIPPER_DOWN)
        self._update_world_arm_from_mount()

        self.gripper_qpos = np.array([0.04, -0.04], dtype=np.float64)
        self.actions: list[np.ndarray] = []
        self.depth_near = 0.02
        self.depth_far = 50.0
        self.invalid_pixels: set[tuple[int, int]] = set()
        self.far_box: tuple[int, int, int, int] | None = None
        self.fixed_world_point = self.mount_origin + self.mount_rotation @ np.array(
            [0.50, 0.15, 0.20]
        )
        self._client = SimpleNamespace(call=self._rpc_call)

    @property
    def mount_origin(self) -> np.ndarray:
        return self.base_xyz + np.array([0.0, 0.0, self.mount_z_offset + self.lift_m])

    @property
    def mount_rotation(self) -> np.ndarray:
        return Rotation.from_euler("z", self.base_yaw).as_matrix()

    def _update_world_arm_from_mount(self) -> None:
        self.site_xyz = self.mount_origin + self.mount_rotation @ self._mount_arm_xyz
        self.site_rotation = self.mount_rotation @ self._mount_arm_rot.as_matrix()

    def _update_mount_arm_from_world(self) -> None:
        self._mount_arm_xyz = self.mount_rotation.T @ (
            self.site_xyz - self.mount_origin
        )
        self._mount_arm_rot = Rotation.from_matrix(
            self.mount_rotation.T @ self.site_rotation
        )

    @property
    def pad_tcp_xyz(self) -> np.ndarray:
        return self.site_xyz + self.site_rotation @ self.pad_offset_site

    @property
    def eef_pos(self) -> np.ndarray:
        return self.site_xyz.copy()

    @property
    def eef_quat(self) -> np.ndarray:
        # Simulate PandaOmron's +90deg Z offset between eef body and grip_site.
        body_rot = self.site_rotation @ Rotation.from_euler("z", np.pi / 2).as_matrix()
        return Rotation.from_matrix(body_rot).as_quat()

    @property
    def current_raw_obs(self) -> dict:
        return {
            "robot0_base_pos": self.base_xyz.copy(),
            "robot0_base_quat": Rotation.from_matrix(self.mount_rotation).as_quat(),
        }

    def get_task_language(self) -> str:
        return "pick the mug from the counter and place it in the cabinet"

    def check_success(self) -> bool:
        return bool(self.success_when and self.success_when(self))

    def _resolve_cam(self, name: str) -> str:
        return {
            "agentview": "robot0_agentview_left",
            "agentview_right": "robot0_agentview_right",
            "wrist": "robot0_eye_in_hand",
        }.get(name, name)

    def _pixel_to_world_transform(self) -> np.ndarray:
        # Top-down pinhole camera located above +x/+z in mount frame so asymmetric pixels map cleanly.
        h, w = self.camera_h, self.camera_w
        fx = float(w)
        fy = float(h)
        cx = w * 0.5
        cy = h * 0.5
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        cam_in_mount = np.eye(4)
        # Camera looks along -z_mount with image +col = -y_mount, +row = -x_mount.
        cam_in_mount[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
        )
        cam_in_mount[:3, 3] = np.array([0.50, 0.15, 1.20])
        mount_in_world = np.eye(4)
        mount_in_world[:3, :3] = self.mount_rotation
        mount_in_world[:3, 3] = self.mount_origin
        cam_to_world = mount_in_world @ cam_in_mount
        K4 = np.eye(4)
        K4[:3, :3] = K
        return cam_to_world @ np.linalg.inv(K4)

    def render_camera(
        self,
        camera_name: str,
        height=None,
        width=None,
        depth=False,
        *,
        mask_invalid_depth=False,
    ):
        h = height or self.camera_h
        w = width or self.camera_w
        rgb = np.full((h, w, 3), 120, dtype=np.uint8)
        if not depth:
            return rgb
        # Default valid metric depth = 1.0 m everywhere inside frustum.
        d = np.full((h, w), 1.0, dtype=np.float64)
        if self.far_box is not None:
            r0, c0, r1, c1 = self.far_box
            d[r0:r1, c0:c1] = self.depth_far
        for r, c in self.invalid_pixels:
            d[r, c] = self.depth_far
        for (r, c), val in getattr(self, "custom_depth_pixels", {}).items():
            d[r, c] = val
        return rgb, d

    def get_camera_meta(self, camera_name: str, height=None, width=None) -> dict:
        h = height or self.camera_h
        w = width or self.camera_w
        return {
            "camera_name": self._resolve_cam(camera_name),
            "height": h,
            "width": w,
            "intrinsic": [
                [float(w), 0.0, w * 0.5],
                [0.0, float(h), h * 0.5],
                [0.0, 0.0, 1.0],
            ],
            "depth_near": self.depth_near,
            "depth_far": self.depth_far,
        }

    def _rpc_call(self, method: str, args=(), kwargs=None):
        kwargs = kwargs or {}
        if method == "env.control_meta":
            return self.meta
        if method == "env.control_frame":
            return {
                "arm_origin_rotation": self.mount_rotation.copy(),
                "base_theta": self.base_yaw - self.init_yaw,
            }
        if method == "env.coordinate_calibration":
            top_key = f"{self.support_kind}_top_xyz"
            return {
                "coordinate_calibration": {
                    "base_xyz": self.mount_origin.copy(),
                    "base_rotation": self.mount_rotation.copy(),
                    top_key: np.array(
                        [self.base_xyz[0], self.base_xyz[1], self.support_z]
                    ),
                },
                "support_surface_kind": self.support_kind,
                "support_surface_source": "counter_main"
                if self.support_kind == "table"
                else "floor_room_g0",
                "mobile_base_xyz": self.base_xyz.copy(),
                "mobile_base_rotation": self.mount_rotation.copy(),
                "lift_m": self.lift_m,
                "pad_tcp_xyz": self.pad_tcp_xyz.copy(),
                "site_xyz": self.site_xyz.copy(),
                "site_rotation": self.site_rotation.copy(),
                "jaw_slide_axes_site": np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            }
        if method == "env.get_camera_transform":
            return self._pixel_to_world_transform()
        raise KeyError(method)

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float64).copy()
        self.actions.append(a)
        dt = 1.0 / self.meta["control_freq"]
        if a[-1] > 0:
            # Base + torso mobility mode: arm tracks mount pose.
            base_cfg = self.meta["parts"]["base"]
            theta = self.base_yaw - self.init_yaw
            rot = Rotation.from_euler("z", -theta).as_matrix()
            low, high = (
                np.asarray(base_cfg["actuator_min"]),
                np.asarray(base_cfg["actuator_max"]),
            )
            vel = rot.T @ (rot @ a[7:10] * ((high - low) / 2) + (high + low) / 2)
            local_dp = np.array([vel[0] * dt, vel[1] * dt, 0.0]) * self.base_scale
            self.base_xyz = self.base_xyz + self.mount_rotation @ local_dp
            self.base_yaw += float(vel[2] * dt * self.base_scale)
            torso_cfg = self.meta["parts"]["torso"]
            t_low, t_high = torso_cfg["output_min"][0], torso_cfg["output_max"][0]
            lift_delta = float(
                (a[10] - torso_cfg["input_min"][0])
                / (torso_cfg["input_max"][0] - torso_cfg["input_min"][0])
                * (t_high - t_low)
                + t_low
            )
            self.lift_m = float(np.clip(self.lift_m + lift_delta, 0.0, self.lift_max_m))
            self._update_world_arm_from_mount()
        else:
            if not self.stall_arm:
                right_cfg = self.meta["parts"]["right"]
                low, high = (
                    np.asarray(right_cfg["output_min"]),
                    np.asarray(right_cfg["output_max"]),
                )
                in_low, in_high = (
                    np.asarray(right_cfg["input_min"]),
                    np.asarray(right_cfg["input_max"]),
                )
                local_delta = (a[:6] - in_low) / (in_high - in_low) * (high - low) + low
                dpos_world = self.mount_rotation @ local_delta[:3]
                drot_world = self.mount_rotation @ local_delta[3:6]
                self.site_xyz = self.site_xyz + dpos_world
                self.site_rotation = (
                    Rotation.from_rotvec(drot_world).as_matrix() @ self.site_rotation
                )
                self._update_mount_arm_from_world()

        # Gripper response: +1 closes (reduces gap), -1 opens (increases gap).
        gap = float(np.abs(self.gripper_qpos).sum())
        gap -= float(a[6]) * 0.012
        min_gap = 0.0 if self.block_gap is None else self.block_gap
        gap = float(np.clip(gap, min_gap, 0.08))
        self.gripper_qpos = np.array([gap / 2.0, -gap / 2.0], dtype=np.float64)
        return self.current_raw_obs, 0.0, self.check_success(), {}


class FakeSam3:
    def __init__(self, mask: np.ndarray):
        self._mask = mask

    def segment(self, image_bytes, *, text_prompt=None, point=None, min_score=0.2):
        return SimpleNamespace(
            found=True,
            score=0.95,
            box=None,
            mask=self._mask,
            reason=None,
        )


@pytest.fixture(autouse=True)
def offline_templates(monkeypatch):
    monkeypatch.setattr("rpent.tools.toolkit.substitute", lambda value: value)
    monkeypatch.setattr(
        "rpent.utils.templates.default_variables", lambda: {"output_dir": "/offline"}
    )


def _make_toolkit(
    tmp_path: Path,
    *,
    env=None,
    sam3=None,
    world_cord: str = "general",
    action_mode: str = "batch",
    image_size: int = 1024,
    **kwargs,
):
    env = env or FakeRoboCasaGeneralEnv(
        camera_size=1024 if world_cord == "general" else 256
    )
    toolkit = RLToolkit(
        runtime_kwargs={"env": env, "sam3_client": sam3, "max_episode_steps": 10000},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        observation_mode="adaptive",
        output_dir=tmp_path / "run",
        environment="robocasa",
        workflow="episode",
        world_cord=world_cord,
        action_mode=action_mode,
        image_size=image_size,
        **kwargs,
    )
    return toolkit, env


# ---------------------------------------------------------------------------
# 1. Option & rejection matrix across all 5 entrypoints
# ---------------------------------------------------------------------------


def test_robocasa_base_batch_rejected_across_all_entrypoints(tmp_path, monkeypatch):
    validate_coordinates("robocasa", "base")
    validate_coordinates("robocasa", "general")
    with pytest.raises(ValueError, match="LIBERO and RoboCasa only"):
        validate_coordinates("robotwin", "general")

    # Direct validator
    validate_options("robocasa", "episode", "single", 3, -1, -1, world_cord="base")
    validate_options("robocasa", "episode", "batch", 3, -1, -1, world_cord="general")
    validate_options("robocasa", "episode", "single", 3, 2, 1, world_cord="general")
    with pytest.raises(ValueError):
        validate_options("robocasa", "episode", "batch", 3, -1, -1, world_cord="base")
    with pytest.raises(ValueError):
        validate_options("robocasa", "episode", "single", 3, -1, 1, world_cord="base")

    # image_size gating
    validate_image_size("robocasa", "general", 512)
    validate_image_size("robocasa", "general", 256)
    validate_image_size("robocasa", "base", 1024)
    with pytest.raises(ValueError):
        validate_image_size("robocasa", "base", 256)

    # CLI _validate_args and _parse_config
    spec = robot_spec.get_robot_spec()
    bad_cli = argparse.Namespace(
        env="robocasa",
        workflow="episode",
        world_cord="base",
        action_mode="batch",
        max_actions_per_call=3,
        reasoning_history=-1,
        image_history=-1,
        perception_mode="single",
        max_observe_calls=8,
        observe_concurrency=8,
        image_size=1024,
        sam3_timeout_s=120.0,
        planner="api",
        dashboard=False,
        tui=False,
        memory_dir=None,
        memory_repo=None,
        observation_mode="adaptive",
        no_images=False,
        remote_toolkit_endpoint=None,
        env_endpoint=None,
        sam3_endpoint=None,
        task_name="OpenDrawer",
        seed=0,
        prompt="simple",
        output_dir=tmp_path,
        suite=None,
        task=None,
    )
    with pytest.raises(ValueError):
        spec.validate_args(bad_cli)
    with pytest.raises(ValueError):
        spec.parse_config(bad_cli)

    # Local RLToolkit
    with pytest.raises(ValueError):
        RLToolkit(
            runtime_kwargs={},
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "mem"),
            observation_mode="adaptive",
            output_dir=tmp_path / "should-not-create",
            environment="robocasa",
            workflow="episode",
            world_cord="base",
            action_mode="batch",
        )
    assert not (tmp_path / "should-not-create").exists()

    # Remote RPC options
    with pytest.raises(ValueError):
        remote.remote_options(
            {
                "environment": "robocasa",
                "task_name": "OpenDrawer",
                "seed": 0,
                "workflow": "episode",
                "world_cord": "base",
                "action_mode": "batch",
            }
        )

    # Batch launcher tools/rl/run_batch.py
    from tools.rl import run_batch as batch

    monkeypatch.setattr(batch, "resolve_run_options", lambda a: None)

    jobs = tmp_path / "jobs.json"
    jobs.write_text(
        json.dumps([{"environment": "robocasa", "task_name": "OpenDrawer", "seed": 0}])
    )
    batch_out = tmp_path / "batch_out"
    batch_out.mkdir()
    batch_args = argparse.Namespace(
        infrastructure_retries=1,
        episode_timeout=14400,
        start_interval=0,
        max_initializing_per_gpu=1,
        jobs=jobs,
        seeds=None,
        workflow="episode",
        world_cord="base",
        action_mode="batch",
        max_actions_per_call=3,
        reasoning_history=-1,
        image_history=-1,
        image_size=1024,
    )
    with pytest.raises(ValueError):
        batch.collect(batch_args, batch_out)
    assert not (batch_out / "manifest.json").exists()


# ---------------------------------------------------------------------------
# 2. Facade calibration: counter vs floor fallback & jaw slide axes
# ---------------------------------------------------------------------------


def _make_fake_facade(*, with_counter: bool, with_floor: bool = True):
    facade = object.__new__(RoboCasaEnvFacade)
    facade._support_surface_cache = None

    mount_pos = np.array([1.2, -0.4, 0.75])
    support_rot = Rotation.from_euler("z", np.pi / 2).as_matrix()
    site_rot = support_rot @ GRIPPER_DOWN
    site_pos = mount_pos + support_rot @ np.array([0.25, 0.0, 0.55])
    pad_l = site_pos + site_rot @ np.array([0.04, 0.0, -0.0036])
    pad_r = site_pos + site_rot @ np.array([-0.04, 0.0, -0.0036])

    geom_names = ["left_pad_g", "right_pad_g"]
    geom_xpos = [pad_l, pad_r]
    geom_xmat = [np.eye(3).reshape(-1), np.eye(3).reshape(-1)]
    geom_type = [6, 6]
    geom_size = [np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01])]
    geom_bodyid = [3, 3]
    geom_contype = [1, 1]
    geom_conaffinity = [1, 1]
    if with_floor:
        geom_names.append("floor_room_g0")
        geom_xpos.append(np.array([0.0, 0.0, -0.05]))
        geom_xmat.append(np.eye(3).reshape(-1))
        geom_type.append(6)  # box with half-height 0.05 -> top at z = 0.0
        geom_size.append(np.array([5.0, 5.0, 0.05]))
        geom_bodyid.append(0)
        geom_contype.append(1)
        geom_conaffinity.append(1)

    body_names = ["world", "mobilebase0_support", "mobilebase0_base", "hand"]
    body_xpos = [
        np.zeros(3),
        mount_pos,
        np.array([1.2, -0.4, 0.0]),
        site_pos,
    ]
    body_xmat = [
        np.eye(3).reshape(-1),
        support_rot.reshape(-1),
        support_rot.reshape(-1),
        site_rot.reshape(-1),
    ]

    model = SimpleNamespace(
        nbody=len(body_names),
        ngeom=len(geom_names),
        body_parentid=np.array([0, 0, 0, 1]),
        site_bodyid=np.array([1, 3]),
        geom_bodyid=np.array(geom_bodyid),
        geom_type=np.array(geom_type),
        geom_size=np.array(geom_size),
        geom_contype=np.array(geom_contype),
        geom_conaffinity=np.array(geom_conaffinity),
        jnt_bodyid=np.array([3, 3]),
        jnt_axis=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        site_name2id=lambda name: {"mobilebase0_center": 0, "grip_site": 1}[name],
        body_name2id=lambda name: body_names.index(name),
        geom_name2id=lambda name: geom_names.index(name),
        joint_name2id=lambda name: {"finger_joint1": 0, "finger_joint2": 1}[name],
        geom_id2name=lambda idx: geom_names[idx],
        body_id2name=lambda idx: body_names[idx],
        get_joint_qpos_addr=lambda name: 0,
    )
    data = SimpleNamespace(
        site_xpos=np.array([mount_pos, site_pos]),
        site_xmat=np.array([support_rot.reshape(-1), site_rot.reshape(-1)]),
        body_xpos=np.array(body_xpos),
        body_xmat=np.array(body_xmat),
        geom_xpos=np.array(geom_xpos),
        geom_xmat=np.array(geom_xmat),
        qpos=np.array([0.05]),
    )
    robot = SimpleNamespace(
        eef_site_id={"right": 1},
        gripper={
            "right": SimpleNamespace(
                important_geoms={
                    "left_fingerpad": ["left_pad_g"],
                    "right_fingerpad": ["right_pad_g"],
                },
                joints=["finger_joint1", "finger_joint2"],
            )
        },
    )
    counter = (
        SimpleNamespace(
            name="counter_main_main_group",
            get_ext_sites=lambda relative=False: [
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.array([1.2, -0.4, 0.92]),
            ],
        )
        if with_counter
        else None
    )
    facade.env = SimpleNamespace(
        sim=SimpleNamespace(model=model, data=data),
        robots=[robot],
        counter=counter,
        model=SimpleNamespace(mujoco_arena=None),
        reset=lambda: {},
    )
    return facade


def test_floor_fallback_ignores_toaster_slot_collision_box():
    facade = _make_fake_facade(with_counter=False)
    sim = facade.env.sim
    # CloseBlenderLid seed 1 has toaster slot bottoms above the room floor.
    sim.model.geom_id2name = [
        "toaster_main_group_slotR_floor",
        "pad",
        "floor_room_g0",
    ].__getitem__
    sim.model.geom_type[0] = 6
    sim.model.geom_contype[0] = 1
    sim.data.geom_xpos[0, 2] = 1.026
    kind, point, source = facade._support_surface()
    assert (kind, source) == ("floor", "floor_room_g0")
    assert point[2] == pytest.approx(0)


def test_facade_calibration_supports_counter_and_floor_fallback():
    from robots.rl.coordinates import Coordinates

    counter_facade = _make_fake_facade(with_counter=True)
    cal = counter_facade.coordinate_calibration()
    frame = Coordinates.from_control("general", cal)
    assert cal["support_surface_kind"] == "table"
    assert cal["support_surface_source"] == "counter_main_main_group"
    assert frame.description()["table_height_above_base_m"] == pytest.approx(0.17)
    np.testing.assert_allclose(
        cal["jaw_slide_axes_site"], [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], atol=1e-9
    )

    # Non-counter task (e.g. NavigateKitchen / CloseFridge / OpenDrawer) falls back to floor.
    floor_facade = _make_fake_facade(with_counter=False, with_floor=True)
    cal_floor = floor_facade.coordinate_calibration()
    frame_floor = Coordinates.from_control("general", cal_floor)
    assert cal_floor["support_surface_kind"] == "floor"
    assert cal_floor["support_surface_source"] == "floor_room_g0"
    desc = frame_floor.description()
    assert desc["support_surface"] == "floor"
    assert desc["table_height_above_base_m"] is None
    assert desc["floor_height_above_base_m"] == pytest.approx(-0.75)

    # Cache invalidation on reset.
    floor_facade.env.counter = counter_facade.env.counter
    assert floor_facade.coordinate_calibration()["support_surface_kind"] == "floor"
    floor_facade.reset()
    assert floor_facade.coordinate_calibration()["support_surface_kind"] == "table"

    # Upward collision plane (geom_type=0) fallback when no named floor box exists.
    plane_facade = _make_fake_facade(with_counter=False, with_floor=True)
    plane_facade.env.sim.model.geom_type[2] = 0
    plane_facade.env.sim.model.geom_id2name = lambda idx: [
        "left_pad_g",
        "right_pad_g",
        "ground_plane",
    ][idx]
    plane_facade.env.sim.data.geom_xpos[2] = np.array([0.0, 0.0, 0.02])
    cal_plane = plane_facade.coordinate_calibration()
    assert cal_plane["support_surface_kind"] == "floor"
    assert cal_plane["support_surface_source"] == "ground_plane"
    assert Coordinates.from_control("general", cal_plane).description()[
        "floor_height_above_base_m"
    ] == pytest.approx(-0.73)

    # Missing both counter and floor fails explicitly.
    broken = _make_fake_facade(with_counter=False, with_floor=False)
    with pytest.raises(ValueError, match="counter or measured floor"):
        broken.coordinate_calibration()


# ---------------------------------------------------------------------------
# 3. Init order, 90deg site alignment, rotation-only move_to & TCP offset
# ---------------------------------------------------------------------------


def test_rotation_only_move_to_preserves_pad_tcp_and_aligns_site_axes(tmp_path):
    env = FakeRoboCasaGeneralEnv(
        base_xyz=(3.1, -1.7, 0.0),
        base_yaw=np.pi / 2,
        lift_m=0.04,
        pad_offset_site=(0.002, -0.001, -0.0036),
        camera_size=64,
    )
    tk, _ = _make_toolkit(tmp_path, env=env, action_mode="single", image_size=1024)
    p = tk._primitives
    assert isinstance(p, RoboCasaGeneralPrimitives)

    # Double-precision transform round-trip < 1e-9 m.
    sample = np.array([0.41, -0.18, 0.33])
    np.testing.assert_allclose(
        p.coordinates.points(p.coordinates.world_position(sample)), sample, atol=1e-9
    )

    # Initial state hides simulator-world base pose and old quaternion field.
    init_state = tk.initial_observation().data["state"]
    assert "orientation_rpy" in init_state["pose"]
    assert "quaternion_wxyz" not in init_state["pose"]
    assert "base_xyz" not in init_state
    assert "base_quaternion_xyzw" not in init_state

    # Immediately execute rotation-only move_to right after connect (verifies init order,
    # 90deg site vs eef_quat alignment, and target-rotation pad-TCP compensation).
    tcp_before = env.pad_tcp_xyz.copy()
    target_rpy = [float(np.pi), 0.0, float(np.pi / 4)]
    result = tk.execute_tool("move_to", {"orientation_rpy": target_rpy})
    assert result.data["status"] == "reached"
    assert result.data["position_error_m"] < 0.004
    assert result.data["rotation_error_rad"] < 0.02
    assert np.linalg.norm(env.pad_tcp_xyz - tcp_before) < 0.004
    expected_world_rot = p.coordinates.world_orientation(
        Rotation.from_euler("xyz", target_rpy).as_matrix()
    )
    rot_err = Rotation.from_matrix(expected_world_rot @ env.site_rotation.T).magnitude()
    assert rot_err < 0.02
    assert "quaternion_wxyz" not in result.data


# ---------------------------------------------------------------------------
# 4. Dynamic mount frame after move_base & move_lift + coordinate_step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mask_shape", ["rectangle", "triangle"])
def test_dynamic_mount_frame_updates_heights_and_historical_back_project(
    tmp_path, mask_shape
):
    env = FakeRoboCasaGeneralEnv(
        base_xyz=(1.5, 0.5, 0.0),
        base_yaw=0.0,
        lift_m=0.05,
        support_z=0.90,
        lift_max_m=0.12,  # forces partial lift when requesting +0.20 m
        camera_size=64,
    )
    if mask_shape == "triangle":
        rows, cols = np.indices((64, 64))
        mask = (rows >= 6) & (cols >= 6) & (rows + cols < 58)
    else:
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:26, 20:36] = True
    tk, _ = _make_toolkit(tmp_path, env=env, sam3=FakeSam3(mask), action_mode="batch")

    # Step 0 perception
    seg0 = tk.execute_tool(
        "segment", {"prompt": "mug", "camera": "agentview", "step": 0}
    )
    assert seg0.data["step"] == 0
    assert seg0.data["coordinate_step"] == 0
    mask_id = seg0.data["mask_id"]

    bp0_pixel = tk.execute_tool(
        "back_project", {"pixel": [18, 28], "camera": "agentview", "step": 0}
    )
    bp0_mask = tk.execute_tool("back_project", {"mask_id": mask_id})
    assert bp0_pixel.data["step"] == 0 and bp0_pixel.data["coordinate_step"] == 0
    assert bp0_mask.data["step"] == 0 and bp0_mask.data["coordinate_step"] == 0

    coords_0 = tk.state.load("step_coordinates.json", step=0)
    origin_0 = np.asarray(coords_0["origin_in_sim_world"])
    rot_0 = np.asarray(coords_0["rotation_to_sim_world"])
    world_pixel_0 = origin_0 + rot_0 @ np.asarray(bp0_pixel.data["world_xyz"])
    world_mask_0 = origin_0 + rot_0 @ np.asarray(bp0_mask.data["world_xyz"])
    native_points = tk.state.load("agentview_world.npz", step=0)[mask]
    np.testing.assert_allclose(
        world_mask_0, np.median(native_points, axis=0), atol=1e-3
    )
    np.testing.assert_allclose(bp0_mask.data["world_xyz"], seg0.data["world_xyz"])

    # Move base (rotate + translate)
    moved = tk.execute_tool(
        "move_base", {"vx": 0.2, "vy": -0.1, "yaw_rate": 0.5, "duration_s": 1.6}
    )
    assert moved.data["status"] == "commanded"
    assert "displacement_world_m" not in moved.data
    assert "displacement_initial_base_m" in moved.data
    assert abs(moved.data["yaw_change_rad"]) > 0.3

    # Lift with joint limit hit: requested +0.20 m, clamped at lift_max_m=0.12 (+0.07 m actual)
    before_table_h = moved.data["coordinates"]["table_height_above_base_m"]
    lifted = tk.execute_tool("move_lift", {"delta_m": 0.20, "duration_s": 2.0})
    assert lifted.data["status"] == "partial"
    assert lifted.data["reason"] == "blocked_or_joint_limit"
    actual_dz = lifted.data["mount_height_change_m"]
    assert 0.06 < actual_dz < 0.08
    after_table_h = lifted.data["coordinates"]["table_height_above_base_m"]
    assert after_table_h - before_table_h == pytest.approx(-actual_dz, abs=1e-3)

    # Re-query the exact same step=0 pixel and mask_id after base rotation + lift
    latest = tk.state.latest_step
    assert latest == 2
    bp2_pixel = tk.execute_tool(
        "back_project", {"pixel": [18, 28], "camera": "agentview", "step": 0}
    )
    bp2_mask = tk.execute_tool("back_project", {"mask_id": mask_id})
    assert bp2_pixel.data["step"] == 0 and bp2_pixel.data["coordinate_step"] == latest
    assert bp2_mask.data["step"] == 0 and bp2_mask.data["coordinate_step"] == latest
    assert not np.allclose(bp2_pixel.data["world_xyz"], bp0_pixel.data["world_xyz"])

    coords_2 = tk.state.load("step_coordinates.json", step=latest)
    origin_2 = np.asarray(coords_2["origin_in_sim_world"])
    rot_2 = np.asarray(coords_2["rotation_to_sim_world"])
    world_pixel_2 = origin_2 + rot_2 @ np.asarray(bp2_pixel.data["world_xyz"])
    world_mask_2 = origin_2 + rot_2 @ np.asarray(bp2_mask.data["world_xyz"])
    np.testing.assert_allclose(world_pixel_2, world_pixel_0, atol=1e-3)
    np.testing.assert_allclose(world_mask_2, world_mask_0, atol=1e-3)
    seg2 = tk.execute_tool(
        "segment", {"prompt": "mug", "camera": "agentview", "step": 0}
    )
    assert seg2.data["coordinate_step"] == latest
    np.testing.assert_allclose(seg2.data["world_xyz"], bp2_mask.data["world_xyz"])
    low, high = np.percentile((native_points - origin_2) @ rot_2, [5, 95], axis=0)
    np.testing.assert_allclose(seg2.data["size_xyz_m"], high - low, atol=1e-3)
    band = [low[2] - 0.001, high[2] + 0.001]
    filtered = tk.execute_tool("back_project", {"mask_id": mask_id, "z_range_m": band})
    assert filtered.data["world_xyz"] == bp2_mask.data["world_xyz"]
    assert tk.execute_tool(
        "back_project",
        {"mask_id": mask_id, "z_range_m": [band[1] + 0.1, band[1] + 0.2]},
    ).is_error


@pytest.mark.parametrize("defect", ["missing_region", "short_sites", "invalid_xyz"])
def test_uncalibrated_counter_uses_measured_floor(defect):
    facade = _make_fake_facade(with_counter=True)

    def ext_sites(*, relative):
        if defect == "missing_region":
            raise ValueError  # Fixture.get_ext_sites without a main or bbox region
        if defect == "short_sites":
            return []
        return [None, None, None, [None, "bad", 1]]

    facade.env.counter.get_ext_sites = ext_sites
    calibration = facade.coordinate_calibration()
    assert calibration["support_surface_source"] == "floor_room_g0"
    assert calibration["coordinate_calibration"]["floor_top_xyz"][2] == pytest.approx(0)


@pytest.mark.parametrize("channel_dim", [False, True])
def test_raw_depth_endpoints_filtered_before_back_projection(
    tmp_path, monkeypatch, channel_dim
):
    from robots.robocasa.env_client import RoboCasaEnvClient

    env = FakeRoboCasaGeneralEnv(camera_size=8)
    env.depth_near, env.depth_far = 0.001, 50.0
    raw = np.full((8, 8), 0.9, dtype=np.float32)
    raw[-1, :6] = [1.0, np.nan, np.inf, -np.inf, 0.0, 1.01]
    raw[-1, 6:] = np.float32(0.9999914765357971)  # native EGL skybox depth
    ids = np.full((8, 8, 2), [5, 252], dtype=np.int32)
    ids[-1, 6] = [-1, -1]  # adjacent geometry at the same depth remains valid
    if channel_dim:
        raw = raw[..., None]
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    facade = object.__new__(RoboCasaEnvFacade)

    def render(*, width, height, camera_name, depth=False, segmentation=False):
        if segmentation:
            return ids
        assert depth
        return rgb, raw

    facade.env = SimpleNamespace(sim=SimpleNamespace(render=render))
    camera_utils = ModuleType("robosuite.utils.camera_utils")

    def real_depth(sim, buffer):
        assert np.all((buffer >= 0) & (buffer <= 1))
        return env.depth_near / (1.0 - buffer * (1.0 - env.depth_near / env.depth_far))

    camera_utils.get_real_depth_map = real_depth
    suite = ModuleType("robosuite")
    suite.utils = ModuleType("robosuite.utils")
    suite.utils.camera_utils = camera_utils
    for module in (suite, suite.utils, camera_utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    client = object.__new__(RoboCasaEnvClient)
    client.camera_h = client.camera_w = 8
    client._client = SimpleNamespace(
        call=lambda method, *, kwargs, timeout_s: facade.render_camera(**kwargs)
    )
    env.render_camera = client.render_camera
    _, legacy = client.render_camera("agentview", depth=True)
    assert np.isfinite(legacy).all()
    assert legacy[0, 0] == pytest.approx(49.932194)
    tk, _ = _make_toolkit(tmp_path, env=env)
    world = tk.state.load("agentview_world.npz", step=0)
    assert np.isnan(world[0, :7]).all()
    assert np.isfinite(world[0, 7]).all()
    for col in (0, 6):
        bad = tk.execute_tool("back_project", {"pixel": [0, col], "step": 0})
        assert "insufficient depth" in bad.data["error"]


@pytest.mark.parametrize(
    ("vx", "yaw_rate", "slip"),
    [
        (0.2, 0.0, "reverse"),
        (0.0, -0.5, "reverse"),
        (0.2, 0.0, "sideways"),
        (0.35, -0.5, "none"),
    ],
)
def test_base_tracks_signed_motion_in_initial_frame(tmp_path, vx, yaw_rate, slip):
    env = FakeRoboCasaGeneralEnv(
        camera_size=16, base_scale=-1 if slip == "reverse" else 1
    )
    if slip == "sideways":
        step = env.step

        def slide(action):
            before = env.base_xyz.copy()
            result = step(action)
            env.base_xyz = before + Rotation.from_euler("z", np.pi / 2).apply(
                env.base_xyz - before
            )
            env._update_world_arm_from_mount()
            return result

        env.step = slide
    tk, _ = _make_toolkit(tmp_path, env=env)
    result = tk.execute_tool(
        "move_base",
        {"vx": vx, "yaw_rate": yaw_rate, "duration_s": 2 if slip == "none" else 1},
    ).data
    if slip == "none":
        assert result["status"] == "commanded"
        assert abs(result["translation_shortfall_m"]) < 0.01
    else:
        assert (result["status"], result["reason"]) == ("partial", "limited_motion")
        if vx:
            expected_shortfall = 0.38 if slip == "reverse" else 0.19
            assert result["translation_shortfall_m"] == pytest.approx(
                expected_shortfall
            )
        else:
            assert result["yaw_shortfall_rad"] == pytest.approx(0.95)


@pytest.mark.parametrize("rebound", [0.007, 0.00404, 0.0039])
def test_lift_judges_unrounded_height_after_stop(tmp_path, rebound):
    env = FakeRoboCasaGeneralEnv(camera_size=16)
    step = env.step

    def rebound_on_stop(action):
        result = step(action)
        if action[-1] > 0 and abs(action[10]) < 1e-9:
            env.lift_m += rebound
            env._update_world_arm_from_mount()
        return result

    env.step = rebound_on_stop
    tk, _ = _make_toolkit(tmp_path, env=env)
    result = tk.execute_tool("move_lift", {"delta_m": 0.05}).data
    assert result["height_error_m"] == pytest.approx(round(-rebound, 4))
    assert result["status"] == ("partial" if rebound > 0.004 else "reached")
    assert (result["reason"] == "completed") == (rebound <= 0.004)


def test_closing_gripper_corrects_tcp_drift(tmp_path):
    env = FakeRoboCasaGeneralEnv(camera_size=16)
    step = env.step
    drift = Rotation.from_euler("z", 0.003).as_matrix()

    def disturb_arm(action):
        result = step(action)
        env.site_xyz[0] += 0.001
        env.site_rotation = drift @ env.site_rotation
        env._update_mount_arm_from_world()
        return result

    env.step = disturb_arm
    tk, _ = _make_toolkit(tmp_path, env=env, action_mode="single")
    tcp, rotation = env.pad_tcp_xyz.copy(), env.site_rotation.copy()
    result = tk.execute_tool("set_gripper", {"opening": 0}).data
    assert result["status"] == "reached"
    assert np.linalg.norm(env.pad_tcp_xyz - tcp) < 0.004
    assert Rotation.from_matrix(env.site_rotation @ rotation.T).magnitude() < 0.02


# ---------------------------------------------------------------------------
# 5. Invalid depth / far-clipping plane masking
# ---------------------------------------------------------------------------


def test_invalid_and_far_clipping_depth_are_excluded_from_geometry(tmp_path):
    env = FakeRoboCasaGeneralEnv(camera_size=64)
    env.invalid_pixels.add((5, 7))
    env.custom_depth_pixels = {
        (6, 7): float("nan"),
        (7, 7): env.depth_near,
    }
    # Right half of the mask lies on the far clipping plane; left half is valid foreground.
    env.far_box = (20, 30, 36, 40)
    mixed_mask = np.zeros((64, 64), dtype=bool)
    mixed_mask[20:36, 20:40] = True
    fg_only_mask = np.zeros((64, 64), dtype=bool)
    fg_only_mask[20:36, 20:30] = True

    tk_mixed, _ = _make_toolkit(
        tmp_path / "mixed", env=env, sam3=FakeSam3(mixed_mask), action_mode="batch"
    )
    tk_fg, _ = _make_toolkit(
        tmp_path / "fg", env=env, sam3=FakeSam3(fg_only_mask), action_mode="batch"
    )

    for bad_pixel in ([5, 7], [6, 7], [7, 7]):
        bg_pixel = tk_mixed.execute_tool(
            "back_project", {"pixel": bad_pixel, "camera": "agentview", "step": 0}
        )
        assert bg_pixel.is_error
        assert "insufficient depth" in bg_pixel.data["error"]

    seg_mixed = tk_mixed.execute_tool(
        "segment", {"prompt": "cup", "camera": "agentview", "step": 0}
    )
    seg_fg = tk_fg.execute_tool(
        "segment", {"prompt": "cup", "camera": "agentview", "step": 0}
    )
    for key in ("world_xyz", "top_z_m", "size_xyz_m"):
        assert seg_mixed.data[key] == pytest.approx(seg_fg.data[key], abs=1e-4)


# ---------------------------------------------------------------------------
# 6. Native 1024 rendering + 1024/512/256 off-center point & edge box scaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("image_size", [1024, 512, 256])
def test_native_1024_and_resized_offcenter_pixel_and_edge_box(tmp_path, image_size):
    env = FakeRoboCasaGeneralEnv(camera_size=1024)
    tk, _ = _make_toolkit(tmp_path, env=env, action_mode="batch", image_size=image_size)
    view = tk.execute_tool("view_env", {"camera": "all"})
    assert len(view.images) == 3
    for info, raw_bytes in zip(view.data["images"], view.images):
        assert info["size"] == [image_size, image_size]
        with Image.open(io.BytesIO(raw_bytes)) as img:
            assert img.size == (image_size, image_size)

    # Check recorded native world map is 1024x1024 across all three cameras.
    for cam in ("agentview", "agentview_right", "wrist"):
        assert tk.state.load(f"{cam}_world.npz", step=0).shape == (1024, 1024, 3)

    # Off-center asymmetric pixel: (row=25%, col=75%) detects flip or row/col swap immediately.
    pixel = [image_size // 4, (3 * image_size) // 4]
    bp = tk.execute_tool(
        "back_project", {"pixel": pixel, "camera": "agentview", "step": 0}
    )
    assert not bp.is_error
    # At depth=1.0m: x_cam=(col-cx)/fx=+0.25, y_cam=(row-cy)/fy=-0.25
    # With cam_in_mount R=[[0,-1,0],[-1,0,0],[0,0,-1]], t=[0.50,0.15,1.20]:
    # p_mount = [0.50 - y_cam, 0.15 - x_cam, 1.20 - 1.0] = [0.75, -0.10, 0.20].
    np.testing.assert_allclose(bp.data["world_xyz"], [0.75, -0.10, 0.20], atol=5e-3)

    # Edge-touching half-open box reaching image_size must pass schema & scale cleanly.
    edge_box = [image_size - 32, image_size - 32, image_size, image_size]
    bp_box = tk.execute_tool(
        "back_project", {"box": edge_box, "camera": "agentview", "step": 0}
    )
    assert not bp_box.is_error


def test_robot_spec_selects_1024_for_general_and_256_for_base(monkeypatch, tmp_path):
    spawned: list[list[str]] = []

    def fake_spawn(
        args, output_dir, name, script, script_args, env_overrides=None, python=None
    ):
        spawned.append(script_args)
        return None, SimpleNamespace()

    monkeypatch.setattr(robot_spec, "_spawn_server", fake_spawn)
    for world_cord, expected_flag in (("base", False), ("general", True)):
        args = argparse.Namespace(
            env="robocasa",
            env_endpoint=None,
            sim_python=None,
            task_name="OpenDrawer",
            seed=0,
            split="target",
            task_config="demo_randomized",
            max_episode_steps=10000,
            assets_path=None,
            world_cord=world_cord,
        )
        robot_spec._start_env(args, tmp_path)
        assert ("--camera-h" in spawned[-1]) is expected_flag
        if expected_flag:
            assert spawned[-1][spawned[-1].index("--camera-h") + 1] == "1024"
            assert spawned[-1][spawned[-1].index("--camera-w") + 1] == "1024"

    connected_meta: list[dict] = []

    class FakeClient:
        def __init__(self, rpc, *, expected_meta):
            connected_meta.append(expected_meta)

    monkeypatch.setattr("robots.robocasa.env_client.RoboCasaEnvClient", FakeClient)
    for world_cord, size in (("base", 256), ("general", 1024)):
        args = argparse.Namespace(
            env="robocasa",
            task_name="OpenDrawer",
            split="target",
            seed=0,
            max_episode_steps=10000,
            world_cord=world_cord,
        )
        robot_spec._connect_env(args, SimpleNamespace())
        assert connected_meta[-1]["camera_h"] == size
        assert connected_meta[-1]["camera_w"] == size


# ---------------------------------------------------------------------------
# 7. Batch execute_actions, gripper hold, blocked & stalled stops
# ---------------------------------------------------------------------------


def test_batch_tools_blocked_gripper_hold_and_stalled_early_stop(tmp_path):
    env = FakeRoboCasaGeneralEnv(block_gap=0.036, camera_size=32)
    tk, _ = _make_toolkit(tmp_path / "blocked", env=env, action_mode="batch")
    assert {tool.name for tool in tk.list_tools()} == {
        "view_env",
        "segment",
        "back_project",
        "execute_actions",
        "move_base",
        "move_lift",
        "finish",
    }

    # Batch stops early on blocked gripper; 3rd item must not run.
    res = tk.execute_tool(
        "execute_actions",
        {
            "actions": [
                {"action": "move_to", "xyz": [0.28, -0.02, 0.55]},
                {"action": "set_gripper", "opening": 0.0},
                {"action": "move_to", "xyz": [0.28, -0.02, 0.65]},
            ]
        },
    )
    assert res.data["status"] == "stopped"
    assert res.data["executed_count"] == 2
    assert res.data["stopped_at"] == 1
    assert res.data["stop_reason"] == "blocked"
    # Gripper hold target stays 0.0 (does not relax to measured ~0.45).
    assert tk._primitives._hold_opening == 0.0
    # Next lift move continues commanding closing torque (> 0) on gripper.
    before_steps = len(env.actions)
    lifted = tk.execute_tool(
        "execute_actions",
        {"actions": [{"action": "move_to", "xyz": [0.28, -0.02, 0.60]}]},
    )
    assert lifted.data["status"] == "completed"
    assert all(a[6] > 0.5 for a in env.actions[before_steps:])

    # Stalled arm stops well before max_motion_steps=300.
    stalled_env = FakeRoboCasaGeneralEnv(stall_arm=True, camera_size=32)
    tk_stall, _ = _make_toolkit(
        tmp_path / "stall", env=stalled_env, action_mode="batch"
    )
    stalled = tk_stall.execute_tool(
        "execute_actions",
        {"actions": [{"action": "move_to", "xyz": [0.40, 0.0, 0.55]}]},
    )
    assert stalled.data["status"] == "stopped"
    assert stalled.data["stop_reason"] == "stalled"
    assert stalled.data["results"][0]["env_steps"] < 40


# ---------------------------------------------------------------------------
# 8. Prompt & HistoryRetention wiring
# ---------------------------------------------------------------------------


def test_prompt_and_history_retention_for_robocasa_general(tmp_path):
    general_batch = system_prompt(
        {
            "environment": "robocasa",
            "workflow": "episode",
            "world_cord": "general",
            "action_mode": "batch",
            "max_actions_per_call": 3,
            "prompt": "simple",
        }
    )
    assert "lifting support" in general_batch
    assert "coordinate_step" in general_batch
    assert "move_base and move_lift remain standalone tools" in general_batch
    assert "use execute_actions for every physical action" not in general_batch

    base_single = system_prompt(
        {
            "environment": "robocasa",
            "workflow": "episode",
            "world_cord": "base",
            "action_mode": "single",
            "prompt": "simple",
        }
    )
    assert "quaternion [w,x,y,z]" in base_single
    assert "move_lift" not in base_single

    assert "selected arm" not in argument_hint("move_to", "robocasa")
    assert "selected arm" in argument_hint("move_to", "robotwin")

    # HistoryRetention attached for RoboCasa general, omitted for RoboCasa base.
    from robots.rl.history import HistoryRetention

    spec = robot_spec.get_robot_spec()
    for world_cord, expect_retention in (("general", True), ("base", False)):
        planner = ApiAgentLoop(
            model="test",
            dashboard_events=NullDashboardEventSink(),
        )
        cfg = SimpleNamespace(
            prompt_vars={
                "environment": "robocasa",
                "workflow": "episode",
                "world_cord": world_cord,
            }
        )
        spec.configure_planner(planner, cfg)
        caps = planner.capabilities_factory(
            SimpleNamespace(list_tools=lambda: [], workflow_stats={}),
            "sys",
            "usr",
            10,
        )
        has_retention = any(isinstance(cap, HistoryRetention) for cap in caps)
        assert has_retention is expect_retention

    # Verify image_history=1 actually prunes earlier camera images on turn 2.
    import asyncio

    from pydantic_ai import BinaryContent
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    tk_hist, _ = _make_toolkit(
        tmp_path / "hist",
        env=FakeRoboCasaGeneralEnv(camera_size=32),
        action_mode="batch",
        image_history=1,
    )
    retention = HistoryRetention(tk_hist)
    req_ctx = SimpleNamespace(
        messages=[
            ModelRequest(
                parts=[
                    UserPromptPart(
                        ["turn 1", BinaryContent(data=b"img1", media_type="image/png")]
                    )
                ]
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        ["turn 2", BinaryContent(data=b"img2", media_type="image/png")]
                    )
                ]
            ),
        ]
    )
    ctx = SimpleNamespace(usage=SimpleNamespace(requests=1))
    filtered = asyncio.run(retention.before_model_request(ctx, req_ctx))
    imgs = [
        item
        for msg in filtered.messages
        for part in msg.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, list)
        for item in part.content
        if isinstance(item, BinaryContent)
    ]
    assert [img.data for img in imgs] == [b"img2"]


# ---------------------------------------------------------------------------
# 9. Remote ToolkitWorker <-> RemoteToolkit handshake & artifact sync
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("support_kind", ["table", "floor"])
@pytest.mark.parametrize("action_mode", ["single", "batch"])
def test_remote_robocasa_general_handshake_and_historical_query(
    tmp_path, monkeypatch, support_kind, action_mode
):
    # Allow running the remote handshake test on non-Linux hosts without /proc.
    orig_read_text = Path.read_text

    def patched_read_text(self, *args, **kwargs):
        if str(self).replace("\\", "/") == "/proc/sys/kernel/random/boot_id":
            return "test-boot-id\n"
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", patched_read_text)

    env = FakeRoboCasaGeneralEnv(support_kind=support_kind, camera_size=32)
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:20, 8:20] = True
    toolkit, _ = _make_toolkit(
        tmp_path / "backend",
        env=env,
        sam3=FakeSam3(mask),
        world_cord="general",
        action_mode=action_mode,
        image_size=1024,
    )
    release = contextlib.ExitStack()
    release.callback(toolkit.close)
    task = {
        "environment": "robocasa",
        "task_name": "PickPlaceCounterToCabinet"
        if support_kind == "table"
        else "OpenDrawer",
        "seed": 0,
        "split": "target",
        "workflow": "episode",
        "world_cord": "general",
        "action_mode": action_mode,
        "max_actions_per_call": 3,
        "image_size": 1024,
        "prompt": "simple",
    }
    req = {"task": task, "gpu": 0, "max_episode_steps": 10000}
    worker = remote.ToolkitWorker(
        toolkit, release, {**req, "output_dir": str(tmp_path / "backend/run")}
    )
    bridge = Bridge(worker)
    server = HttpRpcServer(("127.0.0.1", 0), bridge._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = remote.RemoteToolkit(
            endpoint=f"http://127.0.0.1:{server.server_address[1]}",
            request=req,
            config=SimpleNamespace(output_dir=tmp_path / "frontend"),
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "frontend-memory"),
        )
        try:
            assert remote.declarations(client) == remote.declarations(toolkit)
            init = client.initial_observation().data
            assert init["coordinates"]["world_cord"] == "general"
            assert init["coordinates"]["tcp"] == "finger_pad_midpoint"
            if support_kind == "floor":
                assert init["coordinates"]["support_surface"] == "floor"
                assert init["coordinates"]["table_height_above_base_m"] is None
            else:
                assert init["coordinates"]["table_height_above_base_m"] is not None

            # Fixed tool sequence across RPC boundary:
            # view_env -> segment -> back_project(step=0) -> arm action -> move_base -> old back_project
            assert len(client.execute_tool("view_env", {"camera": "all"}).images) == 3
            seg = client.execute_tool(
                "segment", {"prompt": "mug", "camera": "agentview", "step": 0}
            )
            assert seg.data["step"] == 0 and seg.data["coordinate_step"] == 0
            bp0 = client.execute_tool(
                "back_project", {"pixel": [12, 16], "camera": "agentview", "step": 0}
            )
            assert bp0.data["coordinate_step"] == 0

            if action_mode == "batch":
                act = client.execute_tool(
                    "execute_actions",
                    {
                        "actions": [
                            {"action": "move_to", "xyz": [0.27, -0.02, 0.55]},
                            {"action": "set_gripper", "opening": 1.0},
                        ]
                    },
                )
                assert act.data["status"] == "completed"
            else:
                act = client.execute_tool("move_to", {"xyz": [0.27, -0.02, 0.55]})
                assert act.data["status"] == "reached"

            moved = client.execute_tool(
                "move_base", {"vx": 0.15, "yaw_rate": 0.4, "duration_s": 0.5}
            )
            assert moved.data["status"] == "commanded"

            bp_after = client.execute_tool(
                "back_project", {"pixel": [12, 16], "camera": "agentview", "step": 0}
            )
            assert bp_after.data["step"] == 0
            assert bp_after.data["coordinate_step"] == client.state.latest_step
            assert client.state.records() == toolkit.state.records()
            # Per-step step_coordinates.json and run-level coordinates.json synced across RPC.
            assert client.state.load(
                "coordinates.json", step=None
            ) == toolkit.state.load("coordinates.json", step=None)
            for step_idx in range(client.state.latest_step + 1):
                assert client.state.load(
                    "step_coordinates.json", step=step_idx
                ) == toolkit.state.load("step_coordinates.json", step=step_idx)
        finally:
            client.close()
    finally:
        release.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()

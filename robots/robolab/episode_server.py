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

"""One headless native RoboLab DROID scene, served on Isaac's main thread."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from robots.robolab.access import CAMERAS, require_isaac_eula  # noqa: E402
from robots.robolab.geometry import inner_pad_center, root_control_action  # noqa: E402
from rpent.utils.logging import get_logger  # noqa: E402
from rpent.utils.rpc import RpcFacade  # noqa: E402
from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin  # noqa: E402
from rpent.utils.transforms import compose_transforms, rotation_from_wxyz  # noqa: E402

logger = get_logger(__name__)

_JAWS = ("left_inner_finger", "right_inner_finger")
_GRIPPER_PATH = "Gripper/Robotiq_2F_85"
_PAD_MESH = "Defeatured_2F_85_PAD_OPEN_fingertipsstep"


def _numpy(value) -> np.ndarray:
    """Read IsaacLab's tensor data under either native buffer backend."""
    import torch
    import warp

    if not isinstance(value, torch.Tensor):
        value = warp.to_torch(value)
    return value.detach().cpu().numpy()


def _pad_offsets(usd_path: str) -> np.ndarray:
    """Read each opposed contact-pad centre in its own rigid jaw frame."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    gripper = stage.GetDefaultPrim().GetPath().AppendPath(_GRIPPER_PATH)
    cache = UsdGeom.XformCache()
    flange = cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(gripper.AppendChild("base_link"))
    )
    world_to_flange = flange.GetInverse()
    jaw_frames = [
        cache.GetLocalToWorldTransform(stage.GetPrimAtPath(gripper.AppendChild(name)))
        for name in _JAWS
    ]
    offsets = []
    for index, name in enumerate(_JAWS):
        pad = gripper.AppendPath(f"{name}/{_PAD_MESH}_01/{_PAD_MESH}")
        mesh_prim = stage.GetPrimAtPath(pad)
        mesh = UsdGeom.Mesh(mesh_prim)
        mesh_to_flange = cache.GetLocalToWorldTransform(mesh_prim) * world_to_flange
        points = np.asarray(
            [tuple(mesh_to_flange.Transform(p)) for p in mesh.GetPointsAttr().Get()]
        )
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        polygons = np.split(
            indices, np.cumsum(mesh.GetFaceVertexCountsAttr().Get())[:-1]
        )
        other_pad = gripper.AppendPath(f"{_JAWS[1 - index]}/{_PAD_MESH}_01/{_PAD_MESH}")
        other_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(other_pad))
        other_to_flange = (
            cache.GetLocalToWorldTransform(other_mesh.GetPrim()) * world_to_flange
        )
        other_center = np.mean(
            [
                tuple(other_to_flange.Transform(p))
                for p in other_mesh.GetPointsAttr().Get()
            ],
            axis=0,
        )
        centre = inner_pad_center(points, polygons, other_center)
        flange_to_jaw = (jaw_frames[index] * world_to_flange).GetInverse()
        offsets.append(tuple(flange_to_jaw.Transform(Gf.Vec3d(*centre))))
    return np.asarray(offsets)


def _prim(env, asset):
    from isaacsim.core.utils.stage import get_current_stage

    env_path = env.scene.env_prim_paths[0]
    path = asset.prim_path.replace("/World/envs/env_.*", env_path)
    prim = get_current_stage().GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"No active RoboLab geometry at {path}")
    return prim


def _table_top(env) -> np.ndarray:
    """Measure the task table's rendered top plane in the active USD scene."""
    from pxr import Usd, UsdGeom

    asset = getattr(env.cfg.scene, "table", None) or env.cfg.scene.table_fixture
    table = _prim(env, asset)
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    extent = bounds.ComputeWorldBound(table).ComputeAlignedRange()
    if extent.IsEmpty():
        raise ValueError(f"RoboLab table has no measurable geometry: {table.GetPath()}")
    top = np.asarray(extent.GetMidpoint())
    top[2] = extent.GetMax()[2]
    return top


def episode_status(step: int, step_limit: int, verdict: bool | None) -> dict[str, Any]:
    """Combine the official frozen-environment result with the RPent step cap."""
    return {
        "step": step,
        "success": verdict is True,
        "terminated": verdict is True,
        "truncated": verdict is False or (verdict is None and step >= step_limit),
    }


class RoboLabEpisodeEnvFacade(MainThreadServeMixin, RpcFacade):
    """Expose only measurements and one acknowledged absolute-IK action."""

    def __init__(self, env, instruction: str, meta: dict[str, Any]) -> None:
        from pxr import UsdGeom

        super().__init__()
        self.env = env
        self.meta = {
            **meta,
            "camera_dimensions": {
                name: {
                    "height": env.scene[name].data.output["rgb"].shape[1],
                    "width": env.scene[name].data.output["rgb"].shape[2],
                }
                for name in CAMERAS
            },
        }
        self.instruction = instruction
        self.step_count = 0
        self.robot = env.scene["robot"]
        self._body_indices = {
            name: self.robot.body_names.index(name)
            for name in ("panda_link0", "base_link", *_JAWS)
        }
        self._finger_index = self.robot.joint_names.index("finger_joint")
        self._pad_offsets = _pad_offsets(self.robot.cfg.spawn.usd_path)
        self._calibration = self._measure_calibration()
        wrist = _prim(env, env.cfg.scene.wrist_cam)
        if wrist.GetParent().GetName() != "base_link":
            raise ValueError("RoboLab wrist_cam must be attached to the DROID flange")
        # USD matrices use row vectors. Read the fixed camera transform once.
        self._wrist_in_flange = np.asarray(
            UsdGeom.Xformable(wrist).GetLocalTransformation()
        ).T
        self._rpc.update(
            {
                "env.get_env_meta": self.get_env_meta,
                "env.get_task_language": self.get_task_language,
                "env.get_status": self.get_status,
                "env.get_motion_state": self.get_motion_state,
                "env.get_obs": self.get_obs,
                "env.coordinate_calibration": self.coordinate_calibration,
                "env.step": self.step_to,
            }
        )
        self._readonly_methods.update(self._rpc.keys() - {"env.step"})

    def get_env_meta(self) -> dict[str, Any]:
        """Return the fixed task, seed, camera sizes, and step limits."""
        return dict(self.meta)

    def get_task_language(self) -> str:
        """Return the native task instruction."""
        return self.instruction

    def coordinate_calibration(self) -> dict[str, Any]:
        """Return the measured mounting frame and grasp-axis calibration."""
        return self._calibration

    def _root_pose(self) -> tuple[np.ndarray, np.ndarray]:
        data = self.robot.data
        return (
            _numpy(data.root_pos_w[0]),
            rotation_from_wxyz(_numpy(data.root_quat_w[0])).as_matrix(),
        )

    def _measure_calibration(self) -> dict[str, Any]:
        index = self._body_indices["panda_link0"]
        data = self.robot.data
        motion = self.get_motion_state()
        flange = rotation_from_wxyz(motion["flange_quaternion_wxyz"]).as_matrix()
        pads = motion["pad_centres_world_m"]
        root_xyz, root_rotation = self._root_pose()
        return {
            "coordinate_calibration": {
                "base_xyz": _numpy(data.body_pos_w[0, index]).tolist(),
                "base_rotation": rotation_from_wxyz(_numpy(data.body_quat_w[0, index]))
                .as_matrix()
                .tolist(),
                "table_top_xyz": _table_top(self.env).tolist(),
                "tcp": "robotiq_inner_contact_pad_midpoint",
            },
            "left_to_right_in_flange": (flange.T @ (pads[0] - pads[1])).tolist(),
            "flange_to_tcp_in_flange": (
                flange.T @ (motion["tcp_world_xyz"] - motion["flange_world_xyz"])
            ).tolist(),
            "pad_offsets_in_jaw_m": self._pad_offsets.tolist(),
            "control_root_world_xyz": root_xyz.tolist(),
            "control_root_world_rotation": root_rotation.tolist(),
        }

    def get_status(self) -> dict[str, Any]:
        verdict = self.env.get_env_results()[0]["success"]
        return episode_status(self.step_count, self.meta["max_episode_steps"], verdict)

    def get_motion_state(self) -> dict[str, Any]:
        data = self.robot.data
        indices = [self._body_indices[name] for name in ("base_link", *_JAWS)]
        xyz = _numpy(data.body_pos_w[0, indices])
        quat = _numpy(data.body_quat_w[0, indices])
        pads = xyz[1:] + rotation_from_wxyz(quat[1:]).apply(self._pad_offsets)
        native_close = float(_numpy(data.joint_pos[0, self._finger_index])) / (
            np.pi / 4
        )
        return {
            "flange_world_xyz": xyz[0],
            "flange_quaternion_wxyz": quat[0],
            "tcp_world_xyz": pads.mean(axis=0),
            "pad_centres_world_m": pads,
            "opening": float(np.clip(1 - native_close, 0, 1)),
            "status": self.get_status(),
        }

    def get_obs(self) -> dict[str, Any]:
        frames = {}
        for name in CAMERAS:
            data = self.env.scene[name].data
            camera_to_world = np.eye(4)
            if name == "wrist_cam":
                # IsaacLab 2.2 reads camera poses from USD, which retains the
                # authored arm pose with Fabric enabled. RTX renders the live arm.
                index = self._body_indices["base_link"]
                robot = self.robot.data
                camera_to_world[:3, :3] = rotation_from_wxyz(
                    _numpy(robot.body_quat_w[0, index])
                ).as_matrix()
                camera_to_world[:3, 3] = _numpy(robot.body_pos_w[0, index])
                camera_to_world = compose_transforms(
                    camera_to_world, self._wrist_in_flange
                )
            else:
                camera_to_world[:3, :3] = rotation_from_wxyz(
                    _numpy(data.quat_w_opengl[0])
                ).as_matrix()
                camera_to_world[:3, 3] = _numpy(data.pos_w[0])
            frames[name] = {
                "color": _numpy(data.output["rgb"][0, ..., :3]),
                "distance_to_image_plane": _numpy(
                    data.output["distance_to_image_plane"][0, ..., 0]
                ),
                "intrinsic_matrix": _numpy(data.intrinsic_matrices[0]),
                "camera_to_world": camera_to_world,
            }
        return {"vision": frames, "status": self.get_status()}

    def step_to(self, xyz, quaternion_wxyz, opening, expected_step) -> dict[str, Any]:
        """Validate a world setpoint, advance once, and return measured state."""
        import torch

        if expected_step != self.step_count:
            raise ValueError(
                f"RoboLab step mismatch: expected {expected_step}, "
                f"server is at {self.step_count}"
            )
        status = self.get_status()
        if status["terminated"] or status["truncated"]:
            raise ValueError("RoboLab episode has ended")
        xyz, quat = (
            np.asarray(xyz, dtype=float),
            np.asarray(quaternion_wxyz, dtype=float),
        )
        if (
            xyz.shape != (3,)
            or quat.shape != (4,)
            or not np.isfinite(xyz).all()
            or not np.isfinite(quat).all()
            or not np.isclose(np.linalg.norm(quat), 1, atol=1e-4)
            or not 0 <= opening <= 1
        ):
            raise ValueError(
                "RoboLab expects finite xyz, unit WXYZ, and opening in [0, 1]"
            )
        action = root_control_action(
            xyz,
            rotation_from_wxyz(quat).as_matrix(),
            opening,
            *self._root_pose(),
        )
        self.env.step(
            torch.as_tensor(
                action[None, :], dtype=torch.float32, device=self.env.device
            )
        )
        self.step_count += 1
        return self.get_motion_state()


class RoboLabEpisodeRuntime:
    """Own one Isaac application and rebuild the native scene for each episode."""

    def __init__(
        self, *, cuda_device: int | None, scratch_dir: Path | None = None
    ) -> None:
        require_isaac_eula()
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        kit_args = "--/renderer/multiGpu/enabled=false"
        if cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
            kit_args += f" --/renderer/activeGpu={cuda_device}"
        extra_kit_args = os.environ.get("RPENT_ISAAC_KIT_ARGS", "").strip()
        if extra_kit_args:
            kit_args += f" {extra_kit_args}"

        import cv2  # noqa: F401  # Load OpenCV before Kit, as RoboLab does.
        from isaaclab.app import AppLauncher

        self.device = cuda_device
        self._scratch_dir = scratch_dir
        if scratch_dir is not None:
            scratch_dir.mkdir(exist_ok=True)
        self._app = AppLauncher(
            headless=True,
            enable_cameras=True,
            device="cuda:0",
            renderer="realtime",
            kit_args=kit_args,
        ).app
        self._scene = None
        self._scratch: TemporaryDirectory[str] | None = None
        self._scene_failed = False

    def open_episode(
        self, *, task: str, seed: int, max_episode_steps: int, save_dir: Path | None
    ) -> RoboLabEpisodeEnvFacade:
        """Create a fresh, seeded scene on the worker's main thread."""
        if max_episode_steps <= 0:
            raise ValueError("max episode steps must be positive")
        if self._scene_failed:
            raise RuntimeError(
                "RoboLab scene initialization failed; restart the worker"
            )
        self._release_scene()

        from robolab.constants import set_output_dir
        from robolab.core.environments.config import parse_env_cfg
        from robolab.core.environments.runtime import create_env
        from robolab.core.task.instruction import instruction_text
        from robolab.registrations.droid.auto_env_registrations_abs_ik import (
            auto_register_droid_abs_ik_envs,
        )

        try:
            native_recording = save_dir is not None
            if save_dir is None:
                self._scratch = TemporaryDirectory(
                    prefix="episode-", dir=self._scratch_dir
                )
                save_dir = Path(self._scratch.name)
            save_dir.mkdir(parents=True, exist_ok=True)
            set_output_dir(str(save_dir))
            auto_register_droid_abs_ik_envs(task=task)
            cfg = parse_env_cfg(
                task, device="cuda:0", seed=seed, num_envs=1, use_fabric=True
            )
            if not native_recording:
                cfg.recorders = None
            for name in CAMERAS:
                camera = getattr(cfg.scene, name)
                camera.data_types = list(
                    dict.fromkeys((*camera.data_types, "distance_to_image_plane"))
                )

            env, cfg = create_env(cfg, policy="rpent", renderer="realtime")
            self._scene = env
            env.reset_eval_state()
            env.reset(seed=seed)
            return RoboLabEpisodeEnvFacade(
                env,
                instruction_text(cfg.instruction),
                {
                    "task": task,
                    "seed": seed,
                    "max_episode_steps": max_episode_steps,
                    "native_max_episode_steps": int(env.max_episode_length),
                    "device_id": self.device,
                    "num_envs": 1,
                },
            )
        except BaseException:
            self._scene_failed = True
            try:
                self._release_scene()
            except Exception:
                logger.exception("RoboLab could not release a failed scene")
            raise

    def _release_scene(self) -> None:
        env, self._scene = self._scene, None
        scratch, self._scratch = self._scratch, None
        try:
            if env is not None:
                from robolab.core.environments.runtime import end_episode
                from robolab.core.world.world_state import clear_world_cache

                try:
                    end_episode(env)
                finally:
                    try:
                        env.close()
                    finally:
                        clear_world_cache()
        finally:
            if scratch is not None:
                scratch.cleanup()

    def close_episode(self) -> None:
        """Release the episode and retire workers with uncertain scene state."""
        self._release_scene()
        if self._scene_failed:
            raise RuntimeError(
                "RoboLab scene initialization failed; restart the worker"
            )

    def close(self) -> None:
        """Release the scene and Isaac application."""
        try:
            self._release_scene()
        finally:
            if self._app is not None:
                app, self._app = self._app, None
                app.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Native RoboLab DROID env RPC server")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=700)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--transport", choices=("http", "socket"), default="http")
    parser.add_argument("--parent-watch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_episode_steps <= 0:
        raise ValueError("max episode steps must be positive")
    runtime = RoboLabEpisodeRuntime(cuda_device=args.cuda_device)
    try:
        facade = runtime.open_episode(
            task=args.task,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            save_dir=args.save_dir,
        )
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

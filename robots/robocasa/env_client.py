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

"""RoboCasa env client — thin RPC layer over the env server."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from rpent.robots.components.env_client_base import BaseEnvClient

if TYPE_CHECKING:
    from rpent.utils.rpc import RpcClient

CAM_ALIAS = {
    "agentview": "robot0_agentview_left",
    "agentview_left": "robot0_agentview_left",
    "agentview_right": "robot0_agentview_right",
    "wrist": "robot0_eye_in_hand",
    "eye_in_hand": "robot0_eye_in_hand",
    "robot0_eye_in_hand": "robot0_eye_in_hand",
    "robot0_agentview_left": "robot0_agentview_left",
    "robot0_agentview_right": "robot0_agentview_right",
    "navview": "mobilebase0_navview",  # base-mounted forward-down floor/nav camera
    "mobilebase0_navview": "mobilebase0_navview",
}

ENV_PROTOCOL_VERSION = 1
EXPECTED_ACTION_FIELDS = {
    "action.end_effector_position": 3,
    "action.end_effector_rotation": 3,
    "action.gripper_close": 1,
    "action.base_motion": 4,
    "action.control_mode": 1,
}
EXPECTED_ACTION_DIM = sum(EXPECTED_ACTION_FIELDS.values())
EXPECTED_CAMERAS = {
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
}
MAX_RENDER_DIM = 4096
MAX_ACTION_CHUNK = 512


def _positive_int(value, name, *, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0 or value > maximum:
        raise ValueError(f"{name} must be in [1, {maximum}], got {value}")
    return value


def _finite_array(value, name, *, shape):
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain numeric values")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


class RoboCasaEnvClient(BaseEnvClient):
    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "env.grasp_contact": 10.0,
    }

    def __init__(self, client: RpcClient, *, expected_meta: dict):
        self._client = client
        server_meta = self._client.call(
            "env.get_env_meta", timeout_s=self._TIMEOUT_S["default"]
        )
        if server_meta != expected_meta:
            raise RuntimeError(
                f"env_meta mismatch: expected={expected_meta!r} "
                f"actual={server_meta!r}. The env_server was launched with "
                "different args than this client expects — kill the stale "
                "env_server and relaunch."
            )
        self.camera_h = server_meta["camera_h"]
        self.camera_w = server_meta["camera_w"]
        self.runtime_info = self._client.call(
            "env.get_runtime_info", timeout_s=self._TIMEOUT_S["default"]
        )
        self._validate_runtime_info(self.runtime_info)
        self.runtime_id = self.runtime_info["runtime_id"]
        self._action_dim = int(self.runtime_info["action_schema"]["flat_dim"])
        self._episode_id = int(self.runtime_info["episode_id"])
        self._sim_step = int(self.runtime_info["sim_step"])
        self._success_latched = bool(self.runtime_info["success_latched"])
        self.last_obs = None
        self.reset()

    @staticmethod
    def _validate_runtime_info(info):
        if not isinstance(info, Mapping):
            raise TypeError("env runtime handshake must return a mapping")
        if info.get("protocol_version") != ENV_PROTOCOL_VERSION:
            raise RuntimeError(
                f"unsupported env protocol_version={info.get('protocol_version')!r}; "
                f"expected {ENV_PROTOCOL_VERSION}"
            )
        runtime_id = info.get("runtime_id")
        if not isinstance(runtime_id, str) or not runtime_id:
            raise RuntimeError("env runtime handshake is missing runtime_id")
        schema = info.get("action_schema")
        if (
            not isinstance(schema, Mapping)
            or schema.get("flat_dim") != EXPECTED_ACTION_DIM
        ):
            raise RuntimeError(f"incompatible env action schema: {schema!r}")
        fields = schema.get("fields")
        if not isinstance(fields, list):
            raise RuntimeError("env action schema fields must be a list")
        actual_fields = {
            field.get("name"): field.get("size")
            for field in fields
            if isinstance(field, Mapping)
        }
        if actual_fields != EXPECTED_ACTION_FIELDS:
            raise RuntimeError(
                f"incompatible env action fields: expected={EXPECTED_ACTION_FIELDS!r}, "
                f"actual={actual_fields!r}"
            )
        cameras = info.get("cameras")
        if not isinstance(cameras, list):
            raise RuntimeError("env runtime cameras must be a list")
        perception_isolation = info.get("perception_isolation", False)
        if not isinstance(perception_isolation, bool):
            raise RuntimeError("invalid env runtime perception_isolation flag")
        required_cameras = set(EXPECTED_CAMERAS)
        if perception_isolation:
            required_cameras.add("mobilebase0_navview")
        missing_cameras = required_cameras.difference(cameras)
        if missing_cameras:
            raise RuntimeError(
                f"env runtime is missing required cameras {sorted(missing_cameras)}"
            )
        for key in ("episode_id", "sim_step"):
            value = info.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"invalid env runtime {key}={value!r}")
        if not isinstance(info.get("success_latched"), bool):
            raise RuntimeError("invalid env runtime success_latched flag")

    def _resolve_cam(self, name):
        if not isinstance(name, str) or not name:
            raise TypeError("camera name must be a non-empty string")
        return CAM_ALIAS.get(name, name)

    def _resolve_size(self, height, width):
        height = self.camera_h if height is None else height
        width = self.camera_w if width is None else width
        return (
            _positive_int(height, "height", maximum=MAX_RENDER_DIM),
            _positive_int(width, "width", maximum=MAX_RENDER_DIM),
        )

    # ---- lifecycle ----
    # ---- state accessors ----
    def reset(self):
        obs = self._client.call("env.reset", timeout_s=self._TIMEOUT_S["env.reset"])
        if not isinstance(obs, Mapping):
            raise TypeError(
                f"env.reset returned {type(obs).__name__}, expected mapping"
            )
        previous_episode = self._episode_id
        runtime_info = self._client.call(
            "env.get_runtime_info", timeout_s=self._TIMEOUT_S["default"]
        )
        self._validate_runtime_info(runtime_info)
        if runtime_info["runtime_id"] != self.runtime_id:
            raise RuntimeError("env runtime changed during reset")
        if runtime_info["episode_id"] != previous_episode + 1:
            raise RuntimeError(
                "env reset did not advance exactly one episode: "
                f"before={previous_episode}, after={runtime_info['episode_id']!r}"
            )
        if runtime_info["sim_step"] != 0:
            raise RuntimeError("env reset returned a nonzero simulator step")
        self.last_obs = obs
        self.runtime_info = runtime_info
        self._episode_id = runtime_info["episode_id"]
        self._sim_step = 0
        self._success_latched = runtime_info["success_latched"]
        return self.last_obs

    def step(self, flat_action):
        action = _finite_array(flat_action, "flat_action", shape=(self._action_dim,))
        result = self._client.call(
            "env.step", args=(action,), timeout_s=self._TIMEOUT_S["env.step"]
        )
        if not isinstance(result, (list, tuple)) or len(result) != 4:
            raise RuntimeError("env.step must return (obs, reward, done, info)")
        obs, reward, done, info = result
        if not isinstance(obs, Mapping):
            raise TypeError("env.step observation must be a mapping")
        if isinstance(reward, bool) or not isinstance(reward, (int, float, np.number)):
            raise TypeError("env.step reward must be numeric")
        if not np.isfinite(float(reward)):
            raise ValueError("env.step reward must be finite")
        if not isinstance(done, (bool, np.bool_)):
            raise TypeError("env.step done must be boolean")
        if not isinstance(info, Mapping):
            raise TypeError("env.step info must be a mapping")
        if info.get("rpent_runtime_id") != self.runtime_id:
            raise RuntimeError("env.step response came from a different runtime")
        if info.get("rpent_episode_id") != self._episode_id:
            raise RuntimeError("env.step response episode identity mismatch")
        sim_step = info.get("rpent_sim_step")
        if (
            isinstance(sim_step, bool)
            or not isinstance(sim_step, int)
            or sim_step != self._sim_step + 1
        ):
            raise RuntimeError(
                f"env.step sequence mismatch: expected {self._sim_step + 1}, got {sim_step!r}"
            )
        success_latched = info.get("rpent_success_latched")
        if not isinstance(success_latched, bool):
            raise RuntimeError("env.step response is missing the success latch")
        self._sim_step = sim_step
        self._success_latched = self._success_latched or success_latched
        self.last_obs = obs
        return result

    def chunk_step(self, flat_actions, *, return_all_frames=False):
        """Apply a bounded chunk and verify the server's per-action identities."""
        actions = np.asarray(flat_actions)
        if actions.ndim != 2 or actions.shape[1:] != (self._action_dim,):
            raise ValueError(
                f"flat_actions must have shape (steps, {self._action_dim}), "
                f"got {actions.shape}"
            )
        if not 1 <= actions.shape[0] <= MAX_ACTION_CHUNK:
            raise ValueError(
                f"action chunk length must be in [1, {MAX_ACTION_CHUNK}], "
                f"got {actions.shape[0]}"
            )
        if not np.issubdtype(actions.dtype, np.number) or np.issubdtype(
            actions.dtype, np.bool_
        ):
            raise TypeError("flat_actions must contain numeric values")
        if not np.isfinite(actions).all():
            raise ValueError("flat_actions must contain only finite values")
        if not isinstance(return_all_frames, (bool, np.bool_)):
            raise TypeError("return_all_frames must be boolean")
        result = self._client.call(
            "env.chunk_step",
            args=(actions,),
            kwargs={"return_all_frames": bool(return_all_frames)},
            timeout_s=self._TIMEOUT_S["env.chunk_step"],
        )
        if not isinstance(result, (list, tuple)) or len(result) != 4:
            raise RuntimeError(
                "env.chunk_step must return (obs, rewards, dones, infos)"
            )
        obs_field, rewards, dones, infos = result
        observations = obs_field if return_all_frames else [obs_field]
        if not isinstance(observations, list) or not observations:
            raise RuntimeError("env.chunk_step returned no observations")
        if any(not isinstance(obs, Mapping) for obs in observations):
            raise TypeError("env.chunk_step observations must be mappings")
        if not isinstance(infos, list) or not 1 <= len(infos) <= actions.shape[0]:
            raise RuntimeError("env.chunk_step returned an invalid info sequence")
        rewards = _finite_array(rewards, "chunk rewards", shape=(len(infos),))
        dones = np.asarray(dones)
        if dones.shape != (len(infos),) or dones.dtype != np.bool_:
            raise ValueError(
                f"chunk dones must be boolean with shape {(len(infos),)}, "
                f"got dtype={dones.dtype}, shape={dones.shape}"
            )
        if return_all_frames and len(observations) != len(infos):
            raise RuntimeError("env.chunk_step observation/info lengths differ")
        for info in infos:
            if not isinstance(info, Mapping):
                raise TypeError("env.chunk_step infos must be mappings")
            expected_step = self._sim_step + 1
            if (
                info.get("rpent_runtime_id") != self.runtime_id
                or info.get("rpent_episode_id") != self._episode_id
                or info.get("rpent_sim_step") != expected_step
            ):
                raise RuntimeError("env.chunk_step response identity/sequence mismatch")
            latched = info.get("rpent_success_latched")
            if not isinstance(latched, bool):
                raise RuntimeError(
                    "env.chunk_step response is missing the success latch"
                )
            self._sim_step = expected_step
            self._success_latched = self._success_latched or latched
        if self._success_latched and len(infos) > 1:
            first_success = next(
                (i for i, info in enumerate(infos) if info["rpent_success_latched"]),
                None,
            )
            if first_success is not None and first_success != len(infos) - 1:
                raise RuntimeError("env.chunk_step did not stop at the first success")
        self.last_obs = observations[-1]
        return obs_field, rewards, dones, infos

    def check_success(self):
        if self._success_latched:
            return True
        value = self._client.call(
            "env.check_success", timeout_s=self._TIMEOUT_S["default"]
        )
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("env.check_success must return a boolean")
        self._success_latched = bool(value)
        return self._success_latched

    @property
    def success_latched(self):
        """Return the authoritative latch carried by reset and step responses."""
        return self._success_latched

    @property
    def terminated(self):
        return self.check_success()

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def current_raw_obs(self):
        if self.last_obs is None:
            raise RuntimeError("environment has not been reset")
        return self.last_obs

    @property
    def eef_pos(self):
        return _finite_array(
            self.current_raw_obs["robot0_eef_pos"],
            "robot0_eef_pos",
            shape=(3,),
        )

    @property
    def eef_quat(self):
        return _finite_array(
            self.current_raw_obs["robot0_eef_quat"],
            "robot0_eef_quat",
            shape=(4,),
        )

    @property
    def gripper_qpos(self):
        return _finite_array(
            self.current_raw_obs.get("robot0_gripper_qpos"),
            "robot0_gripper_qpos",
            shape=(2,),
        )

    # ---- task info ----
    def get_task_language(self) -> str | None:
        value = super().get_task_language()
        if value is not None and not isinstance(value, str):
            raise TypeError("env.get_task_language must return a string or null")
        return value

    def get_success_criteria_text(self):
        return self._client.call(
            "env.get_success_criteria_text", timeout_s=self._TIMEOUT_S["default"]
        )

    def get_task_progress(self):
        progress = self._client.call(
            "env.get_task_progress", timeout_s=self._TIMEOUT_S["default"]
        )
        if not isinstance(progress, Mapping):
            raise TypeError("env.get_task_progress must return a mapping")
        return dict(progress)

    # ---- rendering / perception ----
    def _render_native(self, camera_name, height, width, depth):
        """Render through the shared RPC contract and validate its payload."""
        camera_name = self._resolve_cam(camera_name)
        height, width = self._resolve_size(height, width)
        if not isinstance(depth, (bool, np.bool_)):
            raise TypeError("depth must be a boolean")
        result = super().render_camera(
            camera_name,
            height=height,
            width=width,
            depth=bool(depth),
        )
        if depth:
            if not isinstance(result, (list, tuple)) or len(result) != 2:
                raise RuntimeError("depth render must return (rgb, depth)")
            rgb = np.asarray(result[0])
            real = _finite_array(result[1], "depth", shape=(height, width))
            if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
                raise ValueError(
                    f"RGB render must be uint8 with shape {(height, width, 3)}, "
                    f"got dtype={rgb.dtype}, shape={rgb.shape}"
                )
            return rgb, real
        rgb = np.asarray(result)
        if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
            raise ValueError(
                f"RGB render must be uint8 with shape {(height, width, 3)}, "
                f"got dtype={rgb.dtype}, shape={rgb.shape}"
            )
        return rgb

    def render_camera(self, camera_name, height=None, width=None, depth=False):
        """Agent-facing: vertically flip to top-down so the rgb reads naturally and
        is pixel-aligned with world_map()."""
        cam = self._resolve_cam(camera_name)
        h, w = self._resolve_size(height, width)
        if depth:
            rgb, real = self._render_native(cam, h, w, True)
            return rgb[::-1], real[::-1]
        return self._render_native(cam, h, w, False)[::-1]

    def get_camera_meta(self, camera_name, height=None, width=None):
        cam = self._resolve_cam(camera_name)
        h, w = self._resolve_size(height, width)
        meta = super().get_camera_meta(cam, height=h, width=w)
        if not isinstance(meta, Mapping):
            raise TypeError("camera metadata must be a mapping")
        if (
            meta.get("runtime_id") != self.runtime_id
            or meta.get("episode_id") != self._episode_id
        ):
            raise RuntimeError("camera metadata identity mismatch")
        _finite_array(meta.get("intrinsic"), "camera intrinsic", shape=(3, 3))
        _finite_array(meta.get("extrinsic_cam2world"), "camera extrinsic", shape=(4, 4))
        return dict(meta)

    def world_map(self, camera_name, height=None, width=None):
        """HxWx3 world xyz per pixel, TOP-DOWN (row 0 = image top), pixel-aligned
        with render_camera()'s rgb. Uses robosuite's camera transform matrix; the
        sim's OpenGL depth is bottom-up, so it is vertically flipped to match the
        transform's top-down convention (VERIFIED: GT object back-projects within
        ~2cm). Dense vectorized: world = T_p2w @ [col*z, row*z, z, 1]."""
        cam = self._resolve_cam(camera_name)
        h, w = self._resolve_size(height, width)
        _, z_native = self._render_native(cam, h, w, True)  # metric depth, bottom-up
        z = z_native[::-1]  # -> top-down
        T_p2w = self._client.call(
            "env.get_camera_transform",
            kwargs={"camera_name": cam, "height": h, "width": w},
            timeout_s=self._TIMEOUT_S["default"],
        )
        T_p2w = _finite_array(T_p2w, "pixel-to-world transform", shape=(4, 4))
        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        homog = np.stack([cols * z, rows * z, z, np.ones_like(z)], axis=-1)  # HxWx4
        world = homog @ T_p2w.T  # HxWx4
        return world[..., :3]  # top-down, aligned with rgb

    def world_xyz_at(self, camera_name, row, col, height=None, width=None):
        if isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)):
            raise TypeError("row must be an integer")
        if isinstance(col, (bool, np.bool_)) or not isinstance(col, (int, np.integer)):
            raise TypeError("col must be an integer")
        world = self.world_map(camera_name, height, width)
        row, col = int(row), int(col)
        if not (0 <= row < world.shape[0] and 0 <= col < world.shape[1]):
            raise IndexError(f"pixel ({row}, {col}) is outside {world.shape[:2]}")
        return world[row, col]

    # ---- other ----
    def grasp_contact(self):
        result = self._client.call(
            "env.grasp_contact", timeout_s=self._TIMEOUT_S["env.grasp_contact"]
        )
        if not isinstance(result, (list, tuple)) or len(result) != 2:
            raise RuntimeError("env.grasp_contact must return (bool, object_name)")
        grasping, object_name = result
        if not isinstance(grasping, (bool, np.bool_)):
            raise TypeError("env.grasp_contact flag must be boolean")
        if object_name is not None and not isinstance(object_name, str):
            raise TypeError("env.grasp_contact object name must be a string or null")
        return bool(grasping), object_name

    def reassemble_env_action(self, unmap_result):
        if not isinstance(unmap_result, Mapping):
            raise TypeError("unmap_result must be a mapping")
        action = self._client.call(
            "env.reassemble_env_action",
            args=(dict(unmap_result),),
            timeout_s=self._TIMEOUT_S["default"],
        )
        return _finite_array(
            action, "reassembled env action", shape=(self._action_dim,)
        )

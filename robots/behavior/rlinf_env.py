# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Adapt RLinf BEHAVIOR environments to RPent observation and action contracts."""

from __future__ import annotations

import threading
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from robots.behavior.env_client import validate_native_actions
from robots.behavior.observation import (
    CAMERA_SUFFIXES,
    native_policy_observation,
    sensor_key,
    valid_observation_token,
)
from robots.behavior.tools import (
    R1PRO_EVAL_CAMERA_ORDER,
    decode_robot_state,
    extract_public_proprio,
)
from rpent.utils.serialization import to_numpy_tree
from rpent.utils.transforms import compose_transforms, invert_transform

if TYPE_CHECKING:
    from rlinf.envs.sim.behavior import BehaviorSimulator
    from rlinf.envs.sim.behavior.behavior_env import BehaviorEnv


def _pose_matrix(pose: dict[str, Any]) -> np.ndarray:
    position = np.asarray(pose["position"], dtype=np.float64)
    quaternion = np.asarray(pose["orientation_xyzw"], dtype=np.float64)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(position).all()
        or not np.isfinite(quaternion).all()
        or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-3)
    ):
        raise ValueError("Native poses require finite xyz and unit xyzw.")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = position
    return matrix


def _public_proprio(state: dict[str, Any]) -> np.ndarray:
    """Pack public61 from named physical joints, never from model padding."""
    names = state["joint_names"]
    qpos = np.asarray(state["joint_positions"])
    qvel = np.asarray(state["joint_velocities"])
    if (
        len(names) != 28
        or len(set(names)) != 28
        or qpos.shape != (28,)
        or qvel.shape != (28,)
    ):
        raise ValueError("R1Pro state requires 28 distinct named physical joints.")
    indices = {name: index for index, name in enumerate(names)}

    def joints(prefix: str, count: int, values: np.ndarray) -> np.ndarray:
        return values[[indices[f"{prefix}{index}"] for index in range(1, count + 1)]]

    parts = [np.asarray(state["robot_frame_base_velocity"])]
    for arm in ("left", "right"):
        eef = state["eef_poses"][arm]["robot_base"]
        parts.extend(
            (
                joints(f"{arm}_arm_joint", 7, qpos),
                joints(f"{arm}_arm_joint", 7, qvel),
                np.asarray(eef["position"]),
                np.asarray(eef["orientation_xyzw"]),
                joints(f"{arm}_gripper_finger_joint", 2, qpos),
                joints(f"{arm}_gripper_finger_joint", 2, qvel),
            )
        )
    parts.extend((joints("torso_joint", 4, qpos), joints("torso_joint", 4, qvel)))
    proprio = np.concatenate(parts)
    decode_robot_state(proprio)
    return proprio


class _BehaviorSession:
    """Hold the last confirmed frame and serialize actions for one RPent episode."""

    def __init__(
        self, simulator, *, task_id: int, task_language: str, frequency: float
    ) -> None:
        if simulator.num_envs != 1:
            raise ValueError("RPent BEHAVIOR serves one simulator environment.")
        if type(task_id) is not int or task_id < 0 or not task_language.strip():
            raise ValueError(
                "A nonnegative task_id and nonempty instruction are required."
            )
        self.simulator = simulator
        self.task_id = task_id
        self.task_language = task_language
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("Native action_frequency must be finite and positive.")
        self.control_dt_s = 1.0 / frequency
        self._owner_thread = threading.get_ident()
        self._stop = threading.Event()
        self._stop_reason = "closed"
        self._generation = -1
        self._issued = self._confirmed = 0
        self._previous_proprio: np.ndarray | None = None
        self.last_obs: dict[str, Any] | None = None
        self._metrics: dict[str, Any] = {}
        self._terminated = self._truncated = False
        self._fault: str | None = None

    def _check_ready(self, *, resetting: bool = False) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Run BEHAVIOR SDK operations on the owning thread.")
        if not resetting and self.last_obs is None:
            raise RuntimeError("Call reset() before serving BEHAVIOR observations.")

    def _token(self) -> dict[str, int]:
        return {"generation": self._generation, "observation_index": self._confirmed}

    def _motion_context(self) -> dict[str, Any]:
        token = self._token()
        return {
            "source": "confirmed_public_control_observations",
            "control_dt_s": self.control_dt_s,
            "current_token": token,
            "previous_token": {**token, "observation_index": self._confirmed - 1}
            if self._previous_proprio is not None
            else None,
            "current_proprio": extract_public_proprio(self.last_obs),
            "previous_proprio": self._previous_proprio,
            "pair_confirmed": self._previous_proprio is not None
            and self._issued == self._confirmed,
            "initial_bootstrap": self._confirmed == self._issued == 0,
            "issued_actions": self._issued,
            "confirmed_actions": self._confirmed,
            "unconfirmed_actions": self._issued - self._confirmed,
            "actions_pending": False,
        }

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset through RLinf and start a new local episode/frame counter."""
        self._check_ready(resetting=True)
        if self._stop.is_set():
            raise RuntimeError("The BEHAVIOR service has stopped.")
        self._generation += 1
        self.last_obs = None
        self._previous_proprio = None
        self._issued = self._confirmed = 0
        self._terminated = self._truncated = False
        self._fault = None
        observation, reset_info, metrics = self._reset_episode()
        self.last_obs, self._metrics = observation, metrics
        return deepcopy(self.last_obs), deepcopy({**reset_info, **metrics})

    def get_observation(self) -> dict[str, Any]:
        """Copy the latest confirmed frame; reads never advance the simulator."""
        self._check_ready()
        return deepcopy(self.last_obs)

    def get_motion_snapshot(self) -> dict[str, Any]:
        """Copy the current frame together with its real adjacent-step history."""
        self._check_ready()
        return deepcopy(
            {
                "observation": self.last_obs,
                "observation_token": self._token(),
                "motion_context": self._motion_context(),
            }
        )

    def get_camera_meta(self, camera_name: str | None = None) -> dict[str, Any]:
        """Copy calibration and poses paired with the latest confirmed images."""
        self._check_ready()
        metadata = self.last_obs.get("camera_metadata", {})
        if camera_name is not None and camera_name not in CAMERA_SUFFIXES:
            raise KeyError(camera_name)
        if camera_name is None:
            return deepcopy(metadata)
        return deepcopy(metadata.get(camera_name, {}))

    def render_camera(self, camera_name: str = "head") -> np.ndarray:
        """Copy one confirmed RGB image without an additional render or step."""
        self._check_ready()
        from robots.behavior.tools import prepare_behavior_observation

        views = prepare_behavior_observation(
            self.last_obs, task_id=self.task_id, task_language=self.task_language
        )["views"]
        return views[camera_name]["rgb"].copy()

    def stop(self, reason: str = "closed") -> None:
        """Request chunk cancellation from any thread; the owner closes the SDK."""
        self._stop_reason = reason
        self._stop.set()

    def _end_reason(self) -> str | None:
        if self._fault or self._stop.is_set():
            return self._fault or self._stop_reason
        if self._terminated:
            return "terminated"
        return "truncated" if self._truncated else None

    def session_status(self) -> dict[str, Any]:
        """Report native episode outcome and captured metrics without stepping."""
        self._check_ready()
        reason = self._end_reason()
        return {
            "ended": reason is not None,
            "reason": reason,
            "outcome": reason or "running",
            "generation": self._generation,
            "terminated": self._terminated,
            "truncated": self._truncated,
            **deepcopy(self._metrics),
        }

    def chunk_step(
        self,
        actions: Any,
        *,
        return_all_frames: bool = False,
        expected_observation_token: dict[str, int] | None = None,
    ) -> tuple:
        """Execute native actions once, stopping at cancellation, terminal state or error.

        Counts describe this call; motion-context counters describe the episode.
        Confirmation requires both a successful native step and its adapted
        frame. Reward is the sum of confirmed steps. A failed submission retains
        that prefix and marks its action unconfirmed; it is never replayed.
        Failure diagnostics distinguish a native error from an already returned
        step whose observation could not be adapted.
        """
        self._check_ready()
        array = validate_native_actions(actions, chunk=True).copy()
        if expected_observation_token is not None:
            if not valid_observation_token(expected_observation_token):
                raise ValueError(
                    "expected_observation_token must identify one public frame"
                )
            if expected_observation_token != self._token():
                raise RuntimeError("stale_observation: public planning frame changed")
        issued = confirmed = 0
        reward = 0.0
        frames = []
        failure: dict[str, Any] = {}
        for action in array:
            if self._end_reason() is not None:
                break
            self._issued += 1
            issued += 1
            phase = "native_step"
            try:
                result = self._submit_action(action)
                phase = "observation_adapter"
                observation, step_reward, terminated, truncated, metrics = (
                    self._unpack_step(result)
                )
            except Exception as exc:
                self._fault = "execution_outcome_unknown"
                failure = {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "code": f"{phase}_failed",
                    "native_step_returned": phase != "native_step",
                }
                break
            self._previous_proprio = extract_public_proprio(self.last_obs).copy()
            self.last_obs = observation
            self._confirmed += 1
            confirmed += 1
            reward += step_reward
            self._terminated, self._truncated = terminated, truncated
            self._metrics = metrics
            if return_all_frames:
                frames.append(deepcopy(observation))
        reason = self._end_reason()
        info = {
            **self._metrics,
            "requested_actions": len(array),
            "issued_actions": issued,
            "confirmed_actions": confirmed,
            "executed_actions": confirmed,
            "unconfirmed_actions": issued - confirmed,
            "stop_reason": self._fault
            or ("session_ended" if reason is not None else "chunk_completed"),
            "session_outcome": reason or "running",
            "last_known_observation_available": True,
            "observation_after_last_issued_action": issued > 0 and issued == confirmed,
            "observation_token": self._token(),
            "motion_context": self._motion_context(),
            "interrupted": self._stop.is_set(),
            **failure,
        }
        return (
            frames if return_all_frames else deepcopy(self.last_obs),
            reward,
            None if self._fault else self._terminated,
            None if self._fault else self._truncated,
            deepcopy(info),
        )


class RlinfBehaviorEnv(_BehaviorSession):
    """Borrow a single RLinf simulator and serialize tool actions on its thread.

    The caller constructs the simulator and calls reset() once before serving.
    Physics, task termination and scoring stay in RLinf. This adapter retains
    only confirmed frames and counters needed by existing RPent motion tools.
    Advance physics only through this adapter. Native planning utilities may
    inspect the borrowed simulator, but execute their commands via chunk_step()
    so the retained frame and command counters stay aligned.
    stop() may be called from another thread; it does not close the simulator.
    The owner saves results before the simulator's terminal-process close().

    Args:
        simulator: Public RLinf BehaviorSimulator with num_envs=1.
        task_id: Official BEHAVIOR task identifier used by the model and tools.
        task_language: Instruction selected by the launcher.
    """

    def __init__(
        self, simulator: BehaviorSimulator, *, task_id: int, task_language: str
    ) -> None:
        super().__init__(
            simulator,
            task_id=task_id,
            task_language=task_language,
            frequency=float(
                simulator.native_env.env_config.get("action_frequency", 60)
            ),
        )

    def _reset_episode(self):
        observations, infos = self.simulator.reset()
        observation = self._observation(observations[0])
        metrics = to_numpy_tree(self.simulator.get_metrics()[0])
        return observation, infos[0], metrics

    def _submit_action(self, action):
        return self.simulator.step(action[None], render=True)

    def _unpack_step(self, result):
        observations, rewards, terms, truncs, infos = to_numpy_tree(result)
        observation = self._observation(observations[0])
        metrics = {
            key: infos[0][key]
            for key in ("official_metrics", "q_score", "instance_id", "episode_length")
        }
        return observation, float(rewards[0]), bool(terms[0]), bool(truncs[0]), metrics

    def _observation(self, raw: dict[str, Any]) -> dict[str, Any]:
        raw, state, cameras = to_numpy_tree(
            (
                raw,
                self.simulator.get_robot_state(),
                self.simulator.get_camera_metadata(),
            )
        )
        name = state["robot_name"]
        sensors = raw[name]
        model_state = np.asarray(sensors["proprio"])
        if model_state.shape != (256,) or not np.isfinite(model_state).all():
            raise ValueError(
                "RLinf must supply its finite legacy256 model observation."
            )
        result = {
            "task_id": self.task_id,
            f"{name}::proprio": _public_proprio(state),
            # Model and tool states have different base-velocity conventions.
            "policy_proprio": model_state.copy(),
            "camera_metadata": {},
        }
        base_from_world = invert_transform(_pose_matrix(state["world_pose"]))
        poses = {}
        for role, suffix in CAMERA_SUFFIXES.items():
            camera = cameras[role]
            sensor_name = sensor_key(sensors, suffix)
            sensor = sensors[sensor_name]
            rgb = np.asarray(sensor["rgb"])
            if rgb.shape[:2] != (camera["height"], camera["width"]):
                raise ValueError(
                    f"{role} image and native camera calibration disagree."
                )
            for modality in ("rgb", "depth", "depth_linear"):
                if modality in sensor:
                    result[f"{name}::{sensor_name}::{modality}"] = sensor[modality]
            transform = compose_transforms(
                base_from_world, _pose_matrix(camera["world_pose"])
            )
            poses[role] = np.concatenate(
                (transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat())
            )
            depth_type = (
                "distance_to_image_plane"
                if "depth_linear" in sensor
                else "distance_to_camera"
                if "depth" in sensor
                else None
            )
            result["camera_metadata"][role] = {
                "camera_name": role,
                "height": camera["height"],
                "width": camera["width"],
                "pixel_order": "row_col",
                "camera_frame": "opengl",
                "target_frame": "robot_base",
                "camera_to_base_gl": transform.tolist(),
                "pose_xyz_xyzw": poses[role].tolist(),
                "pose_source": "rlinf.BehaviorSimulator.get_camera_metadata",
                "calibration_source": "native VisionSensor.intrinsic_matrix",
                "intrinsic_K": np.asarray(camera["intrinsic_matrix"]).tolist(),
                "calibration_available": True,
                "depth_type": depth_type,
                "depth_unit": "metre" if depth_type else None,
            }
        result[f"{name}::cam_rel_poses"] = np.concatenate(
            [poses[role] for role in R1PRO_EVAL_CAMERA_ORDER]
        )
        return result


class PublicBehaviorEnv(_BehaviorSession):
    """Serve one RLinf BehaviorEnv configured with the native 61-value R1Pro state."""

    def __init__(self, env: BehaviorEnv, *, task_id: int, task_language: str) -> None:
        if env.cfg.get("auto_reset", False) or env.cfg.get(
            "ignore_terminations", False
        ):
            raise ValueError(
                "RPent requires explicit reset and native termination flags."
            )
        super().__init__(
            env,
            task_id=task_id,
            task_language=task_language,
            frequency=float(env.cfg["omni_config"]["env"]["action_frequency"]),
        )

    def _observation(self, raw: dict[str, Any]) -> dict[str, Any]:
        raw = to_numpy_tree(raw)
        frame = native_policy_observation(
            raw, raw.get("task_descriptions", self.task_language)
        )
        frame["task_id"] = self.task_id
        if "main_depths" in raw or "extra_view_depths" in raw:
            for depth, rgb in (
                ("main_depths", "main_images"),
                ("extra_view_depths", "wrist_images"),
            ):
                if np.shape(raw[depth]) != frame[rgb].shape[:-1]:
                    raise ValueError(f"{depth} dimensions must match {rgb}.")
                frame[depth] = raw[depth]
        return frame

    @staticmethod
    def _episode_metrics(info: dict[str, Any]) -> dict[str, Any]:
        episode = {
            key: np.asarray(value).item()
            for key, value in to_numpy_tree(info["episode"]).items()
        }
        length = episode["episode_length"]
        if type(length) is not int or length < 0:
            raise ValueError("RLinf episode_length must be a nonnegative integer.")
        return {"episode": episode, "episode_length": length}

    def _reset_episode(self):
        raw, info = self.simulator.reset()
        return self._observation(raw), {}, self._episode_metrics(info)

    def _submit_action(self, action):
        # RLinf merges done flags at the end of a chunk. One native step per call
        # preserves cancellation, partial confirmation, and the terminal frame.
        return self.simulator.chunk_step(action[None, None, :])

    def _unpack_step(self, result):
        observations, rewards, terms, truncs, infos = result
        if len(observations) != 1 or len(infos) != 1:
            raise ValueError("RLinf must return one observation and one step info.")
        if any(np.shape(value) != (1, 1) for value in (rewards, terms, truncs)):
            raise ValueError("Expected one reward and one pair of terminal flags.")
        metrics = self._episode_metrics(infos[0])
        previous = self._metrics["episode_length"]
        delta = metrics["episode_length"] - previous
        # OG 3.9.3 fills episode_length before incrementing its counter. Reset
        # and the first successful step can both report zero.
        if delta != 1 and not (self._confirmed == 0 and delta == 0):
            raise ValueError(f"Unexpected RLinf episode_length increment {delta}.")
        return (
            self._observation(observations[0]),
            float(rewards[0, 0]),
            bool(terms[0, 0]),
            bool(truncs[0, 0]),
            metrics,
        )

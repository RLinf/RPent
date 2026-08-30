"""Primitive handlers for the standard-main BEHAVIOR toolkit."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from robots.behavior.schemas import (
    DEFAULT_ACTION_CHUNK,
    validate_action_chunk,
    validate_move_both_targets,
    validate_move_both_visual_hand_checks,
    validate_observe_request,
    validate_relative_navigation_motion,
)
from robots.behavior.task_specs import get_task_spec
from robots.behavior.terminal_success import (
    make_raw_success_receipt,
    official_success_receipt_from_info,
    official_task_success,
)
from rpent.tools.toolkit import readonly

_PRIVATE_RESULT_KEYS = {
    "_memory_source",
    "activity_instance_id",
    "ground_truth",
    "gt",
    "hidden_state",
    "native_instance",
    "private_environment_metadata",
    "simulator_state",
    "suggested_next_action",
    "suggested_next_tool",
}
_PUBLIC_IMAGE_BYTE_FIELDS = {
    "_image_bytes",
    "_depth_image_bytes",
    "_image_cam_bytes",
    "_image_wrist_bytes",
    "_image_nav_bytes",
}


def _jsonable(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sanitize_public_result(value: Any) -> Any:
    """Remove privileged diagnostics while preserving public image byte fields."""

    if isinstance(value, dict):
        public: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered == "info":
                public[str(key)] = _public_info_summary(item)
                continue
            if (
                lowered in _PRIVATE_RESULT_KEYS
                or lowered.startswith(("ground_truth_", "gt_", "private_"))
                or lowered.endswith("_ground_truth")
            ):
                continue
            if lowered in _PUBLIC_IMAGE_BYTE_FIELDS:
                public[str(key)] = None if item is None else bytes(item)
                continue
            public[str(key)] = _sanitize_public_result(item)
        return public
    if isinstance(value, (list, tuple)):
        return [_sanitize_public_result(item) for item in value]
    return _jsonable(value)


def _public_info_summary(info: Any) -> dict[str, Any]:
    if not isinstance(info, dict):
        return {}
    public: dict[str, Any] = {}
    if isinstance(info.get("done"), dict):
        public["done"] = _jsonable(info["done"])
    runtime = info.get("_rpent")
    if isinstance(runtime, dict):
        allowed = {
            "total_env_steps",
            "global_env_steps",
            "attempt_index",
            "attempt_nonce",
            "run_nonce",
            "official_success_receipt",
        }
        public["_rpent"] = {
            key: _jsonable(runtime[key])
            for key in allowed
            if key in runtime
        }
    return public


def _info_from_result(value: Any) -> dict[str, Any] | None:
    if isinstance(value, (tuple, list)) and len(value) == 5 and isinstance(value[4], dict):
        return value[4]
    if isinstance(value, dict):
        info = value.get("info")
        if isinstance(info, dict):
            return info
        if isinstance(value.get("done"), dict):
            return value
    return None


def _terminal_capture_pointer_from_info(info: Any) -> dict[str, Any] | None:
    """Reduce a terminal capture to public linkage fields only."""

    runtime = info.get("_rpent") if isinstance(info, dict) else None
    if not isinstance(runtime, dict):
        return None
    capture = runtime.get("terminal_capture")
    if not isinstance(capture, dict):
        capture = runtime.get("vla_after_capture")
    if not isinstance(capture, dict):
        return None
    group_id = capture.get("capture_group_id")
    step = capture.get("capture_env_step", capture.get("env_step", capture.get("simulator_step")))
    frame_ids = capture.get("frame_ids")
    if (
        not isinstance(group_id, str)
        or not group_id
        or isinstance(step, bool)
        or not isinstance(step, (int, np.integer))
        or not isinstance(frame_ids, dict)
    ):
        return None
    cameras = ("head", "left_wrist", "right_wrist")
    if any(not isinstance(frame_ids.get(camera), str) or not frame_ids[camera] for camera in cameras):
        return None
    return {
        "capture_group_id": group_id,
        "capture_env_step": int(step),
        "simulator_step": int(step),
        "frame_ids": {camera: frame_ids[camera] for camera in cameras},
    }


def _observation_summary(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {}
    summary: dict[str, Any] = {}
    for key, value in observation.items():
        if key in {"main_images", "wrist_images", "states"}:
            array = np.asarray(value)
            summary[key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        elif key == "task_descriptions":
            summary[key] = _jsonable(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            summary[str(key)] = value
    return summary


class BehaviorPrimitives:
    """Handlers registered by :class:`robots.behavior.toolkit.BehaviorToolkit`."""

    def __init__(
        self,
        *,
        env: Any = None,
        model: Any = None,
        max_episode_steps: int | None = None,
        output_dir: str | Path | None = None,
        video_path: str | Path | None = None,
        action_horizon: int = DEFAULT_ACTION_CHUNK,
        initial_observation: dict[str, Any] | None = None,
        initial_info: Any = None,
        progress_callback: Any = None,
        behavior_phase: str = "eval",
        task_name: str = "turning_on_radio",
        public_seed: int = 0,
        initial_attempt_index: int = 1,
        job_id: str | None = None,
        max_tool_calls: int | None = 350,
        max_wall_clock_s: float = 86400.0,
        pure_vla_baseline: bool = False,
        memory_index: Any = None,
        dino_component: Any = None,
        close_model_on_shutdown: bool = True,
        **_ignored: Any,
    ) -> None:
        self.env = env
        self.model = model
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.video_path = Path(video_path) if video_path else self.output_dir / "episode.mp4"
        self.action_horizon = int(action_horizon)
        self._current_observation = initial_observation
        self._current_info = initial_info if isinstance(initial_info, dict) else {}
        self.behavior_phase = str(behavior_phase)
        if self.behavior_phase not in {"eval", "explore"}:
            raise ValueError("behavior_phase must be 'eval' or 'explore'")
        self.task_spec = get_task_spec(str(task_name))
        self.task_name = self.task_spec.task_name
        self.public_seed = int(public_seed)
        self.task_spec.instance_for_public_seed(self.public_seed, phase=None)
        self.attempt_index = int(initial_attempt_index)
        if self.attempt_index < 1:
            raise ValueError("initial_attempt_index must be at least 1")
        self.job_id = str(job_id) if job_id is not None else None
        self.max_tool_calls = None if max_tool_calls is None else int(max_tool_calls)
        if self.max_tool_calls is not None and self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be positive")
        if not isinstance(pure_vla_baseline, bool):
            raise TypeError("pure_vla_baseline must be boolean")
        self.max_wall_clock_s = float(max_wall_clock_s)
        if not np.isfinite(self.max_wall_clock_s) or self.max_wall_clock_s <= 0.0:
            raise ValueError("max_wall_clock_s must be positive and finite")
        self.memory_index = memory_index
        self.dino_component = dino_component
        self._close_model_on_shutdown = bool(close_model_on_shutdown)
        self._episode_memory_decision = self._retrieve_episode_memory(
            self._current_observation
        )
        self._progress_callback = progress_callback
        self.started_monotonic = time.monotonic()
        self.last_result: dict[str, Any] | None = None
        self._local_env_steps = 0
        self._vla_invocations = 0
        self._vla_chunks = 0
        self._official_success_latched = official_task_success(self._current_info)
        self._official_success_receipt = (
            official_success_receipt_from_info(self._current_info)
            or make_raw_success_receipt(self._current_info, env_step=self.total_env_steps)
        )

    @property
    def elapsed_wall_clock_s(self) -> float:
        return max(0.0, time.monotonic() - self.started_monotonic)

    @property
    def total_env_steps(self) -> int:
        reported = getattr(self.env, "total_env_steps", None)
        if isinstance(reported, (int, np.integer)) and not isinstance(reported, (bool, np.bool_)):
            return max(self._local_env_steps, int(reported))
        return self._local_env_steps

    @property
    def current_observation(self) -> dict[str, Any] | None:
        return self._current_observation

    def solved(self) -> bool:
        env_solved = bool(getattr(self.env, "official_success_latched", False))
        return bool(self._official_success_latched or env_solved)

    def official_success_receipt(self) -> dict[str, Any] | None:
        env_receipt = getattr(self.env, "official_success_receipt", None)
        if isinstance(env_receipt, dict):
            return _jsonable(env_receipt)
        return _jsonable(self._official_success_receipt) if self._official_success_receipt else None

    def _remaining_steps(self) -> int | None:
        if self.max_episode_steps is None:
            return None
        return max(0, int(self.max_episode_steps) - self.total_env_steps)

    def _require_env(self) -> Any:
        if self.env is None:
            raise RuntimeError("BEHAVIOR env component is unavailable")
        return self.env

    def _require_model(self) -> Any:
        if self.model is None:
            raise RuntimeError("BEHAVIOR VLA component is unavailable")
        return self.model

    def _note_info(self, info: Any) -> None:
        if not isinstance(info, dict):
            return
        self._current_info = info
        runtime = info.get("_rpent")
        if isinstance(runtime, dict):
            steps = runtime.get("total_env_steps", runtime.get("global_env_steps"))
            if isinstance(steps, (int, np.integer)) and not isinstance(steps, (bool, np.bool_)):
                self._local_env_steps = max(self._local_env_steps, int(steps))
        if official_task_success(info):
            self._official_success_latched = True
            self._official_success_receipt = (
                official_success_receipt_from_info(info)
                or make_raw_success_receipt(info, env_step=self.total_env_steps)
            )

    @staticmethod
    def _rgb8(value: Any, *, first: int | None = None) -> np.ndarray | None:
        if value is None:
            return None
        image = np.asarray(value)
        if first is not None:
            if image.ndim != 4 or image.shape[0] <= first:
                return None
            image = image[first]
        elif image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[2] < 3:
            return None
        image = image[..., :3]
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.size and float(np.nanmax(image)) <= 1.0:
                image = np.rint(np.clip(image, 0.0, 1.0) * 255.0)
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    def _retrieve_episode_memory(self, observation: Any) -> dict[str, Any] | None:
        if self.memory_index is None or self.dino_component is None or not isinstance(observation, dict):
            return None
        head = self._rgb8(observation.get("main_images"))
        if head is None:
            return None
        wrists = observation.get("wrist_images")
        left = self._rgb8(wrists, first=0)
        right = self._rgb8(wrists, first=1)
        encoded = self.dino_component.encode_batch([head, left, right])
        head_embedding = encoded[0]
        if head_embedding is None:
            raise RuntimeError("DINO returned no head embedding for episode-memory retrieval")
        shadow = {
            channel: vector
            for channel, vector in zip(("left_wrist", "right_wrist"), encoded[1:])
            if vector is not None
        }
        decision = self.memory_index.retrieve(
            task_name=self.task_name,
            head_embedding=head_embedding,
            wrist_shadow_embeddings=shadow,
        )
        return _jsonable(decision)

    def _envelope(
        self,
        name: str,
        payload: Any,
        *,
        primitive_success: bool | None = None,
        stop_reason: str | None = None,
    ) -> dict[str, Any]:
        info = _info_from_result(payload)
        self._note_info(info)
        public_payload = _sanitize_public_result(payload)
        result: dict[str, Any] = {
            "name": name,
            "primitive_success": (
                bool(primitive_success)
                if primitive_success is not None
                else not (isinstance(public_payload, dict) and public_payload.get("error"))
            ),
            "task_success": self.solved(),
            "official_success_source": 'info["done"]["success"]',
            "total_env_steps": self.total_env_steps,
            "max_episode_steps": self.max_episode_steps,
        }
        if stop_reason is not None:
            result["stop_reason"] = stop_reason
        if isinstance(public_payload, dict):
            result.update(public_payload)
        else:
            result["value"] = public_payload
        if self._episode_memory_decision is not None:
            result["episode_memory"] = self._episode_memory_decision
        if self.solved():
            result["official_success_receipt"] = self.official_success_receipt()
            terminal_capture = _terminal_capture_pointer_from_info(self._current_info)
            if terminal_capture is not None:
                result["terminal_capture"] = terminal_capture
        self.last_result = result
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "public_seed": self.public_seed,
            "behavior_phase": self.behavior_phase,
            "task_success": self.solved(),
            "official_success_source": 'info["done"]["success"]',
            "official_success_receipt": self.official_success_receipt(),
            "total_env_steps": self.total_env_steps,
            "max_episode_steps": self.max_episode_steps,
            "elapsed_wall_clock_s": round(self.elapsed_wall_clock_s, 3),
            "observation": _observation_summary(self._current_observation),
            "episode_memory": self._episode_memory_decision,
        }

    def pi0_nav_pick(self, *, instruction: str, chunks: int) -> dict[str, Any]:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks <= 0:
            raise ValueError("chunks must be a positive integer")
        env = self._require_env()
        model = self._require_model()
        if self.solved():
            return self._envelope(
                "pi0_nav_pick",
                {},
                primitive_success=True,
                stop_reason="already_officially_successful",
            )

        started_steps = self.total_env_steps
        chunks_used = 0
        full_chunks = 0
        stop_reason = "exact_requested_chunks"
        last_info: dict[str, Any] | None = self._current_info
        started = time.monotonic()

        for chunk_index in range(chunks):
            remaining = self._remaining_steps()
            if remaining is not None and remaining <= 0:
                stop_reason = "episode_step_budget_exhausted"
                break
            if self._current_observation is None:
                self._current_observation, last_info = env.current_observation()
                self._note_info(last_info)
                if self.solved():
                    stop_reason = "official_task_success"
                    break
            env_obs = dict(self._current_observation)
            env_obs["task_descriptions"] = instruction.strip()
            actions, model_meta = model.predict_action_batch(env_obs, mode="eval")
            action_array = validate_action_chunk(actions)
            if remaining is not None:
                action_array = action_array[:remaining]
            if action_array.shape[0] <= 0:
                stop_reason = "episode_step_budget_exhausted"
                break
            ret = env.pi0_nav_pick_chunk_step(action_array, chunk_index=chunk_index)
            chunks_used += 1
            self._vla_invocations += 1
            self._vla_chunks += 1
            obs, _reward, terminated, truncated, info = ret
            if isinstance(obs, dict):
                self._current_observation = obs
            last_info = info if isinstance(info, dict) else {}
            self._note_info(last_info)
            monitor = last_info.get("_rpent", {}).get("pi0_nav_pick_monitor") if isinstance(last_info, dict) else None
            executed_steps = None
            if isinstance(monitor, dict):
                value = monitor.get("executed_steps")
                if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
                    executed_steps = int(value)
            if executed_steps is None:
                executed_steps = int(action_array.shape[0])
            self._local_env_steps = max(
                self._local_env_steps,
                started_steps + executed_steps,
            )
            if executed_steps >= int(action_array.shape[0]):
                full_chunks += 1
            if self.solved():
                stop_reason = "official_task_success"
                break
            if bool(terminated):
                stop_reason = "terminated"
                break
            if bool(truncated):
                stop_reason = "truncated"
                break
            if isinstance(model_meta, dict) and model_meta.get("warning"):
                stop_reason = str(model_meta["warning"])
                break

        env_steps_used = max(0, self.total_env_steps - started_steps)
        result = {
            "terminated": stop_reason == "terminated",
            "truncated": stop_reason == "truncated",
            "stop_reason": stop_reason,
            "requested_chunks": int(chunks),
            "chunks_used": chunks_used,
            "full_chunks_executed": full_chunks,
            "exact_requested_chunks_completed": chunks_used == int(chunks) and stop_reason == "exact_requested_chunks",
            "env_steps_used": env_steps_used,
            "total_env_steps": self.total_env_steps,
            "max_episode_steps": self.max_episode_steps,
            "action_horizon": self.action_horizon,
            "required_action_shape": [None, 23],
            "elapsed_s": round(time.monotonic() - started, 3),
            "info": last_info or {},
        }
        return self._envelope(
            "pi0_nav_pick",
            result,
            primitive_success=bool(chunks_used > 0 or self.solved()),
            stop_reason=stop_reason,
        )

    @readonly
    def observe(self, **kwargs: Any) -> dict[str, Any]:
        env = self._require_env()
        request = validate_observe_request(**kwargs)
        result = env.observe(**request)
        info = _info_from_result(result)
        if info is not None:
            self._note_info(info)
        return self._envelope("observe", result, primitive_success=True)

    @readonly
    def pixel_to_world(self, **kwargs: Any) -> dict[str, Any]:
        env = self._require_env()
        result = env.pixel_to_world(**kwargs)
        return self._envelope("pixel_to_world", result, primitive_success=True)

    def navigate_to(self, **kwargs: Any) -> dict[str, Any]:
        env = self._require_env()
        if "relative_motion" in kwargs and kwargs["relative_motion"] is not None:
            kwargs = {**kwargs, "relative_motion": validate_relative_navigation_motion(kwargs["relative_motion"])}
        return self._envelope("navigate_to", env.navigate_to(**kwargs))

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        env = self._require_env()
        return self._envelope("move_to", env.move_to(**kwargs))

    def move_both_to(self, **kwargs: Any) -> dict[str, Any]:
        env = self._require_env()
        kwargs = {
            **kwargs,
            "targets": validate_move_both_targets(kwargs.get("targets")),
            "visual_hand_checks": validate_move_both_visual_hand_checks(kwargs.get("visual_hand_checks")),
        }
        return self._envelope("move_both_to", env.move_both_to(**kwargs))

    @readonly
    def get_prepared_motion_status(self, **kwargs: Any) -> dict[str, Any]:
        env = self._require_env()
        return self._envelope(
            "get_prepared_motion_status",
            env.get_prepared_motion_status(**kwargs),
            primitive_success=True,
        )

    def rotate_wrist(self, **kwargs: Any) -> dict[str, Any]:
        env = self._require_env()
        return self._envelope("rotate_wrist", env.rotate_wrist(**kwargs))

    def close(self, **kwargs: Any) -> dict[str, Any]:
        env = self._require_env()
        return self._envelope("close", env.close(**kwargs))

    def open(self, **kwargs: Any) -> dict[str, Any]:
        env = self._require_env()
        return self._envelope("open", env.open(**kwargs))

    def press(self, **kwargs: Any) -> dict[str, Any]:
        env = self._require_env()
        return self._envelope("press", env.press(**kwargs))

    def save_robot_state_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
        env = self._require_env()
        return self._envelope(
            "save_robot_state_checkpoint",
            env.save_robot_state_checkpoint(**kwargs),
            primitive_success=True,
            stop_reason=kwargs.get("stop_reason"),
        )

    @readonly
    def finish(self, *, status: str, summary: str) -> dict[str, Any]:
        if not isinstance(status, str) or not status.strip():
            raise ValueError("status must be a non-empty string")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("summary must be a non-empty string")
        receipt = {
            "schema_version": 1,
            "kind": "behavior_finish_terminal_receipt",
            "planner_status": status.strip(),
            "summary": summary.strip(),
            "task_success": self.solved(),
            "official_success_source": 'info["done"]["success"]',
            "official_success_receipt": self.official_success_receipt(),
            "total_env_steps": self.total_env_steps,
            "max_episode_steps": self.max_episode_steps,
        }
        result = {"_finish": True, **receipt}
        self.last_result = result
        return result

    def shutdown(self) -> None:
        candidates = [self.env]
        if self._close_model_on_shutdown:
            candidates.insert(0, self.model)
        for candidate in candidates:
            if candidate is None:
                continue
            closer = getattr(candidate, "close_transport", None)
            if not callable(closer):
                closer = getattr(candidate, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass

    def recipe_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if self.last_result is None:
            return records
        records.append(
            {
                "task_name": self.task_name,
                "public_seed": self.public_seed,
                "last_result": _sanitize_public_result(self.last_result),
            }
        )
        return json.loads(json.dumps(records, default=str))


__all__ = [
    "BehaviorPrimitives",
    "_sanitize_public_result",
    "_terminal_capture_pointer_from_info",
    "official_task_success",
]

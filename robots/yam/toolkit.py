# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RPent tools for the real YAM robot."""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robots.yam import tools
from robots.yam.contracts import MODEL_SPEC, YAM_CAMERA_NAMES
from robots.yam.primitives import YamPrimitives
from robots.yam.projection import world_from_depth as _world_from_depth_cv
from robots.yam.tasks import classify_episode
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, readonly
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager


_RECIPE_ACTIONS = {
    "pi05_act",
    "move_to",
    "rotate_wrist",
    "set_gripper",
    "release",
}


class YamToolkit(Toolkit):
    _OPERATOR_WAIT_S = 20.0
    _SPECS = {spec["name"]: spec for spec in tools.TOOLS_SPEC}
    _FRAME_ARTIFACTS = {
        "top": "top_rgb.png",
        "left": "left_rgb.png",
        "right": "right_rgb.png",
    }

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        mode: str = "evaluation",
        attempts_per_session: int = 0,
        state_output_dir: Path | str | None = None,
        run_output_dir: Path | str | None = None,
    ) -> None:
        if mode not in {"evaluation", "exploration"}:
            raise ValueError(f"unsupported YAM toolkit mode: {mode!r}")
        self._state_output_dir = Path(state_output_dir or get_output_dir())
        state = EnvState(self._state_output_dir)
        super().__init__(dashboard_events=dashboard_events, state=state, memory=memory)
        self._mode = mode
        self._attempt = 1
        self._attempts_per_session = max(0, int(attempts_per_session))
        self._session_attempt = 0
        self._run_output_dir = Path(
            run_output_dir or state_output_dir or get_output_dir()
        )
        self._continuation_key = None
        self._continuation_count = 0
        self._latest_status: dict[str, Any] = {}
        self._primitives = YamPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        self._register_yam_tools()
        self.get_env_state(
            command={"action": "observe"},
            result={"success": True},
            elapsed_s=0.0,
        )
        record = self._state.latest_record()
        if record is not None:
            self._publish_step(record)
        # A stopped diagnostic episode may retain its consumed ready receipt.
        # Only adopting a live, continuable episode spends the first attempt.
        self._session_attempt = int(
            classify_episode(self._latest_status)["can_continue"]
        )

    def _register_yam_tools(self) -> None:
        self.add_tool("status", self._SPECS["status"], self.status)
        self._tools.pop("finish", None)
        self.add_tool(
            "view_env_state",
            self._SPECS["view_env_state"],
            partial(tools.view_env_state, state=self._state),
        )
        self.add_tool(
            "sample_world_xyz",
            self._SPECS["sample_world_xyz"],
            partial(tools.sample_world_xyz, self._state),
        )
        self.add_tool(
            "query_world_map",
            self._SPECS["query_world_map"],
            partial(tools.query_world_map, self._state),
        )
        for name in (
            "render",
            "move_to",
            "rotate_wrist",
            "set_gripper",
            "release",
        ):
            self.add_tool(name, self._SPECS[name], partial(self._step, name))
        if self._primitives.model is not None:
            self.add_tool(
                "pi05_act", self._SPECS["pi05_act"], partial(self._step, "pi05_act")
            )
        if self._mode == "exploration":
            self.add_tool("reset", self._SPECS["reset"], self._reset_episode)
        self.add_tool("finish", self._SPECS["finish"], self._finish)

    @readonly
    def status(self) -> dict[str, Any]:
        """Read current episode authority; readiness is not visual clearance."""
        self._latest_status = self._primitives.status()
        return self._latest_status

    def exploration_continuation(self, *, explicit: bool = False) -> str | None:
        if self._mode != "exploration":
            return None
        record = self._state.latest_record()
        if record is not None and record.command.get("action") == "finish":
            return None
        status = self.status()
        if not status["can_continue"]:
            return None
        key = (status["episode_id"], status["take_action_cnt"])
        if explicit or key != self._continuation_key:
            self._continuation_key, self._continuation_count = key, 0
        # Bound text/perception-only loops without spending another episode.
        if self._continuation_count >= 3:
            return None
        self._continuation_count += 1
        return (
            "Program-level explore continuation: current episode is ready, not "
            "terminal, and has action budget. Preserve this scene and grasp; "
            "refresh views and read official memory. Choose a supported recovery "
            "or task action; a rejected plan or recoverable residual alone is not "
            "an episode verdict. Use meaningful measured-pose geometry or VLA "
            "with a suitable horizon. Readiness does not establish "
            "payload clearance: never force uncertain contact or auto-release. "
            "If no supported action exists, record the concrete missing evidence. "
            "Do not reset, fabricate success, or clear a stop."
        )

    def _wait_for_operator(self, event: str) -> dict[str, Any]:
        deadline = time.monotonic() + self._OPERATOR_WAIT_S
        while True:
            self.raise_if_cancelled()
            status = self.status()
            if status.get("terminal_event") == "abort":
                return status
            if event == "ready" and status.get("ready_to_reset"):
                return status
            if event == "verdict" and (
                status.get("eval_success") is True
                or status.get("terminal_event") == "failure"
            ):
                return status
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return status
            time.sleep(min(0.2, remaining))

    def _finish(self, *, status: str, summary: str) -> dict[str, Any]:
        status = status.strip().lower()
        if status != "success":
            return {"_finish": True, "status": status, "summary": summary}
        if self.status().get("eval_success") is True:
            return {"_finish": True, "status": "success", "summary": summary}
        self._primitives.env.request_stop()
        verdict = self._wait_for_operator("verdict")
        if verdict.get("eval_success") is True:
            if self._mode != "exploration":
                return {"_finish": True, "status": "success", "summary": summary}
            return {
                "error": "finish refused: review newly confirmed success",
                "status": "retry",
                "notice": "Operator confirmed success. Read the current evidence, distil the technique to memory inbox, then call finish again.",
            }
        if verdict.get("terminal_event") == "abort":
            return {
                "_finish": True,
                "status": "failure",
                "summary": "Operator aborted.",
            }
        if verdict.get("terminal_event") == "failure":
            if (
                self._attempts_per_session
                and self._session_attempt >= self._attempts_per_session
            ):
                return {"_finish": True, "status": "failure", "summary": summary}
            return {
                "error": "finish refused: operator marked failure",
                "status": "retry",
                "notice": "Record the lesson, then reset after the scene is ready.",
            }
        return {
            "error": "finish refused: pending operator verdict",
            "status": "pending",
            "notice": "Call finish again; this has not ended the session.",
        }

    def _reset_episode(self) -> dict[str, Any]:
        budget = self._attempts_per_session
        if budget and self._session_attempt >= budget:
            return {
                "error": "reset refused",
                "reason": f"This session's attempt budget is spent ({budget} attempts).",
            }
        ready = self._wait_for_operator("ready")
        if ready.get("terminal_event") == "abort":
            return {
                "error": "Operator aborted; call finish(status='failure') now.",
                "status": "failure",
            }
        if not ready.get("ready_to_reset"):
            return {
                "status": "pending",
                "notice": "Waiting for operator ready. Call reset again; the attempt count is unchanged.",
            }
        self._save_episode_video()
        result = self._primitives.reset()
        # The initial ready/reset opens attempt 1, rather than spending it.
        if self._session_attempt:
            self._attempt += 1
        self._session_attempt += 1
        self._latest_status = result.get("episode_status", {})
        result["attempt"] = self._attempt
        result["notice"] = (
            "YAM episode reset completed; re-run perception before acting."
        )
        return result

    def _capture_full_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        env = self._primitives.env
        obs, info = env.observe()
        views: dict[str, dict[str, Any]] = {}
        for camera_name in YAM_CAMERA_NAMES:
            source = obs.get("views", {}).get(camera_name)
            if not isinstance(source, dict):
                raise TypeError(f"YAM observe.views is missing {camera_name!r}")
            rgb = source.get("rgb")
            depth = source.get("depth")
            camera_meta = source.get("camera_meta")
            if rgb is None or depth is None or not isinstance(camera_meta, dict):
                raise TypeError(
                    f"YAM observe.views[{camera_name!r}] must contain rgb, depth, camera_meta"
                )
            projection_error = None
            try:
                world = _world_from_depth_cv(depth, camera_meta)
            except (ValueError, RuntimeError) as error:
                world = np.full((*np.asarray(depth).shape, 3), np.nan, dtype=np.float32)
                projection_error = str(error)
            views[camera_name] = {
                "rgb": np.asarray(rgb),
                "depth": np.asarray(depth, dtype=np.float32),
                "world_xyz": world,
                "camera_meta": camera_meta,
            }
            if projection_error:
                views[camera_name]["world_xyz_limitation"] = projection_error
        return {
            "views": views,
            "robot_state": info["robot_state"],
            "task_name": env.server_meta["task_name"],
            "task_language": env.get_task_language(),
            "depth_unit": "metres",
            "world_frame": "left_base",
            "snapshot_id": obs.get("snapshot_id"),
        }, info["episode_status"]

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        observation, status = self._capture_full_observation()
        self._latest_status = status
        record = tools.dump_observation(
            observation,
            env_state=self._state,
            status=status,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        captured = tools.view_env_state(record.step_idx, state=self._state)
        if result.get("error") or result.get("status") in {"pending", "retry"}:
            captured.update(result)
        if result.get("_finish"):
            # Toolkit.execute_tool replaces stateful handler output with this
            # capture. Keep the planner termination signal at the top level.
            verified = status.get("eval_success") is True
            requested_status = str(result.get("status", "failure"))
            captured.update(
                {
                    **result,
                    "requested_status": requested_status,
                    "requested_success": requested_status.lower() == "success",
                    "verified_success": verified,
                    "episode_status": status,
                    "status": "success" if verified else "failure",
                }
            )
        return captured

    def cancel_active_and_wait(self) -> None:
        with self._operation_lock:
            operation = self._active_operation
            if operation is None:
                return
            operation.cancel_event.set()
        self._primitives.env.request_stop()
        operation.done_event.wait()

    def _step(self, name: str, **kwargs) -> dict[str, Any]:
        self.raise_if_cancelled()
        if name == "render":
            return {"success": True}
        return getattr(self._primitives, name)(**kwargs)

    def close(self) -> None:
        stop_error = None
        try:
            self._primitives.env.request_stop()
        except Exception as error:
            stop_error = error
        self._save_episode_video()
        if stop_error:
            raise RuntimeError(
                "Could not deliver YAM stop request during toolkit close"
            ) from stop_error

    def _save_episode_video(self) -> None:
        frames = self._primitives.stop_recording()
        if frames:
            self._state.save(
                f"episode_{self._attempt:03d}.mp4",
                frames,
                step=None,
                fps=MODEL_SPEC.control_hz,
            )

    def solved(self) -> bool:
        return self.status().get("eval_success") is True

    def write_recipe(self, recipe_tag: str) -> str:
        if not self.solved():
            return ""
        episode_id = self._latest_status["episode_id"]
        successful_records = [
            record
            for record in self._state.records()
            if record.state["episode_status"]["episode_id"] == episode_id
        ]
        if not any(record.terminated for record in successful_records):
            return ""
        recipe = []
        for record in successful_records:
            command = record.command
            result = record.result
            if (
                not isinstance(command, dict)
                or command.get("action") not in _RECIPE_ACTIONS
            ):
                continue
            if isinstance(result, dict) and (
                result.get("error")
                or (
                    result.get("success") is False
                    and not (
                        command.get("action") == "move_to"
                        and result.get("recoverable") is True
                        and result.get("executed_steps", 0) > 0
                    )
                )
            ):
                continue
            recipe.append(command)
        artifacts = EnvState(self._run_output_dir)
        name = f"{recipe_tag}_recipe.jsonl"
        if artifacts.save(name, recipe, step=None) is None:
            raise RuntimeError("failed to save YAM recipe")
        audit = {
            "robot": "yam",
            "recipe_tag": recipe_tag,
            "task_name": successful_records[-1].state.get("task_name"),
            "task_language": successful_records[-1].state.get("task_language"),
            "solved": True,
            "source": "operator_backed_eval_success",
            "episode_status": dict(self._latest_status),
            "terminal_step": successful_records[-1].step_idx,
            "recipe_actions": len(recipe),
            "session_state_dir": str(self._state_output_dir),
            "coordinate_scope": "episode-local; re-localize before reuse",
        }
        if artifacts.save(f"{recipe_tag}.json", audit, step=None) is None:
            raise RuntimeError("failed to save YAM audit")
        return str(artifacts.artifact_path(name, step=None))

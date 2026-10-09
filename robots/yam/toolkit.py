# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RPent tools for the real YAM robot."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from pydantic import Field

from robots.yam import tools
from robots.yam.contracts import MODEL_SPEC, YAM_CAMERA_NAMES
from robots.yam.primitives import YamPrimitives
from robots.yam.projection import world_from_depth as _world_from_depth_cv
from robots.yam.tasks import classify_episode
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, tool
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


def tracking_hold_operator_supervised(
    mode: str, operator_config_path: Path | str | None
) -> bool:
    """Use a human for attended tracking holds unless site YAML opts out."""
    if mode != "exploration":
        return False
    if operator_config_path is None:
        return True
    from robots.yam.runtime_config import load_mapping

    site = load_mapping(operator_config_path)
    enabled = site.get("tracking_hold_operator_input", True)
    return enabled if type(enabled) is bool else True


class YamToolkit(Toolkit):
    _FRAME_ARTIFACTS = {
        "top": "top_rgb.png",
        "left": "left_rgb.png",
        "right": "right_rgb.png",
    }

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        mode: str = "evaluation",
        attempts_per_session: int = 0,
        state_output_dir: Path | str | None = None,
        run_output_dir: Path | str | None = None,
        operator_config_path: Path | str | None = None,
        dashboard_language: str = "en",
    ) -> None:
        if mode not in {"evaluation", "exploration"}:
            raise ValueError(f"unsupported YAM toolkit mode: {mode!r}")
        self._state_output_dir = Path(state_output_dir or get_output_dir())
        state = EnvState(self._state_output_dir)
        super().__init__(dashboard_events=dashboard_events, state=state, memory=memory)
        self._mode = mode
        self._dashboard_language = dashboard_language
        self._attempt = 1
        self._attempts_per_session = max(0, int(attempts_per_session))
        self._session_attempt = 0
        self._run_output_dir = Path(
            run_output_dir or state_output_dir or get_output_dir()
        )
        self._continuation_key = None
        self._continuation_count = 0
        self._operator_waiting_episode: str | None = None
        self._latest_status: dict[str, Any] = {}
        self._primitives = YamPrimitives(
            check_cancelled=self.raise_if_cancelled,
            operator_supervised=tracking_hold_operator_supervised(
                mode, operator_config_path
            ),
            **runtime_kwargs,
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
        self.add_tool(self.status)
        for definition in (
            tools.view_env_state,
            tools.sample_world_xyz,
            tools.query_world_map,
        ):
            handler = (
                partial(definition, state=self._state)
                if definition.name == "view_env_state"
                else partial(definition, self._state)
            )
            self.add_tool(definition.with_handler(handler))
        for definition in (
            self.render,
            self.move_to,
            self.rotate_wrist,
            self.set_gripper,
            self.release,
        ):
            self.add_tool(definition)
        if self._primitives.model is not None:
            self.add_tool(self.pi05_act)
        if self._mode == "exploration":
            self.add_tool(self._reset_episode)
        self.add_tool(self._finish, replace=True)

    @tool(readonly=True, json_schema_extra={"additionalProperties": False})
    def status(self) -> ToolResult:
        """Read fresh episode rules: ready, awaiting ready, terminal, stop or budget. Does not establish visual clearance or drive hardware."""
        self._latest_status = self._primitives.status()
        payload = self._latest_status
        return ToolResult(data=payload)

    def exploration_continuation(self, *, explicit: bool = False) -> str | None:
        if self._mode != "exploration":
            return None
        record = self._state.latest_record()
        if record is not None and record.command.get("action") == "finish":
            result = getattr(record, "result", None)
            finished = isinstance(result, dict) and result.get("_finish")
            if finished or not explicit:
                return None
        status = self.status().data
        if explicit and status["reason"] == "success":
            return (
                "The onsite operator confirmed success for this episode. This is "
                "a terminal verdict, not permission to move. Do not command the "
                "robot or reset the episode. Review the recorded evidence, write "
                "the supported lesson to the memory inbox, then call "
                "finish(status='success') with an accurate summary."
            )
        if explicit and status["reason"] == "abort":
            return (
                "The onsite operator aborted this episode. This is a terminal "
                "verdict, not permission to move or reset. Record the abort "
                "evidence and call finish(status='failure') with an accurate summary."
            )
        if (
            explicit
            and status["reason"] == "failure"
            and self._attempts_per_session
            and self._session_attempt >= self._attempts_per_session
        ):
            return (
                "The onsite operator confirmed failure for this episode, "
                "and the session attempt budget is spent. This is a terminal "
                "verdict, not permission to move. Record the failure evidence "
                "and call finish(status='failure') with an accurate summary."
            )
        if (
            explicit
            and self._attempts_per_session
            and self._session_attempt >= self._attempts_per_session
        ):
            return (
                "The session attempt budget is spent. Do not call reset or command "
                "motion. Ask the onsite operator for a formal verdict if one is "
                "still missing, then finish with the verified outcome."
            )
        if explicit and status.get("ready_to_reset") is True:
            return (
                "The onsite operator supplied a ready receipt for this episode. "
                "This is permission to call reset for episode bookkeeping, not "
                "permission to command motion. Call reset, then inspect fresh "
                "views and measured state before planning any action."
            )
        if explicit and status["reason"] == "failure":
            return (
                "The onsite operator marked this attempt as failed. This is "
                "not permission to move or clear the stop. Record the concrete "
                "failure lesson, ask the operator to prepare the next scene, "
                "then wait for a matching ready receipt and call reset."
            )
        if not status["can_continue"]:
            return None
        if not explicit and self._operator_waiting_episode == status["episode_id"]:
            return None
        key = (status["episode_id"], status["take_action_cnt"])
        if explicit or key != self._continuation_key:
            self._continuation_key, self._continuation_count = key, 0
        if explicit:
            self._operator_waiting_episode = None
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
            "in fixed five-step chunks. Readiness does not establish "
            "payload clearance: never force uncertain contact or auto-release. "
            "If no supported action exists, record the concrete missing evidence. "
            "Do not reset, fabricate success, or clear a stop."
        )

    @tool(name="finish")
    def _finish(self, *, status: str, summary: str) -> ToolResult:
        """Stop the run. Fresh env eval_success is authoritative."""
        status = status.strip().lower()
        verdict = self.status().data
        if verdict.get("eval_success") is True:
            payload = {"_finish": True, "status": "success", "summary": summary}
            return ToolResult(data=payload)
        if verdict.get("terminal_event") == "abort":
            payload = {
                "_finish": True,
                "status": "failure",
                "summary": "Operator aborted.",
            }
            return ToolResult(data=payload)
        if verdict.get("terminal_event") == "failure":
            if (
                self._mode == "exploration"
                and self._attempts_per_session
                and self._session_attempt < self._attempts_per_session
            ):
                payload = {
                    "error": "finish refused: operator marked failure",
                    "status": "retry",
                    "notice": "Record the lesson, then reset after the scene is ready.",
                }
                return ToolResult(data=payload)
            payload = {"_finish": True, "status": "failure", "summary": summary}
            return ToolResult(data=payload)
        if status != "success":
            if self._mode == "exploration":
                payload = {
                    "status": "pending",
                    "awaiting_operator": True,
                    "operator_question": "The agent is blocked. Continue after onsite handling, or formally end this attempt?",
                    "notice": "Ask about the current blocker and wait. A planner claim is not an operator verdict.",
                }
                return ToolResult(data=payload)
            payload = {"_finish": True, "status": status, "summary": summary}
            return ToolResult(data=payload)
        payload = {
            "error": "finish refused: pending operator verdict",
            "status": "pending",
            "awaiting_operator": True,
            "operator_question": "Is the current task complete? Confirm success or explain what remains.",
            "notice": "Ask the operator and wait. Do not poll finish; a current-episode verdict is required.",
        }
        return ToolResult(data=payload)

    @tool(name="reset")
    def _reset_episode(self) -> ToolResult:
        """Operator-approved reset for exploration or a fresh task attempt."""
        budget = self._attempts_per_session
        if budget and self._session_attempt >= budget:
            payload = {
                "error": "reset refused",
                "reason": f"This session's attempt budget is spent ({budget} attempts).",
            }
            return ToolResult(data=payload)
        ready = self.status().data
        if ready.get("terminal_event") == "abort":
            payload = {
                "error": "Operator aborted; call finish(status='failure') now.",
                "status": "failure",
            }
            return ToolResult(data=payload)
        if not ready.get("ready_to_reset"):
            payload = {
                "status": "pending",
                "awaiting_operator": True,
                "operator_question": "Is the scene ready and the held object handled for the next attempt?",
                "notice": "Wait for a matching ready receipt. Do not poll reset; no attempt was spent.",
            }
            return ToolResult(data=payload)
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
        payload = result
        return ToolResult(data=payload)

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
    ) -> ToolResult:
        observation, status = self._capture_full_observation()
        self._latest_status = status
        if result.get("awaiting_operator") or result.get("operator_input_required"):
            self._operator_waiting_episode = status.get("episode_id")
        record = tools.dump_observation(
            observation,
            env_state=self._state,
            status=status,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        captured = tools.view_env_state(record.step_idx, state=self._state)
        if result.get("error") or result.get("status") in {"pending", "retry"}:
            captured.data.update(result)
        if result.get("operator_input_required"):
            captured.data.update(
                {
                    "awaiting_operator": True,
                    "operator_question": (
                        "The arm is holding at its measured pose. Give the next direction "
                        "and distance, or say whether a named gripper may open."
                    ),
                    "notice": "Ask the onsite operator and wait. This tracking residual is not an episode verdict.",
                }
            )
        if result.get("_finish"):
            # Toolkit.execute_tool replaces stateful handler output with this
            # capture. Keep the planner termination signal at the top level.
            verified = status.get("eval_success") is True
            requested_status = str(result.get("status", "failure"))
            captured.data.update(
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
        try:
            self._primitives.env.request_stop()
        finally:
            operation.done_event.wait()

    @tool
    def render(self) -> ToolResult:
        """Capture a fresh synchronized YAM observation."""
        return self._step("render")

    @tool
    def move_to(
        self,
        *,
        arm: Literal["left", "right"],
        xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
        xyz_bounds: Annotated[
            list[Annotated[list[float], Field(min_length=3, max_length=3)]],
            Field(min_length=2, max_length=2),
        ]
        | None = None,
        quat: Annotated[list[float], Field(min_length=4, max_length=4)] | None = None,
        gripper: float | None = None,
        substeps: Annotated[int, Field(ge=0, json_schema_extra={"default": 25})] = 25,
    ) -> ToolResult:
        """Plan and move one YAM arm to a left-base xyz and wxyz orientation.

        Args:
            xyz_bounds: Optional task-valid [lower_xyz, upper_xyz] containing xyz. Try at most three plans within it; execute only one. Reobserve after recoverable motion failure before choosing another point. Does not relax arrival tolerance or collision checks."""
        return self._step(
            "move_to",
            arm=arm,
            xyz=xyz,
            xyz_bounds=xyz_bounds,
            quat=quat,
            gripper=gripper,
            substeps=substeps,
        )

    @tool
    def rotate_wrist(
        self,
        *,
        arm: Literal["left", "right"],
        delta_yaw_deg: float,
        gripper: float | None = None,
        substeps: Annotated[int, Field(ge=0, json_schema_extra={"default": 25})] = 25,
    ) -> ToolResult:
        """Rotate one wrist about world Z by a relative angle in degrees."""
        return self._step(
            "rotate_wrist",
            arm=arm,
            delta_yaw_deg=delta_yaw_deg,
            gripper=gripper,
            substeps=substeps,
        )

    @tool
    def set_gripper(
        self,
        *,
        arm: Literal["left", "right"],
        val: Annotated[float, Field(ge=0, le=1)],
        steps: Annotated[int, Field(ge=1, json_schema_extra={"default": 10})] = 10,
    ) -> ToolResult:
        """Linearly move one normalized YAM gripper, where 0=closed and 1=open."""
        return self._step("set_gripper", arm=arm, val=val, steps=steps)

    @tool
    def release(
        self,
        *,
        arm: Literal["left", "right"],
        val: Annotated[float, Field(json_schema_extra={"default": 1.0})] = 1.0,
        steps: Annotated[int, Field(ge=1, json_schema_extra={"default": 10})] = 10,
    ) -> ToolResult:
        """Open one YAM gripper to 1.0."""
        return self._step("release", arm=arm, val=val, steps=steps)

    @tool
    def pi05_act(
        self,
        *,
        chunks: Annotated[int, Field(ge=1, json_schema_extra={"default": 1})] = 1,
        use_length: Annotated[Literal[5], Field(json_schema_extra={"default": 5})] = 5,
        prompt: str | None = None,
    ) -> ToolResult:
        """Run the YAM Pi0.5 qpos14 policy for one or more short chunks."""
        return self._step(
            "pi05_act", chunks=chunks, use_length=use_length, prompt=prompt
        )

    def _step(self, name: str, **kwargs) -> ToolResult:
        self.raise_if_cancelled()
        if name == "render":
            payload = {"success": True}
            return ToolResult(data=payload)
        payload = getattr(self._primitives, name)(**kwargs)
        return ToolResult(data=payload)

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
        return self.status().data.get("eval_success") is True

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
            if isinstance(result, dict):
                executed = result.get("executed_steps", result.get("executed_actions"))
                if result.get("motion_refused") or executed == 0:
                    continue
            # Keep partial and uncertain execution in the winning attempt's
            # trace. A failed call does not establish that no motion occurred.
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

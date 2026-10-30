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

"""RoboDojo actions composed over the environment RPC boundary."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Annotated, Literal

import numpy as np
from pydantic import Field

from robots.robodojo.access import public_episode_budget
from robots.robodojo.env_client import ActionRejected, ExecutionSyncError
from robots.robodojo.tasks import validate_instruction
from rpent.tools import ToolResult, tool
from rpent.tools.toolkit import ToolCancelled
from rpent.utils.logging import get_logger

logger = get_logger(__name__)


def _qmult(left, right):
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


class RoboDojoPrimitives:
    """Execute bounded actions through the environment RPC interface."""

    def __init__(
        self,
        *,
        env,
        check_cancelled: Callable[[], None],
        sam3_client=None,
        vla_client=None,
        vla_backend="vpp2",
        action_type="joint",
    ):
        self.env = env
        self._check_cancelled = check_cancelled
        self.sam3_client = sam3_client
        self.vla_client = vla_client
        self.action_type = action_type
        self._last_obs = None
        self._segment_index = 0
        self._segment_index_lock = Lock()
        self._uses_awomo05 = vla_backend == "awomo05" and vla_client is not None
        self._awomo05_session_active = False
        self._closed = False
        if self._uses_awomo05:
            try:
                env.register_policy_observer(self._awomo05_execution_event)
            except Exception:
                try:
                    self.close()
                except Exception:
                    logger.warning(
                        "Failed to release Awomo05 after startup error", exc_info=True
                    )
                raise

    def close(self) -> None:
        """Release execution state; Runtime retains environment and model connections."""
        if self._closed:
            return
        try:
            if self._uses_awomo05:
                self.env.unregister_policy_observer(self._awomo05_execution_event)
        finally:
            self._closed = True
            if self._uses_awomo05:
                self._end_awomo05_episode()

    def reserve_segment_index(self) -> int:
        """Allocate a unique artifact index within this trace."""
        with self._segment_index_lock:
            index = self._segment_index
            self._segment_index += 1
        return index

    @staticmethod
    def _public_status(status: dict) -> dict:
        result = public_episode_budget(status)
        if "success" in status:
            result["success"] = status["success"]
        return result

    def _execute(self, updates: list[dict], *, policy_actions: bool = False) -> dict:
        if self._closed:
            raise RuntimeError("Execution was released")
        executed = 0
        status = self.env.get_status()
        reason = None
        error = None
        uncertain = False
        episode_ended = self.env.episode_ended
        for update in updates:
            if (
                episode_ended
                or status.get("success") is True
                or status["step"] >= status["step_limit"]
            ):
                break
            try:
                self._check_cancelled()
                if policy_actions:
                    _, _, _, info = self.env.step(update)
                    result = {**info, "episode_status": info["status"]}
                else:
                    result = self.env.step_joint_update(**update)
            except ToolCancelled as exc:
                reason, error = "cancelled", str(exc)
                self.env.pause_execution()
                break
            except ActionRejected as exc:
                reason, error = "rejected", str(exc)
                break
            except ExecutionSyncError as exc:
                if exc.feedback is not None:
                    executed += exc.feedback["executed_actions"]
                    status = exc.feedback["episode_status"]
                    episode_ended = (
                        exc.feedback["terminated"] or exc.feedback["truncated"]
                    )
                uncertain = exc.execution_uncertain
                reason, error = (
                    ("execution_unknown" if uncertain else "synchronization_lost"),
                    str(exc),
                )
                break
            except Exception as exc:
                # A timed-out mutation may have executed remotely. Do not retry
                # it or claim that only the acknowledged actions took place.
                reason = "execution_unknown"
                error = str(exc)
                uncertain = True
                break
            count = int(result["executed_actions"])
            executed += count
            status = result["episode_status"]
            episode_ended = result["terminated"] or result["truncated"]
            if episode_ended:
                break
            if count != 1:
                reason, error = "runtime_failure", "Native action was not executed"
                break
        completed = (
            executed == len(updates) and error is None and reason != "runtime_failure"
        )
        if reason is None:
            reason = (
                "native_success"
                if status.get("success") is True
                else "budget_exhausted"
                if status["step"] >= status["step_limit"]
                else "episode_end"
                if episode_ended
                else "completed"
            )
        return {
            "success": error is None and reason != "runtime_failure",
            "completed": completed,
            "requested_steps": len(updates),
            "executed_steps": executed,
            "requested_actions": len(updates),
            "executed_actions": executed,
            "stop_reason": reason,
            "episode_status": self._public_status(status),
            "episode_ended": episode_ended
            or status.get("success") is True
            or status["step"] >= status["step_limit"],
            **({"error": error, "execution_uncertain": uncertain} if error else {}),
            **(
                {"code": "tool_cancelled", "interrupted": True}
                if reason == "cancelled"
                else {}
            ),
        }

    def _execute_policy(
        self, *, chunks: int, replan_steps: int, infer: Callable[[], list[dict]]
    ) -> dict:
        if type(chunks) is not int or chunks < 1:
            raise ValueError("chunks must be a positive integer")
        result = self._execute([])
        executed = completed_chunks = 0
        for _ in range(chunks):
            if result["stop_reason"] != "completed":
                break
            try:
                self._check_cancelled()
                self.env.require_policy_ready()
                actions = infer()
            except ToolCancelled as exc:
                self.env.pause_execution()
                result.update(
                    success=False,
                    stop_reason="cancelled",
                    error=str(exc),
                    code="tool_cancelled",
                    interrupted=True,
                )
                break
            except ExecutionSyncError as exc:
                result.update(
                    success=False,
                    stop_reason="synchronization_lost",
                    error=str(exc),
                    execution_uncertain=exc.execution_uncertain,
                )
                break
            except Exception as exc:
                result.update(
                    success=False, stop_reason="inference_failed", error=str(exc)
                )
                break
            try:
                result = self._execute(actions, policy_actions=True)
            except Exception as exc:
                result.update(
                    success=False, stop_reason="runtime_failure", error=str(exc)
                )
                break
            executed += result["executed_actions"]
            completed_chunks += int(result["completed"])
            if not result["completed"]:
                break
        result.update(
            requested_chunks=chunks,
            executed_chunks=completed_chunks,
            requested_steps=chunks * replan_steps,
            requested_actions=chunks * replan_steps,
            executed_steps=executed,
            executed_actions=executed,
            completed=completed_chunks == chunks and "error" not in result,
        )
        return result

    @tool
    def vpp2_act(
        self,
        *,
        instruction: Annotated[str, Field(min_length=1)],
        chunks: Annotated[int, Field(ge=1)] = 1,
    ) -> ToolResult:
        """Execute a natural-language subtask with VPP2. Each chunk predicts 32 native dual-arm EE targets and executes the first 24, then replans from real feedback. Completion means commands executed, not subtask success. Inspect fresh images afterward."""
        from robots.robodojo.vpp2_client import REPLAN_STEPS

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a nonempty subtask string")
        if self.vla_client is None:
            raise RuntimeError("VPP2 is not connected")
        result = self._execute_policy(
            chunks=chunks,
            replan_steps=REPLAN_STEPS,
            infer=lambda: self.vla_client.infer(
                self.env.policy_observation(), instruction
            ),
        )
        result["instruction"] = instruction
        return ToolResult(data=result)

    @tool
    def awomo05_act(
        self,
        *,
        chunks: Annotated[int, Field(ge=1)] = 1,
    ) -> ToolResult:
        """Execute the official task instruction with Awomo05. Each chunk executes 8 native dual-arm joint targets with per-step feedback, then replans. Completion means commands executed, not task success. Inspect fresh images afterward."""
        from robots.robodojo.awomo05_client import REPLAN_STEPS

        if not self._uses_awomo05:
            raise RuntimeError("Awomo05 is not connected")

        def infer():
            try:
                return self.vla_client.get_action()
            except Exception as exc:
                self.env.invalidate_execution()
                raise ExecutionSyncError(f"policy inference uncertain: {exc}") from exc

        result = self._execute_policy(
            chunks=chunks,
            replan_steps=REPLAN_STEPS,
            infer=infer,
        )
        return ToolResult(data=result)

    def _awomo05_execution_event(self, event: dict) -> None:
        """Consume each confirmed observation once, without maintaining model history."""
        kind = event["kind"]
        if kind in ("attached", "reset"):
            self._awomo05_origin_step = event["status"]["step"]
            self._awomo05_instruction = validate_instruction(
                event["observation"]["instruction"]
            )
            # Even a failed reset response may have acquired the remote session.
            self._awomo05_session_active = True
            self.vla_client.reset_session()
            self._observe_awomo05(event["observation"], 0)
        elif kind == "action_feedback" and event["executed_actions"]:
            # Official Awomo deploy.py excludes the terminal observation.
            if not (event["terminated"] or event["truncated"]):
                self._observe_awomo05(
                    event["observation"],
                    event["step_after"] - self._awomo05_origin_step,
                )
        elif kind == "closed":
            self._end_awomo05_episode()

    def _observe_awomo05(self, obs: dict, step: int) -> None:
        self.vla_client.update_obs(
            {**obs, "instruction": self._awomo05_instruction}, step
        )

    def _end_awomo05_episode(self) -> None:
        if self._awomo05_session_active:
            try:
                self.vla_client.end_episode()
            finally:
                self._awomo05_session_active = False

    @tool
    def move_to(
        self,
        *,
        arm: Literal["left", "right"],
        xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
        quat: Annotated[list[float], Field(min_length=4, max_length=4)] | None = None,
        gripper: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None,
        substeps: Annotated[int, Field(ge=0)] = 25,
    ) -> ToolResult:
        """Plan a world-metre xyz and wxyz orientation. Omitted orientation/opening is held. substeps caps sampled waypoints (0: full path, 1: final only), not physics steps. Sampling and the native table-only obstacle model do not guarantee scene-wide collision avoidance."""
        if type(substeps) is not int or substeps < 0:
            raise ValueError("substeps must be a nonnegative integer")
        if gripper is not None and (not np.isfinite(gripper) or not 0 <= gripper <= 1):
            raise ValueError("gripper must be finite and in [0,1]")
        self._check_cancelled()
        if quat is None:
            quat = self.env.get_motion_state()[f"{arm}_eef_pose"][3:]
        planned = self.env.plan_arm_path(arm, [*xyz, *quat])
        self._check_cancelled()
        if planned["status"] != "Success":
            return ToolResult(
                data={
                    "success": False,
                    "completed": False,
                    "requested_steps": 0,
                    "executed_steps": 0,
                    "stop_reason": "plan_failed",
                    "plan_status": planned["status"],
                }
            )
        path = np.asarray(planned["position"], dtype=float)
        if substeps == 1:
            path = path[-1:]
        elif substeps >= 2 and len(path) > substeps:
            path = path[np.linspace(0, len(path) - 1, substeps).astype(int)]
        result = self._execute(
            [{"arm": arm, "arm_qpos": q.tolist(), "gripper": gripper} for q in path]
        )
        result.update(plan_status=planned["status"], waypoints=len(path))
        try:
            final = np.asarray(self.env.get_motion_state()[f"{arm}_eef_pose"][:3])
            result.update(
                final_eef_xyz=final.tolist(),
                final_dist_m=float(np.linalg.norm(final - np.asarray(xyz))),
            )
        except Exception as exc:
            result["feedback_error"] = str(exc)
        return ToolResult(data=result)

    @tool
    def rotate_wrist(
        self,
        *,
        arm: Literal["left", "right"],
        delta_yaw_deg: float,
        gripper: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None,
        substeps: Annotated[int, Field(ge=0)] = 25,
    ) -> ToolResult:
        """Rotate the end effector about world Z by a relative angle in degrees."""
        if not np.isfinite(delta_yaw_deg):
            raise ValueError("delta_yaw_deg must be finite")
        self._check_cancelled()
        pose = self.env.get_motion_state()[f"{arm}_eef_pose"]
        half_yaw = np.deg2rad(delta_yaw_deg) / 2
        quat = _qmult([np.cos(half_yaw), 0, 0, np.sin(half_yaw)], pose[3:])
        result = self.move_to(
            arm=arm,
            xyz=pose[:3],
            quat=quat.tolist(),
            gripper=gripper,
            substeps=substeps,
        )
        result.data["requested_delta_yaw_deg"] = delta_yaw_deg
        return result

    @tool
    def set_gripper(
        self,
        *,
        arm: Literal["left", "right"],
        val: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)],
        steps: Annotated[int, Field(ge=1)] = 10,
    ) -> ToolResult:
        """Move one gripper from measured opening to val (0 closed, 1 open). steps counts gradual native action targets, not physics steps. Other controller targets are retained."""
        if not np.isfinite(val) or not 0 <= val <= 1:
            raise ValueError("val must be finite and in [0,1]")
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        self._check_cancelled()
        measured = float(self.env.get_motion_state()[f"{arm}_gripper"])
        if not np.isfinite(measured):
            raise ValueError("measured gripper state is not finite")
        # Physical overshoot is possible. Saturate the interpolation start,
        # never the user's input; keep the actual measurement in feedback.
        current = float(np.clip(measured, 0, 1))
        values = np.linspace(current, val, steps + 1)[1:]
        result = self._execute([{"arm": arm, "gripper": float(v)} for v in values])
        try:
            result["gripper_val"] = self.env.get_motion_state()[f"{arm}_gripper"]
        except Exception as exc:
            result["feedback_error"] = str(exc)
        return ToolResult(data=result)

    @tool
    def release(
        self,
        *,
        arm: Literal["left", "right"],
        val: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 1.0,
        steps: Annotated[int, Field(ge=1)] = 10,
    ) -> ToolResult:
        """Open one gripper, by default to 1.0 over ten action targets."""
        return self.set_gripper(arm=arm, val=val, steps=steps)

    @tool(readonly=True)
    def finish(self, *, status: str, summary: str) -> ToolResult:
        """Stop the run; the environment, not the requested status, judges task success."""
        try:
            native = self._public_status(self.env.get_status())
        except Exception as exc:
            return ToolResult(
                data={
                    "_finish": True,
                    "status": "error",
                    "summary": summary,
                    "runtime_error": str(exc),
                }
            )
        success = native.get("success")
        reported = (
            "success"
            if success is True
            else "failure"
            if status.lower() == "success"
            else status
        )
        return ToolResult(
            data={
                "_finish": True,
                "status": reported,
                "summary": summary,
                "requested_success": status.lower() == "success",
                "success": success,
                "episode_status": native,
            }
        )

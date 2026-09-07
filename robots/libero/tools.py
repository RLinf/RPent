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

"""LIBERO + OpenPI tool implementation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
from pydantic import Field

from rpent.tools import (
    ToolContext,
    ToolResult,
    parallel,
    readonly,
    tool,
)

if TYPE_CHECKING:
    from robots.libero.toolkit import LiberoRuntime


def _step_env(ctx: ToolContext[LiberoRuntime], action) -> None:
    """Submit and count one physical action through the recording path."""
    ctx.check_cancelled()
    runtime = ctx.robot
    obs, _r, _t, _tr, _i = runtime.env.step(action)
    runtime.executed_steps += 1
    runtime.set_obs(obs)
    ctx.record_frame(obs["main_images"])


def _vlm_chunk(ctx: ToolContext[LiberoRuntime], instruction: str):
    """One model forward + ``chunk_size`` env steps. Overrides prompt."""
    runtime = ctx.robot
    ctx.check_cancelled()
    original_task = runtime._last_obs.get("task_descriptions")
    try:
        runtime._last_obs["task_descriptions"] = instruction
        runtime._last_obs.setdefault("extra_view_images", None)

        actions = runtime.model.predict(runtime._last_obs, options={"mode": "eval"})
        ctx.check_cancelled()

        chunk_obs, _r, _t, _tr, _i = runtime.env.chunk_step(
            actions, return_all_frames=True
        )
        for obs in chunk_obs:
            ctx.record_frame(obs["main_images"])
        runtime.executed_steps += int(np.asarray(_t).size)
        runtime.set_obs(chunk_obs[-1])
        return runtime._last_obs
    finally:
        if original_task is not None:
            runtime._last_obs["task_descriptions"] = original_task


@tool
@readonly
def finish(status: str, summary: str, *, ctx: ToolContext[LiberoRuntime]) -> ToolResult:
    """Call when the task is complete or unrecoverable. Halts the agent loop. Save any artifacts (recipe, audit) BEFORE calling finish.

    Args:
        status: Outcome, e.g. 'success', 'failure', or 'stuck'.
        summary: Short natural-language summary of the run.
    """
    runtime = ctx.robot
    budget = runtime.attempts_per_session
    if (
        runtime.mode == "exploration"
        and budget
        and not runtime.solved
        and runtime.attempt < budget
    ):
        remaining = budget - runtime.attempt
        return ToolResult(
            error=f"This session has {remaining} of its {budget} attempts left and the task is not solved. Archive this attempt, call `reset`, and try another approach."
        )
    return ToolResult(data={"_finish": True, "status": status, "summary": summary})


@tool
def reset(reason: str, *, ctx: ToolContext[LiberoRuntime]) -> ToolResult:
    """EXPLORE MODE ONLY. Abandon the current episode and restore the same initial scene. Archive the failed attempt first and state which strategy lever will change in the next attempt.

    Args:
        reason: Why this episode is unrecoverable and what will change.
    """
    runtime = ctx.robot
    budget = runtime.attempts_per_session
    if budget and runtime.attempt >= budget:
        return ToolResult(
            error=f"This session's attempt budget is spent ({budget} attempts). Archive the attempt, update the handoff notes, and call `finish` so the next session can continue."
        )
    ctx.check_cancelled()
    runtime.attempt += 1
    runtime.reset()
    return ToolResult(
        data={
            "action": "reset",
            "reason": reason,
            "libero_terminated": runtime.env.terminated or runtime.env.truncated,
            "attempt": runtime.attempt,
            "notice": f"Episode restarted; this is attempt {runtime.attempt}. The original "
            "layout was restored. Re-run perception before acting.",
        }
    )


@tool
@readonly
@parallel
def view_env_state(
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    ctx: ToolContext[LiberoRuntime],
) -> ToolResult:
    """Read one recorded state and its observation artifacts. Step -1 selects the latest entry. Embeds policy, agentview, and wrist images when available. Use the calibration-frame images for pixel back-projection; JSON state alone is not enough. Use agentview for global tabletop layout and object locations; use wrist for close-range details near the gripper, occlusions, and container/cabinet interiors.

    Args:
        step: Step number; 0 = initial, -1 = latest.
    """
    # Toolkit imports these tool definitions while assembling the session.
    from robots.libero.toolkit import build_observation

    try:
        record = ctx.state.get(step)
    except Exception as exc:
        return ToolResult(error=f"state step not available: {exc}")
    data, images = build_observation(ctx.state, record)
    return ToolResult(data=data, images=images)


@tool
def move_to(
    xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    gripper: float = -1.0,
    tol: float = 0.012,
    step_clip: float = 0.025,
    max_steps: int = 80,
    action_scale: float = 0.05,
    target_yaw: float | None = None,
    yaw_step_clip: float = 0.1,
    *,
    ctx: ToolContext[LiberoRuntime],
) -> ToolResult:
    """Scripted EEF servo to a world-frame XYZ target via the OSC controller. Holds orientation (use rotate_wrist / rotate_pitch / move_pose to reorient). gripper: -1 = open, +1 = close. NEVER command a single move_to with |Δxy| > 0.30 — OSC flips IK and the run corrupts; split long traversal into 2-3 mid waypoints at carry z.

    Args:
        xyz: World-frame target [x, y, z] in meters
        gripper: Gripper command: -1 open, +1 close (default -1)
        tol: Position tolerance, m (default 0.012)
        step_clip: Per-step Δxyz cap before action_scale, m (default 0.025)
        max_steps: Step budget (default 80)
        action_scale: OSC action scale (default 0.05)
        target_yaw: Optional world-frame yaw target in radians
        yaw_step_clip: Per-step yaw clip, rad (default 0.10)
    """
    # Sends 7-D delta actions; the env's underlying OSC_POSE controller
    # interprets ``action[:3] ∈ [-1, 1]`` as a per-step desired delta scaled
    # by ``action_scale`` (so ``action=1.0`` -> ~5 cm per env step).
    runtime = ctx.robot
    started = runtime.executed_steps
    target = np.asarray(xyz, dtype=np.float32)
    for _ in range(max_steps):
        cur = runtime._last_obs_eef_pos
        diff = target - cur
        dist = float(np.linalg.norm(diff))
        if dist < tol:
            break
        step_dxyz = np.clip(diff, -step_clip, step_clip)
        action = np.zeros(7, dtype=np.float32)
        action[:3] = step_dxyz / action_scale  # -> roughly [-0.5, 0.5]
        action[:3] = np.clip(action[:3], -1.0, 1.0)
        if target_yaw is not None:
            # add wrist yaw control via action[5] (z-axis axis-angle).
            # NOTE: extract world yaw via atan2(R[1,0], R[0,0]), NOT
            # as_euler('zyx')[0] — the latter returns -world_yaw for
            # gripper-down configs (R[2,2]≈-1) and silently flips the
            # commanded rotation direction. See feedback_rotate_wrist_yaw_sign.
            from scipy.spatial.transform import Rotation as _R

            q = runtime.env.raw_obs()["robot0_eef_quat"]
            _R_mat = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
            cur_yaw = float(np.arctan2(_R_mat[1, 0], _R_mat[0, 0]))
            err = (float(target_yaw) - cur_yaw + np.pi) % (2 * np.pi) - np.pi
            step_dyaw = float(np.clip(err, -yaw_step_clip, yaw_step_clip))
            action[5] = float(np.clip(step_dyaw / 0.10, -1.0, 1.0))
        action[6] = gripper
        _step_env(ctx, action)
        if runtime.env.terminated or runtime.env.truncated:
            break
    final = runtime._last_obs_eef_pos
    return ToolResult(
        data={
            "name": "move_to",
            "target_xyz": [float(x) for x in target],
            "final_eef_pos": [round(float(x), 4) for x in final],
            "final_dist_m": round(float(np.linalg.norm(target - final)), 4),
            "steps_used": runtime.executed_steps - started,
            "max_steps": max_steps,
            "terminated": runtime.env.terminated,
            "truncated": runtime.env.truncated,
        }
    )


@tool
def pi0_pick(
    prompt: str,
    max_chunks: int = 24,
    lift_thresh: float = 0.05,
    gripper_closed_thresh: float = 0.06,
    *,
    ctx: ToolContext[LiberoRuntime],
) -> ToolResult:
    """Pi0.5 closed-loop pick. Use it for the grasp; YOU then do every move_to and release. Use modest max_chunks and verify the grasp from EEF lift, gripper closure, and available images.

    Args:
        prompt: Pi0 prompt (e.g. 'pick up the akita black bowl').
        max_chunks: Action-chunk budget (default 24)
        lift_thresh: EEF post-descent ascent threshold for success, m (default 0.05)
        gripper_closed_thresh: Finger-separation closed threshold (default 0.06)
    """
    runtime = ctx.robot
    start_z = float(runtime._last_obs_eef_pos[2])
    peak_z = start_z
    min_z = start_z
    # Track ascent AFTER min_z has been observed — descent then re-ascent
    # is the actual "lift" signal, distinct from raw |peak - min| which
    # also fires at the BOTTOM of the descent.
    post_min_peak_z = start_z
    min_grip = runtime._last_obs_gripper
    last_grip = min_grip
    descent_done = False
    success = False
    chunks_used = 0

    for c in range(max_chunks):
        _vlm_chunk(ctx, prompt)
        chunks_used = c + 1
        z = float(runtime._last_obs_eef_pos[2])
        grip = runtime._last_obs_gripper
        peak_z = max(peak_z, z)
        if z < min_z:
            min_z = z
            post_min_peak_z = z  # reset after a new deeper min
        else:
            post_min_peak_z = max(post_min_peak_z, z)
        if (start_z - min_z) >= 0.10:  # descended ≥ 10 cm — committed to grasp
            descent_done = True
        min_grip = min(min_grip, grip)
        last_grip = grip
        ascended = (post_min_peak_z - min_z) >= lift_thresh
        closed = grip < gripper_closed_thresh
        if descent_done and ascended and closed:
            success = True
            break
        if runtime.env.terminated or runtime.env.truncated:
            success = runtime.env.terminated
            break

    return ToolResult(
        data={
            "name": "pick",
            "instruction": prompt,
            "success": success,
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "peak_lift_m": post_min_peak_z - min_z,  # actual post-descent ascent
            "min_gripper_opening": min_grip,
            "final_gripper_opening": last_grip,
            "terminated": runtime.env.terminated,
            "truncated": runtime.env.truncated,
            "diagnostics": {
                "start_eef_z": round(start_z, 4),
                "peak_eef_z": round(peak_z, 4),
                "min_eef_z": round(min_z, 4),
                "post_min_peak_z": round(post_min_peak_z, 4),
                "descent_m": round(start_z - min_z, 4),
                "post_min_ascent_m": round(post_min_peak_z - min_z, 4),
                "descent_done": descent_done,
                "lift_thresh": lift_thresh,
                "gripper_closed_thresh": gripper_closed_thresh,
            },
        }
    )


@tool
def pi0_doubled(
    prompt: str, max_chunks: int = 20, *, ctx: ToolContext[LiberoRuntime]
) -> ToolResult:
    """Pi0.5 closed-loop contact skill for non-pick interactions (e.g. stove/knob/button/short push). Returned success/task_success only mirrors official termination; for intermediate contact skills, success=false does not necessarily mean the contact interaction failed. Inspect image/state evidence. Do not use it as a general pick/place shortcut.

    Args:
        prompt: Contact-skill prompt, e.g. 'turn on the stove'.
        max_chunks: Action-chunk budget (default 20)
    """
    runtime = ctx.robot
    task_success = False
    chunks_used = 0

    for c in range(max_chunks):
        _vlm_chunk(ctx, prompt)
        chunks_used = c + 1
        if runtime.env.terminated or runtime.env.truncated:
            task_success = runtime.env.terminated
            break

    return ToolResult(
        data={
            "name": "pi0_doubled",
            "instruction": prompt,
            "success": task_success,
            "task_success": task_success,
            "contact_skill_executed": chunks_used > 0,
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "terminated": runtime.env.terminated,
            "truncated": runtime.env.truncated,
            "diagnostics": {
                "mode": "contact_skill_success_by_termination",
                "success_meaning": (
                    "`success` mirrors official LIBERO task termination only; "
                    "for intermediate contact skills, inspect image/state evidence."
                ),
            },
        }
    )


@tool
def release(max_steps: int = 20, *, ctx: ToolContext[LiberoRuntime]) -> ToolResult:
    """Open the gripper for up to max_steps env steps while holding EEF in place. Triggers libero termination if the matching On/In predicate is met.

    Args:
        max_steps: Step budget (default 20)
    """
    runtime = ctx.robot
    started = runtime.executed_steps
    assert max_steps > 0, f"max_steps must be > 0, got {max_steps}"
    start_grip = runtime._last_obs_gripper
    peak_grip = start_grip
    for _ in range(max_steps):
        action = np.zeros(7, dtype=np.float32)
        action[6] = -1.0  # open
        _step_env(ctx, action)
        peak_grip = max(peak_grip, runtime._last_obs_gripper)
        if runtime.env.terminated or runtime.env.truncated:
            break
    return ToolResult(
        data={
            "name": "release",
            "steps_used": runtime.executed_steps - started,
            "start_gripper_opening": round(start_grip, 4),
            "peak_gripper_opening": round(peak_grip, 4),
            "final_gripper_opening": round(runtime._last_obs_gripper, 4),
            "terminated": runtime.env.terminated,
            "truncated": runtime.env.truncated,
        }
    )


@tool
def set_gripper(
    gripper: float = -1.0, steps: int = 5, *, ctx: ToolContext[LiberoRuntime]
) -> ToolResult:
    """Hold the current EEF pose and drive the gripper command for `steps` env steps. Use to firm up a grip mid-carry.

    Args:
        gripper: Gripper command: -1 open, +1 close (default -1)
        steps: Number of env steps (default 5)
    """
    runtime = ctx.robot
    started = runtime.executed_steps
    for _ in range(steps):
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper
        _step_env(ctx, action)
        if runtime.env.terminated or runtime.env.truncated:
            break
    return ToolResult(
        data={
            "name": "set_gripper",
            "gripper": gripper,
            "steps": runtime.executed_steps - started,
            "terminated": runtime.env.terminated,
            "truncated": runtime.env.truncated,
        }
    )


@tool
def rotate_wrist(
    target_yaw: float | None = None,
    delta_yaw: float | None = None,
    gripper: float = 1.0,
    max_steps: int = 40,
    tol: float = 0.02,
    step_clip: float = 0.1,
    *,
    ctx: ToolContext[LiberoRuntime],
) -> ToolResult:
    """Rotate the wrist around the world Z-axis. Provide either target_yaw (absolute) or delta_yaw (relative). Holds xyz fixed.

    Args:
        target_yaw: Absolute world-frame yaw target, rad
        delta_yaw: Relative yaw delta, rad
        gripper: Gripper command held during rotation (default +1)
        max_steps: Step budget (default 40)
        tol: Yaw tolerance, rad (default 0.02)
        step_clip: Per-step yaw clip, rad (default 0.10)
    """
    # Uses ``action[5]`` (axis-angle z component) to drive wrist yaw via the
    # OSC controller. Holds xyz pose constant during rotation.
    from scipy.spatial.transform import Rotation as _R

    runtime = ctx.robot
    started = runtime.executed_steps

    def _yaw_of(quat_xyzw):
        # robot0_eef_quat in libero+robosuite is xyzw (scipy convention).
        q = quat_xyzw
        rot = _R.from_quat([q[0], q[1], q[2], q[3]])
        R = rot.as_matrix()
        # World-frame yaw: angle of the eef x-axis projected onto the
        # world xy plane. Robust to gripper-down (R[2,2]≈-1) which is
        # where the euler 'zyx' chart flips sign.
        return float(np.arctan2(R[1, 0], R[0, 0]))

    raw = runtime.env.raw_obs()
    cur_quat = raw["robot0_eef_quat"]
    start_yaw = _yaw_of(cur_quat)
    if target_yaw is None and delta_yaw is None:
        return ToolResult(
            data={"name": "rotate_wrist"}, error="need target_yaw or delta_yaw"
        )
    if target_yaw is None:
        target_yaw = start_yaw + float(delta_yaw)

    for _ in range(max_steps):
        raw = runtime.env.raw_obs()
        cur_yaw = _yaw_of(raw["robot0_eef_quat"])
        err = float(target_yaw - cur_yaw)
        # wrap to [-pi, pi]
        err = (err + np.pi) % (2 * np.pi) - np.pi
        if abs(err) < tol:
            break
        step_dyaw = float(np.clip(err, -step_clip, step_clip))
        action = np.zeros(7, dtype=np.float32)
        action[5] = step_dyaw / 0.10  # scale to ~[-1,1] action range
        action[5] = float(np.clip(action[5], -1.0, 1.0))
        action[6] = gripper
        _step_env(ctx, action)
        if runtime.env.terminated or runtime.env.truncated:
            break
    final_yaw = _yaw_of(runtime.env.raw_obs()["robot0_eef_quat"])
    return ToolResult(
        data={
            "name": "rotate_wrist",
            "start_yaw": round(start_yaw, 4),
            "target_yaw": round(float(target_yaw), 4),
            "final_yaw": round(final_yaw, 4),
            "final_err": round(
                float((target_yaw - final_yaw + np.pi) % (2 * np.pi) - np.pi), 4
            ),
            "steps_used": runtime.executed_steps - started,
            "terminated": runtime.env.terminated,
            "truncated": runtime.env.truncated,
        }
    )


@tool
def rotate_pitch(
    target_pitch: float | None = None,
    delta_pitch: float | None = None,
    gripper: float = 1.0,
    max_steps: int = 40,
    tol: float = 0.02,
    step_clip: float = 0.1,
    *,
    ctx: ToolContext[LiberoRuntime],
) -> ToolResult:
    """Tilt the gripper around the world X-axis. Provide either target_pitch (absolute) or delta_pitch (relative). Holds xyz and yaw fixed. Use before threading the gripper into a narrow opening whose front face normal is along world ±y (e.g. microwave cavity).

    Args:
        target_pitch: Absolute world-frame pitch target, rad
        delta_pitch: Relative pitch delta, rad
        gripper: Gripper command held during rotation (default +1)
        max_steps: Step budget (default 40)
        tol: Pitch tolerance, rad (default 0.02)
        step_clip: Per-step pitch clip, rad (default 0.10)
    """
    # Pitch is defined as the angle between the eef z-axis and the
    # world -z direction, measured in the world yz-plane:
    #
    #     pitch = atan2(R[1, 2], -R[2, 2])
    #
    # - pitch =  0       -> gripper z-axis aligned with world -z (default
    #                       "gripper down" rest pose).
    # - pitch = +pi/2    -> gripper z-axis points in world +y (gripper
    #                       "looking forward" along world +y).
    # - pitch = -pi/2    -> gripper z-axis points in world -y.
    #
    # Driven by ``action[3]`` (axis-angle X component) of the OSC_POSE
    # controller. Sign verified empirically (probe_pitch.py 2026-05-19):
    # action[3]=+1.0 tilts eef z toward world +y, matching this pitch
    # definition with no sign flip.
    from scipy.spatial.transform import Rotation as _R

    runtime = ctx.robot
    started = runtime.executed_steps

    def _pitch_of(quat_xyzw):
        q = quat_xyzw
        R = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
        return float(np.arctan2(R[1, 2], -R[2, 2]))

    raw = runtime.env.raw_obs()
    start_pitch = _pitch_of(raw["robot0_eef_quat"])
    if target_pitch is None and delta_pitch is None:
        return ToolResult(
            data={"name": "rotate_pitch"}, error="need target_pitch or delta_pitch"
        )
    if target_pitch is None:
        target_pitch = start_pitch + float(delta_pitch)

    for _ in range(max_steps):
        raw = runtime.env.raw_obs()
        cur_pitch = _pitch_of(raw["robot0_eef_quat"])
        err = float(target_pitch - cur_pitch)
        err = (err + np.pi) % (2 * np.pi) - np.pi
        if abs(err) < tol:
            break
        step_dpitch = float(np.clip(err, -step_clip, step_clip))
        action = np.zeros(7, dtype=np.float32)
        action[3] = step_dpitch / 0.10
        action[3] = float(np.clip(action[3], -1.0, 1.0))
        action[6] = gripper
        _step_env(ctx, action)
        if runtime.env.terminated or runtime.env.truncated:
            break
    final_pitch = _pitch_of(runtime.env.raw_obs()["robot0_eef_quat"])
    return ToolResult(
        data={
            "name": "rotate_pitch",
            "start_pitch": round(start_pitch, 4),
            "target_pitch": round(float(target_pitch), 4),
            "final_pitch": round(final_pitch, 4),
            "final_err": round(
                float((target_pitch - final_pitch + np.pi) % (2 * np.pi) - np.pi), 4
            ),
            "steps_used": runtime.executed_steps - started,
            "terminated": runtime.env.terminated,
            "truncated": runtime.env.truncated,
        }
    )


@tool
def move_pose(
    xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    target_pitch: float | None = None,
    target_yaw: float | None = None,
    gripper: float = -1.0,
    step_clip: float = 0.02,
    pitch_step: float = 0.08,
    yaw_step: float = 0.08,
    tol: float = 0.012,
    ori_tol: float = 0.05,
    action_scale: float = 0.05,
    max_steps: int = 150,
    *,
    ctx: ToolContext[LiberoRuntime],
) -> ToolResult:
    """Servo position AND orientation (pitch + yaw) SIMULTANEOUSLY. Unlike move_to (holds orientation) + rotate_pitch (holds xyz), this co-varies xyz and wrist tilt every env.step. Use to thread cabinet-front / low-shelf poses where a decoupled position servo drives the wrist into an IK singularity and stalls.

    Args:
        xyz: World-frame target [x, y, z] in meters
        target_pitch: Absolute pitch target, rad
        target_yaw: Absolute yaw target, rad
        gripper: Gripper command held during the move (default -1)
        step_clip: Per-step Δxyz cap, m (default 0.02)
        pitch_step: Per-step pitch clip, rad (default 0.08)
        yaw_step: Per-step yaw clip, rad (default 0.08)
        tol: Position tolerance, m (default 0.012)
        ori_tol: Orientation tolerance, rad (default 0.05)
        action_scale: OSC action scale (default 0.05)
        max_steps: Step budget (default 150)
    """
    from scipy.spatial.transform import Rotation as _R

    runtime = ctx.robot
    started = runtime.executed_steps

    def _pitch_of(q):
        R = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
        return float(np.arctan2(R[1, 2], -R[2, 2]))

    def _yaw_of(q):
        R = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
        return float(np.arctan2(R[1, 0], R[0, 0]))

    target = np.asarray(xyz, dtype=np.float32)
    for _ in range(max_steps):
        cur = runtime._last_obs_eef_pos
        q = runtime.env.raw_obs()["robot0_eef_quat"]
        diff = target - cur
        dist = float(np.linalg.norm(diff))
        p_err = (
            0.0
            if target_pitch is None
            else float((target_pitch - _pitch_of(q) + np.pi) % (2 * np.pi) - np.pi)
        )
        y_err = (
            0.0
            if target_yaw is None
            else float((target_yaw - _yaw_of(q) + np.pi) % (2 * np.pi) - np.pi)
        )
        if dist < tol and abs(p_err) < ori_tol and abs(y_err) < ori_tol:
            break
        action = np.zeros(7, dtype=np.float32)
        sd = np.clip(diff, -step_clip, step_clip)
        action[:3] = np.clip(sd / action_scale, -1.0, 1.0)
        action[3] = float(
            np.clip(np.clip(p_err, -pitch_step, pitch_step) / 0.10, -1.0, 1.0)
        )
        action[5] = float(
            np.clip(np.clip(y_err, -yaw_step, yaw_step) / 0.10, -1.0, 1.0)
        )
        action[6] = gripper
        _step_env(ctx, action)
        if runtime.env.terminated or runtime.env.truncated:
            break
    final = runtime._last_obs_eef_pos
    fq = runtime.env.raw_obs()["robot0_eef_quat"]
    return ToolResult(
        data={
            "name": "move_pose",
            "final_eef_pos": [round(float(x), 4) for x in final],
            "final_dist_m": round(float(np.linalg.norm(target - final)), 4),
            "final_pitch": round(_pitch_of(fq), 4),
            "steps_used": runtime.executed_steps - started,
            "terminated": runtime.env.terminated,
            "truncated": runtime.env.truncated,
        }
    )


@tool
@readonly
@parallel
def view_camera_meta(
    camera: Literal["agentview", "wrist"] = "agentview",
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    ctx: ToolContext[LiberoRuntime],
) -> ToolResult:
    """Read per-step camera calibration metadata from recorded artifacts.

    Args:
        camera: Camera metadata to read (default agentview).
        step: Metadata step to use; -1 = latest.
    """
    state = ctx.state
    try:
        record = state.get(step)
        metadata_name = f"{camera}_metadata.json"
        if metadata_name not in record.artifacts:
            raise FileNotFoundError(metadata_name)
        meta = state.load(metadata_name, step=record.step_idx)
    except Exception as e:
        return ToolResult(error=f"{camera} camera metadata not found: {e}")

    if camera == "agentview":
        return ToolResult(data={"camera": "agentview", "camera_meta": meta})
    return ToolResult(
        data={"camera": "wrist", "step": record.step_idx, "camera_meta": meta}
    )


@tool
@readonly
def segment(
    prompt: str = "",
    camera: Literal["agentview", "wrist"] = "agentview",
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    point: Annotated[list[int] | None, Field(min_length=2, max_length=2)] = None,
    min_score: float = 0.2,
    *,
    ctx: ToolContext[LiberoRuntime],
) -> ToolResult:
    """SAM3 visual segmentation over an existing run artifact. It never renders a new camera view. Provide exactly one text prompt or single positive point. A successful top-ranked mask is projected through the matching world map to produce world_xyz.

    Args:
        prompt: Object/text prompt to segment.
        camera: Artifact camera to use (default agentview).
        step: Step to segment; -1 = latest.
        point: Optional single positive point as [row, col]. Mutually exclusive with prompt.
        min_score: Minimum accepted mask score (default 0.2).
    """
    runtime = ctx.robot
    state = ctx.state
    try:
        record = state.get(step)
    except Exception as exc:
        return ToolResult(error=f"state step not available: {exc}")
    nn = record.step_idx

    prompt = prompt.strip()
    has_prompt = bool(prompt)
    has_point = point is not None
    if has_prompt == has_point:
        return ToolResult(error="segment needs exactly one of prompt or point")
    artifact_pairs = [
        (f"{camera}_high.png", f"{camera}_world_high.npz"),
        (f"{camera}.png", f"{camera}_world.npz"),
    ]
    for image_name, world_name in artifact_pairs:
        if (
            image_name in record.artifacts
            and world_name in record.artifacts
            and state.exists(image_name, step=nn)
            and state.exists(world_name, step=nn)
        ):
            break
    else:
        return ToolResult(
            data={
                "step": nn,
                "camera": camera,
                "checked_artifacts": [
                    name for image, world in artifact_pairs for name in (image, world)
                ],
            },
            error="complete segment artifacts not found",
        )

    try:
        data = runtime._sam3_client.segment(
            state.load_bytes(image_name, step=nn),
            text_prompt=prompt if has_prompt else None,
            point=point,
            min_score=min_score,
        )
    except ValueError as e:
        return ToolResult(
            data={"step": nn, "camera": camera, "image_artifact": image_name},
            error=str(e),
        )
    except Exception as e:
        return ToolResult(
            data={
                "step": nn,
                "camera": camera,
                "image_artifact": image_name,
                "fallback": "Use manual visual localization and back_project.",
            },
            error=f"segmentation service call failed: {e}",
        )

    segment_index = 0
    while f"segment_{segment_index:02d}.json" in record.artifacts:
        segment_index += 1
    segment_name = f"segment_{segment_index:02d}.json"
    overlay_name = f"segment_overlay_{segment_index:02d}.png"
    saved_overlay = None
    mask = data.mask
    if data.found:
        try:
            world_map = state.load(world_name, step=nn)
        except Exception as exc:
            world_result = {
                "world_xyz": None,
                "world_error": f"world map artifact not available: {exc}",
                "expected_world_artifact": world_name,
            }
        else:
            world_result = _mask_to_world(mask, world_map)
            world_result["world_artifact"] = world_name
        image = state.load(image_name, step=nn)
        if image.ndim == 3 and image.shape[:2] == mask.shape:
            overlay = image.copy()
            red = np.zeros(overlay.shape[-1], dtype=np.float32)
            red[0] = 255
            overlay[mask] = (
                0.55 * overlay[mask].astype(np.float32) + 0.45 * red
            ).astype(np.uint8)
            saved_overlay = state.save(overlay_name, overlay, step=nn)
    else:
        world_result = {
            "world_xyz": None,
            "world_error": data.reason or "segmentation did not find a mask",
        }

    segment_blob = {
        "found": data.found,
        "mode": "text" if has_prompt else "point",
        "camera": camera,
        "source_step": nn,
        "segment_index": segment_index,
        "image_artifact": image_name,
        "min_score": min_score,
        "score": round(float(data.score), 3) if data.score is not None else None,
        "box": data.box,
        "mask_shape": list(data.mask_shape) if data.mask_shape else None,
    }
    if has_prompt:
        segment_blob["prompt"] = prompt
    else:
        segment_blob["point"] = point
    if not data.found:
        segment_blob["error"] = data.reason or "SAM3 found no mask"
    segment_blob.update(world_result)
    saved_segment = state.save(
        segment_name,
        segment_blob,
        step=nn,
    )

    result = {
        "found": data.found,
        "step": nn,
        "camera": camera,
        "image_artifact": image_name,
        "score": segment_blob["score"],
        "box": segment_blob["box"],
        "world_xyz": segment_blob["world_xyz"],
    }
    if data.found:
        result["world_error"] = segment_blob.get("world_error")
    error = None
    if saved_segment is None:
        error = f"failed to persist segment artifact {segment_name}"
        result["attempted_segment_artifact"] = segment_name
        if "error" in segment_blob:
            error += "\n" + json.dumps({"segmentation_error": segment_blob["error"]})
    else:
        result["segment_artifact"] = saved_segment
        result["segment_path"] = str(state.artifact_path(saved_segment, step=nn))
        error = segment_blob.get("error")
    if error is not None:
        result["fallback"] = "Use manual visual localization and back_project."
    images: list[bytes] = []
    if saved_overlay is not None:
        result["overlay_artifact"] = saved_overlay
        result["overlay_path"] = str(state.artifact_path(saved_overlay, step=nn))
        images.append(state.load_bytes(saved_overlay, step=nn))
    return ToolResult(data=result, error=error, images=images)


@tool
@readonly
@parallel
def back_project(
    row: int | None = None,
    col: int | None = None,
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    camera: Literal["agentview", "wrist"] = "agentview",
    resolution: Literal["high", "low"] = "high",
    row_range: list[int] | None = None,
    col_range: list[int] | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    *,
    ctx: ToolContext[LiberoRuntime],
) -> ToolResult:
    """Back-project a pixel (row, col) to a world XYZ point using the selected camera's precomputed world map. Row 0 = top of image, col 0 = left. Returns world_xyz in meters.

    USE THIS to find where an object is in the world — look at the embedded high-resolution image returned by view_env_state to pick a pixel on the target object, then call back_project. The default resolution is high (1024x1024). Pass resolution='low' only for pixels from the embedded/standard 256 image. The pixel coordinates must come from the same camera and resolution requested here. Use camera='agentview' for global tabletop layout and object locations; use camera='wrist' for close-range details near the gripper, occlusions, and container/cabinet interiors. Sample several pixels on the object and median their xy for robustness.

    REGION MODE: pass row_range=[r0,r1] and col_range=[c0,c1] instead of row/col to get the midpoint of world xy over that pixel window, with an optional world-z band (z_min, z_max). Use it for the center of a container cavity or flat region, where a single-pixel or mask-median estimate is biased toward an edge/rim.

    Args:
        row: Pixel row (0=top) in the selected resolution image.
        col: Pixel column (0=left) in the selected resolution image.
        step: Depth/world-map step; 0 = initial, -1 = latest.
        camera: Camera to back-project from (default agentview).
        resolution: Coordinate system for row/col (default high). Use low only when row/col came from the embedded/standard 256 image.
        row_range: Region mode: [r0, r1] pixel row window. Requires col_range.
        col_range: Region mode: [c0, c1] pixel col window. Requires row_range.
        z_min: Region mode: keep only pixels with world z >= z_min.
        z_max: Region mode: keep only pixels with world z <= z_max.
    """
    state = ctx.state
    region_mode = row_range is not None or col_range is not None
    if not region_mode and (row is None or col is None):
        return ToolResult(
            error="provide either (row, col) for a single pixel, or row_range=[r0,r1] and col_range=[c0,c1] for a region center"
        )

    try:
        record = state.get(step)
    except Exception as e:
        return ToolResult(error=f"state step not available: {e}")
    nn = record.step_idx

    hi_artifact = f"{camera}_world_high.npz"
    low_artifact = f"{camera}_world.npz"
    source_artifact = hi_artifact if resolution == "high" else low_artifact
    if source_artifact not in record.artifacts:
        return ToolResult(
            error=f"{camera} {resolution}-resolution world map not recorded for step {nn}"
        )

    try:
        world_map = state.load(source_artifact, step=nn)
    except Exception as e:
        return ToolResult(
            error=f"{camera} {resolution}-resolution artifact not found for step {nn}: {e}"
        )

    height, width = world_map.shape[:2]

    if region_mode:
        if row_range is None or col_range is None:
            return ToolResult(
                error="region mode needs BOTH row_range=[r0,r1] and col_range=[c0,c1]"
            )
        try:
            r0, r1 = row_range[0], row_range[1]
            c0, c1 = col_range[0], col_range[1]
        except IndexError:
            return ToolResult(
                error="row_range/col_range must each be [min, max] integers"
            )
        r0, r1 = sorted((max(0, r0), min(height, r1)))
        c0, c1 = sorted((max(0, c0), min(width, c1)))
        if r1 <= r0 or c1 <= c0:
            return ToolResult(
                error=f"empty region after clamping to image {height}x{width}: rows [{r0},{r1}] cols [{c0},{c1}]"
            )
        window = (
            world_map[r0:r1, c0:c1].reshape(-1, world_map.shape[2]).astype(np.float64)
        )
        finite = np.isfinite(window).all(axis=1) & (
            np.abs(window[:, :3]).sum(axis=1) > 1e-6
        )
        pts = window[finite]
        n_total = int(pts.shape[0])
        if z_min is not None:
            pts = pts[pts[:, 2] >= float(z_min)]
        if z_max is not None:
            pts = pts[pts[:, 2] <= float(z_max)]
        if pts.shape[0] < 8:
            return ToolResult(
                data={"n_valid_before_zfilter": n_total},
                error=f"too few valid pixels in region after z-filter ({int(pts.shape[0])}); widen the window or the z band",
            )
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        center = [
            round(float((xs.min() + xs.max()) / 2.0), 4),
            round(float((ys.min() + ys.max()) / 2.0), 4),
            round(float(np.median(zs)), 4),
        ]
        return ToolResult(
            data={
                "camera": camera,
                "resolution": resolution,
                "mode": "region",
                "row_range": [r0, r1],
                "col_range": [c0, c1],
                "z_band": [z_min, z_max],
                "center_xyz": center,
                "median_xyz": [
                    round(float(np.median(xs)), 4),
                    round(float(np.median(ys)), 4),
                    round(float(np.median(zs)), 4),
                ],
                "n_valid": int(pts.shape[0]),
                "step": nn,
                "image_size": [height, width],
                "source_artifact": source_artifact,
            }
        )

    if row < 0 or row >= height or col < 0 or col >= width:
        return ToolResult(
            error=f"pixel ({row},{col}) out of bounds; {camera} image is {height}x{width}"
        )

    depth_m = None
    if source_artifact == low_artifact:
        try:
            depth_artifact = f"{camera}_depth.npz"
            if depth_artifact not in record.artifacts:
                raise FileNotFoundError(depth_artifact)
            depth = state.load(depth_artifact, step=nn)
            if depth.ndim == 3:
                depth = depth[..., 0]
        except Exception as e:
            return ToolResult(error=f"{camera} depth not found for step {nn}: {e}")
        depth_m = float(depth[row, col])
        if not np.isfinite(depth_m) or depth_m <= 0 or depth_m > 10:
            return ToolResult(
                error=f"invalid {camera} depth {depth_m:.3f}m at pixel ({row},{col}); pick a different pixel"
            )
    world_xyz_raw = world_map[row, col]
    if (
        not np.isfinite(world_xyz_raw).all()
        or float(np.abs(world_xyz_raw[:3]).sum()) <= 1e-6
    ):
        return ToolResult(error=f"invalid {camera} world xyz at pixel ({row},{col})")
    world_xyz = [round(float(v), 4) for v in world_xyz_raw[:3]]

    out = {
        "camera": camera,
        "resolution": resolution,
        "pixel": [row, col],
        "world_xyz": world_xyz,
        "step": nn,
        "image_size": [height, width],
        "source_artifact": source_artifact,
    }
    if depth_m is not None:
        out["depth_m"] = round(depth_m, 4)
    return ToolResult(data=out)


def _mask_to_world(
    mask: np.ndarray, world_map: np.ndarray, min_valid: int = 10
) -> dict:
    if world_map.ndim != 3 or world_map.shape[2] < 3:
        return {
            "world_xyz": None,
            "world_error": f"invalid world map shape: {tuple(world_map.shape)}",
            "n_pixels": int(mask.sum()),
            "n_valid": 0,
            "mask_resized_to_world_shape": False,
        }

    if mask.shape != world_map.shape[:2]:
        return {
            "world_xyz": None,
            "world_error": (
                f"mask/world shape mismatch: mask={tuple(mask.shape)}, "
                f"world={tuple(world_map.shape[:2])}"
            ),
            "n_pixels": int(mask.sum()),
            "n_valid": 0,
            "mask_resized_to_world_shape": False,
        }

    ys, xs = np.where(mask)
    if ys.size == 0:
        return {"world_xyz": None, "world_error": "empty mask"}

    pts = world_map[ys, xs].astype(np.float64)
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 1e-6)
    pts = pts[valid]
    result = {
        "centroid_pixel": [
            int(round(float(np.median(xs)))),
            int(round(float(np.median(ys)))),
        ],
        "n_pixels": int(mask.sum()),
        "n_valid": int(pts.shape[0]),
        "mask_resized_to_world_shape": False,
    }
    if pts.shape[0] < min_valid:
        result.update(
            {
                "world_xyz": None,
                "world_error": f"too few valid depth pixels ({int(pts.shape[0])})",
            }
        )
        return result

    result["world_xyz"] = [
        round(float(np.median(pts[:, 0])), 4),
        round(float(np.median(pts[:, 1])), 4),
        round(float(np.median(pts[:, 2])), 4),
    ]
    return result


LIBERO_TOOLS = (
    finish,
    reset,
    view_env_state,
    move_to,
    pi0_pick,
    pi0_doubled,
    release,
    set_gripper,
    rotate_wrist,
    rotate_pitch,
    move_pose,
    view_camera_meta,
    segment,
    back_project,
)

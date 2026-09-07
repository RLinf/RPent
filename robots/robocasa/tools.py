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

"""Native RoboCasa action and perception tools."""

from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
from pydantic import Field

from rpent.tools import ToolContext, ToolResult, parallel, readonly, tool

if TYPE_CHECKING:
    from robots.robocasa.toolkit import RoboCasaRuntime

OSC_ROT_SCALE = 0.5  # action 1.0 -> 0.5 rad


def _step_env(ctx: ToolContext[RoboCasaRuntime], action: np.ndarray) -> None:
    """Check cancellation and record each physical step, including calibration."""
    ctx.check_cancelled()
    ctx.robot.env.step(action)
    image = ctx.robot.env.render_camera(
        camera_name="agentview", height=256, width=256, depth=False
    )
    ctx.record_frame(np.asarray(image, dtype=np.uint8))


def _calibrate_pos_jacobian(ctx: ToolContext[RoboCasaRuntime], gripper=-1.0):
    """Probe 3 unit arm-xyz actions, measure world dpos -> 3x3 jacobian J s.t.
    world_dpos ~= J @ action_xyz. move_to inverts J to map desired world delta."""
    runtime = ctx.robot
    cols = []
    for axis in range(3):
        p0 = runtime.env.eef_pos.copy()
        a = np.zeros(12)
        a[11] = -1.0
        a[axis] = 0.4
        a[6] = gripper
        for _ in range(3):
            _step_env(ctx, a)
        d = (runtime.env.eef_pos - p0) / (0.4 * 3)  # world dpos per unit action
        cols.append(d)
        # settle back is not needed (closed-loop re-reads); keep going
    runtime._pos_jac = np.stack(cols, axis=1)  # 3x3: world_dpos = J @ a_xyz
    return runtime._pos_jac


def _base_pos(ctx: ToolContext[RoboCasaRuntime]) -> np.ndarray:
    return np.asarray(
        ctx.robot.env.current_raw_obs["robot0_base_pos"], dtype=np.float64
    )


def _calibrate_forward(ctx: ToolContext[RoboCasaRuntime], gripper=1.0):
    """Drive forward briefly, measure the WORLD direction the base actually goes,
    so navigate_to can steer regardless of the base->world frame offset."""
    from scipy.spatial.transform import Rotation as R

    runtime = ctx.robot
    p0 = _base_pos(ctx)
    y0 = float(
        R.from_quat(
            np.asarray(runtime.env.current_raw_obs["robot0_base_quat"])
        ).as_euler("xyz")[2]
    )
    a = np.zeros(12)
    a[11] = 1.0
    a[6] = float(np.clip(gripper, -1, 1))
    a[7] = 1.0
    for _ in range(6):
        _step_env(ctx, a)
    p1 = _base_pos(ctx)
    disp = (p1 - p0)[:2]
    if np.linalg.norm(disp) > 0.005:
        runtime._fwd_offset = np.arctan2(disp[1], disp[0]) - y0
    else:
        runtime._fwd_offset = 0.0
    return runtime._fwd_offset


def _resolve_grip(ctx: ToolContext[RoboCasaRuntime], gripper, target_q):
    """Return the a[6] gripper command for a motion step.
    gripper="hold" (DEFAULT for moves) -> SERVO the fingers back to `target_q`,
    the width they had when the motion began. This is the carry-safe hold: the
    gripper action is a CLOSE-VELOCITY command, so a sustained +1 keeps driving the
    fingers shut and SQUEEZES a small object OUT (verified: bread qpos 0.0376 -> 0.0005
    during a +1 carry). Servoing to the grasped width holds the object without crushing
    it and without letting it drift open. A numeric gripper (+1 close / -1 open) is an
    EXPLICIT override and passes through unchanged."""
    runtime = ctx.robot
    if isinstance(gripper, str):
        cur = float(runtime.env.gripper_qpos[0])
        return float(
            np.clip(60.0 * (cur - target_q), -1.0, 1.0)
        )  # +a[6] closes (qpos↓)
    return float(np.clip(gripper, -1, 1))


def _move_to(
    ctx: ToolContext[RoboCasaRuntime],
    xyz: list[float] | np.ndarray,
    gripper: float | str = "hold",
    step_clip: float = 0.02,
    max_steps: int = 200,
    tol: float = 0.012,
) -> ToolResult:
    runtime = ctx.robot
    runtime._vla_desync = True
    target = np.asarray(xyz, dtype=np.float64)
    target_q = float(runtime.env.gripper_qpos[0])  # finger width to hold
    if runtime._pos_jac is None:
        _calibrate_pos_jacobian(ctx, gripper=_resolve_grip(ctx, gripper, target_q))
    Jinv = np.linalg.pinv(runtime._pos_jac)
    for i in range(max_steps):
        cur = runtime.env.eef_pos
        err = target - cur
        dist = float(np.linalg.norm(err))
        if dist < tol:
            return ToolResult(
                data={
                    "ok": True,
                    "steps": i,
                    "final_dist": dist,
                    "eef": cur.tolist(),
                    "gripper_qpos": round(float(runtime.env.gripper_qpos[0]), 4),
                }
            )
        step_world = err if dist <= step_clip else err / dist * step_clip
        a_xyz = np.clip(Jinv @ step_world, -1, 1)
        a = np.zeros(12)
        a[11] = -1.0
        a[0:3] = a_xyz
        a[6] = _resolve_grip(ctx, gripper, target_q)
        _step_env(ctx, a)

    cur = runtime.env.eef_pos
    return ToolResult(
        data={
            "ok": False,
            "steps": max_steps,
            "final_dist": float(np.linalg.norm(target - cur)),
            "eef": cur.tolist(),
            "gripper_qpos": round(float(runtime.env.gripper_qpos[0]), 4),
        },
        error="move_to did not reach the target within max_steps",
    )


@tool
def move_to(
    xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    gripper: float | str = "hold",
    step_clip: float = 0.02,
    max_steps: int = 200,
    tol: float = 0.012,
    *,
    ctx: ToolContext[RoboCasaRuntime],
) -> ToolResult:
    """Scripted EEF servo to a world-frame XYZ target via the OSC controller. Holds pitch/yaw orientation (use rotate_pitch to reorient). gripper='hold' (DEFAULT) maintains current finger width — carry-safe without crushing small objects. Pass +1 to close, -1 to open. NEVER command a single move_to with |dxyz| > 0.30 — OSC flips IK; split long traversal into 2-3 mid waypoints at carry z.

    Args:
        xyz: World-frame target [x, y, z] in meters
        gripper: Gripper: +1 close, -1 open, or 'hold' to maintain current finger width (default 'hold')
        step_clip: Per-step dxyz cap, m (default 0.02)
        max_steps: Step budget (default 200)
        tol: Position tolerance, m (default 0.012)
    """
    return _move_to(ctx, xyz, gripper, step_clip, max_steps, tol)


@tool
def move_delta(
    dxyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    gripper: float | str = "hold",
    step_clip: float = 0.02,
    max_steps: int = 80,
    *,
    ctx: ToolContext[RoboCasaRuntime],
) -> ToolResult:
    """Relative EEF displacement from the current position. Computes target = current_eef + dxyz and delegates to move_to. Use for small adjustments (micro-align for grasp, approach). gripper='hold' (DEFAULT) maintains current finger width.

    Args:
        dxyz: Relative displacement [dx, dy, dz] in meters
        gripper: Gripper: +1 close, -1 open, or 'hold' (default 'hold')
        step_clip: Per-step dxyz cap, m (default 0.02)
        max_steps: Step budget (default 80)
    """
    runtime = ctx.robot
    return _move_to(
        ctx, runtime.env.eef_pos + np.asarray(dxyz), gripper, step_clip, max_steps
    )


@tool
def rotate_pitch(
    target_pitch: float = 0.6,
    gripper: float = 1,
    n: int = 12,
    *,
    ctx: ToolContext[RoboCasaRuntime],
) -> ToolResult:
    """Tilt the wrist forward (axis-angle about the control X-axis). This pitches the gripper down/up. Holds xyz fixed. Use before threading the gripper into a narrow opening whose front face normal is along world +/-y.

    Args:
        target_pitch: Absolute pitch target, radians (clamped +/-1.5; default 0.6)
        gripper: Gripper command held during rotation (default +1)
        n: Number of env steps for the rotation (default 12)
    """
    runtime = ctx.robot
    runtime._vla_desync = True
    per = float(np.clip(target_pitch, -1.5, 1.5)) / n
    action = np.zeros(12)
    action[11] = -1.0
    action[3] = np.clip(per / OSC_ROT_SCALE, -1, 1)
    action[6] = float(np.clip(gripper, -1, 1))
    for _ in range(n):
        _step_env(ctx, action)
    return ToolResult(data={"ok": True, "eef": runtime.env.eef_pos.tolist()})


def _set_gripper(
    ctx: ToolContext[RoboCasaRuntime], gripper: float = 1, steps: int = 10
) -> ToolResult:
    runtime = ctx.robot
    runtime._vla_desync = True
    a = np.zeros(12)
    a[11] = -1.0
    a[6] = float(np.clip(gripper, -1, 1))
    for _ in range(steps):
        _step_env(ctx, a)
    return ToolResult(
        data={"ok": True, "gripper_qpos": runtime.env.gripper_qpos.tolist()}
    )


@tool
def set_gripper(
    gripper: float = 1, steps: int = 10, *, ctx: ToolContext[RoboCasaRuntime]
) -> ToolResult:
    """Hold the current EEF pose and drive the gripper command for `steps` env steps. Use to firm up a grip mid-carry or to actively open/close the gripper.

    Args:
        gripper: Gripper command: +1 close, -1 open (default +1)
        steps: Number of env steps to hold (default 10)
    """
    return _set_gripper(ctx, gripper, steps)


@tool
def release(steps: int = 10, *, ctx: ToolContext[RoboCasaRuntime]) -> ToolResult:
    """Open the gripper for `steps` env steps while holding EEF in place. Delegates to set_gripper(-1.0, steps=steps). Use to drop a grasped object.

    Args:
        steps: Number of env steps (default 10)
    """
    return _set_gripper(ctx, -1.0, steps=steps)


@tool
def scripted_grasp(
    xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    approach_z: float = 0.10,
    grasp_z_offset: float = 0.0,
    step_clip: float = 0.02,
    *,
    ctx: ToolContext[RoboCasaRuntime],
) -> ToolResult:
    """Coarse scripted grasp sequence: open -> hover above target -> descend -> close -> lift. A fallback when the VLA closed-loop grasp is unavailable. For hard objects prefer rldx_arm. approach_z and grasp_z_offset are RELATIVE offsets from the target xyz.

    Args:
        xyz: World-frame grasp target [x, y, z] in meters
        approach_z: Z offset above target before descent, m (default 0.10)
        grasp_z_offset: Z offset at grasp point (default 0.0; negative = below target)
        step_clip: Per-step dxyz cap during descent, m (default 0.02)
    """
    runtime = ctx.robot
    t = np.asarray(xyz, dtype=np.float64)
    _set_gripper(ctx, -1.0, steps=4)
    r = _move_to(ctx, t + [0, 0, approach_z], gripper=-1.0, step_clip=step_clip)
    if r.is_error:
        r.data["stage"] = "approach"
        return r
    r = _move_to(
        ctx, t + [0, 0, grasp_z_offset], gripper=-1.0, step_clip=0.012, tol=0.01
    )
    if r.is_error:
        r.data["stage"] = "descent"
        return r
    _set_gripper(ctx, +1.0, steps=14)
    r = _move_to(ctx, t + [0, 0, approach_z + 0.05], gripper="hold", step_clip=0.015)
    if r.is_error:
        r.data["stage"] = "lift"
        return r
    return ToolResult(
        data={
            "ok": True,
            "gripper_qpos": runtime.env.gripper_qpos.tolist(),
            "eef": runtime.env.eef_pos.tolist(),
        }
    )


@tool
def move_base(
    forward: float = 0,
    lateral: float = 0,
    turn: float = 0,
    steps: int = 10,
    gripper: float | str = "hold",
    *,
    ctx: ToolContext[RoboCasaRuntime],
) -> ToolResult:
    """Raw base velocity commands in the robot's LOCAL frame. +forward = drive forward, +lateral = strafe right, +turn = rotate CCW (yaw). All values clamped [-1, 1]. Use move_base for fine base adjustments near a target; use navigate_to for long-range navigation. gripper='hold' (DEFAULT) maintains finger width while driving.

    Args:
        forward: Forward velocity, [-1, 1] (default 0)
        lateral: Lateral / strafe velocity, [-1, 1] (default 0)
        turn: Yaw rotation velocity, [-1, 1] (default 0)
        steps: Number of env steps (default 10)
        gripper: Gripper while driving: +1 close, -1 open, or 'hold' (default 'hold')
    """
    runtime = ctx.robot
    runtime._vla_desync = True
    target_q = float(runtime.env.gripper_qpos[0])
    a = np.zeros(12)
    a[11] = 1.0
    a[7:10] = [
        np.clip(forward, -1, 1),
        np.clip(lateral, -1, 1),
        np.clip(turn, -1, 1),
    ]
    bp0 = _base_pos(ctx)
    for _ in range(steps):
        a[6] = _resolve_grip(ctx, gripper, target_q)
        _step_env(ctx, a)
    bp1 = _base_pos(ctx)
    return ToolResult(
        data={
            "ok": True,
            "base_moved": (bp1 - bp0).tolist(),
            "base_pos": bp1.tolist(),
        }
    )


@tool
def navigate_to(
    xy: Annotated[list[float], Field(min_length=2, max_length=2)],
    tol: float = 0.20,
    max_steps: int = 300,
    gripper: float | str = "hold",
    *,
    ctx: ToolContext[RoboCasaRuntime],
) -> ToolResult:
    """Drive the mobile base toward a WORLD (x, y) target. Online-calibrates the base forward-heading, then turns to face + drives forward closed-loop. Holds the arm in place. gripper='hold' (DEFAULT) maintains current finger width while driving (carry-safe). Use tol = expected approach distance + object radius.

    Args:
        xy: World-frame target [x, y] in meters (z ignored if provided)
        tol: Distance threshold to stop, m (default 0.20)
        max_steps: Step budget (default 300)
        gripper: Gripper while driving: +1 close, -1 open, or 'hold' (default 'hold')
    """
    from scipy.spatial.transform import Rotation as R

    runtime = ctx.robot
    runtime._vla_desync = True
    target = np.asarray(xy[:2], dtype=np.float64)
    target_q = float(runtime.env.gripper_qpos[0])
    if runtime._fwd_offset is None:
        _calibrate_forward(ctx, _resolve_grip(ctx, gripper, target_q))
    start = _base_pos(ctx)[:2].copy()
    for i in range(max_steps):
        bp = _base_pos(ctx)
        to = target - bp[:2]
        dist = float(np.linalg.norm(to))
        if dist < tol:
            runtime._pos_jac = None  # base moved -> recalibrate arm
            moved = float(np.linalg.norm(bp[:2] - start))
            return ToolResult(
                data={
                    "ok": True,
                    "steps": i,
                    "final_dist": dist,
                    "moved": moved,
                    "start_pos": start.tolist(),
                    "base_pos": bp.tolist(),
                }
            )
        world_dir = np.arctan2(to[1], to[0])
        yaw = float(
            R.from_quat(
                np.asarray(runtime.env.current_raw_obs["robot0_base_quat"])
            ).as_euler("xyz")[2]
        )
        cur_forward = yaw + runtime._fwd_offset
        dyaw = (world_dir - cur_forward + np.pi) % (2 * np.pi) - np.pi
        a = np.zeros(12)
        a[11] = 1.0
        a[6] = _resolve_grip(ctx, gripper, target_q)
        if abs(dyaw) > 0.30:  # turn to face the target
            a[9] = float(np.sign(dyaw))
        else:  # drive forward + small steer
            a[7] = 1.0
            a[9] = float(np.clip(dyaw * 1.5, -0.4, 0.4))
        _step_env(ctx, a)
    bp = _base_pos(ctx)
    runtime._pos_jac = None
    moved = float(np.linalg.norm(bp[:2] - start))
    # stuck = ran out of steps having barely moved (rammed a fixture, no path-planning)
    return ToolResult(
        data={
            "ok": False,
            "steps": max_steps,
            "final_dist": float(np.linalg.norm(target - bp[:2])),
            "moved": moved,
            "stuck": moved < 0.12,
            "start_pos": start.tolist(),
            "base_pos": bp.tolist(),
        },
        error="navigate_to did not reach the target within max_steps",
    )


@tool
def reset(*, ctx: ToolContext[RoboCasaRuntime]) -> ToolResult:
    """Restart the episode (new layout / object placement sampled). Arm and base calibration are invalidated on reset. DISABLED in no-reset / matched evaluation — the policy must solve the scene in one shot. Only available in EXPLORE mode when RLDX_ALLOW_RESET is enabled."""
    runtime = ctx.robot
    if not runtime._allow_reset:
        return ToolResult(
            error="reset is DISABLED in this run (no-reset/matched evaluation). Solve the scene in one shot; do not restart the episode."
        )
    ctx.check_cancelled()
    runtime.reset()
    return ToolResult(
        data={"ok": True, "reset": True, "eef": runtime.env.eef_pos.tolist()}
    )


def _run_rldx_skill(
    ctx: ToolContext[RoboCasaRuntime],
    prompt: str,
    base_clip: float | None,
    max_chunks: int,
    force_reset: bool,
    n_action_steps: int,
    settle_patience: int,
    settle_eps: float,
) -> ToolResult:
    """Use the full live task and apply environment budget overrides."""
    runtime = ctx.robot
    max_chunks = int(os.environ.get("RLDX_MAX_CHUNKS", max_chunks))
    n_action_steps = int(os.environ.get("RLDX_ACTION_STEPS_PER_CHUNK", n_action_steps))
    settle_patience = int(os.environ.get("RLDX_SETTLE_PATIENCE", settle_patience))
    for name, value in (
        ("max_chunks", max_chunks),
        ("n_action_steps", n_action_steps),
        ("settle_patience", settle_patience),
    ):
        if value < 1:
            return ToolResult(error=f"{name} must be positive; VLA was not executed")
    task_lang = (
        runtime.env.current_raw_obs.get("language") or runtime.env.get_task_language()
    )
    if not task_lang:
        return ToolResult(
            data={"effective_prompt": "", "prompt_overridden": False},
            error="RoboCasa task language is unavailable; VLA was not executed",
        )
    ctx.check_cancelled()
    reset_history = force_reset or runtime._vla_desync
    runtime._vla_desync = False
    result = runtime._rldx.run(
        task_lang,
        max_chunks,
        n_action_steps,
        base_clip=base_clip,
        settle_patience=settle_patience,
        settle_eps=settle_eps,
        force_reset=reset_history,
        step_env=partial(_step_env, ctx),
        check_cancelled=ctx.check_cancelled,
    )
    result["effective_prompt"] = task_lang
    result["effective_max_chunks"] = max_chunks
    result["effective_n_action_steps"] = n_action_steps
    result["effective_settle_patience"] = settle_patience
    result["prompt_overridden"] = prompt != task_lang
    if prompt != task_lang:
        result["requested_prompt"] = prompt
    return ToolResult(data=result)


@tool
def rldx_skill(
    prompt: str,
    base_clip: float | None = None,
    max_chunks: int = 70,
    force_reset: bool = False,
    n_action_steps: int = 8,
    settle_patience: int = 999,
    settle_eps: float = 0.012,
    *,
    ctx: ToolContext[RoboCasaRuntime],
) -> ToolResult:
    """RLDX VLA closed-loop skill — FULL base motion allowed. The VLA drives both arm and mobile base. Use for full-body tasks where the base must reposition (e.g. navigating to a counter while reaching). Pass the complete live task_language verbatim; the runtime always uses that environment language for RLDX. Do NOT interrupt consecutive VLA calls with manual primitives — that breaks VLA frame history continuity (sets vla_desync=True).

    Args:
        prompt: Complete live task_language, copied verbatim
        base_clip: Base motion magnitude cap (default null = no clamp)
        max_chunks: Action-chunk budget (default 70)
        force_reset: Force VLA frame history reset (default False)
        n_action_steps: Actions per VLA chunk (default 8)
        settle_patience: Settle step budget before declaring done (default 999; do NOT set small)
        settle_eps: Settle position tolerance, m (default 0.012)
    """
    return _run_rldx_skill(
        ctx,
        prompt,
        base_clip,
        max_chunks,
        force_reset,
        n_action_steps,
        settle_patience,
        settle_eps,
    )


@tool
def rldx_arm(
    prompt: str,
    base_clip: float | None = 0.1,
    max_chunks: int = 70,
    force_reset: bool = False,
    n_action_steps: int = 8,
    settle_patience: int = 999,
    settle_eps: float = 0.012,
    *,
    ctx: ToolContext[RoboCasaRuntime],
) -> ToolResult:
    """RLDX VLA closed-loop skill — base CLAMPED to small motions (base_clip=0.1 default). The VLA drives the arm for precise micro-alignment (e.g. fine-tuning a grasp approach) but cannot drive the base away. Pass the complete live task_language verbatim; the runtime always uses that environment language for RLDX. Do NOT interrupt consecutive VLA calls with manual primitives.

    Args:
        prompt: Complete live task_language, copied verbatim
        base_clip: Base motion magnitude cap (default 0.1 = small)
        max_chunks: Action-chunk budget (default 70)
        force_reset: Force VLA frame history reset (default False)
        n_action_steps: Actions per VLA chunk (default 8)
        settle_patience: Settle step budget before declaring done (default 999; do NOT set small)
        settle_eps: Settle position tolerance, m (default 0.012)
    """
    return _run_rldx_skill(
        ctx,
        prompt,
        base_clip,
        max_chunks,
        force_reset,
        n_action_steps,
        settle_patience,
        settle_eps,
    )


# Camera -> (low-resolution world map, optional high-resolution world map).
_CAMERA_WORLD_ARTIFACTS = {
    "agentview": ("agentview_world.npz", "agentview_world_high.npz"),
    "navview": ("navview_world.npz", None),
    "wrist": ("wrist_world.npz", None),
}


@tool
@readonly
@parallel
def view_env_state(
    step: int | None = None, *, ctx: ToolContext[RoboCasaRuntime]
) -> ToolResult:
    """Read step NN from states.json + the matching state images in the output dir. If step is null, returns the latest entry. Each entry contains the env state, robocasa_terminated flag, task_progress, vla_desync status, and log. Embeds available PNGs as multimodal image content blocks. Use calibration-frame agentview images for pixel back-projection; use navview for base navigation and floor walkability; use wrist for close-range details near the gripper.

    Args:
        step: Step number; 0 = initial. Null = latest.
    """
    from robots.robocasa.toolkit import build_observation

    try:
        record = ctx.state.get(step if step is not None else -1)
    except Exception as exc:
        return ToolResult(error=f"state step not available: {exc}")
    data, images = build_observation(ctx.state, record)
    return ToolResult(data=data, images=images)


@tool
@readonly
@parallel
def back_project_batch(
    pixels: Annotated[
        list[Annotated[list[int], Field(min_length=2, max_length=2)]],
        Field(min_length=1, max_length=50),
    ],
    step: int | None = None,
    camera: Literal["agentview", "navview", "wrist"] = "agentview",
    resolution: Literal["high", "low"] = "low",
    *,
    ctx: ToolContext[RoboCasaRuntime],
) -> ToolResult:
    """Back-project one or more pixels to world XYZ points in a single call. Loads the world map once and queries all pixels. Returns each pixel's world_xyz plus a summary with median_xyz across valid pixels.

    USE THIS for robust object localization: sample 3-8 pixels on the target object and read summary.median_xyz. Maximum 50 pixels per call.

    Args:
        pixels: List of [row, col] pixel coordinates (max 50)
        step: Depth / world-map step to use (default latest).
        camera: Camera to back-project from (default agentview).
        resolution: Coordinate system for pixels (default low). Use 'low' for the standard 256x256 world map.
    """
    state = ctx.state
    low_name, hi_name = _CAMERA_WORLD_ARTIFACTS[camera]
    source_artifact = hi_name if resolution == "high" else low_name
    if source_artifact is None:
        return ToolResult(error=f"{camera} has no {resolution}-resolution world map")

    try:
        record = state.get(step if step is not None else -1)
    except Exception as exc:
        return ToolResult(error=f"state step not available: {exc}")
    nn = record.step_idx
    if source_artifact not in record.artifacts:
        return ToolResult(
            error=f"{camera} {resolution}-resolution world map not recorded for step {nn}"
        )

    try:
        world_map = state.load(source_artifact, step=nn)
    except Exception as exc:
        return ToolResult(error=f"{source_artifact} not found for step {nn}: {exc}")

    results = []
    valid_xyzs = []
    for pixel in pixels:
        row, col = pixel
        h, w = world_map.shape[:2]
        if row < 0 or row >= h or col < 0 or col >= w:
            results.append(
                {
                    "pixel": pixel,
                    "world_xyz": None,
                    "valid": False,
                    "error": f"pixel ({row},{col}) out of bounds ({h}x{w})",
                }
            )
            continue
        xyz = world_map[row, col, :3]
        if not np.isfinite(xyz).all() or abs(float(xyz.sum())) <= 1e-6:
            results.append(
                {
                    "pixel": pixel,
                    "world_xyz": None,
                    "valid": False,
                    "error": "invalid world xyz at pixel",
                }
            )
            continue
        results.append(
            {
                "pixel": [row, col],
                "world_xyz": [
                    round(float(xyz[0]), 4),
                    round(float(xyz[1]), 4),
                    round(float(xyz[2]), 4),
                ],
                "valid": True,
                "error": None,
            }
        )
        valid_xyzs.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

    summary: dict = {
        "valid_count": len(valid_xyzs),
        "total_count": len(pixels),
    }
    if valid_xyzs:
        median = np.median(valid_xyzs, axis=0)
        summary["median_xyz"] = [
            round(float(median[0]), 4),
            round(float(median[1]), 4),
            round(float(median[2]), 4),
        ]

    return ToolResult(
        data={
            "results": results,
            "summary": summary,
            "step": nn,
            "camera": camera,
            "resolution": resolution,
        }
    )


@tool
@readonly
@parallel
def query_world_map(
    z_min: float = 0.85,
    z_max: float = 0.95,
    x_range: Annotated[list[float], Field(min_length=2, max_length=2)] | None = None,
    y_range: Annotated[list[float], Field(min_length=2, max_length=2)] | None = None,
    camera: Literal["agentview", "navview", "wrist"] = "agentview",
    resolution: Literal["high", "low"] = "low",
    min_cluster_size: int = 10,
    *,
    ctx: ToolContext[RoboCasaRuntime],
) -> ToolResult:
    """Query the world map by Z-range / XY region to find objects at specific heights. Loads the world map once, filters pixels by z_min <= z <= z_max, optionally restricts to x_range / y_range, then clusters contiguous pixels into objects.

    TYPICAL USES:
    - z_min=0.85, z_max=0.95 -> countertop-height objects
    - z_min=0.0, z_max=0.12, camera='navview' -> walkable floor
    - z_min=0.85, z_max=0.95, x_range=[0,2], y_range=[-3,-1] -> counter objects in a specific quadrant

    Args:
        z_min: Minimum Z in meters (default 0.85 for counter height).
        z_max: Maximum Z in meters (default 0.95 for counter height).
        x_range: Optional X range [min, max] in meters; null = no filter.
        y_range: Optional Y range [min, max] in meters; null = no filter.
        camera: Camera world map to query (default agentview).
        resolution: World map resolution (default low).
        min_cluster_size: Minimum pixels per cluster to report (default 10).
    """
    state = ctx.state
    low_name, hi_name = _CAMERA_WORLD_ARTIFACTS[camera]
    source_artifact = hi_name if resolution == "high" else low_name
    if source_artifact is None:
        return ToolResult(error=f"{camera} has no {resolution}-resolution world map")

    try:
        record = state.get(-1)
    except Exception:
        return ToolResult(error="no state trace available")
    nn = record.step_idx
    if source_artifact not in record.artifacts:
        return ToolResult(
            error=f"{camera} {resolution}-resolution world map not found for step {nn}"
        )
    try:
        world_map = state.load(source_artifact, step=nn)
    except Exception:
        return ToolResult(
            error=f"{camera} {resolution}-resolution world map not found for step {nn}"
        )

    z = world_map[:, :, 2]
    mask = (z >= z_min) & (z <= z_max) & np.isfinite(z)
    if x_range:
        x = world_map[:, :, 0]
        mask &= (x >= x_range[0]) & (x <= x_range[1])
    if y_range:
        y = world_map[:, :, 1]
        mask &= (y >= y_range[0]) & (y <= y_range[1])

    ys, xs = np.where(mask)
    total_pixels = len(ys)
    if total_pixels < min_cluster_size:
        return ToolResult(
            data={
                "clusters": [],
                "summary": {"total_clusters": 0, "total_pixels_matched": 0},
            }
        )

    h, w = world_map.shape[:2]
    grid_cells = max(8, min(32, h // 32))
    cell_h = max(1, h // grid_cells)
    cell_w = max(1, w // grid_cells)
    cells: dict[tuple[int, int], dict] = {}
    for i in range(0, len(ys), 5):
        y, x = int(ys[i]), int(xs[i])
        gy, gx = y // cell_h, x // cell_w
        key = (gy, gx)
        if key not in cells:
            cells[key] = {"pixels": [], "world_pts": []}
        cells[key]["pixels"].append((y, x))
        cells[key]["world_pts"].append(world_map[y, x, :3])

    clusters = []
    for data in cells.values():
        if len(data["pixels"]) < min_cluster_size:
            continue
        pts = np.array(data["world_pts"])
        center = np.median(pts, axis=0)
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        center_idx = len(data["pixels"]) // 2
        clusters.append(
            {
                "center_xyz": [
                    round(float(center[0]), 4),
                    round(float(center[1]), 4),
                    round(float(center[2]), 4),
                ],
                "pixel_count": len(data["pixels"]),
                "bbox_xyz": {
                    "min": [
                        round(float(bbox_min[0]), 4),
                        round(float(bbox_min[1]), 4),
                        round(float(bbox_min[2]), 4),
                    ],
                    "max": [
                        round(float(bbox_max[0]), 4),
                        round(float(bbox_max[1]), 4),
                        round(float(bbox_max[2]), 4),
                    ],
                },
                "sample_pixels": [list(data["pixels"][center_idx])],
            }
        )

    clusters.sort(key=lambda c: -c["pixel_count"])
    return ToolResult(
        data={
            "clusters": clusters[:20],
            "summary": {
                "total_clusters": len(clusters[:20]),
                "total_pixels_matched": total_pixels,
            },
        }
    )


@tool
@readonly
def finish(
    status: Literal["success", "failure", "stuck"],
    summary: str,
    *,
    ctx: ToolContext[RoboCasaRuntime],
) -> ToolResult:
    """Declare the task finished. Call when robocasa_terminated becomes True (success detected), or when genuinely stuck after honest exploration. Provide a 1-3 sentence summary of what worked and what failed.

    Args:
        status: Task outcome classification.
        summary: 1-3 sentence summary of what worked / what failed.
    """
    return ToolResult(data={"_finish": True, "status": status, "summary": summary})


ROBOCASA_TOOLS = (
    finish,
    move_to,
    move_delta,
    rotate_pitch,
    set_gripper,
    release,
    scripted_grasp,
    move_base,
    navigate_to,
    rldx_skill,
    rldx_arm,
    reset,
    view_env_state,
    back_project_batch,
    query_world_map,
)

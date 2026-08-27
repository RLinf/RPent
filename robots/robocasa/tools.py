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

"""RoboCasa tool schemas and handlers backed by the run's ``EnvState``.

The state trace (``states.json`` manifest + per-step artifact files under
``<step:02d>/``) is owned by :class:`rpent.session.EnvState`. Tool handlers
that need to read it take a ``state: EnvState`` keyword argument (bound by the
toolkit via :func:`functools.partial`); state-advancing primitive tools capture
state automatically through :meth:`RoboCasaToolkit.get_env_state`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rpent.session import EnvState, StepRecord
from rpent.tools.toolkit import readonly

if TYPE_CHECKING:
    from robots.robocasa.primitives import RoboCasaPrimitives

# ---- TOOLS_SPEC: 17 Anthropic-shaped tool schemas ----

TOOLS_SPEC = [
    # ---- primitive tools (11): dispatched by the toolkit base to primitives.<name> ----
    {
        "name": "move_to",
        "description": (
            "Scripted EEF servo to a world-frame XYZ target via the OSC "
            "controller. Holds pitch/yaw orientation (use rotate_pitch to "
            "reorient). gripper='hold' (DEFAULT) maintains current finger width "
            "— carry-safe without crushing small objects. Pass +1 to close, "
            "-1 to open. NEVER command a single move_to with |dxyz| > 0.30 — "
            "OSC flips IK; split long traversal into 2-3 mid waypoints at carry z."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "description": "World-frame target [x, y, z] in meters",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "gripper": {
                    "type": ["number", "string"],
                    "description": (
                        "Gripper: +1 close, -1 open, or 'hold' to maintain "
                        "current finger width (default 'hold')"
                    ),
                },
                "step_clip": {
                    "type": "number",
                    "description": "Per-step dxyz cap, m (default 0.02)",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Step budget (default 200)",
                },
                "tol": {
                    "type": "number",
                    "description": "Position tolerance, m (default 0.012)",
                },
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "move_delta",
        "description": (
            "Relative EEF displacement from the current position. Computes "
            "target = current_eef + dxyz and delegates to move_to. "
            "Use for small adjustments (micro-align for grasp, approach). "
            "gripper='hold' (DEFAULT) maintains current finger width."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dxyz": {
                    "type": "array",
                    "description": "Relative displacement [dx, dy, dz] in meters",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "gripper": {
                    "type": ["number", "string"],
                    "description": (
                        "Gripper: +1 close, -1 open, or 'hold' (default 'hold')"
                    ),
                },
                "step_clip": {
                    "type": "number",
                    "description": "Per-step dxyz cap, m (default 0.02)",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Step budget (default 80)",
                },
            },
            "required": ["dxyz"],
        },
    },
    {
        "name": "rotate_pitch",
        "description": (
            "Tilt the wrist forward (axis-angle about the control X-axis). "
            "This pitches the gripper down/up. Holds xyz fixed. "
            "Use before threading the gripper into a narrow opening whose "
            "front face normal is along world +/-y."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_pitch": {
                    "type": "number",
                    "description": (
                        "Absolute pitch target, radians (clamped +/-1.5; default 0.6)"
                    ),
                },
                "gripper": {
                    "type": "number",
                    "description": "Gripper command held during rotation (default +1)",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of env steps for the rotation (default 12)",
                },
            },
        },
    },
    {
        "name": "set_gripper",
        "description": (
            "Hold the current EEF pose and drive the gripper command for "
            "`steps` env steps. Use to firm up a grip mid-carry or to "
            "actively open/close the gripper."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gripper": {
                    "type": "number",
                    "description": "Gripper command: +1 close, -1 open (default +1)",
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of env steps to hold (default 10)",
                },
            },
        },
    },
    {
        "name": "release",
        "description": (
            "Open the gripper for `steps` env steps while holding EEF in "
            "place. Delegates to set_gripper(-1.0, steps=steps). "
            "Use to drop a grasped object."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of env steps (default 10)",
                },
            },
        },
    },
    {
        "name": "scripted_grasp",
        "description": (
            "Coarse scripted grasp sequence: open -> hover above target -> "
            "descend -> close -> lift. A fallback when the VLA closed-loop "
            "grasp is unavailable. For hard objects prefer rldx_arm. "
            "approach_z and grasp_z_offset are RELATIVE offsets from the "
            "target xyz."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "description": "World-frame grasp target [x, y, z] in meters",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "approach_z": {
                    "type": "number",
                    "description": "Z offset above target before descent, m (default 0.10)",
                },
                "grasp_z_offset": {
                    "type": "number",
                    "description": "Z offset at grasp point (default 0.0; negative = below target)",
                },
                "step_clip": {
                    "type": "number",
                    "description": "Per-step dxyz cap during descent, m (default 0.02)",
                },
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "rldx_skill",
        "description": (
            "RLDX VLA closed-loop skill — FULL base motion allowed. The VLA "
            "drives both arm and mobile base. Use for full-body tasks where "
            "the base must reposition (e.g. navigating to a counter while "
            "reaching). Do NOT interrupt consecutive VLA calls with manual "
            "primitives — that breaks VLA frame history continuity "
            "(sets vla_desync=True)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "VLA prompt, e.g. 'pick up the red mug'",
                },
                "base_clip": {
                    "type": ["number", "null"],
                    "description": "Base motion magnitude cap (default null = no clamp)",
                },
                "max_chunks": {
                    "type": "integer",
                    "description": "Action-chunk budget (default 70; do NOT set small)",
                },
                "use_prompt": {
                    "type": ["boolean", "null"],
                    "description": (
                        "If true, use the explicit prompt; "
                        "if null/False, use env task language"
                    ),
                },
                "force_reset": {
                    "type": "boolean",
                    "description": "Force VLA frame history reset (default False)",
                },
                "n_action_steps": {
                    "type": "integer",
                    "description": "Actions per VLA chunk (default 8)",
                },
                "settle_patience": {
                    "type": "integer",
                    "description": (
                        "Settle step budget before declaring done "
                        "(default 999; do NOT set small)"
                    ),
                },
                "settle_eps": {
                    "type": "number",
                    "description": "Settle position tolerance, m (default 0.012)",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "rldx_arm",
        "description": (
            "RLDX VLA closed-loop skill — base CLAMPED to small motions "
            "(base_clip=0.1 default). The VLA drives the arm for precise "
            "micro-alignment (e.g. fine-tuning a grasp approach) but cannot "
            "drive the base away. Do NOT interrupt consecutive VLA calls "
            "with manual primitives."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "VLA prompt, e.g. 'pick up the red mug'",
                },
                "base_clip": {
                    "type": ["number", "null"],
                    "description": "Base motion magnitude cap (default 0.1 = small)",
                },
                "max_chunks": {
                    "type": "integer",
                    "description": "Action-chunk budget (default 70; do NOT set small)",
                },
                "use_prompt": {
                    "type": ["boolean", "null"],
                    "description": (
                        "If true, use the explicit prompt; "
                        "if null/False, use env task language"
                    ),
                },
                "force_reset": {
                    "type": "boolean",
                    "description": "Force VLA frame history reset (default False)",
                },
                "n_action_steps": {
                    "type": "integer",
                    "description": "Actions per VLA chunk (default 8)",
                },
                "settle_patience": {
                    "type": "integer",
                    "description": (
                        "Settle step budget before declaring done "
                        "(default 999; do NOT set small)"
                    ),
                },
                "settle_eps": {
                    "type": "number",
                    "description": "Settle position tolerance, m (default 0.012)",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "navigate_to",
        "description": (
            "Drive the mobile base toward a WORLD (x, y) target. "
            "Online-calibrates the base forward-heading, then turns to face "
            "+ drives forward closed-loop. Holds the arm in place. "
            "gripper='hold' (DEFAULT) maintains current finger width while "
            "driving (carry-safe). Use tol = expected approach distance "
            "+ object radius."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xy": {
                    "type": "array",
                    "description": "World-frame target [x, y] in meters (z ignored if provided)",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "tol": {
                    "type": "number",
                    "description": "Distance threshold to stop, m (default 0.20)",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Step budget (default 300)",
                },
                "gripper": {
                    "type": ["number", "string"],
                    "description": (
                        "Gripper while driving: +1 close, -1 open, "
                        "or 'hold' (default 'hold')"
                    ),
                },
            },
            "required": ["xy"],
        },
    },
    {
        "name": "move_base",
        "description": (
            "Raw base velocity commands in the robot's LOCAL frame. "
            "+forward = drive forward, +lateral = strafe right, "
            "+turn = rotate CCW (yaw). All values clamped [-1, 1]. "
            "Use move_base for fine base adjustments near a target; "
            "use navigate_to for long-range navigation. "
            "gripper='hold' (DEFAULT) maintains finger width while driving."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "forward": {
                    "type": "number",
                    "description": "Forward velocity, [-1, 1] (default 0)",
                },
                "lateral": {
                    "type": "number",
                    "description": "Lateral / strafe velocity, [-1, 1] (default 0)",
                },
                "turn": {
                    "type": "number",
                    "description": "Yaw rotation velocity, [-1, 1] (default 0)",
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of env steps (default 10)",
                },
                "gripper": {
                    "type": ["number", "string"],
                    "description": (
                        "Gripper while driving: +1 close, -1 open, "
                        "or 'hold' (default 'hold')"
                    ),
                },
            },
        },
    },
    {
        "name": "reset",
        "description": (
            "Restart the episode (new layout / object placement sampled). "
            "Arm and base calibration are invalidated on reset. "
            "DISABLED in no-reset / matched evaluation — the policy must "
            "solve the scene in one shot. Only available in EXPLORE mode "
            "when RLDX_ALLOW_RESET is enabled."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    # ---- perception tools (6) -- module-level @readonly handlers ----
    {
        "name": "view_env_state",
        "description": (
            "Read step NN from states.json + the matching state images "
            "in the output dir. If step is null, returns the latest entry. "
            "Each entry contains the env state, robocasa_terminated flag, "
            "task_progress, vla_desync status, and log. Embeds available "
            "PNGs as multimodal image content blocks. Use calibration-frame "
            "agentview images for pixel back-projection; use navview for "
            "base navigation and floor walkability; use wrist for close-range "
            "details near the gripper."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": ["integer", "null"],
                    "description": "Step number; 0 = initial. Null = latest.",
                },
            },
        },
    },
    {
        "name": "view_camera_meta",
        "description": (
            "Read camera calibration metadata from the output dir. "
            "camera='agentview' reads the static camera_meta file. "
            "camera='navview' reads the navview camera metadata. "
            "Needed for computing 3D geometry from pixel coordinates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "navview"],
                    "description": "Camera metadata to read (default agentview).",
                },
                "step": {
                    "type": ["integer", "null"],
                    "description": "Step number (default latest).",
                },
            },
        },
    },
    {
        "name": "back_project",
        "description": (
            "Back-project a single pixel (row, col) to a world XYZ point "
            "using the selected camera's precomputed world map. Row 0 = top "
            "of image, col 0 = left. Returns world_xyz in meters.\n\n"
            "Use for a quick single-pixel check. For robust localization, "
            "prefer back_project_batch which samples multiple pixels and "
            "returns a median. camera='agentview' for global layout; "
            "camera='navview' for base navigation (floor pixels); "
            "camera='wrist' for close-range precision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "description": "Pixel row (0=top) in the selected resolution image.",
                },
                "col": {
                    "type": "integer",
                    "description": "Pixel column (0=left) in the selected resolution image.",
                },
                "step": {
                    "type": ["integer", "null"],
                    "description": "Depth / world-map step to use (default latest). 0 for initial.",
                },
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "navview", "wrist"],
                    "description": "Camera to back-project from (default agentview).",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["high", "low"],
                    "description": (
                        "Coordinate system for row/col (default high). "
                        "Use 'low' when row/col came from the standard "
                        "256x256 embedded image."
                    ),
                },
            },
            "required": ["row", "col"],
        },
    },
    {
        "name": "back_project_batch",
        "description": (
            "Back-project MULTIPLE pixels to world XYZ points in a single "
            "call. Loads the world map once and queries all pixels — "
            "replaces N separate back_project calls. "
            "Returns each pixel's world_xyz plus a summary with "
            "median_xyz across valid pixels.\n\n"
            "USE THIS for robust object localization: sample 3-8 pixels "
            "on the target object and read summary.median_xyz. "
            "Maximum 50 pixels per call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pixels": {
                    "type": "array",
                    "description": "List of [row, col] pixel coordinates (max 50)",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "minItems": 1,
                    "maxItems": 50,
                },
                "step": {
                    "type": ["integer", "null"],
                    "description": "Depth / world-map step to use (default latest).",
                },
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "navview", "wrist"],
                    "description": "Camera to back-project from (default agentview).",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["high", "low"],
                    "description": (
                        "Coordinate system for pixels (default low). "
                        "Use 'low' for the standard 256x256 world map."
                    ),
                },
            },
            "required": ["pixels"],
        },
    },
    {
        "name": "query_world_map",
        "description": (
            "Query the world map by Z-range / XY region to find objects "
            "at specific heights. Loads the world map once, filters pixels "
            "by z_min <= z <= z_max, optionally restricts to x_range / "
            "y_range, then clusters contiguous pixels into objects.\n\n"
            "TYPICAL USES:\n"
            "- z_min=0.85, z_max=0.95 -> countertop-height objects\n"
            "- z_min=0.0, z_max=0.12, camera='navview' -> walkable floor\n"
            "- z_min=0.85, z_max=0.95, x_range=[0,2], y_range=[-3,-1] -> "
            "counter objects in a specific quadrant"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "z_min": {
                    "type": "number",
                    "description": "Minimum Z in meters (default 0.85 for counter height).",
                },
                "z_max": {
                    "type": "number",
                    "description": "Maximum Z in meters (default 0.95 for counter height).",
                },
                "x_range": {
                    "type": ["array", "null"],
                    "description": "Optional X range [min, max] in meters; null = no filter.",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "y_range": {
                    "type": ["array", "null"],
                    "description": "Optional Y range [min, max] in meters; null = no filter.",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "navview", "wrist"],
                    "description": "Camera world map to query (default agentview).",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["high", "low"],
                    "description": "World map resolution (default low).",
                },
                "min_cluster_size": {
                    "type": "integer",
                    "description": "Minimum pixels per cluster to report (default 10).",
                },
            },
        },
    },
    {
        "name": "finish",
        "description": (
            "Declare the task finished. Call when robocasa_terminated "
            "becomes True (success detected), or when genuinely stuck "
            "after honest exploration. Provide a 1-3 sentence summary "
            "of what worked and what failed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["success", "failure", "stuck"],
                    "description": "Task outcome classification.",
                },
                "summary": {
                    "type": "string",
                    "description": "1-3 sentence summary of what worked / what failed.",
                },
            },
            "required": ["status", "summary"],
        },
    },
]


# ---- state persistence (dump_state writes through the run's EnvState) ----

# Heavy npy artifacts pruned after the ``_keep_heavy`` window elapses (the
# agent localizes from the latest frame; old world/depth maps are dead weight
# that once filled the 100GB root and deadlocked everything).
_HEAVY_ARTIFACTS = (
    "agentview_depth.npz",
    "agentview_world.npz",
    "wrist_depth.npz",
    "wrist_world.npz",
    "agentview_world_high.npz",
    "navview_world.npz",
)

# Artifact base names exposed to the agent for each camera (high-res first).
_CAMERA_IMAGE_ARTIFACTS = {
    "agentview": ("agentview_high.png", "agentview.png"),
    "navview": ("navview.png",),
    "wrist": ("wrist_high.png", "wrist.png"),
}

# Camera -> (low-res world map artifact, high-res world map artifact or None)
_CAMERA_WORLD_ARTIFACTS = {
    "agentview": ("agentview_world.npz", "agentview_world_high.npz"),
    "navview": ("navview_world.npz", None),
    "wrist": ("wrist_world.npz", None),
}


def dump_state(
    primitives: RoboCasaPrimitives,
    env_state: EnvState,
    log: dict | None = None,
) -> StepRecord:
    """Record one RoboCasa observation through its owned state record.

    Appends a new :class:`StepRecord` (proprio + task + success + vla_desync)
    via :meth:`EnvState.record_step`, saves the rendered RGB / depth / world
    artifacts for that step, and prunes heavy npy artifacts that fell out of
    the ``_keep_heavy`` window.
    """
    state_dict = primitives.current_state_dict()
    log = log or {}
    with env_state.record_step(
        state=state_dict["state"],
        terminated=state_dict["robocasa_terminated"],
        truncated=False,
        command=log.get("command"),
        result=log.get("result"),
        elapsed_s=log.get("elapsed_s"),
        extras={
            "task_language": state_dict["task_language"],
            "success": state_dict["success"],
            "task_progress": state_dict["task_progress"],
            "vla_desync": primitives._vla_desync,
        },
    ) as step_idx:
        _save_observation_artifacts(primitives, env_state, step_idx)
        _prune_heavy_artifacts(primitives, env_state, step_idx)
    return env_state.get(step_idx)


def _save_observation_artifacts(
    primitives: RoboCasaPrimitives,
    env_state: EnvState,
    step_idx: int,
) -> None:
    """Render and save all per-step observation artifacts for ``step_idx``."""
    env = primitives.env
    hi_res = primitives.hi_res

    # ---- agentview + wrist: rgb, depth, world map, camera meta ----
    for cam, image_name, depth_name, world_name, meta_name in (
        (
            "agentview",
            "agentview.png",
            "agentview_depth.npz",
            "agentview_world.npz",
            "agentview_metadata.json",
        ),
        (
            "wrist",
            "wrist.png",
            "wrist_depth.npz",
            "wrist_world.npz",
            "wrist_metadata.json",
        ),
    ):
        rgb, depth = env.render_camera(cam, depth=True)
        env_state.save(image_name, rgb, step=step_idx)
        env_state.save(depth_name, depth.astype(np.float32), step=step_idx)
        env_state.save(world_name, env.world_map(cam).astype(np.float32), step=step_idx)
        if cam not in primitives._cam_meta_cache or cam == "wrist":
            primitives._cam_meta_cache[cam] = env.get_camera_meta(cam)
        env_state.save(meta_name, primitives._cam_meta_cache[cam], step=step_idx)

    # ---- hi-res agentview (SAM grounding / fine localize) ----
    if hi_res:
        hrgb, _ = env.render_camera("agentview", hi_res, hi_res, depth=True)
        env_state.save("agentview_high.png", hrgb, step=step_idx)
        env_state.save(
            "agentview_world_high.npz",
            env.world_map("agentview", hi_res, hi_res).astype(np.float16),
            step=step_idx,
        )

    # ---- navview: base-mounted forward-down floor camera (follows the base) ----
    try:
        nrgb, _ = env.render_camera("navview", depth=True)
        nworld = env.world_map("navview").astype(np.float32)
        env_state.save("navview.png", nrgb, step=step_idx)
        env_state.save("navview_world.npz", nworld, step=step_idx)
        floor = (nworld[:, :, 2] < 0.12) & (nworld[:, :, 2] > -0.2)
        overlay = nrgb.copy()
        overlay[floor] = [0, 255, 0]
        env_state.save("navview_floor.png", overlay, step=step_idx)
    except Exception as e:
        print(f"[tools] navcam render failed at step {step_idx}: {e}", flush=True)


def _prune_heavy_artifacts(
    primitives: RoboCasaPrimitives,
    env_state: EnvState,
    step_idx: int,
) -> None:
    """Delete heavy npy artifacts for the step that fell out of the window."""
    old = step_idx - primitives._keep_heavy
    if old < 0:
        return
    for name in _HEAVY_ARTIFACTS:
        try:
            env_state.artifact_path(name, step=old).unlink(missing_ok=True)
        except Exception:
            pass


# ---- tool handlers (@readonly perception tools take state: EnvState) ----


@readonly
def view_env_state(step: int | None = None, *, state: EnvState) -> dict:
    """Read one recorded state with embedded camera / navview / wrist images."""
    try:
        record = state.get(step if step is not None else -1)
    except Exception as exc:
        return {"error": f"state step not available: {exc}"}

    nn = record.step_idx
    extras = record.extras
    out: dict = {
        "step": nn,
        "task_progress": extras.get("task_progress", {}),
        "task_language": extras.get("task_language", ""),
        "state": record.state,
        "robocasa_terminated": record.terminated,
        "vla_desync": extras.get("vla_desync", False),
        "success": extras.get("success", False),
        "log": {
            "command": record.command,
            "result": record.result,
            "elapsed_s": record.elapsed_s,
        },
        "images": [],
    }

    for kind, candidates in (
        ("_image_cam_bytes", _CAMERA_IMAGE_ARTIFACTS["agentview"]),
        ("_image_nav_bytes", _CAMERA_IMAGE_ARTIFACTS["navview"]),
        ("_image_wrist_bytes", _CAMERA_IMAGE_ARTIFACTS["wrist"]),
    ):
        for name in candidates:
            if name not in record.artifacts:
                continue
            try:
                out[kind] = state.load_bytes(name, step=nn)
            except FileNotFoundError:
                continue
            label = {
                "_image_cam_bytes": ("calibration_frame", "agentview"),
                "_image_nav_bytes": ("nav_view", "navview"),
                "_image_wrist_bytes": ("calibration_frame", "wrist"),
            }[kind]
            out["images"].append(
                {
                    "role": label[0],
                    "camera": label[1],
                    "artifact": name,
                }
            )
            break

    return out


@readonly
def view_camera_meta(
    camera: str = "agentview",
    step: int | None = None,
    *,
    state: EnvState,
) -> dict:
    """Read camera calibration metadata. Skeleton — returns error."""
    return {"error": "not implemented"}


@readonly
def back_project(
    row: int,
    col: int,
    step: int | None = None,
    camera: str = "agentview",
    resolution: str = "high",
    *,
    state: EnvState,
) -> dict:
    """Back-project a single pixel to world XYZ. Skeleton — returns error."""
    return {"error": "not implemented"}


@readonly
def back_project_batch(
    pixels: list[list[int]],
    step: int | None = None,
    camera: str = "agentview",
    resolution: str = "low",
    *,
    state: EnvState,
) -> dict:
    """Back-project multiple pixels to world XYZ in a single call.

    Loads the precomputed world map once and queries all *pixels*, returning
    each result individually plus a summary with the median of valid points.
    """
    camera = camera or "agentview"
    resolution = resolution or "low"
    if camera not in _CAMERA_WORLD_ARTIFACTS:
        return {"error": f"bad camera '{camera}' (use agentview, navview, or wrist)"}
    low_name, hi_name = _CAMERA_WORLD_ARTIFACTS[camera]
    source_artifact = hi_name if resolution == "high" else low_name
    if source_artifact is None:
        return {"error": f"{camera} has no {resolution}-resolution world map"}

    try:
        record = state.get(step if step is not None else -1)
    except Exception as exc:
        return {"error": f"state step not available: {exc}"}
    nn = record.step_idx
    if source_artifact not in record.artifacts:
        return {
            "error": f"{camera} {resolution}-resolution world map not recorded for step {nn}"
        }

    try:
        world_map = state.load(source_artifact, step=nn)
    except Exception as exc:
        return {"error": f"{source_artifact} not found for step {nn}: {exc}"}

    results = []
    valid_xyzs = []
    for pixel in pixels:
        if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
            results.append(
                {
                    "pixel": pixel,
                    "world_xyz": None,
                    "valid": False,
                    "error": "pixel must be [row, col]",
                }
            )
            continue
        row, col = int(pixel[0]), int(pixel[1])
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

    return {
        "results": results,
        "summary": summary,
        "step": nn,
        "camera": camera,
        "resolution": resolution,
    }


@readonly
def query_world_map(
    z_min: float = 0.85,
    z_max: float = 0.95,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    camera: str = "agentview",
    resolution: str = "low",
    min_cluster_size: int = 10,
    *,
    state: EnvState,
) -> dict:
    """Query the world map by z-range and/or region to find objects."""
    if camera not in _CAMERA_WORLD_ARTIFACTS:
        return {"error": f"bad camera '{camera}' (use agentview, navview, or wrist)"}
    low_name, hi_name = _CAMERA_WORLD_ARTIFACTS[camera]
    source_artifact = hi_name if resolution == "high" else low_name
    if source_artifact is None:
        return {"error": f"{camera} has no {resolution}-resolution world map"}

    try:
        record = state.get(-1)
    except Exception:
        return {"error": "no state trace available"}
    nn = record.step_idx
    if source_artifact not in record.artifacts:
        return {
            "error": f"{camera} {resolution}-resolution world map not found for step {nn}"
        }
    try:
        world_map = state.load(source_artifact, step=nn)
    except Exception:
        return {
            "error": f"{camera} {resolution}-resolution world map not found for step {nn}"
        }

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
        return {
            "clusters": [],
            "summary": {"total_clusters": 0, "total_pixels_matched": 0},
        }

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
    return {
        "clusters": clusters[:20],
        "summary": {
            "total_clusters": len(clusters[:20]),
            "total_pixels_matched": total_pixels,
        },
    }


@readonly
def finish(status: str, summary: str) -> dict:
    """Declare the task finished."""
    return {"_finish": True, "status": status, "summary": summary}


# ---- recipe export ----

_PRIMITIVE_ACTIONS = frozenset(
    {
        "move_to",
        "move_delta",
        "rotate_pitch",
        "set_gripper",
        "release",
        "scripted_grasp",
        "rldx_skill",
        "rldx_arm",
        "navigate_to",
        "move_base",
        "reset",
    }
)


def write_recipe_from_states(state: EnvState, recipe_tag: str) -> str:
    """Export non-error RoboCasa primitive commands from the state trace as JSONL."""
    commands = []
    for record in state.records():
        command = record.command
        if not isinstance(command, dict):
            continue
        if command.get("action") not in _PRIMITIVE_ACTIONS:
            continue
        result = record.result
        if isinstance(result, dict) and result.get("error"):
            continue
        commands.append(command)
    recipe_name = f"recipe_{recipe_tag}.jsonl"
    state.save(recipe_name, commands, step=None)
    return recipe_name

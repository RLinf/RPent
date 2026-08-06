"""RoboCasa tool schemas and handler stubs."""
from __future__ import annotations

import glob
import json
import os
import re
from typing import TYPE_CHECKING

import numpy as np

from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from robots.robocasa.primitives import RoboCasaPrimitives

# ---------------------------------------------------------------------------
# TOOLS_SPEC — 17 Anthropic-shaped tool schemas
# 11 primitive tools + 6 perception tools
# ---------------------------------------------------------------------------

TOOLS_SPEC = [
    # ======================================================================
    # Primitive tools (11) — dispatched via _step() in RoboCasaToolkit
    # ======================================================================
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
                        "Absolute pitch target, radians (clamped +/-1.5; "
                        "default 0.6)"
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
    # ======================================================================
    # Perception tools (6) — module-level handler functions
    # ======================================================================
    {
        "name": "view_driver_state",
        "description": (
            "Read step NN from states.json + the matching state images "
            "in the output dir. If step is null, returns the latest entry. "
            "Each entry contains the robot state, robocasa_terminated flag, "
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

# ---------------------------------------------------------------------------
# State reader helpers — accesses dumped run data from the output directory
# ---------------------------------------------------------------------------


def _load_states() -> list:
    """Return the parsed state trace from ``states.json``."""
    out_dir = get_output_dir()
    if out_dir is None:
        return []
    path = out_dir / "states.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _latest_step() -> int | None:
    """Return the highest step index among dumped states, or None."""
    out_dir = get_output_dir()
    if out_dir is None:
        return None
    # Try consolidated states.json first
    states = _load_states()
    if states:
        return int(states[-1]["step"])
    # Fallback: scan done_*.flag files created by dump_state
    matches = sorted(glob.glob(str(out_dir / "done_*.flag")))
    if not matches:
        return None
    m = re.search(r"done_(\d+)\.flag$", matches[-1])
    return int(m.group(1)) if m else None


def _load_step(nn: int) -> dict:
    """Return the state blob for step *nn*.

    Tries the individual ``state_{nn:02d}.json`` written by ``dump_state``
    first, then falls back to searching the consolidated ``states.json``.
    """
    out_dir = get_output_dir()
    if out_dir is None:
        raise FileNotFoundError("no output directory configured")

    path = out_dir / f"state_{nn:02d}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)

    for entry in _load_states():
        if int(entry.get("step", -1)) == nn:
            return entry
    raise FileNotFoundError(f"step {nn} not present in states.json or state_{nn:02d}.json")


def _load_image(nn: int, kind: str) -> bytes | None:
    """Return PNG bytes for a dumped state image, or None if not present.

    Kind values:
        ``"camera"`` — agentview calibration frame (``image_cam_{nn}.png``)
        ``"agent"``  — Pi0-frame (same as camera for RoboCasa)
        ``"nav"``    — navview (``image_nav_{nn}.png``)
        ``"wrist"``  — wrist (``image_cam_wrist_{nn}.png``)
    """
    out_dir = get_output_dir()
    if out_dir is None:
        return None
    if kind in ("camera", "agent"):
        path = out_dir / f"image_cam_{nn:02d}.png"
    elif kind == "nav":
        path = out_dir / f"image_nav_{nn:02d}.png"
    elif kind == "wrist":
        path = out_dir / f"image_cam_wrist_{nn:02d}.png"
    else:
        raise ValueError(f"unknown image kind: {kind!r}")
    if not path.exists():
        return None
    return path.read_bytes()


def _load_camera_meta(camera: str = "agentview", nn: int | None = None) -> dict:
    """Read camera calibration metadata JSON for *camera* at step *nn*.

    When *nn* is ``None`` (default) the latest available step is used.
    Camera values: ``"agentview"``, ``"navview"``, ``"wrist"``.
    """
    out_dir = get_output_dir()
    if out_dir is None:
        raise FileNotFoundError("no output directory configured")
    if nn is None:
        latest = _latest_step()
        if latest is None:
            raise FileNotFoundError("no steps available to infer camera meta step")
        nn = latest
    if camera == "agentview":
        path = out_dir / f"camera_meta_{nn:02d}.json"
    elif camera == "wrist":
        path = out_dir / f"camera_meta_wrist_{nn:02d}.json"
    elif camera == "navview":
        path = out_dir / f"camera_meta_nav_{nn:02d}.json"
    else:
        raise ValueError(f"unknown camera: {camera!r}")
    if not path.exists():
        raise FileNotFoundError(f"camera_meta for {camera} step {nn} not found at {path}")
    with open(path) as f:
        return json.load(f)


def _load_world_map(camera: str, nn: int, resolution: str = "low") -> np.ndarray | None:
    """Load a per-pixel world XYZ map as a float32 numpy array, or None.

    Camera values: ``"agentview"``, ``"navview"``, ``"wrist"``.
    Resolution: ``"low"`` (default) or ``"high"`` (agentview only).
    """
    out_dir = get_output_dir()
    if out_dir is None:
        return None
    if camera == "agentview":
        if resolution == "low":
            path = out_dir / f"world_{nn:02d}.npy"
        elif resolution == "high":
            path = out_dir / f"world_hi_{nn:02d}.npy"
        else:
            raise ValueError(f"unknown resolution: {resolution!r}")
    elif camera == "navview":
        path = out_dir / f"world_nav_{nn:02d}.npy"
    elif camera == "wrist":
        path = out_dir / f"world_wrist_{nn:02d}.npy"
    else:
        raise ValueError(f"unknown camera: {camera!r}")
    if not path.exists():
        return None
    return np.load(path)


# ---------------------------------------------------------------------------
# State persistence — _append_state + dump_state
# ---------------------------------------------------------------------------


def _append_state(output_dir: str, blob: dict) -> None:
    """Append *blob* to ``<output_dir>/states.json`` atomically.

    The merged trace is a top-level JSON array (one entry per step). The
    file is rewritten via a tmp + rename so a reader never sees partial
    content. The entry index equals ``blob['step_idx']``.
    """
    path = os.path.join(output_dir, "states.json")
    tmp = path + ".tmp"
    if os.path.exists(path):
        try:
            with open(path) as f:
                arr = json.load(f)
            if not isinstance(arr, list):
                arr = []
        except Exception:
            arr = []
    else:
        arr = []
    idx = int(blob.get("step_idx", len(arr)))
    # Pad with None if the agent ever skips a step (shouldn't happen,
    # but keeps array index == step_idx).
    while len(arr) < idx:
        arr.append(None)
    if len(arr) == idx:
        arr.append(blob)
    else:
        arr[idx] = blob
    with open(tmp, "w") as f:
        json.dump(arr, f, indent=2)
    os.replace(tmp, path)


def dump_state(
    primitives: RoboCasaPrimitives,
    output_dir: str,
    step_idx: int,
    log: dict | None = None,
) -> dict:
    """Call ``primitives.dump_state()`` then append to ``states.json`` with artifact paths.

    Writes:
      - Standard files via ``RoboCasaPrimitives.dump_state()`` (state json,
        images, depth, world maps, camera meta, navview, hi-res, disk pruning)
      - Appends a step blob to ``<output_dir>/states.json`` that records
        world_map / nav_map / hi_map paths so the agent can locate artifacts.

    If *log* is provided (the return value of :func:`execute`), its
    ``command``, ``result``, and ``elapsed_s`` fields are merged into the
    step blob so a single entry captures everything.
    """
    # Call the existing dump_state to write standard files
    state = primitives.dump_state(step_idx)

    # Build states.json blob with artifact paths
    nn = f"{step_idx:02d}"
    blob = {
        "step_idx": step_idx,
        "robocasa_terminated": primitives.env.terminated,
        "task_language": state.get("task_language", ""),
        "state": state.get("state", {}),
        "task_progress": state.get("task_progress", {}),
        "success": state.get("success", False),
        "world_map": (
            f"world_{nn}.npy"
            if os.path.exists(os.path.join(output_dir, f"world_{nn}.npy"))
            else None
        ),
        "world_nav_map": (
            f"world_nav_{nn}.npy"
            if os.path.exists(os.path.join(output_dir, f"world_nav_{nn}.npy"))
            else None
        ),
        "world_hi_map": (
            f"world_hi_{nn}.npy"
            if os.path.exists(os.path.join(output_dir, f"world_hi_{nn}.npy"))
            else None
        ),
    }
    if log is not None:
        blob["command"] = log.get("command")
        blob["result"] = log.get("result")
        blob["elapsed_s"] = log.get("elapsed_s")
    _append_state(output_dir, blob)
    return state


# ---------------------------------------------------------------------------
# Handler skeletons — fleshed out in later waves
# ---------------------------------------------------------------------------


def back_project(
    row: int,
    col: int,
    step: int | None = None,
    camera: str = "agentview",
    resolution: str = "high",
) -> dict:
    """Back-project a single pixel to world XYZ. Skeleton — returns error."""
    return {"error": "not implemented"}


def view_camera_meta(
    camera: str = "agentview",
    step: int | None = None,
) -> dict:
    """Read camera calibration metadata. Skeleton — returns error."""
    return {"error": "not implemented"}


def finish(status: str, summary: str) -> dict:
    """Declare the task finished."""
    return {"_finish": True, "status": status, "summary": summary}


def query_world_map(
    z_min: float = 0.85,
    z_max: float = 0.95,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    camera: str = "agentview",
    resolution: str = "low",
    min_cluster_size: int = 10,
) -> dict:
    """Query the world map by z-range and/or region to find objects."""
    nn = _latest_step()
    if nn is None:
        return {"error": "no state trace available"}
    world_map = _load_world_map(camera, nn, resolution)
    if world_map is None:
        return {"error": f"{camera} {resolution}-resolution world map not found"}

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
        return {"clusters": [], "summary": {"total_clusters": 0, "total_pixels_matched": 0}}

    # Grid-based clustering
    h, w = world_map.shape[:2]
    grid_cells = max(8, min(32, h // 32))
    cell_h = max(1, h // grid_cells)
    cell_w = max(1, w // grid_cells)
    cells = {}
    for i in range(0, len(ys), 5):
        y, x = int(ys[i]), int(xs[i])
        gy, gx = y // cell_h, x // cell_w
        key = (gy, gx)
        if key not in cells:
            cells[key] = {"pixels": [], "world_pts": []}
        cells[key]["pixels"].append((y, x))
        cells[key]["world_pts"].append(world_map[y, x, :3])

    clusters = []
    for key, data in cells.items():
        if len(data["pixels"]) < min_cluster_size:
            continue
        pts = np.array(data["world_pts"])
        center = np.median(pts, axis=0)
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        center_idx = len(data["pixels"]) // 2
        clusters.append({
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
        })

    clusters.sort(key=lambda c: -c["pixel_count"])
    return {
        "clusters": clusters[:20],
        "summary": {
            "total_clusters": len(clusters[:20]),
            "total_pixels_matched": total_pixels,
        },
    }


def view_driver_state(step: int | None = None) -> dict:
    """Read state trace with embedded images and role-labeled image metadata."""
    nn = _latest_step() if step is None else int(step)
    if nn is None:
        return {"error": "no state entries; primitives not ready"}
    try:
        data = _load_step(nn)
    except Exception as e:
        return {"error": f"step {nn} not found: {e}"}

    out = {
        "step": nn,
        "task_progress": data.get("task_progress", {}),
        "task_language": data.get("task_language", ""),
        "state": data.get("state", data),
        "robocasa_terminated": data.get("robocasa_terminated", False),
        "vla_desync": data.get("vla_desync", data.get("state", {}).get("vla_desync", False)),
        "log": {"command": data.get("command"), "result": data.get("result"), "elapsed_s": data.get("elapsed_s")},
        "images": [],
    }

    # Load images and build role-labeled image list
    img_camera = _load_image(nn, "camera")
    if img_camera:
        out["_image_cam_bytes"] = img_camera
        out["images"].append({"role": "calibration_frame", "camera": "agentview", "resolution": "low", "note": "USE THIS IMAGE for back_project pixel picking. Vertical-flipped raw buffer."})

    img_nav = _load_image(nn, "nav")
    if img_nav:
        out["_image_nav_bytes"] = img_nav
        out["images"].append({"role": "nav_view", "camera": "navview", "resolution": "low", "note": "Base-mounted ground camera. Floor pixels (z=0) are walkable."})

    img_wrist = _load_image(nn, "wrist")
    if img_wrist:
        out["_image_wrist_bytes"] = img_wrist
        out["images"].append({"role": "calibration_frame", "camera": "wrist", "resolution": "low", "note": "Eye-in-hand camera. MOVES with gripper. Good for close-range refinement."})

    # Check for hi-res agentview
    out_dir = get_output_dir()
    if out_dir:
        hi_path = out_dir / "images_cam_hi" / f"image_cam_hi_{nn:02d}.png"
        if hi_path.exists():
            out["image_cam_hi_path"] = str(hi_path)
            out["images"].append({"role": "calibration_frame", "camera": "agentview", "resolution": "high", "path": str(hi_path), "note": "High-res version. Divide pixel coords by 4 to convert to low-res."})

    return out


def back_project_batch(
    pixels: list[list[int]],
    step: int | None = None,
    camera: str = "agentview",
    resolution: str = "low",
) -> dict:
    """Back-project multiple pixels to world XYZ in a single call.

    Loads the precomputed world map once and queries all *pixels*, returning
    each result individually plus a summary with the median of valid points.

    Args:
        pixels: List of ``[row, col]`` pixel coordinates.
        step: Step index; ``None`` (default) uses the latest available step.
        camera: Camera view — ``"agentview"`` (default), ``"navview"``, or ``"wrist"``.
        resolution: ``"low"`` (default) or ``"high"`` (agentview only).

    Returns:
        A dict with keys:
        - ``results``: list of per-pixel results (each with ``pixel``,
          ``world_xyz``, ``valid``, ``error``)
        - ``summary``: ``{median_xyz, valid_count, total_count}``
        - ``step``, ``camera``, ``resolution``
    """
    camera = camera or "agentview"
    resolution = resolution or "low"
    nn = _latest_step() if step is None else int(step)
    if nn is None:
        return {"error": "no state trace available"}

    world_map = _load_world_map(camera, nn, resolution)
    if world_map is None:
        return {
            "error": (
                f"{camera} {resolution}-resolution world map not found "
                f"for step {nn}"
            )
        }

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
                    "error": (
                        f"pixel ({row},{col}) out of bounds ({h}x{w})"
                    ),
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
        result = {
            "pixel": [row, col],
            "world_xyz": [
                round(float(xyz[0]), 4),
                round(float(xyz[1]), 4),
                round(float(xyz[2]), 4),
            ],
            "valid": True,
            "error": None,
        }
        results.append(result)
        valid_xyzs.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

    summary = {
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


def write_recipe_from_states(output_dir: str, recipe_tag: str) -> str:
    """Export non-error RoboCasa primitive commands from the state trace as JSONL."""
    states_path = os.path.join(output_dir, "states.json")
    states = json.load(open(states_path)) if os.path.exists(states_path) else []

    primitive_actions = {
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

    commands = []
    for entry in states:
        if not entry:
            continue
        command = entry.get("command")
        if command is None:
            continue
        if command.get("action") not in primitive_actions:
            continue
        result = entry.get("result")
        if isinstance(result, dict) and result.get("error"):
            continue
        commands.append(command)

    recipe_path = os.path.join(output_dir, f"recipe_{recipe_tag}.jsonl")
    tmp_path = recipe_path + ".tmp"
    with open(tmp_path, "w") as f:
        for command in commands:
            f.write(json.dumps(command, separators=(",", ":")) + "\n")
    os.replace(tmp_path, recipe_path)
    return recipe_path

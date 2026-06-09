"""Tool implementations for the hybrid LLM-in-the-loop agent.

Each tool is a thin wrapper that the agent calls via Anthropic's tool-use
API. Results are JSON-serializable dicts; for image-bearing tools the
caller (runner.py) converts a `_image_path` field into a multimodal
content block.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

WORKDIR = Path(os.environ.get("HYBRID_REPL_WORKDIR", "/tmp/hybrid_repl"))


def set_workdir(path: str | os.PathLike) -> None:
    """Override the REPL working directory used by view_repl_state /
    send_command. Call BEFORE the agent loop starts so each parallel
    worker has its own /tmp/hybrid_repl_<tag>/."""
    global WORKDIR
    WORKDIR = Path(path)


# ---------------------------------------------------------------------------
# Tool schema declarations (passed to the Anthropic API)
# ---------------------------------------------------------------------------

TOOLS_SPEC = [
    {
        "name": "read_text_file",
        "description": (
            "Read a UTF-8 text file. Use for guides (STRICT_HYBRID_GUIDE.md, "
            "PRO_HYBRID_GUIDE.md, env_calibration.md), past recipe JSONLs, "
            "audit JSONs, and memory files. Large files are truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or repo-relative path"},
                "max_chars": {"type": "integer", "description": "Max chars (default 40000)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_text_file",
        "description": (
            "Write a UTF-8 text file (creates parent dirs). Use this to save "
            "the working recipe JSONL and the final audit JSON at the end of "
            "a successful run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": (
            "List files in a directory (non-recursive). Default = /tmp/hybrid_repl. "
            "Use to inspect the REPL working directory or to discover existing "
            "recipes in workspace_pro/results_*_pert/."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Default /tmp/hybrid_repl"},
            },
        },
    },
    {
        "name": "view_repl_state",
        "description": (
            "Read state_NN.json + log_NN.json + image_NN.png from /tmp/hybrid_repl. "
            "If step is null, returns the latest. Returns the state JSON and "
            "embeds the agentview PNG as a multimodal image content block "
            "(use this image — JSON state alone is not enough; see Rule 0)."
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
        "name": "send_command",
        "description": (
            "Write a JSON command to /tmp/hybrid_repl/command.json and BLOCK "
            "until the driver writes the next done_NN.flag. Returns the "
            "new state JSON + log JSON + agentview image.\n\n"
            "Schema for the `command` argument follows STRICT_HYBRID_GUIDE.md "
            "§The command vocabulary. ALLOWED actions:\n"
            "  - move_to: {action, xyz:[x,y,z], gripper:-1|+1, tol, step_clip, "
            "max_steps, target_yaw?}\n"
            "  - pi0_pick: {action, prompt, max_chunks, track_obj, "
            "track_obj_lift_thresh, lift_thresh, gripper_closed_thresh} "
            "— the ONLY allowed Pi0 invocation; use it for the grasp.\n"
            "  - release: {action, max_steps}\n"
            "  - set_gripper: {action, gripper:+1|-1, steps}\n"
            "  - rotate_wrist / rotate_pitch (world-Z / world-X reorient, "
            "see guide §Extended primitives)\n\n"
            "  NO teleport: there is no js_move_to / articulate_to / "
            "set_object_pose. Every motion goes through the OSC controller "
            "or Pi0 (real contact). For Close(articulation) / TurnOn, push "
            "with move_to or use pi0_doubled.\n\n"
            "BLOCKED (returns an error if you try): reset, exit. "
            "You get exactly ONE episode — recover from failures within "
            "the current episode, or call finish(status='stuck')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "object",
                    "description": "Command dict per STRICT_HYBRID_GUIDE.md",
                },
                "timeout_s": {
                    "type": "number",
                    "description": "Seconds to wait for done flag (default 600)",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "view_camera_meta",
        "description": (
            "Read camera_meta.json from the REPL workdir. Returns the camera "
            "intrinsics matrix K (3x3), the camera-to-world extrinsic matrix "
            "(4x4), image dimensions, and the back-projection recipe. Use this "
            "in PERCEPTION-ISOLATED mode to localize objects — you do NOT get "
            "GT world coordinates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "back_project",
        "description": (
            "Back-project a pixel (row, col) to a world XYZ point using the "
            "metric depth at that pixel and the camera calibration. "
            "Row 0 = top of image, col 0 = left. Step NN selects which "
            "depth_NN.npy to use (default latest). Returns world_xyz in meters.\n\n"
            "USE THIS to find where an object is in the world — look at "
            "image_cam_NN.png to pick a pixel on the target object, then "
            "call back_project(row, col). Sample several pixels on the object "
            "and median their xy for robustness."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {"type": "integer", "description": "Pixel row (0=top, 255=bottom)"},
                "col": {"type": "integer", "description": "Pixel column (0=left, 255=right)"},
                "step": {
                    "type": ["integer", "null"],
                    "description": "Depth step to use (default latest). 0 for initial.",
                },
            },
            "required": ["row", "col"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Declare the task finished. Call when state.libero_terminated "
            "becomes True, or when genuinely stuck after honest exploration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["success", "failure", "stuck"],
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
# Tool implementations
# ---------------------------------------------------------------------------

# Auto-detect repo root: this file is at <repo>/examples/embodiment/primitives/workspace_pro/hybrid_agent/tools.py
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + f"\n\n[TRUNCATED — file is {len(text)} chars, showed first {max_chars}]"
    )


def read_text_file(path: str, max_chars: int = 40000) -> dict:
    p = _resolve(path)
    if not p.exists():
        return {"error": f"file not found: {p}"}
    if p.is_dir():
        return {"error": f"is a directory: {p}"}
    try:
        text = p.read_text(errors="replace")
    except Exception as e:
        return {"error": str(e)}
    return {"path": str(p), "size": len(text), "content": _truncate(text, max_chars)}


def write_text_file(path: str, content: str) -> dict:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"path": str(p), "bytes_written": len(content.encode("utf-8"))}


def list_dir(path: str = "") -> dict:
    # Default to the current REPL workdir (so parallel agents see their own).
    p = _resolve(path) if path else WORKDIR
    if not p.exists():
        return {"error": f"directory not found: {p}"}
    files = sorted(os.listdir(p))
    return {"path": str(p), "count": len(files), "files": files}


def _latest_step() -> int | None:
    """Return the highest NN for which done_NN.flag exists, else 0 if state_00 only."""
    if not WORKDIR.exists():
        return None
    flag_nums = []
    for f in WORKDIR.glob("done_*.flag"):
        m = re.match(r"done_(\d+)\.flag", f.name)
        if m:
            flag_nums.append(int(m.group(1)))
    if flag_nums:
        return max(flag_nums)
    if (WORKDIR / "state_00.json").exists():
        return 0
    return None


def view_repl_state(step: int | None = None) -> dict:
    if not WORKDIR.exists():
        return {"error": "WORKDIR /tmp/hybrid_repl does not exist; driver not started"}
    if step is None:
        nn = _latest_step()
        if nn is None:
            return {"error": "no state files; driver not ready"}
    else:
        nn = step
    nn_str = f"{nn:02d}"

    state_path = WORKDIR / f"state_{nn_str}.json"
    log_path = WORKDIR / f"log_{nn_str}.json"
    image_path = WORKDIR / f"image_{nn_str}.png"
    image_cam_path = WORKDIR / f"image_cam_{nn_str}.png"

    out: dict = {"step": nn}
    if state_path.exists():
        with open(state_path) as f:
            data = json.load(f)
        out["state"] = data.get("state", data)
        out["libero_terminated"] = data.get("libero_terminated")
    else:
        out["state_error"] = f"missing {state_path}"
    if log_path.exists():
        with open(log_path) as f:
            out["log"] = json.load(f)
    if image_path.exists():
        out["_image_path"] = str(image_path)
    if image_cam_path.exists():
        out["_image_cam_path"] = str(image_cam_path)
    return out


# Actions the agent is NOT allowed to issue. The driver itself accepts them,
# but exposing them to the agent breaks the single-episode contract:
#   - reset: would let the agent retry forever — defeats the purpose of
#     measuring single-attempt success.
#   - exit: belongs to the runner's cleanup path; if the agent issues it
#     mid-run the driver terminates and we lose the audit.
BLOCKED_ACTIONS = {"reset", "exit"}


def send_command(command: dict, timeout_s: float = 600.0) -> dict:
    if not WORKDIR.exists():
        return {"error": "WORKDIR /tmp/hybrid_repl missing; driver not started"}

    action = command.get("action") if isinstance(command, dict) else None
    if action in BLOCKED_ACTIONS:
        return {
            "error": (
                f"action '{action}' is not available to the agent. "
                f"You get ONE episode; if a pick/move fails, recover within "
                f"the current episode (e.g. set_gripper + move_to to re-stage, "
                f"or another pi0_pick after re-pre-positioning). "
                f"Call finish(status='stuck', summary=...) if truly unrecoverable."
            ),
            "blocked_action": action,
        }

    current = _latest_step()
    if current is None:
        return {"error": "no state_00.json; driver not ready"}
    next_n = current + 1
    next_nn = f"{next_n:02d}"

    cmd_path = WORKDIR / "command.json"
    tmp_path = WORKDIR / "command.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(command, f)
    os.replace(tmp_path, cmd_path)  # atomic

    flag_path = WORKDIR / f"done_{next_nn}.flag"
    t0 = time.time()
    while not flag_path.exists():
        time.sleep(0.5)
        if time.time() - t0 > timeout_s:
            return {
                "error": f"timeout after {timeout_s}s waiting for {flag_path.name}",
                "command_sent": command,
            }

    elapsed = time.time() - t0
    result = view_repl_state(next_n)
    result["agent_elapsed_s"] = round(elapsed, 1)
    return result


def finish(status: str, summary: str) -> dict:
    return {"_finish": True, "status": status, "summary": summary}


def view_camera_meta() -> dict:
    """Read camera_meta.json from the workdir for perception-mode localization."""
    p = WORKDIR / "camera_meta.json"
    if not p.exists():
        return {"error": f"camera_meta.json not found in {WORKDIR}; is the driver running in perception mode?"}
    import json as _json
    meta = _json.load(open(p))
    return {"camera_meta": meta}


def back_project(row: int, col: int, step: int | None = None) -> dict:
    """Back-project a pixel to world XYZ using depth + camera calibration."""
    import json as _json
    import numpy as np

    meta_path = WORKDIR / "camera_meta.json"
    if not meta_path.exists():
        return {"error": "camera_meta.json not found"}

    meta = _json.load(open(meta_path))
    E = np.array(meta["extrinsic_cam2world"])

    if step is None:
        nn = _latest_step()
        if nn is None:
            return {"error": "no depth files available"}
    else:
        nn = step
    nn_str = f"{nn:02d}"

    depth_path = WORKDIR / f"depth_{nn_str}.npy"
    if not depth_path.exists():
        return {"error": f"depth file not found: {depth_path}"}

    depth = np.load(depth_path)
    H, W = depth.shape
    if row < 0 or row >= H or col < 0 or col >= W:
        return {"error": f"pixel ({row},{col}) out of bounds; image is {H}x{W}"}

    z = float(depth[row, col])
    if z <= 0 or z > 10:
        return {"error": f"invalid depth {z:.3f}m at pixel ({row},{col}); pick a different pixel"}

    P = E @ np.array([col * z, row * z, z, 1.0])
    world_xyz = [round(float(v), 4) for v in P[:3]]

    return {
        "pixel": [row, col],
        "depth_m": round(z, 4),
        "world_xyz": world_xyz,
        "step": nn,
        "image_size": [H, W],
    }


TOOL_HANDLERS = {
    "read_text_file": read_text_file,
    "write_text_file": write_text_file,
    "list_dir": list_dir,
    "view_repl_state": view_repl_state,
    "send_command": send_command,
    "view_camera_meta": view_camera_meta,
    "back_project": back_project,
    "finish": finish,
}


def execute_tool(name: str, input_dict: dict) -> dict:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return handler(**input_dict)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}", "got": input_dict}
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Convert tool result -> Anthropic content blocks (text + optional image)
# ---------------------------------------------------------------------------

MAX_TEXT_BYTES_IN_RESULT = 60000


def tool_result_to_content_blocks(result):
    """Build a list of Anthropic content blocks from a tool result dict.

    If the result has an `_image_path`, that PNG is included as a base64
    image block (alongside a text block with the JSON state).
    """
    if not isinstance(result, dict):
        return [{"type": "text", "text": str(result)[:MAX_TEXT_BYTES_IN_RESULT]}]

    image_path = result.pop("_image_path", None)
    image_cam_path = result.pop("_image_cam_path", None)
    text = json.dumps(result, indent=2, default=str)
    if len(text) > MAX_TEXT_BYTES_IN_RESULT:
        text = text[:MAX_TEXT_BYTES_IN_RESULT] + "\n[truncated]"

    blocks = [{"type": "text", "text": text}]

    def _add_image(path):
        p = Path(path)
        if p.exists():
            with open(p, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": data,
                },
            })

    if image_path:
        _add_image(image_path)
    if image_cam_path:
        _add_image(image_cam_path)
    return blocks

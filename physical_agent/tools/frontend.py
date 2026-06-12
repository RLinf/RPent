"""Tool implementations for the hybrid LLM-in-the-loop agent.

Each tool is a thin wrapper that the agent calls via an LLM tool-use API.
Results are JSON-serializable dicts; for image-bearing tools the caller
(runner.py) converts private image byte fields into multimodal content blocks.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from physical_agent.transport import DriverClient, FileDriverClient
from physical_agent.utils.config import get_default_workdir_prefix, get_repo_root

REPO_ROOT = get_repo_root()
WORKDIR = Path(
    os.environ.get(
        "HYBRID_DRIVER_WORKDIR",
        os.environ.get("HYBRID_REPL_WORKDIR", get_default_workdir_prefix()),
    )
)
DRIVER_CLIENT: DriverClient = FileDriverClient(WORKDIR)
# Compatibility name used by existing tests/callers.
TRANSPORT = DRIVER_CLIENT


def _workdir_desc() -> str:
    return str(WORKDIR)


def set_workdir(path: str | os.PathLike) -> None:
    """Override the driver working directory used by view_driver_state /
    send_command. Call BEFORE the agent loop starts so each parallel
    worker has its own workdir."""
    global WORKDIR, DRIVER_CLIENT, TRANSPORT
    WORKDIR = Path(path)
    DRIVER_CLIENT = FileDriverClient(WORKDIR)
    TRANSPORT = DRIVER_CLIENT


def set_transport(client: DriverClient) -> None:
    """Override the process-to-driver client used by agent tools."""
    global DRIVER_CLIENT, TRANSPORT
    DRIVER_CLIENT = client
    TRANSPORT = client


def set_driver_client(client: DriverClient) -> None:
    """Override the driver client used by agent tools."""
    set_transport(client)


# ---------------------------------------------------------------------------
# Tool schema declarations (Anthropic-shaped canonical schema)
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
            "List files in a directory (non-recursive). Default = current driver workdir. "
            "Use to inspect the driver working directory or to discover existing "
            "recipes in workspace_pro/results_*_pert/."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Default: current driver workdir"},
            },
        },
    },
    {
        "name": "view_driver_state",
        "description": (
            "Read step NN from `states.json` + the matching "
            "`images/image_NN.png` in the current driver workdir. If step is "
            "null, returns the latest entry. Each entry contains the robot "
            "state, libero_terminated flag, command log, and result. Embeds "
            "the agentview PNG as a multimodal image content block (use this "
            "image — JSON state alone is not enough; see Rule 0)."
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
            "Send a JSON command to the interactive driver and BLOCK "
            "until the next step is available in `states.json`. "
            "Returns the new state entry + log + agentview image.\n\n"
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
                    "description": "Seconds to wait for the next states.json entry (default 600)",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "view_camera_meta",
        "description": (
            "Read camera_meta.json from the driver workdir. Returns the camera "
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
            "`depths/depth_NN.npy` to use (default latest). Returns world_xyz "
            "in meters.\n\n"
            "USE THIS to find where an object is in the world — look at "
            "`images_cam/image_cam_NN.png` to pick a pixel on the target "
            "object, then call back_project(row, col). Sample several pixels "
            "on the object and median their xy for robustness."
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
    # Default to the current driver workdir (so parallel agents see their own).
    p = _resolve(path) if path else WORKDIR
    if not p.exists():
        return {"error": f"directory not found: {p}"}
    files = sorted(os.listdir(p))
    return {"path": str(p), "count": len(files), "files": files}


def _load_states() -> list:
    """Return the parsed driver state trace."""
    return DRIVER_CLIENT.load_states()


def _latest_step() -> int | None:
    return DRIVER_CLIENT.latest_step()


def view_driver_state(step: int | None = None) -> dict:
    latest = _latest_step()
    if latest is None:
        return {"error": "no driver state entries; driver not ready"}
    nn = latest if step is None else int(step)
    try:
        data = DRIVER_CLIENT.load_step(nn)
    except Exception as e:
        return {"error": f"step {nn} not present in driver state trace: {e}"}

    out: dict = {"step": nn}
    out["state"] = data.get("state", data)
    out["libero_terminated"] = data.get("libero_terminated")
    out["log"] = {
        "command": data.get("command"),
        "result": data.get("result"),
        "elapsed_s": data.get("elapsed_s"),
    }
    image = DRIVER_CLIENT.load_image(nn, "agent")
    image_cam = DRIVER_CLIENT.load_image(nn, "camera")
    if image:
        out["_image_bytes"] = image
    if image_cam:
        out["_image_cam_bytes"] = image_cam
    return out


# Actions the agent is NOT allowed to issue. The driver itself accepts them,
# but exposing them to the agent breaks the single-episode contract:
#   - reset: would let the agent retry forever — defeats the purpose of
#     measuring single-attempt success.
#   - exit: belongs to the runner's cleanup path; if the agent issues it
#     mid-run the driver terminates and we lose the audit.
BLOCKED_ACTIONS = {"reset", "exit"}


def _normalize_command(command: dict) -> dict:
    if not isinstance(command, dict):
        raise ValueError("command must be an object")

    out = dict(command)
    action = out.get("action")

    if action in {"move_to", "move_pose"}:
        xyz = out.get("xyz")
        if isinstance(xyz, dict) and set(xyz) == {"item"}:
            xyz = xyz["item"]
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            raise ValueError(
                "xyz must be a JSON array of three numbers, e.g. "
                '{"xyz":[-0.05,0,0.3]}'
            )
        out["xyz"] = [float(v) for v in xyz]

    float_fields = (
        "gripper tol step_clip action_scale target_yaw yaw_step_clip "
        "target_pitch pitch_step yaw_step ori_tol delta_yaw delta_pitch "
        "track_obj_lift_thresh lift_thresh gripper_closed_thresh"
    )
    for key in float_fields.split():
        if key in out and out[key] is not None:
            out[key] = float(out[key])

    for key in ("max_steps", "steps", "max_chunks"):
        if key in out and out[key] is not None:
            out[key] = int(out[key])

    return out


def send_command(command: dict, timeout_s: float = 600.0) -> dict:
    try:
        command = _normalize_command(command)
    except ValueError as e:
        return {
            "error": str(e),
            "hint": (
                "Use plain JSON for motion vectors, e.g. "
                '{"command":{"action":"move_to","xyz":[-0.05,0,0.3],'
                '"gripper":-1}}'
            ),
        }

    action = command.get("action")
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
        return {"error": "no states.json (or empty); driver not ready"}

    driver_result = DRIVER_CLIENT.send_command(
        command,
        current_step=current,
        timeout_s=timeout_s,
    )
    if driver_result.get("error"):
        return driver_result

    step = int(driver_result.get("step", current + 1))
    result = view_driver_state(step)
    if "agent_elapsed_s" in driver_result:
        result["agent_elapsed_s"] = driver_result["agent_elapsed_s"]
    if "driver_exit" in driver_result:
        result["driver_exit"] = driver_result["driver_exit"]
    return result


def finish(status: str, summary: str) -> dict:
    return {"_finish": True, "status": status, "summary": summary}


def view_camera_meta() -> dict:
    """Read camera calibration metadata for perception-mode localization."""
    try:
        meta = DRIVER_CLIENT.load_camera_meta()
    except Exception:
        return {
            "error": (
                f"camera metadata not found for driver workdir {WORKDIR}; "
                "is the driver running in perception mode?"
            )
        }
    return {"camera_meta": meta}


def back_project(row: int, col: int, step: int | None = None) -> dict:
    """Back-project a pixel to world XYZ using depth + camera calibration."""
    import numpy as np

    try:
        meta = DRIVER_CLIENT.load_camera_meta()
    except Exception:
        return {"error": "camera metadata not found"}

    k_matrix = np.array(meta["intrinsic_K"])
    extrinsic = np.array(meta["extrinsic_cam2world"])

    nn = _latest_step() if step is None else step
    if nn is None:
        return {"error": "no depth files available"}

    try:
        depth = DRIVER_CLIENT.load_depth(nn)
    except Exception as e:
        return {"error": f"depth artifact not found for step {nn}: {e}"}
    height, width = depth.shape
    if row < 0 or row >= height or col < 0 or col >= width:
        return {
            "error": f"pixel ({row},{col}) out of bounds; image is {height}x{width}"
        }

    z = float(depth[row, col])
    if z <= 0 or z > 10:
        return {
            "error": (
                f"invalid depth {z:.3f}m at pixel ({row},{col}); "
                "pick a different pixel"
            )
        }

    pixel_h = np.array([float(col), float(row), 1.0])
    camera_xyz = np.linalg.inv(k_matrix) @ pixel_h * z
    world = extrinsic @ np.array([*camera_xyz, 1.0])
    world_xyz = [round(float(v), 4) for v in world[:3]]

    return {
        "pixel": [row, col],
        "depth_m": round(z, 4),
        "world_xyz": world_xyz,
        "step": nn,
        "image_size": [height, width],
    }


TOOL_HANDLERS = {
    "read_text_file": read_text_file,
    "write_text_file": write_text_file,
    "list_dir": list_dir,
    "view_driver_state": view_driver_state,
    "view_repl_state": view_driver_state,
    "send_command": send_command,
    "view_camera_meta": view_camera_meta,
    "back_project": back_project,
    "finish": finish,
}

# Compatibility function name from the earlier REPL tool.
view_repl_state = view_driver_state


def get_tools_spec() -> list[dict]:
    """Return tool schemas with descriptions bound to the current workdir."""
    tools = json.loads(json.dumps(TOOLS_SPEC))
    replacements = {
        "current driver workdir": _workdir_desc(),
        "Default: current driver workdir": f"Default: {_workdir_desc()}",
    }
    for tool in tools:
        desc = tool.get("description", "")
        for old, new in replacements.items():
            desc = desc.replace(old, new)
        tool["description"] = desc
        props = tool.get("input_schema", {}).get("properties", {})
        for prop in props.values():
            prop_desc = prop.get("description", "")
            for old, new in replacements.items():
                prop_desc = prop_desc.replace(old, new)
            prop["description"] = prop_desc
    return tools


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

    If the result has private image bytes, those PNGs are included as base64
    image blocks (alongside a text block with the JSON state).
    """
    if not isinstance(result, dict):
        return [{"type": "text", "text": str(result)[:MAX_TEXT_BYTES_IN_RESULT]}]

    result_for_text = dict(result)
    image = result_for_text.pop("_image_bytes", None)
    image_cam = result_for_text.pop("_image_cam_bytes", None)
    # Legacy private path fields are still supported for external callers.
    image_path = result_for_text.pop("_image_path", None)
    image_cam_path = result_for_text.pop("_image_cam_path", None)
    text = json.dumps(result_for_text, indent=2, default=str)
    if len(text) > MAX_TEXT_BYTES_IN_RESULT:
        text = text[:MAX_TEXT_BYTES_IN_RESULT] + "\n[truncated]"

    blocks = [{"type": "text", "text": text}]

    def _add_image_bytes(data_bytes: bytes):
        data = base64.b64encode(data_bytes).decode("utf-8")
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": data,
            },
        })

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

    if image:
        _add_image_bytes(image)
    if image_cam:
        _add_image_bytes(image_cam)
    if image_path:
        _add_image(image_path)
    if image_cam_path:
        _add_image(image_cam_path)
    return blocks

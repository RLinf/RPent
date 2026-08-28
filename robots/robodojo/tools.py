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

"""RoboDojo tool specs and state helpers (M1: view only)."""

from __future__ import annotations

from rpent.tools.toolkit import readonly

TOOLS_SPEC: list[dict] = [
    {
        "name": "view_env_state",
        "description": (
            "Return the current RoboDojo observation: three RGB cameras "
            "(cam_head, cam_left_wrist, cam_right_wrist), robot joint/ee "
            "state, depth, camera calibration, and the task instruction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "back_project",
        "description": (
            "Project one pixel from a camera image to a world xyz coordinate "
            "using depth + intrinsics/extrinsics. Use to localize objects."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {"type": "integer", "description": "Pixel row (0=top)"},
                "col": {"type": "integer", "description": "Pixel col (0=left)"},
                "camera": {
                    "type": "string",
                    "enum": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
                    "description": "Camera to project from",
                },
            },
            "required": ["row", "col"],
        },
    },
    {
        "name": "segment",
        "description": (
            "Segment an object in a camera image by text prompt (SAM 3.0). "
            "Returns mask bounding box and score."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text_prompt": {"type": "string", "description": "e.g. 'the bottle'"},
                "camera": {
                    "type": "string",
                    "enum": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
                    "description": "Camera to segment (default cam_head)",
                },
                "min_score": {
                    "type": "number",
                    "description": "Min score (default 0.2)",
                },
            },
            "required": ["text_prompt"],
        },
    },
    {
        "name": "move_to",
        "description": (
            "Move an arm end-effector to a world xyz target (scripted motion, "
            "CuRobo IK). gripper: 1=close, -1=open, 0=keep."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "World xyz target for the end-effector",
                },
                "arm": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "description": "Arm to move",
                },
                "gripper": {
                    "type": "number",
                    "description": "1=close, -1=open, 0=keep current",
                },
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "set_gripper",
        "description": ("Open or close an arm gripper. 1=close, -1=open."),
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {
                    "type": "string",
                    "enum": ["left", "right"],
                },
                "gripper": {
                    "type": "number",
                    "description": "1=close, -1=open",
                },
            },
            "required": ["arm", "gripper"],
        },
    },
    {
        "name": "pi0_pick",
        "description": (
            "Closed-loop Pi_05 pick: feed the observation to the Pi_05 policy, "
            "apply its action chunk, and detect grasp success by eef lift + "
            "gripper closure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Pick instruction"},
                "arm": {"type": "string", "enum": ["left", "right"]},
                "max_chunks": {"type": "integer", "description": "Max policy chunks"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "get_reward_details",
        "description": (
            "Return the environment's reward/score breakdown: per-bottle "
            "is_on_bin_bottom status, grippers open, arms home, current "
            "score tier, and success."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_safety_status",
        "description": (
            "Return the env safety monitor state: any bottle reported as "
            "rolling (moving fast) or off-table. If an alarm is present, "
            "stabilize the bottle FIRST before continuing the task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "stabilize",
        "description": (
            "Emergency stabilization: move an arm's open gripper to a world "
            "xyz at table height to block/stop a rolling bottle (call when a "
            "safety alarm reports a rolling bottle)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "World xyz to block (bottle position)",
                },
                "arm": {"type": "string", "enum": ["left", "right"]},
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "place_in_bin",
        "description": (
            "Place a held object into the dustbin: carry above the bin mouth "
            "center, descend BELOW the rim, release, then retract. Use the "
            "bin mouth center you localized (not the near edge)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "bin_center_xyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Bin mouth center at approach height",
                },
                "approach_z": {
                    "type": "number",
                    "description": "Safe carry height above the bin (default 0.95)",
                },
                "drop_z": {
                    "type": "number",
                    "description": "EE z to descend to before release, BELOW the rim (default 0.78)",
                },
            },
            "required": ["arm", "bin_center_xyz"],
        },
    },
]


@readonly
def view_env_state(primitives, state) -> dict:
    primitives._last_obs = primitives.env.get_obs()
    obs = primitives._last_obs
    return _summarize_obs(obs)


def _summarize_obs(obs: dict) -> dict:
    vision = obs.get("vision", {})
    state_data = obs.get("state", {})
    out = {
        "instruction": obs.get("instruction"),
        "cameras": {
            name: {
                "shape": cam.get("shape"),
                "has_depth": cam.get("distance_to_image_plane") is not None
                or cam.get("depth") is not None,
                "intrinsic_matrix": _jsonable(cam.get("intrinsic_matrix")),
                "extrinsic_matrix": _jsonable(cam.get("extrinsic_matrix")),
                "color_dtype": str(cam.get("color", type(None)).dtype)
                if hasattr(cam.get("color"), "dtype")
                else None,
            }
            for name, cam in vision.items()
        },
        "state": _jsonable(state_data),
        "eef": _jsonable(
            {
                "left": state_data.get("left_ee_pose"),
                "right": state_data.get("right_ee_pose"),
            }
        ),
    }
    # Attach the head camera image so the planner can "see".
    head = vision.get("cam_head", {}).get("color")
    if head is not None:
        out["_image_bytes"] = _png_bytes(head)
    return out


def back_project(primitives, state, row, col, camera="cam_head") -> dict:
    """Pixel -> world xyz via depth + camera calibration."""
    import numpy as np

    if primitives._last_obs is None:
        primitives._last_obs = primitives.env.get_obs()
    cam = primitives._last_obs.get("vision", {}).get(camera)
    if cam is None:
        return {"error": f"camera {camera!r} not in observation"}
    depth = cam.get("distance_to_image_plane")
    if depth is None:
        depth = cam.get("depth")
    if depth is None:
        return {"error": "depth not available in observation"}
    K = np.asarray(cam.get("intrinsic_matrix"), dtype=np.float64)
    T = np.asarray(cam.get("extrinsic_matrix"), dtype=np.float64)
    if K.shape != (3, 3) or T.shape != (4, 4):
        return {"error": f"calibration missing/invalid: K={K.shape} T={T.shape}"}
    h, w = int(depth.shape[0]), int(depth.shape[1])
    if not (0 <= int(row) < h and 0 <= int(col) < w):
        return {"error": f"pixel out of range: row 0..{h - 1}, col 0..{w - 1}"}
    d = float(depth[int(row), int(col)])
    if not np.isfinite(d) or d <= 0:
        return {"error": f"invalid depth at pixel: {d}"}
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    # Isaac/Omniverse cameras look along -Z; the distance_to_image_plane
    # annotator returns distance along the optical axis, so the camera-frame
    # z coordinate is NEGATIVE d. (M5 run: without this the table plane
    # lands mirrored; with -d it validates at the observed z≈0.77.)
    p_cam = np.array(
        [(int(col) - cx) / fx * d, (int(row) - cy) / fy * d, -d, 1.0],
        dtype=np.float64,
    )
    p_world = T @ p_cam
    return {
        "camera": camera,
        "pixel": [int(row), int(col)],
        "depth_m": round(d, 4),
        "world_xyz": [round(float(v), 4) for v in p_world[:3]],
    }


def segment(primitives, state, text_prompt, camera="cam_head", min_score=0.2) -> dict:
    """Segment an object by text prompt using SAM 3.0."""
    if primitives._last_obs is None:
        primitives._last_obs = primitives.env.get_obs()
    cam = primitives._last_obs.get("vision", {}).get(camera)
    if cam is None:
        return {"error": f"camera {camera!r} not in observation"}
    color = cam.get("color")
    if color is None:
        return {"error": "camera color image missing"}
    sam3 = getattr(primitives, "sam3_client", None)
    if sam3 is None:
        return {"error": "sam3_client not configured"}
    result = sam3.segment(color, text_prompt=text_prompt, min_score=min_score)
    if not result.found:
        return {"camera": camera, "found": False, "text_prompt": text_prompt}
    return {
        "camera": camera,
        "found": True,
        "score": result.score,
        "box_px": _jsonable(result.box),
    }


def _arm_ee_pose_key(arm: str) -> str:
    return f"{arm}_ee_pose"


def _arm_ee_joint_key(arm: str) -> str:
    return f"{arm}_ee_joint_state"


def _refresh_obs(primitives) -> dict:
    primitives._last_obs = primitives.env.get_obs()
    return primitives._last_obs


def move_to(
    primitives,
    state,
    xyz,
    arm="right",
    gripper=0,
    tol=0.01,
    max_steps=20,
) -> dict:
    """Scripted ee motion to a world xyz (CuRobo IK via the ee action path)."""
    import numpy as np

    target = [float(v) for v in xyz]
    if len(target) != 3:
        return {"error": "xyz must have 3 values"}
    obs = _refresh_obs(primitives)
    start_ee = obs.get("state", {}).get(_arm_ee_pose_key(arm))
    if start_ee is None:
        return {"error": f"{arm} ee_pose not in state"}
    start = [float(v) for v in np.asarray(start_ee)[:3]]
    final_xyz = list(start)
    dist_to_target = float(np.linalg.norm(np.asarray(target) - np.asarray(final_xyz)))
    steps_used = 0
    reached = False
    last_error = None
    for step in range(max_steps):
        if dist_to_target <= tol:
            reached = True
            break
        obs = _refresh_obs(primitives)
        action: dict
        action_type = "joint"
        # Preferred: position-only IK over several orientation candidates so
        # lateral targets do not diverge; fall back to the fixed-orientation
        # ee path if the solver reports no reachable solution.
        ik = primitives.env.solve_ik_position(arm, target)
        if ik.get("status") == "Success":
            action = {f"{arm}_arm_joint_state": ik["joint_value"]}
            for a in ("left", "right"):
                if a == arm:
                    continue
                joints = obs.get("state", {}).get(f"{a}_arm_joint_state")
                if joints is not None:
                    action[f"{a}_arm_joint_state"] = list(
                        np.asarray(joints, dtype=np.float64)
                    )
        else:
            last_error = ik.get("error")
            action_type = "ee"
            ee_pose = obs.get("state", {}).get(_arm_ee_pose_key(arm))
            if ee_pose is None:
                return {"error": f"{arm} ee_pose missing mid-motion"}
            ee_pose = list(np.asarray(ee_pose, dtype=np.float64))
            ee_pose[:3] = target
            action = {_arm_ee_pose_key(arm): ee_pose}
            for a in ("left", "right"):
                if a == arm:
                    continue
                pose = obs.get("state", {}).get(_arm_ee_pose_key(a))
                if pose is not None:
                    action[_arm_ee_pose_key(a)] = list(
                        np.asarray(pose, dtype=np.float64)
                    )
        for a in ("left", "right"):
            joint = obs.get("state", {}).get(_arm_ee_joint_key(a))
            if joint is None:
                continue
            if a == arm and gripper != 0:
                # Env convention: normalized joint 1.0 = OPEN, 0.0 = CLOSED
                # (reward is_all_gripper_open uses >=0.8). Tool arg semantics:
                # 1=close, -1=open. Map close -> 0.0, open -> 1.0.
                action[_arm_ee_joint_key(a)] = [0.0 if gripper > 0 else 1.0]
            else:
                action[_arm_ee_joint_key(a)] = [float(np.asarray(joint).reshape(-1)[0])]
        result = primitives.env.apply_action(action, action_type)
        steps_used = step + 1
        final = result["obs"].get("state", {}).get(_arm_ee_pose_key(arm))
        if final is None:
            break
        final_xyz = [float(v) for v in np.asarray(final)[:3]]
        dist_to_target = float(
            np.linalg.norm(np.asarray(target) - np.asarray(final_xyz))
        )
        if result["status"].get("step", 0) >= result["status"].get("step_limit", 1):
            break
    return {
        "arm": arm,
        "target_xyz": target,
        "start_xyz": start,
        "final_xyz": final_xyz,
        "dist_to_target_m": round(dist_to_target, 4),
        "steps_used": steps_used,
        "reached": reached,
        "ik_error": last_error,
    }


def set_gripper(primitives, state, arm, gripper) -> dict:
    """Open (<=0) or close (>0) one gripper without moving the arm."""
    import numpy as np

    obs = _refresh_obs(primitives)
    action: dict = {}
    for a in ("left", "right"):
        pose = obs.get("state", {}).get(_arm_ee_pose_key(a))
        joint = obs.get("state", {}).get(_arm_ee_joint_key(a))
        if pose is not None:
            action[_arm_ee_pose_key(a)] = list(np.asarray(pose, dtype=np.float64))
        if joint is not None:
            # Env convention: 1.0 = OPEN, 0.0 = CLOSED. Tool arg: 1=close, -1=open.
            val = 0.0 if (a == arm and gripper > 0) else 1.0
            if a != arm:
                val = float(np.asarray(joint).reshape(-1)[0])
            action[_arm_ee_joint_key(a)] = [val]
    result = primitives.env.apply_action(action, "ee")
    return {
        "arm": arm,
        "gripper": "closed" if gripper > 0 else "open",
        "status": result["status"],
    }


def pi0_pick(
    primitives,
    state,
    prompt,
    arm="right",
    max_chunks=8,
    lift_thresh=0.04,
    gripper_closed_thresh=0.55,
) -> dict:
    """Closed-loop Pi_05 pick driven by the policy's own action chunks."""
    import numpy as np

    vla = getattr(primitives, "vla_client", None)
    if vla is None:
        return {"error": "vla_client not configured"}
    arms = ("left", "right")
    track = {
        a: {
            "start_z": None,
            "min_z": None,
            "peak": None,
            "start_grip": 1.0,
            "last_grip": 1.0,
        }
        for a in arms
    }
    chunks_used = 0
    success = False
    terminated = False
    for c in range(max_chunks):
        obs = _refresh_obs(primitives)
        obs["instruction"] = prompt
        for a in arms:
            t = track[a]
            if t["start_z"] is None:
                pose = np.asarray(obs["state"][f"{a}_ee_pose"], dtype=np.float64)
                t["start_z"] = float(pose[2])
                t["min_z"] = t["start_z"]
                t["peak"] = t["start_z"]
                t["start_grip"] = float(
                    np.asarray(obs["state"][f"{a}_ee_joint_state"]).reshape(-1)[0]
                )
                t["last_grip"] = t["start_grip"]
        actions = vla.predict(obs)
        chunks_used = c + 1
        for action in actions:
            r = primitives.env.apply_action(action, primitives.action_type)
            st = r["obs"]["state"]
            for a in arms:
                t = track[a]
                z = float(np.asarray(st[f"{a}_ee_pose"], dtype=np.float64)[2])
                grip = float(np.asarray(st[f"{a}_ee_joint_state"]).reshape(-1)[0])
                t["last_grip"] = grip
                if z < t["min_z"]:
                    t["min_z"] = z
                    t["peak"] = z
                else:
                    t["peak"] = max(t["peak"], z)
                descended = (t["start_z"] - t["min_z"]) >= 0.06
                if descended:
                    ascended = (t["peak"] - t["min_z"]) >= lift_thresh
                    closed = grip < gripper_closed_thresh
                    if ascended and closed:
                        success = True
                        break
            if r["status"]["step"] >= r["status"]["step_limit"]:
                terminated = True
                break
        if success or terminated:
            break
    per_arm = {}
    for a in arms:
        t = track[a]
        per_arm[a] = {
            "start_eef_z": round(t["start_z"], 4),
            "min_eef_z": round(t["min_z"], 4),
            "peak_eef_z": round(t["peak"], 4),
            "peak_lift_m": round(t["peak"] - t["min_z"], 4),
            "start_gripper": round(t["start_grip"], 3),
            "final_gripper": round(t["last_grip"], 3),
        }
    return {
        "monitored_arm": arm,
        "instruction": prompt,
        "success": success,
        "chunks_used": chunks_used,
        "arms": per_arm,
        "terminated": terminated,
    }


@readonly
def get_reward_details(primitives, state) -> dict:
    """Read the current reward/score breakdown without changing the env."""
    return primitives.env.get_reward_details()


@readonly
def get_safety_status(primitives, state) -> dict:
    """Read the env safety monitor (rolling / off-table alarms)."""
    return primitives.env.get_safety_status()


def stabilize(primitives, state, xyz, arm="right") -> dict:
    """Block a rolling bottle: place the open gripper at table height at xyz."""

    target = [float(v) for v in xyz]
    target[2] = max(target[2], 0.80)  # keep at/above table height
    return move_to(
        primitives, state, target, arm=arm, gripper=-1, tol=0.015, max_steps=20
    )


def place_in_bin(
    primitives,
    state,
    arm,
    bin_center_xyz,
    approach_z=0.95,
    drop_z=0.78,
) -> dict:
    """Carry to the bin mouth center, descend below the rim, release, retract."""

    center = [float(v) for v in bin_center_xyz]
    if len(center) != 3:
        return {"error": "bin_center_xyz must have 3 values"}
    cx, cy = center[0], center[1]
    phases = {
        "approach": [cx, cy, float(approach_z), +1],  # hold the bottle
        "descend": [cx, cy, float(drop_z), +1],  # hold while below rim
        "retract": [cx, cy, float(approach_z), -1],  # already released
    }
    results = {}
    for name, (tx, ty, tz, grip) in phases.items():
        m = move_to(
            primitives,
            state,
            [tx, ty, tz],
            arm=arm,
            gripper=grip,
            tol=0.015,
            max_steps=25,
        )
        results[name] = {
            "target": [tx, ty, tz],
            "gripper": "hold" if grip > 0 else "open",
            "dist_to_target_m": m.get("dist_to_target_m"),
            "reached": m.get("reached"),
            "steps_used": m.get("steps_used"),
            "ik_error": m.get("ik_error"),
        }
        if not m.get("reached", False):
            return {
                "arm": arm,
                "phase": name,
                "reached": False,
                "phase_results": results,
                "error": f"move_to failed at phase {name}: {m.get('ik_error')}",
            }
    # release
    g = set_gripper(primitives, state, arm, -1)
    results["release"] = {"gripper": "open", "status": g.get("status")}
    return {
        "arm": arm,
        "bin_center_xyz": [cx, cy],
        "approach_z": float(approach_z),
        "drop_z": float(drop_z),
        "phases": results,
        "released": True,
    }


def _png_bytes(arr) -> bytes:
    import io

    import imageio.v2 as imageio

    buf = io.BytesIO()
    imageio.imwrite(buf, arr, format="png")
    return buf.getvalue()


def _jsonable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def dump_state(primitives, state, *, log: dict | None = None) -> dict:
    """Capture one state record after a stateful tool (M1: view snapshot)."""
    obs = primitives._last_obs
    summary = _summarize_obs(obs) if obs is not None else {"error": "no obs yet"}
    state_payload = {k: v for k, v in summary.items() if k != "_image_bytes"}
    with state.record_step(
        state={"instruction": (obs or {}).get("instruction"), **state_payload},
        command=(log or {}).get("command"),
        result=(log or {}).get("result"),
        elapsed_s=(log or {}).get("elapsed_s"),
    ) as step_idx:
        pass
    return {**state_payload, "step_idx": step_idx}

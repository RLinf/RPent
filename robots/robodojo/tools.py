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

"""RoboDojo tool specs and backend-specific control adapters."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from rpent.tools import ToolResult, tool

# (camera key in the obs dict, artifact base name). Mirrors the ``frame_channels``
# declared in ``robot_spec.ROBODOJO_DASHBOARD_SPEC``.
CAMERA_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("cam_head", "cam_head.png"),
    ("cam_left_wrist", "cam_left_wrist.png"),
    ("cam_right_wrist", "cam_right_wrist.png"),
)


@tool(readonly=True, exclude=("state",))
def view_env_state(step: int = -1, *, state: Any) -> ToolResult:
    """Read one recorded RoboDojo state: robot joint/ee state, camera shapes and calibration, the task instruction, and the three camera RGB images (cam_head, cam_left_wrist, cam_right_wrist). Step -1 selects the latest record.

    Args:
        step: Recorded step to read (-1 = latest)"""
    try:
        record = state.get(step)
    except Exception as error:
        payload = {"error": f"state step not available: {error}"}
        return ToolResult(data=payload)
    result: dict[str, Any] = {
        "step": record.step_idx,
        "truncated": record.truncated,
        "state": record.state,
        "artifacts": sorted(record.artifacts),
        "task_language": record.extras.get("task_language"),
    }
    result["log"] = {
        "command": record.command,
        "result": record.result,
        "elapsed_s": record.elapsed_s,
    }
    images: list[bytes] = []
    for _, artifact in CAMERA_ARTIFACTS:
        if artifact in record.artifacts:
            try:
                images.append(state.load_bytes(artifact, step=record.step_idx))
            except FileNotFoundError:
                pass
    return ToolResult(data=result, images=images)


def _summarize_obs(obs: dict) -> dict:
    from robots.robodojo.access import public_observation

    obs = public_observation(obs)
    vision = obs.get("vision", {})
    state_data = obs.get("state", {})
    return {
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


@tool(readonly=True, exclude=("primitives", "state"))
def back_project(
    primitives: Any,
    state: Any,
    row: int,
    col: int,
    camera: Literal["cam_head", "cam_left_wrist", "cam_right_wrist"] = "cam_head",
) -> ToolResult:
    """Project one pixel from a camera image to a world xyz coordinate using depth + intrinsics/extrinsics. Use to localize objects.

    Args:
        row: Pixel row (0=top)
        col: Pixel col (0=left)
        camera: Camera to project from"""
    import numpy as np

    if primitives._last_obs is None:
        primitives._last_obs = primitives.env.get_obs()
    cam = primitives._last_obs.get("vision", {}).get(camera)
    if cam is None:
        payload = {"error": f"camera {camera!r} not in observation"}
        return ToolResult(data=payload)
    depth = cam.get("distance_to_image_plane")
    if depth is None:
        depth = cam.get("depth")
    if depth is None:
        payload = {"error": "depth not available in observation"}
        return ToolResult(data=payload)
    K = np.asarray(cam.get("intrinsic_matrix"), dtype=np.float64)
    T = np.asarray(cam.get("extrinsic_matrix"), dtype=np.float64)
    if K.shape != (3, 3) or T.shape != (4, 4):
        payload = {"error": f"calibration missing/invalid: K={K.shape} T={T.shape}"}
        return ToolResult(data=payload)
    h, w = int(depth.shape[0]), int(depth.shape[1])
    if not (0 <= int(row) < h and 0 <= int(col) < w):
        payload = {"error": f"pixel out of range: row 0..{h - 1}, col 0..{w - 1}"}
        return ToolResult(data=payload)
    d = float(depth[int(row), int(col)])
    if not np.isfinite(d) or d <= 0:
        payload = {"error": f"invalid depth at pixel: {d}"}
        return ToolResult(data=payload)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    # Isaac/Omniverse cameras look along -Z; the distance_to_image_plane
    # annotator returns distance along the optical axis, so the camera-frame
    # z coordinate is NEGATIVE d. Image rows grow downward while the USD
    # camera frame has +Y up, so the row offset enters negated.
    p_cam = np.array(
        [(int(col) - cx) / fx * d, -(int(row) - cy) / fy * d, -d, 1.0],
        dtype=np.float64,
    )
    p_world = T @ p_cam
    payload = {
        "camera": camera,
        "pixel": [int(row), int(col)],
        "depth_m": round(d, 4),
        "world_xyz": [round(float(v), 4) for v in p_world[:3]],
    }
    return ToolResult(data=payload)


@tool(readonly=True, exclude=("primitives", "state"))
def segment(
    primitives: Any,
    state: Any,
    text_prompt: str,
    camera: Literal["cam_head", "cam_left_wrist", "cam_right_wrist"] = "cam_head",
    min_score: float = 0.2,
) -> ToolResult:
    """Segment an object in a camera image by text prompt (SAM 3.0). Returns mask bounding box and score.

    Args:
        text_prompt: e.g. 'the bottle'
        camera: Camera to segment (default cam_head)
        min_score: Min score (default 0.2)"""
    if primitives._last_obs is None:
        primitives._last_obs = primitives.env.get_obs()
    cam = primitives._last_obs.get("vision", {}).get(camera)
    if cam is None:
        payload = {"error": f"camera {camera!r} not in observation"}
        return ToolResult(data=payload)
    color = cam.get("color")
    if color is None:
        payload = {"error": "camera color image missing"}
        return ToolResult(data=payload)
    sam3 = getattr(primitives, "sam3_client", None)
    if sam3 is None:
        payload = {"error": "sam3_client not configured"}
        return ToolResult(data=payload)
    result = sam3.segment(color, text_prompt=text_prompt, min_score=min_score)
    if not result.found:
        payload = {"camera": camera, "found": False, "text_prompt": text_prompt}
        return ToolResult(data=payload)
    from robots.robodojo.flash.grounding import mask_geometry

    payload = {
        "camera": camera,
        "found": True,
        "score": result.score,
        "box_px": _jsonable(result.box),
        **mask_geometry(result.mask),
    }
    return ToolResult(data=payload)


def _arm_ee_pose_key(arm: str) -> str:
    return f"{arm}_ee_pose"


def _arm_ee_joint_key(arm: str) -> str:
    return f"{arm}_ee_joint_state"


def _refresh_obs(primitives) -> dict:
    primitives._last_obs = primitives.env.get_obs()
    return primitives._last_obs


@tool(readonly=False, exclude=("primitives", "state", "tol", "max_steps"))
def move_to(
    primitives: Any,
    state: Any,
    xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    arm: Literal["left", "right"] = "right",
    gripper: float = 0,
    tol: Any = 0.01,
    max_steps: Any = 20,
) -> ToolResult:
    """Move an arm end-effector to a world xyz target (scripted motion, CuRobo IK). gripper: 1=close, -1=open, 0=keep.

    Args:
        xyz: World xyz target for the end-effector
        arm: Arm to move
        gripper: 1=close, -1=open, 0=keep current"""
    import numpy as np

    target = [float(v) for v in xyz]
    if len(target) != 3:
        payload = {"error": "xyz must have 3 values"}
        return ToolResult(data=payload)
    obs = _refresh_obs(primitives)
    start_ee = obs.get("state", {}).get(_arm_ee_pose_key(arm))
    if start_ee is None:
        payload = {"error": f"{arm} ee_pose not in state"}
        return ToolResult(data=payload)
    start = [float(v) for v in np.asarray(start_ee)[:3]]
    final_xyz = list(start)
    dist_to_target = float(np.linalg.norm(np.asarray(target) - np.asarray(final_xyz)))
    steps_used = 0
    reached = False
    last_error = None
    for step in range(max_steps):
        primitives._check_cancelled()
        if dist_to_target <= tol:
            reached = True
            break
        obs = _refresh_obs(primitives)
        action: dict
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
            ee_pose = obs.get("state", {}).get(_arm_ee_pose_key(arm))
            if ee_pose is None:
                payload = {"error": f"{arm} ee_pose missing mid-motion"}
                return ToolResult(data=payload)
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
        obs_step, _reward, _done, info = primitives.env.step(action)
        steps_used = step + 1
        final = obs_step.get("state", {}).get(_arm_ee_pose_key(arm))
        if final is None:
            break
        final_xyz = [float(v) for v in np.asarray(final)[:3]]
        dist_to_target = float(
            np.linalg.norm(np.asarray(target) - np.asarray(final_xyz))
        )
        if info["status"].get("step", 0) >= info["status"].get("step_limit", 1):
            break
    payload = {
        "arm": arm,
        "target_xyz": target,
        "start_xyz": start,
        "final_xyz": final_xyz,
        "dist_to_target_m": round(dist_to_target, 4),
        "steps_used": steps_used,
        "reached": reached,
        "ik_error": last_error,
    }
    return ToolResult(data=payload)


@tool(readonly=False, exclude=("primitives", "state"))
def set_gripper(
    primitives: Any, state: Any, arm: Literal["left", "right"], gripper: float
) -> ToolResult:
    """Open or close an arm gripper. 1=close, -1=open.

    Args:
        gripper: 1=close, -1=open"""
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
    _obs_step, _reward, _done, info = primitives.env.step(action)
    payload = {
        "arm": arm,
        "gripper": "closed" if gripper > 0 else "open",
        "status": {
            key: info["status"][key]
            for key in ("step", "step_limit")
            if key in info["status"]
        },
    }
    return ToolResult(data=payload)


@tool(
    readonly=False,
    exclude=("primitives", "state", "lift_thresh", "gripper_closed_thresh"),
    json_schema_extra={"required": []},
)
def pi0_pick(
    primitives: Any,
    state: Any,
    prompt: str | SkipJsonSchema[None] = None,
    arm: Literal["left", "right"] = "right",
    max_chunks: int = 8,
    lift_thresh: Any = 0.04,
    gripper_closed_thresh: Any = 0.55,
) -> ToolResult:
    """Closed-loop Pi_05 pick: feed the observation to the Pi_05 policy, apply its action chunk, and detect grasp success by eef lift + gripper closure.

    Args:
        prompt: Optional specific target instruction; omit to use resolved official task language
        max_chunks: Max policy chunks"""
    import numpy as np

    from robots.robodojo.tasks import validate_instruction

    prompt = validate_instruction(
        primitives.env.get_task_language() if prompt is None else prompt
    )

    vla = getattr(primitives, "vla_client", None)
    if vla is None:
        payload = {"error": "vla_client not configured"}
        return ToolResult(data=payload)
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
        primitives._check_cancelled()
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
            obs_step, _reward, _done, info = primitives.env.step(action)
            st = obs_step["state"]
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
            if success:
                # Stop the chunk at once: the remaining actions were predicted
                # for an ungrasped state and would reopen the gripper, dropping
                # the object while ``success`` stays true.
                break
            if info["status"]["step"] >= info["status"]["step_limit"]:
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
    payload = {
        "monitored_arm": arm,
        "instruction": prompt,
        "success": success,
        "chunks_used": chunks_used,
        "arms": per_arm,
        "terminated": terminated,
    }
    return ToolResult(data=payload)


def get_reward_details(primitives, state) -> dict:
    """Read the current reward/score breakdown without changing the env."""
    return primitives.env.get_reward_details()


def get_safety_status(primitives, state) -> dict:
    """Read the env safety monitor (rolling / off-table alarms)."""
    return primitives.env.get_safety_status()


@tool(readonly=False, exclude=("primitives", "state"))
def stabilize(
    primitives: Any,
    state: Any,
    xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    arm: Literal["left", "right"] = "right",
) -> ToolResult:
    """Emergency stabilization: move an arm's open gripper to a world xyz at table height to block/stop a bottle observed rolling in the camera images.

    Args:
        xyz: World xyz to block (bottle position)"""

    target = [float(v) for v in xyz]
    target[2] = max(target[2], 0.80)  # keep at/above table height
    return move_to(
        primitives, state, target, arm=arm, gripper=-1, tol=0.015, max_steps=20
    )


@tool(readonly=False, exclude=("primitives", "state"))
def place_in_bin(
    primitives: Any,
    state: Any,
    arm: Literal["left", "right"],
    bin_center_xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    approach_z: float = 0.95,
    drop_z: float = 0.78,
) -> ToolResult:
    """Place a held object into the dustbin: carry above the bin mouth center, descend BELOW the rim, release, then retract. Use the bin mouth center you localized (not the near edge).

    Args:
        bin_center_xyz: Bin mouth center at approach height
        approach_z: Safe carry height above the bin (default 0.95)
        drop_z: EE z to descend to before release, BELOW the rim (default 0.78)"""

    center = [float(v) for v in bin_center_xyz]
    if len(center) != 3:
        payload = {"error": "bin_center_xyz must have 3 values"}
        return ToolResult(data=payload)
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
            "dist_to_target_m": m.data.get("dist_to_target_m"),
            "reached": m.data.get("reached"),
            "steps_used": m.data.get("steps_used"),
            "ik_error": m.data.get("ik_error"),
        }
        if not m.data.get("reached", False):
            payload = {
                "arm": arm,
                "phase": name,
                "reached": False,
                "phase_results": results,
                "error": f"move_to failed at phase {name}: {m.data.get('ik_error')}",
            }
            return ToolResult(data=payload)
    # release
    g = set_gripper(primitives, state, arm, -1)
    results["release"] = {"gripper": "open", "status": g.data.get("status")}
    payload = {
        "arm": arm,
        "bin_center_xyz": [cx, cy],
        "approach_z": float(approach_z),
        "drop_z": float(drop_z),
        "phases": results,
        "released": True,
    }
    return ToolResult(data=payload)


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


def dump_state(primitives, state, *, log: dict | None = None, perception=None):
    """Record one RoboDojo observation and save its camera artifacts.

    Fetches a live observation, appends a :class:`StepRecord` through
    :meth:`EnvState.record_step`, writes the three camera RGB PNGs for that
    step, and returns the record.
    """
    obs = _refresh_obs(primitives)
    status = primitives.env.get_status()
    log = log or {}
    with state.record_step(
        state=_summarize_obs(obs),
        # Official success is read by the runner after planner execution,
        # not written into planner-readable state artifacts.
        terminated=False,
        truncated=int(status.get("step", 0)) >= int(status.get("step_limit", 0) or 0),
        command=log.get("command"),
        result=log.get("result"),
        elapsed_s=log.get("elapsed_s"),
        extras={
            "task_language": primitives.env.get_task_language(),
            **({"perception": perception} if perception else {}),
        },
    ) as step_idx:
        vision = obs.get("vision", {})
        for camera, artifact in CAMERA_ARTIFACTS:
            color = (vision.get(camera) or {}).get("color")
            if color is not None:
                state.save(artifact, color, step=step_idx)
    return state.get(step_idx)

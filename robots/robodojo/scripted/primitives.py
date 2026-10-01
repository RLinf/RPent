# Copyright 2026 The RPent Authors.
"""Numerical motion helpers embedded in trusted recipe workers."""

from __future__ import annotations


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
    _obs_step, _reward, _done, info = primitives.env.step(action)
    return {
        "arm": arm,
        "gripper": "closed" if gripper > 0 else "open",
        "status": {
            key: info["status"][key]
            for key in ("step", "step_limit")
            if key in info["status"]
        },
    }

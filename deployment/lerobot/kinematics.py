"""SO101 forward / inverse kinematics in the arm base frame.

Thin wrapper over LeRobot's placo-based
:class:`lerobot.model.kinematics.RobotKinematics`, pinned to the SO101 arm
joints and the ``gripper_frame_link`` tip frame. Forward kinematics returns
``T_base_gripper`` (the gripper pose in the ``base_link`` world frame); inverse
kinematics maps a desired base-frame gripper pose back to joint targets.
"""
from __future__ import annotations

import os

import numpy as np

# Arm joints in bus-ID order (no gripper); must match the URDF joint names.
_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
_DEFAULT_URDF = "~/.cache/huggingface/lerobot/urdf/so101.urdf"
_TARGET_FRAME = "gripper_frame_link"


class SO101Kinematics:
    """FK/IK for the SO101 arm, expressed in the ``base_link`` world frame."""

    def __init__(self, urdf_path: str | None = None, target_frame: str = _TARGET_FRAME):
        from lerobot.model.kinematics import RobotKinematics

        urdf = os.path.expanduser(urdf_path or _DEFAULT_URDF)
        if not os.path.isfile(urdf):
            raise FileNotFoundError(
                f"SO101 URDF not found at {urdf}. Set urdf_path or download the "
                "SO101 URDF (see toolkits/lerobot/compute_ee_pose.py --urdf help)."
            )
        self._kin = RobotKinematics(
            urdf_path=urdf,
            target_frame_name=target_frame,
            joint_names=list(_ARM_JOINTS),
        )

    def fk(self, joints_deg) -> np.ndarray:
        """Forward kinematics: 5 arm joint angles (deg) -> ``T_base_gripper`` (4x4)."""
        q = np.asarray(joints_deg, dtype=np.float64).reshape(-1)[: len(_ARM_JOINTS)]
        return np.asarray(self._kin.forward_kinematics(q), dtype=np.float64)

    def ik(
        self,
        current_joints_deg,
        T_base_gripper_des,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        max_iters: int = 60,
        pos_tol_m: float = 0.002,
        orient_tol_deg: float = 2.0,
    ) -> np.ndarray:
        """Inverse kinematics: desired ``T_base_gripper`` (4x4) -> arm joints (deg).

        placo's solver advances one step per call, so a single solve undershoots
        on large targets. We iterate (feeding each solution back as the seed)
        until the FK error is within tolerance or ``max_iters`` is hit; the
        result is the best effort (check the FK error if it matters).

        With ``orientation_weight <= 0`` only the position error (``pos_tol_m``)
        gates convergence and the target rotation is ignored. With a positive
        ``orientation_weight`` the solve also matches the rotation of
        ``T_base_gripper_des``, and convergence additionally requires the
        orientation error below ``orient_tol_deg`` (used by ``move_to``'s
        top-down mode).

        ``current_joints_deg`` seeds the first iteration (its gripper entry, if
        any, is preserved by LeRobot's solver).
        """
        q = np.asarray(current_joints_deg, dtype=np.float64).reshape(-1)
        T = np.asarray(T_base_gripper_des, dtype=np.float64)
        target = T[:3, 3]
        R_des = T[:3, :3]
        n = len(_ARM_JOINTS)
        for _ in range(max_iters):
            q = np.asarray(
                self._kin.inverse_kinematics(
                    q, T,
                    position_weight=position_weight,
                    orientation_weight=orientation_weight,
                ),
                dtype=np.float64,
            )
            T_cur = self.fk(q[:n])
            if np.linalg.norm(T_cur[:3, 3] - target) >= pos_tol_m:
                continue
            if orientation_weight <= 0.0:
                break
            cos_ang = (np.trace(R_des.T @ T_cur[:3, :3]) - 1.0) / 2.0
            if np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))) < orient_tol_deg:
                break
        return q

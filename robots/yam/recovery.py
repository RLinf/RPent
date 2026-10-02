# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Operator-confirmed upward retreat from a positive table guard margin."""

import numpy as np

from robots.yam.geometry import ARM_SLICES, enforce_hard_limits, pose_to_matrix


class UpwardRetreat:
    def __init__(self, geometry, qpos, arm, distance_m):
        if (
            arm not in ARM_SLICES
            or not np.isfinite(distance_m)
            or not 0.02 <= distance_m <= 0.05
        ):
            raise ValueError("upward retreat requires left/right and 20-50 mm")
        self.geometry = geometry
        self.arm = arm
        self.start = enforce_hard_limits(qpos, geometry.lower, geometry.upper)
        self.distance = distance_m
        self.pose = geometry.eef_pose(arm, self.start)
        self.rotation = pose_to_matrix(self.pose)[:3, :3]
        self.indices = ARM_SLICES[arm]
        self.other = ARM_SLICES["right" if arm == "left" else "left"]
        if self.start[self.indices][-1] < 0.9:
            raise ValueError("retreat requires an already open, empty gripper")
        check = geometry.check_qpos_transition(self.start, self.start)
        if (
            check.get("reason") != "link_table_guard"
            or set(check.get("bodies", [])) != {"world", arm + "_linear_module"}
            or check.get("distance_m", 0) < 0.02
        ):
            raise ValueError(
                "retreat only supports positive linear-module/table clearance >=20 mm"
            )
        self.escape = (check["bodies"], check["distance_m"] - 0.0005)

    def check(self, start, target):
        g = self.geometry
        start = enforce_hard_limits(start, g.lower, g.upper)
        target = enforce_hard_limits(target, g.lower, g.upper)
        count = max(8, int(np.ceil(np.max(np.abs(target - start)) / 0.005)) + 1)
        previous_z = None
        for q in np.linspace(start, target, count):
            if (
                np.max(np.abs(q[self.other] - self.start[self.other])) > 0.015
                or abs(q[self.indices][-1] - self.start[self.indices][-1]) > 0.02
            ):
                return {"ok": False, "reason": "retreat_changed_other_arm_or_gripper"}
            pose = g.eef_pose(self.arm, q)
            angle = np.arccos(
                np.clip(
                    (np.trace(self.rotation.T @ pose_to_matrix(pose)[:3, :3]) - 1) / 2,
                    -1,
                    1,
                )
            )
            if (
                np.linalg.norm(pose[:2] - self.pose[:2]) > 0.005
                or angle > 0.03
                or pose[2] < self.pose[2] - 0.001
                or pose[2] > self.pose[2] + self.distance + 0.005
                or (previous_z is not None and pose[2] < previous_z - 0.0005)
            ):
                return {"ok": False, "reason": "retreat_not_upward"}
            previous_z = pose[2]
            check = g._collision_guard.check(q, table_escape=self.escape)
            if not check["ok"]:
                return check
            if g._tcp_table_clearance(pose[:3]) < g.table_clearance_m:
                return {"ok": False, "reason": "table_guard"}
        return {"ok": True}

    def plan(self):
        g = self.geometry
        kin = g._kinematics_for(self.arm)
        previous = self.start.copy()
        path = []
        for height in np.linspace(0, self.distance, 41)[1:]:
            pose = self.pose.copy()
            pose[2] += height
            result = kin.solve(
                g.calibration.arm_target_from_world(self.arm, pose),
                previous[self.indices][:6],
                self.start[self.indices][-1],
            )
            if not result.success:
                raise ValueError("upward retreat IK failed")
            q = self.start.copy()
            q[self.indices][:6] = result.q_target
            check = self.check(previous, q)
            if not check["ok"]:
                raise ValueError("upward retreat rejected: " + str(check))
            path.append(q)
            previous = q
        if not g.check_qpos_transition(previous, previous)["ok"]:
            raise ValueError("retreat endpoint does not clear normal guards")
        return np.asarray(path)

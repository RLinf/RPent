# Copyright 2026 The RLinf Authors.
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

"""Static feasibility of grip-site paths in a LIBERO scene.

Runs inside the env worker on a copy of the simulator state, so the episode is
not touched. For every waypoint the Panda arm joints are solved by damped least
squares IK (warm-started from the previous waypoint), an object held between
the fingers is carried rigidly with the grip site, and contacts that penetrate
deeper than ``depth`` between the robot (or the held object) and the rest of
the scene are reported. Uses the scene's ground-truth collision geometry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

GRIP_SITE = "gripper0_right_grip_site"
ROBOT_PREFIXES = ("robot0_", "gripper0_")
#: Robot bodies whose contacts with the mount / table are part of the setup.
_BASE_BODIES = ("robot0_base", "robot0_link0", "mount0_")


def _family(name: str) -> str:
    """Object instance a body belongs to: ``microwave_1_microdoorroot`` -> ``microwave_1``."""
    parts = name.split("_")
    for i, part in enumerate(parts):
        if part.isdigit():
            return "_".join(parts[: i + 1])
    return name


class SceneChecker:
    """Feasibility queries against one MuJoCo model/state (copied per query)."""

    def __init__(self, model: Any, data: Any):
        import mujoco

        self.mj = mujoco
        self.model = model
        self.src = data
        self.data = mujoco.MjData(model)
        m = model
        self.body_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            for i in range(m.nbody)
        ]
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
        self.arm_joints = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}")
            for k in range(1, 8)
        ]
        self.arm_qpos = np.array([m.jnt_qposadr[j] for j in self.arm_joints])
        self.arm_dof = np.array([m.jnt_dofadr[j] for j in self.arm_joints])
        self.lo = m.jnt_range[self.arm_joints, 0]
        self.hi = m.jnt_range[self.arm_joints, 1]
        self.robot_body = np.array(
            [n.startswith(ROBOT_PREFIXES) for n in self.body_names]
        )
        self.base_body = np.array([n.startswith(_BASE_BODIES) for n in self.body_names])
        #: Bodies whose root floats freely (objects that a touch can push aside)
        #: as opposed to fixtures, which stop the arm.
        free = mujoco.mjtJoint.mjJNT_FREE
        self.movable_body = np.array(
            [
                m.body_jntadr[r] >= 0 and m.jnt_type[m.body_jntadr[r]] == free
                for r in m.body_rootid
            ]
        )

    # ---- state helpers ----

    def _reset_copy(self) -> None:
        d, s = self.data, self.src
        d.qpos[:] = s.qpos
        d.qvel[:] = 0.0
        d.mocap_pos[:] = s.mocap_pos
        d.mocap_quat[:] = s.mocap_quat
        self.mj.mj_kinematics(self.model, d)

    def _site_pose(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[self.site]
        return pose

    def _body_pose(self, b: int) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[b].reshape(3, 3)
        pose[:3, 3] = self.data.xpos[b]
        return pose

    def held_body(self) -> int | None:
        """Free-floating body touching both finger pads in the live state, if any."""
        m, d = self.model, self.src
        touching: dict[int, set[str]] = {}
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for a, b in ((b1, b2), (b2, b1)):
                name = self.body_names[a]
                if "finger" in name and not self.robot_body[b]:
                    touching.setdefault(int(m.body_rootid[b]), set()).add(name)
        for root, fingers in touching.items():
            jnt = m.body_jntadr[root]
            if (
                len(fingers) >= 2
                and jnt >= 0
                and m.jnt_type[jnt] == self.mj.mjtJoint.mjJNT_FREE
            ):
                return root
        return None

    # ---- IK ----

    def solve(self, target: np.ndarray, iters: int = 100) -> tuple[float, float]:
        """Damped least squares on the arm joints toward ``target``; returns the
        remaining position [m] and rotation [deg] error."""
        mj, m, d = self.mj, self.model, self.data
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        pos_err, rot_err = np.inf, np.inf
        for _ in range(iters):
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            cur = self._site_pose()
            dp = target[:3, 3] - cur[:3, 3]
            rel = target[:3, :3] @ cur[:3, :3].T
            dr = _rotvec(rel)
            pos_err, rot_err = (
                float(np.linalg.norm(dp)),
                float(np.degrees(np.linalg.norm(dr))),
            )
            if pos_err < 1e-3 and rot_err < 0.5:
                break
            mj.mj_jacSite(m, d, jacp, jacr, self.site)
            # Damping 1e-2 avoids the local minima that near-singular poses at
            # the edge of the workspace produce with lighter damping.
            jac = np.vstack([jacp[:, self.arm_dof], 0.5 * jacr[:, self.arm_dof]])
            err = np.concatenate([dp, 0.5 * dr])
            inv = jac.T @ np.linalg.inv(jac @ jac.T + 1e-2 * np.eye(6))
            dq = inv @ err
            # Null-space pull toward the starting posture, as the OSC
            # controller does, so the walk does not drift into far branches.
            dq += (np.eye(7) - inv @ jac) @ (0.1 * (self.q0 - d.qpos[self.arm_qpos]))
            q = d.qpos[self.arm_qpos] + np.clip(dq, -0.2, 0.2)
            d.qpos[self.arm_qpos] = np.clip(q, self.lo, self.hi)
        return pos_err, rot_err

    # ---- collisions ----

    def collisions(
        self, held: int | None, depth: float, ignore: set[int]
    ) -> list[tuple[str, str, float, bool]]:
        mj, m, d = self.mj, self.model, self.data
        mj.mj_forward(m, d)
        hits = []
        for k in range(d.ncon):
            c = d.contact[k]
            if c.dist > -depth:
                continue
            r1, r2 = (
                int(m.body_rootid[m.geom_bodyid[c.geom1]]),
                int(m.body_rootid[m.geom_bodyid[c.geom2]]),
            )
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            mover1 = self.robot_body[b1] or (held is not None and r1 == held)
            mover2 = self.robot_body[b2] or (held is not None and r2 == held)
            if mover1 == mover2:
                continue  # robot-robot, robot-held, or scene-scene
            mover, other = (b1, b2) if mover1 else (b2, b1)
            if (
                self.base_body[mover]
                or other in ignore
                or int(m.body_rootid[other]) in ignore
            ):
                continue
            hits.append(
                (
                    self.body_names[mover],
                    self.body_names[other],
                    float(-c.dist),
                    bool(self.movable_body[other]),
                )
            )
        return hits

    def check(
        self,
        path: list[np.ndarray],
        depth: float = 0.002,
        ik_pos_tol: float = 0.03,
        ik_rot_tol: float = 8.0,
        ignore_families: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Walk ``path`` (4x4 grip-site poses) from the live state."""
        self._reset_copy()
        self.q0 = self.data.qpos[self.arm_qpos].copy()
        held = self.held_body()
        held_offset = None
        if held is not None:
            held_offset = np.linalg.inv(self._site_pose()) @ self._body_pose(held)
        fams = set(ignore_families)
        ignore = {
            i for i, n in enumerate(self.body_names) if fams and _family(n) in fams
        }
        # Contacts present before moving (the held object on the table, a
        # finger already against a wall) count only once they press 5 mm
        # deeper than at the start.
        pre: dict[tuple[str, str], float] = {}
        for h in self.collisions(held, depth, ignore):
            pre[h[:2]] = max(pre.get(h[:2], 0.0), h[2])
        first_collision, first_ik_fail, bodies = None, None, set()
        first_movable, movable_pairs = None, set()
        goal_pos_err, goal_rot_err = np.inf, np.inf
        worst_pos, min_margin, deepest = 0.0, np.inf, 0.0
        for i, target in enumerate(path):
            pos_err, rot_err = self.solve(np.asarray(target))
            worst_pos = max(worst_pos, pos_err)
            q = self.data.qpos[self.arm_qpos]
            min_margin = min(
                min_margin,
                float(np.degrees(np.min(np.minimum(q - self.lo, self.hi - q)))),
            )
            if first_ik_fail is None and (pos_err > ik_pos_tol or rot_err > ik_rot_tol):
                first_ik_fail = i
            goal_pos_err, goal_rot_err = pos_err, rot_err
            if held is not None:
                pose = self._site_pose() @ held_offset
                adr = self.model.jnt_qposadr[self.model.body_jntadr[held]]
                self.data.qpos[adr : adr + 3] = pose[:3, 3]
                self.data.qpos[adr + 3 : adr + 7] = _quat_wxyz(pose[:3, :3])
            hits = [
                h
                for h in self.collisions(held, depth, ignore)
                if h[2] > pre.get(h[:2], -1.0) + 0.005
            ]
            fixed = [h for h in hits if not h[3]]
            if fixed:
                bodies.update(f"{h[0]}|{h[1]}" for h in fixed)
                deepest = max(deepest, max(h[2] for h in fixed))
                if first_collision is None:
                    first_collision = i
            if len(fixed) < len(hits):
                movable_pairs.update(f"{h[0]}|{h[1]}" for h in hits if h[3])
                if first_movable is None:
                    first_movable = i
        return {
            "n": len(path),
            "held": self.body_names[held] if held is not None else None,
            "first_collision": first_collision,
            "first_ik_fail": first_ik_fail,
            "collision_pairs": sorted(bodies)[:8],
            # Touches of free objects: they are pushed rather than blocking,
            # but can knock an object over.
            "first_movable_contact": first_movable,
            "movable_pairs": sorted(movable_pairs)[:8],
            "max_ik_pos_err_cm": round(100 * worst_pos, 2),
            "max_penetration_cm": round(100 * deepest, 2),
            "min_joint_margin_deg": round(min_margin, 1),
            "goal_ik_pos_err_cm": round(100 * goal_pos_err, 2),
            "goal_ik_rot_err_deg": round(goal_rot_err, 1),
            # Unreachable intermediate waypoints are not disqualifying: the
            # controller cuts the corner as long as nothing is hit and the
            # final pose can be reached.
            "feasible": first_collision is None
            and goal_pos_err <= ik_pos_tol
            and goal_rot_err <= ik_rot_tol,
        }


def _rotvec(rot: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rot).as_rotvec()


def _quat_wxyz(rot: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(rot).as_quat()
    return np.array([w, x, y, z])


def check_paths(sim: Any, paths: list, **kwargs: Any) -> list[dict[str, Any]]:
    """Check each path in ``paths`` against ``sim`` (robosuite MjSim)."""
    checker = SceneChecker(sim.model._model, sim.data._data)
    return [checker.check([np.asarray(p) for p in path], **kwargs) for path in paths]

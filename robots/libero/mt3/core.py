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
"""MT3 retrieval, alignment and replay, independent of how the env is reached.

A task library (one ``.npz`` per task, written by an offline builder from the
official demonstrations) holds, per demonstration frame, the grip-site pose
``eef``, the commanded target pose ``target``, the gripper command and the raw
action, plus each demonstration's segments. A segment is one interaction with
one reference body: ``c`` is its first contact frame (gripper change or touch),
``end`` the frame where its interaction ends, ``start`` the earliest frame its
bottleneck may sit at. Arrays:

- ``offsets`` (N+1,): demo ``d`` spans rows ``offsets[d]:offsets[d+1]``.
- ``eef``, ``target`` (T, 4, 4); ``gripper`` (T,); ``actions`` (T, 7).
- ``pose/<body>`` (T, 4, 4): world pose of each reference body.
- ``keep`` (N,): the demo's segment sequence is the task's usual one.
- ``segments``: JSON, per demo a list of ``{"ref", "start", "c", "end",
  "events"}`` with demo-local frames.
- ``ref_bodies``, ``meta``: JSON.

An episode retrieves the kept demo whose reference bodies sit closest to the
live ones, then per segment moves the arm along a lift/transit/descend path to
the demo's bottleneck pose carried by the reference body's 4-DOF (x, y, z, yaw)
pose change, and replays the demo from there (Dreczkowski et al., "Learning a
Thousand Tasks in a Day", 2025). ``mt3`` tracks the carried target poses
closed loop; ``mt3_open`` applies the demo's per-step relative commands to the
current pose (MT3's open-loop twist replay); ``replay`` sends the retrieved
demo's raw actions unchanged. ``continuous`` retains all source frames and
smoothly blends object-relative target poses during each approach, with no
separate alignment or settling steps. It still uses the interaction events
as reference anchors; object poses are read only at the start of each segment.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

MODES = ("mt3", "mt3_open", "continuous", "replay")
#: OSC_POSE delta scaling of LIBERO's Panda: 1.0 = 5 cm / 0.5 rad per step.
POS_SCALE = 0.05
ROT_SCALE = 0.5
GRIP_SITE = "gripper0_right_grip_site"


@dataclass
class MT3Config:
    """Execution settings; read from ``mt3.json`` beside the library."""

    mode: str = "mt3"
    #: Bottleneck distance before the first contact [m].
    radius: float = 0.10
    #: Alignment path spacing [m per step].
    align_step: float = 0.01
    settle_pos_tol: float = 0.003
    settle_rot_tol_deg: float = 2.0
    settle_max: int = 50
    #: Steps the last target is held after the demo ends, waiting for success.
    hold_steps: int = 30
    #: Extra high-clearance alignment attempts after the ordinary route stalls.
    #: Length [m] of the demo's own approach path, ending at the bottleneck,
    #: that is replayed (transferred) after aligning to its entry; 0 aligns
    #: straight to the bottleneck.
    approach_corridor: float = 0.0
    #: Extra transit height [m] while the gripper holds an object.
    carry_clearance: float = 0.0
    #: "log": before each alignment, check candidate routes (top-k demos x
    #: corridor x transit height) for reach and collisions and log the result;
    #: execution is unchanged. "" disables the check.
    feasibility_check: str = ""
    feasibility_topk: int = 5
    alignment_replans: int = 0
    alignment_guard_pos: float = 0.0
    alignment_guard_rot_deg: float = 15.0
    #: Repeat a source target when its tracking residual exceeds these limits.
    track_extra_steps: int = 0
    track_pos_tol: float = 0.015
    track_rot_tol_deg: float = 5.0
    #: Compensate measured in-hand pose differences before a placement segment.
    grasp_compensation: bool = False
    #: Rotational symmetry about z per body-name substring: 0 ignores the yaw
    #: change (continuous symmetry), P folds it into +-P/2 degrees. Bodies that
    #: match nothing use their full yaw change.
    yaw_symmetry: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "MT3Config":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        if cfg.mode not in MODES:
            raise ValueError(f"unknown MT3 mode {cfg.mode!r}; expected one of {MODES}")
        return cfg

    def symmetry(self, body: str) -> float | None:
        for key, period in self.yaw_symmetry.items():
            if key in body:
                return float(period)
        return None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def make_pose(pos, rot) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    return pose


def pose_from_wxyz(values) -> np.ndarray:
    """4x4 pose from ``[x, y, z, qw, qx, qy, qz]`` (MuJoCo quaternion order)."""
    x, y, z, qw, qx, qy, qz = values
    return make_pose([x, y, z], Rotation.from_quat([qx, qy, qz, qw]).as_matrix())


def rot_angle_deg(rot: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1, 1))))


def rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_change(live: np.ndarray, demo: np.ndarray, period: float | None) -> float:
    """Rotation about world z [rad] taking ``demo`` to ``live``, symmetry-folded."""
    rel = live[:3, :3] @ demo[:3, :3].T
    angle = float(np.arctan2(rel[1, 0], rel[0, 0]))
    if period is None:
        return angle
    if period <= 0:
        return 0.0
    p = np.radians(period)
    return angle - np.round(angle / p) * p


def transfer_4dof(live: np.ndarray, demo: np.ndarray, period: float | None):
    """World transform moving the demo's reference body onto the live one in
    x, y, z and yaw only. Returns (transform, yaw change [rad])."""
    dyaw = yaw_change(live, demo, period)
    rot = rz(dyaw)
    return make_pose(live[:3, 3] - rot @ demo[:3, 3], rot), dyaw


def slerp(r0: np.ndarray, r1: np.ndarray, tau: float) -> np.ndarray:
    return (
        r0
        @ Rotation.from_rotvec(
            tau * Rotation.from_matrix(r0.T @ r1).as_rotvec()
        ).as_matrix()
    )


def alignment_path(
    start: np.ndarray, goal: np.ndarray, step: float, clearance: float = 0.0
) -> list[np.ndarray]:
    """Lift / transit / descend: rise (or stay) to the higher of the two heights,
    move horizontally, then go down to ``goal``, with orientation slerped along
    the arc length. Returns the poses after ``start``."""
    p0, p3 = start[:3, 3], goal[:3, 3]
    zh = max(p0[2], p3[2]) + clearance
    pts = [p0, np.array([p0[0], p0[1], zh]), np.array([p3[0], p3[1], zh]), p3]
    legs = [(a, b) for a, b in zip(pts[:-1], pts[1:]) if np.linalg.norm(b - a) > 1e-9]
    lengths = [float(np.linalg.norm(b - a)) for a, b in legs]
    total = sum(lengths)
    n = max(1, int(np.ceil(total / step)))
    poses = []
    for j in range(1, n + 1):
        s = total * j / n
        pos, acc = p3, 0.0
        for (a, b), length in zip(legs, lengths):
            if s <= acc + length + 1e-12:
                pos = a + (b - a) * ((s - acc) / length)
                break
            acc += length
        tau = s / total if total > 0 else 1.0
        poses.append(make_pose(pos, slerp(start[:3, :3], goal[:3, :3], tau)))
    return poses


def bottleneck_frame(eef: np.ndarray, start: int, c: int, radius: float) -> int:
    """Last frame in ``[start, c]`` at least ``radius`` from the eef at ``c``;
    ``start`` when none is."""
    d = np.linalg.norm(eef[start : c + 1, :3, 3] - eef[c, :3, 3], axis=-1)
    far = np.nonzero(d >= radius)[0]
    return start + int(far[-1]) if len(far) else start


def target_to_action(eef: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
    """OSC_POSE delta action driving the grip site from ``eef`` toward ``target``.

    LIBERO's Panda base is unrotated, so the controller's base-frame deltas are
    world-frame deltas, applied to the current (achieved) pose.
    """
    action = np.zeros(7)
    action[:3] = np.clip((target[:3, 3] - eef[:3, 3]) / POS_SCALE, -1.0, 1.0)
    rotvec = Rotation.from_matrix(target[:3, :3] @ eef[:3, :3].T).as_rotvec()
    action[3:6] = np.clip(rotvec / ROT_SCALE, -1.0, 1.0)
    action[6] = gripper
    return action


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


class TaskLibrary:
    """One task's segmented demonstrations (see the module docstring)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        with np.load(self.path) as data:
            self.offsets = data["offsets"]
            self.eef = data["eef"]
            self.target = data["target"]
            self.gripper = data["gripper"]
            self.actions = data["actions"]
            self.keep = data["keep"]
            self.segments = json.loads(str(data["segments"]))
            self.ref_bodies = json.loads(str(data["ref_bodies"]))
            self.meta = json.loads(str(data["meta"]))
            self.pose = {
                k.split("/", 1)[1]: data[k] for k in data.files if k.startswith("pose/")
            }

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def frames(self, demo: int) -> slice:
        return slice(int(self.offsets[demo]), int(self.offsets[demo + 1]))

    def ranked(self, scene: dict[str, np.ndarray], cfg: MT3Config) -> list[int]:
        """Kept demos ordered by the :meth:`nearest` distance to ``scene``."""
        dist = {}
        for d in np.nonzero(self.keep)[0]:
            first = int(self.offsets[d])
            dist[int(d)] = sum(
                float(np.linalg.norm(scene[b][:3, 3] - self.pose[b][first][:3, 3]))
                + abs(yaw_change(scene[b], self.pose[b][first], cfg.symmetry(b)))
                for b in self.ref_bodies
            )
        return sorted(dist, key=dist.get)

    def nearest(
        self, scene: dict[str, np.ndarray], cfg: MT3Config
    ) -> tuple[int, dict[str, Any]]:
        """Kept demo whose reference bodies, at its first frame, are closest to
        ``scene``: summed position distance [m] + symmetry-folded yaw [rad]."""
        best, best_dist, dists = -1, np.inf, []
        for d in np.nonzero(self.keep)[0]:
            first = int(self.offsets[d])
            dist = 0.0
            for body in self.ref_bodies:
                demo_pose = self.pose[body][first]
                dist += float(np.linalg.norm(scene[body][:3, 3] - demo_pose[:3, 3]))
                dist += abs(yaw_change(scene[body], demo_pose, cfg.symmetry(body)))
            dists.append(dist)
            if dist < best_dist:
                best, best_dist = int(d), dist
        if best < 0:
            raise RuntimeError(f"{self.path} keeps no demonstration")
        return best, {
            "demo": best,
            "dist": round(best_dist, 4),
            "dist_median": round(float(np.median(dists)), 4),
            "candidates": len(dists),
        }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class EnvHooks:
    """How the runner reaches the environment."""

    #: Send one 7-D action.
    step: Callable[[np.ndarray], None]
    #: Current grip-site pose (4x4).
    eef_pose: Callable[[], np.ndarray]
    #: World poses (4x4) of the named bodies.
    body_poses: Callable[[list[str]], dict[str, np.ndarray]]
    #: Whether the episode has ended (success or truncation).
    done: Callable[[], bool]
    #: Static feasibility of grip-site paths on a copy of the live scene
    #: (``robots.libero.mt3.feasibility``); None when unavailable.
    check_paths: Callable[[list[list[np.ndarray]]], list[dict[str, Any]]] | None = None


class MT3Runner:
    """Replays one retrieved demonstration, segment by segment."""

    def __init__(self, lib: TaskLibrary, demo: int, cfg: MT3Config, env: EnvHooks):
        self.lib = lib
        self.demo = demo
        self.cfg = cfg
        self.env = env
        rows = lib.frames(demo)
        self.first = rows.start
        self.eef = lib.eef[rows]
        self.target = lib.target[rows]
        self.gripper = lib.gripper[rows]
        self.actions = lib.actions[rows]
        self.segments = lib.segments[demo]
        #: One diagnostics dict per executed call, in order.
        self.logs: list[dict[str, Any]] = []
        self.steps = 0
        self._last_target: np.ndarray | None = None
        self._last_gripper = -1.0
        self._previous_transform = np.eye(4)

    def _send(self, target: np.ndarray, gripper: float) -> bool:
        """Track ``target`` for one step; False once the episode has ended."""
        if self.env.done():
            return False
        self.env.step(target_to_action(self.env.eef_pose(), target, gripper))
        self.steps += 1
        self._last_target, self._last_gripper = target, float(gripper)
        return not self.env.done()

    def run_segment(self, k: int) -> dict[str, Any]:
        seg = self.segments[k]
        ref, start, c, end = seg["ref"], seg["start"], seg["c"], seg["end"]
        radius = self.cfg.radius
        b = bottleneck_frame(self.eef, start, c, radius)
        live = self.env.body_poses([ref])[ref]
        demo_ref = self.lib.pose[ref][self.first + b]
        transform, dyaw = transfer_4dof(live, demo_ref, self.cfg.symmetry(ref))
        log: dict[str, Any] = {
            "segment": k,
            "ref": ref,
            "frames": [start, b, c, end],
            "bottleneck_dist_cm": round(
                100 * float(np.linalg.norm(self.eef[b, :3, 3] - self.eef[c, :3, 3])), 2
            ),
            "ref_shift_cm": round(
                100 * float(np.linalg.norm(live[:2, 3] - demo_ref[:2, 3])), 2
            ),
            "ref_yaw_change_deg": round(float(np.degrees(dyaw)), 1),
        }
        n0 = self.steps

        if self.cfg.mode == "continuous":
            return self._run_continuous(seg, log, transform, b, n0)

        goal = transform @ self.eef[b]
        grasp_delta = np.eye(4)
        if self.cfg.grasp_compensation and k > 0 and self.gripper[b] > 0:
            previous = self.segments[k - 1]
            held = previous["ref"]
            placing = any(e[0] in {"release", "end"} for e in seg["events"])
            grasped = any(e[0] == "grasp" for e in previous["events"])
            if placing and grasped and held in self.lib.pose:
                live_grasp = (
                    np.linalg.inv(self.env.eef_pose())
                    @ self.env.body_poses([held])[held]
                )
                demo_grasp = (
                    np.linalg.inv(self.eef[b]) @ self.lib.pose[held][self.first + b]
                )
                if np.linalg.norm(live_grasp[:3, 3]) < 0.15:
                    grasp_delta = demo_grasp @ np.linalg.inv(live_grasp)
                    log["grasp_correction_cm"] = round(
                        100 * float(np.linalg.norm(grasp_delta[:3, 3])), 3
                    )
                    log["grasp_correction_deg"] = round(
                        rot_angle_deg(grasp_delta[:3, :3]), 2
                    )
                    goal = goal @ grasp_delta
        hold = self.gripper[b]
        if self.cfg.feasibility_check and self.env.check_paths is not None:
            log["feasibility"] = self._feasibility(k, live, hold)
        corridor: list[np.ndarray] = []
        if self.cfg.approach_corridor > 0:
            entry = bottleneck_frame(self.eef, start, b, self.cfg.approach_corridor)
            corridor = [transform @ self.eef[f] @ grasp_delta for f in range(entry, b)]
            log["corridor_frames"] = [entry, b]
        attempts = []
        align_steps, total_settle = 0, 0
        for attempt in range(self.cfg.alignment_replans + 1):
            clearance = 0.06 * attempt + (self.cfg.carry_clearance if hold > 0 else 0.0)
            if corridor:
                path = alignment_path(
                    self.env.eef_pose(), corridor[0], self.cfg.align_step, clearance
                ) + corridor[1:] + [goal]
            else:
                path = alignment_path(
                    self.env.eef_pose(), goal, self.cfg.align_step, clearance
                )
            for pose in path:
                if not self._send(pose, hold):
                    return self._close(log, n0, "done during alignment")
            align_steps += len(path)
            settle, pos_err, rot_err = 0, np.inf, np.inf
            while True:
                eef = self.env.eef_pose()
                pos_err = float(np.linalg.norm(eef[:3, 3] - goal[:3, 3]))
                rot_err = rot_angle_deg(eef[:3, :3].T @ goal[:3, :3])
                if (
                    pos_err <= self.cfg.settle_pos_tol
                    and rot_err <= self.cfg.settle_rot_tol_deg
                ) or settle >= self.cfg.settle_max:
                    break
                if not self._send(goal, hold):
                    return self._close(log, n0, "done while settling")
                settle += 1
            total_settle += settle
            attempts.append(
                {
                    "clearance": clearance,
                    "pos_error_cm": round(100 * pos_err, 3),
                    "rot_error_deg": round(rot_err, 2),
                }
            )
            if not self.cfg.alignment_guard_pos or (
                pos_err <= self.cfg.alignment_guard_pos
                and rot_err <= self.cfg.alignment_guard_rot_deg
            ):
                break
        after = self.env.body_poses([ref])[ref]
        log.update(
            align_steps=align_steps,
            settle_steps=total_settle,
            alignment_attempts=attempts,
            settle_pos_err_cm=round(100 * pos_err, 3),
            settle_rot_err_deg=round(rot_err, 2),
            ref_moved_during_align_cm=round(
                100 * float(np.linalg.norm(after[:3, 3] - live[:3, 3])), 2
            ),
        )
        if self.cfg.alignment_guard_pos and (
            pos_err > self.cfg.alignment_guard_pos
            or rot_err > self.cfg.alignment_guard_rot_deg
        ):
            log["blocked"] = True
            return self._close(
                log, n0, "alignment blocked; source interaction not executed"
            )

        for f in range(b, end):
            if self.cfg.mode == "mt3":
                target = transform @ self.target[f] @ grasp_delta
            else:  # mt3_open: the demo's command relative to its own pose
                target = (
                    self.env.eef_pose() @ np.linalg.inv(self.eef[f]) @ self.target[f]
                )
            if f == c:
                ref_pos = (transform @ self.eef[c] @ grasp_delta)[:3, 3]
                log["contact_err_cm"] = round(
                    100 * float(np.linalg.norm(self.env.eef_pose()[:3, 3] - ref_pos)), 2
                )
            if not self._send(target, self.gripper[f]):
                return self._close(log, n0, "done during interaction")
            if self.cfg.mode == "mt3":
                for _ in range(self.cfg.track_extra_steps):
                    achieved = self.env.eef_pose()
                    if (
                        np.linalg.norm(achieved[:3, 3] - target[:3, 3])
                        <= self.cfg.track_pos_tol
                        and rot_angle_deg(achieved[:3, :3].T @ target[:3, :3])
                        <= self.cfg.track_rot_tol_deg
                    ):
                        break
                    log["tracking_extra_steps"] = log.get("tracking_extra_steps", 0) + 1
                    if not self._send(target, self.gripper[f]):
                        return self._close(log, n0, "done during tracking correction")
        return self._close(log, n0, "segment replayed")

    def _feasibility(self, k: int, live: np.ndarray, hold: float) -> list[dict[str, Any]]:
        """Candidate alignment routes for segment ``k`` and their static feasibility.

        Candidates: the ``feasibility_topk`` kept demos nearest the live scene
        (this runner's demo first) x approach corridor {0, 0.1, 0.2} m x extra
        transit height {0, 0.06, 0.12, 0.2} m. Each route ends at that demo's
        transferred bottleneck. Nothing is executed.
        """
        lib, seg = self.lib, self.segments[k]
        ref = seg["ref"]
        scene = self.env.body_poses(lib.ref_bodies)
        ranked = [self.demo] + [
            d for d in lib.ranked(scene, self.cfg) if d != self.demo
        ][: max(0, self.cfg.feasibility_topk - 1)]
        start = self.env.eef_pose()
        names, paths = [], []
        for d in ranked:
            other = lib.segments[d][k] if k < len(lib.segments[d]) else None
            if other is None or other["ref"] != ref:
                continue
            rows = lib.frames(d)
            eef = lib.eef[rows]
            b = bottleneck_frame(eef, other["start"], other["c"], self.cfg.radius)
            transform, _ = transfer_4dof(
                live, lib.pose[ref][rows.start + b], self.cfg.symmetry(ref)
            )
            goal = transform @ eef[b]
            for length in (0.0, 0.1, 0.2):
                entry = bottleneck_frame(eef, other["start"], b, length) if length else b
                lead = [transform @ eef[f] for f in range(entry, b)]
                for clearance in (0.0, 0.06, 0.12, 0.2):
                    head = lead[0] if lead else goal
                    path = alignment_path(start, head, self.cfg.align_step, clearance)
                    paths.append(path + lead[1:] + ([goal] if lead else []))
                    names.append({"demo": d, "corridor": length, "clearance": clearance})
        results = self.env.check_paths(paths) if paths else []
        return [dict(n, **r) for n, r in zip(names, results)]

    def _run_continuous(
        self,
        seg: dict[str, Any],
        log: dict[str, Any],
        transform: np.ndarray,
        bottleneck: int,
        n0: int,
    ) -> dict[str, Any]:
        """Keep every source frame, blending reference frames before contact.

        At a segment boundary, the reference body's live pose is measured once.
        During the approach, smoothstep blends the previous and new transforms
        of each source target. From the bottleneck through contact, the new
        transform is fixed. There is no separate alignment path or settling.
        """
        start = seg["start"]
        log.update(align_steps=0, settle_steps=0, source_frames=seg["end"] - start)
        for frame in range(start, seg["end"]):
            tau = min(1.0, (frame - start) / max(1, bottleneck - start))
            if bottleneck == start:
                tau = 1.0
            weight = tau * tau * (3.0 - 2.0 * tau)
            old = self._previous_transform @ self.target[frame]
            new = transform @ self.target[frame]
            target = make_pose(
                (1.0 - weight) * old[:3, 3] + weight * new[:3, 3],
                slerp(old[:3, :3], new[:3, :3], weight),
            )
            if frame == seg["c"]:
                contact = (transform @ self.eef[frame])[:3, 3]
                log["contact_err_cm"] = round(
                    100 * float(np.linalg.norm(self.env.eef_pose()[:3, 3] - contact)), 2
                )
            if not self._send(target, self.gripper[frame]):
                return self._close(log, n0, "done during continuous replay")
        self._previous_transform = transform
        return self._close(log, n0, "source frames replayed")

    def _close(self, log: dict[str, Any], n0: int, status: str) -> dict[str, Any]:
        log.update(steps=self.steps - n0, status=status, done=self.env.done())
        self.logs.append(log)
        return log

    def hold(self) -> dict[str, Any]:
        """Keep the last command for ``hold_steps`` (objects settling into place)."""
        n0 = self.steps
        if self._last_target is not None:
            for _ in range(self.cfg.hold_steps):
                if not self._send(self._last_target, self._last_gripper):
                    break
        return self._close({"call": "hold"}, n0, "held")

    def replay_actions(self) -> dict[str, Any]:
        """``replay`` mode: the demo's raw actions, no alignment."""
        n0 = self.steps
        for action in self.actions:
            if self.env.done():
                break
            self.env.step(np.asarray(action, dtype=np.float64))
            self.steps += 1
        return self._close(
            {"call": "replay", "demo_length": len(self.actions)}, n0, "replayed"
        )

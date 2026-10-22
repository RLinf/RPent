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
  "events"}`` with demo-local frames, plus an optional bottleneck ``b``.
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
from dataclasses import dataclass, field, replace
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
    #: execution is unchanged. "select": execute the segment with the shortest
    #: feasible candidate (its demo, corridor and transit height), or the
    #: configured route when none is feasible. "" disables the check.
    feasibility_check: str = ""
    #: Use joint-space RRT-Connect when all static alignment routes fail.
    rrt_fallback: bool = False
    feasibility_topk: int = 5
    #: Extra rotations [deg] about the vertical axis tried by "minimal" when
    #: this demo's routes are infeasible: placements (object held) turn about
    #: where the held object is put down, since placement goals ignore its yaw;
    #: other segments only for references matching ``yaw_free``.
    yaw_search_deg: list[float] = field(default_factory=list)
    #: Escape offsets [m, world frame] tried by "minimal" before the transit
    #: when this demo's routes and yaw variants are blocked: the arm first
    #: moves by the offset from where it is (e.g. up, or back toward the base).
    via_offsets: list[list[float]] = field(default_factory=list)
    #: Body-name substrings whose grasp may be turned about their own axis.
    yaw_free: list[str] = field(default_factory=list)
    #: Point [m, body frame] that grasp yaw variants of a ``yaw_free``
    #: reference turn about, per body-name substring; the body origin when
    #: nothing matches. A handled mug's origin is off its cup axis, so turning
    #: about the origin moves the grasp off the rim.
    yaw_pivot: dict[str, list[float]] = field(default_factory=dict)
    #: Tilts [deg] of a grasp about the fingers' closing axis through the
    #: contact pose, tried by "minimal" (before yaw turns) when the grasp
    #: route is infeasible. A pinch keeps its contact faces under this turn
    #: while the wrist swings to the other side.
    grasp_tilt_deg: list[float] = field(default_factory=list)
    #: Grip-site axis the fingers close along (LIBERO Panda: x).
    grasp_tilt_axis: int = 0
    #: Shift [m, reference body frame] of every segment transferred onto a
    #: reference whose name contains the key, e.g. to move a demo's marginal
    #: release toward the middle of a container.
    ref_offset: dict[str, list[float]] = field(default_factory=dict)
    #: Also check the source interaction from the bottleneck to contact,
    #: ignoring the segment's reference object.
    feasibility_interaction: bool = False
    #: Extra alignment transit height [m], always applied.
    transit_clearance: float = 0.0
    #: Alignment route: "path" (lift / transit / descend, plus the approach
    #: corridor) or "dmp" (a DMP fitted to the demo's own approach from the
    #: segment start to the bottleneck, transferred, replayed from the live pose;
    #: as in DynaMimicGen).
    approach: str = "path"
    #: Save the executed episode (state before each step, the action) as
    #: ``mt3_trace.npz`` for building a library from successful runs.
    record_trace: bool = False
    #: Retries of a grasp segment whose object was not picked up.
    grasp_retries: int = 0
    #: Demo retrieval: "nearest" (reference-body distance) or "reach" (most
    #: segments whose transferred bottleneck the arm can reach, then nearest).
    retrieval: str = "nearest"
    alignment_replans: int = 0
    #: Additional alignment attempts for references matching a body-name
    #: substring. Absent entries preserve the global setting.
    alignment_replans_by_ref: dict[str, int] = field(default_factory=dict)
    #: Continue a transferred segment's final target direction by this distance
    #: [m], for references matching a body-name substring. Empty by default.
    segment_overshoot_by_ref: dict[str, float] = field(default_factory=dict)
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

    def transfer(
        self, ref: str, live: np.ndarray, demo: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """``transfer_4dof`` of reference ``ref``, plus its ``ref_offset``."""
        transform, dyaw = transfer_4dof(live, demo, self.symmetry(ref))
        for key, offset in self.ref_offset.items():
            if key in ref:
                shift = np.eye(4)
                shift[:3, 3] = live[:3, :3] @ np.asarray(offset, dtype=float)
                return shift @ transform, dyaw
        return transform, dyaw

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


def yaw_about(transform: np.ndarray, angle: float, pivot: np.ndarray) -> np.ndarray:
    """``transform`` followed by a rotation of ``angle`` [rad] about the
    vertical axis through ``pivot``."""
    rot = rz(angle)
    turn = make_pose(pivot - rot @ pivot, rot)
    return turn @ transform


def tilt_about(
    transform: np.ndarray, angle: float, grip: np.ndarray, axis: int
) -> np.ndarray:
    """``transform`` followed by a rotation of ``angle`` [rad] about grip-site
    axis ``axis`` of the transferred contact pose ``transform @ grip``."""
    contact = transform @ grip
    rot = Rotation.from_rotvec(angle * contact[:3, axis]).as_matrix()
    turn = make_pose(contact[:3, 3] - rot @ contact[:3, 3], rot)
    return turn @ transform


def via_path(
    start: np.ndarray, head: np.ndarray, offset, step: float, clearance: float
) -> list[np.ndarray]:
    """Move by ``offset`` from ``start`` first, then the usual lift / transit /
    descend to ``head``. ``offset`` None is the plain alignment path."""
    if offset is None:
        return alignment_path(start, head, step, clearance)
    via = start.copy()
    via[:3, 3] = start[:3, 3] + np.asarray(offset, dtype=float)
    n = max(1, int(np.ceil(np.linalg.norm(offset) / step)))
    lead = [
        make_pose(start[:3, 3] + (via[:3, 3] - start[:3, 3]) * j / n, start[:3, :3])
        for j in range(1, n + 1)
    ]
    return lead + alignment_path(via, head, step, clearance)


def _family(name: str) -> str:
    """Object instance a body belongs to: ``microwave_1_microdoorroot`` -> ``microwave_1``."""
    parts = name.split("_")
    for i, part in enumerate(parts):
        if part.isdigit():
            return "_".join(parts[: i + 1])
    return name


def dmp_path(
    start: np.ndarray, demo: list[np.ndarray], step: float, n_basis: int = 30
) -> list[np.ndarray]:
    """Demo-shaped route from ``start`` to ``demo[-1]``.

    Positions follow a discrete DMP (Hoffmann et al. 2009 form, which does not
    flip when the start-goal vector changes) fitted to ``demo`` and integrated
    from ``start``; orientation is the demo's, with the initial offset to
    ``start`` decayed along the canonical phase. Waypoints are resampled to at
    most ``step`` apart. Returns the poses after ``start``.
    """
    y = np.array([p[:3, 3] for p in demo])
    n = len(y)
    if n < 3:
        return alignment_path(start, demo[-1], step)
    t = np.linspace(0.0, 1.0, n)
    dt = t[1] - t[0]
    yd = np.gradient(y, dt, axis=0)
    ydd = np.gradient(yd, dt, axis=0)
    k_gain, d_gain, alpha_x = 100.0, 20.0, 4.0
    x = np.exp(-alpha_x * t)
    g, y0 = y[-1], y[0]
    # Forcing term the demo implies, regressed on Gaussian bases in the phase.
    f_target = (ydd - k_gain * (g - y) + d_gain * yd) / k_gain + (g - y0) * x[:, None]
    centers = np.exp(-alpha_x * np.linspace(0.0, 1.0, n_basis))
    widths = n_basis**1.5 / centers / alpha_x
    psi = np.exp(-widths * (x[:, None] - centers) ** 2)
    feat = psi * x[:, None] / (psi.sum(axis=1, keepdims=True) + 1e-10)
    weights = np.linalg.lstsq(feat, f_target, rcond=None)[0]
    # Integrate from the live start (zero velocity) toward the same goal.
    p, v = start[:3, 3].copy(), np.zeros(3)
    p_start = p.copy()
    out = []
    for i in range(n):
        f = feat[i] @ weights
        acc = k_gain * (g - p) - d_gain * v - k_gain * (g - p_start) * x[i] + k_gain * f
        v = v + acc * dt
        p = p + v * dt
        out.append(p.copy())
    out[-1] = g.copy()
    offset = start[:3, :3] @ demo[0][:3, :3].T
    poses = []
    for i, pos in enumerate(out):
        decay = Rotation.from_matrix(offset).as_rotvec() * x[i] / x[0]
        rot = Rotation.from_rotvec(decay).as_matrix() @ demo[i][:3, :3]
        poses.append(make_pose(pos, rot))
    # Resample so consecutive waypoints are at most ``step`` apart.
    dense, prev = [], start
    for pose in poses:
        gap = float(np.linalg.norm(pose[:3, 3] - prev[:3, 3]))
        m = max(1, int(np.ceil(gap / step)))
        for j in range(1, m + 1):
            dense.append(
                make_pose(
                    prev[:3, 3] + (pose[:3, 3] - prev[:3, 3]) * j / m,
                    slerp(prev[:3, :3], pose[:3, :3], j / m),
                )
            )
        prev = pose
    return dense


def bottleneck_frame(eef: np.ndarray, start: int, c: int, radius: float) -> int:
    """Last frame in ``[start, c]`` at least ``radius`` from the eef at ``c``;
    ``start`` when none is."""
    d = np.linalg.norm(eef[start : c + 1, :3, 3] - eef[c, :3, 3], axis=-1)
    far = np.nonzero(d >= radius)[0]
    return start + int(far[-1]) if len(far) else start


def segment_bottleneck(eef: np.ndarray, seg: dict[str, Any], radius: float) -> int:
    """A segment's bottleneck frame: its stored ``b`` when the library gives one
    (MT3-generated demos, where it is the frame the executed interaction
    started from rest), else :func:`bottleneck_frame`."""
    if seg.get("b") is not None:
        return int(seg["b"])
    return bottleneck_frame(eef, seg["start"], seg["c"], radius)


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
    check_paths: Callable[..., list[dict[str, Any]]] | None = None
    #: Plan one grip-site pose on a copied simulator state.
    plan_path: Callable[[np.ndarray], dict[str, Any]] | None = None


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
        #: Extra yaw [rad] applied to the next segment's transfer (see
        #: ``yaw_search_deg``).
        self._yaw = 0.0
        #: Extra tilt [rad] of the next grasp (see ``grasp_tilt_deg``).
        self._tilt = 0.0
        #: Yaw [deg] the last grasp was turned by; the placement that follows
        #: turns by the same amount so the object sits where the demo put it.
        self._carry_yaw = 0.0
        #: Escape offset [m] for the next alignment (see ``via_offsets``).
        self._via: np.ndarray | None = None

    def _send(self, target: np.ndarray, gripper: float) -> bool:
        """Track ``target`` for one step; False once the episode has ended."""
        if self.env.done():
            return False
        self.env.step(target_to_action(self.env.eef_pose(), target, gripper))
        self.steps += 1
        self._last_target, self._last_gripper = target, float(gripper)
        return not self.env.done()

    def _alignment_replans(self, ref: str) -> int:
        """Return the global count plus reference-specific alignment attempts."""
        return self.cfg.alignment_replans + sum(
            max(0, int(count))
            for key, count in self.cfg.alignment_replans_by_ref.items()
            if key in ref
        )

    def run_segment(self, k: int) -> dict[str, Any]:
        """Run segment ``k``; a grasp that did not pick its object up is retried
        (open, lift, re-align from the object's new pose) up to
        ``grasp_retries`` times."""
        seg = self.segments[k]
        grasping = self.cfg.grasp_retries > 0 and any(
            e[0] == "grasp" for e in seg.get("events", ())
        )
        ref = seg["ref"]
        before = self.env.body_poses([ref])[ref] if grasping else None
        log = self._dispatch(k)
        retries = []
        while (
            grasping
            and len(retries) < self.cfg.grasp_retries
            and not self.env.done()
            and not self._picked_up(ref, before)
        ):
            self._release_and_lift()
            before = self.env.body_poses([ref])[ref]
            log = self._dispatch(k)
            retries.append(log.get("status"))
        if grasping:
            log["grasp_held"] = bool(self._picked_up(ref, before))
            if retries:
                log["grasp_retries"] = retries
        return log

    def _picked_up(self, body: str, before: np.ndarray) -> bool:
        """``body`` rose with the closed gripper and stays near the grip site."""
        now = self.env.body_poses([body])[body]
        eef = self.env.eef_pose()
        return bool(
            self._last_gripper > 0
            and now[2, 3] - before[2, 3] >= 0.015
            and np.linalg.norm(now[:3, 3] - eef[:3, 3]) <= 0.10
        )

    def _release_and_lift(self, lift: float = 0.08) -> None:
        """Open the gripper in place, then rise ``lift`` m."""
        pose = self.env.eef_pose()
        for _ in range(10):
            if not self._send(pose, -1.0):
                return
        up = pose.copy()
        up[2, 3] += lift
        for target in alignment_path(pose, up, self.cfg.align_step):
            if not self._send(target, -1.0):
                return

    def _dispatch(self, k: int) -> dict[str, Any]:
        if self.env.check_paths is None:
            return self._segment(k)
        if self.cfg.feasibility_check == "minimal":
            return self._run_minimal(k)
        if self.cfg.feasibility_check != "select":
            return self._segment(k)
        ref = self.segments[k]["ref"]
        live = self.env.body_poses([ref])[ref]
        candidates = self._feasibility(k, live, 0.0)
        feasible = [x for x in candidates if x["feasible"]]
        if not feasible:
            log = self._segment(k)
            log.update(
                selection="none feasible; configured route", candidates=candidates
            )
            return log
        # Shortest route; ties keep this runner's demo, then the shorter corridor.
        # Routes that touch no free object first, then the shortest; ties keep
        # this runner's demo, then the shorter corridor.
        pick = min(
            feasible,
            key=lambda x: (
                x["first_movable_contact"] is not None,
                x["n"],
                x["demo"] != self.demo,
                x["corridor"],
            ),
        )
        saved = (
            self.demo,
            self.first,
            self.eef,
            self.target,
            self.gripper,
            self.actions,
            self.segments,
            self.cfg,
        )
        rows = self.lib.frames(pick["demo"])
        self.demo, self.first = pick["demo"], rows.start
        self.eef, self.target = self.lib.eef[rows], self.lib.target[rows]
        self.gripper, self.actions = self.lib.gripper[rows], self.lib.actions[rows]
        self.segments = self.lib.segments[pick["demo"]]
        self.cfg = replace(
            self.cfg,
            approach_corridor=pick["corridor"],
            transit_clearance=pick["clearance"],
            carry_clearance=0.0,
        )
        try:
            log = self._segment(k)
        finally:
            (
                self.demo,
                self.first,
                self.eef,
                self.target,
                self.gripper,
                self.actions,
                self.segments,
                self.cfg,
            ) = saved
        log.update(
            selection={
                key: pick[key] for key in ("demo", "corridor", "clearance", "n")
            },
            candidates=candidates,
        )
        return log

    def _use_demo(self, demo: int) -> None:
        """Replay ``demo`` from now on (this and all later segments)."""
        rows = self.lib.frames(demo)
        self.demo, self.first = demo, rows.start
        self.eef, self.target = self.lib.eef[rows], self.lib.target[rows]
        self.gripper, self.actions = self.lib.gripper[rows], self.lib.actions[rows]
        self.segments = self.lib.segments[demo]

    def _run_minimal(self, k: int) -> dict[str, Any]:
        """Change the configured route only when it is infeasible.

        1. The configured route (this demo, configured corridor, no extra
           height) when it is feasible.
        2. Else this demo with another corridor / transit height.
        3. Else this demo turned about the vertical axis (``yaw_search_deg``),
           for placements and for ``yaw_free`` references.
        4. Else, only while nothing is held, another of the nearest demos;
           the switch persists for the later segments, so a grasp and the
           placement that follows come from the same demonstration.
        5. Else the configured route.
        """
        ref = self.segments[k]["ref"]
        live = self.env.body_poses([ref])[ref]
        held = (
            self.gripper[
                segment_bottleneck(self.eef, self.segments[k], self.cfg.radius)
            ]
            > 0
        )
        corridor = self.cfg.approach_corridor
        tried: list[dict[str, Any]] = []

        def best(cands):
            ok = [x for x in cands if x["feasible"]]
            return min(
                ok,
                key=lambda x: (x["first_movable_contact"] is not None, x["n"]),
                default=None,
            )

        # With grasp compensation the held object follows the demo's own path,
        # so a turned grasp needs no turned placement.
        base = self._carry_yaw if held and not self.cfg.grasp_compensation else 0.0
        stage, pick = "configured", None
        cands = self._feasibility(
            k, live, 0.0, [self.demo], (corridor,), (0.0,), (base,)
        )
        tried += cands
        if cands and cands[0]["feasible"]:
            pick = cands[0]
        if pick is None:
            stage = "same demo"
            cands = self._feasibility(
                k,
                live,
                0.0,
                [self.demo],
                tuple(sorted({corridor, 0.2, 0.1, 0.0}, reverse=True)),
                (0.0, 0.06, 0.12, 0.2),
                (base,),
            )
            tried += cands
            pick = best(cands)
        grasping = any(e[0] == "grasp" for e in self.segments[k].get("events", ()))
        if pick is None and self.cfg.grasp_tilt_deg and grasping and not held:
            stage = "tilt"
            cands = self._feasibility(
                k,
                live,
                0.0,
                [self.demo],
                tuple(sorted({corridor, 0.0}, reverse=True)),
                (0.0, 0.06, 0.12),
                tilts=tuple(
                    sign * a for a in self.cfg.grasp_tilt_deg for sign in (1.0, -1.0)
                ),
            )
            tried += cands
            ok = [x for x in cands if x["feasible"]]
            pick = min(
                ok,
                key=lambda x: (
                    abs(x["tilt"]),
                    x["first_movable_contact"] is not None,
                    x["n"],
                ),
                default=None,
            )
        if pick is None and self._yaw_allowed(k, held):
            stage = "yaw"
            turns = tuple(
                base + sign * a for a in self.cfg.yaw_search_deg for sign in (1.0, -1.0)
            )
            cands = self._feasibility(
                k,
                live,
                0.0,
                [self.demo],
                tuple(sorted({corridor, 0.0}, reverse=True)),
                (0.0, 0.06, 0.12),
                turns,
            )
            tried += cands
            # Smallest turn first, then the usual order.
            ok = [x for x in cands if x["feasible"]]
            pick = min(
                ok,
                key=lambda x: (
                    abs(x["yaw"] - base),
                    x["first_movable_contact"] is not None,
                    x["n"],
                ),
                default=None,
            )
        if pick is None and self.cfg.via_offsets:
            stage = "via"
            turns = (base,)
            if self._yaw_allowed(k, held):
                turns += tuple(
                    base + sign * a
                    for a in self.cfg.yaw_search_deg
                    for sign in (1.0, -1.0)
                )
            cands = self._feasibility(
                k,
                live,
                0.0,
                [self.demo],
                tuple(sorted({corridor, 0.0}, reverse=True)),
                (0.0, 0.06),
                turns,
                tuple(tuple(v) for v in self.cfg.via_offsets),
            )
            tried += cands
            ok = [x for x in cands if x["feasible"]]
            pick = min(
                ok,
                key=lambda x: (
                    abs(x["yaw"] - base),
                    x["first_movable_contact"] is not None,
                    x["n"],
                ),
                default=None,
            )
        if pick is None and not held:
            stage = "other demo"
            scene = self.env.body_poses(self.lib.ref_bodies)
            others = [d for d in self.lib.ranked(scene, self.cfg) if d != self.demo]
            cands = self._feasibility(
                k,
                live,
                0.0,
                others[: max(1, self.cfg.feasibility_topk - 1)],
                (corridor,),
                (0.0, 0.06, 0.12),
            )
            tried += cands
            pick = best(cands)
            if pick is not None:
                self._use_demo(pick["demo"])
        if pick is None:
            if self.cfg.rrt_fallback and self.env.plan_path is not None:
                b = segment_bottleneck(self.eef, self.segments[k], self.cfg.radius)
                demo_ref = self.lib.pose[ref][self.first + b]
                transform, _ = self.cfg.transfer(ref, live, demo_ref)
                if base:
                    transform = yaw_about(
                        transform,
                        np.radians(base),
                        self._pivot(k, self.demo, transform, live),
                    )
                planned = self.env.plan_path(transform @ self.eef[b])
                if planned.get("success"):
                    self._yaw = np.radians(base)
                    try:
                        log = self._segment(
                            k, precomputed_path=[np.asarray(p) for p in planned["path"]]
                        )
                    finally:
                        self._yaw = 0.0
                    self._carry_yaw = 0.0 if held else float(base)
                    log.update(
                        selection={
                            "stage": "rrt",
                            **planned,
                            "path": None,
                            "yaw": base,
                        },
                        candidates=tried,
                    )
                    return log
            self._yaw = np.radians(base)
            try:
                log = self._segment(k)
            finally:
                self._yaw = 0.0
            self._carry_yaw = 0.0
            log.update(
                selection={"stage": "none feasible; configured route", "yaw": base},
                candidates=tried,
            )
            return log
        saved = self.cfg
        self.cfg = replace(
            self.cfg,
            approach_corridor=pick["corridor"],
            transit_clearance=pick["clearance"],
        )
        self._yaw = np.radians(pick.get("yaw", 0.0))
        self._tilt = np.radians(pick.get("tilt", 0.0))
        self._via = None if pick.get("via") is None else np.asarray(pick["via"])
        try:
            log = self._segment(k)
        finally:
            self.cfg = saved
            self._yaw = 0.0
            self._tilt = 0.0
            self._via = None
        self._carry_yaw = 0.0 if held else float(pick.get("yaw", 0.0))
        log.update(
            selection={
                "stage": stage,
                **{
                    key: pick.get(key)
                    for key in (
                        "demo",
                        "corridor",
                        "clearance",
                        "yaw",
                        "tilt",
                        "via",
                        "n",
                    )
                },
            },
            candidates=tried if stage != "configured" else [],
        )
        return log

    def _pivot(
        self, k: int, demo: int, transform: np.ndarray, live: np.ndarray
    ) -> np.ndarray:
        """Point a yaw variant of segment ``k`` turns about: where the held
        object (the previous segment's reference) is at contact for a
        placement, else the live reference body."""
        seg = self.lib.segments[demo][k]
        rows = self.lib.frames(demo)
        if k > 0:
            held = self.lib.segments[demo][k - 1]["ref"]
            if self.lib.gripper[rows][seg["c"]] > 0 and held in self.lib.pose:
                return (transform @ self.lib.pose[held][rows.start + seg["c"]])[:3, 3]
        ref = seg["ref"]
        offset = next(
            (v for key, v in self.cfg.yaw_pivot.items() if key in ref), (0.0, 0.0, 0.0)
        )
        return (live @ np.append(np.asarray(offset, dtype=float), 1.0))[:3]

    def _grasp_delta(self, k: int, demo: int, b: int) -> np.ndarray | None:
        """In-hand correction ``demo_grasp @ inv(live_grasp)`` for a placement.

        Applies (with ``grasp_compensation``) to segment ``k`` of ``demo`` when
        it releases the object grasped in the previous segment and that object
        is between the fingers now; ``None`` otherwise. Right-multiplied onto
        the demo's grip-site poses, it makes the held object follow the demo's.
        """
        if not self.cfg.grasp_compensation or k == 0:
            return None
        rows = self.lib.frames(demo)
        segs = self.lib.segments[demo]
        seg, previous = segs[k], segs[k - 1]
        held = previous["ref"]
        if (
            self.lib.gripper[rows][b] <= 0
            or held not in self.lib.pose
            or not any(e[0] in {"release", "end"} for e in seg["events"])
            or not any(e[0] == "grasp" for e in previous["events"])
        ):
            return None
        live_grasp = (
            np.linalg.inv(self.env.eef_pose()) @ self.env.body_poses([held])[held]
        )
        if np.linalg.norm(live_grasp[:3, 3]) >= 0.15:
            return None
        demo_grasp = (
            np.linalg.inv(self.lib.eef[rows][b]) @ self.lib.pose[held][rows.start + b]
        )
        return demo_grasp @ np.linalg.inv(live_grasp)

    def _yaw_allowed(self, k: int, held: bool) -> bool:
        ref = self.segments[k]["ref"]
        return bool(self.cfg.yaw_search_deg) and (
            held or any(key in ref for key in self.cfg.yaw_free)
        )

    def _segment(
        self, k: int, precomputed_path: list[np.ndarray] | None = None
    ) -> dict[str, Any]:
        seg = self.segments[k]
        ref, start, c, end = seg["ref"], seg["start"], seg["c"], seg["end"]
        radius = self.cfg.radius
        b = segment_bottleneck(self.eef, seg, radius)
        live = self.env.body_poses([ref])[ref]
        demo_ref = self.lib.pose[ref][self.first + b]
        transform, dyaw = self.cfg.transfer(ref, live, demo_ref)
        if self._yaw:
            transform = yaw_about(
                transform, self._yaw, self._pivot(k, self.demo, transform, live)
            )
        if self._tilt:
            transform = tilt_about(
                transform, self._tilt, self.eef[c], self.cfg.grasp_tilt_axis
            )
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
        grasp_delta = self._grasp_delta(k, self.demo, b)
        if grasp_delta is None:
            grasp_delta = np.eye(4)
        else:
            log["grasp_correction_cm"] = round(
                100 * float(np.linalg.norm(grasp_delta[:3, 3])), 3
            )
            log["grasp_correction_deg"] = round(rot_angle_deg(grasp_delta[:3, :3]), 2)
            goal = goal @ grasp_delta
        hold = self.gripper[b]
        if self.cfg.feasibility_check == "log" and self.env.check_paths is not None:
            log["feasibility"] = self._feasibility(k, live, hold)
        corridor: list[np.ndarray] = []
        if self.cfg.approach_corridor > 0:
            entry = bottleneck_frame(self.eef, start, b, self.cfg.approach_corridor)
            corridor = [transform @ self.eef[f] @ grasp_delta for f in range(entry, b)]
            log["corridor_frames"] = [entry, b]
        attempts = []
        align_steps, total_settle = 0, 0
        for attempt in range(self._alignment_replans(ref) + 1):
            clearance = (
                0.06 * attempt
                + self.cfg.transit_clearance
                + (self.cfg.carry_clearance if hold > 0 else 0.0)
            )
            if precomputed_path is not None and attempt == 0:
                path = precomputed_path
            elif self.cfg.approach == "dmp" and attempt == 0 and b > start:
                demo_path = [
                    transform @ self.eef[f] @ grasp_delta for f in range(start, b + 1)
                ]
                path = dmp_path(self.env.eef_pose(), demo_path, self.cfg.align_step)
                log["approach"] = "dmp"
            elif corridor:
                path = (
                    via_path(
                        self.env.eef_pose(),
                        corridor[0],
                        self._via if attempt == 0 else None,
                        self.cfg.align_step,
                        clearance,
                    )
                    + corridor[1:]
                    + [goal]
                )
            else:
                path = via_path(
                    self.env.eef_pose(),
                    goal,
                    self._via if attempt == 0 else None,
                    self.cfg.align_step,
                    clearance,
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
        overshoot = next(
            (
                float(distance)
                for key, distance in self.cfg.segment_overshoot_by_ref.items()
                if key in ref
            ),
            0.0,
        )
        if overshoot > 0 and end > b:
            tail_start = max(b, end - 5)
            first_target = transform @ self.target[tail_start] @ grasp_delta
            last_target = transform @ self.target[end - 1] @ grasp_delta
            direction = last_target[:3, 3] - first_target[:3, 3]
            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                direction /= norm
                count = int(np.ceil(overshoot / self.cfg.align_step))
                for i in range(1, count + 1):
                    extra = min(i * self.cfg.align_step, overshoot)
                    target = last_target.copy()
                    target[:3, 3] += direction * extra
                    if not self._send(target, self.gripper[end - 1]):
                        return self._close(log, n0, "done during segment overshoot")
                log["overshoot_m"] = overshoot
        return self._close(log, n0, "segment replayed")

    def _feasibility(
        self,
        k: int,
        live: np.ndarray,
        hold: float,
        demos: list[int] | None = None,
        lengths: tuple[float, ...] = (0.0, 0.1, 0.2),
        clearances: tuple[float, ...] = (0.0, 0.06, 0.12, 0.2),
        yaws: tuple[float, ...] = (0.0,),
        vias: tuple = (None,),
        tilts: tuple[float, ...] = (0.0,),
    ) -> list[dict[str, Any]]:
        """Candidate alignment routes for segment ``k`` and their static feasibility.

        Candidates: the ``feasibility_topk`` kept demos nearest the live scene
        (this runner's demo first) x approach corridor {0, 0.1, 0.2} m x extra
        transit height {0, 0.06, 0.12, 0.2} m. Each route ends at that demo's
        transferred bottleneck. Nothing is executed.
        """
        lib, seg = self.lib, self.segments[k]
        ref = seg["ref"]
        if demos is None:
            scene = self.env.body_poses(lib.ref_bodies)
            demos = [self.demo] + [
                d for d in lib.ranked(scene, self.cfg) if d != self.demo
            ][: max(0, self.cfg.feasibility_topk - 1)]
        ranked = demos
        start = self.env.eef_pose()
        names, paths, tails = [], [], []
        for d in ranked:
            other = lib.segments[d][k] if k < len(lib.segments[d]) else None
            if other is None or other["ref"] != ref:
                continue
            rows = lib.frames(d)
            eef = lib.eef[rows]
            b = segment_bottleneck(eef, other, self.cfg.radius)
            transform, _ = self.cfg.transfer(ref, live, lib.pose[ref][rows.start + b])
            base = transform
            for yaw, tilt in [(y, t) for y in yaws for t in tilts]:
                transform = (
                    yaw_about(base, np.radians(yaw), self._pivot(k, d, base, live))
                    if yaw
                    else base
                )
                if tilt:
                    transform = tilt_about(
                        transform,
                        np.radians(tilt),
                        eef[other["c"]],
                        self.cfg.grasp_tilt_axis,
                    )
                self._candidate_routes(
                    lib,
                    rows,
                    eef,
                    other,
                    b,
                    transform,
                    start,
                    lengths,
                    clearances,
                    {"demo": d, "yaw": yaw, **({"tilt": tilt} if tilt else {})},
                    names,
                    paths,
                    tails,
                    vias,
                    self._grasp_delta(k, d, b),
                )
        options = None
        if self.cfg.feasibility_interaction:
            placing = (
                self.gripper[segment_bottleneck(self.eef, seg, self.cfg.radius)] > 0
            )
            # A grasp may touch its object all the way in; a placement into a
            # container (the reference) only right before release.
            options = (
                {
                    "ignore_families": ["table"],
                    "tail_families": [_family(ref)],
                    "tails": tails,
                }
                if placing
                else {"ignore_families": [_family(ref), "table"]}
            )
        results = self.env.check_paths(paths, options) if paths else []
        return [dict(n, **r) for n, r in zip(names, results)]

    def _candidate_routes(
        self,
        lib,
        rows,
        eef,
        other,
        b,
        transform,
        start,
        lengths,
        clearances,
        tag,
        names,
        paths,
        tails,
        vias=(None,),
        grasp_delta: np.ndarray | None = None,
    ) -> None:
        """Append the routes of one (demo, yaw) to ``paths`` / ``names``.

        ``grasp_delta`` (see ``_grasp_delta``) corrects every demo pose the way
        execution does.
        """
        delta = np.eye(4) if grasp_delta is None else grasp_delta
        goal = transform @ eef[b] @ delta
        # The source interaction up to contact: it must not run into a
        # fixture either (the reference object itself is ignored).
        reach_in = (
            [
                transform @ lib.target[rows][f] @ delta
                for f in range(b + 1, other["c"] + 1)
            ]
            if self.cfg.feasibility_interaction
            else []
        )
        # Where the reference may be touched: the last 3 cm before contact.
        near = sum(
            1
            for f in range(b + 1, other["c"] + 1)
            if np.linalg.norm(eef[f, :3, 3] - eef[other["c"], :3, 3]) <= 0.03
        )
        for length in lengths:
            entry = bottleneck_frame(eef, other["start"], b, length) if length else b
            lead = [transform @ eef[f] @ delta for f in range(entry, b)]
            for clearance in clearances:
                for via in vias:
                    head = lead[0] if lead else goal
                    path = via_path(start, head, via, self.cfg.align_step, clearance)
                    paths.append(path + lead[1:] + ([goal] if lead else []) + reach_in)
                    names.append(
                        dict(tag, corridor=length, clearance=clearance, via=via)
                    )
                    tails.append(near if reach_in else 0)

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


def reach_retrieval(
    lib: TaskLibrary, scene: dict[str, np.ndarray], cfg: MT3Config, env: EnvHooks
) -> tuple[int, dict[str, Any]]:
    """Kept demo with the most segments whose transferred bottleneck is
    reachable (IK only, on a copy of the live scene); ties go to the nearest.

    Each segment's bottleneck is transferred with the reference body's pose in
    ``scene`` (the start of the episode), so it is exact for fixtures and
    placement targets that stay put, approximate for objects moved earlier.
    """
    ranked = lib.ranked(scene, cfg)
    start = env.eef_pose()
    paths, owners = [], []
    for d in ranked:
        rows = lib.frames(d)
        eef = lib.eef[rows]
        for seg in lib.segments[d]:
            ref = seg["ref"]
            b = segment_bottleneck(eef, seg, cfg.radius)
            transform, _ = cfg.transfer(ref, scene[ref], lib.pose[ref][rows.start + b])
            goal = transform @ eef[b]
            # A coarse straight approach warm-starts the IK toward the goal.
            n = 8
            paths.append(
                [
                    make_pose(
                        start[:3, 3] + (goal[:3, 3] - start[:3, 3]) * (j / n),
                        slerp(start[:3, :3], goal[:3, :3], j / n),
                    )
                    for j in range(1, n + 1)
                ]
            )
            owners.append(d)
    results = env.check_paths(paths)
    reach: dict[int, int] = dict.fromkeys(ranked, 0)
    for d, r in zip(owners, results):
        reach[d] += int(
            r["goal_ik_pos_err_cm"] <= 3.0 and r["goal_ik_rot_err_deg"] <= 8.0
        )
    best = max(ranked, key=lambda d: (reach[d], -ranked.index(d)))
    return best, {
        "demo": best,
        "reachable_segments": reach[best],
        "segments": len(lib.segments[best]),
        "nearest_demo": ranked[0],
        "nearest_reachable_segments": reach[ranked[0]],
        "candidates": len(ranked),
    }

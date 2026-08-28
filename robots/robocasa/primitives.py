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

"""RoboCasaPrimitives — action primitives, perception, and VLA execution for RoboCasa."""

import os

import numpy as np

from robots.robocasa.rldx_skill import RLDXSkill

OSC_POS_SCALE = 0.05  # action 1.0 -> 0.05 m target delta
OSC_ROT_SCALE = 0.5  # action 1.0 -> 0.5 rad
MAX_ENV_STEPS = 10_000
MAX_VLA_CHUNKS = 1_000
MAX_VLA_ACTION_STEPS = 512
MAX_HI_RES = 4096


def _bounded_int(value, name, *, minimum=1, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


def _finite_scalar(value, name, *, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


def _finite_vector(value, name, size):
    array = np.asarray(value)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain numeric values")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _env_flag(name, default):
    raw = os.environ.get(name, default)
    if raw not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1, got {raw!r}")
    return raw == "1"


class RoboCasaPrimitives:
    def __init__(self, env_client, workdir, hi_res, vla_client, check_cancelled=None):
        self.env = env_client
        self.workdir = workdir
        self.hi_res = (
            None
            if hi_res is None
            else _bounded_int(hi_res, "hi_res", maximum=MAX_HI_RES)
        )
        # Cancellation checkpoint callback (e.g. Toolkit.raise_if_cancelled),
        # invoked before long-running env.step loops so an interrupted tool
        # operation stops promptly instead of draining the whole episode.
        self._check_cancelled = check_cancelled
        # World maps (per-pixel world xyz) are dumped EVERY step — they are how the agent
        # localizes every object, so they are never optional. Disk is bounded by keeping only
        # the last `_keep_heavy` steps' heavy npy on disk (pruned in dump_state) + each
        # launcher's per-cell workdir cleanup.
        try:
            keep_heavy = int(os.environ.get("RLDX_KEEP_HEAVY_NPY", "25"))
        except ValueError as exc:
            raise ValueError("RLDX_KEEP_HEAVY_NPY must be an integer") from exc
        self._keep_heavy = _bounded_int(
            keep_heavy, "RLDX_KEEP_HEAVY_NPY", maximum=10_000
        )
        # reset (restart the episode) is ONLY legitimate in EXPLORE mode (reset-based recipe
        # search). It is FORBIDDEN in multi-seed / matched-scene evaluation (a give-up-and-
        # restart). Gated by RLDX_ALLOW_RESET (default 0 = off); explore runs opt in.
        self._allow_reset = _env_flag("RLDX_ALLOW_RESET", "0")
        self._perception_isolation = _env_flag("RLDX_PERCEPTION_ISOLATION", "0")
        os.makedirs(workdir, exist_ok=True)
        if self.env.current_raw_obs is None:
            raise RuntimeError(
                "RoboCasaEnvClient must complete its initial reset first"
            )
        self._pos_jac = None  # 3x3 action(arm xyz) -> world dpos
        self._fwd_offset = None  # world_forward_heading = base_yaw + offset
        self._cam_meta_cache = {}
        self._rldx = RLDXSkill(
            self.env, vla_client=vla_client, check_cancelled=check_cancelled
        )
        # "mid-call" desync guard: True whenever a NON-VLA primitive (move/navigate/
        # manual grasp) or a reset has stepped the env since the last rldx_skill call.
        # The next rldx_skill then reseeds its per-sim-step frame history (else the VLA
        # sees a pre-manual history stitched onto a post-manual current frame -> OOD).
        # Consecutive VLA calls with NO manual step in between keep history continuity.
        self._vla_desync = True
        # Recording (off by default — zero overhead when not recording)
        self._recording = False
        self._frames = []

    def _invalidate_geometry(self, *, base_moved=False, vla_moved=False):
        """Drop pose-dependent calibration after motions outside the OSC servo."""
        if base_moved or vla_moved:
            self._pos_jac = None
            self._cam_meta_cache.pop("agentview", None)
            self._cam_meta_cache.pop("navview", None)
        if vla_moved:
            self._fwd_offset = None
            self._cam_meta_cache.pop("wrist", None)

    # ---- recording methods ----
    def start_recording(self):
        self._recording = True
        self._frames = []

    def record_frame(self, img=None):
        """Snapshot the current agentview to the frame buffer, if recording."""
        if img is None:
            img = self.env.render_camera(
                camera_name="agentview",
                height=256,
                width=256,
                depth=False,
            )
            img = np.asarray(img, dtype=np.uint8)
            img = np.ascontiguousarray(img)
        self._frames.append(img)

    def recorded_frame_count(self) -> int:
        return len(self._frames)

    def frame_slice(self, start: int) -> list[np.ndarray]:
        return list(self._frames[int(start) :])

    def stop_recording(self) -> list[np.ndarray]:
        frames = list(self._frames)
        self._recording = False
        self._frames = []
        return frames

    # ---- action helpers ----
    def _zero(self, base_mode=-1.0):
        a = np.zeros(12)
        a[11] = base_mode
        return a

    def _hold_gripper_val(self, g):
        # +1 close/hold, -1 open
        g = _finite_scalar(g, "gripper")
        return float(np.clip(g, -1, 1))

    @staticmethod
    def _validate_gripper(gripper):
        if gripper is None:
            return gripper
        if isinstance(gripper, str):
            if gripper == "hold":
                return gripper
            raise ValueError("gripper string must be 'hold'")
        return _finite_scalar(gripper, "gripper")

    def _resolve_grip(self, gripper, target_q):
        """Return the a[6] gripper command for a motion step.
        gripper="hold"/None (DEFAULT for moves) -> SERVO the fingers back to `target_q`,
        the width they had when the motion began. This is the carry-safe hold: the
        gripper action is a CLOSE-VELOCITY command, so a sustained +1 keeps driving the
        fingers shut and SQUEEZES a small object OUT (verified: bread qpos 0.0376 -> 0.0005
        during a +1 carry). Servoing to the grasped width holds the object without crushing
        it and without letting it drift open. A numeric gripper (+1 close / -1 open) is an
        EXPLICIT override and passes through unchanged."""
        gripper = self._validate_gripper(gripper)
        if isinstance(gripper, str) or gripper is None:
            cur = float(self.env.gripper_qpos[0])
            return float(
                np.clip(60.0 * (cur - target_q), -1.0, 1.0)
            )  # +a[6] closes (qpos↓)
        return self._hold_gripper_val(gripper)

    def _success_latched(self):
        value = getattr(self.env, "success_latched", None)
        if value is not None:
            return bool(value)
        return bool(self.env.check_success())

    def _apply_step(self, action, *, record=False):
        """Apply at most one physical step and stop permanently on first success."""
        if self._success_latched():
            return False
        if self._check_cancelled is not None:
            self._check_cancelled()
        self.env.step(action)
        if record and self._recording:
            self.record_frame()
        return not self._success_latched()

    def _step_arm(self, dpos=(0, 0, 0), drot=(0, 0, 0), gripper=-1.0, n=1):
        dpos = _finite_vector(dpos, "dpos", 3)
        drot = _finite_vector(drot, "drot", 3)
        gripper = _finite_scalar(gripper, "gripper")
        n = _bounded_int(n, "n", maximum=MAX_ENV_STEPS)
        a = self._zero(base_mode=-1.0)
        a[0:3] = np.clip(np.asarray(dpos) / OSC_POS_SCALE, -1, 1)
        a[3:6] = np.clip(np.asarray(drot) / OSC_ROT_SCALE, -1, 1)
        a[6] = self._hold_gripper_val(gripper)
        for _ in range(n):
            if not self._apply_step(a, record=True):
                break
        return self.env.eef_pos

    # ---- Phase 2: online jacobian + move_to ----
    def _calibrate_pos_jacobian(self, gripper=-1.0):
        """Probe 3 unit arm-xyz actions, measure world dpos -> 3x3 jacobian J s.t.
        world_dpos ~= J @ action_xyz. move_to inverts J to map desired world delta."""
        cols = []
        for axis in range(3):
            if self._success_latched():
                return None
            p0 = self.env.eef_pos.copy()
            a = self._zero()
            a[axis] = 0.4
            a[6] = gripper
            applied = 0
            for _ in range(3):
                applied += 1
                if not self._apply_step(a):
                    return None
            d = (self.env.eef_pos - p0) / (0.4 * applied)
            cols.append(d)
            # settle back is not needed (closed-loop re-reads); keep going
        self._pos_jac = np.stack(cols, axis=1)  # 3x3: world_dpos = J @ a_xyz
        if (
            not np.isfinite(self._pos_jac).all()
            or np.linalg.matrix_rank(self._pos_jac) < 3
        ):
            self._pos_jac = None
            raise RuntimeError(
                "arm position Jacobian calibration is singular or non-finite"
            )
        return self._pos_jac

    def move_to(self, xyz, gripper="hold", step_clip=0.02, max_steps=200, tol=0.012):
        """Closed-loop OSC servo of the eef to a WORLD xyz target.
        gripper="hold" (DEFAULT) maintains the CURRENT finger width — carry a grasped
        object WITHOUT crushing it (a sustained +1 squeezes small objects out) or letting
        it drop. Pass +1 to actively CLOSE, -1 to actively OPEN."""
        self._vla_desync = True
        target = _finite_vector(xyz, "xyz", 3)
        gripper = self._validate_gripper(gripper)
        step_clip = _finite_scalar(
            step_clip, "step_clip", minimum=np.finfo(float).eps, maximum=0.30
        )
        max_steps = _bounded_int(max_steps, "max_steps", maximum=MAX_ENV_STEPS)
        tol = _finite_scalar(tol, "tol", minimum=np.finfo(float).eps, maximum=1.0)
        target_q = float(self.env.gripper_qpos[0])  # finger width to hold
        if self._success_latched():
            cur = self.env.eef_pos
            return {
                "ok": True,
                "status": "task_success",
                "steps": 0,
                "final_dist": float(np.linalg.norm(target - cur)),
                "eef": cur.tolist(),
                "gripper_qpos": round(float(self.env.gripper_qpos[0]), 4),
            }
        if self._pos_jac is None:
            calibrated = self._calibrate_pos_jacobian(
                gripper=self._resolve_grip(gripper, target_q)
            )
            if calibrated is None:
                cur = self.env.eef_pos
                return {
                    "ok": True,
                    "status": "task_success",
                    "steps": 0,
                    "final_dist": float(np.linalg.norm(target - cur)),
                    "eef": cur.tolist(),
                    "gripper_qpos": round(float(self.env.gripper_qpos[0]), 4),
                }
        Jinv = np.linalg.pinv(self._pos_jac)
        for i in range(max_steps):
            cur = self.env.eef_pos
            err = target - cur
            dist = float(np.linalg.norm(err))
            if dist < tol:
                return {
                    "ok": True,
                    "steps": i,
                    "final_dist": dist,
                    "eef": cur.tolist(),
                    "gripper_qpos": round(float(self.env.gripper_qpos[0]), 4),
                }
            step_world = err if dist <= step_clip else err / dist * step_clip
            a_xyz = np.clip(Jinv @ step_world, -1, 1)
            a = self._zero()
            a[0:3] = a_xyz
            a[6] = self._resolve_grip(gripper, target_q)
            if not self._apply_step(a, record=True):
                cur = self.env.eef_pos
                return {
                    "ok": True,
                    "status": "task_success",
                    "steps": i + 1,
                    "final_dist": float(np.linalg.norm(target - cur)),
                    "eef": cur.tolist(),
                    "gripper_qpos": round(float(self.env.gripper_qpos[0]), 4),
                }
        cur = self.env.eef_pos
        return {
            "ok": False,
            "steps": max_steps,
            "final_dist": float(np.linalg.norm(target - cur)),
            "eef": cur.tolist(),
            "gripper_qpos": round(float(self.env.gripper_qpos[0]), 4),
        }

    def move_delta(self, dxyz, gripper="hold", step_clip=0.02, max_steps=80):
        self._vla_desync = True
        dxyz = _finite_vector(dxyz, "dxyz", 3)
        gripper = self._validate_gripper(gripper)
        return self.move_to(self.env.eef_pos + dxyz, gripper, step_clip, max_steps)

    def rotate_pitch(self, target_pitch=0.6, gripper=1, n=12):
        """Tilt the wrist forward (axis-angle about control-x). Reuses arm drot."""
        self._vla_desync = True
        target_pitch = _finite_scalar(
            target_pitch, "target_pitch", minimum=-1.5, maximum=1.5
        )
        gripper = _finite_scalar(gripper, "gripper")
        n = _bounded_int(n, "n", maximum=MAX_ENV_STEPS)
        per = target_pitch / n
        for _ in range(n):
            self._step_arm(drot=(per, 0, 0), gripper=gripper, n=1)
            if self._success_latched():
                break
        return {"ok": True, "eef": self.env.eef_pos.tolist()}

    def set_gripper(self, gripper=1, steps=10):
        self._vla_desync = True
        gripper = _finite_scalar(gripper, "gripper")
        steps = _bounded_int(steps, "steps", maximum=MAX_ENV_STEPS)
        g = self._hold_gripper_val(gripper)
        a = self._zero()
        a[6] = g
        for _ in range(steps):
            if not self._apply_step(a):
                break
        return {"ok": True, "gripper_qpos": self.env.gripper_qpos.tolist()}

    def release(self, steps=10):
        return self.set_gripper(-1.0, steps=steps)

    # ---- Phase 4: scripted grasp ----
    def scripted_grasp(self, xyz, approach_z=0.10, grasp_z_offset=0.0, step_clip=0.02):
        """Open -> hover above target -> descend -> close -> lift. Coarse; replace
        with RLDX closed-loop grasp for hard objects."""
        t = _finite_vector(xyz, "xyz", 3)
        approach_z = _finite_scalar(approach_z, "approach_z", minimum=0.0, maximum=1.0)
        grasp_z_offset = _finite_scalar(
            grasp_z_offset, "grasp_z_offset", minimum=-1.0, maximum=1.0
        )
        step_clip = _finite_scalar(
            step_clip, "step_clip", minimum=np.finfo(float).eps, maximum=0.30
        )
        self.set_gripper(-1.0, steps=4)
        r = self.move_to(t + [0, 0, approach_z], gripper=-1.0, step_clip=step_clip)
        if not r["ok"]:
            return {**r, "stage": "approach"}
        r = self.move_to(
            t + [0, 0, grasp_z_offset], gripper=-1.0, step_clip=0.012, tol=0.01
        )
        if not r["ok"]:
            return {**r, "stage": "descent"}
        self.set_gripper(+1.0, steps=14)
        r = self.move_to(t + [0, 0, approach_z + 0.05], gripper="hold", step_clip=0.015)
        if not r["ok"]:
            return {**r, "stage": "lift"}
        return {
            "ok": True,
            "gripper_qpos": self.env.gripper_qpos.tolist(),
            "eef": self.env.eef_pos.tolist(),
        }

    # ---- Phase 3: navigation ----
    def _base_pose(self):
        o = self.env.current_raw_obs
        return (
            _finite_vector(o["robot0_base_pos"], "robot0_base_pos", 3),
            _finite_vector(o["robot0_base_quat"], "robot0_base_quat", 4),
        )

    def move_base(self, forward=0, lateral=0, turn=0, steps=10, gripper="hold"):
        """Raw base velocity command (robot-local: +fwd, +lateral, +turn yaw).
        gripper="hold" (DEFAULT) maintains the finger width while driving (carry-safe)."""
        self._vla_desync = True
        forward = _finite_scalar(forward, "forward")
        lateral = _finite_scalar(lateral, "lateral")
        turn = _finite_scalar(turn, "turn")
        steps = _bounded_int(steps, "steps", maximum=MAX_ENV_STEPS)
        gripper = self._validate_gripper(gripper)
        self._invalidate_geometry(base_moved=True)
        target_q = float(self.env.gripper_qpos[0])
        a = self._zero(base_mode=1.0)
        a[7:10] = [
            np.clip(forward, -1, 1),
            np.clip(lateral, -1, 1),
            np.clip(turn, -1, 1),
        ]
        bp0, _ = self._base_pose()
        for _ in range(steps):
            a[6] = self._resolve_grip(gripper, target_q)
            if not self._apply_step(a):
                break
        bp1, _ = self._base_pose()
        return {
            "ok": True,
            "base_moved": (bp1 - bp0).tolist(),
            "base_pos": bp1.tolist(),
        }

    def _yaw(self):
        from scipy.spatial.transform import Rotation as R

        quat = _finite_vector(
            self.env.current_raw_obs["robot0_base_quat"], "robot0_base_quat", 4
        )
        if float(np.linalg.norm(quat)) <= np.finfo(float).eps:
            raise ValueError("robot0_base_quat must be nonzero")
        return float(R.from_quat(quat).as_euler("xyz")[2])

    def _calibrate_forward(self, gripper=1.0):
        """Drive forward briefly, measure the WORLD direction the base actually goes,
        so navigate_to can steer regardless of the base->world frame offset."""
        gripper = _finite_scalar(gripper, "gripper")
        self._invalidate_geometry(base_moved=True)
        p0, _ = self._base_pose()
        y0 = self._yaw()
        a = self._zero(base_mode=1.0)
        a[6] = self._hold_gripper_val(gripper)
        a[7] = 1.0
        for _ in range(6):
            if not self._apply_step(a):
                break
        p1, _ = self._base_pose()
        disp = (p1 - p0)[:2]
        if np.linalg.norm(disp) > 0.005:
            self._fwd_offset = np.arctan2(disp[1], disp[0]) - y0
        else:
            self._fwd_offset = 0.0
        return self._fwd_offset

    def navigate_to(self, xy, tol=0.20, max_steps=300, gripper="hold"):
        """Drive the mobile base toward a WORLD (x,y) target. Online-calibrates the
        base forward->world heading, then turns to face + drives forward at full
        speed (closed-loop). Holds the arm (base_mode>0). Base ~2.3mm/step.
        gripper="hold" (DEFAULT) maintains the finger width while driving (carry-safe)."""
        self._vla_desync = True
        target = _finite_vector(xy, "xy", 2)
        tol = _finite_scalar(tol, "tol", minimum=np.finfo(float).eps, maximum=5.0)
        max_steps = _bounded_int(max_steps, "max_steps", maximum=MAX_ENV_STEPS)
        gripper = self._validate_gripper(gripper)
        self._invalidate_geometry(base_moved=True)
        target_q = float(self.env.gripper_qpos[0])
        if self._fwd_offset is None:
            self._calibrate_forward(self._resolve_grip(gripper, target_q))
        start = self._base_pose()[0][:2].copy()
        for i in range(max_steps):
            bp, _ = self._base_pose()
            if self._success_latched():
                return {
                    "ok": True,
                    "status": "task_success",
                    "steps": i,
                    "final_dist": float(np.linalg.norm(target - bp[:2])),
                    "moved": float(np.linalg.norm(bp[:2] - start)),
                    "start_pos": start.tolist(),
                    "base_pos": bp.tolist(),
                }
            to = target - bp[:2]
            dist = float(np.linalg.norm(to))
            if dist < tol:
                moved = float(np.linalg.norm(bp[:2] - start))
                return {
                    "ok": True,
                    "steps": i,
                    "final_dist": dist,
                    "moved": moved,
                    "start_pos": start.tolist(),
                    "base_pos": bp.tolist(),
                }
            world_dir = np.arctan2(to[1], to[0])
            cur_forward = self._yaw() + self._fwd_offset
            dyaw = (world_dir - cur_forward + np.pi) % (2 * np.pi) - np.pi
            a = self._zero(base_mode=1.0)
            a[6] = self._resolve_grip(gripper, target_q)
            if abs(dyaw) > 0.30:  # turn to face the target
                a[9] = float(np.sign(dyaw))
            else:  # drive forward + small steer
                a[7] = 1.0
                a[9] = float(np.clip(dyaw * 1.5, -0.4, 0.4))
            if not self._apply_step(a):
                bp = self._base_pose()[0]
                return {
                    "ok": True,
                    "status": "task_success",
                    "steps": i + 1,
                    "final_dist": float(np.linalg.norm(target - bp[:2])),
                    "moved": float(np.linalg.norm(bp[:2] - start)),
                    "start_pos": start.tolist(),
                    "base_pos": bp.tolist(),
                }
        bp, _ = self._base_pose()
        moved = float(np.linalg.norm(bp[:2] - start))
        # stuck = ran out of steps having barely moved (rammed a fixture, no path-planning)
        return {
            "ok": False,
            "steps": max_steps,
            "final_dist": float(np.linalg.norm(target - bp[:2])),
            "moved": moved,
            "stuck": moved < 0.12,
            "start_pos": start.tolist(),
            "base_pos": bp.tolist(),
        }

    def dump_success_criteria(self):
        """Return this task's EXACT success condition text (the env's _check_success +
        the helper/fixture predicates it calls), so the agent knows WHAT counts as done.
        This is the success LOGIC (conditions on named objects/fixtures + thresholds) —
        it does NOT reveal GT object COORDINATES (the agent still localizes every object
        from perception). The caller writes it to the run's EnvState."""
        if self._perception_isolation:
            raise PermissionError(
                "success criteria are unavailable in perception-isolated evaluation"
            )
        return self.env.get_success_criteria_text()

    def task_progress(self):
        """NUMERIC progress signal toward this task's success — generic for ALL tasks.
        The state json only exposes the final `success` bool; an agent driving a task
        whose success is a COUNTER/THRESHOLD (washed_time>=25, success_time>=5) or a set
        of sub-predicates (gripper_obj_far, is_dishwasher_closed, contact_check) is then
        flying blind — it can't tell water-on from water-off, or 3/25 from 24/25.
        We surface the env's OWN intermediate quantities WITHOUT hand-coding any task:
          1) self.<attr> int/float/bool COUNTERS/FLAGS referenced in _check_success
             (washed_time, success_time, _turned_on, ...), read live.
          2) the INTERMEDIATE LOCALS the real _check_success computes this step
             (each sub-predicate bool / distance / range), captured by tracing one
             read-only call of _check_success.
        This is the success CRITERION's current value — NOT ground-truth object coords
        (the agent still localizes objects from perception)."""
        if self._perception_isolation:
            return {}
        try:
            return self.env.get_task_progress()
        except Exception:
            return {}

    # ---- Phase 1: perception (state dict + rendered arrays) ----
    def current_state_dict(self) -> dict:
        """Return the proprio + task + success state dict (NO object coords).

        Rendered RGB/depth/world arrays are produced by the caller via
        ``_render_observation_artifacts`` (kept out of primitives so the primitives
        object stays free of any output-dir / file-IO concern).
        """
        o = self.env.current_raw_obs
        if not isinstance(o, dict):
            raise TypeError("current_raw_obs must be a dict")
        task_language = o.get("language") or self.env.get_task_language() or ""
        if not isinstance(task_language, str):
            raise TypeError("task language must be a string")
        success = self.env.check_success()
        base_pos = _finite_vector(o["robot0_base_pos"], "robot0_base_pos", 3)
        base_quat = _finite_vector(o["robot0_base_quat"], "robot0_base_quat", 4)
        result = {
            "task_language": task_language,
            "success": success,
            "state": {
                "robot0_eef_pos": self.env.eef_pos.tolist(),
                "robot0_eef_quat": self.env.eef_quat.tolist(),
                "robot0_gripper_qpos": self.env.gripper_qpos.tolist(),
                "robot0_base_pos": base_pos.tolist(),
                "robot0_base_quat": base_quat.tolist(),
            },
        }
        if not self._perception_isolation:
            # Exploratory runs may opt into predicate progress. Formal evaluation
            # omits both fields entirely and never invokes the progress RPC.
            result["task_progress"] = self.task_progress()
            result["robocasa_terminated"] = success
        return result

    # ---- VLA execution ----
    def run_rldx_skill(
        self,
        base_clip,
        max_chunks,
        use_prompt,
        prompt,
        force_reset,
        n_action_steps,
        settle_patience,
        settle_eps,
    ):
        """Execute a VLA skill (RLDX). Resolves task_lang, computes force_reset with
        _vla_desync, clears _vla_desync."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if use_prompt not in (None, True, False):
            raise TypeError("use_prompt must be true, false, or null")
        if not isinstance(force_reset, (bool, np.bool_)):
            raise TypeError("force_reset must be a boolean")
        if base_clip is not None:
            base_clip = _finite_scalar(base_clip, "base_clip", minimum=0.0, maximum=1.0)
        max_chunks = _bounded_int(max_chunks, "max_chunks", maximum=MAX_VLA_CHUNKS)
        n_action_steps = _bounded_int(
            n_action_steps, "n_action_steps", maximum=MAX_VLA_ACTION_STEPS
        )
        settle_patience = _bounded_int(
            settle_patience, "settle_patience", maximum=MAX_VLA_CHUNKS
        )
        settle_eps = _finite_scalar(
            settle_eps, "settle_eps", minimum=np.finfo(float).eps, maximum=1.0
        )
        if use_prompt:
            task_lang = prompt
        else:
            task_lang = (
                self.env.current_raw_obs.get("language") or self.env.get_task_language()
            ) or prompt
        # Auto-reseed history if a non-VLA primitive ran since the last VLA call
        # (read _vla_desync BEFORE clearing it)
        fr = bool(force_reset) or self._vla_desync
        self._vla_desync = False
        try:
            return self._rldx.run(
                task_lang,
                max_chunks,
                n_action_steps,
                base_clip=base_clip,
                settle_patience=settle_patience,
                settle_eps=settle_eps,
                force_reset=fr,
                recording=self._recording,
                record_frame=self.record_frame,
            )
        except Exception:
            self._vla_desync = True
            raise
        finally:
            self._invalidate_geometry(vla_moved=True)

    def vla_act(self, prompt):
        """Run the frozen formal VLA primitive on the benchmark instruction."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise TypeError("prompt must be a non-empty string")
        observation = self.env.current_raw_obs
        if not isinstance(observation, dict):
            raise TypeError("current_raw_obs must be a dict")
        task_language = (
            observation.get("language") or self.env.get_task_language() or ""
        )
        if not isinstance(task_language, str) or not task_language.strip():
            raise RuntimeError("environment task language is unavailable")
        if prompt != task_language:
            raise ValueError("vla_act prompt must equal state.task_language verbatim")
        try:
            max_chunks = int(os.environ.get("RLDX_MAX_CHUNKS", "40"))
            settle_patience = int(os.environ.get("RLDX_SETTLE_PATIENCE", "999"))
        except ValueError as exc:
            raise ValueError(
                "RLDX_MAX_CHUNKS and RLDX_SETTLE_PATIENCE must be integers"
            ) from exc
        max_chunks = _bounded_int(max_chunks, "RLDX_MAX_CHUNKS", maximum=MAX_VLA_CHUNKS)
        settle_patience = _bounded_int(
            settle_patience,
            "RLDX_SETTLE_PATIENCE",
            maximum=MAX_VLA_CHUNKS,
        )
        result = self.run_rldx_skill(
            None,
            max_chunks,
            True,
            prompt,
            False,
            8,
            settle_patience,
            0.012,
        )
        if not self._perception_isolation:
            return result
        if not isinstance(result, dict):
            raise TypeError("VLA result must be a dict")
        return {
            "ok": bool(result.get("ok")),
            "status": result.get("status"),
            "chunks": result.get("chunks"),
            "steps_applied": result.get("steps_applied"),
            "contact_made": bool(result.get("grasp_detected")),
            "holding": bool(result.get("grasped")),
        }

    # ---- reset ----
    def reset(self):
        # reset is ONLY for EXPLORE mode (reset-based multi-attempt recipe search).
        # In no-reset / matched-scene evaluation it is a give-up-and-restart and is
        # FORBIDDEN — the policy must solve the scene in one shot like the fullshot eval.
        # Gated by RLDX_ALLOW_RESET (default 0 = disabled); explore launchers opt in.
        if not self._allow_reset:
            return {
                "error": "reset is DISABLED in this run (no-reset/matched evaluation). "
                "Solve the scene in one shot; do not restart the episode."
            }
        # EXPLORE MODE: restart the episode for a fresh attempt. New layout/
        # object placement sampled; arm/base calibration is invalidated.
        self._vla_desync = True
        self.env.reset()
        self._pos_jac = None
        self._fwd_offset = None
        self._cam_meta_cache.clear()
        self._rldx.reset_session()
        return {"ok": True, "reset": True, "eef": self.env.eef_pos.tolist()}

    # ---- VLA wrappers (public API for execute) ----
    def rldx_skill(
        self,
        base_clip=None,
        max_chunks=int(os.environ.get("RLDX_MAX_CHUNKS", 70)),
        use_prompt=None,
        prompt="",
        force_reset=False,
        n_action_steps=8,
        settle_patience=int(os.environ.get("RLDX_SETTLE_PATIENCE", 999)),
        settle_eps=0.012,
    ):
        # rldx_skill = full base motion (fullshot); base NOT clamped.
        return self.run_rldx_skill(
            base_clip,
            max_chunks,
            use_prompt,
            prompt,
            force_reset,
            n_action_steps,
            settle_patience,
            settle_eps,
        )

    def rldx_arm(
        self,
        base_clip=0.1,
        max_chunks=int(os.environ.get("RLDX_MAX_CHUNKS", 70)),
        use_prompt=None,
        prompt="",
        force_reset=False,
        n_action_steps=8,
        settle_patience=int(os.environ.get("RLDX_SETTLE_PATIENCE", 999)),
        settle_eps=0.012,
    ):
        # rldx_arm = base CLAMPED to a small range so the VLA can micro-align
        # for the grasp but can't drive away.
        return self.run_rldx_skill(
            base_clip,
            max_chunks,
            use_prompt,
            prompt,
            force_reset,
            n_action_steps,
            settle_patience,
            settle_eps,
        )

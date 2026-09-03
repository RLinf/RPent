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


class RoboCasaPrimitives:
    def __init__(self, env_client, workdir, hi_res, vla_client, check_cancelled=None):
        self.env = env_client
        self.workdir = workdir
        self.hi_res = hi_res
        # Cancellation checkpoint callback (e.g. Toolkit.raise_if_cancelled),
        # invoked before long-running env.step loops so an interrupted tool
        # operation stops promptly instead of draining the whole episode.
        self._check_cancelled = check_cancelled
        # World maps (per-pixel world xyz) are dumped EVERY step — they are how the agent
        # localizes every object, so they are never optional. Disk is bounded by keeping only
        # the last `_keep_heavy` steps' heavy npy on disk (pruned in dump_state) + each
        # launcher's per-cell workdir cleanup.
        self._keep_heavy = int(os.environ.get("RLDX_KEEP_HEAVY_NPY", "25"))
        # reset (restart the episode) is ONLY legitimate in EXPLORE mode (reset-based recipe
        # search). It is FORBIDDEN in multi-seed / matched-scene evaluation (a give-up-and-
        # restart). Gated by RLDX_ALLOW_RESET (default 0 = off); explore runs opt in.
        self._allow_reset = bool(int(os.environ.get("RLDX_ALLOW_RESET", "0")))
        os.makedirs(workdir, exist_ok=True)
        self.env.reset()
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
        return float(np.clip(g, -1, 1))

    def _resolve_grip(self, gripper, target_q):
        """Return the a[6] gripper command for a motion step.
        gripper="hold"/None (DEFAULT for moves) -> SERVO the fingers back to `target_q`,
        the width they had when the motion began. This is the carry-safe hold: the
        gripper action is a CLOSE-VELOCITY command, so a sustained +1 keeps driving the
        fingers shut and SQUEEZES a small object OUT (verified: bread qpos 0.0376 -> 0.0005
        during a +1 carry). Servoing to the grasped width holds the object without crushing
        it and without letting it drift open. A numeric gripper (+1 close / -1 open) is an
        EXPLICIT override and passes through unchanged."""
        if isinstance(gripper, str) or gripper is None:
            cur = float(self.env.gripper_qpos[0])
            return float(
                np.clip(60.0 * (cur - target_q), -1.0, 1.0)
            )  # +a[6] closes (qpos↓)
        return self._hold_gripper_val(gripper)

    def _step_arm(self, dpos=(0, 0, 0), drot=(0, 0, 0), gripper=-1.0, n=1):
        a = self._zero(base_mode=-1.0)
        a[0:3] = np.clip(np.asarray(dpos) / OSC_POS_SCALE, -1, 1)
        a[3:6] = np.clip(np.asarray(drot) / OSC_ROT_SCALE, -1, 1)
        a[6] = self._hold_gripper_val(gripper)
        for _ in range(n):
            if self._check_cancelled is not None:
                self._check_cancelled()
            self.env.step(a)
            if self._recording:
                self.record_frame()
        return self.env.eef_pos

    # ---- Phase 2: online jacobian + move_to ----
    def _calibrate_pos_jacobian(self, gripper=-1.0):
        """Probe 3 unit arm-xyz actions, measure world dpos -> 3x3 jacobian J s.t.
        world_dpos ~= J @ action_xyz. move_to inverts J to map desired world delta."""
        cols = []
        for axis in range(3):
            p0 = self.env.eef_pos.copy()
            a = self._zero()
            a[axis] = 0.4
            a[6] = gripper
            for _ in range(3):
                if self._check_cancelled is not None:
                    self._check_cancelled()
                self.env.step(a)
            d = (self.env.eef_pos - p0) / (0.4 * 3)  # world dpos per unit action
            cols.append(d)
            # settle back is not needed (closed-loop re-reads); keep going
        self._pos_jac = np.stack(cols, axis=1)  # 3x3: world_dpos = J @ a_xyz
        return self._pos_jac

    def move_to(self, xyz, gripper="hold", step_clip=0.02, max_steps=200, tol=0.012):
        """Closed-loop OSC servo of the eef to a WORLD xyz target.
        gripper="hold" (DEFAULT) maintains the CURRENT finger width — carry a grasped
        object WITHOUT crushing it (a sustained +1 squeezes small objects out) or letting
        it drop. Pass +1 to actively CLOSE, -1 to actively OPEN."""
        self._vla_desync = True
        target = np.asarray(xyz, dtype=np.float64)
        target_q = float(self.env.gripper_qpos[0])  # finger width to hold
        if self._pos_jac is None:
            self._calibrate_pos_jacobian(gripper=self._resolve_grip(gripper, target_q))
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
            if self._check_cancelled is not None:
                self._check_cancelled()
            self.env.step(a)
            if self._recording:
                self.record_frame()
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
        return self.move_to(
            self.env.eef_pos + np.asarray(dxyz), gripper, step_clip, max_steps
        )

    def rotate_pitch(self, target_pitch=0.6, gripper=1, n=12):
        """Tilt the wrist forward (axis-angle about control-x). Reuses arm drot."""
        self._vla_desync = True
        per = float(np.clip(target_pitch, -1.5, 1.5)) / n
        for _ in range(n):
            self._step_arm(drot=(per, 0, 0), gripper=gripper, n=1)
        return {"ok": True, "eef": self.env.eef_pos.tolist()}

    def set_gripper(self, gripper=1, steps=10):
        self._vla_desync = True
        g = self._hold_gripper_val(gripper)
        a = self._zero()
        a[6] = g
        for _ in range(steps):
            if self._check_cancelled is not None:
                self._check_cancelled()
            self.env.step(a)
        return {"ok": True, "gripper_qpos": self.env.gripper_qpos.tolist()}

    def release(self, steps=10):
        return self.set_gripper(-1.0, steps=steps)

    # ---- Phase 4: scripted grasp ----
    def scripted_grasp(self, xyz, approach_z=0.10, grasp_z_offset=0.0, step_clip=0.02):
        """Open -> hover above target -> descend -> close -> lift. Coarse; replace
        with RLDX closed-loop grasp for hard objects."""
        t = np.asarray(xyz, dtype=np.float64)
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
            np.asarray(o["robot0_base_pos"], dtype=np.float64),
            np.asarray(o["robot0_base_quat"], dtype=np.float64),
        )

    def move_base(self, forward=0, lateral=0, turn=0, steps=10, gripper="hold"):
        """Raw base velocity command (robot-local: +fwd, +lateral, +turn yaw).
        gripper="hold" (DEFAULT) maintains the finger width while driving (carry-safe)."""
        self._vla_desync = True
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
            if self._check_cancelled is not None:
                self._check_cancelled()
            self.env.step(a)
        bp1, _ = self._base_pose()
        return {
            "ok": True,
            "base_moved": (bp1 - bp0).tolist(),
            "base_pos": bp1.tolist(),
        }

    def _yaw(self):
        from scipy.spatial.transform import Rotation as R

        return float(
            R.from_quat(
                np.asarray(self.env.current_raw_obs["robot0_base_quat"])
            ).as_euler("xyz")[2]
        )

    def _calibrate_forward(self, gripper=1.0):
        """Drive forward briefly, measure the WORLD direction the base actually goes,
        so navigate_to can steer regardless of the base->world frame offset."""
        p0, _ = self._base_pose()
        y0 = self._yaw()
        a = self._zero(base_mode=1.0)
        a[6] = self._hold_gripper_val(gripper)
        a[7] = 1.0
        for _ in range(6):
            if self._check_cancelled is not None:
                self._check_cancelled()
            self.env.step(a)
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
        target = np.asarray(xy[:2], dtype=np.float64)
        target_q = float(self.env.gripper_qpos[0])
        if self._fwd_offset is None:
            self._calibrate_forward(self._resolve_grip(gripper, target_q))
        start = self._base_pose()[0][:2].copy()
        for i in range(max_steps):
            bp, _ = self._base_pose()
            to = target - bp[:2]
            dist = float(np.linalg.norm(to))
            if dist < tol:
                self._pos_jac = None  # base moved -> recalibrate arm
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
            if self._check_cancelled is not None:
                self._check_cancelled()
            self.env.step(a)
        bp, _ = self._base_pose()
        self._pos_jac = None
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
        return {
            "task_language": o.get("language", self.env.get_task_language()) or "",
            "success": self.env.check_success(),
            # NUMERIC progress toward success (counters/sub-predicates the env's own
            # _check_success computes) so the agent has a feedback loop, not just a bool.
            "task_progress": self.task_progress(),
            "robocasa_terminated": self.env.terminated,
            "state": {
                "robot0_eef_pos": self.env.eef_pos.tolist(),
                "robot0_eef_quat": self.env.eef_quat.tolist(),
                "robot0_gripper_qpos": self.env.gripper_qpos.tolist(),
                "robot0_base_pos": np.asarray(o["robot0_base_pos"]).tolist(),
                "robot0_base_quat": np.asarray(o["robot0_base_quat"]).tolist(),
            },
        }

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
        """Execute RLDX with the environment's live, full task language."""
        del use_prompt  # Accepted for compatibility with historical task recipes.
        task_lang = (
            self.env.current_raw_obs.get("language") or self.env.get_task_language()
        )
        if not task_lang:
            return {
                "error": "RoboCasa task language is unavailable; VLA was not executed",
                "effective_prompt": "",
                "prompt_overridden": False,
            }

        prompt_overridden = prompt != task_lang
        # Auto-reseed history if a non-VLA primitive ran since the last VLA call
        # (read _vla_desync BEFORE clearing it)
        fr = bool(force_reset) or self._vla_desync
        self._vla_desync = False
        result = self._rldx.run(
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
        result["effective_prompt"] = task_lang
        result["prompt_overridden"] = prompt_overridden
        if prompt_overridden:
            result["requested_prompt"] = prompt
        return result

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

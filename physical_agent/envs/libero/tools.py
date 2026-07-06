"""LIBERO + OpenPI tool implementation."""
from __future__ import annotations

import base64
import json
import os

import imageio.v2 as imageio
import numpy as np

from physical_agent.rpc_driver.vla_client import VLAClient
from physical_agent.envs.libero.libero_env_client import LiberoEnvClient
from physical_agent.utils.logging import get_logger, get_output_dir

logger = get_logger("libero")


def _normalize_xyz(xyz):
    """Coerce an LLM-supplied xyz into a length-3 list[float]."""
    if isinstance(xyz, dict) and set(xyz) == {"item"}:
        xyz = xyz["item"]
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        raise ValueError(
            'xyz must be a JSON array of three numbers, e.g. "xyz":[-0.05,0,0.3]'
        )
    return [float(v) for v in xyz]


class LiberoPrimitives:
    """Wraps a single-env LIBERO-shaped env + VLA policy with primitive-
    level methods.

    ``pick`` and ``place`` override ``obs['task_descriptions']`` with a
    sub-instruction then run a fixed-length action chunk loop until a
    termination predicate fires. ``move_to`` and friends are scripted
    (no VLM call) and drive the underlying OSC controller directly.
    """

    def __init__(
        self,
        env: LiberoEnvClient,
        model: VLAClient,
    ):
        self.env = env
        self.model = model
        self._last_obs = None
        self._last_obs_eef_pos = None
        self._last_obs_eef_z = None
        self._last_obs_gripper = None
        # Per-env-step frame buffer for diagnostic video rendering.
        # Toggled via start_recording() / stop_recording_and_save().
        self._recording = False
        self._frames = []

    def start_recording(self):
        self._recording = True
        self._frames = []

    def record_frame(self):
        """Snapshot the current agentview to the frame buffer, if recording."""
        if self._recording:
            img = self.env.render_camera(
                camera_name="agentview",
                height=256,
                width=256,
                depth=False,
            )
            img = np.asarray(img)
            self._frames.append(np.ascontiguousarray(img[::-1, ::-1]))

    def recorded_frame_count(self) -> int:
        return len(self._frames)

    def stop_recording_and_save(self, path: str, fps: int = 20,
                                 keep_recording: bool = False):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        n = len(self._frames)
        if n > 0:
            imageio.mimwrite(path, self._frames, fps=fps)
        if not keep_recording:
            self._recording = False
            self._frames = []
        return {"path": path, "n_frames": n}

    def save_frame_slice(self, start: int, path: str, fps: int = 20):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        frames = list(self._frames[int(start):])
        n = len(frames)
        if n > 0:
            imageio.mimwrite(path, frames, fps=fps)
        return {"path": path, "n_frames": n, "fps": fps}

    def set_obs(self, obs):
        self._last_obs = obs
        states_arr = np.asarray(obs["states"])
        self._last_obs_eef_pos = np.asarray(states_arr[:3], dtype=np.float32)
        self._last_obs_eef_z = float(self._last_obs_eef_pos[2])
        # robosuite 2f85: qpos[6] in [~0, ~0.04], qpos[7] in [~-0.04, ~0].
        # Use |qpos[6]| + |qpos[7]| ≈ finger separation proxy.
        # When open ≈ 0.08; when closed ≈ 0.
        gp = np.asarray(states_arr[6:8], dtype=np.float32)
        self._last_obs_gripper = float(abs(gp[0]) + abs(gp[1]))

    def reset(self):
        obs, info = self.env.reset()
        self.set_obs(obs)
        return self._last_obs, info

    def _vlm_chunk(self, instruction: str):
        """One model forward + ``chunk_size`` env steps. Overrides prompt."""
        # Stash & override task_descriptions (one prompt).
        original_td = self._last_obs.get("task_descriptions")
        self._last_obs["task_descriptions"] = instruction
        self._last_obs.setdefault("extra_view_images", None)

        actions, _ = self.model.predict_action_batch(self._last_obs, mode="eval")
        # actions: [chunk_size, action_dim] The whole chunk
        # runs in a single env.chunk_step RPC; the env owns the per-step
        # loop server-side.
        chunk_obs,  _r, _t, _tr, _i = self.env.chunk_step(actions)
        obs = chunk_obs[-1] if self.env.return_all_frames else chunk_obs
        self.set_obs(obs)
        # One frame per chunk (matches the previous chunk_step cadence).
        self.record_frame()
        # Restore original task_descriptions on the obs dict for fairness
        # with future steps (no leaked state if caller switches primitives).
        if original_td is not None:
            self._last_obs["task_descriptions"] = original_td
        return self._last_obs

    def pi0_pick(
        self,
        prompt: str,
        *,
        max_chunks: int = 30,
        lift_thresh: float = 0.05,
        gripper_closed_thresh: float = 0.06,
    ) -> dict:
        """Closed-loop Pi0.5 pick driven by ``prompt`` as the VLA instruction.

        Success := eef lifted by >= ``lift_thresh`` AND gripper_opening
        below ``gripper_closed_thresh``. Terminates early on libero
        ``terminated`` (official success) or ``max_chunks``.
        """
        instr = prompt
        start_z = self._last_obs_eef_z
        peak_z = start_z
        min_z = start_z
        # Track ascent AFTER min_z has been observed — descent then re-ascent
        # is the actual "lift" signal, distinct from raw |peak - min| which
        # also fires at the BOTTOM of the descent.
        post_min_peak_z = start_z
        min_grip = self._last_obs_gripper
        last_grip = min_grip
        descent_done = False
        success = False
        chunks_used = 0

        for c in range(max_chunks):
            self._vlm_chunk(instr)
            chunks_used = c + 1
            z = self._last_obs_eef_z
            grip = self._last_obs_gripper
            peak_z = max(peak_z, z)
            if z < min_z:
                min_z = z
                post_min_peak_z = z  # reset after a new deeper min
            else:
                post_min_peak_z = max(post_min_peak_z, z)
            if (start_z - min_z) >= 0.10:  # descended ≥ 10 cm — committed to grasp
                descent_done = True
            min_grip = min(min_grip, grip)
            last_grip = grip
            ascended = (post_min_peak_z - min_z) >= lift_thresh
            closed = grip < gripper_closed_thresh
            if descent_done and ascended and closed:
                success = True
                break
            if self.env.episode_done:
                success = True
                break

        return {
            "name": "pick",
            "instruction": instr,
            "success": success,
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "peak_lift_m": post_min_peak_z - min_z,  # actual post-descent ascent
            "min_gripper_opening": min_grip,
            "final_gripper_opening": last_grip,
            "libero_terminated": self.env.episode_done,
            "diagnostics": {
                "start_eef_z": round(start_z, 4),
                "peak_eef_z": round(peak_z, 4),
                "min_eef_z": round(min_z, 4),
                "post_min_peak_z": round(post_min_peak_z, 4),
                "descent_m": round(start_z - min_z, 4),
                "post_min_ascent_m": round(post_min_peak_z - min_z, 4),
                "descent_done": descent_done,
                "lift_thresh": lift_thresh,
                "gripper_closed_thresh": gripper_closed_thresh,
            },
        }

    def pi0_doubled(
        self,
        prompt: str,
        *,
        max_chunks: int = 20,
    ) -> dict:
        """Closed-loop Pi0.5 contact skill.

        Intended for non-pick contact interactions such as turning knobs,
        toggling stoves, or short pushes. Success is the official LIBERO
        termination predicate, not a private object-pose oracle.
        """
        instr = prompt
        task_success = False
        chunks_used = 0

        for c in range(max_chunks):
            self._vlm_chunk(instr)
            chunks_used = c + 1
            if self.env.episode_done:
                task_success = True
                break

        return {
            "name": "pi0_doubled",
            "instruction": instr,
            "success": task_success,
            "task_success": task_success,
            "contact_skill_executed": chunks_used > 0,
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "libero_terminated": self.env.episode_done,
            "diagnostics": {
                "mode": "contact_skill_success_by_libero_terminated",
                "success_meaning": (
                    "`success` mirrors official LIBERO task termination only; "
                    "for intermediate contact skills, inspect image/state evidence."
                ),
            },
        }

    def place(
        self,
        target_text: str,
        *,
        max_chunks: int = 24,
        release_thresh: float = 0.04,
        instruction_template: str = "place it on {tgt}",
    ) -> dict:
        """Run VLA with the place sub-instruction until gripper opens or budget."""
        instr = instruction_template.format(tgt=target_text)
        start_z = self._last_obs_eef_z
        peak_z = start_z
        min_grip = self._last_obs_gripper
        last_grip = min_grip
        success = False
        chunks_used = 0

        for c in range(max_chunks):
            self._vlm_chunk(instr)
            chunks_used = c + 1
            z = self._last_obs_eef_z
            grip = self._last_obs_gripper
            peak_z = max(peak_z, z)
            min_grip = min(min_grip, grip)
            last_grip = grip
            if grip >= release_thresh:
                success = True
                break
            if self.env.episode_done:
                success = True
                break

        return {
            "name": "place",
            "instruction": instr,
            "success": success,
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "peak_lift_m": peak_z - start_z,
            "min_gripper_opening": min_grip,
            "final_gripper_opening": last_grip,
            "libero_terminated": self.env.episode_done,
            "diagnostics": {"release_thresh": release_thresh},
        }

    def move_to(
        self,
        xyz,
        *,
        max_steps: int = 80,
        gripper: float = -1.0,
        step_clip: float = 0.025,
        tol: float = 0.012,
        action_scale: float = 0.05,
        target_yaw: float | None = None,
        yaw_step_clip: float = 0.10,
    ) -> dict:
        """Scripted EEF servo to a world-frame target xyz.

        Sends 7-D delta actions; the env's underlying OSC_POSE controller
        interprets ``action[:3] ∈ [-1, 1]`` as a per-step desired delta scaled
        by ``action_scale`` (so ``action=1.0`` -> ~5 cm per env step).
        ``gripper``: +1.0 keeps it closed (holding object), -1.0 opens.
        """
        target = np.asarray(_normalize_xyz(xyz), dtype=np.float32)
        traj = []
        for step in range(max_steps):
            cur = self._last_obs_eef_pos
            diff = target - cur
            dist = float(np.linalg.norm(diff))
            traj.append({
                "step": step,
                "eef_pos": [round(float(x), 4) for x in cur],
                "dist_to_target_m": round(dist, 4),
            })
            if dist < tol:
                break
            step_dxyz = np.clip(diff, -step_clip, step_clip)
            action = np.zeros(7, dtype=np.float32)
            action[:3] = step_dxyz / action_scale  # -> roughly [-0.5, 0.5]
            action[:3] = np.clip(action[:3], -1.0, 1.0)
            if target_yaw is not None:
                # add wrist yaw control via action[5] (z-axis axis-angle).
                # NOTE: extract world yaw via atan2(R[1,0], R[0,0]), NOT
                # as_euler('zyx')[0] — the latter returns -world_yaw for
                # gripper-down configs (R[2,2]≈-1) and silently flips the
                # commanded rotation direction. See feedback_rotate_wrist_yaw_sign.
                from scipy.spatial.transform import Rotation as _R
                q = self.env.raw_obs()["robot0_eef_quat"]
                _R_mat = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
                cur_yaw = float(np.arctan2(_R_mat[1, 0], _R_mat[0, 0]))
                err = (float(target_yaw) - cur_yaw + np.pi) % (2 * np.pi) - np.pi
                step_dyaw = float(np.clip(err, -yaw_step_clip, yaw_step_clip))
                action[5] = float(np.clip(step_dyaw / 0.10, -1.0, 1.0))
            action[6] = gripper
            obs, _r, _t, _tr, _i = self.env.step(action)
            self.set_obs(obs)
            self.record_frame()
            if self.env.episode_done:
                break
        final = self._last_obs_eef_pos
        return {
            "name": "move_to",
            "target_xyz": [float(x) for x in target],
            "final_eef_pos": [round(float(x), 4) for x in final],
            "final_dist_m": round(float(np.linalg.norm(target - final)), 4),
            "steps_used": len(traj),
            "max_steps": max_steps,
            "libero_terminated": self.env.episode_done,
        }

    def rotate_wrist(
        self,
        *,
        target_yaw: float | None = None,
        delta_yaw: float | None = None,
        gripper: float = 1.0,
        max_steps: int = 40,
        tol: float = 0.02,
        step_clip: float = 0.10,
    ) -> dict:
        """Rotate wrist around world z-axis. Provide EITHER target_yaw (absolute)
        or delta_yaw (relative, applied as a single rotation goal).

        Uses ``action[5]`` (axis-angle z component) to drive wrist yaw via the
        OSC controller. Holds xyz pose constant during rotation.

        Yaw is the world-frame z-rotation, recovered as
        ``atan2(R[1,0], R[0,0])`` where R is the eef rotation matrix in the
        world frame. (Note: ``as_euler('zyx')[0]`` returns the *negative*
        of this value for gripper-down configurations because the Z-Y-X
        decomposition picks the chart with γ ≈ π, flipping α. Bug fixed
        2026-05-19 — previous implementation rotated the wrist in the
        opposite direction of the commanded yaw.)
        """
        from scipy.spatial.transform import Rotation as _R

        def _yaw_of(quat_xyzw):
            # robot0_eef_quat in libero+robosuite is xyzw (scipy convention).
            q = quat_xyzw
            rot = _R.from_quat([q[0], q[1], q[2], q[3]])
            R = rot.as_matrix()
            # World-frame yaw: angle of the eef x-axis projected onto the
            # world xy plane. Robust to gripper-down (R[2,2]≈-1) which is
            # where the euler 'zyx' chart flips sign.
            return float(np.arctan2(R[1, 0], R[0, 0]))

        raw = self.env.raw_obs()
        cur_quat = raw["robot0_eef_quat"]
        start_yaw = _yaw_of(cur_quat)
        if target_yaw is None and delta_yaw is None:
            return {"name": "rotate_wrist", "error": "need target_yaw or delta_yaw"}
        if target_yaw is None:
            target_yaw = start_yaw + float(delta_yaw)

        traj = []
        for step in range(max_steps):
            raw = self.env.raw_obs()
            cur_yaw = _yaw_of(raw["robot0_eef_quat"])
            err = float(target_yaw - cur_yaw)
            # wrap to [-pi, pi]
            err = (err + np.pi) % (2 * np.pi) - np.pi
            traj.append({"step": step, "yaw": round(cur_yaw, 4), "err": round(err, 4)})
            if abs(err) < tol:
                break
            step_dyaw = float(np.clip(err, -step_clip, step_clip))
            action = np.zeros(7, dtype=np.float32)
            action[5] = step_dyaw / 0.10  # scale to ~[-1,1] action range
            action[5] = float(np.clip(action[5], -1.0, 1.0))
            action[6] = float(gripper)
            obs, _r, _t, _tr, _i = self.env.step(action)
            self.set_obs(obs)
            self.record_frame()
            if self.env.episode_done:
                break
        final_yaw = _yaw_of(self.env.raw_obs()["robot0_eef_quat"])
        return {
            "name": "rotate_wrist",
            "start_yaw": round(start_yaw, 4),
            "target_yaw": round(float(target_yaw), 4),
            "final_yaw": round(final_yaw, 4),
            "final_err": round(float((target_yaw - final_yaw + np.pi) % (2 * np.pi) - np.pi), 4),
            "steps_used": len(traj),
            "libero_terminated": self.env.episode_done,
        }

    def rotate_pitch(
        self,
        *,
        target_pitch: float | None = None,
        delta_pitch: float | None = None,
        gripper: float = 1.0,
        max_steps: int = 40,
        tol: float = 0.02,
        step_clip: float = 0.10,
    ) -> dict:
        """Tilt the gripper around the world X-axis ("pitch").

        Pitch is defined as the angle between the eef z-axis and the
        world -z direction, measured in the world yz-plane:

            pitch = atan2(R[1, 2], -R[2, 2])

        - pitch =  0       -> gripper z-axis aligned with world -z (default
                              "gripper down" rest pose).
        - pitch = +pi/2    -> gripper z-axis points in world +y (gripper
                              "looking forward" along world +y).
        - pitch = -pi/2    -> gripper z-axis points in world -y.

        Driven by ``action[3]`` (axis-angle X component) of the OSC_POSE
        controller. Sign verified empirically (probe_pitch.py 2026-05-19):
        action[3]=+1.0 tilts eef z toward world +y, matching this pitch
        definition with no sign flip.

        Holds xyz, yaw, and gripper constant during rotation. Use BEFORE
        threading the gripper into a narrow opening whose front face
        normal is along world ±y (e.g. microwave cavity in libero_10 t9).

        Provide EITHER ``target_pitch`` (absolute) or ``delta_pitch``
        (relative). Both in radians.
        """
        from scipy.spatial.transform import Rotation as _R

        def _pitch_of(quat_xyzw):
            q = quat_xyzw
            R = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
            return float(np.arctan2(R[1, 2], -R[2, 2]))

        raw = self.env.raw_obs()
        start_pitch = _pitch_of(raw["robot0_eef_quat"])
        if target_pitch is None and delta_pitch is None:
            return {"name": "rotate_pitch",
                    "error": "need target_pitch or delta_pitch"}
        if target_pitch is None:
            target_pitch = start_pitch + float(delta_pitch)

        traj = []
        for step in range(max_steps):
            raw = self.env.raw_obs()
            cur_pitch = _pitch_of(raw["robot0_eef_quat"])
            err = float(target_pitch - cur_pitch)
            err = (err + np.pi) % (2 * np.pi) - np.pi
            traj.append({"step": step,
                         "pitch": round(cur_pitch, 4),
                         "err": round(err, 4)})
            if abs(err) < tol:
                break
            step_dpitch = float(np.clip(err, -step_clip, step_clip))
            action = np.zeros(7, dtype=np.float32)
            action[3] = step_dpitch / 0.10
            action[3] = float(np.clip(action[3], -1.0, 1.0))
            action[6] = float(gripper)
            obs, _r, _t, _tr, _i = self.env.step(action)
            self.set_obs(obs)
            self.record_frame()
            if self.env.episode_done:
                break
        final_pitch = _pitch_of(self.env.raw_obs()["robot0_eef_quat"])
        return {
            "name": "rotate_pitch",
            "start_pitch": round(start_pitch, 4),
            "target_pitch": round(float(target_pitch), 4),
            "final_pitch": round(final_pitch, 4),
            "final_err": round(float(
                (target_pitch - final_pitch + np.pi) % (2 * np.pi) - np.pi), 4),
            "steps_used": len(traj),
            "libero_terminated": self.env.episode_done,
        }

    def move_pose(
        self,
        xyz,
        *,
        target_pitch: float | None = None,
        target_yaw: float | None = None,
        gripper: float = -1.0,
        step_clip: float = 0.02,
        pitch_step: float = 0.08,
        yaw_step: float = 0.08,
        tol: float = 0.012,
        ori_tol: float = 0.05,
        action_scale: float = 0.05,
        max_steps: int = 150,
    ) -> dict:
        """Servo position AND orientation (pitch + yaw) SIMULTANEOUSLY.

        Unlike ``move_to`` (holds orientation) + ``rotate_pitch`` (holds
        xyz), this co-varies xyz and wrist tilt every env.step. Co-variation
        lets the OSC controller thread cabinet-front-low poses where a
        decoupled position servo (fixed gripper-down orientation) drives
        the wrist into a singularity and stalls — mimicking pi0's curved
        reach-in.
        """
        from scipy.spatial.transform import Rotation as _R

        def _pitch_of(q):
            R = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
            return float(np.arctan2(R[1, 2], -R[2, 2]))

        def _yaw_of(q):
            R = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
            return float(np.arctan2(R[1, 0], R[0, 0]))

        target = np.asarray(_normalize_xyz(xyz), dtype=np.float32)
        traj = []
        step = 0
        for step in range(max_steps):
            cur = self._last_obs_eef_pos
            q = self.env.raw_obs()["robot0_eef_quat"]
            diff = target - cur
            dist = float(np.linalg.norm(diff))
            p_err = 0.0 if target_pitch is None else \
                float((target_pitch - _pitch_of(q) + np.pi) % (2 * np.pi) - np.pi)
            y_err = 0.0 if target_yaw is None else \
                float((target_yaw - _yaw_of(q) + np.pi) % (2 * np.pi) - np.pi)
            traj.append({"step": step, "eef": [round(float(x), 4) for x in cur],
                         "dist": round(dist, 4), "p_err": round(p_err, 3)})
            if dist < tol and abs(p_err) < ori_tol and abs(y_err) < ori_tol:
                break
            action = np.zeros(7, dtype=np.float32)
            sd = np.clip(diff, -step_clip, step_clip)
            action[:3] = np.clip(sd / action_scale, -1.0, 1.0)
            action[3] = float(np.clip(np.clip(p_err, -pitch_step, pitch_step) / 0.10, -1.0, 1.0))
            action[5] = float(np.clip(np.clip(y_err, -yaw_step, yaw_step) / 0.10, -1.0, 1.0))
            action[6] = float(gripper)
            obs, _r, _t, _tr, _i = self.env.step(action)
            self.set_obs(obs)
            self.record_frame()
            if self.env.episode_done:
                break
        final = self._last_obs_eef_pos
        fq = self.env.raw_obs()["robot0_eef_quat"]
        return {
            "name": "move_pose",
            "final_eef_pos": [round(float(x), 4) for x in final],
            "final_dist_m": round(float(np.linalg.norm(target - final)), 4),
            "final_pitch": round(_pitch_of(fq), 4),
            "steps_used": step + 1,
            "libero_terminated": self.env.episode_done,
        }

    def release(
        self,
        *,
        max_steps: int = 20,
        hold_pos: bool = True,
    ) -> dict:
        """Open gripper for ``max_steps`` env steps, optionally keeping eef in place.

        Returns once libero terminates (success) or step budget exhausted.
        """
        start_grip = self._last_obs_gripper
        peak_grip = start_grip
        for step in range(max_steps):
            action = np.zeros(7, dtype=np.float32)
            action[6] = -1.0  # open
            obs, _r, _t, _tr, _i = self.env.step(action)
            self.set_obs(obs)
            self.record_frame()
            peak_grip = max(peak_grip, self._last_obs_gripper)
            if self.env.episode_done:
                break
        return {
            "name": "release",
            "steps_used": step + 1,
            "start_gripper_opening": round(start_grip, 4),
            "peak_gripper_opening": round(peak_grip, 4),
            "final_gripper_opening": round(self._last_obs_gripper, 4),
            "libero_terminated": self.env.episode_done,
        }

    def set_gripper(
        self,
        *,
        gripper: float = -1.0,
        steps: int = 5,
    ) -> dict:
        """Hold the current EEF pose and drive ``gripper`` for ``steps`` env steps."""
        g = float(gripper)
        n = int(steps)
        for _ in range(n):
            action = np.zeros(7, dtype=np.float32)
            action[6] = g
            obs, _r, _t, _tr, _i = self.env.step(action)
            self.set_obs(obs)
            self.record_frame()
            if self.env.episode_done:
                break
        return {
            "name": "set_gripper",
            "gripper": g,
            "steps": n,
            "libero_terminated": self.env.episode_done,
        }

    # ---- introspection helpers (for LLM-in-the-loop) ----

    def get_privileged_state(self) -> dict:
        """Pull world-frame positions of EEF + all named objects from raw_obs.

        These keys come straight from libero's robosuite observables (e.g.
        ``plate_1_pos``, ``akita_black_bowl_1_pos``). Use them as 'ground
        truth' for LLM-in-the-loop planning or post-hoc evaluation.
        """
        raw = self.env.raw_obs()
        out = {
            "robot0_eef_pos": [float(x) for x in raw["robot0_eef_pos"]],
            "robot0_eef_quat": [float(x) for x in raw["robot0_eef_quat"]],
            "robot0_gripper_qpos": [float(x) for x in raw["robot0_gripper_qpos"]],
            "objects": {},
            "obj_of_interest": None,
        }
        for k, v in raw.items():
            if k.endswith("_pos") and "robot0" not in k and "to_robot" not in k:
                obj_name = k[:-4]
                out["objects"][obj_name] = [float(x) for x in v]
        return out

    # ---- full task baseline (no instruction override) ----

    def run_full_task(
        self,
        *,
        max_chunks: int = 48,
    ) -> dict:
        """Just run the VLA with the ORIGINAL ``task_descriptions``, no override.

        Used as the baseline (Pi0.5 in its natural prompting mode).
        """
        chunks_used = 0
        start_z = self._last_obs_eef_z
        peak_z = start_z
        for c in range(max_chunks):
            # _vlm_chunk overrides prompt; but we want the ORIGINAL. Bypass.
            self._last_obs.setdefault("extra_view_images", None)
            actions, _ = self.model.predict_action_batch(self._last_obs, mode="eval")
            chunk_obs,  _r, _t, _tr, _i = self.env.chunk_step(actions)
            obs = chunk_obs[-1] if self.env.return_all_frames else chunk_obs
            self.set_obs(obs)
            chunks_used = c + 1
            peak_z = max(peak_z, self._last_obs_eef_z)
            if self.env.episode_done:
                break
        return {
            "name": "full_task",
            "instruction": str(self._last_obs.get("task_descriptions") or ""),
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "peak_lift_m": peak_z - start_z,
            "final_gripper_opening": self._last_obs_gripper,
            "libero_terminated": self.env.episode_done,
        }


# ---------------------------------------------------------------------------
# State artifacts
# ---------------------------------------------------------------------------


def _append_state(output_dir: str, blob: dict) -> None:
    """Append *blob* to ``<output_dir>/states.json`` atomically.

    The merged trace is a top-level JSON array (one entry per step). The
    file is rewritten via a tmp + rename so a reader never sees partial
    content. The entry index equals ``blob['step_idx']``.
    """
    path = os.path.join(output_dir, "states.json")
    tmp = path + ".tmp"
    if os.path.exists(path):
        try:
            with open(path) as f:
                arr = json.load(f)
            if not isinstance(arr, list):
                arr = []
        except Exception:
            arr = []
    else:
        arr = []
    idx = int(blob.get("step_idx", len(arr)))
    # Pad with None if the agent ever skips a step (shouldn't happen,
    # but keeps array index == step_idx).
    while len(arr) < idx:
        arr.append(None)
    if len(arr) == idx:
        arr.append(blob)
    else:
        arr[idx] = blob
    with open(tmp, "w") as f:
        json.dump(arr, f, indent=2)
    os.replace(tmp, path)


def write_recipe_from_states(output_dir: str, recipe_tag: str) -> str:
    """Find a command sequence that gets ``libero_terminated=True``.

    Export only non-error LIBERO primitive commands (move_to, pi0_pick,
    release, set_gripper, rotate_*, move_pose). Exclude inspection/file/
    finalization tools such as Read, Bash, back_project, view_driver_state,
    write_text_file, and finish.
    """
    states_path = os.path.join(output_dir, "states.json")
    states = json.load(open(states_path)) if os.path.exists(states_path) else []
    primitive_actions = {
        "move_to",
        "pi0_pick",
        "pi0_doubled",
        "release",
        "set_gripper",
        "rotate_wrist",
        "rotate_pitch",
        "move_pose",
    }

    commands = []
    for entry in states:
        if not entry:
            continue
        command = entry.get("command")
        if command is None:
            continue
        if command.get("action") not in primitive_actions:
            continue
        result = entry.get("result")
        if isinstance(result, dict) and result.get("error"):
            continue
        commands.append(command)

    recipe_path = os.path.join(output_dir, f"recipe_{recipe_tag}.jsonl")
    tmp_path = recipe_path + ".tmp"
    with open(tmp_path, "w") as f:
        for command in commands:
            f.write(json.dumps(command, separators=(",", ":")) + "\n")
    os.replace(tmp_path, recipe_path)
    return recipe_path


def _metric_depth(depth: Any, camera_meta: dict) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    near = camera_meta.get("depth_near")
    far = camera_meta.get("depth_far")
    if near is not None and far is not None:
        d = near / (1.0 - d * (1.0 - near / far))
    return d


def _world_from_depth(depth_metric: np.ndarray, camera_meta: dict) -> np.ndarray:
    k_matrix = np.array(camera_meta["intrinsic_K"], dtype=np.float64)
    extrinsic = np.array(camera_meta["extrinsic_cam2world"], dtype=np.float64)
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    height, width = depth_metric.shape
    rr, cc = np.mgrid[0:height, 0:width]
    z = depth_metric.astype(np.float64)
    camera_points = np.stack(
        [(cc - cx) * z / fx, (rr - cy) * z / fy, z, np.ones_like(z)],
        axis=-1,
    )
    return (camera_points @ extrinsic.T)[..., :3]


def dump_state(driver: LiberoPrimitives, output_dir: str, step_idx: int,
               log: dict | None = None) -> dict:
    """Dump state snapshot, images, and depth for step *step_idx*.

    Writes:
      - ``<output_dir>/images/image_NN.png``       (Pi0-frame agentview)
      - ``<output_dir>/images_cam/image_cam_NN.png`` (calibration-frame agentview)
      - ``<output_dir>/depths/depth_NN.npy``        (metric depth, meters)
      - ``<output_dir>/world/world_NN.npy``         (agentview world xyz map)
      - ``<output_dir>/images_wrist/image_wrist_NN.png``
      - ``<output_dir>/depths_wrist/depth_wrist_NN.npy``
      - ``<output_dir>/world_wrist/world_wrist_NN.npy``
      - ``<output_dir>/wrist_meta/wrist_meta_NN.json``
      - high-res ``images_cam_hi`` / ``world_hi`` artifacts
      - high-res ``images_wrist_hi`` / ``world_wrist_hi`` artifacts
      - ``<output_dir>/camera_meta.json``           (static, once)
      - appends the step blob to ``<output_dir>/states.json``

    If *log* is provided (the return value of :func:`execute`), its
    ``command``, ``result``, and ``elapsed_s`` fields are merged into the
    step blob so a single entry captures everything.
    """
    images_dir = os.path.join(output_dir, "images")
    images_cam_dir = os.path.join(output_dir, "images_cam")
    depths_dir = os.path.join(output_dir, "depths")
    world_dir = os.path.join(output_dir, "world")
    images_wrist_dir = os.path.join(output_dir, "images_wrist")
    depths_wrist_dir = os.path.join(output_dir, "depths_wrist")
    world_wrist_dir = os.path.join(output_dir, "world_wrist")
    wrist_meta_dir = os.path.join(output_dir, "wrist_meta")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(images_cam_dir, exist_ok=True)
    os.makedirs(depths_dir, exist_ok=True)
    os.makedirs(world_dir, exist_ok=True)
    os.makedirs(images_wrist_dir, exist_ok=True)
    os.makedirs(depths_wrist_dir, exist_ok=True)
    os.makedirs(world_wrist_dir, exist_ok=True)
    os.makedirs(wrist_meta_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images_cam_hi"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "world_hi"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images_wrist_hi"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "world_wrist_hi"), exist_ok=True)
    agent_world_map = None
    wrist_world_map = None
    agent_world_map_hi = None
    wrist_world_map_hi = None
    state = driver.get_privileged_state()
    # force perception: drop object world coords (the agent must
    # localize via depth_NN.npy + camera_meta.json). Keep the object NAMES
    # (what's in the scene / which is the target) + robot proprioception —
    # names are not coordinate info and are also implied by the task language.
    objs = state.get("objects", {})
    state["object_names"] = sorted(objs.keys())
    state.pop("objects", None)
    # Render live agentview and convert to the Pi0-frame artifact. Fall back
    # to the most recent valid cached frame if active rendering is unavailable.
    try:
        img = driver.env.render_camera(
            camera_name="agentview",
            height=256,
            width=256,
            depth=False,
        )
        img = np.asarray(img)
        img = np.ascontiguousarray(img[::-1, ::-1])
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3 \
                or img.shape[0] < 32 or img.shape[1] < 32:
            raise ValueError(f"bad img shape/dtype: {img.shape} {img.dtype}")
    except Exception:
        # cached_image() is already 180°-flipped (get_libero_image does
        # the flip), so just hand it through. No double-flip.
        img = driver.env.cached_image()
        if img is None:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
    imageio.imwrite(os.path.join(images_dir, f"image_{step_idx:02d}.png"), img)

    # --- camera calibration (static for agentview): fetch + dump once ---
    agentview_meta = getattr(driver, "_agentview_camera_meta", None)
    if agentview_meta is None:
        agentview_meta = driver.env.get_camera_meta(
            camera_name="agentview",
            height=256,
            width=256,
        )
        if agentview_meta is None:
            agentview_meta = {}
        driver._agentview_camera_meta = agentview_meta
        if agentview_meta:
            cam_meta_out = dict(agentview_meta)
            cam_meta_out["projection"] = (
                "Prefer the back_project(row, col, step=NN) MCP tool; it "
                "uses the 1024x1024 high-resolution world map by default. "
                "Pass resolution='low' only when row/col came from the "
                "256x256 calibration-frame image. "
                "For reference: world->pixel first computes "
                "camera_xyz = (inv(extrinsic_cam2world) @ [X,Y,Z,1])[:3], "
                "then q = K @ camera_xyz; "
                "col=q[0]/q[2], row=q[1]/q[2], metric_depth=q[2]. "
                "Back-project a pixel with z=depth_NN[row,col] by computing "
                "camera_xyz = inv(K) @ [col,row,1] * z, then "
                "P_world = extrinsic_cam2world @ [camera_xyz,1].")
            cam_meta_out["note"] = (
                "depth_NN.npy is in this camera frame (vertical-flipped raw "
                "buffer). image_NN.png is rotated 180deg (Pi0 convention) and "
                "is NOT in the same frame as depth/K.")
            with open(os.path.join(output_dir, "camera_meta.json"), "w") as f:
                json.dump(cam_meta_out, f, indent=2)

    # Fetch one raw observation snapshot for all per-step camera artifacts.
    raw = driver.env.raw_obs()

    # --- per-step RGB in the depth/K frame (vertical-flip of the raw buffer) ---
    # The agent picks object pixels HERE (same frame as depth_NN.npy + K), so
    # pixel -> depth -> back-project is direct. (image_NN.png is the 180°-rotated
    # Pi0-convention frame and must NOT be used for back-projection.)
    try:
        ci = raw.get("agentview_image")
        if ci is not None:
            ci = np.asarray(ci)
            if ci.dtype != np.uint8:
                ci = ci.astype(np.uint8)
            imageio.imwrite(
                os.path.join(images_cam_dir, f"image_cam_{step_idx:02d}.png"),
                ci[::-1],
            )
    except Exception as e:
        logger.warning("image_cam dump failed: %s", e)

    # --- per-step metric depth (agentview), native orientation, in meters ---
    try:
        d = raw.get("agentview_depth")
        if d is not None:
            # Vertical flip to align with the camera matrices: robosuite's
            # camera_utils projection M = K_exp @ inv(extrinsic) expects the
            # depth map in this frame. VERIFIED 5/5: projecting each GT object
            # world pos via M lands on a pixel whose depth_flip[row,col] matches
            # the object's surface depth (plate Δ6mm, cookies Δ14mm). So
            # pixel(row,col) in depth_NN.npy back-projects correctly with
            # camera_meta.json (NOT the same frame as the 180°-rotated
            # image_NN.png — see camera_meta note).
            d = _metric_depth(d, agentview_meta)[::-1]
            np.save(os.path.join(depths_dir, f"depth_{step_idx:02d}.npy"),
                    d.astype(np.float32))
            world = _world_from_depth(d, agentview_meta).astype(np.float32)
            world_name = f"world_{step_idx:02d}.npy"
            np.save(os.path.join(world_dir, world_name), world)
            agent_world_map = f"world/{world_name}"
    except Exception as e:
        logger.warning("depth dump failed: %s", e)

    # --- per-step wrist camera (robot0_eye_in_hand), calibration frame ---
    try:
        wimg = raw.get("robot0_eye_in_hand_image")
        if wimg is None:
            logger.warning("wrist image missing from raw_obs")
        else:
            wimg = np.asarray(wimg)
            if wimg.dtype != np.uint8:
                wimg = wimg.astype(np.uint8)
            imageio.imwrite(
                os.path.join(images_wrist_dir, f"image_wrist_{step_idx:02d}.png"),
                wimg[::-1],
            )
    except Exception as e:
        logger.warning("wrist image dump failed: %s", e)

    try:
        wdpt = raw.get("robot0_eye_in_hand_depth")
        if wdpt is None:
            logger.warning("wrist depth missing from raw_obs")
        else:
            wdpt_arr = np.asarray(wdpt, dtype=np.float32)
            height, width = wdpt_arr.shape[:2]
            wmeta = driver.env.get_camera_meta(
                camera_name="robot0_eye_in_hand",
                height=int(height),
                width=int(width),
            )
            if wmeta is None:
                logger.warning("wrist camera meta missing; skipping wrist depth/world")
            else:
                wdpt_metric = _metric_depth(wdpt_arr, wmeta)[::-1]
                depth_name = f"depth_wrist_{step_idx:02d}.npy"
                np.save(
                    os.path.join(depths_wrist_dir, depth_name),
                    wdpt_metric.astype(np.float32),
                )
                world_w = _world_from_depth(wdpt_metric, wmeta).astype(np.float32)
                world_name = f"world_wrist_{step_idx:02d}.npy"
                np.save(os.path.join(world_wrist_dir, world_name), world_w)
                wrist_world_map = f"world_wrist/{world_name}"

                wmeta_out = dict(wmeta)
                wmeta_out["note"] = (
                    "MOVING camera: extrinsic_cam2world is for THIS step "
                    "only. world_wrist_NN.npy[row,col] gives world "
                    "(x,y,z) for that pixel, in the SAME world frame as "
                    "agentview world_NN.npy."
                )
                with open(
                    os.path.join(wrist_meta_dir, f"wrist_meta_{step_idx:02d}.json"),
                    "w",
                ) as f:
                    json.dump(wmeta_out, f, indent=2)
    except Exception as e:
        logger.warning("wrist depth/world dump failed: %s", e)

    try:
        rgb_hi, depth_hi = driver.env.render_camera(
            camera_name="agentview",
            height=1024,
            width=1024,
            depth=True,
        )
        meta_hi = driver.env.get_camera_meta("agentview", 1024, 1024)
        if meta_hi is None:
            raise RuntimeError("agentview camera metadata missing")
        imageio.imwrite(
            os.path.join(
                output_dir,
                "images_cam_hi",
                f"image_cam_hi_{step_idx:02d}.png",
            ),
            np.asarray(rgb_hi)[::-1],
        )
        world_hi = _world_from_depth(
            _metric_depth(depth_hi, meta_hi)[::-1],
            meta_hi,
        ).astype(np.float16)
        world_name = f"world_hi_{step_idx:02d}.npy"
        np.save(os.path.join(output_dir, "world_hi", world_name), world_hi)
        agent_world_map_hi = f"world_hi/{world_name}"
    except Exception as e:
        logger.warning("agentview high-res dump failed: %s", e)

    try:
        rgb_wrist_hi, depth_wrist_hi = driver.env.render_camera(
            camera_name="robot0_eye_in_hand",
            height=1024,
            width=1024,
            depth=True,
        )
        meta_wrist_hi = driver.env.get_camera_meta(
            "robot0_eye_in_hand", 1024, 1024
        )
        if meta_wrist_hi is None:
            raise RuntimeError("robot0_eye_in_hand camera metadata missing")
        imageio.imwrite(
            os.path.join(
                output_dir,
                "images_wrist_hi",
                f"image_wrist_hi_{step_idx:02d}.png",
            ),
            np.asarray(rgb_wrist_hi)[::-1],
        )
        world_wrist_hi = _world_from_depth(
            _metric_depth(depth_wrist_hi, meta_wrist_hi)[::-1],
            meta_wrist_hi,
        ).astype(np.float16)
        world_name = f"world_wrist_hi_{step_idx:02d}.npy"
        np.save(
            os.path.join(output_dir, "world_wrist_hi", world_name),
            world_wrist_hi,
        )
        wrist_world_map_hi = f"world_wrist_hi/{world_name}"
    except Exception as e:
        logger.warning("wrist high-res dump failed: %s", e)

    for old_step in range(max(0, int(step_idx) - 4)):
        for path in (
            os.path.join(
                output_dir,
                "images_cam_hi",
                f"image_cam_hi_{old_step:02d}.png",
            ),
            os.path.join(output_dir, "world_hi", f"world_hi_{old_step:02d}.npy"),
            os.path.join(
                output_dir,
                "images_wrist_hi",
                f"image_wrist_hi_{old_step:02d}.png",
            ),
            os.path.join(
                output_dir,
                "world_wrist_hi",
                f"world_wrist_hi_{old_step:02d}.npy",
            ),
        ):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    blob = {
        "step_idx": step_idx,
        "libero_terminated": driver.env.episode_done,
        "task_language": driver.env.get_task_language(),
        "state": state,
        "world_map": agent_world_map,
        "wrist_world_map": wrist_world_map,
        "world_map_hi": agent_world_map_hi,
        "wrist_world_map_hi": wrist_world_map_hi,
    }
    # Merge the execution log (command + result + elapsed_s) into the
    # state blob so a single entry captures everything for the step.
    if log is not None:
        blob["command"] = log.get("command")
        blob["result"] = log.get("result")
        blob["elapsed_s"] = log.get("elapsed_s")
    _append_state(output_dir, blob)
    return blob


# ---------------------------------------------------------------------------
# Tool schema declarations (Anthropic-shaped canonical schema)
# ---------------------------------------------------------------------------

TOOLS_SPEC = [
    {
        "name": "view_driver_state",
        "description": (
            "Read step NN from `states.json` + the matching "
            "state images in {{output_dir}}. If step is "
            "null, returns the latest entry. Each entry contains the robot "
            "state, libero_terminated flag, command log, and result. Embeds "
            "available PNGs as multimodal image content blocks in this stable "
            "order: 1) `images/image_NN.png` (Pi0-frame agentview), "
            "2) `images_cam/image_cam_NN.png` (calibration-frame agentview), "
            "3) `images_wrist/image_wrist_NN.png` (calibration-frame wrist). "
            "High-resolution calibration-frame images are returned as file "
            "paths, not embedded as image bytes. "
            "Use the calibration-frame images for pixel back-projection; JSON "
            "state alone is not enough. Use agentview for global tabletop "
            "layout and object locations; use wrist for close-range details "
            "near the gripper, occlusions, and container/cabinet interiors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": ["integer", "null"],
                    "description": "Step number; 0 = initial. Null = latest.",
                },
            },
        },
    },
    {
        "name": "move_to",
        "description": (
            "Scripted EEF servo to a world-frame XYZ target via the OSC "
            "controller. Holds orientation (use rotate_wrist / rotate_pitch "
            "/ move_pose to reorient). gripper: -1 = open, +1 = close. NEVER "
            "command a single move_to with |Δxy| > 0.30 — OSC flips IK and "
            "the run corrupts; split long traversal into 2-3 mid waypoints "
            "at carry z."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "description": "World-frame target [x, y, z] in meters",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "gripper": {
                    "type": "number",
                    "description": "Gripper command: -1 open, +1 close (default -1)",
                },
                "tol": {"type": "number", "description": "Position tolerance, m (default 0.012)"},
                "step_clip": {"type": "number", "description": "Per-step Δxyz cap before action_scale, m (default 0.025)"},
                "max_steps": {"type": "integer", "description": "Step budget (default 80)"},
                "action_scale": {"type": "number", "description": "OSC action scale (default 0.05)"},
                "target_yaw": {
                    "type": ["number", "null"],
                    "description": "Optional world-frame yaw target in radians",
                },
                "yaw_step_clip": {"type": "number", "description": "Per-step yaw clip, rad (default 0.10)"},
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "pi0_pick",
        "description": (
            "Pi0.5 closed-loop pick. Use it for the grasp; YOU then do "
            "every move_to and release. Use modest max_chunks and verify "
            "the grasp from EEF lift, gripper closure, and available images."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Pi0 prompt (e.g. 'pick up the akita black bowl').",
                },
                "max_chunks": {"type": "integer", "description": "Action-chunk budget (default 30)"},
                "lift_thresh": {"type": "number", "description": "EEF post-descent ascent threshold for success, m (default 0.05)"},
                "gripper_closed_thresh": {"type": "number", "description": "Finger-separation closed threshold (default 0.06)"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "pi0_doubled",
        "description": (
            "Pi0.5 closed-loop contact skill for non-pick interactions "
            "(e.g. stove/knob/button/short push). Returned success/task_success "
            "only mirrors official libero_terminated; for intermediate contact "
            "skills, success=false does not necessarily mean the contact "
            "interaction failed. Inspect image/state evidence. Do not use it "
            "as a general pick/place shortcut."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Contact-skill prompt, e.g. 'turn on the stove'.",
                },
                "max_chunks": {"type": "integer", "description": "Action-chunk budget (default 20)"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "release",
        "description": (
            "Open the gripper for up to max_steps env steps while holding "
            "EEF in place. Triggers libero termination if the matching "
            "On/In predicate is met."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_steps": {"type": "integer", "description": "Step budget (default 20)"},
            },
        },
    },
    {
        "name": "set_gripper",
        "description": (
            "Hold the current EEF pose and drive the gripper command for "
            "`steps` env steps. Use to firm up a grip mid-carry."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gripper": {
                    "type": "number",
                    "description": "Gripper command: -1 open, +1 close (default -1)",
                },
                "steps": {"type": "integer", "description": "Number of env steps (default 5)"},
            },
        },
    },
    {
        "name": "rotate_wrist",
        "description": (
            "Rotate the wrist around the world Z-axis. Provide either "
            "target_yaw (absolute) or delta_yaw (relative). Holds xyz fixed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_yaw": {"type": ["number", "null"], "description": "Absolute world-frame yaw target, rad"},
                "delta_yaw": {"type": ["number", "null"], "description": "Relative yaw delta, rad"},
                "gripper": {"type": "number", "description": "Gripper command held during rotation (default +1)"},
                "max_steps": {"type": "integer", "description": "Step budget (default 40)"},
                "tol": {"type": "number", "description": "Yaw tolerance, rad (default 0.02)"},
                "step_clip": {"type": "number", "description": "Per-step yaw clip, rad (default 0.10)"},
            },
        },
    },
    {
        "name": "rotate_pitch",
        "description": (
            "Tilt the gripper around the world X-axis. Provide either "
            "target_pitch (absolute) or delta_pitch (relative). Holds xyz "
            "and yaw fixed. Use before threading the gripper into a narrow "
            "opening whose front face normal is along world ±y (e.g. "
            "microwave cavity)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_pitch": {"type": ["number", "null"], "description": "Absolute world-frame pitch target, rad"},
                "delta_pitch": {"type": ["number", "null"], "description": "Relative pitch delta, rad"},
                "gripper": {"type": "number", "description": "Gripper command held during rotation (default +1)"},
                "max_steps": {"type": "integer", "description": "Step budget (default 40)"},
                "tol": {"type": "number", "description": "Pitch tolerance, rad (default 0.02)"},
                "step_clip": {"type": "number", "description": "Per-step pitch clip, rad (default 0.10)"},
            },
        },
    },
    {
        "name": "move_pose",
        "description": (
            "Servo position AND orientation (pitch + yaw) SIMULTANEOUSLY. "
            "Unlike move_to (holds orientation) + rotate_pitch (holds xyz), "
            "this co-varies xyz and wrist tilt every env.step. Use to thread "
            "cabinet-front / low-shelf poses where a decoupled position "
            "servo drives the wrist into an IK singularity and stalls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "description": "World-frame target [x, y, z] in meters",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "target_pitch": {"type": ["number", "null"], "description": "Absolute pitch target, rad"},
                "target_yaw": {"type": ["number", "null"], "description": "Absolute yaw target, rad"},
                "gripper": {"type": "number", "description": "Gripper command held during the move (default -1)"},
                "step_clip": {"type": "number", "description": "Per-step Δxyz cap, m (default 0.02)"},
                "pitch_step": {"type": "number", "description": "Per-step pitch clip, rad (default 0.08)"},
                "yaw_step": {"type": "number", "description": "Per-step yaw clip, rad (default 0.08)"},
                "tol": {"type": "number", "description": "Position tolerance, m (default 0.012)"},
                "ori_tol": {"type": "number", "description": "Orientation tolerance, rad (default 0.05)"},
                "max_steps": {"type": "integer", "description": "Step budget (default 150)"},
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "view_camera_meta",
        "description": (
            "Read camera calibration metadata from the output dir. "
            "camera='agentview' reads static camera_meta.json. "
            "camera='wrist' reads the per-step wrist metadata."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "wrist"],
                    "description": "Camera metadata to read (default agentview).",
                },
                "step": {
                    "type": ["integer", "null"],
                    "description": "Wrist metadata step to use (default latest).",
                },
            },
        },
    },
    {
        "name": "segment",
        "description": (
            "Optional visual segmentation helper. It consumes existing "
            "run artifacts only and never renders new camera views. Use it "
            "as a fallback localization aid when a configured segmentation "
            "service and the requested image artifact already exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Object/text prompt to segment.",
                },
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "wrist"],
                    "description": "Artifact camera to use (default agentview).",
                },
                "step": {
                    "type": ["integer", "null"],
                    "description": "Step NN to segment; null = latest.",
                },
                "point": {
                    "type": ["array", "null"],
                    "description": "Optional [row, col] point prompt.",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum accepted mask score (default 0.2).",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "back_project",
        "description": (
            "Back-project a pixel (row, col) to a world XYZ point using the "
            "selected camera's precomputed world map. Row 0 = top of image, "
            "col 0 = left. Returns world_xyz in meters.\n\n"
            "USE THIS to find where an object is in the world — look at "
            "the high-resolution paths returned by view_driver_state "
            "to pick a pixel on the target object, then call back_project. "
            "The default resolution is high (1024x1024). Pass "
            "resolution='low' only for pixels from the embedded/standard "
            "256 image. The pixel coordinates must come "
            "from the same camera and resolution requested here. Use "
            "camera='agentview' for global tabletop layout and object "
            "locations; use camera='wrist' for close-range details near the "
            "gripper, occlusions, and container/cabinet interiors. "
            "Sample several pixels on the object and median their xy for "
            "robustness."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "description": "Pixel row (0=top) in the selected resolution image.",
                },
                "col": {
                    "type": "integer",
                    "description": "Pixel column (0=left) in the selected resolution image.",
                },
                "step": {
                    "type": ["integer", "null"],
                    "description": "Depth/world-map step to use (default latest). 0 for initial.",
                },
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "wrist"],
                    "description": "Camera to back-project from (default agentview).",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["high", "low"],
                    "description": (
                        "Coordinate system for row/col (default high). "
                        "Use low only when row/col came from the "
                        "embedded/standard 256 image."
                    ),
                },
            },
            "required": ["row", "col"],
        },
    },
]


# ---------------------------------------------------------------------------
# State trace readers
# ---------------------------------------------------------------------------


def _load_states() -> list:
    """Return the parsed driver state trace from the local output dir."""
    path = get_output_dir() / "states.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _latest_step() -> int | None:
    states = _load_states()
    if not states:
        return None
    return int(states[-1]["step_idx"])


def _load_step(nn: int) -> dict:
    """Look up the state blob for step ``nn`` from states.json."""
    for entry in _load_states():
        if int(entry.get("step_idx", -1)) == nn:
            return entry
    raise FileNotFoundError(f"step {nn} not present in states.json")


def _load_image(nn: int, kind: str) -> bytes | None:
    """Return PNG bytes for a dumped state image. None if not present."""
    out_dir = get_output_dir()
    if kind == "agent":
        path = out_dir / "images" / f"image_{nn:02d}.png"
    elif kind == "camera":
        path = out_dir / "images_cam" / f"image_cam_{nn:02d}.png"
    elif kind == "wrist":
        path = out_dir / "images_wrist" / f"image_wrist_{nn:02d}.png"
    else:
        raise ValueError(f"unknown image kind: {kind}")
    if not path.exists():
        return None
    return path.read_bytes()


def _load_camera_meta(camera: str = "agentview", nn: int | None = None) -> dict:
    out_dir = get_output_dir()
    if camera == "agentview":
        path = out_dir / "camera_meta.json"
    elif camera == "wrist" and nn is not None:
        path = out_dir / "wrist_meta" / f"wrist_meta_{nn:02d}.json"
    else:
        raise ValueError("camera must be 'agentview' or 'wrist' with nn")
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {out_dir}")
    with open(path) as f:
        return json.load(f)


def _load_depth(camera: str, nn: int) -> np.ndarray:
    out_dir = get_output_dir()
    if camera == "agentview":
        path = out_dir / "depths" / f"depth_{nn:02d}.npy"
    elif camera == "wrist":
        path = out_dir / "depths_wrist" / f"depth_wrist_{nn:02d}.npy"
    else:
        raise ValueError("camera must be 'agentview' or 'wrist'")
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {out_dir}")
    depth = np.load(path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def view_driver_state(step: int | None = None) -> dict:
    latest = _latest_step()
    if latest is None:
        return {"error": "no driver state entries; driver not ready"}
    nn = latest if step is None else int(step)
    try:
        data = _load_step(nn)
    except Exception as e:
        return {"error": f"step {nn} not present in driver state trace: {e}"}

    out: dict = {"step": nn}
    out["task_language"] = data.get("task_language")
    out["state"] = data.get("state", data)
    out["libero_terminated"] = data.get("libero_terminated")
    out["world_map"] = data.get("world_map")
    out["wrist_world_map"] = data.get("wrist_world_map")
    out["world_map_hi"] = data.get("world_map_hi")
    out["wrist_world_map_hi"] = data.get("wrist_world_map_hi")
    out["log"] = {
        "command": data.get("command"),
        "result": data.get("result"),
        "elapsed_s": data.get("elapsed_s"),
    }
    image = _load_image(nn, "agent")
    image_cam = _load_image(nn, "camera")
    image_wrist = _load_image(nn, "wrist")
    if image:
        out["_image_bytes"] = image
    if image_cam:
        out["_image_cam_bytes"] = image_cam
    if image_wrist:
        out["_image_wrist_bytes"] = image_wrist
    out_dir = get_output_dir()
    image_cam_hi_path = out_dir / "images_cam_hi" / f"image_cam_hi_{nn:02d}.png"
    image_wrist_hi_path = (
        out_dir / "images_wrist_hi" / f"image_wrist_hi_{nn:02d}.png"
    )
    if image_cam_hi_path.exists():
        out["image_cam_hi_path"] = str(image_cam_hi_path)
    if image_wrist_hi_path.exists():
        out["image_wrist_hi_path"] = str(image_wrist_hi_path)
    return out


def _segment_image_candidates(nn: int, camera: str) -> list:
    out_dir = get_output_dir()
    if camera == "agentview":
        return [
            out_dir / "images_cam_hi" / f"image_cam_hi_{nn:02d}.png",
            out_dir / f"image_cam_hi_{nn:02d}.png",
            out_dir / "images_cam" / f"image_cam_{nn:02d}.png",
        ]
    if camera == "wrist":
        return [
            out_dir / "images_wrist_hi" / f"image_wrist_hi_{nn:02d}.png",
            out_dir / f"image_wrist_hi_{nn:02d}.png",
            out_dir / "images_wrist" / f"image_wrist_{nn:02d}.png",
            out_dir / f"image_wrist_{nn:02d}.png",
            out_dir / "images" / f"image_wrist_{nn:02d}.png",
        ]
    raise ValueError(f"unknown segment camera: {camera}")


def segment(
    prompt: str,
    camera: str = "agentview",
    step: int | None = None,
    point: list[int] | None = None,
    min_score: float = 0.2,
) -> dict:
    """Call an optional segmentation service on an existing image artifact.

    This tool deliberately does not render camera views or create wrist/high-res
    artifacts. If the requested artifact or service is missing, it returns a
    structured fallback error so the agent can continue with image inspection
    and back_project.
    """
    nn = _latest_step() if step is None else int(step)
    if nn is None:
        return {"error": "no driver state entries; cannot select segment image"}

    camera = camera or "agentview"
    try:
        candidates = _segment_image_candidates(nn, camera)
    except ValueError as e:
        return {"error": str(e)}
    image_path = next((p for p in candidates if p.exists()), None)
    if image_path is None:
        return {
            "error": "segment image artifact not found",
            "step": nn,
            "camera": camera,
            "checked_paths": [str(p) for p in candidates],
            "fallback": "Read the available image artifact and use back_project.",
        }

    server_url = (
        os.environ.get("SAM3_SERVER_URL")
        or os.environ.get("SEGMENT_SERVER_URL")
        or ""
    ).strip()
    if not server_url:
        return {
            "error": "segmentation service not configured",
            "step": nn,
            "camera": camera,
            "image_path": str(image_path),
            "fallback": "Use manual visual localization and back_project.",
        }

    payload = {
        "prompt": prompt,
        "camera": camera,
        "step": nn,
        "image_path": str(image_path),
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        "point": point,
        "min_score": min_score,
    }
    try:
        import httpx

        response = httpx.post(
            server_url.rstrip("/") + "/segment",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return {
            "error": f"segmentation service call failed: {e}",
            "step": nn,
            "camera": camera,
            "image_path": str(image_path),
            "fallback": "Use manual visual localization and back_project.",
        }

    out_dir = get_output_dir()
    segment_path = out_dir / f"segment_{nn:02d}.json"
    segment_blob = {
        "prompt": prompt,
        "camera": camera,
        "step": nn,
        "image_path": str(image_path),
        "min_score": min_score,
        "response": data,
    }
    segment_path.write_text(json.dumps(segment_blob, indent=2, default=str))
    overlay_b64 = data.get("overlay_base64") if isinstance(data, dict) else None
    overlay_path = None
    if overlay_b64:
        overlay_path = out_dir / f"segment_overlay_{nn:02d}.png"
        try:
            overlay_path.write_bytes(base64.b64decode(overlay_b64))
        except Exception:
            overlay_path = None

    result = {
        "step": nn,
        "camera": camera,
        "image_path": str(image_path),
        "segment_path": str(segment_path),
        "response": data,
    }
    if overlay_path is not None:
        result["overlay_path"] = str(overlay_path)
        result["_image_bytes"] = overlay_path.read_bytes()
    return result


def view_camera_meta(camera: str = "agentview", step: int | None = None) -> dict:
    """Read camera calibration metadata for localization."""
    if camera not in ("agentview", "wrist"):
        return {"error": f"bad camera '{camera}' (use 'agentview' or 'wrist')"}

    nn = None
    if camera == "wrist":
        nn = _latest_step() if step is None else int(step)
        if nn is None:
            return {"error": "no wrist metadata available"}

    try:
        meta = _load_camera_meta(camera, nn)
    except Exception as e:
        return {"error": f"{camera} camera metadata not found: {e}"}

    if camera == "agentview":
        return {"camera": "agentview", "camera_meta": meta}
    else:
        return {"camera": "wrist", "step": nn, "camera_meta": meta}


def back_project(
    row: int,
    col: int,
    step: int | None = None,
    camera: str = "agentview",
    resolution: str = "high",
) -> dict:
    """Look up a pixel's world XYZ in the precomputed world map."""
    if camera not in ("agentview", "wrist"):
        return {"error": f"bad camera '{camera}' (use 'agentview' or 'wrist')"}
    if resolution not in ("high", "low"):
        return {"error": f"bad resolution '{resolution}' (use 'high' or 'low')"}

    nn = _latest_step() if step is None else int(step)
    if nn is None:
        return {"error": "no depth/world-map files available"}

    try:
        data = _load_step(nn)
    except Exception as e:
        return {"error": f"step {nn} not present in driver state trace: {e}"}

    if camera == "agentview":
        hi_artifact = data.get("world_map_hi")
        low_artifact = data.get("world_map")
    else:
        hi_artifact = data.get("wrist_world_map_hi")
        low_artifact = data.get("wrist_world_map")
    source_artifact = hi_artifact if resolution == "high" else low_artifact
    if not source_artifact:
        return {
            "error": (
                f"{camera} {resolution}-resolution world map not recorded "
                f"for step {nn}"
            )
        }

    try:
        world_map = np.load(get_output_dir() / source_artifact)
    except Exception as e:
        return {
            "error": (
                f"{camera} {resolution}-resolution artifact not found "
                f"for step {nn}: {e}"
            )
        }

    height, width = world_map.shape[:2]
    if row < 0 or row >= height or col < 0 or col >= width:
        return {
            "error": (
                f"pixel ({row},{col}) out of bounds; {camera} image is "
                f"{height}x{width}"
            )
        }

    depth_m = None
    if source_artifact == low_artifact:
        try:
            depth = _load_depth(camera, nn)
        except Exception as e:
            return {"error": f"{camera} depth not found for step {nn}: {e}"}
        depth_m = float(depth[row, col])
        if not np.isfinite(depth_m) or depth_m <= 0 or depth_m > 10:
            return {
                "error": (
                    f"invalid {camera} depth {depth_m:.3f}m at pixel "
                    f"({row},{col}); pick a different pixel"
                )
            }
    world_xyz_raw = world_map[row, col]
    if (
        not np.isfinite(world_xyz_raw).all()
        or float(np.abs(world_xyz_raw[:3]).sum()) <= 1e-6
    ):
        return {"error": f"invalid {camera} world xyz at pixel ({row},{col})"}
    world_xyz = [round(float(v), 4) for v in world_xyz_raw[:3]]

    out = {
        "camera": camera,
        "resolution": resolution,
        "pixel": [row, col],
        "world_xyz": world_xyz,
        "step": nn,
        "image_size": [height, width],
        "source_artifact": source_artifact,
    }
    if depth_m is not None:
        out["depth_m"] = round(depth_m, 4)
    return out

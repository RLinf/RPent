"""LIBERO + OpenPI tool implementation."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from robots.libero.env_client import LiberoEnvClient
from rpent.tools.state import EnvState, StepRecord
from rpent.tools.toolkit import parallel, readonly
from rpent.utils.logging import get_logger
from rpent.utils.sam3_client import Sam3Client
from rpent.utils.vla_client import VLAClient

logger = get_logger("libero")


def _normalize_xyz(xyz):
    """Coerce an LLM-supplied xyz into a length-3 list[float]."""
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        raise ValueError(
            'xyz must be a JSON array of three numbers, e.g. "xyz":[-0.05,0,0.3]'
        )
    return [float(v) for v in xyz]


class LiberoPrimitives:
    """Wraps a single-env LIBERO-shaped env + VLA policy with primitive-
    level methods.

    ``pi0_pick`` and ``pi0_doubled`` override ``obs['task_descriptions']``
    with a sub-instruction. ``move_to`` and friends are scripted (no VLM
    call) and drive the underlying OSC controller directly.
    """

    def __init__(
        self,
        env: LiberoEnvClient,
        model: VLAClient,
        sam3_client: Sam3Client,
        check_cancelled: Callable[[], None],
    ):
        self.env = env
        self.model = model
        self._sam3_client = sam3_client
        self._check_cancelled = check_cancelled
        self._last_obs = None
        self._last_obs_eef_pos = None
        self._last_obs_eef_z = None
        self._last_obs_gripper = None
        # Per-env-step frame buffer for diagnostic video rendering.
        # Toggled via start_recording() / stop_recording().
        self._recording = False
        self._frames = []

    def start_recording(self):
        self._recording = True
        self._frames = []

    def record_frame(self, obs):
        """Append one agentview frame extracted from ``obs`` to the buffer."""
        self._frames.append(np.ascontiguousarray(np.asarray(obs["main_images"])))

    def recorded_frame_count(self) -> int:
        return len(self._frames)

    def stop_recording(self) -> list[np.ndarray]:
        frames = list(self._frames)
        self._recording = False
        self._frames = []
        return frames

    def frame_slice(self, start: int) -> list[np.ndarray]:
        return list(self._frames[int(start):])

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

    def _step_env(self, action) -> None:
        """Execute one env action between cancellation checkpoints."""
        self._check_cancelled()
        obs, _r, _t, _tr, _i = self.env.step(action)
        self.set_obs(obs)
        if self._recording:
            self.record_frame(obs)

    def _vlm_chunk(self, instruction: str):
        """One model forward + ``chunk_size`` env steps. Overrides prompt."""
        self._check_cancelled()
        original_task = self._last_obs.get("task_descriptions")
        try:
            self._last_obs["task_descriptions"] = instruction
            self._last_obs.setdefault("extra_view_images", None)

            actions, _ = self.model.predict_action_batch(self._last_obs, mode="eval")
            self._check_cancelled()

            if not self._recording:
                chunk_obs,  _r, _t, _tr, _i = self.env.chunk_step(actions)
                obs = chunk_obs[-1] if self.env.return_all_frames else chunk_obs
            else:
                chunk_obs,  _r, _t, _tr, _i = self.env.chunk_step(
                    actions, return_all_frames=True
                )
                for obs in chunk_obs:
                    self.record_frame(obs)
                obs = chunk_obs[-1]
            self.set_obs(obs)
            return self._last_obs
        finally:
            if original_task is not None:
                self._last_obs["task_descriptions"] = original_task

    def pi0_pick(
        self,
        prompt: str,
        *,
        max_chunks: int = 24,
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
            if self.env.terminated or self.env.truncated:
                success = self.env.terminated
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
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
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
            if self.env.terminated or self.env.truncated:
                task_success = self.env.terminated
                break

        return {
            "name": "pi0_doubled",
            "instruction": instr,
            "success": task_success,
            "task_success": task_success,
            "contact_skill_executed": chunks_used > 0,
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
            "diagnostics": {
                "mode": "contact_skill_success_by_termination",
                "success_meaning": (
                    "`success` mirrors official LIBERO task termination only; "
                    "for intermediate contact skills, inspect image/state evidence."
                ),
            },
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
            self._step_env(action)
            if self.env.terminated or self.env.truncated:
                break
        final = self._last_obs_eef_pos
        return {
            "name": "move_to",
            "target_xyz": [float(x) for x in target],
            "final_eef_pos": [round(float(x), 4) for x in final],
            "final_dist_m": round(float(np.linalg.norm(target - final)), 4),
            "steps_used": len(traj),
            "max_steps": max_steps,
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
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
            self._step_env(action)
            if self.env.terminated or self.env.truncated:
                break
        final_yaw = _yaw_of(self.env.raw_obs()["robot0_eef_quat"])
        return {
            "name": "rotate_wrist",
            "start_yaw": round(start_yaw, 4),
            "target_yaw": round(float(target_yaw), 4),
            "final_yaw": round(final_yaw, 4),
            "final_err": round(float((target_yaw - final_yaw + np.pi) % (2 * np.pi) - np.pi), 4),
            "steps_used": len(traj),
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
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
            self._step_env(action)
            if self.env.terminated or self.env.truncated:
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
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
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
            self._step_env(action)
            if self.env.terminated or self.env.truncated:
                break
        final = self._last_obs_eef_pos
        fq = self.env.raw_obs()["robot0_eef_quat"]
        return {
            "name": "move_pose",
            "final_eef_pos": [round(float(x), 4) for x in final],
            "final_dist_m": round(float(np.linalg.norm(target - final)), 4),
            "final_pitch": round(_pitch_of(fq), 4),
            "steps_used": step + 1,
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
        }

    def release(
        self,
        *,
        max_steps: int = 20,
    ) -> dict:
        """Open gripper for ``max_steps`` env steps while keeping eef in place.

        Returns once libero terminates (success) or step budget exhausted.
        """
        assert max_steps > 0, f"max_steps must be > 0, got {max_steps}"
        start_grip = self._last_obs_gripper
        peak_grip = start_grip
        for step in range(max_steps):
            action = np.zeros(7, dtype=np.float32)
            action[6] = -1.0  # open
            self._step_env(action)
            peak_grip = max(peak_grip, self._last_obs_gripper)
            if self.env.terminated or self.env.truncated:
                break
        return {
            "name": "release",
            "steps_used": step + 1,
            "start_gripper_opening": round(start_grip, 4),
            "peak_gripper_opening": round(peak_grip, 4),
            "final_gripper_opening": round(self._last_obs_gripper, 4),
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
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
            self._step_env(action)
            if self.env.terminated or self.env.truncated:
                break
        return {
            "name": "set_gripper",
            "gripper": g,
            "steps": n,
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
        }

    # ---- introspection helpers (for LLM-in-the-loop) ----

    @readonly
    def segment(
        self,
        prompt: str = "",
        camera: str = "agentview",
        step: int = -1,
        point: list[int] | None = None,
        min_score: float = 0.2,
        *,
        state: EnvState,
    ) -> dict:
        """Call SAM3 on an existing image artifact without advancing the env.

        This tool deliberately does not render camera views or create wrist/high-res
        artifacts. Errors are structured so the agent can continue with image
        inspection and ``back_project``.
        """
        try:
            record = state.get(step)
        except Exception as exc:
            return {"error": f"state step not available: {exc}"}
        nn = record.step_idx

        camera = camera or "agentview"
        prompt = prompt.strip()
        has_prompt = bool(prompt)
        has_point = point is not None
        if has_prompt == has_point:
            return {"error": "segment needs exactly one of prompt or point"}
        try:
            image_name, world_name, artifact_pairs = _select_segment_artifacts(
                state, record, camera
            )
        except ValueError as e:
            return {"error": str(e)}
        if image_name is None or world_name is None:
            return {
                "error": "complete segment artifacts not found",
                "step": nn,
                "camera": camera,
                "checked_artifacts": [
                    name
                    for image, world in artifact_pairs
                    for name in (image, world)
                    if name
                ],
            }

        try:
            data = self._sam3_client.segment(
                state.load_bytes(image_name, step=nn),
                text_prompt=prompt if has_prompt else None,
                point=point,
                min_score=min_score,
            )
        except ValueError as e:
            return {
                "error": str(e),
                "step": nn,
                "camera": camera,
                "image_artifact": image_name,
            }
        except Exception as e:
            return {
                "error": f"segmentation service call failed: {e}",
                "step": nn,
                "camera": camera,
                "image_artifact": image_name,
                "fallback": "Use manual visual localization and back_project.",
            }

        segment_index = _next_segment_index(record)
        segment_name = f"segment_{segment_index:02d}.json"
        overlay_name = f"segment_overlay_{segment_index:02d}.png"
        saved_overlay = None
        mask = data.mask
        if data.found and isinstance(mask, np.ndarray):
            try:
                world_map = state.load(world_name, step=nn)
            except Exception as exc:
                world_result = {
                    "world_xyz": None,
                    "world_error": f"world map artifact not available: {exc}",
                    "expected_world_artifact": world_name,
                }
            else:
                world_result = _mask_to_world(mask, world_map)
                world_result["world_artifact"] = world_name
            overlay = _make_segment_overlay(state.load(image_name, step=nn), mask)
            if overlay is not None and state.save(
                overlay_name,
                overlay,
                step=nn,
            ):
                saved_overlay = overlay_name
        else:
            world_result = {
                "world_xyz": None,
                "world_error": data.reason or "segmentation did not find a mask",
            }

        segment_blob = {
            "found": data.found,
            "mode": "text" if has_prompt else "point",
            "camera": camera,
            "source_step": nn,
            "segment_index": segment_index,
            "image_artifact": image_name,
            "min_score": min_score,
            "score": round(float(data.score), 3) if data.score is not None else None,
            "box": data.box,
            "mask_shape": list(data.mask_shape) if data.mask_shape else None,
        }
        if has_prompt:
            segment_blob["prompt"] = prompt
        else:
            segment_blob["point"] = point
        if not data.found:
            segment_blob["error"] = data.reason or "SAM3 found no mask"
        segment_blob.update(world_result)
        saved_segment = state.save(
            segment_name,
            segment_blob,
            step=nn,
        )

        result = {
            "found": data.found,
            "step": nn,
            "camera": camera,
            "image_artifact": image_name,
            "score": segment_blob["score"],
            "box": segment_blob["box"],
            "world_xyz": segment_blob["world_xyz"],
            "world_error": segment_blob.get("world_error"),
        }
        if saved_segment is None:
            result["error"] = f"failed to persist segment artifact {segment_name}"
            result["code"] = "segment_artifact_save_failed"
            result["attempted_segment_artifact"] = segment_name
            if "error" in segment_blob:
                result["segmentation_error"] = segment_blob["error"]
            result["fallback"] = "Use manual visual localization and back_project."
        else:
            result["segment_artifact"] = saved_segment
        if saved_segment is not None and "error" in segment_blob:
            result["error"] = segment_blob["error"]
            result["fallback"] = "Use manual visual localization and back_project."
        if saved_overlay is not None:
            result["overlay_artifact"] = saved_overlay
            result["_image_bytes"] = state.load_bytes(saved_overlay, step=nn)
        return result


def _is_primitive_action(name: object) -> bool:
    """Whether ``name`` is a state-advancing LIBERO primitive.

    A primitive is any non-read-only method on :class:`LiberoPrimitives`;
    read-only tools and non-strings read as ``False``.
    """
    if not isinstance(name, str):
        return False
    method = getattr(LiberoPrimitives, name, None)
    return method is not None and not bool(getattr(method, "_readonly", False))


def write_recipe_from_states(state: EnvState, recipe_tag: str) -> str:
    """Find a command sequence that gets ``terminated=True``.

    Export non-error LIBERO primitive commands and successful segment calls.
    """
    command_events = []
    for record in state.records():
        command = record.command
        result = record.result
        if (
            command is not None
            and _is_primitive_action(command.get("action"))
            and not (isinstance(result, dict) and result.get("error"))
        ):
            command_events.append(((record.step_idx, -1), command))

        for name in sorted(record.artifacts):
            if not (name.startswith("segment_") and name.endswith(".json")):
                continue
            segment = state.load(name, step=record.step_idx)
            if segment.get("error"):
                continue
            if segment["mode"] == "text":
                segment_command = {
                    "action": "segment",
                    "prompt": segment["prompt"],
                    "camera": segment["camera"],
                }
            else:
                segment_command = {
                    "action": "segment",
                    "point": segment["point"],
                    "camera": segment["camera"],
                }
            event_order = (record.step_idx, int(segment["segment_index"]))
            command_events.append((event_order, segment_command))

    command_events.sort(key=lambda event: event[0])
    recipe_name = f"recipe_{recipe_tag}.jsonl"
    state.save(
        recipe_name,
        [command for _, command in command_events],
        step=None,
    )
    return recipe_name


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


def dump_state(
    primitives: LiberoPrimitives,
    env_state: EnvState,
    log: dict | None = None,
) -> StepRecord:
    """Save one Libero observation through its owned state record."""
    raw = primitives.env.raw_obs()
    state = {
        "robot0_eef_pos": [float(x) for x in raw["robot0_eef_pos"]],
        "robot0_eef_quat": [float(x) for x in raw["robot0_eef_quat"]],
        "robot0_gripper_qpos": [float(x) for x in raw["robot0_gripper_qpos"]],
        "object_names": sorted(
            k[:-4]
            for k in raw
            if k.endswith("_pos") and "robot0" not in k and "to_robot" not in k
        ),
    }
    log = log or {}
    with env_state.record_step(
        state=state,
        terminated=primitives.env.terminated,
        truncated=primitives.env.truncated,
        command=log.get("command"),
        result=log.get("result"),
        elapsed_s=log.get("elapsed_s"),
        extras={
            "task_language": primitives.env.get_task_language(),
        },
    ) as step_idx:
        _save_observation_artifacts(primitives, env_state, step_idx, raw)
    return env_state.get(step_idx)


def _save_observation_artifacts(
    primitives: LiberoPrimitives,
    env_state: EnvState,
    step_idx: int,
    raw: dict[str, Any],
) -> None:
    env_state.save(
        "agentview_policy.png",
        primitives._last_obs["main_images"],
        step=step_idx,
    )

    # --- camera calibration (static for agentview): fetch metadata as needed ---
    agentview_meta = primitives.env.get_camera_meta(
        camera_name="agentview",
        height=256,
        width=256,
    ) or {}
    if agentview_meta:
        cam_meta_out = dict(agentview_meta)
        cam_meta_out["projection"] = (
            "Prefer the back_project(row, col, step=NN) MCP tool; it "
            "uses the 1024x1024 high-resolution world map by default. "
            "Pass resolution='low' only when row/col came from the "
            "256x256 calibration-frame image."
        )
        cam_meta_out["note"] = (
            "The agentview_depth.npz observation is aligned with agentview.png. "
            "agentview_policy.png uses the Pi0 orientation and must not supply "
            "pixels for back-projection."
        )
        env_state.save(
            "agentview_metadata.json",
            cam_meta_out,
            step=step_idx,
        )

    # --- per-step RGB in the depth/K frame (vertical-flip of the raw buffer) ---
    # Agentview pixels align with the matching depth and calibration. The policy
    # image uses Pi0 orientation and must not supply back-projection pixels.
    try:
        ci = raw.get("agentview_image")
        if ci is not None:
            ci = np.asarray(ci)
            if ci.dtype != np.uint8:
                ci = ci.astype(np.uint8)
            env_state.save(
                "agentview.png",
                ci[::-1],
                step=step_idx,
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
            # Pixels in agentview.png align with agentview_depth.npz and the
            # per-step agentview metadata saved with the record artifacts.
            d = _metric_depth(d, agentview_meta)[::-1]
            env_state.save(
                "agentview_depth.npz",
                d.astype(np.float32),
                step=step_idx,
            )
            world = _world_from_depth(d, agentview_meta).astype(np.float32)
            env_state.save(
                "agentview_world.npz",
                world,
                step=step_idx,
            )
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
            env_state.save(
                "wrist.png",
                wimg[::-1],
                step=step_idx,
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
            wmeta = primitives.env.get_camera_meta(
                camera_name="robot0_eye_in_hand",
                height=int(height),
                width=int(width),
            )
            if wmeta is None:
                logger.warning("wrist camera meta missing; skipping wrist depth/world")
            else:
                wdpt_metric = _metric_depth(wdpt_arr, wmeta)[::-1]
                env_state.save(
                    "wrist_depth.npz",
                    wdpt_metric.astype(np.float32),
                    step=step_idx,
                )
                world_w = _world_from_depth(wdpt_metric, wmeta).astype(np.float32)
                env_state.save(
                    "wrist_world.npz",
                    world_w,
                    step=step_idx,
                )

                wmeta_out = dict(wmeta)
                wmeta_out["note"] = (
                    "MOVING camera: extrinsic_cam2world is for THIS step "
                    "only. The matching wrist world-map observation gives world "
                    "(x,y,z) for that pixel in the same world frame as the "
                    "agentview world-map artifact."
                )
                env_state.save(
                    "wrist_metadata.json",
                    wmeta_out,
                    step=step_idx,
                )
    except Exception as e:
        logger.warning("wrist depth/world dump failed: %s", e)

    try:
        rgb_hi, depth_hi = primitives.env.render_camera(
            camera_name="agentview",
            height=1024,
            width=1024,
            depth=True,
        )
        meta_hi = primitives.env.get_camera_meta("agentview", 1024, 1024)
        if meta_hi is None:
            raise RuntimeError("agentview camera metadata missing")
        env_state.save(
            "agentview_high.png",
            np.asarray(rgb_hi)[::-1],
            step=step_idx,
        )
        world_hi = _world_from_depth(
            _metric_depth(depth_hi, meta_hi)[::-1],
            meta_hi,
        ).astype(np.float16)
        env_state.save(
            "agentview_world_high.npz",
            world_hi,
            step=step_idx,
        )
    except Exception as e:
        logger.warning("agentview high-res dump failed: %s", e)

    try:
        rgb_wrist_hi, depth_wrist_hi = primitives.env.render_camera(
            camera_name="robot0_eye_in_hand",
            height=1024,
            width=1024,
            depth=True,
        )
        meta_wrist_hi = primitives.env.get_camera_meta(
            "robot0_eye_in_hand", 1024, 1024
        )
        if meta_wrist_hi is None:
            raise RuntimeError("robot0_eye_in_hand camera metadata missing")
        env_state.save(
            "wrist_high.png",
            np.asarray(rgb_wrist_hi)[::-1],
            step=step_idx,
        )
        world_wrist_hi = _world_from_depth(
            _metric_depth(depth_wrist_hi, meta_wrist_hi)[::-1],
            meta_wrist_hi,
        ).astype(np.float16)
        env_state.save(
            "wrist_world_high.npz",
            world_wrist_hi,
            step=step_idx,
        )
    except Exception as e:
        logger.warning("wrist high-res dump failed: %s", e)


# ---------------------------------------------------------------------------
# Tool schema declarations (Anthropic-shaped canonical schema)
# ---------------------------------------------------------------------------

TOOLS_SPEC = [
    {
        "name": "view_env_state",
        "description": (
            "Read one recorded state and its observation artifacts. Step -1 "
            "selects the latest entry. Embeds policy, agentview, and wrist "
            "images when available. "
            "Use the calibration-frame images for pixel back-projection; JSON "
            "state alone is not enough. Use agentview for global tabletop "
            "layout and object locations; use wrist for close-range details "
            "near the gripper, occlusions, and container/cabinet interiors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": "Step number; 0 = initial, -1 = latest.",
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
                "max_chunks": {"type": "integer", "description": "Action-chunk budget (default 24)"},
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
            "only mirrors official termination; for intermediate contact "
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
                "action_scale": {"type": "number", "description": "OSC action scale (default 0.05)"},
                "max_steps": {"type": "integer", "description": "Step budget (default 150)"},
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "view_camera_meta",
        "description": (
            "Read per-step camera calibration metadata from recorded artifacts."
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
                    "type": "integer",
                    "default": -1,
                    "description": "Metadata step to use; -1 = latest.",
                },
            },
        },
    },
    {
        "name": "segment",
        "description": (
            "SAM3 visual segmentation over an existing run artifact. It never "
            "renders a new camera view. Provide exactly one text prompt or "
            "single positive point. A successful top-ranked mask is projected "
            "through the matching world map to produce world_xyz."
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
                    "type": "integer",
                    "default": -1,
                    "description": "Step to segment; -1 = latest.",
                },
                "point": {
                    "type": ["array", "null"],
                    "description": (
                        "Optional single positive point as [row, col]. "
                        "Mutually exclusive with prompt."
                    ),
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum accepted mask score (default 0.2).",
                },
            },
        },
    },
    {
        "name": "back_project",
        "description": (
            "Back-project a pixel (row, col) to a world XYZ point using the "
            "selected camera's precomputed world map. Row 0 = top of image, "
            "col 0 = left. Returns world_xyz in meters.\n\n"
            "USE THIS to find where an object is in the world — look at "
            "the embedded high-resolution image returned by view_env_state "
            "to pick a pixel on the target object, then call back_project. "
            "The default resolution is high (1024x1024). Pass "
            "resolution='low' only for pixels from the embedded/standard "
            "256 image. The pixel coordinates must come "
            "from the same camera and resolution requested here. Use "
            "camera='agentview' for global tabletop layout and object "
            "locations; use camera='wrist' for close-range details near the "
            "gripper, occlusions, and container/cabinet interiors. "
            "Sample several pixels on the object and median their xy for "
            "robustness.\n\n"
            "REGION MODE: pass row_range=[r0,r1] and col_range=[c0,c1] instead "
            "of row/col to get the midpoint of world xy over that pixel window, "
            "with an optional world-z band (z_min, z_max). Use it for the "
            "center of a container cavity or flat region, where a single-pixel "
            "or mask-median estimate is biased toward an edge/rim."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {
                    "type": ["integer", "null"],
                    "description": "Pixel row (0=top) in the selected resolution image.",
                },
                "col": {
                    "type": ["integer", "null"],
                    "description": "Pixel column (0=left) in the selected resolution image.",
                },
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": "Depth/world-map step; 0 = initial, -1 = latest.",
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
                "row_range": {
                    "type": ["array", "null"],
                    "items": {"type": "integer"},
                    "description": "Region mode: [r0, r1] pixel row window. Requires col_range.",
                },
                "col_range": {
                    "type": ["array", "null"],
                    "items": {"type": "integer"},
                    "description": "Region mode: [c0, c1] pixel col window. Requires row_range.",
                },
                "z_min": {
                    "type": ["number", "null"],
                    "description": "Region mode: keep only pixels with world z >= z_min.",
                },
                "z_max": {
                    "type": ["number", "null"],
                    "description": "Region mode: keep only pixels with world z <= z_max.",
                },
            },
        },
    },
]


@parallel
@readonly
def view_env_state(step: int = -1, *, state: EnvState) -> dict:
    try:
        record = state.get(step)
    except Exception as exc:
        return {"error": f"state step not available: {exc}"}

    nn = record.step_idx
    extras = record.extras
    out: dict = {
        "step": nn,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "state": record.state,
        "artifacts": sorted(record.artifacts),
    }
    out["task_language"] = extras.get("task_language")
    out["log"] = {
        "command": record.command,
        "result": record.result,
        "elapsed_s": record.elapsed_s,
    }
    for slot, names in (
        ("_image_bytes", ("agentview_policy.png",)),
        ("_image_cam_bytes", ("agentview_high.png", "agentview.png")),
        ("_image_wrist_bytes", ("wrist_high.png", "wrist.png")),
    ):
        name = next((name for name in names if name in record.artifacts), None)
        if name:
            try:
                out[slot] = state.load_bytes(name, step=nn)
            except FileNotFoundError:
                pass
    return out


def _select_segment_artifacts(
    state: EnvState,
    record: StepRecord,
    camera: str,
) -> tuple[str | None, str | None, list[tuple[str | None, str | None]]]:
    if camera not in ("agentview", "wrist"):
        raise ValueError(f"unknown segment camera: {camera}")
    pairs = [
        (f"{camera}_high.png", f"{camera}_world_high.npz"),
        (f"{camera}.png", f"{camera}_world.npz"),
    ]
    for image_name, world_name in pairs:
        if (
            image_name in record.artifacts
            and world_name in record.artifacts
            and state.exists(image_name, step=record.step_idx)
            and state.exists(world_name, step=record.step_idx)
        ):
            return image_name, world_name, pairs
    return None, None, pairs


def _next_segment_index(record: StepRecord) -> int:
    idx = 0
    while f"segment_{idx:02d}.json" in record.artifacts:
        idx += 1
    return idx


def _mask_to_world(mask: np.ndarray, world_map: np.ndarray,
                   min_valid: int = 10) -> dict:
    if world_map.ndim != 3 or world_map.shape[2] < 3:
        return {
            "world_xyz": None,
            "world_error": f"invalid world map shape: {tuple(world_map.shape)}",
            "n_pixels": int(mask.sum()),
            "n_valid": 0,
            "mask_resized_to_world_shape": False,
        }

    if mask.shape != world_map.shape[:2]:
        return {
            "world_xyz": None,
            "world_error": (
                f"mask/world shape mismatch: mask={tuple(mask.shape)}, "
                f"world={tuple(world_map.shape[:2])}"
            ),
            "n_pixels": int(mask.sum()),
            "n_valid": 0,
            "mask_resized_to_world_shape": False,
        }

    ys, xs = np.where(mask)
    if ys.size == 0:
        return {"world_xyz": None, "world_error": "empty mask"}

    pts = world_map[ys, xs].astype(np.float64)
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 1e-6)
    pts = pts[valid]
    result = {
        "centroid_pixel": [
            int(round(float(np.median(xs)))),
            int(round(float(np.median(ys)))),
        ],
        "n_pixels": int(mask.sum()),
        "n_valid": int(pts.shape[0]),
        "mask_resized_to_world_shape": False,
    }
    if pts.shape[0] < min_valid:
        result.update({
            "world_xyz": None,
            "world_error": f"too few valid depth pixels ({int(pts.shape[0])})",
        })
        return result

    result["world_xyz"] = [
        round(float(np.median(pts[:, 0])), 4),
        round(float(np.median(pts[:, 1])), 4),
        round(float(np.median(pts[:, 2])), 4),
    ]
    return result


def _make_segment_overlay(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    if image.ndim != 3 or image.shape[:2] != mask.shape:
        return None
    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[mask] = (
        0.55 * overlay[mask].astype(np.float32)
        + 0.45 * red[mask].astype(np.float32)
    ).astype(np.uint8)
    return overlay


@parallel
@readonly
def view_camera_meta(
    camera: str = "agentview",
    step: int = -1,
    *,
    state: EnvState,
) -> dict:
    """Read camera calibration metadata for localization."""
    if camera not in ("agentview", "wrist"):
        return {"error": f"bad camera '{camera}' (use 'agentview' or 'wrist')"}

    try:
        record = state.get(step)
        metadata_name = f"{camera}_metadata.json"
        if metadata_name not in record.artifacts:
            raise FileNotFoundError(metadata_name)
        meta = state.load(metadata_name, step=record.step_idx)
    except Exception as e:
        return {"error": f"{camera} camera metadata not found: {e}"}

    if camera == "agentview":
        return {"camera": "agentview", "camera_meta": meta}
    return {"camera": "wrist", "step": record.step_idx, "camera_meta": meta}


@parallel
@readonly
def back_project(
    row: int | None = None,
    col: int | None = None,
    step: int = -1,
    camera: str = "agentview",
    resolution: str = "high",
    row_range: list | None = None,
    col_range: list | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    *,
    state: EnvState,
) -> dict:
    """Look up a pixel's world XYZ in the precomputed world map."""
    if camera not in ("agentview", "wrist"):
        return {"error": f"bad camera '{camera}' (use 'agentview' or 'wrist')"}
    if resolution not in ("high", "low"):
        return {"error": f"bad resolution '{resolution}' (use 'high' or 'low')"}

    region_mode = row_range is not None or col_range is not None
    if not region_mode and (row is None or col is None):
        return {
            "error": (
                "provide either (row, col) for a single pixel, or "
                "row_range=[r0,r1] and col_range=[c0,c1] for a region center"
            )
        }

    try:
        record = state.get(step)
    except Exception as e:
        return {"error": f"state step not available: {e}"}
    nn = record.step_idx

    hi_artifact = f"{camera}_world_high.npz"
    low_artifact = f"{camera}_world.npz"
    source_artifact = hi_artifact if resolution == "high" else low_artifact
    if source_artifact not in record.artifacts:
        return {
            "error": (
                f"{camera} {resolution}-resolution world map not recorded "
                f"for step {nn}"
            )
        }

    try:
        world_map = state.load(str(source_artifact), step=nn)
    except Exception as e:
        return {
            "error": (
                f"{camera} {resolution}-resolution artifact not found "
                f"for step {nn}: {e}"
            )
        }

    height, width = world_map.shape[:2]

    if region_mode:
        if row_range is None or col_range is None:
            return {
                "error": "region mode needs BOTH row_range=[r0,r1] and col_range=[c0,c1]"
            }
        try:
            r0, r1 = int(row_range[0]), int(row_range[1])
            c0, c1 = int(col_range[0]), int(col_range[1])
        except Exception:
            return {"error": "row_range/col_range must each be [min, max] integers"}
        r0, r1 = sorted((max(0, r0), min(height, r1)))
        c0, c1 = sorted((max(0, c0), min(width, c1)))
        if r1 <= r0 or c1 <= c0:
            return {
                "error": (
                    f"empty region after clamping to image {height}x{width}: "
                    f"rows [{r0},{r1}] cols [{c0},{c1}]"
                )
            }
        window = world_map[r0:r1, c0:c1].reshape(-1, world_map.shape[2]).astype(
            np.float64
        )
        finite = np.isfinite(window).all(axis=1) & (
            np.abs(window[:, :3]).sum(axis=1) > 1e-6
        )
        pts = window[finite]
        n_total = int(pts.shape[0])
        if z_min is not None:
            pts = pts[pts[:, 2] >= float(z_min)]
        if z_max is not None:
            pts = pts[pts[:, 2] <= float(z_max)]
        if pts.shape[0] < 8:
            return {
                "error": (
                    f"too few valid pixels in region after z-filter "
                    f"({int(pts.shape[0])}); widen the window or the z band"
                ),
                "n_valid_before_zfilter": n_total,
            }
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        center = [
            round(float((xs.min() + xs.max()) / 2.0), 4),
            round(float((ys.min() + ys.max()) / 2.0), 4),
            round(float(np.median(zs)), 4),
        ]
        return {
            "camera": camera,
            "resolution": resolution,
            "mode": "region",
            "row_range": [r0, r1],
            "col_range": [c0, c1],
            "z_band": [z_min, z_max],
            "center_xyz": center,
            "median_xyz": [
                round(float(np.median(xs)), 4),
                round(float(np.median(ys)), 4),
                round(float(np.median(zs)), 4),
            ],
            "n_valid": int(pts.shape[0]),
            "step": nn,
            "image_size": [height, width],
            "source_artifact": source_artifact,
        }

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
            depth_artifact = f"{camera}_depth.npz"
            if depth_artifact not in record.artifacts:
                raise FileNotFoundError(depth_artifact)
            depth = state.load(depth_artifact, step=nn)
            if depth.ndim == 3:
                depth = depth[..., 0]
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

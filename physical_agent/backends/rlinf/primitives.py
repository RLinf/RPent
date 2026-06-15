"""LIBERO primitive driver: pick / place / move_to over a frozen Pi0.5 VLA.

Unlike BuilderBench's scripted keyframe-PD primitives, LIBERO primitives are
closed-loop VLM rollouts: each primitive overrides the language prompt fed to
the VLA, runs a fixed-length action chunk loop until a termination predicate
fires, and reports diagnostics (lift height, gripper opening, official success).

Usage (single-task smoke run):
    python -m physical_agent.backends.rlinf.primitives --task 0 --mode subinstr --seed 0

The driver bypasses Worker / Cluster — it constructs LiberoEnv and Pi0.5 directly
so the same code can be reused for offline evaluation, RL warm-starts, or
LLM-orchestrated multi-primitive scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from physical_agent.utils.config import get_repo_root, get_pi05_checkpoint_path
from physical_agent.backends import add_external_rlinf_to_path

PHYSICALAGENT_ROOT = get_repo_root()
RLINF_REPO_PATH = add_external_rlinf_to_path(PHYSICALAGENT_ROOT)
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.libero.libero_env import LiberoEnv

from physical_agent.backends.rlinf.vla_client import VLAClient


# ----- Config builders ----------------------------------------------------

CHECKPOINT_PATH = get_pi05_checkpoint_path()


def build_env_cfg(
    *,
    task_suite_name: str = "libero_spatial",
    specific_reset_id: int = 0,
    seed: int = 0,
    max_episode_steps: int = 240,
) -> Any:
    cfg = OmegaConf.create(
        {
            "env_type": "libero",
            "task_suite_name": task_suite_name,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_steps_per_rollout_epoch": max_episode_steps,
            "max_episode_steps": max_episode_steps,
            "use_rel_reward": False,
            "use_step_penalty": False,
            "reward_coef": 1.0,
            "reset_gripper_open": True,
            "is_eval": True,
            "seed": seed,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": True,
            "specific_reset_id": specific_reset_id,
            "video_cfg": {
                "save_video": True,
                "info_on_video": True,
                "video_base_dir": "/tmp/primitive_videos",
            },
            "init_params": {
                "camera_heights": 256,
                "camera_widths": 256,
                # Render depth too, so the perception-isolated protocol can
                # back-project pixels to world from depth + camera calibration
                # (no GT object poses). agentview_depth/robot0_eye_in_hand_depth
                # observables appear in raw_obs; toggled with the image
                # observables by LiberoEnv.set_image_render_enabled.
                "camera_depths": True,
                **({"robots": [os.environ["LIBERO_ROBOT_BASE"]]}
                   if os.environ.get("LIBERO_ROBOT_BASE") else {}),
            },
        }
    )
    return cfg


def build_model_cfg(model_path: str = CHECKPOINT_PATH) -> Any:
    model_path = model_path or get_pi05_checkpoint_path()
    if not model_path:
        raise RuntimeError(
            "PI05_CHECKPOINT_PATH is not set; provide the Pi0.5 checkpoint "
            "path via environment before launching LIBERO primitives."
        )
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": model_path,
            "precision": None,
            "num_action_chunks": 5,
            "action_dim": 7,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": 5,
            "add_value_head": False,
            "openpi": {
                "config_name": "pi05_libero",
                "num_images_in_input": 2,
                "noise_level": 0.5,
                "action_chunk": 5,
                "num_steps": 5,
                "train_expert_only": True,
                "action_env_dim": 7,
                "noise_method": "flow_sde",
                "add_value_head": False,
                "value_after_vlm": False,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": None,
                "use_dsrl": False,
            },
        }
    )


# ----- Primitive driver ---------------------------------------------------


@dataclass
class PrimitiveResult:
    name: str
    instruction: str
    success: bool
    chunks_used: int
    max_chunks: int
    peak_lift_m: float
    min_gripper_opening: float
    final_gripper_opening: float
    libero_terminated: bool = False
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "name": self.name,
            "instruction": self.instruction,
            "success": self.success,
            "chunks_used": self.chunks_used,
            "max_chunks": self.max_chunks,
            "peak_lift_m": round(self.peak_lift_m, 4),
            "min_gripper_opening": round(self.min_gripper_opening, 4),
            "final_gripper_opening": round(self.final_gripper_opening, 4),
            "libero_terminated": self.libero_terminated,
            "diagnostics": self.diagnostics,
        }


class LiberoPrimitiveDriver:
    """Wraps a single-env LiberoEnv + Pi0.5 VLA with primitive-level methods.

    `pick` and `place` reuse the VLA but override task_descriptions in the obs
    dict to a sub-instruction. `move_to` is scripted (no VLM call).
    """

    def __init__(
        self,
        env: LiberoEnv,
        model,
        action_chunk: int = 5,
        env_idx: int = 0,
    ):
        self.env = env
        self.model = model
        self.action_chunk = action_chunk
        self.env_idx = env_idx
        self._last_obs = None
        self._start_eef_z = None
        self._libero_terminated = False
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
            self._frames.append(self.render_agentview())

    def stop_recording_and_save(self, path: str, fps: int = 20,
                                 keep_recording: bool = False):
        import imageio.v2 as imageio
        import os as _os
        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        n = len(self._frames)
        if n > 0:
            imageio.mimwrite(path, self._frames, fps=fps)
        if not keep_recording:
            self._recording = False
            self._frames = []
        return {"path": path, "n_frames": n}

    # ---- helpers ----

    def _state(self, obs):
        s = obs["states"][self.env_idx]
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        return {
            "eef_pos": np.asarray(s[:3], dtype=np.float32),
            "eef_aa": np.asarray(s[3:6], dtype=np.float32),
            "gripper_qpos": np.asarray(s[6:8], dtype=np.float32),
        }

    def _eef_z(self, obs):
        return float(self._state(obs)["eef_pos"][2])

    def _gripper_opening(self, obs):
        # robosuite 2f85: qpos[6] in [~0, ~0.04], qpos[7] in [~-0.04, ~0].
        # Use |qpos[6]| + |qpos[7]| ≈ finger separation proxy.
        # When open ≈ 0.08; when closed ≈ 0.
        gp = self._state(obs)["gripper_qpos"]
        return float(abs(gp[0]) + abs(gp[1]))

    # ---- reset ----

    def reset(self):
        obs, info = self.env.reset()
        self._last_obs = obs
        self._start_eef_z = self._eef_z(obs)
        self._libero_terminated = False
        return obs, info

    # ---- VLM chunk step (used by pick / place) ----

    @torch.no_grad()
    def _vlm_chunk(self, instruction: str):
        """One Pi0.5 inference + one chunk of env steps. Overrides prompt."""
        obs = self._last_obs
        n_envs = obs["main_images"].shape[0]
        # Stash & override task_descriptions (one prompt per env).
        original_td = obs.get("task_descriptions")
        obs["task_descriptions"] = [instruction] * n_envs
        obs.setdefault("extra_view_images", None)

        actions, _result = self.model.predict_action_batch(obs, mode="eval")
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = actions

        # Pi0.5 emits float in normalized action space already unwrapped by
        # output_transform -> safe to feed directly. prepare_actions_for_libero
        # is a no-op for openpi.
        obs_list, _rew, term, _trunc, _info = self.env.chunk_step(actions_np)
        # Per-step rendering for diagnostic video. chunk_step ran action_chunk
        # env steps; we get one frame per chunk (since env.render is only valid
        # at chunk boundaries here). For finer detail use the per-env-step
        # branches in move_to / release / hold_gripper below.
        self.record_frame()
        # chunk_step returns obs_list of length action_chunk; take the latest.
        if isinstance(obs_list, (list, tuple)):
            self._last_obs = obs_list[-1]
        else:
            self._last_obs = obs_list
        # chunk_step returns terminations of shape [num_envs, chunk_size];
        # use .any() over the chunk dim, indexed by env.
        if isinstance(term, torch.Tensor):
            term_b = bool(term[self.env_idx].any().item())
        else:
            term_b = bool(np.asarray(term)[self.env_idx].any())
        if term_b:
            self._libero_terminated = True
        # Restore original task_descriptions on the obs dict for fairness with
        # future steps (no leaked state if caller switches primitives).
        if original_td is not None:
            self._last_obs["task_descriptions"] = original_td
        return self._last_obs

    # ---- primitives ----

    def pick(
        self,
        object_text: str,
        *,
        max_chunks: int = 24,
        lift_thresh: float = 0.05,
        gripper_closed_thresh: float = 0.06,
        instruction_template: str = "pick up the {obj}",
        track_obj: str = None,
        track_obj_lift_thresh: float = 0.05,
    ) -> PrimitiveResult:
        """Run VLA with prompt `instruction_template.format(obj=object_text)`.

        Success := eef lifted by >= lift_thresh AND gripper_opening below
        `gripper_closed_thresh`. Terminates early on libero `terminated`
        (official success) or max_chunks.
        """
        instr = instruction_template.format(obj=object_text)
        start_z = self._eef_z(self._last_obs)
        peak_z = start_z
        min_z = start_z
        # Track ascent AFTER min_z has been observed — descent then re-ascent
        # is the actual "lift" signal, distinct from raw |peak - min| which
        # also fires at the BOTTOM of the descent.
        post_min_peak_z = start_z
        min_grip = self._gripper_opening(self._last_obs)
        last_grip = min_grip
        descent_done = False
        success = False
        chunks_used = 0
        # Track an external object's z to break out as soon as it's lifted —
        # useful when you want Pi0.5 to ONLY pick (not place) and need a hard
        # mid-rollout stop right after the grasp.
        track_obj_init_z = None
        if track_obj is not None:
            raw = self.env.current_raw_obs[self.env_idx]
            track_obj_init_z = float(raw[f"{track_obj}_pos"][2])
        track_obj_lifted_to = None

        for c in range(max_chunks):
            self._vlm_chunk(instr)
            chunks_used = c + 1
            z = self._eef_z(self._last_obs)
            grip = self._gripper_opening(self._last_obs)
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
            # External-object lift signal (hard cut for hybrid LLM+VLA).
            if track_obj is not None:
                raw = self.env.current_raw_obs[self.env_idx]
                obj_z = float(raw[f"{track_obj}_pos"][2])
                track_obj_lifted_to = obj_z
                if (obj_z - track_obj_init_z) >= track_obj_lift_thresh:
                    success = True
                    break
            if self._libero_terminated:
                success = True
                break

        return PrimitiveResult(
            name="pick",
            instruction=instr,
            success=success,
            chunks_used=chunks_used,
            max_chunks=max_chunks,
            peak_lift_m=post_min_peak_z - min_z,  # actual post-descent ascent
            min_gripper_opening=min_grip,
            final_gripper_opening=last_grip,
            libero_terminated=self._libero_terminated,
            diagnostics={
                "start_eef_z": round(start_z, 4),
                "peak_eef_z": round(peak_z, 4),
                "min_eef_z": round(min_z, 4),
                "post_min_peak_z": round(post_min_peak_z, 4),
                "descent_m": round(start_z - min_z, 4),
                "post_min_ascent_m": round(post_min_peak_z - min_z, 4),
                "descent_done": descent_done,
                "lift_thresh": lift_thresh,
                "gripper_closed_thresh": gripper_closed_thresh,
                "track_obj": track_obj,
                "track_obj_init_z": track_obj_init_z,
                "track_obj_final_z": track_obj_lifted_to,
            },
        )

    def place(
        self,
        target_text: str,
        *,
        max_chunks: int = 24,
        release_thresh: float = 0.04,
        instruction_template: str = "place it on {tgt}",
    ) -> PrimitiveResult:
        """Run VLA with the place sub-instruction until gripper opens or budget."""
        instr = instruction_template.format(tgt=target_text)
        start_z = self._eef_z(self._last_obs)
        peak_z = start_z
        min_grip = self._gripper_opening(self._last_obs)
        last_grip = min_grip
        success = False
        chunks_used = 0

        for c in range(max_chunks):
            self._vlm_chunk(instr)
            chunks_used = c + 1
            z = self._eef_z(self._last_obs)
            grip = self._gripper_opening(self._last_obs)
            peak_z = max(peak_z, z)
            min_grip = min(min_grip, grip)
            last_grip = grip
            if grip >= release_thresh:
                success = True
                break
            if self._libero_terminated:
                success = True
                break

        return PrimitiveResult(
            name="place",
            instruction=instr,
            success=success,
            chunks_used=chunks_used,
            max_chunks=max_chunks,
            peak_lift_m=peak_z - start_z,
            min_gripper_opening=min_grip,
            final_gripper_opening=last_grip,
            libero_terminated=self._libero_terminated,
            diagnostics={"release_thresh": release_thresh},
        )

    # ---- scripted primitives (no VLM) ----

    def move_to(
        self,
        target_xyz,
        *,
        max_steps: int = 80,
        gripper_action: float = 1.0,
        step_clip: float = 0.025,
        tol: float = 0.012,
        action_scale: float = 0.05,
        target_yaw: float = None,
        yaw_step_clip: float = 0.10,
    ) -> dict:
        """Scripted EEF servo to a world-frame target xyz.

        Sends 7-D delta actions; the env's underlying OSC_POSE controller
        interprets ``action[:3] ∈ [-1, 1]`` as a per-step desired delta scaled
        by ``action_scale`` (so ``action=1.0`` -> ~5 cm per env step).
        gripper_action: +1.0 keeps it closed (holding object), -1.0 opens.
        """
        target = np.asarray(target_xyz, dtype=np.float32)
        traj = []
        for step in range(max_steps):
            cur = self._state(self._last_obs)["eef_pos"]
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
                q = self.env.current_raw_obs[self.env_idx]["robot0_eef_quat"]
                _R_mat = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
                cur_yaw = float(np.arctan2(_R_mat[1, 0], _R_mat[0, 0]))
                err = (float(target_yaw) - cur_yaw + np.pi) % (2 * np.pi) - np.pi
                step_dyaw = float(np.clip(err, -yaw_step_clip, yaw_step_clip))
                action[5] = float(np.clip(step_dyaw / 0.10, -1.0, 1.0))
            action[6] = gripper_action
            obs, _rew, term, _trunc, _info = self.env.step(action[None])
            self._last_obs = obs
            self.record_frame()
            if isinstance(term, torch.Tensor):
                if bool(term[self.env_idx].any().item()):
                    self._libero_terminated = True
                    break
        final = self._state(self._last_obs)["eef_pos"]
        return {
            "name": "move_to",
            "target_xyz": [float(x) for x in target],
            "final_eef_pos": [round(float(x), 4) for x in final],
            "final_dist_m": round(float(np.linalg.norm(target - final)), 4),
            "steps_used": len(traj),
            "max_steps": max_steps,
            "libero_terminated": self._libero_terminated,
        }

    def rotate_wrist(
        self,
        *,
        target_yaw: float = None,
        delta_yaw: float = None,
        gripper_action: float = 1.0,
        max_steps: int = 40,
        tol: float = 0.02,
        step_clip: float = 0.10,
    ) -> dict:
        """Rotate wrist around world z-axis. Provide EITHER target_yaw (absolute)
        or delta_yaw (relative, applied as a single rotation goal).

        Uses action[5] (axis-angle z component) to drive wrist yaw via the
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
            # Empirically verified via /tmp/probe_yaw.py: passing the raw
            # quat directly to _R.from_quat gives a matrix whose col0 is
            # the eef x-axis expressed in the world frame.
            q = quat_xyzw
            rot = _R.from_quat([q[0], q[1], q[2], q[3]])
            R = rot.as_matrix()
            # World-frame yaw: angle of the eef x-axis projected onto the
            # world xy plane. Robust to gripper-down (R[2,2]≈-1) which is
            # where the euler 'zyx' chart flips sign.
            return float(np.arctan2(R[1, 0], R[0, 0]))

        raw = self.env.current_raw_obs[self.env_idx]
        cur_quat = raw["robot0_eef_quat"]
        start_yaw = _yaw_of(cur_quat)
        if target_yaw is None and delta_yaw is None:
            return {"name": "rotate_wrist", "error": "need target_yaw or delta_yaw"}
        if target_yaw is None:
            target_yaw = start_yaw + float(delta_yaw)

        traj = []
        for step in range(max_steps):
            raw = self.env.current_raw_obs[self.env_idx]
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
            action[6] = float(gripper_action)
            obs, _r, term, _t, _i = self.env.step(action[None])
            self._last_obs = obs
            self.record_frame()
            if isinstance(term, torch.Tensor) and bool(term[self.env_idx].any().item()):
                self._libero_terminated = True
                break
        final_yaw = _yaw_of(self.env.current_raw_obs[self.env_idx]["robot0_eef_quat"])
        return {
            "name": "rotate_wrist",
            "start_yaw": round(start_yaw, 4),
            "target_yaw": round(float(target_yaw), 4),
            "final_yaw": round(final_yaw, 4),
            "final_err": round(float((target_yaw - final_yaw + np.pi) % (2 * np.pi) - np.pi), 4),
            "steps_used": len(traj),
            "libero_terminated": self._libero_terminated,
        }

    def rotate_pitch(
        self,
        *,
        target_pitch: float = None,
        delta_pitch: float = None,
        gripper_action: float = 1.0,
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

        raw = self.env.current_raw_obs[self.env_idx]
        start_pitch = _pitch_of(raw["robot0_eef_quat"])
        if target_pitch is None and delta_pitch is None:
            return {"name": "rotate_pitch",
                    "error": "need target_pitch or delta_pitch"}
        if target_pitch is None:
            target_pitch = start_pitch + float(delta_pitch)

        traj = []
        for step in range(max_steps):
            raw = self.env.current_raw_obs[self.env_idx]
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
            action[6] = float(gripper_action)
            obs, _r, term, _t, _i = self.env.step(action[None])
            self._last_obs = obs
            self.record_frame()
            if isinstance(term, torch.Tensor) and bool(term[self.env_idx].any().item()):
                self._libero_terminated = True
                break
        final_pitch = _pitch_of(self.env.current_raw_obs[self.env_idx]["robot0_eef_quat"])
        return {
            "name": "rotate_pitch",
            "start_pitch": round(start_pitch, 4),
            "target_pitch": round(float(target_pitch), 4),
            "final_pitch": round(final_pitch, 4),
            "final_err": round(float(
                (target_pitch - final_pitch + np.pi) % (2 * np.pi) - np.pi), 4),
            "steps_used": len(traj),
            "libero_terminated": self._libero_terminated,
        }

    def move_pose(
        self,
        target_xyz,
        *,
        target_pitch: float = None,
        target_yaw: float = None,
        gripper_action: float = -1.0,
        step_clip: float = 0.02,
        pitch_step: float = 0.08,
        yaw_step: float = 0.08,
        tol: float = 0.012,
        ori_tol: float = 0.05,
        action_scale: float = 0.05,
        max_steps: int = 150,
    ) -> dict:
        """Servo position AND orientation (pitch + yaw) SIMULTANEOUSLY.

        Unlike move_to (holds orientation) + rotate_pitch (holds xyz), this
        co-varies xyz and wrist tilt every env.step. Co-variation lets the
        OSC controller thread cabinet-front-low poses where a decoupled
        position servo (fixed gripper-down orientation) drives the wrist
        into a singularity and stalls — mimicking pi0's curved reach-in.
        """
        from scipy.spatial.transform import Rotation as _R

        def _pitch_of(q):
            R = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
            return float(np.arctan2(R[1, 2], -R[2, 2]))

        def _yaw_of(q):
            R = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
            return float(np.arctan2(R[1, 0], R[0, 0]))

        target = np.asarray(target_xyz, dtype=np.float32)
        traj = []
        step = 0
        for step in range(max_steps):
            cur = self._state(self._last_obs)["eef_pos"]
            q = self.env.current_raw_obs[self.env_idx]["robot0_eef_quat"]
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
            action[6] = float(gripper_action)
            obs, _r, term, _t, _i = self.env.step(action[None])
            self._last_obs = obs
            self.record_frame()
            if isinstance(term, torch.Tensor) and bool(term[self.env_idx].any().item()):
                self._libero_terminated = True
                break
        final = self._state(self._last_obs)["eef_pos"]
        fq = self.env.current_raw_obs[self.env_idx]["robot0_eef_quat"]
        return {
            "name": "move_pose",
            "final_eef_pos": [round(float(x), 4) for x in final],
            "final_dist_m": round(float(np.linalg.norm(target - final)), 4),
            "final_pitch": round(_pitch_of(fq), 4),
            "steps_used": step + 1,
            "libero_terminated": self._libero_terminated,
        }

    def release(
        self,
        *,
        max_steps: int = 20,
        hold_pos: bool = True,
    ) -> dict:
        """Open gripper for `max_steps` env steps, optionally keeping eef in place.

        Returns once libero terminates (success) or step budget exhausted.
        """
        start_grip = self._gripper_opening(self._last_obs)
        peak_grip = start_grip
        for step in range(max_steps):
            action = np.zeros(7, dtype=np.float32)
            action[6] = -1.0  # open
            obs, _rew, term, _trunc, _info = self.env.step(action[None])
            self._last_obs = obs
            self.record_frame()
            peak_grip = max(peak_grip, self._gripper_opening(self._last_obs))
            if isinstance(term, torch.Tensor):
                if bool(term[self.env_idx].any().item()):
                    self._libero_terminated = True
                    break
        return {
            "name": "release",
            "steps_used": step + 1,
            "start_gripper_opening": round(start_grip, 4),
            "peak_gripper_opening": round(peak_grip, 4),
            "final_gripper_opening": round(self._gripper_opening(self._last_obs), 4),
            "libero_terminated": self._libero_terminated,
        }

    # ---- introspection helpers (for LLM-in-the-loop) ----

    def render_agentview(self) -> np.ndarray:
        """Return an HxWx3 uint8 RGB image of the agentview camera (180° rotated
        to match Pi0.5 training convention)."""
        raw = self.env.current_raw_obs[self.env_idx]
        img = raw["agentview_image"]
        return img[::-1, ::-1]  # same flip as rlinf.envs.libero.utils.get_libero_image

    def get_privileged_state(self) -> dict:
        """Pull world-frame positions of EEF + all named objects from raw_obs.

        These keys come straight from libero's robosuite observables (e.g.
        plate_1_pos, akita_black_bowl_1_pos). Use them as 'ground truth' for
        LLM-in-the-loop planning or post-hoc evaluation.
        """
        raw = self.env.current_raw_obs[self.env_idx]
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
        """Just run the VLA with the ORIGINAL task_descriptions, no override.

        Used as the baseline (Pi0.5 in its natural prompting mode)."""
        chunks_used = 0
        start_z = self._eef_z(self._last_obs)
        peak_z = start_z
        for c in range(max_chunks):
            # _vlm_chunk overrides prompt; but we want the ORIGINAL. Bypass.
            obs = self._last_obs
            obs.setdefault("extra_view_images", None)
            with torch.no_grad():
                actions, _ = self.model.predict_action_batch(obs, mode="eval")
            actions_np = (
                actions.detach().cpu().numpy()
                if isinstance(actions, torch.Tensor)
                else actions
            )
            obs_list, _rew, term, _trunc, _info = self.env.chunk_step(actions_np)
            self._last_obs = obs_list[-1] if isinstance(obs_list, (list, tuple)) else obs_list
            chunks_used = c + 1
            peak_z = max(peak_z, self._eef_z(self._last_obs))
            term_b = (
                bool(term[self.env_idx].any().item())
                if isinstance(term, torch.Tensor)
                else bool(np.asarray(term)[self.env_idx].any())
            )
            if term_b:
                self._libero_terminated = True
                break
        return {
            "name": "full_task",
            "instruction": self._last_obs.get("task_descriptions", [""])[self.env_idx]
            if isinstance(self._last_obs.get("task_descriptions"), list)
            else "",
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "peak_lift_m": peak_z - start_z,
            "final_gripper_opening": self._gripper_opening(self._last_obs),
            "libero_terminated": self._libero_terminated,
        }


# ----- Single-env factory --------------------------------------------------


def build_driver(
    *,
    task_id: int,
    model: VLAClient,
    seed: int = 0,
):
    """Build a single-env LiberoPrimitiveDriver pinned to one libero_spatial task.

    The caller owns the ``VLAClient`` (and the matching ``vla_server``
    process behind it).

    Uses the task's first trial (specific_reset_id = task_id * trials_per_task
    is not what we want — instead we lock task via specific_task_id-style trick:
    we pick the first reset state whose task_id matches).
    """
    # Compute a reset_state_id whose task_id == task_id (uses suite's bins).
    from libero.libero.benchmark import get_benchmark

    suite = get_benchmark("libero_spatial")()
    # cumulative trial counts up to task_id give the first reset state for that task.
    first_id = 0
    for t in range(task_id):
        first_id += len(suite.get_task_init_states(t))

    env_cfg = build_env_cfg(
        task_suite_name="libero_spatial",
        specific_reset_id=first_id + (seed % len(suite.get_task_init_states(task_id))),
        seed=seed,
    )
    env = LiberoEnv(
        cfg=env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )

    driver = LiberoPrimitiveDriver(env=env, model=model, action_chunk=5)
    return driver, env


# ----- CLI smoke-run -------------------------------------------------------


def parse_task_object(task_id: int) -> tuple[str, str]:
    """Pull (object_text, target_text) out of the libero_spatial task language."""
    from libero.libero.benchmark import get_benchmark

    suite = get_benchmark("libero_spatial")()
    lang = suite.get_task(task_id).language
    m = re.match(r"pick up the (.+?) and place it on (.+)", lang)
    if not m:
        return ("", "")
    return m.group(1), m.group(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, default=0, help="libero_spatial task id 0-9")
    p.add_argument(
        "--mode",
        choices=["full", "subinstr", "pick_only"],
        default="pick_only",
        help="full=baseline; subinstr=pick->place sub-instructions; "
        "pick_only=just pick sub-instruction",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_chunks_pick", type=int, default=24)
    p.add_argument("--max_chunks_place", type=int, default=24)
    p.add_argument("--max_chunks_full", type=int, default=48)
    p.add_argument("--out", type=str, default=None, help="JSON result path")
    p.add_argument("--vla-endpoint", default=None,
                   help="Base URL for a remote Pi0.5 VLA /predict server "
                        "(e.g. http://localhost:8000). When unset, the model "
                        "is loaded in-process via get_openpi_model.")
    args = p.parse_args()

    t0 = time.time()
    obj, tgt = parse_task_object(args.task)
    print(f"[task {args.task}] obj=\"{obj}\"  tgt=\"{tgt}\"")

    local_server = None
    vla_endpoint = args.vla_endpoint
    if vla_endpoint is None:
        import atexit
        from physical_agent.backends.rlinf.vla_server import spawn_local_server
        vla_endpoint, local_server = spawn_local_server()
        atexit.register(local_server.terminate)
        print(f"[setup] spawned local vla_server at {vla_endpoint}; "
              "waiting for /healthz ...")

    model = VLAClient(vla_endpoint)
    model.wait_for_healthz(timeout_s=60.0)
    print(f"[setup] vla_server ready at {vla_endpoint}")

    driver, env = build_driver(
        task_id=args.task,
        seed=args.seed,
        model=model,
    )
    print(f"[setup] driver built in {time.time() - t0:.1f}s")

    obs, _ = driver.reset()
    start_td = obs.get("task_descriptions", [""])[0]
    print(f"[reset] start_eef_z={driver._start_eef_z:.4f}m  full_prompt=\"{start_td}\"")

    out = {"task_id": args.task, "seed": args.seed, "mode": args.mode,
           "object_text": obj, "target_text": tgt,
           "full_prompt": start_td, "start_eef_z": driver._start_eef_z}

    if args.mode == "full":
        r = driver.run_full_task(max_chunks=args.max_chunks_full)
        out["full_task"] = r
        print(f"[full] {r}")
    elif args.mode == "pick_only":
        r = driver.pick(obj, max_chunks=args.max_chunks_pick)
        out["pick"] = r.to_dict()
        print(f"[pick] {r.to_dict()}")
    elif args.mode == "subinstr":
        r1 = driver.pick(obj, max_chunks=args.max_chunks_pick)
        out["pick"] = r1.to_dict()
        print(f"[pick] {r1.to_dict()}")
        r2 = driver.place(tgt, max_chunks=args.max_chunks_place)
        out["place"] = r2.to_dict()
        print(f"[place] {r2.to_dict()}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()

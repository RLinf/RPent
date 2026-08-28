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

"""RLDX-1 closed-loop grasp/contact skill for the RoboCasa agent.

Loads RLDX-1 once (in the sim venv) and drives it for a few action chunks against
the live cameras — the equivalent of LIBERO's pi0_pick: a reusable VLA "delivery
service" for the grasp, after which the LLM scripts the carry+place.

Obs format mirrors robocasa's gym get_observation (state.* + video.{3 cams} +
annotation), stacked over the policy's video_delta_indices history, batch dim = 1,
with a per-call session id + reset_memory for the RLDX memory module.
"""

import os
import uuid
from collections import deque
from collections.abc import Mapping

import imageio.v2 as imageio
import numpy as np

from robots.robocasa.env_client import RoboCasaEnvClient

ACTION_FIELDS = {
    "action.end_effector_position": 3,
    "action.end_effector_rotation": 3,
    "action.gripper_close": 1,
    "action.base_motion": 4,
    "action.control_mode": 1,
}
STATE_FIELDS = {
    "state.gripper_qpos": ("robot0_gripper_qpos", 2),
    "state.base_position": ("robot0_base_pos", 3),
    "state.base_rotation": ("robot0_base_quat", 4),
    "state.end_effector_position_relative": ("robot0_base_to_eef_pos", 3),
    "state.end_effector_rotation_relative": ("robot0_base_to_eef_quat", 4),
}
MAX_VLA_CHUNKS = 1_000
MAX_ACTION_STEPS = 512


def _bounded_int(value, name, *, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0 or value > maximum:
        raise ValueError(f"{name} must be in [1, {maximum}], got {value}")
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


def _numeric_vector(value, name, size, *, dtype=np.float64):
    array = np.asarray(value)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain numeric values")
    array = array.astype(dtype, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _identity_token(value):
    return "".join(ch for ch in str(value) if ch.isalnum())[:16] or "unknown"


class RLDXSkill:
    def __init__(
        self, env_client: RoboCasaEnvClient, vla_client=None, check_cancelled=None
    ):
        self.env = env_client  # RoboCasaEnvClient
        self._vla_client = vla_client  # VLA RPC client (when set, _load() uses it instead of loading the model directly)
        self._check_cancelled = (
            check_cancelled  # optional cancellation checkpoint callback
        )
        self._vdi = None  # video delta indices, e.g. [-6,-4,-2,0]
        self._hist = None  # deque of raw frame dicts
        env_runtime = _identity_token(getattr(env_client, "runtime_id", "env"))
        vla_runtime = _identity_token(getattr(vla_client, "runtime_id", "vla"))
        self._sid = f"rc:{env_runtime}:{vla_runtime}:{uuid.uuid4().hex}"
        self._unmap = None  # lazy: eval's PandaOmronKeyConverter.unmap_action
        # OPTIONAL per-sim-step video capture. OFF by default (env RLDX_VIDEO_DIR unset):
        # the VLA rollout is closed-loop over 100s of sim-steps but the primitives only dump
        # single frames at command boundaries, so the actual motion is never recorded.
        # When RLDX_VIDEO_DIR is set, we collect the 3 VLA camera frames (already rendered
        # for the obs — ZERO extra render cost) per step and write one mp4 per run() call.
        self._video_dir = os.environ.get("RLDX_VIDEO_DIR") or None
        try:
            video_fps = int(os.environ.get("RLDX_VIDEO_FPS", "20"))
        except ValueError as exc:
            raise ValueError("RLDX_VIDEO_FPS must be an integer") from exc
        self._video_fps = _bounded_int(video_fps, "RLDX_VIDEO_FPS", maximum=240)
        self._video_idx = 0  # cmd counter for cmd_NN.mp4 naming
        self._frames = None  # list of HxWx3 uint8 for the current run() call

    def _load(self):
        if self._vdi is not None:
            return
        if self._vla_client is None:
            raise RuntimeError("RLDXSkill requires a RoboCasaVLAClient")
        action_schema = getattr(self._vla_client, "action_schema", None)
        if not isinstance(action_schema, Mapping):
            raise RuntimeError(
                "VLA client has not completed its action-schema handshake"
            )
        if action_schema.get("flat_dim") != self.env.action_dim:
            raise RuntimeError(
                f"VLA/env action dimension mismatch: VLA={action_schema.get('flat_dim')!r}, "
                f"env={self.env.action_dim}"
            )
        fields = action_schema.get("fields")
        actual_fields = {
            item.get("name"): item.get("size")
            for item in fields or []
            if isinstance(item, Mapping)
        }
        if actual_fields != ACTION_FIELDS:
            raise RuntimeError(
                f"VLA action fields mismatch: expected={ACTION_FIELDS!r}, "
                f"actual={actual_fields!r}"
            )
        mod = self._vla_client.get_modality_config()
        self._vdi = np.asarray(mod["video_delta_indices"])
        if (
            self._vdi.ndim != 1
            or self._vdi.size == 0
            or not np.issubdtype(self._vdi.dtype, np.integer)
            or np.any(np.diff(self._vdi) <= 0)
            or int(self._vdi[-1]) != 0
        ):
            raise RuntimeError(f"invalid video_delta_indices={self._vdi!r}")
        self._vdi = self._vdi.astype(np.int64, copy=False)
        expected_hist = int(self._vdi.max() - self._vdi.min()) + 2
        if mod["hist_maxlen"] != expected_hist:
            raise RuntimeError(
                f"hist_maxlen={mod['hist_maxlen']!r}, expected {expected_hist}"
            )
        self._hist = deque(maxlen=expected_hist)
        print(
            f"[rldx_skill] modality loaded via RPC; video_delta_indices={self._vdi.tolist()} "
            f"hist_maxlen={self._hist.maxlen}",
            flush=True,
        )

    # ---- obs construction (mirror robocasa gym get_observation) ----
    VLA_OBS_RES = (
        256  # matches the eval: robocasa365 gym_wrapper get_camera_config renders
    )
    # the 3 VLA cameras at 256 (overriding create_env's 128 DEFAULT).
    # Verified: eval gym obs are (256,256,3) and primitives@256 is byte-identical.

    def _raw_frame(self, task_text):
        if not isinstance(task_text, str) or not task_text.strip():
            raise ValueError("task_text must be a non-empty string")
        e = self.env
        o = e.current_raw_obs
        # render the 3 VLA cameras at the eval's resolution (256)
        r = self.VLA_OBS_RES
        vL = e.render_camera("robot0_agentview_left", height=r, width=r)
        vR = e.render_camera("robot0_agentview_right", height=r, width=r)
        vW = e.render_camera("robot0_eye_in_hand", height=r, width=r)
        frame = {
            output_name: _numeric_vector(
                o[input_name], input_name, size, dtype=np.float32
            )
            for output_name, (input_name, size) in STATE_FIELDS.items()
        }
        for name, image in (
            ("video.robot0_agentview_left", vL),
            ("video.robot0_agentview_right", vR),
            ("video.robot0_eye_in_hand", vW),
        ):
            image = np.asarray(image)
            if image.shape != (r, r, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"{name} must be uint8 with shape {(r, r, 3)}, "
                    f"got dtype={image.dtype}, shape={image.shape}"
                )
            frame[name] = image
        frame["annotation.human.task_description"] = task_text
        return frame

    def _seed_hist(self, task_text):
        """Pad the per-sim-step history with copies of the CURRENT frame, mirroring the
        eval MultiStepWrapper.reset (which fills its obs deque with n copies of obs0)."""
        frame = self._raw_frame(task_text)
        self._hist.clear()
        for _ in range(self._hist.maxlen):
            self._hist.append(frame)

    def _record_frame(self, task_text):
        """Append ONE frame for the just-executed sim-step (call after every env.step)."""
        self._hist.append(self._raw_frame(task_text))

    def _capture_video_frame(self):
        """Collect the 3 VLA camera frames (left|right|wrist) already rendered into the
        latest history entry and stack them side-by-side. Called after every env.step
        when RLDX_VIDEO_DIR is set. Reuses the obs render -> no extra render cost."""
        if self._frames is None or not self._hist:
            return
        f = self._hist[-1]
        try:
            tiles = [
                f["video.robot0_agentview_left"],
                f["video.robot0_agentview_right"],
                f["video.robot0_eye_in_hand"],
            ]
            self._frames.append(np.concatenate(tiles, axis=1))  # (H, 3W, 3) uint8
        except Exception:
            pass

    def _flush_video(self):
        """Write the collected frames of the current run() call to cmd_NN.mp4."""
        if self._video_dir is None or not self._frames:
            self._frames = None
            return None
        os.makedirs(self._video_dir, exist_ok=True)
        path = os.path.join(self._video_dir, f"cmd_{self._video_idx:02d}.mp4")
        try:
            imageio.mimwrite(
                path,
                self._frames,
                fps=self._video_fps,
                codec="libx264",
                macro_block_size=1,
            )
            print(
                f"[rldx_skill] wrote video ({len(self._frames)} frames) -> {path}",
                flush=True,
            )
        except Exception as e:
            print(f"[rldx_skill] video write failed: {e}", flush=True)
            path = None
        self._frames = None
        return path

    def _build_obs(self, task_text):
        """Build the batched (n_envs=1), history-stacked obs by indexing the per-sim-step
        history at vdi-1 = [-7,-5,-3,-1], EXACTLY like the eval MultiStepWrapper._get_obs
        (delta_indices = video_delta_indices - 1, over self.obs[i])."""
        if self._hist is None or not self._hist:
            raise RuntimeError("VLA observation history has not been seeded")
        buf = list(self._hist)
        idx = [len(buf) + (int(d) - 1) for d in self._vdi]  # vdi-1 offset from end
        idx = [max(0, min(len(buf) - 1, k)) for k in idx]
        cur = buf[-1]  # state/annotation = current
        obs = {}
        for k, v in cur.items():
            if k.startswith("video."):
                stack = np.stack([buf[j][k] for j in idx], axis=0)  # (T,H,W,3)
                obs[k] = stack[None]  # (1,T,H,W,3)
            elif k.startswith("state."):
                obs[k] = np.asarray(v)[None][None]  # (1,1,D)
            else:
                obs[k] = [task_text]  # (1,) annotation
        return obs

    def _build_env_action(
        self, eef_pos, eef_rot, gripper_close, base_motion, control_mode
    ):
        """Assemble the native robosuite action EXACTLY like the EVAL gym wrapper
        (robocasa365 wrappers/gym_wrapper.py: PandaOmronKeyConverter.unmap_action +
        composite-controller split-index assembly). We route the VLA through the eval's
        OWN conversion instead of a hand-written 12-d concat so the policy sees the
        IDENTICAL action mode it was trained/evaluated with. This is where gripper_close
        and control_mode get binarized to +-1 (raw [0,1] -> -1 if <0.5 else +1);
        re-implementing that by hand and getting it wrong was the original no-grasp bug."""
        if self._unmap is None:
            from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter

            self._unmap = PandaOmronKeyConverter.unmap_action
        ad = self._unmap(
            {
                "action.end_effector_position": _numeric_vector(
                    eef_pos, "action.end_effector_position", 3
                ),
                "action.end_effector_rotation": _numeric_vector(
                    eef_rot, "action.end_effector_rotation", 3
                ),
                "action.gripper_close": _numeric_vector(
                    gripper_close, "action.gripper_close", 1
                ),
                "action.base_motion": _numeric_vector(
                    base_motion, "action.base_motion", 4
                ),
                "action.control_mode": _numeric_vector(
                    control_mode, "action.control_mode", 1
                ),
            }
        )
        return self.env.reassemble_env_action(ad)

    @staticmethod
    def _validate_actions(actions):
        if not isinstance(actions, Mapping):
            raise TypeError("VLA predict must return an action mapping")
        normalized = {}
        horizon = None
        for name, size in ACTION_FIELDS.items():
            if name not in actions:
                raise ValueError(f"VLA action output is missing {name!r}")
            array = np.asarray(actions[name])
            if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != size:
                raise ValueError(
                    f"{name} must have shape (1, horizon, {size}), got {array.shape}"
                )
            if not 0 < array.shape[1] <= MAX_ACTION_STEPS:
                raise ValueError(f"{name} has invalid horizon {array.shape[1]}")
            if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
                array.dtype, np.bool_
            ):
                raise TypeError(f"{name} must contain numeric values")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
            if horizon is None:
                horizon = array.shape[1]
            elif array.shape[1] != horizon:
                raise ValueError("VLA action fields have inconsistent horizons")
            normalized[name] = array
        return normalized, horizon

    def _grasp_contact(self):
        """Direction-AGNOSTIC grasp check: True iff BOTH gripper fingerpads are in
        contact with the SAME task object (robosuite `_check_grasp`, the same primitive
        the benchmark uses). Unlike `peak_lift` (which only sees Z-axis lift and MISSES
        sideways/horizontal grasps — a bar, a handle, a lateral pull — and anything
        grasped-but-not-yet-raised), this fires for ANY grasp orientation and even for
        holding a static object. Returns (grasping: bool, obj_name: str|None)."""
        return self.env.grasp_contact()

    # ---- the skill ----
    def run(
        self,
        prompt,
        max_chunks=15,
        n_action_steps=8,
        base_clip=None,
        settle_eps=0.012,
        settle_patience=2,
        force_reset=False,
        recording=False,
        record_frame=None,
    ):
        """Drive RLDX-1 closed-loop until it FINISHES, not a fixed tiny budget. The VLA
        has no terminate signal (like the eval, which runs to env-success), so we stop
        on a completion criterion and report WHY — so a closed-empty gripper means the
        VLA actually executed and missed, not that we cut it off mid-grasp.
        Stops when: env `success` fires / the VLA SETTLES (eef+gripper stop changing for
        `settle_patience` chunks = it's done) / `max_chunks` safety cap.
        base_clip=None -> full base motion; base_clip=v -> clamp base_motion to [-v,v]
        (whole-body policy: never zero it). Returns status + grasp signals so the LLM
        decides: continue (call again) vs done vs genuinely-failed."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        max_chunks = _bounded_int(max_chunks, "max_chunks", maximum=MAX_VLA_CHUNKS)
        n_action_steps = _bounded_int(
            n_action_steps, "n_action_steps", maximum=MAX_ACTION_STEPS
        )
        settle_patience = _bounded_int(
            settle_patience, "settle_patience", maximum=MAX_VLA_CHUNKS
        )
        settle_eps = _finite_scalar(
            settle_eps, "settle_eps", minimum=np.finfo(float).eps, maximum=1.0
        )
        if base_clip is not None:
            base_clip = _finite_scalar(base_clip, "base_clip", minimum=0.0, maximum=1.0)
        if not isinstance(force_reset, (bool, np.bool_)):
            raise TypeError("force_reset must be boolean")
        if not isinstance(recording, (bool, np.bool_)):
            raise TypeError("recording must be boolean")
        if recording and not callable(record_frame):
            raise TypeError("record_frame must be callable when recording=True")
        self._load()
        # Start a fresh video buffer for this run() call (no-op if RLDX_VIDEO_DIR unset).
        self._frames = [] if self._video_dir is not None else None
        # Reset the VLA memory + frame history ONLY when the INSTRUCTION CHANGES (a new
        # sub-task). Same-prompt calls (retrying one grasp) KEEP memory continuity — the
        # grasp needs it (per-call reset regressed grasping to 0). A switched instruction
        # ("Pick the pan" -> "Turn on the faucet") with stale memory makes the VLA idle,
        # so reset there. (The eval never switches instruction mid-episode.)
        # force_reset=True -> treat this call as a brand-new episode (clears the session's
        # RTC chunk + memory_tokens + frame history). Use this when the ENV was reset under
        # the same instruction (e.g. multi-trial fullshot comparison): the eval resets the
        # policy session EVERY episode, so without this, stale RTC/memory from the prior
        # episode bleeds into the next and degrades it.
        new_task = force_reset or (prompt != getattr(self, "_last_prompt", None))
        self._last_prompt = prompt
        # Reseed the per-sim-step history with the current frame on a new instruction or
        # the first call ever; same-prompt retries KEEP the continuous history (matches
        # the eval, which never switches instruction and never re-seeds mid-episode).
        if new_task or not self._hist:
            self._seed_hist(prompt)
        fresh = new_task
        base0 = np.asarray(self.env.current_raw_obs["robot0_base_pos"], np.float64)[
            :2
        ].copy()
        eef_prev = np.asarray(self.env.eef_pos, np.float64).copy()
        grip_prev = float(self.env.gripper_qpos[0])
        eef_min_z = float(self.env.eef_pos[2])
        peak_lift = 0.0
        applied = 0
        settled = 0
        status = "cap"
        chunks_executed = 0
        success_latched = bool(self.env.check_success())
        if success_latched:
            status = "success"
        grasp_ever = False
        grasp_obj = None
        last_cmd_close = False
        for _ in range(max_chunks if not success_latched else 0):
            chunks_executed += 1
            obs = self._build_obs(prompt)
            options = {"reset_memory": [fresh], "session_ids": [self._sid]}
            fresh = False
            if self._check_cancelled is not None:
                self._check_cancelled()
            actions = self._vla_client.predict(obs, options)
            actions, horizon = self._validate_actions(actions)
            # gym Dict -> native flat 12-d [eef_pos(3),eef_rot(3),gripper(1),base(4),mode(1)]
            for step in range(min(n_action_steps, horizon)):
                base_motion = np.asarray(actions["action.base_motion"])[0, step]
                if base_clip is not None:
                    # rldx_arm only: clamp base velocities so the VLA micro-aligns but
                    # can't drive off (eval/rldx_skill pass base_clip=None = full motion).
                    base_motion = np.clip(base_motion, -base_clip, base_clip)
                # Build the action through the EVAL's own conversion (binarizes gripper +
                # control_mode); identical to how the policy is evaluated. See _build_env_action.
                last_cmd_close = (
                    float(
                        np.asarray(actions["action.gripper_close"])[0, step].reshape(
                            -1
                        )[0]
                    )
                    >= 0.5
                )
                a = self._build_env_action(
                    np.asarray(actions["action.end_effector_position"])[0, step],
                    np.asarray(actions["action.end_effector_rotation"])[0, step],
                    np.asarray(actions["action.gripper_close"])[0, step],
                    base_motion,
                    np.asarray(actions["action.control_mode"])[0, step],
                )
                if self._check_cancelled is not None:
                    self._check_cancelled()
                self.env.step(a)
                if recording:
                    record_frame()
                applied += 1
                self._record_frame(
                    prompt
                )  # PER-SIM-STEP history (matches eval cadence)
                self._capture_video_frame()  # OPTIONAL video (no-op if RLDX_VIDEO_DIR unset)
                if self.env.check_success():
                    status = "success"
                    success_latched = True
                    break
            if success_latched:
                break
            # ROBUST grasp check once per chunk (direction-agnostic; see _grasp_contact)
            g_now, g_obj = self._grasp_contact()
            if g_now:
                grasp_ever = True
                grasp_obj = grasp_obj or g_obj
            eef_now = np.asarray(self.env.eef_pos, np.float64)
            grip_now = float(self.env.gripper_qpos[0])
            eef_min_z = min(eef_min_z, float(eef_now[2]))
            peak_lift = max(peak_lift, float(eef_now[2] - eef_min_z))
            if (
                float(np.linalg.norm(eef_now - eef_prev)) < settle_eps
                and abs(grip_now - grip_prev) < 0.003
            ):
                settled += 1
                if settled >= settle_patience:
                    status = "settled"
                    break  # VLA idle = finished its action
            else:
                settled = 0
            eef_prev = eef_now.copy()
            grip_prev = grip_now
        grip = float(self.env.gripper_qpos[0])
        base1 = np.asarray(self.env.current_raw_obs["robot0_base_pos"], np.float64)[:2]
        # ---- ROBUST, DIRECTION-AGNOSTIC GRASP DETERMINATION ----
        # Primary (gold): fingerpad↔object CONTACT now (works for any grasp orientation,
        # incl. sideways; does NOT depend on Z-lift). Secondary: the commanded-close-but-
        # held-apart signal — the gripper was told to CLOSE but the fingers stopped before
        # fully-closed-empty (~0), i.e. something is wedged between them. This catches
        # holding a FIXTURE handle (drawer/door) that isn't in env.objects. We expose all
        # signals so the caller never has to trust a single brittle proxy (e.g. peak_lift).
        grasp_now, gobj_now = self._grasp_contact()
        grasp_obj = gobj_now or grasp_obj
        held_apart = bool(
            last_cmd_close and 0.004 < grip < 0.039
        )  # closed onto something
        grasped_now = bool(grasp_now or held_apart)  # holding at END of call
        grasp_detected = bool(grasp_ever or grasped_now)  # held at SOME point
        video_path = self._flush_video()  # write cmd_NN.mp4 (no-op if disabled)
        self._video_idx += 1
        result = {
            "ok": True,
            "prompt": prompt,
            "status": status,
            "chunks": chunks_executed,
            "steps_applied": applied,
            "grasped": grasped_now,  # holding at end-of-call (carry-ready)
            "grasp_detected": grasp_detected,  # grabbed at some chunk (may have placed since)
            "grasp_contact": bool(
                grasp_now
            ),  # fingerpad↔object contact (gold standard)
            "held_apart": held_apart,  # commanded-close + fingers wedged apart
            "grasp_obj": grasp_obj,  # which task object, if contact-identified
            "gripper_qpos": round(grip, 3),
            "peak_lift": round(peak_lift, 3),
            "base_clip": base_clip,
            "base_drift": round(float(np.linalg.norm(base1 - base0)), 3),
        }
        if video_path:
            result["video"] = video_path
        return result

    def reset_session(self):
        if self._vla_client is not None:
            self._vla_client.reset_session(self._sid)
        self._last_prompt = None  # post-reset: next call is a fresh task
        if self._hist is not None:
            self._hist.clear()

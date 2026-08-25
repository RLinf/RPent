"""RLDX-1 closed-loop grasp/contact skill for the RoboCasa agent.

Loads RLDX-1 once (in the sim venv) and drives it for a few action chunks against
the live cameras — the equivalent of LIBERO's pi0_pick: a reusable VLA "delivery
service" for the grasp, after which the LLM scripts the carry+place.

Obs format mirrors robocasa's gym get_observation (state.* + video.{3 cams} +
annotation), stacked over the policy's video_delta_indices history, batch dim = 1,
with a per-call session id + reset_memory for the RLDX memory module.
"""
import numpy as np
from collections import deque

from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.vla_client import RoboCasaVLAClient


class RLDXSkill:
    def __init__(
        self,
        env_client: RoboCasaEnvClient,
        vla_client: RoboCasaVLAClient,
        check_cancelled=None,
        record_frame_callback=None,
    ):
        self.env = env_client               # RoboCasaEnvClient
        # session seed is in self.vla rpc server
        self.vla = vla_client               # VLA RPC client (when set, _load() uses it instead of loading the model directly)
        self._check_cancelled = check_cancelled  # optional cancellation checkpoint callback
        # Optional per-step frame callback (e.g. primitives.record_frame(obs))
        # so the VLA's per-step agentview lands in the unified episode buffer
        # the toolkit owns — instead of a parallel cmd_NN.mp4 written here.
        # Called once per sim-step with the chunk_step per-step obs (which
        # carries the agentview as 'robot0_agentview_left_rgb'); matches the
        # manual primitives' (move_to / _step_arm) per-step record_frame
        # cadence so the unified video has consistent frame density across
        # VLA chunks and scripted moves.
        self._record_frame_callback = record_frame_callback
        self._last_prompt = None
        self.load()

    def load(self):
        # lazy: eval's PandaOmronKeyConverter.unmap_action
        from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter
        self.unmap_action = PandaOmronKeyConverter.unmap_action

        self.video_delta_indices = self.vla.get_video_delta_indices() # video delta indices, e.g. [-6,-4,-2,0]
        video_history_maxlen = int(self.video_delta_indices.max() - self.video_delta_indices.min()) + 2
        self.video_history = deque(maxlen=video_history_maxlen) # deque of raw frame dicts
        self._action_dim = self.env.action_dim # chunk_step array shape
        print(f"[rldx_skill] modality loaded via RPC; hist_maxlen={video_history_maxlen} action_dim={self._action_dim}", flush=True)

    # ---- obs construction (mirror robocasa gym get_observation) ----
    VLA_OBS_RES = 256   # matches the eval: robocasa365 gym_wrapper get_camera_config renders
                        # the 3 VLA cameras at 256 (overriding create_env's 128 DEFAULT).
                        # Verified: eval gym obs are (256,256,3) and primitives@256 is byte-identical.

    def _raw_frame(self, task_text, raw_obs=None, frames=None):
        if raw_obs is None:
            raw_obs = self.env.current_raw_obs
        # frames from chunk_step are rendered at per-step sim state
        if frames is None:
            # fall back to render_camera
            # render the 3 VLA cameras at the eval's resolution (256)
            r = self.VLA_OBS_RES
            frames = {}
            for key in [
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ]:
                frames[key] = self.env.render_camera(key, height=r, width=r)
        vL = frames["robot0_agentview_left"]
        vR = frames["robot0_agentview_right"]
        vW = frames["robot0_eye_in_hand"]
        return {
            "state.gripper_qpos": np.asarray(raw_obs["robot0_gripper_qpos"], np.float32),
            "state.base_position": np.asarray(raw_obs["robot0_base_pos"], np.float32),
            "state.base_rotation": np.asarray(raw_obs["robot0_base_quat"], np.float32),
            "state.end_effector_position_relative": np.asarray(raw_obs["robot0_base_to_eef_pos"], np.float32),
            "state.end_effector_rotation_relative": np.asarray(raw_obs["robot0_base_to_eef_quat"], np.float32),
            "video.robot0_agentview_left": vL.astype(np.uint8),
            "video.robot0_agentview_right": vR.astype(np.uint8),
            "video.robot0_eye_in_hand": vW.astype(np.uint8),
            "annotation.human.task_description": task_text,
        }

    def _seed_hist(self, task_text):
        """Pad the per-sim-step history with copies of the CURRENT frame, mirroring the
        eval MultiStepWrapper.reset (which fills its obs deque with n copies of obs0)."""
        frame = self._raw_frame(task_text)
        self.video_history.clear()
        for _ in range(self.video_history.maxlen):
            self.video_history.append(frame)

    def _build_obs(self, task_text):
        """Build the batched (n_envs=1), history-stacked obs by indexing the per-sim-step
        history at vdi-1 = [-7,-5,-3,-1], EXACTLY like the eval MultiStepWrapper._get_obs
        (delta_indices = video_delta_indices - 1, over self.obs[i])."""
        buf = list(self.video_history)
        idx = [len(buf) + (int(d) - 1) for d in self.video_delta_indices]   # vdi-1 offset from end
        idx = [max(0, min(len(buf) - 1, k)) for k in idx]
        cur = buf[-1]                                                 # state/annotation = current
        obs = {}
        for k, v in cur.items():
            if k.startswith("video."):
                stack = np.stack([buf[j][k] for j in idx], axis=0)    # (T,H,W,3)
                obs[k] = stack[None]                                  # (1,T,H,W,3)
            elif k.startswith("state."):
                obs[k] = np.asarray(v)[None][None]                    # (1,1,D)
            else:
                obs[k] = [task_text]                                  # (1,) annotation
        return obs

    def _build_env_action(self, eef_pos, eef_rot, gripper_close, base_motion, control_mode):
        """Assemble the native robosuite action EXACTLY like the EVAL gym wrapper
        (robocasa365 wrappers/gym_wrapper.py: PandaOmronKeyConverter.unmap_action +
        composite-controller split-index assembly). We route the VLA through the eval's
        OWN conversion instead of a hand-written 12-d concat so the policy sees the
        IDENTICAL action mode it was trained/evaluated with. This is where gripper_close
        and control_mode get binarized to +-1 (raw [0,1] -> -1 if <0.5 else +1);
        re-implementing that by hand and getting it wrong was the original no-grasp bug."""
        ad = self.unmap_action({
            "action.end_effector_position": np.asarray(eef_pos, np.float64).reshape(-1),
            "action.end_effector_rotation": np.asarray(eef_rot, np.float64).reshape(-1),
            "action.gripper_close": np.asarray(gripper_close, np.float64).reshape(-1),
            "action.base_motion": np.asarray(base_motion, np.float64).reshape(-1),
            "action.control_mode": np.asarray(control_mode, np.float64).reshape(-1),
        })
        return self.env.reassemble_env_action(ad)

    def _grasp_contact(self):
        """Direction-AGNOSTIC grasp check: True iff BOTH gripper fingerpads are in
        contact with the SAME task object (robosuite `_check_grasp`, the same primitive
        the benchmark uses). Unlike `peak_lift` (which only sees Z-axis lift and MISSES
        sideways/horizontal grasps — a bar, a handle, a lateral pull — and anything
        grasped-but-not-yet-raised), this fires for ANY grasp orientation and even for
        holding a static object. Returns (grasping: bool, obj_name: str|None)."""
        return self.env.grasp_contact()

    # ---- the skill ----
    def run(self, prompt, max_chunks=15, n_action_steps=8, base_clip=None,
            settle_eps=0.012, settle_patience=2, force_reset=False,
            recording=False, record_frame=None):
        """Drive RLDX-1 closed-loop until it FINISHES, not a fixed tiny budget. The VLA
        has no terminate signal (like the eval, which runs to env-success), so we stop
        on a completion criterion and report WHY — so a closed-empty gripper means the
        VLA actually executed and missed, not that we cut it off mid-grasp.
        Stops when: env `success` fires / the VLA SETTLES (eef+gripper stop changing for
        `settle_patience` chunks = it's done) / `max_chunks` safety cap.
        base_clip=None -> full base motion; base_clip=v -> clamp base_motion to [-v,v]
        (whole-body policy: never zero it). Returns status + grasp signals so the LLM
        decides: continue (call again) vs done vs genuinely-failed."""
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
        new_task = force_reset or prompt != self._last_prompt
        self._last_prompt = prompt
        # Reseed the per-sim-step history with the current frame on a new instruction or
        # the first call ever; same-prompt retries KEEP the continuous history (matches
        # the eval, which never switches instruction and never re-seeds mid-episode).
        if new_task or not self.video_history:
            self._seed_hist(prompt)
        fresh = new_task
        base0 = np.asarray(self.env.current_raw_obs["robot0_base_pos"], np.float64)[:2].copy()
        eef_prev = np.asarray(self.env.eef_pos, np.float64).copy()
        grip_prev = float(self.env.gripper_qpos[0])
        eef_min_z = float(self.env.eef_pos[2])
        peak_lift = 0.0
        applied = 0
        settled = 0
        status = "cap"
        grasp_ever = False
        grasp_obj = None
        last_cmd_close = False
        for c in range(max_chunks):
            obs = self._build_obs(prompt)
            # Do NOT set options["session_ids"] — the vla_server processed
            options = {"reset_memory": [fresh]}
            fresh = False
            if self._check_cancelled is not None:
                self._check_cancelled()
            actions = self.vla.predict(obs, options)
            # gym Dict -> native flat 12-d [eef_pos(3),eef_rot(3),gripper(1),base(4),mode(1)]
            horizon = actions["action.gripper_close"].shape[1]
            chunked_actions = []
            for step in range(min(n_action_steps, horizon)):
                base_motion = np.asarray(actions["action.base_motion"])[0, step]
                if base_clip is not None:
                    # rldx_arm only: clamp base velocities so the VLA micro-aligns but
                    # can't drive off (eval/rldx_skill pass base_clip=None = full motion).
                    base_motion = np.clip(base_motion, -base_clip, base_clip)
                # Build the action through the EVAL's own conversion (binarizes gripper +
                # control_mode); identical to how the policy is evaluated. See _build_env_action.
                last_cmd_close = (
                    float(np.asarray(actions["action.gripper_close"])[0, step].reshape(-1)[0]) >= 0.5)
                a = self._build_env_action(
                    np.asarray(actions["action.end_effector_position"])[0, step],
                    np.asarray(actions["action.end_effector_rotation"])[0, step],
                    np.asarray(actions["action.gripper_close"])[0, step],
                    base_motion,
                    np.asarray(actions["action.control_mode"])[0, step],
                )
                chunked_actions.append(a)
            if self._check_cancelled is not None:
                self._check_cancelled()
            recording = self._record_frame_callback is not None
            chunked_obs, _, _, _, chunked_frames, n_applied = self.env.chunk_step(chunked_actions)
            applied += n_applied
            # Per-step agentview into the unified episode buffer (matches
            # manual primitives' per-step cadence — fixes the per-chunk vs
            # per-step density mismatch that produced inconsistent video).
            for step_obs, step_frames in zip(chunked_obs, chunked_frames):
                self._record_frame_callback(step_obs)
                self._raw_frame(prompt, raw_obs=step_obs, frames=step_frames)
                if recording:
                    self._record_frame_callback(xxx)
            # ROBUST grasp check once per chunk (direction-agnostic; see _grasp_contact)
            g_now, g_obj = self._grasp_contact()
            if g_now:
                grasp_ever = True
                grasp_obj = grasp_obj or g_obj
            if self.env.check_success():
                status = "success"
                break
            eef_now = np.asarray(self.env.eef_pos, np.float64)
            grip_now = float(self.env.gripper_qpos[0])
            eef_min_z = min(eef_min_z, float(eef_now[2]))
            peak_lift = max(peak_lift, float(eef_now[2] - eef_min_z))
            if float(np.linalg.norm(eef_now - eef_prev)) < settle_eps and abs(grip_now - grip_prev) < 0.003:
                settled += 1
                if settled >= settle_patience:
                    status = "settled"
                    break          # VLA idle = finished its action
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
        held_apart = bool(last_cmd_close and 0.004 < grip < 0.039)   # closed onto something
        grasped_now = bool(grasp_now or held_apart)                  # holding at END of call
        grasp_detected = bool(grasp_ever or grasped_now)             # held at SOME point
        result = {"ok": True, "prompt": prompt, "status": status, "chunks": c + 1,
                "steps_applied": applied,
                "grasped": grasped_now,            # holding at end-of-call (carry-ready)
                "grasp_detected": grasp_detected,  # grabbed at some chunk (may have placed since)
                "grasp_contact": bool(grasp_now),  # fingerpad↔object contact (gold standard)
                "held_apart": held_apart,          # commanded-close + fingers wedged apart
                "grasp_obj": grasp_obj,            # which task object, if contact-identified
                "gripper_qpos": round(grip, 3), "peak_lift": round(peak_lift, 3),
                "base_clip": base_clip, "base_drift": round(float(np.linalg.norm(base1 - base0)), 3)}
        return result

    def reset_session(self):
        if self.vla is not None:
            try:
                self.vla.reset_session()
            except Exception:
                pass
        self._last_prompt = None          # post-reset: next call is a fresh task
        if self.video_history is not None:
            self.video_history.clear()

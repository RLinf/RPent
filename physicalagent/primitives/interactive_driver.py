"""REPL-style LLM-in-the-loop driver.

One process: env + Pi0.5 loaded once. Reads single-command JSON files from
`/tmp/hybrid_repl/command.json`, executes, dumps `state_<step>.json` and
`image_<step>.png`, then blocks waiting for the next command.

Commands (in command.json):
    {"action": "move_to", "xyz": [x, y, z], "gripper": -1,
     "max_steps": 80, "step_clip": 0.025}
    {"action": "pi0_pick", "prompt": "pick up the black bowl", "max_chunks": 30}
    {"action": "rotate_wrist", "delta_yaw": 1.57, "gripper": 1}    # world-Z rotation
    {"action": "rotate_pitch", "delta_pitch": 0.5, "gripper": 1}   # world-X rotation
    {"action": "release", "max_steps": 20}
    {"action": "set_gripper", "gripper": -1, "steps": 5}     # hold pose, change gripper
    {"action": "exit"}                                        # clean exit

Usage:
    Launch in background, then write commands one at a time:
        $ echo '{"action":"move_to","xyz":[...],"gripper":-1}' > /tmp/hybrid_repl/command.json
        $ # wait for /tmp/hybrid_repl/done_<step>.flag
        $ # read state + image
        $ # write next command
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_PATH = "/mnt/public2/zhangyixian/RLinf_agentic"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

import imageio.v2 as imageio
import numpy as np
import torch

from examples.embodiment.primitives.primitives import (
    CHECKPOINT_PATH,
    LiberoPrimitiveDriver,
    build_env_cfg,
    build_model_cfg,
)
from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.models.embodiment.openpi import get_model as get_openpi_model


def make_env(task_id: int, seed: int, suite_name: str = "libero_spatial",
             max_episode_steps: int = 240):
    from rlinf.envs.libero.utils import benchmark as _bench_mod
    suite = _bench_mod.get_benchmark(suite_name)()
    first_id = sum(len(suite.get_task_init_states(t)) for t in range(task_id))
    trials = len(suite.get_task_init_states(task_id))
    rid = first_id + (seed % trials)
    cfg = build_env_cfg(
        task_suite_name=suite_name,
        specific_reset_id=rid,
        seed=seed,
        max_episode_steps=max_episode_steps,
    )
    return LiberoEnv(cfg=cfg, num_envs=1, seed_offset=0,
                     total_num_processes=1, worker_info=None)


def hold_gripper(driver, gripper: float, steps: int):
    """Hold current EEF pose and command gripper for N env steps."""
    for _ in range(steps):
        action = np.zeros(7, dtype=np.float32)
        action[6] = float(gripper)
        obs, _r, term, _t, _i = driver.env.step(action[None])
        driver._last_obs = obs
        driver.record_frame()
        if isinstance(term, torch.Tensor):
            if bool(term[driver.env_idx].any().item()):
                driver._libero_terminated = True


def dump_state(driver, workdir: str, step_idx: int) -> dict:
    state = driver.get_privileged_state()
    # PERCEPTION-ISOLATED mode: drop object world coords (the agent must
    # localize via depth_NN.npy + camera_meta.json). Keep the object NAMES
    # (what's in the scene / which is the target) + robot proprioception —
    # names are not coordinate info and are also implied by the task language.
    if getattr(driver, "_hide_object_coords", False):
        objs = state.get("objects", {})
        state["object_names"] = sorted(objs.keys())
        state.pop("objects", None)
    # Try render_agentview (live obs from env). When the image observable
    # is disabled, current_raw_obs has no image key OR robosuite returns
    # a degenerate (1,1,3) float64 placeholder. Fall back to the most
    # recent valid frame cached in LiberoEnv._cached_full_image (set in
    # _extract_image_and_state whenever a render-enabled step ran).
    try:
        img = driver.render_agentview()
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3 \
                or img.shape[0] < 32 or img.shape[1] < 32:
            raise ValueError(f"bad img shape/dtype: {img.shape} {img.dtype}")
    except Exception:
        # _cached_full_image is already 180°-flipped (get_libero_image does
        # the flip), so just hand it through. No double-flip.
        cached = getattr(driver.env, "_cached_full_image", None)
        if cached is not None:
            img = cached.cpu().numpy() if hasattr(cached, "cpu") else np.asarray(cached)
        else:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
    imageio.imwrite(os.path.join(workdir, f"image_{step_idx:02d}.png"), img)

    # --- camera calibration (static for agentview): fetch + dump once ---
    cam_meta = getattr(driver, "_camera_meta", None)
    if cam_meta is None:
        try:
            cam_meta = driver.env.get_camera_meta(
                camera_name="agentview", height=256, width=256)
            driver._camera_meta = cam_meta
            cam_meta_out = dict(cam_meta)
            cam_meta_out["projection"] = (
                "world->pixel: M = K_exp @ inv(extrinsic_cam2world), where "
                "K_exp is 4x4 (K in top-left). q = M @ [X,Y,Z,1]; "
                "col=q[0]/q[2], row=q[1]/q[2], metric_depth=q[2]. "
                "(row,col) indexes depth_NN.npy directly. Back-project a pixel: "
                "P_world = extrinsic_cam2world @ [col*z, row*z, z, 1] with "
                "z=depth_NN[row,col]. VERIFIED 5/5 vs GT object poses.")
            cam_meta_out["note"] = (
                "depth_NN.npy is in this camera frame (vertical-flipped raw "
                "buffer). image_NN.png is rotated 180deg (Pi0 convention) and "
                "is NOT in the same frame as depth/K.")
            with open(os.path.join(workdir, "camera_meta.json"), "w") as f:
                json.dump(cam_meta_out, f, indent=2)
        except Exception as e:
            print(f"[dump_state] get_camera_meta failed: {e}", flush=True)
            driver._camera_meta = cam_meta = {}

    # --- per-step RGB in the depth/K frame (vertical-flip of the raw buffer) ---
    # The agent picks object pixels HERE (same frame as depth_NN.npy + K), so
    # pixel -> depth -> back-project is direct. (image_NN.png is the 180°-rotated
    # Pi0-convention frame and must NOT be used for back-projection.)
    try:
        _raw = driver.env.current_raw_obs[driver.env_idx]
        ci = _raw.get("agentview_image")
        if ci is not None:
            ci = np.asarray(ci)
            if ci.dtype != np.uint8:
                ci = ci.astype(np.uint8)
            imageio.imwrite(os.path.join(workdir, f"image_cam_{step_idx:02d}.png"),
                            ci[::-1])
    except Exception as e:
        print(f"[dump_state] image_cam dump failed: {e}", flush=True)

    # --- per-step metric depth (agentview), native orientation, in meters ---
    try:
        raw = driver.env.current_raw_obs[driver.env_idx]
        d = raw.get("agentview_depth")
        if d is not None:
            d = np.asarray(d, dtype=np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            near = cam_meta.get("depth_near")
            far = cam_meta.get("depth_far")
            if near is not None and far is not None:
                # robosuite normalized OpenGL depth -> metric (get_real_depth_map)
                d = near / (1.0 - d * (1.0 - near / far))
            # Vertical flip to align with the camera matrices: robosuite's
            # camera_utils projection M = K_exp @ inv(extrinsic) expects the
            # depth map in this frame. VERIFIED 5/5: projecting each GT object
            # world pos via M lands on a pixel whose depth_flip[row,col] matches
            # the object's surface depth (plate Δ6mm, cookies Δ14mm). So
            # pixel(row,col) in depth_NN.npy back-projects correctly with
            # camera_meta.json (NOT the same frame as the 180°-rotated
            # image_NN.png — see camera_meta note).
            d = d[::-1]
            np.save(os.path.join(workdir, f"depth_{step_idx:02d}.npy"),
                    d.astype(np.float32))
    except Exception as e:
        print(f"[dump_state] depth dump failed: {e}", flush=True)

    blob = {
        "step_idx": step_idx,
        "libero_terminated": driver._libero_terminated,
        "state": state,
    }
    with open(os.path.join(workdir, f"state_{step_idx:02d}.json"), "w") as f:
        json.dump(blob, f, indent=2)
    return blob


def execute(driver, cmd: dict, workdir: str, step_idx: int):
    action = cmd.get("action")
    t0 = time.time()
    log = {"step_idx": step_idx, "command": cmd}

    # EGL accumulates if we render an image every env.step (which robosuite
    # does by default). Pure-OSC primitives (move_to, set_gripper, release,
    # rotate_*) don't need images during their internal step loops, so
    # disable image observables before them and re-enable for pi0_pick (which
    # feeds images to Pi0). After every primitive we re-enable + step zero so
    # the cached `agentview_image` / `current_raw_obs` are fresh for the
    # post-step image dump.
    _always = getattr(driver, "_always_render", False)
    _osc_only = action in {"move_to", "release", "set_gripper", "rotate_wrist",
                            "rotate_pitch", "move_pose"}
    if _osc_only and not _always:
        try:
            driver.env.set_image_render_enabled(False)
        except Exception:
            pass  # older env without the toggle — no-op
    elif action == "pi0_pick" or _always:
        try:
            driver.env.set_image_render_enabled(True)
        except Exception:
            pass

    if action == "move_to":
        result = driver.move_to(
            cmd["xyz"],
            max_steps=cmd.get("max_steps", 80),
            gripper_action=float(cmd.get("gripper", +1.0)),
            step_clip=cmd.get("step_clip", 0.025),
            tol=cmd.get("tol", 0.012),
            action_scale=cmd.get("action_scale", 0.05),
            target_yaw=cmd.get("target_yaw"),
            yaw_step_clip=cmd.get("yaw_step_clip", 0.10),
        )
        log["result"] = result
    elif action == "rotate_wrist":
        log["result"] = driver.rotate_wrist(
            target_yaw=cmd.get("target_yaw"),
            delta_yaw=cmd.get("delta_yaw"),
            gripper_action=float(cmd.get("gripper", +1.0)),
            max_steps=cmd.get("max_steps", 40),
            tol=cmd.get("tol", 0.02),
            step_clip=cmd.get("step_clip", 0.10),
        )
    elif action == "move_pose":
        log["result"] = driver.move_pose(
            cmd["xyz"],
            target_pitch=cmd.get("target_pitch"),
            target_yaw=cmd.get("target_yaw"),
            gripper_action=float(cmd.get("gripper", -1.0)),
            step_clip=cmd.get("step_clip", 0.02),
            pitch_step=cmd.get("pitch_step", 0.08),
            yaw_step=cmd.get("yaw_step", 0.08),
            tol=cmd.get("tol", 0.012),
            ori_tol=cmd.get("ori_tol", 0.05),
            max_steps=cmd.get("max_steps", 150),
        )
    elif action == "rotate_pitch":
        log["result"] = driver.rotate_pitch(
            target_pitch=cmd.get("target_pitch"),
            delta_pitch=cmd.get("delta_pitch"),
            gripper_action=float(cmd.get("gripper", +1.0)),
            max_steps=cmd.get("max_steps", 40),
            tol=cmd.get("tol", 0.02),
            step_clip=cmd.get("step_clip", 0.10),
        )
    elif action == "pi0_pick":
        pres = driver.pick(
            cmd.get("prompt", "pick up the black bowl"),
            max_chunks=cmd.get("max_chunks", 30),
            instruction_template="{obj}",
            lift_thresh=cmd.get("lift_thresh", 0.05),
            gripper_closed_thresh=cmd.get("gripper_closed_thresh", 0.06),
            track_obj=cmd.get("track_obj"),
            track_obj_lift_thresh=cmd.get("track_obj_lift_thresh", 0.05),
        )
        log["result"] = pres.to_dict()
    elif action == "reset":
        driver.reset()
        log["result"] = {"name": "reset"}
    elif action == "start_recording":
        driver.start_recording()
        log["result"] = {"name": "start_recording"}
    elif action == "save_video":
        path = cmd.get("path",
            "/mnt/public2/zhangyixian/RLinf_agentic/"
            "examples/embodiment/primitives/videos/last_run.mp4")
        log["result"] = driver.stop_recording_and_save(
            path, fps=cmd.get("fps", 20),
            keep_recording=cmd.get("keep_recording", False),
        )
    elif action == "release":
        log["result"] = driver.release(max_steps=cmd.get("max_steps", 20))
    elif action == "set_gripper":
        hold_gripper(driver,
                     gripper=float(cmd.get("gripper", -1.0)),
                     steps=int(cmd.get("steps", 5)))
        log["result"] = {"name": "set_gripper", "ok": True}
    elif action == "snapshot":
        log["result"] = {"name": "snapshot"}
    elif action == "exit":
        log["result"] = {"name": "exit"}
    else:
        log["result"] = {"error": f"unknown action {action}"}

    # NOTE: no per-primitive "render refresh" env.step. Robosuite's
    # observable when re-enabled returns garbage (1,1,3 float64) on the
    # first sample after disable, which crashes imageio. Instead we leave
    # rendering OFF for OSC primitives and rely on dump_state's fallback
    # to LiberoEnv._cached_full_image (last good image from the most
    # recent render-enabled step, e.g. the previous pi0_pick or reset).

    log["elapsed_s"] = round(time.time() - t0, 2)
    with open(os.path.join(workdir, f"log_{step_idx:02d}.json"), "w") as f:
        json.dump(log, f, indent=2)
    return log


def wait_for_command(cmd_path: str, poll_s: float = 0.5, timeout_s: float = 3600.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(cmd_path):
            try:
                with open(cmd_path) as f:
                    cmd = json.load(f)
                os.remove(cmd_path)
                return cmd
            except Exception as e:
                print(f"[wait] error reading cmd: {e}, retrying")
                time.sleep(poll_s)
                continue
        time.sleep(poll_s)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suite", type=str, default="libero_spatial")
    p.add_argument("--max_episode_steps", type=int, default=240)
    p.add_argument("--workdir", type=str, default="/tmp/hybrid_repl")
    p.add_argument("--max_steps", type=int, default=40,
                   help="hard cap on the number of REPL commands")
    p.add_argument("--always_render", action="store_true",
                   help="disable the auto render-skip toggle; render every "
                        "env.step. Use for video recording (frames are valid "
                        "throughout primitive loops). Costs ~30× more env.step "
                        "wall time and risks the old EGL accumulation cap.")
    p.add_argument("--hide_object_coords", action="store_true",
                   help="PERCEPTION-ISOLATED mode: drop object world coords from "
                        "state_NN.json (keep object_names + obj_of_interest + "
                        "robot proprioception). The agent must localize objects "
                        "from depth_NN.npy + camera_meta.json. Default off "
                        "(oracle mode: full GT coords).")
    p.add_argument("--video_path", default=None,
                   help="Path to save episode video (default: <workdir>/episode.mp4)")
    args = p.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    # clean stale state
    for f in os.listdir(args.workdir):
        os.remove(os.path.join(args.workdir, f))

    print(f"[setup] task={args.task}  seed={args.seed}  workdir={args.workdir}")
    print(f"[setup] loading Pi0.5 ...")
    t0 = time.time()
    model_cfg = build_model_cfg(model_path=CHECKPOINT_PATH)
    model = get_openpi_model(model_cfg, torch_dtype=None).cuda().eval()
    print(f"[setup] model ready in {time.time() - t0:.1f}s")

    env = make_env(args.task, args.seed, suite_name=args.suite,
                   max_episode_steps=args.max_episode_steps)
    driver = LiberoPrimitiveDriver(env=env, model=model, action_chunk=5)
    driver._always_render = args.always_render
    driver._hide_object_coords = args.hide_object_coords
    driver.reset()
    dump_state(driver, args.workdir, step_idx=0)
    print(f"[reset] initial state dumped (step 0).")

    # Auto-start recording so video is always available
    video_path = getattr(args, "video_path", None) or os.path.join(
        args.workdir, "episode.mp4")
    driver.start_recording()
    print(f"[record] started; will save to {video_path}")

    cmd_path = os.path.join(args.workdir, "command.json")
    step = 1
    while step <= args.max_steps:
        print(f"\n[step {step}] BLOCKED — waiting for {cmd_path}")
        cmd = wait_for_command(cmd_path, timeout_s=3600.0)
        if cmd is None:
            print(f"[step {step}] TIMEOUT")
            break
        print(f"[step {step}] received: {cmd}")
        log = execute(driver, cmd, args.workdir, step)
        dump_state(driver, args.workdir, step)
        # signal done
        with open(os.path.join(args.workdir, f"done_{step:02d}.flag"), "w") as f:
            f.write("ok")
        print(f"[step {step}] done in {log['elapsed_s']}s  result={log['result']}")
        if cmd.get("action") == "exit":
            result = driver.stop_recording_and_save(video_path)
            print(f"[record] saved {result['n_frames']} frames to {video_path}")
            break
        step += 1

    # final summary
    print(f"\n[final] libero_terminated = {driver._libero_terminated}")


if __name__ == "__main__":
    main()

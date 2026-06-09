"""Hybrid LLM + VLA experiment: Pi0.5 picks, LLM-in-the-loop decides where to drop.

Flow:
  1. Build env + Pi0.5 model (~90s).
  2. Reset to a libero_spatial task.
  3. Pi0.5 executes pick (its strong suit — works regardless of prompt).
  4. Render post-pick agentview to {workdir}/post_pick.png.
  5. Dump privileged state (EEF pose + all object world-frame xyz) to
     {workdir}/state_after_pick.json — this includes plate position so the LLM
     can VERIFY its visual judgment against ground truth.
  6. BLOCK until {workdir}/decision.json appears with {"target_xyz": [x, y, z]}.
  7. driver.move_to(target_xyz) keeping gripper closed.
  8. driver.release().
  9. Save final agentview to {workdir}/final.png. Dump full result JSON.

To run end-to-end with the LLM (Claude) as the planner:
  - Launch in background: nohup python hybrid_pick_place.py --task 0 ... &
  - LLM reads {workdir}/post_pick.png and state_after_pick.json.
  - LLM writes {workdir}/decision.json with its chosen target.
  - Script unblocks and finishes.
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
    parse_task_object,
)
from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.models.embodiment.openpi import get_model as get_openpi_model


def make_env(task_id: int, seed: int):
    from libero.libero.benchmark import get_benchmark

    suite = get_benchmark("libero_spatial")()
    first_id = sum(len(suite.get_task_init_states(t)) for t in range(task_id))
    trials = len(suite.get_task_init_states(task_id))
    rid = first_id + (seed % trials)
    cfg = build_env_cfg(
        task_suite_name="libero_spatial",
        specific_reset_id=rid,
        seed=seed,
    )
    return LiberoEnv(
        cfg=cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None,
    )


def wait_for_file(path: str, timeout_s: float = 600.0, poll_s: float = 1.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(poll_s)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workdir", type=str, default="/tmp/hybrid")
    p.add_argument("--max_chunks_pick", type=int, default=20,
                   help="Pi0.5 chunk budget for pick — stops on lift+close.")
    p.add_argument("--move_max_steps", type=int, default=80)
    p.add_argument("--release_max_steps", type=int, default=20)
    p.add_argument("--pick_prompt", type=str, default="pick up the black bowl")
    p.add_argument("--decision_timeout_s", type=float, default=600.0)
    p.add_argument("--auto_target_above_plate_z", type=float, default=None,
                   help="If set, skip the LLM wait and use plate_1_pos + this Δz "
                        "directly (sanity-check the move+release primitives).")
    args = p.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    print(f"[setup] workdir={args.workdir}  task={args.task}  seed={args.seed}")

    obj_text, tgt_text = parse_task_object(args.task)
    print(f"[task {args.task}] obj=\"{obj_text}\"  tgt=\"{tgt_text}\"")

    t0 = time.time()
    print(f"[setup] loading Pi0.5 from {CHECKPOINT_PATH}...")
    model_cfg = build_model_cfg(model_path=CHECKPOINT_PATH)
    model = get_openpi_model(model_cfg, torch_dtype=None).cuda().eval()
    print(f"[setup] model ready in {time.time() - t0:.1f}s")

    env = make_env(args.task, args.seed)
    driver = LiberoPrimitiveDriver(env=env, model=model, action_chunk=5)
    obs, _ = driver.reset()

    # ---- initial snapshot ----
    init_img = driver.render_agentview()
    imageio.imwrite(os.path.join(args.workdir, "initial.png"), init_img)
    init_state = driver.get_privileged_state()
    print(f"[reset] eef_pos={init_state['robot0_eef_pos']}")
    print(f"[reset] plate_1_pos={init_state['objects'].get('plate_1')}")
    print(f"[reset] akita_black_bowl_1_pos={init_state['objects'].get('akita_black_bowl_1')}")

    # ---- Phase 1: Pi0.5 pick ----
    print(f"\n[phase 1] Pi0.5 pick with prompt=\"{args.pick_prompt}\"")
    t_pick = time.time()
    pick_res = driver.pick(
        args.pick_prompt,
        max_chunks=args.max_chunks_pick,
        instruction_template="{obj}",  # use prompt verbatim, no template
    )
    print(f"[phase 1] done in {time.time() - t_pick:.1f}s  {pick_res.to_dict()}")

    # ---- snapshot after pick ----
    post_img = driver.render_agentview()
    imageio.imwrite(os.path.join(args.workdir, "post_pick.png"), post_img)
    post_state = driver.get_privileged_state()
    print(f"[post-pick] eef_pos={post_state['robot0_eef_pos']}")
    print(f"[post-pick] akita_black_bowl_1_pos={post_state['objects'].get('akita_black_bowl_1')}")
    state_dump = {
        "task_id": args.task,
        "seed": args.seed,
        "obj_text": obj_text,
        "tgt_text": tgt_text,
        "pick_result": pick_res.to_dict(),
        "post_pick_state": post_state,
    }
    with open(os.path.join(args.workdir, "state_after_pick.json"), "w") as f:
        json.dump(state_dump, f, indent=2)
    print(f"[post-pick] dumped state_after_pick.json + post_pick.png")

    # ---- Phase 2: wait for LLM decision (or use auto target) ----
    decision_path = os.path.join(args.workdir, "decision.json")
    if args.auto_target_above_plate_z is not None:
        plate_xyz = post_state["objects"]["plate_1"]
        target = [
            float(plate_xyz[0]),
            float(plate_xyz[1]),
            float(plate_xyz[2]) + float(args.auto_target_above_plate_z),
        ]
        decision = {"target_xyz": target, "source": "auto_plate_pos"}
        print(f"[phase 2] AUTO target = {target} (plate + Δz)")
    else:
        # Remove stale decision if present.
        if os.path.exists(decision_path):
            os.remove(decision_path)
        print(f"\n[phase 2] BLOCKED — waiting for {decision_path} ({args.decision_timeout_s:.0f}s)")
        ok = wait_for_file(decision_path, timeout_s=args.decision_timeout_s)
        if not ok:
            print(f"[phase 2] TIMEOUT — no decision after {args.decision_timeout_s}s")
            return
        with open(decision_path) as f:
            decision = json.load(f)
        target = decision["target_xyz"]
        print(f"[phase 2] decision received: target_xyz={target}  "
              f"({decision.get('rationale','')})")

    # ---- Phase 3: scripted move + release ----
    t_move = time.time()
    move_res = driver.move_to(
        target,
        max_steps=args.move_max_steps,
        gripper_action=+1.0,  # stay closed during transit
    )
    print(f"[phase 3] move_to done in {time.time() - t_move:.1f}s  {move_res}")

    # snapshot before release
    pre_release_img = driver.render_agentview()
    imageio.imwrite(os.path.join(args.workdir, "pre_release.png"), pre_release_img)

    t_rel = time.time()
    release_res = driver.release(max_steps=args.release_max_steps)
    print(f"[phase 3] release done in {time.time() - t_rel:.1f}s  {release_res}")

    # ---- final snapshot ----
    final_img = driver.render_agentview()
    imageio.imwrite(os.path.join(args.workdir, "final.png"), final_img)
    final_state = driver.get_privileged_state()

    # libero check_success
    libero_term = driver._libero_terminated
    bowl_final = final_state["objects"].get("akita_black_bowl_1")
    plate_xyz = final_state["objects"].get("plate_1")
    bowl_to_plate_xy = (
        float(np.linalg.norm(np.array(bowl_final[:2]) - np.array(plate_xyz[:2])))
        if bowl_final and plate_xyz
        else None
    )

    result = {
        **state_dump,
        "decision": decision,
        "move_result": move_res,
        "release_result": release_res,
        "final_state": final_state,
        "libero_terminated": libero_term,
        "bowl_to_plate_xy_m": bowl_to_plate_xy,
    }
    with open(os.path.join(args.workdir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n========== RESULT ==========")
    print(f"libero_terminated = {libero_term}")
    print(f"final bowl pos    = {bowl_final}")
    print(f"target plate pos  = {plate_xyz}")
    print(f"bowl→plate xy err = {bowl_to_plate_xy:.3f} m" if bowl_to_plate_xy is not None else "")
    print(f"files: initial.png post_pick.png pre_release.png final.png result.json")


if __name__ == "__main__":
    main()

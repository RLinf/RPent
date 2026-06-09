"""Hybrid (Pi0.5 pick + scripted offset-compensated place) on all libero_spatial tasks.

Encodes the two rules derived interactively from task 0 into a formula so we
can sweep all 10 tasks without per-task LLM intervention:

    target_eef_xy = plate_xy - (bowl_pos - eef_pos)_xy        # offset compensation
    target_eef_z  = plate_z + bowl_half_h + margin - bowl_eef_dz  # release-on-contact

Per task × seed:
  1. Build env, reset.
  2. Pi0.5 picks with prompt "pick up the black bowl" (model is largely
     prompt-blind so the exact wording doesn't matter — see prior negctrl finding).
  3. Read privileged state. Compute target_xyz from the formula.
  4. move_to(target) keeping gripper closed → release.
  5. Record libero_term + per-rollout JSON.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_PATH = "/mnt/public/jxqiu/physicalagent"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

import numpy as np
import torch

from physicalagent.primitives.primitives import (
    CHECKPOINT_PATH,
    LiberoPrimitiveDriver,
    build_env_cfg,
    build_model_cfg,
)
from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.models.embodiment.openpi import get_model as get_openpi_model


# Approximate physical constants used by the formula. Values measured from
# task-0 runs; libero_spatial uses the same akita black bowl + plate across all
# 10 tasks so these generalize.
BOWL_HALF_HEIGHT_M = 0.050  # akita_black_bowl_1 vertical half-extent.
RELEASE_MARGIN_M = 0.005    # gap between bowl-bottom and plate-top at release.


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


def compute_target_from_state(state: dict) -> tuple[list, dict]:
    """Apply the two formulas to derive a placement target_xyz.

    Returns (target_xyz, diagnostics).
    """
    eef = np.array(state["robot0_eef_pos"])
    bowl = np.array(state["objects"]["akita_black_bowl_1"])
    plate = np.array(state["objects"]["plate_1"])
    offset = bowl - eef  # (Δx, Δy, Δz) of bowl relative to EEF
    target_xy = plate[:2] - offset[:2]
    target_z = plate[2] + BOWL_HALF_HEIGHT_M + RELEASE_MARGIN_M - offset[2]
    target = [float(target_xy[0]), float(target_xy[1]), float(target_z)]
    diag = {
        "eef": [round(float(x), 4) for x in eef],
        "bowl": [round(float(x), 4) for x in bowl],
        "plate": [round(float(x), 4) for x in plate],
        "bowl_minus_eef": [round(float(x), 4) for x in offset],
    }
    return target, diag


def run_one(driver, max_chunks_pick: int, move_max_steps: int, release_max_steps: int) -> dict:
    obs, _ = driver.reset()
    init_state = driver.get_privileged_state()

    pick_res = driver.pick(
        "pick up the black bowl",
        max_chunks=max_chunks_pick,
        instruction_template="{obj}",
    )
    post_state = driver.get_privileged_state()

    if pick_res.libero_terminated:
        return {
            "pick_result": pick_res.to_dict(),
            "post_pick_state": post_state,
            "libero_terminated": True,
            "stopped_at_pick": True,
        }

    # Sanity guard: if bowl didn't actually move from its spawn, the grasp
    # was effectively a miss — proceeding to move would just drag empty air.
    bowl_init = np.array(init_state["objects"]["akita_black_bowl_1"])
    bowl_post = np.array(post_state["objects"]["akita_black_bowl_1"])
    bowl_lifted = float(np.linalg.norm(bowl_post - bowl_init))

    target_xyz, target_diag = compute_target_from_state(post_state)

    move_res = driver.move_to(
        target_xyz,
        max_steps=move_max_steps,
        gripper_action=+1.0,
    )
    release_res = driver.release(max_steps=release_max_steps)
    final_state = driver.get_privileged_state()
    bowl_final = final_state["objects"]["akita_black_bowl_1"]
    plate_xyz = final_state["objects"]["plate_1"]
    bowl_to_plate_xy = float(
        np.linalg.norm(np.array(bowl_final[:2]) - np.array(plate_xyz[:2]))
    )

    return {
        "pick_result": pick_res.to_dict(),
        "post_pick_state": post_state,
        "bowl_lifted_during_pick_m": round(bowl_lifted, 4),
        "target_xyz": target_xyz,
        "target_diag": target_diag,
        "move_result": move_res,
        "release_result": release_res,
        "final_state": final_state,
        "libero_terminated": driver._libero_terminated,
        "bowl_to_plate_xy_m": round(bowl_to_plate_xy, 4),
        "stopped_at_pick": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--max_chunks_pick", type=int, default=30)
    p.add_argument("--move_max_steps", type=int, default=80)
    p.add_argument("--release_max_steps", type=int, default=20)
    p.add_argument("--out_dir", type=str,
                   default="/mnt/public/jxqiu/physicalagent/"
                           "physicalagent/primitives/results_all_spatial")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[plan] tasks={args.tasks} seeds={args.seeds}")
    print(f"[plan] {len(args.tasks)*len(args.seeds)} rollouts, out_dir={args.out_dir}")

    # Build model once.
    t0 = time.time()
    print(f"[setup] loading Pi0.5 from {CHECKPOINT_PATH}...")
    model_cfg = build_model_cfg(model_path=CHECKPOINT_PATH)
    model = get_openpi_model(model_cfg, torch_dtype=None).cuda().eval()
    print(f"[setup] model ready in {time.time() - t0:.1f}s")

    rows = []
    for task_id in args.tasks:
        for seed in args.seeds:
            t1 = time.time()
            env = make_env(task_id, seed)
            driver = LiberoPrimitiveDriver(env=env, model=model, action_chunk=5)
            try:
                result = run_one(
                    driver,
                    max_chunks_pick=args.max_chunks_pick,
                    move_max_steps=args.move_max_steps,
                    release_max_steps=args.release_max_steps,
                )
            except Exception as e:
                result = {"error": repr(e)}
            elapsed = time.time() - t1
            row = {"task_id": task_id, "seed": seed, "elapsed_s": round(elapsed, 1),
                   **result}
            rows.append(row)

            term = result.get("libero_terminated", False)
            err = result.get("bowl_to_plate_xy_m", None)
            lifted = result.get("bowl_lifted_during_pick_m", None)
            print(f"  [t{task_id} s{seed}]  libero_term={str(term)[:5]:>5}  "
                  f"bowl_lift={lifted}  bowl→plate_xy={err}  ({elapsed:.1f}s)")

            with open(os.path.join(args.out_dir, f"t{task_id}_s{seed}.json"), "w") as f:
                json.dump(row, f, indent=2)
            try:
                env.env.close()
            except Exception:
                pass
            del env, driver
            gc.collect()
            torch.cuda.empty_cache()

    # Aggregate.
    with open(os.path.join(args.out_dir, "all_rows.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print("\n=========== SUMMARY ===========")
    print(f"{'task':>4}  {'seeds':<5}  {'libero_term':>11}  {'mean_err_mm':>11}")
    for task_id in args.tasks:
        subset = [r for r in rows if r["task_id"] == task_id]
        term = sum(r.get("libero_terminated", False) for r in subset)
        errs = [r.get("bowl_to_plate_xy_m", None) for r in subset]
        errs = [e for e in errs if e is not None]
        mean_err = np.mean(errs) * 1000 if errs else float("nan")
        print(f"{task_id:>4}  {len(subset):<5}  {term}/{len(subset):>10}  {mean_err:>10.1f}")
    n = len(rows)
    overall = sum(r.get("libero_terminated", False) for r in rows)
    print(f"\noverall: {overall}/{n} libero_term")


if __name__ == "__main__":
    main()

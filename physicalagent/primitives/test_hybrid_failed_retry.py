"""Retry the 6 failing (task, seed) pairs from test_hybrid_all_spatial.py
using the ORIGINAL task description as the pick prompt instead of the
prompt-blind sub-instruction "pick up the black bowl".

If full prompt rescues t4/t9 (bowl in drawer / on cabinet), it means
Pi0.5 needs spatial context in the prompt for non-tabletop pickups —
this clarifies the limits of the prompt-blind regime."""
from __future__ import annotations

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
    parse_task_object,
)
from physicalagent.primitives.test_hybrid_all_spatial import (
    compute_target_from_state,
    make_env,
)
from rlinf.models.embodiment.openpi import get_model as get_openpi_model


FAILED_PAIRS = [(1, 1), (4, 0), (4, 1), (6, 0), (9, 0), (9, 1)]


def main():
    from libero.libero.benchmark import get_benchmark
    suite = get_benchmark("libero_spatial")()

    out_dir = ("/mnt/public/jxqiu/physicalagent/"
               "physicalagent/primitives/results_all_spatial_retry")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print(f"[setup] loading Pi0.5 ...")
    model_cfg = build_model_cfg(model_path=CHECKPOINT_PATH)
    model = get_openpi_model(model_cfg, torch_dtype=None).cuda().eval()
    print(f"[setup] model ready in {time.time() - t0:.1f}s")

    for task_id, seed in FAILED_PAIRS:
        t1 = time.time()
        full_prompt = suite.get_task(task_id).language
        print(f"\n========== task {task_id} seed {seed} ==========")
        print(f"  prompt = \"{full_prompt}\"")

        env = make_env(task_id, seed)
        driver = LiberoPrimitiveDriver(env=env, model=model, action_chunk=5)
        init_state_unused, _ = driver.reset()
        init_state = driver.get_privileged_state()

        pick_res = driver.pick(
            full_prompt,
            max_chunks=40,  # extra budget for harder pickups
            instruction_template="{obj}",
        )
        post_state = driver.get_privileged_state()
        bowl_lifted = float(np.linalg.norm(
            np.array(post_state["objects"]["akita_black_bowl_1"])
            - np.array(init_state["objects"]["akita_black_bowl_1"])
        ))

        if pick_res.libero_terminated:
            result = {
                "task_id": task_id, "seed": seed,
                "prompt": full_prompt,
                "pick_result": pick_res.to_dict(),
                "post_pick_state": post_state,
                "bowl_lifted_during_pick_m": round(bowl_lifted, 4),
                "libero_terminated": True,
                "stopped_at_pick": True,
            }
        else:
            target_xyz, target_diag = compute_target_from_state(post_state)
            move_res = driver.move_to(target_xyz, max_steps=80, gripper_action=+1.0)
            release_res = driver.release(max_steps=20)
            final_state = driver.get_privileged_state()
            bowl_final = final_state["objects"]["akita_black_bowl_1"]
            plate_xyz = final_state["objects"]["plate_1"]
            err = float(np.linalg.norm(
                np.array(bowl_final[:2]) - np.array(plate_xyz[:2])))
            result = {
                "task_id": task_id, "seed": seed,
                "prompt": full_prompt,
                "pick_result": pick_res.to_dict(),
                "post_pick_state": post_state,
                "bowl_lifted_during_pick_m": round(bowl_lifted, 4),
                "target_xyz": target_xyz,
                "target_diag": target_diag,
                "move_result": move_res,
                "release_result": release_res,
                "final_state": final_state,
                "libero_terminated": driver._libero_terminated,
                "bowl_to_plate_xy_m": round(err, 4),
            }

        elapsed = time.time() - t1
        result["elapsed_s"] = round(elapsed, 1)
        with open(os.path.join(out_dir, f"t{task_id}_s{seed}.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(f"  [t{task_id} s{seed}]  libero_term={result.get('libero_terminated', False)}  "
              f"bowl_lift={result.get('bowl_lifted_during_pick_m', 0):.3f}  "
              f"bowl→plate_xy={result.get('bowl_to_plate_xy_m', '-')}  ({elapsed:.1f}s)")

        try:
            env.env.close()
        except Exception:
            pass
        del env, driver
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

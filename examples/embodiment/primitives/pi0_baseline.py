#!/usr/bin/env python
"""Pi0.5 fullshot baseline runner.

Loads the Pi0.5 LIBERO-130 checkpoint, builds a single-env sim for the
requested (suite, task, seed), then calls `driver.run_full_task` which
drives Pi0 end-to-end with the env-supplied task_descriptions
(= the BDDL's :language tag) as prompt. Records the per-trial outcome
to an audit JSON.

Usage:
    LIBERO_TYPE=pro CUDA_VISIBLE_DEVICES=0 /opt/venv/openpi/bin/python \\
        examples/embodiment/primitives/pi0_baseline.py \\
        --suite libero_spatial_task --task 0 --seed 0 --max_chunks 60 \\
        --out workspace_pro/results_spatial_pert/baseline_pi0_spatial_task_t0_s0.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

REPO = "/mnt/public2/zhangyixian/RLinf_agentic"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import imageio.v2 as imageio
import numpy as np

from examples.embodiment.primitives.primitives import (
    CHECKPOINT_PATH,
    LiberoPrimitiveDriver,
    build_env_cfg,
    build_model_cfg,
)
from examples.embodiment.primitives.interactive_driver import make_env
from rlinf.models.embodiment.openpi import get_model as get_openpi_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=str, required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_chunks", type=int, default=60)
    p.add_argument("--max_episode_steps", type=int, default=600)
    p.add_argument("--out", type=str, required=True,
                   help="Output audit JSON path (relative to REPO or absolute)")
    p.add_argument("--save_image_dir", type=str, default=None,
                   help="If set, save initial and final agentview PNGs here")
    args = p.parse_args()

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[setup] suite={args.suite}  task={args.task}  seed={args.seed}")
    t_load_start = time.time()
    env = make_env(args.task, args.seed, suite_name=args.suite,
                   max_episode_steps=args.max_episode_steps)
    model_cfg = build_model_cfg(model_path=CHECKPOINT_PATH)
    model = get_openpi_model(model_cfg).cuda().eval()
    driver = LiberoPrimitiveDriver(env=env, model=model, action_chunk=5)
    t_load = time.time() - t_load_start
    driver.reset()

    # Initial observation + prompt
    init_state = driver.get_privileged_state()
    init_obs = driver._last_obs
    task_desc = init_obs.get("task_descriptions", [""])
    if isinstance(task_desc, list):
        task_desc = task_desc[driver.env_idx]
    print(f"[info] task_descriptions (= BDDL :language): {task_desc!r}")
    print(f"[info] Pi0 model load + reset: {t_load:.1f}s")

    if args.save_image_dir:
        os.makedirs(args.save_image_dir, exist_ok=True)
        imageio.imwrite(os.path.join(args.save_image_dir, "initial.png"),
                        driver.render_agentview())

    # Run Pi0 end-to-end
    t0 = time.time()
    result = driver.run_full_task(max_chunks=args.max_chunks)
    elapsed = time.time() - t0

    final_state = driver.get_privileged_state()
    if args.save_image_dir:
        imageio.imwrite(os.path.join(args.save_image_dir, "final.png"),
                        driver.render_agentview())

    audit = {
        "suite": args.suite,
        "task_id": args.task,
        "seed": args.seed,
        "regime": "pi0_fullshot_baseline",
        "rule_1_compliant": False,
        "note": ("Pi0 drives the entire task. The env-supplied "
                 "task_descriptions field carries the perturbed BDDL "
                 ":language tag — Pi0 sees exactly what the LLM-in-the-loop "
                 "hybrid receives."),
        "task_descriptions_seen_by_pi0": task_desc,
        "max_chunks_budget": args.max_chunks,
        "result": result,
        "libero_terminated": result.get("libero_terminated", False),
        "wall_time_s": round(elapsed, 1),
        "model_load_time_s": round(t_load, 1),
        "initial_state": init_state,
        "final_state": final_state,
    }
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n[result] libero_terminated={audit['libero_terminated']} "
          f"chunks_used={result['chunks_used']}/{args.max_chunks} "
          f"wall_time={elapsed:.1f}s")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

"""Generalization test: does pi05_libero130_fullshot accept short sub-instructions?

For each (task_id, mode, seed) combination:
  - mode=full      : use the ORIGINAL task description ("pick up the X and place it on Y").
  - mode=pick_only : feed only "pick up the X" — measure pick.
  - mode=subinstr  : feed "pick up the X" → then "place it on Y".

Measures:
  - peak_lift_m      : max EEF z - start EEF z.
  - min_gripper_opening : how closed the gripper got (proxy for "grabbed something").
  - libero_terminated   : official success flag (only fires after full pick+place).

The expensive ops (model load 90s, model weights to GPU) happen ONCE.
LiberoEnv is rebuilt per task because specific_reset_id is set at __init__.

Output: writes one JSON per run + a summary table to stdout.
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

REPO_PATH = "/mnt/public2/zhangyixian/RLinf_agentic"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

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


def make_env(task_id: int, seed: int, suite_name: str = "libero_spatial"):
    """Build a one-env LiberoEnv pinned to task_id (first trial offset by seed)."""
    from libero.libero.benchmark import get_benchmark

    suite = get_benchmark(suite_name)()
    first_id = sum(len(suite.get_task_init_states(t)) for t in range(task_id))
    trials = len(suite.get_task_init_states(task_id))
    rid = first_id + (seed % trials)

    cfg = build_env_cfg(
        task_suite_name=suite_name,
        specific_reset_id=rid,
        seed=seed,
    )
    env = LiberoEnv(
        cfg=cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None,
    )
    return env


NEG_CONTROL_OBJ = "alphabet soup"  # libero_object item — never in libero_spatial scenes


def run_one(driver, mode: str, obj: str, tgt: str,
            max_chunks_pick: int, max_chunks_place: int, max_chunks_full: int):
    if mode == "full":
        r = driver.run_full_task(max_chunks=max_chunks_full)
        return {"full": r}
    if mode == "pick_only":
        r = driver.pick(obj, max_chunks=max_chunks_pick)
        return {"pick": r.to_dict()}
    if mode == "pick_negctrl":
        # Negative control: prompt names an object not in the scene. If the
        # model still picks the actual scene object → it's ignoring the prompt.
        r = driver.pick(NEG_CONTROL_OBJ, max_chunks=max_chunks_pick)
        return {"pick": r.to_dict(), "neg_obj": NEG_CONTROL_OBJ}
    if mode == "subinstr":
        r1 = driver.pick(obj, max_chunks=max_chunks_pick)
        out = {"pick": r1.to_dict()}
        if r1.peak_lift_m >= 0.02:
            r2 = driver.place(tgt, max_chunks=max_chunks_place)
            out["place"] = r2.to_dict()
        else:
            out["place"] = {"skipped": True, "reason": "pick stroke < 2cm"}
        return out
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--modes", type=str, nargs="+",
                   default=["full", "pick_only", "subinstr"],
                   choices=["full", "pick_only", "subinstr", "pick_negctrl"])
    p.add_argument("--max_chunks_pick", type=int, default=20)
    p.add_argument("--max_chunks_place", type=int, default=24)
    p.add_argument("--max_chunks_full", type=int, default=48)
    p.add_argument("--out_dir", type=str,
                   default="/mnt/public2/zhangyixian/RLinf_agentic/"
                           "examples/embodiment/primitives/results")
    p.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[plan] tasks={args.tasks} seeds={args.seeds} modes={args.modes}")
    print(f"[plan] {len(args.tasks)*len(args.seeds)*len(args.modes)} rollouts total")
    print(f"[plan] out_dir={args.out_dir}")

    # Build model ONCE.
    t0 = time.time()
    print(f"[setup] loading Pi0.5 from {args.checkpoint}...")
    model_cfg = build_model_cfg(model_path=args.checkpoint)
    model = get_openpi_model(model_cfg, torch_dtype=None).cuda().eval()
    print(f"[setup] model ready in {time.time() - t0:.1f}s")

    rows = []
    for task_id in args.tasks:
        obj, tgt = parse_task_object(task_id)
        print(f"\n========== TASK {task_id}: obj=\"{obj}\"  tgt=\"{tgt}\" ==========")
        for seed in args.seeds:
            for mode in args.modes:
                t1 = time.time()
                env = make_env(task_id, seed)
                driver = LiberoPrimitiveDriver(env=env, model=model, action_chunk=5)
                driver.reset()
                result = run_one(
                    driver, mode, obj, tgt,
                    max_chunks_pick=args.max_chunks_pick,
                    max_chunks_place=args.max_chunks_place,
                    max_chunks_full=args.max_chunks_full,
                )
                elapsed = time.time() - t1
                row = {
                    "task_id": task_id, "seed": seed, "mode": mode,
                    "obj": obj, "tgt": tgt,
                    "elapsed_s": round(elapsed, 1),
                    **result,
                }
                rows.append(row)
                # Compact one-line summary.
                summ = _summary_line(row)
                print(f"  [t{task_id} s{seed} {mode:>10}] {summ}  ({elapsed:.1f}s)")
                # Per-run JSON.
                fpath = os.path.join(
                    args.out_dir, f"t{task_id}_s{seed}_{mode}.json"
                )
                with open(fpath, "w") as f:
                    json.dump(row, f, indent=2)
                # Clean up env (Subproc workers).
                try:
                    env.env.close()
                except Exception:
                    pass
                del env, driver
                gc.collect()
                torch.cuda.empty_cache()

    # ---- aggregate summary ----
    print("\n\n=========== SUMMARY TABLE ===========")
    for task_id in args.tasks:
        for mode in args.modes:
            xs = [r for r in rows
                  if r["task_id"] == task_id and r["mode"] == mode]
            if not xs:
                continue
            print(f"[task {task_id}] mode={mode:>10}  "
                  f"{_aggregate_line(xs)}")

    out_all = os.path.join(args.out_dir, "all_rows.json")
    with open(out_all, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[done] {len(rows)} rows → {out_all}")


def _summary_line(row):
    if "pick" in row and isinstance(row["pick"], dict):
        pick = row["pick"]
        s = (f"pick_lift={pick.get('peak_lift_m', 0):.3f}m  "
             f"grip_min={pick.get('min_gripper_opening', 0):.3f}  "
             f"pick_ok={pick.get('success', False)}  "
             f"chunks={pick.get('chunks_used', 0)}/{pick.get('max_chunks', 0)}")
        if "place" in row:
            p = row["place"]
            if p.get("skipped"):
                s += "  place=skip"
            else:
                s += (f"  place_ok={p.get('success', False)}  "
                      f"libero_term={p.get('libero_terminated', False)}")
        return s
    if "full" in row:
        f = row["full"]
        return (f"full_lift={f['peak_lift_m']:.3f}m  "
                f"libero_term={f['libero_terminated']}  "
                f"chunks={f['chunks_used']}/{f['max_chunks']}")
    return ""


def _aggregate_line(rows):
    n = len(rows)
    if "full" in rows[0]:
        ok = sum(r["full"]["libero_terminated"] for r in rows)
        lifts = [r["full"]["peak_lift_m"] for r in rows]
        return (f"n={n}  libero_term={ok}/{n}  "
                f"lift_mean={np.mean(lifts):.3f}  lift_max={np.max(lifts):.3f}")
    if "pick" in rows[0]:
        pick_ok = sum(r["pick"]["success"] for r in rows)
        lifts = [r["pick"]["peak_lift_m"] for r in rows]
        grips = [r["pick"]["min_gripper_opening"] for r in rows]
        s = (f"n={n}  pick_ok={pick_ok}/{n}  "
             f"lift_mean={np.mean(lifts):.3f}  lift_max={np.max(lifts):.3f}  "
             f"grip_min_mean={np.mean(grips):.3f}")
        if any("place" in r and not r["place"].get("skipped") for r in rows):
            placed = [r for r in rows if "place" in r and not r["place"].get("skipped")]
            term = sum(r["place"]["libero_terminated"] for r in placed)
            s += f"  place_n={len(placed)}  libero_term={term}/{len(placed)}"
        return s
    return ""


if __name__ == "__main__":
    main()

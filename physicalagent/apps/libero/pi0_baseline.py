"""Pi0 full-task baseline runner for LIBERO cells.

This script runs the OpenPI/Pi0.5 policy end-to-end with the task language
provided by the environment. It is intended as the apples-to-apples baseline
for hybrid LLM+primitive runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

import imageio.v2 as imageio
import torch

from physicalagent.backends import add_external_rlinf_to_path
from physicalagent.config import get_pi05_checkpoint_path, get_repo_root

PHYSICALAGENT_ROOT = get_repo_root()
add_external_rlinf_to_path(PHYSICALAGENT_ROOT)

from physicalagent.backends.rlinf.primitives import (  # noqa: E402
    LiberoPrimitiveDriver,
    build_env_cfg,
    build_model_cfg,
)
from rlinf.envs.libero.libero_env import LiberoEnv  # noqa: E402
from rlinf.envs.libero.utils import benchmark as benchmark_mod  # noqa: E402
from rlinf.models.embodiment.openpi import get_model as get_openpi_model  # noqa: E402


def _reset_id_for_cell(suite_name: str, task_id: int, seed: int) -> int:
    suite = benchmark_mod.get_benchmark(suite_name)()
    first_id = sum(len(suite.get_task_init_states(t)) for t in range(task_id))
    trials = len(suite.get_task_init_states(task_id))
    return first_id + (seed % trials)


def _task_language(suite_name: str, task_id: int) -> str:
    try:
        suite = benchmark_mod.get_benchmark(suite_name)()
        return str(suite.get_task(task_id).language)
    except Exception:
        return ""


def _make_env(suite_name: str, task_id: int, seed: int, max_episode_steps: int):
    cfg = build_env_cfg(
        task_suite_name=suite_name,
        specific_reset_id=_reset_id_for_cell(suite_name, task_id, seed),
        seed=seed,
        max_episode_steps=max_episode_steps,
    )
    return LiberoEnv(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def run_baseline(
    *,
    suite: str,
    task: int,
    seed: int,
    max_chunks: int,
    max_episode_steps: int,
    model_path: str,
    out: str | None,
    save_image_dir: str | None,
) -> dict:
    t0 = time.time()
    env = _make_env(suite, task, seed, max_episode_steps)

    model_cfg = build_model_cfg(model_path=model_path)
    model = get_openpi_model(model_cfg, torch_dtype=None).cuda().eval()

    driver = LiberoPrimitiveDriver(env=env, model=model, action_chunk=5)
    driver.reset()

    image_dir = Path(save_image_dir) if save_image_dir else None
    if image_dir:
        image_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(image_dir / "initial.png", driver.render_agentview())

    result = driver.run_full_task(max_chunks=max_chunks)

    if image_dir:
        imageio.imwrite(image_dir / "final.png", driver.render_agentview())

    audit = {
        "suite": suite,
        "task_id": task,
        "seed": seed,
        "regime": "pi0_fullshot_baseline",
        "task_language": _task_language(suite, task),
        "max_chunks": max_chunks,
        "max_episode_steps": max_episode_steps,
        "full_task": result,
        "final_state": driver.get_privileged_state(),
        "libero_terminated": bool(result.get("libero_terminated")),
        "elapsed_s": round(time.time() - t0, 2),
    }

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(audit, indent=2))
        print(f"[done] wrote {out_path}")
    else:
        print(json.dumps(audit, indent=2))
    return audit


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Pi0 full-task LIBERO baseline")
    parser.add_argument("--suite", required=True, help="e.g. libero_spatial_task")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_chunks", type=int, default=60)
    parser.add_argument("--max_episode_steps", type=int, default=600)
    parser.add_argument("--model_path", default=None,
                        help="Pi0.5 checkpoint path; defaults to PI05_CHECKPOINT_PATH")
    parser.add_argument("--out", default=None, help="Audit JSON path")
    parser.add_argument("--save_image_dir", default=None,
                        help="Directory for initial.png and final.png")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    model_path = args.model_path or get_pi05_checkpoint_path()
    if not model_path:
        print("ERROR: set PI05_CHECKPOINT_PATH or pass --model_path", file=sys.stderr)
        return 2
    run_baseline(
        suite=args.suite,
        task=args.task,
        seed=args.seed,
        max_chunks=args.max_chunks,
        max_episode_steps=args.max_episode_steps,
        model_path=model_path,
        out=args.out,
        save_image_dir=args.save_image_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

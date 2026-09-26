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

"""Construct the RoboDojo agent environment and serve RPC on the main thread."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.robodojo.env_facade import RoboDojoEnvFacade  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoboDojo Isaac Sim env RPC server")
    parser.add_argument("--task", default="put_bottles_into_dustbin")
    parser.add_argument("--layout", type=int, default=0, help="layout id == seed")
    parser.add_argument("--env-cfg-type", default="arx_x5")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=700)
    parser.add_argument("--save-dir", default=os.getcwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Directory for per-camera episode mp4 files",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sample a fresh random scene layout per episode "
        "(random template + fresh env seed, official-eval style)",
    )
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument("--mode", choices=["dev", "eval-fair"], default="dev")

    parser.add_argument("--source-root", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--kit_args", default="")
    return parser


def build_config(args: argparse.Namespace):
    """Translate CLI settings into the RLinf wrapper and runtime task config."""
    from omegaconf import OmegaConf

    if args.num_envs != 1:
        raise ValueError("RoboDojo RPC serves exactly one environment")
    return OmegaConf.create(
        {
            "seed": args.layout,
            "group_size": 1,
            "auto_reset": False,
            "ignore_terminations": False,
            "use_rel_reward": False,
            "use_custom_reward": False,
            "use_fixed_reset_state_ids": True,
            "max_episode_steps": args.max_episode_steps,
            "task_config": {
                "task_name": args.task,
                "env_cfg_type": args.env_cfg_type,
                "source_root": args.source_root,
                "cuda_device": args.cuda_device,
                "save_dir": args.save_dir,
                "headless": args.headless,
                "kit_args": args.kit_args,
                "random": args.random,
                "max_episode_steps": args.max_episode_steps,
            },
        }
    )


def main() -> None:
    args = build_parser().parse_args()
    cfg = build_config(args)
    from robots.robodojo.rlinf_env import RoboDojoAgentEnv

    env = RoboDojoAgentEnv(
        cfg,
        video_dir=args.video_dir or "",
        meta={
            "task": args.task,
            "layout": args.layout,
            "env_cfg_type": args.env_cfg_type,
            "device_id": args.cuda_device,
            "num_envs": args.num_envs,
            "max_episode_steps": args.max_episode_steps,
            "random": args.random,
            **({"mode": "eval-fair"} if args.mode == "eval-fair" else {}),
        },
    )
    try:
        RoboDojoEnvFacade(env).serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()

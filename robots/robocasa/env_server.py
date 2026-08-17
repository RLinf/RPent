"""RoboCasa env server — hosts an rlinf Robocasa365Env in a subprocess and
exposes basic calls via RPC.

The raw robosuite env lives in an rlinf subprocess (``RobocasaSubprocEnv``)
so the parent process never touches the MuJoCo/EGL context. This facade
translates ``env.*`` RPC calls into ``Robocasa365Env`` method calls.

All worker returns are numpy arrays or plain Python — robosuite/MuJoCo output
is numpy, and obs cross the subprocess boundary via numpy buffers. The
``_NumpyEncoder`` in http_rpc tags numpy arrays at JSON serialization time,
so no torch → numpy conversion is needed here. If a future env backend ever
returns torch tensors (e.g. a GPU-resident renderer), add a
``.detach().cpu().numpy()`` step at the affected call sites — ``_NumpyEncoder``
only handles numpy, not torch.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
from omegaconf import OmegaConf

from rpent.utils.config import (
    get_repo_root,
    get_rlinf_repo_path,
)
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

# MuJoCo env vars must be set BEFORE importing anything that touches MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
assert "mujoco" not in sys.modules, \
    "mujoco must not be imported before MUJOCO_GL/PYOPENGL_PLATFORM are set"

logger = get_logger("env_server")

RPENT_ROOT = get_repo_root()


def _resolve_rlinf_repo_path() -> Any:
    """Pick the rlinf checkout to put on sys.path.

    Order: explicit env var → sibling ``rlinf`` checkout → in-repo
    ``rlinf_robocasa`` (the local development layout). Falls back to the
    sibling layout for parity with other envs.
    """
    env_path = get_rlinf_repo_path()
    if env_path is not None:
        return env_path
    candidates = [
        (RPENT_ROOT.parent / "rlinf").resolve(),
        (RPENT_ROOT / "rlinf_robocasa").resolve(),
    ]
    for path in candidates:
        if (path / "rlinf" / "__init__.py").is_file():
            return path
    # Last resort: keep parity with the legacy default even if it doesn't
    # exist; an explicit env var (RPENT_RLINF_ROOT / RLINF_REPO_PATH) is the
    # supported way to override.
    return candidates[0]


RLINF_REPO_PATH = _resolve_rlinf_repo_path()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))
# multiprocessing.spawn on macOS does NOT inherit sys.path mutations, so the
# RobocasaSubprocEnvWorker subprocess (spawned by Robocasa365Env) can't import
# rlinf unless rlinf is also exposed via PYTHONPATH. Mirror the sys.path entry.
pythonpath = os.environ.get("PYTHONPATH", "")
if str(RLINF_REPO_PATH) not in pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        str(RLINF_REPO_PATH) + (os.pathsep + pythonpath if pythonpath else "")
    )
os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")

# Robocasa365Env is only imported at call time (after --cuda-device sets
# CUDA_VISIBLE_DEVICES in main()); it transitively imports torch via rlinf.
if TYPE_CHECKING:
    from rlinf.envs.robocasa365.robocasa365_env import Robocasa365Env


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

# RPent's ``--robocasa-split`` choices are ``target`` / ``pretrain`` / ``all``.
# RoboCasa365's official dataset registry only recognises ``pretrain`` and
# ``target`` (see ``robocasa.utils.dataset_registry_utils.get_ds_soup``).
# ``all`` is mapped to ``pretrain`` (the larger of the two splits) with a
# warning so existing CLI scripts that pass ``all`` keep working.
_SPLIT_ALIASES = {"target": "target", "pretrain": "pretrain", "all": "pretrain"}


def _normalize_split(split: str) -> str:
    mapped = _SPLIT_ALIASES.get(split)
    if mapped is None:
        raise ValueError(
            f"--split must be one of {sorted(_SPLIT_ALIASES)}, got {split!r}"
        )
    if split == "all":
        logger.warning(
            "--split all is not a RoboCasa365 benchmark split; "
            "mapping to pretrain. Use --split target or --split pretrain "
            "for an unambiguous benchmark slice."
        )
    return mapped


def build_env_cfg(
    *,
    env_name: str,
    split: str,
    seed: int,
    camera_h: int = 256,
    camera_w: int = 256,
    max_episode_steps: int = 600,
) -> Any:
    """Build the OmegaConf cfg consumed by ``Robocasa365Env``.

    The cfg mirrors ``examples/embodiment/config/env/robocasa365.yaml`` from
    rlinf: ``task_source=dataset_registry`` selects tasks through the
    official RoboCasa dataset registry, ``split`` picks the benchmark slice
    (``pretrain`` / ``target``), and ``task_names=[env_name]`` restricts
    the selection to the single task the user passed on the CLI.  The
    agent-side facade renders images on demand via ``render_raw``
    (EGL-safe); ``has_renderer=False`` keeps robosuite's on-screen window
    off, and ``use_camera_obs=False`` is the intent (note: robocasa's
    ``create_env`` currently derives ``use_camera_obs`` from
    ``render_onscreen`` itself, so the per-step camera render still runs —
    it is simply ignored by the facade, which only reads raw state obs).
    """
    normalized_split = _normalize_split(split)
    return OmegaConf.create({
        "env_type": "robocasa365",
        "task_source": "dataset_registry",
        "dataset_source": "human",
        "split": normalized_split,
        "task_names": [env_name],
        "task_sampling_strategy": "ordered",  # single-env, single-task
        "rotate_tasks_on_auto_reset": False,
        "robot_name": "PandaOmron",
        "camera_names": [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        "render_camera": "robot0_agentview_left",
        "has_renderer": False,
        "use_camera_obs": False,
        "camera_depths": False,
        "translucent_robot": False,
        "auto_reset": False,
        "ignore_terminations": True,
        "use_rel_reward": False,
        "reward_coef": 1.0,
        "episode_horizon_source": "max_episode_steps",
        "max_episode_steps": max_episode_steps,
        "max_steps_per_rollout_epoch": max_episode_steps,
        "use_fixed_reset_state_ids": True,
        "is_eval": True,
        "group_size": 1,
        "seed": seed,
        "seed_strategy": "worker_offset",
        "init_params": {
            "camera_widths": camera_w,
            "camera_heights": camera_h,
        },
        "video_cfg": {
            "save_video": False,
            "info_on_video": False,
            "video_base_dir": "/tmp/primitive_videos",
        },
    })


def make_env(
    env_name: str,
    split: str = "target",
    seed: int = 0,
    camera_h: int = 256,
    camera_w: int = 256,
    max_episode_steps: int = 600,
) -> Robocasa365Env:
    """Build a single-env ``Robocasa365Env`` pinned to ``env_name`` / ``seed``."""
    from rlinf.envs.robocasa365.robocasa365_env import Robocasa365Env
    cfg = build_env_cfg(
        env_name=env_name,
        split=split,
        seed=seed,
        camera_h=camera_h,
        camera_w=camera_w,
        max_episode_steps=max_episode_steps,
    )
    return Robocasa365Env(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class RoboCasaEnvFacade(RpcFacade):
    """Implements :class:`robots.robocasa.env_client.RoboCasaEnvClient` over
    :class:`rlinf.envs.robocasa365.robocasa365_env.Robocasa365Env`.
    """

    def __init__(self, env: Robocasa365Env, *, meta: dict):
        super().__init__()
        self.env = env
        self.env_idx = 0
        self._terminated = False
        self._last_obs = None
        # Identifies what env/seed/split this server was launched with — the
        # client compares against its own expected values at construction and
        # refuses to talk to a stale or mis-configured server.
        self._meta = dict(meta)

    def get_env_meta(self):
        return self._meta

    # ---- RPC dispatch ----
    def _dispatch(self, method: str, args: tuple, kwargs: dict, *, session_id: str | None = None) -> Any:
        """Route ``env.*`` calls to the matching facade method."""
        with self._lock:
            if method.startswith("env."):
                attr = method[len("env."):]
                try:
                    return getattr(self, attr)(*args, **kwargs)
                except Exception as e:
                    logger.warning("run method %s failed: %s", method, e)
                    raise e
            raise ValueError(f"unknown RPC method: {method!r}")

    # ---- lifecycle ----
    def reset(self):
        # RLDX_RESET_SEED=<episode_seed> -> reproduce the EXACT scene the fullshot eval
        # generated for that episode, seeded the SAME way as the eval's
        # VideoRecordingWrapper (random.seed + np.random.seed + robosuite
        # env.rng/seed) BEFORE reset.  Lets the hybrid run on the IDENTICAL
        # reset layouts fullshot was scored on (true paired comparison).
        # The eval formula: episode_seed = (run_seed + env_idx)*100000 + episode_id.
        rs_env = os.environ.get("RLDX_RESET_SEED")
        if rs_env:
            self.env.set_seed(int(rs_env), env_idx=self.env_idx)
        obs, _info = self.env.raw_reset(env_idx=self.env_idx)
        self._last_obs = obs
        self._terminated = False
        return self._last_obs

    def step(self, flat_action):
        """flat_action: np.ndarray[12] = [eef_pos(3), eef_rot(3), gripper(1),
        base_motion(4), control_mode(1)] in the PandaOmron composite layout."""
        a = np.asarray(flat_action, dtype=np.float64).reshape(-1)
        action_dim = self.env.get_action_dim(env_idx=self.env_idx)
        assert a.shape[0] == action_dim, (
            f"action dim {a.shape[0]} != env.action_dim {action_dim}"
        )
        obs, reward, done, info = self.env.raw_step(a, env_idx=self.env_idx)
        self._last_obs = obs
        if self.env.check_success(env_idx=self.env_idx):
            self._terminated = True
        return obs, reward, done, info

    def check_success(self):
        return bool(self.env.check_success(env_idx=self.env_idx))

    def raw_obs(self):
        return self._last_obs

    def render_raw(self, cam, h, w, depth):
        """sim.render in ROBOSUITE-NATIVE orientation (matches the camera
        transform matrices). rgb uint8 HxWx3, depth metric HxW."""
        return self.env.render_raw(cam, h, w, depth, env_idx=self.env_idx)

    def get_camera_meta(self, camera_name, height=None, width=None):
        meta = self.env.get_camera_meta(
            camera_name=camera_name,
            height=height,
            width=width,
            env_idx=self.env_idx,
        )
        return meta

    def get_camera_transform(self, camera_name, height=None, width=None):
        # Worker returns the pixel-to-world 4x4 (inv of the camera transform).
        return self.env.get_camera_transform(
            camera_name=camera_name,
            height=height,
            width=width,
            env_idx=self.env_idx,
        )

    def get_ep_meta(self):
        return self.env.get_ep_meta(env_idx=self.env_idx)

    def get_terminated(self):
        return self._terminated or self.check_success()

    def get_action_dim(self):
        return self.env.get_action_dim(env_idx=self.env_idx)

    def grasp_contact(self):
        """Check if the gripper is currently contacting a task object."""
        return self.env.grasp_contact(env_idx=self.env_idx)

    def reassemble_env_action(self, unmap_result):
        """Reassemble the unmap result into a flat action using the env's robots."""
        return self.env.reassemble_env_action(unmap_result, env_idx=self.env_idx)

    def get_success_criteria_text(self):
        """Return the success_criteria.md text for this task."""
        return self.env.get_success_criteria_text(env_idx=self.env_idx)

    def get_task_progress(self):
        """Return the progress dict for this task."""
        return self.env.get_task_progress(env_idx=self.env_idx)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--parent-watch", action="store_true",
                   help="watch parent process via stdin pipe and exit when it dies")
    p.add_argument("--cuda-device", type=int, default=None,
                   help="GPU device to pin MuJoCo EGL rendering and the torch "
                        "default device to (physical CUDA ordinal).")
    p.add_argument("--env", dest="env_name", default="OpenDrawer")
    p.add_argument("--split", default="target")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=0,
                   help="RoboCasa365 eval horizon. 0 -> server-side default (600). "
                        "Must be positive — RoboCasa365 validates it on init.")
    args = p.parse_args()

    if args.cuda_device is not None:
        # Deliberately do NOT set CUDA_VISIBLE_DEVICES. robosuite (imported
        # transitively via libero) asserts at import time that
        # ``MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`` (substring check),
        # which assumes the EGL index equals the CUDA ordinal and crashes on
        # multi-GPU boxes where the EGL order differs. That assertion is gated
        # on ``CUDA_VISIBLE_DEVICES != ""``, so leaving it unset skips it in
        # both this process and the multiprocessing-spawned render workers
        # (which inherit the env). Pin the two backends directly instead:
        #   - MuJoCo render device <- MUJOCO_EGL_DEVICE_ID (configure_egl_device)
        #   - torch default device  <- torch.cuda.set_device(N)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is set; clearing it and pinning via "
                "MUJOCO_EGL_DEVICE_ID + torch.cuda.set_device(--cuda-device=%s) "
                "instead (robosuite's CVD assertion is incompatible with EGL<->CUDA mapping)",
                prev, args.cuda_device,
            )
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from rpent.utils.egl import configure_egl_device
        configure_egl_device(args.cuda_device)
        import torch
        torch.cuda.set_device(args.cuda_device)

    # RoboCasa365 requires max_episode_steps > 0 when
    # episode_horizon_source='max_episode_steps' (which build_env_cfg sets), so
    # a 0 / unset CLI value falls back to the server-side default of 600.
    max_episode_steps = args.max_episode_steps if args.max_episode_steps > 0 else 600
    raw_env = make_env(
        args.env_name,
        split=args.split,
        seed=args.seed,
        max_episode_steps=max_episode_steps,
    )
    facade = RoboCasaEnvFacade(
        raw_env,
        meta={
            "env_name": args.env_name,
            "split": args.split,
            "seed": args.seed,
            "camera_h": 256,
            "camera_w": 256,
        },
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()

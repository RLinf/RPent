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

"""RPC server wrapping a single-env LIBERO environment."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from rpent.robots.components.action_spec import box_action_spec
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.logging import get_logger
from rpent.utils.serialization import to_numpy_tree

# MuJoCo env vars must be set BEFORE importing anything that touches MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
assert "mujoco" not in sys.modules, (
    "mujoco must not be imported before MUJOCO_GL/PYOPENGL_PLATFORM are set"
)

logger = get_logger("libero_env_server")

os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

# torch and LiberoEnv are only imported at call time (after --cuda-device
# sets CUDA_VISIBLE_DEVICES in main()); LiberoEnv transitively imports torch.
if TYPE_CHECKING:
    import torch  # noqa: F401  (transitive dep of LiberoEnv; type-check only)
    from rlinf.envs.sim.libero.libero_env import LiberoEnv


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_env_cfg(
    *,
    task_suite_name: str = "libero_spatial",
    specific_reset_id: int = 0,
    seed: int = 0,
    max_episode_steps: int = 10000,
) -> Any:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "env_type": "libero",
            "task_suite_name": task_suite_name,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_steps_per_rollout_epoch": max_episode_steps,
            "max_episode_steps": max_episode_steps,
            "use_rel_reward": False,
            "use_step_penalty": False,
            "reward_coef": 1.0,
            "reset_gripper_open": True,
            "is_eval": True,
            "seed": seed,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": True,
            "specific_reset_id": specific_reset_id,
            "video_cfg": {
                "save_video": True,
                "info_on_video": True,
                "video_base_dir": "/tmp/primitive_videos",
            },
            "init_params": {
                "camera_heights": 256,
                "camera_widths": 256,
                # Render depth too, so we can back-project pixels to world
                # from depth + camera calibration
                "camera_depths": True,
                "horizon": max_episode_steps,
                # RLinf counts policy steps after reset settling and owns truncation.
                # LIBERO replaces robosuite's done flag with task success.
                "ignore_done": True,
                **(
                    {"robots": [os.environ["LIBERO_ROBOT_BASE"]]}
                    if os.environ.get("LIBERO_ROBOT_BASE")
                    else {}
                ),
            },
        }
    )
    return cfg


def make_env(
    task_id: int,
    seed: int,
    suite_name: str = "libero_spatial",
    max_episode_steps: int = 10000,
) -> LiberoEnv:
    """Build a single-env LiberoEnv pinned to ``task_id`` / ``seed``."""
    from rlinf.envs.sim.libero.libero_env import LiberoEnv
    from rlinf.envs.sim.libero.utils import benchmark as _bench_mod

    suite = _bench_mod.get_benchmark(suite_name)()
    if not 0 <= task_id < suite.n_tasks:
        raise ValueError(
            f"task_id must be in [0, {suite.n_tasks}) for suite {suite_name!r}; got {task_id}"
        )
    trials = len(suite.get_task_init_states(task_id))
    if trials == 0:
        raise ValueError(f"Suite {suite_name!r}, task {task_id} has no initial states")
    first_id = sum(len(suite.get_task_init_states(t)) for t in range(task_id))
    rid = first_id + (seed % trials)
    cfg = build_env_cfg(
        task_suite_name=suite_name,
        specific_reset_id=rid,
        seed=seed,
        max_episode_steps=max_episode_steps,
    )
    return LiberoEnv(
        cfg=cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None
    )


# ---------------------------------------------------------------------------
# Facade implementing the robots.libero.env_client protocol
# ---------------------------------------------------------------------------


class LiberoEnvFacade(BaseEnvFacade):
    """Implements :class:`robots.libero.env_client.LiberoEnvClient`
    over :class:`rlinf.envs.sim.libero.libero_env.LiberoEnv`.

    All return values are converted to CPU numpy so the agent process
    (which does not import torch) can consume them after the pickle round
    trip.
    """

    def __init__(self, env: LiberoEnv, *, meta: dict):
        self._env = env
        self._env_idx = 0
        self._closed = False
        self._episode_id = None
        self._measured = False
        self._sealed = False
        self._success: bool | None = None
        self._first_success = None
        self._truncated = False
        # Identifies what task/seed this server was launched with — the
        # client compares against its own expected values at construction
        # and refuses to talk to a stale or mis-configured server.
        self._meta = dict(meta)
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc.update(
            {
                "env.begin_trial": self.begin_trial,
                "env.trial_actions": self.trial_actions,
                "env.seal_episode": self.seal_episode,
                "env.raw_obs": self.raw_obs,
                "env.render_camera": self.render_camera,
                "env.get_camera_meta": self.get_camera_meta,
                "env.get_task_language": self.get_task_language,
            }
        )
        self._readonly_methods.add("env.get_task_language")

    # ---- shape helpers ----

    def _strip(self, v):
        """Drop the leading env dim. ``v`` is either a batched numpy array
        (shape ``[B, ...]``), a length-B list (e.g. ``task_descriptions``),
        or ``None`` (optional images). LiberoEnv runs ``num_envs=1`` so
        index ``self._env_idx`` is always present."""
        if v is None:
            return None
        return v[self._env_idx]

    def _strip_obs(self, obs: dict) -> dict:
        """Strip the leading env dim from every value of a LIBERO obs dict."""
        return {k: self._strip(v) for k, v in obs.items()}

    def _expand_action(self, action) -> np.ndarray:
        """Inject the env dim onto a single-env action shaped ``[action_dim]``."""
        return np.asarray(action)[None]

    def _expand_chunk(self, actions) -> np.ndarray:
        """Inject the env dim onto a single-env chunk shaped
        ``[chunk_size, action_dim]``."""
        return np.asarray(actions)[None]

    # ---- gym-like surface ----

    def reset(self):
        if self._measured and not self._sealed:
            raise RuntimeError("Cannot reset an active evaluation episode")
        self._episode_id = uuid4().hex
        self._measured = False
        self._sealed = False
        obs, info = self._env.reset()
        obs = self._strip_obs(to_numpy_tree(obs))
        return obs, {**to_numpy_tree(info), "episode_id": self._episode_id}

    def step(self, action):
        if self._measured:
            raise RuntimeError("Measured actions require an active episode ID")
        return self._step(action)

    def _step(self, action):
        obs, rew, term, trunc, info = self._env.step(self._expand_action(action))
        obs = self._strip_obs(to_numpy_tree(obs))
        term = self._strip(to_numpy_tree(term))
        trunc = self._strip(to_numpy_tree(trunc))
        return (
            obs,
            self._strip(to_numpy_tree(rew)),
            term,
            trunc,
            to_numpy_tree(info),
        )

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        """Run a full action chunk in one RPC. ``actions`` shape
        ``[chunk_size, action_dim]`` (single env).

        Returns the 5-positional tuple
        ``(obs_or_list, reward, terminated, truncated, info)``. ``obs`` is
        ``list[Obs]`` when ``return_all_frames=True`` (full per-step
        trajectory), or just the final ``Obs`` dict when False (default).
        ``terminated`` / ``truncated`` carry shape ``[chunk_size]`` after
        the leading env dim is stripped — the agent reduces across the
        chunk itself.
        """
        if self._measured:
            raise RuntimeError("Measured actions require an active episode ID")
        obs_list, rew, term, trunc, info = self._env.chunk_step(
            self._expand_chunk(actions)
        )
        obs_list = [self._strip_obs(to_numpy_tree(o)) for o in obs_list]
        term = self._strip(to_numpy_tree(term))
        trunc = self._strip(to_numpy_tree(trunc))
        obs_field = obs_list if return_all_frames else obs_list[-1]
        return (
            obs_field,
            self._strip(to_numpy_tree(rew)),
            term,
            trunc,
            to_numpy_tree(info),
        )

    def begin_trial(self, episode_id: str) -> dict:
        """Bind one reset to its evaluator and read the settled native predicate."""
        if not episode_id or episode_id != self._episode_id or self._measured:
            raise RuntimeError("Cannot admit a stale or previously admitted episode")
        self._success = bool(self._env.env.check_success()[self._env_idx])
        self._first_success = None
        self._truncated = False
        self._measured = True
        return self._trial_status()

    def _trial_status(self) -> dict:
        return {
            "episode_id": self._episode_id,
            "success": self._success,
            "truncated": self._truncated,
            "sealed": self._sealed,
            "native_success_age_s": (
                time.monotonic() - self._first_success
                if self._first_success is not None
                else None
            ),
        }

    def seal_episode(self, episode_id: str) -> dict:
        """Fence further actions under the same write lock as execution."""
        if not episode_id or episode_id != self._episode_id:
            raise RuntimeError("Cannot seal a stale episode")
        if not self._measured:
            # The client may have lost its begin response or overtaken it. Revoke
            # the reservation without querying an environment of uncertain state.
            self._measured = self._sealed = True
            return {"episode_id": episode_id, "sealed": True}
        self._sealed = True
        status = self._trial_status()
        # Report the fresh native final predicate separately: success at any
        # earlier admitted step remains the authoritative latched verdict.
        status["final_native_success"] = bool(
            self._env.env.check_success()[self._env_idx]
        )
        status["executed_steps"] = int(self._env.elapsed_steps[self._env_idx])
        return status

    def trial_actions(
        self, episode_id, actions, *, return_all_frames=False, remaining_s=None
    ):
        """Execute only the native control steps admitted within this trial."""
        if episode_id != self._episode_id or not self._measured or self._sealed:
            raise RuntimeError("Evaluation episode is stale or sealed")
        if self._success or self._truncated:
            raise RuntimeError("Evaluation episode is already terminal")
        deadline = None if remaining_s is None else time.monotonic() + remaining_s
        frames, rewards, terminated, truncated, infos = [], [], [], [], []
        for action in actions:
            if deadline is not None and time.monotonic() >= deadline:
                break
            # If an exception interrupts a state change, its outcome is unknown.
            self._success = None
            obs, reward, term, trunc, info = self._step(action)
            self._success = bool(term)
            self._truncated = bool(trunc)
            if self._success:
                self._first_success = time.monotonic()
            if return_all_frames:
                frames.append(obs)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            infos.append(info)
            if self._success or self._truncated:
                break
        if not rewards:
            raise ValueError("No native steps were admitted")
        return (
            frames if return_all_frames else obs,
            np.asarray(rewards),
            np.asarray(terminated),
            np.asarray(truncated),
            infos,
            self._trial_status(),
        )

    def raw_obs(self) -> dict:
        return to_numpy_tree(self._env.current_raw_obs[self._env_idx])

    def get_action_spec(self) -> dict:
        """Describe the active single-arm OSC layout and native input bounds."""
        worker = self._env.env.workers[self._env_idx]
        low, high = worker.env_call(
            "__getattribute__", args=["action_spec"], target="robosuite"
        )
        robot_config = worker.env_call(
            "__getattribute__", args=["robot_configs"], target="robosuite"
        )[0]
        # Standard LIBERO uses a composite controller; LIBERO-plus uses
        # robosuite 1.4's single arm controller configuration.
        if "composite_controller_config" in robot_config:
            arm = robot_config["composite_controller_config"]["body_parts"]["right"]
            frame = arm.get("input_ref_frame", "base")
            mode = arm.get("input_type", "delta")
        else:
            arm = robot_config["controller_config"]
            frame = "world"
            mode = "delta" if arm.get("control_delta", True) else "absolute"
        impedance = arm.get("impedance_mode", "fixed")
        gains = {
            "fixed": "",
            "variable_kp": "6 stiffness gains, then ",
            "variable": "6 damping ratios, 6 stiffness gains, then ",
        }[impedance]
        description = (
            f"Native {arm['type']} action: {gains}"
            "[x, y, z, rx, ry, rz], then gripper. "
            f"Arm commands use the {frame} reference frame and {mode} control; "
            "rotation is an axis-angle vector, not Euler angles. "
        )
        if mode == "delta":
            description += (
                "Arm inputs are normalized controller commands, linearly mapped "
                f"from input_min={arm['input_min']} / input_max={arm['input_max']} "
                f"to output_min={arm['output_min']} / output_max={arm['output_max']} "
                "(xyz in metres per action; rotation in radians per action). "
            )
        else:
            description += (
                "Arm inputs are absolute xyz in metres and axis-angle in radians. "
            )
        description += "Gripper: -1 opens, +1 closes, 0 keeps the current command."
        return {"default": box_action_spec(low, high, description)}

    def get_env_meta(self) -> dict:
        """Return the meta info this server was launched with."""
        return dict(self._meta)

    def close(self) -> None:
        """Release the server-owned LIBERO environment once."""
        if self._closed:
            return
        self._env.env.close()
        self._closed = True

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        return to_numpy_tree(
            self._env.render_camera(
                camera_name=camera_name,
                height=height,
                width=width,
                depth=depth,
            )
        )

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict | None:
        return to_numpy_tree(
            self._env.get_camera_meta(
                camera_name=camera_name, height=height, width=width
            )
        )

    def get_task_language(self) -> str | None:
        return self._env.task_descriptions[self._env_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--suite", type=str, default="libero_spatial")
    p.add_argument("--task", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=10000)
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device to pin MuJoCo EGL rendering and the torch "
        "default device to (physical CUDA ordinal).",
    )
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
                prev,
                args.cuda_device,
            )
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from rpent.utils.egl import configure_egl_device

        configure_egl_device(args.cuda_device)
        import torch

        torch.cuda.set_device(args.cuda_device)

    raw_env = make_env(
        args.task,
        args.seed,
        suite_name=args.suite,
        max_episode_steps=args.max_episode_steps,
    )
    facade = LiberoEnvFacade(
        raw_env,
        meta={
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
        },
    )
    try:
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        facade.close()


if __name__ == "__main__":
    main()

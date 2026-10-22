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
from typing import TYPE_CHECKING, Any

import numpy as np

from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.logging import get_logger
from rpent.utils.serialization import to_numpy_tree

# MuJoCo env vars must be set BEFORE importing anything that touches MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
assert "mujoco" not in sys.modules, (
    "mujoco must not be imported before MUJOCO_GL/PYOPENGL_PLATFORM are set"
)

logger = get_logger("env_server")

os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

# Some managed Python installs mark robosuite sources without a numba cache
# locator. Disable only persistent caching before LIBERO imports robosuite;
# numerical kernels remain JIT compiled.
try:
    import importlib.util
    import site
    from pathlib import Path

    for _site in site.getsitepackages():
        _macro_file = Path(_site) / "robosuite" / "macros.py"
        if _macro_file.is_file():
            _spec = importlib.util.spec_from_file_location("robosuite.macros", _macro_file)
            _macros = importlib.util.module_from_spec(_spec)
            _macros.CACHE_NUMBA = False
            sys.modules["robosuite.macros"] = _macros
            _spec.loader.exec_module(_macros)
            _macros.CACHE_NUMBA = False
            break
except ImportError:
    pass

# torch and LiberoEnv are only imported at call time (after --cuda-device
# sets CUDA_VISIBLE_DEVICES in main()); LiberoEnv transitively imports torch.
if TYPE_CHECKING:
    import torch  # noqa: F401  (transitive dep of LiberoEnv; type-check only)
    from rlinf.envs.libero.libero_env import LiberoEnv


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
                **(
                    {"robots": [os.environ["LIBERO_ROBOT_BASE"]]}
                    if os.environ.get("LIBERO_ROBOT_BASE")
                    else {}
                ),
            },
        }
    )
    return cfg


def _attach_pose_reader(env):
    """Give a worker-side LIBERO env a read-only ``read_poses`` method.

    The simulator lives in rlinf's subprocess worker, and fixtures (cabinets,
    stoves, microwaves) and their articulated links are not in the observation
    dict. ``read_poses`` returns world poses straight from MuJoCo so the
    server can expose them as ``env.poses``.
    """
    import mujoco

    def read_poses(bodies=None, sites=()):
        """``{name: [x, y, z, qw, qx, qy, qz]}``; ``bodies=None`` reads every body."""
        sim = env.env.sim
        model = sim.model
        if bodies is None:
            bodies = [model.body_id2name(i) for i in range(model.nbody)]
        out = {}
        for name in bodies:
            i = model.body_name2id(name)
            out[name] = [float(v) for v in sim.data.body_xpos[i]] + [
                float(v) for v in sim.data.body_xquat[i]
            ]
        for name in sites:
            i = model.site_name2id(name)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(
                quat, np.asarray(sim.data.site_xmat[i], dtype=np.float64)
            )
            out[name] = [float(v) for v in sim.data.site_xpos[i]] + [
                float(v) for v in quat
            ]
        return out

    def check_paths(paths, options=None):
        """Static feasibility of grip-site paths; see ``robots.libero.mt3.feasibility``."""
        from robots.libero.mt3.feasibility import check_paths as _check

        return _check(env.env.sim, paths, **(options or {}))

    def sim_state():
        """The flattened MuJoCo state (LIBERO demo ``states`` format)."""
        return [float(v) for v in env.get_sim_state()]

    def plan_path(goal_pose, options=None):
        """Joint-space RRT path for one grip-site goal on copied scene state."""
        from robots.libero.mt3.feasibility import plan_path as _plan

        return _plan(env.env.sim, goal_pose, **(options or {}))

    env.read_poses = read_poses
    env.check_paths = check_paths
    env.sim_state = sim_state
    env.plan_path = plan_path
    return env


def _with_pose_reader(env_fn):
    """Wrap an rlinf env factory so the env it builds can report poses."""

    def make():
        env = _configure_fixture_seed(env_fn())
        return _attach_pose_reader(_fresh_scene(_restore_scene_after_reset(env)))

    return make


def _restore_scene_after_reset(env):
    """Restore saved fixture roots for a paired replay experiment."""
    snapshot_path = os.environ.get("RPENT_MT3_SCENE_SNAPSHOT")
    if not snapshot_path:
        return env
    import json
    from pathlib import Path

    snapshot = json.loads(Path(snapshot_path).read_text())
    original_reset = env.reset

    def reset():
        original_reset()
        sim = env.env.sim
        for fixture in env.env.fixtures_dict.values():
            name = fixture.root_body
            body = sim.model.body_name2id(name)
            if int(sim.model.body_parentid[body]) != 0:
                raise ValueError(f"fixture {name} is not world-rooted")
            values = snapshot[name]
            sim.model.body_pos[body] = values[:3]
            sim.model.body_quat[body] = values[3:]
        sim.forward()
        return env.regenerate_obs_from_state(env.get_sim_state())

    env.reset = reset
    return env


def _fresh_scene(env):
    """Sample a new scene instead of a benchmark init state (data generation).

    With ``RPENT_MT3_FRESH_SCENE=<seed>``, ``set_init_state`` ignores the
    benchmark state and resets the env again with the NumPy global RNG seeded,
    so object and fixture placements are drawn anew from the task's BDDL
    regions. These scenes are disjoint from the 50 evaluation init states.
    """
    fresh = os.environ.get("RPENT_MT3_FRESH_SCENE")
    if not fresh:
        return env

    def set_init_state(init_state=None):
        np.random.seed(int(fresh))
        env.reset()
        return env.regenerate_obs_from_state(env.get_sim_state())

    env.set_init_state = set_init_state
    return env


def _configure_fixture_seed(env):
    """Select native RPent initialization or the deterministic pilot repair."""
    policy = os.environ.get("RPENT_MT3_FIXTURE_SEED_POLICY", "deterministic")
    if policy == "rpent":
        return env
    if policy == "deterministic":
        return _seed_fixture_sampler(env)
    raise ValueError(f"unknown RPENT_MT3_FIXTURE_SEED_POLICY: {policy!r}")


def _seed_fixture_sampler(env):
    """Seed LIBERO's legacy placement RNG along with robosuite's generator.

    LIBERO's fixture samplers use the NumPy global RNG. Under robosuite 1.5,
    the wrapper's seed method resets only the generator owned by robosuite;
    fixture placements would otherwise vary between processes at the same seed.
    """
    original_seed = env.seed

    def seed(value):
        np.random.seed(value)
        return original_seed(value)

    env.seed = seed
    return env


def make_env(
    task_id: int,
    seed: int,
    suite_name: str = "libero_spatial",
    max_episode_steps: int = 10000,
) -> LiberoEnv:
    """Build a single-env LiberoEnv pinned to ``task_id`` / ``seed``."""
    from rlinf.envs.libero.libero_env import LiberoEnv
    from rlinf.envs.libero.utils import benchmark as _bench_mod

    class _PoseReadingLiberoEnv(LiberoEnv):
        """LiberoEnv whose workers also answer ``read_poses``; nothing else changes."""

        def get_env_fns(self):
            return [_with_pose_reader(fn) for fn in super().get_env_fns()]

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
    return _PoseReadingLiberoEnv(
        cfg=cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None
    )


# ---------------------------------------------------------------------------
# Facade implementing the robots.libero.env_client protocol
# ---------------------------------------------------------------------------


class LiberoEnvFacade(BaseEnvFacade):
    """Implements :class:`robots.libero.env_client.LiberoEnvClient`
    over :class:`rlinf.envs.libero.libero_env.LiberoEnv`.

    All return values are converted to CPU numpy so the agent process
    (which does not import torch) can consume them after the pickle round
    trip.
    """

    def __init__(self, env: LiberoEnv, *, meta: dict):
        self._env = env
        self._env_idx = 0
        self._closed = False
        # Identifies what task/seed this server was launched with — the
        # client compares against its own expected values at construction
        # and refuses to talk to a stale or mis-configured server.
        self._meta = dict(meta)
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc.update(
            {
                "env.raw_obs": self.raw_obs,
                "env.render_camera": self.render_camera,
                "env.get_camera_meta": self.get_camera_meta,
                "env.get_task_language": self.get_task_language,
                "env.poses": self.poses,
                "env.check_paths": self.check_paths,
                "env.sim_state": self.sim_state,
                "env.plan_path": self.plan_path,
            }
        )
        self._readonly_methods.add("env.get_task_language")
        self._readonly_methods.add("env.poses")
        self._readonly_methods.add("env.check_paths")
        self._readonly_methods.add("env.sim_state")
        self._readonly_methods.add("env.plan_path")

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
        obs, info = self._env.reset()
        obs = self._strip_obs(to_numpy_tree(obs))
        return obs, to_numpy_tree(info)

    def step(self, action):
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

    def raw_obs(self) -> dict:
        return to_numpy_tree(self._env.current_raw_obs[self._env_idx])

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

    def poses(self, bodies=None, sites=()) -> dict:
        """Ground-truth world poses of named bodies and sites (see ``_attach_pose_reader``)."""
        return self._env.env.workers[self._env_idx].env_call(
            "read_poses",
            args=[None if bodies is None else list(bodies), list(sites)],
            target="self",
        )

    def sim_state(self) -> list:
        """Flattened MuJoCo state of the live simulator."""
        return self._env.env.workers[self._env_idx].env_call(
            "sim_state", args=[], target="self"
        )

    def check_paths(self, paths, options=None) -> list:
        """Feasibility of grip-site paths on a copy of the live simulator state."""
        return self._env.env.workers[self._env_idx].env_call(
            "check_paths",
            args=[[np.asarray(p).tolist() for p in paths], options or {}],
            target="self",
        )

    def plan_path(self, goal_pose, options=None) -> dict:
        """Plan to a grip-site pose on a copy of the live simulator state."""
        return self._env.env.workers[self._env_idx].env_call(
            "plan_path",
            args=[np.asarray(goal_pose).tolist(), options or {}],
            target="self",
        )


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

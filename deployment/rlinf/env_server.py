"""LIBERO host process for the LLM-in-the-loop agent (env-only).

Owns the RLinf/LIBERO bootstrap (path setup, env config builder) and
exposes a pickle-framed RPC server over the
:class:`~physical_agent.rpc_driver.socket.SocketRpcServer`. The agent
process drives a :class:`~physical_agent.envs.libero.tools.LiberoPrimitives`
locally and reaches in only for ``env.*`` method calls; the model side
goes over HTTP to a separate ``deployment/rlinf/vla_server.py`` process
(see :class:`~physical_agent.rpc_driver.vla_client.VLAClient`).

Launched as a subprocess by :func:`cli.main.start_driver`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import Any

# MuJoCo env vars must be set BEFORE importing anything that touches MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from physical_agent.rpc_driver.socket import SocketRpcServer
from physical_agent.utils.config import (
    get_repo_root,
    get_rlinf_repo_path,
)
from physical_agent.utils.logging import get_logger, init_output_dir

logger = get_logger("driver")

PHYSICALAGENT_ROOT = get_repo_root()
RLINF_REPO_PATH = get_rlinf_repo_path() or (PHYSICALAGENT_ROOT.parent / "rlinf").resolve()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from rlinf.envs.libero.libero_env import LiberoEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_env_cfg(
    *,
    task_suite_name: str = "libero_spatial",
    specific_reset_id: int = 0,
    seed: int = 0,
    max_episode_steps: int = 600,
) -> Any:
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
                **({"robots": [os.environ["LIBERO_ROBOT_BASE"]]}
                   if os.environ.get("LIBERO_ROBOT_BASE") else {}),
            },
        }
    )
    return cfg


def make_env(task_id: int, seed: int, suite_name: str = "libero_spatial",
             max_episode_steps: int = 600) -> LiberoEnv:
    """Build a single-env LiberoEnv pinned to ``task_id`` / ``seed``."""
    from rlinf.envs.libero.utils import benchmark as _bench_mod
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
    return LiberoEnv(cfg=cfg, num_envs=1, seed_offset=0,
                     total_num_processes=1, worker_info=None)


# ---------------------------------------------------------------------------
# Facades implementing the envs.libero.tools protocols
# ---------------------------------------------------------------------------


def _to_numpy_tree(x):
    """Recursively convert torch tensors to CPU numpy arrays so the result
    pickles cleanly across the agent/driver wire."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_numpy_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_numpy_tree(v) for v in x)
    return x


class LiberoEnvFacade:
    """Implements :class:`physical_agent.envs.libero.libero_env_client.LiberoEnvClient`
    over :class:`rlinf.envs.libero.libero_env.LiberoEnv`.

    All return values are converted to CPU numpy so the agent process
    (which does not import torch) can consume them after the pickle round
    trip.
    """

    def __init__(self, env: LiberoEnv, *, meta: dict):
        self._env = env
        self._env_idx = 0
        self._done = False
        # Identifies what task/seed this server was launched with — the
        # client compares against its own expected values at construction
        # and refuses to talk to a stale or mis-configured server.
        self._meta = dict(meta)

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

    def _record_done(self, *signals: Any) -> None:
        """OR the truthiness of every termination/truncation signal into
        ``self._done`` so subsequent step() calls short-circuit."""
        for s in signals:
            if np.asarray(s).any():
                self._done = True
                return

    # ---- gym-like surface ----

    def reset(self):
        obs, info = self._env.reset()
        obs = self._strip_obs(_to_numpy_tree(obs))
        self._done = False
        return obs, _to_numpy_tree(info)

    def step(self, action):
        assert not self._done, "step called after episode done"
        obs, rew, term, trunc, info = self._env.step(self._expand_action(action))
        obs = self._strip_obs(_to_numpy_tree(obs))
        term = self._strip(_to_numpy_tree(term))
        trunc = self._strip(_to_numpy_tree(trunc))
        self._record_done(term, trunc)
        return (
            obs,
            self._strip(_to_numpy_tree(rew)),
            term,
            trunc,
            _to_numpy_tree(info),
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
        assert not self._done, "chunk_step called after episode done"
        obs_list, rew, term, trunc, info = self._env.chunk_step(
            self._expand_chunk(actions)
        )
        obs_list = [self._strip_obs(_to_numpy_tree(o)) for o in obs_list]
        term = self._strip(_to_numpy_tree(term))
        trunc = self._strip(_to_numpy_tree(trunc))
        self._record_done(term, trunc)
        obs_field = obs_list if return_all_frames else obs_list[-1]
        return (
            obs_field,
            self._strip(_to_numpy_tree(rew)),
            term,
            trunc,
            _to_numpy_tree(info),
        )

    def raw_obs(self) -> dict:
        return _to_numpy_tree(self._env.current_raw_obs[self._env_idx])

    def get_env_meta(self) -> dict:
        """Return the meta info this server was launched with. """
        return dict(self._meta)

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        return _to_numpy_tree(
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
        return _to_numpy_tree(
            self._env.get_camera_meta(
                camera_name=camera_name, height=height, width=width
            )
        )

    def get_task_language(self) -> str | None:
        return self._env.task_descriptions[self._env_idx]

    def cached_image(self) -> np.ndarray | None:
        cached = getattr(self._env, "_cached_full_image", None)
        if cached is None:
            return None
        return cached.cpu().numpy() if hasattr(cached, "cpu") else np.asarray(cached)


# ---------------------------------------------------------------------------
# RPC dispatcher
# ---------------------------------------------------------------------------


_INITIAL_PPID = os.getppid()


def _start_parent_watchdog(server: SocketRpcServer, shutdown_event: threading.Event,
                           poll_s: float = 2.0) -> None:
    """Shut the RPC server down if the agent (parent) process dies."""

    def _watch() -> None:
        while not shutdown_event.is_set():
            time.sleep(poll_s)
            ppid = os.getppid()
            if ppid != _INITIAL_PPID or ppid == 1:
                logger.warning(
                    "parent died (ppid %s -> %s); stopping RPC server",
                    _INITIAL_PPID,
                    ppid,
                )
                shutdown_event.set()
                threading.Thread(target=server.shutdown, daemon=True).start()
                return

    threading.Thread(target=_watch, daemon=True).start()


def _build_dispatcher(env: LiberoEnvFacade,
                      shutdown_event: threading.Event):
    """Route ``env.*`` / ``shutdown`` to the right callable."""

    def dispatch(method: str, args: tuple, kwargs: dict):
        if method.startswith("env."):
            attr = method[len("env."):]
            try:
                return getattr(env, attr)(*args, **kwargs)
            except Exception as e:
                logger.warning("run method %s failed: %s", method, e)
                raise e
        if method == "shutdown":
            shutdown_event.set()
            return {"ok": True}
        raise ValueError(f"unknown RPC method: {method!r}")

    return dispatch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suite", type=str, default="libero_spatial")
    p.add_argument("--max-episode-steps", type=int, default=600)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--transport-host", type=str, default="127.0.0.1")
    p.add_argument("--transport-port", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialise unified logging for this run
    init_output_dir(args.output_dir)

    logger.info("task=%d  seed=%d  output_dir=%s", args.task, args.seed, args.output_dir)

    raw_env = make_env(args.task, args.seed, suite_name=args.suite,
                       max_episode_steps=args.max_episode_steps)
    env_facade = LiberoEnvFacade(
        raw_env,
        meta={
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
        },
    )

    shutdown_event = threading.Event()
    dispatch = _build_dispatcher(env_facade, shutdown_event)

    server = SocketRpcServer(
        (args.transport_host, args.transport_port), dispatch,
    )
    bound_host, bound_port = server.server_address
    client_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
    print(
        json.dumps({
            "event": "transport_ready",
            "kind": "socket",
            "host": client_host,
            "port": bound_port,
        }),
        flush=True,
    )
    logger.info("RPC server listening on %s:%s", client_host, bound_port)

    _start_parent_watchdog(server, shutdown_event)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        shutdown_event.wait()
    finally:
        server.shutdown()
        server.server_close()
    logger.info("driver exited cleanly")


if __name__ == "__main__":
    main()

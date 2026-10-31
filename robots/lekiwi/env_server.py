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

"""RPC server owning one RLinf LeKiwi ``RealWorldEnv`` in this process.

The env runs in-process — no Ray, no scheduler cluster — because the robot's
computer is a small board where five scheduler manager processes (~0.7 GB)
would exhaust memory. ``node_rank`` stays ``None`` throughout the hardware
config, which is how RLinf's parts know to open their devices locally rather
than hosting them on a scheduler worker (``RobotPart.connect``). The RPC
surface is unchanged, so the client and the agent attach exactly as before.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from robots.lekiwi.runtime_config import load_runtime_config
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.logging import get_logger
from rpent.utils.serialization import to_numpy_tree

logger = get_logger("lekiwi_env_server")


@dataclass(frozen=True)
class _HeldVelocity:
    """One operator hold: the clipped action and the instant it expires."""

    action: np.ndarray
    expires_at: float


class LeKiwiEnvFacade(BaseEnvFacade):
    """Expose the LeKiwi ``env.*`` protocol from an in-process env backend."""

    _METHODS = (
        "get_env_meta",
        "reset",
        "get_robot_state",
        "get_observation",
        "get_camera_meta",
        "drive",
        "hold",
        "stop",
    )

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        super().__init__()

    def _register_rpc(self) -> None:
        for name in self._METHODS:
            self._rpc[f"env.{name}"] = getattr(self._backend, name)
        self._readonly_methods.update(
            {
                "env.get_env_meta",
                "env.get_robot_state",
                "env.get_observation",
                "env.get_camera_meta",
            }
        )


def _local_worker_info(cfg: Any) -> Any:
    """Describe a one-worker, one-robot placement that owns hardware locally.

    ``cluster_node_rank`` and the hardware config's ``node_rank`` are ``None``:
    that is what makes every part open its device in this process. The cluster
    section's ``node_rank: 0`` would host each part on a scheduler worker and
    drag in the whole Ray cluster, which the robot's computer cannot carry.
    """
    from omegaconf import OmegaConf
    from rlinf.robotics.discovery import RobotInfo
    from rlinf.scheduler import AcceleratorType, WorkerAddress, WorkerInfo

    hardware = OmegaConf.to_container(cfg.cluster.node_groups[0].hardware, resolve=True)
    from rlinf.robotics.robots.lekiwi import LeKiwiConfig

    robot_config = LeKiwiConfig(**{**hardware["configs"][0], "node_rank": None})
    # LeKiwi reports its type as its model (RobotConfig.hardware_model).
    robot_info = RobotInfo(
        type=hardware["type"],
        model=hardware["type"],
        config=robot_config,
    )
    return WorkerInfo(
        address=WorkerAddress("lekiwi", 0),
        rank=0,
        group_world_size=1,
        cluster_node_rank=None,
        accelerator_type=AcceleratorType.NO_ACCEL,
        accelerator_model="",
        accelerator_rank=0,
        node_ip="",
        node_port=0,
        available_accelerators=[],
        hardware_infos=[robot_info],
    )


class LeKiwiEnvBackend:
    """Own the physical LeKiwi base and camera through one RLinf env."""

    def __init__(self, cfg: Any, controller_config: dict[str, Any]) -> None:
        from rlinf.envs.real.env import RealWorldEnv

        from robots.lekiwi.rpent_env import register_rpent_lekiwi_env

        register_rpent_lekiwi_env()
        self.cfg = cfg
        self.controller = dict(controller_config)
        worker_info = _local_worker_info(cfg)
        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=worker_info,
        )
        self.action_dim = int(self.env.action_space.shape[-1])
        env_config = self.env.env.call("get_wrapper_attr", "config")[0]
        self.step_frequency = float(env_config.step_frequency)
        # The vector env's action space is batched, (1, 3); clip against the
        # single-env space so a commanded (3,) velocity stays (3,).
        self.velocity_limits = np.asarray(
            self.env.env.single_action_space.high, dtype=np.float32
        )
        self.step_period_s = 1.0 / self.step_frequency
        # A held command drives the env from a control thread, paced by the
        # env's own step, so motion does not depend on RPC latency. RPC
        # threads only swap the command and read the last observation; the
        # bus lock keeps them off the wheels while a step is in flight.
        self._condition = threading.Condition()
        self._bus_lock = threading.Lock()
        self._command: _HeldVelocity | None = None
        self._closed = False
        self._steps = 0
        self._latest_observation: Any = None
        self._latest_observed_at = 0.0
        self._control_error: str | None = None
        self._control = threading.Thread(
            target=self._control_loop, name="lekiwi-control", daemon=True
        )
        self._control.start()

    def get_env_meta(self) -> dict[str, Any]:
        return {
            "ok": True,
            "action_dim": self.action_dim,
            "velocity_limits": self.velocity_limits.tolist(),
            "step_frequency": self.step_frequency,
        }

    def close_env(self) -> None:
        """Stop the control thread, zero the wheels, then close the env."""
        with self._condition:
            self._closed = True
            self._command = None
            self._condition.notify_all()
        self._control.join(timeout=5.0)
        if self._control.is_alive():
            logger.warning("the LeKiwi control thread did not stop within 5s")
        with self._bus_lock:
            try:
                self._raw_rlinf_env().stop()
            except Exception:  # noqa: BLE001 - shutdown must not mask the error
                logger.warning("stopping the LeKiwi base failed", exc_info=True)
        try:
            self.env.close()
        except Exception:  # noqa: BLE001 - shutdown must not mask the error
            logger.warning("closing the LeKiwi env failed", exc_info=True)

    @staticmethod
    def _strip_batch(value: Any) -> Any:
        array = to_numpy_tree(value)
        if isinstance(array, np.ndarray) and array.ndim > 0 and array.shape[0] == 1:
            return array[0]
        if isinstance(array, list) and len(array) == 1:
            return array[0]
        return array

    def _raw_rlinf_env(self) -> Any:
        """Return RPent's unwrapped RLinf environment compatibility layer."""
        return self.env.env.envs[0].unwrapped

    def _clear_command(self) -> None:
        with self._condition:
            self._command = None
            self._condition.notify_all()

    def reset(self) -> dict[str, Any]:
        self._clear_command()
        with self._bus_lock:
            observation, info = self.env.reset()
            steps = self._steps
        return {
            "ok": True,
            "info": to_numpy_tree(info),
            "robot_state": self.get_robot_state(),
            "states": self._strip_batch(observation.get("states")),
            "main_images": self._strip_batch(observation.get("main_images")),
            "steps": steps,
        }

    def get_robot_state(self) -> dict[str, Any]:
        with self._bus_lock:
            velocity = self._raw_rlinf_env().get_measured_velocity().tolist()
        with self._condition:
            steps = self._steps
            control_error = self._control_error
        return {
            "velocity": velocity,
            "action_dim": self.action_dim,
            "velocity_limits": self.velocity_limits.tolist(),
            "step_frequency": self.step_frequency,
            "steps": steps,
            "control_error": control_error,
        }

    def get_observation(self) -> dict[str, Any]:
        """Return the last step observation, or read the sensors when idle.

        While a hold is live the control thread captured an observation in
        the step it just took, so serving that copy costs neither a bus read
        nor a camera wait.
        """
        observation = self._latest_observation
        fresh = time.monotonic() - self._latest_observed_at <= 2 * self.step_period_s
        if self._command is not None and observation is not None and fresh:
            return self._observation_payload(observation, self._steps)
        try:
            with self._bus_lock:
                return self._read_live_observation()
        except Exception as exc:
            logger.warning("live observation refresh failed: %s", exc)
            return {}

    def _observation_payload(self, observation: Any, steps: int) -> dict[str, Any]:
        """Convert a batched step observation to the client's flat shape."""
        payload: dict[str, Any] = {
            "states": self._strip_batch(observation.get("states")),
            "main_images": self._strip_batch(observation.get("main_images")),
            "steps": steps,
        }
        extras = observation.get("extra_view_images")
        if extras is not None:
            payload["extra_view_images"] = self._strip_batch(extras)
        return payload

    def _read_live_observation(self) -> dict[str, Any]:
        """Re-read the sensors live and map them to policy image keys.

        Frames pass through the observation wrappers unchanged, so
        re-reading them without a robot step yields the same format as a
        stepped obs.
        """
        raw = self._raw_rlinf_env().get_live_observation()
        frames = raw.get("frames", {})
        output: dict[str, Any] = {
            "states": np.asarray(raw["state"]["velocity"], dtype=np.float32)
        }
        main_key = self.cfg.env.eval.get("main_image_key")
        output["main_images"] = np.asarray(frames[main_key])
        extras = [
            np.asarray(frames[name]) for name in sorted(frames) if name != main_key
        ]
        if extras:
            output["extra_view_images"] = np.stack(extras, axis=0)
        output["steps"] = self._steps
        return output

    def get_camera_meta(self) -> dict[str, Any]:
        metadata = to_numpy_tree(self._raw_rlinf_env().get_camera_metadata())
        main_key = self.cfg.env.eval.get("main_image_key")
        names = sorted(metadata.get("cameras", {}))
        extras = [name for name in names if name != main_key]
        metadata["observation_camera_map"] = {
            "main": main_key,
            **{f"extra_{index}": name for index, name in enumerate(extras)},
        }
        return metadata

    def drive(self, velocity: Any, *, duration_s: float) -> dict[str, Any]:
        """Hold one body velocity for a bounded duration, then stop.

        An explicit drive takes over from a Dashboard hold. The env's
        dead-man semantics stop the wheels after every control period, so
        the base keeps moving only while this loop keeps stepping.
        """
        requested = self._validate_velocity(velocity)
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")
        duration_s = min(float(duration_s), self.controller["drive_max_duration_s"])

        action = self._clip(requested)
        self._clear_command()
        steps = max(1, int(round(duration_s * self.step_frequency)))
        states: Any = None
        observation: Any = None
        deadline = time.time() + duration_s + self.controller["drive_max_duration_s"]
        for _ in range(steps):
            if time.time() >= deadline:
                break
            with self._bus_lock:
                observation, _reward, _term, _trunc, _info = self.env.step(
                    action[None, :]
                )
        if observation is not None:
            states = self._strip_batch(observation.get("states"))
        return {
            "ok": True,
            "requested_velocity": requested.tolist(),
            "commanded_velocity": np.asarray(action, dtype=np.float32).tolist(),
            "steps": steps,
            "duration_s": duration_s,
            "robot_state": self.get_robot_state(),
            "states": states,
        }

    def hold(self, velocity: Any, *, ttl_s: float = 1.0) -> dict[str, Any]:
        """Drive a body velocity until the command expires or ``stop`` arrives.

        Returns without waiting for the motion: a control thread steps the
        env at its own rate, so the base keeps moving while the caller
        refreshes the command, however long each refresh takes on the wire.
        """
        requested = self._validate_velocity(velocity)
        ttl = float(ttl_s)
        if not np.isfinite(ttl) or ttl <= 0:
            raise ValueError("ttl_s must be finite and positive")
        ttl = min(ttl, self.controller["hold_max_ttl_s"])
        action = self._clip(requested)
        with self._condition:
            if self._closed:
                raise RuntimeError("the LeKiwi environment is closed")
            self._command = (
                _HeldVelocity(action=action, expires_at=time.monotonic() + ttl)
                if action.any()
                else None
            )
            self._condition.notify_all()
            steps = self._steps
        return {
            "ok": True,
            "requested_velocity": requested.tolist(),
            "commanded_velocity": action.tolist(),
            "ttl_s": ttl,
            "steps": steps,
        }

    def stop(self) -> dict[str, Any]:
        """Drop any held command and zero the wheels now."""
        with self._condition:
            self._command = None
            self._condition.notify_all()
        with self._bus_lock:
            try:
                self._raw_rlinf_env().stop()
            except Exception:  # noqa: BLE001 - report through robot_state
                logger.warning("stopping the LeKiwi base failed", exc_info=True)
        return {
            "ok": True,
            "robot_state": self.get_robot_state(),
        }

    def _control_loop(self) -> None:
        """Step the env at its control rate while a held command is live."""
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or self._command is not None
                )
                if self._closed:
                    return
            while self._step_once():
                pass

    def _step_once(self) -> bool:
        """Take one control step if the held command is still live."""
        with self._bus_lock:
            command = self._command
            if (
                self._closed
                or command is None
                or time.monotonic() >= command.expires_at
            ):
                return False
            try:
                observation, _reward, _term, _trunc, _info = self.env.step(
                    command.action[None, :]
                )
            except Exception as exc:  # noqa: BLE001 - clear it and report
                logger.warning("LeKiwi control step failed: %s", exc)
                with self._condition:
                    self._control_error = f"{type(exc).__name__}: {exc}"
                    self._command = None
                    self._condition.notify_all()
                return False
        with self._condition:
            self._steps += 1
            self._latest_observation = observation
            self._latest_observed_at = time.monotonic()
        return True

    def _validate_velocity(self, velocity: Any) -> np.ndarray:
        requested = np.asarray(velocity, dtype=np.float32)
        if requested.shape != (self.action_dim,):
            raise ValueError(
                f"velocity must contain exactly {self.action_dim} values, "
                f"got shape {requested.shape}"
            )
        if not np.isfinite(requested).all():
            raise ValueError("velocity must contain only finite values")
        return requested

    def _clip(self, requested: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.clip(requested, self.velocity_limits * -1.0, self.velocity_limits),
            dtype=np.float32,
        )


def main(
    *,
    backend_class: type[LeKiwiEnvBackend] = LeKiwiEnvBackend,
    load_runtime_config: Any = load_runtime_config,
    facade_class: type[LeKiwiEnvFacade] = LeKiwiEnvFacade,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--robot-config", default=None)
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved RLinf config and exit without launching.",
    )
    args = parser.parse_args()

    runtime = load_runtime_config(args.robot_config)
    if args.print_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(runtime.rlinf))
        return 0
    backend = backend_class(runtime.rlinf, runtime.controller)
    facade = facade_class(backend)
    try:
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        backend.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
